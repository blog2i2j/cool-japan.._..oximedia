//! Multi-track frame-level pipeline executor with DTS-ordered interleaving.
//!
//! This module provides [`MultiTrackExecutor`] — the real decode → filter →
//! encode engine for `oximedia-transcode`.  It connects the `FrameDecoder` /
//! `FilterGraph` / `FrameEncoder` plumbing from [`pipeline_context`] to a
//! container-level output via a [`Muxer`] and performs *DTS-ordered
//! interleaving* across all tracks using a min-heap.
//!
//! # Architecture
//!
//! ```text
//!    [FrameDecoder₀] → FilterGraph₀ → [FrameEncoder₀] ─┐
//!    [FrameDecoder₁] → FilterGraph₁ → [FrameEncoder₁] ─┤→ DTS min-heap → Muxer
//!           …                                            ┘
//! ```
//!
//! ## Execute loop
//!
//! 1. For each active track, call [`FrameDecoder::decode_next`].
//! 2. Apply the track's [`FilterGraph::apply`] to the decoded frame.
//! 3. Pass filtered frames to the track's [`FrameEncoder::encode_frame`].
//! 4. Push resulting encoded bytes as a `StagedPacket` onto the DTS min-heap.
//! 5. After all tracks are exhausted, flush each encoder.
//! 6. Pop the heap in DTS order and write every packet to the [`Muxer`].
//!
//! The [`MultiTrackExecutor::step`] method performs one packet-cycle (one pass
//! through all tracks) and pushes ready encoded data into the internal staging
//! buffer, so a segment or parallel driver can call it externally.
//!
//! [`pipeline_context`]: crate::pipeline_context

#![allow(clippy::module_name_repetitions)]

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use bytes::Bytes;
use oximedia_container::{Muxer, Packet, PacketFlags, StreamInfo};
use oximedia_core::{Rational, Timestamp};

use crate::pipeline_context::{FilterGraph, FrameDecoder, FrameEncoder};
use crate::{Result, TranscodeError};

// ─── PerTrack ─────────────────────────────────────────────────────────────────

/// One logical media track wired through decode → filter → encode.
///
/// Created by the caller with concrete decoder, filter graph, and encoder
/// implementations, then handed to [`MultiTrackExecutor::add_track`].
pub struct PerTrack {
    /// The stream index in the output muxer for packets from this track.
    pub stream_index: usize,
    /// Decoder for this track.
    pub decoder: Box<dyn FrameDecoder>,
    /// Filter graph applied between decode and encode.
    pub filter_graph: FilterGraph,
    /// Encoder for this track.
    pub encoder: Box<dyn FrameEncoder>,
    /// `true` when the decoder has reported EOF and the encoder has been flushed.
    pub flushed: bool,
    /// Frame counter; used to derive synthetic PTS for flush packets.
    frame_count: u64,
    /// Accumulated encoded-bytes count (public for stats queries).
    pub encoded_bytes: u64,
    /// Accumulated encoded-frame count (public for stats queries).
    pub encoded_frames: u64,
    /// Whether this track carries audio (`true`) or video (`false`).
    ///
    /// Determined from the first decoded frame and used to populate the
    /// `is_audio` flag on flush tail-packets — ensuring the muxer receives
    /// correct stream-type information at EOS even when no frames flow.
    is_audio: Option<bool>,
}

impl PerTrack {
    /// Create a new [`PerTrack`] with the given stream index, decoder,
    /// filter graph, and encoder.
    #[must_use]
    pub fn new(
        stream_index: usize,
        decoder: Box<dyn FrameDecoder>,
        filter_graph: FilterGraph,
        encoder: Box<dyn FrameEncoder>,
    ) -> Self {
        Self {
            stream_index,
            decoder,
            filter_graph,
            encoder,
            flushed: false,
            frame_count: 0,
            encoded_bytes: 0,
            encoded_frames: 0,
            is_audio: None,
        }
    }

    /// Create a new [`PerTrack`] whose track type is known at construction.
    ///
    /// Use this constructor when the stream kind (audio vs video) is
    /// available from the container's [`StreamInfo`] before decoding starts,
    /// so that `flush_encoder` emits packets with the correct type even if
    /// no frames were decoded (e.g., a very short audio track).
    #[must_use]
    pub fn new_typed(
        stream_index: usize,
        decoder: Box<dyn FrameDecoder>,
        filter_graph: FilterGraph,
        encoder: Box<dyn FrameEncoder>,
        is_audio: bool,
    ) -> Self {
        Self {
            stream_index,
            decoder,
            filter_graph,
            encoder,
            flushed: false,
            frame_count: 0,
            encoded_bytes: 0,
            encoded_frames: 0,
            is_audio: Some(is_audio),
        }
    }

    /// Step this track by one frame: decode → filter → encode.
    ///
    /// Returns `Ok(Some(TrackEncoded))` when a frame was successfully encoded,
    /// `Ok(None)` when the decoder produced no frame (EOF or frame dropped by
    /// filter), or an error if encoding or filter operations fail.
    fn step_frame(&mut self) -> Result<Option<TrackEncoded>> {
        if self.flushed || self.decoder.eof() {
            return Ok(None);
        }

        let frame = match self.decoder.decode_next() {
            Some(f) => f,
            None => return Ok(None),
        };

        let pts_ms = frame.pts_ms;
        let is_audio = frame.is_audio;

        // Latch the track kind from the first frame so flush_encoder can use it.
        if self.is_audio.is_none() {
            self.is_audio = Some(is_audio);
        }

        let filtered = match self.filter_graph.apply(frame)? {
            Some(f) => f,
            None => {
                // Frame dropped by filter — counts as dropped, not an error.
                return Ok(None);
            }
        };

        let encoded = self.encoder.encode_frame(&filtered)?;
        let n = encoded.len() as u64;
        self.encoded_bytes += n;
        self.encoded_frames += 1;
        self.frame_count += 1;

        Ok(Some(TrackEncoded {
            data: encoded,
            pts_ms,
            is_audio,
        }))
    }

    /// Flush the encoder and return remaining encoded bytes (if any).
    ///
    /// Sets `self.flushed = true` after the first call; subsequent calls are
    /// no-ops.
    ///
    /// The `is_audio` flag on the returned tail-packet is taken from the track
    /// type latched during [`step_frame`](Self::step_frame) (or set at
    /// construction via [`new_typed`](Self::new_typed)).  If neither path has
    /// provided a type yet (a zero-frame track), the flush packet is omitted
    /// entirely since there is no stream kind to report.
    fn flush_encoder(&mut self) -> Result<Option<TrackEncoded>> {
        if self.flushed {
            return Ok(None);
        }
        self.flushed = true;
        let data = self.encoder.flush()?;
        if data.is_empty() {
            return Ok(None);
        }
        // If the track type is still unknown (zero-frame track), skip the
        // flush packet rather than reporting a wrong stream kind to the muxer.
        let is_audio = match self.is_audio {
            Some(v) => v,
            None => return Ok(None),
        };
        self.encoded_bytes += data.len() as u64;
        // Derive a synthetic PTS from the frame count (33 ms/frame ≈ 30 fps).
        let pts_ms = self.frame_count as i64 * 33;
        Ok(Some(TrackEncoded {
            data,
            pts_ms,
            is_audio,
        }))
    }
}

// ─── TrackEncoded ─────────────────────────────────────────────────────────────

/// Encoded output produced by a single [`PerTrack::step_frame`] call.
#[derive(Debug)]
struct TrackEncoded {
    data: Vec<u8>,
    pts_ms: i64,
    is_audio: bool,
}

// ─── StagedPacket ─────────────────────────────────────────────────────────────

/// An encoded packet waiting in the DTS min-heap for muxer output.
#[derive(Debug)]
struct StagedPacket {
    /// Effective DTS for heap ordering.
    dts_ms: i64,
    /// Stream index for the muxer.
    stream_index: usize,
    /// Encoded payload.
    data: Vec<u8>,
    /// `true` for audio packets.
    is_audio: bool,
}

impl PartialEq for StagedPacket {
    fn eq(&self, other: &Self) -> bool {
        self.dts_ms == other.dts_ms && self.stream_index == other.stream_index
    }
}

impl Eq for StagedPacket {}

impl PartialOrd for StagedPacket {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StagedPacket {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Primary: DTS ascending; secondary: stream_index ascending for determinism.
        self.dts_ms
            .cmp(&other.dts_ms)
            .then(self.stream_index.cmp(&other.stream_index))
    }
}

// ─── MultiTrackStats ──────────────────────────────────────────────────────────

/// Statistics returned by [`MultiTrackExecutor::execute`].
#[derive(Debug, Clone, Default)]
pub struct MultiTrackStats {
    /// Total encoded frames across all tracks.
    pub total_encoded_frames: u64,
    /// Total encoded bytes across all tracks.
    pub total_encoded_bytes: u64,
    /// Number of packets written to the muxer in DTS order.
    pub packets_muxed: u64,
    /// Number of frames dropped by filter graphs.
    pub frames_dropped: u64,
}

// ─── MultiTrackExecutor ───────────────────────────────────────────────────────

/// Frame-level multi-track decode → filter → encode executor with
/// DTS-ordered muxing.
///
/// # Usage
///
/// ```rust,ignore
/// use oximedia_transcode::multi_track::{MultiTrackExecutor, PerTrack};
/// use oximedia_transcode::pipeline_context::{FilterGraph, Frame};
///
/// // Supply concrete FrameDecoder / FrameEncoder implementations:
/// let mut executor = MultiTrackExecutor::new(muxer);
/// executor.add_track(PerTrack::new(0, decoder0, FilterGraph::new(), encoder0));
/// executor.add_track(PerTrack::new(1, decoder1, FilterGraph::new(), encoder1));
/// let stats = executor.execute(&streams).await?;
/// ```
pub struct MultiTrackExecutor<M: Muxer> {
    /// Per-track decode/filter/encode pipelines.
    tracks: Vec<PerTrack>,
    /// The output container muxer.
    muxer: M,
    /// DTS min-heap: `Reverse` turns `BinaryHeap` (max-heap) into a min-heap.
    heap: BinaryHeap<Reverse<StagedPacket>>,
    /// Timebase used for `Timestamp` construction (1 ms resolution by default).
    timebase: Rational,
    /// Drain the heap to the muxer every `flush_interval` step cycles.
    flush_interval: u64,
    /// Step counter for flush scheduling.
    step_count: u64,
    /// `true` after all tracks have reached EOF.
    tracks_done: bool,
    /// Accumulated statistics.
    stats: MultiTrackStats,
}

impl<M: Muxer> MultiTrackExecutor<M> {
    /// Default flush interval (drain heap every 30 steps).
    const DEFAULT_FLUSH_INTERVAL: u64 = 30;

    /// Creates a new executor wrapping `muxer`.
    ///
    /// Tracks must be added with [`add_track`](Self::add_track) before calling
    /// [`execute`](Self::execute) or [`step`](Self::step).
    pub fn new(muxer: M) -> Self {
        Self {
            tracks: Vec::new(),
            muxer,
            heap: BinaryHeap::new(),
            timebase: Rational::new(1, 1_000),
            flush_interval: Self::DEFAULT_FLUSH_INTERVAL,
            step_count: 0,
            tracks_done: false,
            stats: MultiTrackStats::default(),
        }
    }

    /// Adds a [`PerTrack`] to the executor.
    pub fn add_track(&mut self, track: PerTrack) {
        self.tracks.push(track);
    }

    /// Overrides the heap flush interval (default: 30 steps).
    pub fn set_flush_interval(&mut self, n: u64) {
        self.flush_interval = n.max(1);
    }

    /// Returns a shared reference to the inner muxer.
    #[must_use]
    pub fn muxer(&self) -> &M {
        &self.muxer
    }

    /// Consumes the executor and returns the inner muxer.
    #[must_use]
    pub fn into_muxer(self) -> M {
        self.muxer
    }

    /// Returns the accumulated execution statistics.
    #[must_use]
    pub fn stats(&self) -> &MultiTrackStats {
        &self.stats
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Push a [`TrackEncoded`] result onto the DTS heap.
    fn push_to_heap(&mut self, stream_index: usize, encoded: TrackEncoded) {
        let packet = StagedPacket {
            dts_ms: encoded.pts_ms,
            stream_index,
            data: encoded.data,
            is_audio: encoded.is_audio,
        };
        self.heap.push(Reverse(packet));
    }

    /// Drain all packets in the heap to the muxer in DTS order.
    async fn drain_heap_to_muxer(&mut self) -> Result<()> {
        while let Some(Reverse(staged)) = self.heap.pop() {
            self.write_staged_packet(staged).await?;
        }
        Ok(())
    }

    /// Drain heap packets whose DTS is strictly less than `horizon_ms`.
    ///
    /// This "safe drain" strategy ensures packets behind the current minimum
    /// active DTS are flushed promptly, while packets that might still be
    /// overtaken by a slower track are retained.
    async fn drain_heap_until(&mut self, horizon_ms: i64) -> Result<()> {
        loop {
            match self.heap.peek() {
                Some(Reverse(staged)) if staged.dts_ms < horizon_ms => {
                    let Reverse(pkt) = self.heap.pop().expect("non-empty after peek");
                    self.write_staged_packet(pkt).await?;
                }
                _ => break,
            }
        }
        Ok(())
    }

    /// Write a single [`StagedPacket`] to the muxer.
    async fn write_staged_packet(&mut self, staged: StagedPacket) -> Result<()> {
        let ts = Timestamp::new(staged.dts_ms, self.timebase);
        let flags = if staged.is_audio {
            PacketFlags::empty()
        } else {
            PacketFlags::KEYFRAME
        };
        let pkt = Packet::new(staged.stream_index, Bytes::from(staged.data), ts, flags);
        self.muxer.write_packet(&pkt).await.map_err(|e| {
            TranscodeError::ContainerError(format!("muxer write_packet failed: {e}"))
        })?;
        self.stats.packets_muxed += 1;
        Ok(())
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Perform one step of the pipeline: attempt to decode one frame from
    /// every active track, filter and encode it, then push the result onto
    /// the DTS heap.
    ///
    /// Periodically drains the heap to the muxer based on the minimum active
    /// DTS (safe-drain strategy).
    ///
    /// Returns `true` if at least one track produced an encoded packet this
    /// step, `false` when all tracks are exhausted.
    ///
    /// # Errors
    ///
    /// Propagates errors from the filter graph, encoder, or muxer.
    pub async fn step(&mut self) -> Result<bool> {
        if self.tracks_done {
            return Ok(false);
        }

        // Collect encoded output from all tracks before mutating `self.heap`.
        // This avoids a double-borrow of `self` when `push_to_heap` is called
        // inside the loop that also borrows `self.tracks`.
        let mut pending: Vec<(usize, TrackEncoded)> = Vec::new();
        let mut min_active_dts: Option<i64> = None;

        for track in &mut self.tracks {
            if track.flushed || track.decoder.eof() {
                continue;
            }

            if let Some(encoded) = track.step_frame()? {
                let dts = encoded.pts_ms;
                min_active_dts = Some(match min_active_dts {
                    Some(prev) => prev.min(dts),
                    None => dts,
                });
                pending.push((track.stream_index, encoded));
            }
        }

        let any_produced = !pending.is_empty();
        let encoded_this_step = pending.len() as u64;

        // Push collected results onto the DTS heap.
        for (stream_index, encoded) in pending {
            self.push_to_heap(stream_index, encoded);
        }

        // Aggregate byte stats.
        self.stats.total_encoded_bytes = self.tracks.iter().map(|t| t.encoded_bytes).sum();
        self.stats.total_encoded_frames += encoded_this_step;

        self.step_count += 1;

        // Safe-drain the heap on schedule.
        if self.step_count % self.flush_interval == 0 {
            if let Some(horizon) = min_active_dts {
                self.drain_heap_until(horizon).await?;
            }
        }

        // Update done flag.
        let all_done = self.tracks.iter().all(|t| t.decoder.eof() || t.flushed);
        if all_done {
            self.tracks_done = true;
        }

        Ok(any_produced)
    }

    /// Execute the full pipeline end-to-end.
    ///
    /// 1. Registers `streams` with the muxer and writes the header.
    /// 2. Calls [`step`](Self::step) in a loop until all tracks are exhausted.
    /// 3. Flushes each track's encoder.
    /// 4. Drains the remaining heap to the muxer in DTS order.
    /// 5. Writes the muxer trailer.
    ///
    /// Returns accumulated [`MultiTrackStats`].
    ///
    /// # Errors
    ///
    /// Returns an error if any stage (filter, encode, mux header/packet/trailer)
    /// fails.
    pub async fn execute(&mut self, streams: &[StreamInfo]) -> Result<MultiTrackStats> {
        // Register streams with the muxer.
        for stream in streams {
            self.muxer
                .add_stream(stream.clone())
                .map_err(|e| TranscodeError::ContainerError(format!("add_stream failed: {e}")))?;
        }

        self.muxer
            .write_header()
            .await
            .map_err(|e| TranscodeError::ContainerError(format!("write_header failed: {e}")))?;

        // Main decode/encode loop.
        loop {
            let produced = self.step().await?;
            if self.tracks_done {
                break;
            }
            if !produced {
                // No track produced a packet — check whether they are all at EOF.
                let all_eof = self.tracks.iter().all(|t| t.decoder.eof() || t.flushed);
                if all_eof {
                    self.tracks_done = true;
                    break;
                }
            }
        }

        // Flush each encoder.
        for idx in 0..self.tracks.len() {
            let stream_index = self.tracks[idx].stream_index;
            if let Some(encoded) = self.tracks[idx].flush_encoder()? {
                self.push_to_heap(stream_index, encoded);
            }
        }

        // Final full heap drain in DTS order.
        self.drain_heap_to_muxer().await?;

        // Finalise stats.
        self.stats.total_encoded_bytes = self.tracks.iter().map(|t| t.encoded_bytes).sum();
        self.stats.total_encoded_frames = self.tracks.iter().map(|t| t.encoded_frames).sum();

        self.muxer
            .write_trailer()
            .await
            .map_err(|e| TranscodeError::ContainerError(format!("write_trailer failed: {e}")))?;

        Ok(self.stats.clone())
    }
}
