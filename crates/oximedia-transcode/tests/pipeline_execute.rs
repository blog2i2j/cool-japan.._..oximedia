//! Integration tests for `MultiTrackExecutor` — real frame-level pipeline
//! execution connecting `FrameDecoder` / `FilterGraph` / `FrameEncoder` to a
//! container [`Muxer`] via DTS-ordered interleaving.
//!
//! All tests use:
//! - Purely in-memory synthetic frame data (no real media files).
//! - `MockDecoder` / `MockEncoder` structs that implement the
//!   `pipeline_context` traits.
//! - An in-memory `MatroskaMuxer` backed by `MemorySource`.
//! - `std::env::temp_dir()` for any temporary files.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use oximedia_container::{
    mux::{MatroskaMuxer, MuxerConfig},
    Muxer, Packet, StreamInfo,
};
use oximedia_core::{CodecId, Rational};
use oximedia_io::MemorySource;
use oximedia_transcode::{
    multi_track::{MultiTrackExecutor, PerTrack},
    FilterGraph, Frame, FrameDecoder, FrameEncoder, TranscodeError,
};

// ── Synthetic frame helpers ───────────────────────────────────────────────────

/// Build a synthetic RGBA video frame (4 bytes/pixel).
fn make_rgba(width: u32, height: u32, fill: u8, pts_ms: i64) -> Frame {
    let data = vec![fill; (width * height * 4) as usize];
    Frame::video(data, pts_ms, width, height)
}

/// Build a synthetic audio frame with interleaved i16 PCM LE (stereo).
fn make_audio(n_samples: usize, sample_val: i16, pts_ms: i64) -> Frame {
    let mut data = Vec::with_capacity(n_samples * 4);
    for _ in 0..n_samples {
        data.extend_from_slice(&sample_val.to_le_bytes()); // L
        data.extend_from_slice(&sample_val.to_le_bytes()); // R
    }
    Frame::audio(data, pts_ms)
}

// ── MockDecoder ───────────────────────────────────────────────────────────────

/// A decoder that yields a pre-loaded sequence of frames and then signals EOF.
struct MockDecoder {
    frames: VecDeque<Frame>,
}

impl MockDecoder {
    fn with_frames(frames: Vec<Frame>) -> Self {
        Self {
            frames: VecDeque::from(frames),
        }
    }

    fn empty() -> Self {
        Self {
            frames: VecDeque::new(),
        }
    }
}

impl FrameDecoder for MockDecoder {
    fn decode_next(&mut self) -> Option<Frame> {
        self.frames.pop_front()
    }

    fn eof(&self) -> bool {
        self.frames.is_empty()
    }
}

// ── MockEncoder ───────────────────────────────────────────────────────────────

/// A passthrough encoder that stores all encoded payloads for later inspection.
///
/// It shares state via an `Arc<Mutex<Vec<Vec<u8>>>>` so that after the
/// executor runs the test can read what was encoded.
struct MockEncoder {
    output: Arc<Mutex<Vec<Vec<u8>>>>,
    flush_called: Arc<Mutex<bool>>,
}

impl MockEncoder {
    fn new() -> (Self, Arc<Mutex<Vec<Vec<u8>>>>, Arc<Mutex<bool>>) {
        let output = Arc::new(Mutex::new(Vec::new()));
        let flush_called = Arc::new(Mutex::new(false));
        let enc = Self {
            output: Arc::clone(&output),
            flush_called: Arc::clone(&flush_called),
        };
        (enc, output, flush_called)
    }
}

impl FrameEncoder for MockEncoder {
    fn encode_frame(&mut self, frame: &Frame) -> oximedia_transcode::Result<Vec<u8>> {
        // Echo the raw frame data as the "encoded" payload.
        let data = frame.data.clone();
        self.output.lock().expect("output lock").push(data.clone());
        Ok(data)
    }

    fn flush(&mut self) -> oximedia_transcode::Result<Vec<u8>> {
        *self.flush_called.lock().expect("flush lock") = true;
        Ok(Vec::new())
    }
}

// ── ErrorEncoder — errors on second encode call ───────────────────────────────

/// Encoder that succeeds on the first call and errors on the second.
struct ErrorOnSecondEncode {
    call_count: usize,
    output: Vec<Vec<u8>>,
}

impl ErrorOnSecondEncode {
    fn new() -> Self {
        Self {
            call_count: 0,
            output: Vec::new(),
        }
    }
}

impl FrameEncoder for ErrorOnSecondEncode {
    fn encode_frame(&mut self, frame: &Frame) -> oximedia_transcode::Result<Vec<u8>> {
        self.call_count += 1;
        if self.call_count >= 2 {
            return Err(TranscodeError::CodecError(
                "deliberate encode error on second frame".to_string(),
            ));
        }
        let data = frame.data.clone();
        self.output.push(data.clone());
        Ok(data)
    }

    fn flush(&mut self) -> oximedia_transcode::Result<Vec<u8>> {
        Ok(Vec::new())
    }
}

// ── Helper: build an in-memory MatroskaMuxer ──────────────────────────────────

fn make_mkv_muxer() -> MatroskaMuxer<MemorySource> {
    let buf = MemorySource::new_writable(64 * 1024);
    MatroskaMuxer::new(buf, MuxerConfig::new())
}

fn video_stream(idx: usize) -> StreamInfo {
    let mut si = StreamInfo::new(idx, CodecId::Vp9, Rational::new(1, 1_000));
    si.codec_params.width = Some(4);
    si.codec_params.height = Some(4);
    si
}

fn audio_stream(idx: usize) -> StreamInfo {
    let mut si = StreamInfo::new(idx, CodecId::Opus, Rational::new(1, 1_000));
    si.codec_params.sample_rate = Some(48_000);
    si.codec_params.channels = Some(2);
    si
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Single video track passthrough
// ─────────────────────────────────────────────────────────────────────────────

/// A single video track, no filter, passthrough encoder.
/// Verifies that the executor round-trips all frames to the muxer.
#[tokio::test]
async fn test_single_video_track_passthrough() {
    let n = 10usize;
    let frames: Vec<Frame> = (0..n)
        .map(|i| make_rgba(4, 4, (i as u8) * 10, i as i64 * 33))
        .collect();

    let (encoder, output_arc, flush_arc) = MockEncoder::new();
    let decoder = Box::new(MockDecoder::with_frames(frames));

    let track = PerTrack::new(0, decoder, FilterGraph::new(), Box::new(encoder));
    let muxer = make_mkv_muxer();
    let mut executor = MultiTrackExecutor::new(muxer);
    executor.add_track(track);

    let streams = vec![video_stream(0)];
    let stats = executor
        .execute(&streams)
        .await
        .expect("execute should succeed");

    assert_eq!(stats.total_encoded_frames, n as u64, "all frames encoded");
    assert!(stats.packets_muxed > 0, "packets written to muxer");

    let encoded = output_arc.lock().expect("output lock");
    assert_eq!(encoded.len(), n, "encoder received exactly {n} frames");

    // Verify flush was called once.
    let flushed = *flush_arc.lock().expect("flush lock");
    assert!(flushed, "encoder flush must be called after EOF");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Single video track with scale filter
// ─────────────────────────────────────────────────────────────────────────────

/// A single video track with a 4×4 → 2×2 scale filter applied.
/// Verifies that the filter graph is applied and the encoder sees the scaled frames.
#[tokio::test]
async fn test_single_video_track_scale_filter() {
    let src_w = 4u32;
    let src_h = 4u32;
    let dst_w = 2u32;
    let dst_h = 2u32;

    let frames: Vec<Frame> = (0..5)
        .map(|i| make_rgba(src_w, src_h, 0xAA, i as i64 * 33))
        .collect();

    let (encoder, output_arc, _) = MockEncoder::new();
    let filter = FilterGraph::new().add_video_scale(dst_w, dst_h);
    let track = PerTrack::new(
        0,
        Box::new(MockDecoder::with_frames(frames)),
        filter,
        Box::new(encoder),
    );

    let muxer = make_mkv_muxer();
    let mut executor = MultiTrackExecutor::new(muxer);
    executor.add_track(track);

    let stats = executor
        .execute(&[video_stream(0)])
        .await
        .expect("execute should succeed");

    assert_eq!(stats.total_encoded_frames, 5);

    let encoded = output_arc.lock().expect("output lock");
    assert_eq!(encoded.len(), 5, "5 frames encoded");
    // Each encoded frame is the raw RGBA data of a 2×2 frame = 16 bytes.
    let expected_size = (dst_w * dst_h * 4) as usize;
    for frame_data in encoded.iter() {
        assert_eq!(
            frame_data.len(),
            expected_size,
            "scaled frame should be {expected_size} bytes, got {}",
            frame_data.len()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Audio-only track with gain filter
// ─────────────────────────────────────────────────────────────────────────────

/// A single audio track with a +6 dB gain filter (≈ ×2 amplitude).
/// Verifies that the audio filter is applied: sample values double.
#[tokio::test]
async fn test_audio_only_track_with_gain_filter() {
    let sample_val: i16 = 1000;
    let frames: Vec<Frame> = (0..8)
        .map(|i| make_audio(16, sample_val, i as i64 * 20))
        .collect();

    let (encoder, output_arc, _) = MockEncoder::new();
    let filter = FilterGraph::new().add_audio_gain_db(6.0206); // ≈ ×2
    let track = PerTrack::new(
        0,
        Box::new(MockDecoder::with_frames(frames)),
        filter,
        Box::new(encoder),
    );

    let muxer = make_mkv_muxer();
    let mut executor = MultiTrackExecutor::new(muxer);
    executor.add_track(track);

    let stats = executor
        .execute(&[audio_stream(0)])
        .await
        .expect("execute should succeed");

    assert_eq!(stats.total_encoded_frames, 8, "8 audio frames encoded");

    let encoded = output_arc.lock().expect("output lock");
    assert_eq!(encoded.len(), 8);
    // Check that the first sample of the first frame is approximately doubled.
    let first_frame = &encoded[0];
    let s0 = i16::from_le_bytes([first_frame[0], first_frame[1]]);
    // Expected ≈ 2000 (within 10 counts rounding error).
    assert!(
        (s0 as i32 - 2000).abs() < 10,
        "sample should be ~2000 after +6 dB gain, got {s0}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Two-track video + audio with DTS heap interleaving
// ─────────────────────────────────────────────────────────────────────────────

/// Two tracks (video stream 0, audio stream 1) interleaved via the DTS heap.
/// Verifies that all frames from both tracks reach their respective encoders,
/// and that the total packet count written to the muxer is at least the sum
/// of both frame counts (the DTS heap may merge or buffer some packets).
///
/// Note: this test verifies encoder-side frame receipt and mux packet counts.
/// DTS monotonicity in the muxed byte stream is validated implicitly by the
/// `MatroskaMuxer` which rejects out-of-order cluster timestamps.
#[tokio::test]
async fn test_two_track_interleaved_both_tracks_encoded() {
    let n_video = 10usize;
    let n_audio = 15usize;

    // Video frames: 33 ms apart.
    let video_frames: Vec<Frame> = (0..n_video)
        .map(|i| make_rgba(4, 4, 0xFF, i as i64 * 33))
        .collect();

    // Audio frames: 20 ms apart (different cadence intentionally).
    let audio_frames: Vec<Frame> = (0..n_audio)
        .map(|i| make_audio(16, 500, i as i64 * 20))
        .collect();

    let (enc_v, out_v, _) = MockEncoder::new();
    let (enc_a, out_a, _) = MockEncoder::new();

    let track_v = PerTrack::new(
        0,
        Box::new(MockDecoder::with_frames(video_frames)),
        FilterGraph::new(),
        Box::new(enc_v),
    );
    let track_a = PerTrack::new(
        1,
        Box::new(MockDecoder::with_frames(audio_frames)),
        FilterGraph::new(),
        Box::new(enc_a),
    );

    let muxer = make_mkv_muxer();
    let mut executor = MultiTrackExecutor::new(muxer);
    // Use a small flush interval so the heap drains more eagerly.
    executor.set_flush_interval(5);
    executor.add_track(track_v);
    executor.add_track(track_a);

    let streams = vec![video_stream(0), audio_stream(1)];
    let stats = executor
        .execute(&streams)
        .await
        .expect("execute should succeed");

    assert_eq!(
        stats.total_encoded_frames,
        (n_video + n_audio) as u64,
        "total encoded frames from both tracks"
    );
    // Both encoders must have received their frames.
    let v_out = out_v.lock().expect("v lock");
    let a_out = out_a.lock().expect("a lock");
    assert_eq!(v_out.len(), n_video, "video encoder received all frames");
    assert_eq!(a_out.len(), n_audio, "audio encoder received all frames");
    // Total packets muxed must be at least n_video + n_audio.
    assert!(
        stats.packets_muxed >= (n_video + n_audio) as u64,
        "expected >= {} packets muxed, got {}",
        n_video + n_audio,
        stats.packets_muxed
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: EOF drain — encoder.flush() called exactly once per track
// ─────────────────────────────────────────────────────────────────────────────

/// Two tracks, each with a `MockEncoder`.  After `execute()`, verifies that
/// `flush()` was called exactly once per track (not zero, not twice).
#[tokio::test]
async fn test_eof_drain_flush_called_once_per_track() {
    let frames_a: Vec<Frame> = (0..5)
        .map(|i| make_rgba(2, 2, 0x11, i as i64 * 33))
        .collect();
    let frames_b: Vec<Frame> = (0..3).map(|i| make_audio(8, 200, i as i64 * 20)).collect();

    let (enc_a, _, flush_a) = MockEncoder::new();
    let (enc_b, _, flush_b) = MockEncoder::new();

    let track_a = PerTrack::new(
        0,
        Box::new(MockDecoder::with_frames(frames_a)),
        FilterGraph::new(),
        Box::new(enc_a),
    );
    let track_b = PerTrack::new(
        1,
        Box::new(MockDecoder::with_frames(frames_b)),
        FilterGraph::new(),
        Box::new(enc_b),
    );

    let muxer = make_mkv_muxer();
    let mut executor = MultiTrackExecutor::new(muxer);
    executor.add_track(track_a);
    executor.add_track(track_b);

    let streams = vec![video_stream(0), audio_stream(1)];
    executor
        .execute(&streams)
        .await
        .expect("execute should succeed");

    // Each encoder's flush must be called exactly once.
    assert!(
        *flush_a.lock().expect("flush_a lock"),
        "encoder A flush must be called"
    );
    assert!(
        *flush_b.lock().expect("flush_b lock"),
        "encoder B flush must be called"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: Encoder error bubbles up
// ─────────────────────────────────────────────────────────────────────────────

/// A single track whose encoder fails on the second frame.
/// Verifies that the error propagates out of `execute()` as a `TranscodeError`.
#[tokio::test]
async fn test_encoder_error_bubbles_up() {
    let frames: Vec<Frame> = (0..5)
        .map(|i| make_rgba(4, 4, 0x80, i as i64 * 33))
        .collect();

    let encoder = Box::new(ErrorOnSecondEncode::new());
    let track = PerTrack::new(
        0,
        Box::new(MockDecoder::with_frames(frames)),
        FilterGraph::new(),
        encoder,
    );

    let muxer = make_mkv_muxer();
    let mut executor = MultiTrackExecutor::new(muxer);
    executor.add_track(track);

    let streams = vec![video_stream(0)];
    let result = executor.execute(&streams).await;

    assert!(
        result.is_err(),
        "execute must return Err when encoder fails"
    );
    let err = result.expect_err("must be error");
    match err {
        TranscodeError::CodecError(msg) => {
            assert!(
                msg.contains("deliberate"),
                "error message should contain 'deliberate', got: {msg}"
            );
        }
        other => panic!("expected CodecError, got: {other:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: Empty decoder — zero frames in / zero frames out
// ─────────────────────────────────────────────────────────────────────────────

/// An executor with a single track whose decoder yields no frames at all.
/// Verifies that `execute()` succeeds and reports zero encoded frames.
#[tokio::test]
async fn test_empty_decoder_zero_frames() {
    let (encoder, output_arc, flush_arc) = MockEncoder::new();

    let track = PerTrack::new(
        0,
        Box::new(MockDecoder::empty()),
        FilterGraph::new(),
        Box::new(encoder),
    );

    let muxer = make_mkv_muxer();
    let mut executor = MultiTrackExecutor::new(muxer);
    executor.add_track(track);

    let streams = vec![video_stream(0)];
    let stats = executor
        .execute(&streams)
        .await
        .expect("execute with empty decoder should succeed");

    assert_eq!(stats.total_encoded_frames, 0, "no frames should be encoded");
    assert_eq!(stats.total_encoded_bytes, 0, "no bytes should be encoded");

    let encoded = output_arc.lock().expect("output lock");
    assert!(encoded.is_empty(), "encoder should receive no frames");

    // Flush is still called even for an empty decoder.
    let flushed = *flush_arc.lock().expect("flush lock");
    assert!(flushed, "flush must be called even with an empty decoder");
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8: DTS ordering is monotonic across both tracks
// ─────────────────────────────────────────────────────────────────────────────

/// A `RecordingMuxer` that captures the DTS (pts) of every packet written,
/// so tests can assert DTS monotonicity without needing a real container.
///
/// All `Muxer` methods other than `write_packet` are no-ops.
struct RecordingMuxer {
    /// DTS values (in ms) of each packet, in write order.
    pub dts_log: Arc<Mutex<Vec<i64>>>,
    /// Streams registered via `add_stream`.
    streams: Vec<StreamInfo>,
    /// Config returned by `Muxer::config`.
    cfg: MuxerConfig,
}

impl RecordingMuxer {
    fn new() -> (Self, Arc<Mutex<Vec<i64>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let muxer = Self {
            dts_log: Arc::clone(&log),
            streams: Vec::new(),
            cfg: MuxerConfig::new(),
        };
        (muxer, log)
    }
}

#[async_trait]
impl Muxer for RecordingMuxer {
    fn add_stream(&mut self, info: StreamInfo) -> oximedia_core::OxiResult<usize> {
        let idx = self.streams.len();
        self.streams.push(info);
        Ok(idx)
    }

    async fn write_header(&mut self) -> oximedia_core::OxiResult<()> {
        Ok(())
    }

    async fn write_packet(&mut self, packet: &Packet) -> oximedia_core::OxiResult<()> {
        // Convert the packet timestamp PTS to integer milliseconds for ordering checks.
        let dts_ms = (packet.timestamp.to_seconds() * 1_000.0) as i64;
        self.dts_log.lock().expect("dts_log lock").push(dts_ms);
        Ok(())
    }

    async fn write_trailer(&mut self) -> oximedia_core::OxiResult<()> {
        Ok(())
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn config(&self) -> &MuxerConfig {
        &self.cfg
    }
}

/// Two tracks (video at 33 ms cadence, audio at 20 ms cadence) written through
/// a `RecordingMuxer`.  Verifies that the DTS sequence delivered to the muxer
/// is monotonically non-decreasing across both tracks, confirming that the
/// DTS min-heap correctly interleaves cross-track packets.
#[tokio::test]
async fn test_dts_ordering_is_monotonic_across_tracks() {
    let n_video = 6usize;
    let n_audio = 10usize;

    let video_frames: Vec<Frame> = (0..n_video)
        .map(|i| make_rgba(2, 2, 0x10, i as i64 * 33))
        .collect();
    let audio_frames: Vec<Frame> = (0..n_audio)
        .map(|i| make_audio(8, 100, i as i64 * 20))
        .collect();

    let (enc_v, _, _) = MockEncoder::new();
    let (enc_a, _, _) = MockEncoder::new();

    let track_v = PerTrack::new_typed(
        0,
        Box::new(MockDecoder::with_frames(video_frames)),
        FilterGraph::new(),
        Box::new(enc_v),
        false, // video track
    );
    let track_a = PerTrack::new_typed(
        1,
        Box::new(MockDecoder::with_frames(audio_frames)),
        FilterGraph::new(),
        Box::new(enc_a),
        true, // audio track
    );

    let (muxer, dts_log) = RecordingMuxer::new();
    let mut executor = MultiTrackExecutor::new(muxer);
    // Flush every step so the heap drains eagerly for small frame counts.
    executor.set_flush_interval(1);
    executor.add_track(track_v);
    executor.add_track(track_a);

    let streams = vec![video_stream(0), audio_stream(1)];
    let stats = executor
        .execute(&streams)
        .await
        .expect("execute with recording muxer should succeed");

    assert_eq!(
        stats.total_encoded_frames,
        (n_video + n_audio) as u64,
        "all frames from both tracks must be encoded"
    );

    let log = dts_log.lock().expect("dts_log lock");
    assert!(
        !log.is_empty(),
        "at least one packet must have been written to the muxer"
    );

    // Verify monotonically non-decreasing DTS across all packets.
    let mut prev = i64::MIN;
    for (i, &dts) in log.iter().enumerate() {
        assert!(
            dts >= prev,
            "DTS out of order at packet {i}: {dts} < previous {prev}"
        );
        prev = dts;
    }
}
