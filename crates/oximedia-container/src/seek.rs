//! Seeking infrastructure for demuxers.
//!
//! This module provides types and utilities for seeking in media containers,
//! including keyframe-based seeking, sample-accurate seeking, and a seek index
//! for fast random access.
//!
//! # Sample-Accurate Seeking
//!
//! Sample-accurate seeking goes beyond simple keyframe-based seeking by:
//! 1. Finding the nearest keyframe before the target
//! 2. Tracking samples between that keyframe and the target
//! 3. Providing a `SeekPlan` that tells the decoder which samples to decode
//!    and which to discard
//!
//! ```ignore
//! let index = SeekIndex::new(90000); // 90kHz timescale
//! // ... populate with sample entries ...
//! let plan = index.plan_seek(target_pts, SeekAccuracy::SampleAccurate)?;
//! // plan.decode_from_pts: start decoding here (keyframe)
//! // plan.discard_count: number of frames to decode but discard
//! // plan.target_pts: the actual target presentation time
//! ```

use bitflags::bitflags;
use std::cmp::Ordering;
use std::collections::HashMap;

bitflags! {
    /// Flags controlling seek behavior.
    ///
    /// These flags allow fine-grained control over how a seek operation
    /// is performed and what position is targeted.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
    pub struct SeekFlags: u32 {
        /// Seek backward (to position <= target).
        ///
        /// Without this flag, seeks go forward (to position >= target).
        /// This is useful for finding the keyframe before a target position.
        const BACKWARD = 0x0001;

        /// Allow seeking to any frame, not just keyframes.
        ///
        /// By default, seeks target keyframes for clean decoding.
        /// Setting this flag allows seeking to any position, which may
        /// require decoding from the previous keyframe.
        const ANY = 0x0002;

        /// Seek to the nearest keyframe.
        ///
        /// This is the default behavior and ensures the seek position
        /// can be decoded immediately without reference frames.
        const KEYFRAME = 0x0004;

        /// Seek by bytes rather than time.
        ///
        /// When set, the seek target is interpreted as a byte offset
        /// in the file rather than a timestamp.
        const BYTE = 0x0008;

        /// Seek to exact position (frame-accurate).
        ///
        /// Attempts to seek to the exact target timestamp, which may
        /// require additional parsing and decoding.
        const FRAME_ACCURATE = 0x0010;
    }
}

/// Target for a seek operation.
///
/// Specifies where to seek and which stream to use as reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeekTarget {
    /// Target timestamp in seconds, or byte offset if `SeekFlags::BYTE` is set.
    pub position: f64,

    /// Stream index to use for seeking, or `None` for the default stream.
    ///
    /// The default stream is typically the first video stream, or the
    /// first audio stream if there are no video streams.
    pub stream_index: Option<usize>,

    /// Seek flags controlling behavior.
    pub flags: SeekFlags,
}

impl SeekTarget {
    /// Creates a new seek target to a timestamp in seconds.
    ///
    /// # Arguments
    ///
    /// * `position` - Target timestamp in seconds
    #[must_use]
    pub const fn time(position: f64) -> Self {
        Self {
            position,
            stream_index: None,
            flags: SeekFlags::KEYFRAME,
        }
    }

    /// Creates a new seek target to a byte offset.
    ///
    /// # Arguments
    ///
    /// * `offset` - Target byte offset in the file
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn byte(offset: u64) -> Self {
        Self {
            position: offset as f64,
            stream_index: None,
            flags: SeekFlags::BYTE,
        }
    }

    /// Creates a sample-accurate seek target.
    ///
    /// This will seek to the exact position, decoding from the
    /// preceding keyframe and discarding intermediate frames.
    #[must_use]
    pub const fn sample_accurate(position: f64) -> Self {
        Self {
            position,
            stream_index: None,
            flags: SeekFlags::from_bits_truncate(
                SeekFlags::FRAME_ACCURATE.bits() | SeekFlags::BACKWARD.bits(),
            ),
        }
    }

    /// Sets the stream index for this seek target.
    #[must_use]
    pub const fn with_stream(mut self, stream_index: usize) -> Self {
        self.stream_index = Some(stream_index);
        self
    }

    /// Sets the seek flags for this seek target.
    #[must_use]
    pub const fn with_flags(mut self, flags: SeekFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Adds additional flags to this seek target.
    #[must_use]
    pub const fn add_flags(mut self, flags: SeekFlags) -> Self {
        self.flags = SeekFlags::from_bits_truncate(self.flags.bits() | flags.bits());
        self
    }

    /// Returns true if this is a backward seek.
    #[must_use]
    pub const fn is_backward(&self) -> bool {
        self.flags.contains(SeekFlags::BACKWARD)
    }

    /// Returns true if this allows seeking to any frame.
    #[must_use]
    pub const fn is_any(&self) -> bool {
        self.flags.contains(SeekFlags::ANY)
    }

    /// Returns true if this seeks to a keyframe.
    #[must_use]
    pub const fn is_keyframe(&self) -> bool {
        self.flags.contains(SeekFlags::KEYFRAME)
    }

    /// Returns true if this is a byte-based seek.
    #[must_use]
    pub const fn is_byte(&self) -> bool {
        self.flags.contains(SeekFlags::BYTE)
    }

    /// Returns true if this is a frame-accurate seek.
    #[must_use]
    pub const fn is_frame_accurate(&self) -> bool {
        self.flags.contains(SeekFlags::FRAME_ACCURATE)
    }
}

// ─── Seek Accuracy ──────────────────────────────────────────────────────────

/// Desired accuracy level for seeking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekAccuracy {
    /// Seek to the nearest keyframe (fastest, least accurate).
    Keyframe,
    /// Seek to the exact sample/frame (requires decoding from prior keyframe).
    SampleAccurate,
    /// Seek to within a specified tolerance in timescale ticks.
    WithinTolerance(u64),
}

// ─── Seek Index Entry ───────────────────────────────────────────────────────

/// An entry in the seek index representing one sample/frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeekIndexEntry {
    /// Presentation timestamp in timescale ticks.
    pub pts: i64,
    /// Decode timestamp in timescale ticks.
    pub dts: i64,
    /// Byte offset in the container file.
    pub file_offset: u64,
    /// Sample size in bytes.
    pub size: u32,
    /// Sample duration in timescale ticks.
    pub duration: u32,
    /// Whether this is a keyframe (sync sample).
    pub is_keyframe: bool,
    /// Sample number (0-based).
    pub sample_number: u32,
}

impl SeekIndexEntry {
    /// Creates a new keyframe entry.
    #[must_use]
    pub const fn keyframe(
        pts: i64,
        dts: i64,
        file_offset: u64,
        size: u32,
        duration: u32,
        sample_number: u32,
    ) -> Self {
        Self {
            pts,
            dts,
            file_offset,
            size,
            duration,
            is_keyframe: true,
            sample_number,
        }
    }

    /// Creates a new non-keyframe entry.
    #[must_use]
    pub const fn non_keyframe(
        pts: i64,
        dts: i64,
        file_offset: u64,
        size: u32,
        duration: u32,
        sample_number: u32,
    ) -> Self {
        Self {
            pts,
            dts,
            file_offset,
            size,
            duration,
            is_keyframe: false,
            sample_number,
        }
    }

    /// Returns the end PTS (pts + duration).
    #[must_use]
    pub const fn end_pts(&self) -> i64 {
        self.pts + self.duration as i64
    }
}

// ─── Seek Plan ──────────────────────────────────────────────────────────────

/// A plan for executing a sample-accurate seek.
///
/// Contains information about where to start decoding (keyframe),
/// how many samples to decode-and-discard, and the final target sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeekPlan {
    /// The keyframe entry to seek to in the container (start decoding here).
    pub keyframe_entry: SeekIndexEntry,
    /// Number of samples between the keyframe and the target to decode
    /// but discard (not presented).
    pub discard_count: u32,
    /// The target sample entry.
    pub target_entry: SeekIndexEntry,
    /// File offset to seek to.
    pub file_offset: u64,
    /// The target PTS that was requested.
    pub requested_pts: i64,
    /// Whether the seek is exact (target PTS matches a sample boundary).
    pub is_exact: bool,
}

// ─── Seek Index ─────────────────────────────────────────────────────────────

/// Index of sample positions for fast seeking.
///
/// Maintains a sorted list of sample entries that enables both keyframe-based
/// and sample-accurate seeking. Entries are sorted by DTS for efficient
/// binary search.
#[derive(Debug, Clone)]
pub struct SeekIndex {
    /// Timescale (ticks per second) for interpreting timestamps.
    timescale: u32,
    /// All sample entries sorted by DTS.
    entries: Vec<SeekIndexEntry>,
    /// Indices of keyframe entries within `entries` (for fast keyframe lookup).
    keyframe_indices: Vec<usize>,
}

impl SeekIndex {
    /// Creates a new empty seek index.
    #[must_use]
    pub fn new(timescale: u32) -> Self {
        Self {
            timescale,
            entries: Vec::new(),
            keyframe_indices: Vec::new(),
        }
    }

    /// Creates a seek index with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(timescale: u32, capacity: usize) -> Self {
        Self {
            timescale,
            entries: Vec::with_capacity(capacity),
            keyframe_indices: Vec::new(),
        }
    }

    /// Returns the timescale.
    #[must_use]
    pub const fn timescale(&self) -> u32 {
        self.timescale
    }

    /// Returns the number of entries in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the number of keyframes in the index.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.keyframe_indices.len()
    }

    /// Returns all entries.
    #[must_use]
    pub fn entries(&self) -> &[SeekIndexEntry] {
        &self.entries
    }

    /// Adds a sample entry to the index.
    ///
    /// Entries should be added in DTS order for optimal performance.
    /// If entries are added out of order, call [`sort`](SeekIndex::sort)
    /// before seeking.
    pub fn add_entry(&mut self, entry: SeekIndexEntry) {
        let idx = self.entries.len();
        if entry.is_keyframe {
            self.keyframe_indices.push(idx);
        }
        self.entries.push(entry);
    }

    /// Sorts entries by DTS and rebuilds the keyframe index.
    pub fn sort(&mut self) {
        self.entries.sort_by_key(|e| e.dts);
        self.keyframe_indices.clear();
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.is_keyframe {
                self.keyframe_indices.push(i);
            }
        }
    }

    /// Finalizes the index after all entries have been added.
    ///
    /// Equivalent to [`SeekIndex::sort`]: sorts all entries by DTS and
    /// rebuilds the keyframe lookup table.  Call this once after all calls
    /// to [`SeekIndex::add_entry`] are complete.
    pub fn finalize(&mut self) {
        self.sort();
    }

    /// Converts a time in seconds to ticks in this index's timescale.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn seconds_to_ticks(&self, seconds: f64) -> i64 {
        (seconds * f64::from(self.timescale)) as i64
    }

    /// Converts ticks to seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn ticks_to_seconds(&self, ticks: i64) -> f64 {
        if self.timescale == 0 {
            return 0.0;
        }
        ticks as f64 / f64::from(self.timescale)
    }

    /// Finds the nearest keyframe at or before the given PTS.
    ///
    /// Returns `None` if the index is empty or has no keyframes.
    #[must_use]
    pub fn find_keyframe_before(&self, target_pts: i64) -> Option<&SeekIndexEntry> {
        if self.keyframe_indices.is_empty() {
            return None;
        }

        // Binary search for the last keyframe with pts <= target_pts
        let mut best: Option<usize> = None;
        let mut lo = 0usize;
        let mut hi = self.keyframe_indices.len();

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let kf_idx = self.keyframe_indices[mid];
            let kf = &self.entries[kf_idx];

            match kf.pts.cmp(&target_pts) {
                Ordering::Less | Ordering::Equal => {
                    best = Some(kf_idx);
                    lo = mid + 1;
                }
                Ordering::Greater => {
                    hi = mid;
                }
            }
        }

        best.map(|idx| &self.entries[idx])
    }

    /// Finds the nearest keyframe at or after the given PTS.
    ///
    /// Returns `None` if no keyframe exists at or after the target.
    #[must_use]
    pub fn find_keyframe_after(&self, target_pts: i64) -> Option<&SeekIndexEntry> {
        if self.keyframe_indices.is_empty() {
            return None;
        }

        let mut lo = 0usize;
        let mut hi = self.keyframe_indices.len();

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let kf_idx = self.keyframe_indices[mid];
            let kf = &self.entries[kf_idx];

            if kf.pts < target_pts {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        if lo < self.keyframe_indices.len() {
            Some(&self.entries[self.keyframe_indices[lo]])
        } else {
            None
        }
    }

    /// Finds the nearest keyframe (before or after) to the given PTS.
    ///
    /// Returns whichever keyframe is closer in PTS distance.
    #[must_use]
    pub fn find_nearest_keyframe(&self, target_pts: i64) -> Option<&SeekIndexEntry> {
        let before = self.find_keyframe_before(target_pts);
        let after = self.find_keyframe_after(target_pts);

        match (before, after) {
            (None, None) => None,
            (Some(b), None) => Some(b),
            (None, Some(a)) => Some(a),
            (Some(b), Some(a)) => {
                let dist_before = (target_pts - b.pts).unsigned_abs();
                let dist_after = (a.pts - target_pts).unsigned_abs();
                if dist_before <= dist_after {
                    Some(b)
                } else {
                    Some(a)
                }
            }
        }
    }

    /// Finds the exact sample entry whose PTS range contains the target.
    ///
    /// Returns `None` if no sample covers the target PTS.
    #[must_use]
    pub fn find_sample_at(&self, target_pts: i64) -> Option<&SeekIndexEntry> {
        // Binary search for the sample whose pts <= target_pts < pts+duration
        let result = self.entries.binary_search_by(|entry| {
            if entry.pts > target_pts {
                Ordering::Greater
            } else if entry.end_pts() <= target_pts {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        });

        match result {
            Ok(idx) => Some(&self.entries[idx]),
            Err(_) => {
                // Fallback: find the last entry with pts <= target_pts
                let mut best = None;
                for entry in &self.entries {
                    if entry.pts <= target_pts {
                        best = Some(entry);
                    } else {
                        break;
                    }
                }
                best
            }
        }
    }

    /// Plans a sample-accurate seek to the given PTS.
    ///
    /// Returns a `SeekPlan` describing how to execute the seek:
    /// - Which keyframe to seek to in the container
    /// - How many samples to decode and discard
    /// - The target sample
    ///
    /// # Errors
    ///
    /// Returns `None` if the index is empty or the target is out of range.
    #[must_use]
    pub fn plan_seek(&self, target_pts: i64, accuracy: SeekAccuracy) -> Option<SeekPlan> {
        if self.entries.is_empty() || self.keyframe_indices.is_empty() {
            return None;
        }

        match accuracy {
            SeekAccuracy::Keyframe => {
                let kf = self.find_keyframe_before(target_pts)?;
                Some(SeekPlan {
                    keyframe_entry: *kf,
                    discard_count: 0,
                    target_entry: *kf,
                    file_offset: kf.file_offset,
                    requested_pts: target_pts,
                    is_exact: kf.pts == target_pts,
                })
            }
            SeekAccuracy::SampleAccurate => self.plan_sample_accurate_seek(target_pts),
            SeekAccuracy::WithinTolerance(tolerance) => {
                // First try exact, fall back to keyframe if within tolerance
                if let Some(plan) = self.plan_sample_accurate_seek(target_pts) {
                    let distance = (plan.target_entry.pts - target_pts).unsigned_abs();
                    if distance <= tolerance {
                        return Some(plan);
                    }
                }
                // Fall back to nearest keyframe
                let kf = self.find_nearest_keyframe(target_pts)?;
                let distance = (kf.pts - target_pts).unsigned_abs();
                if distance <= tolerance {
                    Some(SeekPlan {
                        keyframe_entry: *kf,
                        discard_count: 0,
                        target_entry: *kf,
                        file_offset: kf.file_offset,
                        requested_pts: target_pts,
                        is_exact: kf.pts == target_pts,
                    })
                } else {
                    None
                }
            }
        }
    }

    fn plan_sample_accurate_seek(&self, target_pts: i64) -> Option<SeekPlan> {
        // 1. Find the preceding keyframe
        let kf = self.find_keyframe_before(target_pts)?;
        let kf_copy = *kf;

        // 2. Find the target sample
        let target_sample = self.find_sample_at(target_pts);
        let target = match target_sample {
            Some(s) => *s,
            None => {
                // Target is past all samples; use the last sample
                *self.entries.last()?
            }
        };

        // 3. Count samples between keyframe and target (exclusive of keyframe, inclusive of target)
        let mut discard_count: u32 = 0;
        for entry in &self.entries {
            if entry.dts > kf_copy.dts && entry.dts < target.dts {
                discard_count += 1;
            }
        }

        Some(SeekPlan {
            keyframe_entry: kf_copy,
            discard_count,
            target_entry: target,
            file_offset: kf_copy.file_offset,
            requested_pts: target_pts,
            is_exact: target.pts <= target_pts && target_pts < target.end_pts(),
        })
    }

    /// Returns the duration of the indexed content in timescale ticks.
    #[must_use]
    pub fn duration_ticks(&self) -> i64 {
        self.entries.last().map_or(0, |e| e.pts + e.duration as i64)
    }

    /// Returns the duration of the indexed content in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        self.ticks_to_seconds(self.duration_ticks())
    }

    /// Returns the average keyframe interval in timescale ticks.
    #[must_use]
    pub fn average_keyframe_interval(&self) -> Option<f64> {
        if self.keyframe_indices.len() < 2 {
            return None;
        }

        let mut total_interval: i64 = 0;
        for i in 1..self.keyframe_indices.len() {
            let prev = &self.entries[self.keyframe_indices[i - 1]];
            let curr = &self.entries[self.keyframe_indices[i]];
            total_interval += curr.pts - prev.pts;
        }

        #[allow(clippy::cast_precision_loss)]
        let avg = total_interval as f64 / (self.keyframe_indices.len() - 1) as f64;
        Some(avg)
    }
}

/// Type alias for [`SeekIndex`] used in pre-roll seeking contexts.
///
/// `SampleIndex` is the same type as `SeekIndex`; the alias provides a
/// more descriptive name when building per-stream sample tables for
/// pre-roll seek planning.
pub type SampleIndex = SeekIndex;

// ─── TrackIndex ─────────────────────────────────────────────────────────────

/// A lightweight index of keyframe positions within a single track.
///
/// Used by [`SampleAccurateSeeker`] to locate the nearest keyframe before a
/// target PTS and compute the number of samples that must be decoded and
/// discarded to reach a sample-accurate position.
#[derive(Debug, Clone)]
pub struct TrackIndex {
    /// The underlying seek index (sorted by DTS).
    pub seek_index: SeekIndex,
    /// Codec delay in samples (e.g. 512 for Opus, 0 for most video codecs).
    /// Added to the `preroll_samples` field of the returned [`SeekResult`].
    pub codec_delay_samples: u32,
}

impl TrackIndex {
    /// Creates a `TrackIndex` from an existing [`SeekIndex`].
    #[must_use]
    pub fn new(seek_index: SeekIndex) -> Self {
        Self {
            seek_index,
            codec_delay_samples: 0,
        }
    }

    /// Creates a `TrackIndex` with an explicit codec delay.
    #[must_use]
    pub fn with_codec_delay(seek_index: SeekIndex, codec_delay_samples: u32) -> Self {
        Self {
            seek_index,
            codec_delay_samples,
        }
    }
}

// ─── SeekResult ─────────────────────────────────────────────────────────────

/// The result of a sample-accurate seek operation.
///
/// Returned by [`SampleAccurateSeeker::seek_to_sample`] when a suitable
/// keyframe is found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeekResult {
    /// The PTS of the keyframe that the decoder must start from.
    ///
    /// The container should be seeked to the file offset corresponding to
    /// this keyframe before decoding begins.
    pub keyframe_pts: u64,
    /// Byte offset of the keyframe in the container file.
    pub sample_offset: u64,
    /// Number of samples to decode and discard between `keyframe_pts` and the
    /// target PTS, plus any codec delay.
    ///
    /// A value of 0 means the seek landed exactly on a keyframe boundary.
    pub preroll_samples: u32,
}

/// A shared decode-and-skip cursor for sample-accurate seek planning.
///
/// Demuxers can return this when the caller must seek to `byte_offset`,
/// begin decoding at `sample_index`, and discard `skip_samples` decoded
/// samples before presenting the first target sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeSkipCursor {
    /// Byte offset of the keyframe/sample where decoding should begin.
    pub byte_offset: u64,
    /// 0-based sample index where decoding should begin.
    pub sample_index: usize,
    /// Number of decoded samples to discard before presentation.
    pub skip_samples: u32,
    /// Requested target presentation timestamp in track timescale units.
    pub target_pts: i64,
}

// ─── SampleAccurateSeeker ───────────────────────────────────────────────────

/// Performs sample-accurate seeking on a single media track.
///
/// Wraps a [`TrackIndex`] and provides a high-level `seek_to_sample` method
/// that finds the nearest keyframe at or before a target PTS, returning the
/// exact byte offset and the number of samples that must be decoded and
/// discarded to reach the target position.
///
/// # Example
///
/// ```ignore
/// use oximedia_container::seek::{SeekIndex, SeekIndexEntry, TrackIndex, SampleAccurateSeeker};
///
/// let mut index = SeekIndex::new(90000);
/// // … populate index …
/// let track = TrackIndex::new(index);
/// let seeker = SampleAccurateSeeker::with_track(track);
///
/// let result = seeker.seek_to_sample(45000, &seeker.track).expect("seek should succeed");
/// println!("Seek to file offset {}", result.sample_offset);
/// println!("Discard {} samples after decoding", result.preroll_samples);
/// ```
pub struct SampleAccurateSeeker {
    /// The primary track index used for single-track seek operations.
    ///
    /// This field is set when constructed with [`SampleAccurateSeeker::with_track`].
    /// It holds a default empty `TrackIndex` when constructed with
    /// [`SampleAccurateSeeker::new`] (multi-stream mode).
    pub track: TrackIndex,
    /// Per-stream sample indices for multi-stream pre-roll seeking.
    ///
    /// Keys are stream IDs (typically matching `StreamInfo.index`).
    streams: HashMap<u32, SampleIndex>,
}

impl SampleAccurateSeeker {
    /// Creates a new multi-stream `SampleAccurateSeeker` with no pre-loaded
    /// streams.
    ///
    /// Use [`SampleAccurateSeeker::add_stream`] to register stream indices,
    /// then call [`SampleAccurateSeeker::plan_preroll_seek`] to plan seeks.
    #[must_use]
    pub fn new() -> Self {
        let empty_index = SeekIndex::new(90_000);
        let empty_track = TrackIndex::new(empty_index);
        Self {
            track: empty_track,
            streams: HashMap::new(),
        }
    }

    /// Creates a `SampleAccurateSeeker` from a single pre-built [`TrackIndex`].
    ///
    /// This is the single-track constructor.  Use `new()` instead for
    /// multi-stream workflows.
    #[must_use]
    pub fn with_track(track: TrackIndex) -> Self {
        Self {
            track,
            streams: HashMap::new(),
        }
    }

    /// Registers a per-stream `SampleIndex` for multi-stream pre-roll seeking.
    ///
    /// The `stream_id` must match the stream identifier used when calling
    /// [`SampleAccurateSeeker::plan_preroll_seek`].
    pub fn add_stream(&mut self, stream_id: u32, index: SampleIndex) {
        self.streams.insert(stream_id, index);
    }

    /// Plans a pre-roll seek for `stream_id` to `target_pts`.
    ///
    /// Returns a [`crate::preroll::PreRollSeekPlan`] describing the keyframe to
    /// start decoding from and the chain of samples to decode (some discarded,
    /// some presented) in order to reach `target_pts` sample-accurately.
    ///
    /// `max_preroll` optionally limits the maximum number of decode-and-discard
    /// samples.  When the distance from keyframe to target exceeds this limit the
    /// chain is truncated.
    ///
    /// Returns `None` if `stream_id` has not been registered or the index has
    /// no keyframe at or before `target_pts`.
    #[must_use]
    pub fn plan_preroll_seek(
        &self,
        stream_id: u32,
        target_pts: i64,
        max_preroll: Option<u32>,
    ) -> Option<crate::preroll::PreRollSeekPlan> {
        use crate::preroll::{PreRollAction, PreRollSample, PreRollSeekPlan};

        let index = self.streams.get(&stream_id)?;
        let keyframe = index.find_keyframe_before(target_pts)?;

        // Collect samples from the keyframe onwards.
        // - Before target_pts: decode-and-discard (up to max_preroll).
        // - At target_pts or after: present (stop after the first present sample
        //   since the caller only needs to reach the target, not process all
        //   future samples).
        let all_from_kf: Vec<&SeekIndexEntry> = index
            .entries()
            .iter()
            .filter(|e| e.pts >= keyframe.pts)
            .collect();

        // Separate discard candidates (before target) from present candidates.
        let discard_candidates: Vec<&&SeekIndexEntry> =
            all_from_kf.iter().filter(|e| e.pts < target_pts).collect();
        let present_candidate: Option<&SeekIndexEntry> =
            all_from_kf.iter().find(|e| e.pts >= target_pts).copied();

        // Apply max_preroll cap: if capped, take the LAST N discard candidates
        // (closest to target) so the decoder has the fewest samples to skip.
        let capped_discards: Vec<&SeekIndexEntry> = if let Some(max) = max_preroll {
            let max = max as usize;
            if discard_candidates.len() > max {
                discard_candidates[discard_candidates.len() - max..]
                    .iter()
                    .copied()
                    .copied()
                    .collect()
            } else {
                discard_candidates.iter().copied().copied().collect()
            }
        } else {
            discard_candidates.iter().copied().copied().collect()
        };

        let mut samples: Vec<PreRollSample> = capped_discards
            .iter()
            .map(|e| PreRollSample {
                entry: **e,
                action: PreRollAction::Decode,
            })
            .collect();

        let discard_count = samples.len() as u32;
        let mut present_count: u32 = 0;

        if let Some(entry) = present_candidate {
            samples.push(PreRollSample {
                entry: *entry,
                action: PreRollAction::Present,
            });
            present_count = 1;
        } else if discard_count == 0 {
            // Nothing found — no useful plan.
            return None;
        } else {
            // No explicit present sample found (target is beyond the index).
            // Promote the last discard to present.
            if let Some(last) = samples.last_mut() {
                last.action = PreRollAction::Present;
                present_count = 1;
            }
        }

        let final_discard_count = samples
            .iter()
            .filter(|s| matches!(s.action, PreRollAction::Decode))
            .count() as u32;

        Some(PreRollSeekPlan {
            keyframe: *keyframe,
            target_pts,
            samples,
            discard_count: final_discard_count,
            present_count,
            file_offset: keyframe.file_offset,
        })
    }

    /// Returns the number of samples that must be decoded and discarded
    /// (pre-roll count) to achieve sample-accurate positioning at `target_pts`
    /// in `stream_id`.
    ///
    /// Returns `None` if the stream is not registered or has no suitable
    /// keyframe.
    #[must_use]
    pub fn preroll_count(&self, stream_id: u32, target_pts: i64) -> Option<u32> {
        let plan = self.plan_preroll_seek(stream_id, target_pts, None)?;
        Some(plan.discard_count)
    }

    /// Seeks to the sample-accurate position for `target_pts` within `track`.
    ///
    /// The algorithm:
    /// 1. Finds the nearest keyframe at or before `target_pts` using binary
    ///    search on the track's seek index.
    /// 2. Counts every sample between the keyframe (exclusive) and the target
    ///    sample (exclusive) — these must be decoded and discarded.
    /// 3. Adds the track's `codec_delay_samples` to the discard count.
    ///
    /// # Returns
    ///
    /// `Some(SeekResult)` if a keyframe is found, or `None` if the index is
    /// empty or no keyframe exists before `target_pts`.
    ///
    /// # Arguments
    ///
    /// * `target_pts` — desired presentation timestamp in the track's timescale.
    /// * `track` — the [`TrackIndex`] to search (typically `&self.track`).
    #[must_use]
    pub fn seek_to_sample(&self, target_pts: u64, track: &TrackIndex) -> Option<SeekResult> {
        let target_i64 = i64::try_from(target_pts).unwrap_or(i64::MAX);

        let plan = track
            .seek_index
            .plan_seek(target_i64, SeekAccuracy::SampleAccurate)?;

        let keyframe_pts = u64::try_from(plan.keyframe_entry.pts.max(0)).unwrap_or(0);
        let sample_offset = plan.keyframe_entry.file_offset;
        let preroll_samples = plan.discard_count.saturating_add(track.codec_delay_samples);

        Some(SeekResult {
            keyframe_pts,
            sample_offset,
            preroll_samples,
        })
    }
}

impl Default for SampleAccurateSeeker {
    fn default() -> Self {
        Self::new()
    }
}

// ─── MultiTrackSeeker ────────────────────────────────────────────────────────

/// A compact index entry describing a single sample within a track.
///
/// Used by [`MultiTrackSeeker`] to build a per-track PTS→byte-offset index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleIndexEntry {
    /// Presentation timestamp in the track's timescale ticks.
    pub pts: i64,
    /// Byte offset of the sample within the container file.
    pub byte_offset: u64,
    /// Whether this sample is a sync (key) sample.
    pub is_sync: bool,
}

impl SampleIndexEntry {
    /// Creates a new keyframe [`SampleIndexEntry`].
    #[must_use]
    pub const fn keyframe(pts: i64, byte_offset: u64) -> Self {
        Self {
            pts,
            byte_offset,
            is_sync: true,
        }
    }

    /// Creates a new non-keyframe [`SampleIndexEntry`].
    #[must_use]
    pub const fn delta(pts: i64, byte_offset: u64) -> Self {
        Self {
            pts,
            byte_offset,
            is_sync: false,
        }
    }
}

/// The result of a [`MultiTrackSeeker::seek_to_pts`] operation.
///
/// Contains the actual sample PTS found (which may differ from the requested
/// target if the target fell between samples), the byte offset of that sample
/// in the container file, and the 0-based index of the sample within the
/// track's index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtsSeekResult {
    /// Presentation timestamp of the sample that was found (in timescale ticks).
    pub found_pts: i64,
    /// Byte offset of the found sample in the container file.
    pub byte_offset: u64,
    /// 0-based index of the sample within the track's sorted index array.
    pub sample_idx: usize,
}

/// Error type returned by [`MultiTrackSeeker`] operations.
#[derive(Debug, thiserror::Error)]
pub enum MultiTrackSeekerError {
    /// The requested track ID has not been indexed yet.
    #[error("no index for track {0}")]
    NoIndex(u32),
    /// The index for the given track is empty.
    #[error("empty index for track {0}")]
    EmptyIndex(u32),
    /// The requested PTS is before all samples in the track.
    #[error("pts {0} is before the first sample in track {1}")]
    BeforeFirstSample(i64, u32),
}

/// Multi-track sample-accurate seeker with a per-track PTS→byte-offset index.
///
/// Unlike [`SampleAccurateSeeker`] (which wraps a single [`TrackIndex`]),
/// `MultiTrackSeeker` manages a separate sorted index for each track and
/// exposes an O(log n) [`seek_to_pts`](MultiTrackSeeker::seek_to_pts) lookup.
///
/// # Example
///
/// ```ignore
/// use oximedia_container::seek::{MultiTrackSeeker, SampleIndexEntry};
///
/// let mut seeker = MultiTrackSeeker::new();
///
/// let samples = vec![
///     SampleIndexEntry::keyframe(0,     1000),
///     SampleIndexEntry::delta(3000,  1200),
///     SampleIndexEntry::keyframe(6000,  1500),
/// ];
/// seeker.build_index(1, &samples).expect("index built");
///
/// let result = seeker.seek_to_pts(1, 4500).expect("seek ok");
/// println!("found_pts={} offset={} idx={}", result.found_pts, result.byte_offset, result.sample_idx);
/// ```
pub struct MultiTrackSeeker {
    /// Per-track index: track_id → sorted `Vec<SampleIndexEntry>`.
    indices: HashMap<u32, Vec<SampleIndexEntry>>,
}

impl MultiTrackSeeker {
    /// Creates an empty [`MultiTrackSeeker`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            indices: HashMap::new(),
        }
    }

    /// Builds (or replaces) the index for `track_id` from the provided sample list.
    ///
    /// The entries are sorted by PTS so that [`seek_to_pts`](Self::seek_to_pts)
    /// can use binary search.
    ///
    /// # Errors
    ///
    /// This method currently always succeeds.  It is defined with `Result` for
    /// forward compatibility (e.g. if validation logic is added later).
    pub fn build_index(
        &mut self,
        track_id: u32,
        samples: &[SampleIndexEntry],
    ) -> Result<(), MultiTrackSeekerError> {
        let mut sorted = samples.to_vec();
        sorted.sort_unstable_by_key(|e| e.pts);
        self.indices.insert(track_id, sorted);
        Ok(())
    }

    /// Seeks to the sample-accurate position for `target_pts` within `track_id`.
    ///
    /// Uses binary search (O(log n)) to locate the last sample whose PTS is ≤
    /// `target_pts`.  If the `target_pts` is exactly a sample boundary the
    /// result is exact; otherwise the sample immediately preceding the target
    /// is returned (the decoder must decode from that point).
    ///
    /// # Errors
    ///
    /// - [`MultiTrackSeekerError::NoIndex`] — the track has no index.
    /// - [`MultiTrackSeekerError::EmptyIndex`] — the index is empty.
    /// - [`MultiTrackSeekerError::BeforeFirstSample`] — `target_pts` is earlier
    ///   than the first indexed sample.
    pub fn seek_to_pts(
        &self,
        track_id: u32,
        target_pts: i64,
    ) -> Result<PtsSeekResult, MultiTrackSeekerError> {
        let entries = self
            .indices
            .get(&track_id)
            .ok_or(MultiTrackSeekerError::NoIndex(track_id))?;

        if entries.is_empty() {
            return Err(MultiTrackSeekerError::EmptyIndex(track_id));
        }

        // Binary search: find the last entry with pts <= target_pts.
        // `partition_point` returns the index of the first element where the
        // predicate is false, i.e. the first entry with pts > target_pts.
        let insertion = entries.partition_point(|e| e.pts <= target_pts);

        if insertion == 0 {
            // All entries have pts > target_pts → before the first sample
            return Err(MultiTrackSeekerError::BeforeFirstSample(
                target_pts, track_id,
            ));
        }

        let sample_idx = insertion - 1;
        let entry = &entries[sample_idx];

        Ok(PtsSeekResult {
            found_pts: entry.pts,
            byte_offset: entry.byte_offset,
            sample_idx,
        })
    }

    /// Returns the number of tracks that have been indexed.
    #[must_use]
    pub fn indexed_track_count(&self) -> usize {
        self.indices.len()
    }

    /// Returns the number of indexed samples for `track_id`, or `None`.
    #[must_use]
    pub fn sample_count(&self, track_id: u32) -> Option<usize> {
        self.indices.get(&track_id).map(Vec::len)
    }

    /// Clears the index for `track_id`.
    pub fn clear_index(&mut self, track_id: u32) {
        self.indices.remove(&track_id);
    }

    /// Returns a sorted slice of index entries for `track_id`, or `None`.
    #[must_use]
    pub fn entries(&self, track_id: u32) -> Option<&[SampleIndexEntry]> {
        self.indices.get(&track_id).map(Vec::as_slice)
    }
}

impl Default for MultiTrackSeeker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── SeekFlags tests ─────────────────────────────────────────────────

    #[test]
    fn test_seek_flags() {
        let flags = SeekFlags::BACKWARD | SeekFlags::KEYFRAME;
        assert!(flags.contains(SeekFlags::BACKWARD));
        assert!(flags.contains(SeekFlags::KEYFRAME));
        assert!(!flags.contains(SeekFlags::ANY));
    }

    // ── SeekTarget tests ────────────────────────────────────────────────

    #[test]
    fn test_seek_target_time() {
        let target = SeekTarget::time(10.5);
        assert_eq!(target.position, 10.5);
        assert!(target.is_keyframe());
        assert!(!target.is_byte());
        assert_eq!(target.stream_index, None);
    }

    #[test]
    fn test_seek_target_byte() {
        let target = SeekTarget::byte(1024);
        assert_eq!(target.position, 1024.0);
        assert!(target.is_byte());
        assert_eq!(target.stream_index, None);
    }

    #[test]
    fn test_seek_target_sample_accurate() {
        let target = SeekTarget::sample_accurate(5.0);
        assert!(target.is_frame_accurate());
        assert!(target.is_backward());
        assert!(!target.is_keyframe());
    }

    #[test]
    fn test_seek_target_with_stream() {
        let target = SeekTarget::time(5.0).with_stream(1);
        assert_eq!(target.stream_index, Some(1));
        assert_eq!(target.position, 5.0);
    }

    #[test]
    fn test_seek_target_with_flags() {
        let target = SeekTarget::time(3.0)
            .with_flags(SeekFlags::BACKWARD)
            .add_flags(SeekFlags::ANY);

        assert!(target.is_backward());
        assert!(target.is_any());
    }

    #[test]
    fn test_seek_target_predicates() {
        let target =
            SeekTarget::time(1.0).add_flags(SeekFlags::BACKWARD | SeekFlags::FRAME_ACCURATE);

        assert!(target.is_backward());
        assert!(!target.is_any());
        assert!(target.is_keyframe());
        assert!(!target.is_byte());
        assert!(target.is_frame_accurate());
    }

    // ── SeekIndexEntry tests ────────────────────────────────────────────

    #[test]
    fn test_entry_keyframe() {
        let e = SeekIndexEntry::keyframe(0, 0, 100, 500, 3000, 0);
        assert!(e.is_keyframe);
        assert_eq!(e.pts, 0);
        assert_eq!(e.file_offset, 100);
        assert_eq!(e.end_pts(), 3000);
    }

    #[test]
    fn test_entry_non_keyframe() {
        let e = SeekIndexEntry::non_keyframe(3000, 3000, 600, 200, 3000, 1);
        assert!(!e.is_keyframe);
        assert_eq!(e.sample_number, 1);
        assert_eq!(e.end_pts(), 6000);
    }

    // ── SeekIndex basic tests ───────────────────────────────────────────

    fn build_test_index() -> SeekIndex {
        // 90kHz timescale, 30fps video (3000 ticks per frame)
        // GOP size = 5 frames (keyframe every 5th frame)
        let mut index = SeekIndex::new(90000);
        for i in 0u32..20 {
            let pts = i64::from(i) * 3000;
            let is_kf = i % 5 == 0;
            let offset = u64::from(i) * 500 + 1000; // arbitrary offsets
            if is_kf {
                index.add_entry(SeekIndexEntry::keyframe(pts, pts, offset, 500, 3000, i));
            } else {
                index.add_entry(SeekIndexEntry::non_keyframe(pts, pts, offset, 200, 3000, i));
            }
        }
        index
    }

    #[test]
    fn test_index_new() {
        let index = SeekIndex::new(48000);
        assert_eq!(index.timescale(), 48000);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert_eq!(index.keyframe_count(), 0);
    }

    #[test]
    fn test_index_add_entries() {
        let index = build_test_index();
        assert_eq!(index.len(), 20);
        assert_eq!(index.keyframe_count(), 4); // frames 0, 5, 10, 15
    }

    #[test]
    fn test_seconds_to_ticks() {
        let index = SeekIndex::new(90000);
        assert_eq!(index.seconds_to_ticks(1.0), 90000);
        assert_eq!(index.seconds_to_ticks(0.5), 45000);
    }

    #[test]
    fn test_ticks_to_seconds() {
        let index = SeekIndex::new(90000);
        let s = index.ticks_to_seconds(90000);
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ticks_to_seconds_zero_timescale() {
        let index = SeekIndex::new(0);
        assert_eq!(index.ticks_to_seconds(12345), 0.0);
    }

    // ── Keyframe search tests ───────────────────────────────────────────

    #[test]
    fn test_find_keyframe_before_exact() {
        let index = build_test_index();
        // Keyframes at pts: 0, 15000, 30000, 45000
        let kf = index.find_keyframe_before(15000).expect("should find");
        assert_eq!(kf.pts, 15000);
        assert!(kf.is_keyframe);
    }

    #[test]
    fn test_find_keyframe_before_between() {
        let index = build_test_index();
        // Target at 20000 (between kf@15000 and kf@30000)
        let kf = index.find_keyframe_before(20000).expect("should find");
        assert_eq!(kf.pts, 15000);
    }

    #[test]
    fn test_find_keyframe_before_start() {
        let index = build_test_index();
        let kf = index.find_keyframe_before(0).expect("should find");
        assert_eq!(kf.pts, 0);
    }

    #[test]
    fn test_find_keyframe_before_none() {
        let index = build_test_index();
        // Target before all entries
        let kf = index.find_keyframe_before(-1);
        assert!(kf.is_none());
    }

    #[test]
    fn test_find_keyframe_after() {
        let index = build_test_index();
        let kf = index.find_keyframe_after(20000).expect("should find");
        assert_eq!(kf.pts, 30000);
    }

    #[test]
    fn test_find_keyframe_after_exact() {
        let index = build_test_index();
        let kf = index.find_keyframe_after(15000).expect("should find");
        assert_eq!(kf.pts, 15000);
    }

    #[test]
    fn test_find_keyframe_after_past_end() {
        let index = build_test_index();
        let kf = index.find_keyframe_after(99999);
        assert!(kf.is_none());
    }

    #[test]
    fn test_find_nearest_keyframe_closer_before() {
        let index = build_test_index();
        // 16000 is closer to kf@15000 than kf@30000
        let kf = index.find_nearest_keyframe(16000).expect("should find");
        assert_eq!(kf.pts, 15000);
    }

    #[test]
    fn test_find_nearest_keyframe_closer_after() {
        let index = build_test_index();
        // 28000 is closer to kf@30000 than kf@15000
        let kf = index.find_nearest_keyframe(28000).expect("should find");
        assert_eq!(kf.pts, 30000);
    }

    #[test]
    fn test_find_nearest_keyframe_equidistant() {
        let index = build_test_index();
        // 22500 is equidistant between kf@15000 and kf@30000
        // Should prefer the earlier one (backward preference)
        let kf = index.find_nearest_keyframe(22500).expect("should find");
        assert_eq!(kf.pts, 15000);
    }

    // ── Sample-at tests ─────────────────────────────────────────────────

    #[test]
    fn test_find_sample_at_exact_start() {
        let index = build_test_index();
        let sample = index.find_sample_at(3000).expect("should find");
        assert_eq!(sample.pts, 3000);
    }

    #[test]
    fn test_find_sample_at_mid_frame() {
        let index = build_test_index();
        // 4000 is within frame at pts=3000 (duration=3000, so [3000..6000))
        let sample = index.find_sample_at(4000).expect("should find");
        assert_eq!(sample.pts, 3000);
    }

    #[test]
    fn test_find_sample_at_last() {
        let index = build_test_index();
        // Last frame: pts=57000
        let sample = index.find_sample_at(58000).expect("should find");
        assert_eq!(sample.pts, 57000);
    }

    // ── Seek Plan tests ─────────────────────────────────────────────────

    #[test]
    fn test_plan_seek_keyframe() {
        let index = build_test_index();
        let plan = index
            .plan_seek(20000, SeekAccuracy::Keyframe)
            .expect("should plan");
        // Should go to keyframe at 15000
        assert_eq!(plan.keyframe_entry.pts, 15000);
        assert_eq!(plan.discard_count, 0);
        assert_eq!(plan.target_entry.pts, 15000);
    }

    #[test]
    fn test_plan_seek_sample_accurate() {
        let index = build_test_index();
        // Target: pts=21000 (frame 7 at pts=21000)
        let plan = index
            .plan_seek(21000, SeekAccuracy::SampleAccurate)
            .expect("should plan");

        // Keyframe should be at pts=15000 (frame 5)
        assert_eq!(plan.keyframe_entry.pts, 15000);
        // Target should be the frame at pts=21000
        assert_eq!(plan.target_entry.pts, 21000);
        // Discard count: frames 6 (18000) between keyframe 5 (15000) and target 7 (21000)
        assert_eq!(plan.discard_count, 1); // frame at 18000
        assert!(plan.is_exact);
    }

    #[test]
    fn test_plan_seek_sample_accurate_on_keyframe() {
        let index = build_test_index();
        let plan = index
            .plan_seek(15000, SeekAccuracy::SampleAccurate)
            .expect("should plan");
        assert_eq!(plan.keyframe_entry.pts, 15000);
        assert_eq!(plan.discard_count, 0);
        assert!(plan.is_exact);
    }

    #[test]
    fn test_plan_seek_sample_accurate_first_frame() {
        let index = build_test_index();
        let plan = index
            .plan_seek(0, SeekAccuracy::SampleAccurate)
            .expect("should plan");
        assert_eq!(plan.keyframe_entry.pts, 0);
        assert_eq!(plan.target_entry.pts, 0);
        assert_eq!(plan.discard_count, 0);
    }

    #[test]
    fn test_plan_seek_within_tolerance_exact() {
        let index = build_test_index();
        // Within 5000 ticks of a keyframe at 15000
        let plan = index
            .plan_seek(16000, SeekAccuracy::WithinTolerance(5000))
            .expect("should plan");
        assert_eq!(plan.target_entry.pts, 15000);
    }

    #[test]
    fn test_plan_seek_within_tolerance_out_of_range() {
        let index = build_test_index();
        // Only 1 tick tolerance, and 20000 is not within 1 tick of any keyframe
        let plan = index.plan_seek(20000, SeekAccuracy::WithinTolerance(1));
        assert!(plan.is_none());
    }

    #[test]
    fn test_plan_seek_empty_index() {
        let index = SeekIndex::new(90000);
        let plan = index.plan_seek(0, SeekAccuracy::Keyframe);
        assert!(plan.is_none());
    }

    // ── Duration and statistics tests ───────────────────────────────────

    #[test]
    fn test_duration_ticks() {
        let index = build_test_index();
        // 20 frames, each 3000 ticks, last frame ends at 20*3000 = 60000
        assert_eq!(index.duration_ticks(), 60000);
    }

    #[test]
    fn test_duration_seconds() {
        let index = build_test_index();
        let dur = index.duration_seconds();
        // 60000 ticks at 90kHz = 0.6667 seconds
        assert!((dur - 60000.0 / 90000.0).abs() < 1e-6);
    }

    #[test]
    fn test_average_keyframe_interval() {
        let index = build_test_index();
        let avg = index.average_keyframe_interval().expect("should calculate");
        // Keyframes at 0, 15000, 30000, 45000 -> intervals: 15000, 15000, 15000
        assert!((avg - 15000.0).abs() < 1e-6);
    }

    #[test]
    fn test_average_keyframe_interval_single() {
        let mut index = SeekIndex::new(90000);
        index.add_entry(SeekIndexEntry::keyframe(0, 0, 0, 100, 3000, 0));
        assert!(index.average_keyframe_interval().is_none());
    }

    // ── Sort test ───────────────────────────────────────────────────────

    #[test]
    fn test_sort_reorders_entries() {
        let mut index = SeekIndex::new(90000);
        // Add out of order
        index.add_entry(SeekIndexEntry::non_keyframe(6000, 6000, 300, 100, 3000, 2));
        index.add_entry(SeekIndexEntry::keyframe(0, 0, 100, 100, 3000, 0));
        index.add_entry(SeekIndexEntry::non_keyframe(3000, 3000, 200, 100, 3000, 1));

        index.sort();

        assert_eq!(index.entries()[0].pts, 0);
        assert_eq!(index.entries()[1].pts, 3000);
        assert_eq!(index.entries()[2].pts, 6000);
        assert_eq!(index.keyframe_count(), 1);
    }

    // ── Edge cases ──────────────────────────────────────────────────────

    #[test]
    fn test_find_keyframe_empty() {
        let index = SeekIndex::new(90000);
        assert!(index.find_keyframe_before(0).is_none());
        assert!(index.find_keyframe_after(0).is_none());
        assert!(index.find_nearest_keyframe(0).is_none());
    }

    #[test]
    fn test_single_keyframe_index() {
        let mut index = SeekIndex::new(90000);
        index.add_entry(SeekIndexEntry::keyframe(0, 0, 0, 100, 90000, 0));

        let kf = index.find_keyframe_before(45000).expect("should find");
        assert_eq!(kf.pts, 0);

        let plan = index
            .plan_seek(45000, SeekAccuracy::SampleAccurate)
            .expect("should plan");
        assert_eq!(plan.keyframe_entry.pts, 0);
    }

    #[test]
    fn test_all_keyframes_index() {
        let mut index = SeekIndex::new(48000);
        // Audio: every frame is a keyframe
        for i in 0u32..100 {
            let pts = i64::from(i) * 960;
            index.add_entry(SeekIndexEntry::keyframe(
                pts,
                pts,
                u64::from(i) * 100,
                100,
                960,
                i,
            ));
        }

        assert_eq!(index.keyframe_count(), 100);

        let plan = index
            .plan_seek(48000, SeekAccuracy::SampleAccurate)
            .expect("should plan");
        // Should find the exact frame
        assert_eq!(plan.keyframe_entry.pts, 48000);
        assert_eq!(plan.discard_count, 0);
    }

    #[test]
    fn test_with_capacity() {
        let index = SeekIndex::with_capacity(90000, 1000);
        assert_eq!(index.timescale(), 90000);
        assert!(index.is_empty());
    }

    // ── SampleAccurateSeeker tests ───────────────────────────────────────

    fn build_seeker_index() -> SeekIndex {
        // 90kHz, 30fps (3000 ticks/frame), GOP=5
        let mut index = SeekIndex::new(90000);
        for i in 0u32..20 {
            let pts = i64::from(i) * 3000;
            let is_kf = i % 5 == 0;
            let offset = u64::from(i) * 500 + 1000;
            if is_kf {
                index.add_entry(SeekIndexEntry::keyframe(pts, pts, offset, 500, 3000, i));
            } else {
                index.add_entry(SeekIndexEntry::non_keyframe(pts, pts, offset, 200, 3000, i));
            }
        }
        index
    }

    #[test]
    fn test_sample_accurate_seeker_on_keyframe() {
        let track = TrackIndex::new(build_seeker_index());
        let seeker = SampleAccurateSeeker::with_track(TrackIndex::new(build_seeker_index()));
        let result = seeker.seek_to_sample(15000, &track).expect("should find");
        assert_eq!(result.keyframe_pts, 15000);
        assert_eq!(result.preroll_samples, 0);
        assert_eq!(result.sample_offset, 1000 + 5 * 500); // frame 5 offset
    }

    #[test]
    fn test_sample_accurate_seeker_between_keyframes() {
        let track = TrackIndex::new(build_seeker_index());
        let seeker = SampleAccurateSeeker::with_track(TrackIndex::new(build_seeker_index()));
        // Target pts=21000 (frame 7) — keyframe is at pts=15000 (frame 5)
        // Frames 6 (pts=18000) must be discarded → preroll = 1
        let result = seeker.seek_to_sample(21000, &track).expect("should find");
        assert_eq!(result.keyframe_pts, 15000);
        assert_eq!(result.preroll_samples, 1);
    }

    #[test]
    fn test_sample_accurate_seeker_codec_delay_added() {
        let track = TrackIndex::with_codec_delay(build_seeker_index(), 512);
        let seeker = SampleAccurateSeeker::with_track(TrackIndex::new(build_seeker_index()));
        // On a keyframe: preroll = 0 inter-frame + 512 codec_delay = 512
        let result = seeker.seek_to_sample(0, &track).expect("should find");
        assert_eq!(result.keyframe_pts, 0);
        assert_eq!(result.preroll_samples, 512);
    }

    #[test]
    fn test_sample_accurate_seeker_empty_index() {
        let track = TrackIndex::new(SeekIndex::new(90000));
        let seeker = SampleAccurateSeeker::with_track(TrackIndex::new(SeekIndex::new(90000)));
        let result = seeker.seek_to_sample(0, &track);
        assert!(result.is_none());
    }

    #[test]
    fn test_track_index_default_codec_delay() {
        let idx = SeekIndex::new(90000);
        let track = TrackIndex::new(idx);
        assert_eq!(track.codec_delay_samples, 0);
    }

    #[test]
    fn test_seek_result_fields() {
        let track = TrackIndex::new(build_seeker_index());
        let seeker = SampleAccurateSeeker::with_track(TrackIndex::new(build_seeker_index()));
        let result = seeker.seek_to_sample(0, &track).expect("should find");
        // frame 0 is a keyframe at file offset 1000
        assert_eq!(result.keyframe_pts, 0);
        assert_eq!(result.sample_offset, 1000);
        assert_eq!(result.preroll_samples, 0);
    }

    // ── MultiTrackSeeker tests ───────────────────────────────────────────

    fn build_multi_track_samples() -> Vec<SampleIndexEntry> {
        // 30fps video at 90kHz: keyframe every 5 frames (GOP=5)
        (0u32..20)
            .map(|i| {
                let pts = i64::from(i) * 3000;
                let offset = 1000 + u64::from(i) * 500;
                if i % 5 == 0 {
                    SampleIndexEntry::keyframe(pts, offset)
                } else {
                    SampleIndexEntry::delta(pts, offset)
                }
            })
            .collect()
    }

    #[test]
    fn test_multi_track_seeker_new() {
        let seeker = MultiTrackSeeker::new();
        assert_eq!(seeker.indexed_track_count(), 0);
    }

    #[test]
    fn test_multi_track_build_index() {
        let mut seeker = MultiTrackSeeker::new();
        let samples = build_multi_track_samples();
        seeker.build_index(1, &samples).expect("build ok");
        assert_eq!(seeker.indexed_track_count(), 1);
        assert_eq!(seeker.sample_count(1), Some(20));
    }

    #[test]
    fn test_multi_track_entries_sorted() {
        let mut seeker = MultiTrackSeeker::new();
        // Insert out of order
        let samples = vec![
            SampleIndexEntry::delta(9000, 3000),
            SampleIndexEntry::keyframe(0, 1000),
            SampleIndexEntry::delta(3000, 1500),
            SampleIndexEntry::delta(6000, 2000),
        ];
        seeker.build_index(1, &samples).expect("ok");
        let entries = seeker.entries(1).expect("entries exist");
        assert_eq!(entries[0].pts, 0);
        assert_eq!(entries[1].pts, 3000);
        assert_eq!(entries[2].pts, 6000);
        assert_eq!(entries[3].pts, 9000);
    }

    #[test]
    fn test_seek_to_pts_exact() {
        let mut seeker = MultiTrackSeeker::new();
        seeker
            .build_index(1, &build_multi_track_samples())
            .expect("ok");
        // Seek exactly to frame 5 (pts=15000)
        let result = seeker.seek_to_pts(1, 15000).expect("seek ok");
        assert_eq!(result.found_pts, 15000);
        assert_eq!(result.byte_offset, 1000 + 5 * 500);
        assert_eq!(result.sample_idx, 5);
    }

    #[test]
    fn test_seek_to_pts_between_samples() {
        let mut seeker = MultiTrackSeeker::new();
        seeker
            .build_index(1, &build_multi_track_samples())
            .expect("ok");
        // PTS 16000 falls between frame 5 (15000) and frame 6 (18000)
        let result = seeker.seek_to_pts(1, 16000).expect("seek ok");
        assert_eq!(
            result.found_pts, 15000,
            "should return the preceding sample"
        );
        assert_eq!(result.sample_idx, 5);
    }

    #[test]
    fn test_seek_to_pts_first_sample() {
        let mut seeker = MultiTrackSeeker::new();
        seeker
            .build_index(1, &build_multi_track_samples())
            .expect("ok");
        let result = seeker.seek_to_pts(1, 0).expect("seek ok");
        assert_eq!(result.found_pts, 0);
        assert_eq!(result.sample_idx, 0);
    }

    #[test]
    fn test_seek_to_pts_last_sample() {
        let mut seeker = MultiTrackSeeker::new();
        seeker
            .build_index(1, &build_multi_track_samples())
            .expect("ok");
        // PTS beyond the last sample should return the last sample
        let result = seeker.seek_to_pts(1, 99999).expect("seek ok");
        assert_eq!(result.found_pts, 19 * 3000); // last sample
        assert_eq!(result.sample_idx, 19);
    }

    #[test]
    fn test_seek_to_pts_before_first_sample() {
        let mut seeker = MultiTrackSeeker::new();
        seeker
            .build_index(1, &build_multi_track_samples())
            .expect("ok");
        let err = seeker.seek_to_pts(1, -1);
        assert!(matches!(
            err,
            Err(MultiTrackSeekerError::BeforeFirstSample(-1, 1))
        ));
    }

    #[test]
    fn test_seek_to_pts_no_index() {
        let seeker = MultiTrackSeeker::new();
        let err = seeker.seek_to_pts(42, 0);
        assert!(matches!(err, Err(MultiTrackSeekerError::NoIndex(42))));
    }

    #[test]
    fn test_seek_to_pts_empty_index() {
        let mut seeker = MultiTrackSeeker::new();
        seeker.build_index(1, &[]).expect("ok");
        let err = seeker.seek_to_pts(1, 0);
        assert!(matches!(err, Err(MultiTrackSeekerError::EmptyIndex(1))));
    }

    #[test]
    fn test_multi_track_multiple_tracks() {
        let mut seeker = MultiTrackSeeker::new();
        let video = build_multi_track_samples();
        // 51 audio frames: pts 0, 960, 1920, ..., 50*960=48000
        let audio: Vec<SampleIndexEntry> = (0u32..=50)
            .map(|i| SampleIndexEntry::keyframe(i64::from(i) * 960, u64::from(i) * 100 + 500))
            .collect();

        seeker.build_index(1, &video).expect("video ok");
        seeker.build_index(2, &audio).expect("audio ok");

        assert_eq!(seeker.indexed_track_count(), 2);

        let v_result = seeker.seek_to_pts(1, 15000).expect("video seek ok");
        let a_result = seeker.seek_to_pts(2, 48000).expect("audio seek ok");

        assert_eq!(v_result.found_pts, 15000);
        assert_eq!(a_result.found_pts, 48000);
    }

    #[test]
    fn test_clear_index() {
        let mut seeker = MultiTrackSeeker::new();
        seeker
            .build_index(1, &build_multi_track_samples())
            .expect("ok");
        assert_eq!(seeker.indexed_track_count(), 1);
        seeker.clear_index(1);
        assert_eq!(seeker.indexed_track_count(), 0);
        let err = seeker.seek_to_pts(1, 0);
        assert!(matches!(err, Err(MultiTrackSeekerError::NoIndex(1))));
    }

    #[test]
    fn test_build_index_replaces_existing() {
        let mut seeker = MultiTrackSeeker::new();
        let old: Vec<SampleIndexEntry> = vec![SampleIndexEntry::keyframe(0, 100)];
        let new: Vec<SampleIndexEntry> = vec![
            SampleIndexEntry::keyframe(0, 200),
            SampleIndexEntry::keyframe(3000, 300),
        ];
        seeker.build_index(1, &old).expect("ok");
        seeker.build_index(1, &new).expect("replace ok");
        assert_eq!(seeker.sample_count(1), Some(2));
        let result = seeker.seek_to_pts(1, 0).expect("ok");
        assert_eq!(result.byte_offset, 200);
    }

    #[test]
    fn test_sample_index_entry_constructors() {
        let kf = SampleIndexEntry::keyframe(1000, 9999);
        assert!(kf.is_sync);
        assert_eq!(kf.pts, 1000);
        assert_eq!(kf.byte_offset, 9999);

        let df = SampleIndexEntry::delta(2000, 8888);
        assert!(!df.is_sync);
        assert_eq!(df.pts, 2000);
    }

    #[test]
    fn test_pts_seek_result_fields() {
        let mut seeker = MultiTrackSeeker::new();
        seeker
            .build_index(1, &[SampleIndexEntry::keyframe(5000, 12345)])
            .expect("ok");
        let r = seeker.seek_to_pts(1, 5000).expect("ok");
        assert_eq!(r.found_pts, 5000);
        assert_eq!(r.byte_offset, 12345);
        assert_eq!(r.sample_idx, 0);
    }
}
