//! Segment lifecycle management for adaptive bitrate streaming.
//!
//! This module tracks the full lifecycle of media segments from download request
//! through decode, playback, and eventual eviction from the buffer. It provides:
//!
//! - [`SegmentState`] — state machine for individual segment lifecycle.
//! - [`SegmentBuffer`] — ring buffer with watermark-based eviction events.
//! - [`LifecycleEvent`] — typed events emitted at each state transition.
//!
//! # Design
//!
//! Each segment progresses through a well-defined state machine:
//!
//! ```text
//! Requested → Downloading → Downloaded → Decoding → Buffered → Playing → Evicted
//!                                ↓
//!                             Failed
//! ```
//!
//! The [`SegmentBuffer`] maintains a sliding window of segments limited by a
//! configurable maximum buffer duration. When the high-watermark is breached,
//! a [`LifecycleEvent::HighWatermark`] is emitted. When it drops below the
//! low-watermark, a [`LifecycleEvent::LowWatermark`] is emitted.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::error::{StreamError, StreamResult};

/// The lifecycle state of a single media segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentState {
    /// A download request has been issued for this segment.
    Requested,
    /// The segment is currently being downloaded.
    Downloading {
        /// Monotonic timestamp when the download started.
        started_at: Instant,
    },
    /// The segment has been fully downloaded and is awaiting decode.
    Downloaded {
        /// How long the download took.
        download_duration: Duration,
        /// Byte length of the downloaded data.
        byte_length: u64,
    },
    /// The segment is being decoded.
    Decoding {
        /// Monotonic timestamp when decode began.
        started_at: Instant,
    },
    /// The segment is decoded and held in the playback buffer.
    Buffered {
        /// Total duration (download + decode) before entering buffer.
        total_prep_duration: Duration,
    },
    /// The segment is currently being rendered/played.
    Playing,
    /// The segment has been played and removed from the buffer.
    Evicted,
    /// A non-recoverable error occurred during this segment's lifecycle.
    Failed {
        /// Human-readable reason for failure.
        reason: String,
    },
}

/// A descriptor for a single media segment.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Monotonically increasing sequence number.
    pub sequence_number: u64,
    /// Presentation timestamp in seconds.
    pub pts_secs: f64,
    /// Duration of the media content in seconds.
    pub duration_secs: f64,
    /// Quality level index this segment belongs to.
    pub quality_index: usize,
    /// URL from which this segment should be fetched.
    pub url: String,
    /// Current lifecycle state.
    pub state: SegmentState,
}

impl Segment {
    /// Create a new segment in the [`SegmentState::Requested`] state.
    pub fn new(
        sequence_number: u64,
        pts_secs: f64,
        duration_secs: f64,
        quality_index: usize,
        url: String,
    ) -> Self {
        Self {
            sequence_number,
            pts_secs,
            duration_secs,
            quality_index,
            url,
            state: SegmentState::Requested,
        }
    }

    /// Transition to the `Downloading` state.
    ///
    /// # Errors
    /// Returns `StreamError::Segment` if the segment is not in [`SegmentState::Requested`].
    pub fn begin_download(&mut self) -> StreamResult<()> {
        match &self.state {
            SegmentState::Requested => {
                self.state = SegmentState::Downloading {
                    started_at: Instant::now(),
                };
                Ok(())
            }
            other => Err(StreamError::Segment(format!(
                "cannot begin_download from state {other:?}"
            ))),
        }
    }

    /// Transition to the `Downloaded` state.
    ///
    /// # Errors
    /// Returns `StreamError::Segment` if not currently `Downloading`.
    pub fn finish_download(&mut self, byte_length: u64) -> StreamResult<()> {
        match &self.state {
            SegmentState::Downloading { started_at } => {
                let download_duration = started_at.elapsed();
                self.state = SegmentState::Downloaded {
                    download_duration,
                    byte_length,
                };
                Ok(())
            }
            other => Err(StreamError::Segment(format!(
                "cannot finish_download from state {other:?}"
            ))),
        }
    }

    /// Transition to the `Decoding` state.
    ///
    /// # Errors
    /// Returns `StreamError::Segment` if not currently `Downloaded`.
    pub fn begin_decode(&mut self) -> StreamResult<()> {
        match &self.state {
            SegmentState::Downloaded { .. } => {
                self.state = SegmentState::Decoding {
                    started_at: Instant::now(),
                };
                Ok(())
            }
            other => Err(StreamError::Segment(format!(
                "cannot begin_decode from state {other:?}"
            ))),
        }
    }

    /// Transition to the `Buffered` state.
    ///
    /// # Errors
    /// Returns `StreamError::Segment` if not currently `Decoding`.
    pub fn finish_decode(&mut self) -> StreamResult<Duration> {
        match &self.state {
            SegmentState::Decoding { started_at } => {
                let decode_duration = started_at.elapsed();
                // Approximate total prep duration (we lost the download_duration in Decoding state).
                let total_prep_duration = decode_duration;
                self.state = SegmentState::Buffered {
                    total_prep_duration,
                };
                Ok(decode_duration)
            }
            other => Err(StreamError::Segment(format!(
                "cannot finish_decode from state {other:?}"
            ))),
        }
    }

    /// Transition to the `Playing` state.
    ///
    /// # Errors
    /// Returns `StreamError::Segment` if not currently `Buffered`.
    pub fn begin_play(&mut self) -> StreamResult<()> {
        match &self.state {
            SegmentState::Buffered { .. } => {
                self.state = SegmentState::Playing;
                Ok(())
            }
            other => Err(StreamError::Segment(format!(
                "cannot begin_play from state {other:?}"
            ))),
        }
    }

    /// Transition to the `Evicted` state.
    ///
    /// # Errors
    /// Returns `StreamError::Segment` if not currently `Playing`.
    pub fn evict(&mut self) -> StreamResult<()> {
        match &self.state {
            SegmentState::Playing => {
                self.state = SegmentState::Evicted;
                Ok(())
            }
            other => Err(StreamError::Segment(format!(
                "cannot evict from state {other:?}"
            ))),
        }
    }

    /// Mark the segment as failed from any state.
    pub fn fail(&mut self, reason: impl Into<String>) {
        self.state = SegmentState::Failed {
            reason: reason.into(),
        };
    }

    /// Return true if the segment occupies buffer space (i.e., is `Buffered` or `Playing`).
    pub fn is_in_buffer(&self) -> bool {
        matches!(
            &self.state,
            SegmentState::Buffered { .. } | SegmentState::Playing
        )
    }
}

/// An event emitted by the [`SegmentBuffer`] during lifecycle transitions.
#[derive(Debug, Clone, PartialEq)]
pub enum LifecycleEvent {
    /// A segment transitioned from one state to another.
    StateChanged {
        /// Sequence number of the affected segment.
        sequence_number: u64,
        /// The new state (as a label string for logging/monitoring).
        new_state: &'static str,
    },
    /// Buffer occupancy crossed above the high-watermark threshold.
    HighWatermark {
        /// Current occupancy in seconds.
        occupancy_secs: f64,
        /// Configured high-watermark threshold.
        threshold_secs: f64,
    },
    /// Buffer occupancy dropped below the low-watermark threshold.
    LowWatermark {
        /// Current occupancy in seconds.
        occupancy_secs: f64,
        /// Configured low-watermark threshold.
        threshold_secs: f64,
    },
    /// A segment was automatically evicted to make room (oldest `Evicted` entries pruned).
    SegmentPruned {
        /// Sequence number of the pruned segment.
        sequence_number: u64,
    },
}

/// Watermark configuration for the [`SegmentBuffer`].
#[derive(Debug, Clone)]
pub struct WatermarkConfig {
    /// Low-watermark in seconds. Emits [`LifecycleEvent::LowWatermark`] when crossed downward.
    pub low_secs: f64,
    /// High-watermark in seconds. Emits [`LifecycleEvent::HighWatermark`] when crossed upward.
    pub high_secs: f64,
    /// Maximum buffer duration in seconds. Segments are automatically pruned beyond this.
    pub max_secs: f64,
}

impl WatermarkConfig {
    /// Create a new [`WatermarkConfig`] with validation.
    ///
    /// # Errors
    /// Returns `StreamError::InvalidParameter` if values are inconsistent.
    pub fn new(low_secs: f64, high_secs: f64, max_secs: f64) -> StreamResult<Self> {
        if low_secs <= 0.0 {
            return Err(StreamError::InvalidParameter("low_secs must be > 0".into()));
        }
        if high_secs <= low_secs {
            return Err(StreamError::InvalidParameter(
                "high_secs must be > low_secs".into(),
            ));
        }
        if max_secs <= high_secs {
            return Err(StreamError::InvalidParameter(
                "max_secs must be > high_secs".into(),
            ));
        }
        Ok(Self {
            low_secs,
            high_secs,
            max_secs,
        })
    }
}

/// Ring buffer tracking segment lifecycle with watermark events.
pub struct SegmentBuffer {
    segments: VecDeque<Segment>,
    watermarks: WatermarkConfig,
    /// Track whether we are currently above the high-watermark (to avoid repeated events).
    above_high_watermark: bool,
    /// Track whether we are currently below the low-watermark.
    below_low_watermark: bool,
    /// Pending events to be consumed by the caller.
    pending_events: Vec<LifecycleEvent>,
}

impl SegmentBuffer {
    /// Create a new [`SegmentBuffer`] with the given watermark configuration.
    pub fn new(watermarks: WatermarkConfig) -> Self {
        Self {
            segments: VecDeque::new(),
            watermarks,
            above_high_watermark: false,
            below_low_watermark: true,
            pending_events: Vec::new(),
        }
    }

    /// Return total playback buffer occupancy from buffered segments (seconds).
    pub fn occupancy_secs(&self) -> f64 {
        self.segments
            .iter()
            .filter(|s| s.is_in_buffer())
            .map(|s| s.duration_secs)
            .sum()
    }

    /// Push a new segment into the buffer in the `Requested` state.
    ///
    /// # Errors
    /// Returns `StreamError::BufferFull` if adding the segment would exceed `max_secs`.
    pub fn push(&mut self, segment: Segment) -> StreamResult<()> {
        let projected = self.occupancy_secs() + segment.duration_secs;
        if projected > self.watermarks.max_secs {
            return Err(StreamError::BufferFull {
                capacity: self.segments.len(),
            });
        }
        self.segments.push_back(segment);
        self.check_watermarks();
        Ok(())
    }

    /// Look up a mutable reference to a segment by sequence number.
    pub fn get_mut(&mut self, sequence_number: u64) -> Option<&mut Segment> {
        self.segments
            .iter_mut()
            .find(|s| s.sequence_number == sequence_number)
    }

    /// Apply a state transition function to the segment with the given sequence number,
    /// then record a [`LifecycleEvent::StateChanged`] and re-check watermarks.
    ///
    /// # Errors
    /// Returns `StreamError::Segment` if the segment is not found or the transition fails.
    pub fn transition<F>(&mut self, sequence_number: u64, f: F) -> StreamResult<()>
    where
        F: FnOnce(&mut Segment) -> StreamResult<()>,
    {
        let seg = self
            .segments
            .iter_mut()
            .find(|s| s.sequence_number == sequence_number)
            .ok_or_else(|| StreamError::Segment(format!("segment {sequence_number} not found")))?;

        f(seg)?;

        let label = state_label(&seg.state);
        self.pending_events.push(LifecycleEvent::StateChanged {
            sequence_number,
            new_state: label,
        });

        self.check_watermarks();
        self.prune_evicted();

        Ok(())
    }

    /// Drain and return all pending lifecycle events.
    pub fn drain_events(&mut self) -> Vec<LifecycleEvent> {
        std::mem::take(&mut self.pending_events)
    }

    /// Return the number of segments currently tracked (in any state).
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Return true if the buffer contains no segments.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Check and emit watermark events based on current occupancy.
    fn check_watermarks(&mut self) {
        let occ = self.occupancy_secs();
        if occ >= self.watermarks.high_secs && !self.above_high_watermark {
            self.above_high_watermark = true;
            self.below_low_watermark = false;
            self.pending_events.push(LifecycleEvent::HighWatermark {
                occupancy_secs: occ,
                threshold_secs: self.watermarks.high_secs,
            });
        } else if occ < self.watermarks.low_secs && !self.below_low_watermark {
            self.below_low_watermark = true;
            self.above_high_watermark = false;
            self.pending_events.push(LifecycleEvent::LowWatermark {
                occupancy_secs: occ,
                threshold_secs: self.watermarks.low_secs,
            });
        } else if occ < self.watermarks.high_secs {
            self.above_high_watermark = false;
        }
    }

    /// Remove old `Evicted` segments from the front of the queue.
    fn prune_evicted(&mut self) {
        while let Some(front) = self.segments.front() {
            if front.state == SegmentState::Evicted {
                let seq = front.sequence_number;
                self.segments.pop_front();
                self.pending_events.push(LifecycleEvent::SegmentPruned {
                    sequence_number: seq,
                });
            } else {
                break;
            }
        }
    }
}

/// Map a segment state to a static label string.
fn state_label(state: &SegmentState) -> &'static str {
    match state {
        SegmentState::Requested => "Requested",
        SegmentState::Downloading { .. } => "Downloading",
        SegmentState::Downloaded { .. } => "Downloaded",
        SegmentState::Decoding { .. } => "Decoding",
        SegmentState::Buffered { .. } => "Buffered",
        SegmentState::Playing => "Playing",
        SegmentState::Evicted => "Evicted",
        SegmentState::Failed { .. } => "Failed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn make_segment(seq: u64, duration: f64) -> Segment {
        Segment::new(
            seq,
            seq as f64 * duration,
            duration,
            0,
            format!("http://example.com/{seq}.ts"),
        )
    }

    fn default_watermarks() -> StreamResult<WatermarkConfig> {
        WatermarkConfig::new(5.0, 15.0, 30.0)
    }

    #[test]
    fn test_watermark_config_valid() {
        assert!(WatermarkConfig::new(5.0, 15.0, 30.0).is_ok());
    }

    #[test]
    fn test_watermark_config_invalid_order() {
        assert!(WatermarkConfig::new(15.0, 5.0, 30.0).is_err());
        assert!(WatermarkConfig::new(5.0, 30.0, 20.0).is_err());
    }

    #[test]
    fn test_watermark_config_zero_low() {
        assert!(WatermarkConfig::new(0.0, 15.0, 30.0).is_err());
    }

    #[test]
    fn test_segment_state_machine() -> TestResult {
        let mut seg = make_segment(1, 4.0);
        assert_eq!(seg.state, SegmentState::Requested);

        seg.begin_download()?;
        assert!(matches!(seg.state, SegmentState::Downloading { .. }));

        seg.finish_download(500_000)?;
        assert!(matches!(seg.state, SegmentState::Downloaded { .. }));

        seg.begin_decode()?;
        assert!(matches!(seg.state, SegmentState::Decoding { .. }));

        seg.finish_decode()?;
        assert!(matches!(seg.state, SegmentState::Buffered { .. }));

        seg.begin_play()?;
        assert_eq!(seg.state, SegmentState::Playing);

        seg.evict()?;
        assert_eq!(seg.state, SegmentState::Evicted);
        Ok(())
    }

    #[test]
    fn test_segment_invalid_transition() {
        let mut seg = make_segment(2, 4.0);
        // Cannot finish_download before begin_download.
        assert!(seg.finish_download(100).is_err());
    }

    #[test]
    fn test_segment_fail_from_any_state() -> TestResult {
        let mut seg = make_segment(3, 4.0);
        seg.begin_download()?;
        seg.fail("network error");
        assert!(matches!(seg.state, SegmentState::Failed { .. }));
        Ok(())
    }

    #[test]
    fn test_buffer_push_and_occupancy() -> TestResult {
        let wm = default_watermarks()?;
        let mut buf = SegmentBuffer::new(wm);
        let seg = make_segment(1, 4.0);
        buf.push(seg)?;
        // Segment in Requested state doesn't count toward occupancy.
        assert_eq!(buf.occupancy_secs(), 0.0);
        Ok(())
    }

    #[test]
    fn test_buffer_full_error() -> TestResult {
        let wm = WatermarkConfig::new(2.0, 5.0, 8.0)?;
        let mut buf = SegmentBuffer::new(wm);
        // Use transition to put segment in Buffered state so it counts toward occupancy.
        buf.push(make_segment(1, 4.0))?;
        buf.transition(1, |s| {
            s.begin_download()?;
            s.finish_download(0)?;
            s.begin_decode()?;
            s.finish_decode().map(|_| ())
        })?;
        // Occupancy is now 4.0s. Trying to push a 5.0s segment (would be 9.0 > 8.0) should fail.
        let big = make_segment(2, 5.0);
        assert!(matches!(buf.push(big), Err(StreamError::BufferFull { .. })));
        Ok(())
    }

    #[test]
    fn test_high_watermark_event() -> TestResult {
        let wm = WatermarkConfig::new(2.0, 10.0, 30.0)?;
        let mut buf = SegmentBuffer::new(wm);

        // Add and buffer 12 seconds of segments.
        for i in 0..3u64 {
            buf.push(make_segment(i, 4.0))?;
            buf.transition(i, |s| {
                s.begin_download()?;
                s.finish_download(0)?;
                s.begin_decode()?;
                s.finish_decode().map(|_| ())
            })?;
        }

        let events = buf.drain_events();
        let has_high = events
            .iter()
            .any(|e| matches!(e, LifecycleEvent::HighWatermark { .. }));
        assert!(
            has_high,
            "should emit HighWatermark when crossing threshold"
        );
        Ok(())
    }

    #[test]
    fn test_evicted_segments_pruned() -> TestResult {
        let wm = default_watermarks()?;
        let mut buf = SegmentBuffer::new(wm);
        buf.push(make_segment(1, 4.0))?;

        // Walk segment through full lifecycle.
        buf.transition(1, |s| {
            s.begin_download()?;
            s.finish_download(0)?;
            s.begin_decode()?;
            s.finish_decode().map(|_| ())
        })?;
        buf.transition(1, |s| s.begin_play())?;
        buf.transition(1, |s| s.evict())?;

        let events = buf.drain_events();
        let pruned = events
            .iter()
            .any(|e| matches!(e, LifecycleEvent::SegmentPruned { sequence_number: 1 }));
        assert!(pruned, "evicted segment should be pruned and event emitted");
        assert!(buf.is_empty(), "buffer should be empty after eviction");
        Ok(())
    }

    #[test]
    fn test_get_mut_not_found() -> TestResult {
        let wm = default_watermarks()?;
        let mut buf = SegmentBuffer::new(wm);
        assert!(buf.get_mut(999).is_none());
        Ok(())
    }

    #[test]
    fn test_download_duration_captured() -> TestResult {
        let mut seg = make_segment(10, 4.0);
        seg.begin_download()?;
        sleep(Duration::from_millis(1));
        seg.finish_download(1024)?;
        if let SegmentState::Downloaded {
            download_duration,
            byte_length,
        } = &seg.state
        {
            assert!(*download_duration >= Duration::from_millis(1));
            assert_eq!(*byte_length, 1024);
        } else {
            panic!("expected Downloaded state");
        }
        Ok(())
    }
}
