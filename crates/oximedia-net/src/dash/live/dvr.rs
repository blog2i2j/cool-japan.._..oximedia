//! DVR (time-shift buffer) management for DASH live streaming.
//!
//! This module implements a Digital Video Recorder (DVR) buffer that allows
//! viewers to seek back in time during a live stream. It manages the retention
//! of segments and provides access to historical segments within the configured
//! time window.

#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]

use bytes::Bytes;
use std::collections::VecDeque;
use std::time::{Duration, SystemTime};

/// Maximum number of segments to keep in the DVR buffer.
const MAX_DVR_SEGMENTS: usize = 500;

/// DVR buffer for managing time-shift capability in live streams.
///
/// The DVR buffer maintains a sliding window of segments, allowing clients
/// to seek backwards in time up to the configured buffer depth.
#[derive(Debug)]
pub struct DvrBuffer {
    /// Buffer of segments.
    segments: VecDeque<DvrSegment>,
    /// Maximum buffer depth (time window).
    buffer_depth: Duration,
    /// Maximum number of segments to retain.
    max_segments: usize,
    /// Total size in bytes.
    total_size: u64,
    /// Maximum total size in bytes (0 = unlimited).
    max_size: u64,
}

/// A segment stored in the DVR buffer.
#[derive(Debug, Clone)]
pub struct DvrSegment {
    /// Segment number.
    pub number: u64,
    /// Representation ID.
    pub representation_id: String,
    /// Segment data.
    pub data: Bytes,
    /// Presentation timestamp (start time).
    pub pts: Duration,
    /// Segment duration.
    pub duration: Duration,
    /// Wall clock time when added.
    pub wall_clock_time: SystemTime,
    /// Timescale for this segment.
    pub timescale: u32,
}

impl DvrBuffer {
    /// Creates a new DVR buffer.
    ///
    /// # Arguments
    ///
    /// * `buffer_depth` - Maximum time window to retain
    #[must_use]
    pub fn new(buffer_depth: Duration) -> Self {
        Self {
            segments: VecDeque::new(),
            buffer_depth,
            max_segments: MAX_DVR_SEGMENTS,
            total_size: 0,
            max_size: 0,
        }
    }

    /// Creates a DVR buffer with a maximum size constraint.
    ///
    /// # Arguments
    ///
    /// * `buffer_depth` - Maximum time window to retain
    /// * `max_size` - Maximum buffer size in bytes
    #[must_use]
    pub fn with_max_size(buffer_depth: Duration, max_size: u64) -> Self {
        Self {
            segments: VecDeque::new(),
            buffer_depth,
            max_segments: MAX_DVR_SEGMENTS,
            total_size: 0,
            max_size,
        }
    }

    /// Adds a segment to the DVR buffer.
    ///
    /// # Arguments
    ///
    /// * `segment` - The segment to add
    pub fn add_segment(&mut self, segment: DvrSegment) {
        self.total_size += segment.data.len() as u64;
        self.segments.push_back(segment);

        // Trim based on time, count, and size
        self.trim_buffer();
    }

    /// Retrieves a segment by number and representation ID.
    ///
    /// # Arguments
    ///
    /// * `number` - Segment number
    /// * `representation_id` - Representation identifier
    ///
    /// # Returns
    ///
    /// The segment if found, `None` otherwise
    #[must_use]
    pub fn get_segment(&self, number: u64, representation_id: &str) -> Option<&DvrSegment> {
        self.segments
            .iter()
            .find(|s| s.number == number && s.representation_id == representation_id)
    }

    /// Retrieves segments in a time range.
    ///
    /// # Arguments
    ///
    /// * `start` - Start time
    /// * `end` - End time
    /// * `representation_id` - Representation identifier
    ///
    /// # Returns
    ///
    /// Vector of segments in the range
    pub fn get_segments_in_range(
        &self,
        start: Duration,
        end: Duration,
        representation_id: &str,
    ) -> Vec<&DvrSegment> {
        self.segments
            .iter()
            .filter(|s| {
                s.representation_id == representation_id
                    && s.pts < end
                    && s.pts + s.duration > start
            })
            .collect()
    }

    /// Returns all segments for a specific representation.
    ///
    /// # Arguments
    ///
    /// * `representation_id` - Representation identifier
    pub fn get_segments_for_representation(&self, representation_id: &str) -> Vec<&DvrSegment> {
        self.segments
            .iter()
            .filter(|s| s.representation_id == representation_id)
            .collect()
    }

    /// Returns the earliest available segment.
    #[must_use]
    pub fn earliest_segment(&self) -> Option<&DvrSegment> {
        self.segments.front()
    }

    /// Returns the latest available segment.
    #[must_use]
    pub fn latest_segment(&self) -> Option<&DvrSegment> {
        self.segments.back()
    }

    /// Returns the earliest presentation time.
    #[must_use]
    pub fn earliest_time(&self) -> Option<Duration> {
        self.segments.front().map(|s| s.pts)
    }

    /// Returns the latest presentation time.
    #[must_use]
    pub fn latest_time(&self) -> Option<Duration> {
        self.segments.back().map(|s| s.pts + s.duration)
    }

    /// Returns the available time window.
    #[must_use]
    pub fn available_window(&self) -> Option<Duration> {
        match (self.earliest_time(), self.latest_time()) {
            (Some(start), Some(end)) if end > start => Some(end - start),
            _ => None,
        }
    }

    /// Returns the number of segments in the buffer.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Returns the total size of buffered data in bytes.
    #[must_use]
    pub const fn total_size(&self) -> u64 {
        self.total_size
    }

    /// Returns the buffer depth configuration.
    #[must_use]
    pub const fn buffer_depth(&self) -> Duration {
        self.buffer_depth
    }

    /// Sets a new buffer depth.
    pub fn set_buffer_depth(&mut self, depth: Duration) {
        self.buffer_depth = depth;
        self.trim_buffer();
    }

    /// Sets the maximum number of segments.
    pub fn set_max_segments(&mut self, max: usize) {
        self.max_segments = max;
        self.trim_buffer();
    }

    /// Clears all segments from the buffer.
    pub fn clear(&mut self) {
        self.segments.clear();
        self.total_size = 0;
    }

    /// Removes segments older than the buffer depth.
    fn trim_buffer(&mut self) {
        // Get the current latest time
        let latest_time = match self.latest_time() {
            Some(t) => t,
            None => return,
        };

        // Calculate cutoff time
        let cutoff_time = latest_time.saturating_sub(self.buffer_depth);

        // Remove old segments by time
        while let Some(segment) = self.segments.front() {
            if segment.pts + segment.duration <= cutoff_time {
                // Front was confirmed by `while let Some(segment)`.
                if let Some(removed) = self.segments.pop_front() {
                    self.total_size = self.total_size.saturating_sub(removed.data.len() as u64);
                }
            } else {
                break;
            }
        }

        // Enforce max segment count
        while self.segments.len() > self.max_segments {
            if let Some(removed) = self.segments.pop_front() {
                self.total_size = self.total_size.saturating_sub(removed.data.len() as u64);
            }
        }

        // Enforce max size constraint
        if self.max_size > 0 {
            while self.total_size > self.max_size {
                if let Some(removed) = self.segments.pop_front() {
                    self.total_size = self.total_size.saturating_sub(removed.data.len() as u64);
                } else {
                    break;
                }
            }
        }
    }

    /// Returns statistics about the DVR buffer.
    #[must_use]
    pub fn stats(&self) -> DvrStats {
        DvrStats {
            segment_count: self.segment_count(),
            total_size: self.total_size,
            earliest_time: self.earliest_time(),
            latest_time: self.latest_time(),
            available_window: self.available_window(),
            buffer_depth: self.buffer_depth,
        }
    }

    /// Checks if a segment is available.
    #[must_use]
    pub fn has_segment(&self, number: u64, representation_id: &str) -> bool {
        self.segments
            .iter()
            .any(|s| s.number == number && s.representation_id == representation_id)
    }

    /// Returns the segment numbers available for a representation.
    pub fn available_segment_numbers(&self, representation_id: &str) -> Vec<u64> {
        self.segments
            .iter()
            .filter(|s| s.representation_id == representation_id)
            .map(|s| s.number)
            .collect()
    }
}

/// DVR buffer statistics.
#[derive(Debug, Clone)]
pub struct DvrStats {
    /// Number of segments in buffer.
    pub segment_count: usize,
    /// Total size in bytes.
    pub total_size: u64,
    /// Earliest presentation time.
    pub earliest_time: Option<Duration>,
    /// Latest presentation time.
    pub latest_time: Option<Duration>,
    /// Available time window.
    pub available_window: Option<Duration>,
    /// Configured buffer depth.
    pub buffer_depth: Duration,
}

impl DvrStats {
    /// Returns the buffer utilization as a ratio (0.0 to 1.0).
    #[must_use]
    pub fn utilization(&self) -> f64 {
        match self.available_window {
            Some(window) => {
                let window_secs = window.as_secs_f64();
                let depth_secs = self.buffer_depth.as_secs_f64();
                if depth_secs > 0.0 {
                    (window_secs / depth_secs).min(1.0)
                } else {
                    0.0
                }
            }
            None => 0.0,
        }
    }
}

impl DvrSegment {
    /// Creates a new DVR segment.
    #[must_use]
    pub fn new(
        number: u64,
        representation_id: impl Into<String>,
        data: Bytes,
        pts: Duration,
        duration: Duration,
        timescale: u32,
    ) -> Self {
        Self {
            number,
            representation_id: representation_id.into(),
            data,
            pts,
            duration,
            wall_clock_time: SystemTime::now(),
            timescale,
        }
    }

    /// Returns the segment size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns the end time of the segment.
    #[must_use]
    pub fn end_time(&self) -> Duration {
        self.pts + self.duration
    }

    /// Checks if this segment contains the given timestamp.
    #[must_use]
    pub fn contains_time(&self, time: Duration) -> bool {
        time >= self.pts && time < self.end_time()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_segment(number: u64, pts_secs: u64, duration_secs: u64) -> DvrSegment {
        DvrSegment::new(
            number,
            "test-repr",
            Bytes::from(vec![0u8; 1024]),
            Duration::from_secs(pts_secs),
            Duration::from_secs(duration_secs),
            90000,
        )
    }

    #[test]
    fn test_dvr_buffer_creation() {
        let buffer = DvrBuffer::new(Duration::from_secs(30));
        assert_eq!(buffer.segment_count(), 0);
        assert_eq!(buffer.total_size(), 0);
        assert_eq!(buffer.buffer_depth(), Duration::from_secs(30));
    }

    #[test]
    fn test_add_segment() {
        let mut buffer = DvrBuffer::new(Duration::from_secs(30));
        let segment = create_test_segment(1, 0, 2);

        buffer.add_segment(segment);
        assert_eq!(buffer.segment_count(), 1);
        assert_eq!(buffer.total_size(), 1024);
    }

    #[test]
    fn test_get_segment() {
        let mut buffer = DvrBuffer::new(Duration::from_secs(30));
        buffer.add_segment(create_test_segment(1, 0, 2));
        buffer.add_segment(create_test_segment(2, 2, 2));

        let seg = buffer.get_segment(1, "test-repr");
        assert!(seg.is_some());
        assert_eq!(seg.expect("should succeed in test").number, 1);

        let seg = buffer.get_segment(99, "test-repr");
        assert!(seg.is_none());
    }

    #[test]
    fn test_time_based_trimming() {
        let mut buffer = DvrBuffer::new(Duration::from_secs(10));

        // Add segments spanning 20 seconds
        for i in 0..10 {
            buffer.add_segment(create_test_segment(i + 1, i * 2, 2));
        }

        // Should have trimmed to keep only the last 10 seconds
        assert!(buffer.segment_count() <= 6);
    }

    #[test]
    fn test_earliest_latest_time() {
        let mut buffer = DvrBuffer::new(Duration::from_secs(30));
        buffer.add_segment(create_test_segment(1, 0, 2));
        buffer.add_segment(create_test_segment(2, 2, 2));
        buffer.add_segment(create_test_segment(3, 4, 2));

        assert_eq!(buffer.earliest_time(), Some(Duration::from_secs(0)));
        assert_eq!(buffer.latest_time(), Some(Duration::from_secs(6)));
    }

    #[test]
    fn test_available_window() {
        let mut buffer = DvrBuffer::new(Duration::from_secs(30));
        buffer.add_segment(create_test_segment(1, 0, 2));
        buffer.add_segment(create_test_segment(2, 2, 2));

        assert_eq!(buffer.available_window(), Some(Duration::from_secs(4)));
    }

    #[test]
    fn test_get_segments_in_range() {
        let mut buffer = DvrBuffer::new(Duration::from_secs(30));
        buffer.add_segment(create_test_segment(1, 0, 2));
        buffer.add_segment(create_test_segment(2, 2, 2));
        buffer.add_segment(create_test_segment(3, 4, 2));

        let segments = buffer.get_segments_in_range(
            Duration::from_secs(1),
            Duration::from_secs(5),
            "test-repr",
        );

        assert_eq!(segments.len(), 3);
    }

    #[test]
    fn test_clear() {
        let mut buffer = DvrBuffer::new(Duration::from_secs(30));
        buffer.add_segment(create_test_segment(1, 0, 2));
        buffer.add_segment(create_test_segment(2, 2, 2));

        buffer.clear();
        assert_eq!(buffer.segment_count(), 0);
        assert_eq!(buffer.total_size(), 0);
    }

    #[test]
    fn test_dvr_stats() {
        let mut buffer = DvrBuffer::new(Duration::from_secs(30));
        buffer.add_segment(create_test_segment(1, 0, 2));

        let stats = buffer.stats();
        assert_eq!(stats.segment_count, 1);
        assert!(stats.utilization() < 1.0);
    }

    #[test]
    fn test_max_size_constraint() {
        let mut buffer = DvrBuffer::with_max_size(Duration::from_secs(30), 2048);
        buffer.add_segment(create_test_segment(1, 0, 2));
        buffer.add_segment(create_test_segment(2, 2, 2));
        buffer.add_segment(create_test_segment(3, 4, 2));

        // Should have trimmed to stay under 2048 bytes
        assert!(buffer.total_size() <= 2048);
    }
}
