//! Caption synchronization and timing adjustment.

use crate::caption::Caption;
use crate::error::{AccessError, AccessResult};
use serde::{Deserialize, Serialize};

/// Synchronization quality level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncQuality {
    /// Basic frame-level sync (within 1 frame).
    Frame,
    /// Word-level sync (each word timed).
    Word,
    /// Phoneme-level sync (highest precision).
    Phoneme,
}

/// Caption synchronizer for timing adjustments.
pub struct CaptionSynchronizer {
    #[allow(dead_code)]
    quality: SyncQuality,
    frame_rate: f64,
}

impl CaptionSynchronizer {
    /// Create a new synchronizer.
    #[must_use]
    pub const fn new(quality: SyncQuality, frame_rate: f64) -> Self {
        Self {
            quality,
            frame_rate,
        }
    }

    /// Synchronize captions to frame boundaries.
    pub fn sync_to_frames(&self, captions: &mut [Caption]) -> AccessResult<()> {
        for caption in captions {
            caption.subtitle.start_time = self.snap_to_frame(caption.subtitle.start_time);
            caption.subtitle.end_time = self.snap_to_frame(caption.subtitle.end_time);
        }

        Ok(())
    }

    /// Snap time to nearest frame boundary using floating-point precision.
    fn snap_to_frame(&self, time_ms: i64) -> i64 {
        let frame_duration = 1000.0 / self.frame_rate;
        let frame_num = (time_ms as f64 / frame_duration).round() as i64;
        (frame_num as f64 * frame_duration).round() as i64
    }

    /// Adjust timing offset for all captions.
    pub fn adjust_offset(&self, captions: &mut [Caption], offset_ms: i64) {
        for caption in captions {
            caption.subtitle.start_time += offset_ms;
            caption.subtitle.end_time += offset_ms;
        }
    }

    /// Detect and fix timing gaps.
    pub fn fix_gaps(&self, captions: &mut [Caption], max_gap_ms: i64) -> AccessResult<usize> {
        let mut fixed_count = 0;

        for i in 0..captions.len().saturating_sub(1) {
            let gap = captions[i + 1].subtitle.start_time - captions[i].subtitle.end_time;

            if gap > max_gap_ms {
                captions[i].subtitle.end_time = captions[i + 1].subtitle.start_time - 100;
                fixed_count += 1;
            }
        }

        Ok(fixed_count)
    }

    /// Detect and fix overlapping captions.
    pub fn fix_overlaps(&self, captions: &mut [Caption]) -> AccessResult<usize> {
        let mut fixed_count = 0;

        for i in 0..captions.len().saturating_sub(1) {
            if captions[i].subtitle.end_time > captions[i + 1].subtitle.start_time {
                captions[i].subtitle.end_time = captions[i + 1].subtitle.start_time;
                fixed_count += 1;
            }
        }

        Ok(fixed_count)
    }

    /// Validate caption timing.
    pub fn validate(&self, captions: &[Caption]) -> AccessResult<()> {
        for (i, caption) in captions.iter().enumerate() {
            if caption.subtitle.end_time <= caption.subtitle.start_time {
                return Err(AccessError::SyncError(format!(
                    "Caption {} has invalid timing: start={}, end={}",
                    i, caption.subtitle.start_time, caption.subtitle.end_time
                )));
            }
        }

        for i in 0..captions.len().saturating_sub(1) {
            if captions[i].subtitle.end_time > captions[i + 1].subtitle.start_time {
                return Err(AccessError::SyncError(format!(
                    "Captions {} and {} overlap",
                    i,
                    i + 1
                )));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caption::CaptionType;
    use oximedia_subtitle::Subtitle;

    #[test]
    fn test_sync_to_frames() {
        let synchronizer = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);

        let mut captions = vec![Caption::new(
            Subtitle::new(1003, 2007, "Test".to_string()),
            CaptionType::Closed,
        )];

        synchronizer
            .sync_to_frames(&mut captions)
            .expect("sync_to_frames should succeed");

        // Should snap to 24fps frame boundaries (41.67ms per frame)
        assert_eq!(captions[0].subtitle.start_time, 1000);
    }

    #[test]
    fn test_adjust_offset() {
        let synchronizer = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);

        let mut captions = vec![Caption::new(
            Subtitle::new(1000, 2000, "Test".to_string()),
            CaptionType::Closed,
        )];

        synchronizer.adjust_offset(&mut captions, 500);

        assert_eq!(captions[0].subtitle.start_time, 1500);
        assert_eq!(captions[0].subtitle.end_time, 2500);
    }

    #[test]
    fn test_fix_overlaps() {
        let synchronizer = CaptionSynchronizer::new(SyncQuality::Frame, 24.0);

        let mut captions = vec![
            Caption::new(
                Subtitle::new(1000, 3000, "First".to_string()),
                CaptionType::Closed,
            ),
            Caption::new(
                Subtitle::new(2000, 4000, "Second".to_string()),
                CaptionType::Closed,
            ),
        ];

        let fixed = synchronizer
            .fix_overlaps(&mut captions)
            .expect("fixed should be valid");
        assert_eq!(fixed, 1);
        assert_eq!(captions[0].subtitle.end_time, 2000);
    }
}
