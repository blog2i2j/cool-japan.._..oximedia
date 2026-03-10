//! Shot change detection for caption timing

use crate::error::Result;
use crate::types::{Caption, CaptionTrack, Duration, Timestamp};

/// Shot change detector
pub struct ShotChangeDetector {
    /// Minimum shot duration (frames)
    min_shot_duration: u32,
    /// Frame rate
    fps: f64,
}

impl ShotChangeDetector {
    /// Create a new shot change detector
    #[must_use]
    pub fn new(fps: f64) -> Self {
        Self {
            min_shot_duration: 12, // Minimum 12 frames
            fps,
        }
    }

    /// Set minimum shot duration in frames
    pub fn set_min_shot_duration(&mut self, frames: u32) {
        self.min_shot_duration = frames;
    }

    /// Detect shot changes from video frames (placeholder for actual implementation)
    pub fn detect_shot_changes(&self, _video_data: &[u8]) -> Result<Vec<Timestamp>> {
        // This would integrate with video analysis
        // Placeholder implementation
        Ok(Vec::new())
    }

    /// Snap caption boundaries to shot changes
    pub fn snap_to_shots(
        &self,
        track: &mut CaptionTrack,
        shot_changes: &[Timestamp],
        tolerance_frames: u32,
    ) -> Result<usize> {
        let tolerance =
            Duration::from_micros((f64::from(tolerance_frames) * 1_000_000.0 / self.fps) as i64);

        let mut snapped_count = 0;

        for caption in &mut track.captions {
            // Try to snap start time to nearest shot change
            if let Some(&shot_time) = self.find_nearest_shot(caption.start, shot_changes, tolerance)
            {
                caption.start = shot_time;
                snapped_count += 1;
            }

            // Try to snap end time to nearest shot change
            if let Some(&shot_time) = self.find_nearest_shot(caption.end, shot_changes, tolerance) {
                caption.end = shot_time;
                snapped_count += 1;
            }
        }

        Ok(snapped_count)
    }

    /// Find nearest shot change within tolerance
    fn find_nearest_shot<'a>(
        &self,
        timestamp: Timestamp,
        shot_changes: &'a [Timestamp],
        tolerance: Duration,
    ) -> Option<&'a Timestamp> {
        shot_changes
            .iter()
            .filter(|&&shot_time| {
                let diff = if shot_time > timestamp {
                    shot_time.duration_since(timestamp)
                } else {
                    timestamp.duration_since(shot_time)
                };
                diff <= tolerance
            })
            .min_by_key(|&&shot_time| {
                if shot_time > timestamp {
                    shot_time.duration_since(timestamp).as_micros()
                } else {
                    timestamp.duration_since(shot_time).as_micros()
                }
            })
    }

    /// Check if caption boundaries align with shots
    #[must_use]
    pub fn check_alignment(
        &self,
        caption: &Caption,
        shot_changes: &[Timestamp],
        tolerance_frames: u32,
    ) -> bool {
        let tolerance =
            Duration::from_micros((f64::from(tolerance_frames) * 1_000_000.0 / self.fps) as i64);

        let start_aligned = self
            .find_nearest_shot(caption.start, shot_changes, tolerance)
            .is_some();
        let end_aligned = self
            .find_nearest_shot(caption.end, shot_changes, tolerance)
            .is_some();

        start_aligned && end_aligned
    }

    /// Calculate shot-based metrics
    #[must_use]
    pub fn calculate_shot_metrics(
        &self,
        track: &CaptionTrack,
        shot_changes: &[Timestamp],
    ) -> ShotMetrics {
        let mut metrics = ShotMetrics::default();

        for caption in &track.captions {
            // Count shots spanned by this caption
            let shots_in_range: Vec<&Timestamp> = shot_changes
                .iter()
                .filter(|&&t| t >= caption.start && t < caption.end)
                .collect();

            let shot_count = shots_in_range.len() + 1; // +1 for the initial shot
            metrics.total_shots_spanned += shot_count;

            if shot_count > 1 {
                metrics.captions_spanning_shots += 1;
            }

            if shot_count > metrics.max_shots_per_caption {
                metrics.max_shots_per_caption = shot_count;
            }
        }

        if !track.captions.is_empty() {
            metrics.avg_shots_per_caption =
                metrics.total_shots_spanned as f64 / track.captions.len() as f64;
        }

        metrics
    }
}

/// Shot-based caption metrics
#[derive(Debug, Clone, Default)]
pub struct ShotMetrics {
    /// Total shots spanned by all captions
    pub total_shots_spanned: usize,
    /// Number of captions that span multiple shots
    pub captions_spanning_shots: usize,
    /// Maximum shots spanned by a single caption
    pub max_shots_per_caption: usize,
    /// Average shots per caption
    pub avg_shots_per_caption: f64,
}

/// Scene-based captioning utilities
pub struct SceneCaptioning;

impl SceneCaptioning {
    /// Split long captions at shot boundaries
    pub fn split_at_shots(
        track: &mut CaptionTrack,
        shot_changes: &[Timestamp],
        min_duration_ms: i64,
    ) -> Result<usize> {
        let mut new_captions = Vec::new();
        let mut split_count = 0;

        for caption in &track.captions {
            let shots_in_range: Vec<Timestamp> = shot_changes
                .iter()
                .filter(|&&t| t > caption.start && t < caption.end)
                .copied()
                .collect();

            if shots_in_range.is_empty() {
                // No shots to split at
                new_captions.push(caption.clone());
                continue;
            }

            // Split the caption at shot boundaries
            let words: Vec<&str> = caption.text.split_whitespace().collect();
            let mut current_start = caption.start;

            for &shot_time in &shots_in_range {
                let duration_ms = shot_time.duration_since(current_start).as_millis();
                if duration_ms < min_duration_ms {
                    continue; // Too short, skip
                }

                // Estimate how many words fit in this segment
                let total_duration = caption.end.duration_since(caption.start).as_millis();
                let segment_ratio = duration_ms as f64 / total_duration as f64;
                let words_in_segment = (words.len() as f64 * segment_ratio).ceil() as usize;

                if words_in_segment > 0 && words_in_segment < words.len() {
                    let segment_text = words[..words_in_segment].join(" ");
                    let mut new_caption = caption.clone();
                    new_caption.start = current_start;
                    new_caption.end = shot_time;
                    new_caption.text = segment_text;
                    new_captions.push(new_caption);

                    current_start = shot_time;
                    split_count += 1;
                }
            }

            // Add final segment
            if current_start < caption.end {
                let mut new_caption = caption.clone();
                new_caption.start = current_start;
                new_caption.end = caption.end;
                // Remaining text would need to be calculated
                new_captions.push(new_caption);
            }
        }

        track.captions = new_captions;
        Ok(split_count)
    }

    /// Merge consecutive captions within the same shot
    pub fn merge_within_shots(
        track: &mut CaptionTrack,
        shot_changes: &[Timestamp],
        max_chars: usize,
    ) -> Result<usize> {
        let mut merged_captions = Vec::new();
        let mut merge_count = 0;
        let mut i = 0;

        while i < track.captions.len() {
            let mut current = track.captions[i].clone();
            let mut j = i + 1;

            // Try to merge with following captions in the same shot
            while j < track.captions.len() {
                let next = &track.captions[j];

                // Check if they're in the same shot
                let shot_between = shot_changes
                    .iter()
                    .any(|&t| t > current.end && t < next.start);

                if shot_between {
                    break; // Different shots
                }

                // Check if merged text would be too long
                let merged_text = format!("{} {}", current.text, next.text);
                if merged_text.len() > max_chars {
                    break;
                }

                // Merge
                current.end = next.end;
                current.text = merged_text;
                merge_count += 1;
                j += 1;
            }

            merged_captions.push(current);
            i = j;
        }

        track.captions = merged_captions;
        Ok(merge_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Language;

    #[test]
    fn test_detector_creation() {
        let detector = ShotChangeDetector::new(25.0);
        assert_eq!(detector.fps, 25.0);
        assert_eq!(detector.min_shot_duration, 12);
    }

    #[test]
    fn test_snap_to_shots() {
        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_millis(1000),
                Timestamp::from_millis(3000),
                "Test".to_string(),
            ))
            .expect("operation should succeed in test");

        let shot_changes = vec![Timestamp::from_millis(990), Timestamp::from_millis(2995)];

        let detector = ShotChangeDetector::new(25.0);
        let snapped = detector
            .snap_to_shots(&mut track, &shot_changes, 5)
            .expect("operation should succeed in test");

        assert!(snapped > 0);
        // Caption should be snapped to shot changes
        assert_eq!(track.captions[0].start, Timestamp::from_millis(990));
    }

    #[test]
    fn test_shot_metrics() {
        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(0),
                Timestamp::from_secs(5),
                "Test".to_string(),
            ))
            .expect("operation should succeed in test");

        let shot_changes = vec![Timestamp::from_secs(2), Timestamp::from_secs(4)];

        let detector = ShotChangeDetector::new(25.0);
        let metrics = detector.calculate_shot_metrics(&track, &shot_changes);

        assert!(metrics.captions_spanning_shots > 0);
        assert!(metrics.max_shots_per_caption >= 1);
    }
}
