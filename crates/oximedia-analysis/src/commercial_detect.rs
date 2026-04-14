//! Commercial / advertisement break detection.
//!
//! Uses heuristics based on logo presence transitions and audio level drops to
//! detect commercial breaks within a video timeline.  The approach is designed
//! to work on the output of per-frame analysis already produced by the
//! `brand_detection` and `audio` modules.

#![allow(dead_code)]

/// Per-frame analysis summary used by the commercial detector.
///
/// Each field represents the analysis result for one frame.
#[derive(Debug, Clone)]
pub struct FrameAnalysis {
    /// Frame index (0-based).
    pub frame_idx: usize,
    /// Whether a recognisable logo was detected in this frame.
    pub logo_detected: bool,
    /// Average audio level for this frame (0.0–1.0; 0.0 = silence).
    pub audio_level: f32,
}

impl FrameAnalysis {
    /// Create a new `FrameAnalysis`.
    #[must_use]
    pub fn new(frame_idx: usize, logo_detected: bool, audio_level: f32) -> Self {
        Self {
            frame_idx,
            logo_detected,
            audio_level: audio_level.clamp(0.0, 1.0),
        }
    }
}

/// A detected commercial break segment.
#[derive(Debug, Clone)]
pub struct CommercialBreak {
    /// Index of the first frame in the commercial break.
    pub start_frame: usize,
    /// Index of the last frame in the commercial break (inclusive).
    pub end_frame: usize,
    /// Estimated duration in seconds (requires caller to provide frame rate).
    pub duration_frames: usize,
    /// Detection confidence (0.0–1.0).
    pub confidence: f32,
}

impl CommercialBreak {
    /// Compute duration in seconds given a frame rate.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self, fps: f32) -> f32 {
        self.duration_frames as f32 / fps.max(1.0)
    }
}

/// Commercial break detector.
///
/// Analyses a timeline of [`FrameAnalysis`] records and returns segments that
/// appear to be commercial breaks.
///
/// # Algorithm
///
/// A frame is flagged as a "commercial candidate" when:
/// 1. Its audio level is below a configurable threshold (audio drop heuristic), **and**
/// 2. A logo was detected in the frame (logo presence heuristic).
///
/// Consecutive candidate frames are grouped into segments.  Only segments
/// longer than `min_duration_frames` survive as actual commercial breaks.
pub struct CommercialDetector {
    /// Minimum length of a commercial break in frames.
    pub min_duration_frames: usize,
    /// Fraction of frames in a window that must show a logo to count.
    pub logo_threshold: f32,
    /// Audio level below which the frame is considered a "quiet" frame.
    pub audio_silence_threshold: f32,
}

impl CommercialDetector {
    /// Create a new `CommercialDetector` with explicit parameters.
    ///
    /// * `min_duration_s` – minimum break duration in seconds (converted to frames using 25 fps).
    /// * `logo_threshold` – fraction of frames (0.0–1.0) that must contain a logo.
    #[must_use]
    pub fn new(min_duration_s: f32, logo_threshold: f32) -> Self {
        let min_duration_frames = (min_duration_s * 25.0).round() as usize;
        Self {
            min_duration_frames: min_duration_frames.max(1),
            logo_threshold: logo_threshold.clamp(0.0, 1.0),
            audio_silence_threshold: 0.15,
        }
    }

    /// Create a detector with default parameters:
    /// * 5 seconds minimum break duration
    /// * 50% logo coverage threshold
    #[must_use]
    pub fn default_params() -> Self {
        Self::new(5.0, 0.50)
    }

    /// Detect commercial breaks in the given frame timeline.
    ///
    /// Returns a list of [`CommercialBreak`] segments.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn detect_breaks(&self, timeline: &[FrameAnalysis]) -> Vec<CommercialBreak> {
        if timeline.is_empty() {
            return Vec::new();
        }

        // Slide a window of `min_duration_frames` across the timeline.
        // A window is a commercial candidate when:
        //   - ≥ logo_threshold fraction of frames have a logo, AND
        //   - average audio level is below audio_silence_threshold.
        let win = self.min_duration_frames.max(1);
        let n = timeline.len();
        let mut is_candidate = vec![false; n];

        for i in 0..n.saturating_sub(win).saturating_add(1) {
            let end = (i + win).min(n);
            let slice = &timeline[i..end];
            let logo_count = slice.iter().filter(|f| f.logo_detected).count();
            let logo_ratio = logo_count as f32 / slice.len() as f32;
            let avg_audio = slice.iter().map(|f| f.audio_level).sum::<f32>() / slice.len() as f32;

            if logo_ratio >= self.logo_threshold && avg_audio < self.audio_silence_threshold {
                for j in i..end {
                    is_candidate[j] = true;
                }
            }
        }

        // Merge consecutive candidate frames into segments.
        let mut breaks = Vec::new();
        let mut seg_start: Option<usize> = None;

        for (idx, &cand) in is_candidate.iter().enumerate() {
            if cand && seg_start.is_none() {
                seg_start = Some(idx);
            } else if !cand {
                if let Some(start) = seg_start.take() {
                    let len = idx - start;
                    if len >= self.min_duration_frames {
                        let logo_count = timeline[start..idx]
                            .iter()
                            .filter(|f| f.logo_detected)
                            .count();
                        let confidence = logo_count as f32 / len as f32;
                        breaks.push(CommercialBreak {
                            start_frame: start,
                            end_frame: idx - 1,
                            duration_frames: len,
                            confidence,
                        });
                    }
                }
            }
        }

        // Handle segment that extends to end of timeline
        if let Some(start) = seg_start {
            let len = n - start;
            if len >= self.min_duration_frames {
                let logo_count = timeline[start..n]
                    .iter()
                    .filter(|f| f.logo_detected)
                    .count();
                let confidence = logo_count as f32 / len as f32;
                breaks.push(CommercialBreak {
                    start_frame: start,
                    end_frame: n - 1,
                    duration_frames: len,
                    confidence,
                });
            }
        }

        breaks
    }
}

impl Default for CommercialDetector {
    fn default() -> Self {
        Self::default_params()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_timeline(n: usize, logo: bool, audio: f32) -> Vec<FrameAnalysis> {
        (0..n).map(|i| FrameAnalysis::new(i, logo, audio)).collect()
    }

    #[test]
    fn test_no_breaks_in_empty_timeline() {
        let det = CommercialDetector::default();
        let breaks = det.detect_breaks(&[]);
        assert!(breaks.is_empty());
    }

    #[test]
    fn test_no_breaks_when_no_logo() {
        let timeline = make_timeline(200, false, 0.05);
        let det = CommercialDetector::new(5.0, 0.5);
        let breaks = det.detect_breaks(&timeline);
        assert!(breaks.is_empty(), "no logo → no commercial breaks");
    }

    #[test]
    fn test_no_breaks_when_audio_too_loud() {
        let timeline = make_timeline(200, true, 0.8);
        let det = CommercialDetector::new(5.0, 0.5);
        let breaks = det.detect_breaks(&timeline);
        assert!(breaks.is_empty(), "loud audio → no commercial breaks");
    }

    #[test]
    fn test_detects_break_with_logo_and_silence() {
        // 10 seconds at 25fps = 250 frames
        let timeline = make_timeline(250, true, 0.05);
        let det = CommercialDetector::new(5.0, 0.5);
        let breaks = det.detect_breaks(&timeline);
        assert!(!breaks.is_empty(), "should detect at least one break");
        assert!(breaks[0].confidence >= 0.5);
    }

    #[test]
    fn test_break_duration_seconds() {
        let b = CommercialBreak {
            start_frame: 0,
            end_frame: 124,
            duration_frames: 125,
            confidence: 0.8,
        };
        let dur = b.duration_seconds(25.0);
        assert!((dur - 5.0).abs() < 0.1, "duration={dur}");
    }

    #[test]
    fn test_frame_analysis_audio_clamp() {
        let f = FrameAnalysis::new(0, false, 1.5);
        assert!((f.audio_level - 1.0).abs() < f32::EPSILON);
        let f2 = FrameAnalysis::new(0, false, -0.1);
        assert!(f2.audio_level.abs() < f32::EPSILON);
    }
}
