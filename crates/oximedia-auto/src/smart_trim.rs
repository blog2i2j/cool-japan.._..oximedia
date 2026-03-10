//! Intelligent clip trimming — suggests in/out points based on content analysis.

#![allow(dead_code)]

/// Semantic category of a trim suggestion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrimPoint {
    /// Detected scene boundary.
    SceneChange,
    /// Start of a silent region.
    SilenceStart,
    /// End of a silent region.
    SilenceEnd,
    /// Peak of action / motion.
    ActionPeak,
    /// Manually specified in-point.
    ManualIn,
    /// Manually specified out-point.
    ManualOut,
}

impl TrimPoint {
    /// Returns `true` for automatically detected trim points.
    pub fn is_automatic(&self) -> bool {
        !matches!(self, TrimPoint::ManualIn | TrimPoint::ManualOut)
    }
}

/// A candidate trim suggestion at a specific frame.
#[derive(Debug, Clone, PartialEq)]
pub struct TrimSuggestion {
    /// Frame index for this trim point.
    pub frame: u64,
    /// What kind of trim point this is.
    pub point_type: TrimPoint,
    /// Confidence in this suggestion (0.0 – 1.0).
    pub confidence: f32,
}

impl TrimSuggestion {
    /// Returns `true` when `confidence >= threshold`.
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

/// Produces trim suggestions from scene-change and silence data.
pub struct SmartTrimmer {
    /// Sensitivity for scene-change trim points (0.0 – 1.0).
    pub scene_change_sensitivity: f32,
    /// dB level below which audio is considered silent (negative, e.g. -40.0).
    pub silence_threshold_db: f32,
}

impl SmartTrimmer {
    /// Create a trimmer with default parameters.
    pub fn default() -> Self {
        Self {
            scene_change_sensitivity: 0.75,
            silence_threshold_db: -40.0,
        }
    }

    /// Generate sorted trim suggestions.
    ///
    /// - Every entry in `scene_changes` produces a `SceneChange` suggestion with confidence
    ///   equal to `scene_change_sensitivity`.
    /// - Every entry in `silence_regions` produces a `SilenceStart` at `region.0` and a
    ///   `SilenceEnd` at `region.1`, both with confidence `0.85`.
    /// - Frame boundaries (frame 0 and `frame_count - 1`) are added as `ManualIn` /
    ///   `ManualOut` with confidence `1.0` when `frame_count > 0`.
    /// - Results are sorted ascending by frame; ties are stable in insertion order.
    pub fn find_trim_points(
        &self,
        frame_count: u64,
        scene_changes: &[u64],
        silence_regions: &[(u64, u64)],
    ) -> Vec<TrimSuggestion> {
        let mut suggestions: Vec<TrimSuggestion> = Vec::new();

        // Boundaries
        if frame_count > 0 {
            suggestions.push(TrimSuggestion {
                frame: 0,
                point_type: TrimPoint::ManualIn,
                confidence: 1.0,
            });
            suggestions.push(TrimSuggestion {
                frame: frame_count - 1,
                point_type: TrimPoint::ManualOut,
                confidence: 1.0,
            });
        }

        // Scene changes
        for &f in scene_changes {
            suggestions.push(TrimSuggestion {
                frame: f,
                point_type: TrimPoint::SceneChange,
                confidence: self.scene_change_sensitivity,
            });
        }

        // Silence regions
        for &(start, end) in silence_regions {
            suggestions.push(TrimSuggestion {
                frame: start,
                point_type: TrimPoint::SilenceStart,
                confidence: 0.85,
            });
            suggestions.push(TrimSuggestion {
                frame: end,
                point_type: TrimPoint::SilenceEnd,
                confidence: 0.85,
            });
        }

        // Sort by frame ascending
        suggestions.sort_by_key(|s| s.frame);
        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- TrimPoint tests ----

    #[test]
    fn test_scene_change_is_automatic() {
        assert!(TrimPoint::SceneChange.is_automatic());
    }

    #[test]
    fn test_silence_start_is_automatic() {
        assert!(TrimPoint::SilenceStart.is_automatic());
    }

    #[test]
    fn test_silence_end_is_automatic() {
        assert!(TrimPoint::SilenceEnd.is_automatic());
    }

    #[test]
    fn test_action_peak_is_automatic() {
        assert!(TrimPoint::ActionPeak.is_automatic());
    }

    #[test]
    fn test_manual_in_not_automatic() {
        assert!(!TrimPoint::ManualIn.is_automatic());
    }

    #[test]
    fn test_manual_out_not_automatic() {
        assert!(!TrimPoint::ManualOut.is_automatic());
    }

    // ---- TrimSuggestion tests ----

    #[test]
    fn test_is_confident_above_threshold() {
        let s = TrimSuggestion {
            frame: 10,
            point_type: TrimPoint::SceneChange,
            confidence: 0.9,
        };
        assert!(s.is_confident(0.8));
    }

    #[test]
    fn test_is_confident_below_threshold() {
        let s = TrimSuggestion {
            frame: 10,
            point_type: TrimPoint::SceneChange,
            confidence: 0.3,
        };
        assert!(!s.is_confident(0.8));
    }

    #[test]
    fn test_is_confident_at_exact_threshold() {
        let s = TrimSuggestion {
            frame: 5,
            point_type: TrimPoint::SilenceEnd,
            confidence: 0.5,
        };
        assert!(s.is_confident(0.5));
    }

    // ---- SmartTrimmer tests ----

    #[test]
    fn test_default_trimmer_fields() {
        let t = SmartTrimmer::default();
        assert!((t.scene_change_sensitivity - 0.75).abs() < 1e-6);
        assert!((t.silence_threshold_db + 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_trim_points_empty_frame_count() {
        let t = SmartTrimmer::default();
        let result = t.find_trim_points(0, &[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_find_trim_points_boundaries_only() {
        let t = SmartTrimmer::default();
        let result = t.find_trim_points(100, &[], &[]);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].frame, 0);
        assert!(matches!(result[0].point_type, TrimPoint::ManualIn));
        assert_eq!(result[1].frame, 99);
        assert!(matches!(result[1].point_type, TrimPoint::ManualOut));
    }

    #[test]
    fn test_find_trim_points_scene_changes_added() {
        let t = SmartTrimmer::default();
        let result = t.find_trim_points(200, &[50, 100], &[]);
        // 2 boundaries + 2 scene changes
        assert_eq!(result.len(), 4);
        let frames: Vec<u64> = result.iter().map(|s| s.frame).collect();
        assert!(frames.contains(&50));
        assert!(frames.contains(&100));
    }

    #[test]
    fn test_find_trim_points_silence_regions_added() {
        let t = SmartTrimmer::default();
        let result = t.find_trim_points(300, &[], &[(30, 60), (120, 150)]);
        // 2 boundaries + 4 silence points
        assert_eq!(result.len(), 6);
        let frames: Vec<u64> = result.iter().map(|s| s.frame).collect();
        assert!(frames.contains(&30));
        assert!(frames.contains(&60));
        assert!(frames.contains(&120));
        assert!(frames.contains(&150));
    }

    #[test]
    fn test_find_trim_points_sorted_ascending() {
        let t = SmartTrimmer::default();
        let result = t.find_trim_points(500, &[300, 100, 200], &[(50, 80)]);
        let frames: Vec<u64> = result.iter().map(|s| s.frame).collect();
        for window in frames.windows(2) {
            assert!(window[0] <= window[1], "not sorted: {frames:?}");
        }
    }

    #[test]
    fn test_scene_change_confidence_equals_sensitivity() {
        let t = SmartTrimmer {
            scene_change_sensitivity: 0.6,
            silence_threshold_db: -40.0,
        };
        let result = t.find_trim_points(100, &[25], &[]);
        let sc = result
            .iter()
            .find(|s| matches!(s.point_type, TrimPoint::SceneChange))
            .expect("test expectation failed");
        assert!((sc.confidence - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_silence_confidence_is_fixed() {
        let t = SmartTrimmer::default();
        let result = t.find_trim_points(200, &[], &[(40, 90)]);
        for s in &result {
            if matches!(
                s.point_type,
                TrimPoint::SilenceStart | TrimPoint::SilenceEnd
            ) {
                assert!((s.confidence - 0.85).abs() < 1e-6);
            }
        }
    }
}
