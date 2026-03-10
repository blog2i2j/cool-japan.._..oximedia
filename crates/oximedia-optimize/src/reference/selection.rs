//! Reference frame selection optimization.

/// Reference frame score.
#[derive(Debug, Clone, Copy)]
pub struct RefFrameScore {
    /// Frame index.
    pub frame_idx: usize,
    /// Prediction quality score.
    pub quality_score: f64,
    /// Temporal distance.
    pub temporal_distance: i32,
    /// Total score (higher is better).
    pub total_score: f64,
}

impl RefFrameScore {
    /// Creates a new reference frame score.
    #[must_use]
    pub fn new(frame_idx: usize, quality_score: f64, temporal_distance: i32) -> Self {
        // Prefer closer frames and higher quality
        let distance_penalty = f64::from(temporal_distance.abs()) * 0.1;
        let total_score = quality_score - distance_penalty;

        Self {
            frame_idx,
            quality_score,
            temporal_distance,
            total_score,
        }
    }
}

/// Reference selection optimizer.
pub struct ReferenceSelection {
    max_references: usize,
    #[allow(dead_code)]
    temporal_bias: f64,
}

impl Default for ReferenceSelection {
    fn default() -> Self {
        Self::new(3, 0.1)
    }
}

impl ReferenceSelection {
    /// Creates a new reference selector.
    #[must_use]
    pub fn new(max_references: usize, temporal_bias: f64) -> Self {
        Self {
            max_references,
            temporal_bias,
        }
    }

    /// Selects best reference frames.
    #[allow(dead_code)]
    #[must_use]
    pub fn select_references(
        &self,
        current_poc: i32,
        available_frames: &[(usize, i32)], // (index, POC)
        src: &[u8],
    ) -> Vec<usize> {
        let mut scores: Vec<RefFrameScore> = available_frames
            .iter()
            .map(|&(idx, poc)| {
                let temporal_distance = current_poc - poc;
                let quality = self.estimate_quality(src, idx);
                RefFrameScore::new(idx, quality, temporal_distance)
            })
            .collect();

        // Sort by total score (descending)
        scores.sort_by(|a, b| {
            b.total_score
                .partial_cmp(&a.total_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top N references
        scores
            .iter()
            .take(self.max_references)
            .map(|s| s.frame_idx)
            .collect()
    }

    fn estimate_quality(&self, _src: &[u8], _frame_idx: usize) -> f64 {
        // Simplified quality estimation
        // In production, would compare actual pixels
        100.0
    }

    /// Determines if a frame should be kept as reference.
    #[must_use]
    pub fn should_keep_as_reference(&self, frame_type: FrameType, layer: u8) -> bool {
        match frame_type {
            FrameType::Key => true,         // Always keep key frames
            FrameType::Inter => layer == 0, // Keep base layer inter frames
        }
    }
}

/// Frame types for reference management.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// Key frame (I-frame).
    Key,
    /// Inter frame (P/B-frame).
    Inter,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ref_frame_score() {
        let score = RefFrameScore::new(0, 100.0, 2);
        assert_eq!(score.frame_idx, 0);
        assert_eq!(score.quality_score, 100.0);
        assert_eq!(score.temporal_distance, 2);
        assert!(score.total_score < 100.0); // Has distance penalty
    }

    #[test]
    fn test_reference_selection_creation() {
        let selector = ReferenceSelection::default();
        assert_eq!(selector.max_references, 3);
    }

    #[test]
    fn test_select_references() {
        let selector = ReferenceSelection::default();
        let available = vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)];
        let src = vec![128u8; 64];
        let selected = selector.select_references(5, &available, &src);
        assert!(selected.len() <= 3);
    }

    #[test]
    fn test_should_keep_reference() {
        let selector = ReferenceSelection::default();
        assert!(selector.should_keep_as_reference(FrameType::Key, 0));
        assert!(selector.should_keep_as_reference(FrameType::Inter, 0));
        assert!(!selector.should_keep_as_reference(FrameType::Inter, 1));
    }
}
