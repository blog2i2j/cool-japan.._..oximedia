//! Lookahead analysis for temporal optimization.

use crate::OptimizerConfig;
use oximedia_core::OxiResult;

/// Lookahead frame analysis.
#[derive(Debug, Clone)]
pub struct LookaheadFrame {
    /// Frame index.
    pub frame_idx: usize,
    /// Temporal complexity.
    pub complexity: f64,
    /// Scene change score.
    pub scene_change_score: f64,
    /// Average motion magnitude.
    pub avg_motion: f64,
    /// Whether this is likely a scene change.
    pub is_scene_change: bool,
}

impl LookaheadFrame {
    /// Creates a new lookahead frame.
    #[must_use]
    pub fn new(frame_idx: usize) -> Self {
        Self {
            frame_idx,
            complexity: 0.0,
            scene_change_score: 0.0,
            avg_motion: 0.0,
            is_scene_change: false,
        }
    }
}

/// Lookahead analyzer.
pub struct LookaheadAnalyzer {
    #[allow(dead_code)]
    buffer_size: usize,
    scene_change_threshold: f64,
    enable_complexity_analysis: bool,
}

impl LookaheadAnalyzer {
    /// Creates a new lookahead analyzer.
    pub fn new(config: &OptimizerConfig) -> OxiResult<Self> {
        Ok(Self {
            buffer_size: config.lookahead_frames,
            scene_change_threshold: 0.4,
            enable_complexity_analysis: true,
        })
    }

    /// Analyzes a sequence of frames.
    #[allow(dead_code)]
    #[must_use]
    pub fn analyze(&self, frames: &[&[u8]], width: usize, height: usize) -> Vec<LookaheadFrame> {
        let mut results = Vec::new();

        for (idx, &frame) in frames.iter().enumerate() {
            let mut analysis = LookaheadFrame::new(idx);

            if self.enable_complexity_analysis {
                analysis.complexity = self.calculate_complexity(frame);
            }

            // Scene change detection
            if idx > 0 {
                let prev_frame = frames[idx - 1];
                analysis.scene_change_score =
                    self.calculate_scene_change(prev_frame, frame, width, height);
                analysis.is_scene_change =
                    analysis.scene_change_score > self.scene_change_threshold;
            }

            // Motion estimation (simplified)
            if idx > 0 {
                let prev_frame = frames[idx - 1];
                analysis.avg_motion = self.estimate_motion(prev_frame, frame, width, height);
            }

            results.push(analysis);
        }

        results
    }

    fn calculate_complexity(&self, frame: &[u8]) -> f64 {
        if frame.is_empty() {
            return 0.0;
        }

        // Calculate variance as complexity metric
        let mean = frame.iter().map(|&p| f64::from(p)).sum::<f64>() / frame.len() as f64;
        frame
            .iter()
            .map(|&p| {
                let diff = f64::from(p) - mean;
                diff * diff
            })
            .sum::<f64>()
            / frame.len() as f64
    }

    fn calculate_scene_change(&self, prev: &[u8], curr: &[u8], width: usize, height: usize) -> f64 {
        if prev.len() != curr.len() || prev.is_empty() {
            return 0.0;
        }

        // Calculate SAD between frames
        let sad: u64 = prev
            .iter()
            .zip(curr)
            .map(|(&p, &c)| u64::from(p.abs_diff(c)))
            .sum();

        let pixels = (width * height) as f64;

        sad as f64 / (pixels * 255.0)
    }

    fn estimate_motion(&self, prev: &[u8], curr: &[u8], width: usize, height: usize) -> f64 {
        // Simplified motion estimation using block-based SAD
        const BLOCK_SIZE: usize = 16;
        let blocks_x = width / BLOCK_SIZE;
        let blocks_y = height / BLOCK_SIZE;

        if blocks_x == 0 || blocks_y == 0 {
            return 0.0;
        }

        let mut total_motion = 0.0;
        let mut block_count = 0;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let motion = self.estimate_block_motion(
                    prev,
                    curr,
                    width,
                    bx * BLOCK_SIZE,
                    by * BLOCK_SIZE,
                    BLOCK_SIZE,
                );
                total_motion += motion;
                block_count += 1;
            }
        }

        if block_count > 0 {
            total_motion / f64::from(block_count)
        } else {
            0.0
        }
    }

    fn estimate_block_motion(
        &self,
        prev: &[u8],
        curr: &[u8],
        width: usize,
        block_x: usize,
        block_y: usize,
        block_size: usize,
    ) -> f64 {
        // Simplified: just calculate SAD for the block
        let mut sad = 0u32;

        for y in 0..block_size {
            for x in 0..block_size {
                let px = block_x + x;
                let py = block_y + y;
                let idx = py * width + px;

                if idx < prev.len() {
                    sad += u32::from(prev[idx].abs_diff(curr[idx]));
                }
            }
        }

        f64::from(sad) / (block_size * block_size) as f64
    }

    /// Determines optimal GOP structure based on analysis.
    #[allow(dead_code)]
    #[must_use]
    pub fn determine_gop_structure(&self, analysis: &[LookaheadFrame]) -> GopStructure {
        let mut keyframe_positions = Vec::new();

        for frame in analysis {
            if frame.is_scene_change {
                keyframe_positions.push(frame.frame_idx);
            }
        }

        // Add first frame if not already included
        if keyframe_positions.first() != Some(&0) {
            keyframe_positions.insert(0, 0);
        }

        GopStructure {
            keyframe_positions,
            total_frames: analysis.len(),
        }
    }

    /// Calculates bit allocation for frames.
    #[allow(dead_code)]
    #[must_use]
    pub fn allocate_bits(&self, analysis: &[LookaheadFrame], total_bits: u64) -> Vec<u64> {
        if analysis.is_empty() {
            return Vec::new();
        }

        // Calculate total complexity
        let total_complexity: f64 = analysis.iter().map(|f| f.complexity + 1.0).sum();

        // Allocate bits proportional to complexity
        analysis
            .iter()
            .map(|f| {
                let proportion = (f.complexity + 1.0) / total_complexity;
                (total_bits as f64 * proportion) as u64
            })
            .collect()
    }
}

/// GOP (Group of Pictures) structure.
#[derive(Debug, Clone)]
pub struct GopStructure {
    /// Positions of keyframes.
    pub keyframe_positions: Vec<usize>,
    /// Total number of frames.
    pub total_frames: usize,
}

impl GopStructure {
    /// Gets the distance to next keyframe from a position.
    #[must_use]
    pub fn distance_to_next_keyframe(&self, position: usize) -> usize {
        for &kf_pos in &self.keyframe_positions {
            if kf_pos > position {
                return kf_pos - position;
            }
        }
        self.total_frames - position
    }

    /// Checks if a position is a keyframe.
    #[must_use]
    pub fn is_keyframe(&self, position: usize) -> bool {
        self.keyframe_positions.contains(&position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookahead_frame_creation() {
        let frame = LookaheadFrame::new(42);
        assert_eq!(frame.frame_idx, 42);
        assert_eq!(frame.complexity, 0.0);
        assert!(!frame.is_scene_change);
    }

    #[test]
    fn test_lookahead_analyzer_creation() {
        let config = OptimizerConfig::default();
        let analyzer =
            LookaheadAnalyzer::new(&config).expect("lookahead analyzer creation should succeed");
        assert_eq!(analyzer.buffer_size, config.lookahead_frames);
    }

    #[test]
    fn test_complexity_calculation() {
        let config = OptimizerConfig::default();
        let analyzer =
            LookaheadAnalyzer::new(&config).expect("lookahead analyzer creation should succeed");

        let flat = vec![128u8; 256];
        let complexity_flat = analyzer.calculate_complexity(&flat);
        assert_eq!(complexity_flat, 0.0);

        let varied: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let complexity_varied = analyzer.calculate_complexity(&varied);
        assert!(complexity_varied > 0.0);
    }

    #[test]
    fn test_scene_change_detection() {
        let config = OptimizerConfig::default();
        let analyzer =
            LookaheadAnalyzer::new(&config).expect("lookahead analyzer creation should succeed");

        let frame1 = vec![100u8; 256];
        let frame2 = vec![100u8; 256]; // Same
        let score_same = analyzer.calculate_scene_change(&frame1, &frame2, 16, 16);
        assert_eq!(score_same, 0.0);

        let frame3 = vec![200u8; 256]; // Different
        let score_diff = analyzer.calculate_scene_change(&frame1, &frame3, 16, 16);
        assert!(score_diff > 0.0);
    }

    #[test]
    fn test_gop_structure() {
        let gop = GopStructure {
            keyframe_positions: vec![0, 10, 20],
            total_frames: 30,
        };

        assert!(gop.is_keyframe(0));
        assert!(gop.is_keyframe(10));
        assert!(!gop.is_keyframe(5));

        assert_eq!(gop.distance_to_next_keyframe(0), 10);
        assert_eq!(gop.distance_to_next_keyframe(5), 5);
        assert_eq!(gop.distance_to_next_keyframe(25), 5);
    }

    #[test]
    fn test_bit_allocation() {
        let config = OptimizerConfig::default();
        let analyzer =
            LookaheadAnalyzer::new(&config).expect("lookahead analyzer creation should succeed");

        let analysis = vec![
            LookaheadFrame {
                frame_idx: 0,
                complexity: 100.0,
                scene_change_score: 0.0,
                avg_motion: 0.0,
                is_scene_change: false,
            },
            LookaheadFrame {
                frame_idx: 1,
                complexity: 200.0,
                scene_change_score: 0.0,
                avg_motion: 0.0,
                is_scene_change: false,
            },
        ];

        let allocation = analyzer.allocate_bits(&analysis, 1000);
        assert_eq!(allocation.len(), 2);
        assert!(allocation[1] > allocation[0]); // Higher complexity gets more bits
    }
}
