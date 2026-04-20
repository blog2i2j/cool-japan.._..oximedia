//! Depth-based compositing for ICVFX
//!
//! Provides depth map processing and depth-based alpha generation
//! for realistic compositing.

use crate::{Result, VirtualProductionError};

/// Depth processor
pub struct DepthProcessor {
    near_depth: f32,
    far_depth: f32,
}

impl DepthProcessor {
    /// Create new depth processor
    pub fn new() -> Result<Self> {
        Ok(Self {
            near_depth: 0.1,
            far_depth: 100.0,
        })
    }

    /// Process depth map to alpha values
    pub fn process(&mut self, depth: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
        if depth.len() != width * height {
            return Err(VirtualProductionError::Compositing(format!(
                "Invalid depth map size: expected {}, got {}",
                width * height,
                depth.len()
            )));
        }

        // Convert depth to alpha (0.0 = background, 1.0 = foreground)
        let alpha: Vec<f32> = depth
            .iter()
            .map(|&d| {
                if d < self.near_depth {
                    1.0
                } else if d > self.far_depth {
                    0.0
                } else {
                    1.0 - (d - self.near_depth) / (self.far_depth - self.near_depth)
                }
            })
            .collect();

        Ok(alpha)
    }

    /// Set depth range
    pub fn set_depth_range(&mut self, near: f32, far: f32) {
        self.near_depth = near;
        self.far_depth = far;
    }
}

impl Default for DepthProcessor {
    fn default() -> Self {
        Self {
            near_depth: 0.1,
            far_depth: 100.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depth_processor() {
        let processor = DepthProcessor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_depth_processing() {
        let mut processor = DepthProcessor::new().expect("should succeed in test");
        processor.set_depth_range(0.0, 10.0);

        let depth = vec![0.0, 5.0, 10.0, 15.0];
        let alpha = processor
            .process(&depth, 2, 2)
            .expect("should succeed in test");

        assert_eq!(alpha.len(), 4);
        assert_eq!(alpha[0], 1.0); // Near
        assert_eq!(alpha[1], 0.5); // Middle
        assert_eq!(alpha[2], 0.0); // Far
    }
}
