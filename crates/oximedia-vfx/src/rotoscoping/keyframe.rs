//! Keyframe animation for masks.

use super::bezier::{BezierMask, BezierPoint};
use crate::{EasingFunction, Frame, VfxResult};
use serde::{Deserialize, Serialize};

/// A keyframed mask at a specific time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaskKeyframe {
    /// Time in seconds.
    pub time: f64,
    /// Mask at this time.
    pub mask: BezierMask,
    /// Easing to next keyframe.
    pub easing: EasingFunction,
}

/// A mask with keyframe animation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyframedMask {
    keyframes: Vec<MaskKeyframe>,
}

impl KeyframedMask {
    /// Create a new keyframed mask.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            keyframes: Vec::new(),
        }
    }

    /// Add a keyframe.
    pub fn add_keyframe(&mut self, time: f64, mask: BezierMask, easing: EasingFunction) {
        let keyframe = MaskKeyframe { time, mask, easing };

        match self.keyframes.binary_search_by(|k| {
            k.time
                .partial_cmp(&time)
                .unwrap_or(std::cmp::Ordering::Less)
        }) {
            Ok(idx) => self.keyframes[idx] = keyframe,
            Err(idx) => self.keyframes.insert(idx, keyframe),
        }
    }

    /// Evaluate mask at given time.
    #[must_use]
    pub fn evaluate(&self, time: f64) -> Option<BezierMask> {
        if self.keyframes.is_empty() {
            return None;
        }

        if self.keyframes.len() == 1 {
            return Some(self.keyframes[0].mask.clone());
        }

        // Find surrounding keyframes
        let idx = match self.keyframes.binary_search_by(|k| {
            k.time
                .partial_cmp(&time)
                .unwrap_or(std::cmp::Ordering::Less)
        }) {
            Ok(idx) => return Some(self.keyframes[idx].mask.clone()),
            Err(idx) => idx,
        };

        if idx == 0 {
            return Some(self.keyframes[0].mask.clone());
        }

        if idx >= self.keyframes.len() {
            return Some(self.keyframes[self.keyframes.len() - 1].mask.clone());
        }

        let k1 = &self.keyframes[idx - 1];
        let k2 = &self.keyframes[idx];

        // Interpolate
        let dt = k2.time - k1.time;
        if dt <= 0.0 {
            return Some(k1.mask.clone());
        }

        let t = ((time - k1.time) / dt) as f32;
        let eased_t = k1.easing.apply(t);

        Some(self.interpolate_masks(&k1.mask, &k2.mask, eased_t))
    }

    fn interpolate_masks(&self, mask1: &BezierMask, mask2: &BezierMask, t: f32) -> BezierMask {
        let mut result = mask1.clone();

        // Interpolate feather and opacity
        result.feather = mask1.feather + (mask2.feather - mask1.feather) * t;
        result.opacity = mask1.opacity + (mask2.opacity - mask1.opacity) * t;

        // Interpolate curve points
        let len1 = mask1.curve.points().len();
        let len2 = mask2.curve.points().len();

        if len1 == len2 {
            // Simple case: same number of points
            for i in 0..len1 {
                let p1 = &mask1.curve.points()[i];
                let p2 = &mask2.curve.points()[i];

                result.curve.points_mut()[i] = BezierPoint {
                    x: p1.x + (p2.x - p1.x) * t,
                    y: p1.y + (p2.y - p1.y) * t,
                    handle_in_x: p1.handle_in_x + (p2.handle_in_x - p1.handle_in_x) * t,
                    handle_in_y: p1.handle_in_y + (p2.handle_in_y - p1.handle_in_y) * t,
                    handle_out_x: p1.handle_out_x + (p2.handle_out_x - p1.handle_out_x) * t,
                    handle_out_y: p1.handle_out_y + (p2.handle_out_y - p1.handle_out_y) * t,
                };
            }
        }
        // If point counts differ, return mask1 (could implement more sophisticated interpolation)

        result
    }

    /// Render mask at given time to frame.
    pub fn render(&self, time: f64, output: &mut Frame) -> VfxResult<()> {
        if let Some(mask) = self.evaluate(time) {
            mask.render(output)?;
        }
        Ok(())
    }

    /// Get number of keyframes.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }
}

impl Default for KeyframedMask {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotoscoping::bezier::BezierCurve;

    #[test]
    fn test_keyframed_mask_creation() {
        let mask = KeyframedMask::new();
        assert_eq!(mask.keyframe_count(), 0);
    }

    #[test]
    fn test_keyframed_mask_single_keyframe() {
        let mut mask = KeyframedMask::new();
        let bezier = BezierMask::new(BezierCurve::new());
        mask.add_keyframe(0.0, bezier, EasingFunction::Linear);

        let evaluated = mask.evaluate(0.5);
        assert!(evaluated.is_some());
    }

    #[test]
    fn test_keyframed_mask_interpolation() {
        let mut mask = KeyframedMask::new();
        let bezier1 = BezierMask::new(BezierCurve::new()).with_opacity(0.0);
        let bezier2 = BezierMask::new(BezierCurve::new()).with_opacity(1.0);

        mask.add_keyframe(0.0, bezier1, EasingFunction::Linear);
        mask.add_keyframe(1.0, bezier2, EasingFunction::Linear);

        let evaluated = mask.evaluate(0.5).expect("should succeed in test");
        assert!((evaluated.opacity - 0.5).abs() < 0.01);
    }
}
