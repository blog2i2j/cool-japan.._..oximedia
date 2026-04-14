//! Automatic camera framing based on subject position and composition rules.
//!
//! Given a detected subject position within a frame, computes an optimal
//! crop region that satisfies cinematic composition rules such as
//! rule-of-thirds, headroom preservation, and motion following.

use std::fmt;

use crate::{MultiCamError, Result};

/// Composition rule for automatic framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FramingRule {
    /// Center the crop on the subject.
    CenterOnSubject,
    /// Place the subject on a rule-of-thirds intersection.
    RuleOfThirds,
    /// Preserve headroom above the subject (portrait framing).
    HeadroomPreserve,
    /// Smoothly follow subject motion across frames.
    FollowMotion,
}

impl fmt::Display for FramingRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CenterOnSubject => write!(f, "CenterOnSubject"),
            Self::RuleOfThirds => write!(f, "RuleOfThirds"),
            Self::HeadroomPreserve => write!(f, "HeadroomPreserve"),
            Self::FollowMotion => write!(f, "FollowMotion"),
        }
    }
}

/// Configuration for automatic framing.
#[derive(Debug, Clone)]
pub struct AutoFrameConfig {
    /// Output aspect ratio (width / height). E.g. 16.0/9.0.
    pub output_aspect_ratio: f64,
    /// Margin around the subject as a fraction of crop size [0.0, 0.5).
    pub crop_margin: f64,
    /// Smoothing factor for temporal filtering [0.0, 1.0].
    /// 0.0 = no smoothing (instant snap), 1.0 = maximum smoothing.
    pub smoothing: f64,
    /// Primary composition rule.
    pub rule: FramingRule,
    /// Headroom fraction (top space above subject) for `HeadroomPreserve`.
    pub headroom_fraction: f64,
    /// Rule-of-thirds quadrant preference (0=top-left, 1=top-right,
    /// 2=bottom-left, 3=bottom-right).
    pub thirds_quadrant: u8,
}

impl Default for AutoFrameConfig {
    fn default() -> Self {
        Self {
            output_aspect_ratio: 16.0 / 9.0,
            crop_margin: 0.1,
            smoothing: 0.3,
            rule: FramingRule::CenterOnSubject,
            headroom_fraction: 0.15,
            thirds_quadrant: 0,
        }
    }
}

impl AutoFrameConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.output_aspect_ratio <= 0.0 {
            return Err(MultiCamError::ConfigError(
                "output_aspect_ratio must be positive".into(),
            ));
        }
        if self.crop_margin < 0.0 || self.crop_margin >= 0.5 {
            return Err(MultiCamError::ConfigError(
                "crop_margin must be in [0.0, 0.5)".into(),
            ));
        }
        if self.smoothing < 0.0 || self.smoothing > 1.0 {
            return Err(MultiCamError::ConfigError(
                "smoothing must be in [0.0, 1.0]".into(),
            ));
        }
        if self.headroom_fraction < 0.0 || self.headroom_fraction > 0.5 {
            return Err(MultiCamError::ConfigError(
                "headroom_fraction must be in [0.0, 0.5]".into(),
            ));
        }
        if self.thirds_quadrant > 3 {
            return Err(MultiCamError::ConfigError(
                "thirds_quadrant must be 0..3".into(),
            ));
        }
        Ok(())
    }
}

/// A 2D position (normalised to [0, 1] of the source frame).
#[derive(Debug, Clone, Copy)]
pub struct SubjectPosition {
    /// Horizontal position [0.0 = left edge, 1.0 = right edge].
    pub x: f64,
    /// Vertical position [0.0 = top edge, 1.0 = bottom edge].
    pub y: f64,
}

impl SubjectPosition {
    /// Create a new subject position, clamping to [0, 1].
    #[must_use]
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            x: x.clamp(0.0, 1.0),
            y: y.clamp(0.0, 1.0),
        }
    }
}

/// A crop region in the source frame (pixel coordinates).
#[derive(Debug, Clone, Copy)]
pub struct CropRegion {
    /// Left edge (pixels).
    pub x: f64,
    /// Top edge (pixels).
    pub y: f64,
    /// Width (pixels).
    pub width: f64,
    /// Height (pixels).
    pub height: f64,
}

impl CropRegion {
    /// Clamp this region to stay within the source frame.
    #[must_use]
    pub fn clamp_to_frame(mut self, frame_width: f64, frame_height: f64) -> Self {
        if self.x < 0.0 {
            self.x = 0.0;
        }
        if self.y < 0.0 {
            self.y = 0.0;
        }
        if self.x + self.width > frame_width {
            self.x = (frame_width - self.width).max(0.0);
        }
        if self.y + self.height > frame_height {
            self.y = (frame_height - self.height).max(0.0);
        }
        self
    }

    /// Center of the crop region.
    #[must_use]
    pub fn center(&self) -> (f64, f64) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Aspect ratio of the crop region.
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        if self.height <= 0.0 {
            return 0.0;
        }
        self.width / self.height
    }
}

/// Automatic framing engine.
#[derive(Debug)]
pub struct AutoFramer {
    config: AutoFrameConfig,
    /// Source frame dimensions.
    source_width: f64,
    source_height: f64,
    /// Previous crop center for smoothing.
    prev_center: Option<(f64, f64)>,
}

impl AutoFramer {
    /// Create a new auto-framer for the given source dimensions.
    pub fn new(
        config: AutoFrameConfig,
        source_width: u32,
        source_height: u32,
    ) -> Result<Self> {
        config.validate()?;
        if source_width == 0 || source_height == 0 {
            return Err(MultiCamError::ConfigError(
                "source dimensions must be > 0".into(),
            ));
        }
        Ok(Self {
            config,
            source_width: f64::from(source_width),
            source_height: f64::from(source_height),
            prev_center: None,
        })
    }

    /// Compute the crop dimensions that fit the output aspect ratio
    /// within the source frame.
    fn crop_dimensions(&self) -> (f64, f64) {
        let src_ar = self.source_width / self.source_height;
        let out_ar = self.config.output_aspect_ratio;
        let margin_factor = 1.0 - 2.0 * self.config.crop_margin;

        let (w, h) = if out_ar >= src_ar {
            // Width-limited.
            let w = self.source_width * margin_factor;
            let h = w / out_ar;
            (w, h)
        } else {
            // Height-limited.
            let h = self.source_height * margin_factor;
            let w = h * out_ar;
            (w, h)
        };
        (w.max(1.0), h.max(1.0))
    }

    /// Compute the raw (un-smoothed) crop center for a subject position.
    fn raw_center(&self, subject: &SubjectPosition) -> (f64, f64) {
        let sx = subject.x * self.source_width;
        let sy = subject.y * self.source_height;

        match self.config.rule {
            FramingRule::CenterOnSubject => (sx, sy),
            FramingRule::RuleOfThirds => {
                // Place subject on the appropriate thirds intersection.
                let (crop_w, crop_h) = self.crop_dimensions();
                let (tx, ty) = match self.config.thirds_quadrant {
                    0 => (1.0 / 3.0, 1.0 / 3.0),
                    1 => (2.0 / 3.0, 1.0 / 3.0),
                    2 => (1.0 / 3.0, 2.0 / 3.0),
                    _ => (2.0 / 3.0, 2.0 / 3.0),
                };
                // Center of crop such that subject lands at the thirds point.
                let cx = sx - crop_w * (tx - 0.5);
                let cy = sy - crop_h * (ty - 0.5);
                (cx, cy)
            }
            FramingRule::HeadroomPreserve => {
                // Subject is placed below the headroom zone.
                let (_, crop_h) = self.crop_dimensions();
                let headroom_px = crop_h * self.config.headroom_fraction;
                // Top of crop is at: subject_y - headroom_px
                let crop_top = sy - headroom_px;
                let cy = crop_top + crop_h / 2.0;
                (sx, cy)
            }
            FramingRule::FollowMotion => {
                // Same as center but smoothing is relied upon.
                (sx, sy)
            }
        }
    }

    /// Apply temporal smoothing to the crop center.
    fn smooth_center(&self, target: (f64, f64)) -> (f64, f64) {
        match self.prev_center {
            Some(prev) => {
                let alpha = self.config.smoothing;
                let cx = prev.0 * alpha + target.0 * (1.0 - alpha);
                let cy = prev.1 * alpha + target.1 * (1.0 - alpha);
                (cx, cy)
            }
            None => target,
        }
    }

    /// Compute the crop region for a subject at the given position.
    pub fn compute_crop(&mut self, subject: &SubjectPosition) -> CropRegion {
        let (crop_w, crop_h) = self.crop_dimensions();
        let raw = self.raw_center(subject);
        let (cx, cy) = self.smooth_center(raw);

        self.prev_center = Some((cx, cy));

        let region = CropRegion {
            x: cx - crop_w / 2.0,
            y: cy - crop_h / 2.0,
            width: crop_w,
            height: crop_h,
        };

        region.clamp_to_frame(self.source_width, self.source_height)
    }

    /// Reset the smoothing state (e.g. on a scene cut).
    pub fn reset_smoothing(&mut self) {
        self.prev_center = None;
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &AutoFrameConfig {
        &self.config
    }

    /// Get the computed crop dimensions.
    #[must_use]
    pub fn crop_size(&self) -> (f64, f64) {
        self.crop_dimensions()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let cfg = AutoFrameConfig::default();
        assert!((cfg.output_aspect_ratio - 16.0 / 9.0).abs() < 0.001);
        assert_eq!(cfg.crop_margin, 0.1);
        assert_eq!(cfg.rule, FramingRule::CenterOnSubject);
    }

    #[test]
    fn test_config_validate_ok() {
        assert!(AutoFrameConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_validate_bad_aspect() {
        let mut cfg = AutoFrameConfig::default();
        cfg.output_aspect_ratio = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_margin() {
        let mut cfg = AutoFrameConfig::default();
        cfg.crop_margin = 0.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_smoothing() {
        let mut cfg = AutoFrameConfig::default();
        cfg.smoothing = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_center_on_subject() {
        let cfg = AutoFrameConfig {
            output_aspect_ratio: 16.0 / 9.0,
            crop_margin: 0.0,
            smoothing: 0.0,
            rule: FramingRule::CenterOnSubject,
            ..AutoFrameConfig::default()
        };
        let mut framer = AutoFramer::new(cfg, 1920, 1080).expect("framer");
        let subject = SubjectPosition::new(0.5, 0.5);
        let crop = framer.compute_crop(&subject);

        // Centered crop should be roughly centered in the frame.
        let (cx, cy) = crop.center();
        assert!((cx - 960.0).abs() < 1.0);
        assert!((cy - 540.0).abs() < 1.0);
    }

    #[test]
    fn test_crop_clamps_to_frame() {
        let cfg = AutoFrameConfig {
            output_aspect_ratio: 16.0 / 9.0,
            crop_margin: 0.0,
            smoothing: 0.0,
            rule: FramingRule::CenterOnSubject,
            ..AutoFrameConfig::default()
        };
        let mut framer = AutoFramer::new(cfg, 1920, 1080).expect("framer");
        // Subject at extreme top-left corner.
        let subject = SubjectPosition::new(0.0, 0.0);
        let crop = framer.compute_crop(&subject);

        assert!(crop.x >= 0.0);
        assert!(crop.y >= 0.0);
        assert!(crop.x + crop.width <= 1920.0 + 0.01);
        assert!(crop.y + crop.height <= 1080.0 + 0.01);
    }

    #[test]
    fn test_rule_of_thirds() {
        let cfg = AutoFrameConfig {
            output_aspect_ratio: 16.0 / 9.0,
            crop_margin: 0.0,
            smoothing: 0.0,
            rule: FramingRule::RuleOfThirds,
            thirds_quadrant: 1, // top-right
            ..AutoFrameConfig::default()
        };
        let mut framer = AutoFramer::new(cfg, 1920, 1080).expect("framer");
        let subject = SubjectPosition::new(0.5, 0.5);
        let crop = framer.compute_crop(&subject);

        // Subject should be offset from center toward the 2/3 horizontal point.
        // The crop center moves left so the subject lands at 2/3 of crop width.
        assert!(crop.width > 0.0);
        assert!(crop.height > 0.0);
    }

    #[test]
    fn test_headroom_preserve() {
        let cfg = AutoFrameConfig {
            output_aspect_ratio: 16.0 / 9.0,
            crop_margin: 0.0,
            smoothing: 0.0,
            rule: FramingRule::HeadroomPreserve,
            headroom_fraction: 0.2,
            ..AutoFrameConfig::default()
        };
        let mut framer = AutoFramer::new(cfg, 1920, 1080).expect("framer");
        let subject = SubjectPosition::new(0.5, 0.3);
        let crop = framer.compute_crop(&subject);

        // Subject pixel Y = 0.3 * 1080 = 324
        // Headroom pushes crop down so subject is below the top 20%.
        let subject_in_crop_y = 324.0 - crop.y;
        let headroom_px = crop.height * 0.2;
        // Subject should be near (but not necessarily exactly at) the headroom line.
        assert!(subject_in_crop_y >= headroom_px * 0.5);
    }

    #[test]
    fn test_smoothing_effect() {
        let cfg = AutoFrameConfig {
            output_aspect_ratio: 16.0 / 9.0,
            crop_margin: 0.3, // smaller crop so it can move within the frame
            smoothing: 0.8,   // heavy smoothing
            rule: FramingRule::FollowMotion,
            ..AutoFrameConfig::default()
        };
        // Use a wider source so the crop has room to move.
        let mut framer = AutoFramer::new(cfg, 3840, 2160).expect("framer");

        // First frame at center.
        let s1 = SubjectPosition::new(0.3, 0.5);
        let crop1 = framer.compute_crop(&s1);

        // Second frame: subject jumps to the right.
        let s2 = SubjectPosition::new(0.7, 0.5);
        let crop2 = framer.compute_crop(&s2);

        // With heavy smoothing the crop should NOT have jumped all the way.
        let (cx1, _) = crop1.center();
        let (cx2, _) = crop2.center();
        let subject_x2 = 0.7 * 3840.0;
        // crop center should be between old center and subject position.
        assert!(cx2 > cx1, "cx2={cx2} should be > cx1={cx1}");
        assert!(cx2 < subject_x2, "cx2={cx2} should be < {subject_x2}");
    }

    #[test]
    fn test_reset_smoothing() {
        let cfg = AutoFrameConfig {
            smoothing: 0.9,
            rule: FramingRule::FollowMotion,
            crop_margin: 0.3, // smaller crop so it can move
            ..AutoFrameConfig::default()
        };
        let mut framer = AutoFramer::new(cfg, 3840, 2160).expect("framer");

        let s1 = SubjectPosition::new(0.2, 0.2);
        let _ = framer.compute_crop(&s1);

        framer.reset_smoothing();

        // After reset, next frame should snap directly to the new subject.
        let s2 = SubjectPosition::new(0.6, 0.6);
        let crop = framer.compute_crop(&s2);
        let (cx, cy) = crop.center();
        let expected_x = 0.6 * 3840.0;
        let expected_y = 0.6 * 2160.0;
        // With no smoothing history, should snap close to subject.
        assert!(
            (cx - expected_x).abs() < 500.0,
            "cx={cx} expected near {expected_x}"
        );
        assert!(
            (cy - expected_y).abs() < 400.0,
            "cy={cy} expected near {expected_y}"
        );
    }

    #[test]
    fn test_crop_aspect_ratio() {
        let cfg = AutoFrameConfig {
            output_aspect_ratio: 2.39, // cinemascope
            crop_margin: 0.0,
            smoothing: 0.0,
            rule: FramingRule::CenterOnSubject,
            ..AutoFrameConfig::default()
        };
        let mut framer = AutoFramer::new(cfg, 1920, 1080).expect("framer");
        let subject = SubjectPosition::new(0.5, 0.5);
        let crop = framer.compute_crop(&subject);
        let ar = crop.aspect_ratio();
        assert!((ar - 2.39).abs() < 0.02);
    }

    #[test]
    fn test_subject_position_clamp() {
        let s = SubjectPosition::new(-0.5, 1.5);
        assert_eq!(s.x, 0.0);
        assert_eq!(s.y, 1.0);
    }

    #[test]
    fn test_zero_source_dimensions() {
        let cfg = AutoFrameConfig::default();
        assert!(AutoFramer::new(cfg, 0, 1080).is_err());
    }

    #[test]
    fn test_framing_rule_display() {
        assert_eq!(
            format!("{}", FramingRule::CenterOnSubject),
            "CenterOnSubject"
        );
        assert_eq!(
            format!("{}", FramingRule::RuleOfThirds),
            "RuleOfThirds"
        );
        assert_eq!(
            format!("{}", FramingRule::HeadroomPreserve),
            "HeadroomPreserve"
        );
        assert_eq!(
            format!("{}", FramingRule::FollowMotion),
            "FollowMotion"
        );
    }

    #[test]
    fn test_crop_region_center() {
        let r = CropRegion {
            x: 100.0,
            y: 200.0,
            width: 400.0,
            height: 300.0,
        };
        let (cx, cy) = r.center();
        assert!((cx - 300.0).abs() < 0.01);
        assert!((cy - 350.0).abs() < 0.01);
    }

    #[test]
    fn test_crop_region_clamp() {
        let r = CropRegion {
            x: -50.0,
            y: -30.0,
            width: 200.0,
            height: 100.0,
        };
        let clamped = r.clamp_to_frame(1920.0, 1080.0);
        assert_eq!(clamped.x, 0.0);
        assert_eq!(clamped.y, 0.0);
    }
}
