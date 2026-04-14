//! Per-scene encoding parameter selection.
//!
//! Selects optimal encoding parameters (QP offset, B-frame depth, reference
//! count, psychovisual strength) for each scene based on lookahead analysis.
//! The selector uses a rule-based decision tree that maps scene features to
//! encoder knobs, enabling content-adaptive quality targeting without a full
//! RD search.
//!
//! # Decision pipeline
//!
//! 1. Classify scene into a [`SceneClass`] using motion, complexity, and
//!    temporal correlation from [`SceneFeatures`].
//! 2. Look up a base [`SceneEncodeParams`] for that class.
//! 3. Fine-tune the parameters using scene-specific metrics (grain level,
//!    fade flag, HDR flag).
//! 4. Return the final [`SceneEncodeParams`].

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

/// Scene classification used internally by the parameter selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneClass {
    /// Very static scene (slide show, title card, talking head).
    Static,
    /// Moderate movement (conversation, slow pan, controlled camera).
    Moderate,
    /// High-motion scene (sports, action, fast camera movement).
    Action,
    /// Scene change / I-frame boundary with high intra cost.
    SceneCut,
    /// Animation / screen content with sharp edges and flat regions.
    Animation,
}

/// Scene features extracted from lookahead analysis.
#[derive(Debug, Clone)]
pub struct SceneFeatures {
    /// Average motion magnitude (sum of MV norms / number of MVs).
    pub avg_motion: f32,
    /// Spatial complexity score in \[0.0, 1.0\].
    pub spatial_complexity: f32,
    /// Temporal correlation with the previous scene (1.0 = identical).
    pub temporal_correlation: f32,
    /// Estimated film grain level in \[0.0, 1.0\].
    pub grain_level: f32,
    /// True if a brightness fade is detected.
    pub has_fade: bool,
    /// True if the content is animation / synthetic.
    pub is_animation: bool,
    /// True if this scene starts with a hard cut.
    pub is_scene_cut: bool,
}

impl Default for SceneFeatures {
    fn default() -> Self {
        Self {
            avg_motion: 0.0,
            spatial_complexity: 0.5,
            temporal_correlation: 0.8,
            grain_level: 0.0,
            has_fade: false,
            is_animation: false,
            is_scene_cut: false,
        }
    }
}

/// Encoding parameters selected for a single scene.
#[derive(Debug, Clone)]
pub struct SceneEncodeParams {
    /// QP offset relative to the base CRF/QP (negative = better quality).
    pub qp_offset: i8,
    /// Maximum B-frame depth (0 = I/P only, 3 = deep B hierarchy).
    pub b_frames: u8,
    /// Number of reference frames.
    pub ref_frames: u8,
    /// Psychovisual RD strength (higher = more perceptual tuning).
    pub psy_rd_strength: f32,
    /// Adaptive quantization strength for this scene.
    pub aq_strength: f32,
    /// Whether to enable grain synthesis preservation.
    pub preserve_grain: bool,
    /// Lookahead depth override (None = use global default).
    pub lookahead_override: Option<u32>,
}

impl Default for SceneEncodeParams {
    fn default() -> Self {
        Self {
            qp_offset: 0,
            b_frames: 3,
            ref_frames: 4,
            psy_rd_strength: 1.0,
            aq_strength: 1.0,
            preserve_grain: false,
            lookahead_override: None,
        }
    }
}

/// Selects per-scene encoding parameters based on scene features.
pub struct SceneParamSelector {
    /// QP offset budget available for content-adaptive adjustments.
    pub qp_budget: i8,
    /// Global AQ strength multiplier.
    pub aq_strength_global: f32,
    /// Maximum allowed B-frame depth.
    pub max_b_frames: u8,
}

impl Default for SceneParamSelector {
    fn default() -> Self {
        Self {
            qp_budget: 4,
            aq_strength_global: 1.0,
            max_b_frames: 3,
        }
    }
}

impl SceneParamSelector {
    /// Creates a new selector with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Classifies a scene from its features.
    #[must_use]
    pub fn classify(&self, feat: &SceneFeatures) -> SceneClass {
        if feat.is_scene_cut {
            return SceneClass::SceneCut;
        }
        if feat.is_animation {
            return SceneClass::Animation;
        }
        if feat.avg_motion < 2.0 && feat.temporal_correlation > 0.9 {
            return SceneClass::Static;
        }
        if feat.avg_motion > 20.0 || feat.spatial_complexity > 0.75 {
            return SceneClass::Action;
        }
        SceneClass::Moderate
    }

    /// Selects encoding parameters for a scene with the given features.
    #[must_use]
    pub fn select_params(&self, feat: &SceneFeatures) -> SceneEncodeParams {
        let class = self.classify(feat);
        let mut params = self.base_params(class);

        // Fine-tune based on scene-specific features.
        if feat.grain_level > 0.3 {
            params.preserve_grain = true;
            // Reduce AQ to avoid disturbing grain texture.
            params.aq_strength = (params.aq_strength * 0.7).max(0.3);
        }

        if feat.has_fade {
            // Fades benefit from a small positive QP offset (dark frames waste bits).
            params.qp_offset = (params.qp_offset + 1).min(self.qp_budget);
            params.b_frames = self.max_b_frames;
        }

        if feat.spatial_complexity > 0.8 {
            // Very complex frame: give it more bits and avoid deep B-frames.
            params.qp_offset = (params.qp_offset - 1).max(-self.qp_budget);
            params.b_frames = params.b_frames.saturating_sub(1);
        }

        // Apply global AQ multiplier.
        params.aq_strength *= self.aq_strength_global;

        params
    }

    fn base_params(&self, class: SceneClass) -> SceneEncodeParams {
        match class {
            SceneClass::Static => SceneEncodeParams {
                qp_offset: 2, // Flat scene → save bits
                b_frames: self.max_b_frames,
                ref_frames: 2,
                psy_rd_strength: 0.6,
                aq_strength: 0.8,
                preserve_grain: false,
                lookahead_override: Some(8),
            },
            SceneClass::Moderate => SceneEncodeParams {
                qp_offset: 0,
                b_frames: self.max_b_frames,
                ref_frames: 4,
                psy_rd_strength: 1.0,
                aq_strength: 1.0,
                preserve_grain: false,
                lookahead_override: None,
            },
            SceneClass::Action => SceneEncodeParams {
                qp_offset: -1, // Fast motion → give more bits
                b_frames: 1,   // Fewer B-frames for high-motion
                ref_frames: 3,
                psy_rd_strength: 1.2,
                aq_strength: 1.2,
                preserve_grain: false,
                lookahead_override: Some(16),
            },
            SceneClass::SceneCut => SceneEncodeParams {
                qp_offset: -2, // I-frame: high cost, needs bits
                b_frames: 0,
                ref_frames: 1,
                psy_rd_strength: 1.0,
                aq_strength: 1.0,
                preserve_grain: false,
                lookahead_override: Some(4),
            },
            SceneClass::Animation => SceneEncodeParams {
                qp_offset: 1,
                b_frames: self.max_b_frames,
                ref_frames: 4,
                psy_rd_strength: 0.5, // Less psy-RD for flat regions
                aq_strength: 1.5,     // Strong AQ to exploit flat areas
                preserve_grain: false,
                lookahead_override: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feat_static() -> SceneFeatures {
        SceneFeatures {
            avg_motion: 0.5,
            temporal_correlation: 0.97,
            ..Default::default()
        }
    }

    fn feat_action() -> SceneFeatures {
        SceneFeatures {
            avg_motion: 30.0,
            spatial_complexity: 0.9,
            ..Default::default()
        }
    }

    fn feat_scene_cut() -> SceneFeatures {
        SceneFeatures {
            is_scene_cut: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_classify_static() {
        let sel = SceneParamSelector::new();
        assert_eq!(sel.classify(&feat_static()), SceneClass::Static);
    }

    #[test]
    fn test_classify_action() {
        let sel = SceneParamSelector::new();
        assert_eq!(sel.classify(&feat_action()), SceneClass::Action);
    }

    #[test]
    fn test_classify_scene_cut() {
        let sel = SceneParamSelector::new();
        assert_eq!(sel.classify(&feat_scene_cut()), SceneClass::SceneCut);
    }

    #[test]
    fn test_classify_animation() {
        let sel = SceneParamSelector::new();
        let feat = SceneFeatures {
            is_animation: true,
            ..Default::default()
        };
        assert_eq!(sel.classify(&feat), SceneClass::Animation);
    }

    #[test]
    fn test_static_scene_positive_qp_offset() {
        let sel = SceneParamSelector::new();
        let params = sel.select_params(&feat_static());
        assert!(
            params.qp_offset > 0,
            "static scenes should save bits (positive QP offset)"
        );
    }

    #[test]
    fn test_action_scene_more_bits() {
        let sel = SceneParamSelector::new();
        let action_params = sel.select_params(&feat_action());
        let static_params = sel.select_params(&feat_static());
        assert!(
            action_params.qp_offset < static_params.qp_offset,
            "action should have lower (or equal) QP offset than static"
        );
    }

    #[test]
    fn test_scene_cut_no_b_frames() {
        let sel = SceneParamSelector::new();
        let params = sel.select_params(&feat_scene_cut());
        assert_eq!(params.b_frames, 0, "scene cuts should not use B-frames");
    }

    #[test]
    fn test_grain_enables_preserve_grain() {
        let sel = SceneParamSelector::new();
        let feat = SceneFeatures {
            grain_level: 0.6,
            ..Default::default()
        };
        let params = sel.select_params(&feat);
        assert!(
            params.preserve_grain,
            "high grain should enable preservation"
        );
    }

    #[test]
    fn test_qp_offset_within_budget() {
        let sel = SceneParamSelector::new();
        let feats = [feat_static(), feat_action(), feat_scene_cut()];
        for feat in &feats {
            let params = sel.select_params(feat);
            assert!(
                params.qp_offset.abs() <= sel.qp_budget,
                "qp_offset {} exceeds budget {}",
                params.qp_offset,
                sel.qp_budget
            );
        }
    }
}
