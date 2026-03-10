//! AAF effects and operations.
//!
//! This module provides structures and builders for AAF effects including
//! transitions (dissolves, wipes, dips), audio effects (gain, pan/vol),
//! color correction, and keyframe animation.

/// The type of an AAF effect or operation.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AafEffectType {
    /// Video dissolve transition.
    Dissolve,
    /// Video wipe transition.
    Wipe,
    /// Dip-to-color transition.
    Dip,
    /// Color correction operation.
    ColorCorrect,
    /// Audio gain effect.
    AudioGain,
    /// Audio pan and volume effect.
    PanVol,
}

impl std::fmt::Display for AafEffectType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Dissolve => "Dissolve",
            Self::Wipe => "Wipe",
            Self::Dip => "Dip",
            Self::ColorCorrect => "ColorCorrect",
            Self::AudioGain => "AudioGain",
            Self::PanVol => "PanVol",
        };
        write!(f, "{s}")
    }
}

/// A single parameter attached to an AAF effect.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AafParameter {
    /// Parameter name (e.g. `"Level"`, `"Angle"`).
    pub name: String,
    /// Numeric value.
    pub value: f64,
    /// Unit string (e.g. `"dB"`, `"degrees"`, `"frames"`).
    pub unit: String,
}

impl AafParameter {
    /// Create a new parameter.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(name: impl Into<String>, value: f64, unit: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value,
            unit: unit.into(),
        }
    }
}

/// Interpolation mode for keyframes.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyframeInterp {
    /// Value jumps at the keyframe (hold).
    Constant,
    /// Linear interpolation between keyframes.
    Linear,
    /// Bezier curve interpolation.
    Bezier,
}

impl std::fmt::Display for KeyframeInterp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Constant => "Constant",
            Self::Linear => "Linear",
            Self::Bezier => "Bezier",
        };
        write!(f, "{s}")
    }
}

/// A single keyframe in an animated parameter curve.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AafKeyframe {
    /// Time position (in edit units / frames).
    pub time: u64,
    /// Value at this keyframe.
    pub value: f64,
    /// Interpolation to apply from this keyframe to the next.
    pub interpolation: KeyframeInterp,
}

impl AafKeyframe {
    /// Create a new keyframe.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(time: u64, value: f64, interpolation: KeyframeInterp) -> Self {
        Self {
            time,
            value,
            interpolation,
        }
    }
}

/// A complete AAF effect, combining type, unique ID, and parameters.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AafEffect {
    /// Unique identifier for this effect instance.
    pub effect_id: u64,
    /// The kind of effect.
    pub effect_type: AafEffectType,
    /// List of parameters controlling the effect.
    pub parameters: Vec<AafParameter>,
}

impl AafEffect {
    /// Create a new effect with no parameters.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(effect_id: u64, effect_type: AafEffectType) -> Self {
        Self {
            effect_id,
            effect_type,
            parameters: Vec::new(),
        }
    }

    /// Add a parameter to this effect.
    #[allow(dead_code)]
    pub fn add_parameter(&mut self, param: AafParameter) {
        self.parameters.push(param);
    }

    /// Look up a parameter by name.
    #[allow(dead_code)]
    #[must_use]
    pub fn get_parameter(&self, name: &str) -> Option<&AafParameter> {
        self.parameters.iter().find(|p| p.name == name)
    }
}

/// Builder for constructing collections of [`AafEffect`]s and producing
/// individual effect instances.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct AafEffectBuilder {
    effects: Vec<AafEffect>,
    next_id: u64,
}

impl AafEffectBuilder {
    /// Create a new, empty builder.
    #[allow(dead_code)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
            next_id: 1,
        }
    }

    /// Create a dissolve transition of the given length (frames).
    ///
    /// The resulting effect has a `"Length"` parameter set to `length`.
    #[allow(dead_code)]
    #[must_use]
    pub fn add_dissolve(length: u64) -> AafEffect {
        let mut effect = AafEffect::new(0, AafEffectType::Dissolve);
        effect.add_parameter(AafParameter::new("Length", length as f64, "frames"));
        effect
    }

    /// Create an audio gain effect at the given level in decibels.
    ///
    /// The resulting effect has a `"Level"` parameter set to `level_db`.
    #[allow(dead_code)]
    #[must_use]
    pub fn add_gain(level_db: f64) -> AafEffect {
        let mut effect = AafEffect::new(0, AafEffectType::AudioGain);
        effect.add_parameter(AafParameter::new("Level", level_db, "dB"));
        effect
    }

    /// Create a wipe transition of the given length.
    #[allow(dead_code)]
    #[must_use]
    pub fn add_wipe(length: u64) -> AafEffect {
        let mut effect = AafEffect::new(0, AafEffectType::Wipe);
        effect.add_parameter(AafParameter::new("Length", length as f64, "frames"));
        effect
    }

    /// Create a dip-to-color transition of the given length.
    #[allow(dead_code)]
    #[must_use]
    pub fn add_dip(length: u64) -> AafEffect {
        let mut effect = AafEffect::new(0, AafEffectType::Dip);
        effect.add_parameter(AafParameter::new("Length", length as f64, "frames"));
        effect
    }

    /// Create a pan/vol audio effect.
    #[allow(dead_code)]
    #[must_use]
    pub fn add_pan_vol(pan: f64, vol_db: f64) -> AafEffect {
        let mut effect = AafEffect::new(0, AafEffectType::PanVol);
        effect.add_parameter(AafParameter::new("Pan", pan, ""));
        effect.add_parameter(AafParameter::new("Volume", vol_db, "dB"));
        effect
    }

    /// Interpolate across a keyframe array to find the value at `time`.
    ///
    /// - If `keyframes` is empty, returns `0.0`.
    /// - If `time` is before the first keyframe, returns the first value.
    /// - If `time` is after the last keyframe, returns the last value.
    /// - Between keyframes the interpolation follows the *preceding*
    ///   keyframe's [`KeyframeInterp`] mode.
    #[allow(dead_code)]
    #[must_use]
    pub fn interpolate_keyframes(keyframes: &[AafKeyframe], time: u64) -> f64 {
        if keyframes.is_empty() {
            return 0.0;
        }

        // Before first keyframe
        if time <= keyframes[0].time {
            return keyframes[0].value;
        }

        // After last keyframe
        let last = &keyframes[keyframes.len() - 1];
        if time >= last.time {
            return last.value;
        }

        // Find the surrounding pair
        for i in 0..keyframes.len() - 1 {
            let kf_a = &keyframes[i];
            let kf_b = &keyframes[i + 1];

            if time >= kf_a.time && time < kf_b.time {
                let span = (kf_b.time - kf_a.time) as f64;
                let t = (time - kf_a.time) as f64 / span; // 0.0 .. 1.0

                let result = match kf_a.interpolation {
                    KeyframeInterp::Constant => kf_a.value,
                    KeyframeInterp::Linear => kf_a.value + t * (kf_b.value - kf_a.value),
                    KeyframeInterp::Bezier => {
                        // Simple cubic smoothstep approximation
                        let smooth_t = t * t * (3.0 - 2.0 * t);
                        kf_a.value + smooth_t * (kf_b.value - kf_a.value)
                    }
                };
                return result;
            }
        }

        last.value
    }

    /// Add an effect to the internal collection, assigning it a unique ID.
    ///
    /// Returns the assigned effect ID.
    #[allow(dead_code)]
    pub fn push_effect(&mut self, mut effect: AafEffect) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        effect.effect_id = id;
        self.effects.push(effect);
        id
    }

    /// Get the list of effects held by this builder.
    #[allow(dead_code)]
    #[must_use]
    pub fn effects(&self) -> &[AafEffect] {
        &self.effects
    }

    /// Consume the builder and return all effects.
    #[allow(dead_code)]
    #[must_use]
    pub fn build(self) -> Vec<AafEffect> {
        self.effects
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- AafEffectType ---

    #[test]
    fn test_effect_type_display() {
        assert_eq!(AafEffectType::Dissolve.to_string(), "Dissolve");
        assert_eq!(AafEffectType::AudioGain.to_string(), "AudioGain");
        assert_eq!(AafEffectType::PanVol.to_string(), "PanVol");
    }

    #[test]
    fn test_effect_type_equality() {
        assert_eq!(AafEffectType::Wipe, AafEffectType::Wipe);
        assert_ne!(AafEffectType::Wipe, AafEffectType::Dissolve);
    }

    // --- AafParameter ---

    #[test]
    fn test_parameter_creation() {
        let p = AafParameter::new("Level", -6.0, "dB");
        assert_eq!(p.name, "Level");
        assert!((p.value - (-6.0)).abs() < f64::EPSILON);
        assert_eq!(p.unit, "dB");
    }

    // --- KeyframeInterp ---

    #[test]
    fn test_keyframe_interp_display() {
        assert_eq!(KeyframeInterp::Linear.to_string(), "Linear");
        assert_eq!(KeyframeInterp::Constant.to_string(), "Constant");
        assert_eq!(KeyframeInterp::Bezier.to_string(), "Bezier");
    }

    // --- AafKeyframe ---

    #[test]
    fn test_keyframe_creation() {
        let kf = AafKeyframe::new(10, 0.5, KeyframeInterp::Linear);
        assert_eq!(kf.time, 10);
        assert!((kf.value - 0.5).abs() < f64::EPSILON);
        assert_eq!(kf.interpolation, KeyframeInterp::Linear);
    }

    // --- AafEffect ---

    #[test]
    fn test_effect_creation_and_parameter() {
        let mut effect = AafEffect::new(42, AafEffectType::Dissolve);
        effect.add_parameter(AafParameter::new("Length", 25.0, "frames"));
        assert_eq!(effect.effect_id, 42);
        assert_eq!(effect.effect_type, AafEffectType::Dissolve);
        assert!(effect.get_parameter("Length").is_some());
        assert!(effect.get_parameter("Missing").is_none());
    }

    // --- AafEffectBuilder helpers ---

    #[test]
    fn test_add_dissolve() {
        let effect = AafEffectBuilder::add_dissolve(50);
        assert_eq!(effect.effect_type, AafEffectType::Dissolve);
        let len_param = effect
            .get_parameter("Length")
            .expect("len_param should be valid");
        assert!((len_param.value - 50.0).abs() < f64::EPSILON);
        assert_eq!(len_param.unit, "frames");
    }

    #[test]
    fn test_add_gain() {
        let effect = AafEffectBuilder::add_gain(-3.0);
        assert_eq!(effect.effect_type, AafEffectType::AudioGain);
        let param = effect
            .get_parameter("Level")
            .expect("param should be valid");
        assert!((param.value - (-3.0)).abs() < f64::EPSILON);
        assert_eq!(param.unit, "dB");
    }

    #[test]
    fn test_add_pan_vol() {
        let effect = AafEffectBuilder::add_pan_vol(0.5, -6.0);
        assert_eq!(effect.effect_type, AafEffectType::PanVol);
        let pan = effect.get_parameter("Pan").expect("pan should be valid");
        assert!((pan.value - 0.5).abs() < f64::EPSILON);
        let vol = effect.get_parameter("Volume").expect("vol should be valid");
        assert!((vol.value - (-6.0)).abs() < f64::EPSILON);
    }

    // --- interpolate_keyframes ---

    #[test]
    fn test_interpolate_empty() {
        let result = AafEffectBuilder::interpolate_keyframes(&[], 10);
        assert!((result - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_interpolate_before_first() {
        let kfs = vec![AafKeyframe::new(10, 1.0, KeyframeInterp::Linear)];
        let result = AafEffectBuilder::interpolate_keyframes(&kfs, 0);
        assert!((result - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_interpolate_after_last() {
        let kfs = vec![AafKeyframe::new(0, 0.0, KeyframeInterp::Linear)];
        let result = AafEffectBuilder::interpolate_keyframes(&kfs, 100);
        assert!((result - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_interpolate_linear_midpoint() {
        let kfs = vec![
            AafKeyframe::new(0, 0.0, KeyframeInterp::Linear),
            AafKeyframe::new(100, 1.0, KeyframeInterp::Linear),
        ];
        let result = AafEffectBuilder::interpolate_keyframes(&kfs, 50);
        assert!((result - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_interpolate_constant() {
        let kfs = vec![
            AafKeyframe::new(0, 0.0, KeyframeInterp::Constant),
            AafKeyframe::new(100, 1.0, KeyframeInterp::Constant),
        ];
        // With Constant interpolation, the value holds until the next keyframe.
        let result = AafEffectBuilder::interpolate_keyframes(&kfs, 50);
        assert!((result - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_interpolate_bezier_midpoint() {
        let kfs = vec![
            AafKeyframe::new(0, 0.0, KeyframeInterp::Bezier),
            AafKeyframe::new(100, 1.0, KeyframeInterp::Bezier),
        ];
        let result = AafEffectBuilder::interpolate_keyframes(&kfs, 50);
        // Smoothstep at t=0.5: 0.5*0.5*(3-2*0.5) = 0.25*2 = 0.5
        assert!((result - 0.5).abs() < 1e-9);
    }

    // --- AafEffectBuilder push/build ---

    #[test]
    fn test_builder_push_and_build() {
        let mut builder = AafEffectBuilder::new();
        let e1 = AafEffectBuilder::add_dissolve(25);
        let e2 = AafEffectBuilder::add_gain(-6.0);
        let id1 = builder.push_effect(e1);
        let id2 = builder.push_effect(e2);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(builder.effects().len(), 2);
        let effects = builder.build();
        assert_eq!(effects.len(), 2);
    }
}
