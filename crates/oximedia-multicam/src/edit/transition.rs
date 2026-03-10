//! Transition effects between camera angles.

use super::TransitionType;
use crate::FrameNumber;

/// Transition engine
#[derive(Debug)]
pub struct TransitionEngine {
    /// Current transition
    current_transition: Option<ActiveTransition>,
}

/// Active transition state
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ActiveTransition {
    /// Transition type
    transition_type: TransitionType,
    /// Start frame
    start_frame: FrameNumber,
    /// Duration in frames
    duration: u32,
    /// Source angle
    from_angle: usize,
    /// Target angle
    to_angle: usize,
}

impl TransitionEngine {
    /// Create a new transition engine
    #[must_use]
    pub fn new() -> Self {
        Self {
            current_transition: None,
        }
    }

    /// Start a new transition
    pub fn start_transition(
        &mut self,
        transition_type: TransitionType,
        start_frame: FrameNumber,
        duration: u32,
        from_angle: usize,
        to_angle: usize,
    ) {
        self.current_transition = Some(ActiveTransition {
            transition_type,
            start_frame,
            duration,
            from_angle,
            to_angle,
        });
    }

    /// Check if currently in a transition
    #[must_use]
    pub fn is_transitioning(&self, current_frame: FrameNumber) -> bool {
        if let Some(ref transition) = self.current_transition {
            current_frame >= transition.start_frame
                && current_frame < transition.start_frame + u64::from(transition.duration)
        } else {
            false
        }
    }

    /// Get transition progress (0.0 to 1.0)
    #[must_use]
    pub fn get_progress(&self, current_frame: FrameNumber) -> f32 {
        if let Some(ref transition) = self.current_transition {
            if current_frame < transition.start_frame {
                return 0.0;
            }
            if current_frame >= transition.start_frame + u64::from(transition.duration) {
                return 1.0;
            }

            let elapsed = current_frame - transition.start_frame;
            (elapsed as f32 / transition.duration as f32).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Get mix level for source angle (1.0 to 0.0)
    #[must_use]
    pub fn get_source_mix(&self, current_frame: FrameNumber) -> f32 {
        1.0 - self.get_target_mix(current_frame)
    }

    /// Get mix level for target angle (0.0 to 1.0)
    #[must_use]
    pub fn get_target_mix(&self, current_frame: FrameNumber) -> f32 {
        let progress = self.get_progress(current_frame);

        if let Some(ref transition) = self.current_transition {
            match transition.transition_type {
                TransitionType::Cut => 1.0,
                TransitionType::Dissolve => self.dissolve_curve(progress),
                TransitionType::Wipe => progress,
                TransitionType::DipToBlack => self.dip_to_black_curve(progress),
            }
        } else {
            1.0
        }
    }

    /// Dissolve curve (linear)
    fn dissolve_curve(&self, t: f32) -> f32 {
        t
    }

    /// Dip to black curve (parabolic)
    fn dip_to_black_curve(&self, t: f32) -> f32 {
        if t < 0.5 {
            // Fade out first half
            0.0
        } else {
            // Fade in second half
            (t - 0.5) * 2.0
        }
    }

    /// Get wipe position for wipe transition
    #[must_use]
    pub fn get_wipe_position(&self, current_frame: FrameNumber, width: u32) -> u32 {
        let progress = self.get_progress(current_frame);
        (width as f32 * progress) as u32
    }

    /// Clear current transition
    pub fn clear(&mut self) {
        self.current_transition = None;
    }

    /// Get current transition type
    #[must_use]
    pub fn current_transition_type(&self) -> Option<TransitionType> {
        self.current_transition.as_ref().map(|t| t.transition_type)
    }

    /// Apply easing function to progress
    #[must_use]
    pub fn apply_easing(&self, progress: f32, easing: EasingFunction) -> f32 {
        match easing {
            EasingFunction::Linear => progress,
            EasingFunction::EaseIn => progress * progress,
            EasingFunction::EaseOut => progress * (2.0 - progress),
            EasingFunction::EaseInOut => {
                if progress < 0.5 {
                    2.0 * progress * progress
                } else {
                    -1.0 + (4.0 - 2.0 * progress) * progress
                }
            }
            EasingFunction::Cubic => progress * progress * progress,
            EasingFunction::Sine => {
                let pi = std::f32::consts::PI;
                1.0 - ((progress * pi / 2.0).cos())
            }
        }
    }
}

impl Default for TransitionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Easing functions for transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EasingFunction {
    /// Linear interpolation
    Linear,
    /// Ease in (accelerate)
    EaseIn,
    /// Ease out (decelerate)
    EaseOut,
    /// Ease in and out
    EaseInOut,
    /// Cubic easing
    Cubic,
    /// Sine easing
    Sine,
}

/// Transition preset
#[derive(Debug, Clone)]
pub struct TransitionPreset {
    /// Preset name
    pub name: String,
    /// Transition type
    pub transition_type: TransitionType,
    /// Default duration (frames)
    pub duration: u32,
    /// Easing function
    pub easing: EasingFunction,
}

impl TransitionPreset {
    /// Create a dissolve preset
    #[must_use]
    pub fn dissolve(duration: u32) -> Self {
        Self {
            name: "Dissolve".to_string(),
            transition_type: TransitionType::Dissolve,
            duration,
            easing: EasingFunction::Linear,
        }
    }

    /// Create a wipe preset
    #[must_use]
    pub fn wipe(duration: u32) -> Self {
        Self {
            name: "Wipe".to_string(),
            transition_type: TransitionType::Wipe,
            duration,
            easing: EasingFunction::Linear,
        }
    }

    /// Create a dip to black preset
    #[must_use]
    pub fn dip_to_black(duration: u32) -> Self {
        Self {
            name: "Dip to Black".to_string(),
            transition_type: TransitionType::DipToBlack,
            duration,
            easing: EasingFunction::EaseInOut,
        }
    }

    /// Create a cut preset
    #[must_use]
    pub fn cut() -> Self {
        Self {
            name: "Cut".to_string(),
            transition_type: TransitionType::Cut,
            duration: 0,
            easing: EasingFunction::Linear,
        }
    }
}

/// Transition library
#[derive(Debug)]
pub struct TransitionLibrary {
    /// Available presets
    presets: Vec<TransitionPreset>,
}

impl TransitionLibrary {
    /// Create a new transition library with default presets
    #[must_use]
    pub fn new() -> Self {
        let mut library = Self {
            presets: Vec::new(),
        };

        // Add default presets
        library.add_preset(TransitionPreset::cut());
        library.add_preset(TransitionPreset::dissolve(25));
        library.add_preset(TransitionPreset::wipe(25));
        library.add_preset(TransitionPreset::dip_to_black(50));

        library
    }

    /// Add a preset
    pub fn add_preset(&mut self, preset: TransitionPreset) {
        self.presets.push(preset);
    }

    /// Get preset by name
    #[must_use]
    pub fn get_preset(&self, name: &str) -> Option<&TransitionPreset> {
        self.presets.iter().find(|p| p.name == name)
    }

    /// Get all presets
    #[must_use]
    pub fn presets(&self) -> &[TransitionPreset] {
        &self.presets
    }

    /// Remove preset by name
    pub fn remove_preset(&mut self, name: &str) -> bool {
        if let Some(pos) = self.presets.iter().position(|p| p.name == name) {
            self.presets.remove(pos);
            true
        } else {
            false
        }
    }
}

impl Default for TransitionLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_engine_creation() {
        let engine = TransitionEngine::new();
        assert!(!engine.is_transitioning(0));
    }

    #[test]
    fn test_transition_progress() {
        let mut engine = TransitionEngine::new();
        engine.start_transition(TransitionType::Dissolve, 100, 10, 0, 1);

        assert_eq!(engine.get_progress(99), 0.0);
        assert_eq!(engine.get_progress(100), 0.0);
        assert_eq!(engine.get_progress(105), 0.5);
        assert_eq!(engine.get_progress(109), 0.9);
        assert_eq!(engine.get_progress(110), 1.0);
    }

    #[test]
    fn test_mix_levels() {
        let mut engine = TransitionEngine::new();
        engine.start_transition(TransitionType::Dissolve, 100, 10, 0, 1);

        // At 50% progress
        assert!((engine.get_source_mix(105) - 0.5).abs() < 0.01);
        assert!((engine.get_target_mix(105) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_wipe_position() {
        let mut engine = TransitionEngine::new();
        engine.start_transition(TransitionType::Wipe, 100, 10, 0, 1);

        let pos = engine.get_wipe_position(105, 1920);
        assert_eq!(pos, 960); // 50% of 1920
    }

    #[test]
    fn test_easing_functions() {
        let engine = TransitionEngine::new();

        assert_eq!(engine.apply_easing(0.5, EasingFunction::Linear), 0.5);
        assert!(engine.apply_easing(0.5, EasingFunction::EaseIn) < 0.5);
        assert!(engine.apply_easing(0.5, EasingFunction::EaseOut) > 0.5);
    }

    #[test]
    fn test_transition_presets() {
        let dissolve = TransitionPreset::dissolve(25);
        assert_eq!(dissolve.transition_type, TransitionType::Dissolve);
        assert_eq!(dissolve.duration, 25);

        let cut = TransitionPreset::cut();
        assert_eq!(cut.transition_type, TransitionType::Cut);
        assert_eq!(cut.duration, 0);
    }

    #[test]
    fn test_transition_library() {
        let library = TransitionLibrary::new();
        assert!(!library.presets().is_empty());

        let cut = library.get_preset("Cut");
        assert!(cut.is_some());
        assert_eq!(
            cut.expect("multicam test operation should succeed")
                .transition_type,
            TransitionType::Cut
        );
    }

    #[test]
    fn test_library_add_remove() {
        let mut library = TransitionLibrary::new();
        let initial_count = library.presets().len();

        let custom = TransitionPreset {
            name: "Custom".to_string(),
            transition_type: TransitionType::Dissolve,
            duration: 15,
            easing: EasingFunction::EaseIn,
        };

        library.add_preset(custom);
        assert_eq!(library.presets().len(), initial_count + 1);

        assert!(library.remove_preset("Custom"));
        assert_eq!(library.presets().len(), initial_count);
    }
}
