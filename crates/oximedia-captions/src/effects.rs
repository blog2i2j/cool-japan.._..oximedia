//! Caption visual effects and animations

use crate::error::Result;
use crate::types::{Caption, CaptionTrack, Duration, Timestamp};
use serde::{Deserialize, Serialize};

/// Visual effect type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectType {
    /// Fade in at start
    FadeIn,
    /// Fade out at end
    FadeOut,
    /// Scroll up
    ScrollUp,
    /// Scroll down
    ScrollDown,
    /// Typewriter (character by character)
    Typewriter,
    /// Karaoke (word by word highlighting)
    Karaoke,
    /// Bounce in
    BounceIn,
    /// Slide from left
    SlideLeft,
    /// Slide from right
    SlideRight,
    /// Slide from top
    SlideTop,
    /// Slide from bottom
    SlideBottom,
}

/// Effect parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectParams {
    /// Effect type
    pub effect_type: EffectType,
    /// Duration of the effect (milliseconds)
    pub duration_ms: i64,
    /// Delay before effect starts (milliseconds)
    pub delay_ms: i64,
    /// Easing function
    pub easing: EasingFunction,
}

impl Default for EffectParams {
    fn default() -> Self {
        Self {
            effect_type: EffectType::FadeIn,
            duration_ms: 300,
            delay_ms: 0,
            easing: EasingFunction::Linear,
        }
    }
}

/// Easing function for animations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EasingFunction {
    /// Linear interpolation
    Linear,
    /// Ease in (slow start)
    EaseIn,
    /// Ease out (slow end)
    EaseOut,
    /// Ease in-out (slow start and end)
    EaseInOut,
    /// Cubic bezier
    Cubic,
    /// Elastic
    Elastic,
    /// Bounce
    Bounce,
}

impl EasingFunction {
    /// Calculate eased value (0.0 to 1.0)
    #[must_use]
    pub fn ease(&self, t: f64) -> f64 {
        match self {
            Self::Linear => t,
            Self::EaseIn => t * t,
            Self::EaseOut => t * (2.0 - t),
            Self::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    -1.0 + (4.0 - 2.0 * t) * t
                }
            }
            Self::Cubic => t * t * t,
            Self::Elastic => {
                if t == 0.0 || t == 1.0 {
                    t
                } else {
                    -(2.0_f64.powf(10.0 * (t - 1.0)))
                        * ((t - 1.1) * 5.0 * std::f64::consts::PI).sin()
                }
            }
            Self::Bounce => {
                if t < 1.0 / 2.75 {
                    7.5625 * t * t
                } else if t < 2.0 / 2.75 {
                    let t = t - 1.5 / 2.75;
                    7.5625 * t * t + 0.75
                } else if t < 2.5 / 2.75 {
                    let t = t - 2.25 / 2.75;
                    7.5625 * t * t + 0.9375
                } else {
                    let t = t - 2.625 / 2.75;
                    7.5625 * t * t + 0.984375
                }
            }
        }
    }
}

/// Effect renderer
pub struct EffectRenderer {
    fps: f64,
}

impl EffectRenderer {
    /// Create a new effect renderer
    #[must_use]
    pub fn new(fps: f64) -> Self {
        Self { fps }
    }

    /// Apply effect to caption
    pub fn apply_effect(&self, caption: &mut Caption, params: &EffectParams) -> Result<()> {
        // Store effect parameters in caption metadata
        caption.metadata.insert(
            "effect_type".to_string(),
            format!("{:?}", params.effect_type),
        );
        caption.metadata.insert(
            "effect_duration".to_string(),
            params.duration_ms.to_string(),
        );
        caption
            .metadata
            .insert("effect_delay".to_string(), params.delay_ms.to_string());
        caption
            .metadata
            .insert("easing".to_string(), format!("{:?}", params.easing));

        Ok(())
    }

    /// Calculate effect progress at a given timestamp
    #[must_use]
    pub fn calculate_progress(
        &self,
        caption: &Caption,
        timestamp: Timestamp,
        params: &EffectParams,
    ) -> f64 {
        if timestamp < caption.start {
            return 0.0;
        }

        let elapsed_ms = timestamp.duration_since(caption.start).as_millis();
        let effect_start_ms = params.delay_ms;
        let effect_end_ms = effect_start_ms + params.duration_ms;

        if elapsed_ms < effect_start_ms {
            return 0.0;
        }

        if elapsed_ms > effect_end_ms {
            return 1.0;
        }

        let progress = (elapsed_ms - effect_start_ms) as f64 / params.duration_ms as f64;
        params.easing.ease(progress)
    }

    /// Generate keyframes for an effect
    #[must_use]
    pub fn generate_keyframes(&self, caption: &Caption, params: &EffectParams) -> Vec<Keyframe> {
        let mut keyframes = Vec::new();

        let effect_duration = Duration::from_millis(params.duration_ms);
        let frame_duration = Duration::from_micros((1_000_000.0 / self.fps) as i64);

        let mut current_time = caption.start;
        let end_time = caption.start.add(effect_duration);

        while current_time <= end_time {
            let progress = self.calculate_progress(caption, current_time, params);

            let keyframe = Keyframe {
                timestamp: current_time,
                progress,
                opacity: self.calculate_opacity(params.effect_type, progress),
                position: self.calculate_position(params.effect_type, progress),
            };

            keyframes.push(keyframe);
            current_time = current_time.add(frame_duration);
        }

        keyframes
    }

    fn calculate_opacity(&self, effect_type: EffectType, progress: f64) -> f64 {
        match effect_type {
            EffectType::FadeIn => progress,
            EffectType::FadeOut => 1.0 - progress,
            _ => 1.0,
        }
    }

    fn calculate_position(&self, effect_type: EffectType, progress: f64) -> (f64, f64) {
        match effect_type {
            EffectType::SlideLeft => (progress - 1.0, 0.0),
            EffectType::SlideRight => (1.0 - progress, 0.0),
            EffectType::SlideTop => (0.0, progress - 1.0),
            EffectType::SlideBottom => (0.0, 1.0 - progress),
            EffectType::ScrollUp => (0.0, -progress),
            EffectType::ScrollDown => (0.0, progress),
            _ => (0.0, 0.0),
        }
    }
}

/// Animation keyframe
#[derive(Debug, Clone, PartialEq)]
pub struct Keyframe {
    /// Timestamp
    pub timestamp: Timestamp,
    /// Animation progress (0.0 to 1.0)
    pub progress: f64,
    /// Opacity (0.0 to 1.0)
    pub opacity: f64,
    /// Position offset (x, y)
    pub position: (f64, f64),
}

/// Karaoke effect generator
#[allow(dead_code)]
pub struct KaraokeEffect {
    word_highlight_duration_ms: i64,
}

impl KaraokeEffect {
    /// Create a new karaoke effect generator
    #[must_use]
    pub fn new() -> Self {
        Self {
            word_highlight_duration_ms: 500,
        }
    }

    /// Generate karaoke timing for a caption
    #[must_use]
    pub fn generate_timing(&self, caption: &Caption) -> Vec<WordTiming> {
        let words: Vec<&str> = caption.text.split_whitespace().collect();
        let total_duration = caption.duration();
        let word_duration = total_duration.as_millis() / words.len() as i64;

        let mut timings = Vec::new();
        let mut current_time = caption.start;

        for word in words {
            let timing = WordTiming {
                word: word.to_string(),
                start: current_time,
                end: current_time.add(Duration::from_millis(word_duration)),
            };

            timings.push(timing);
            current_time = current_time.add(Duration::from_millis(word_duration));
        }

        timings
    }
}

impl Default for KaraokeEffect {
    fn default() -> Self {
        Self::new()
    }
}

/// Word timing for karaoke effect
#[derive(Debug, Clone, PartialEq)]
pub struct WordTiming {
    /// Word text
    pub word: String,
    /// Start timestamp
    pub start: Timestamp,
    /// End timestamp
    pub end: Timestamp,
}

/// Roll-up caption generator
pub struct RollUpEffect {
    /// Number of lines to display
    lines_displayed: usize,
    /// Scroll speed (lines per second)
    scroll_speed: f64,
}

impl RollUpEffect {
    /// Create a new roll-up effect
    #[must_use]
    pub fn new(lines_displayed: usize) -> Self {
        Self {
            lines_displayed,
            scroll_speed: 1.0,
        }
    }

    /// Set scroll speed
    #[must_use]
    pub fn with_scroll_speed(mut self, speed: f64) -> Self {
        self.scroll_speed = speed;
        self
    }

    /// Generate roll-up captions from a track
    pub fn generate(&self, track: &CaptionTrack) -> Result<Vec<RollUpCaption>> {
        let mut roll_ups = Vec::new();
        let mut line_buffer: Vec<String> = Vec::new();

        for caption in &track.captions {
            // Split caption into lines
            let lines: Vec<String> = caption.text.lines().map(String::from).collect();

            for line in lines {
                line_buffer.push(line.clone());

                // Keep only the last N lines
                if line_buffer.len() > self.lines_displayed {
                    line_buffer.remove(0);
                }

                let roll_up = RollUpCaption {
                    timestamp: caption.start,
                    lines: line_buffer.clone(),
                };

                roll_ups.push(roll_up);
            }
        }

        Ok(roll_ups)
    }
}

/// Roll-up caption frame
#[derive(Debug, Clone)]
pub struct RollUpCaption {
    /// Timestamp
    pub timestamp: Timestamp,
    /// Lines to display (oldest to newest)
    pub lines: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Language;

    #[test]
    fn test_easing_functions() {
        let linear = EasingFunction::Linear;
        assert_eq!(linear.ease(0.0), 0.0);
        assert_eq!(linear.ease(0.5), 0.5);
        assert_eq!(linear.ease(1.0), 1.0);

        let ease_in = EasingFunction::EaseIn;
        assert_eq!(ease_in.ease(0.0), 0.0);
        assert!(ease_in.ease(0.5) < 0.5);
        assert_eq!(ease_in.ease(1.0), 1.0);
    }

    #[test]
    fn test_effect_renderer() {
        let renderer = EffectRenderer::new(25.0);
        let caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(5),
            "Test".to_string(),
        );

        let params = EffectParams {
            effect_type: EffectType::FadeIn,
            duration_ms: 1000,
            delay_ms: 0,
            easing: EasingFunction::Linear,
        };

        let progress = renderer.calculate_progress(&caption, Timestamp::from_millis(500), &params);
        assert!((progress - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_keyframe_generation() {
        let renderer = EffectRenderer::new(25.0);
        let caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(5),
            "Test".to_string(),
        );

        let params = EffectParams {
            effect_type: EffectType::FadeIn,
            duration_ms: 1000,
            delay_ms: 0,
            easing: EasingFunction::Linear,
        };

        let keyframes = renderer.generate_keyframes(&caption, &params);
        assert!(!keyframes.is_empty());
        assert!(
            keyframes
                .first()
                .expect("first element should exist")
                .opacity
                >= 0.0
        );
        assert!(keyframes.last().expect("last element should exist").opacity <= 1.0);
    }

    #[test]
    fn test_karaoke_timing() {
        let karaoke = KaraokeEffect::new();
        let caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(5),
            "Hello world test".to_string(),
        );

        let timings = karaoke.generate_timing(&caption);
        assert_eq!(timings.len(), 3); // 3 words
        assert_eq!(timings[0].word, "Hello");
        assert_eq!(timings[1].word, "world");
        assert_eq!(timings[2].word, "test");
    }

    #[test]
    fn test_roll_up_generation() {
        let roll_up = RollUpEffect::new(2);
        let mut track = CaptionTrack::new(Language::english());

        track
            .add_caption(Caption::new(
                Timestamp::from_secs(0),
                Timestamp::from_secs(2),
                "Line 1".to_string(),
            ))
            .expect("operation should succeed in test");

        track
            .add_caption(Caption::new(
                Timestamp::from_secs(2),
                Timestamp::from_secs(4),
                "Line 2".to_string(),
            ))
            .expect("operation should succeed in test");

        track
            .add_caption(Caption::new(
                Timestamp::from_secs(4),
                Timestamp::from_secs(6),
                "Line 3".to_string(),
            ))
            .expect("operation should succeed in test");

        let captions = roll_up.generate(&track).expect("generation should succeed");
        assert!(!captions.is_empty());
        assert!(
            captions
                .last()
                .expect("last element should exist")
                .lines
                .len()
                <= 2
        );
    }
}
