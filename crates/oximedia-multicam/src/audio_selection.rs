//! Multi-angle audio source selection.
//!
//! Allows independent selection of an audio source angle separate from the
//! currently active video angle.  In a typical live production workflow, the
//! director can follow one camera visually while keeping a boom-mic angle or
//! wireless-lav angle for clean audio.
//!
//! # Core types
//!
//! - [`AudioAngle`] — describes the audio capabilities of one camera angle.
//! - [`AudioMixConfig`] — configuration for mixing, including an optional
//!   independent audio source angle.
//! - [`MultiAngleAudioMixer`] — stateful object that tracks the current video
//!   angle, optional explicit audio angle, and a list of registered angles.
//!
//! # Example
//!
//! ```
//! use oximedia_multicam::audio_selection::{AudioAngle, MultiAngleAudioMixer};
//!
//! let angles = vec![
//!     AudioAngle { angle_id: 0, channels: 2, sample_rate: 48_000 },
//!     AudioAngle { angle_id: 1, channels: 4, sample_rate: 48_000 },
//! ];
//!
//! let mut mixer = MultiAngleAudioMixer::new(0, None, angles);
//!
//! // Follow video angle by default.
//! assert_eq!(mixer.current_audio_angle(), 0);
//!
//! // Switch audio to angle 1 independently.
//! mixer.select_audio_source(Some(1));
//! assert_eq!(mixer.current_audio_angle(), 1);
//!
//! // Revert to following video.
//! mixer.select_audio_source(None);
//! assert_eq!(mixer.current_audio_angle(), 0);
//! ```

// ── AudioAngle ────────────────────────────────────────────────────────────────

/// Audio track descriptor for one camera angle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioAngle {
    /// Angle index (matches the video angle identifier).
    pub angle_id: usize,
    /// Number of audio channels (e.g. 1 = mono, 2 = stereo, 4 = quad).
    pub channels: u32,
    /// Sample rate in Hz (e.g. 48000).
    pub sample_rate: u32,
}

impl AudioAngle {
    /// Create a new `AudioAngle`.
    #[must_use]
    pub fn new(angle_id: usize, channels: u32, sample_rate: u32) -> Self {
        Self {
            angle_id,
            channels,
            sample_rate,
        }
    }

    /// Returns `true` when this angle carries stereo audio (exactly 2 channels).
    #[must_use]
    pub fn is_stereo(&self) -> bool {
        self.channels == 2
    }
}

// ── AudioMixConfig ────────────────────────────────────────────────────────────

/// Configuration controlling how audio is mixed for a multicam output.
#[derive(Debug, Clone)]
pub struct AudioMixConfig {
    /// Master output gain in linear scale (1.0 = unity).
    pub master_gain: f32,
    /// If `Some(id)`: use the specified angle as the audio source regardless
    /// of the current video angle.  If `None`: follow the active video angle.
    pub source_angle: Option<usize>,
    /// Target output channel count.
    pub output_channels: u32,
    /// Target output sample rate in Hz.
    pub output_sample_rate: u32,
}

impl Default for AudioMixConfig {
    fn default() -> Self {
        Self {
            master_gain: 1.0,
            source_angle: None,
            output_channels: 2,
            output_sample_rate: 48_000,
        }
    }
}

// ── MultiAngleAudioMixer ──────────────────────────────────────────────────────

/// Stateful multi-angle audio mixer that supports independent audio / video
/// angle selection.
///
/// The mixer maintains:
/// - `video_angle` — the currently active video angle (updated externally).
/// - `audio_angle` — an optional explicit audio source angle (`None` = follow
///   video).
/// - `angles` — the set of registered [`AudioAngle`] descriptors.
#[derive(Debug, Clone)]
pub struct MultiAngleAudioMixer {
    /// Currently active video angle index.
    pub video_angle: usize,
    /// Explicit audio source angle override.  `None` = follow `video_angle`.
    pub audio_angle: Option<usize>,
    /// Registered audio-capable angles.
    pub angles: Vec<AudioAngle>,
}

impl MultiAngleAudioMixer {
    /// Create a new `MultiAngleAudioMixer`.
    ///
    /// - `video_angle`: initial active video angle.
    /// - `audio_angle`: optional explicit audio override (`None` = follow video).
    /// - `angles`: registered [`AudioAngle`] descriptors (can be empty; angles
    ///   can be added later via [`add_angle`](Self::add_angle)).
    #[must_use]
    pub fn new(video_angle: usize, audio_angle: Option<usize>, angles: Vec<AudioAngle>) -> Self {
        Self {
            video_angle,
            audio_angle,
            angles,
        }
    }

    /// Register a new [`AudioAngle`].
    ///
    /// If an angle with the same `angle_id` already exists it is replaced.
    pub fn add_angle(&mut self, angle: AudioAngle) {
        if let Some(existing) = self.angles.iter_mut().find(|a| a.angle_id == angle.angle_id) {
            *existing = angle;
        } else {
            self.angles.push(angle);
        }
    }

    /// Set the active video angle.  If `audio_angle` is `None` (following
    /// video), the effective audio source will also switch.
    pub fn set_video_angle(&mut self, angle_id: usize) {
        self.video_angle = angle_id;
    }

    /// Select the audio source angle.
    ///
    /// - `Some(id)`: lock audio to the specified angle regardless of video.
    /// - `None`: revert to following the video angle.
    pub fn select_audio_source(&mut self, angle_id: Option<usize>) {
        self.audio_angle = angle_id;
    }

    /// Return the effective audio angle index.
    ///
    /// When `audio_angle` is `None` this returns `video_angle`; otherwise it
    /// returns the explicitly set audio angle.
    #[must_use]
    pub fn current_audio_angle(&self) -> usize {
        self.audio_angle.unwrap_or(self.video_angle)
    }

    /// Look up the [`AudioAngle`] descriptor for the currently active audio
    /// source.  Returns `None` when no matching angle is registered.
    #[must_use]
    pub fn current_audio_info(&self) -> Option<&AudioAngle> {
        let id = self.current_audio_angle();
        self.angles.iter().find(|a| a.angle_id == id)
    }

    /// Returns `true` when the audio and video angles are the same (i.e. audio
    /// is following video or the override matches the current video angle).
    #[must_use]
    pub fn is_audio_following_video(&self) -> bool {
        self.current_audio_angle() == self.video_angle
    }

    /// Return all registered angles that match the given `sample_rate`.
    #[must_use]
    pub fn angles_with_sample_rate(&self, sample_rate: u32) -> Vec<&AudioAngle> {
        self.angles
            .iter()
            .filter(|a| a.sample_rate == sample_rate)
            .collect()
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mixer() -> MultiAngleAudioMixer {
        let angles = vec![
            AudioAngle::new(0, 2, 48_000),
            AudioAngle::new(1, 4, 48_000),
            AudioAngle::new(2, 2, 44_100),
        ];
        MultiAngleAudioMixer::new(0, None, angles)
    }

    #[test]
    fn test_default_audio_follows_video() {
        let mixer = make_mixer();
        assert_eq!(mixer.current_audio_angle(), 0);
        assert!(mixer.is_audio_following_video());
    }

    #[test]
    fn test_select_audio_source_independent() {
        let mut mixer = make_mixer();
        mixer.select_audio_source(Some(1));
        assert_eq!(mixer.current_audio_angle(), 1);
        assert!(!mixer.is_audio_following_video());
    }

    #[test]
    fn test_revert_to_follow_video() {
        let mut mixer = make_mixer();
        mixer.select_audio_source(Some(2));
        assert_eq!(mixer.current_audio_angle(), 2);

        mixer.select_audio_source(None);
        assert_eq!(mixer.current_audio_angle(), 0);
        assert!(mixer.is_audio_following_video());
    }

    #[test]
    fn test_video_angle_change_propagates_when_following() {
        let mut mixer = make_mixer();
        // audio_angle is None → follows video
        mixer.set_video_angle(2);
        assert_eq!(mixer.current_audio_angle(), 2);
    }

    #[test]
    fn test_video_angle_change_does_not_affect_independent_audio() {
        let mut mixer = make_mixer();
        mixer.select_audio_source(Some(1));
        mixer.set_video_angle(2);
        // Audio should still be angle 1
        assert_eq!(mixer.current_audio_angle(), 1);
    }

    #[test]
    fn test_current_audio_info_found() {
        let mixer = make_mixer();
        let info = mixer.current_audio_info().expect("angle 0 should be registered");
        assert_eq!(info.angle_id, 0);
        assert_eq!(info.channels, 2);
    }

    #[test]
    fn test_current_audio_info_not_found_for_unknown_angle() {
        let mut mixer = make_mixer();
        mixer.select_audio_source(Some(99)); // not registered
        assert!(mixer.current_audio_info().is_none());
    }

    #[test]
    fn test_add_angle_insert() {
        let mut mixer = make_mixer();
        mixer.add_angle(AudioAngle::new(3, 8, 96_000));
        let found = mixer.angles.iter().find(|a| a.angle_id == 3);
        assert!(found.is_some());
    }

    #[test]
    fn test_add_angle_replace() {
        let mut mixer = make_mixer();
        mixer.add_angle(AudioAngle::new(0, 6, 96_000)); // replace id=0
        let found = mixer.angles.iter().find(|a| a.angle_id == 0).expect("should exist");
        assert_eq!(found.channels, 6);
    }

    #[test]
    fn test_angles_with_sample_rate() {
        let mixer = make_mixer();
        let stereo_48k = mixer.angles_with_sample_rate(48_000);
        assert_eq!(stereo_48k.len(), 2); // id=0 and id=1
    }

    #[test]
    fn test_audio_angle_is_stereo() {
        let a = AudioAngle::new(0, 2, 48_000);
        assert!(a.is_stereo());
        let b = AudioAngle::new(1, 4, 48_000);
        assert!(!b.is_stereo());
    }

    #[test]
    fn test_audio_mix_config_default() {
        let cfg = AudioMixConfig::default();
        assert!((cfg.master_gain - 1.0).abs() < 1e-6);
        assert!(cfg.source_angle.is_none());
        assert_eq!(cfg.output_channels, 2);
        assert_eq!(cfg.output_sample_rate, 48_000);
    }
}
