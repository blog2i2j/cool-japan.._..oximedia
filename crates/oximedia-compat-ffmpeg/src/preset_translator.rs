//! `-preset` / `-tune` / `-profile` translation for OxiMedia.
//!
//! FFmpeg encoding commands frequently include quality/speed tradeoff options
//! via `-preset`, codec-specific tuning via `-tune`, and encoder profiles via
//! `-profile:v`. This module translates those options to their OxiMedia
//! equivalents using the mapping tables in [`crate::codec_mapping`].
//!
//! ## Preset semantics
//!
//! | FFmpeg preset  | OxiMedia speed (AV1) | Rationale                        |
//! |----------------|----------------------|----------------------------------|
//! | `ultrafast`    | 13                   | Maximum throughput, lower quality |
//! | `superfast`    | 12                   |                                  |
//! | `veryfast`     | 10                   |                                  |
//! | `faster`       | 9                    |                                  |
//! | `fast`         | 8                    |                                  |
//! | `medium`       | 6                    | Balanced default                 |
//! | `slow`         | 4                    |                                  |
//! | `slower`       | 2                    |                                  |
//! | `veryslow`     | 1                    |                                  |
//! | `placebo`      | 0                    | Maximum quality, slowest         |
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_compat_ffmpeg::preset_translator::{
//!     PresetTuneProfileTranslator, TranslatedEncoderOptions,
//! };
//!
//! let args: Vec<String> = vec![
//!     "-i".into(), "input.mkv".into(),
//!     "-c:v".into(), "libaom-av1".into(),
//!     "-preset".into(), "medium".into(),
//!     "-tune".into(), "film".into(),
//!     "-profile:v".into(), "main".into(),
//!     "-crf".into(), "28".into(),
//!     "output.webm".into(),
//! ];
//!
//! let options = PresetTuneProfileTranslator::from_args(&args);
//! assert_eq!(options.oxi_speed, Some(6));
//! assert_eq!(options.oxi_tune.as_deref(), Some("film"));
//! assert_eq!(options.oxi_profile.as_deref(), Some("main"));
//! ```

use crate::codec_mapping::CodecMapper;
use crate::diagnostics::{Diagnostic, DiagnosticSink};

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Error returned when a preset, tune, or profile cannot be translated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresetTranslationError {
    /// The preset name is not recognised.
    UnknownPreset(String),
    /// The tune name is not recognised.
    UnknownTune(String),
    /// The profile name is not recognised.
    UnknownProfile(String),
}

impl std::fmt::Display for PresetTranslationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownPreset(p) => write!(f, "unknown preset: '{}'", p),
            Self::UnknownTune(t) => write!(f, "unknown tune: '{}'", t),
            Self::UnknownProfile(p) => write!(f, "unknown profile: '{}'", p),
        }
    }
}

impl std::error::Error for PresetTranslationError {}

// ─────────────────────────────────────────────────────────────────────────────
// TranslatedEncoderOptions
// ─────────────────────────────────────────────────────────────────────────────

/// The result of translating FFmpeg `-preset`, `-tune`, and `-profile:v`
/// into OxiMedia-native encoder option values.
#[derive(Debug, Clone, Default)]
pub struct TranslatedEncoderOptions {
    /// OxiMedia AV1/VP9 speed level (0 = slowest/best, 13 = fastest).
    ///
    /// Derived from `-preset` translation. `None` if no preset was found.
    pub oxi_speed: Option<u8>,

    /// The original FFmpeg preset string (e.g. `"medium"`, `"veryslow"`).
    pub ffmpeg_preset: Option<String>,

    /// OxiMedia tune identifier (e.g. `"film"`, `"animation"`, `"grain"`).
    ///
    /// Derived from `-tune` translation. `None` if no tune was found.
    pub oxi_tune: Option<String>,

    /// The original FFmpeg tune string.
    pub ffmpeg_tune: Option<String>,

    /// OxiMedia profile identifier (e.g. `"main"`, `"high"`, `"professional"`).
    ///
    /// Derived from `-profile:v` / `-profile` translation.
    pub oxi_profile: Option<String>,

    /// The original FFmpeg profile string.
    pub ffmpeg_profile: Option<String>,

    /// Diagnostics produced during translation (unknown/unsupported options).
    pub diagnostics: Vec<Diagnostic>,
}

impl TranslatedEncoderOptions {
    /// Return `true` if no preset, tune, or profile was recognised.
    pub fn is_empty(&self) -> bool {
        self.oxi_speed.is_none()
            && self.oxi_tune.is_none()
            && self.oxi_profile.is_none()
    }

    /// Return `true` if any diagnostic is at error level.
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(|d| d.is_error())
    }

    /// Return a human-readable summary of the translated options.
    pub fn summary(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(speed) = self.oxi_speed {
            parts.push(format!(
                "preset={}→speed={}",
                self.ffmpeg_preset.as_deref().unwrap_or("?"),
                speed
            ));
        }
        if let Some(ref tune) = self.oxi_tune {
            parts.push(format!(
                "tune={}→{}",
                self.ffmpeg_tune.as_deref().unwrap_or("?"),
                tune
            ));
        }
        if let Some(ref profile) = self.oxi_profile {
            parts.push(format!(
                "profile={}→{}",
                self.ffmpeg_profile.as_deref().unwrap_or("?"),
                profile
            ));
        }
        if parts.is_empty() {
            "no preset/tune/profile specified".to_string()
        } else {
            parts.join(", ")
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PresetTuneProfileTranslator
// ─────────────────────────────────────────────────────────────────────────────

/// Translates FFmpeg `-preset`, `-tune`, and `-profile:v` argument values
/// into OxiMedia encoder option structures.
///
/// The translator performs case-insensitive, hyphen/underscore-normalised
/// lookups against the static tables in [`CodecMapper`]. Unknown values
/// produce [`Diagnostic`] warnings rather than hard errors, allowing the
/// translation pipeline to continue with partial results.
pub struct PresetTuneProfileTranslator;

impl PresetTuneProfileTranslator {
    /// Parse a raw FFmpeg argument slice and extract all preset/tune/profile
    /// options, returning a [`TranslatedEncoderOptions`].
    ///
    /// Recognised argument forms:
    /// - `-preset <name>` — speed/quality preset
    /// - `-tune <name>` — codec tuning
    /// - `-profile:v <name>`, `-profile <name>` — encoder profile
    pub fn from_args(args: &[String]) -> TranslatedEncoderOptions {
        let mut opts = TranslatedEncoderOptions::default();
        let mut sink = DiagnosticSink::new();

        let mut it = args.iter().peekable();
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-preset" => {
                    if let Some(val) = it.next() {
                        let result = Self::translate_preset(val);
                        match result {
                            Ok(speed) => {
                                opts.oxi_speed = Some(speed);
                                opts.ffmpeg_preset = Some(val.clone());
                            }
                            Err(PresetTranslationError::UnknownPreset(ref p)) => {
                                sink.push(Diagnostic::warning(format!(
                                    "unknown -preset '{}'; using default speed",
                                    p
                                )));
                                opts.ffmpeg_preset = Some(val.clone());
                            }
                            Err(e) => {
                                sink.push(Diagnostic::warning(e.to_string()));
                            }
                        }
                    }
                }
                "-tune" => {
                    if let Some(val) = it.next() {
                        let result = Self::translate_tune(val);
                        match result {
                            Ok(tune) => {
                                opts.oxi_tune = Some(tune);
                                opts.ffmpeg_tune = Some(val.clone());
                            }
                            Err(PresetTranslationError::UnknownTune(ref t)) => {
                                sink.push(Diagnostic::warning(format!(
                                    "unknown -tune '{}'; ignoring",
                                    t
                                )));
                                opts.ffmpeg_tune = Some(val.clone());
                            }
                            Err(e) => {
                                sink.push(Diagnostic::warning(e.to_string()));
                            }
                        }
                    }
                }
                "-profile" | "-profile:v" => {
                    if let Some(val) = it.next() {
                        let result = Self::translate_profile(val);
                        match result {
                            Ok(profile) => {
                                opts.oxi_profile = Some(profile);
                                opts.ffmpeg_profile = Some(val.clone());
                            }
                            Err(PresetTranslationError::UnknownProfile(ref p)) => {
                                sink.push(Diagnostic::warning(format!(
                                    "unknown -profile '{}'; using default",
                                    p
                                )));
                                opts.ffmpeg_profile = Some(val.clone());
                            }
                            Err(e) => {
                                sink.push(Diagnostic::warning(e.to_string()));
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        opts.diagnostics = sink.into_diagnostics();
        opts
    }

    /// Translate a single FFmpeg preset name to an OxiMedia speed level.
    ///
    /// Returns the `oxi_speed` value (0–13) on success, or
    /// [`PresetTranslationError::UnknownPreset`] if the preset is not in the table.
    pub fn translate_preset(ffmpeg_preset: &str) -> Result<u8, PresetTranslationError> {
        CodecMapper::preset(ffmpeg_preset)
            .map(|m| m.oxi_speed)
            .ok_or_else(|| PresetTranslationError::UnknownPreset(ffmpeg_preset.to_string()))
    }

    /// Translate a single FFmpeg tune name to an OxiMedia tune identifier.
    ///
    /// Returns the `oxi_tune` string on success, or
    /// [`PresetTranslationError::UnknownTune`] if the tune is not in the table.
    pub fn translate_tune(ffmpeg_tune: &str) -> Result<String, PresetTranslationError> {
        CodecMapper::tune(ffmpeg_tune)
            .map(|m| m.oxi_tune.to_string())
            .ok_or_else(|| PresetTranslationError::UnknownTune(ffmpeg_tune.to_string()))
    }

    /// Translate a single FFmpeg profile name to an OxiMedia profile identifier.
    ///
    /// Returns the `oxi_profile` string on success, or
    /// [`PresetTranslationError::UnknownProfile`] if the profile is not in the table.
    pub fn translate_profile(ffmpeg_profile: &str) -> Result<String, PresetTranslationError> {
        CodecMapper::profile(ffmpeg_profile)
            .map(|m| m.oxi_profile.to_string())
            .ok_or_else(|| PresetTranslationError::UnknownProfile(ffmpeg_profile.to_string()))
    }

    /// Return all known FFmpeg preset names (for help/completion output).
    pub fn known_presets() -> Vec<&'static str> {
        CodecMapper::all_presets()
            .iter()
            .map(|m| m.ffmpeg_name)
            .collect()
    }

    /// Return all known FFmpeg tune names.
    pub fn known_tunes() -> Vec<&'static str> {
        CodecMapper::all_tunes()
            .iter()
            .map(|m| m.ffmpeg_name)
            .collect()
    }

    /// Return all known FFmpeg profile names.
    pub fn known_profiles() -> Vec<&'static str> {
        CodecMapper::all_profiles()
            .iter()
            .map(|m| m.ffmpeg_name)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn args(slice: &[&str]) -> Vec<String> {
        slice.iter().map(|s| s.to_string()).collect()
    }

    // ── translate_preset ─────────────────────────────────────────────────────

    #[test]
    fn test_translate_preset_medium() {
        let speed = PresetTuneProfileTranslator::translate_preset("medium").expect("should succeed");
        assert_eq!(speed, 6, "medium should map to AV1 speed 6");
    }

    #[test]
    fn test_translate_preset_ultrafast() {
        let speed =
            PresetTuneProfileTranslator::translate_preset("ultrafast").expect("should succeed");
        assert_eq!(speed, 13, "ultrafast should map to AV1 speed 13");
    }

    #[test]
    fn test_translate_preset_veryslow() {
        let speed =
            PresetTuneProfileTranslator::translate_preset("veryslow").expect("should succeed");
        assert_eq!(speed, 1, "veryslow should map to AV1 speed 1");
    }

    #[test]
    fn test_translate_preset_placebo() {
        let speed =
            PresetTuneProfileTranslator::translate_preset("placebo").expect("should succeed");
        assert_eq!(speed, 0, "placebo should map to AV1 speed 0");
    }

    #[test]
    fn test_translate_preset_numeric_svt() {
        // SVT-AV1 numeric presets
        let speed = PresetTuneProfileTranslator::translate_preset("6").expect("should succeed");
        assert_eq!(speed, 6);
    }

    #[test]
    fn test_translate_preset_case_insensitive() {
        let speed =
            PresetTuneProfileTranslator::translate_preset("MEDIUM").expect("should succeed");
        assert_eq!(speed, 6, "case-insensitive lookup should work");
    }

    #[test]
    fn test_translate_preset_unknown_returns_error() {
        let err =
            PresetTuneProfileTranslator::translate_preset("superduperslow").expect_err("should fail");
        assert!(
            matches!(err, PresetTranslationError::UnknownPreset(_)),
            "should be UnknownPreset"
        );
    }

    // ── translate_tune ───────────────────────────────────────────────────────

    #[test]
    fn test_translate_tune_film() {
        let tune =
            PresetTuneProfileTranslator::translate_tune("film").expect("should succeed");
        assert_eq!(tune, "film");
    }

    #[test]
    fn test_translate_tune_animation() {
        let tune =
            PresetTuneProfileTranslator::translate_tune("animation").expect("should succeed");
        assert_eq!(tune, "animation");
    }

    #[test]
    fn test_translate_tune_grain() {
        let tune =
            PresetTuneProfileTranslator::translate_tune("grain").expect("should succeed");
        assert_eq!(tune, "grain");
    }

    #[test]
    fn test_translate_tune_zerolatency() {
        let tune =
            PresetTuneProfileTranslator::translate_tune("zerolatency").expect("should succeed");
        assert_eq!(tune, "zerolatency");
    }

    #[test]
    fn test_translate_tune_unknown_returns_error() {
        let err =
            PresetTuneProfileTranslator::translate_tune("totally_unknown_tune").expect_err("should fail");
        assert!(
            matches!(err, PresetTranslationError::UnknownTune(_)),
            "should be UnknownTune"
        );
    }

    // ── translate_profile ────────────────────────────────────────────────────

    #[test]
    fn test_translate_profile_main() {
        let profile =
            PresetTuneProfileTranslator::translate_profile("main").expect("should succeed");
        assert_eq!(profile, "main");
    }

    #[test]
    fn test_translate_profile_high() {
        let profile =
            PresetTuneProfileTranslator::translate_profile("high").expect("should succeed");
        assert_eq!(profile, "high");
    }

    #[test]
    fn test_translate_profile_high10_to_professional() {
        let profile =
            PresetTuneProfileTranslator::translate_profile("high10").expect("should succeed");
        assert_eq!(profile, "professional", "high10 should map to professional");
    }

    #[test]
    fn test_translate_profile_main10_to_professional() {
        let profile =
            PresetTuneProfileTranslator::translate_profile("main10").expect("should succeed");
        assert_eq!(profile, "professional");
    }

    #[test]
    fn test_translate_profile_vp9_profile0() {
        let profile =
            PresetTuneProfileTranslator::translate_profile("profile0").expect("should succeed");
        assert_eq!(profile, "main");
    }

    #[test]
    fn test_translate_profile_vp9_profile2() {
        let profile =
            PresetTuneProfileTranslator::translate_profile("profile2").expect("should succeed");
        assert_eq!(profile, "professional");
    }

    #[test]
    fn test_translate_profile_unknown_returns_error() {
        let err =
            PresetTuneProfileTranslator::translate_profile("superultra422").expect_err("should fail");
        assert!(
            matches!(err, PresetTranslationError::UnknownProfile(_)),
            "should be UnknownProfile"
        );
    }

    // ── from_args ────────────────────────────────────────────────────────────

    #[test]
    fn test_from_args_full_options() {
        let a = args(&[
            "-i", "input.mkv",
            "-c:v", "libaom-av1",
            "-preset", "medium",
            "-tune", "film",
            "-profile:v", "main",
            "-crf", "28",
            "output.webm",
        ]);
        let opts = PresetTuneProfileTranslator::from_args(&a);
        assert_eq!(opts.oxi_speed, Some(6), "medium -> speed 6");
        assert_eq!(opts.oxi_tune.as_deref(), Some("film"));
        assert_eq!(opts.oxi_profile.as_deref(), Some("main"));
        assert_eq!(opts.ffmpeg_preset.as_deref(), Some("medium"));
        assert_eq!(opts.ffmpeg_tune.as_deref(), Some("film"));
        assert!(!opts.has_errors(), "no errors for valid options");
    }

    #[test]
    fn test_from_args_preset_only() {
        let a = args(&["-i", "in.mkv", "-preset", "slow", "out.webm"]);
        let opts = PresetTuneProfileTranslator::from_args(&a);
        assert_eq!(opts.oxi_speed, Some(4), "slow -> speed 4");
        assert!(opts.oxi_tune.is_none());
        assert!(opts.oxi_profile.is_none());
        assert!(!opts.is_empty());
    }

    #[test]
    fn test_from_args_profile_flag_without_colon() {
        let a = args(&["-i", "in.mkv", "-profile", "high", "out.webm"]);
        let opts = PresetTuneProfileTranslator::from_args(&a);
        assert_eq!(opts.oxi_profile.as_deref(), Some("high"));
    }

    #[test]
    fn test_from_args_no_options() {
        let a = args(&["-i", "in.mkv", "-c:v", "av1", "out.webm"]);
        let opts = PresetTuneProfileTranslator::from_args(&a);
        assert!(opts.is_empty(), "should be empty when no preset/tune/profile");
    }

    #[test]
    fn test_from_args_unknown_preset_produces_warning() {
        let a = args(&["-i", "in.mkv", "-preset", "banana", "out.webm"]);
        let opts = PresetTuneProfileTranslator::from_args(&a);
        assert!(opts.oxi_speed.is_none(), "unknown preset => no speed");
        assert!(
            !opts.diagnostics.is_empty(),
            "should have a diagnostic for unknown preset"
        );
    }

    #[test]
    fn test_from_args_unknown_tune_produces_warning() {
        let a = args(&["-i", "in.mkv", "-tune", "mystery", "out.webm"]);
        let opts = PresetTuneProfileTranslator::from_args(&a);
        assert!(opts.oxi_tune.is_none(), "unknown tune => no tune");
        assert!(
            !opts.diagnostics.is_empty(),
            "should have a diagnostic for unknown tune"
        );
    }

    #[test]
    fn test_summary_full() {
        let opts = TranslatedEncoderOptions {
            oxi_speed: Some(6),
            ffmpeg_preset: Some("medium".into()),
            oxi_tune: Some("film".into()),
            ffmpeg_tune: Some("film".into()),
            oxi_profile: Some("main".into()),
            ffmpeg_profile: Some("main".into()),
            diagnostics: vec![],
        };
        let s = opts.summary();
        assert!(s.contains("medium"), "summary should mention ffmpeg preset");
        assert!(s.contains("speed"), "summary should mention speed");
        assert!(s.contains("film"), "summary should mention tune");
        assert!(s.contains("main"), "summary should mention profile");
    }

    #[test]
    fn test_summary_empty() {
        let opts = TranslatedEncoderOptions::default();
        let s = opts.summary();
        assert!(
            s.contains("no preset"),
            "empty options should say 'no preset'"
        );
    }

    #[test]
    fn test_known_presets_not_empty() {
        let presets = PresetTuneProfileTranslator::known_presets();
        assert!(presets.len() >= 10, "should have at least 10 preset entries");
        assert!(presets.contains(&"medium"), "medium should be present");
        assert!(presets.contains(&"ultrafast"), "ultrafast should be present");
    }

    #[test]
    fn test_known_tunes_not_empty() {
        let tunes = PresetTuneProfileTranslator::known_tunes();
        assert!(tunes.len() >= 5, "should have at least 5 tune entries");
        assert!(tunes.contains(&"film"), "film should be present");
    }

    #[test]
    fn test_known_profiles_not_empty() {
        let profiles = PresetTuneProfileTranslator::known_profiles();
        assert!(profiles.len() >= 5, "should have at least 5 profile entries");
        assert!(profiles.contains(&"main"), "main should be present");
    }

    #[test]
    fn test_preset_speed_ordering() {
        // veryslow < slow < medium < fast < ultrafast
        let veryslow = PresetTuneProfileTranslator::translate_preset("veryslow").unwrap();
        let slow = PresetTuneProfileTranslator::translate_preset("slow").unwrap();
        let medium = PresetTuneProfileTranslator::translate_preset("medium").unwrap();
        let fast = PresetTuneProfileTranslator::translate_preset("fast").unwrap();
        let ultrafast = PresetTuneProfileTranslator::translate_preset("ultrafast").unwrap();

        assert!(
            veryslow < slow && slow < medium && medium < fast && fast < ultrafast,
            "preset speed values should increase from veryslow to ultrafast"
        );
    }
}
