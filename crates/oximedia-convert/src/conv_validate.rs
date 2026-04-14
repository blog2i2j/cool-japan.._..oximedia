/// Pre-conversion input validation for `OxiMedia`.
///
/// Validates conversion parameters against configurable constraints and
/// checks basic input file properties before the conversion pipeline starts.
///
/// Errors reported by the conversion validator.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// The file extension or container format is not supported.
    UnsupportedFormat(String),
    /// The requested resolution exceeds the maximum allowed.
    ResolutionTooLarge {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// The requested bitrate is outside the acceptable range.
    InvalidBitrate(u32),
    /// The codec is not compatible with the target container.
    IncompatibleCodec {
        /// Codec identifier.
        codec: String,
        /// Container/format identifier.
        format: String,
    },
    /// A required field is missing or empty.
    MissingRequiredField(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedFormat(fmt) => write!(f, "Unsupported format: {fmt}"),
            Self::ResolutionTooLarge { width, height } => {
                write!(f, "Resolution {width}x{height} exceeds maximum")
            }
            Self::InvalidBitrate(bps) => write!(f, "Invalid bitrate: {bps} kbps"),
            Self::IncompatibleCodec { codec, format } => {
                write!(f, "Codec '{codec}' incompatible with format '{format}'")
            }
            Self::MissingRequiredField(field) => {
                write!(f, "Missing required field: {field}")
            }
        }
    }
}

// ── ValidateProfile ───────────────────────────────────────────────────────────

/// A flat set of parameters that describes a requested conversion.
///
/// Used as the input type for [`ConvertValidation::validate_profile`].
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ValidateProfile {
    /// Video codec identifier (e.g., "av1", "vp9").
    pub codec: String,
    /// Target bitrate in kilobits per second.
    pub bitrate_kbps: u32,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Target container format (e.g., "webm", "mp4", "mkv").
    pub container: String,
}

impl ValidateProfile {
    /// Creates a new validate profile with the given parameters.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(codec: &str, bitrate_kbps: u32, width: u32, height: u32, container: &str) -> Self {
        Self {
            codec: codec.to_string(),
            bitrate_kbps,
            width,
            height,
            container: container.to_string(),
        }
    }
}

// ── ConvertValidation ─────────────────────────────────────────────────────────

/// Validates conversion profiles and input files against configurable limits.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ConvertValidation {
    /// Maximum allowed output width in pixels.
    pub max_width: u32,
    /// Maximum allowed output height in pixels.
    pub max_height: u32,
    /// Maximum allowed bitrate in kilobits per second.
    pub max_bitrate_kbps: u32,
    /// Set of recognised codec identifiers.
    pub allowed_codecs: Vec<String>,
}

impl Default for ConvertValidation {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvertValidation {
    /// Creates a new validator with sensible defaults.
    ///
    /// * Max resolution: 7680 × 4320 (8K)
    /// * Max bitrate: 100 000 kbps
    /// * Allowed codecs: AV1, VP9, VP8, FLAC, Opus, Vorbis (all patent-free)
    #[allow(dead_code)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_width: 7_680,
            max_height: 4_320,
            max_bitrate_kbps: 100_000,
            allowed_codecs: vec![
                "av1".to_string(),
                "vp9".to_string(),
                "vp8".to_string(),
                "flac".to_string(),
                "opus".to_string(),
                "vorbis".to_string(),
                "theora".to_string(),
                "aom-av1".to_string(),
            ],
        }
    }

    /// Returns `true` if the codec is in the allowed list (case-insensitive).
    #[allow(dead_code)]
    #[must_use]
    pub fn is_codec_supported(&self, codec: &str) -> bool {
        let lower = codec.to_lowercase();
        self.allowed_codecs
            .iter()
            .any(|c| c.to_lowercase() == lower)
    }

    /// Validates all fields of a [`ValidateProfile`] and returns a (possibly
    /// empty) list of [`ValidationError`]s.
    #[allow(dead_code)]
    #[must_use]
    pub fn validate_profile(&self, profile: &ValidateProfile) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // Codec must not be empty
        if profile.codec.is_empty() {
            errors.push(ValidationError::MissingRequiredField("codec".to_string()));
        } else if !self.is_codec_supported(&profile.codec) {
            errors.push(ValidationError::UnsupportedFormat(profile.codec.clone()));
        }

        // Container must not be empty
        if profile.container.is_empty() {
            errors.push(ValidationError::MissingRequiredField(
                "container".to_string(),
            ));
        }

        // Resolution check
        if profile.width > self.max_width || profile.height > self.max_height {
            errors.push(ValidationError::ResolutionTooLarge {
                width: profile.width,
                height: profile.height,
            });
        }

        // Zero resolution check
        if profile.width == 0 {
            errors.push(ValidationError::MissingRequiredField("width".to_string()));
        }
        if profile.height == 0 {
            errors.push(ValidationError::MissingRequiredField("height".to_string()));
        }

        // Bitrate check
        if profile.bitrate_kbps == 0 || profile.bitrate_kbps > self.max_bitrate_kbps {
            errors.push(ValidationError::InvalidBitrate(profile.bitrate_kbps));
        }

        // Codec–container compatibility check
        if !profile.codec.is_empty() && !profile.container.is_empty() {
            if let Some(err) = check_codec_container_compat(&profile.codec, &profile.container) {
                errors.push(err);
            }
        }

        errors
    }
}

/// Checks whether the codec is compatible with the target container.
///
/// Returns `Some(ValidationError)` on incompatibility, `None` if OK.
fn check_codec_container_compat(codec: &str, container: &str) -> Option<ValidationError> {
    let codec_lower = codec.to_lowercase();
    let container_lower = container.to_lowercase();

    // Known incompatibilities (non-exhaustive but representative)
    let incompatible = match container_lower.as_str() {
        "mp4" => matches!(codec_lower.as_str(), "vorbis" | "theora" | "vp8"),
        "webm" => !matches!(
            codec_lower.as_str(),
            "vp8" | "vp9" | "av1" | "opus" | "vorbis" | "aom-av1"
        ),
        "ogg" => !matches!(codec_lower.as_str(), "vorbis" | "opus" | "flac" | "theora"),
        _ => false,
    };

    if incompatible {
        Some(ValidationError::IncompatibleCodec {
            codec: codec.to_string(),
            format: container.to_string(),
        })
    } else {
        None
    }
}

// ── validate_input_file ───────────────────────────────────────────────────────

/// Checks basic properties of an input file path.
///
/// Returns a list of [`ValidationError`]s (empty means valid).
/// Checks performed:
/// 1. Path must not be empty.
/// 2. Extension must be a known media format.
#[allow(dead_code)]
#[must_use]
pub fn validate_input_file(path: &str) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    if path.is_empty() {
        errors.push(ValidationError::MissingRequiredField("path".to_string()));
        return errors;
    }

    // Extract extension
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let known_extensions = [
        "webm", "mkv", "ogg", "ogv", "oga", "flac", "opus", "mp4", "mov", "avi", "wav", "mp3",
        "aac", "m4a", "ts", "m2ts", "mts",
    ];

    if ext.is_empty() || !known_extensions.contains(&ext.as_str()) {
        errors.push(ValidationError::UnsupportedFormat(format!(
            "unknown extension '{ext}'"
        )));
    }

    errors
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn valid_profile() -> ValidateProfile {
        ValidateProfile::new("av1", 4_000, 1920, 1080, "webm")
    }

    // ── ValidationError display ───────────────────────────────────────────────

    #[test]
    fn error_display_unsupported_format() {
        let e = ValidationError::UnsupportedFormat("h264".to_string());
        assert!(e.to_string().contains("h264"));
    }

    #[test]
    fn error_display_resolution_too_large() {
        let e = ValidationError::ResolutionTooLarge {
            width: 10_000,
            height: 8_000,
        };
        assert!(e.to_string().contains("10000"));
    }

    #[test]
    fn error_display_invalid_bitrate() {
        let e = ValidationError::InvalidBitrate(0);
        assert!(e.to_string().contains("0 kbps"));
    }

    // ── ConvertValidation ─────────────────────────────────────────────────────

    #[test]
    fn validation_valid_profile_no_errors() {
        let v = ConvertValidation::new();
        let p = valid_profile();
        assert!(v.validate_profile(&p).is_empty());
    }

    #[test]
    fn validation_unsupported_codec() {
        let v = ConvertValidation::new();
        let p = ValidateProfile::new("h264", 4_000, 1920, 1080, "mp4");
        let errors = v.validate_profile(&p);
        assert!(!errors.is_empty());
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::UnsupportedFormat(_))));
    }

    #[test]
    fn validation_resolution_too_large() {
        let v = ConvertValidation::new();
        let p = ValidateProfile::new("av1", 4_000, 10_000, 10_000, "webm");
        let errors = v.validate_profile(&p);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::ResolutionTooLarge { .. })));
    }

    #[test]
    fn validation_invalid_bitrate_zero() {
        let v = ConvertValidation::new();
        let p = ValidateProfile::new("av1", 0, 1920, 1080, "webm");
        let errors = v.validate_profile(&p);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidBitrate(_))));
    }

    #[test]
    fn validation_bitrate_too_high() {
        let v = ConvertValidation::new();
        let p = ValidateProfile::new("av1", 200_000, 1920, 1080, "webm");
        let errors = v.validate_profile(&p);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::InvalidBitrate(_))));
    }

    #[test]
    fn validation_missing_codec_field() {
        let v = ConvertValidation::new();
        let p = ValidateProfile::new("", 4_000, 1920, 1080, "webm");
        let errors = v.validate_profile(&p);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::MissingRequiredField(f) if f == "codec")));
    }

    #[test]
    fn validation_incompatible_codec_container() {
        let v = ConvertValidation::new();
        // vorbis is not compatible with mp4
        let p = ValidateProfile::new("vorbis", 128, 1920, 1080, "mp4");
        let errors = v.validate_profile(&p);
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::IncompatibleCodec { .. })));
    }

    #[test]
    fn validation_codec_supported_case_insensitive() {
        let v = ConvertValidation::new();
        assert!(v.is_codec_supported("AV1"));
        assert!(v.is_codec_supported("Opus"));
        assert!(!v.is_codec_supported("h264"));
    }

    // ── validate_input_file ───────────────────────────────────────────────────

    #[test]
    fn validate_file_empty_path() {
        let errors = validate_input_file("");
        assert!(!errors.is_empty());
        assert!(errors
            .iter()
            .any(|e| matches!(e, ValidationError::MissingRequiredField(_))));
    }

    #[test]
    fn validate_file_known_extension() {
        assert!(validate_input_file("video.webm").is_empty());
        assert!(validate_input_file("audio.flac").is_empty());
    }

    #[test]
    fn validate_file_unknown_extension() {
        let errors = validate_input_file("archive.xyz");
        assert!(!errors.is_empty());
    }

    #[test]
    fn validate_file_no_extension() {
        let errors = validate_input_file("noextension");
        assert!(!errors.is_empty());
    }
}
