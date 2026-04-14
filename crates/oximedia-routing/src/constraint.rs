//! Connection constraint validation for NMOS IS-05.
//!
//! IS-05 allows senders and receivers to declare constraints on the media
//! formats and codecs they support.  Before activating a connection the
//! controller should verify that the sender's format and codec are compatible
//! with the receiver's constraints.
//!
//! # Example
//!
//! ```
//! use oximedia_routing::constraint::ConnectionConstraint;
//!
//! let c = ConnectionConstraint::new()
//!     .allow_format("video/raw")
//!     .allow_format("video/jxsv")
//!     .allow_codec("raw")
//!     .allow_codec("jpeg-xs");
//!
//! assert!(c.check("video/raw", "raw"));
//! assert!(c.check("video/jxsv", "jpeg-xs"));
//! assert!(!c.check("audio/L24", "pcm"));
//! ```

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Represents an IS-05 connection constraint specifying the allowed media
/// formats and codecs for a sender or receiver endpoint.
///
/// An empty `allowed_formats` list means **any format is accepted** (wildcard).
/// Likewise, an empty `allowed_codecs` list means any codec is accepted.
#[derive(Debug, Clone, Default)]
pub struct ConnectionConstraint {
    /// Set of accepted media-type strings (e.g., `"video/raw"`, `"audio/L24"`).
    pub allowed_formats: Vec<String>,
    /// Set of accepted codec identifiers (e.g., `"raw"`, `"h264"`, `"opus"`).
    pub allowed_codecs: Vec<String>,
}

impl ConnectionConstraint {
    /// Create a new, permissive constraint (no restrictions).
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an allowed format (builder pattern).
    pub fn allow_format(mut self, format: impl Into<String>) -> Self {
        self.allowed_formats.push(format.into());
        self
    }

    /// Add an allowed codec (builder pattern).
    pub fn allow_codec(mut self, codec: impl Into<String>) -> Self {
        self.allowed_codecs.push(codec.into());
        self
    }

    /// Register an allowed format in-place.
    pub fn add_format(&mut self, format: impl Into<String>) {
        self.allowed_formats.push(format.into());
    }

    /// Register an allowed codec in-place.
    pub fn add_codec(&mut self, codec: impl Into<String>) {
        self.allowed_codecs.push(codec.into());
    }

    /// Check whether a given `format` and `codec` pair satisfies this constraint.
    ///
    /// The comparison is **case-insensitive** to match common IS-05 practice.
    ///
    /// # Rules
    ///
    /// * If `allowed_formats` is empty → any format is accepted.
    /// * If `allowed_codecs` is empty  → any codec is accepted.
    /// * Both conditions must hold simultaneously.
    ///
    /// # Returns
    ///
    /// `true` if the format/codec pair is permitted; `false` otherwise.
    pub fn check(&self, format: &str, codec: &str) -> bool {
        let format_ok = self.allowed_formats.is_empty()
            || self
                .allowed_formats
                .iter()
                .any(|f| f.eq_ignore_ascii_case(format));

        let codec_ok = self.allowed_codecs.is_empty()
            || self
                .allowed_codecs
                .iter()
                .any(|c| c.eq_ignore_ascii_case(codec));

        format_ok && codec_ok
    }

    /// Return `true` if this constraint has no restrictions (wildcard).
    pub fn is_permissive(&self) -> bool {
        self.allowed_formats.is_empty() && self.allowed_codecs.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_constraint_is_permissive() {
        let c = ConnectionConstraint::new();
        assert!(c.is_permissive());
        assert!(c.check("video/raw", "h264"));
        assert!(c.check("audio/L24", "pcm"));
    }

    #[test]
    fn test_format_restriction_blocks_unknown() {
        let c = ConnectionConstraint::new().allow_format("video/raw");
        assert!(c.check("video/raw", "any"));
        assert!(!c.check("audio/L24", "any"));
    }

    #[test]
    fn test_codec_restriction_blocks_unknown() {
        let c = ConnectionConstraint::new().allow_codec("opus");
        assert!(c.check("any/format", "opus"));
        assert!(!c.check("any/format", "aac"));
    }

    #[test]
    fn test_both_restrictions_must_match() {
        let c = ConnectionConstraint::new()
            .allow_format("video/raw")
            .allow_codec("raw");
        assert!(c.check("video/raw", "raw"));
        assert!(!c.check("video/raw", "h264")); // wrong codec
        assert!(!c.check("audio/L24", "raw")); // wrong format
        assert!(!c.check("audio/L24", "aac")); // both wrong
    }

    #[test]
    fn test_case_insensitive_format() {
        let c = ConnectionConstraint::new().allow_format("Video/Raw");
        assert!(c.check("video/raw", "x"));
        assert!(c.check("VIDEO/RAW", "x"));
    }

    #[test]
    fn test_case_insensitive_codec() {
        let c = ConnectionConstraint::new().allow_codec("OPUS");
        assert!(c.check("x", "opus"));
        assert!(c.check("x", "Opus"));
    }

    #[test]
    fn test_multiple_allowed_formats() {
        let c = ConnectionConstraint::new()
            .allow_format("video/raw")
            .allow_format("video/jxsv");
        assert!(c.check("video/raw", "x"));
        assert!(c.check("video/jxsv", "x"));
        assert!(!c.check("audio/L24", "x"));
    }

    #[test]
    fn test_builder_and_mutating_api_equivalent() {
        let c_builder = ConnectionConstraint::new()
            .allow_format("video/raw")
            .allow_codec("raw");

        let mut c_mut = ConnectionConstraint::new();
        c_mut.add_format("video/raw");
        c_mut.add_codec("raw");

        assert_eq!(
            c_builder.check("video/raw", "raw"),
            c_mut.check("video/raw", "raw")
        );
        assert_eq!(
            c_builder.check("audio/L24", "pcm"),
            c_mut.check("audio/L24", "pcm")
        );
    }

    #[test]
    fn test_is_permissive_false_with_restrictions() {
        let c = ConnectionConstraint::new().allow_format("video/raw");
        assert!(!c.is_permissive());
    }
}
