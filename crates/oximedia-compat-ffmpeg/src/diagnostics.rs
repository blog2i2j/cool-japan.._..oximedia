//! Structured diagnostics for FFmpeg-compat translation.
//!
//! During argument parsing and translation, various warnings and errors may be
//! generated — for example, patent-encumbered codecs, unsupported options, or
//! unknown filter names. [`Diagnostic`] collects these in a structured way so
//! callers can present them to users or programmatically inspect them.

use thiserror::Error;

/// The category/kind of a diagnostic message, with rich semantic detail.
#[derive(Debug, Clone, PartialEq)]
pub enum DiagnosticKind {
    /// A patent-encumbered codec was requested and substituted with a free alternative.
    PatentCodecSubstituted {
        /// The originally requested FFmpeg codec name.
        from: String,
        /// The OxiMedia patent-free codec used instead.
        to: String,
    },
    /// An FFmpeg option is not supported and was silently ignored.
    UnknownOptionIgnored {
        /// The option string that was ignored.
        option: String,
    },
    /// A filter in the filtergraph is not supported by OxiMedia.
    FilterNotSupported {
        /// The filter name that was skipped.
        filter: String,
    },
    /// A feature is known but not yet implemented.
    UnsupportedFeature {
        /// Human-readable description of what is not supported.
        description: String,
    },
    /// An informational note about how an option was mapped.
    Info {
        /// The informational message.
        message: String,
    },
    /// A hard error that prevents translation from completing.
    Error {
        /// Description of the error.
        message: String,
    },
    /// A non-fatal warning.
    Warning {
        /// Description of the warning.
        message: String,
    },
}

impl DiagnosticKind {
    /// Return `true` if this kind represents a fatal error.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Short severity label for display.
    pub fn severity_label(&self) -> &'static str {
        match self {
            Self::Error { .. } => "error",
            Self::Warning { .. } => "warning",
            Self::PatentCodecSubstituted { .. } => "warning",
            Self::UnknownOptionIgnored { .. } => "warning",
            Self::FilterNotSupported { .. } => "warning",
            Self::UnsupportedFeature { .. } => "warning",
            Self::Info { .. } => "info",
        }
    }
}

impl std::fmt::Display for DiagnosticKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.severity_label())
    }
}

/// A single diagnostic message produced during FFmpeg-compat translation.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// Semantic kind of this diagnostic (determines severity and message format).
    pub kind: DiagnosticKind,
    /// Optional additional hint about how to resolve the issue.
    pub suggestion: Option<String>,
}

impl Diagnostic {
    /// Create a `PatentCodecSubstituted` diagnostic.
    pub fn patent_substituted(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            kind: DiagnosticKind::PatentCodecSubstituted {
                from: from.into(),
                to: to.into(),
            },
            suggestion: None,
        }
    }

    /// Create an `UnknownOptionIgnored` diagnostic.
    pub fn unknown_option(option: impl Into<String>) -> Self {
        Self {
            kind: DiagnosticKind::UnknownOptionIgnored {
                option: option.into(),
            },
            suggestion: None,
        }
    }

    /// Create a `FilterNotSupported` diagnostic.
    pub fn filter_not_supported(filter: impl Into<String>) -> Self {
        Self {
            kind: DiagnosticKind::FilterNotSupported {
                filter: filter.into(),
            },
            suggestion: None,
        }
    }

    /// Create an `UnsupportedFeature` diagnostic.
    pub fn unsupported_feature(description: impl Into<String>) -> Self {
        Self {
            kind: DiagnosticKind::UnsupportedFeature {
                description: description.into(),
            },
            suggestion: None,
        }
    }

    /// Create an info diagnostic.
    pub fn info(message: impl Into<String>) -> Self {
        Self {
            kind: DiagnosticKind::Info {
                message: message.into(),
            },
            suggestion: None,
        }
    }

    /// Create an error-level diagnostic.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            kind: DiagnosticKind::Error {
                message: message.into(),
            },
            suggestion: None,
        }
    }

    /// Create a warning-level diagnostic.
    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            kind: DiagnosticKind::Warning {
                message: message.into(),
            },
            suggestion: None,
        }
    }

    /// Attach a suggestion to this diagnostic.
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Return `true` if this diagnostic represents a fatal error.
    pub fn is_error(&self) -> bool {
        self.kind.is_error()
    }

    /// Format this diagnostic in FFmpeg's stderr style.
    ///
    /// ```text
    /// oximedia-ff: Codec 'libx264' is a patent codec. Using 'av1' instead.
    /// oximedia-ff: Option '-hwaccel' not supported. Ignoring.
    /// ```
    pub fn format_ffmpeg_style(&self, program: &str) -> String {
        let base = match &self.kind {
            DiagnosticKind::PatentCodecSubstituted { from, to } => {
                format!(
                    "{}: Codec '{}' is a patent codec. Using '{}' instead.",
                    program, from, to
                )
            }
            DiagnosticKind::UnknownOptionIgnored { option } => {
                format!("{}: Option '{}' not supported. Ignoring.", program, option)
            }
            DiagnosticKind::FilterNotSupported { filter } => {
                format!("{}: Filter '{}' not supported. Skipping.", program, filter)
            }
            DiagnosticKind::UnsupportedFeature { description } => {
                format!("{}: {}.", program, description)
            }
            DiagnosticKind::Info { message } => {
                format!("{}: {}", program, message)
            }
            DiagnosticKind::Error { message } => {
                format!("{}: error: {}", program, message)
            }
            DiagnosticKind::Warning { message } => {
                format!("{}: warning: {}", program, message)
            }
        };

        if let Some(hint) = &self.suggestion {
            format!("{}\n  hint: {}", base, hint)
        } else {
            base
        }
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format_ffmpeg_style("oximedia-ff"))
    }
}

/// Error type for fatal translation failures.
#[derive(Debug, Error)]
pub enum TranslationError {
    /// The input arguments could not be parsed.
    #[error("argument parse error: {0}")]
    ParseError(String),

    /// A required input file was not specified.
    #[error("no input file specified")]
    NoInput,

    /// A required output file was not specified.
    #[error("no output file specified")]
    NoOutput,

    /// A filter expression could not be parsed.
    #[error("filter parse error: {0}")]
    FilterParseError(String),

    /// An anyhow error converted into a translation error.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Collect diagnostics produced during a translation pass.
#[derive(Debug, Default)]
pub struct DiagnosticSink {
    items: Vec<Diagnostic>,
}

impl DiagnosticSink {
    /// Create an empty sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a diagnostic into the sink.
    pub fn push(&mut self, diag: Diagnostic) {
        self.items.push(diag);
    }

    /// Return all diagnostics.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.items
    }

    /// Return `true` if any error-level diagnostics were collected.
    pub fn has_errors(&self) -> bool {
        self.items.iter().any(|d| d.is_error())
    }

    /// Consume the sink and return all diagnostics.
    pub fn into_diagnostics(self) -> Vec<Diagnostic> {
        self.items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patent_format_message() {
        let d = Diagnostic::patent_substituted("libx264", "av1");
        let msg = d.format_ffmpeg_style("oximedia-ff");
        assert!(msg.contains("libx264"), "should mention original codec");
        assert!(msg.contains("av1"), "should mention replacement");
        assert!(msg.contains("oximedia-ff"), "should mention program name");
        // Patent substitution is a warning, not an error
        assert!(!d.is_error(), "patent substitution is a warning, not error");
    }

    #[test]
    fn test_patent_substituted_kind() {
        let d = Diagnostic::patent_substituted("aac", "opus");
        match &d.kind {
            DiagnosticKind::PatentCodecSubstituted { from, to } => {
                assert_eq!(from, "aac");
                assert_eq!(to, "opus");
            }
            _ => panic!("expected PatentCodecSubstituted"),
        }
    }

    #[test]
    fn test_unknown_option_format() {
        let d = Diagnostic::unknown_option("-movflags");
        let msg = d.format_ffmpeg_style("oximedia-ff");
        assert!(msg.contains("-movflags"), "should mention the option");
        assert!(
            msg.contains("Ignoring") || msg.contains("not supported"),
            "should indicate the option is ignored or not supported"
        );
        assert!(!d.is_error(), "unknown option is a warning, not error");
    }

    #[test]
    fn test_filter_not_supported_format() {
        let d = Diagnostic::filter_not_supported("drawtext");
        let msg = d.format_ffmpeg_style("oximedia-ff");
        assert!(msg.contains("drawtext"), "should mention the filter name");
        assert!(
            !d.is_error(),
            "filter not supported is a warning, not error"
        );
    }

    #[test]
    fn test_unsupported_feature_format() {
        let d = Diagnostic::unsupported_feature("Hardware decoding via NVDEC");
        let msg = d.format_ffmpeg_style("myapp");
        assert!(msg.contains("Hardware decoding via NVDEC"));
        assert!(msg.starts_with("myapp:"), "should start with program name");
    }

    #[test]
    fn test_error_diagnostic_is_error() {
        let d = Diagnostic::error("something went wrong");
        assert!(
            d.is_error(),
            "error diagnostic should report is_error() = true"
        );
        let msg = d.format_ffmpeg_style("oximedia-ff");
        assert!(
            msg.contains("error"),
            "error message should contain 'error'"
        );
        assert!(msg.contains("something went wrong"));
    }

    #[test]
    fn test_warning_diagnostic_not_error() {
        let d = Diagnostic::warning("this is a warning");
        assert!(!d.is_error(), "warning should not be an error");
        let msg = d.format_ffmpeg_style("oximedia-ff");
        assert!(msg.contains("warning"));
    }

    #[test]
    fn test_info_diagnostic_not_error() {
        let d = Diagnostic::info("informational message");
        assert!(!d.is_error());
        let msg = d.format_ffmpeg_style("oximedia-ff");
        assert!(msg.contains("informational message"));
    }

    #[test]
    fn test_diagnostic_with_suggestion() {
        let d = Diagnostic::unknown_option("-hwaccel")
            .with_suggestion("Use -c:v av1 for software AV1 encoding");
        let msg = d.format_ffmpeg_style("oximedia-ff");
        assert!(msg.contains("hint:"), "suggestion should appear as a hint");
        assert!(msg.contains("Use -c:v av1"), "hint text should be present");
    }

    #[test]
    fn test_diagnostic_sink_collects_items() {
        let mut sink = DiagnosticSink::new();
        sink.push(Diagnostic::warning("w1"));
        sink.push(Diagnostic::info("i1"));
        sink.push(Diagnostic::warning("w2"));
        assert_eq!(sink.diagnostics().len(), 3);
        assert!(!sink.has_errors());
    }

    #[test]
    fn test_diagnostic_sink_detects_errors() {
        let mut sink = DiagnosticSink::new();
        sink.push(Diagnostic::warning("harmless"));
        assert!(!sink.has_errors());
        sink.push(Diagnostic::error("fatal"));
        assert!(sink.has_errors());
    }

    #[test]
    fn test_diagnostic_sink_into_diagnostics() {
        let mut sink = DiagnosticSink::new();
        sink.push(Diagnostic::info("one"));
        sink.push(Diagnostic::info("two"));
        let items = sink.into_diagnostics();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_severity_labels() {
        assert_eq!(
            DiagnosticKind::Error { message: "".into() }.severity_label(),
            "error"
        );
        assert_eq!(
            DiagnosticKind::Warning { message: "".into() }.severity_label(),
            "warning"
        );
        assert_eq!(
            DiagnosticKind::Info { message: "".into() }.severity_label(),
            "info"
        );
        assert_eq!(
            DiagnosticKind::PatentCodecSubstituted {
                from: "".into(),
                to: "".into()
            }
            .severity_label(),
            "warning"
        );
        assert_eq!(
            DiagnosticKind::UnknownOptionIgnored { option: "".into() }.severity_label(),
            "warning"
        );
        assert_eq!(
            DiagnosticKind::FilterNotSupported { filter: "".into() }.severity_label(),
            "warning"
        );
    }

    #[test]
    fn test_format_starts_with_program_name() {
        let d = Diagnostic::patent_substituted("libx264", "av1");
        let msg = d.format_ffmpeg_style("oximedia-ff");
        assert!(
            msg.starts_with("oximedia-ff:"),
            "should start with program name colon"
        );
    }

    #[test]
    fn test_display_impl_uses_default_program() {
        let d = Diagnostic::warning("test warning");
        let display = format!("{}", d);
        // Display impl calls format_ffmpeg_style("oximedia-ff")
        assert!(display.contains("oximedia-ff"));
        assert!(display.contains("test warning"));
    }

    #[test]
    fn test_translation_error_display() {
        let e = TranslationError::NoInput;
        assert!(e.to_string().contains("no input file"));

        let e2 = TranslationError::NoOutput;
        assert!(e2.to_string().contains("no output file"));

        let e3 = TranslationError::ParseError("bad arg".into());
        assert!(e3.to_string().contains("bad arg"));

        let e4 = TranslationError::FilterParseError("bad filter".into());
        assert!(e4.to_string().contains("bad filter"));
    }

    #[test]
    fn test_diagnostic_kind_is_error_only_for_error_variant() {
        assert!(DiagnosticKind::Error { message: "".into() }.is_error());
        assert!(!DiagnosticKind::Warning { message: "".into() }.is_error());
        assert!(!DiagnosticKind::Info { message: "".into() }.is_error());
        assert!(!DiagnosticKind::PatentCodecSubstituted {
            from: "".into(),
            to: "".into()
        }
        .is_error());
        assert!(!DiagnosticKind::UnknownOptionIgnored { option: "".into() }.is_error());
        assert!(!DiagnosticKind::FilterNotSupported { filter: "".into() }.is_error());
        assert!(!DiagnosticKind::UnsupportedFeature {
            description: "".into()
        }
        .is_error());
    }
}
