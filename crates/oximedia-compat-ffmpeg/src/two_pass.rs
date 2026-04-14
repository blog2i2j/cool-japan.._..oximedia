//! Two-pass encoding translation for FFmpeg `-pass 1` / `-pass 2`.
//!
//! FFmpeg's two-pass encoding workflow uses separate invocations for analysis
//! and encoding:
//!
//! ```text
//! ffmpeg -i input.mkv -c:v libaom-av1 -b:v 2M -pass 1 -f null /dev/null
//! ffmpeg -i input.mkv -c:v libaom-av1 -b:v 2M -pass 2 output.webm
//! ```
//!
//! This module detects and translates these patterns into a [`TwoPassPlan`]
//! that OxiMedia's pipeline can execute natively — either sequentially (pass 1
//! then pass 2) or as a single combined operation when the codec supports it.
//!
//! ## Pass Log Files
//!
//! FFmpeg stores inter-pass statistics in `ffmpeg2pass-0.log` by default, or in
//! a file named by `-passlogfile PREFIX`. OxiMedia uses a [`PassLogConfig`] to
//! abstract over this.
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_compat_ffmpeg::two_pass::{
//!     TwoPassPlan, PassSpec, PassLogConfig, translate_two_pass_args,
//! };
//!
//! let pass1_args: Vec<String> = vec![
//!     "-i".into(), "input.mkv".into(),
//!     "-c:v".into(), "libaom-av1".into(),
//!     "-b:v".into(), "2M".into(),
//!     "-pass".into(), "1".into(),
//!     "-f".into(), "null".into(),
//!     "/dev/null".into(),
//! ];
//! let pass2_args: Vec<String> = vec![
//!     "-i".into(), "input.mkv".into(),
//!     "-c:v".into(), "libaom-av1".into(),
//!     "-b:v".into(), "2M".into(),
//!     "-pass".into(), "2".into(),
//!     "output.webm".into(),
//! ];
//!
//! let plan = translate_two_pass_args(&pass1_args, Some(&pass2_args)).unwrap();
//! assert_eq!(plan.pass1.pass_number, 1);
//! assert_eq!(plan.pass2.as_ref().unwrap().pass_number, 2);
//! assert!(plan.is_complete());
//! ```

use std::collections::HashMap;

use thiserror::Error;

use crate::arg_parser::FfmpegArgs;
use crate::codec_map::{CodecCategory, CodecMap};
use crate::diagnostics::Diagnostic;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Error returned when two-pass arguments cannot be translated.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TwoPassError {
    /// No `-pass` flag was found in the arguments.
    #[error("no -pass flag found in arguments")]
    NoPassFlag,

    /// The `-pass` value is not `1` or `2`.
    #[error("invalid -pass value '{0}': must be 1 or 2")]
    InvalidPassValue(String),

    /// Pass 1 and Pass 2 arguments specify different input files.
    #[error("pass 1 input '{pass1}' differs from pass 2 input '{pass2}'")]
    InputMismatch {
        /// Pass 1 input path.
        pass1: String,
        /// Pass 2 input path.
        pass2: String,
    },

    /// Pass 1 and Pass 2 specify different video codecs.
    #[error("pass 1 codec '{pass1}' differs from pass 2 codec '{pass2}'")]
    CodecMismatch {
        /// Pass 1 codec name.
        pass1: String,
        /// Pass 2 codec name.
        pass2: String,
    },

    /// The argument list could not be parsed.
    #[error("argument parse error: {0}")]
    ArgumentParseError(String),

    /// A pass-1 `null` output was expected but not found.
    #[error("pass 1 must output to a null sink (e.g. /dev/null or NUL)")]
    Pass1NotNullOutput,
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass log configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the pass-log file used between the two encoding passes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassLogConfig {
    /// The log file prefix (default: `"ffmpeg2pass"`).
    pub prefix: String,
    /// The full derived log file path (e.g. `"ffmpeg2pass-0.log"`).
    pub log_path: String,
}

impl PassLogConfig {
    /// Build a [`PassLogConfig`] from an optional `-passlogfile` prefix.
    ///
    /// If `prefix` is `None`, the FFmpeg default `"ffmpeg2pass"` is used.
    pub fn from_prefix(prefix: Option<&str>) -> Self {
        let prefix = prefix.unwrap_or("ffmpeg2pass").to_string();
        let log_path = format!("{}-0.log", prefix);
        Self { prefix, log_path }
    }
}

impl Default for PassLogConfig {
    fn default() -> Self {
        Self::from_prefix(None)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PassSpec — a single pass description
// ─────────────────────────────────────────────────────────────────────────────

/// Description of a single encoding pass (1 or 2).
#[derive(Debug, Clone)]
pub struct PassSpec {
    /// Pass number: `1` (analysis) or `2` (encoding).
    pub pass_number: u8,
    /// The source input file.
    pub input_path: String,
    /// The output path (`/dev/null` for pass 1, the real output for pass 2).
    pub output_path: String,
    /// Target video codec (OxiMedia canonical name).
    pub video_codec: Option<String>,
    /// Video bitrate string (e.g. `"2M"`).
    pub video_bitrate: Option<String>,
    /// CRF value, if any.
    pub crf: Option<f64>,
    /// Container format override.
    pub format: Option<String>,
    /// Pass-log file configuration for this pass.
    pub log_config: PassLogConfig,
    /// Additional metadata from `-metadata` flags.
    pub metadata: HashMap<String, String>,
    /// Diagnostics produced while parsing this pass.
    pub diagnostics: Vec<Diagnostic>,
}

impl PassSpec {
    /// Return `true` if this pass outputs to a null sink.
    pub fn is_null_output(&self) -> bool {
        let p = self.output_path.as_str();
        p == "/dev/null" || p == "NUL" || p.eq_ignore_ascii_case("nul") || p == "-"
    }

    /// Return a human-readable description of this pass.
    pub fn description(&self) -> String {
        format!(
            "Pass {} | {} -> {} | codec={}",
            self.pass_number,
            self.input_path,
            self.output_path,
            self.video_codec.as_deref().unwrap_or("copy"),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TwoPassPlan
// ─────────────────────────────────────────────────────────────────────────────

/// A complete two-pass encoding plan.
#[derive(Debug, Clone)]
pub struct TwoPassPlan {
    /// Pass 1 (analysis) specification.
    pub pass1: PassSpec,
    /// Pass 2 (encoding) specification, or `None` if only pass 1 was provided.
    pub pass2: Option<PassSpec>,
    /// The shared pass-log file configuration.
    pub log_config: PassLogConfig,
    /// Whether the codec supports combined two-pass as a single operation.
    pub can_combine: bool,
}

impl TwoPassPlan {
    /// Return `true` if both pass 1 and pass 2 have been specified.
    pub fn is_complete(&self) -> bool {
        self.pass2.is_some()
    }

    /// Return the final output path (from pass 2).
    pub fn final_output(&self) -> Option<&str> {
        self.pass2.as_ref().map(|p| p.output_path.as_str())
    }

    /// Return all diagnostics from both passes.
    pub fn all_diagnostics(&self) -> Vec<&Diagnostic> {
        let mut result: Vec<&Diagnostic> = self.pass1.diagnostics.iter().collect();
        if let Some(ref p2) = self.pass2 {
            result.extend(p2.diagnostics.iter());
        }
        result
    }

    /// Return `true` if any diagnostic from either pass is error-level.
    pub fn has_errors(&self) -> bool {
        self.all_diagnostics().iter().any(|d| d.is_error())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Detection helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Detect whether a raw argument slice contains a `-pass` flag and return its value.
///
/// Returns `None` if no `-pass` flag is present.
pub fn detect_pass_number(args: &[String]) -> Option<u8> {
    let mut it = args.iter().peekable();
    while let Some(arg) = it.next() {
        if arg == "-pass" {
            if let Some(val) = it.next() {
                return val.parse::<u8>().ok();
            }
        }
    }
    None
}

/// Extract the `-passlogfile` prefix from a raw argument slice.
///
/// Returns `None` if not present.
pub fn detect_passlogfile(args: &[String]) -> Option<&str> {
    let mut it = args.iter().peekable();
    while let Some(arg) = it.next() {
        if arg == "-passlogfile" {
            return it.next().map(|s| s.as_str());
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-pass translation
// ─────────────────────────────────────────────────────────────────────────────

/// Translate a single set of FFmpeg arguments that contain `-pass N`.
///
/// Used internally and also useful when processing one pass at a time.
pub fn translate_single_pass(args: &[String]) -> Result<PassSpec, TwoPassError> {
    let pass_number = detect_pass_number(args)
        .ok_or(TwoPassError::NoPassFlag)?;

    if pass_number != 1 && pass_number != 2 {
        return Err(TwoPassError::InvalidPassValue(pass_number.to_string()));
    }

    let log_prefix = detect_passlogfile(args);
    let log_config = PassLogConfig::from_prefix(log_prefix);

    // Build a minimal argv without `-pass` and `-passlogfile` for FfmpegArgs.
    let filtered: Vec<String> = {
        let mut out = Vec::new();
        let mut skip_next = false;
        for arg in args {
            if skip_next {
                skip_next = false;
                continue;
            }
            if arg == "-pass" || arg == "-passlogfile" {
                skip_next = true;
                continue;
            }
            out.push(arg.clone());
        }
        out
    };

    let parsed = FfmpegArgs::parse(&filtered)
        .map_err(|e| TwoPassError::ArgumentParseError(e.to_string()))?;

    let input_path = parsed
        .inputs
        .first()
        .map(|i| i.path.clone())
        .unwrap_or_default();

    let output_path = parsed
        .outputs
        .first()
        .and_then(|o| Some(o.path.clone()))
        .unwrap_or_else(|| "/dev/null".into());

    let codec_map = CodecMap::new();
    let mut diagnostics: Vec<Diagnostic> = Vec::new();
    let mut video_codec: Option<String> = None;
    let mut video_bitrate: Option<String> = None;
    let mut crf: Option<f64> = None;
    let mut format: Option<String> = None;
    let mut metadata: HashMap<String, String> = HashMap::new();

    if let Some(out) = parsed.outputs.first() {
        format = out.format.clone();
        metadata = out.metadata.clone();
        for stream_opt in &out.stream_options {
            use crate::arg_parser::StreamType;
            match stream_opt.stream_type {
                StreamType::Video | StreamType::All => {
                    if let Some(codec_name) = &stream_opt.codec {
                        let resolved = resolve_codec_for_pass(&codec_map, codec_name, &mut diagnostics);
                        video_codec = Some(resolved);
                    }
                    if let Some(br) = &stream_opt.bitrate {
                        video_bitrate = Some(br.clone());
                    }
                    if let Some(c) = stream_opt.crf {
                        crf = Some(c);
                    }
                }
                StreamType::Audio | StreamType::Subtitle => {}
            }
        }
    }

    Ok(PassSpec {
        pass_number,
        input_path,
        output_path,
        video_codec,
        video_bitrate,
        crf,
        format,
        log_config,
        metadata,
        diagnostics,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Two-pass translation entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Translate pass-1 and optionally pass-2 FFmpeg argument arrays into a
/// [`TwoPassPlan`].
///
/// # Arguments
///
/// - `pass1_args` — raw argument slice for the first pass (must contain `-pass 1`)
/// - `pass2_args` — optional raw argument slice for the second pass
pub fn translate_two_pass_args(
    pass1_args: &[String],
    pass2_args: Option<&[String]>,
) -> Result<TwoPassPlan, TwoPassError> {
    let pass1 = translate_single_pass(pass1_args)?;

    if pass1.pass_number != 1 {
        return Err(TwoPassError::InvalidPassValue(pass1.pass_number.to_string()));
    }

    let pass2 = if let Some(p2_args) = pass2_args {
        let spec = translate_single_pass(p2_args)?;
        if spec.pass_number != 2 {
            return Err(TwoPassError::InvalidPassValue(spec.pass_number.to_string()));
        }

        // Validate input consistency.
        if !pass1.input_path.is_empty()
            && !spec.input_path.is_empty()
            && pass1.input_path != spec.input_path
        {
            return Err(TwoPassError::InputMismatch {
                pass1: pass1.input_path.clone(),
                pass2: spec.input_path.clone(),
            });
        }

        // Validate codec consistency.
        if pass1.video_codec.is_some()
            && spec.video_codec.is_some()
            && pass1.video_codec != spec.video_codec
        {
            return Err(TwoPassError::CodecMismatch {
                pass1: pass1.video_codec.clone().unwrap_or_default(),
                pass2: spec.video_codec.clone().unwrap_or_default(),
            });
        }

        Some(spec)
    } else {
        None
    };

    // Determine if the codec can be combined into a single pass.
    let can_combine = pass1
        .video_codec
        .as_deref()
        .map(codec_supports_single_pass)
        .unwrap_or(false);

    let log_config = pass1.log_config.clone();

    Ok(TwoPassPlan {
        pass1,
        pass2,
        log_config,
        can_combine,
    })
}

/// Return `true` if the given OxiMedia codec name supports single-pass encoding
/// with quality equivalent to two-pass (e.g. CRF-based codecs like AV1, VP9).
pub fn codec_supports_single_pass(codec: &str) -> bool {
    matches!(codec, "av1" | "vp9" | "vp8" | "ffv1" | "opus" | "vorbis" | "flac")
}

/// Return a short description of the two-pass plan for dry-run output.
pub fn describe_two_pass_plan(plan: &TwoPassPlan) -> String {
    let mut lines = Vec::new();
    lines.push(format!("Two-pass plan [log={}]:", plan.log_config.log_path));
    lines.push(format!("  {}", plan.pass1.description()));
    if let Some(ref p2) = plan.pass2 {
        lines.push(format!("  {}", p2.description()));
    } else {
        lines.push("  Pass 2: (not yet provided)".to_string());
    }
    if plan.can_combine {
        lines.push("  NOTE: codec supports single-pass CRF; two-pass may be unnecessary".to_string());
    }
    lines.join("\n")
}

// ─────────────────────────────────────────────────────────────────────────────
// TwoPassPlanBuilder — fluent builder API
// ─────────────────────────────────────────────────────────────────────────────

/// A fluent builder for constructing a [`TwoPassPlan`] without requiring raw
/// FFmpeg argument slices.
///
/// Useful for tests, configuration-file driven encoding, and OxiMedia-native
/// callers that already have structured encoder parameters.
///
/// ## Example
///
/// ```rust
/// use oximedia_compat_ffmpeg::two_pass::TwoPassPlanBuilder;
///
/// let plan = TwoPassPlanBuilder::new()
///     .input("input.mkv")
///     .video_codec("av1")
///     .video_bitrate("2M")
///     .log_prefix("my_log")
///     .build_pass1("/dev/null")
///     .expect("pass1 build failed");
///
/// assert_eq!(plan.pass1.pass_number, 1);
/// assert!(plan.pass1.is_null_output());
/// ```
#[derive(Debug, Default, Clone)]
pub struct TwoPassPlanBuilder {
    /// Input file path.
    input_path: Option<String>,
    /// Video codec name (OxiMedia canonical).
    video_codec: Option<String>,
    /// Target video bitrate string (e.g. `"2M"`).
    video_bitrate: Option<String>,
    /// CRF quality value.
    crf: Option<f64>,
    /// Container format override.
    format: Option<String>,
    /// Pass-log file prefix.
    log_prefix: Option<String>,
    /// Extra metadata key/value pairs.
    metadata: HashMap<String, String>,
}

impl TwoPassPlanBuilder {
    /// Create a new builder with all fields unset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input file path.
    pub fn input(mut self, path: impl Into<String>) -> Self {
        self.input_path = Some(path.into());
        self
    }

    /// Set the video codec (OxiMedia canonical name, e.g. `"av1"`, `"vp9"`).
    pub fn video_codec(mut self, codec: impl Into<String>) -> Self {
        self.video_codec = Some(codec.into());
        self
    }

    /// Set the video target bitrate (e.g. `"2M"`, `"500k"`).
    pub fn video_bitrate(mut self, bitrate: impl Into<String>) -> Self {
        self.video_bitrate = Some(bitrate.into());
        self
    }

    /// Set a CRF value for quality-based encoding.
    pub fn crf(mut self, value: f64) -> Self {
        self.crf = Some(value);
        self
    }

    /// Set the container format (e.g. `"webm"`, `"mkv"`).
    pub fn format(mut self, fmt: impl Into<String>) -> Self {
        self.format = Some(fmt.into());
        self
    }

    /// Set the pass-log file prefix (default: `"ffmpeg2pass"`).
    pub fn log_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.log_prefix = Some(prefix.into());
        self
    }

    /// Add a metadata key/value pair.
    pub fn metadata_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Build a complete [`TwoPassPlan`] for pass 1, with `output_path` as the
    /// null sink.
    ///
    /// The returned plan will have `pass2 = None`; call
    /// [`TwoPassPlanBuilder::add_pass2`] to complete it.
    pub fn build_pass1(self, null_output: impl Into<String>) -> Result<TwoPassPlan, TwoPassError> {
        let input_path = self.input_path.clone().unwrap_or_default();
        let output_path = null_output.into();
        let log_config = PassLogConfig::from_prefix(self.log_prefix.as_deref());
        let can_combine = self
            .video_codec
            .as_deref()
            .map(codec_supports_single_pass)
            .unwrap_or(false);

        let pass1 = PassSpec {
            pass_number: 1,
            input_path,
            output_path,
            video_codec: self.video_codec.clone(),
            video_bitrate: self.video_bitrate.clone(),
            crf: self.crf,
            format: self.format.clone(),
            log_config: log_config.clone(),
            metadata: self.metadata.clone(),
            diagnostics: Vec::new(),
        };

        Ok(TwoPassPlan {
            pass1,
            pass2: None,
            log_config,
            can_combine,
        })
    }

    /// Extend an existing pass-1-only [`TwoPassPlan`] with pass 2 information.
    ///
    /// Validates that `plan.pass1.input_path == self.input_path` (if both are
    /// set) and that the video codecs match.
    pub fn add_pass2(
        self,
        plan: TwoPassPlan,
        output_path: impl Into<String>,
    ) -> Result<TwoPassPlan, TwoPassError> {
        let input_path = self.input_path.clone().unwrap_or_else(|| plan.pass1.input_path.clone());
        let output_path_str = output_path.into();

        // Validate input consistency.
        if !plan.pass1.input_path.is_empty()
            && !input_path.is_empty()
            && plan.pass1.input_path != input_path
        {
            return Err(TwoPassError::InputMismatch {
                pass1: plan.pass1.input_path.clone(),
                pass2: input_path,
            });
        }

        // Validate codec consistency.
        if let (Some(ref c1), Some(ref c2)) = (&plan.pass1.video_codec, &self.video_codec) {
            if c1 != c2 {
                return Err(TwoPassError::CodecMismatch {
                    pass1: c1.clone(),
                    pass2: c2.clone(),
                });
            }
        }

        let video_codec = self
            .video_codec
            .clone()
            .or_else(|| plan.pass1.video_codec.clone());
        let video_bitrate = self
            .video_bitrate
            .clone()
            .or_else(|| plan.pass1.video_bitrate.clone());
        let format = self.format.clone().or_else(|| plan.pass1.format.clone());
        let log_config = plan.log_config.clone();

        let pass2 = PassSpec {
            pass_number: 2,
            input_path,
            output_path: output_path_str,
            video_codec: video_codec.clone(),
            video_bitrate,
            crf: self.crf.or(plan.pass1.crf),
            format,
            log_config: log_config.clone(),
            metadata: if self.metadata.is_empty() {
                plan.pass1.metadata.clone()
            } else {
                self.metadata
            },
            diagnostics: Vec::new(),
        };

        let can_combine = video_codec
            .as_deref()
            .map(codec_supports_single_pass)
            .unwrap_or(false);

        Ok(TwoPassPlan {
            pass1: plan.pass1,
            pass2: Some(pass2),
            log_config,
            can_combine,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn resolve_codec_for_pass(
    map: &CodecMap,
    ffmpeg_name: &str,
    diags: &mut Vec<Diagnostic>,
) -> String {
    match map.lookup(ffmpeg_name) {
        Some(entry) => {
            if entry.category == CodecCategory::PatentSubstituted {
                diags.push(Diagnostic::patent_substituted(ffmpeg_name, entry.oxi_name));
            }
            entry.oxi_name.to_string()
        }
        None => {
            diags.push(Diagnostic::unknown_option(ffmpeg_name));
            ffmpeg_name.to_string()
        }
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

    // ── detect_pass_number ───────────────────────────────────────────────────

    #[test]
    fn test_detect_pass_number_one() {
        let a = args(&["-i", "in.mkv", "-pass", "1", "-f", "null", "/dev/null"]);
        assert_eq!(detect_pass_number(&a), Some(1));
    }

    #[test]
    fn test_detect_pass_number_two() {
        let a = args(&["-i", "in.mkv", "-pass", "2", "out.webm"]);
        assert_eq!(detect_pass_number(&a), Some(2));
    }

    #[test]
    fn test_detect_pass_number_absent() {
        let a = args(&["-i", "in.mkv", "-c:v", "av1", "out.webm"]);
        assert_eq!(detect_pass_number(&a), None);
    }

    #[test]
    fn test_detect_passlogfile_present() {
        let a = args(&["-i", "in.mkv", "-passlogfile", "mylog", "-pass", "1", "/dev/null"]);
        assert_eq!(detect_passlogfile(&a), Some("mylog"));
    }

    #[test]
    fn test_detect_passlogfile_absent() {
        let a = args(&["-i", "in.mkv", "-pass", "1", "/dev/null"]);
        assert_eq!(detect_passlogfile(&a), None);
    }

    // ── PassLogConfig ────────────────────────────────────────────────────────

    #[test]
    fn test_pass_log_config_default() {
        let cfg = PassLogConfig::default();
        assert_eq!(cfg.prefix, "ffmpeg2pass");
        assert_eq!(cfg.log_path, "ffmpeg2pass-0.log");
    }

    #[test]
    fn test_pass_log_config_custom_prefix() {
        let cfg = PassLogConfig::from_prefix(Some("myenc"));
        assert_eq!(cfg.prefix, "myenc");
        assert_eq!(cfg.log_path, "myenc-0.log");
    }

    // ── translate_single_pass ────────────────────────────────────────────────

    #[test]
    fn test_translate_single_pass_pass1() {
        let a = args(&[
            "-i", "input.mkv",
            "-c:v", "libaom-av1",
            "-b:v", "2M",
            "-pass", "1",
            "-f", "null",
            "/dev/null",
        ]);
        let spec = translate_single_pass(&a).expect("should succeed");
        assert_eq!(spec.pass_number, 1);
        assert_eq!(spec.video_codec.as_deref(), Some("av1"));
        assert_eq!(spec.video_bitrate.as_deref(), Some("2M"));
        assert!(spec.is_null_output(), "pass 1 output should be null");
    }

    #[test]
    fn test_translate_single_pass_pass2() {
        let a = args(&[
            "-i", "input.mkv",
            "-c:v", "libaom-av1",
            "-b:v", "2M",
            "-pass", "2",
            "output.webm",
        ]);
        let spec = translate_single_pass(&a).expect("should succeed");
        assert_eq!(spec.pass_number, 2);
        assert!(!spec.is_null_output(), "pass 2 output should not be null");
    }

    #[test]
    fn test_translate_single_pass_no_flag_errors() {
        let a = args(&["-i", "in.mkv", "-c:v", "av1", "out.webm"]);
        let err = translate_single_pass(&a).expect_err("should fail without -pass");
        assert!(matches!(err, TwoPassError::NoPassFlag));
    }

    #[test]
    fn test_translate_single_pass_patent_substitution() {
        let a = args(&[
            "-i", "in.mp4",
            "-c:v", "libx264",
            "-b:v", "1M",
            "-pass", "1",
            "-f", "null",
            "/dev/null",
        ]);
        let spec = translate_single_pass(&a).expect("should succeed");
        assert_eq!(spec.video_codec.as_deref(), Some("av1"));
        let has_patent = spec.diagnostics.iter().any(|d| {
            matches!(&d.kind, crate::diagnostics::DiagnosticKind::PatentCodecSubstituted { .. })
        });
        assert!(has_patent, "should have patent diagnostic for libx264");
    }

    // ── translate_two_pass_args ──────────────────────────────────────────────

    #[test]
    fn test_translate_two_pass_complete() {
        let p1 = args(&[
            "-i", "input.mkv",
            "-c:v", "libaom-av1",
            "-b:v", "2M",
            "-pass", "1",
            "-f", "null",
            "/dev/null",
        ]);
        let p2 = args(&[
            "-i", "input.mkv",
            "-c:v", "libaom-av1",
            "-b:v", "2M",
            "-pass", "2",
            "output.webm",
        ]);

        let plan = translate_two_pass_args(&p1, Some(&p2)).expect("should succeed");
        assert!(plan.is_complete(), "plan should be complete");
        assert_eq!(plan.pass1.pass_number, 1);
        assert_eq!(plan.pass2.as_ref().map(|p| p.pass_number), Some(2));
        assert_eq!(plan.final_output(), Some("output.webm"));
    }

    #[test]
    fn test_translate_two_pass_pass1_only() {
        let p1 = args(&[
            "-i", "input.mkv",
            "-c:v", "libaom-av1",
            "-b:v", "2M",
            "-pass", "1",
            "-f", "null",
            "/dev/null",
        ]);

        let plan = translate_two_pass_args(&p1, None).expect("should succeed");
        assert!(!plan.is_complete(), "single-pass plan should not be complete");
        assert!(plan.final_output().is_none());
    }

    #[test]
    fn test_translate_two_pass_codec_mismatch_error() {
        let p1 = args(&[
            "-i", "in.mkv",
            "-c:v", "libaom-av1",
            "-b:v", "2M",
            "-pass", "1",
            "-f", "null", "/dev/null",
        ]);
        let p2 = args(&[
            "-i", "in.mkv",
            "-c:v", "libvpx-vp9",
            "-b:v", "2M",
            "-pass", "2",
            "out.webm",
        ]);

        let err = translate_two_pass_args(&p1, Some(&p2)).expect_err("should fail");
        assert!(matches!(err, TwoPassError::CodecMismatch { .. }));
    }

    #[test]
    fn test_translate_two_pass_can_combine_av1() {
        let p1 = args(&[
            "-i", "in.mkv",
            "-c:v", "libaom-av1",
            "-crf", "28",
            "-pass", "1",
            "-f", "null", "/dev/null",
        ]);

        let plan = translate_two_pass_args(&p1, None).expect("should succeed");
        assert!(plan.can_combine, "AV1 should support single-pass");
    }

    #[test]
    fn test_codec_supports_single_pass() {
        assert!(codec_supports_single_pass("av1"));
        assert!(codec_supports_single_pass("vp9"));
        assert!(codec_supports_single_pass("opus"));
        assert!(!codec_supports_single_pass("h264_nvenc"));
    }

    #[test]
    fn test_describe_two_pass_plan() {
        let p1 = args(&[
            "-i", "input.mkv",
            "-c:v", "libaom-av1",
            "-b:v", "2M",
            "-pass", "1",
            "-f", "null", "/dev/null",
        ]);
        let p2 = args(&[
            "-i", "input.mkv",
            "-c:v", "libaom-av1",
            "-b:v", "2M",
            "-pass", "2",
            "output.webm",
        ]);

        let plan = translate_two_pass_args(&p1, Some(&p2)).expect("should succeed");
        let desc = describe_two_pass_plan(&plan);
        assert!(desc.contains("Two-pass plan"), "description should have header");
        assert!(desc.contains("Pass 1"), "description should mention pass 1");
        assert!(desc.contains("Pass 2"), "description should mention pass 2");
    }

    #[test]
    fn test_pass_spec_is_null_output_variants() {
        let spec_null = PassSpec {
            pass_number: 1,
            input_path: "in.mkv".into(),
            output_path: "/dev/null".into(),
            video_codec: None,
            video_bitrate: None,
            crf: None,
            format: None,
            log_config: PassLogConfig::default(),
            metadata: HashMap::new(),
            diagnostics: vec![],
        };
        assert!(spec_null.is_null_output());

        let spec_nul = PassSpec { output_path: "NUL".into(), ..spec_null.clone() };
        assert!(spec_nul.is_null_output());

        let spec_real = PassSpec { output_path: "output.webm".into(), ..spec_null };
        assert!(!spec_real.is_null_output());
    }

    #[test]
    fn test_translate_two_pass_with_passlogfile() {
        let p1 = args(&[
            "-i", "in.mkv",
            "-c:v", "libaom-av1",
            "-b:v", "2M",
            "-passlogfile", "custom_log",
            "-pass", "1",
            "-f", "null", "/dev/null",
        ]);

        let plan = translate_two_pass_args(&p1, None).expect("should succeed");
        assert_eq!(plan.log_config.prefix, "custom_log");
        assert_eq!(plan.log_config.log_path, "custom_log-0.log");
    }

    // ── TwoPassPlanBuilder ───────────────────────────────────────────────────

    #[test]
    fn test_builder_pass1_basic() {
        let plan = TwoPassPlanBuilder::new()
            .input("input.mkv")
            .video_codec("av1")
            .video_bitrate("2M")
            .build_pass1("/dev/null")
            .expect("build_pass1 should succeed");

        assert_eq!(plan.pass1.pass_number, 1);
        assert_eq!(plan.pass1.input_path, "input.mkv");
        assert_eq!(plan.pass1.video_codec.as_deref(), Some("av1"));
        assert_eq!(plan.pass1.video_bitrate.as_deref(), Some("2M"));
        assert!(plan.pass1.is_null_output(), "should be null output");
        assert!(!plan.is_complete(), "pass2 should be None");
    }

    #[test]
    fn test_builder_with_custom_log_prefix() {
        let plan = TwoPassPlanBuilder::new()
            .input("in.mkv")
            .video_codec("vp9")
            .log_prefix("custom_log")
            .build_pass1("/dev/null")
            .expect("build_pass1 should succeed");

        assert_eq!(plan.log_config.prefix, "custom_log");
        assert_eq!(plan.log_config.log_path, "custom_log-0.log");
    }

    #[test]
    fn test_builder_can_combine_av1() {
        let plan = TwoPassPlanBuilder::new()
            .video_codec("av1")
            .build_pass1("/dev/null")
            .expect("should succeed");

        assert!(plan.can_combine, "AV1 supports single-pass CRF");
    }

    #[test]
    fn test_builder_can_combine_vp9() {
        let plan = TwoPassPlanBuilder::new()
            .video_codec("vp9")
            .build_pass1("/dev/null")
            .expect("should succeed");

        assert!(plan.can_combine, "VP9 supports single-pass CRF");
    }

    #[test]
    fn test_builder_add_pass2_complete() {
        let pass1_plan = TwoPassPlanBuilder::new()
            .input("input.mkv")
            .video_codec("av1")
            .video_bitrate("2M")
            .build_pass1("/dev/null")
            .expect("build_pass1 should succeed");

        let complete_plan = TwoPassPlanBuilder::new()
            .input("input.mkv")
            .video_codec("av1")
            .video_bitrate("2M")
            .add_pass2(pass1_plan, "output.webm")
            .expect("add_pass2 should succeed");

        assert!(complete_plan.is_complete(), "plan should be complete");
        assert_eq!(complete_plan.final_output(), Some("output.webm"));
        assert_eq!(
            complete_plan.pass2.as_ref().map(|p| p.pass_number),
            Some(2)
        );
    }

    #[test]
    fn test_builder_add_pass2_codec_mismatch() {
        let pass1_plan = TwoPassPlanBuilder::new()
            .input("input.mkv")
            .video_codec("av1")
            .build_pass1("/dev/null")
            .expect("build_pass1 should succeed");

        let err = TwoPassPlanBuilder::new()
            .input("input.mkv")
            .video_codec("vp9") // different codec
            .add_pass2(pass1_plan, "output.webm")
            .expect_err("codec mismatch should fail");

        assert!(matches!(err, TwoPassError::CodecMismatch { .. }));
    }

    #[test]
    fn test_builder_add_pass2_input_mismatch() {
        let pass1_plan = TwoPassPlanBuilder::new()
            .input("file_a.mkv")
            .video_codec("av1")
            .build_pass1("/dev/null")
            .expect("build_pass1 should succeed");

        let err = TwoPassPlanBuilder::new()
            .input("file_b.mkv") // different input
            .video_codec("av1")
            .add_pass2(pass1_plan, "output.webm")
            .expect_err("input mismatch should fail");

        assert!(matches!(err, TwoPassError::InputMismatch { .. }));
    }

    #[test]
    fn test_builder_metadata_tag() {
        let plan = TwoPassPlanBuilder::new()
            .input("in.mkv")
            .video_codec("av1")
            .metadata_tag("title", "My Documentary")
            .metadata_tag("artist", "Director")
            .build_pass1("/dev/null")
            .expect("should succeed");

        assert_eq!(
            plan.pass1.metadata.get("title").map(|s| s.as_str()),
            Some("My Documentary")
        );
        assert_eq!(
            plan.pass1.metadata.get("artist").map(|s| s.as_str()),
            Some("Director")
        );
    }

    #[test]
    fn test_builder_crf_value() {
        let plan = TwoPassPlanBuilder::new()
            .input("in.mkv")
            .video_codec("av1")
            .crf(28.0)
            .build_pass1("/dev/null")
            .expect("should succeed");

        assert_eq!(plan.pass1.crf, Some(28.0));
    }

    #[test]
    fn test_builder_format_override() {
        let plan = TwoPassPlanBuilder::new()
            .video_codec("av1")
            .format("webm")
            .build_pass1("/dev/null")
            .expect("should succeed");

        assert_eq!(plan.pass1.format.as_deref(), Some("webm"));
    }

    #[test]
    fn test_builder_add_pass2_inherits_bitrate() {
        let pass1_plan = TwoPassPlanBuilder::new()
            .input("in.mkv")
            .video_codec("av1")
            .video_bitrate("4M")
            .build_pass1("/dev/null")
            .expect("should succeed");

        // Pass2 builder doesn't repeat bitrate — should inherit from pass1
        let complete = TwoPassPlanBuilder::new()
            .input("in.mkv")
            .video_codec("av1")
            .add_pass2(pass1_plan, "out.webm")
            .expect("add_pass2 should succeed");

        assert_eq!(
            complete.pass2.as_ref().and_then(|p| p.video_bitrate.as_deref()),
            Some("4M"),
            "pass2 should inherit bitrate from pass1"
        );
    }
}
