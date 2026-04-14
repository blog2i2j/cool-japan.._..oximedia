//! Batch mode translation for converting multiple files in a single invocation.
//!
//! FFmpeg supports batch processing via shell glob patterns or explicit file lists.
//! This module provides a pure-Rust translation layer that converts a batch job
//! specification into a sequence of [`BatchTranscodeJob`] entries, each carrying
//! full per-file encoding parameters ready for the OxiMedia transcoding pipeline.
//!
//! ## Batch Input Sources
//!
//! | Source kind        | Example                               |
//! |--------------------|---------------------------------------|
//! | Explicit file list | `["a.mp4", "b.mp4", "c.mkv"]`         |
//! | Glob pattern       | `"input_*.mp4"`                        |
//! | Directory scan     | a directory path scanned for media    |
//!
//! ## Output Naming
//!
//! | Strategy          | Description                                             |
//! |-------------------|---------------------------------------------------------|
//! | `SameDir`         | Replace extension in the same directory                 |
//! | `OutputDir`       | Write all files to a fixed output directory             |
//! | `Suffix`          | Append a suffix before the extension (e.g. `_converted`)|
//! | `Template`        | Use `{name}`, `{ext}`, `{dir}`, `{index}` placeholders |
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_compat_ffmpeg::batch_mode::{
//!     BatchSpec, InputSource, OutputNaming, BatchTranscodeJob, translate_batch,
//! };
//!
//! let spec = BatchSpec {
//!     inputs: InputSource::FileList(vec![
//!         "clip1.mp4".into(),
//!         "clip2.mp4".into(),
//!     ]),
//!     output_naming: OutputNaming::Suffix { suffix: "_hq".into(), ext: "mkv".into() },
//!     shared_args: vec![
//!         "-c:v".into(), "libaom-av1".into(),
//!         "-crf".into(), "30".into(),
//!         "-c:a".into(), "libopus".into(),
//!     ],
//!     overwrite: false,
//!     dry_run: false,
//! };
//!
//! let result = translate_batch(&spec).unwrap();
//! assert_eq!(result.jobs.len(), 2);
//! assert!(result.jobs[0].output_path.contains("clip1"));
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::arg_parser::FfmpegArgs;
use crate::codec_map::{CodecCategory, CodecMap};
use crate::diagnostics::Diagnostic;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Error returned when a batch specification cannot be processed.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BatchError {
    /// No input files were found.
    #[error("batch job has no input files")]
    NoInputs,

    /// A glob pattern could not be expanded because it is malformed.
    #[error("malformed glob pattern: '{0}'")]
    MalformedGlob(String),

    /// An output template contains an unknown placeholder.
    #[error("unknown placeholder '{placeholder}' in output template '{template}'")]
    UnknownTemplatePlaceholder {
        /// The unrecognised placeholder name.
        placeholder: String,
        /// The full template string.
        template: String,
    },

    /// The per-file argument list could not be parsed.
    #[error("failed to parse per-file arguments: {0}")]
    ArgumentParseError(String),

    /// An output directory path is invalid.
    #[error("invalid output directory: '{0}'")]
    InvalidOutputDir(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Input source
// ─────────────────────────────────────────────────────────────────────────────

/// How input files are specified for a batch job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputSource {
    /// An explicit list of file paths.
    FileList(Vec<String>),
    /// A glob pattern to be expanded by the caller (OxiMedia does not call the OS).
    GlobPattern(String),
    /// A directory path whose direct children matching optional extensions are used.
    Directory {
        /// The directory path.
        path: String,
        /// Optional list of extensions to include (e.g. `["mp4", "mkv"]`).
        /// If empty, all files are included.
        extensions: Vec<String>,
    },
}

impl InputSource {
    /// Resolve to a list of paths.
    ///
    /// For [`InputSource::FileList`] this is a direct conversion.
    /// For [`InputSource::GlobPattern`] the pattern is returned as-is in a
    /// single-element list (the caller is expected to expand it before passing
    /// files to OxiMedia's transcoding pipeline).
    /// For [`InputSource::Directory`] returns the path itself so the caller
    /// can enumerate its contents.
    pub fn to_path_hints(&self) -> Vec<&str> {
        match self {
            Self::FileList(paths) => paths.iter().map(|s| s.as_str()).collect(),
            Self::GlobPattern(pat) => vec![pat.as_str()],
            Self::Directory { path, .. } => vec![path.as_str()],
        }
    }

    /// Expand `FileList` inputs to resolved file path strings.
    /// For `GlobPattern` and `Directory`, returns the raw pattern/dir hint so
    /// the integration layer can expand it.
    pub fn resolve_files(&self) -> Result<Vec<String>, BatchError> {
        match self {
            Self::FileList(files) => {
                if files.is_empty() {
                    return Err(BatchError::NoInputs);
                }
                Ok(files.clone())
            }
            Self::GlobPattern(pat) => {
                if pat.trim().is_empty() {
                    return Err(BatchError::MalformedGlob(pat.clone()));
                }
                // Return the pattern as-is; callers resolve via OS glob.
                Ok(vec![pat.clone()])
            }
            Self::Directory { path, extensions } => {
                if path.trim().is_empty() {
                    return Err(BatchError::InvalidOutputDir(path.clone()));
                }
                // Return the directory as a hint; callers enumerate on disk.
                let hint = if extensions.is_empty() {
                    path.clone()
                } else {
                    format!("{}/*.({})", path, extensions.join("|"))
                };
                Ok(vec![hint])
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Output naming strategy
// ─────────────────────────────────────────────────────────────────────────────

/// How output file paths are derived from each input path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputNaming {
    /// Replace the file extension, keeping the same directory.
    ///
    /// `input/foo.mkv` → `input/foo.mp4` (when `ext = "mp4"`)
    SameDir {
        /// The output container extension (without leading dot).
        ext: String,
    },
    /// Write all output files to a fixed directory.
    ///
    /// `input/foo.mkv` → `out_dir/foo.mp4`
    OutputDir {
        /// Destination directory path.
        dir: String,
        /// The output container extension.
        ext: String,
    },
    /// Append a suffix before the extension in the same directory.
    ///
    /// `input/foo.mkv` → `input/foo_converted.mkv` (when `suffix = "_converted"`)
    Suffix {
        /// The suffix string to append to the stem (e.g. `"_hq"`).
        suffix: String,
        /// The output container extension.
        ext: String,
    },
    /// Use a template string with named placeholders.
    ///
    /// Supported placeholders: `{name}` (stem), `{ext}` (extension with dot),
    /// `{dir}` (parent dir), `{index}` (zero-padded job index).
    Template(String),
}

impl OutputNaming {
    /// Derive the output path for a given input path and job index.
    pub fn derive_output(
        &self,
        input_path: &str,
        job_index: usize,
    ) -> Result<String, BatchError> {
        let path = Path::new(input_path);
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let parent = path
            .parent()
            .and_then(|p| p.to_str())
            .unwrap_or(".");

        match self {
            Self::SameDir { ext } => {
                let out = format!("{}/{}.{}", parent, stem, ext.trim_start_matches('.'));
                Ok(out)
            }
            Self::OutputDir { dir, ext } => {
                let out = format!("{}/{}.{}", dir.trim_end_matches('/'), stem, ext.trim_start_matches('.'));
                Ok(out)
            }
            Self::Suffix { suffix, ext } => {
                let out = format!("{}/{}{}.{}", parent, stem, suffix, ext.trim_start_matches('.'));
                Ok(out)
            }
            Self::Template(template) => {
                let original_ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");
                let result = template
                    .replace("{name}", stem)
                    .replace("{ext}", original_ext)
                    .replace("{dir}", parent)
                    .replace("{index}", &format!("{:04}", job_index));

                // Check for any remaining unresolved placeholders.
                if let Some(start) = result.find('{') {
                    if let Some(end) = result[start..].find('}') {
                        let placeholder = result[start + 1..start + end].to_string();
                        return Err(BatchError::UnknownTemplatePlaceholder {
                            placeholder,
                            template: template.clone(),
                        });
                    }
                }
                Ok(result)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch specification
// ─────────────────────────────────────────────────────────────────────────────

/// A complete batch transcoding specification.
#[derive(Debug, Clone)]
pub struct BatchSpec {
    /// How input files are discovered.
    pub inputs: InputSource,
    /// How each output file path is derived from the input.
    pub output_naming: OutputNaming,
    /// FFmpeg-style arguments applied uniformly to every file in the batch.
    pub shared_args: Vec<String>,
    /// If `true`, overwrite existing output files without asking.
    pub overwrite: bool,
    /// If `true`, compute jobs but do not schedule them for execution.
    pub dry_run: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch transcode job
// ─────────────────────────────────────────────────────────────────────────────

/// A single file-level transcode job produced by batch translation.
#[derive(Debug, Clone)]
pub struct BatchTranscodeJob {
    /// Job index within the batch (0-based).
    pub index: usize,
    /// The resolved input file path.
    pub input_path: String,
    /// The derived output file path.
    pub output_path: String,
    /// Target video codec (OxiMedia canonical name).
    pub video_codec: Option<String>,
    /// Target audio codec (OxiMedia canonical name).
    pub audio_codec: Option<String>,
    /// Video bitrate string (e.g. `"2M"`).
    pub video_bitrate: Option<String>,
    /// Audio bitrate string (e.g. `"128k"`).
    pub audio_bitrate: Option<String>,
    /// CRF value for quality-based encoding.
    pub crf: Option<f64>,
    /// Whether to overwrite the output if it already exists.
    pub overwrite: bool,
    /// Whether this job is a dry-run (no actual transcoding).
    pub dry_run: bool,
    /// Metadata key/value pairs applied to this file.
    pub metadata: HashMap<String, String>,
    /// Diagnostics (patent substitutions, unknown options, etc.).
    pub diagnostics: Vec<Diagnostic>,
    /// Container format override, if any.
    pub format: Option<String>,
    /// Encoding preset.
    pub preset: Option<String>,
}

impl BatchTranscodeJob {
    /// Return `true` if any of the diagnostics are error-level.
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(|d| d.is_error())
    }

    /// Return a summary string suitable for dry-run output.
    pub fn dry_run_summary(&self) -> String {
        format!(
            "[{}] {} -> {} (video={}, audio={}, crf={:?})",
            self.index,
            self.input_path,
            self.output_path,
            self.video_codec.as_deref().unwrap_or("copy"),
            self.audio_codec.as_deref().unwrap_or("copy"),
            self.crf,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch result
// ─────────────────────────────────────────────────────────────────────────────

/// The result of translating a [`BatchSpec`].
#[derive(Debug)]
pub struct BatchResult {
    /// All generated jobs, in order.
    pub jobs: Vec<BatchTranscodeJob>,
    /// Top-level diagnostics (e.g. glob expansion hints).
    pub diagnostics: Vec<Diagnostic>,
}

impl BatchResult {
    /// Return `true` if any job or top-level diagnostic is error-level.
    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(|d| d.is_error())
            || self.jobs.iter().any(|j| j.has_errors())
    }

    /// Return all job output paths.
    pub fn output_paths(&self) -> Vec<&str> {
        self.jobs.iter().map(|j| j.output_path.as_str()).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Translation
// ─────────────────────────────────────────────────────────────────────────────

/// Translate a [`BatchSpec`] into a sequence of [`BatchTranscodeJob`] entries.
///
/// The shared FFmpeg arguments are parsed once via [`FfmpegArgs`] and then
/// applied to each input file in turn. Codec names are resolved through
/// [`CodecMap`] for patent-motivated substitution warnings.
pub fn translate_batch(spec: &BatchSpec) -> Result<BatchResult, BatchError> {
    let input_files = spec.inputs.resolve_files()?;

    let codec_map = CodecMap::new();
    let mut top_diagnostics: Vec<Diagnostic> = Vec::new();

    // Parse shared args once to extract codec/bitrate/etc. settings.
    // We synthesise a minimal FFmpeg argv: `-i PLACEHOLDER <shared_args> output.mkv`
    let mut argv: Vec<String> = vec![
        "-i".into(),
        "__batch_placeholder__.mkv".into(),
    ];
    argv.extend_from_slice(&spec.shared_args);
    argv.push("__batch_output__.mkv".into());

    let parsed = FfmpegArgs::parse(&argv)
        .map_err(|e| BatchError::ArgumentParseError(e.to_string()))?;

    // Extract settings from the first (and only) output spec.
    let output_spec = parsed.outputs.into_iter().next();

    // Resolved codec names.
    let mut video_codec: Option<String> = None;
    let mut audio_codec: Option<String> = None;
    let mut video_bitrate: Option<String> = None;
    let mut audio_bitrate: Option<String> = None;
    let mut crf: Option<f64> = None;
    let mut format: Option<String> = None;
    let mut preset: Option<String> = None;
    let mut metadata: HashMap<String, String> = HashMap::new();
    let mut per_file_diagnostics: Vec<Diagnostic> = Vec::new();

    if let Some(ref out) = output_spec {
        format = out.format.clone();
        preset = out.preset.clone();
        metadata = out.metadata.clone();

        for stream_opt in &out.stream_options {
            use crate::arg_parser::StreamType;
            match stream_opt.stream_type {
                StreamType::Video | StreamType::All => {
                    if let Some(codec_name) = &stream_opt.codec {
                        let resolved = resolve_codec(&codec_map, codec_name, &mut per_file_diagnostics);
                        video_codec = Some(resolved);
                    }
                    if let Some(br) = &stream_opt.bitrate {
                        video_bitrate = Some(br.clone());
                    }
                    if let Some(c) = stream_opt.crf {
                        crf = Some(c);
                    }
                }
                StreamType::Audio => {
                    if let Some(codec_name) = &stream_opt.codec {
                        let resolved = resolve_codec(&codec_map, codec_name, &mut per_file_diagnostics);
                        audio_codec = Some(resolved);
                    }
                    if let Some(br) = &stream_opt.bitrate {
                        audio_bitrate = Some(br.clone());
                    }
                }
                StreamType::Subtitle => {}
            }
        }
    }

    // If the input source is a glob/directory pattern, emit an info diagnostic.
    if matches!(spec.inputs, InputSource::GlobPattern(_) | InputSource::Directory { .. }) {
        top_diagnostics.push(Diagnostic::info(
            "Batch glob/directory inputs require caller-side expansion before execution",
        ));
    }

    // Build one job per input file.
    let mut jobs: Vec<BatchTranscodeJob> = Vec::with_capacity(input_files.len());
    for (index, input_path) in input_files.iter().enumerate() {
        let output_path = spec.output_naming.derive_output(input_path, index)?;

        jobs.push(BatchTranscodeJob {
            index,
            input_path: input_path.clone(),
            output_path,
            video_codec: video_codec.clone(),
            audio_codec: audio_codec.clone(),
            video_bitrate: video_bitrate.clone(),
            audio_bitrate: audio_bitrate.clone(),
            crf,
            overwrite: spec.overwrite,
            dry_run: spec.dry_run,
            metadata: metadata.clone(),
            diagnostics: per_file_diagnostics.clone(),
            format: format.clone(),
            preset: preset.clone(),
        });
    }

    Ok(BatchResult {
        jobs,
        diagnostics: top_diagnostics,
    })
}

/// Resolve a codec name via the [`CodecMap`], recording patent-substitution diagnostics.
fn resolve_codec(
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

/// Parse a batch file list from a newline-separated text block.
///
/// Lines beginning with `#` are treated as comments.
/// Empty lines are ignored.
///
/// ```
/// use oximedia_compat_ffmpeg::batch_mode::parse_batch_file_list;
///
/// let text = "# my batch list\nclip1.mp4\nclip2.mp4\n\nclip3.mkv\n";
/// let files = parse_batch_file_list(text);
/// assert_eq!(files, vec!["clip1.mp4", "clip2.mp4", "clip3.mkv"]);
/// ```
pub fn parse_batch_file_list(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.to_string())
        .collect()
}

/// Estimate the total duration of a batch job given per-file duration hints.
///
/// Returns `None` if no durations are available.
pub fn estimate_batch_duration(durations_secs: &[f64]) -> Option<f64> {
    if durations_secs.is_empty() {
        return None;
    }
    Some(durations_secs.iter().copied().sum())
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build a PathBuf from a string slice (useful for caller code)
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a string path to a [`PathBuf`].
pub fn to_path_buf(s: &str) -> PathBuf {
    PathBuf::from(s)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── InputSource tests ────────────────────────────────────────────────────

    #[test]
    fn test_input_source_file_list_resolve() {
        let src = InputSource::FileList(vec!["a.mp4".into(), "b.mp4".into()]);
        let files = src.resolve_files().expect("should resolve");
        assert_eq!(files, vec!["a.mp4", "b.mp4"]);
    }

    #[test]
    fn test_input_source_file_list_empty() {
        let src = InputSource::FileList(vec![]);
        let err = src.resolve_files().expect_err("empty list should error");
        assert!(matches!(err, BatchError::NoInputs));
    }

    #[test]
    fn test_input_source_glob_pattern_resolve() {
        let src = InputSource::GlobPattern("input_*.mp4".into());
        let files = src.resolve_files().expect("glob should not error");
        assert_eq!(files, vec!["input_*.mp4"]);
    }

    #[test]
    fn test_input_source_glob_empty_errors() {
        let src = InputSource::GlobPattern("   ".into());
        let err = src.resolve_files().expect_err("empty glob should error");
        assert!(matches!(err, BatchError::MalformedGlob(_)));
    }

    #[test]
    fn test_input_source_directory_resolve() {
        let src = InputSource::Directory {
            path: "/media/input".into(),
            extensions: vec!["mp4".into(), "mkv".into()],
        };
        let files = src.resolve_files().expect("directory should resolve");
        assert!(!files.is_empty());
        assert!(files[0].contains("input"));
    }

    // ── OutputNaming tests ───────────────────────────────────────────────────

    #[test]
    fn test_output_naming_same_dir() {
        let naming = OutputNaming::SameDir { ext: "mkv".into() };
        let out = naming.derive_output("/videos/clip.mp4", 0).expect("should derive");
        assert!(out.ends_with("clip.mkv"), "got: {}", out);
        assert!(out.contains("/videos/"), "got: {}", out);
    }

    #[test]
    fn test_output_naming_output_dir() {
        let naming = OutputNaming::OutputDir {
            dir: "/out".into(),
            ext: "webm".into(),
        };
        let out = naming.derive_output("/source/video.avi", 0).expect("should derive");
        assert!(out.starts_with("/out/"), "got: {}", out);
        assert!(out.ends_with("video.webm"), "got: {}", out);
    }

    #[test]
    fn test_output_naming_suffix() {
        let naming = OutputNaming::Suffix {
            suffix: "_hq".into(),
            ext: "mp4".into(),
        };
        let out = naming.derive_output("clip.mkv", 0).expect("should derive");
        assert!(out.contains("clip_hq.mp4"), "got: {}", out);
    }

    #[test]
    fn test_output_naming_template_basic() {
        let naming = OutputNaming::Template("{dir}/out_{name}_{index}.{ext}".into());
        let out = naming.derive_output("/media/foo.mkv", 3).expect("should derive");
        assert!(out.contains("out_foo_0003"), "got: {}", out);
        assert!(out.contains(".mkv"), "got: {}", out);
    }

    #[test]
    fn test_output_naming_template_unknown_placeholder() {
        let naming = OutputNaming::Template("{unknown_field}.mp4".into());
        let err = naming.derive_output("clip.mkv", 0).expect_err("should fail");
        match err {
            BatchError::UnknownTemplatePlaceholder { placeholder, .. } => {
                assert_eq!(placeholder, "unknown_field");
            }
            other => panic!("unexpected error: {:?}", other),
        }
    }

    // ── translate_batch tests ────────────────────────────────────────────────

    #[test]
    fn test_translate_batch_basic_av1() {
        let spec = BatchSpec {
            inputs: InputSource::FileList(vec!["a.mp4".into(), "b.mp4".into()]),
            output_naming: OutputNaming::Suffix {
                suffix: "_out".into(),
                ext: "mkv".into(),
            },
            shared_args: vec![
                "-c:v".into(), "libaom-av1".into(),
                "-crf".into(), "28".into(),
                "-c:a".into(), "libopus".into(),
            ],
            overwrite: false,
            dry_run: false,
        };

        let result = translate_batch(&spec).expect("should succeed");
        assert_eq!(result.jobs.len(), 2);
        assert_eq!(result.jobs[0].video_codec.as_deref(), Some("av1"));
        assert_eq!(result.jobs[0].audio_codec.as_deref(), Some("opus"));
        assert_eq!(result.jobs[0].crf, Some(28.0));
        assert!(!result.has_errors());
    }

    #[test]
    fn test_translate_batch_patent_substitution_diagnostic() {
        let spec = BatchSpec {
            inputs: InputSource::FileList(vec!["input.avi".into()]),
            output_naming: OutputNaming::SameDir { ext: "mp4".into() },
            shared_args: vec!["-c:v".into(), "libx264".into()],
            overwrite: true,
            dry_run: false,
        };

        let result = translate_batch(&spec).expect("should succeed");
        assert_eq!(result.jobs.len(), 1);
        // libx264 is patent-substituted to av1
        assert_eq!(result.jobs[0].video_codec.as_deref(), Some("av1"));
        let has_patent_diag = result.jobs[0].diagnostics.iter().any(|d| {
            matches!(&d.kind, crate::diagnostics::DiagnosticKind::PatentCodecSubstituted { .. })
        });
        assert!(has_patent_diag, "should have patent substitution diagnostic");
    }

    #[test]
    fn test_translate_batch_dry_run_flag() {
        let spec = BatchSpec {
            inputs: InputSource::FileList(vec!["x.mp4".into()]),
            output_naming: OutputNaming::SameDir { ext: "mkv".into() },
            shared_args: vec!["-c:v".into(), "av1".into()],
            overwrite: false,
            dry_run: true,
        };

        let result = translate_batch(&spec).expect("should succeed");
        assert!(result.jobs[0].dry_run, "job should be marked dry_run");
        let summary = result.jobs[0].dry_run_summary();
        assert!(summary.contains("x.mp4"), "summary should mention input");
        assert!(summary.contains("x.mkv"), "summary should mention output");
    }

    #[test]
    fn test_translate_batch_overwrite_flag_propagated() {
        let spec = BatchSpec {
            inputs: InputSource::FileList(vec!["clip.mp4".into()]),
            output_naming: OutputNaming::SameDir { ext: "webm".into() },
            shared_args: vec![],
            overwrite: true,
            dry_run: false,
        };

        let result = translate_batch(&spec).expect("should succeed");
        assert!(result.jobs[0].overwrite);
    }

    #[test]
    fn test_translate_batch_job_indices() {
        let spec = BatchSpec {
            inputs: InputSource::FileList(vec!["a.mp4".into(), "b.mp4".into(), "c.mp4".into()]),
            output_naming: OutputNaming::SameDir { ext: "mkv".into() },
            shared_args: vec![],
            overwrite: false,
            dry_run: false,
        };

        let result = translate_batch(&spec).expect("should succeed");
        for (i, job) in result.jobs.iter().enumerate() {
            assert_eq!(job.index, i, "job index should match position");
        }
    }

    #[test]
    fn test_translate_batch_output_paths() {
        let spec = BatchSpec {
            inputs: InputSource::FileList(vec![
                "/source/clip1.avi".into(),
                "/source/clip2.avi".into(),
            ]),
            output_naming: OutputNaming::OutputDir {
                dir: "/dest".into(),
                ext: "mp4".into(),
            },
            shared_args: vec![],
            overwrite: false,
            dry_run: false,
        };

        let result = translate_batch(&spec).expect("should succeed");
        let paths = result.output_paths();
        assert!(paths[0].starts_with("/dest/"), "path: {}", paths[0]);
        assert!(paths[0].ends_with("clip1.mp4"), "path: {}", paths[0]);
        assert!(paths[1].ends_with("clip2.mp4"), "path: {}", paths[1]);
    }

    #[test]
    fn test_translate_batch_glob_info_diagnostic() {
        let spec = BatchSpec {
            inputs: InputSource::GlobPattern("*.mp4".into()),
            output_naming: OutputNaming::SameDir { ext: "webm".into() },
            shared_args: vec![],
            overwrite: false,
            dry_run: true,
        };

        let result = translate_batch(&spec).expect("should succeed");
        // Should have an info diagnostic about glob expansion
        let has_info = result.diagnostics.iter().any(|d| {
            matches!(&d.kind, crate::diagnostics::DiagnosticKind::Info { .. })
        });
        assert!(has_info, "should have info diagnostic for glob input");
    }

    // ── parse_batch_file_list tests ──────────────────────────────────────────

    #[test]
    fn test_parse_batch_file_list_basic() {
        let text = "clip1.mp4\nclip2.mp4\nclip3.mkv\n";
        let files = parse_batch_file_list(text);
        assert_eq!(files, vec!["clip1.mp4", "clip2.mp4", "clip3.mkv"]);
    }

    #[test]
    fn test_parse_batch_file_list_comments_and_blanks() {
        let text = "# header\n\nclip1.mp4\n# comment\n\nclip2.mp4\n";
        let files = parse_batch_file_list(text);
        assert_eq!(files, vec!["clip1.mp4", "clip2.mp4"]);
    }

    #[test]
    fn test_parse_batch_file_list_empty_input() {
        let files = parse_batch_file_list("");
        assert!(files.is_empty());
    }

    #[test]
    fn test_estimate_batch_duration() {
        let durations = [30.0_f64, 45.0, 120.0];
        let total = estimate_batch_duration(&durations).expect("should return total");
        assert!((total - 195.0).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_batch_duration_empty() {
        let total = estimate_batch_duration(&[]);
        assert!(total.is_none());
    }

    #[test]
    fn test_to_path_buf() {
        let pb = to_path_buf("/tmp/test.mp4");
        assert_eq!(pb, PathBuf::from("/tmp/test.mp4"));
    }
}
