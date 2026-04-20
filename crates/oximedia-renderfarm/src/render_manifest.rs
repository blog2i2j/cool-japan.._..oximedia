//! Render manifest — describes what must be produced by a render job.
//!
//! A [`RenderManifest`] lists the output targets together with their expected
//! file paths and format descriptors. [`ManifestValidator`] checks that a
//! manifest is self-consistent before the farm accepts the job.

#![allow(dead_code)]

use std::collections::HashSet;
use std::path::PathBuf;

/// Errors raised during manifest validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    /// The manifest contains no targets.
    NoTargets,
    /// Two or more targets share the same output path.
    DuplicatePath(PathBuf),
    /// A target has an empty or blank output path.
    EmptyPath,
    /// A required field is missing in a target.
    MissingField(String),
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoTargets => write!(f, "manifest has no render targets"),
            Self::DuplicatePath(p) => write!(f, "duplicate output path: {}", p.display()),
            Self::EmptyPath => write!(f, "output path must not be empty"),
            Self::MissingField(n) => write!(f, "required field missing: {n}"),
        }
    }
}

impl std::error::Error for ManifestError {}

/// The output format / container of a render target.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OutputFormat {
    /// Single-frame EXR image.
    OpenExr,
    /// PNG image sequence.
    Png,
    /// JPEG image sequence.
    Jpeg,
    /// Tiff image sequence.
    Tiff,
    /// MP4 video container.
    Mp4,
    /// MOV video container.
    Mov,
    /// Custom format identifier.
    Custom(String),
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenExr => write!(f, "openexr"),
            Self::Png => write!(f, "png"),
            Self::Jpeg => write!(f, "jpeg"),
            Self::Tiff => write!(f, "tiff"),
            Self::Mp4 => write!(f, "mp4"),
            Self::Mov => write!(f, "mov"),
            Self::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// A single output target within a render manifest.
///
/// Describes one stream of rendered output — e.g. the beauty pass as EXR, or
/// a proxy MP4 alongside it.
#[derive(Debug, Clone)]
pub struct RenderTarget {
    /// Human-readable name for this target (e.g. `"beauty"`, `"proxy"`).
    pub name: String,
    /// Filesystem path where output files will be written.
    pub output_path: PathBuf,
    /// Container / image format for this target.
    pub format: OutputFormat,
    /// Width of the rendered output in pixels.
    pub width: u32,
    /// Height of the rendered output in pixels.
    pub height: u32,
    /// Whether alpha channel is present in the output.
    pub has_alpha: bool,
}

impl RenderTarget {
    /// Create a new render target with the minimum required fields.
    pub fn new(
        name: impl Into<String>,
        output_path: impl Into<PathBuf>,
        format: OutputFormat,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            name: name.into(),
            output_path: output_path.into(),
            format,
            width,
            height,
            has_alpha: false,
        }
    }

    /// Total pixel count for one frame of this target.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Estimated bytes per frame assuming 4 channels × 4 bytes each.
    #[must_use]
    pub fn estimated_frame_bytes(&self) -> u64 {
        self.pixel_count() * if self.has_alpha { 16 } else { 12 }
    }
}

/// A complete description of what a render job must produce.
///
/// Contains one or more [`RenderTarget`]s and job-level metadata.
#[derive(Debug, Clone, Default)]
pub struct RenderManifest {
    /// Unique identifier for this manifest.
    pub id: String,
    /// Human-readable project name.
    pub project: String,
    /// List of output targets.
    pub targets: Vec<RenderTarget>,
    /// Optional notes attached to the manifest.
    pub notes: Option<String>,
}

impl RenderManifest {
    /// Create a new manifest with the given id and project name.
    #[must_use]
    pub fn new(id: impl Into<String>, project: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            project: project.into(),
            targets: Vec::new(),
            notes: None,
        }
    }

    /// Add a target to this manifest.
    pub fn add_target(&mut self, target: RenderTarget) {
        self.targets.push(target);
    }

    /// Number of targets.
    #[must_use]
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Sum of estimated frame bytes across all targets.
    #[must_use]
    pub fn total_estimated_frame_bytes(&self) -> u64 {
        self.targets
            .iter()
            .map(RenderTarget::estimated_frame_bytes)
            .sum()
    }
}

/// Validates a [`RenderManifest`] for consistency before job submission.
#[derive(Debug, Default)]
pub struct ManifestValidator;

impl ManifestValidator {
    /// Create a new validator.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Validate `manifest` and return the first error found, or `Ok(())`.
    ///
    /// Checks performed:
    /// 1. At least one target exists.
    /// 2. No target has an empty output path.
    /// 3. No two targets share an output path.
    pub fn validate(&self, manifest: &RenderManifest) -> Result<(), ManifestError> {
        if manifest.targets.is_empty() {
            return Err(ManifestError::NoTargets);
        }

        let mut seen = HashSet::new();
        for t in &manifest.targets {
            if t.output_path.as_os_str().is_empty() {
                return Err(ManifestError::EmptyPath);
            }
            if !seen.insert(t.output_path.clone()) {
                return Err(ManifestError::DuplicatePath(t.output_path.clone()));
            }
        }

        Ok(())
    }

    /// Return `true` if the manifest passes validation.
    #[must_use]
    pub fn is_valid(&self, manifest: &RenderManifest) -> bool {
        self.validate(manifest).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_target(name: &str, path: &str) -> RenderTarget {
        RenderTarget::new(name, path, OutputFormat::OpenExr, 1920, 1080)
    }

    fn tmp_path(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-renderfarm-manifest-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn render_target_pixel_count() {
        let t = make_target("beauty", &tmp_path("beauty.exr"));
        assert_eq!(t.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn render_target_frame_bytes_without_alpha() {
        let t = make_target("beauty", &tmp_path("beauty.exr"));
        assert_eq!(t.estimated_frame_bytes(), 1920 * 1080 * 12);
    }

    #[test]
    fn render_target_frame_bytes_with_alpha() {
        let mut t = make_target("beauty", &tmp_path("beauty.exr"));
        t.has_alpha = true;
        assert_eq!(t.estimated_frame_bytes(), 1920 * 1080 * 16);
    }

    #[test]
    fn manifest_add_target_increments_count() {
        let mut m = RenderManifest::new("m1", "proj");
        m.add_target(make_target("beauty", &tmp_path("beauty.exr")));
        assert_eq!(m.target_count(), 1);
    }

    #[test]
    fn manifest_total_bytes_sums_targets() {
        let mut m = RenderManifest::new("m1", "proj");
        m.add_target(make_target("beauty", &tmp_path("beauty.exr")));
        m.add_target(make_target("z", &tmp_path("z.exr")));
        assert_eq!(m.total_estimated_frame_bytes(), 2 * 1920 * 1080 * 12);
    }

    #[test]
    fn validator_accepts_valid_manifest() {
        let mut m = RenderManifest::new("m1", "proj");
        m.add_target(make_target("beauty", &tmp_path("beauty.exr")));
        assert!(ManifestValidator::new().validate(&m).is_ok());
    }

    #[test]
    fn validator_rejects_no_targets() {
        let m = RenderManifest::new("m1", "proj");
        let err = ManifestValidator::new().validate(&m).unwrap_err();
        assert_eq!(err, ManifestError::NoTargets);
    }

    #[test]
    fn validator_rejects_duplicate_paths() {
        let mut m = RenderManifest::new("m1", "proj");
        let dup = tmp_path("out.exr");
        m.add_target(make_target("a", &dup));
        m.add_target(make_target("b", &dup));
        let err = ManifestValidator::new().validate(&m).unwrap_err();
        matches!(err, ManifestError::DuplicatePath(_));
    }

    #[test]
    fn is_valid_convenience() {
        let mut m = RenderManifest::new("m1", "proj");
        m.add_target(make_target("beauty", &tmp_path("beauty.exr")));
        assert!(ManifestValidator::new().is_valid(&m));
    }

    #[test]
    fn output_format_display() {
        assert_eq!(OutputFormat::Png.to_string(), "png");
        assert_eq!(OutputFormat::Custom("dpx".into()).to_string(), "dpx");
    }

    #[test]
    fn manifest_default_is_empty() {
        let m = RenderManifest::default();
        assert_eq!(m.target_count(), 0);
    }

    #[test]
    fn manifest_error_display_no_targets() {
        let e = ManifestError::NoTargets;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn manifest_error_display_duplicate() {
        let e = ManifestError::DuplicatePath(
            std::env::temp_dir().join("oximedia-renderfarm-manifest-x.exr"),
        );
        assert!(e.to_string().contains("duplicate"));
    }
}
