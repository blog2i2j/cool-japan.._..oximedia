//! Hardware acceleration option mapping for OxiMedia.
//!
//! Translates FFmpeg `-hwaccel`, `-hwaccel_device`, and hardware-accelerated
//! codec names (e.g. `h264_nvenc`, `hevc_vaapi`, `av1_amf`) into structured
//! [`HwAccelConfig`] records that describe how OxiMedia should configure its
//! GPU pipeline.
//!
//! ## Supported backends
//!
//! | FFmpeg name  | OxiMedia backend  | Platform    |
//! |-------------|-------------------|-------------|
//! | `cuda`      | `Cuda`            | NVIDIA      |
//! | `nvenc`     | `Cuda`            | NVIDIA      |
//! | `cuvid`     | `Cuda`            | NVIDIA (decode) |
//! | `vaapi`     | `Vaapi`           | Linux (VA-API) |
//! | `qsv`       | `QuickSync`       | Intel       |
//! | `amf`       | `Amf`             | AMD         |
//! | `videotoolbox` | `VideoToolbox` | macOS       |
//! | `v4l2m2m`   | `V4l2M2m`         | Linux (V4L2)|
//! | `opencl`    | `OpenCl`          | Cross-platform |
//! | `vulkan`    | `Vulkan`          | Cross-platform |
//! | `dxva2`     | `Dxva2`           | Windows (legacy) |
//! | `d3d11va`   | `D3d11Va`         | Windows     |
//! | `auto`      | `Auto`            | Runtime detection |
//! | `none` / `software` | `Software` | Disable HW |
//!
//! ## Example
//!
//! ```rust
//! use oximedia_compat_ffmpeg::hwaccel_compat::{translate_hwaccel, HwBackend};
//!
//! let cfg = translate_hwaccel("cuda", None, None);
//! assert_eq!(cfg.backend, HwBackend::Cuda);
//! assert!(cfg.is_gpu_enabled());
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Error returned when an hwaccel option cannot be interpreted.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum HwAccelError {
    /// The hwaccel method name is completely unknown.
    #[error("unknown hwaccel method: '{0}'")]
    UnknownMethod(String),

    /// A device index or path is invalid.
    #[error("invalid hwaccel device specification: '{0}'")]
    InvalidDevice(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend enum
// ─────────────────────────────────────────────────────────────────────────────

/// The hardware acceleration backend as understood by OxiMedia.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HwBackend {
    /// NVIDIA CUDA (encoding via NVENC, decoding via CUVID/NVDEC).
    Cuda,
    /// Linux VA-API (VAAPI).
    Vaapi,
    /// Intel Quick Sync Video (QSV).
    QuickSync,
    /// AMD Advanced Media Framework (AMF).
    Amf,
    /// Apple VideoToolbox (macOS).
    VideoToolbox,
    /// Linux Video4Linux2 mem-to-mem (V4L2M2M).
    V4l2M2m,
    /// OpenCL compute backend.
    OpenCl,
    /// Vulkan GPU compute backend.
    Vulkan,
    /// Windows DXVA2 (legacy DirectX Video Acceleration).
    Dxva2,
    /// Windows D3D11VA (DirectX 11 Video Acceleration).
    D3d11Va,
    /// macOS / iOS OpenGL ES decode.
    MediaCodec,
    /// Runtime auto-detection of best available backend.
    Auto,
    /// Software-only (no hardware acceleration).
    Software,
}

impl HwBackend {
    /// Return `true` if this backend implies GPU hardware acceleration.
    pub fn is_gpu(&self) -> bool {
        !matches!(self, Self::Software | Self::Auto)
    }

    /// Return the canonical string representation (lowercase).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Vaapi => "vaapi",
            Self::QuickSync => "qsv",
            Self::Amf => "amf",
            Self::VideoToolbox => "videotoolbox",
            Self::V4l2M2m => "v4l2m2m",
            Self::OpenCl => "opencl",
            Self::Vulkan => "vulkan",
            Self::Dxva2 => "dxva2",
            Self::D3d11Va => "d3d11va",
            Self::MediaCodec => "mediacodec",
            Self::Auto => "auto",
            Self::Software => "software",
        }
    }
}

impl std::fmt::Display for HwBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Codec role
// ─────────────────────────────────────────────────────────────────────────────

/// Whether the hardware codec is used for encoding, decoding, or both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HwCodecRole {
    /// Hardware encoder (e.g. `h264_nvenc`).
    Encode,
    /// Hardware decoder (e.g. `h264_cuvid`).
    Decode,
    /// Hardware used for both encode and decode.
    Both,
}

// ─────────────────────────────────────────────────────────────────────────────
// Codec hint
// ─────────────────────────────────────────────────────────────────────────────

/// A hardware codec hint extracted from a codec name like `h264_nvenc`.
#[derive(Debug, Clone)]
pub struct HwCodecHint {
    /// The base codec (patent-free OxiMedia equivalent), e.g. `"av1"`.
    pub base_codec: String,
    /// The detected hardware backend.
    pub backend: HwBackend,
    /// Whether this codec performs encode, decode, or both.
    pub role: HwCodecRole,
    /// Whether the original codec is patent-encumbered (and was substituted).
    pub is_patent_substituted: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// HwAccelConfig
// ─────────────────────────────────────────────────────────────────────────────

/// A fully resolved hardware acceleration configuration for OxiMedia.
#[derive(Debug, Clone)]
pub struct HwAccelConfig {
    /// The resolved hardware backend.
    pub backend: HwBackend,
    /// Device specification, e.g. `"0"` (GPU index) or `"/dev/dri/renderD128"`.
    pub device: Option<String>,
    /// Optional codec hint when the config was derived from a codec name.
    pub codec_hint: Option<HwCodecHint>,
    /// Fallback to software if the hardware backend is unavailable.
    pub allow_software_fallback: bool,
    /// Human-readable description of what OxiMedia will do.
    pub description: String,
}

impl HwAccelConfig {
    /// Return `true` if GPU hardware acceleration is enabled.
    pub fn is_gpu_enabled(&self) -> bool {
        self.backend.is_gpu()
    }

    /// Return `true` if this configuration represents pure software processing.
    pub fn is_software_only(&self) -> bool {
        matches!(self.backend, HwBackend::Software)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend parsing
// ─────────────────────────────────────────────────────────────────────────────

/// Parse an FFmpeg `-hwaccel METHOD` value into a [`HwBackend`].
///
/// Returns `None` for completely unknown backend names.
pub fn parse_hwaccel_method(method: &str) -> Option<HwBackend> {
    match method.to_lowercase().replace('-', "_").as_str() {
        "cuda" | "nvenc" | "nvdec" | "cuvid" => Some(HwBackend::Cuda),
        "vaapi" => Some(HwBackend::Vaapi),
        "qsv" | "quicksync" | "quick_sync" => Some(HwBackend::QuickSync),
        "amf" | "d3d12va_enc" => Some(HwBackend::Amf),
        "videotoolbox" | "vt" => Some(HwBackend::VideoToolbox),
        "v4l2m2m" | "v4l2" => Some(HwBackend::V4l2M2m),
        "opencl" => Some(HwBackend::OpenCl),
        "vulkan" => Some(HwBackend::Vulkan),
        "dxva2" => Some(HwBackend::Dxva2),
        "d3d11va" | "d3d11" => Some(HwBackend::D3d11Va),
        "mediacodec" => Some(HwBackend::MediaCodec),
        "auto" => Some(HwBackend::Auto),
        "none" | "software" | "sw" => Some(HwBackend::Software),
        _ => None,
    }
}

/// Extract the hardware backend from a hardware-accelerated codec name
/// like `h264_nvenc` or `hevc_vaapi`.
///
/// Returns `None` if the codec name contains no recognisable hardware suffix.
pub fn backend_from_codec_name(codec_name: &str) -> Option<HwBackend> {
    let lower = codec_name.to_lowercase();
    // Order matters: check longer suffixes first.
    if lower.contains("_nvenc") || lower.contains("_cuvid") || lower.ends_with("_cuda") {
        Some(HwBackend::Cuda)
    } else if lower.contains("_vaapi") {
        Some(HwBackend::Vaapi)
    } else if lower.contains("_qsv") {
        Some(HwBackend::QuickSync)
    } else if lower.contains("_amf") {
        Some(HwBackend::Amf)
    } else if lower.contains("_videotoolbox") || lower.contains("_vt") {
        Some(HwBackend::VideoToolbox)
    } else if lower.contains("_v4l2m2m") {
        Some(HwBackend::V4l2M2m)
    } else if lower.contains("_mf") {
        // Windows MediaFoundation
        Some(HwBackend::D3d11Va)
    } else if lower.contains("_omx") {
        Some(HwBackend::MediaCodec)
    } else {
        None
    }
}

/// Determine whether a hardware codec name is an encoder or decoder.
fn role_from_codec_name(codec_name: &str) -> HwCodecRole {
    let lower = codec_name.to_lowercase();
    if lower.contains("_cuvid") || lower.contains("_dec") {
        HwCodecRole::Decode
    } else {
        HwCodecRole::Encode
    }
}

/// Map the base codec from a hardware codec name to its patent-free OxiMedia equivalent.
fn base_codec_from_hw_name(codec_name: &str) -> (String, bool) {
    let lower = codec_name.to_lowercase();
    if lower.starts_with("av1") {
        ("av1".to_string(), false)
    } else if lower.starts_with("vp9") {
        ("vp9".to_string(), false)
    } else if lower.starts_with("vp8") {
        ("vp8".to_string(), false)
    } else if lower.starts_with("h264") || lower.starts_with("libx264") {
        ("av1".to_string(), true) // patent codec → AV1
    } else if lower.starts_with("hevc") || lower.starts_with("h265") || lower.starts_with("libx265") {
        ("av1".to_string(), true) // patent codec → AV1
    } else if lower.starts_with("mpeg2") || lower.starts_with("mpeg4") {
        ("av1".to_string(), true)
    } else if lower.starts_with("aac") {
        ("opus".to_string(), true)
    } else if lower.starts_with("mp3") {
        ("opus".to_string(), true)
    } else {
        // Unknown base codec — return as-is without substitution.
        (lower.split('_').next().unwrap_or(codec_name).to_string(), false)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public translation API
// ─────────────────────────────────────────────────────────────────────────────

/// Translate a `-hwaccel METHOD` option (plus optional device) into a
/// [`HwAccelConfig`].
///
/// # Arguments
///
/// * `method` — The value of `-hwaccel` (e.g. `"cuda"`, `"vaapi"`, `"auto"`).
/// * `device` — Optional value of `-hwaccel_device` (e.g. `"0"`, `"/dev/dri/renderD128"`).
/// * `output_device` — Optional value of `-hwaccel_output_format`.
///
/// # Fallback behaviour
///
/// If the method name is not recognised, [`HwBackend::Auto`] is used and a
/// note is embedded in the description. This avoids hard failures for exotic
/// or future backends.
pub fn translate_hwaccel(
    method: &str,
    device: Option<&str>,
    output_device: Option<&str>,
) -> HwAccelConfig {
    let backend = parse_hwaccel_method(method).unwrap_or(HwBackend::Auto);

    let description = build_description(&backend, device, output_device, method);

    HwAccelConfig {
        backend,
        device: device.map(str::to_string),
        codec_hint: None,
        allow_software_fallback: true,
        description,
    }
}

/// Translate a hardware-accelerated codec name (e.g. `"h264_nvenc"`,
/// `"hevc_vaapi"`) into a [`HwAccelConfig`] that describes both the
/// patent-free codec substitution and the GPU backend to use.
///
/// # Returns
///
/// Returns `Err(HwAccelError::UnknownMethod)` if the codec name has no
/// recognisable hardware suffix **and** is not a known base codec.
pub fn translate_hw_codec(codec_name: &str) -> Result<HwAccelConfig, HwAccelError> {
    let backend = backend_from_codec_name(codec_name)
        .ok_or_else(|| HwAccelError::UnknownMethod(codec_name.to_string()))?;

    let role = role_from_codec_name(codec_name);
    let (base_codec, is_patent) = base_codec_from_hw_name(codec_name);

    let description = if is_patent {
        format!(
            "Hardware codec '{}' is patent-encumbered; using '{}' with {} backend instead.",
            codec_name, base_codec, backend
        )
    } else {
        format!(
            "Hardware codec '{}' maps to '{}' with {} backend.",
            codec_name, base_codec, backend
        )
    };

    let hint = HwCodecHint {
        base_codec,
        backend: backend.clone(),
        role,
        is_patent_substituted: is_patent,
    };

    Ok(HwAccelConfig {
        backend,
        device: None,
        codec_hint: Some(hint),
        allow_software_fallback: true,
        description,
    })
}

/// Build a human-readable description for an [`HwAccelConfig`].
fn build_description(
    backend: &HwBackend,
    device: Option<&str>,
    output_format: Option<&str>,
    original_method: &str,
) -> String {
    let backend_str = backend.as_str();
    let mut parts = Vec::new();

    if backend.is_gpu() {
        parts.push(format!(
            "GPU acceleration enabled via {} backend",
            backend_str
        ));
    } else if matches!(backend, HwBackend::Auto) {
        if original_method.to_lowercase() != "auto" {
            parts.push(format!(
                "Unknown hwaccel '{}'; using runtime auto-detection",
                original_method
            ));
        } else {
            parts.push("Runtime hwaccel auto-detection enabled".to_string());
        }
    } else {
        parts.push("Software processing (no GPU acceleration)".to_string());
    }

    if let Some(dev) = device {
        parts.push(format!("device: {}", dev));
    }

    if let Some(fmt) = output_format {
        parts.push(format!("output format: {}", fmt));
    }

    parts.join("; ")
}

/// A summary of all hardware codec options detected in a set of FFmpeg arguments.
#[derive(Debug, Default)]
pub struct HwAccelSummary {
    /// The global `-hwaccel` config, if present.
    pub global_config: Option<HwAccelConfig>,
    /// Per-codec hardware configs extracted from codec names.
    pub codec_configs: Vec<HwAccelConfig>,
    /// Codec names that contained no recognisable hardware suffix.
    pub unrecognised_hw_codecs: Vec<String>,
}

impl HwAccelSummary {
    /// Return `true` if any hardware acceleration is configured.
    pub fn has_hw_accel(&self) -> bool {
        self.global_config
            .as_ref()
            .is_some_and(|c| c.is_gpu_enabled())
            || self.codec_configs.iter().any(|c| c.is_gpu_enabled())
    }

    /// Return all backends referenced in this summary (deduplicated).
    pub fn backends(&self) -> Vec<HwBackend> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();

        if let Some(cfg) = &self.global_config {
            if seen.insert(cfg.backend.clone()) {
                result.push(cfg.backend.clone());
            }
        }
        for cfg in &self.codec_configs {
            if seen.insert(cfg.backend.clone()) {
                result.push(cfg.backend.clone());
            }
        }

        result
    }
}

/// Scan a list of codec names and produce a consolidated [`HwAccelSummary`].
///
/// This is useful when translating a full FFmpeg command that may mix hardware
/// and software codecs.
pub fn build_hw_accel_summary(
    hwaccel_method: Option<&str>,
    hwaccel_device: Option<&str>,
    codec_names: &[&str],
) -> HwAccelSummary {
    let mut summary = HwAccelSummary::default();

    if let Some(method) = hwaccel_method {
        summary.global_config = Some(translate_hwaccel(method, hwaccel_device, None));
    }

    for &codec in codec_names {
        match translate_hw_codec(codec) {
            Ok(cfg) => summary.codec_configs.push(cfg),
            Err(_) => {
                // Only record as unrecognised if it actually looks like a hw codec name.
                if codec.contains('_') && !codec.starts_with("lib") {
                    summary.unrecognised_hw_codecs.push(codec.to_string());
                }
            }
        }
    }

    summary
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_hwaccel_cuda() {
        let cfg = translate_hwaccel("cuda", None, None);
        assert_eq!(cfg.backend, HwBackend::Cuda);
        assert!(cfg.is_gpu_enabled());
        assert!(!cfg.is_software_only());
    }

    #[test]
    fn test_translate_hwaccel_vaapi_with_device() {
        let cfg = translate_hwaccel("vaapi", Some("/dev/dri/renderD128"), None);
        assert_eq!(cfg.backend, HwBackend::Vaapi);
        assert_eq!(cfg.device.as_deref(), Some("/dev/dri/renderD128"));
    }

    #[test]
    fn test_translate_hwaccel_software() {
        let cfg = translate_hwaccel("none", None, None);
        assert_eq!(cfg.backend, HwBackend::Software);
        assert!(cfg.is_software_only());
        assert!(!cfg.is_gpu_enabled());
    }

    #[test]
    fn test_translate_hwaccel_auto() {
        let cfg = translate_hwaccel("auto", None, None);
        assert_eq!(cfg.backend, HwBackend::Auto);
        assert!(!cfg.is_gpu_enabled());
    }

    #[test]
    fn test_translate_hwaccel_unknown_falls_back_to_auto() {
        let cfg = translate_hwaccel("future_accelerator_xyz", None, None);
        assert_eq!(cfg.backend, HwBackend::Auto);
        assert!(cfg.description.contains("Unknown hwaccel"));
    }

    #[test]
    fn test_translate_hw_codec_h264_nvenc() {
        let cfg = translate_hw_codec("h264_nvenc").expect("should succeed");
        assert_eq!(cfg.backend, HwBackend::Cuda);
        let hint = cfg.codec_hint.as_ref().expect("should have hint");
        assert_eq!(hint.base_codec, "av1"); // patent substitution
        assert!(hint.is_patent_substituted);
        assert_eq!(hint.role, HwCodecRole::Encode);
    }

    #[test]
    fn test_translate_hw_codec_hevc_vaapi() {
        let cfg = translate_hw_codec("hevc_vaapi").expect("should succeed");
        assert_eq!(cfg.backend, HwBackend::Vaapi);
        let hint = cfg.codec_hint.as_ref().expect("should have hint");
        assert_eq!(hint.base_codec, "av1");
        assert!(hint.is_patent_substituted);
    }

    #[test]
    fn test_translate_hw_codec_av1_nvenc() {
        let cfg = translate_hw_codec("av1_nvenc").expect("should succeed");
        assert_eq!(cfg.backend, HwBackend::Cuda);
        let hint = cfg.codec_hint.as_ref().expect("should have hint");
        assert_eq!(hint.base_codec, "av1");
        assert!(!hint.is_patent_substituted);
    }

    #[test]
    fn test_translate_hw_codec_h264_cuvid_is_decode() {
        let cfg = translate_hw_codec("h264_cuvid").expect("should succeed");
        let hint = cfg.codec_hint.as_ref().expect("should have hint");
        assert_eq!(hint.role, HwCodecRole::Decode);
    }

    #[test]
    fn test_translate_hw_codec_no_hw_suffix_returns_error() {
        // "libopus" has no hardware suffix and starts with "lib"
        let result = translate_hw_codec("libopus");
        assert!(result.is_err());
    }

    #[test]
    fn test_build_hw_accel_summary() {
        let summary = build_hw_accel_summary(
            Some("cuda"),
            Some("0"),
            &["h264_nvenc", "hevc_vaapi", "libopus"],
        );
        assert!(summary.has_hw_accel());
        assert_eq!(summary.codec_configs.len(), 2); // h264_nvenc + hevc_vaapi
        let backends = summary.backends();
        assert!(backends.contains(&HwBackend::Cuda));
        assert!(backends.contains(&HwBackend::Vaapi));
    }

    #[test]
    fn test_hw_backend_display() {
        assert_eq!(HwBackend::Cuda.to_string(), "cuda");
        assert_eq!(HwBackend::Vaapi.to_string(), "vaapi");
        assert_eq!(HwBackend::QuickSync.to_string(), "qsv");
        assert_eq!(HwBackend::Software.to_string(), "software");
    }

    #[test]
    fn test_parse_hwaccel_all_known_methods() {
        let cases = [
            ("cuda", HwBackend::Cuda),
            ("nvenc", HwBackend::Cuda),
            ("vaapi", HwBackend::Vaapi),
            ("qsv", HwBackend::QuickSync),
            ("amf", HwBackend::Amf),
            ("videotoolbox", HwBackend::VideoToolbox),
            ("v4l2m2m", HwBackend::V4l2M2m),
            ("opencl", HwBackend::OpenCl),
            ("vulkan", HwBackend::Vulkan),
            ("dxva2", HwBackend::Dxva2),
            ("d3d11va", HwBackend::D3d11Va),
            ("auto", HwBackend::Auto),
            ("none", HwBackend::Software),
            ("software", HwBackend::Software),
        ];
        for (method, expected) in &cases {
            let result = parse_hwaccel_method(method);
            assert_eq!(result.as_ref(), Some(expected), "method={}", method);
        }
    }

    #[test]
    fn test_parse_hwaccel_unknown_returns_none() {
        assert!(parse_hwaccel_method("totally_unknown_hw_xyz").is_none());
    }

    #[test]
    fn test_summary_no_hw_accel_when_software() {
        let summary = build_hw_accel_summary(Some("none"), None, &[]);
        assert!(!summary.has_hw_accel());
    }

    #[test]
    fn test_backend_from_codec_name_all_variants() {
        assert_eq!(backend_from_codec_name("h264_nvenc"), Some(HwBackend::Cuda));
        assert_eq!(backend_from_codec_name("h264_vaapi"), Some(HwBackend::Vaapi));
        assert_eq!(backend_from_codec_name("h264_qsv"), Some(HwBackend::QuickSync));
        assert_eq!(backend_from_codec_name("h264_amf"), Some(HwBackend::Amf));
        assert_eq!(
            backend_from_codec_name("h264_videotoolbox"),
            Some(HwBackend::VideoToolbox)
        );
        assert_eq!(
            backend_from_codec_name("h264_v4l2m2m"),
            Some(HwBackend::V4l2M2m)
        );
        assert_eq!(backend_from_codec_name("libopus"), None);
        assert_eq!(backend_from_codec_name("flac"), None);
    }
}
