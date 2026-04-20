//! Hardware acceleration capability types.
//!
//! Provides the rich capability model returned by platform probing:
//! [`HwKind`], [`HwAccelDevice`], and [`HwAccelCapabilities`].

use std::path::PathBuf;

// ─── HwKind ──────────────────────────────────────────────────────────────────

/// High-level hardware acceleration family.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HwKind {
    /// Apple VideoToolbox (macOS / iOS).
    VideoToolbox,
    /// Video Acceleration API (Linux, via DRM/KMS — Intel, AMD, Nouveau).
    Vaapi,
    /// CUDA-based hardware acceleration (NVIDIA NVENC/NVDEC).
    ///
    /// Included for completeness; probing requires presence of
    /// `/dev/nvidia0` on Linux. Windows probing is not yet implemented.
    Cuda,
}

// ─── HwAccelDevice ───────────────────────────────────────────────────────────

/// A single hardware-accelerated device discovered during platform probing.
///
/// Codec names follow the OxiMedia convention (`"h264"`, `"hevc"`, `"av1"`,
/// `"vp9"`, `"vp8"`, `"mjpeg"`).  Note that OxiMedia's Green-List-only
/// [`CodecId`](oximedia_core::types::CodecId) does not include H.264/HEVC;
/// those names appear here because VideoToolbox and VAAPI do in fact support
/// them on the underlying OS hardware — they are reported verbatim so callers
/// can decide whether to use them via FFmpeg compat or platform APIs.
#[derive(Debug, Clone)]
pub struct HwAccelDevice {
    /// Backend kind (VideoToolbox, VAAPI, …).
    pub kind: HwKind,
    /// Kernel-level driver name, if detectable (e.g., `"i915"`, `"amdgpu"`).
    pub driver: Option<String>,
    /// DRI render node path (Linux only, e.g., `/dev/dri/renderD128`).
    pub render_node: Option<PathBuf>,
    /// Codec names supported by this device.
    pub supported_codecs: Vec<String>,
    /// Maximum encode/decode width in pixels.
    pub max_width: u32,
    /// Maximum encode/decode height in pixels.
    pub max_height: u32,
    /// Whether HDR (10-bit, PQ/HLG) is supported.
    pub supports_hdr: bool,
}

impl HwAccelDevice {
    /// Returns `true` if this device advertises support for `codec`.
    ///
    /// Comparison is case-insensitive.
    #[must_use]
    pub fn supports_codec(&self, codec: &str) -> bool {
        let lower = codec.to_lowercase();
        self.supported_codecs
            .iter()
            .any(|c| c.to_lowercase() == lower)
    }
}

// ─── HwAccelCapabilities ─────────────────────────────────────────────────────

/// Set of hardware-accelerated devices found on the current system.
///
/// Obtained via [`detect_hw_accel_caps`](crate::hw_accel::detect_hw_accel_caps)
/// (cached for the process lifetime) or
/// [`detect_hw_accel_with_probe`](crate::hw_accel::detect_hw_accel_with_probe)
/// (for tests / dependency injection).
#[derive(Debug, Clone, Default)]
pub struct HwAccelCapabilities {
    /// Discovered devices, in probe order.
    pub devices: Vec<HwAccelDevice>,
}

impl HwAccelCapabilities {
    /// Empty capability set — used when no hardware is found or detection is
    /// unsupported on this platform.
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    /// Returns `true` if no hardware devices were found.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.devices.is_empty()
    }

    /// Returns the first device that supports `codec`, if any.
    #[must_use]
    pub fn device_for_codec(&self, codec: &str) -> Option<&HwAccelDevice> {
        self.devices.iter().find(|d| d.supports_codec(codec))
    }
}
