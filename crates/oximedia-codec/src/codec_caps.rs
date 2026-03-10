//! Codec capability discovery and hardware acceleration detection.
//!
//! Provides a stub-based capability model for querying which hardware
//! accelerators are available on the current platform, and a registry
//! for looking up codec capabilities by codec identifier.

#![allow(dead_code)]

/// Hardware acceleration backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HwAccelType {
    /// Software (no hardware acceleration).
    None,
    /// NVIDIA CUDA / NVENC / NVDEC.
    Cuda,
    /// Intel Quick Sync Video.
    Qsv,
    /// VA-API (Linux open-source GPU acceleration).
    Vaapi,
    /// Apple VideoToolbox.
    VideoToolbox,
    /// AMD AMF (Advanced Media Framework).
    Amf,
}

impl HwAccelType {
    /// Returns a stub availability check.
    ///
    /// In production this would probe the system; here it always returns `false`
    /// for `None` and `true` for every named accelerator (for testability).
    pub fn is_available_stub(self) -> bool {
        !matches!(self, HwAccelType::None)
    }

    /// Human-readable name of the accelerator.
    pub fn name(self) -> &'static str {
        match self {
            HwAccelType::None => "Software",
            HwAccelType::Cuda => "CUDA",
            HwAccelType::Qsv => "Intel QSV",
            HwAccelType::Vaapi => "VA-API",
            HwAccelType::VideoToolbox => "VideoToolbox",
            HwAccelType::Amf => "AMD AMF",
        }
    }
}

/// Capabilities record for a single codec.
#[derive(Debug, Clone)]
pub struct CodecCaps {
    /// Codec identifier string (e.g. `"h264"`, `"av1"`).
    pub codec_id: String,
    /// Maximum supported width in pixels.
    pub max_width: u32,
    /// Maximum supported height in pixels.
    pub max_height: u32,
    /// Supported hardware accelerators.
    pub hw_accels: Vec<HwAccelType>,
    /// Whether the codec supports B-frames.
    pub b_frames: bool,
    /// Whether the codec supports lossless mode.
    pub lossless: bool,
}

impl CodecCaps {
    /// Create a new capability record.
    pub fn new(codec_id: impl Into<String>) -> Self {
        Self {
            codec_id: codec_id.into(),
            max_width: 7680,
            max_height: 4320,
            hw_accels: Vec::new(),
            b_frames: false,
            lossless: false,
        }
    }

    /// Builder: set maximum resolution.
    pub fn with_max_resolution(mut self, w: u32, h: u32) -> Self {
        self.max_width = w;
        self.max_height = h;
        self
    }

    /// Builder: add a hardware accelerator.
    pub fn with_hw_accel(mut self, accel: HwAccelType) -> Self {
        self.hw_accels.push(accel);
        self
    }

    /// Builder: enable B-frame support.
    pub fn with_b_frames(mut self) -> Self {
        self.b_frames = true;
        self
    }

    /// Builder: enable lossless mode.
    pub fn with_lossless(mut self) -> Self {
        self.lossless = true;
        self
    }

    /// Returns `true` if this codec supports the given hardware accelerator.
    pub fn supports_hw_accel(&self, accel: HwAccelType) -> bool {
        self.hw_accels.contains(&accel)
    }

    /// Returns `true` if the codec can encode frames of the given dimensions.
    pub fn supports_resolution(&self, width: u32, height: u32) -> bool {
        width <= self.max_width && height <= self.max_height
    }
}

/// A registry of [`CodecCaps`] keyed by codec identifier string.
#[derive(Debug, Default)]
pub struct CodecCapsRegistry {
    entries: Vec<CodecCaps>,
}

impl CodecCapsRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register a codec's capabilities.
    ///
    /// If an entry with the same `codec_id` already exists it is replaced.
    pub fn register(&mut self, caps: CodecCaps) {
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|c| c.codec_id == caps.codec_id)
        {
            *existing = caps;
        } else {
            self.entries.push(caps);
        }
    }

    /// Find the capabilities for a codec by its identifier.
    pub fn find(&self, codec_id: &str) -> Option<&CodecCaps> {
        self.entries.iter().find(|c| c.codec_id == codec_id)
    }

    /// Returns the number of registered codecs.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the registry contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns all registered codec IDs.
    pub fn codec_ids(&self) -> Vec<&str> {
        self.entries.iter().map(|c| c.codec_id.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hw_accel_none_not_available() {
        assert!(!HwAccelType::None.is_available_stub());
    }

    #[test]
    fn test_hw_accel_cuda_available() {
        assert!(HwAccelType::Cuda.is_available_stub());
    }

    #[test]
    fn test_hw_accel_vaapi_available() {
        assert!(HwAccelType::Vaapi.is_available_stub());
    }

    #[test]
    fn test_hw_accel_name() {
        assert_eq!(HwAccelType::Cuda.name(), "CUDA");
        assert_eq!(HwAccelType::Qsv.name(), "Intel QSV");
        assert_eq!(HwAccelType::VideoToolbox.name(), "VideoToolbox");
    }

    #[test]
    fn test_codec_caps_default_max_resolution() {
        let caps = CodecCaps::new("av1");
        assert_eq!(caps.max_width, 7680);
        assert_eq!(caps.max_height, 4320);
    }

    #[test]
    fn test_codec_caps_supports_hw_accel_true() {
        let caps = CodecCaps::new("h264").with_hw_accel(HwAccelType::Cuda);
        assert!(caps.supports_hw_accel(HwAccelType::Cuda));
    }

    #[test]
    fn test_codec_caps_supports_hw_accel_false() {
        let caps = CodecCaps::new("av1");
        assert!(!caps.supports_hw_accel(HwAccelType::Cuda));
    }

    #[test]
    fn test_codec_caps_supports_resolution_within() {
        let caps = CodecCaps::new("vp9").with_max_resolution(3840, 2160);
        assert!(caps.supports_resolution(1920, 1080));
    }

    #[test]
    fn test_codec_caps_supports_resolution_too_large() {
        let caps = CodecCaps::new("vp9").with_max_resolution(3840, 2160);
        assert!(!caps.supports_resolution(7680, 4320));
    }

    #[test]
    fn test_codec_caps_lossless_flag() {
        let caps = CodecCaps::new("av1").with_lossless();
        assert!(caps.lossless);
    }

    #[test]
    fn test_registry_register_and_find() {
        let mut reg = CodecCapsRegistry::new();
        reg.register(CodecCaps::new("av1"));
        assert!(reg.find("av1").is_some());
    }

    #[test]
    fn test_registry_find_missing() {
        let reg = CodecCapsRegistry::new();
        assert!(reg.find("h264").is_none());
    }

    #[test]
    fn test_registry_replaces_existing() {
        let mut reg = CodecCapsRegistry::new();
        reg.register(CodecCaps::new("av1").with_max_resolution(1920, 1080));
        reg.register(CodecCaps::new("av1").with_max_resolution(3840, 2160));
        let caps = reg.find("av1").expect("should succeed");
        assert_eq!(caps.max_width, 3840);
    }

    #[test]
    fn test_registry_len_and_is_empty() {
        let mut reg = CodecCapsRegistry::new();
        assert!(reg.is_empty());
        reg.register(CodecCaps::new("av1"));
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
    }

    #[test]
    fn test_registry_codec_ids() {
        let mut reg = CodecCapsRegistry::new();
        reg.register(CodecCaps::new("av1"));
        reg.register(CodecCaps::new("vp9"));
        let ids = reg.codec_ids();
        assert!(ids.contains(&"av1"));
        assert!(ids.contains(&"vp9"));
    }
}
