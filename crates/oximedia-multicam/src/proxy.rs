//! Multi-angle proxy generation for lightweight preview.
//!
//! Creates lower-resolution proxy specifications for each camera angle,
//! enabling simultaneous preview of all angles without decoding
//! full-resolution media. Also maps proxy frames back to original
//! source coordinates for final conform.

use std::collections::HashMap;
use std::fmt;

use crate::{AngleId, MultiCamError, Result};

/// Codec hint for the proxy file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProxyCodec {
    /// VP9 proxy (patent-free, good compression).
    Vp9,
    /// AV1 proxy (patent-free, best compression).
    Av1,
    /// MJPEG proxy (simple, fast scrubbing).
    Mjpeg,
    /// Raw / uncompressed proxy (fastest decode).
    Raw,
}

impl fmt::Display for ProxyCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vp9 => write!(f, "VP9"),
            Self::Av1 => write!(f, "AV1"),
            Self::Mjpeg => write!(f, "MJPEG"),
            Self::Raw => write!(f, "RAW"),
        }
    }
}

/// Quality preset for proxy generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProxyQuality {
    /// Low quality, smallest files.
    Draft,
    /// Medium quality, balanced.
    Preview,
    /// High quality, larger files.
    HighQuality,
}

impl ProxyQuality {
    /// Suggested CRF/QP value for this quality level (lower = better).
    #[must_use]
    pub fn suggested_crf(&self) -> u32 {
        match self {
            Self::Draft => 42,
            Self::Preview => 32,
            Self::HighQuality => 22,
        }
    }
}

impl fmt::Display for ProxyQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Draft => write!(f, "Draft"),
            Self::Preview => write!(f, "Preview"),
            Self::HighQuality => write!(f, "HighQuality"),
        }
    }
}

/// Configuration for proxy generation.
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Scale factor (0.0, 1.0]. 0.25 means 1/4 resolution.
    pub resolution_scale: f64,
    /// Codec to use for the proxy.
    pub codec: ProxyCodec,
    /// Quality preset.
    pub quality: ProxyQuality,
    /// Frame rate reduction factor. 1 = same as original, 2 = half, etc.
    pub frame_rate_divisor: u32,
    /// Whether to include audio in the proxy.
    pub include_audio: bool,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            resolution_scale: 0.25,
            codec: ProxyCodec::Vp9,
            quality: ProxyQuality::Preview,
            frame_rate_divisor: 1,
            include_audio: true,
        }
    }
}

impl ProxyConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.resolution_scale <= 0.0 || self.resolution_scale > 1.0 {
            return Err(MultiCamError::ConfigError(
                "resolution_scale must be in (0.0, 1.0]".into(),
            ));
        }
        if self.frame_rate_divisor == 0 {
            return Err(MultiCamError::ConfigError(
                "frame_rate_divisor must be >= 1".into(),
            ));
        }
        Ok(())
    }

    /// Compute the proxy dimensions from original dimensions.
    #[must_use]
    pub fn proxy_dimensions(&self, original_width: u32, original_height: u32) -> (u32, u32) {
        let w = ((f64::from(original_width) * self.resolution_scale).round() as u32).max(2);
        let h = ((f64::from(original_height) * self.resolution_scale).round() as u32).max(2);
        // Ensure even dimensions for codec compatibility.
        let w = if w % 2 != 0 { w + 1 } else { w };
        let h = if h % 2 != 0 { h + 1 } else { h };
        (w, h)
    }
}

/// Specification for a single proxy stream (one camera angle).
#[derive(Debug, Clone)]
pub struct ProxySpec {
    /// Angle identifier.
    pub angle_id: AngleId,
    /// Original resolution.
    pub original_width: u32,
    /// Original height.
    pub original_height: u32,
    /// Proxy resolution width.
    pub proxy_width: u32,
    /// Proxy resolution height.
    pub proxy_height: u32,
    /// Codec for the proxy.
    pub codec: ProxyCodec,
    /// Suggested CRF/QP.
    pub crf: u32,
    /// Frame rate divisor.
    pub frame_rate_divisor: u32,
    /// Whether audio is included.
    pub include_audio: bool,
}

/// Generates proxy specifications for camera angles.
#[derive(Debug)]
pub struct ProxyGenerator {
    config: ProxyConfig,
    /// Registered angles: angle_id -> (width, height, label).
    angles: HashMap<AngleId, (u32, u32, String)>,
}

impl ProxyGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: ProxyConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            angles: HashMap::new(),
        })
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Result<Self> {
        Self::new(ProxyConfig::default())
    }

    /// Register a camera angle with its original resolution.
    pub fn add_angle(
        &mut self,
        angle_id: AngleId,
        width: u32,
        height: u32,
        label: &str,
    ) -> Result<()> {
        if width == 0 || height == 0 {
            return Err(MultiCamError::ConfigError(
                "width and height must be > 0".into(),
            ));
        }
        self.angles
            .insert(angle_id, (width, height, label.to_owned()));
        Ok(())
    }

    /// Remove an angle.
    pub fn remove_angle(&mut self, angle_id: AngleId) -> bool {
        self.angles.remove(&angle_id).is_some()
    }

    /// Number of registered angles.
    #[must_use]
    pub fn angle_count(&self) -> usize {
        self.angles.len()
    }

    /// Generate a proxy specification for a single angle.
    pub fn generate_spec(&self, angle_id: AngleId) -> Result<ProxySpec> {
        let (w, h, _label) = self
            .angles
            .get(&angle_id)
            .ok_or(MultiCamError::AngleNotFound(angle_id))?;

        let (pw, ph) = self.config.proxy_dimensions(*w, *h);
        Ok(ProxySpec {
            angle_id,
            original_width: *w,
            original_height: *h,
            proxy_width: pw,
            proxy_height: ph,
            codec: self.config.codec,
            crf: self.config.quality.suggested_crf(),
            frame_rate_divisor: self.config.frame_rate_divisor,
            include_audio: self.config.include_audio,
        })
    }

    /// Generate proxy specifications for all registered angles.
    pub fn generate_all_specs(&self) -> Result<Vec<ProxySpec>> {
        let mut specs = Vec::with_capacity(self.angles.len());
        let mut ids: Vec<_> = self.angles.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            specs.push(self.generate_spec(id)?);
        }
        Ok(specs)
    }

    /// Estimated storage ratio compared to original (rough heuristic).
    #[must_use]
    pub fn estimated_storage_ratio(&self) -> f64 {
        let scale_sq = self.config.resolution_scale * self.config.resolution_scale;
        let quality_factor = match self.config.quality {
            ProxyQuality::Draft => 0.3,
            ProxyQuality::Preview => 0.5,
            ProxyQuality::HighQuality => 0.8,
        };
        let fps_factor = 1.0 / f64::from(self.config.frame_rate_divisor.max(1));
        scale_sq * quality_factor * fps_factor
    }
}

/// Maps proxy frame coordinates back to original source coordinates.
#[derive(Debug, Clone)]
pub struct ProxyMapping {
    /// Scale factor that was applied (original -> proxy).
    pub scale: f64,
    /// Original width.
    pub original_width: u32,
    /// Original height.
    pub original_height: u32,
    /// Proxy width.
    pub proxy_width: u32,
    /// Proxy height.
    pub proxy_height: u32,
    /// Frame rate divisor.
    pub frame_rate_divisor: u32,
}

impl ProxyMapping {
    /// Create a mapping from a proxy specification.
    #[must_use]
    pub fn from_spec(spec: &ProxySpec, scale: f64) -> Self {
        Self {
            scale,
            original_width: spec.original_width,
            original_height: spec.original_height,
            proxy_width: spec.proxy_width,
            proxy_height: spec.proxy_height,
            frame_rate_divisor: spec.frame_rate_divisor,
        }
    }

    /// Map a pixel coordinate in proxy space back to original space.
    #[must_use]
    pub fn proxy_to_original(&self, proxy_x: f64, proxy_y: f64) -> (f64, f64) {
        if self.scale <= 0.0 {
            return (0.0, 0.0);
        }
        let orig_x = proxy_x / self.scale;
        let orig_y = proxy_y / self.scale;
        (orig_x, orig_y)
    }

    /// Map a pixel coordinate in original space to proxy space.
    #[must_use]
    pub fn original_to_proxy(&self, orig_x: f64, orig_y: f64) -> (f64, f64) {
        (orig_x * self.scale, orig_y * self.scale)
    }

    /// Map a proxy frame number to the original frame number.
    #[must_use]
    pub fn proxy_frame_to_original(&self, proxy_frame: u64) -> u64 {
        proxy_frame * u64::from(self.frame_rate_divisor.max(1))
    }

    /// Map an original frame number to the nearest proxy frame number.
    #[must_use]
    pub fn original_frame_to_proxy(&self, original_frame: u64) -> u64 {
        let divisor = u64::from(self.frame_rate_divisor.max(1));
        original_frame / divisor
    }

    /// Map a crop rectangle from proxy to original space.
    #[must_use]
    pub fn proxy_rect_to_original(
        &self,
        x: f64,
        y: f64,
        w: f64,
        h: f64,
    ) -> (f64, f64, f64, f64) {
        let (ox, oy) = self.proxy_to_original(x, y);
        if self.scale <= 0.0 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let ow = w / self.scale;
        let oh = h / self.scale;
        (ox, oy, ow, oh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proxy_config_defaults() {
        let cfg = ProxyConfig::default();
        assert_eq!(cfg.resolution_scale, 0.25);
        assert_eq!(cfg.codec, ProxyCodec::Vp9);
        assert_eq!(cfg.quality, ProxyQuality::Preview);
        assert!(cfg.include_audio);
    }

    #[test]
    fn test_proxy_config_validate_ok() {
        assert!(ProxyConfig::default().validate().is_ok());
    }

    #[test]
    fn test_proxy_config_validate_bad_scale() {
        let mut cfg = ProxyConfig::default();
        cfg.resolution_scale = 0.0;
        assert!(cfg.validate().is_err());
        cfg.resolution_scale = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_proxy_config_validate_bad_fps_divisor() {
        let mut cfg = ProxyConfig::default();
        cfg.frame_rate_divisor = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_proxy_dimensions_quarter() {
        let cfg = ProxyConfig {
            resolution_scale: 0.25,
            ..ProxyConfig::default()
        };
        let (w, h) = cfg.proxy_dimensions(1920, 1080);
        assert_eq!(w, 480);
        assert_eq!(h, 270);
    }

    #[test]
    fn test_proxy_dimensions_even() {
        let cfg = ProxyConfig {
            resolution_scale: 0.5,
            ..ProxyConfig::default()
        };
        // 1921 * 0.5 = 960.5 -> 961 (odd) -> 962
        let (w, h) = cfg.proxy_dimensions(1921, 1081);
        assert_eq!(w % 2, 0);
        assert_eq!(h % 2, 0);
    }

    #[test]
    fn test_proxy_dimensions_minimum() {
        let cfg = ProxyConfig {
            resolution_scale: 0.01,
            ..ProxyConfig::default()
        };
        let (w, h) = cfg.proxy_dimensions(100, 100);
        assert!(w >= 2);
        assert!(h >= 2);
    }

    #[test]
    fn test_generator_add_angle() {
        let mut gen = ProxyGenerator::with_defaults().expect("gen");
        gen.add_angle(0, 1920, 1080, "Camera A").expect("add");
        assert_eq!(gen.angle_count(), 1);
    }

    #[test]
    fn test_generator_add_zero_dimensions() {
        let mut gen = ProxyGenerator::with_defaults().expect("gen");
        assert!(gen.add_angle(0, 0, 1080, "Bad").is_err());
    }

    #[test]
    fn test_generate_spec() {
        let mut gen = ProxyGenerator::with_defaults().expect("gen");
        gen.add_angle(0, 3840, 2160, "Cam1").expect("add");
        let spec = gen.generate_spec(0).expect("spec");
        assert_eq!(spec.original_width, 3840);
        assert_eq!(spec.proxy_width, 960);
        assert_eq!(spec.proxy_height, 540);
        assert_eq!(spec.codec, ProxyCodec::Vp9);
    }

    #[test]
    fn test_generate_spec_not_found() {
        let gen = ProxyGenerator::with_defaults().expect("gen");
        assert!(gen.generate_spec(99).is_err());
    }

    #[test]
    fn test_generate_all_specs() {
        let mut gen = ProxyGenerator::with_defaults().expect("gen");
        gen.add_angle(0, 1920, 1080, "A").expect("add");
        gen.add_angle(1, 3840, 2160, "B").expect("add");
        let specs = gen.generate_all_specs().expect("all");
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].angle_id, 0);
        assert_eq!(specs[1].angle_id, 1);
    }

    #[test]
    fn test_estimated_storage_ratio() {
        let gen = ProxyGenerator::with_defaults().expect("gen");
        let ratio = gen.estimated_storage_ratio();
        // 0.25^2 * 0.5 * 1.0 = 0.03125
        assert!((ratio - 0.03125).abs() < 0.001);
    }

    #[test]
    fn test_proxy_mapping_coordinate() {
        let spec = ProxySpec {
            angle_id: 0,
            original_width: 1920,
            original_height: 1080,
            proxy_width: 480,
            proxy_height: 270,
            codec: ProxyCodec::Vp9,
            crf: 32,
            frame_rate_divisor: 1,
            include_audio: true,
        };
        let mapping = ProxyMapping::from_spec(&spec, 0.25);
        let (ox, oy) = mapping.proxy_to_original(100.0, 50.0);
        assert!((ox - 400.0).abs() < 0.01);
        assert!((oy - 200.0).abs() < 0.01);
    }

    #[test]
    fn test_proxy_mapping_round_trip() {
        let spec = ProxySpec {
            angle_id: 0,
            original_width: 1920,
            original_height: 1080,
            proxy_width: 960,
            proxy_height: 540,
            codec: ProxyCodec::Av1,
            crf: 22,
            frame_rate_divisor: 2,
            include_audio: false,
        };
        let mapping = ProxyMapping::from_spec(&spec, 0.5);
        let (px, py) = mapping.original_to_proxy(800.0, 400.0);
        let (ox, oy) = mapping.proxy_to_original(px, py);
        assert!((ox - 800.0).abs() < 0.01);
        assert!((oy - 400.0).abs() < 0.01);
    }

    #[test]
    fn test_proxy_mapping_frame_numbers() {
        let spec = ProxySpec {
            angle_id: 0,
            original_width: 1920,
            original_height: 1080,
            proxy_width: 480,
            proxy_height: 270,
            codec: ProxyCodec::Vp9,
            crf: 32,
            frame_rate_divisor: 3,
            include_audio: true,
        };
        let mapping = ProxyMapping::from_spec(&spec, 0.25);
        assert_eq!(mapping.proxy_frame_to_original(10), 30);
        assert_eq!(mapping.original_frame_to_proxy(30), 10);
        assert_eq!(mapping.original_frame_to_proxy(31), 10); // floor
    }

    #[test]
    fn test_proxy_mapping_rect() {
        let spec = ProxySpec {
            angle_id: 0,
            original_width: 1920,
            original_height: 1080,
            proxy_width: 960,
            proxy_height: 540,
            codec: ProxyCodec::Mjpeg,
            crf: 42,
            frame_rate_divisor: 1,
            include_audio: false,
        };
        let mapping = ProxyMapping::from_spec(&spec, 0.5);
        let (ox, oy, ow, oh) = mapping.proxy_rect_to_original(10.0, 20.0, 100.0, 50.0);
        assert!((ox - 20.0).abs() < 0.01);
        assert!((oy - 40.0).abs() < 0.01);
        assert!((ow - 200.0).abs() < 0.01);
        assert!((oh - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_proxy_codec_display() {
        assert_eq!(format!("{}", ProxyCodec::Av1), "AV1");
        assert_eq!(format!("{}", ProxyCodec::Raw), "RAW");
    }

    #[test]
    fn test_proxy_quality_crf() {
        assert_eq!(ProxyQuality::Draft.suggested_crf(), 42);
        assert_eq!(ProxyQuality::Preview.suggested_crf(), 32);
        assert_eq!(ProxyQuality::HighQuality.suggested_crf(), 22);
    }

    #[test]
    fn test_remove_angle() {
        let mut gen = ProxyGenerator::with_defaults().expect("gen");
        gen.add_angle(0, 1920, 1080, "A").expect("add");
        assert!(gen.remove_angle(0));
        assert!(!gen.remove_angle(0));
        assert_eq!(gen.angle_count(), 0);
    }
}
