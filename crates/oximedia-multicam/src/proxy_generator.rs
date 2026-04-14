//! Multi-angle proxy generation for lightweight preview of all angles
//! simultaneously.
//!
//! This module provides a higher-level API over the existing [`crate::proxy_gen`]
//! module.  Where `proxy_gen` operates on individual decoded-pixel buffers,
//! `proxy_generator` provides:
//!
//! - [`AngleProxy`] — a complete proxy bundle for one angle (including
//!   per-angle RGBA stub data).
//! - [`ProxyConfig`] — target dimensions, fps, and quality byte.
//! - [`MultiAngleProxyGenerator`] — creates stub proxy data for all angles
//!   without requiring a live decoded source frame.
//!
//! The stub approach is intentional: in a lightweight preview context you
//! often only need to confirm that all angles are present and correctly indexed
//! before any source decoding has occurred.  Each angle is assigned a distinct
//! solid colour so the multi-view layout is immediately recognisable.

// ── AngleProxy ────────────────────────────────────────────────────────────────

/// A proxy bundle for a single camera angle.
///
/// `data` contains raw RGBA pixel data (`width × height × 4` bytes).
#[derive(Debug, Clone)]
pub struct AngleProxy {
    /// Camera angle identifier (0-based).
    pub angle_id: usize,
    /// Proxy frame width in pixels.
    pub width: u32,
    /// Proxy frame height in pixels.
    pub height: u32,
    /// Output frame rate of the proxy.
    pub fps: f32,
    /// Raw RGBA pixel data (`width × height × 4` bytes).
    pub data: Vec<u8>,
}

impl AngleProxy {
    /// Return the expected byte length of `data` for a valid RGBA proxy.
    #[must_use]
    pub fn expected_data_len(&self) -> usize {
        (self.width * self.height) as usize * 4
    }

    /// `true` when `data.len() == expected_data_len()`.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.data.len() == self.expected_data_len()
    }
}

// ── ProxyConfig ───────────────────────────────────────────────────────────────

/// Configuration for proxy generation.
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Target proxy width in pixels.
    pub target_width: u32,
    /// Target proxy height in pixels.
    pub target_height: u32,
    /// Target proxy frame rate.
    pub fps: f32,
    /// Quality hint \[0, 100\].  Higher values imply better quality
    /// (actual encoding fidelity depends on the downstream codec).
    pub quality: u8,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            target_width: 960,
            target_height: 540,
            fps: 25.0,
            quality: 70,
        }
    }
}

// ── MultiAngleProxyGenerator ──────────────────────────────────────────────────

/// Generator that creates placeholder proxy data for all camera angles.
///
/// Each angle receives a solid-colour RGBA frame so that multi-view preview
/// panels can be populated before any source media is decoded.  The colours
/// cycle through a built-in palette of 16 distinct hues.
#[derive(Debug, Clone)]
pub struct MultiAngleProxyGenerator {
    config: ProxyConfig,
}

impl MultiAngleProxyGenerator {
    /// Create a new `MultiAngleProxyGenerator` with the given configuration.
    #[must_use]
    pub fn new(config: ProxyConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration (960×540, 25 fps, quality 70).
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(ProxyConfig::default())
    }

    /// Return the proxy width from the configuration.
    #[must_use]
    pub fn target_width(&self) -> u32 {
        self.config.target_width
    }

    /// Return the proxy height from the configuration.
    #[must_use]
    pub fn target_height(&self) -> u32 {
        self.config.target_height
    }

    /// Generate a single placeholder proxy for one angle.
    ///
    /// The proxy contains `frame_count` worth of RGBA data where each frame
    /// is a solid colour chosen from the built-in palette based on `angle_id`.
    /// The actual `data` field is sized to hold **one representative frame**
    /// (`width × height × 4` bytes); `frame_count` is stored only for
    /// informational purposes in the metadata.
    ///
    /// In production you would replace the stub `data` with actual downscaled
    /// frame data.
    #[must_use]
    pub fn generate_stub(&self, angle_id: usize, frame_count: u32) -> AngleProxy {
        let w = self.config.target_width;
        let h = self.config.target_height;
        let [r, g, b] = Self::angle_colour(angle_id);
        // One solid-colour RGBA frame as a representative stub.
        // frame_count can be used by callers to pre-allocate multi-frame
        // buffers; here we store a single frame so tests and previews can
        // verify the stub without allocating gigabytes.
        let _ = frame_count; // acknowledged; stored as metadata externally
        let pixel_count = (w * h) as usize;
        let mut data = Vec::with_capacity(pixel_count * 4);
        for _ in 0..pixel_count {
            data.push(r);
            data.push(g);
            data.push(b);
            data.push(255); // alpha = fully opaque
        }
        AngleProxy {
            angle_id,
            width: w,
            height: h,
            fps: self.config.fps,
            data,
        }
    }

    /// Generate one placeholder proxy per angle.
    ///
    /// Returns a `Vec<AngleProxy>` with exactly `angle_count` entries, one per
    /// angle in order `0..angle_count`.
    #[must_use]
    pub fn generate_all_angles(&self, angle_count: usize, frame_count: u32) -> Vec<AngleProxy> {
        (0..angle_count)
            .map(|id| self.generate_stub(id, frame_count))
            .collect()
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Return a visually distinct RGB colour for an angle index.
    ///
    /// Cycles through 16 pre-defined hues so up to 16 angles receive a unique
    /// colour; beyond that colours repeat with a different shade.
    fn angle_colour(angle_id: usize) -> [u8; 3] {
        // 16-colour palette derived from the HSV colour wheel at S=0.8, V=0.8.
        const PALETTE: [[u8; 3]; 16] = [
            [204,  51,  51], // 0  red
            [204, 128,  51], // 1  orange
            [204, 204,  51], // 2  yellow
            [128, 204,  51], // 3  yellow-green
            [ 51, 204,  51], // 4  green
            [ 51, 204, 128], // 5  spring green
            [ 51, 204, 204], // 6  cyan
            [ 51, 128, 204], // 7  azure
            [ 51,  51, 204], // 8  blue
            [128,  51, 204], // 9  violet
            [204,  51, 204], // 10 magenta
            [204,  51, 128], // 11 rose
            [153, 153, 153], // 12 light grey
            [102, 102, 102], // 13 mid grey
            [ 51,  51,  51], // 14 dark grey
            [230, 179, 102], // 15 tan
        ];
        PALETTE[angle_id % PALETTE.len()]
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_stub_valid_data_length() {
        let cfg = ProxyConfig {
            target_width: 16,
            target_height: 9,
            fps: 25.0,
            quality: 70,
        };
        let gen = MultiAngleProxyGenerator::new(cfg);
        let proxy = gen.generate_stub(0, 10);
        assert!(proxy.is_valid(), "Stub data length should match declared dimensions");
        assert_eq!(proxy.data.len(), 16 * 9 * 4);
    }

    #[test]
    fn test_generate_stub_angle_id_preserved() {
        let gen = MultiAngleProxyGenerator::with_defaults();
        let proxy = gen.generate_stub(3, 1);
        assert_eq!(proxy.angle_id, 3);
    }

    #[test]
    fn test_generate_stub_fps_from_config() {
        let cfg = ProxyConfig {
            target_width: 8,
            target_height: 8,
            fps: 59.94,
            quality: 90,
        };
        let gen = MultiAngleProxyGenerator::new(cfg);
        let proxy = gen.generate_stub(0, 1);
        assert!((proxy.fps - 59.94_f32).abs() < 0.01);
    }

    #[test]
    fn test_generate_stub_alpha_opaque() {
        let cfg = ProxyConfig {
            target_width: 4,
            target_height: 4,
            fps: 25.0,
            quality: 70,
        };
        let gen = MultiAngleProxyGenerator::new(cfg);
        let proxy = gen.generate_stub(0, 1);
        // Every 4th byte is the alpha channel and should be 255.
        for chunk in proxy.data.chunks_exact(4) {
            assert_eq!(chunk[3], 255, "Alpha channel should be fully opaque");
        }
    }

    #[test]
    fn test_generate_all_angles_count() {
        let gen = MultiAngleProxyGenerator::with_defaults();
        let proxies = gen.generate_all_angles(5, 30);
        assert_eq!(proxies.len(), 5);
    }

    #[test]
    fn test_generate_all_angles_ids_sequential() {
        let gen = MultiAngleProxyGenerator::with_defaults();
        let proxies = gen.generate_all_angles(4, 1);
        for (i, p) in proxies.iter().enumerate() {
            assert_eq!(p.angle_id, i);
        }
    }

    #[test]
    fn test_generate_all_angles_each_valid() {
        let cfg = ProxyConfig {
            target_width: 8,
            target_height: 6,
            fps: 24.0,
            quality: 50,
        };
        let gen = MultiAngleProxyGenerator::new(cfg);
        for proxy in gen.generate_all_angles(3, 5) {
            assert!(proxy.is_valid(), "Proxy {} should be valid", proxy.angle_id);
        }
    }

    #[test]
    fn test_angles_have_distinct_colours() {
        let cfg = ProxyConfig {
            target_width: 2,
            target_height: 2,
            fps: 25.0,
            quality: 70,
        };
        let gen = MultiAngleProxyGenerator::new(cfg);
        let p0 = gen.generate_stub(0, 1);
        let p1 = gen.generate_stub(1, 1);
        // Different angles should produce different RGB triples.
        // Compare the full first RGBA pixel.
        let px0 = &p0.data[0..3];
        let px1 = &p1.data[0..3];
        assert_ne!(
            px0, px1,
            "Angles 0 and 1 should have different RGB colours: {:?} vs {:?}",
            px0, px1
        );
    }

    #[test]
    fn test_proxy_config_default() {
        let cfg = ProxyConfig::default();
        assert_eq!(cfg.target_width, 960);
        assert_eq!(cfg.target_height, 540);
        assert!((cfg.fps - 25.0).abs() < 0.01);
        assert_eq!(cfg.quality, 70);
    }

    #[test]
    fn test_angle_proxy_expected_len() {
        let proxy = AngleProxy {
            angle_id: 0,
            width: 10,
            height: 10,
            fps: 25.0,
            data: vec![0u8; 400], // 10*10*4
        };
        assert_eq!(proxy.expected_data_len(), 400);
        assert!(proxy.is_valid());
    }

    #[test]
    fn test_angle_proxy_invalid_when_wrong_length() {
        let proxy = AngleProxy {
            angle_id: 0,
            width: 10,
            height: 10,
            fps: 25.0,
            data: vec![0u8; 100], // too short
        };
        assert!(!proxy.is_valid());
    }
}
