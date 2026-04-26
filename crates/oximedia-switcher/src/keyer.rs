//! Upstream and downstream keyer implementation for video switchers.
//!
//! Keyers allow compositing multiple video layers with transparency.

use crate::chroma::{ChromaKey, ChromaKeyParams};
use crate::composite::{apply_clip_gain_invert, attach_alpha_plane, composite_over};
use crate::luma::{LumaKey, LumaKeyParams};
use oximedia_codec::VideoFrame;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur with keyer operations.
#[derive(Error, Debug, Clone)]
pub enum KeyerError {
    #[error("Invalid keyer ID: {0}")]
    InvalidKeyerId(usize),

    #[error("Keyer {0} not found")]
    KeyerNotFound(usize),

    #[error("Invalid fill source: {0}")]
    InvalidFillSource(usize),

    #[error("Invalid key source: {0}")]
    InvalidKeySource(usize),

    #[error("Processing error: {0}")]
    ProcessingError(String),
}

/// Type of keyer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum KeyerType {
    /// Luma key (brightness-based)
    Luma,
    /// Chroma key (color-based)
    Chroma,
    /// Linear key (uses external matte)
    Linear,
    /// Pattern key (uses pattern generator)
    Pattern,
}

/// Key source configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeySource {
    /// Fill source (foreground video)
    pub fill: usize,
    /// Key source (matte/alpha channel)
    pub key: Option<usize>,
}

impl KeySource {
    /// Create a new key source.
    pub fn new(fill: usize) -> Self {
        Self { fill, key: None }
    }

    /// Create with both fill and key.
    pub fn with_key(fill: usize, key: usize) -> Self {
        Self {
            fill,
            key: Some(key),
        }
    }

    /// Set the fill source.
    pub fn set_fill(&mut self, fill: usize) {
        self.fill = fill;
    }

    /// Set the key source.
    pub fn set_key(&mut self, key: Option<usize>) {
        self.key = key;
    }
}

/// Keyer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyerConfig {
    /// Keyer ID
    pub id: usize,
    /// Keyer type
    pub keyer_type: KeyerType,
    /// Source configuration
    pub source: KeySource,
    /// Enabled state
    pub enabled: bool,
    /// On-air state
    pub on_air: bool,
    /// Tie (enables fill and key together)
    pub tie: bool,
    /// Pre-multiplied alpha
    pub pre_multiplied: bool,
}

impl KeyerConfig {
    /// Create a new keyer configuration.
    pub fn new(id: usize, keyer_type: KeyerType, fill: usize) -> Self {
        Self {
            id,
            keyer_type,
            source: KeySource::new(fill),
            enabled: true,
            on_air: false,
            tie: true,
            pre_multiplied: false,
        }
    }
}

/// Upstream keyer (USK) - part of the M/E row, affected by transitions.
pub struct UpstreamKeyer {
    config: KeyerConfig,
    luma_key: LumaKey,
    chroma_key: ChromaKey,
}

impl UpstreamKeyer {
    /// Create a new upstream keyer.
    pub fn new(id: usize, keyer_type: KeyerType, fill: usize) -> Self {
        Self {
            config: KeyerConfig::new(id, keyer_type, fill),
            luma_key: LumaKey::new(),
            chroma_key: ChromaKey::new_green(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &KeyerConfig {
        &self.config
    }

    /// Get mutable configuration.
    pub fn config_mut(&mut self) -> &mut KeyerConfig {
        &mut self.config
    }

    /// Set the keyer type.
    pub fn set_type(&mut self, keyer_type: KeyerType) {
        self.config.keyer_type = keyer_type;
    }

    /// Get the keyer type.
    pub fn keyer_type(&self) -> KeyerType {
        self.config.keyer_type
    }

    /// Enable or disable the keyer.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
    }

    /// Check if the keyer is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Set on-air state.
    pub fn set_on_air(&mut self, on_air: bool) {
        self.config.on_air = on_air;
    }

    /// Check if the keyer is on-air.
    pub fn is_on_air(&self) -> bool {
        self.config.on_air
    }

    /// Get the luma key processor.
    pub fn luma_key(&self) -> &LumaKey {
        &self.luma_key
    }

    /// Get mutable luma key processor.
    pub fn luma_key_mut(&mut self) -> &mut LumaKey {
        &mut self.luma_key
    }

    /// Get the chroma key processor.
    pub fn chroma_key(&self) -> &ChromaKey {
        &self.chroma_key
    }

    /// Get mutable chroma key processor.
    pub fn chroma_key_mut(&mut self) -> &mut ChromaKey {
        &mut self.chroma_key
    }

    /// Set luma key parameters.
    pub fn set_luma_params(&mut self, params: LumaKeyParams) {
        self.luma_key.set_params(params);
    }

    /// Set chroma key parameters.
    pub fn set_chroma_params(&mut self, params: ChromaKeyParams) {
        self.chroma_key.set_params(params);
    }

    /// Process video through the keyer.
    ///
    /// Returns a clone of `fill` with an alpha plane appended as the last plane.
    /// The alpha plane is generated according to the configured keyer type:
    ///
    /// * `Luma`    — luma-based matte extracted from `fill`.
    /// * `Chroma`  — chroma-based matte extracted from `fill`.
    /// * `Linear`  — takes the first plane of the supplied `key` frame as alpha.
    /// * `Pattern` — not yet supported; returns `ProcessingError`.
    pub fn process(
        &self,
        fill: &VideoFrame,
        key: Option<&VideoFrame>,
    ) -> Result<VideoFrame, KeyerError> {
        match self.config.keyer_type {
            KeyerType::Luma => {
                let alpha = self
                    .luma_key
                    .process_frame(fill)
                    .map_err(|e| KeyerError::ProcessingError(e.to_string()))?;
                Ok(attach_alpha_plane(fill, alpha))
            }
            KeyerType::Chroma => {
                let alpha = self
                    .chroma_key
                    .process_frame(fill)
                    .map_err(|e| KeyerError::ProcessingError(e.to_string()))?;
                Ok(attach_alpha_plane(fill, alpha))
            }
            KeyerType::Linear => {
                let key_frame = key.ok_or(KeyerError::InvalidKeySource(self.config.id))?;
                let alpha = key_frame
                    .planes
                    .first()
                    .map(|p| p.data.clone())
                    .ok_or_else(|| {
                        KeyerError::ProcessingError(
                            "key frame has no planes for linear keyer".to_string(),
                        )
                    })?;
                Ok(attach_alpha_plane(fill, alpha))
            }
            KeyerType::Pattern => Err(KeyerError::ProcessingError(
                "Pattern key requires a configured pattern generator".to_string(),
            )),
        }
    }
}

/// Downstream keyer (DSK) - applied after M/E processing, not affected by transitions.
pub struct DownstreamKeyer {
    config: KeyerConfig,
    #[allow(dead_code)]
    luma_key: LumaKey,
    /// Clip level (0.0 - 1.0)
    clip: f32,
    /// Gain (0.0 - 2.0)
    gain: f32,
    /// Invert key
    invert: bool,
}

impl DownstreamKeyer {
    /// Create a new downstream keyer.
    pub fn new(id: usize, fill: usize, key: usize) -> Self {
        let mut config = KeyerConfig::new(id, KeyerType::Linear, fill);
        config.source.set_key(Some(key));

        Self {
            config,
            luma_key: LumaKey::new(),
            clip: 0.5,
            gain: 1.0,
            invert: false,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &KeyerConfig {
        &self.config
    }

    /// Get mutable configuration.
    pub fn config_mut(&mut self) -> &mut KeyerConfig {
        &mut self.config
    }

    /// Set on-air state.
    pub fn set_on_air(&mut self, on_air: bool) {
        self.config.on_air = on_air;
    }

    /// Check if the keyer is on-air.
    pub fn is_on_air(&self) -> bool {
        self.config.on_air
    }

    /// Set tie state.
    pub fn set_tie(&mut self, tie: bool) {
        self.config.tie = tie;
    }

    /// Check if tie is enabled.
    pub fn is_tie(&self) -> bool {
        self.config.tie
    }

    /// Set clip level.
    pub fn set_clip(&mut self, clip: f32) {
        self.clip = clip.clamp(0.0, 1.0);
    }

    /// Get clip level.
    pub fn clip(&self) -> f32 {
        self.clip
    }

    /// Set gain.
    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain.clamp(0.0, 2.0);
    }

    /// Get gain.
    pub fn gain(&self) -> f32 {
        self.gain
    }

    /// Set invert state.
    pub fn set_invert(&mut self, invert: bool) {
        self.invert = invert;
    }

    /// Check if invert is enabled.
    pub fn is_invert(&self) -> bool {
        self.invert
    }

    /// Auto-transition DSK on or off.
    pub fn auto_transition(&mut self, duration_frames: u32) -> Result<(), KeyerError> {
        // In a real implementation, this would:
        // 1. Start an automatic transition
        // 2. Fade the DSK on or off over duration_frames
        // 3. Update on_air state when complete

        // For now, just toggle immediately
        self.config.on_air = !self.config.on_air;
        let _ = duration_frames; // Suppress unused warning
        Ok(())
    }

    /// Process video through the DSK.
    ///
    /// Takes the first plane of `key` as the raw alpha matte, applies
    /// `clip`/`gain`/`invert`, then composites `fill` over `program`.
    pub fn process(
        &self,
        program: &VideoFrame,
        fill: &VideoFrame,
        key: &VideoFrame,
    ) -> Result<VideoFrame, KeyerError> {
        let mut alpha = key.planes.first().map(|p| p.data.clone()).ok_or_else(|| {
            KeyerError::ProcessingError("key frame has no planes for DSK".to_string())
        })?;

        apply_clip_gain_invert(&mut alpha, self.clip, self.gain, self.invert);

        composite_over(fill, program, &alpha)
    }
}

/// Keyer manager for a switcher.
pub struct KeyerManager {
    upstream_keyers: Vec<UpstreamKeyer>,
    downstream_keyers: Vec<DownstreamKeyer>,
}

impl KeyerManager {
    /// Create a new keyer manager.
    pub fn new(num_upstream: usize, num_downstream: usize) -> Self {
        let upstream_keyers = (0..num_upstream)
            .map(|i| UpstreamKeyer::new(i, KeyerType::Luma, 0))
            .collect();

        let downstream_keyers = (0..num_downstream)
            .map(|i| DownstreamKeyer::new(i, 0, 0))
            .collect();

        Self {
            upstream_keyers,
            downstream_keyers,
        }
    }

    /// Get an upstream keyer.
    pub fn get_upstream(&self, id: usize) -> Result<&UpstreamKeyer, KeyerError> {
        self.upstream_keyers
            .get(id)
            .ok_or(KeyerError::KeyerNotFound(id))
    }

    /// Get a mutable upstream keyer.
    pub fn get_upstream_mut(&mut self, id: usize) -> Result<&mut UpstreamKeyer, KeyerError> {
        self.upstream_keyers
            .get_mut(id)
            .ok_or(KeyerError::KeyerNotFound(id))
    }

    /// Get a downstream keyer.
    pub fn get_downstream(&self, id: usize) -> Result<&DownstreamKeyer, KeyerError> {
        self.downstream_keyers
            .get(id)
            .ok_or(KeyerError::KeyerNotFound(id))
    }

    /// Get a mutable downstream keyer.
    pub fn get_downstream_mut(&mut self, id: usize) -> Result<&mut DownstreamKeyer, KeyerError> {
        self.downstream_keyers
            .get_mut(id)
            .ok_or(KeyerError::KeyerNotFound(id))
    }

    /// Get all upstream keyers.
    pub fn upstream_keyers(&self) -> &[UpstreamKeyer] {
        &self.upstream_keyers
    }

    /// Get all downstream keyers.
    pub fn downstream_keyers(&self) -> &[DownstreamKeyer] {
        &self.downstream_keyers
    }

    /// Get the number of upstream keyers.
    pub fn upstream_count(&self) -> usize {
        self.upstream_keyers.len()
    }

    /// Get the number of downstream keyers.
    pub fn downstream_count(&self) -> usize {
        self.downstream_keyers.len()
    }

    /// Get all on-air upstream keyer IDs.
    pub fn on_air_upstream(&self) -> Vec<usize> {
        self.upstream_keyers
            .iter()
            .filter(|k| k.is_on_air())
            .map(|k| k.config().id)
            .collect()
    }

    /// Get all on-air downstream keyer IDs.
    pub fn on_air_downstream(&self) -> Vec<usize> {
        self.downstream_keyers
            .iter()
            .filter(|k| k.is_on_air())
            .map(|k| k.config().id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_source() {
        let source = KeySource::new(1);
        assert_eq!(source.fill, 1);
        assert_eq!(source.key, None);

        let source_with_key = KeySource::with_key(1, 2);
        assert_eq!(source_with_key.fill, 1);
        assert_eq!(source_with_key.key, Some(2));
    }

    #[test]
    fn test_keyer_config() {
        let config = KeyerConfig::new(0, KeyerType::Luma, 1);
        assert_eq!(config.id, 0);
        assert_eq!(config.keyer_type, KeyerType::Luma);
        assert_eq!(config.source.fill, 1);
        assert!(config.enabled);
        assert!(!config.on_air);
        assert!(config.tie);
    }

    #[test]
    fn test_upstream_keyer_creation() {
        let usk = UpstreamKeyer::new(0, KeyerType::Chroma, 1);
        assert_eq!(usk.config().id, 0);
        assert_eq!(usk.keyer_type(), KeyerType::Chroma);
        assert!(usk.is_enabled());
        assert!(!usk.is_on_air());
    }

    #[test]
    fn test_upstream_keyer_on_air() {
        let mut usk = UpstreamKeyer::new(0, KeyerType::Luma, 1);
        assert!(!usk.is_on_air());

        usk.set_on_air(true);
        assert!(usk.is_on_air());

        usk.set_on_air(false);
        assert!(!usk.is_on_air());
    }

    #[test]
    fn test_downstream_keyer_creation() {
        let dsk = DownstreamKeyer::new(0, 1, 2);
        assert_eq!(dsk.config().id, 0);
        assert_eq!(dsk.config().source.fill, 1);
        assert_eq!(dsk.config().source.key, Some(2));
        assert!(!dsk.is_on_air());
    }

    #[test]
    fn test_downstream_keyer_clip_gain() {
        let mut dsk = DownstreamKeyer::new(0, 1, 2);

        assert_eq!(dsk.clip(), 0.5);
        assert_eq!(dsk.gain(), 1.0);

        dsk.set_clip(0.3);
        assert_eq!(dsk.clip(), 0.3);

        dsk.set_gain(1.5);
        assert_eq!(dsk.gain(), 1.5);

        // Test clamping
        dsk.set_clip(1.5);
        assert_eq!(dsk.clip(), 1.0);

        dsk.set_gain(3.0);
        assert_eq!(dsk.gain(), 2.0);
    }

    #[test]
    fn test_downstream_keyer_tie() {
        let mut dsk = DownstreamKeyer::new(0, 1, 2);
        assert!(dsk.is_tie());

        dsk.set_tie(false);
        assert!(!dsk.is_tie());
    }

    #[test]
    fn test_downstream_keyer_invert() {
        let mut dsk = DownstreamKeyer::new(0, 1, 2);
        assert!(!dsk.is_invert());

        dsk.set_invert(true);
        assert!(dsk.is_invert());
    }

    #[test]
    fn test_keyer_manager_creation() {
        let manager = KeyerManager::new(4, 2);
        assert_eq!(manager.upstream_count(), 4);
        assert_eq!(manager.downstream_count(), 2);
    }

    #[test]
    fn test_keyer_manager_get_upstream() {
        let mut manager = KeyerManager::new(4, 2);

        let usk = manager.get_upstream(0).expect("should succeed in test");
        assert_eq!(usk.config().id, 0);

        let usk_mut = manager.get_upstream_mut(1).expect("should succeed in test");
        usk_mut.set_on_air(true);
        assert!(manager
            .get_upstream(1)
            .expect("should succeed in test")
            .is_on_air());

        assert!(manager.get_upstream(10).is_err());
    }

    #[test]
    fn test_keyer_manager_get_downstream() {
        let mut manager = KeyerManager::new(4, 2);

        let dsk = manager.get_downstream(0).expect("should succeed in test");
        assert_eq!(dsk.config().id, 0);

        let dsk_mut = manager
            .get_downstream_mut(1)
            .expect("should succeed in test");
        dsk_mut.set_on_air(true);
        assert!(manager
            .get_downstream(1)
            .expect("should succeed in test")
            .is_on_air());

        assert!(manager.get_downstream(10).is_err());
    }

    #[test]
    fn test_keyer_manager_on_air_lists() {
        let mut manager = KeyerManager::new(4, 2);

        assert_eq!(manager.on_air_upstream().len(), 0);
        assert_eq!(manager.on_air_downstream().len(), 0);

        manager
            .get_upstream_mut(0)
            .expect("should succeed in test")
            .set_on_air(true);
        manager
            .get_upstream_mut(2)
            .expect("should succeed in test")
            .set_on_air(true);
        manager
            .get_downstream_mut(0)
            .expect("should succeed in test")
            .set_on_air(true);

        let on_air_usk = manager.on_air_upstream();
        assert_eq!(on_air_usk.len(), 2);
        assert!(on_air_usk.contains(&0));
        assert!(on_air_usk.contains(&2));

        let on_air_dsk = manager.on_air_downstream();
        assert_eq!(on_air_dsk.len(), 1);
        assert!(on_air_dsk.contains(&0));
    }

    #[test]
    fn test_keyer_type_variants() {
        assert_eq!(KeyerType::Luma, KeyerType::Luma);
        assert_ne!(KeyerType::Luma, KeyerType::Chroma);
        assert!(matches!(KeyerType::Linear, KeyerType::Linear));
        assert!(matches!(KeyerType::Pattern, KeyerType::Pattern));
    }

    #[test]
    fn test_upstream_keyer_type_change() {
        let mut usk = UpstreamKeyer::new(0, KeyerType::Luma, 1);
        assert_eq!(usk.keyer_type(), KeyerType::Luma);

        usk.set_type(KeyerType::Chroma);
        assert_eq!(usk.keyer_type(), KeyerType::Chroma);
    }

    #[test]
    fn test_auto_transition() {
        let mut dsk = DownstreamKeyer::new(0, 1, 2);
        assert!(!dsk.is_on_air());

        dsk.auto_transition(30).expect("should succeed in test");
        assert!(dsk.is_on_air());

        dsk.auto_transition(30).expect("should succeed in test");
        assert!(!dsk.is_on_air());
    }

    // -----------------------------------------------------------------------
    // process() tests for UpstreamKeyer and DownstreamKeyer
    // -----------------------------------------------------------------------

    use oximedia_codec::{Plane, VideoFrame};
    use oximedia_core::{PixelFormat, Rational, Timestamp};

    /// Build a minimal single-plane RGB24 VideoFrame filled with `val`.
    fn make_rgb24(w: u32, h: u32, val: u8) -> VideoFrame {
        let mut f = VideoFrame::new(PixelFormat::Rgb24, w, h);
        f.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        f.planes.push(Plane::with_dimensions(
            vec![val; (w * h * 3) as usize],
            (w * 3) as usize,
            w,
            h,
        ));
        f
    }

    /// Build a single-plane frame that carries a raw alpha matte (1 byte/pixel).
    fn make_alpha_frame(w: u32, h: u32, val: u8) -> VideoFrame {
        let mut f = VideoFrame::new(PixelFormat::Rgb24, w, h);
        f.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        f.planes.push(Plane::with_dimensions(
            vec![val; (w * h) as usize],
            w as usize,
            w,
            h,
        ));
        f
    }

    #[test]
    fn test_upstream_luma_returns_extra_alpha_plane() {
        let w = 4u32;
        let h = 4u32;
        let fill = make_rgb24(w, h, 200);
        let original_plane_count = fill.planes.len();

        let usk = UpstreamKeyer::new(0, KeyerType::Luma, 0);
        let result = usk
            .process(&fill, None)
            .expect("luma upstream must succeed");

        assert_eq!(
            result.planes.len(),
            original_plane_count + 1,
            "luma upstream must append exactly one alpha plane"
        );
    }

    #[test]
    fn test_upstream_chroma_returns_extra_alpha_plane() {
        let w = 4u32;
        let h = 4u32;
        let fill = make_rgb24(w, h, 128);
        let original_plane_count = fill.planes.len();

        let usk = UpstreamKeyer::new(0, KeyerType::Chroma, 0);
        let result = usk
            .process(&fill, None)
            .expect("chroma upstream must succeed");

        assert_eq!(
            result.planes.len(),
            original_plane_count + 1,
            "chroma upstream must append exactly one alpha plane"
        );
    }

    #[test]
    fn test_upstream_linear_with_explicit_key() {
        let w = 4u32;
        let h = 4u32;
        let fill = make_rgb24(w, h, 100);
        let key = make_alpha_frame(w, h, 200);
        let original_plane_count = fill.planes.len();

        let usk = UpstreamKeyer::new(0, KeyerType::Linear, 0);
        let result = usk
            .process(&fill, Some(&key))
            .expect("linear upstream with key must succeed");

        assert_eq!(
            result.planes.len(),
            original_plane_count + 1,
            "linear upstream must append exactly one alpha plane"
        );
        // Alpha data must come from the key frame's first plane.
        let alpha_plane = result.planes.last().expect("last plane must exist");
        assert!(
            alpha_plane.data.iter().all(|&b| b == 200),
            "alpha plane data must match key frame values"
        );
    }

    #[test]
    fn test_upstream_linear_missing_key_returns_invalid_key_source() {
        let w = 4u32;
        let h = 4u32;
        let fill = make_rgb24(w, h, 100);

        let usk = UpstreamKeyer::new(7, KeyerType::Linear, 0);
        let err = usk
            .process(&fill, None)
            .expect_err("linear upstream without key must fail");

        assert!(
            matches!(err, KeyerError::InvalidKeySource(7)),
            "error must be InvalidKeySource with the keyer id, got {err:?}"
        );
    }

    #[test]
    fn test_upstream_pattern_returns_processing_error() {
        let w = 4u32;
        let h = 4u32;
        let fill = make_rgb24(w, h, 100);

        let usk = UpstreamKeyer::new(0, KeyerType::Pattern, 0);
        let err = usk
            .process(&fill, None)
            .expect_err("pattern upstream must fail with ProcessingError");

        assert!(
            matches!(err, KeyerError::ProcessingError(_)),
            "error must be ProcessingError, got {err:?}"
        );
    }

    #[test]
    fn test_downstream_keyer_round_trip() {
        let w = 4u32;
        let h = 4u32;

        // program = 50, fill = 200, key alpha = 255 (fully opaque)
        let program = make_rgb24(w, h, 50);
        let fill = make_rgb24(w, h, 200);
        let key = make_alpha_frame(w, h, 255);

        let mut dsk = DownstreamKeyer::new(0, 0, 0);
        // clip=0 so no clipping, gain=1.0, invert=false
        dsk.set_clip(0.0);
        dsk.set_gain(1.0);

        let result = dsk
            .process(&program, &fill, &key)
            .expect("DSK process must succeed");

        // With fully-opaque alpha, result must match fill (200).
        for byte in &result.planes[0].data {
            assert_eq!(*byte, 200, "full alpha DSK must output fill values");
        }
    }

    #[test]
    fn test_downstream_keyer_zero_alpha_passes_program_through() {
        let w = 4u32;
        let h = 4u32;

        let program = make_rgb24(w, h, 50);
        let fill = make_rgb24(w, h, 200);
        let key = make_alpha_frame(w, h, 0); // fully transparent

        let mut dsk = DownstreamKeyer::new(0, 0, 0);
        dsk.set_clip(0.0);
        dsk.set_gain(1.0);

        let result = dsk
            .process(&program, &fill, &key)
            .expect("DSK process must succeed");

        // With zero alpha, result must equal program (50).
        for byte in &result.planes[0].data {
            assert_eq!(*byte, 50, "zero alpha DSK must output program values");
        }
    }

    #[test]
    fn test_downstream_keyer_invert_flips_alpha() {
        let w = 4u32;
        let h = 4u32;

        let program = make_rgb24(w, h, 50);
        let fill = make_rgb24(w, h, 200);
        // key alpha = 255 (fully opaque), but invert=true → becomes 0 (transparent)
        let key = make_alpha_frame(w, h, 255);

        let mut dsk = DownstreamKeyer::new(0, 0, 0);
        dsk.set_clip(0.0);
        dsk.set_gain(1.0);
        dsk.set_invert(true);

        let result = dsk
            .process(&program, &fill, &key)
            .expect("DSK process with invert must succeed");

        // After invert, alpha becomes 0 → output must equal program (50).
        for byte in &result.planes[0].data {
            assert_eq!(
                *byte, 50,
                "inverted full alpha DSK must output program values"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// SIMD-accelerated alpha blending
// ---------------------------------------------------------------------------

/// Scalar alpha compositing: `dst = fg * alpha/255 + bg * (255-alpha)/255`.
///
/// Processes RGBA bytes (4 bytes per pixel).  All slices must be the same
/// length and a multiple of 4 (one RGBA pixel = 4 bytes).
pub fn alpha_composite_scalar(fg: &[u8], bg: &[u8], alpha: &[u8], dst: &mut [u8]) {
    let n_pixels = fg.len() / 4;
    for p in 0..n_pixels {
        let base = p * 4;
        let a = alpha[p] as u32; // alpha for this pixel
        let ia = 255 - a; // inverse alpha
                          // Process R, G, B channels.
        for c in 0..3 {
            let idx = base + c;
            let blended = (fg[idx] as u32 * a + bg[idx] as u32 * ia) >> 8;
            dst[idx] = blended.min(255) as u8;
        }
        // Alpha channel of dst = alpha of fg (over operation).
        dst[base + 3] = alpha[p];
    }
}

/// AVX2-style alpha compositing — processes 8 RGBA pixels per outer iteration.
///
/// Uses a `>> 8` divide-by-256 approximation in 8-wide unrolled loops that
/// LLVM maps to 256-bit SIMD instructions on x86-64 targets that support AVX2.
/// No intrinsics or `unsafe` are required; the compiler auto-vectorises.
///
/// Formula: `dst ≈ (fg * alpha + bg * (255 - alpha)) >> 8`
pub fn alpha_composite_avx2(fg: &[u8], bg: &[u8], alpha: &[u8], dst: &mut [u8]) {
    let n_pixels = fg.len() / 4;
    let mut p = 0usize;

    // 8-pixel chunks — maps to a 256-bit AVX2 pass under auto-vectorisation.
    while p + 8 <= n_pixels {
        for i in 0..8 {
            let base = (p + i) * 4;
            let a = alpha[p + i] as u16;
            let ia = 255u16 - a;
            dst[base] = ((fg[base] as u16 * a + bg[base] as u16 * ia) >> 8) as u8;
            dst[base + 1] = ((fg[base + 1] as u16 * a + bg[base + 1] as u16 * ia) >> 8) as u8;
            dst[base + 2] = ((fg[base + 2] as u16 * a + bg[base + 2] as u16 * ia) >> 8) as u8;
            dst[base + 3] = alpha[p + i];
        }
        p += 8;
    }

    // Scalar tail for any remaining pixels.
    while p < n_pixels {
        let base = p * 4;
        let a = alpha[p] as u16;
        let ia = 255u16 - a;
        for c in 0..3 {
            let idx = base + c;
            dst[idx] = ((fg[idx] as u16 * a + bg[idx] as u16 * ia) >> 8) as u8;
        }
        dst[base + 3] = alpha[p];
        p += 1;
    }
}

/// SSE4.2-style alpha compositing — processes 4 RGBA pixels per outer iteration.
///
/// Uses the same `>> 8` approximation as [`alpha_composite_avx2`] but in
/// 4-wide unrolled loops, which LLVM maps to 128-bit SSE instructions.
pub fn alpha_composite_sse42(fg: &[u8], bg: &[u8], alpha: &[u8], dst: &mut [u8]) {
    let n_pixels = fg.len() / 4;
    let mut p = 0usize;

    // 4-pixel chunks — maps to a 128-bit SSE4.2 pass under auto-vectorisation.
    while p + 4 <= n_pixels {
        for i in 0..4 {
            let base = (p + i) * 4;
            let a = alpha[p + i] as u16;
            let ia = 255u16 - a;
            dst[base] = ((fg[base] as u16 * a + bg[base] as u16 * ia) >> 8) as u8;
            dst[base + 1] = ((fg[base + 1] as u16 * a + bg[base + 1] as u16 * ia) >> 8) as u8;
            dst[base + 2] = ((fg[base + 2] as u16 * a + bg[base + 2] as u16 * ia) >> 8) as u8;
            dst[base + 3] = alpha[p + i];
        }
        p += 4;
    }

    // Scalar tail.
    while p < n_pixels {
        let base = p * 4;
        let a = alpha[p] as u16;
        let ia = 255u16 - a;
        for c in 0..3 {
            let idx = base + c;
            dst[idx] = ((fg[idx] as u16 * a + bg[idx] as u16 * ia) >> 8) as u8;
        }
        dst[base + 3] = alpha[p];
        p += 1;
    }
}

/// Runtime-dispatched alpha compositing: AVX2 > SSE4.2 > scalar.
pub fn alpha_composite_dispatch(fg: &[u8], bg: &[u8], alpha: &[u8], dst: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        alpha_composite_avx2(fg, bg, alpha, dst);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("sse4.2") {
        alpha_composite_sse42(fg, bg, alpha, dst);
        return;
    }
    alpha_composite_scalar(fg, bg, alpha, dst);
}

#[cfg(test)]
mod blend_tests {
    use super::*;

    /// Build an RGBA image of `n_pixels` pixels where each pixel is `(r, g, b, a)`.
    fn make_rgba(n_pixels: usize, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(n_pixels * 4);
        for _ in 0..n_pixels {
            v.extend_from_slice(&[r, g, b, a]);
        }
        v
    }

    /// Build a per-pixel alpha slice of `n_pixels`.
    fn make_alpha(n_pixels: usize, a: u8) -> Vec<u8> {
        vec![a; n_pixels]
    }

    /// Check that the scalar and SIMD paths agree within ±1 for rounding.
    fn assert_within_1(scalar: &[u8], simd: &[u8], label: &str) {
        assert_eq!(scalar.len(), simd.len(), "{label}: length mismatch");
        for (i, (&s, &d)) in scalar.iter().zip(simd.iter()).enumerate() {
            let diff = (s as i16 - d as i16).unsigned_abs();
            assert!(
                diff <= 1,
                "{label}: byte {i}: scalar={s} simd={d} diff={diff}"
            );
        }
    }

    #[test]
    fn test_alpha_composite_fully_transparent() {
        // Alpha = 0 → dst should equal bg.
        let n = 8;
        let fg = make_rgba(n, 255, 0, 0, 255);
        let bg = make_rgba(n, 0, 255, 0, 255);
        let alpha = make_alpha(n, 0);
        let mut dst = vec![0u8; n * 4];
        alpha_composite_scalar(&fg, &bg, &alpha, &mut dst);
        for p in 0..n {
            let base = p * 4;
            // bg is (0, 255, 0, 255); at alpha=0 dst channels ≈ bg channels.
            assert!(dst[base] <= 1, "R should be ~0, got {}", dst[base]);
            assert!(
                (dst[base + 1] as i16 - 255).abs() <= 1,
                "G should be ~255, got {}",
                dst[base + 1]
            );
            assert!(dst[base + 2] <= 1, "B should be ~0, got {}", dst[base + 2]);
        }
    }

    #[test]
    fn test_alpha_composite_fully_opaque() {
        // Alpha = 255 → dst should equal fg.
        let n = 8;
        let fg = make_rgba(n, 255, 128, 64, 255);
        let bg = make_rgba(n, 0, 0, 0, 255);
        let alpha = make_alpha(n, 255);
        let mut dst = vec![0u8; n * 4];
        alpha_composite_scalar(&fg, &bg, &alpha, &mut dst);
        for p in 0..n {
            let base = p * 4;
            assert!(
                (dst[base] as i16 - 255).abs() <= 1,
                "R should be ~255, got {}",
                dst[base]
            );
            assert!(
                (dst[base + 1] as i16 - 128).abs() <= 1,
                "G should be ~128, got {}",
                dst[base + 1]
            );
            assert!(
                (dst[base + 2] as i16 - 64).abs() <= 1,
                "B should be ~64, got {}",
                dst[base + 2]
            );
        }
    }

    #[test]
    fn test_alpha_composite_50pct_blend() {
        // Alpha = 128 ≈ 50% → dst ≈ (fg + bg) / 2.
        let n = 8;
        let fg = make_rgba(n, 200, 100, 50, 255);
        let bg = make_rgba(n, 100, 200, 150, 255);
        let alpha = make_alpha(n, 128);
        let mut dst = vec![0u8; n * 4];
        alpha_composite_scalar(&fg, &bg, &alpha, &mut dst);
        for p in 0..n {
            let base = p * 4;
            // Expected mid-point (±2 for rounding).
            let exp_r = (200u16 * 128 + 100u16 * 127) / 255;
            let exp_g = (100u16 * 128 + 200u16 * 127) / 255;
            let exp_b = (50u16 * 128 + 150u16 * 127) / 255;
            assert!((dst[base] as i16 - exp_r as i16).abs() <= 2);
            assert!((dst[base + 1] as i16 - exp_g as i16).abs() <= 2);
            assert!((dst[base + 2] as i16 - exp_b as i16).abs() <= 2);
        }
    }

    #[test]
    fn test_avx2_matches_scalar_transparent() {
        let n = 16;
        let fg = make_rgba(n, 255, 0, 0, 255);
        let bg = make_rgba(n, 0, 255, 0, 255);
        let alpha = make_alpha(n, 0);
        let mut scalar_dst = vec![0u8; n * 4];
        let mut avx2_dst = vec![0u8; n * 4];
        alpha_composite_scalar(&fg, &bg, &alpha, &mut scalar_dst);
        alpha_composite_avx2(&fg, &bg, &alpha, &mut avx2_dst);
        assert_within_1(&scalar_dst, &avx2_dst, "avx2 transparent");
    }

    #[test]
    fn test_avx2_matches_scalar_opaque() {
        let n = 16;
        let fg = make_rgba(n, 0, 128, 255, 255);
        let bg = make_rgba(n, 100, 50, 25, 255);
        let alpha = make_alpha(n, 255);
        let mut scalar_dst = vec![0u8; n * 4];
        let mut avx2_dst = vec![0u8; n * 4];
        alpha_composite_scalar(&fg, &bg, &alpha, &mut scalar_dst);
        alpha_composite_avx2(&fg, &bg, &alpha, &mut avx2_dst);
        assert_within_1(&scalar_dst, &avx2_dst, "avx2 opaque");
    }

    #[test]
    fn test_sse42_matches_scalar_50pct() {
        let n = 16;
        let fg = make_rgba(n, 200, 100, 50, 255);
        let bg = make_rgba(n, 100, 200, 150, 255);
        let alpha = make_alpha(n, 128);
        let mut scalar_dst = vec![0u8; n * 4];
        let mut sse_dst = vec![0u8; n * 4];
        alpha_composite_scalar(&fg, &bg, &alpha, &mut scalar_dst);
        alpha_composite_sse42(&fg, &bg, &alpha, &mut sse_dst);
        assert_within_1(&scalar_dst, &sse_dst, "sse42 50pct");
    }

    #[test]
    fn test_dispatch_matches_scalar() {
        let n = 32;
        let fg = make_rgba(n, 180, 90, 45, 255);
        let bg = make_rgba(n, 20, 40, 60, 255);
        let alpha: Vec<u8> = (0..n).map(|i| (i * 8) as u8).collect();
        let mut scalar_dst = vec![0u8; n * 4];
        let mut disp_dst = vec![0u8; n * 4];
        alpha_composite_scalar(&fg, &bg, &alpha, &mut scalar_dst);
        alpha_composite_dispatch(&fg, &bg, &alpha, &mut disp_dst);
        assert_within_1(&scalar_dst, &disp_dst, "dispatch varying alpha");
    }

    #[test]
    fn test_alpha_composite_tail_pixels() {
        // 3 pixels — exercises the scalar tail path in AVX2/SSE42 (< 8 / < 4 pixels).
        let n = 3;
        let fg = make_rgba(n, 100, 150, 200, 255);
        let bg = make_rgba(n, 50, 75, 100, 255);
        let alpha = make_alpha(n, 200);
        let mut scalar_dst = vec![0u8; n * 4];
        let mut avx2_dst = vec![0u8; n * 4];
        alpha_composite_scalar(&fg, &bg, &alpha, &mut scalar_dst);
        alpha_composite_avx2(&fg, &bg, &alpha, &mut avx2_dst);
        assert_within_1(&scalar_dst, &avx2_dst, "avx2 tail");
    }

    #[test]
    fn test_alpha_stored_in_dst_alpha_channel() {
        // The dst alpha channel should hold the per-pixel input alpha.
        let n = 4;
        let fg = make_rgba(n, 255, 0, 0, 255);
        let bg = make_rgba(n, 0, 255, 0, 255);
        let alpha: Vec<u8> = vec![0, 64, 128, 255];
        let mut dst = vec![0u8; n * 4];
        alpha_composite_scalar(&fg, &bg, &alpha, &mut dst);
        for (p, &expected_a) in alpha.iter().enumerate() {
            assert_eq!(
                dst[p * 4 + 3],
                expected_a,
                "pixel {p}: alpha channel mismatch"
            );
        }
    }
}
