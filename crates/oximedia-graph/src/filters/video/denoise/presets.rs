/// Denoise presets for common use cases.
use super::{DenoiseConfig, DenoiseMethod, MotionQuality, TemporalMode};

/// Light denoising for slightly noisy footage.
#[must_use]
pub fn light() -> DenoiseConfig {
    DenoiseConfig::new()
        .with_method(DenoiseMethod::Bilateral)
        .with_strength(0.4)
        .with_spatial_radius(3)
        .with_bilateral_params(30.0, 8.0)
}

/// Medium denoising for moderately noisy footage.
#[must_use]
pub fn medium() -> DenoiseConfig {
    DenoiseConfig::new()
        .with_method(DenoiseMethod::Combined)
        .with_strength(0.7)
        .with_spatial_radius(5)
        .with_temporal_depth(3)
        .with_bilateral_params(50.0, 10.0)
}

/// Strong denoising for very noisy footage.
#[must_use]
pub fn strong() -> DenoiseConfig {
    DenoiseConfig::new()
        .with_method(DenoiseMethod::NonLocalMeans)
        .with_strength(0.9)
        .with_spatial_radius(7)
        .with_nlm_params(15.0, 21, 7)
}

/// Temporal-only denoising.
#[must_use]
pub fn temporal() -> DenoiseConfig {
    DenoiseConfig::new()
        .with_method(DenoiseMethod::Temporal)
        .with_strength(0.7)
        .with_temporal_depth(5)
        .with_temporal_mode(TemporalMode::WeightedAverage)
}

/// Motion-compensated temporal denoising.
#[must_use]
pub fn motion_compensated() -> DenoiseConfig {
    DenoiseConfig::new()
        .with_method(DenoiseMethod::MotionCompensated)
        .with_strength(0.8)
        .with_temporal_depth(3)
        .with_motion_quality(MotionQuality::High)
}

/// BM3D-style denoising.
#[must_use]
pub fn bm3d() -> DenoiseConfig {
    DenoiseConfig::new()
        .with_method(DenoiseMethod::BlockMatching3D)
        .with_strength(0.8)
        .with_spatial_radius(8)
        .with_nlm_params(12.0, 21, 8)
}

/// Chroma-only denoising.
#[must_use]
pub fn chroma_only() -> DenoiseConfig {
    DenoiseConfig::new()
        .with_method(DenoiseMethod::Bilateral)
        .with_strength(0.8)
        .with_luma_strength(0.0)
        .with_chroma_strength(1.5)
        .with_spatial_radius(5)
}

/// Fast denoising (lower quality, faster).
#[must_use]
pub fn fast() -> DenoiseConfig {
    DenoiseConfig::new()
        .with_method(DenoiseMethod::Gaussian)
        .with_strength(0.6)
        .with_spatial_radius(3)
}
