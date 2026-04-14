//! AV1-compatible film grain parameters (simplified subset).
//!
//! Provides [`FilmGrainParams`] for encoding grain characteristics that match
//! the AV1 film grain synthesis specification, along with helper functions for
//! estimating parameters from real frame noise, synthesising grain samples, and
//! applying grain to normalised f32 pixel buffers.

// ============================================================================
// AV1-compatible film grain parameters (simplified task-spec API)
// ============================================================================

/// AV1 film grain parameters (simplified subset).
#[derive(Debug, Clone)]
pub struct FilmGrainParams {
    /// Whether to apply grain at all.
    pub apply_grain: bool,
    /// Pseudo-random seed for grain generation.
    pub grain_seed: u16,
    /// Luma grain scaling in [0, 15].  0 = no grain, 15 = maximum grain.
    pub luma_scaling: u8,
    /// Derive chroma grain parameters from luma when `true`.
    pub chroma_scaling_from_luma: bool,
    /// Chroma-blue grain multiplier.
    pub cb_mult: u8,
    /// Chroma-red grain multiplier.
    pub cr_mult: u8,
    /// Enable overlap removal between neighbouring grain blocks.
    pub overlap_flag: bool,
    /// Clip output to restricted (studio-swing) range.
    pub clip_to_restricted_range: bool,
}

impl FilmGrainParams {
    /// No-grain preset: `apply_grain = false`, all fields zero.
    pub fn none() -> Self {
        Self {
            apply_grain: false,
            grain_seed: 0,
            luma_scaling: 0,
            chroma_scaling_from_luma: false,
            cb_mult: 0,
            cr_mult: 0,
            overlap_flag: false,
            clip_to_restricted_range: false,
        }
    }

    /// Light grain preset: `luma_scaling = 4`.
    pub fn light() -> Self {
        Self {
            apply_grain: true,
            grain_seed: 0x1234,
            luma_scaling: 4,
            chroma_scaling_from_luma: true,
            cb_mult: 128,
            cr_mult: 128,
            overlap_flag: true,
            clip_to_restricted_range: false,
        }
    }

    /// Medium grain preset: `luma_scaling = 8`.
    pub fn medium() -> Self {
        Self {
            apply_grain: true,
            grain_seed: 0x5678,
            luma_scaling: 8,
            chroma_scaling_from_luma: true,
            cb_mult: 128,
            cr_mult: 128,
            overlap_flag: true,
            clip_to_restricted_range: false,
        }
    }

    /// Heavy grain preset: `luma_scaling = 12`.
    pub fn heavy() -> Self {
        Self {
            apply_grain: true,
            grain_seed: 0x9abc,
            luma_scaling: 12,
            chroma_scaling_from_luma: false,
            cb_mult: 192,
            cr_mult: 192,
            overlap_flag: true,
            clip_to_restricted_range: false,
        }
    }

    /// Create params from a continuous noise level in [0.0, 1.0].
    ///
    /// Maps linearly to `luma_scaling` in [0, 15].
    pub fn from_noise_level(noise_level: f32) -> Self {
        let scaling = (noise_level.clamp(0.0, 1.0) * 15.0).round() as u8;
        if scaling == 0 {
            return Self::none();
        }
        Self {
            apply_grain: true,
            grain_seed: (noise_level * 65535.0) as u16,
            luma_scaling: scaling,
            chroma_scaling_from_luma: true,
            cb_mult: 128,
            cr_mult: 128,
            overlap_flag: true,
            clip_to_restricted_range: false,
        }
    }

    /// Serialize parameters to a minimal JSON string.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"apply_grain":{apply_grain},"grain_seed":{grain_seed},"luma_scaling":{luma_scaling},"chroma_scaling_from_luma":{csl},"cb_mult":{cb_mult},"cr_mult":{cr_mult},"overlap_flag":{overlap},"clip_to_restricted_range":{clip}}}"#,
            apply_grain = self.apply_grain,
            grain_seed = self.grain_seed,
            luma_scaling = self.luma_scaling,
            csl = self.chroma_scaling_from_luma,
            cb_mult = self.cb_mult,
            cr_mult = self.cr_mult,
            overlap = self.overlap_flag,
            clip = self.clip_to_restricted_range,
        )
    }
}

/// Estimate `FilmGrainParams` from the noise level measured in a luma plane.
///
/// Analyses local 3×3 variance in flat regions to estimate noise power, then
/// converts that to a `FilmGrainParams` via [`FilmGrainParams::from_noise_level`].
///
/// # Arguments
///
/// * `luma_pixels` – flat row-major array of f32 luma values in [0.0, 1.0].
/// * `width` / `height` – frame dimensions.
pub fn estimate_av1_grain_params(
    luma_pixels: &[f32],
    width: usize,
    height: usize,
) -> Result<FilmGrainParams, String> {
    if width == 0 || height == 0 {
        return Err(format!("invalid dimensions: {width}x{height}"));
    }
    if luma_pixels.len() != width * height {
        return Err(format!(
            "buffer length {} does not match {width}x{height}",
            luma_pixels.len()
        ));
    }

    let mut variance_sum = 0.0f64;
    let mut count = 0u64;

    for y in 1..(height.saturating_sub(1)) {
        for x in 1..(width.saturating_sub(1)) {
            // Collect 3×3 neighbourhood
            let mut local = [0.0f32; 9];
            let mut idx = 0;
            for ky in -1i32..=1 {
                for kx in -1i32..=1 {
                    let ny = (y as i32 + ky) as usize;
                    let nx = (x as i32 + kx) as usize;
                    local[idx] = luma_pixels[ny * width + nx];
                    idx += 1;
                }
            }
            let mean = local.iter().sum::<f32>() / 9.0;
            let var: f32 = local.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / 9.0;
            variance_sum += var as f64;
            count += 1;
        }
    }

    let noise_power = if count > 0 {
        variance_sum / count as f64
    } else {
        0.0
    };
    // Scale factor converts typical noise variance to [0, 1] range.
    // Typical mild noise is ~0.001 in normalised [0,1] units.
    let noise_level = (noise_power.sqrt() as f32 * 10.0).clamp(0.0, 1.0);
    Ok(FilmGrainParams::from_noise_level(noise_level))
}

/// Synthesise additive grain samples using a simple LCG random number generator.
///
/// * `seed` – initial RNG state.
/// * `scaling` – grain amplitude in [0, 15]; scales maximum amplitude to 5%.
/// * `count` – number of grain samples to generate.
///
/// Returns samples in the range `[-scaling/15 * 0.05, +scaling/15 * 0.05]`.
pub fn synthesize_grain(seed: u64, scaling: u8, count: usize) -> Vec<f32> {
    let mut state = seed;
    let amplitude = (scaling as f32 / 15.0) * 0.05;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Map high bits to [-0.5, 0.5]
        let raw = (state >> 48) as f32 / 65536.0 - 0.5;
        out.push(raw * amplitude);
    }
    out
}

/// Apply synthesised film grain to a normalised f32 pixel buffer.
///
/// Uses `params.grain_seed` as the LCG seed and clamps output to [0.0, 1.0].
/// Returns `Err` if `params.apply_grain == false` (no-op guard).
pub fn apply_film_grain_params(pixels: &mut [f32], params: &FilmGrainParams) -> Result<(), String> {
    if !params.apply_grain {
        return Err("apply_grain is false; no grain applied".to_string());
    }
    let grain = synthesize_grain(params.grain_seed as u64, params.luma_scaling, pixels.len());
    for (p, g) in pixels.iter_mut().zip(grain.iter()) {
        *p = (*p + g).clamp(0.0, 1.0);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1. FilmGrainParams::none() has apply_grain=false
    #[test]
    fn test_film_grain_params_none() {
        let p = FilmGrainParams::none();
        assert!(!p.apply_grain);
        assert_eq!(p.luma_scaling, 0);
    }

    // 2. FilmGrainParams::light() has luma_scaling=4
    #[test]
    fn test_film_grain_params_light() {
        let p = FilmGrainParams::light();
        assert!(p.apply_grain);
        assert_eq!(p.luma_scaling, 4);
    }

    // 3. FilmGrainParams::medium() has luma_scaling=8
    #[test]
    fn test_film_grain_params_medium() {
        let p = FilmGrainParams::medium();
        assert!(p.apply_grain);
        assert_eq!(p.luma_scaling, 8);
    }

    // 4. FilmGrainParams::heavy() has luma_scaling=12
    #[test]
    fn test_film_grain_params_heavy() {
        let p = FilmGrainParams::heavy();
        assert!(p.apply_grain);
        assert_eq!(p.luma_scaling, 12);
    }

    // 5. from_noise_level maps 0.0 → luma_scaling=0 (apply_grain=false)
    #[test]
    fn test_from_noise_level_zero() {
        let p = FilmGrainParams::from_noise_level(0.0);
        assert!(!p.apply_grain);
        assert_eq!(p.luma_scaling, 0);
    }

    // 6. from_noise_level maps 1.0 → luma_scaling=15
    #[test]
    fn test_from_noise_level_one() {
        let p = FilmGrainParams::from_noise_level(1.0);
        assert!(p.apply_grain);
        assert_eq!(p.luma_scaling, 15);
    }

    // 7. synthesize_grain returns correct length
    #[test]
    fn test_synthesize_grain_length() {
        let v = synthesize_grain(42, 8, 100);
        assert_eq!(v.len(), 100);
    }

    // 8. synthesize_grain values are in valid amplitude range
    #[test]
    fn test_synthesize_grain_range() {
        let scaling = 15u8;
        let v = synthesize_grain(0xdead_beef, scaling, 1000);
        let max_amp = (scaling as f32 / 15.0) * 0.05;
        for &g in &v {
            assert!(
                g.abs() <= max_amp + 1e-6,
                "grain sample {g} out of range [-{max_amp}, {max_amp}]"
            );
        }
    }

    // 9. apply_film_grain_params modifies pixels
    #[test]
    fn test_apply_film_grain_modifies_pixels() {
        let mut pixels = vec![0.5f32; 64];
        let original = pixels.clone();
        let params = FilmGrainParams::medium();
        apply_film_grain_params(&mut pixels, &params).expect("should apply grain");
        assert!(
            pixels
                .iter()
                .zip(original.iter())
                .any(|(a, b)| (a - b).abs() > 1e-9),
            "grain application must modify at least one pixel"
        );
    }

    // 10. to_json contains luma_scaling field
    #[test]
    fn test_to_json_contains_luma_scaling() {
        let p = FilmGrainParams::medium();
        let json = p.to_json();
        assert!(
            json.contains("luma_scaling"),
            "JSON must contain 'luma_scaling': {json}"
        );
        assert!(
            json.contains("\"luma_scaling\":8"),
            "JSON luma_scaling must be 8: {json}"
        );
    }

    // 11. estimate_av1_grain_params returns valid params on noisy input
    #[test]
    fn test_estimate_av1_grain_params_valid() {
        let mut pixels = Vec::with_capacity(64 * 64);
        let mut lcg = 0xabcdef01u64;
        for _ in 0..64 * 64 {
            lcg = lcg
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = (lcg >> 48) as f32 / 65536.0;
            pixels.push(v * 0.1 + 0.45);
        }
        let params = estimate_av1_grain_params(&pixels, 64, 64).expect("estimation should succeed");
        assert!(params.luma_scaling <= 15);
    }

    // 12. apply_film_grain_params with apply_grain=false returns Err
    #[test]
    fn test_apply_film_grain_no_op_on_false() {
        let mut pixels = vec![0.5f32; 16];
        let params = FilmGrainParams::none();
        let result = apply_film_grain_params(&mut pixels, &params);
        assert!(result.is_err(), "should return Err when apply_grain=false");
    }
}
