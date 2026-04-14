//! Film grain simulation using a linear congruential generator (LCG).
//!
//! Provides per-pixel noise injection for vintage film-stock looks,
//! archival restoration simulation, and cinematic texture grading.
//!
//! The LCG parameters follow Knuth's constants:
//! - multiplier  = 6 364 136 223 846 793 005
//! - increment   = 1 442 695 040 888 963 407

/// Linear Congruential Generator (LCG) for deterministic noise.
///
/// State advances as: `state = state × M + A (mod 2^64)`.
#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    /// Create a new LCG seeded with `seed`.
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance and return the next raw 64-bit value.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005_u64)
            .wrapping_add(1_442_695_040_888_963_407_u64);
        self.state
    }

    /// Return a float in `[-1.0, 1.0)`.
    fn next_f32(&mut self) -> f32 {
        // Map upper 32 bits to [-1, 1)
        let raw = (self.next_u64() >> 32) as u32;
        (raw as f32 / 2_147_483_648.0) - 1.0
    }
}

// ── FilmGrain ─────────────────────────────────────────────────────────────────

/// Film grain effect that adds per-pixel LCG noise to a raw RGBA image.
///
/// # Example
///
/// ```rust
/// use oximedia_vfx::grain::FilmGrain;
///
/// let grain = FilmGrain::new(0.05, 42);
/// let mut img = vec![128u8; 4 * 4 * 4]; // 4×4 RGBA
/// grain.apply(&mut img, 4, 4);
/// ```
#[derive(Debug, Clone)]
pub struct FilmGrain {
    /// Noise amplitude in the range [0, 1] where 1 adds ±255 to each channel.
    strength: f32,
    /// Initial LCG seed, making the grain pattern deterministic.
    seed: u64,
}

impl FilmGrain {
    /// Create a new `FilmGrain` effect.
    ///
    /// # Arguments
    ///
    /// * `strength` — noise amplitude in [0, 1]; clamped if out of range.
    /// * `seed`     — LCG seed for deterministic noise.
    #[must_use]
    pub fn new(strength: f32, seed: u64) -> Self {
        Self {
            strength: strength.clamp(0.0, 1.0),
            seed,
        }
    }

    /// Return the configured noise strength.
    #[must_use]
    pub fn strength(&self) -> f32 {
        self.strength
    }

    /// Return the configured LCG seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Apply film grain noise to `img` in-place.
    ///
    /// `img` is a flat RGBA byte slice of length `w * h * 4`.
    /// The alpha channel is left unchanged.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `img.len() != w * h * 4`.
    pub fn apply(&self, img: &mut Vec<u8>, w: u32, h: u32) {
        let expected = (w as usize) * (h as usize) * 4;
        debug_assert_eq!(img.len(), expected, "buffer length mismatch");

        if self.strength == 0.0 || img.len() < expected {
            return;
        }

        let amplitude = self.strength * 255.0;
        let mut rng = Lcg::new(self.seed);

        // Process RGB channels, skip alpha (index 3)
        let pixels = (w as usize) * (h as usize);
        for p in 0..pixels {
            let base = p * 4;

            // One LCG call per pixel — derive individual channel offsets
            // by splitting bits across R, G, B.
            let noise_r = rng.next_f32();
            let noise_g = rng.next_f32();
            let noise_b = rng.next_f32();

            let r = img[base] as f32 + noise_r * amplitude;
            let g = img[base + 1] as f32 + noise_g * amplitude;
            let b = img[base + 2] as f32 + noise_b * amplitude;

            img[base] = r.clamp(0.0, 255.0) as u8;
            img[base + 1] = g.clamp(0.0, 255.0) as u8;
            img[base + 2] = b.clamp(0.0, 255.0) as u8;
            // img[base + 3]: alpha untouched
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcg_deterministic() {
        let mut a = Lcg::new(12345);
        let mut b = Lcg::new(12345);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn test_lcg_different_seeds_differ() {
        let mut a = Lcg::new(1);
        let mut b = Lcg::new(2);
        // At least one of the first 8 outputs should differ
        let same = (0..8).all(|_| a.next_u64() == b.next_u64());
        assert!(!same);
    }

    #[test]
    fn test_lcg_f32_range() {
        let mut rng = Lcg::new(99);
        for _ in 0..10_000 {
            let v = rng.next_f32();
            assert!(v >= -1.0 && v < 1.0, "out of range: {v}");
        }
    }

    #[test]
    fn test_film_grain_new_clamps_strength() {
        let g = FilmGrain::new(2.0, 0);
        assert_eq!(g.strength(), 1.0);

        let g2 = FilmGrain::new(-0.5, 0);
        assert_eq!(g2.strength(), 0.0);
    }

    #[test]
    fn test_film_grain_zero_strength_no_change() {
        let grain = FilmGrain::new(0.0, 7);
        let mut img: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect(); // 4×4
        let before = img.clone();
        grain.apply(&mut img, 4, 4);
        assert_eq!(img, before, "zero-strength should leave image unchanged");
    }

    #[test]
    fn test_film_grain_alpha_preserved() {
        let grain = FilmGrain::new(1.0, 42);
        let mut img = vec![0u8; 4 * 4 * 4]; // 4×4 RGBA, all zeroes
        // Set all alphas to 0xFF
        for p in 0..16 {
            img[p * 4 + 3] = 0xFF;
        }
        grain.apply(&mut img, 4, 4);
        for p in 0..16 {
            assert_eq!(img[p * 4 + 3], 0xFF, "alpha must not be modified");
        }
    }

    #[test]
    fn test_film_grain_output_length_unchanged() {
        let grain = FilmGrain::new(0.3, 1);
        let mut img = vec![128u8; 8 * 8 * 4];
        grain.apply(&mut img, 8, 8);
        assert_eq!(img.len(), 8 * 8 * 4);
    }

    #[test]
    fn test_film_grain_deterministic() {
        let grain = FilmGrain::new(0.5, 777);
        let mut a = vec![100u8; 4 * 4 * 4];
        let mut b = vec![100u8; 4 * 4 * 4];
        grain.apply(&mut a, 4, 4);
        grain.apply(&mut b, 4, 4);
        assert_eq!(a, b, "same seed must produce identical output");
    }

    #[test]
    fn test_film_grain_different_seeds_differ() {
        let g1 = FilmGrain::new(0.5, 1);
        let g2 = FilmGrain::new(0.5, 2);
        let mut a = vec![128u8; 16 * 16 * 4];
        let mut b = a.clone();
        g1.apply(&mut a, 16, 16);
        g2.apply(&mut b, 16, 16);
        assert_ne!(a, b, "different seeds should produce different noise");
    }

    #[test]
    fn test_film_grain_clamps_output_to_valid_range() {
        // Start with max values — grain should not overflow
        let grain = FilmGrain::new(1.0, 99);
        let mut img = vec![255u8; 4 * 4 * 4];
        grain.apply(&mut img, 4, 4);
        for (i, &v) in img.iter().enumerate() {
            if i % 4 != 3 {
                // RGB must remain [0, 255]
                assert!(v <= 255, "pixel out of range");
            }
        }

        // Start with min values
        let mut img2 = vec![0u8; 4 * 4 * 4];
        grain.apply(&mut img2, 4, 4);
        for (i, &v) in img2.iter().enumerate() {
            if i % 4 != 3 {
                assert!(v <= 255);
            }
        }
    }
}
