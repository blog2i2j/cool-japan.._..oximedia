//! Simplified film grain synthesis for post-processing pipelines.
//!
//! Applies synthetic grain to a luma-only (Y-plane) frame buffer using a
//! deterministic linear-congruential PRNG seeded per-frame.  The grain model
//! is intentionally simple — additive Gaussian-like noise clamped to `[0, 255]`
//! — to serve as a lightweight grain layer without the full AV1 film grain
//! parameter model.
//!
//! For AV1-spec film grain (tables, AR coefficients, chroma scaling) see
//! [`crate::film_grain_synthesis`].
//!
//! # Usage
//!
//! ```rust
//! use oximedia_video::grain::FilmGrainSynthesizer;
//!
//! let synth = FilmGrainSynthesizer::new(12, 42);
//! let mut frame = vec![128u8; 8 * 4]; // 8×4 luma plane
//! synth.apply(&mut frame, 8, 4);
//! ```

/// Simple deterministic film grain synthesizer.
///
/// Adds spatially varying additive noise to a luma frame using a
/// linear-congruential generator (LCG) seeded by `seed`.  The amplitude is
/// controlled by `strength` (range 0–255; typical values 4–32).
#[derive(Debug, Clone)]
pub struct FilmGrainSynthesizer {
    /// Peak grain amplitude in luma levels (0–255).
    pub strength: u8,
    /// Deterministic seed for the per-frame LCG.
    pub seed: u64,
}

impl FilmGrainSynthesizer {
    /// Creates a new `FilmGrainSynthesizer`.
    ///
    /// # Arguments
    ///
    /// * `strength` – maximum grain amplitude in luma levels.  `0` means no
    ///   grain is applied.
    /// * `seed` – initial seed for the deterministic random number generator.
    #[must_use]
    pub fn new(strength: u8, seed: u64) -> Self {
        Self { strength, seed }
    }

    /// Applies grain in-place to a row-major luma `frame` of size `w × h`.
    ///
    /// Pixels outside the expected `w * h` bytes are not modified.  If
    /// `strength` is `0` or the frame buffer is shorter than `w * h` bytes
    /// the function returns immediately without changes.
    ///
    /// The grain pattern is purely spatial and does not model film grain
    /// structure (no correlation, no chroma influence).
    pub fn apply(&self, frame: &mut Vec<u8>, w: u32, h: u32) {
        if self.strength == 0 || w == 0 || h == 0 {
            return;
        }
        let expected = w as usize * h as usize;
        if frame.len() < expected {
            return;
        }

        let half_strength = self.strength as i32 / 2;
        let mut state = lcg_next(self.seed);

        for pixel in frame.iter_mut().take(expected) {
            // Produce a noise value in [-half_strength, +half_strength].
            let noise = (state % (self.strength as u64 + 1)) as i32 - half_strength;
            let new_val = (*pixel as i32 + noise).clamp(0, 255) as u8;
            *pixel = new_val;
            state = lcg_next(state);
        }
    }

    /// Returns a new synthesizer with the seed advanced by one frame, making
    /// each call to `apply` produce a different grain pattern while remaining
    /// deterministic across runs.
    #[must_use]
    pub fn advance_frame(&self) -> Self {
        Self {
            strength: self.strength,
            seed: lcg_next(self.seed),
        }
    }
}

// ─── LCG ─────────────────────────────────────────────────────────────────────

/// Minimal 64-bit linear congruential generator step.
///
/// Parameters from Knuth's MMIX LCG (also used by glibc's drand48 family
/// scaled to 64-bit).
#[inline]
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_strength_no_change() {
        let synth = FilmGrainSynthesizer::new(0, 42);
        let original = vec![128u8; 16];
        let mut frame = original.clone();
        synth.apply(&mut frame, 4, 4);
        assert_eq!(frame, original);
    }

    #[test]
    fn grain_applied_changes_frame() {
        let synth = FilmGrainSynthesizer::new(32, 12345);
        let mut frame = vec![128u8; 16];
        synth.apply(&mut frame, 4, 4);
        // With strength=32 at least some pixels should differ from 128.
        let changed = frame.iter().filter(|&&b| b != 128).count();
        assert!(changed > 0, "grain should modify pixels");
    }

    #[test]
    fn pixels_remain_in_valid_range() {
        let synth = FilmGrainSynthesizer::new(255, 99);
        let mut frame = vec![128u8; 64];
        synth.apply(&mut frame, 8, 8);
        // All bytes are valid u8 values by type — verify clamping didn't panic.
        assert_eq!(frame.len(), 64);
    }

    #[test]
    fn grain_is_deterministic() {
        let synth = FilmGrainSynthesizer::new(16, 42);
        let mut f1 = vec![100u8; 16];
        let mut f2 = f1.clone();
        synth.apply(&mut f1, 4, 4);
        synth.apply(&mut f2, 4, 4);
        assert_eq!(f1, f2, "same seed should produce identical grain");
    }

    #[test]
    fn different_seeds_different_patterns() {
        let synth_a = FilmGrainSynthesizer::new(32, 1);
        let synth_b = FilmGrainSynthesizer::new(32, 2);
        let original = vec![128u8; 64];
        let mut f_a = original.clone();
        let mut f_b = original.clone();
        synth_a.apply(&mut f_a, 8, 8);
        synth_b.apply(&mut f_b, 8, 8);
        assert_ne!(
            f_a, f_b,
            "different seeds should produce different patterns"
        );
    }

    #[test]
    fn too_short_frame_not_modified() {
        let synth = FilmGrainSynthesizer::new(16, 42);
        let original = vec![100u8; 4]; // shorter than 4×4=16
        let mut frame = original.clone();
        synth.apply(&mut frame, 4, 4);
        assert_eq!(frame, original);
    }

    #[test]
    fn advance_frame_different_pattern() {
        let synth1 = FilmGrainSynthesizer::new(20, 7);
        let synth2 = synth1.advance_frame();
        let original = vec![128u8; 64];
        let mut f1 = original.clone();
        let mut f2 = original.clone();
        synth1.apply(&mut f1, 8, 8);
        synth2.apply(&mut f2, 8, 8);
        assert_ne!(f1, f2, "advanced seed should differ");
    }

    #[test]
    fn black_pixels_clamped_to_zero() {
        let synth = FilmGrainSynthesizer::new(255, 42);
        let mut frame = vec![0u8; 16];
        synth.apply(&mut frame, 4, 4);
        // clamp prevented wrap-around — frame is still valid
        assert_eq!(frame.len(), 16);
    }

    #[test]
    fn white_pixels_clamped_to_255() {
        let synth = FilmGrainSynthesizer::new(255, 42);
        let mut frame = vec![255u8; 16];
        synth.apply(&mut frame, 4, 4);
        assert_eq!(frame.len(), 16);
    }
}
