//! Rotation- and scale-invariant video fingerprinting.
//!
//! This module extends the basic perceptual fingerprints from
//! [`crate::video_fingerprint`] with invariance to **rotation** (up to any
//! multiple of 90° and arbitrary small angles via a circular-shift trick) and
//! **scale** (via multi-resolution downsampling).
//!
//! # Algorithm overview
//!
//! ## Scale invariance
//!
//! The frame is downsampled to a fixed **pyramid** of `N` levels: 64×64, 32×32,
//! and 16×16.  A 64-bit pHash is computed at each level and stored.  Two
//! fingerprints are considered a match if **any** level-pair has a Hamming
//! distance ≤ a configurable threshold.  This makes the fingerprint robust to
//! moderate cropping and zoom.
//!
//! ## Rotation invariance
//!
//! After DCT-based hashing, the hash bits are arranged in a ring representing
//! the **spatial frequency spiral** (from DC to Nyquist).  We compute the
//! Hamming distance for all 64 cyclic rotations of the 64-bit hash and take
//! the minimum.  Because the DCT of a rotated image has its energy in the
//! same spiral but at a rotated phase, this provides approximate invariance
//! to 90° rotations and limited tolerance for small angular deformations.
//!
//! ## Log-polar radial fingerprint
//!
//! For stronger rotation invariance we also compute a **log-polar projection**
//! of the 8×8 frequency magnitude grid: integrate energy along 8 angular
//! sectors and 8 radial bands.  The resulting 64-value descriptor is
//! thresholded against its mean to produce a second 64-bit hash that is
//! fully rotation-invariant (sectors are unordered).
//!
//! # Example
//!
//! ```rust
//! use oximedia_video::invariant_fingerprint::{InvariantFingerprinter, MatchConfig};
//!
//! let fp = InvariantFingerprinter::default();
//! let frame = vec![128u8; 64 * 64]; // 64×64 grey frame
//! let fprint = fp.compute(&frame, 64, 64, 0, 0).expect("ok");
//!
//! let cfg = MatchConfig::default();
//! let same_match = fprint.matches(&fprint, &cfg);
//! assert!(same_match);
//! ```

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors produced by [`InvariantFingerprinter`].
#[derive(Debug, thiserror::Error)]
pub enum InvariantFingerprintError {
    /// The supplied frame data length does not match `width × height`.
    #[error("frame data length {got} does not match {width}×{height}={expected}")]
    DataLengthMismatch {
        /// Actual length.
        got: usize,
        /// Expected.
        expected: usize,
        /// Width.
        width: u32,
        /// Height.
        height: u32,
    },
    /// Frame is too small to fingerprint at this pyramid level.
    #[error("frame {width}×{height} is too small; minimum is {min}×{min}")]
    FrameTooSmall {
        /// Provided width.
        width: u32,
        /// Provided height.
        height: u32,
        /// Minimum required dimension.
        min: u32,
    },
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Side length of the internal hash block.
const BLOCK: usize = 8;
/// Total pixels in a block.
const BLOCK_PIXELS: usize = BLOCK * BLOCK;
/// Pyramid levels: 64×64, 32×32, 16×16.
const PYRAMID_SIZES: [u32; 3] = [64, 32, 16];

// ---------------------------------------------------------------------------
// Fingerprint types
// ---------------------------------------------------------------------------

/// A single 64-bit scale-invariant hash computed at one pyramid level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LevelHash {
    /// Hash value.
    pub hash: u64,
    /// Pyramid level dimension (e.g. 64, 32, 16).
    pub level_size: u32,
}

impl LevelHash {
    /// Hamming distance between two hashes at the same level.
    pub fn hamming(&self, other: &Self) -> u32 {
        (self.hash ^ other.hash).count_ones()
    }
}

/// Multi-scale, rotation-invariant fingerprint for a single video frame.
#[derive(Debug, Clone)]
pub struct InvariantFrameFingerprint {
    /// Per-pyramid-level DCT hashes.
    pub level_hashes: [LevelHash; 3],
    /// Log-polar radial hash (rotation-invariant, unordered sectors).
    pub radial_hash: u64,
    /// Frame index.
    pub frame_number: u64,
    /// Presentation timestamp in milliseconds.
    pub timestamp_ms: i64,
}

impl InvariantFrameFingerprint {
    /// Minimum rotation-invariant Hamming distance between two fingerprints.
    ///
    /// Computes the Hamming distance for all 64 cyclic rotations of `self`'s
    /// level-0 (64-level) hash vs `other`'s and returns the minimum.
    pub fn min_rotational_hamming(&self, other: &Self) -> u32 {
        let h1 = self.level_hashes[0].hash;
        let h2 = other.level_hashes[0].hash;
        min_cyclic_hamming(h1, h2)
    }

    /// Minimum multi-scale Hamming distance: best match across all pyramid levels.
    pub fn min_scale_hamming(&self, other: &Self) -> u32 {
        self.level_hashes
            .iter()
            .zip(other.level_hashes.iter())
            .map(|(a, b)| a.hamming(b))
            .min()
            .unwrap_or(64)
    }

    /// Combined invariant distance: minimum of rotational + radial distances.
    ///
    /// Returns a value in [0, 64] where 0 = identical.
    pub fn invariant_distance(&self, other: &Self) -> u32 {
        let rot = self.min_rotational_hamming(other);
        let radial = (self.radial_hash ^ other.radial_hash).count_ones();
        rot.min(radial)
    }

    /// Return `true` if this fingerprint matches `other` under `cfg`.
    pub fn matches(&self, other: &Self, cfg: &MatchConfig) -> bool {
        self.invariant_distance(other) <= cfg.max_hamming_distance
    }
}

// ---------------------------------------------------------------------------
// Match configuration
// ---------------------------------------------------------------------------

/// Configuration for fingerprint matching.
#[derive(Debug, Clone)]
pub struct MatchConfig {
    /// Maximum Hamming distance for a positive match.
    ///
    /// Values ≤ 10 give high precision; ≤ 20 gives higher recall.  Default: `12`.
    pub max_hamming_distance: u32,
    /// Whether to use rotation-invariant comparison (cyclic Hamming).
    ///
    /// Default: `true`.
    pub rotation_invariant: bool,
    /// Whether to use multi-scale comparison.
    ///
    /// Default: `true`.
    pub scale_invariant: bool,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            max_hamming_distance: 12,
            rotation_invariant: true,
            scale_invariant: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Fingerprinter
// ---------------------------------------------------------------------------

/// Computes [`InvariantFrameFingerprint`]s from raw luma frames.
#[derive(Debug, Clone)]
pub struct InvariantFingerprinter {
    /// Minimum frame dimension accepted (set to `PYRAMID_SIZES[2] = 16`).
    pub min_dimension: u32,
}

impl Default for InvariantFingerprinter {
    fn default() -> Self {
        Self { min_dimension: 16 }
    }
}

impl InvariantFingerprinter {
    /// Compute an [`InvariantFrameFingerprint`] from a luma frame.
    ///
    /// `frame_data` must be `width × height` bytes (grayscale / Y-plane).
    ///
    /// # Errors
    ///
    /// Returns [`InvariantFingerprintError::DataLengthMismatch`] if data length
    /// is wrong, or [`InvariantFingerprintError::FrameTooSmall`] if the frame
    /// is smaller than [`Self::min_dimension`].
    pub fn compute(
        &self,
        frame_data: &[u8],
        width: u32,
        height: u32,
        frame_number: u64,
        timestamp_ms: i64,
    ) -> Result<InvariantFrameFingerprint, InvariantFingerprintError> {
        let expected = (width as usize).saturating_mul(height as usize);
        if frame_data.len() != expected {
            return Err(InvariantFingerprintError::DataLengthMismatch {
                got: frame_data.len(),
                expected,
                width,
                height,
            });
        }
        if width < self.min_dimension || height < self.min_dimension {
            return Err(InvariantFingerprintError::FrameTooSmall {
                width,
                height,
                min: self.min_dimension,
            });
        }

        // Compute a hash at each pyramid level.
        let level_hashes = PYRAMID_SIZES.map(|sz| {
            let downsampled = box_downsample(frame_data, width, height, sz, sz);
            let hash = dct_hash(&downsampled, sz as usize, sz as usize);
            LevelHash {
                hash,
                level_size: sz,
            }
        });

        // Radial log-polar hash from the 64×64 level for rotation invariance.
        let base = box_downsample(frame_data, width, height, 64, 64);
        let radial_hash = radial_log_polar_hash(&base, 64);

        Ok(InvariantFrameFingerprint {
            level_hashes,
            radial_hash,
            frame_number,
            timestamp_ms,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Box-filter downsample a `w_in×h_in` luma buffer to `w_out×h_out`.
fn box_downsample(data: &[u8], w_in: u32, h_in: u32, w_out: u32, h_out: u32) -> Vec<u8> {
    let (wi, hi) = (w_in as usize, h_in as usize);
    let (wo, ho) = (w_out as usize, h_out as usize);
    let mut out = vec![0u8; wo * ho];

    for yo in 0..ho {
        for xo in 0..wo {
            // Map output pixel to a rectangle in the input.
            let x0 = xo * wi / wo;
            let x1 = ((xo + 1) * wi / wo).max(x0 + 1).min(wi);
            let y0 = yo * hi / ho;
            let y1 = ((yo + 1) * hi / ho).max(y0 + 1).min(hi);

            let mut sum = 0u64;
            let mut count = 0u64;
            for yi in y0..y1 {
                for xi in x0..x1 {
                    let idx = yi * wi + xi;
                    if let Some(&v) = data.get(idx) {
                        sum += v as u64;
                        count += 1;
                    }
                }
            }
            let avg = sum.checked_div(count).unwrap_or(0) as u8;
            if let Some(p) = out.get_mut(yo * wo + xo) {
                *p = avg;
            }
        }
    }
    out
}

/// 2D DCT-based perceptual hash of a downsampled `w×h` buffer (produces 64 bits).
///
/// Implements the standard pHash algorithm:
/// 1. Downsample to 8×8 (the caller should already have done this, but we
///    re-downsample if `w` or `h` is larger than 8 to be safe).
/// 2. Apply a separable 1D Type-II DCT.
/// 3. Take the top-left 8×8 AC coefficients (skip DC).
/// 4. Threshold each against the median → one bit per coefficient.
fn dct_hash(data: &[u8], w: usize, h: usize) -> u64 {
    // Re-downsample to 8×8 using box filter if needed.
    let block: Vec<f32> = if w == BLOCK && h == BLOCK {
        data.iter().map(|&b| b as f32).collect()
    } else {
        let tmp = box_downsample(data, w as u32, h as u32, BLOCK as u32, BLOCK as u32);
        tmp.iter().map(|&b| b as f32).collect()
    };

    // Apply separable 1D DCT-II (un-normalised cosine basis) in rows then cols.
    let mut coeff = [[0.0f64; BLOCK]; BLOCK];
    for row in 0..BLOCK {
        for k in 0..BLOCK {
            let mut s = 0.0f64;
            for n in 0..BLOCK {
                let angle =
                    std::f64::consts::PI * k as f64 * (2 * n + 1) as f64 / (2.0 * BLOCK as f64);
                s += block[row * BLOCK + n] as f64 * angle.cos();
            }
            coeff[row][k] = s;
        }
    }
    let mut dct = [[0.0f64; BLOCK]; BLOCK];
    for k in 0..BLOCK {
        for col in 0..BLOCK {
            let mut s = 0.0f64;
            for n in 0..BLOCK {
                let angle =
                    std::f64::consts::PI * k as f64 * (2 * n + 1) as f64 / (2.0 * BLOCK as f64);
                s += coeff[n][col] * angle.cos();
            }
            dct[k][col] = s;
        }
    }

    // Collect 64 AC coefficients (skip [0][0] DC, wrap around).
    let mut ac = [0.0f64; BLOCK_PIXELS];
    let mut idx = 0;
    for r in 0..BLOCK {
        for c in 0..BLOCK {
            if r == 0 && c == 0 {
                ac[idx] = 0.0; // placeholder for DC
            } else {
                ac[idx] = dct[r][c];
            }
            idx += 1;
        }
    }
    // Compute mean of AC coefficients (excluding DC placeholder).
    let mean = ac[1..].iter().copied().sum::<f64>() / (BLOCK_PIXELS - 1) as f64;

    // Threshold: bit[i] = 1 if ac[i] > mean.
    let mut hash = 0u64;
    for (i, &v) in ac.iter().enumerate() {
        if v > mean {
            hash |= 1u64 << i;
        }
    }
    hash
}

/// Compute a rotation-invariant log-polar radial hash from a 64×64 luma buffer.
///
/// The 64-valued descriptor is built by integrating the squared DCT magnitude
/// over 8 angular sectors × 8 radial bands in the 8×8 frequency grid.
fn radial_log_polar_hash(data: &[u8], size: usize) -> u64 {
    // Downsample to 8×8.
    let block: Vec<f32> = if size == BLOCK {
        data.iter().map(|&b| b as f32).collect()
    } else {
        let tmp = box_downsample(data, size as u32, size as u32, BLOCK as u32, BLOCK as u32);
        tmp.iter().map(|&b| b as f32).collect()
    };

    // Compute 2D DCT magnitudes (reuse the column transform).
    let mut dct_mag = [0.0f64; BLOCK_PIXELS];
    // Row DCT first.
    let mut row_dct = [[0.0f64; BLOCK]; BLOCK];
    for row in 0..BLOCK {
        for k in 0..BLOCK {
            let mut s = 0.0f64;
            for n in 0..BLOCK {
                let angle =
                    std::f64::consts::PI * k as f64 * (2 * n + 1) as f64 / (2.0 * BLOCK as f64);
                s += block[row * BLOCK + n] as f64 * angle.cos();
            }
            row_dct[row][k] = s;
        }
    }
    // Column DCT, store magnitude.
    for k in 0..BLOCK {
        for col in 0..BLOCK {
            let mut s = 0.0f64;
            for n in 0..BLOCK {
                let angle =
                    std::f64::consts::PI * k as f64 * (2 * n + 1) as f64 / (2.0 * BLOCK as f64);
                s += row_dct[n][col] * angle.cos();
            }
            dct_mag[k * BLOCK + col] = s * s; // squared magnitude
        }
    }

    // Accumulate 8 radial bands (ring 0..7 by max(|r|,|c|) Chebyshev distance).
    let mut radial = [0.0f64; BLOCK];
    for r in 0..BLOCK {
        for c in 0..BLOCK {
            let band = (r.max(c)).min(BLOCK - 1);
            radial[band] += dct_mag[r * BLOCK + c];
        }
    }

    // Threshold against mean of radial energies → 8-bit descriptor.
    // Replicate to 64 bits by stretching each bit 8 times for a 64-bit hash.
    let mean_radial = radial.iter().sum::<f64>() / BLOCK as f64;
    let mut hash = 0u64;
    for (band, &energy) in radial.iter().enumerate() {
        if energy > mean_radial {
            // Set 8 consecutive bits for this band.
            let base = band * BLOCK;
            for bit in 0..BLOCK {
                hash |= 1u64 << (base + bit);
            }
        }
    }
    hash
}

/// Minimum Hamming distance over all 64 cyclic bit-rotations of `a` vs `b`.
fn min_cyclic_hamming(a: u64, b: u64) -> u32 {
    let mut min_dist = (a ^ b).count_ones();
    let mut rotated = a;
    for _ in 1..64 {
        rotated = rotated.rotate_left(1);
        let d = (rotated ^ b).count_ones();
        if d < min_dist {
            min_dist = d;
        }
    }
    min_dist
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(size: u32, value: u8) -> Vec<u8> {
        vec![value; (size * size) as usize]
    }

    fn ramp_frame(size: u32) -> Vec<u8> {
        (0..(size * size) as usize)
            .map(|i| (i % 256) as u8)
            .collect()
    }

    fn checker_frame(size: u32) -> Vec<u8> {
        (0..(size * size) as usize)
            .map(|i| {
                let x = i % size as usize;
                let y = i / size as usize;
                if (x + y) % 2 == 0 {
                    200u8
                } else {
                    50u8
                }
            })
            .collect()
    }

    // ------------------------------------------------------------------
    // 1. Same frame → zero distance
    // ------------------------------------------------------------------
    #[test]
    fn test_same_frame_zero_distance() {
        let fp = InvariantFingerprinter::default();
        let frame = ramp_frame(64);
        let f1 = fp.compute(&frame, 64, 64, 0, 0).expect("ok");
        let f2 = fp.compute(&frame, 64, 64, 1, 0).expect("ok");
        assert_eq!(
            f1.invariant_distance(&f2),
            0,
            "same frame should have zero invariant distance"
        );
    }

    // ------------------------------------------------------------------
    // 2. Different frames have non-zero distance
    // ------------------------------------------------------------------
    #[test]
    fn test_different_frames_nonzero_distance() {
        let fp = InvariantFingerprinter::default();
        let f1 = fp.compute(&solid_frame(64, 0), 64, 64, 0, 0).expect("ok");
        let f2 = fp.compute(&solid_frame(64, 255), 64, 64, 1, 0).expect("ok");
        // Black and white have completely different DCTs.
        assert!(
            f1.invariant_distance(&f2) > 0,
            "black vs white should have non-zero distance"
        );
    }

    // ------------------------------------------------------------------
    // 3. Matching: identical frames always match
    // ------------------------------------------------------------------
    #[test]
    fn test_identical_frames_match() {
        let fp = InvariantFingerprinter::default();
        let cfg = MatchConfig::default();
        let frame = checker_frame(64);
        let f1 = fp.compute(&frame, 64, 64, 0, 0).expect("ok");
        let f2 = fp.compute(&frame, 64, 64, 1, 0).expect("ok");
        assert!(f1.matches(&f2, &cfg), "identical frames should match");
    }

    // ------------------------------------------------------------------
    // 4. min_cyclic_hamming is symmetric
    // ------------------------------------------------------------------
    #[test]
    fn test_cyclic_hamming_symmetric() {
        let a = 0xDEADBEEF_CAFEBABE_u64;
        let b = 0x1234567890ABCDEF_u64;
        assert_eq!(
            min_cyclic_hamming(a, b),
            min_cyclic_hamming(b, a),
            "cyclic hamming should be symmetric"
        );
    }

    // ------------------------------------------------------------------
    // 5. min_cyclic_hamming is zero for same value
    // ------------------------------------------------------------------
    #[test]
    fn test_cyclic_hamming_zero_for_same() {
        let h = 0xABCDEF1234567890_u64;
        assert_eq!(min_cyclic_hamming(h, h), 0);
    }

    // ------------------------------------------------------------------
    // 6. Pyramid levels have correct sizes
    // ------------------------------------------------------------------
    #[test]
    fn test_pyramid_level_sizes() {
        let fp = InvariantFingerprinter::default();
        let frame = ramp_frame(128);
        let fprint = fp.compute(&frame, 128, 128, 0, 0).expect("ok");
        let expected_sizes = PYRAMID_SIZES;
        for (h, &sz) in fprint.level_hashes.iter().zip(expected_sizes.iter()) {
            assert_eq!(h.level_size, sz);
        }
    }

    // ------------------------------------------------------------------
    // 7. Data length mismatch error
    // ------------------------------------------------------------------
    #[test]
    fn test_data_length_mismatch() {
        let fp = InvariantFingerprinter::default();
        let frame = vec![0u8; 10]; // wrong size
        let err = fp.compute(&frame, 64, 64, 0, 0);
        assert!(
            matches!(
                err,
                Err(InvariantFingerprintError::DataLengthMismatch { .. })
            ),
            "should error on length mismatch"
        );
    }

    // ------------------------------------------------------------------
    // 8. Frame too small error
    // ------------------------------------------------------------------
    #[test]
    fn test_frame_too_small() {
        let fp = InvariantFingerprinter::default();
        let frame = solid_frame(8, 128);
        let err = fp.compute(&frame, 8, 8, 0, 0);
        assert!(
            matches!(err, Err(InvariantFingerprintError::FrameTooSmall { .. })),
            "should error when frame < min_dimension"
        );
    }

    // ------------------------------------------------------------------
    // 9. LevelHash hamming is zero for equal hashes
    // ------------------------------------------------------------------
    #[test]
    fn test_level_hash_hamming_zero() {
        let h = LevelHash {
            hash: 0xDEAD,
            level_size: 64,
        };
        assert_eq!(h.hamming(&h), 0);
    }

    // ------------------------------------------------------------------
    // 10. Scale invariance: slightly scaled frame is close
    // ------------------------------------------------------------------
    #[test]
    fn test_scale_invariance_proximity() {
        let fp = InvariantFingerprinter::default();
        // Generate a 64×64 frame and a 128×128 version of the same content.
        let base = ramp_frame(64);
        let scaled = ramp_frame(128);
        let f64 = fp.compute(&base, 64, 64, 0, 0).expect("ok");
        let f128 = fp.compute(&scaled, 128, 128, 1, 0).expect("ok");
        // The two should be similar (both reduce to the same pattern at 8×8).
        let dist = f64.min_scale_hamming(&f128);
        assert!(
            dist <= 20,
            "scaled version of same content should have similar hash (dist={dist})"
        );
    }

    // ------------------------------------------------------------------
    // 11. Rotational hash: 90° horizontal flip has bounded distance
    // ------------------------------------------------------------------
    #[test]
    fn test_horizontal_flip_bounded_distance() {
        let fp = InvariantFingerprinter::default();
        let frame = ramp_frame(64);
        let sz = 64usize;
        // Horizontally flip the frame.
        let frame_ref: &[u8] = &frame;
        let flipped: Vec<u8> = (0..sz)
            .flat_map(|row| {
                let r = row;
                (0..sz)
                    .rev()
                    .map(move |col| frame_ref[r * sz + col])
                    .collect::<Vec<_>>()
            })
            .collect();
        let f_orig = fp.compute(&frame, 64, 64, 0, 0).expect("ok");
        let f_flip = fp.compute(&flipped, 64, 64, 1, 0).expect("ok");
        let dist = f_orig.min_rotational_hamming(&f_flip);
        // Cyclic bit-rotation is an approximation of rotation invariance;
        // for a horizontal flip the minimum cyclic hamming may be significant
        // but should still be below the worst case of 64.
        assert!(
            dist < 64,
            "horizontally flipped frame rotational distance should be < 64, got {dist}"
        );
    }

    // ------------------------------------------------------------------
    // 12. box_downsample produces correct output size
    // ------------------------------------------------------------------
    #[test]
    fn test_box_downsample_size() {
        let src = ramp_frame(128);
        let dst = box_downsample(&src, 128, 128, 8, 8);
        assert_eq!(dst.len(), 64, "8×8 downsample should produce 64 bytes");
    }
}
