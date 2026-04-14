//! Motion-Compensated Temporal Filtering (MCTF).
//!
//! MCTF improves temporal noise reduction by first aligning previous and next
//! frames to the current frame via block-based motion compensation before
//! performing the temporal average.  This avoids the motion-blur artefacts
//! that arise when static temporal blending is applied to moving regions.
//!
//! # Algorithm overview
//!
//! For each `N×N` block in the *current* frame:
//!
//! 1. A full-search block-matching pass finds the best matching block in
//!    each of the supplied reference frames (up to `max_ref_frames`).
//! 2. Each reference block is motion-compensated by copying its pixels to
//!    the current block position.
//! 3. The current block and all compensated reference blocks are blended with
//!    weights that decay geometrically with temporal distance:
//!    `w_ref[i] = (1 − decay)^i`, where `i ∈ {1 … n_refs}`.
//!    The current-frame weight is 1.0.
//!    All weights are normalised so they sum to 1.
//! 4. The SAD of the best match is compared against a *motion threshold*.
//!    If the SAD per pixel exceeds the threshold the block is treated as
//!    *high-motion* and the current-frame weight is increased to `motion_boost`,
//!    reducing the contribution of reference frames and preserving sharpness.
//!
//! # Example
//!
//! ```rust
//! use oximedia_video::mctf::{MctfFilter, MctfConfig};
//!
//! let w = 32u32;
//! let h = 32u32;
//! let n = (w * h) as usize;
//! let current: Vec<u8> = (0..n).map(|i| (i % 200) as u8).collect();
//! let ref1: Vec<u8>    = (0..n).map(|i| ((i + 8) % 200) as u8).collect();
//!
//! let cfg = MctfConfig::default();
//! let filter = MctfFilter::new(cfg);
//! let filtered = filter.filter(&current, &[ref1.as_slice()], w, h);
//! assert_eq!(filtered.len(), n);
//! ```

// ---------------------------------------------------------------------------
// MctfConfig
// ---------------------------------------------------------------------------

/// Configuration for the MCTF filter.
#[derive(Debug, Clone)]
pub struct MctfConfig {
    /// Block size (pixels per side, must be ≥ 4).
    pub block_size: u32,
    /// Block-matching search radius in pixels.
    pub search_range: i32,
    /// Geometric weight decay factor per reference frame step
    /// `(0 < decay ≤ 1)`.  A higher value gives less weight to older frames.
    pub decay: f64,
    /// SAD-per-pixel threshold above which a block is classified as
    /// *high-motion*.  Typical values: 8–32.
    pub motion_threshold: f64,
    /// Weight boost applied to the current frame in high-motion blocks.
    /// Must be ≥ 1.0 (1.0 = no boost).
    pub motion_boost: f64,
    /// Maximum number of reference frames to use (≥ 1).
    pub max_ref_frames: usize,
}

impl Default for MctfConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            search_range: 8,
            decay: 0.5,
            motion_threshold: 16.0,
            motion_boost: 4.0,
            max_ref_frames: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// MctfBlockResult
// ---------------------------------------------------------------------------

/// Diagnostics for a single filtered block.
#[derive(Debug, Clone)]
pub struct MctfBlockResult {
    /// Top-left X of the block in the current frame.
    pub block_x: u32,
    /// Top-left Y of the block in the current frame.
    pub block_y: u32,
    /// `true` when the block was classified as high-motion.
    pub high_motion: bool,
    /// Minimum SAD-per-pixel among all reference matches.
    pub min_sad_per_pixel: f64,
    /// Number of reference frames actually used (may be less than configured
    /// `max_ref_frames` if fewer were supplied).
    pub refs_used: usize,
}

// ---------------------------------------------------------------------------
// MctfResult
// ---------------------------------------------------------------------------

/// Output of [`MctfFilter::filter_with_diagnostics`].
#[derive(Debug, Clone)]
pub struct MctfResult {
    /// Filtered luma frame, `width × height` bytes.
    pub filtered: Vec<u8>,
    /// Per-block diagnostics.
    pub blocks: Vec<MctfBlockResult>,
    /// Fraction of blocks classified as high-motion.
    pub high_motion_rate: f64,
}

// ---------------------------------------------------------------------------
// MctfFilter
// ---------------------------------------------------------------------------

/// Motion-compensated temporal noise filter.
#[derive(Debug, Clone)]
pub struct MctfFilter {
    cfg: MctfConfig,
}

impl MctfFilter {
    /// Create a new filter from the given configuration.
    pub fn new(cfg: MctfConfig) -> Self {
        let block_size = cfg.block_size.max(4);
        Self {
            cfg: MctfConfig { block_size, ..cfg },
        }
    }

    /// Filter `current` using `references` (ordered oldest-to-newest, most
    /// recent last).  Returns the filtered frame only.
    ///
    /// All buffers must be 8-bit luma, `width × height` bytes.
    pub fn filter(&self, current: &[u8], references: &[&[u8]], width: u32, height: u32) -> Vec<u8> {
        self.filter_with_diagnostics(current, references, width, height)
            .filtered
    }

    /// Filter `current` and return full per-block diagnostics.
    pub fn filter_with_diagnostics(
        &self,
        current: &[u8],
        references: &[&[u8]],
        width: u32,
        height: u32,
    ) -> MctfResult {
        assert_eq!(
            current.len(),
            (width * height) as usize,
            "current frame length must equal width × height"
        );
        for (i, r) in references.iter().enumerate() {
            assert_eq!(
                r.len(),
                current.len(),
                "reference frame {i} has wrong length"
            );
        }

        let n_refs = references.len().min(self.cfg.max_ref_frames);
        let bs = self.cfg.block_size;
        let mut output = current.to_vec();
        let mut block_results = Vec::new();

        let mut by = 0u32;
        while by < height {
            let bh = bs.min(height.saturating_sub(by));
            let mut bx = 0u32;
            while bx < width {
                let bw = bs.min(width.saturating_sub(bx));

                let diag = self.process_block(
                    current,
                    references,
                    &mut output,
                    width,
                    height,
                    bx,
                    by,
                    bw,
                    bh,
                    n_refs,
                );
                block_results.push(diag);

                bx += bs;
            }
            by += bs;
        }

        let hm_count = block_results.iter().filter(|b| b.high_motion).count();
        let high_motion_rate = if block_results.is_empty() {
            0.0
        } else {
            hm_count as f64 / block_results.len() as f64
        };

        MctfResult {
            filtered: output,
            blocks: block_results,
            high_motion_rate,
        }
    }

    /// Process a single block: motion-compensate all reference frames, blend,
    /// write results into `output`, return diagnostics.
    fn process_block(
        &self,
        current: &[u8],
        references: &[&[u8]],
        output: &mut Vec<u8>,
        width: u32,
        height: u32,
        bx: u32,
        by: u32,
        bw: u32,
        bh: u32,
        n_refs: usize,
    ) -> MctfBlockResult {
        let block_pixels = (bw * bh) as f64;

        // --- Find motion vectors for each reference frame ---
        let mut matched_blocks: Vec<Vec<u8>> = Vec::with_capacity(n_refs);
        let mut min_sad_per_pixel = f64::MAX;

        for ref_idx in 0..n_refs {
            let reference = references[ref_idx];
            let (mv, sad) = full_search(
                current,
                reference,
                width,
                height,
                bx,
                by,
                bw,
                bh,
                self.cfg.search_range,
            );

            let sad_per_pixel = sad as f64 / block_pixels.max(1.0);
            if sad_per_pixel < min_sad_per_pixel {
                min_sad_per_pixel = sad_per_pixel;
            }

            // Extract the displaced reference block.
            let comp = extract_displaced_block(reference, width, height, bx, by, bw, bh, mv);
            matched_blocks.push(comp);
        }

        if min_sad_per_pixel == f64::MAX {
            min_sad_per_pixel = 0.0;
        }

        let high_motion = min_sad_per_pixel > self.cfg.motion_threshold;

        // --- Build weights ---
        // w[0] = current frame weight (boosted for high-motion blocks).
        let current_weight = if high_motion {
            self.cfg.motion_boost.max(1.0)
        } else {
            1.0
        };
        let mut weights: Vec<f64> = Vec::with_capacity(1 + n_refs);
        weights.push(current_weight);
        for i in 1..=(n_refs as u32) {
            weights.push((1.0 - self.cfg.decay).powi(i as i32).max(0.0));
        }
        let total_w: f64 = weights.iter().sum();

        // --- Blend pixels ---
        for row in 0..bh {
            for col in 0..bw {
                let dst_x = bx + col;
                let dst_y = by + row;
                let local_idx = (row * bw + col) as usize;
                let frame_idx = (dst_y * width + dst_x) as usize;

                let cur_val = current.get(frame_idx).copied().unwrap_or(0) as f64;
                let mut blended = cur_val * weights[0];

                for (ri, comp) in matched_blocks.iter().enumerate() {
                    let ref_val = comp.get(local_idx).copied().unwrap_or(0) as f64;
                    blended += ref_val * weights[ri + 1];
                }

                blended /= total_w.max(1e-9);
                output[frame_idx] = blended.round().clamp(0.0, 255.0) as u8;
            }
        }

        MctfBlockResult {
            block_x: bx,
            block_y: by,
            high_motion,
            min_sad_per_pixel,
            refs_used: n_refs,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Full-search block matching: returns `(best_mv (dx, dy), best_sad)`.
fn full_search(
    current: &[u8],
    reference: &[u8],
    width: u32,
    height: u32,
    bx: u32,
    by: u32,
    bw: u32,
    bh: u32,
    search_range: i32,
) -> ((i32, i32), u32) {
    let mut best_sad = u32::MAX;
    let mut best_dx = 0i32;
    let mut best_dy = 0i32;

    for dy in -search_range..=search_range {
        for dx in -search_range..=search_range {
            let mut sad = 0u32;
            let mut valid = true;
            'outer: for row in 0..bh {
                for col in 0..bw {
                    let cx = (bx + col) as usize;
                    let cy = (by + row) as usize;
                    let cur_idx = cy * width as usize + cx;

                    let rx = clamp_coord(bx as i32 + col as i32 + dx, width) as usize;
                    let ry = clamp_coord(by as i32 + row as i32 + dy, height) as usize;
                    let ref_idx = ry * width as usize + rx;

                    let a = match current.get(cur_idx) {
                        Some(&v) => v as i32,
                        None => {
                            valid = false;
                            break 'outer;
                        }
                    };
                    let b = reference.get(ref_idx).copied().unwrap_or(0) as i32;
                    sad = sad.saturating_add((a - b).unsigned_abs());
                }
            }
            if valid && sad < best_sad {
                best_sad = sad;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    ((best_dx, best_dy), best_sad)
}

/// Extract a displaced block from `reference` at position `(bx + dx, by + dy)`
/// into a flat `bw × bh` buffer.
fn extract_displaced_block(
    reference: &[u8],
    width: u32,
    height: u32,
    bx: u32,
    by: u32,
    bw: u32,
    bh: u32,
    mv: (i32, i32),
) -> Vec<u8> {
    let mut block = Vec::with_capacity((bw * bh) as usize);
    for row in 0..bh {
        for col in 0..bw {
            let rx = clamp_coord(bx as i32 + col as i32 + mv.0, width) as usize;
            let ry = clamp_coord(by as i32 + row as i32 + mv.1, height) as usize;
            let idx = ry * width as usize + rx;
            block.push(reference.get(idx).copied().unwrap_or(0));
        }
    }
    block
}

/// Clamp a signed coordinate to `[0, dim − 1]`.
#[inline]
fn clamp_coord(v: i32, dim: u32) -> u32 {
    v.clamp(0, dim.saturating_sub(1) as i32) as u32
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, v: u8) -> Vec<u8> {
        vec![v; (w * h) as usize]
    }

    fn ramp(w: u32, h: u32) -> Vec<u8> {
        (0..(w * h) as usize).map(|i| (i % 256) as u8).collect()
    }

    fn noisy(w: u32, h: u32, base: u8, noise: i32) -> Vec<u8> {
        (0..(w * h) as usize)
            .map(|i| {
                let n = if i % 3 == 0 { noise } else { -noise };
                ((base as i32) + n).clamp(0, 255) as u8
            })
            .collect()
    }

    // --- output size ---

    #[test]
    fn test_filter_output_size() {
        let w = 32u32;
        let h = 32u32;
        let cur = ramp(w, h);
        let r1 = ramp(w, h);
        let filter = MctfFilter::new(MctfConfig::default());
        let out = filter.filter(&cur, &[r1.as_slice()], w, h);
        assert_eq!(out.len(), (w * h) as usize);
    }

    // --- identical solid frames → output unchanged ---

    #[test]
    fn test_identical_solid_frames_unchanged() {
        let w = 16u32;
        let h = 16u32;
        let cur = solid(w, h, 150);
        let refs: Vec<Vec<u8>> = vec![solid(w, h, 150); 3];
        let ref_slices: Vec<&[u8]> = refs.iter().map(|v| v.as_slice()).collect();
        let filter = MctfFilter::new(MctfConfig::default());
        let out = filter.filter(&cur, &ref_slices, w, h);
        for &v in &out {
            assert_eq!(v, 150);
        }
    }

    // --- zero reference frames → current frame returned ---

    #[test]
    fn test_no_references_returns_current() {
        let w = 16u32;
        let h = 16u32;
        let cur = ramp(w, h);
        let filter = MctfFilter::new(MctfConfig::default());
        let out = filter.filter(&cur, &[], w, h);
        // With zero refs, output should equal input (weight sum = 1 for current).
        assert_eq!(out, cur);
    }

    // --- noisy frames are smoothed towards reference ---

    #[test]
    fn test_noisy_frame_is_smoothed() {
        let w = 32u32;
        let h = 32u32;
        let clean = solid(w, h, 128);
        let noisy_frame = noisy(w, h, 128, 30);
        let refs: Vec<Vec<u8>> = vec![clean.clone(); 3];
        let ref_slices: Vec<&[u8]> = refs.iter().map(|v| v.as_slice()).collect();
        let filter = MctfFilter::new(MctfConfig {
            block_size: 16,
            search_range: 4,
            decay: 0.3,
            motion_threshold: 1000.0, // force low-motion path
            ..Default::default()
        });
        let out = filter.filter(&noisy_frame, &ref_slices, w, h);
        // Compute mean absolute error vs clean.
        let mae: f64 = out
            .iter()
            .zip(clean.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / out.len() as f64;
        let input_mae: f64 = noisy_frame
            .iter()
            .zip(clean.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / noisy_frame.len() as f64;
        assert!(
            mae < input_mae,
            "filtered MAE ({mae:.2}) should be less than noisy input MAE ({input_mae:.2})"
        );
    }

    // --- high_motion_rate is in [0, 1] ---

    #[test]
    fn test_high_motion_rate_range() {
        let w = 32u32;
        let h = 32u32;
        let cur = ramp(w, h);
        let r1 = solid(w, h, 0);
        let filter = MctfFilter::new(MctfConfig::default());
        let result = filter.filter_with_diagnostics(&cur, &[r1.as_slice()], w, h);
        assert!((0.0..=1.0).contains(&result.high_motion_rate));
    }

    // --- diagnostics block count matches expected ---

    #[test]
    fn test_diagnostics_block_count() {
        let w = 32u32;
        let h = 32u32;
        let cur = ramp(w, h);
        let refs = vec![ramp(w, h)];
        let ref_slices: Vec<&[u8]> = refs.iter().map(|v| v.as_slice()).collect();
        let filter = MctfFilter::new(MctfConfig {
            block_size: 16,
            ..Default::default()
        });
        let result = filter.filter_with_diagnostics(&cur, &ref_slices, w, h);
        // 32×32 / 16×16 = 4 blocks
        assert_eq!(result.blocks.len(), 4);
    }

    // --- max_ref_frames cap respected ---

    #[test]
    fn test_max_ref_frames_cap() {
        let w = 16u32;
        let h = 16u32;
        let cur = solid(w, h, 100);
        let refs: Vec<Vec<u8>> = (0..10).map(|_| solid(w, h, 100)).collect();
        let ref_slices: Vec<&[u8]> = refs.iter().map(|v| v.as_slice()).collect();
        let filter = MctfFilter::new(MctfConfig {
            block_size: 16,
            max_ref_frames: 3,
            ..Default::default()
        });
        let result = filter.filter_with_diagnostics(&cur, &ref_slices, w, h);
        for blk in &result.blocks {
            assert!(
                blk.refs_used <= 3,
                "refs_used {} exceeds max_ref_frames 3",
                blk.refs_used
            );
        }
    }

    // --- motion_boost increases current-frame weight for moving blocks ---

    #[test]
    fn test_motion_boost_preserves_sharpness() {
        // current = 200, reference = 0. High motion → should stay close to 200.
        let w = 16u32;
        let h = 16u32;
        let cur = solid(w, h, 200);
        let r1 = solid(w, h, 0);
        let filter = MctfFilter::new(MctfConfig {
            block_size: 16,
            search_range: 4,
            motion_threshold: 0.0, // force high-motion path
            motion_boost: 10.0,
            decay: 0.5,
            max_ref_frames: 1,
        });
        let out = filter.filter(&cur, &[r1.as_slice()], w, h);
        let mean: f64 = out.iter().map(|&v| v as f64).sum::<f64>() / out.len() as f64;
        // With boost=10, current has weight 10 vs reference weight ~0.5 → mean ≈ 190
        assert!(
            mean > 150.0,
            "high-motion boost should keep output close to current frame (mean={mean:.1})"
        );
    }

    // --- clamp_coord helper ---

    #[test]
    fn test_clamp_coord_helper() {
        assert_eq!(clamp_coord(-1, 10), 0);
        assert_eq!(clamp_coord(10, 10), 9);
        assert_eq!(clamp_coord(5, 10), 5);
    }

    // --- output pixel values stay in [0, 255] ---

    #[test]
    fn test_output_values_in_range() {
        let w = 32u32;
        let h = 32u32;
        let cur = ramp(w, h);
        let refs: Vec<Vec<u8>> = (0..4).map(|k| solid(w, h, (k * 50) as u8)).collect();
        let ref_slices: Vec<&[u8]> = refs.iter().map(|v| v.as_slice()).collect();
        let filter = MctfFilter::new(MctfConfig::default());
        let out = filter.filter(&cur, &ref_slices, w, h);
        // All pixel values are u8, so they are always in [0, 255].
        // We just verify the buffer has the expected number of elements.
        assert_eq!(out.len(), (w * h) as usize, "output length mismatch");
    }
}
