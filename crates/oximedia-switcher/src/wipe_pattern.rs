// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Wipe pattern generators for video transitions.
//!
//! Each function returns a *transition mask* — a `Vec<f32>` of length `w * h`
//! where each value is in `[0.0, 1.0]`.  A value of `0.0` means "show source A"
//! and `1.0` means "show source B".  The transition position `t ∈ [0.0, 1.0]`
//! drives the wipe from fully-A to fully-B.
//!
//! Pixels are stored in row-major order: `mask[y * w + x]`.

/// Wipe pattern generator.
pub struct WipePattern;

impl WipePattern {
    /// **Box wipe** — a rectangular region expands from the centre outward.
    ///
    /// At `t = 0.0` the entire frame shows source A.
    /// At `t = 1.0` the entire frame shows source B.
    ///
    /// The box grows symmetrically: the inner rectangle has half-widths
    /// `(t * w/2, t * h/2)` centred at `(w/2, h/2)`.
    pub fn box_wipe(t: f32, w: u32, h: u32) -> Vec<f32> {
        let t = t.clamp(0.0, 1.0);
        let w = w.max(1) as usize;
        let h = h.max(1) as usize;
        let mut mask = Vec::with_capacity(w * h);

        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        // At t=1 the box covers the whole frame, so half-extents are cx and cy.
        let half_w = t * cx;
        let half_h = t * cy;

        for y in 0..h {
            let fy = y as f32 + 0.5;
            let dy = (fy - cy).abs();
            for x in 0..w {
                let fx = x as f32 + 0.5;
                let dx = (fx - cx).abs();
                let inside = dx <= half_w && dy <= half_h;
                mask.push(if inside { 1.0 } else { 0.0 });
            }
        }

        mask
    }

    /// **Diagonal wipe** — source B enters from the top-left corner along a
    /// diagonal sweep.
    ///
    /// The diagonal threshold moves from `(0,0)` at `t=0` to `(w+h, w+h)` at
    /// `t=1`.  A pixel `(x, y)` transitions once `x + y < t * (w + h)`.
    pub fn diagonal_wipe(t: f32, w: u32, h: u32) -> Vec<f32> {
        let t = t.clamp(0.0, 1.0);
        let w = w.max(1) as usize;
        let h = h.max(1) as usize;
        let threshold = t * (w + h) as f32;
        let mut mask = Vec::with_capacity(w * h);

        for y in 0..h {
            let fy = y as f32 + 0.5;
            for x in 0..w {
                let fx = x as f32 + 0.5;
                let sum = fx + fy;
                let v = if sum < threshold {
                    1.0f32
                } else if sum < threshold + 1.0 {
                    // One-pixel soft edge
                    threshold + 1.0 - sum
                } else {
                    0.0
                };
                mask.push(v.clamp(0.0, 1.0));
            }
        }

        mask
    }

    /// **Iris wipe** — a circle expands from the centre of the frame.
    ///
    /// At `t = 0.0` the frame shows source A.
    /// At `t = 1.0` the entire frame shows source B (the circle fills the frame
    /// with radius equal to the half-diagonal).
    ///
    /// The mask has a one-pixel soft edge for anti-aliasing.
    pub fn iris_wipe(t: f32, w: u32, h: u32) -> Vec<f32> {
        let t = t.clamp(0.0, 1.0);
        let w = w.max(1) as usize;
        let h = h.max(1) as usize;

        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        // Maximum radius to cover the entire frame (half-diagonal)
        let max_r = ((cx * cx) + (cy * cy)).sqrt();
        let r = t * max_r;
        let feather = 1.0f32; // pixels

        let mut mask = Vec::with_capacity(w * h);

        for y in 0..h {
            let fy = y as f32 + 0.5 - cy;
            for x in 0..w {
                let fx = x as f32 + 0.5 - cx;
                let dist = (fx * fx + fy * fy).sqrt();

                let v = if dist < r - feather {
                    1.0
                } else if dist < r + feather {
                    // Soft edge: linear falloff across 2×feather pixels
                    ((r + feather - dist) / (2.0 * feather)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                mask.push(v);
            }
        }

        mask
    }

    /// **Horizontal wipe** (left → right).
    ///
    /// Source B enters from the left edge; the boundary sweeps right as `t`
    /// increases from 0 to 1.
    pub fn horizontal_wipe(t: f32, w: u32, h: u32) -> Vec<f32> {
        let t = t.clamp(0.0, 1.0);
        let w = w.max(1) as usize;
        let h = h.max(1) as usize;
        let boundary = t * w as f32;
        let mut mask = Vec::with_capacity(w * h);

        for _y in 0..h {
            for x in 0..w {
                let fx = x as f32 + 0.5;
                let v = if fx < boundary {
                    1.0f32
                } else if boundary > 0.0 && fx < boundary + 1.0 {
                    boundary + 1.0 - fx
                } else {
                    0.0
                };
                mask.push(v.clamp(0.0, 1.0));
            }
        }

        mask
    }

    /// **Vertical wipe** (top → bottom).
    ///
    /// Source B enters from the top edge; the boundary sweeps downward as `t`
    /// increases from 0 to 1.
    pub fn vertical_wipe(t: f32, w: u32, h: u32) -> Vec<f32> {
        let t = t.clamp(0.0, 1.0);
        let w = w.max(1) as usize;
        let h = h.max(1) as usize;
        let boundary = t * h as f32;
        let mut mask = Vec::with_capacity(w * h);

        for y in 0..h {
            let fy = y as f32 + 0.5;
            let v = if fy < boundary {
                1.0f32
            } else if fy < boundary + 1.0 {
                boundary + 1.0 - fy
            } else {
                0.0
            };
            for _x in 0..w {
                mask.push(v.clamp(0.0, 1.0));
            }
        }

        mask
    }

    /// Composite two RGBA frames using a wipe mask.
    ///
    /// `mask[i] = 0.0` → show `src_a`; `mask[i] = 1.0` → show `src_b`.
    ///
    /// Both frames must be RGBA-interleaved and the same length.
    /// Returns an empty `Vec` on length mismatch.
    pub fn apply_mask(src_a: &[u8], src_b: &[u8], mask: &[f32]) -> Vec<u8> {
        let num_pixels = mask.len();
        if src_a.len() != num_pixels * 4 || src_b.len() != num_pixels * 4 {
            return Vec::new();
        }

        let mut out = Vec::with_capacity(num_pixels * 4);
        for p in 0..num_pixels {
            let m = mask[p].clamp(0.0, 1.0);
            for ch in 0..4 {
                let a = src_a[p * 4 + ch] as f32;
                let b = src_b[p * 4 + ch] as f32;
                out.push((a * (1.0 - m) + b * m).round() as u8);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_wipe_length() {
        let mask = WipePattern::box_wipe(0.5, 16, 9);
        assert_eq!(mask.len(), 16 * 9);
    }

    #[test]
    fn box_wipe_all_zero_at_t0() {
        let mask = WipePattern::box_wipe(0.0, 16, 9);
        for &v in &mask {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn box_wipe_all_one_at_t1() {
        let mask = WipePattern::box_wipe(1.0, 16, 9);
        for &v in &mask {
            assert_eq!(v, 1.0);
        }
    }

    #[test]
    fn diagonal_wipe_length() {
        let mask = WipePattern::diagonal_wipe(0.5, 20, 20);
        assert_eq!(mask.len(), 20 * 20);
    }

    #[test]
    fn diagonal_wipe_at_t0_all_zero() {
        let mask = WipePattern::diagonal_wipe(0.0, 10, 10);
        for &v in &mask {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn iris_wipe_length() {
        let mask = WipePattern::iris_wipe(0.5, 32, 18);
        assert_eq!(mask.len(), 32 * 18);
    }

    #[test]
    fn iris_wipe_centre_is_one_at_t1() {
        let w = 32u32;
        let h = 18u32;
        let mask = WipePattern::iris_wipe(1.0, w, h);
        let cx = w as usize / 2;
        let cy = h as usize / 2;
        let centre_val = mask[cy * w as usize + cx];
        assert!(
            centre_val >= 0.99,
            "centre should be ~1.0 at t=1, got {centre_val}"
        );
    }

    #[test]
    fn horizontal_wipe_at_t0_all_zero() {
        let mask = WipePattern::horizontal_wipe(0.0, 8, 4);
        for &v in &mask {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn vertical_wipe_length() {
        let mask = WipePattern::vertical_wipe(0.5, 8, 4);
        assert_eq!(mask.len(), 32);
    }

    #[test]
    fn all_patterns_values_in_range() {
        let fns: Vec<fn(f32, u32, u32) -> Vec<f32>> = vec![
            WipePattern::box_wipe,
            WipePattern::diagonal_wipe,
            WipePattern::iris_wipe,
            WipePattern::horizontal_wipe,
            WipePattern::vertical_wipe,
        ];
        for f in &fns {
            for &t in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
                let mask = f(t, 16, 16);
                for &v in &mask {
                    assert!(v >= 0.0 && v <= 1.0, "v={v} out of range for t={t}");
                }
            }
        }
    }

    #[test]
    fn apply_mask_full_a() {
        let a = vec![100u8; 4];
        let b = vec![200u8; 4];
        let mask = vec![0.0f32; 1];
        let out = WipePattern::apply_mask(&a, &b, &mask);
        assert_eq!(out[0], 100);
    }

    #[test]
    fn apply_mask_full_b() {
        let a = vec![100u8; 4];
        let b = vec![200u8; 4];
        let mask = vec![1.0f32; 1];
        let out = WipePattern::apply_mask(&a, &b, &mask);
        assert_eq!(out[0], 200);
    }
}
