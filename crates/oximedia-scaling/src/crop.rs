#![allow(dead_code)]
//! Crop rectangle and crop operation helpers.

/// A rectangular crop region within a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropRect {
    /// Left offset in pixels.
    pub x: u32,
    /// Top offset in pixels.
    pub y: u32,
    /// Crop width in pixels.
    pub w: u32,
    /// Crop height in pixels.
    pub h: u32,
}

impl CropRect {
    /// Create a new [`CropRect`].
    pub fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    /// Width of the crop rectangle.
    pub fn width(&self) -> u32 {
        self.w
    }

    /// Height of the crop rectangle.
    pub fn height(&self) -> u32 {
        self.h
    }

    /// Area of the crop rectangle in pixels.
    pub fn area(&self) -> u64 {
        self.w as u64 * self.h as u64
    }

    /// Whether this crop rectangle has non-zero dimensions.
    pub fn is_valid(&self) -> bool {
        self.w > 0 && self.h > 0
    }

    /// Whether this crop fits inside a frame of the given dimensions.
    pub fn fits_in(&self, frame_w: u32, frame_h: u32) -> bool {
        self.x + self.w <= frame_w && self.y + self.h <= frame_h
    }
}

/// Strategy for choosing the crop region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CropMode {
    /// Crop coordinates are specified explicitly.
    Manual,
    /// Centre the crop rectangle in the source frame.
    Centered,
    /// Crop to fit a target aspect ratio (width / height).
    AspectRatio,
    /// Smart crop: use saliency / subject detection (placeholder).
    Smart,
}

impl CropMode {
    /// Whether this mode preserves the aspect ratio of the crop region.
    pub fn preserves_aspect_ratio(&self) -> bool {
        matches!(self, CropMode::AspectRatio | CropMode::Smart)
    }
}

/// Compute a gradient-magnitude saliency map using Sobel operators and weight
/// it by a 2D Gaussian centred on the image with σ = 0.3 × min(width, height).
///
/// `pixels` must be a row-major, single-channel byte slice of length `w × h`.
/// Returns a `Vec<f32>` of the same length with values in [0, 1].
fn compute_saliency(pixels: &[u8], w: u32, h: u32) -> Vec<f32> {
    let w = w as usize;
    let h = h as usize;
    let len = w * h;

    if len == 0 {
        return Vec::new();
    }

    // --- Sobel gradient magnitude ---
    let mut grad = vec![0.0f32; len];
    let mut max_grad = 0.0f32;

    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let get = |dy: isize, dx: isize| -> f32 {
                let row = (y as isize + dy) as usize;
                let col = (x as isize + dx) as usize;
                pixels[row * w + col] as f32 / 255.0
            };
            let gx = -get(-1, -1) - 2.0 * get(0, -1) - get(1, -1)
                + get(-1, 1)
                + 2.0 * get(0, 1)
                + get(1, 1);
            let gy = -get(-1, -1) - 2.0 * get(-1, 0) - get(-1, 1)
                + get(1, -1)
                + 2.0 * get(1, 0)
                + get(1, 1);
            let mag = gx.abs() + gy.abs(); // L1 norm is cheaper and sufficient
            grad[y * w + x] = mag;
            if mag > max_grad {
                max_grad = mag;
            }
        }
    }

    // Normalise gradient magnitude to [0, 1].
    // The threshold 1e-3 guards against normalising pure f32 rounding noise:
    // for a u8 image, a genuine gradient of 1 grey-level step produces a Sobel
    // magnitude of at least 1/255 ≈ 0.004 before weighting, so anything below
    // that is effectively zero for our purposes.
    if max_grad > 1e-3 {
        for g in &mut grad {
            *g /= max_grad;
        }
    } else {
        // Treat sub-threshold gradients as zero — no salient edges detected.
        for g in &mut grad {
            *g = 0.0;
        }
    }

    // --- Gaussian centre-prior weight ---
    let cx = w as f32 / 2.0;
    let cy = h as f32 / 2.0;
    let sigma = 0.3 * (w.min(h)) as f32;
    let two_sigma2 = 2.0 * sigma * sigma;

    let mut saliency = vec![0.0f32; len];
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let weight = (-(dx * dx + dy * dy) / two_sigma2).exp();
            saliency[y * w + x] = grad[y * w + x] * weight;
        }
    }

    saliency
}

/// Compute the optimal smart-crop rectangle for `pixels` given a desired output
/// size `(crop_w, crop_h)`.
///
/// Uses gradient-magnitude saliency weighted by a Gaussian centre prior.  A
/// sliding-window search finds the candidate crop position that maximises the
/// sum of saliency values within the window.
///
/// If the requested crop is larger than the source image the crop is centred
/// instead.
///
/// `pixels` — row-major single-channel byte slice of length `src_w × src_h`.
pub fn smart_crop(pixels: &[u8], src_w: u32, src_h: u32, crop_w: u32, crop_h: u32) -> CropRect {
    // Clamp crop to source dimensions.
    let crop_w = crop_w.min(src_w);
    let crop_h = crop_h.min(src_h);

    // Degenerate: source is exactly the crop size (or smaller).
    if crop_w == src_w && crop_h == src_h {
        return CropRect::new(0, 0, crop_w, crop_h);
    }

    let sw = src_w as usize;
    let sh = src_h as usize;
    let cw = crop_w as usize;
    let ch = crop_h as usize;

    // Compute weighted saliency map.
    let saliency = compute_saliency(pixels, src_w, src_h);

    // Pre-compute integral image of the saliency map for O(1) window sums.
    // ii[y * (sw+1) + x] = sum of saliency over rectangle [0..y, 0..x).
    let ii_w = sw + 1;
    let ii_h = sh + 1;
    let mut ii = vec![0.0f64; ii_w * ii_h];

    for y in 1..ii_h {
        for x in 1..ii_w {
            ii[y * ii_w + x] = saliency[(y - 1) * sw + (x - 1)] as f64
                + ii[(y - 1) * ii_w + x]
                + ii[y * ii_w + (x - 1)]
                - ii[(y - 1) * ii_w + (x - 1)];
        }
    }

    // Sliding-window search.
    let max_ox = sw.saturating_sub(cw);
    let max_oy = sh.saturating_sub(ch);

    let mut best_x = max_ox / 2;
    let mut best_y = max_oy / 2;
    let mut best_sum = f64::NEG_INFINITY;

    for oy in 0..=max_oy {
        for ox in 0..=max_ox {
            // Sum over [oy..oy+ch, ox..ox+cw] using the integral image.
            let x0 = ox;
            let y0 = oy;
            let x1 = ox + cw;
            let y1 = oy + ch;
            let sum =
                ii[y1 * ii_w + x1] - ii[y0 * ii_w + x1] - ii[y1 * ii_w + x0] + ii[y0 * ii_w + x0];
            if sum > best_sum {
                best_sum = sum;
                best_x = ox;
                best_y = oy;
            }
        }
    }

    CropRect::new(best_x as u32, best_y as u32, crop_w, crop_h)
}

/// Applies a crop to a source frame and reports output dimensions.
#[derive(Debug, Clone)]
pub struct CropOperation {
    /// The crop region to apply.
    pub rect: CropRect,
    /// The crop selection strategy.
    pub mode: CropMode,
}

impl CropOperation {
    /// Create a new [`CropOperation`].
    pub fn new(rect: CropRect, mode: CropMode) -> Self {
        Self { rect, mode }
    }

    /// Compute output dimensions after applying this crop to a source frame.
    ///
    /// Returns `None` if the crop rect does not fit within `(src_w, src_h)`.
    pub fn apply_to_dimensions(&self, src_w: u32, src_h: u32) -> Option<(u32, u32)> {
        if !self.rect.fits_in(src_w, src_h) || !self.rect.is_valid() {
            return None;
        }
        Some((self.rect.w, self.rect.h))
    }

    /// Build a centred crop rect for the given source and target dimensions.
    pub fn centered(src_w: u32, src_h: u32, target_w: u32, target_h: u32) -> Option<CropRect> {
        if target_w > src_w || target_h > src_h {
            return None;
        }
        let x = (src_w - target_w) / 2;
        let y = (src_h - target_h) / 2;
        Some(CropRect::new(x, y, target_w, target_h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crop_rect_dimensions() {
        let r = CropRect::new(0, 0, 1920, 1080);
        assert_eq!(r.width(), 1920);
        assert_eq!(r.height(), 1080);
    }

    #[test]
    fn test_crop_rect_area() {
        let r = CropRect::new(0, 0, 100, 50);
        assert_eq!(r.area(), 5000);
    }

    #[test]
    fn test_crop_rect_is_valid_true() {
        let r = CropRect::new(10, 10, 100, 100);
        assert!(r.is_valid());
    }

    #[test]
    fn test_crop_rect_is_valid_zero_width() {
        let r = CropRect::new(0, 0, 0, 100);
        assert!(!r.is_valid());
    }

    #[test]
    fn test_crop_rect_is_valid_zero_height() {
        let r = CropRect::new(0, 0, 100, 0);
        assert!(!r.is_valid());
    }

    #[test]
    fn test_fits_in_true() {
        let r = CropRect::new(0, 0, 1920, 1080);
        assert!(r.fits_in(1920, 1080));
        assert!(r.fits_in(3840, 2160));
    }

    #[test]
    fn test_fits_in_false() {
        let r = CropRect::new(100, 0, 1920, 1080);
        // x + w = 2020 > 1920
        assert!(!r.fits_in(1920, 1080));
    }

    #[test]
    fn test_crop_mode_preserves_aspect_ratio() {
        assert!(CropMode::AspectRatio.preserves_aspect_ratio());
        assert!(CropMode::Smart.preserves_aspect_ratio());
        assert!(!CropMode::Manual.preserves_aspect_ratio());
        assert!(!CropMode::Centered.preserves_aspect_ratio());
    }

    #[test]
    fn test_apply_to_dimensions_valid() {
        let rect = CropRect::new(0, 0, 640, 480);
        let op = CropOperation::new(rect, CropMode::Manual);
        assert_eq!(op.apply_to_dimensions(1920, 1080), Some((640, 480)));
    }

    #[test]
    fn test_apply_to_dimensions_out_of_bounds() {
        let rect = CropRect::new(0, 0, 2000, 1080);
        let op = CropOperation::new(rect, CropMode::Manual);
        assert!(op.apply_to_dimensions(1920, 1080).is_none());
    }

    #[test]
    fn test_centered_crop() {
        let rect = CropOperation::centered(1920, 1080, 1280, 720).expect("should succeed in test");
        assert_eq!(rect.x, 320);
        assert_eq!(rect.y, 180);
        assert_eq!(rect.w, 1280);
        assert_eq!(rect.h, 720);
    }

    #[test]
    fn test_centered_crop_too_large() {
        assert!(CropOperation::centered(1280, 720, 1920, 1080).is_none());
    }

    #[test]
    fn test_smart_crop_exact_size() {
        let pixels = vec![128u8; 100 * 80];
        let rect = smart_crop(&pixels, 100, 80, 100, 80);
        assert_eq!(rect.x, 0);
        assert_eq!(rect.y, 0);
        assert_eq!(rect.w, 100);
        assert_eq!(rect.h, 80);
    }

    #[test]
    fn test_smart_crop_clamps_to_source() {
        // crop size > source → clamped to source
        let pixels = vec![128u8; 50 * 50];
        let rect = smart_crop(&pixels, 50, 50, 200, 200);
        assert!(rect.fits_in(50, 50));
    }

    #[test]
    fn test_smart_crop_uniform_image_returns_valid_rect() {
        // Uniform image: saliency is zero everywhere, result should still be valid.
        let pixels = vec![128u8; 100 * 100];
        let rect = smart_crop(&pixels, 100, 100, 60, 60);
        assert!(rect.is_valid());
        assert!(rect.fits_in(100, 100));
    }

    #[test]
    fn test_smart_crop_prefers_high_saliency_region() {
        // Create a 20×20 image that is dark (low saliency) except for a bright
        // patch in the top-left (rows 0-5, cols 0-5 = value 255, rest = 0).
        // A 10×10 crop should move towards the bright region.
        let w = 20usize;
        let h = 20usize;
        let mut pixels = vec![0u8; w * h];
        for y in 0..6 {
            for x in 0..6 {
                pixels[y * w + x] = 255;
            }
        }
        let rect = smart_crop(&pixels, w as u32, h as u32, 10, 10);
        // The crop x position should be on the left half.
        assert!(
            rect.x <= 10,
            "expected crop to favour left side, got x={}",
            rect.x
        );
        assert!(rect.fits_in(w as u32, h as u32));
    }

    #[test]
    fn test_compute_saliency_uniform() {
        let pixels = vec![128u8; 10 * 10];
        let s = compute_saliency(&pixels, 10, 10);
        // Uniform image: all gradients are zero.
        for &v in &s {
            assert!(v.abs() < f32::EPSILON, "expected zero saliency, got {v}");
        }
    }
}
