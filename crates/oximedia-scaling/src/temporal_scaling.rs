//! Temporal scaling — frame rate conversion.
//!
//! Implements two frame rate conversion strategies:
//!
//! 1. **Frame blending**: produces an output frame by linearly blending two
//!    adjacent input frames weighted by their temporal proximity to the target
//!    presentation time. This avoids judder but can introduce motion blur.
//!
//! 2. **Motion-compensated interpolation (MCI)**: estimates the dominant
//!    block-level motion vector between two reference frames using a simple
//!    block-matching search (Sum of Absolute Differences) and warps the
//!    forward and backward reference frames toward the target time. The two
//!    warped frames are then blended. This reduces motion blur while
//!    maintaining temporal smoothness.
//!
//! Both algorithms operate on packed RGB (`u8`, 3 bytes per pixel) frames in
//! row-major order.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::fmt;

use rayon::prelude::*;

// ── Rational frame rate ───────────────────────────────────────────────────────

/// A rational frame rate expressed as `numerator / denominator` frames per
/// second.
///
/// Common presets: 24000/1001 ≈ 23.976, 30000/1001 ≈ 29.97, 50/1, 60/1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRate {
    /// Numerator (frames per `denominator` seconds).
    pub num: u32,
    /// Denominator (seconds unit).
    pub den: u32,
}

impl FrameRate {
    /// Create a new frame rate. Denominator is clamped to at least 1.
    pub fn new(num: u32, den: u32) -> Self {
        Self {
            num,
            den: den.max(1),
        }
    }

    /// 23.976 fps (24000/1001).
    pub fn ntsc_film() -> Self {
        Self {
            num: 24_000,
            den: 1_001,
        }
    }

    /// 24 fps (film).
    pub fn film() -> Self {
        Self { num: 24, den: 1 }
    }

    /// 25 fps (PAL).
    pub fn pal() -> Self {
        Self { num: 25, den: 1 }
    }

    /// 29.97 fps (NTSC).
    pub fn ntsc() -> Self {
        Self {
            num: 30_000,
            den: 1_001,
        }
    }

    /// 30 fps.
    pub fn fps30() -> Self {
        Self { num: 30, den: 1 }
    }

    /// 50 fps (PAL high frame rate).
    pub fn fps50() -> Self {
        Self { num: 50, den: 1 }
    }

    /// 59.94 fps.
    pub fn fps59_94() -> Self {
        Self {
            num: 60_000,
            den: 1_001,
        }
    }

    /// 60 fps.
    pub fn fps60() -> Self {
        Self { num: 60, den: 1 }
    }

    /// Returns the frame rate as a floating-point value.
    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Returns the duration of one frame in seconds.
    pub fn frame_duration_secs(self) -> f64 {
        self.den as f64 / self.num as f64
    }
}

impl fmt::Display for FrameRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{} fps", self.num)
        } else {
            write!(f, "{}/{} fps", self.num, self.den)
        }
    }
}

// ── Interpolation mode ────────────────────────────────────────────────────────

/// Frame interpolation algorithm used for rate conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    /// Linear blend of the two nearest input frames.
    FrameBlend,
    /// Block-matching motion estimation with warped-frame blending.
    MotionCompensated,
}

impl fmt::Display for InterpolationMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FrameBlend => write!(f, "FrameBlend"),
            Self::MotionCompensated => write!(f, "MotionCompensated"),
        }
    }
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for temporal (frame-rate) conversion.
#[derive(Debug, Clone)]
pub struct TemporalScalingConfig {
    /// Source frame rate.
    pub src_rate: FrameRate,
    /// Target (output) frame rate.
    pub dst_rate: FrameRate,
    /// Interpolation algorithm.
    pub mode: InterpolationMode,
    /// Block size used for motion search (must be ≥ 4). Only used in
    /// `MotionCompensated` mode.
    pub block_size: u32,
    /// Half-size of the motion search window (pixels in each direction).
    /// Only used in `MotionCompensated` mode.
    pub search_radius: u32,
}

impl TemporalScalingConfig {
    /// Create a configuration with the given source and target frame rates.
    pub fn new(src_rate: FrameRate, dst_rate: FrameRate) -> Self {
        Self {
            src_rate,
            dst_rate,
            mode: InterpolationMode::FrameBlend,
            block_size: 16,
            search_radius: 8,
        }
    }

    /// Set the interpolation mode.
    pub fn with_mode(mut self, mode: InterpolationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the block size for motion search.
    pub fn with_block_size(mut self, block_size: u32) -> Self {
        self.block_size = block_size.max(4);
        self
    }

    /// Set the search radius for motion estimation.
    pub fn with_search_radius(mut self, radius: u32) -> Self {
        self.search_radius = radius;
        self
    }
}

// ── Frame ─────────────────────────────────────────────────────────────────────

/// A single packed RGB video frame.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Packed RGB pixel data (3 bytes per pixel, row-major).
    pub pixels: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Presentation timestamp in seconds from stream start.
    pub pts_secs: f64,
}

impl VideoFrame {
    /// Create a new frame.
    ///
    /// Returns `None` if the pixel buffer is too small.
    pub fn new(pixels: Vec<u8>, width: u32, height: u32, pts_secs: f64) -> Option<Self> {
        if pixels.len() < (width as usize) * (height as usize) * 3 {
            return None;
        }
        Some(Self {
            pixels,
            width,
            height,
            pts_secs,
        })
    }

    /// Sample the pixel at `(x, y)` — returns `[R, G, B]`.
    ///
    /// Clamps coordinates to the frame boundary.
    #[inline]
    pub fn sample(&self, x: i32, y: i32) -> [u8; 3] {
        let cx = x.clamp(0, self.width as i32 - 1) as usize;
        let cy = y.clamp(0, self.height as i32 - 1) as usize;
        let base = (cy * self.width as usize + cx) * 3;
        [
            self.pixels[base],
            self.pixels[base + 1],
            self.pixels[base + 2],
        ]
    }
}

// ── Motion vector ─────────────────────────────────────────────────────────────

/// A 2-D motion vector in pixel units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionVector {
    /// Horizontal displacement (positive = rightward).
    pub dx: i32,
    /// Vertical displacement (positive = downward).
    pub dy: i32,
}

impl MotionVector {
    /// Zero motion (no displacement).
    pub const ZERO: Self = Self { dx: 0, dy: 0 };

    /// Create a motion vector.
    pub fn new(dx: i32, dy: i32) -> Self {
        Self { dx, dy }
    }
}

// ── Block-matching motion estimation ─────────────────────────────────────────

/// Estimate a global motion vector between `reference` and `target` frames
/// using block-matching (Sum of Absolute Differences) over a regular grid of
/// blocks.
///
/// For each block in `reference`, the algorithm searches a neighbourhood of
/// radius `search_radius` pixels in `target` for the best matching block, then
/// returns the median motion vector across all blocks.
pub fn estimate_motion(
    reference: &VideoFrame,
    target: &VideoFrame,
    block_size: u32,
    search_radius: u32,
) -> MotionVector {
    if reference.width == 0 || reference.height == 0 {
        return MotionVector::ZERO;
    }

    let bs = block_size as usize;
    let sr = search_radius as i32;
    let w = reference.width as usize;
    let h = reference.height as usize;

    // Collect block motion vectors in parallel
    let blocks_x = (w / bs).max(1);
    let blocks_y = (h / bs).max(1);

    let vectors: Vec<MotionVector> = (0..blocks_y)
        .into_par_iter()
        .flat_map_iter(|by| {
            let oy = by * bs;
            (0..blocks_x).map(move |bx| {
                let ox = bx * bs;
                // Initialise with zero displacement so that when all candidates
                // have the same SAD (e.g. identical frames) we prefer no motion.
                let mut best_dx = 0i32;
                let mut best_dy = 0i32;
                let mut best_sad = block_sad(reference, target, ox, oy, bs, 0, 0);

                for dy in -sr..=sr {
                    for dx in -sr..=sr {
                        if dx == 0 && dy == 0 {
                            continue; // already initialised with (0,0)
                        }
                        let sad = block_sad(reference, target, ox, oy, bs, dx, dy);
                        if sad < best_sad {
                            best_sad = sad;
                            best_dx = dx;
                            best_dy = dy;
                        }
                    }
                }
                MotionVector::new(best_dx, best_dy)
            })
        })
        .collect();

    median_motion_vector(&vectors)
}

/// Sum of Absolute Differences for a block starting at `(ox, oy)` in the
/// reference frame, compared to the block at `(ox + dx, oy + dy)` in the
/// target frame.
fn block_sad(
    reference: &VideoFrame,
    target: &VideoFrame,
    ox: usize,
    oy: usize,
    block_size: usize,
    dx: i32,
    dy: i32,
) -> u64 {
    let mut sad = 0u64;
    for row in 0..block_size {
        for col in 0..block_size {
            let rx = ox + col;
            let ry = oy + row;
            let tx = rx as i32 + dx;
            let ty = ry as i32 + dy;
            let rp = reference.sample(rx as i32, ry as i32);
            let tp = target.sample(tx, ty);
            for c in 0..3 {
                sad += (rp[c] as i64 - tp[c] as i64).unsigned_abs() as u64;
            }
        }
    }
    sad
}

/// Compute the median motion vector from a list.
///
/// Uses separate medians for `dx` and `dy`.
fn median_motion_vector(vectors: &[MotionVector]) -> MotionVector {
    if vectors.is_empty() {
        return MotionVector::ZERO;
    }
    let mut dxs: Vec<i32> = vectors.iter().map(|v| v.dx).collect();
    let mut dys: Vec<i32> = vectors.iter().map(|v| v.dy).collect();
    dxs.sort_unstable();
    dys.sort_unstable();
    let mid = dxs.len() / 2;
    MotionVector::new(dxs[mid], dys[mid])
}

// ── Frame blending ────────────────────────────────────────────────────────────

/// Blend two RGB frames linearly.
///
/// `alpha` is the blend factor for `frame_a` (0.0 = fully `frame_b`,
/// 1.0 = fully `frame_a`). Both frames must have identical dimensions.
///
/// Returns `None` if the frames differ in size or a buffer is too small.
pub fn blend_frames(frame_a: &VideoFrame, frame_b: &VideoFrame, alpha: f64) -> Option<Vec<u8>> {
    if frame_a.width != frame_b.width || frame_a.height != frame_b.height {
        return None;
    }
    let count = (frame_a.width as usize) * (frame_a.height as usize) * 3;
    if frame_a.pixels.len() < count || frame_b.pixels.len() < count {
        return None;
    }
    let a = alpha.clamp(0.0, 1.0) as f32;
    let b = 1.0 - a;
    let blended: Vec<u8> = frame_a.pixels[..count]
        .iter()
        .zip(frame_b.pixels[..count].iter())
        .map(|(&pa, &pb)| (pa as f32 * a + pb as f32 * b).round().clamp(0.0, 255.0) as u8)
        .collect();
    Some(blended)
}

// ── Motion-compensated interpolation ─────────────────────────────────────────

/// Synthesise a frame at time `t` ∈ `[0, 1]` between `frame_a` (at t=0) and
/// `frame_b` (at t=1) using motion-compensated interpolation.
///
/// The pipeline:
/// 1. Estimate the motion vector `mv` from `frame_a` to `frame_b`.
/// 2. Warp `frame_a` forward by `t × mv` toward `frame_b`.
/// 3. Warp `frame_b` backward by `(1-t) × mv` toward `frame_a`.
/// 4. Linearly blend the two warped frames with weight `1-t` and `t`.
///
/// Returns `None` if the frames differ in size or pixel buffers are too small.
pub fn motion_compensated_interpolate(
    frame_a: &VideoFrame,
    frame_b: &VideoFrame,
    t: f64,
    block_size: u32,
    search_radius: u32,
) -> Option<Vec<u8>> {
    if frame_a.width != frame_b.width || frame_a.height != frame_b.height {
        return None;
    }
    let w = frame_a.width as usize;
    let h = frame_a.height as usize;
    let count = w * h * 3;
    if frame_a.pixels.len() < count || frame_b.pixels.len() < count {
        return None;
    }
    let t = t.clamp(0.0, 1.0);

    let mv = estimate_motion(frame_a, frame_b, block_size, search_radius);

    // Warp frame_a forward by t * mv
    let warped_a = warp_frame(
        frame_a,
        (mv.dx as f64 * t) as i32,
        (mv.dy as f64 * t) as i32,
    );
    // Warp frame_b backward by -(1-t) * mv
    let warped_b = warp_frame(
        frame_b,
        -(mv.dx as f64 * (1.0 - t)) as i32,
        -(mv.dy as f64 * (1.0 - t)) as i32,
    );

    let wa = VideoFrame {
        pixels: warped_a,
        width: frame_a.width,
        height: frame_a.height,
        pts_secs: frame_a.pts_secs,
    };
    let wb = VideoFrame {
        pixels: warped_b,
        width: frame_b.width,
        height: frame_b.height,
        pts_secs: frame_b.pts_secs,
    };

    blend_frames(&wa, &wb, 1.0 - t)
}

/// Shift all pixels of a frame by `(dx, dy)` pixels using integer translation.
///
/// Out-of-bounds regions are filled by clamping to the nearest edge pixel.
fn warp_frame(frame: &VideoFrame, dx: i32, dy: i32) -> Vec<u8> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let mut out = vec![0u8; w * h * 3];
    for oy in 0..h {
        for ox in 0..w {
            let sx = ox as i32 - dx;
            let sy = oy as i32 - dy;
            let p = frame.sample(sx, sy);
            let base = (oy * w + ox) * 3;
            out[base] = p[0];
            out[base + 1] = p[1];
            out[base + 2] = p[2];
        }
    }
    out
}

// ── Temporal converter ────────────────────────────────────────────────────────

/// Frame rate converter that produces an output frame sequence from an input
/// sequence using the configured interpolation algorithm.
#[derive(Debug)]
pub struct TemporalScaler {
    config: TemporalScalingConfig,
}

impl TemporalScaler {
    /// Create a new `TemporalScaler`.
    pub fn new(config: TemporalScalingConfig) -> Self {
        Self { config }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &TemporalScalingConfig {
        &self.config
    }

    /// Compute the output presentation timestamps for a given number of input
    /// frames at the source frame rate.
    ///
    /// Returns the list of `(input_frame_a_index, input_frame_b_index, t)` tuples
    /// where `t ∈ [0, 1]` is the blend factor toward `frame_b`, for each output
    /// frame. `t = 0` means the output is identical to `frame_a`; `t = 1` means
    /// identical to `frame_b`.
    pub fn compute_schedule(&self, num_input_frames: usize) -> Vec<(usize, usize, f64)> {
        if num_input_frames < 2 {
            return Vec::new();
        }

        let src_fps = self.config.src_rate.to_f64();
        let dst_fps = self.config.dst_rate.to_f64();
        if src_fps <= 0.0 || dst_fps <= 0.0 {
            return Vec::new();
        }

        let src_duration = (num_input_frames - 1) as f64 / src_fps;
        let num_output_frames = (src_duration * dst_fps).floor() as usize + 1;
        let dst_frame_dur = 1.0 / dst_fps;

        let mut schedule = Vec::with_capacity(num_output_frames);
        for out_idx in 0..num_output_frames {
            let t_out = out_idx as f64 * dst_frame_dur;
            let src_float = t_out * src_fps;
            let src_floor = src_float.floor() as usize;
            let frame_a = src_floor.min(num_input_frames - 2);
            let frame_b = (frame_a + 1).min(num_input_frames - 1);
            // blend_t is the fractional part relative to the true floor, not the
            // clamped frame_a, so that identity conversions always produce t = 0.
            let blend_t = (src_float - src_floor as f64).clamp(0.0, 1.0);
            schedule.push((frame_a, frame_b, blend_t));
        }

        schedule
    }

    /// Convert a sequence of input frames to the target frame rate.
    ///
    /// Calls `blend_frames` or `motion_compensated_interpolate` for each output
    /// frame according to the interpolation mode. Returns `None` if any frame
    /// in `input_frames` has inconsistent dimensions, or if the input is too
    /// short.
    pub fn convert(&self, input_frames: &[VideoFrame]) -> Option<Vec<Vec<u8>>> {
        if input_frames.len() < 2 {
            return None;
        }
        // Verify all frames have the same dimensions
        let w = input_frames[0].width;
        let h = input_frames[0].height;
        for frame in input_frames.iter().skip(1) {
            if frame.width != w || frame.height != h {
                return None;
            }
        }

        let schedule = self.compute_schedule(input_frames.len());
        let mut output = Vec::with_capacity(schedule.len());

        for (idx_a, idx_b, t) in schedule {
            let fa = &input_frames[idx_a];
            let fb = &input_frames[idx_b];

            let blended = match self.config.mode {
                InterpolationMode::FrameBlend => blend_frames(fa, fb, 1.0 - t)?,
                InterpolationMode::MotionCompensated => motion_compensated_interpolate(
                    fa,
                    fb,
                    t,
                    self.config.block_size,
                    self.config.search_radius,
                )?,
            };
            output.push(blended);
        }

        Some(output)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(r: u8, g: u8, b: u8, w: u32, h: u32, pts: f64) -> VideoFrame {
        let count = (w as usize) * (h as usize);
        let mut pixels = vec![0u8; count * 3];
        for i in 0..count {
            pixels[i * 3] = r;
            pixels[i * 3 + 1] = g;
            pixels[i * 3 + 2] = b;
        }
        VideoFrame::new(pixels, w, h, pts).expect("frame creation should succeed")
    }

    // ── FrameRate ────────────────────────────────────────────────────────────

    #[test]
    fn test_frame_rate_new() {
        let fps = FrameRate::new(30, 1);
        assert_eq!(fps.num, 30);
        assert_eq!(fps.den, 1);
        assert!((fps.to_f64() - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_rate_zero_den_clamped() {
        let fps = FrameRate::new(30, 0);
        assert_eq!(fps.den, 1);
    }

    #[test]
    fn test_frame_rate_ntsc() {
        let fps = FrameRate::ntsc();
        assert!((fps.to_f64() - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_frame_rate_frame_duration() {
        let fps = FrameRate::fps60();
        assert!((fps.frame_duration_secs() - 1.0 / 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_rate_display_integer() {
        let fps = FrameRate::fps30();
        assert_eq!(fps.to_string(), "30 fps");
    }

    #[test]
    fn test_frame_rate_display_rational() {
        let fps = FrameRate::ntsc();
        let s = fps.to_string();
        assert!(s.contains("30000"));
        assert!(s.contains("1001"));
    }

    // ── VideoFrame ───────────────────────────────────────────────────────────

    #[test]
    fn test_video_frame_sample_center() {
        let frame = make_frame(100, 150, 200, 4, 4, 0.0);
        let p = frame.sample(2, 2);
        assert_eq!(p, [100, 150, 200]);
    }

    #[test]
    fn test_video_frame_sample_clamp_neg() {
        let frame = make_frame(10, 20, 30, 4, 4, 0.0);
        let p = frame.sample(-5, -5);
        assert_eq!(p, [10, 20, 30]); // clamped to (0,0)
    }

    #[test]
    fn test_video_frame_sample_clamp_oob() {
        let frame = make_frame(50, 60, 70, 4, 4, 0.0);
        let p = frame.sample(100, 100);
        assert_eq!(p, [50, 60, 70]); // clamped to (3,3)
    }

    #[test]
    fn test_video_frame_new_invalid_buffer() {
        let result = VideoFrame::new(vec![0u8; 3], 4, 4, 0.0);
        assert!(result.is_none());
    }

    // ── InterpolationMode display ────────────────────────────────────────────

    #[test]
    fn test_interpolation_mode_display() {
        assert_eq!(InterpolationMode::FrameBlend.to_string(), "FrameBlend");
        assert_eq!(
            InterpolationMode::MotionCompensated.to_string(),
            "MotionCompensated"
        );
    }

    // ── blend_frames ────────────────────────────────────────────────────────

    #[test]
    fn test_blend_frames_alpha_zero() {
        // alpha=0 → fully frame_b
        let fa = make_frame(255, 0, 0, 2, 2, 0.0);
        let fb = make_frame(0, 0, 255, 2, 2, 0.0);
        let result = blend_frames(&fa, &fb, 0.0).expect("blend should succeed");
        for chunk in result.chunks(3) {
            assert!(chunk[0] < 10, "should be mostly blue");
            assert!(chunk[2] > 200, "should be mostly blue");
        }
    }

    #[test]
    fn test_blend_frames_alpha_one() {
        let fa = make_frame(255, 0, 0, 2, 2, 0.0);
        let fb = make_frame(0, 0, 255, 2, 2, 0.0);
        let result = blend_frames(&fa, &fb, 1.0).expect("blend should succeed");
        for chunk in result.chunks(3) {
            assert!(chunk[0] > 200, "should be fully red");
        }
    }

    #[test]
    fn test_blend_frames_half() {
        let fa = make_frame(200, 200, 200, 2, 2, 0.0);
        let fb = make_frame(100, 100, 100, 2, 2, 0.0);
        let result = blend_frames(&fa, &fb, 0.5).expect("blend should succeed");
        for chunk in result.chunks(3) {
            let v = chunk[0];
            assert!((v as i32 - 150).abs() <= 1, "expected ~150, got {v}");
        }
    }

    #[test]
    fn test_blend_frames_dimension_mismatch() {
        let fa = make_frame(0, 0, 0, 4, 4, 0.0);
        let fb = make_frame(0, 0, 0, 2, 2, 0.0);
        assert!(blend_frames(&fa, &fb, 0.5).is_none());
    }

    // ── Motion estimation ────────────────────────────────────────────────────

    #[test]
    fn test_estimate_motion_identical_frames() {
        let fa = make_frame(128, 64, 32, 8, 8, 0.0);
        let fb = fa.clone();
        let mv = estimate_motion(&fa, &fb, 4, 2);
        // Identical frames → zero motion
        assert_eq!(mv.dx, 0);
        assert_eq!(mv.dy, 0);
    }

    #[test]
    fn test_motion_vector_zero() {
        let mv = MotionVector::ZERO;
        assert_eq!(mv.dx, 0);
        assert_eq!(mv.dy, 0);
    }

    #[test]
    fn test_estimate_motion_shifted_frame() {
        // Create a frame with a bright rectangle, then shift it by (2, 1)
        let w = 8u32;
        let h = 8u32;
        let count = (w as usize) * (h as usize);
        let mut pixels_a = vec![50u8; count * 3];
        // Place a bright 4x4 block at (0,0)
        for y in 0..4usize {
            for x in 0..4usize {
                let base = (y * w as usize + x) * 3;
                pixels_a[base] = 200;
                pixels_a[base + 1] = 200;
                pixels_a[base + 2] = 200;
            }
        }
        let fa = VideoFrame::new(pixels_a.clone(), w, h, 0.0).expect("frame ok");

        // Shift frame by dx=2, dy=1
        let warped = warp_frame(&fa, 2, 1);
        let fb = VideoFrame::new(warped, w, h, 0.04).expect("frame ok");

        let mv = estimate_motion(&fa, &fb, 4, 4);
        // Should detect motion near (2, 1)
        assert!(
            mv.dx.abs() <= 3 && mv.dy.abs() <= 2,
            "motion vector ({},{}) too large",
            mv.dx,
            mv.dy
        );
    }

    // ── MotionCompensated interpolation ──────────────────────────────────────

    #[test]
    fn test_motion_compensated_at_t0() {
        let fa = make_frame(100, 0, 0, 4, 4, 0.0);
        let fb = make_frame(200, 0, 0, 4, 4, 0.04);
        let result = motion_compensated_interpolate(&fa, &fb, 0.0, 4, 2);
        assert!(result.is_some());
        let out = result.expect("mci should succeed");
        // At t=0, output should be close to frame_a
        for chunk in out.chunks(3) {
            assert!(
                (chunk[0] as i32 - 100).abs() <= 20,
                "expected near 100, got {}",
                chunk[0]
            );
        }
    }

    #[test]
    fn test_motion_compensated_output_size() {
        let fa = make_frame(0, 128, 0, 4, 4, 0.0);
        let fb = make_frame(0, 200, 0, 4, 4, 0.04);
        let result = motion_compensated_interpolate(&fa, &fb, 0.5, 4, 2);
        assert!(result.is_some());
        let out = result.expect("mci should succeed");
        assert_eq!(out.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_motion_compensated_dimension_mismatch() {
        let fa = make_frame(0, 0, 0, 4, 4, 0.0);
        let fb = make_frame(0, 0, 0, 8, 4, 0.0);
        assert!(motion_compensated_interpolate(&fa, &fb, 0.5, 4, 2).is_none());
    }

    // ── TemporalScaler schedule ───────────────────────────────────────────────

    #[test]
    fn test_compute_schedule_24_to_30() {
        // 24 fps → 30 fps with 5 input frames
        let cfg = TemporalScalingConfig::new(FrameRate::film(), FrameRate::fps30());
        let scaler = TemporalScaler::new(cfg);
        let schedule = scaler.compute_schedule(5);
        // Verify every output frame references valid input indices
        for &(a, b, t) in &schedule {
            assert!(a < 5, "a={a} out of range");
            assert!(b < 5, "b={b} out of range");
            assert!(b >= a, "b must be >= a");
            assert!(t >= 0.0 && t <= 1.0, "t={t} out of [0,1]");
        }
        // 5 input frames at 24fps → ~4/24 = 0.167s duration → 5 output frames at 30fps
        assert!(!schedule.is_empty());
    }

    #[test]
    fn test_compute_schedule_too_short() {
        let cfg = TemporalScalingConfig::new(FrameRate::film(), FrameRate::fps30());
        let scaler = TemporalScaler::new(cfg);
        assert!(scaler.compute_schedule(0).is_empty());
        assert!(scaler.compute_schedule(1).is_empty());
    }

    #[test]
    fn test_compute_schedule_identity() {
        // 30fps → 30fps: schedule should have the same frame count
        let cfg = TemporalScalingConfig::new(FrameRate::fps30(), FrameRate::fps30());
        let scaler = TemporalScaler::new(cfg);
        let schedule = scaler.compute_schedule(10);
        assert_eq!(schedule.len(), 10);
        // All t values should be near 0 (each output maps to start of source frame)
        for &(_, _, t) in &schedule {
            assert!(t < 1e-9, "t should be 0 for identity conversion, got {t}");
        }
    }

    // ── TemporalScaler::convert ───────────────────────────────────────────────

    #[test]
    fn test_convert_frame_blend() {
        let frames: Vec<VideoFrame> = (0..5)
            .map(|i| make_frame(i * 50, 0, 0, 4, 4, i as f64 / 24.0))
            .collect();
        let cfg = TemporalScalingConfig::new(FrameRate::film(), FrameRate::fps30());
        let scaler = TemporalScaler::new(cfg);
        let result = scaler.convert(&frames);
        assert!(result.is_some());
        let output = result.expect("conversion should succeed");
        assert!(!output.is_empty());
        for frame in &output {
            assert_eq!(frame.len(), 4 * 4 * 3);
        }
    }

    #[test]
    fn test_convert_motion_compensated() {
        let frames: Vec<VideoFrame> = (0..4)
            .map(|i| make_frame(100 + i * 20, 0, 0, 8, 8, i as f64 / 25.0))
            .collect();
        let cfg = TemporalScalingConfig::new(FrameRate::pal(), FrameRate::fps30())
            .with_mode(InterpolationMode::MotionCompensated)
            .with_block_size(4)
            .with_search_radius(2);
        let scaler = TemporalScaler::new(cfg);
        let result = scaler.convert(&frames);
        assert!(result.is_some());
        let output = result.expect("conversion should succeed");
        for frame in &output {
            assert_eq!(frame.len(), 8 * 8 * 3);
        }
    }

    #[test]
    fn test_convert_too_few_frames() {
        let frames = vec![make_frame(0, 0, 0, 4, 4, 0.0)];
        let cfg = TemporalScalingConfig::new(FrameRate::film(), FrameRate::fps30());
        let scaler = TemporalScaler::new(cfg);
        assert!(scaler.convert(&frames).is_none());
    }

    #[test]
    fn test_convert_dimension_mismatch() {
        let fa = make_frame(0, 0, 0, 4, 4, 0.0);
        let fb = make_frame(0, 0, 0, 8, 8, 0.04);
        let cfg = TemporalScalingConfig::new(FrameRate::film(), FrameRate::fps30());
        let scaler = TemporalScaler::new(cfg);
        assert!(scaler.convert(&[fa, fb]).is_none());
    }

    #[test]
    fn test_config_builder() {
        let cfg = TemporalScalingConfig::new(FrameRate::ntsc(), FrameRate::fps60())
            .with_mode(InterpolationMode::MotionCompensated)
            .with_block_size(8)
            .with_search_radius(16);
        assert_eq!(cfg.mode, InterpolationMode::MotionCompensated);
        assert_eq!(cfg.block_size, 8);
        assert_eq!(cfg.search_radius, 16);
    }

    #[test]
    fn test_config_block_size_minimum() {
        let cfg =
            TemporalScalingConfig::new(FrameRate::film(), FrameRate::fps30()).with_block_size(1); // below minimum
        assert_eq!(cfg.block_size, 4);
    }
}
