//! Stereoscopic 3D support for 360° VR video.
//!
//! Handles frame splitting/merging for side-by-side and top-bottom stereo
//! layouts, depth-based stereo synthesis, and basic quality metrics.

use crate::VrError;

// ─── Layout enum ─────────────────────────────────────────────────────────────

/// Arrangement of left/right eye views within a single video frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StereoLayout {
    /// Top half = left eye, bottom half = right eye.
    TopBottom,
    /// Left half = left eye, right half = right eye.
    LeftRight,
    /// Odd frames = left eye, even frames = right eye.
    Alternating,
    /// No stereo: single monoscopic view.
    Mono,
}

// ─── Metadata ────────────────────────────────────────────────────────────────

/// Stereoscopic recording / display parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct StereoMetadata {
    /// How the two eye views are packed in the frame.
    pub layout: StereoLayout,
    /// Camera eye separation expressed as angular separation in degrees.
    pub eye_separation_deg: f32,
    /// Inter-pupillary distance of the intended viewer in millimetres.
    pub ipd_mm: f32,
}

impl StereoMetadata {
    /// Create stereo metadata with sensible defaults (65 mm IPD).
    pub fn new(layout: StereoLayout) -> Self {
        Self {
            layout,
            eye_separation_deg: 0.065,
            ipd_mm: 65.0,
        }
    }
}

// ─── Frame split / merge ─────────────────────────────────────────────────────

/// Split a packed stereo frame into separate left and right eye buffers.
///
/// * `data`     — packed frame pixel data (RGB, 3 bytes per pixel, row-major)
/// * `width`    — full frame width in pixels
/// * `height`   — full frame height in pixels
/// * `layout`   — stereo packing layout
///
/// Returns `(left, right)` eye buffers.
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if the frame dimensions are incompatible
/// with the requested layout (e.g. odd height for `TopBottom`).
/// Returns [`VrError::UnsupportedLayout`] for `Mono` and `Alternating`.
pub fn split_stereo_frame(
    data: &[u8],
    width: u32,
    height: u32,
    layout: StereoLayout,
) -> Result<(Vec<u8>, Vec<u8>), VrError> {
    const CH: usize = 3;
    let expected = width as usize * height as usize * CH;
    if data.len() < expected {
        return Err(VrError::BufferTooSmall {
            expected,
            got: data.len(),
        });
    }
    if width == 0 || height == 0 {
        return Err(VrError::InvalidDimensions(
            "width/height must be > 0".into(),
        ));
    }

    match layout {
        StereoLayout::TopBottom => {
            if height % 2 != 0 {
                return Err(VrError::InvalidDimensions(
                    "TopBottom layout requires even height".into(),
                ));
            }
            let half_h = height / 2;
            let stride = width as usize * CH;
            let half_bytes = half_h as usize * stride;
            let left = data[..half_bytes].to_vec();
            let right = data[half_bytes..half_bytes * 2].to_vec();
            Ok((left, right))
        }

        StereoLayout::LeftRight => {
            if width % 2 != 0 {
                return Err(VrError::InvalidDimensions(
                    "LeftRight layout requires even width".into(),
                ));
            }
            let half_w = (width / 2) as usize;
            let full_w = width as usize;
            let row_bytes = full_w * CH;
            let eye_row_bytes = half_w * CH;
            let mut left = vec![0u8; height as usize * eye_row_bytes];
            let mut right = vec![0u8; height as usize * eye_row_bytes];

            for row in 0..height as usize {
                let src_row = &data[row * row_bytes..(row + 1) * row_bytes];
                left[row * eye_row_bytes..(row + 1) * eye_row_bytes]
                    .copy_from_slice(&src_row[..eye_row_bytes]);
                right[row * eye_row_bytes..(row + 1) * eye_row_bytes]
                    .copy_from_slice(&src_row[eye_row_bytes..]);
            }
            Ok((left, right))
        }

        StereoLayout::Mono | StereoLayout::Alternating => Err(VrError::UnsupportedLayout(format!(
            "{layout:?} cannot be split into two eye views from a single frame"
        ))),
    }
}

/// Merge separate left and right eye buffers into a single packed frame.
///
/// * `left`, `right` — per-eye pixel data (RGB, 3 bpp, row-major)
/// * `width`         — per-eye image width (for `LeftRight`: each eye is this wide)
/// * `height`        — per-eye image height (for `TopBottom`: each eye is this tall)
/// * `layout`        — stereo packing layout
///
/// # Errors
/// Returns [`VrError::UnsupportedLayout`] for `Mono` and `Alternating`.
pub fn merge_stereo_frames(
    left: &[u8],
    right: &[u8],
    width: u32,
    height: u32,
    layout: StereoLayout,
) -> Result<Vec<u8>, VrError> {
    const CH: usize = 3;

    match layout {
        StereoLayout::TopBottom => {
            let expected = width as usize * height as usize * CH;
            if left.len() < expected || right.len() < expected {
                return Err(VrError::BufferTooSmall {
                    expected,
                    got: left.len().min(right.len()),
                });
            }
            let mut out = Vec::with_capacity(expected * 2);
            out.extend_from_slice(&left[..expected]);
            out.extend_from_slice(&right[..expected]);
            Ok(out)
        }

        StereoLayout::LeftRight => {
            let eye_row = width as usize * CH;
            let expected = height as usize * eye_row;
            if left.len() < expected || right.len() < expected {
                return Err(VrError::BufferTooSmall {
                    expected,
                    got: left.len().min(right.len()),
                });
            }
            let full_row = eye_row * 2;
            let mut out = vec![0u8; height as usize * full_row];
            for row in 0..height as usize {
                out[row * full_row..row * full_row + eye_row]
                    .copy_from_slice(&left[row * eye_row..(row + 1) * eye_row]);
                out[row * full_row + eye_row..(row + 1) * full_row]
                    .copy_from_slice(&right[row * eye_row..(row + 1) * eye_row]);
            }
            Ok(out)
        }

        StereoLayout::Mono | StereoLayout::Alternating => Err(VrError::UnsupportedLayout(format!(
            "{layout:?} cannot be composed from two separate eye views"
        ))),
    }
}

// ─── Calibration & depth ──────────────────────────────────────────────────────

/// Physical stereo rig calibration parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct StereoCalibration {
    /// Distance to the convergence plane in metres.
    pub convergence_dist_m: f32,
    /// Physical screen width in metres.
    pub screen_width_m: f32,
    /// Viewer's distance from the screen in metres.
    pub viewer_dist_m: f32,
    /// Inter-pupillary distance in metres.
    pub ipd_m: f32,
}

impl StereoCalibration {
    /// Compute the pixel disparity for an object at `depth_m` metres.
    ///
    /// Uses the formula:
    /// `disparity = ipd × focal_length / depth`
    /// where `focal_length = screen_width × viewer_dist / image_width`.
    pub fn compute_disparity_pixels(&self, depth_m: f32, image_width: u32) -> f32 {
        if depth_m <= 0.0 || image_width == 0 {
            return 0.0;
        }
        let focal_length = self.screen_width_m * self.viewer_dist_m / image_width as f32;
        self.ipd_m * focal_length / depth_m
    }
}

// ─── Depth map ────────────────────────────────────────────────────────────────

/// Normalised depth map accompanying a video frame.
///
/// `data` values are in `0.0..=1.0` where `0.0` is closest and `1.0` is
/// farthest (or vice-versa depending on the acquisition pipeline — the caller
/// is responsible for the convention).
#[derive(Debug, Clone)]
pub struct DepthMap {
    /// Normalised depth values, row-major.
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl DepthMap {
    /// Create a new depth map filled with `fill_value`.
    pub fn new_filled(width: u32, height: u32, fill_value: f32) -> Self {
        Self {
            data: vec![fill_value; width as usize * height as usize],
            width,
            height,
        }
    }

    /// Sample the depth at normalised coordinates `(u, v)` using nearest-neighbour.
    pub fn sample(&self, u: f32, v: f32) -> f32 {
        let x = ((u * self.width as f32) as usize).min(self.width as usize - 1);
        let y = ((v * self.height as f32) as usize).min(self.height as usize - 1);
        self.data[y * self.width as usize + x]
    }
}

/// Generate a stereo pair from a mono frame and a depth map via horizontal
/// pixel shift (parallax mapping).
///
/// * `frame`            — source RGB frame (3 bpp, row-major)
/// * `depth`            — normalised depth map (must match frame dimensions)
/// * `max_disparity_px` — maximum horizontal shift in pixels at depth == 0
///
/// Returns `(left_eye, right_eye)` where left eye is shifted right and right
/// eye is shifted left relative to the source frame.
///
/// Regions with no source data after shifting are filled with black.
pub fn stereo_from_depth(
    frame: &[u8],
    depth: &DepthMap,
    max_disparity_px: u32,
) -> (Vec<u8>, Vec<u8>) {
    const CH: usize = 3;
    let w = depth.width as usize;
    let h = depth.height as usize;
    let row_bytes = w * CH;

    let mut left = vec![0u8; h * row_bytes];
    let mut right = vec![0u8; h * row_bytes];

    for row in 0..h {
        for col in 0..w {
            let depth_val = depth.data[row * w + col];
            // Closer objects (low depth value) get larger disparity
            let disparity = ((1.0 - depth_val) * max_disparity_px as f32) as usize;
            let half = disparity / 2;

            let src_base = row * row_bytes + col * CH;
            let pixel = &frame[src_base..src_base + CH];

            // Left eye: shift right (add to column)
            let left_col = col + half;
            if left_col < w {
                let dst = row * row_bytes + left_col * CH;
                left[dst..dst + CH].copy_from_slice(pixel);
            }

            // Right eye: shift left (subtract from column)
            if col >= half {
                let right_col = col - half;
                let dst = row * row_bytes + right_col * CH;
                right[dst..dst + CH].copy_from_slice(pixel);
            }
        }
    }

    (left, right)
}

// ─── RGBA stereo split / merge ────────────────────────────────────────────────

/// Split a packed RGBA stereo frame into separate left and right eye buffers.
///
/// Works identically to [`split_stereo_frame`] but with 4-byte pixels (RGBA).
///
/// * `data`   — packed RGBA frame (4 bytes per pixel, row-major)
/// * `width`  — full frame width in pixels
/// * `height` — full frame height in pixels
/// * `layout` — stereo packing layout
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if dimensions are incompatible.
/// Returns [`VrError::UnsupportedLayout`] for `Mono` and `Alternating`.
pub fn split_stereo_frame_rgba(
    data: &[u8],
    width: u32,
    height: u32,
    layout: StereoLayout,
) -> Result<(Vec<u8>, Vec<u8>), VrError> {
    const CH: usize = 4;
    let expected = width as usize * height as usize * CH;
    if data.len() < expected {
        return Err(VrError::BufferTooSmall {
            expected,
            got: data.len(),
        });
    }
    if width == 0 || height == 0 {
        return Err(VrError::InvalidDimensions(
            "width/height must be > 0".into(),
        ));
    }

    match layout {
        StereoLayout::TopBottom => {
            if height % 2 != 0 {
                return Err(VrError::InvalidDimensions(
                    "TopBottom layout requires even height".into(),
                ));
            }
            let half_h = height / 2;
            let stride = width as usize * CH;
            let half_bytes = half_h as usize * stride;
            let left = data[..half_bytes].to_vec();
            let right = data[half_bytes..half_bytes * 2].to_vec();
            Ok((left, right))
        }

        StereoLayout::LeftRight => {
            if width % 2 != 0 {
                return Err(VrError::InvalidDimensions(
                    "LeftRight layout requires even width".into(),
                ));
            }
            let half_w = (width / 2) as usize;
            let full_w = width as usize;
            let row_bytes = full_w * CH;
            let eye_row_bytes = half_w * CH;
            let mut left = vec![0u8; height as usize * eye_row_bytes];
            let mut right = vec![0u8; height as usize * eye_row_bytes];

            for row in 0..height as usize {
                let src_row = &data[row * row_bytes..(row + 1) * row_bytes];
                left[row * eye_row_bytes..(row + 1) * eye_row_bytes]
                    .copy_from_slice(&src_row[..eye_row_bytes]);
                right[row * eye_row_bytes..(row + 1) * eye_row_bytes]
                    .copy_from_slice(&src_row[eye_row_bytes..]);
            }
            Ok((left, right))
        }

        StereoLayout::Mono | StereoLayout::Alternating => Err(VrError::UnsupportedLayout(format!(
            "{layout:?} cannot be split into two eye views from a single frame"
        ))),
    }
}

/// Merge separate RGBA left and right eye buffers into a single packed frame.
///
/// Works identically to [`merge_stereo_frames`] but with 4-byte pixels (RGBA).
///
/// * `left`, `right` — per-eye pixel data (RGBA, 4 bpp, row-major)
/// * `width`         — per-eye image width
/// * `height`        — per-eye image height
/// * `layout`        — stereo packing layout
///
/// # Errors
/// Returns [`VrError::UnsupportedLayout`] for `Mono` and `Alternating`.
pub fn merge_stereo_frames_rgba(
    left: &[u8],
    right: &[u8],
    width: u32,
    height: u32,
    layout: StereoLayout,
) -> Result<Vec<u8>, VrError> {
    const CH: usize = 4;

    match layout {
        StereoLayout::TopBottom => {
            let expected = width as usize * height as usize * CH;
            if left.len() < expected || right.len() < expected {
                return Err(VrError::BufferTooSmall {
                    expected,
                    got: left.len().min(right.len()),
                });
            }
            let mut out = Vec::with_capacity(expected * 2);
            out.extend_from_slice(&left[..expected]);
            out.extend_from_slice(&right[..expected]);
            Ok(out)
        }

        StereoLayout::LeftRight => {
            let eye_row = width as usize * CH;
            let expected = height as usize * eye_row;
            if left.len() < expected || right.len() < expected {
                return Err(VrError::BufferTooSmall {
                    expected,
                    got: left.len().min(right.len()),
                });
            }
            let full_row = eye_row * 2;
            let mut out = vec![0u8; height as usize * full_row];
            for row in 0..height as usize {
                out[row * full_row..row * full_row + eye_row]
                    .copy_from_slice(&left[row * eye_row..(row + 1) * eye_row]);
                out[row * full_row + eye_row..(row + 1) * full_row]
                    .copy_from_slice(&right[row * eye_row..(row + 1) * eye_row]);
            }
            Ok(out)
        }

        StereoLayout::Mono | StereoLayout::Alternating => Err(VrError::UnsupportedLayout(format!(
            "{layout:?} cannot be composed from two separate eye views"
        ))),
    }
}

// ─── Quality metric ───────────────────────────────────────────────────────────

/// Stereo quality metrics.
pub struct StereoQuality;

impl StereoQuality {
    /// Compute the average Sum of Absolute Differences (SAD) between left and
    /// right eye frames as a measure of parallax error.
    ///
    /// Lower values indicate better left–right alignment (i.e. less unwanted
    /// vertical parallax or colour mismatch).
    ///
    /// Returns a value in the range `0.0 ..= 255.0`.
    ///
    /// # Errors
    /// Returns [`VrError::BufferTooSmall`] if either buffer is too small.
    pub fn compute_parallax_error(
        left: &[u8],
        right: &[u8],
        width: u32,
        height: u32,
    ) -> Result<f32, VrError> {
        const CH: usize = 3;
        let expected = width as usize * height as usize * CH;
        if left.len() < expected {
            return Err(VrError::BufferTooSmall {
                expected,
                got: left.len(),
            });
        }
        if right.len() < expected {
            return Err(VrError::BufferTooSmall {
                expected,
                got: right.len(),
            });
        }

        let total_sad: u64 = left[..expected]
            .iter()
            .zip(right[..expected].iter())
            .map(|(&l, &r)| (l as i32 - r as i32).unsigned_abs() as u64)
            .sum();

        let num_samples = expected as f64;
        Ok((total_sad as f64 / num_samples) as f32)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(w as usize * h as usize * 3);
        for _ in 0..(w * h) {
            v.push(r);
            v.push(g);
            v.push(b);
        }
        v
    }

    // ── StereoMetadata ───────────────────────────────────────────────────────

    #[test]
    fn metadata_defaults() {
        let m = StereoMetadata::new(StereoLayout::TopBottom);
        assert_eq!(m.layout, StereoLayout::TopBottom);
        assert_eq!(m.ipd_mm, 65.0);
    }

    // ── TopBottom split / merge roundtrip ────────────────────────────────────

    #[test]
    fn topbottom_split_sizes() {
        let frame = solid_frame(64, 32, 10, 20, 30);
        let (left, right) =
            split_stereo_frame(&frame, 64, 32, StereoLayout::TopBottom).expect("split");
        assert_eq!(left.len(), 64 * 16 * 3);
        assert_eq!(right.len(), 64 * 16 * 3);
    }

    #[test]
    fn topbottom_merge_roundtrip() {
        let left = solid_frame(16, 8, 100, 0, 0);
        let right = solid_frame(16, 8, 0, 200, 0);
        let merged =
            merge_stereo_frames(&left, &right, 16, 8, StereoLayout::TopBottom).expect("merge");
        assert_eq!(merged.len(), 16 * 16 * 3);
        assert_eq!(merged[0], 100); // left eye red
        let half = 16 * 8 * 3;
        assert_eq!(merged[half + 1], 200); // right eye green
    }

    #[test]
    fn topbottom_split_odd_height_error() {
        let frame = solid_frame(8, 7, 0, 0, 0);
        let result = split_stereo_frame(&frame, 8, 7, StereoLayout::TopBottom);
        assert!(result.is_err());
    }

    #[test]
    fn topbottom_split_then_merge_identity() {
        let original = solid_frame(16, 8, 42, 84, 126);
        // Make unique left/right halves
        let mut frame = solid_frame(16, 16, 0, 0, 0);
        for i in 0..16 * 8 * 3 {
            frame[i] = original[i % original.len()];
        }
        for i in (16 * 8 * 3)..(16 * 16 * 3) {
            frame[i] = (i % 200) as u8;
        }

        let (left, right) =
            split_stereo_frame(&frame, 16, 16, StereoLayout::TopBottom).expect("split");
        let merged =
            merge_stereo_frames(&left, &right, 16, 8, StereoLayout::TopBottom).expect("merge");
        assert_eq!(merged.len(), frame.len());
        assert_eq!(merged, frame);
    }

    // ── LeftRight split / merge roundtrip ────────────────────────────────────

    #[test]
    fn leftright_split_sizes() {
        let frame = solid_frame(64, 32, 50, 100, 150);
        let (left, right) =
            split_stereo_frame(&frame, 64, 32, StereoLayout::LeftRight).expect("split");
        assert_eq!(left.len(), 32 * 32 * 3);
        assert_eq!(right.len(), 32 * 32 * 3);
    }

    #[test]
    fn leftright_merge_roundtrip() {
        let left = solid_frame(8, 4, 200, 0, 0);
        let right = solid_frame(8, 4, 0, 0, 200);
        let merged =
            merge_stereo_frames(&left, &right, 8, 4, StereoLayout::LeftRight).expect("merge");
        assert_eq!(merged.len(), 16 * 4 * 3);
        assert_eq!(merged[0], 200); // left eye red first pixel
        assert_eq!(merged[8 * 3 + 2], 200); // right eye blue first pixel
    }

    #[test]
    fn leftright_split_odd_width_error() {
        let frame = solid_frame(7, 4, 0, 0, 0);
        let result = split_stereo_frame(&frame, 7, 4, StereoLayout::LeftRight);
        assert!(result.is_err());
    }

    #[test]
    fn leftright_split_then_merge_identity() {
        let w = 16u32;
        let h = 4u32;
        // Build a frame where left half is all-red, right half is all-blue
        let mut frame = vec![0u8; w as usize * h as usize * 3];
        for row in 0..h as usize {
            for col in 0..w as usize {
                let base = row * w as usize * 3 + col * 3;
                if col < (w / 2) as usize {
                    frame[base] = 255; // red
                } else {
                    frame[base + 2] = 255; // blue
                }
            }
        }
        let (left, right) =
            split_stereo_frame(&frame, w, h, StereoLayout::LeftRight).expect("split");
        let merged =
            merge_stereo_frames(&left, &right, w / 2, h, StereoLayout::LeftRight).expect("merge");
        assert_eq!(merged, frame);
    }

    // ── Mono / Alternating unsupported ───────────────────────────────────────

    #[test]
    fn mono_split_unsupported() {
        let frame = solid_frame(8, 8, 0, 0, 0);
        let result = split_stereo_frame(&frame, 8, 8, StereoLayout::Mono);
        assert!(result.is_err());
    }

    #[test]
    fn alternating_split_unsupported() {
        let frame = solid_frame(8, 8, 0, 0, 0);
        let result = split_stereo_frame(&frame, 8, 8, StereoLayout::Alternating);
        assert!(result.is_err());
    }

    #[test]
    fn mono_merge_unsupported() {
        let left = solid_frame(8, 8, 0, 0, 0);
        let right = solid_frame(8, 8, 0, 0, 0);
        let result = merge_stereo_frames(&left, &right, 8, 8, StereoLayout::Mono);
        assert!(result.is_err());
    }

    // ── StereoCalibration ────────────────────────────────────────────────────

    #[test]
    fn disparity_at_infinity_is_near_zero() {
        let cal = StereoCalibration {
            convergence_dist_m: 2.0,
            screen_width_m: 0.5,
            viewer_dist_m: 0.6,
            ipd_m: 0.065,
        };
        // Very large depth → disparity → 0
        let d = cal.compute_disparity_pixels(10000.0, 1920);
        assert!(d < 0.01);
    }

    #[test]
    fn disparity_increases_for_closer_objects() {
        let cal = StereoCalibration {
            convergence_dist_m: 2.0,
            screen_width_m: 0.5,
            viewer_dist_m: 0.6,
            ipd_m: 0.065,
        };
        let d_far = cal.compute_disparity_pixels(5.0, 1920);
        let d_near = cal.compute_disparity_pixels(1.0, 1920);
        assert!(d_near > d_far);
    }

    #[test]
    fn disparity_zero_depth_returns_zero() {
        let cal = StereoCalibration {
            convergence_dist_m: 2.0,
            screen_width_m: 0.5,
            viewer_dist_m: 0.6,
            ipd_m: 0.065,
        };
        let d = cal.compute_disparity_pixels(0.0, 1920);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn disparity_zero_width_returns_zero() {
        let cal = StereoCalibration {
            convergence_dist_m: 2.0,
            screen_width_m: 0.5,
            viewer_dist_m: 0.6,
            ipd_m: 0.065,
        };
        let d = cal.compute_disparity_pixels(1.0, 0);
        assert_eq!(d, 0.0);
    }

    // ── DepthMap ─────────────────────────────────────────────────────────────

    #[test]
    fn depth_map_fill_and_sample() {
        let dm = DepthMap::new_filled(64, 64, 0.5);
        let v = dm.sample(0.5, 0.5);
        assert!((v - 0.5).abs() < 1e-5);
    }

    #[test]
    fn depth_map_sample_corners() {
        let dm = DepthMap::new_filled(8, 8, 0.75);
        assert!((dm.sample(0.0, 0.0) - 0.75).abs() < 1e-5);
        assert!((dm.sample(1.0, 1.0) - 0.75).abs() < 1e-5);
    }

    // ── stereo_from_depth ────────────────────────────────────────────────────

    #[test]
    fn stereo_from_depth_produces_correct_sizes() {
        let frame = solid_frame(32, 16, 128, 128, 128);
        let depth = DepthMap::new_filled(32, 16, 0.5);
        let (left, right) = stereo_from_depth(&frame, &depth, 4);
        assert_eq!(left.len(), frame.len());
        assert_eq!(right.len(), frame.len());
    }

    #[test]
    fn stereo_from_depth_zero_disparity_mirrors_source() {
        // depth == 1.0 → disparity == 0 → both eyes identical to source
        let frame = solid_frame(8, 4, 200, 100, 50);
        let depth = DepthMap::new_filled(8, 4, 1.0);
        let (left, right) = stereo_from_depth(&frame, &depth, 4);
        assert_eq!(left, frame);
        assert_eq!(right, frame);
    }

    // ── StereoQuality ────────────────────────────────────────────────────────

    #[test]
    fn parallax_error_identical_frames_is_zero() {
        let frame = solid_frame(8, 4, 100, 150, 200);
        let err = StereoQuality::compute_parallax_error(&frame, &frame, 8, 4).expect("compute");
        assert_eq!(err, 0.0);
    }

    #[test]
    fn parallax_error_different_frames_is_positive() {
        let left = solid_frame(8, 4, 0, 0, 0);
        let right = solid_frame(8, 4, 255, 255, 255);
        let err = StereoQuality::compute_parallax_error(&left, &right, 8, 4).expect("compute");
        assert!(err > 0.0);
    }

    #[test]
    fn parallax_error_buffer_too_small() {
        let left = vec![0u8; 10];
        let right = vec![0u8; 10];
        let result = StereoQuality::compute_parallax_error(&left, &right, 100, 100);
        assert!(result.is_err());
    }

    #[test]
    fn parallax_error_maximum_is_255() {
        let left = vec![0u8; 8 * 4 * 3];
        let right = vec![255u8; 8 * 4 * 3];
        let err = StereoQuality::compute_parallax_error(&left, &right, 8, 4).expect("compute");
        assert!((err - 255.0).abs() < 1.0);
    }

    // ── RGBA split / merge ───────────────────────────────────────────────────

    fn solid_frame_rgba(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(w as usize * h as usize * 4);
        for _ in 0..(w * h) {
            v.push(r);
            v.push(g);
            v.push(b);
            v.push(a);
        }
        v
    }

    #[test]
    fn rgba_topbottom_split_sizes() {
        let frame = solid_frame_rgba(64, 32, 10, 20, 30, 255);
        let (left, right) =
            split_stereo_frame_rgba(&frame, 64, 32, StereoLayout::TopBottom).expect("split");
        // Each eye is half the height
        assert_eq!(left.len(), 64 * 16 * 4);
        assert_eq!(right.len(), 64 * 16 * 4);
    }

    #[test]
    fn rgba_topbottom_merge_roundtrip() {
        let left = solid_frame_rgba(16, 8, 100, 0, 0, 255);
        let right = solid_frame_rgba(16, 8, 0, 200, 0, 128);
        let merged =
            merge_stereo_frames_rgba(&left, &right, 16, 8, StereoLayout::TopBottom).expect("merge");
        assert_eq!(merged.len(), 16 * 16 * 4);
        assert_eq!(merged[0], 100); // left eye red
        assert_eq!(merged[3], 255); // left eye alpha
        let half = 16 * 8 * 4;
        assert_eq!(merged[half + 1], 200); // right eye green
        assert_eq!(merged[half + 3], 128); // right eye alpha
    }

    #[test]
    fn rgba_topbottom_split_then_merge_identity() {
        let left_half = solid_frame_rgba(16, 8, 42, 84, 126, 200);
        let right_half = solid_frame_rgba(16, 8, 10, 20, 30, 100);
        let mut frame = Vec::with_capacity(16 * 16 * 4);
        frame.extend_from_slice(&left_half);
        frame.extend_from_slice(&right_half);

        let (l, r) =
            split_stereo_frame_rgba(&frame, 16, 16, StereoLayout::TopBottom).expect("split");
        let merged =
            merge_stereo_frames_rgba(&l, &r, 16, 8, StereoLayout::TopBottom).expect("merge");
        assert_eq!(merged, frame);
    }

    #[test]
    fn rgba_leftright_split_sizes() {
        let frame = solid_frame_rgba(64, 32, 50, 100, 150, 200);
        let (left, right) =
            split_stereo_frame_rgba(&frame, 64, 32, StereoLayout::LeftRight).expect("split");
        assert_eq!(left.len(), 32 * 32 * 4);
        assert_eq!(right.len(), 32 * 32 * 4);
    }

    #[test]
    fn rgba_leftright_merge_roundtrip() {
        let left = solid_frame_rgba(8, 4, 200, 0, 0, 255);
        let right = solid_frame_rgba(8, 4, 0, 0, 200, 128);
        let merged =
            merge_stereo_frames_rgba(&left, &right, 8, 4, StereoLayout::LeftRight).expect("merge");
        assert_eq!(merged.len(), 16 * 4 * 4);
        assert_eq!(merged[0], 200); // left eye red
        assert_eq!(merged[3], 255); // left eye alpha
        assert_eq!(merged[8 * 4 + 2], 200); // right eye blue
        assert_eq!(merged[8 * 4 + 3], 128); // right eye alpha
    }

    #[test]
    fn rgba_leftright_split_then_merge_identity() {
        let w = 16u32;
        let h = 4u32;
        // Build RGBA frame where left half is all red+opaque, right half is all blue+transparent
        let mut frame = vec![0u8; w as usize * h as usize * 4];
        for row in 0..h as usize {
            for col in 0..w as usize {
                let base = row * w as usize * 4 + col * 4;
                if col < (w / 2) as usize {
                    frame[base] = 255; // R
                    frame[base + 3] = 255; // A
                } else {
                    frame[base + 2] = 255; // B
                    frame[base + 3] = 128; // A
                }
            }
        }
        let (left, right) =
            split_stereo_frame_rgba(&frame, w, h, StereoLayout::LeftRight).expect("split");
        let merged = merge_stereo_frames_rgba(&left, &right, w / 2, h, StereoLayout::LeftRight)
            .expect("merge");
        assert_eq!(merged, frame);
    }

    #[test]
    fn rgba_odd_height_topbottom_error() {
        let frame = solid_frame_rgba(8, 7, 0, 0, 0, 255);
        assert!(split_stereo_frame_rgba(&frame, 8, 7, StereoLayout::TopBottom).is_err());
    }

    #[test]
    fn rgba_odd_width_leftright_error() {
        let frame = solid_frame_rgba(7, 4, 0, 0, 0, 255);
        assert!(split_stereo_frame_rgba(&frame, 7, 4, StereoLayout::LeftRight).is_err());
    }

    #[test]
    fn rgba_mono_unsupported() {
        let frame = solid_frame_rgba(8, 8, 0, 0, 0, 255);
        assert!(split_stereo_frame_rgba(&frame, 8, 8, StereoLayout::Mono).is_err());
        let l = solid_frame_rgba(8, 8, 0, 0, 0, 255);
        let r = solid_frame_rgba(8, 8, 0, 0, 0, 255);
        assert!(merge_stereo_frames_rgba(&l, &r, 8, 8, StereoLayout::Mono).is_err());
    }

    #[test]
    fn rgba_alternating_unsupported() {
        let frame = solid_frame_rgba(8, 8, 0, 0, 0, 255);
        assert!(split_stereo_frame_rgba(&frame, 8, 8, StereoLayout::Alternating).is_err());
    }

    #[test]
    fn rgba_buffer_too_small_error() {
        assert!(split_stereo_frame_rgba(&[0u8; 5], 64, 32, StereoLayout::TopBottom).is_err());
    }
}
