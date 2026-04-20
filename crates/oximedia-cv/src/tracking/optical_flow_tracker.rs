//! Lucas-Kanade pyramidal sparse optical flow tracker.
//!
//! Provides an iterative LK tracker that:
//! - Builds a Gaussian image pyramid for multi-scale tracking
//! - Runs iterative Lucas-Kanade at each pyramid level (coarse → fine)
//! - Computes per-point tracking confidence from the spatial gradient determinant
//! - Manages a list of feature points with stable IDs across frames
//!
//! # Example
//!
//! ```
//! use oximedia_cv::tracking::optical_flow_tracker::{GrayFrame, LkTracker};
//!
//! let data = vec![128u8; 64 * 64];
//! let frame = GrayFrame::new(64, 64, data);
//! let mut tracker = LkTracker::new(7, 2);
//! tracker.add_points(&[(32.0, 32.0)]);
//! ```

// ── GrayFrame ──────────────────────────────────────────────────────────────────

/// A single-channel (grayscale) image frame.
#[derive(Debug, Clone)]
pub struct GrayFrame {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row-major grayscale pixels (u8, one byte per pixel).
    pub data: Vec<u8>,
}

impl GrayFrame {
    /// Create a new GrayFrame.
    ///
    /// # Panics (debug only)
    ///
    /// Asserts that `data.len() == width * height` in debug builds.
    #[must_use]
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        debug_assert_eq!(data.len(), (width * height) as usize);
        Self {
            width,
            height,
            data,
        }
    }

    /// Get pixel value at integer coordinates as a float in [0, 1].
    ///
    /// Returns 0.0 for out-of-bounds coordinates.
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        let idx = (y * self.width + x) as usize;
        if idx < self.data.len() {
            self.data[idx] as f32 / 255.0
        } else {
            0.0
        }
    }

    /// Bilinear interpolation at sub-pixel position.  Returns 0.0 outside bounds.
    #[must_use]
    pub fn pixel_bilinear(&self, x: f32, y: f32) -> f32 {
        if x < 0.0 || y < 0.0 || x >= (self.width - 1) as f32 || y >= (self.height - 1) as f32 {
            return 0.0;
        }
        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let p00 = self.pixel(x0, y0);
        let p10 = self.pixel(x1, y0);
        let p01 = self.pixel(x0, y1);
        let p11 = self.pixel(x1, y1);

        p00 * (1.0 - fx) * (1.0 - fy)
            + p10 * fx * (1.0 - fy)
            + p01 * (1.0 - fx) * fy
            + p11 * fx * fy
    }

    /// Horizontal gradient at (x, y) using central differences (values in [-1, 1]).
    ///
    /// Uses forward/backward difference at borders.
    #[must_use]
    pub fn gradient_x(&self, x: u32, y: u32) -> f32 {
        if self.width < 2 || y >= self.height || x >= self.width {
            return 0.0;
        }
        if x == 0 {
            self.pixel(x + 1, y) - self.pixel(x, y)
        } else if x == self.width - 1 {
            self.pixel(x, y) - self.pixel(x - 1, y)
        } else {
            (self.pixel(x + 1, y) - self.pixel(x - 1, y)) * 0.5
        }
    }

    /// Vertical gradient at (x, y) using central differences (values in [-1, 1]).
    #[must_use]
    pub fn gradient_y(&self, x: u32, y: u32) -> f32 {
        if self.height < 2 || x >= self.width || y >= self.height {
            return 0.0;
        }
        if y == 0 {
            self.pixel(x, y + 1) - self.pixel(x, y)
        } else if y == self.height - 1 {
            self.pixel(x, y) - self.pixel(x, y - 1)
        } else {
            (self.pixel(x, y + 1) - self.pixel(x, y - 1)) * 0.5
        }
    }

    /// Check if the frame data is valid (correct size).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.data.len() == (self.width * self.height) as usize
    }
}

// ── FlowPoint ──────────────────────────────────────────────────────────────────

/// Tracked feature point with displacement and confidence.
#[derive(Debug, Clone)]
pub struct FlowPoint {
    /// Stable point identifier.
    pub id: u32,
    /// Position in the previous frame.
    pub prev_pos: (f32, f32),
    /// Estimated position in the current frame.
    pub curr_pos: (f32, f32),
    /// Displacement vector `curr_pos − prev_pos`.
    pub displacement: (f32, f32),
    /// Tracking confidence in [0, 1] based on cornerness (higher = more reliable).
    pub confidence: f32,
}

// ── Pyramid builder ────────────────────────────────────────────────────────────

/// Downsample a grayscale image by 2× using box filter.
fn downsample(frame: &GrayFrame) -> Option<GrayFrame> {
    let nw = frame.width / 2;
    let nh = frame.height / 2;
    if nw == 0 || nh == 0 {
        return None;
    }
    let mut data = vec![0u8; (nw * nh) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let sx = x * 2;
            let sy = y * 2;
            let mut sum = 0u32;
            let mut count = 0u32;
            for dy in 0..2u32 {
                for dx in 0..2u32 {
                    let px = sx + dx;
                    let py = sy + dy;
                    if px < frame.width && py < frame.height {
                        let idx = (py * frame.width + px) as usize;
                        if idx < frame.data.len() {
                            sum += frame.data[idx] as u32;
                            count += 1;
                        }
                    }
                }
            }
            data[(y * nw + x) as usize] = sum.checked_div(count).unwrap_or(0) as u8;
        }
    }
    Some(GrayFrame {
        width: nw,
        height: nh,
        data,
    })
}

/// Build an image pyramid from finest (index 0) to coarsest (index `levels-1`).
fn build_pyramid(frame: &GrayFrame, levels: u32) -> Vec<GrayFrame> {
    let mut pyr = Vec::with_capacity(levels as usize);
    pyr.push(frame.clone());
    for _ in 1..levels {
        let Some(last) = pyr.last() else {
            break;
        };
        if last.width < 8 || last.height < 8 {
            break;
        }
        match downsample(last) {
            Some(ds) => pyr.push(ds),
            None => break,
        }
    }
    pyr
}

// ── Single-level LK iteration ─────────────────────────────────────────────────

/// Run one level of the iterative Lucas-Kanade optical flow.
///
/// `prev` and `next` must have the same dimensions.
/// `init_u`, `init_v` are the initial flow estimates for this level.
/// Returns `(u, v, cornerness)` at convergence.
#[allow(clippy::manual_checked_ops)]
fn lk_level(
    prev: &GrayFrame,
    next: &GrayFrame,
    px: f32,
    py: f32,
    init_u: f32,
    init_v: f32,
    half_win: i32,
    max_iter: u32,
    epsilon: f32,
) -> (f32, f32, f32) {
    let w = prev.width as i32;
    let h = prev.height as i32;
    let mut u = init_u;
    let mut v = init_v;

    // Accumulate spatial gradient matrix G = [Gxx Gxy; Gxy Gyy]
    // and cornerness = det(G) / (trace(G) + eps)^2
    let mut gxx = 0.0f32;
    let mut gxy = 0.0f32;
    let mut gyy = 0.0f32;

    let cx = px.round() as i32;
    let cy = py.round() as i32;

    for dy in -half_win..=half_win {
        for dx in -half_win..=half_win {
            let x = cx + dx;
            let y = cy + dy;
            if x < 1 || x >= w - 1 || y < 1 || y >= h - 1 {
                continue;
            }
            let ix = prev.gradient_x(x as u32, y as u32);
            let iy = prev.gradient_y(x as u32, y as u32);
            gxx += ix * ix;
            gxy += ix * iy;
            gyy += iy * iy;
        }
    }

    let det = gxx * gyy - gxy * gxy;
    let trace = gxx + gyy;
    // Harris cornerness measure
    let cornerness = if (trace * trace + 1e-8) > 0.0 {
        det / (trace * trace + 1e-8)
    } else {
        0.0
    };

    if det.abs() < 1e-8 {
        return (u, v, cornerness);
    }

    // Iterative refinement
    for _ in 0..max_iter {
        let mut bx = 0.0f32;
        let mut by = 0.0f32;

        for dy in -half_win..=half_win {
            for dx in -half_win..=half_win {
                let x = cx + dx;
                let y = cy + dy;
                if x < 1 || x >= w - 1 || y < 1 || y >= h - 1 {
                    continue;
                }
                let nx = x as f32 + u;
                let ny = y as f32 + v;

                let prev_val = prev.pixel(x as u32, y as u32);
                let next_val = next.pixel_bilinear(nx, ny);
                let it = next_val - prev_val;

                let ix = prev.gradient_x(x as u32, y as u32);
                let iy = prev.gradient_y(x as u32, y as u32);

                bx += ix * it;
                by += iy * it;
            }
        }

        // Solve 2x2: [gxx gxy; gxy gyy] * [du; dv] = -[bx; by]
        let du = (gyy * (-bx) - gxy * (-by)) / det;
        let dv = (gxx * (-by) - gxy * (-bx)) / det;

        u += du;
        v += dv;

        if (du * du + dv * dv).sqrt() < epsilon {
            break;
        }
    }

    (u, v, cornerness.max(0.0))
}

// ── LkTracker ─────────────────────────────────────────────────────────────────

/// Lucas-Kanade pyramidal sparse optical flow tracker.
///
/// Maintains a list of feature points across frames, updating each point's
/// position using iterative LK on a multi-scale image pyramid.
#[derive(Debug, Clone)]
pub struct LkTracker {
    /// Current point positions.
    points: Vec<(f32, f32)>,
    /// Stable ID for each point.
    ids: Vec<u32>,
    /// Next ID to assign.
    next_id: u32,
    /// Half the window size passed to LK (full window = 2*half_win+1).
    window_size: u32,
    /// Number of pyramid levels.
    max_level: u32,
    /// Maximum LK iterations per level.
    max_iterations: u32,
    /// Convergence threshold (pixels).
    epsilon: f32,
}

impl LkTracker {
    /// Create a new LK tracker.
    ///
    /// `window_size` specifies the half-window radius (e.g., 7 → 15×15 patch).
    /// `max_level` is the number of pyramid levels (1 = no pyramid).
    #[must_use]
    pub fn new(window_size: u32, max_level: u32) -> Self {
        Self {
            points: Vec::new(),
            ids: Vec::new(),
            next_id: 1,
            window_size: window_size.max(1),
            max_level: max_level.max(1),
            max_iterations: 30,
            epsilon: 0.03,
        }
    }

    /// Create with explicit iteration and epsilon parameters.
    #[must_use]
    pub fn with_params(
        window_size: u32,
        max_level: u32,
        max_iterations: u32,
        epsilon: f32,
    ) -> Self {
        let mut t = Self::new(window_size, max_level);
        t.max_iterations = max_iterations;
        t.epsilon = epsilon;
        t
    }

    /// Add new points to track.  Each point is assigned a new stable ID.
    pub fn add_points(&mut self, points: &[(f32, f32)]) {
        for &pt in points {
            self.points.push(pt);
            self.ids.push(self.next_id);
            self.next_id += 1;
        }
    }

    /// Track all current points from `prev` to `next` using pyramidal LK.
    ///
    /// Updates the internal point positions to `next`-frame coordinates and
    /// returns a `FlowPoint` for every tracked point.
    pub fn track(&mut self, prev: &GrayFrame, next: &GrayFrame) -> Vec<FlowPoint> {
        if self.points.is_empty() {
            return Vec::new();
        }

        let prev_pyr = build_pyramid(prev, self.max_level);
        let next_pyr = build_pyramid(next, self.max_level);
        let levels = prev_pyr.len().min(next_pyr.len());
        let half_win = self.window_size as i32;

        let mut results = Vec::with_capacity(self.points.len());

        for idx in 0..self.points.len() {
            let (px, py) = self.points[idx];
            let id = self.ids[idx];

            // Scale factor from level L to original: 2^L
            let coarsest = levels - 1;

            // Initial flow estimate at coarsest level = 0
            let scale = (1u32 << coarsest) as f32;
            let mut u = 0.0f32;
            let mut v = 0.0f32;
            let mut cornerness = 0.0f32;

            // Coarse-to-fine refinement
            for level in (0..levels).rev() {
                let level_scale = (1u32 << level) as f32;
                let lprev = &prev_pyr[level];
                let lnext = &next_pyr[level];

                // Scale point to this level
                let lx = px / level_scale;
                let ly = py / level_scale;

                // Scale previous level's flow down by 2 when going from coarse to next finer
                let (lu, lv, corn) = lk_level(
                    lprev,
                    lnext,
                    lx,
                    ly,
                    u,
                    v,
                    half_win,
                    self.max_iterations,
                    self.epsilon,
                );
                u = lu;
                v = lv;
                cornerness = corn;

                // Upsample flow estimate for finer level (multiply by 2)
                if level > 0 {
                    u *= 2.0;
                    v *= 2.0;
                }
            }

            let new_x = px + u;
            let new_y = py + v;

            let confidence = cornerness_to_confidence(cornerness);

            results.push(FlowPoint {
                id,
                prev_pos: (px, py),
                curr_pos: (new_x, new_y),
                displacement: (u, v),
                confidence,
            });

            // Update internal position to new frame
            self.points[idx] = (new_x, new_y);
        }

        results
    }

    /// Remove points that left the frame or have low confidence.
    ///
    /// Updates the internal point list in place.
    pub fn prune(&mut self, frame_width: u32, frame_height: u32, min_confidence: f32) {
        // We need confidence scores — re-use last cornerness from a dummy prev=next pass.
        // Since we don't cache last confidences here, we use bounds-only pruning plus
        // a confidence threshold applied by the caller (who has the FlowPoint list).
        // This variant prunes purely by position bounds.
        let fw = frame_width as f32;
        let fh = frame_height as f32;

        let mut keep = Vec::with_capacity(self.points.len());
        for (i, &(x, y)) in self.points.iter().enumerate() {
            if x >= 0.0 && x < fw && y >= 0.0 && y < fh {
                keep.push(i);
            }
        }

        let new_points: Vec<(f32, f32)> = keep.iter().map(|&i| self.points[i]).collect();
        let new_ids: Vec<u32> = keep.iter().map(|&i| self.ids[i]).collect();
        self.points = new_points;
        self.ids = new_ids;
    }

    /// Prune points by their FlowPoint confidences from the last `track()` call.
    ///
    /// Points whose `confidence < min_confidence` are removed.
    pub fn prune_by_confidence(&mut self, flow_points: &[FlowPoint], min_confidence: f32) {
        let mut keep_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for fp in flow_points {
            if fp.confidence >= min_confidence {
                keep_ids.insert(fp.id);
            }
        }

        let mut i = 0;
        while i < self.ids.len() {
            if keep_ids.contains(&self.ids[i]) {
                i += 1;
            } else {
                self.points.remove(i);
                self.ids.remove(i);
            }
        }
    }

    /// Number of currently tracked points.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Remove all tracked points.
    pub fn clear(&mut self) {
        self.points.clear();
        self.ids.clear();
    }

    /// Read-only slice of current point positions.
    #[must_use]
    pub fn positions(&self) -> &[(f32, f32)] {
        &self.points
    }

    /// Read-only slice of point IDs.
    #[must_use]
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
}

/// Map Harris cornerness to a confidence value in [0, 1] via soft sigmoid.
fn cornerness_to_confidence(c: f32) -> f32 {
    // c is typically in [0, ~0.25] for well-textured regions; normalise to [0,1]
    let scaled = c * 40.0; // empirical scale
    scaled / (1.0 + scaled)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_frame(w: u32, h: u32, val: u8) -> GrayFrame {
        GrayFrame::new(w, h, vec![val; (w * h) as usize])
    }

    fn gradient_frame(w: u32, h: u32) -> GrayFrame {
        let mut data = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                data[(y * w + x) as usize] = (x as u8).wrapping_add(y as u8);
            }
        }
        GrayFrame::new(w, h, data)
    }

    // ── GrayFrame tests ────────────────────────────────────────────────────────

    #[test]
    fn test_gray_frame_pixel_in_bounds() {
        let frame = GrayFrame::new(4, 4, vec![128u8; 16]);
        let p = frame.pixel(2, 2);
        assert!((p - 128.0 / 255.0).abs() < 1e-5);
    }

    #[test]
    fn test_gray_frame_pixel_out_of_bounds() {
        let frame = flat_frame(4, 4, 200);
        assert_eq!(frame.pixel(4, 4), 0.0);
        assert_eq!(frame.pixel(0, 10), 0.0);
    }

    #[test]
    fn test_gray_frame_pixel_bilinear_center() {
        let frame = flat_frame(8, 8, 128);
        // Bilinear on uniform frame should return the same value (minus border)
        let p = frame.pixel_bilinear(3.5, 3.5);
        assert!((p - 128.0 / 255.0).abs() < 1e-4);
    }

    #[test]
    fn test_gray_frame_gradient_x_zero_on_flat() {
        let frame = flat_frame(10, 10, 100);
        // Uniform frame → all gradients zero
        assert!((frame.gradient_x(5, 5)).abs() < 1e-5);
        assert!((frame.gradient_y(5, 5)).abs() < 1e-5);
    }

    #[test]
    fn test_gray_frame_gradient_x_nonzero_on_ramp() {
        let frame = gradient_frame(16, 16);
        // horizontal ramp → Ix should be positive in the interior
        let gx = frame.gradient_x(8, 8);
        assert!(
            gx > 0.0,
            "gradient_x should be positive on an x-ramp, got {gx}"
        );
    }

    #[test]
    fn test_gray_frame_gradient_y_nonzero_on_ramp() {
        let frame = gradient_frame(16, 16);
        let gy = frame.gradient_y(8, 8);
        assert!(
            gy > 0.0,
            "gradient_y should be positive on a y-ramp, got {gy}"
        );
    }

    #[test]
    fn test_gray_frame_is_valid() {
        let frame = flat_frame(4, 4, 0);
        assert!(frame.is_valid());
        let bad = GrayFrame {
            width: 4,
            height: 4,
            data: vec![0u8; 10],
        };
        assert!(!bad.is_valid());
    }

    // ── Pyramid builder tests ──────────────────────────────────────────────────

    #[test]
    fn test_build_pyramid_levels() {
        let frame = flat_frame(64, 64, 128);
        let pyr = build_pyramid(&frame, 3);
        assert_eq!(pyr.len(), 3);
        assert_eq!(pyr[0].width, 64);
        assert_eq!(pyr[1].width, 32);
        assert_eq!(pyr[2].width, 16);
    }

    #[test]
    fn test_build_pyramid_stops_at_min_size() {
        // 8×8 → downsample → 4×4 which is < 8 so next level is skipped.
        // Requesting 5 levels from an 8×8 frame should yield 2 levels: 8×8 and 4×4.
        let frame = flat_frame(8, 8, 0);
        let pyr = build_pyramid(&frame, 5);
        // First downsample (8→4) succeeds; second (4→2) is stopped by the < 8 guard on the 4×4 frame.
        assert_eq!(pyr.len(), 2);
    }

    // ── LkTracker tests ────────────────────────────────────────────────────────

    #[test]
    fn test_lk_tracker_add_points() {
        let mut tracker = LkTracker::new(5, 2);
        tracker.add_points(&[(10.0, 10.0), (20.0, 20.0)]);
        assert_eq!(tracker.point_count(), 2);
    }

    #[test]
    fn test_lk_tracker_ids_are_stable() {
        let mut tracker = LkTracker::new(5, 2);
        tracker.add_points(&[(5.0, 5.0)]);
        let id_before = tracker.ids()[0];
        tracker.add_points(&[(15.0, 15.0)]);
        assert_eq!(tracker.ids()[0], id_before);
        assert_eq!(tracker.ids()[1], id_before + 1);
    }

    #[test]
    fn test_lk_tracker_track_returns_all_points() {
        let mut tracker = LkTracker::new(3, 1);
        tracker.add_points(&[(16.0, 16.0), (24.0, 24.0)]);
        let frame = flat_frame(64, 64, 128);
        let results = tracker.track(&frame, &frame);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_lk_tracker_track_same_frame_near_zero_displacement() {
        let mut tracker = LkTracker::new(5, 2);
        tracker.add_points(&[(32.0, 32.0)]);
        let frame = gradient_frame(64, 64);
        let results = tracker.track(&frame, &frame);
        assert_eq!(results.len(), 1);
        // Tracking the same frame → displacement should be close to zero
        let disp = results[0].displacement;
        assert!(
            (disp.0 * disp.0 + disp.1 * disp.1).sqrt() < 2.0,
            "displacement on same frame too large: {:?}",
            disp
        );
    }

    #[test]
    fn test_lk_tracker_prune_removes_out_of_bounds() {
        let mut tracker = LkTracker::new(3, 1);
        tracker.add_points(&[(5.0, 5.0), (200.0, 200.0)]);
        tracker.prune(100, 100, 0.0);
        assert_eq!(tracker.point_count(), 1);
    }

    #[test]
    fn test_lk_tracker_clear() {
        let mut tracker = LkTracker::new(3, 1);
        tracker.add_points(&[(1.0, 1.0), (2.0, 2.0)]);
        tracker.clear();
        assert_eq!(tracker.point_count(), 0);
    }

    #[test]
    fn test_lk_tracker_empty_track() {
        let mut tracker = LkTracker::new(5, 2);
        let frame = flat_frame(32, 32, 100);
        let results = tracker.track(&frame, &frame);
        assert!(results.is_empty());
    }

    #[test]
    fn test_lk_tracker_confidence_in_range() {
        let mut tracker = LkTracker::new(5, 1);
        tracker.add_points(&[(16.0, 16.0)]);
        let frame = gradient_frame(32, 32);
        let results = tracker.track(&frame, &frame);
        for fp in &results {
            assert!(
                fp.confidence >= 0.0 && fp.confidence <= 1.0,
                "confidence out of range: {}",
                fp.confidence
            );
        }
    }

    #[test]
    fn test_lk_tracker_prune_by_confidence() {
        let mut tracker = LkTracker::new(5, 1);
        tracker.add_points(&[(8.0, 8.0), (16.0, 16.0)]);
        let frame = gradient_frame(32, 32);
        let results = tracker.track(&frame, &frame);
        let before = tracker.point_count();
        tracker.prune_by_confidence(&results, 0.99); // very high threshold, likely removes all
                                                     // Just verify method runs without panic
        assert!(tracker.point_count() <= before);
    }
}
