//! AI-guided smart reframing for aspect ratio conversion.
//!
//! Provides saliency-based crop window computation, trajectory smoothing,
//! and interpolated reframe sequences for converting between aspect ratios
//! (e.g., 16:9 landscape → 9:16 portrait) without losing key content.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Saliency map
// ---------------------------------------------------------------------------

/// Per-pixel importance map for a single video frame.
///
/// Values are in `[0, 1]` with higher values indicating more visually
/// important regions. Stored in row-major (y, x) order.
#[derive(Debug, Clone)]
pub struct SaliencyMap {
    /// Width of the source frame in pixels.
    pub width: u32,
    /// Height of the source frame in pixels.
    pub height: u32,
    /// Importance value for each pixel, in row-major order.
    ///
    /// Length must equal `width * height`.
    pub values: Vec<f32>,
}

impl SaliencyMap {
    /// Create a new saliency map filled with uniform importance.
    pub fn uniform(width: u32, height: u32) -> Self {
        let n = (width * height) as usize;
        Self {
            width,
            height,
            values: vec![1.0 / n as f32; n],
        }
    }

    /// Create a saliency map from a raw values vector.
    ///
    /// Returns `None` if the vector length does not match `width × height`.
    pub fn from_values(width: u32, height: u32, values: Vec<f32>) -> Option<Self> {
        if values.len() != (width * height) as usize {
            return None;
        }
        Some(Self {
            width,
            height,
            values,
        })
    }

    /// Get the saliency value at pixel `(x, y)`.
    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        let idx = (y * self.width + x) as usize;
        self.values.get(idx).copied().unwrap_or(0.0)
    }

    /// Total saliency sum.
    pub fn total(&self) -> f32 {
        self.values.iter().sum()
    }

    /// Compute the weighted centroid (cx, cy) of the saliency map.
    ///
    /// Returns `(width/2, height/2)` if total saliency is zero.
    pub fn weighted_centroid(&self) -> (f32, f32) {
        let total = self.total();
        if total < 1e-10 {
            return (self.width as f32 / 2.0, self.height as f32 / 2.0);
        }

        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        for y in 0..self.height {
            for x in 0..self.width {
                let v = self.get(x, y) as f64;
                sum_x += v * x as f64;
                sum_y += v * y as f64;
            }
        }

        ((sum_x / total as f64) as f32, (sum_y / total as f64) as f32)
    }

    /// Compute the bounding box of pixels whose saliency is in the top
    /// `top_fraction` (e.g., 0.20 for top 20%).
    ///
    /// Returns `(min_x, min_y, max_x, max_y)` or the full frame bounds
    /// if no pixels exceed the threshold.
    pub fn top_fraction_bounds(&self, top_fraction: f32) -> (u32, u32, u32, u32) {
        if self.values.is_empty() {
            return (
                0,
                0,
                self.width.saturating_sub(1),
                self.height.saturating_sub(1),
            );
        }

        // Find the threshold value
        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cutoff_idx = ((1.0 - top_fraction.clamp(0.0, 1.0)) * sorted.len() as f32) as usize;
        let threshold = sorted.get(cutoff_idx).copied().unwrap_or(0.0);

        let mut min_x = self.width;
        let mut min_y = self.height;
        let mut max_x = 0u32;
        let mut max_y = 0u32;
        let mut found = false;

        for y in 0..self.height {
            for x in 0..self.width {
                if self.get(x, y) >= threshold {
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                    found = true;
                }
            }
        }

        if !found {
            (
                0,
                0,
                self.width.saturating_sub(1),
                self.height.saturating_sub(1),
            )
        } else {
            (min_x, min_y, max_x, max_y)
        }
    }
}

// ---------------------------------------------------------------------------
// Reframe types
// ---------------------------------------------------------------------------

/// The target aspect ratio and output resolution for reframing.
#[derive(Debug, Clone, Copy)]
pub struct ReframeTarget {
    /// Numerator of the target aspect ratio (e.g., 9 for 9:16).
    pub aspect_width: u32,
    /// Denominator of the target aspect ratio (e.g., 16 for 9:16).
    pub aspect_height: u32,
    /// Desired output pixel width.
    pub output_width: u32,
    /// Desired output pixel height.
    pub output_height: u32,
}

impl ReframeTarget {
    /// Create a standard 9:16 (vertical/portrait) target at 1080×1920.
    pub fn vertical_1080p() -> Self {
        Self {
            aspect_width: 9,
            aspect_height: 16,
            output_width: 1080,
            output_height: 1920,
        }
    }

    /// Create a standard 16:9 (landscape) target at 1920×1080.
    pub fn landscape_1080p() -> Self {
        Self {
            aspect_width: 16,
            aspect_height: 9,
            output_width: 1920,
            output_height: 1080,
        }
    }

    /// Create a 1:1 (square) target at 1080×1080.
    pub fn square_1080() -> Self {
        Self {
            aspect_width: 1,
            aspect_height: 1,
            output_width: 1080,
            output_height: 1080,
        }
    }

    /// Compute the crop window size needed to fill this target from a source
    /// of `(src_w, src_h)`.
    ///
    /// Returns `(crop_w, crop_h)`.
    pub fn crop_size_for_source(&self, src_w: u32, src_h: u32) -> (u32, u32) {
        // The crop region must fit inside the source and have the target ratio.
        let target_ratio = self.aspect_width as f64 / self.aspect_height as f64;
        let source_ratio = src_w as f64 / src_h as f64;

        if target_ratio < source_ratio {
            // Source is wider than target → crop width
            let crop_h = src_h;
            let crop_w = (crop_h as f64 * target_ratio).round() as u32;
            (crop_w.min(src_w), crop_h)
        } else {
            // Source is taller than target → crop height
            let crop_w = src_w;
            let crop_h = (crop_w as f64 / target_ratio).round() as u32;
            (crop_w, crop_h.min(src_h))
        }
    }
}

/// A rectangular crop window within a source frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReframeWindow {
    /// Left edge of the crop window in source pixels.
    pub x: i32,
    /// Top edge of the crop window in source pixels.
    pub y: i32,
    /// Width of the crop window in source pixels.
    pub width: u32,
    /// Height of the crop window in source pixels.
    pub height: u32,
}

impl ReframeWindow {
    /// Create a center crop for a given source and crop size.
    pub fn center(src_w: u32, src_h: u32, crop_w: u32, crop_h: u32) -> Self {
        let x = ((src_w as i32 - crop_w as i32) / 2).max(0);
        let y = ((src_h as i32 - crop_h as i32) / 2).max(0);
        Self {
            x,
            y,
            width: crop_w.min(src_w),
            height: crop_h.min(src_h),
        }
    }

    /// Clamp the window to stay within source bounds.
    pub fn clamped(mut self, src_w: u32, src_h: u32) -> Self {
        self.x = self.x.max(0).min(src_w.saturating_sub(self.width) as i32);
        self.y = self.y.max(0).min(src_h.saturating_sub(self.height) as i32);
        self.width = self.width.min(src_w - self.x as u32);
        self.height = self.height.min(src_h - self.y as u32);
        self
    }

    /// Interpolate (bilinear blend) between two windows at factor `t` (0–1).
    pub fn lerp(a: &ReframeWindow, b: &ReframeWindow, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let x = lerp_i32(a.x, b.x, t);
        let y = lerp_i32(a.y, b.y, t);
        let width = lerp_u32(a.width, b.width, t);
        let height = lerp_u32(a.height, b.height, t);
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

fn lerp_i32(a: i32, b: i32, t: f32) -> i32 {
    (a as f32 + (b - a) as f32 * t).round() as i32
}

fn lerp_u32(a: u32, b: u32, t: f32) -> u32 {
    (a as f32 + (b as f32 - a as f32) * t).round() as u32
}

// ---------------------------------------------------------------------------
// SmartReframer
// ---------------------------------------------------------------------------

/// Computes saliency-driven crop windows and smooth reframe trajectories.
#[derive(Debug, Clone)]
pub struct SmartReframer {
    /// Optional saliency map for the current frame.
    pub saliency_map: Option<SaliencyMap>,
    /// Smoothing factor for EMA trajectory (0 = no smoothing, 1 = full lag).
    pub motion_smoothing: f32,
}

impl SmartReframer {
    /// Create a new reframer with optional saliency and a smoothing coefficient.
    pub fn new(saliency_map: Option<SaliencyMap>, motion_smoothing: f32) -> Self {
        Self {
            saliency_map,
            motion_smoothing: motion_smoothing.clamp(0.0, 1.0),
        }
    }

    /// Compute the optimal crop window for a given source frame and target.
    ///
    /// - If no saliency map is provided: returns a centered crop.
    /// - With saliency: finds the bounding box of the top 20% salient pixels,
    ///   centers the crop window on the weighted centroid of that region, and
    ///   clamps to source bounds.
    pub fn compute_optimal_crop(
        src_w: u32,
        src_h: u32,
        saliency: &SaliencyMap,
        target: &ReframeTarget,
    ) -> ReframeWindow {
        let (crop_w, crop_h) = target.crop_size_for_source(src_w, src_h);

        // Get the bounding box of the top 20% salient pixels
        let (min_x, min_y, max_x, max_y) = saliency.top_fraction_bounds(0.20);
        let region_cx = ((min_x + max_x) as f32 / 2.0).round();
        let region_cy = ((min_y + max_y) as f32 / 2.0).round();

        // Also use the weighted centroid within that region for fine-grained centering
        let (centroid_x, centroid_y) = saliency.weighted_centroid();
        // Blend: 60% bounding-box center, 40% full-map centroid
        let focus_x = 0.6 * region_cx + 0.4 * centroid_x;
        let focus_y = 0.6 * region_cy + 0.4 * centroid_y;

        // Center crop window on focus point
        let x = (focus_x - crop_w as f32 / 2.0).round() as i32;
        let y = (focus_y - crop_h as f32 / 2.0).round() as i32;

        ReframeWindow {
            x,
            y,
            width: crop_w,
            height: crop_h,
        }
        .clamped(src_w, src_h)
    }

    /// Apply exponential moving average smoothing to a trajectory of windows.
    ///
    /// `alpha` in `[0, 1]`: higher values retain more of the previous position
    /// (more smoothing / more lag). At `alpha = 0`, returns the input unchanged.
    pub fn smooth_trajectory(windows: &[ReframeWindow], alpha: f32) -> Vec<ReframeWindow> {
        if windows.is_empty() {
            return Vec::new();
        }
        let alpha = alpha.clamp(0.0, 1.0);
        if alpha < 1e-6 {
            return windows.to_vec();
        }

        let mut smoothed = Vec::with_capacity(windows.len());
        let mut prev = windows[0];
        smoothed.push(prev);

        for &current in &windows[1..] {
            let sx = (prev.x as f32 * alpha + current.x as f32 * (1.0 - alpha)).round() as i32;
            let sy = (prev.y as f32 * alpha + current.y as f32 * (1.0 - alpha)).round() as i32;
            let sw =
                (prev.width as f32 * alpha + current.width as f32 * (1.0 - alpha)).round() as u32;
            let sh =
                (prev.height as f32 * alpha + current.height as f32 * (1.0 - alpha)).round() as u32;
            let w = ReframeWindow {
                x: sx,
                y: sy,
                width: sw,
                height: sh,
            };
            smoothed.push(w);
            prev = w;
        }

        smoothed
    }
}

// ---------------------------------------------------------------------------
// Saliency detection
// ---------------------------------------------------------------------------

/// Compute a simple saliency map from a raw `[u8]` frame using Laplacian
/// magnitude as a proxy for visual interest (edges and high-frequency detail).
///
/// `frame` is expected to be a grayscale (luma-only) byte buffer in row-major
/// order of length `width × height`. If a multi-channel buffer is passed, only
/// the first channel value at each pixel position `i * channels` is used — the
/// caller should extract luma first.
pub fn detect_salient_regions(frame: &[u8], width: u32, height: u32) -> SaliencyMap {
    let w = width as usize;
    let h = height as usize;
    let expected_len = w * h;

    if frame.len() < expected_len || w < 3 || h < 3 {
        // Fall back to uniform saliency for undersized inputs
        return SaliencyMap::uniform(width, height);
    }

    let mut laplacian = vec![0.0f32; w * h];
    let mut max_val = 0.0f32;

    // Discrete Laplacian kernel: [0,1,0; 1,-4,1; 0,1,0]
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let idx = y * w + x;
            let center = frame[idx] as f32;
            let north = frame[(y - 1) * w + x] as f32;
            let south = frame[(y + 1) * w + x] as f32;
            let west = frame[y * w + (x - 1)] as f32;
            let east = frame[y * w + (x + 1)] as f32;

            let lap = (north + south + west + east - 4.0 * center).abs();
            laplacian[idx] = lap;
            if lap > max_val {
                max_val = lap;
            }
        }
    }

    // Normalize to [0, 1]
    if max_val > 1e-6 {
        for v in &mut laplacian {
            *v /= max_val;
        }
    }

    SaliencyMap {
        width,
        height,
        values: laplacian,
    }
}

// ---------------------------------------------------------------------------
// ReframeSequence
// ---------------------------------------------------------------------------

/// A sequence of keyframe crop windows that can be interpolated at arbitrary
/// frame positions.
#[derive(Debug, Clone)]
pub struct ReframeSequence {
    /// Keyframe windows. Each entry is `(frame_index, ReframeWindow)`.
    /// Must be sorted by `frame_index` in ascending order.
    pub frames: Vec<(u64, ReframeWindow)>,
    /// Total number of frames in the sequence.
    pub total_frames: u64,
}

impl ReframeSequence {
    /// Create a new sequence with the given keyframes and total frame count.
    ///
    /// Keyframes are sorted internally by frame index.
    pub fn new(mut frames: Vec<(u64, ReframeWindow)>, total_frames: u64) -> Self {
        frames.sort_by_key(|(idx, _)| *idx);
        Self {
            frames,
            total_frames,
        }
    }

    /// Interpolate the crop window at `frame_idx` using bilinear interpolation
    /// between the surrounding keyframes.
    ///
    /// - If `frame_idx` is before the first keyframe, returns the first keyframe.
    /// - If `frame_idx` is after the last keyframe, returns the last keyframe.
    /// - Otherwise, bilinearly interpolates between the bracketing keyframes.
    pub fn interpolate_at(&self, frame_idx: u64) -> ReframeWindow {
        if self.frames.is_empty() {
            return ReframeWindow {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            };
        }

        let first = self
            .frames
            .first()
            .map(|(_, w)| *w)
            .unwrap_or(ReframeWindow {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            });
        let last = self.frames.last().map(|(_, w)| *w).unwrap_or(first);

        if frame_idx <= self.frames[0].0 {
            return first;
        }
        if frame_idx >= self.frames[self.frames.len() - 1].0 {
            return last;
        }

        // Binary search for the surrounding keyframes
        let pos = self.frames.partition_point(|(idx, _)| *idx <= frame_idx);

        if pos == 0 {
            return first;
        }
        if pos >= self.frames.len() {
            return last;
        }

        let (prev_idx, prev_w) = self.frames[pos - 1];
        let (next_idx, next_w) = self.frames[pos];

        let span = (next_idx - prev_idx) as f32;
        if span < 1.0 {
            return prev_w;
        }

        let t = (frame_idx - prev_idx) as f32 / span;
        ReframeWindow::lerp(&prev_w, &next_w, t)
    }

    /// Number of keyframes in the sequence.
    pub fn keyframe_count(&self) -> usize {
        self.frames.len()
    }
}

// ---------------------------------------------------------------------------
// SubjectTracker — multi-frame subject tracking for smooth panning
// ---------------------------------------------------------------------------

/// A bounding box for a tracked subject within a frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubjectBounds {
    /// Center X in source pixels.
    pub cx: f32,
    /// Center Y in source pixels.
    pub cy: f32,
    /// Bounding-box width in source pixels.
    pub width: f32,
    /// Bounding-box height in source pixels.
    pub height: f32,
    /// Detection confidence `[0, 1]`.
    pub confidence: f32,
}

impl SubjectBounds {
    /// Create bounds, clamping `confidence` to `[0, 1]`.
    #[must_use]
    pub fn new(cx: f32, cy: f32, width: f32, height: f32, confidence: f32) -> Self {
        Self {
            cx,
            cy,
            width,
            height,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Return the centre `(cx, cy)` as the recommended crop focus point.
    #[must_use]
    pub fn focus_point(&self) -> (f32, f32) {
        (self.cx, self.cy)
    }
}

/// Tracks a subject across multiple frames using an exponential moving average
/// (EMA) to produce a smooth panning trajectory for reframing.
#[derive(Debug, Clone)]
pub struct SubjectTracker {
    /// EMA smoothing factor in `[0, 1]`. Higher = more lag / smoother.
    pub smoothing: f32,
    /// Detections below this threshold are ignored.
    pub min_confidence: f32,
    history: Vec<(u64, SubjectBounds)>,
    ema_cx: Option<f32>,
    ema_cy: Option<f32>,
}

impl SubjectTracker {
    /// Create a new tracker.
    #[must_use]
    pub fn new(smoothing: f32, min_confidence: f32) -> Self {
        Self {
            smoothing: smoothing.clamp(0.0, 1.0),
            min_confidence: min_confidence.clamp(0.0, 1.0),
            history: Vec::new(),
            ema_cx: None,
            ema_cy: None,
        }
    }

    /// Feed a new detection at `frame_idx`.
    ///
    /// Detections below `self.min_confidence` carry the previous EMA forward.
    pub fn update(&mut self, frame_idx: u64, bounds: SubjectBounds) {
        if bounds.confidence < self.min_confidence {
            if let (Some(ex), Some(ey)) = (self.ema_cx, self.ema_cy) {
                let synth = SubjectBounds::new(ex, ey, bounds.width, bounds.height, 0.0);
                self.history.push((frame_idx, synth));
            }
            return;
        }
        let new_cx = self.ema_cx.map_or(bounds.cx, |p| {
            p * self.smoothing + bounds.cx * (1.0 - self.smoothing)
        });
        let new_cy = self.ema_cy.map_or(bounds.cy, |p| {
            p * self.smoothing + bounds.cy * (1.0 - self.smoothing)
        });
        self.ema_cx = Some(new_cx);
        self.ema_cy = Some(new_cy);
        self.history.push((
            frame_idx,
            SubjectBounds::new(
                new_cx,
                new_cy,
                bounds.width,
                bounds.height,
                bounds.confidence,
            ),
        ));
    }

    /// Clear all history and reset the EMA.
    pub fn reset(&mut self) {
        self.history.clear();
        self.ema_cx = None;
        self.ema_cy = None;
    }

    /// Borrow the raw tracking history.
    #[must_use]
    pub fn history(&self) -> &[(u64, SubjectBounds)] {
        &self.history
    }

    /// Return the latest EMA position `(cx, cy)`, if any.
    #[must_use]
    pub fn current_position(&self) -> Option<(f32, f32)> {
        self.ema_cx.zip(self.ema_cy)
    }

    /// Build a [`ReframeSequence`] centered on tracked positions.
    #[must_use]
    pub fn generate_sequence(
        &self,
        src_w: u32,
        src_h: u32,
        target: &ReframeTarget,
        total_frames: u64,
    ) -> ReframeSequence {
        let (crop_w, crop_h) = target.crop_size_for_source(src_w, src_h);
        let kfs = self
            .history
            .iter()
            .map(|(idx, b)| {
                let (fx, fy) = b.focus_point();
                let w = ReframeWindow {
                    x: (fx - crop_w as f32 / 2.0).round() as i32,
                    y: (fy - crop_h as f32 / 2.0).round() as i32,
                    width: crop_w,
                    height: crop_h,
                }
                .clamped(src_w, src_h);
                (*idx, w)
            })
            .collect();
        ReframeSequence::new(kfs, total_frames)
    }

    /// Build a smooth [`ReframeSequence`] with an extra trajectory-smoothing pass.
    #[must_use]
    pub fn generate_smooth_sequence(
        &self,
        src_w: u32,
        src_h: u32,
        target: &ReframeTarget,
        total_frames: u64,
        trajectory_smoothing: f32,
    ) -> ReframeSequence {
        let raw = self.generate_sequence(src_w, src_h, target, total_frames);
        let wins: Vec<ReframeWindow> = raw.frames.iter().map(|(_, w)| *w).collect();
        let smoothed = SmartReframer::smooth_trajectory(&wins, trajectory_smoothing);
        let kfs = raw
            .frames
            .iter()
            .zip(smoothed.iter())
            .map(|((idx, _), w)| (*idx, *w))
            .collect();
        ReframeSequence::new(kfs, total_frames)
    }
}

// ---------------------------------------------------------------------------
// Vertical-to-horizontal reframing
// ---------------------------------------------------------------------------

/// Orientation of a video frame derived from its dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameOrientation {
    /// Width > Height.
    Landscape,
    /// Height > Width.
    Portrait,
    /// Width == Height.
    Square,
}

impl FrameOrientation {
    /// Detect from pixel dimensions.
    #[must_use]
    pub fn from_dimensions(width: u32, height: u32) -> Self {
        match width.cmp(&height) {
            std::cmp::Ordering::Greater => Self::Landscape,
            std::cmp::Ordering::Less => Self::Portrait,
            std::cmp::Ordering::Equal => Self::Square,
        }
    }
}

/// Strategy for filling the side bars when placing a portrait source in a
/// landscape output canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerticalToHorizontalStrategy {
    /// Black pillarboxes.
    Pillarbox,
    /// Blurred / stretched fill (common on platforms).
    BlurredSides,
    /// Horizontal stretch (distorts).
    Stretch,
    /// Mirror the left/right edges.
    MirrorSides,
    /// Saliency-guided dual crop.
    SmartCrop,
}

/// Parameters for converting a portrait (vertical) source frame to a
/// landscape (horizontal) output canvas.
#[derive(Debug, Clone)]
pub struct VerticalToHorizontalParams {
    /// Source frame width.
    pub src_width: u32,
    /// Source frame height.
    pub src_height: u32,
    /// Output canvas width.
    pub output_width: u32,
    /// Output canvas height.
    pub output_height: u32,
    /// Conversion strategy.
    pub strategy: VerticalToHorizontalStrategy,
}

impl VerticalToHorizontalParams {
    /// Convenience preset: 9:16 (1080×1920) → 16:9 (1920×1080).
    #[must_use]
    pub fn vertical_to_widescreen() -> Self {
        Self {
            src_width: 1080,
            src_height: 1920,
            output_width: 1920,
            output_height: 1080,
            strategy: VerticalToHorizontalStrategy::BlurredSides,
        }
    }

    /// Create with explicit dimensions and strategy.
    #[must_use]
    pub fn new(
        src_width: u32,
        src_height: u32,
        output_width: u32,
        output_height: u32,
        strategy: VerticalToHorizontalStrategy,
    ) -> Self {
        Self {
            src_width,
            src_height,
            output_width,
            output_height,
            strategy,
        }
    }

    /// Placement of the primary (subject) area in the output canvas.
    ///
    /// Returns `(out_x, out_y, placed_width, placed_height)` in output pixels.
    #[must_use]
    pub fn primary_placement(&self) -> (i32, i32, u32, u32) {
        let scale = self.output_height as f64 / self.src_height as f64;
        let placed_w = (self.src_width as f64 * scale).round() as u32;
        let placed_h = self.output_height;
        let out_x = ((self.output_width as i32 - placed_w as i32) / 2).max(0);
        (out_x, 0, placed_w.min(self.output_width), placed_h)
    }

    /// Side-bar regions in output coordinates.
    ///
    /// Returns `(left_x, left_w, right_x, right_w)`.
    #[must_use]
    pub fn side_regions(&self) -> (u32, u32, u32, u32) {
        let (out_x, _, placed_w, _) = self.primary_placement();
        let left_w = out_x as u32;
        let right_x = (out_x as u32 + placed_w).min(self.output_width);
        let right_w = self.output_width.saturating_sub(right_x);
        (0, left_w, right_x, right_w)
    }

    /// Saliency-guided crop window (in *source* coordinates) for the primary area.
    #[must_use]
    pub fn saliency_crop_window(&self, saliency: &SaliencyMap) -> ReframeWindow {
        let scale = self.output_height as f64 / self.src_height as f64;
        let placed_w = (self.src_width as f64 * scale).round() as u32;
        if placed_w >= self.output_width {
            let target = ReframeTarget {
                aspect_width: self.output_width,
                aspect_height: self.output_height,
                output_width: self.output_width,
                output_height: self.output_height,
            };
            SmartReframer::compute_optimal_crop(self.src_width, self.src_height, saliency, &target)
        } else {
            ReframeWindow {
                x: 0,
                y: 0,
                width: self.src_width,
                height: self.src_height,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_window(x: i32, y: i32, w: u32, h: u32) -> ReframeWindow {
        ReframeWindow {
            x,
            y,
            width: w,
            height: h,
        }
    }

    // -- SaliencyMap tests --

    #[test]
    fn test_saliency_map_uniform_sum() {
        let s = SaliencyMap::uniform(10, 10);
        assert_eq!(s.values.len(), 100);
        assert!((s.total() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_saliency_map_from_values_mismatch() {
        let result = SaliencyMap::from_values(10, 10, vec![0.0; 50]);
        assert!(result.is_none());
    }

    #[test]
    fn test_saliency_map_get_oob() {
        let s = SaliencyMap::uniform(4, 4);
        assert_eq!(s.get(10, 10), 0.0);
    }

    #[test]
    fn test_saliency_weighted_centroid_uniform() {
        let s = SaliencyMap::uniform(10, 10);
        let (cx, cy) = s.weighted_centroid();
        // Uniform → centroid at center
        assert!((cx - 4.5).abs() < 0.5, "cx={cx}");
        assert!((cy - 4.5).abs() < 0.5, "cy={cy}");
    }

    #[test]
    fn test_saliency_top_fraction_bounds() {
        // Put high saliency only in top-left 2×2 of a 10×10 map
        let mut vals = vec![0.0f32; 100];
        vals[0] = 1.0; // (0,0)
        vals[1] = 1.0; // (1,0)
        vals[10] = 1.0; // (0,1)
        vals[11] = 1.0; // (1,1)
        let s = SaliencyMap::from_values(10, 10, vals).expect("from values should succeed");
        let (min_x, min_y, max_x, max_y) = s.top_fraction_bounds(0.04);
        assert_eq!(min_x, 0);
        assert_eq!(min_y, 0);
        assert!(max_x <= 1);
        assert!(max_y <= 1);
    }

    // -- ReframeTarget tests --

    #[test]
    fn test_crop_size_for_source_16_to_9() {
        // Source 1920×1080 (16:9), target 9:16 (vertical)
        let target = ReframeTarget::vertical_1080p();
        let (cw, ch) = target.crop_size_for_source(1920, 1080);
        // target ratio 9/16 = 0.5625; source 16/9 ≈ 1.78 → wider → crop width
        // crop_h = 1080, crop_w = 1080 * 9/16 = 607.5 → 608 (rounds)
        assert_eq!(ch, 1080);
        assert!(cw < 1920);
        // Check ratio matches
        let ratio = cw as f64 / ch as f64;
        let expected = 9.0 / 16.0;
        assert!((ratio - expected).abs() < 0.01, "ratio={ratio}");
    }

    #[test]
    fn test_reframe_window_center() {
        let w = ReframeWindow::center(1920, 1080, 608, 1080);
        assert_eq!(w.x, (1920 - 608) / 2);
        assert_eq!(w.y, 0);
    }

    #[test]
    fn test_reframe_window_clamped() {
        let w = ReframeWindow {
            x: -50,
            y: -50,
            width: 200,
            height: 200,
        };
        let clamped = w.clamped(1920, 1080);
        assert_eq!(clamped.x, 0);
        assert_eq!(clamped.y, 0);
    }

    #[test]
    fn test_reframe_window_lerp_midpoint() {
        let a = make_window(0, 0, 100, 100);
        let b = make_window(100, 100, 200, 200);
        let mid = ReframeWindow::lerp(&a, &b, 0.5);
        assert_eq!(mid.x, 50);
        assert_eq!(mid.y, 50);
        assert_eq!(mid.width, 150);
        assert_eq!(mid.height, 150);
    }

    #[test]
    fn test_reframe_window_lerp_t0() {
        let a = make_window(0, 0, 100, 100);
        let b = make_window(100, 100, 200, 200);
        let result = ReframeWindow::lerp(&a, &b, 0.0);
        assert_eq!(result, a);
    }

    #[test]
    fn test_reframe_window_lerp_t1() {
        let a = make_window(0, 0, 100, 100);
        let b = make_window(100, 100, 200, 200);
        let result = ReframeWindow::lerp(&a, &b, 1.0);
        assert_eq!(result, b);
    }

    // -- SmartReframer::compute_optimal_crop tests --

    #[test]
    fn test_compute_optimal_crop_center_uniform() {
        let saliency = SaliencyMap::uniform(1920, 1080);
        let target = ReframeTarget::vertical_1080p();
        let window = SmartReframer::compute_optimal_crop(1920, 1080, &saliency, &target);
        // With uniform saliency, should produce a center crop
        assert!(window.x >= 0);
        assert!(window.y >= 0);
        assert!(window.width > 0 && window.height > 0);
    }

    #[test]
    fn test_compute_optimal_crop_clamped() {
        let saliency = SaliencyMap::uniform(1920, 1080);
        let target = ReframeTarget::vertical_1080p();
        let window = SmartReframer::compute_optimal_crop(1920, 1080, &saliency, &target);
        // Window must be within source bounds
        assert!(window.x >= 0);
        assert!(window.y >= 0);
        assert!(window.x as u32 + window.width <= 1920);
        assert!(window.y as u32 + window.height <= 1080);
    }

    // -- SmartReframer::smooth_trajectory tests --

    #[test]
    fn test_smooth_trajectory_empty() {
        let result = SmartReframer::smooth_trajectory(&[], 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_smooth_trajectory_no_smoothing() {
        let windows = vec![make_window(0, 0, 100, 100), make_window(50, 50, 100, 100)];
        let result = SmartReframer::smooth_trajectory(&windows, 0.0);
        assert_eq!(result[0], windows[0]);
        assert_eq!(result[1], windows[1]);
    }

    #[test]
    fn test_smooth_trajectory_preserves_length() {
        let windows: Vec<ReframeWindow> =
            (0..10).map(|i| make_window(i * 10, 0, 100, 100)).collect();
        let result = SmartReframer::smooth_trajectory(&windows, 0.5);
        assert_eq!(result.len(), windows.len());
    }

    #[test]
    fn test_smooth_trajectory_reduces_jumps() {
        // Large jump at frame 1: 0 → 200
        let windows = vec![make_window(0, 0, 100, 100), make_window(200, 0, 100, 100)];
        let result = SmartReframer::smooth_trajectory(&windows, 0.7);
        // Smoothed x should be between 0 and 200
        assert!(result[1].x > 0 && result[1].x < 200, "x={}", result[1].x);
    }

    // -- detect_salient_regions tests --

    #[test]
    fn test_detect_salient_regions_uniform_image() {
        // Uniform image → Laplacian is zero everywhere → uniform saliency fallback
        let frame = vec![128u8; 10 * 10];
        let s = detect_salient_regions(&frame, 10, 10);
        assert_eq!(s.width, 10);
        assert_eq!(s.height, 10);
        assert_eq!(s.values.len(), 100);
    }

    #[test]
    fn test_detect_salient_regions_edge_highlighted() {
        // Checkerboard-like pattern creates high Laplacian response
        let mut frame = vec![0u8; 20 * 20];
        for y in 0..20usize {
            for x in 0..20usize {
                frame[y * 20 + x] = if (x + y) % 2 == 0 { 255 } else { 0 };
            }
        }
        let s = detect_salient_regions(&frame, 20, 20);
        // Interior pixels should have high saliency
        assert!(s.get(5, 5) > 0.0 || s.get(10, 10) > 0.0);
    }

    #[test]
    fn test_detect_salient_regions_too_small() {
        let frame = vec![100u8; 4];
        let s = detect_salient_regions(&frame, 2, 2);
        // Too small for Laplacian → uniform fallback
        assert_eq!(s.values.len(), 4);
    }

    // -- ReframeSequence tests --

    #[test]
    fn test_reframe_sequence_interpolate_between() {
        let kf0 = (0u64, make_window(0, 0, 100, 100));
        let kf1 = (100u64, make_window(100, 0, 100, 100));
        let seq = ReframeSequence::new(vec![kf0, kf1], 200);
        let mid = seq.interpolate_at(50);
        assert_eq!(mid.x, 50);
    }

    #[test]
    fn test_reframe_sequence_before_first_keyframe() {
        let kf0 = (10u64, make_window(50, 0, 100, 100));
        let seq = ReframeSequence::new(vec![kf0], 200);
        let result = seq.interpolate_at(0);
        assert_eq!(result.x, 50);
    }

    #[test]
    fn test_reframe_sequence_after_last_keyframe() {
        let kf0 = (0u64, make_window(0, 0, 100, 100));
        let kf1 = (50u64, make_window(100, 0, 100, 100));
        let seq = ReframeSequence::new(vec![kf0, kf1], 200);
        let result = seq.interpolate_at(150);
        assert_eq!(result.x, 100);
    }

    #[test]
    fn test_reframe_sequence_empty() {
        let seq = ReframeSequence::new(vec![], 100);
        let result = seq.interpolate_at(50);
        assert_eq!(result.width, 0);
    }

    #[test]
    fn test_reframe_sequence_keyframe_count() {
        let kfs = vec![
            (0u64, make_window(0, 0, 100, 100)),
            (50u64, make_window(50, 0, 100, 100)),
            (100u64, make_window(100, 0, 100, 100)),
        ];
        let seq = ReframeSequence::new(kfs, 120);
        assert_eq!(seq.keyframe_count(), 3);
    }

    // SubjectBounds

    #[test]
    fn test_subject_bounds_clamps_confidence() {
        let hi = SubjectBounds::new(0.0, 0.0, 10.0, 10.0, 2.0);
        assert!(hi.confidence <= 1.0);
        let lo = SubjectBounds::new(0.0, 0.0, 10.0, 10.0, -0.5);
        assert!(lo.confidence >= 0.0);
    }

    #[test]
    fn test_subject_bounds_focus_point() {
        let b = SubjectBounds::new(100.0, 200.0, 50.0, 80.0, 0.9);
        let (fx, fy) = b.focus_point();
        assert!((fx - 100.0).abs() < 0.01);
        assert!((fy - 200.0).abs() < 0.01);
    }

    // SubjectTracker

    fn make_bounds(cx: f32, cy: f32, conf: f32) -> SubjectBounds {
        SubjectBounds::new(cx, cy, 100.0, 150.0, conf)
    }

    #[test]
    fn test_tracker_no_smoothing_follows_raw() {
        let mut t = SubjectTracker::new(0.0, 0.0);
        t.update(0, make_bounds(100.0, 200.0, 1.0));
        t.update(1, make_bounds(300.0, 400.0, 1.0));
        let (cx, cy) = t.current_position().expect("pos");
        assert!((cx - 300.0).abs() < 1.0, "cx={cx}");
        assert!((cy - 400.0).abs() < 1.0, "cy={cy}");
    }

    #[test]
    fn test_tracker_ema_reduces_jump() {
        let mut t = SubjectTracker::new(0.5, 0.0);
        t.update(0, make_bounds(0.0, 0.0, 1.0));
        t.update(1, make_bounds(200.0, 200.0, 1.0));
        let (cx, _) = t.current_position().expect("pos");
        assert!((cx - 100.0).abs() < 1.0, "cx={cx}");
    }

    #[test]
    fn test_tracker_ignores_low_confidence() {
        let mut t = SubjectTracker::new(0.0, 0.8);
        t.update(0, make_bounds(100.0, 100.0, 1.0));
        t.update(1, make_bounds(900.0, 900.0, 0.1));
        let (cx, _) = t.current_position().expect("pos");
        assert!((cx - 100.0).abs() < 1.0, "cx={cx}");
    }

    #[test]
    fn test_tracker_reset() {
        let mut t = SubjectTracker::new(0.0, 0.0);
        t.update(0, make_bounds(100.0, 100.0, 1.0));
        t.reset();
        assert!(t.current_position().is_none());
        assert!(t.history().is_empty());
    }

    #[test]
    fn test_tracker_history_count() {
        let mut t = SubjectTracker::new(0.0, 0.0);
        for i in 0..5u64 {
            t.update(i, make_bounds(i as f32 * 10.0, 0.0, 1.0));
        }
        assert_eq!(t.history().len(), 5);
    }

    #[test]
    fn test_tracker_generate_sequence_count() {
        let mut t = SubjectTracker::new(0.0, 0.0);
        t.update(0, make_bounds(540.0, 960.0, 1.0));
        t.update(30, make_bounds(600.0, 900.0, 1.0));
        let seq = t.generate_sequence(1080, 1920, &ReframeTarget::landscape_1080p(), 60);
        assert_eq!(seq.keyframe_count(), 2);
    }

    #[test]
    fn test_tracker_generate_sequence_clamped() {
        let mut t = SubjectTracker::new(0.0, 0.0);
        t.update(0, make_bounds(5.0, 5.0, 1.0));
        let seq = t.generate_sequence(1080, 1920, &ReframeTarget::landscape_1080p(), 30);
        let w = seq.interpolate_at(0);
        assert!(w.x >= 0 && w.y >= 0);
    }

    #[test]
    fn test_tracker_smooth_sequence_count() {
        let mut t = SubjectTracker::new(0.0, 0.0);
        for i in 0..8u64 {
            t.update(i * 5, make_bounds(i as f32 * 30.0, 500.0, 1.0));
        }
        let seq =
            t.generate_smooth_sequence(1080, 1920, &ReframeTarget::landscape_1080p(), 60, 0.5);
        assert_eq!(seq.keyframe_count(), 8);
    }

    // FrameOrientation

    #[test]
    fn test_frame_orientation_landscape() {
        assert_eq!(
            FrameOrientation::from_dimensions(1920, 1080),
            FrameOrientation::Landscape
        );
    }

    #[test]
    fn test_frame_orientation_portrait() {
        assert_eq!(
            FrameOrientation::from_dimensions(1080, 1920),
            FrameOrientation::Portrait
        );
    }

    #[test]
    fn test_frame_orientation_square() {
        assert_eq!(
            FrameOrientation::from_dimensions(1080, 1080),
            FrameOrientation::Square
        );
    }

    // VerticalToHorizontalParams

    #[test]
    fn test_primary_placement_centered() {
        let p = VerticalToHorizontalParams::vertical_to_widescreen();
        let (out_x, out_y, placed_w, placed_h) = p.primary_placement();
        assert_eq!(placed_h, 1080);
        assert!(placed_w < 1920);
        let expected_x = (1920 - placed_w as i32) / 2;
        assert_eq!(out_x, expected_x);
        assert_eq!(out_y, 0);
    }

    #[test]
    fn test_side_regions_sum() {
        let p = VerticalToHorizontalParams::vertical_to_widescreen();
        let (_, _, placed_w, _) = p.primary_placement();
        let (_, left_w, _, right_w) = p.side_regions();
        assert_eq!(left_w + placed_w + right_w, p.output_width);
    }

    #[test]
    fn test_saliency_crop_full_source() {
        let p = VerticalToHorizontalParams::vertical_to_widescreen();
        let sal = SaliencyMap::uniform(p.src_width, p.src_height);
        let w = p.saliency_crop_window(&sal);
        assert_eq!(w.width, p.src_width);
        assert_eq!(w.height, p.src_height);
    }

    #[test]
    fn test_square_source_in_landscape() {
        let p = VerticalToHorizontalParams::new(
            1080,
            1080,
            1920,
            1080,
            VerticalToHorizontalStrategy::Pillarbox,
        );
        let (out_x, _, placed_w, _) = p.primary_placement();
        assert_eq!(placed_w, 1080);
        assert!(out_x > 0);
    }
}
