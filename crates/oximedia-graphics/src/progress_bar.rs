//! Animated progress bar widget for broadcast graphics overlays.
//!
//! A progress bar communicates completion progress (0 %–100 %) in a visual and
//! optionally animated way. This module provides:
//!
//! - Horizontal and vertical orientations.
//! - Solid-colour, gradient (left-to-right or bottom-to-top), or segmented fill.
//! - Animated fill with configurable easing so the bar smoothly transitions
//!   from one fill level to another.
//! - Optional label text encoded as a simple pixel-level description (the
//!   module is self-contained and does not depend on a font renderer).
//! - Corner rounding via a simple distance-based masking approach so the bar
//!   looks polished at broadcast resolution.
//! - Configurable border and background.
//!
//! All rendering is done into an RGBA `Vec<u8>` that can be composited over a
//! video frame.

// ---------------------------------------------------------------------------
// Easing
// ---------------------------------------------------------------------------

/// Easing function applied to the animated fill progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressEasing {
    /// No easing — the fill jumps to its target instantly.
    None,
    /// Linear interpolation.
    Linear,
    /// Cubic ease-in-out — gentle start and end.
    EaseInOut,
    /// Cubic ease-out — fast start, gentle end.
    EaseOut,
    /// Cubic ease-in — gentle start, fast end.
    EaseIn,
}

impl Default for ProgressEasing {
    fn default() -> Self {
        Self::EaseInOut
    }
}

impl ProgressEasing {
    /// Apply the easing to a normalised time value `t` in [0.0, 1.0].
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::None => if t >= 1.0 { 1.0 } else { 0.0 },
            Self::Linear => t,
            Self::EaseInOut => t * t * (3.0 - 2.0 * t),
            Self::EaseOut => {
                let u = 1.0 - t;
                1.0 - u * u * u
            }
            Self::EaseIn => t * t * t,
        }
    }
}

// ---------------------------------------------------------------------------
// Fill style
// ---------------------------------------------------------------------------

/// Fill style for the progress bar.
#[derive(Debug, Clone, PartialEq)]
pub enum FillStyle {
    /// Solid single colour.
    Solid([u8; 4]),
    /// Linear gradient from `start_color` to `end_color` along the fill axis.
    Gradient {
        /// Colour at the empty end of the bar.
        start_color: [u8; 4],
        /// Colour at the full end of the bar.
        end_color: [u8; 4],
    },
    /// Bar is divided into `count` equal segments with alternating colors.
    Segmented {
        /// Number of segments.
        count: u32,
        /// Primary segment colour.
        primary_color: [u8; 4],
        /// Gap / separator colour.
        gap_color: [u8; 4],
        /// Gap width as a fraction of one segment width (0.0–0.5).
        gap_fraction: f32,
    },
}

impl Default for FillStyle {
    fn default() -> Self {
        Self::Solid([0, 180, 60, 255])
    }
}

// ---------------------------------------------------------------------------
// Orientation
// ---------------------------------------------------------------------------

/// Orientation of the progress bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarOrientation {
    /// Fills from left to right.
    Horizontal,
    /// Fills from bottom to top.
    Vertical,
}

impl Default for BarOrientation {
    fn default() -> Self {
        Self::Horizontal
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Full configuration for a progress bar widget.
#[derive(Debug, Clone)]
pub struct ProgressBarConfig {
    /// Width of the rendered output in pixels.
    pub width: u32,
    /// Height of the rendered output in pixels.
    pub height: u32,
    /// Background colour of the track (behind the fill).
    pub track_color: [u8; 4],
    /// Border colour.  Set alpha to 0 to disable the border.
    pub border_color: [u8; 4],
    /// Border thickness in pixels.
    pub border_thickness_px: u32,
    /// Corner radius in pixels.  0 = sharp corners.
    pub corner_radius_px: u32,
    /// Fill style.
    pub fill_style: FillStyle,
    /// Bar orientation.
    pub orientation: BarOrientation,
    /// Duration of the fill animation in seconds.  0 = instant snap.
    pub animation_duration_secs: f32,
    /// Easing function for the animation.
    pub easing: ProgressEasing,
}

impl Default for ProgressBarConfig {
    fn default() -> Self {
        Self {
            width: 400,
            height: 40,
            track_color: [40, 40, 40, 220],
            border_color: [120, 120, 120, 255],
            border_thickness_px: 2,
            corner_radius_px: 6,
            fill_style: FillStyle::default(),
            orientation: BarOrientation::default(),
            easing: ProgressEasing::default(),
            animation_duration_secs: 0.4,
        }
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Runtime state for an animated progress bar.
#[derive(Debug, Clone)]
pub struct ProgressBarState {
    /// Current *display* fill in [0.0, 1.0] (the animated value).
    pub display_fill: f32,
    /// Target fill level in [0.0, 1.0].
    pub target_fill: f32,
    /// Previous fill level (animation starts from here).
    pub from_fill: f32,
    /// Elapsed time within the current animation (seconds).
    pub elapsed_secs: f32,
    /// Whether the animation is currently active.
    pub animating: bool,
}

impl Default for ProgressBarState {
    fn default() -> Self {
        Self {
            display_fill: 0.0,
            target_fill: 0.0,
            from_fill: 0.0,
            elapsed_secs: 0.0,
            animating: false,
        }
    }
}

impl ProgressBarState {
    /// Create a new state with a specific initial fill level in [0.0, 1.0].
    pub fn with_fill(fill: f32) -> Self {
        let fill = fill.clamp(0.0, 1.0);
        Self {
            display_fill: fill,
            target_fill: fill,
            from_fill: fill,
            elapsed_secs: 0.0,
            animating: false,
        }
    }

    /// Set a new target fill level.  The bar will animate toward `fill`.
    ///
    /// If `config.animation_duration_secs` is 0, the fill snaps immediately.
    pub fn set_fill(&mut self, fill: f32, config: &ProgressBarConfig) {
        let fill = fill.clamp(0.0, 1.0);
        if (fill - self.display_fill).abs() < f32::EPSILON {
            return;
        }
        self.from_fill = self.display_fill;
        self.target_fill = fill;
        self.elapsed_secs = 0.0;
        if config.animation_duration_secs <= 0.0 {
            self.display_fill = fill;
            self.animating = false;
        } else {
            self.animating = true;
        }
    }

    /// Advance the animation by `dt_secs`.
    ///
    /// Returns `true` when the animation has completed (or was never active).
    pub fn advance(&mut self, dt_secs: f32, config: &ProgressBarConfig) -> bool {
        if !self.animating {
            return true;
        }
        self.elapsed_secs += dt_secs;
        let dur = config.animation_duration_secs;
        let t = if dur > 0.0 {
            (self.elapsed_secs / dur).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let eased = config.easing.apply(t);
        self.display_fill = self.from_fill + (self.target_fill - self.from_fill) * eased;
        if t >= 1.0 {
            self.display_fill = self.target_fill;
            self.animating = false;
            return true;
        }
        false
    }

    /// Snap the display fill to the target without animation.
    pub fn snap_to_target(&mut self) {
        self.display_fill = self.target_fill;
        self.animating = false;
    }
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

/// Renders a progress bar into an RGBA pixel buffer.
pub struct ProgressBarRenderer;

impl ProgressBarRenderer {
    /// Render the progress bar.
    ///
    /// Returns a `Vec<u8>` of length `width * height * 4`.
    pub fn render(state: &ProgressBarState, config: &ProgressBarConfig) -> Vec<u8> {
        let w = config.width as usize;
        let h = config.height as usize;
        let mut data = vec![0u8; w * h * 4];

        let radius = config.corner_radius_px as f32;
        let border_t = config.border_thickness_px as usize;

        for row in 0..h {
            for col in 0..w {
                let in_widget = is_inside_rounded_rect(col, row, w, h, radius);
                if !in_widget {
                    continue;
                }

                // Determine if this pixel is in the border region.
                let in_border = border_t > 0
                    && (col < border_t
                        || col >= w.saturating_sub(border_t)
                        || row < border_t
                        || row >= h.saturating_sub(border_t));

                let pixel_color = if in_border {
                    config.border_color
                } else {
                    // Is this pixel inside the filled region?
                    let in_fill = is_in_fill_region(
                        col,
                        row,
                        w,
                        h,
                        state.display_fill,
                        config.orientation,
                        border_t,
                    );

                    if in_fill {
                        sample_fill_color(col, row, w, h, &config.fill_style, config.orientation)
                    } else {
                        config.track_color
                    }
                };

                let idx = (row * w + col) * 4;
                data[idx] = pixel_color[0];
                data[idx + 1] = pixel_color[1];
                data[idx + 2] = pixel_color[2];
                data[idx + 3] = pixel_color[3];
            }
        }

        data
    }
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

/// Test whether pixel `(col, row)` is inside a rounded rectangle.
fn is_inside_rounded_rect(col: usize, row: usize, w: usize, h: usize, radius: f32) -> bool {
    if w == 0 || h == 0 {
        return false;
    }
    let r = radius.min(w as f32 / 2.0).min(h as f32 / 2.0);
    if r < 1.0 {
        return true; // sharp corners — all pixels inside
    }

    let fx = col as f32 + 0.5;
    let fy = row as f32 + 0.5;
    let fw = w as f32;
    let fh = h as f32;

    // Closest corner centre.
    let cx = fx.clamp(r, fw - r);
    let cy = fy.clamp(r, fh - r);
    let dx = fx - cx;
    let dy = fy - cy;
    (dx * dx + dy * dy).sqrt() <= r
}

/// Test whether pixel `(col, row)` is inside the filled region of the bar.
fn is_in_fill_region(
    col: usize,
    row: usize,
    w: usize,
    h: usize,
    fill: f32,
    orientation: BarOrientation,
    border_t: usize,
) -> bool {
    // Track inner dimensions (excluding border).
    let inner_x_start = border_t;
    let inner_y_start = border_t;
    let inner_w = w.saturating_sub(border_t * 2);
    let inner_h = h.saturating_sub(border_t * 2);

    if inner_w == 0 || inner_h == 0 {
        return false;
    }

    match orientation {
        BarOrientation::Horizontal => {
            if col < inner_x_start {
                return false;
            }
            let local_x = col - inner_x_start;
            let fill_px = (inner_w as f32 * fill) as usize;
            local_x < fill_px && row >= inner_y_start && row < inner_y_start + inner_h
        }
        BarOrientation::Vertical => {
            if row < inner_y_start {
                return false;
            }
            let local_y = row - inner_y_start;
            // Fills from bottom.
            let fill_px = (inner_h as f32 * fill) as usize;
            let fill_start = inner_h.saturating_sub(fill_px);
            local_y >= fill_start
                && local_y < inner_h
                && col >= inner_x_start
                && col < inner_x_start + inner_w
        }
    }
}

/// Sample the fill colour at pixel `(col, row)`.
fn sample_fill_color(
    col: usize,
    row: usize,
    w: usize,
    h: usize,
    fill_style: &FillStyle,
    orientation: BarOrientation,
) -> [u8; 4] {
    match fill_style {
        FillStyle::Solid(color) => *color,
        FillStyle::Gradient {
            start_color,
            end_color,
        } => {
            let t = match orientation {
                BarOrientation::Horizontal => {
                    if w > 1 {
                        col as f32 / (w - 1) as f32
                    } else {
                        0.0
                    }
                }
                BarOrientation::Vertical => {
                    // From bottom (t=0) to top (t=1).
                    if h > 1 {
                        1.0 - row as f32 / (h - 1) as f32
                    } else {
                        0.0
                    }
                }
            };
            lerp_color(*start_color, *end_color, t)
        }
        FillStyle::Segmented {
            count,
            primary_color,
            gap_color,
            gap_fraction,
        } => {
            if *count == 0 {
                return *primary_color;
            }
            let count = *count as usize;
            let t = match orientation {
                BarOrientation::Horizontal => {
                    if w > 1 {
                        col as f32 / w as f32
                    } else {
                        0.0
                    }
                }
                BarOrientation::Vertical => {
                    if h > 1 {
                        1.0 - row as f32 / h as f32
                    } else {
                        0.0
                    }
                }
            };
            // Position within a single segment [0.0, 1.0).
            let seg_t = (t * count as f32).fract();
            let gap_f = gap_fraction.clamp(0.0, 0.5);
            if seg_t < gap_f || seg_t > 1.0 - gap_f {
                *gap_color
            } else {
                *primary_color
            }
        }
    }
}

/// Linear interpolation between two RGBA colours.
fn lerp_color(a: [u8; 4], b: [u8; 4], t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    [
        (a[0] as f32 + (b[0] as f32 - a[0] as f32) * t) as u8,
        (a[1] as f32 + (b[1] as f32 - a[1] as f32) * t) as u8,
        (a[2] as f32 + (b[2] as f32 - a[2] as f32) * t) as u8,
        (a[3] as f32 + (b[3] as f32 - a[3] as f32) * t) as u8,
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> ProgressBarConfig {
        ProgressBarConfig {
            width: 200,
            height: 30,
            border_thickness_px: 2,
            corner_radius_px: 4,
            ..ProgressBarConfig::default()
        }
    }

    #[test]
    fn test_render_output_size() {
        let state = ProgressBarState::with_fill(0.5);
        let cfg = small_config();
        let data = ProgressBarRenderer::render(&state, &cfg);
        assert_eq!(data.len(), 200 * 30 * 4);
    }

    #[test]
    fn test_empty_bar_has_track_colour() {
        let state = ProgressBarState::with_fill(0.0);
        let cfg = small_config();
        let data = ProgressBarRenderer::render(&state, &cfg);
        // Centre pixel should be track colour (no fill).
        let cx = 100usize;
        let cy = 15usize;
        let idx = (cy * 200 + cx) * 4;
        let track = cfg.track_color;
        assert_eq!(data[idx], track[0]);
        assert_eq!(data[idx + 1], track[1]);
        assert_eq!(data[idx + 2], track[2]);
    }

    #[test]
    fn test_full_bar_centre_is_fill_colour() {
        let state = ProgressBarState::with_fill(1.0);
        let cfg = ProgressBarConfig {
            fill_style: FillStyle::Solid([255, 0, 0, 255]),
            border_thickness_px: 0,
            corner_radius_px: 0,
            ..small_config()
        };
        let data = ProgressBarRenderer::render(&state, &cfg);
        let cx = 100usize;
        let cy = 15usize;
        let idx = (cy * 200 + cx) * 4;
        assert_eq!(data[idx], 255, "red channel");
        assert_eq!(data[idx + 1], 0, "green channel");
        assert_eq!(data[idx + 2], 0, "blue channel");
    }

    #[test]
    fn test_easing_none_snaps() {
        assert!((ProgressEasing::None.apply(0.0) - 0.0).abs() < f32::EPSILON);
        assert!((ProgressEasing::None.apply(1.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_easing_linear() {
        assert!((ProgressEasing::Linear.apply(0.5) - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_easing_ease_in_out_midpoint() {
        // At t=0.5, cubic ease-in-out = 0.5 * 0.5 * (3 - 1) = 0.5.
        let v = ProgressEasing::EaseInOut.apply(0.5);
        assert!((v - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_state_set_fill_snaps_when_no_animation() {
        let mut cfg = small_config();
        cfg.animation_duration_secs = 0.0;
        let mut state = ProgressBarState::with_fill(0.0);
        state.set_fill(0.75, &cfg);
        assert!((state.display_fill - 0.75).abs() < f32::EPSILON);
        assert!(!state.animating);
    }

    #[test]
    fn test_state_set_fill_animates() {
        let cfg = small_config(); // 0.4s animation
        let mut state = ProgressBarState::with_fill(0.0);
        state.set_fill(1.0, &cfg);
        assert!(state.animating);
        assert!((state.display_fill - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_state_advance_completes() {
        let cfg = small_config();
        let mut state = ProgressBarState::with_fill(0.0);
        state.set_fill(1.0, &cfg);
        let done = state.advance(cfg.animation_duration_secs + 0.1, &cfg);
        assert!(done);
        assert!((state.display_fill - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_state_snap_to_target() {
        let cfg = small_config();
        let mut state = ProgressBarState::with_fill(0.2);
        state.set_fill(0.9, &cfg);
        state.snap_to_target();
        assert!((state.display_fill - 0.9).abs() < f32::EPSILON);
        assert!(!state.animating);
    }

    #[test]
    fn test_gradient_fill_produces_different_colours_at_ends() {
        let state = ProgressBarState::with_fill(1.0);
        let cfg = ProgressBarConfig {
            width: 200,
            height: 20,
            border_thickness_px: 0,
            corner_radius_px: 0,
            fill_style: FillStyle::Gradient {
                start_color: [0, 0, 255, 255],
                end_color: [255, 0, 0, 255],
            },
            ..ProgressBarConfig::default()
        };
        let data = ProgressBarRenderer::render(&state, &cfg);
        // Left edge pixel: should be close to start_color (blue).
        let left_idx = (10 * 200 + 2) * 4;
        // Right edge pixel: should be close to end_color (red).
        let right_idx = (10 * 200 + 197) * 4;
        assert!(
            data[left_idx + 2] > data[right_idx + 2],
            "left edge should be more blue"
        );
        assert!(
            data[right_idx] > data[left_idx],
            "right edge should be more red"
        );
    }

    #[test]
    fn test_vertical_bar_fills_from_bottom() {
        let state = ProgressBarState::with_fill(0.5);
        let cfg = ProgressBarConfig {
            width: 40,
            height: 100,
            border_thickness_px: 0,
            corner_radius_px: 0,
            fill_style: FillStyle::Solid([200, 100, 50, 255]),
            orientation: BarOrientation::Vertical,
            ..ProgressBarConfig::default()
        };
        let data = ProgressBarRenderer::render(&state, &cfg);
        // Bottom row (row 99) should be filled.
        let bottom_idx = (99 * 40 + 20) * 4;
        assert_eq!(data[bottom_idx], 200);
        // Top row (row 0) should be track colour.
        let top_idx = 20 * 4;
        let track = cfg.track_color;
        assert_eq!(data[top_idx], track[0]);
    }

    #[test]
    fn test_segmented_fill() {
        let state = ProgressBarState::with_fill(1.0);
        let cfg = ProgressBarConfig {
            width: 100,
            height: 20,
            border_thickness_px: 0,
            corner_radius_px: 0,
            fill_style: FillStyle::Segmented {
                count: 10,
                primary_color: [0, 200, 0, 255],
                gap_color: [0, 0, 0, 255],
                gap_fraction: 0.1,
            },
            ..ProgressBarConfig::default()
        };
        let data = ProgressBarRenderer::render(&state, &cfg);
        assert_eq!(data.len(), 100 * 20 * 4);
        // Should have both primary and gap pixels.
        let has_primary = data.chunks_exact(4).any(|p| p[1] == 200);
        let has_gap = data.chunks_exact(4).any(|p| p[0] == 0 && p[1] == 0 && p[3] == 255);
        assert!(has_primary);
        assert!(has_gap);
    }

    #[test]
    fn test_lerp_color_at_endpoints() {
        let a = [0u8, 0, 0, 255];
        let b = [200u8, 100, 50, 255];
        assert_eq!(lerp_color(a, b, 0.0), a);
        assert_eq!(lerp_color(a, b, 1.0), b);
    }

    #[test]
    fn test_is_inside_rounded_rect_corners_excluded() {
        // With radius=10 on a 100x50 widget, corner pixel (0,0) should be outside.
        assert!(!is_inside_rounded_rect(0, 0, 100, 50, 10.0));
        // Centre should always be inside.
        assert!(is_inside_rounded_rect(50, 25, 100, 50, 10.0));
    }

    #[test]
    fn test_render_zero_fill_all_track() {
        let state = ProgressBarState::with_fill(0.0);
        let cfg = ProgressBarConfig {
            width: 100,
            height: 20,
            border_thickness_px: 0,
            corner_radius_px: 0,
            fill_style: FillStyle::Solid([255, 0, 0, 255]),
            track_color: [10, 10, 10, 255],
            ..ProgressBarConfig::default()
        };
        let data = ProgressBarRenderer::render(&state, &cfg);
        // No pixel should have red=255 (fill colour).
        let has_red = data.chunks_exact(4).any(|p| p[0] == 255 && p[3] == 255);
        assert!(!has_red, "zero fill should show no red fill pixels");
    }
}
