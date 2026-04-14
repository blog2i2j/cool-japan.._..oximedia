//! Picture-in-picture layout with position-preset and keyframe animation.
//!
//! Provides named corner positions, a pixel-rect compositor helper, a
//! simple keyframe timeline for animating PiP position/size/opacity over time,
//! and a pixel-level RGBA compositor (`PictureInPicture`).

#![allow(dead_code)]

// ─────────────────────────────────────────────────────────────────────────────
// PictureInPicture — pixel-level RGBA compositor
// ─────────────────────────────────────────────────────────────────────────────

/// Pixel-level picture-in-picture compositor that blends an overlay ("PiP")
/// frame onto a main (background) frame.
///
/// Both frames use packed RGBA byte buffers (`width * height * 4` bytes).
/// Alpha blending uses the PiP frame's per-pixel alpha channel.
///
/// # Example
/// ```no_run
/// use oximedia_edit::picture_in_picture::PictureInPicture;
///
/// let pip = PictureInPicture::new(1920, 1080, 480, 270, 10, 10);
/// let main = vec![0u8; 1920 * 1080 * 4];
/// let overlay = vec![255u8; 480 * 270 * 4];
/// let result = pip.composite(&main, &overlay);
/// assert_eq!(result.len(), 1920 * 1080 * 4);
/// ```
#[derive(Debug, Clone)]
pub struct PictureInPicture {
    /// Width of the main (background) frame in pixels.
    pub main_w: u32,
    /// Height of the main (background) frame in pixels.
    pub main_h: u32,
    /// Width of the PiP overlay frame in pixels.
    pub pip_w: u32,
    /// Height of the PiP overlay frame in pixels.
    pub pip_h: u32,
    /// X offset (pixels from left) of the PiP within the main frame.
    pub x: u32,
    /// Y offset (pixels from top) of the PiP within the main frame.
    pub y: u32,
}

impl PictureInPicture {
    /// Create a new `PictureInPicture` compositor.
    ///
    /// # Arguments
    /// * `main_w`, `main_h` – dimensions of the background (main) frame.
    /// * `pip_w`, `pip_h`   – dimensions of the overlay (PiP) frame.
    /// * `x`, `y`           – top-left position of the PiP inside the main frame.
    #[must_use]
    pub const fn new(main_w: u32, main_h: u32, pip_w: u32, pip_h: u32, x: u32, y: u32) -> Self {
        Self {
            main_w,
            main_h,
            pip_w,
            pip_h,
            x,
            y,
        }
    }

    /// Composite the PiP frame over the main frame using alpha blending.
    ///
    /// Both `main` and `pip` must be packed RGBA buffers:
    /// - `main` length: `main_w * main_h * 4`
    /// - `pip`  length: `pip_w * pip_h * 4`
    ///
    /// Pixels outside `[x, x+pip_w) × [y, y+pip_h)` are copied verbatim from
    /// `main`.  Pixels inside that region are alpha-blended:
    /// ```text
    /// out_rgb = pip_alpha/255 * pip_rgb + (1 - pip_alpha/255) * main_rgb
    /// out_a   = 255  (fully opaque output)
    /// ```
    ///
    /// # Panics
    /// Panics in debug mode if buffer sizes do not match the declared dimensions.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    pub fn composite(&self, main: &[u8], pip: &[u8]) -> Vec<u8> {
        debug_assert_eq!(
            main.len(),
            (self.main_w * self.main_h * 4) as usize,
            "main buffer size mismatch"
        );
        debug_assert_eq!(
            pip.len(),
            (self.pip_w * self.pip_h * 4) as usize,
            "pip buffer size mismatch"
        );

        let mut out = main.to_vec();

        // Clamp the PiP rect to within the main frame boundaries.
        let x_end = (self.x + self.pip_w).min(self.main_w);
        let y_end = (self.y + self.pip_h).min(self.main_h);

        for row in self.y..y_end {
            for col in self.x..x_end {
                let pip_row = row - self.y;
                let pip_col = col - self.x;

                // Guard against out-of-bounds for partial PiP frames.
                if pip_row >= self.pip_h || pip_col >= self.pip_w {
                    continue;
                }

                let main_idx = ((row * self.main_w + col) * 4) as usize;
                let pip_idx = ((pip_row * self.pip_w + pip_col) * 4) as usize;

                if main_idx + 3 >= main.len() || pip_idx + 3 >= pip.len() {
                    continue;
                }

                let alpha = pip[pip_idx + 3] as f32 / 255.0;
                let inv_alpha = 1.0 - alpha;

                out[main_idx] =
                    (alpha * pip[pip_idx] as f32 + inv_alpha * main[main_idx] as f32) as u8;
                out[main_idx + 1] =
                    (alpha * pip[pip_idx + 1] as f32 + inv_alpha * main[main_idx + 1] as f32) as u8;
                out[main_idx + 2] =
                    (alpha * pip[pip_idx + 2] as f32 + inv_alpha * main[main_idx + 2] as f32) as u8;
                out[main_idx + 3] = 255;
            }
        }

        out
    }
}

#[cfg(test)]
mod pip_compositor_tests {
    use super::*;

    #[test]
    fn test_pip_compositor_output_size() {
        let pip = PictureInPicture::new(4, 4, 2, 2, 0, 0);
        let main = vec![0u8; 4 * 4 * 4];
        let overlay = vec![255u8; 2 * 2 * 4];
        let out = pip.composite(&main, &overlay);
        assert_eq!(out.len(), 4 * 4 * 4);
    }

    #[test]
    fn test_pip_compositor_fully_opaque_overwrites() {
        // 2×2 main, 1×1 pip at (0,0), fully opaque white pip over black main
        let pip = PictureInPicture::new(2, 2, 1, 1, 0, 0);
        let main = vec![0u8; 2 * 2 * 4]; // All black
        let overlay = vec![255u8; 1 * 1 * 4]; // Opaque white
        let out = pip.composite(&main, &overlay);
        // Top-left pixel should be white (255,255,255,255)
        assert_eq!(out[0], 255);
        assert_eq!(out[1], 255);
        assert_eq!(out[2], 255);
        assert_eq!(out[3], 255);
        // Bottom-right pixel (offset 12) should remain black
        assert_eq!(out[12], 0);
    }

    #[test]
    fn test_pip_compositor_fully_transparent_keeps_main() {
        let pip = PictureInPicture::new(2, 2, 1, 1, 0, 0);
        let main = vec![100u8; 2 * 2 * 4];
        // Alpha = 0 → fully transparent
        let mut overlay = vec![255u8; 1 * 1 * 4];
        overlay[3] = 0; // alpha = 0
        let out = pip.composite(&main, &overlay);
        // Top-left pixel: main channel values should be preserved
        assert_eq!(out[0], 100);
        assert_eq!(out[1], 100);
        assert_eq!(out[2], 100);
    }

    #[test]
    fn test_pip_compositor_clamped_to_main_bounds() {
        // PiP at (3,3) in a 4×4 main, pip is 2×2 – extends beyond bounds
        let pip = PictureInPicture::new(4, 4, 2, 2, 3, 3);
        let main = vec![50u8; 4 * 4 * 4];
        let overlay = vec![200u8; 2 * 2 * 4];
        // Should not panic
        let out = pip.composite(&main, &overlay);
        assert_eq!(out.len(), 4 * 4 * 4);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PipPosition
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-defined or custom anchor position for a PiP overlay.
#[derive(Debug, Clone, PartialEq)]
pub enum PipPosition {
    /// Top-left corner of the main frame.
    TopLeft,
    /// Top-right corner of the main frame.
    TopRight,
    /// Bottom-left corner of the main frame.
    BottomLeft,
    /// Bottom-right corner of the main frame.
    BottomRight,
    /// Centred on the main frame.
    Center,
    /// Arbitrary anchor expressed as percentages from the top-left (0–100).
    Custom {
        /// Horizontal anchor (% from left edge).
        x_pct: f32,
        /// Vertical anchor (% from top edge).
        y_pct: f32,
    },
}

impl PipPosition {
    /// Anchor point expressed as `(x_pct, y_pct)` (0–100 % of main frame).
    ///
    /// These values are the top-left corner of the PiP window before any
    /// margin is applied.
    #[must_use]
    pub fn anchor_pct(&self) -> (f32, f32) {
        match self {
            Self::TopLeft => (5.0, 5.0),
            Self::TopRight => (70.0, 5.0),
            Self::BottomLeft => (5.0, 70.0),
            Self::BottomRight => (70.0, 70.0),
            Self::Center => (37.5, 37.5),
            Self::Custom { x_pct, y_pct } => (*x_pct, *y_pct),
        }
    }

    /// Human-readable name for this position.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::TopLeft => "Top Left",
            Self::TopRight => "Top Right",
            Self::BottomLeft => "Bottom Left",
            Self::BottomRight => "Bottom Right",
            Self::Center => "Center",
            Self::Custom { .. } => "Custom",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PipLayout
// ─────────────────────────────────────────────────────────────────────────────

/// Layout configuration for a picture-in-picture overlay.
#[derive(Debug, Clone)]
pub struct PipLayout {
    /// Where to anchor the PiP within the main frame.
    pub position: PipPosition,
    /// PiP width as a percentage of the main frame width (default 25 %).
    pub width_pct: f32,
    /// PiP height as a percentage of the main frame height (default 25 %).
    pub height_pct: f32,
    /// Opacity of the PiP layer: 0.0 (transparent) – 1.0 (opaque).
    pub opacity: f32,
    /// Border thickness in pixels (0 = no border).
    pub border_width: u32,
    /// Border colour as `[R, G, B]` bytes.
    pub border_color: [u8; 3],
    /// Margin from the nearest frame edge in pixels.
    pub margin_px: u32,
}

impl PipLayout {
    /// Create a layout with sensible defaults:
    /// - size: 25 % of main frame
    /// - opacity: 1.0
    /// - border: 2 px white
    /// - margin: 10 px
    #[must_use]
    pub fn new(position: PipPosition) -> Self {
        Self {
            position,
            width_pct: 25.0,
            height_pct: 25.0,
            opacity: 1.0,
            border_width: 2,
            border_color: [255, 255, 255],
            margin_px: 10,
        }
    }

    /// Override the PiP size (builder pattern).
    #[must_use]
    pub fn with_size(mut self, width_pct: f32, height_pct: f32) -> Self {
        self.width_pct = width_pct;
        self.height_pct = height_pct;
        self
    }

    /// Override the opacity (clamped to 0.0–1.0, builder pattern).
    #[must_use]
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Compute the pixel rectangle `(x, y, w, h)` for this PiP overlay.
    ///
    /// - `w = main_w × width_pct / 100`  (rounded down)
    /// - `h = main_h × height_pct / 100` (rounded down)
    /// - `x = main_w × anchor.x / 100 + margin_px`
    /// - `y = main_h × anchor.y / 100 + margin_px`
    ///
    /// The result is clamped so the PiP stays fully within the main frame.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn compute_rect(&self, main_w: u32, main_h: u32) -> (u32, u32, u32, u32) {
        let w = ((main_w as f32 * self.width_pct / 100.0) as u32).max(1);
        let h = ((main_h as f32 * self.height_pct / 100.0) as u32).max(1);

        let (ax, ay) = self.position.anchor_pct();
        let raw_x = (main_w as f32 * ax / 100.0) as u32 + self.margin_px;
        let raw_y = (main_h as f32 * ay / 100.0) as u32 + self.margin_px;

        // Clamp so PiP stays within the main frame.
        let x = raw_x.min(main_w.saturating_sub(w));
        let y = raw_y.min(main_h.saturating_sub(h));

        (x, y, w, h)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PipAnimation
// ─────────────────────────────────────────────────────────────────────────────

/// A keyframe that associates a [`PipLayout`] with a specific time.
#[derive(Debug, Clone)]
pub struct PipAnimKeyframe {
    /// Timeline position in seconds.
    pub time_secs: f64,
    /// Layout to use from this keyframe forward.
    pub layout: PipLayout,
}

/// A simple keyframe-driven PiP animation.
///
/// Keyframes are kept sorted by `time_secs`.  `layout_at` performs a
/// step-hold lookup (returns the last keyframe whose time ≤ t).
#[derive(Debug, Clone, Default)]
pub struct PipAnimation {
    /// Sorted keyframes.
    pub keyframes: Vec<PipAnimKeyframe>,
}

impl PipAnimation {
    /// Create an empty animation.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a keyframe (builder pattern).  Keyframes are kept sorted by time.
    #[must_use]
    pub fn add_keyframe(mut self, time_secs: f64, layout: PipLayout) -> Self {
        self.keyframes.push(PipAnimKeyframe { time_secs, layout });
        self.keyframes.sort_by(|a, b| {
            a.time_secs
                .partial_cmp(&b.time_secs)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self
    }

    /// Return the active layout at `time_secs` (last keyframe with time ≤ t).
    ///
    /// Returns `None` when there are no keyframes or `time_secs` is before
    /// the first keyframe.
    #[must_use]
    pub fn layout_at(&self, time_secs: f64) -> Option<&PipLayout> {
        self.keyframes
            .iter()
            .rev()
            .find(|kf| kf.time_secs <= time_secs)
            .map(|kf| &kf.layout)
    }

    /// Total duration covered by keyframes (time of last keyframe).
    ///
    /// Returns `0.0` when there are no keyframes.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        self.keyframes.last().map_or(0.0, |kf| kf.time_secs)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pip_position_anchor_pct_named() {
        let (x, y) = PipPosition::TopLeft.anchor_pct();
        assert!((x - 5.0).abs() < f32::EPSILON);
        assert!((y - 5.0).abs() < f32::EPSILON);

        let (x, y) = PipPosition::BottomRight.anchor_pct();
        assert!((x - 70.0).abs() < f32::EPSILON);
        assert!((y - 70.0).abs() < f32::EPSILON);

        let (x, y) = PipPosition::Center.anchor_pct();
        assert!((x - 37.5).abs() < f32::EPSILON);
        assert!((y - 37.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pip_position_anchor_pct_custom() {
        let pos = PipPosition::Custom {
            x_pct: 20.0,
            y_pct: 30.0,
        };
        let (x, y) = pos.anchor_pct();
        assert!((x - 20.0).abs() < f32::EPSILON);
        assert!((y - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pip_position_name() {
        assert_eq!(PipPosition::TopLeft.name(), "Top Left");
        assert_eq!(PipPosition::BottomRight.name(), "Bottom Right");
        assert_eq!(
            PipPosition::Custom {
                x_pct: 0.0,
                y_pct: 0.0
            }
            .name(),
            "Custom"
        );
    }

    #[test]
    fn test_pip_layout_compute_rect_proportions() {
        // 25% of 1920x1080
        let layout = PipLayout::new(PipPosition::TopLeft);
        let (_, _, w, h) = layout.compute_rect(1920, 1080);
        assert_eq!(w, 480); // 1920 * 25 / 100
        assert_eq!(h, 270); // 1080 * 25 / 100
    }

    #[test]
    fn test_pip_layout_compute_rect_stays_within_frame() {
        // Use BottomRight with large size; PiP must not exceed main bounds
        let layout = PipLayout::new(PipPosition::BottomRight).with_size(40.0, 40.0);
        let (x, y, w, h) = layout.compute_rect(1920, 1080);
        assert!(
            x + w <= 1920,
            "PiP right edge {x}+{w}={} exceeds main width 1920",
            x + w
        );
        assert!(
            y + h <= 1080,
            "PiP bottom edge {y}+{h}={} exceeds main height 1080",
            y + h
        );
    }

    #[test]
    fn test_pip_layout_with_size() {
        let layout = PipLayout::new(PipPosition::Center).with_size(50.0, 30.0);
        assert!((layout.width_pct - 50.0).abs() < f32::EPSILON);
        assert!((layout.height_pct - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pip_layout_with_opacity_clamped() {
        let layout = PipLayout::new(PipPosition::Center).with_opacity(1.5);
        assert!((layout.opacity - 1.0).abs() < f32::EPSILON);
        let layout2 = PipLayout::new(PipPosition::Center).with_opacity(-0.5);
        assert!((layout2.opacity).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pip_animation_no_keyframes_returns_none() {
        let anim = PipAnimation::new();
        assert!(anim.layout_at(5.0).is_none());
    }

    #[test]
    fn test_pip_animation_add_keyframes_sorted() {
        let anim = PipAnimation::new()
            .add_keyframe(10.0, PipLayout::new(PipPosition::TopRight))
            .add_keyframe(2.0, PipLayout::new(PipPosition::TopLeft))
            .add_keyframe(5.0, PipLayout::new(PipPosition::Center));

        // Keyframes must be sorted by time
        let times: Vec<f64> = anim.keyframes.iter().map(|kf| kf.time_secs).collect();
        assert_eq!(times, vec![2.0, 5.0, 10.0]);
    }

    #[test]
    fn test_pip_animation_layout_at_step_hold() {
        let anim = PipAnimation::new()
            .add_keyframe(0.0, PipLayout::new(PipPosition::TopLeft))
            .add_keyframe(30.0, PipLayout::new(PipPosition::BottomRight));

        // Before any keyframe
        assert!(anim.layout_at(-1.0).is_none());
        // At exactly first keyframe
        let l = anim.layout_at(0.0);
        assert!(l.is_some());
        assert_eq!(l.expect("layout").position, PipPosition::TopLeft);
        // Between keyframes: still returns first (step-hold)
        let l2 = anim.layout_at(15.0);
        assert_eq!(l2.expect("layout").position, PipPosition::TopLeft);
        // At second keyframe
        let l3 = anim.layout_at(30.0);
        assert_eq!(l3.expect("layout").position, PipPosition::BottomRight);
        // After last keyframe
        let l4 = anim.layout_at(999.0);
        assert_eq!(l4.expect("layout").position, PipPosition::BottomRight);
    }

    #[test]
    fn test_pip_animation_duration_secs() {
        let anim = PipAnimation::new()
            .add_keyframe(0.0, PipLayout::new(PipPosition::TopLeft))
            .add_keyframe(45.0, PipLayout::new(PipPosition::BottomRight));
        assert!((anim.duration_secs() - 45.0).abs() < 1e-9);
    }

    #[test]
    fn test_pip_animation_duration_empty() {
        let anim = PipAnimation::new();
        assert!((anim.duration_secs()).abs() < 1e-9);
    }
}
