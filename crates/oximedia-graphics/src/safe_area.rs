//! Broadcast safe area guide rendering.
//!
//! A *safe area* is a region of the television or broadcast frame that is
//! guaranteed to be visible on all consumer displays. Two classic safe-area
//! conventions originated in the analogue CRT era and remain in use today as
//! *action safe* and *title safe*:
//!
//! - **Action safe** (typically 90 % of the frame): important action should stay
//!   inside this boundary to avoid being cropped on older displays.
//! - **Title safe** (typically 80 % of the frame): graphics, subtitles, and
//!   other critical text should stay inside this inner boundary.
//!
//! Modern broadcast standards (EBU, SMPTE) define additional zones, and some
//! production facilities use custom percentages. This module supports all of
//! these use-cases through the [`SafeAreaConfig`] type and [`SafeAreaRenderer`].
//!
//! The renderer produces an RGBA pixel buffer that can be composited over a
//! video frame as a guide overlay. Each zone is drawn as a rectangular outline
//! (optionally filled) so the guide is non-destructive: it does not obscure the
//! underlying video content.

// ---------------------------------------------------------------------------
// Zone definitions
// ---------------------------------------------------------------------------

/// A single safe-area zone defined as a percentage of the frame dimensions.
///
/// The zone is centred within the frame.  For example, a `coverage` of `0.9`
/// means the zone rectangle spans 90 % of the frame width and 90 % of the
/// frame height.
#[derive(Debug, Clone, PartialEq)]
pub struct SafeAreaZone {
    /// Human-readable label for this zone, e.g. "Action Safe" or "Title Safe".
    pub label: String,
    /// Coverage fraction in (0.0, 1.0].  `1.0` is the full frame.
    pub coverage: f32,
    /// RGBA colour used to draw the zone outline (and fill, if enabled).
    pub color: [u8; 4],
    /// Whether to fill the *outside* of the zone with a semi-transparent shade.
    ///
    /// When `true` the region outside the zone boundary is tinted with
    /// `color` at a reduced opacity to help visualise unsafe areas.
    pub shade_exterior: bool,
    /// Opacity used for the exterior shade in [0.0, 1.0].
    pub shade_opacity: f32,
    /// Outline thickness in pixels.
    pub outline_thickness_px: u32,
}

impl SafeAreaZone {
    /// Construct an action-safe zone (90 % coverage, green outline).
    pub fn action_safe() -> Self {
        Self {
            label: "Action Safe (90%)".to_string(),
            coverage: 0.9,
            color: [0, 200, 80, 220],
            shade_exterior: false,
            shade_opacity: 0.15,
            outline_thickness_px: 2,
        }
    }

    /// Construct a title-safe zone (80 % coverage, yellow outline).
    pub fn title_safe() -> Self {
        Self {
            label: "Title Safe (80%)".to_string(),
            coverage: 0.8,
            color: [255, 200, 0, 220],
            shade_exterior: false,
            shade_opacity: 0.15,
            outline_thickness_px: 2,
        }
    }

    /// Construct an EBU-R95 centre zone (safe for 16:9 graphics, 88 % coverage).
    pub fn ebu_r95() -> Self {
        Self {
            label: "EBU R95 Graphics Safe (88%)".to_string(),
            coverage: 0.88,
            color: [100, 160, 255, 200],
            shade_exterior: false,
            shade_opacity: 0.12,
            outline_thickness_px: 2,
        }
    }

    /// Construct a custom zone with the given label, coverage, and RGBA colour.
    ///
    /// # Errors
    /// Returns an error string if `coverage` is not in (0.0, 1.0].
    pub fn custom(label: impl Into<String>, coverage: f32, color: [u8; 4]) -> Result<Self, String> {
        if !(coverage > 0.0 && coverage <= 1.0) {
            return Err(format!(
                "coverage must be in (0.0, 1.0], got {coverage}"
            ));
        }
        Ok(Self {
            label: label.into(),
            coverage,
            color,
            shade_exterior: false,
            shade_opacity: 0.15,
            outline_thickness_px: 2,
        })
    }

    /// Return the pixel rectangle `(x, y, width, height)` for this zone inside
    /// a frame of the given dimensions.
    pub fn pixel_rect(&self, frame_width: u32, frame_height: u32) -> (u32, u32, u32, u32) {
        let w = (frame_width as f32 * self.coverage) as u32;
        let h = (frame_height as f32 * self.coverage) as u32;
        let x = frame_width.saturating_sub(w) / 2;
        let y = frame_height.saturating_sub(h) / 2;
        (x, y, w, h)
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the safe-area overlay renderer.
#[derive(Debug, Clone)]
pub struct SafeAreaConfig {
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
    /// Zones to render, from outermost to innermost.
    ///
    /// Zones are composited in order; later zones are drawn on top.
    pub zones: Vec<SafeAreaZone>,
    /// Draw a cross-hair at the frame centre.
    pub show_centre_crosshair: bool,
    /// RGBA colour of the centre crosshair.
    pub crosshair_color: [u8; 4],
    /// Length of each crosshair arm in pixels.
    pub crosshair_arm_px: u32,
    /// Thickness of the crosshair lines in pixels.
    pub crosshair_thickness_px: u32,
}

impl Default for SafeAreaConfig {
    fn default() -> Self {
        Self {
            frame_width: 1920,
            frame_height: 1080,
            zones: vec![SafeAreaZone::action_safe(), SafeAreaZone::title_safe()],
            show_centre_crosshair: true,
            crosshair_color: [255, 255, 255, 180],
            crosshair_arm_px: 30,
            crosshair_thickness_px: 1,
        }
    }
}

impl SafeAreaConfig {
    /// Create a config with a full broadcast preset: action safe + title safe +
    /// EBU R95 + centre crosshair.
    pub fn broadcast_preset(frame_width: u32, frame_height: u32) -> Self {
        Self {
            frame_width,
            frame_height,
            zones: vec![
                SafeAreaZone::action_safe(),
                SafeAreaZone::ebu_r95(),
                SafeAreaZone::title_safe(),
            ],
            show_centre_crosshair: true,
            crosshair_color: [255, 255, 255, 180],
            crosshair_arm_px: 40,
            crosshair_thickness_px: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

/// Renders safe-area guide overlays onto an RGBA pixel buffer.
pub struct SafeAreaRenderer;

impl SafeAreaRenderer {
    /// Render the safe-area guide for the given configuration.
    ///
    /// Returns an RGBA `Vec<u8>` of length `frame_width * frame_height * 4`.
    /// The buffer starts fully transparent; guide graphics are composited on
    /// top using the alpha values specified in each zone.
    pub fn render(config: &SafeAreaConfig) -> Vec<u8> {
        let w = config.frame_width as usize;
        let h = config.frame_height as usize;
        let mut data = vec![0u8; w * h * 4];

        // Render zones in order (outermost first so inner zones draw on top).
        for zone in &config.zones {
            render_zone(&mut data, w, h, zone);
        }

        // Render crosshair last so it appears on top of all zones.
        if config.show_centre_crosshair {
            render_crosshair(
                &mut data,
                w,
                h,
                config.crosshair_color,
                config.crosshair_arm_px as usize,
                config.crosshair_thickness_px as usize,
            );
        }

        data
    }

    /// Composite the safe-area guide over an existing RGBA frame in-place.
    ///
    /// `frame` must have length `frame_width * frame_height * 4` as defined by
    /// `config`.  The guide is alpha-composited using a simple over operation.
    pub fn composite_onto(config: &SafeAreaConfig, frame: &mut [u8]) {
        let guide = Self::render(config);
        let len = frame.len().min(guide.len());
        for i in (0..len).step_by(4) {
            let src_a = guide[i + 3] as f32 / 255.0;
            if src_a < f32::EPSILON {
                continue;
            }
            let dst_a = frame[i + 3] as f32 / 255.0;
            let out_a = src_a + dst_a * (1.0 - src_a);
            if out_a < f32::EPSILON {
                continue;
            }
            for c in 0..3 {
                let src = guide[i + c] as f32 / 255.0;
                let dst = frame[i + c] as f32 / 255.0;
                let out = (src * src_a + dst * dst_a * (1.0 - src_a)) / out_a;
                frame[i + c] = (out * 255.0) as u8;
            }
            frame[i + 3] = (out_a * 255.0) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Internal rendering helpers
// ---------------------------------------------------------------------------

/// Render a single safe-area zone onto the pixel buffer.
fn render_zone(data: &mut [u8], w: usize, h: usize, zone: &SafeAreaZone) {
    let (rx, ry, rw, rh) = zone.pixel_rect(w as u32, h as u32);
    let rx = rx as usize;
    let ry = ry as usize;
    let rw = rw as usize;
    let rh = rh as usize;
    let thickness = zone.outline_thickness_px as usize;

    // Optionally shade the exterior region.
    if zone.shade_exterior && zone.shade_opacity > 0.0 {
        let shade_a = (zone.color[3] as f32 * zone.shade_opacity) as u8;
        let shade_color = [zone.color[0], zone.color[1], zone.color[2], shade_a];
        for row in 0..h {
            for col in 0..w {
                let inside = col >= rx
                    && col < rx + rw
                    && row >= ry
                    && row < ry + rh;
                if !inside {
                    blend_pixel(data, w, col, row, shade_color);
                }
            }
        }
    }

    // Draw the rectangular outline.
    draw_rect_outline(data, w, h, rx, ry, rw, rh, thickness, zone.color);
}

/// Draw a hollow rectangle outline.
fn draw_rect_outline(
    data: &mut [u8],
    w: usize,
    h: usize,
    rx: usize,
    ry: usize,
    rw: usize,
    rh: usize,
    thickness: usize,
    color: [u8; 4],
) {
    let thickness = thickness.max(1);
    let x_end = rx + rw;
    let y_end = ry + rh;

    // Top and bottom horizontal bars.
    for t in 0..thickness {
        // Top bar
        let row_top = ry + t;
        if row_top < h {
            for col in rx..x_end.min(w) {
                blend_pixel(data, w, col, row_top, color);
            }
        }
        // Bottom bar
        if y_end > t {
            let row_bot = y_end - 1 - t;
            if row_bot < h {
                for col in rx..x_end.min(w) {
                    blend_pixel(data, w, col, row_bot, color);
                }
            }
        }
    }

    // Left and right vertical bars (excluding corners already drawn).
    for row in (ry + thickness)..(y_end.saturating_sub(thickness)).min(h) {
        for t in 0..thickness {
            // Left bar
            let col_left = rx + t;
            if col_left < w {
                blend_pixel(data, w, col_left, row, color);
            }
            // Right bar
            if x_end > t {
                let col_right = x_end - 1 - t;
                if col_right < w {
                    blend_pixel(data, w, col_right, row, color);
                }
            }
        }
    }
}

/// Render a crosshair at the frame centre.
fn render_crosshair(
    data: &mut [u8],
    w: usize,
    h: usize,
    color: [u8; 4],
    arm_px: usize,
    thickness_px: usize,
) {
    let cx = w / 2;
    let cy = h / 2;
    let half_t = thickness_px / 2;
    let thickness = thickness_px.max(1);

    // Horizontal arm.
    let x_start = cx.saturating_sub(arm_px);
    let x_end = (cx + arm_px + 1).min(w);
    for dx in x_start..x_end {
        for t in 0..thickness {
            let row = cy.saturating_sub(half_t) + t;
            if row < h {
                blend_pixel(data, w, dx, row, color);
            }
        }
    }

    // Vertical arm.
    let y_start = cy.saturating_sub(arm_px);
    let y_end = (cy + arm_px + 1).min(h);
    for dy in y_start..y_end {
        for t in 0..thickness {
            let col = cx.saturating_sub(half_t) + t;
            if col < w {
                blend_pixel(data, w, col, dy, color);
            }
        }
    }
}

/// Alpha-composite a solid `color` over the pixel at `(col, row)` using the
/// Porter-Duff *over* operation.
#[inline]
fn blend_pixel(data: &mut [u8], w: usize, col: usize, row: usize, color: [u8; 4]) {
    let idx = (row * w + col) * 4;
    if idx + 3 >= data.len() {
        return;
    }
    let src_a = color[3] as f32 / 255.0;
    if src_a < f32::EPSILON {
        return;
    }
    let dst_a = data[idx + 3] as f32 / 255.0;
    let out_a = src_a + dst_a * (1.0 - src_a);
    if out_a < f32::EPSILON {
        data[idx + 3] = 0;
        return;
    }
    for c in 0..3 {
        let src = color[c] as f32 / 255.0;
        let dst = data[idx + c] as f32 / 255.0;
        let out = (src * src_a + dst * dst_a * (1.0 - src_a)) / out_a;
        data[idx + c] = (out * 255.0) as u8;
    }
    data[idx + 3] = (out_a * 255.0) as u8;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> SafeAreaConfig {
        SafeAreaConfig {
            frame_width: 320,
            frame_height: 180,
            zones: vec![SafeAreaZone::action_safe(), SafeAreaZone::title_safe()],
            show_centre_crosshair: true,
            crosshair_color: [255, 255, 255, 200],
            crosshair_arm_px: 20,
            crosshair_thickness_px: 1,
        }
    }

    #[test]
    fn test_render_output_size() {
        let cfg = small_config();
        let data = SafeAreaRenderer::render(&cfg);
        assert_eq!(data.len(), 320 * 180 * 4);
    }

    #[test]
    fn test_render_has_nonzero_pixels() {
        let cfg = small_config();
        let data = SafeAreaRenderer::render(&cfg);
        assert!(data.iter().any(|&b| b > 0), "expected some visible pixels");
    }

    #[test]
    fn test_zone_pixel_rect_action_safe() {
        let zone = SafeAreaZone::action_safe();
        let (x, y, w, h) = zone.pixel_rect(320, 180);
        // 90% of 320 = 288, centred: (320-288)/2 = 16
        assert_eq!(w, 288);
        assert_eq!(h, 162);
        assert_eq!(x, 16);
        assert_eq!(y, 9);
    }

    #[test]
    fn test_zone_pixel_rect_title_safe() {
        let zone = SafeAreaZone::title_safe();
        let (x, y, w, h) = zone.pixel_rect(320, 180);
        // 80% of 320 = 256, centred: (320-256)/2 = 32
        assert_eq!(w, 256);
        assert_eq!(h, 144);
        assert_eq!(x, 32);
        assert_eq!(y, 18);
    }

    #[test]
    fn test_zone_pixel_rect_full_frame() {
        let zone = SafeAreaZone::custom("Full", 1.0, [255, 0, 0, 255]).unwrap();
        let (x, y, w, h) = zone.pixel_rect(1920, 1080);
        assert_eq!(x, 0);
        assert_eq!(y, 0);
        assert_eq!(w, 1920);
        assert_eq!(h, 1080);
    }

    #[test]
    fn test_custom_zone_invalid_coverage_zero() {
        let result = SafeAreaZone::custom("Bad", 0.0, [255, 0, 0, 255]);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_zone_invalid_coverage_over_one() {
        let result = SafeAreaZone::custom("Bad", 1.1, [255, 0, 0, 255]);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_zone_valid() {
        let zone = SafeAreaZone::custom("Custom 85%", 0.85, [200, 100, 50, 255]);
        assert!(zone.is_ok());
        let zone = zone.unwrap();
        assert!((zone.coverage - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn test_broadcast_preset_has_three_zones() {
        let cfg = SafeAreaConfig::broadcast_preset(1920, 1080);
        assert_eq!(cfg.zones.len(), 3);
    }

    #[test]
    fn test_crosshair_produces_pixels_at_centre() {
        let cfg = SafeAreaConfig {
            frame_width: 100,
            frame_height: 100,
            zones: vec![],
            show_centre_crosshair: true,
            crosshair_color: [255, 0, 0, 255],
            crosshair_arm_px: 10,
            crosshair_thickness_px: 1,
        };
        let data = SafeAreaRenderer::render(&cfg);
        // Centre pixel should be red (R=255, A=255).
        let cx = 50usize;
        let cy = 50usize;
        let idx = (cy * 100 + cx) * 4;
        assert!(data[idx] > 0, "red channel at centre should be non-zero");
        assert_eq!(data[idx + 3], 255, "alpha at centre crosshair pixel");
    }

    #[test]
    fn test_composite_onto_modifies_frame() {
        let cfg = SafeAreaConfig {
            frame_width: 64,
            frame_height: 64,
            zones: vec![SafeAreaZone::action_safe()],
            show_centre_crosshair: false,
            ..SafeAreaConfig::default()
        };
        let mut frame = vec![50u8; 64 * 64 * 4];
        // Set alpha to 255 everywhere so the frame is opaque.
        for chunk in frame.chunks_exact_mut(4) {
            chunk[3] = 255;
        }
        let original_frame = frame.clone();
        SafeAreaRenderer::composite_onto(&cfg, &mut frame);
        // At least some pixels should have changed.
        assert_ne!(frame, original_frame);
    }

    #[test]
    fn test_shade_exterior_produces_exterior_pixels() {
        let mut zone = SafeAreaZone::action_safe();
        zone.shade_exterior = true;
        zone.shade_opacity = 0.5;
        let cfg = SafeAreaConfig {
            frame_width: 100,
            frame_height: 100,
            zones: vec![zone],
            show_centre_crosshair: false,
            ..SafeAreaConfig::default()
        };
        let data = SafeAreaRenderer::render(&cfg);
        // Top-left corner (0,0) is outside the action safe zone, should be shaded.
        let idx = 0;
        assert!(data[idx + 3] > 0, "exterior corner pixel should be shaded");
    }

    #[test]
    fn test_ebu_r95_zone_coverage() {
        let zone = SafeAreaZone::ebu_r95();
        assert!((zone.coverage - 0.88).abs() < f32::EPSILON);
    }

    #[test]
    fn test_no_zones_only_crosshair() {
        let cfg = SafeAreaConfig {
            frame_width: 50,
            frame_height: 50,
            zones: vec![],
            show_centre_crosshair: true,
            crosshair_color: [0, 255, 0, 255],
            crosshair_arm_px: 5,
            crosshair_thickness_px: 1,
        };
        let data = SafeAreaRenderer::render(&cfg);
        assert_eq!(data.len(), 50 * 50 * 4);
        // Should have non-transparent pixels from the crosshair.
        let has_pixels = data.chunks_exact(4).any(|p| p[3] > 0);
        assert!(has_pixels);
    }
}
