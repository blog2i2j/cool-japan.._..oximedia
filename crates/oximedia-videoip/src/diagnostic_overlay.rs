//! Diagnostic overlay: burns network statistics onto video frames.
//!
//! The `DiagnosticOverlay` renders text statistics (latency, packet loss,
//! bitrate, jitter) into an RGBA8 frame buffer.  It uses a minimal built-in
//! monospace bitmap font so there are no external font dependencies.

#![allow(dead_code)]

/// Network statistics to display in the overlay.
#[derive(Debug, Clone, Default)]
pub struct NetworkStats {
    /// Round-trip latency in milliseconds.
    pub latency_ms: f64,
    /// Packet loss percentage (0.0–100.0).
    pub packet_loss_pct: f32,
    /// Current bitrate in kbps.
    pub bitrate_kbps: u32,
    /// Network jitter in milliseconds.
    pub jitter_ms: f64,
    /// Number of FEC-recovered packets.
    pub fec_recovered: u64,
    /// Frame sequence number.
    pub frame_seq: u64,
}

impl NetworkStats {
    /// Formats the stats into a multi-line display string.
    #[must_use]
    pub fn format_lines(&self) -> Vec<String> {
        vec![
            format!("Latency : {:.1} ms", self.latency_ms),
            format!("Loss    : {:.1} %", self.packet_loss_pct),
            format!("Bitrate : {} kbps", self.bitrate_kbps),
            format!("Jitter  : {:.1} ms", self.jitter_ms),
            format!("FEC rec : {}", self.fec_recovered),
            format!("Seq     : {}", self.frame_seq),
        ]
    }
}

/// Configuration for the diagnostic overlay.
#[derive(Debug, Clone)]
pub struct OverlayConfig {
    /// X offset of the overlay within the frame (pixels).
    pub x: u32,
    /// Y offset of the overlay within the frame (pixels).
    pub y: u32,
    /// Text foreground colour (RGBA8).
    pub fg_colour: [u8; 4],
    /// Background colour drawn behind the text (RGBA8). Alpha controls blending.
    pub bg_colour: [u8; 4],
    /// Scale factor for the bitmap font (1 = 8×8 pixels per char).
    pub scale: u32,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            x: 8,
            y: 8,
            fg_colour: [0, 255, 0, 255],   // green
            bg_colour: [0, 0, 0, 160],      // semi-transparent black
        scale: 1,
        }
    }
}

// ── Minimal 8×8 bitmap font (ASCII printable 32–127) ─────────────────────────
//
// Each glyph is 8 rows of 8 bits stored as a u8.  The font is the classic
// "IBM PC" 8×8 character ROM encoded inline.  Only the characters actually
// needed for digit/letter/punctuation rendering are fully defined; others
// fall back to a block glyph.

/// Retrieves the 8 row bytes for a given ASCII character.
fn glyph(c: u8) -> [u8; 8] {
    // A minimal monospace 8×8 font for 0-9, A-Z, a-z, space, colon, dot, %
    match c {
        b' ' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        b'0' => [0x3C, 0x66, 0x6E, 0x76, 0x66, 0x66, 0x3C, 0x00],
        b'1' => [0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00],
        b'2' => [0x3C, 0x66, 0x06, 0x0C, 0x18, 0x30, 0x7E, 0x00],
        b'3' => [0x3C, 0x66, 0x06, 0x1C, 0x06, 0x66, 0x3C, 0x00],
        b'4' => [0x0E, 0x1E, 0x36, 0x66, 0x7F, 0x06, 0x06, 0x00],
        b'5' => [0x7E, 0x60, 0x7C, 0x06, 0x06, 0x66, 0x3C, 0x00],
        b'6' => [0x1C, 0x30, 0x60, 0x7C, 0x66, 0x66, 0x3C, 0x00],
        b'7' => [0x7E, 0x06, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00],
        b'8' => [0x3C, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x3C, 0x00],
        b'9' => [0x3C, 0x66, 0x66, 0x3E, 0x06, 0x0C, 0x38, 0x00],
        b'.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00],
        b':' => [0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00],
        b'%' => [0x62, 0x66, 0x0C, 0x18, 0x30, 0x66, 0x46, 0x00],
        b'-' => [0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00],
        b'/' => [0x02, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00],
        // Letters A-Z
        b'A' => [0x18, 0x3C, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00],
        b'B' => [0x7C, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x7C, 0x00],
        b'C' => [0x3C, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3C, 0x00],
        b'D' => [0x78, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0x78, 0x00],
        b'E' => [0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x7E, 0x00],
        b'F' => [0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x00],
        b'G' => [0x3C, 0x66, 0x60, 0x6E, 0x66, 0x66, 0x3E, 0x00],
        b'J' => [0x06, 0x06, 0x06, 0x06, 0x66, 0x66, 0x3C, 0x00],
        b'L' => [0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E, 0x00],
        b'S' => [0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0x00],
        b'T' => [0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00],
        b'a' => [0x00, 0x00, 0x3C, 0x06, 0x3E, 0x66, 0x3E, 0x00],
        b'b' => [0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x7C, 0x00],
        b'c' => [0x00, 0x00, 0x3C, 0x60, 0x60, 0x60, 0x3C, 0x00],
        b'd' => [0x06, 0x06, 0x3E, 0x66, 0x66, 0x66, 0x3E, 0x00],
        b'e' => [0x00, 0x00, 0x3C, 0x66, 0x7E, 0x60, 0x3C, 0x00],
        b'i' => [0x18, 0x00, 0x18, 0x18, 0x18, 0x18, 0x0E, 0x00],
        b'k' => [0x60, 0x66, 0x6C, 0x78, 0x6C, 0x66, 0x63, 0x00],
        b'm' => [0x00, 0x00, 0x66, 0x7F, 0x7F, 0x6B, 0x63, 0x00],
        b'n' => [0x00, 0x00, 0x7C, 0x66, 0x66, 0x66, 0x66, 0x00],
        b'o' => [0x00, 0x00, 0x3C, 0x66, 0x66, 0x66, 0x3C, 0x00],
        b'p' => [0x00, 0x00, 0x7C, 0x66, 0x7C, 0x60, 0x60, 0x00],
        b'r' => [0x00, 0x00, 0x7C, 0x66, 0x60, 0x60, 0x60, 0x00],
        b's' => [0x00, 0x00, 0x3E, 0x60, 0x3C, 0x06, 0x7C, 0x00],
        b't' => [0x18, 0x18, 0x7E, 0x18, 0x18, 0x18, 0x0E, 0x00],
        b'u' => [0x00, 0x00, 0x66, 0x66, 0x66, 0x66, 0x3E, 0x00],
        b'y' => [0x00, 0x00, 0x66, 0x66, 0x3E, 0x06, 0x3C, 0x00],
        // Fallback: filled rectangle
        _ => [0x00, 0x7E, 0x42, 0x42, 0x42, 0x42, 0x7E, 0x00],
    }
}

/// Draws a single 8×8 (or scaled) glyph into an RGBA8 frame buffer.
///
/// Pixels outside the frame boundary are clipped silently.
fn draw_glyph(
    frame: &mut [u8],
    frame_w: u32,
    frame_h: u32,
    x: u32,
    y: u32,
    scale: u32,
    c: u8,
    fg: [u8; 4],
    bg: [u8; 4],
) {
    let rows = glyph(c);
    let glyph_px = 8 * scale;

    for row in 0..8u32 {
        for col in 0..8u32 {
            let bit = (rows[row as usize] >> (7 - col)) & 1;
            let colour = if bit != 0 { fg } else { bg };

            for sy in 0..scale {
                for sx in 0..scale {
                    let px = x + col * scale + sx;
                    let py = y + row * scale + sy;
                    if px >= frame_w || py >= frame_h {
                        continue;
                    }
                    let off = ((py * frame_w + px) * 4) as usize;
                    if off + 3 >= frame.len() {
                        continue;
                    }
                    // Alpha-blend background colour.
                    let bg_alpha = bg[3] as f32 / 255.0;
                    let inv_alpha = 1.0 - bg_alpha;
                    if bit != 0 {
                        // Foreground pixel: full opacity.
                        frame[off..off + 4].copy_from_slice(&fg);
                    } else if bg[3] > 0 {
                        // Background pixel: alpha blend.
                        for c_idx in 0..3 {
                            let existing = frame[off + c_idx] as f32;
                            let blended = existing * inv_alpha + colour[c_idx] as f32 * bg_alpha;
                            frame[off + c_idx] = blended.clamp(0.0, 255.0) as u8;
                        }
                        frame[off + 3] = 255;
                    }
                }
            }
        }
    }
    let _ = glyph_px; // suppress unused warning
}

/// Draws a string of text starting at `(x, y)` in the frame.
fn draw_text(
    frame: &mut [u8],
    frame_w: u32,
    frame_h: u32,
    x: u32,
    y: u32,
    scale: u32,
    text: &str,
    fg: [u8; 4],
    bg: [u8; 4],
) {
    let char_w = 8 * scale;
    for (i, c) in text.bytes().enumerate() {
        let cx = x + i as u32 * char_w;
        draw_glyph(frame, frame_w, frame_h, cx, y, scale, c, fg, bg);
    }
}

/// Diagnostic overlay that burns network stats onto video frames.
#[derive(Debug, Clone)]
pub struct DiagnosticOverlay {
    /// Overlay configuration.
    pub config: OverlayConfig,
}

impl DiagnosticOverlay {
    /// Creates a new overlay with the given configuration.
    #[must_use]
    pub fn new(config: OverlayConfig) -> Self {
        Self { config }
    }

    /// Creates an overlay with default configuration.
    #[must_use]
    pub fn default_overlay() -> Self {
        Self::new(OverlayConfig::default())
    }

    /// Burns the given `stats` onto the RGBA8 `frame` buffer in-place.
    ///
    /// `frame` must be exactly `frame_w * frame_h * 4` bytes.
    /// If the dimensions are inconsistent, the frame is returned unchanged.
    pub fn render_onto(
        &self,
        frame: &mut [u8],
        frame_w: u32,
        frame_h: u32,
        stats: &NetworkStats,
    ) {
        let expected = (frame_w * frame_h * 4) as usize;
        if frame.len() != expected {
            return;
        }

        let lines = stats.format_lines();
        let char_h = 8 * self.config.scale;

        for (i, line) in lines.iter().enumerate() {
            let ly = self.config.y + i as u32 * (char_h + 1);
            draw_text(
                frame,
                frame_w,
                frame_h,
                self.config.x,
                ly,
                self.config.scale,
                line,
                self.config.fg_colour,
                self.config.bg_colour,
            );
        }
    }

    /// Returns the pixel height of the full overlay block (all stats lines).
    #[must_use]
    pub fn overlay_height(&self) -> u32 {
        let char_h = 8 * self.config.scale;
        let n_lines = 6u32; // matches NetworkStats::format_lines()
        n_lines * (char_h + 1)
    }

    /// Returns the pixel width of the widest stats line (approximate).
    #[must_use]
    pub fn overlay_width(&self) -> u32 {
        let max_chars = 24u32; // "Latency : XXXXX.X ms"
        max_chars * 8 * self.config.scale
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(w: u32, h: u32) -> Vec<u8> {
        vec![0u8; (w * h * 4) as usize]
    }

    #[test]
    fn test_stats_format_lines() {
        let stats = NetworkStats {
            latency_ms: 12.5,
            packet_loss_pct: 0.3,
            bitrate_kbps: 5000,
            jitter_ms: 2.1,
            fec_recovered: 7,
            frame_seq: 42,
        };
        let lines = stats.format_lines();
        assert_eq!(lines.len(), 6);
        assert!(lines[0].contains("12.5"));
        assert!(lines[2].contains("5000"));
    }

    #[test]
    fn test_overlay_modifies_frame() {
        let ov = DiagnosticOverlay::default_overlay();
        let mut frame = make_frame(320, 240);
        let stats = NetworkStats::default();
        ov.render_onto(&mut frame, 320, 240, &stats);
        // At least some pixels should be non-zero (green foreground)
        assert!(frame.iter().any(|&v| v != 0));
    }

    #[test]
    fn test_overlay_bad_dimensions_ignored() {
        let ov = DiagnosticOverlay::default_overlay();
        let mut frame = vec![0u8; 100]; // wrong size
        let stats = NetworkStats::default();
        // Should not panic
        ov.render_onto(&mut frame, 320, 240, &stats);
        assert!(frame.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_overlay_height() {
        let cfg = OverlayConfig { scale: 2, ..Default::default() };
        let ov = DiagnosticOverlay::new(cfg);
        // 6 lines × (8*2 + 1) = 6 × 17 = 102
        assert_eq!(ov.overlay_height(), 102);
    }

    #[test]
    fn test_overlay_width() {
        let ov = DiagnosticOverlay::default_overlay();
        assert_eq!(ov.overlay_width(), 24 * 8);
    }

    #[test]
    fn test_glyph_nonzero_for_known_chars() {
        // Known glyphs like '0'-'9' should have at least one set bit
        for c in b'0'..=b'9' {
            let g = glyph(c);
            assert!(g.iter().any(|&r| r != 0), "glyph for '{c}' should be non-empty");
        }
    }

    #[test]
    fn test_network_stats_default_zeroed() {
        let stats = NetworkStats::default();
        assert!((stats.latency_ms).abs() < f64::EPSILON);
        assert!((stats.packet_loss_pct as f64).abs() < f64::EPSILON);
        assert_eq!(stats.bitrate_kbps, 0);
        assert_eq!(stats.fec_recovered, 0);
    }

    #[test]
    fn test_stats_format_lines_count() {
        let stats = NetworkStats::default();
        assert_eq!(stats.format_lines().len(), 6);
    }

    #[test]
    fn test_overlay_config_default_values() {
        let cfg = OverlayConfig::default();
        assert_eq!(cfg.x, 8);
        assert_eq!(cfg.y, 8);
        assert_eq!(cfg.scale, 1);
        // Foreground should be green.
        assert_eq!(cfg.fg_colour, [0, 255, 0, 255]);
    }

    #[test]
    fn test_overlay_scale2_height_doubled() {
        let cfg1 = OverlayConfig { scale: 1, ..Default::default() };
        let cfg2 = OverlayConfig { scale: 2, ..Default::default() };
        let ov1 = DiagnosticOverlay::new(cfg1);
        let ov2 = DiagnosticOverlay::new(cfg2);
        assert_eq!(ov2.overlay_height(), ov1.overlay_height() * 2);
    }

    #[test]
    fn test_render_onto_large_frame_no_panic() {
        // A very large frame should not panic.
        let ov = DiagnosticOverlay::default_overlay();
        let mut frame = make_frame(1920, 1080);
        let stats = NetworkStats {
            latency_ms: 5.5,
            packet_loss_pct: 1.2,
            bitrate_kbps: 8000,
            jitter_ms: 3.3,
            fec_recovered: 100,
            frame_seq: 9999,
        };
        ov.render_onto(&mut frame, 1920, 1080, &stats);
        // Should have written some non-zero pixels.
        assert!(frame.iter().any(|&v| v != 0));
    }

    #[test]
    fn test_render_onto_out_of_bounds_offset_no_panic() {
        // Overlay positioned near the edge — draw_glyph clips, should not panic.
        let cfg = OverlayConfig {
            x: 310,
            y: 230,
            scale: 1,
            ..Default::default()
        };
        let ov = DiagnosticOverlay::new(cfg);
        let mut frame = make_frame(320, 240);
        ov.render_onto(&mut frame, 320, 240, &NetworkStats::default());
        // No panic is the assertion.
    }

    #[test]
    fn test_glyph_fallback_for_unknown_char() {
        // A character not in the match table falls back to the box glyph.
        let g = glyph(b'@');  // '@' not in our table
        // The fallback is [0x00, 0x7E, 0x42, ...] — non-zero.
        assert!(g.iter().any(|&r| r != 0));
    }
}
