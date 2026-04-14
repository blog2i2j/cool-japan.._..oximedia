//! RGB parade scope — separate R/G/B channel parades with individual histograms.
//!
//! The RGB parade is a professional broadcast scope that renders three
//! independent waveform displays side-by-side, one per colour channel.
//! Each pane shows the full-height intensity distribution (0–255, 0–100 IRE)
//! of its channel as a column-by-column histogram, allowing the colorist to
//! quickly assess:
//!
//! - Colour balance (differences in trace height between channels).
//! - Clipping (traces hitting the top of any channel pane).
//! - Lift/gamma/gain per channel.
//!
//! # Layout
//!
//! ```text
//! ┌────────────────┬────────────────┬────────────────┐
//! │   Red (R)      │  Green (G)     │   Blue (B)     │
//! │   waveform     │  waveform      │   waveform     │
//! └────────────────┴────────────────┴────────────────┘
//! ```
//!
//! # Output
//!
//! The renderer produces an RGBA byte buffer (`width × height × 4`).  Each
//! channel pane uses its canonical colour (red/green/blue); an optional
//! horizontal graticule marks 0 %, 25 %, 50 %, 75 %, and 100 % IRE.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use oximedia_core::{OxiError, OxiResult};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Rendering configuration for the RGB parade scope.
#[derive(Debug, Clone)]
pub struct RgbParadeConfig {
    /// Output image width in pixels.  Each channel pane gets roughly a third.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Width in pixels of the black gutter between channel panes.
    pub gutter_width: u32,
    /// Whether to render a horizontal graticule (0/25/50/75/100 % IRE lines).
    pub show_graticule: bool,
    /// Whether to overlay the current channel's clipping count as a brightness
    /// pulse at the top of the pane when pixels clip.
    pub highlight_clipping: bool,
    /// RGBA colour for the red channel waveform.
    pub red_color: [u8; 4],
    /// RGBA colour for the green channel waveform.
    pub green_color: [u8; 4],
    /// RGBA colour for the blue channel waveform.
    pub blue_color: [u8; 4],
    /// RGBA colour for the graticule lines.
    pub graticule_color: [u8; 4],
    /// RGBA colour for the background.
    pub background_color: [u8; 4],
}

impl Default for RgbParadeConfig {
    fn default() -> Self {
        Self {
            width: 768,
            height: 256,
            gutter_width: 4,
            show_graticule: true,
            highlight_clipping: true,
            red_color: [230, 60, 60, 220],
            green_color: [60, 230, 60, 220],
            blue_color: [60, 60, 230, 220],
            graticule_color: [60, 60, 60, 160],
            background_color: [10, 10, 12, 255],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-channel waveform data
// ─────────────────────────────────────────────────────────────────────────────

/// Column-by-column histogram data for a single channel.
///
/// `columns[x][v]` counts how many pixels in column `x` have channel value `v`.
#[derive(Debug, Clone)]
pub struct ChannelWaveform {
    /// Which channel (0=R, 1=G, 2=B).
    pub channel_index: usize,
    /// Number of source columns mapped onto this waveform.
    pub num_columns: usize,
    /// Per-column, per-intensity-bin counts.  Outer index = column, inner = value 0..256.
    pub columns: Vec<[u32; 256]>,
    /// Total pixels with value ≥ 252 (near-clipping).
    pub clipping_count: u64,
    /// Total pixels with value ≤ 3 (near-black / crushed blacks).
    pub black_count: u64,
}

impl ChannelWaveform {
    /// Accumulate pixel data from an RGB24 frame into per-column histograms.
    ///
    /// `num_columns` is the number of output columns to use for the waveform.
    fn build(
        frame: &[u8],
        frame_w: u32,
        frame_h: u32,
        channel_index: usize,
        num_columns: usize,
    ) -> Self {
        let fw = frame_w as usize;
        let fh = frame_h as usize;
        let total_pixels = fw * fh;
        let num_columns = num_columns.max(1);

        let mut columns = vec![[0u32; 256]; num_columns];
        let mut clipping_count = 0u64;
        let mut black_count = 0u64;

        for i in 0..total_pixels {
            let base = i * 3;
            if base + 2 >= frame.len() {
                break;
            }
            let value = frame[base + channel_index];
            let frame_x = i % fw;
            let col = (frame_x * num_columns) / fw;
            if col < num_columns {
                columns[col][value as usize] += 1;
            }
            if value >= 252 {
                clipping_count += 1;
            }
            if value <= 3 {
                black_count += 1;
            }
        }

        Self {
            channel_index,
            num_columns,
            columns,
            clipping_count,
            black_count,
        }
    }

    /// Returns the maximum bin count across all columns and values (for normalisation).
    #[must_use]
    pub fn max_count(&self) -> u32 {
        self.columns
            .iter()
            .flat_map(|col| col.iter().copied())
            .max()
            .unwrap_or(1)
            .max(1)
    }

    /// Returns the mean value (0.0–255.0) for the given output column.
    ///
    /// Returns `None` if `col_index` is out of range.
    #[must_use]
    pub fn column_mean(&self, col_index: usize) -> Option<f32> {
        let col = self.columns.get(col_index)?;
        let total: u64 = col.iter().map(|&c| u64::from(c)).sum();
        if total == 0 {
            return Some(0.0);
        }
        let weighted: u64 = col
            .iter()
            .enumerate()
            .map(|(v, &c)| v as u64 * u64::from(c))
            .sum();
        Some(weighted as f32 / total as f32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RGB Parade
// ─────────────────────────────────────────────────────────────────────────────

/// A complete RGB parade scope with per-channel waveform data.
#[derive(Debug, Clone)]
pub struct RgbParade {
    /// Red channel waveform.
    pub red: ChannelWaveform,
    /// Green channel waveform.
    pub green: ChannelWaveform,
    /// Blue channel waveform.
    pub blue: ChannelWaveform,
}

impl RgbParade {
    /// Analyse an RGB24 frame and build a parade scope.
    ///
    /// `pane_columns` is the number of output columns assigned to each channel
    /// pane; typically `(output_width - 2 * gutter_width) / 3`.
    #[must_use]
    pub fn generate(frame: &[u8], frame_w: u32, frame_h: u32, pane_columns: usize) -> Self {
        Self {
            red: ChannelWaveform::build(frame, frame_w, frame_h, 0, pane_columns),
            green: ChannelWaveform::build(frame, frame_w, frame_h, 1, pane_columns),
            blue: ChannelWaveform::build(frame, frame_w, frame_h, 2, pane_columns),
        }
    }

    /// Returns `true` if any channel has clipping pixels (value ≥ 252).
    #[must_use]
    pub fn has_clipping(&self) -> bool {
        self.red.clipping_count > 0
            || self.green.clipping_count > 0
            || self.blue.clipping_count > 0
    }

    /// Returns the channel with the most clipping pixels (0=R, 1=G, 2=B),
    /// or `None` if there is no clipping.
    #[must_use]
    pub fn dominant_clipping_channel(&self) -> Option<usize> {
        let counts = [
            self.red.clipping_count,
            self.green.clipping_count,
            self.blue.clipping_count,
        ];
        let max = *counts.iter().max()?;
        if max == 0 {
            None
        } else {
            counts.iter().position(|&c| c == max)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rendering
// ─────────────────────────────────────────────────────────────────────────────

/// Render an RGB24 frame as an RGB parade RGBA image.
///
/// # Errors
///
/// Returns an error if the frame buffer is smaller than `frame_w * frame_h * 3`.
pub fn render_rgb_parade(
    frame: &[u8],
    frame_w: u32,
    frame_h: u32,
    config: &RgbParadeConfig,
) -> OxiResult<Vec<u8>> {
    let total_pixels = (frame_w as usize) * (frame_h as usize);
    let expected = total_pixels * 3;
    if frame.len() < expected {
        return Err(OxiError::InvalidData(format!(
            "Frame too small: need {expected}, got {}",
            frame.len()
        )));
    }

    let out_w = config.width as usize;
    let out_h = config.height as usize;
    let gutter = config.gutter_width as usize;
    let total_gutters = gutter * 4; // gutters: |g|R|g|G|g|B|g|
    let pane_w = if out_w > total_gutters {
        (out_w - total_gutters) / 3
    } else {
        4
    };

    let parade = RgbParade::generate(frame, frame_w, frame_h, pane_w);

    let mut buf = vec![0u8; out_w * out_h * 4];
    // Fill background
    for chunk in buf.chunks_exact_mut(4) {
        chunk.copy_from_slice(&config.background_color);
    }

    let pane_offsets = [gutter, gutter * 2 + pane_w, gutter * 3 + pane_w * 2];
    let waveforms = [&parade.red, &parade.green, &parade.blue];
    let colors = [config.red_color, config.green_color, config.blue_color];

    for (pane_idx, (waveform, color)) in waveforms.iter().zip(colors.iter()).enumerate() {
        let x_offset = pane_offsets[pane_idx];
        let max_count = waveform.max_count();

        for (col, histogram) in waveform.columns.iter().enumerate() {
            let abs_x = x_offset + col;
            if abs_x >= out_w {
                break;
            }
            for (value, &count) in histogram.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                // Normalise brightness by square-root mapping (perceptually even)
                let brightness = ((count as f64 / max_count as f64).sqrt() * 255.0) as u8;
                // Map value 0..255 to y coordinate (0 at bottom, 255 at top)
                let py = out_h.saturating_sub(1) - (value * out_h / 256).min(out_h - 1);
                let idx = (py * out_w + abs_x) * 4;
                if idx + 3 < buf.len() {
                    let src_a = color[3] as f64 * brightness as f64 / 255.0 / 255.0;
                    let dst_a = 1.0 - src_a;
                    buf[idx] = (color[0] as f64 * src_a + buf[idx] as f64 * dst_a) as u8;
                    buf[idx + 1] =
                        (color[1] as f64 * src_a + buf[idx + 1] as f64 * dst_a) as u8;
                    buf[idx + 2] =
                        (color[2] as f64 * src_a + buf[idx + 2] as f64 * dst_a) as u8;
                    buf[idx + 3] = 255;
                }
            }
        }

        // Clipping highlight: bright row at the very top of the pane
        if config.highlight_clipping && waveform.clipping_count > 0 {
            for col in 0..pane_w {
                let abs_x = x_offset + col;
                if abs_x >= out_w {
                    break;
                }
                let idx = abs_x * 4; // row 0
                if idx + 3 < buf.len() {
                    buf[idx] = 255;
                    buf[idx + 1] = 80;
                    buf[idx + 2] = 80;
                    buf[idx + 3] = 255;
                }
            }
        }
    }

    // Graticule: horizontal lines at 0/25/50/75/100 %
    if config.show_graticule {
        let pct_marks = [0u32, 25, 50, 75, 100];
        for &pct in &pct_marks {
            // pct = 0 → y = out_h-1 (bottom), pct = 100 → y = 0 (top)
            let y = out_h.saturating_sub(1).saturating_sub(pct as usize * (out_h - 1) / 100);
            for px in 0..out_w {
                let idx = (y * out_w + px) * 4;
                if idx + 3 < buf.len() {
                    let gc = config.graticule_color;
                    let a = gc[3] as f64 / 255.0;
                    let ia = 1.0 - a;
                    buf[idx] = (gc[0] as f64 * a + buf[idx] as f64 * ia) as u8;
                    buf[idx + 1] = (gc[1] as f64 * a + buf[idx + 1] as f64 * ia) as u8;
                    buf[idx + 2] = (gc[2] as f64 * a + buf[idx + 2] as f64 * ia) as u8;
                    buf[idx + 3] = 255;
                }
            }
        }
    }

    Ok(buf)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgb(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        vec![[r, g, b]; (w * h) as usize]
            .into_iter()
            .flatten()
            .collect()
    }

    #[test]
    fn test_render_rgb_parade_output_size() {
        let frame = solid_rgb(16, 16, 128, 128, 128);
        let config = RgbParadeConfig {
            width: 192,
            height: 64,
            ..Default::default()
        };
        let result = render_rgb_parade(&frame, 16, 16, &config).expect("ok");
        assert_eq!(result.len(), 192 * 64 * 4);
    }

    #[test]
    fn test_render_rgb_parade_frame_too_small() {
        let frame = vec![0u8; 5];
        let config = RgbParadeConfig::default();
        assert!(render_rgb_parade(&frame, 10, 10, &config).is_err());
    }

    #[test]
    fn test_channel_waveform_bin_count_solid() {
        // 4×4 red=200 frame → all 16 pixels should land in bin 200 for R channel
        let frame = solid_rgb(4, 4, 200, 100, 50);
        let wf = ChannelWaveform::build(&frame, 4, 4, 0, 4);
        // Total across all columns for value 200
        let total_200: u32 = wf.columns.iter().map(|c| c[200]).sum();
        assert_eq!(total_200, 16, "expected 16 pixels at red=200, got {total_200}");
    }

    #[test]
    fn test_channel_waveform_clipping_detection() {
        let frame = solid_rgb(4, 4, 255, 100, 50);
        let wf = ChannelWaveform::build(&frame, 4, 4, 0, 4);
        assert!(wf.clipping_count > 0);
        // Green (100) should not clip
        let wf_g = ChannelWaveform::build(&frame, 4, 4, 1, 4);
        assert_eq!(wf_g.clipping_count, 0);
    }

    #[test]
    fn test_channel_waveform_black_count() {
        let frame = solid_rgb(4, 4, 0, 0, 0);
        let wf = ChannelWaveform::build(&frame, 4, 4, 0, 4);
        assert_eq!(wf.black_count, 16);
    }

    #[test]
    fn test_rgb_parade_has_clipping_true() {
        let frame = solid_rgb(4, 4, 255, 100, 50);
        let parade = RgbParade::generate(&frame, 4, 4, 4);
        assert!(parade.has_clipping());
    }

    #[test]
    fn test_rgb_parade_has_clipping_false() {
        let frame = solid_rgb(4, 4, 128, 100, 80);
        let parade = RgbParade::generate(&frame, 4, 4, 4);
        assert!(!parade.has_clipping());
    }

    #[test]
    fn test_dominant_clipping_channel() {
        let frame = solid_rgb(4, 4, 255, 100, 50);
        let parade = RgbParade::generate(&frame, 4, 4, 4);
        let ch = parade.dominant_clipping_channel();
        assert_eq!(ch, Some(0), "red should dominate clipping");
    }

    #[test]
    fn test_dominant_clipping_channel_none_when_clean() {
        let frame = solid_rgb(4, 4, 100, 100, 100);
        let parade = RgbParade::generate(&frame, 4, 4, 4);
        assert!(parade.dominant_clipping_channel().is_none());
    }

    #[test]
    fn test_column_mean_solid_frame() {
        let frame = solid_rgb(4, 4, 128, 0, 0);
        let wf = ChannelWaveform::build(&frame, 4, 4, 0, 4);
        for col in 0..4 {
            let mean = wf.column_mean(col).expect("should be Some");
            assert!(
                (mean - 128.0).abs() < 1.0,
                "col {col} mean={mean}"
            );
        }
    }

    #[test]
    fn test_column_mean_out_of_range_returns_none() {
        let frame = solid_rgb(4, 4, 128, 0, 0);
        let wf = ChannelWaveform::build(&frame, 4, 4, 0, 4);
        assert!(wf.column_mean(100).is_none());
    }

    #[test]
    fn test_config_default_values() {
        let cfg = RgbParadeConfig::default();
        assert_eq!(cfg.width, 768);
        assert_eq!(cfg.height, 256);
        assert!(cfg.show_graticule);
        assert!(cfg.highlight_clipping);
    }

    #[test]
    fn test_render_zero_dimension_frame() {
        let frame = vec![];
        let config = RgbParadeConfig {
            width: 48,
            height: 32,
            ..Default::default()
        };
        let result = render_rgb_parade(&frame, 0, 0, &config);
        // 0×0 frame: 0 expected bytes, frame is empty ⟹ not an error
        assert!(result.is_ok());
    }

    #[test]
    fn test_max_count_solid_frame() {
        let frame = solid_rgb(4, 4, 128, 128, 128);
        let wf = ChannelWaveform::build(&frame, 4, 4, 0, 4);
        // 4 columns × 4 rows = 16 pixels; all in one column = 4 per column
        assert!(wf.max_count() >= 1);
    }
}
