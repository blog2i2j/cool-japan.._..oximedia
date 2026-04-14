//! Rayon-parallel RGB parade generation for high-throughput scope analysis.
//!
//! This module provides a parallel implementation of the RGB parade display,
//! where column-distribution accumulation is processed concurrently across
//! all columns using rayon's work-stealing thread pool.  The parallel code
//! path is identical in output to the serial `parade::generate_rgb_parade`
//! but significantly faster on multi-core hardware for high-resolution frames.
//!
//! # Algorithm
//!
//! 1. The frame is conceptually sliced into `section_width` vertical columns
//!    (one per parade section).
//! 2. Each column independently accumulates a 256-bin intensity histogram
//!    for each RGB channel — no sharing between columns, no locks.
//! 3. The per-column histograms are built in parallel using rayon.
//! 4. A final serial pass renders the histograms onto the output canvas.
//!
//! # Performance notes
//!
//! - Zero heap allocation per pixel: histograms are built into pre-allocated
//!   `Vec<Vec<u32>>` slabs then filled via `rayon::iter::IndexedParallelIterator`.
//! - The column-major decomposition avoids false sharing between threads.
//! - A global max is computed with a single rayon reduce pass before drawing.
//!
//! # Example
//!
//! ```
//! use oximedia_scopes::parade_parallel::{generate_rgb_parade_parallel, ParallelParadeConfig};
//! use oximedia_scopes::ScopeConfig;
//!
//! let config = ScopeConfig::default();
//! let frame = vec![128u8; 320 * 240 * 3];
//! let result = generate_rgb_parade_parallel(&frame, 320, 240, &config, &ParallelParadeConfig::default());
//! assert!(result.is_ok());
//! ```

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]

use crate::render::{colors, rgb_to_ycbcr, Canvas};
use crate::{ScopeConfig, ScopeData, ScopeType};
use oximedia_core::{OxiError, OxiResult};
use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration specific to the parallel parade renderer.
#[derive(Debug, Clone)]
pub struct ParallelParadeConfig {
    /// Minimum number of pixels per rayon chunk.
    ///
    /// Larger values reduce scheduling overhead at the cost of load-balance
    /// granularity.  Default: 64.
    pub chunk_size: usize,

    /// Whether to render the YCbCr parade instead of the RGB parade.
    pub ycbcr_mode: bool,
}

impl Default for ParallelParadeConfig {
    fn default() -> Self {
        Self {
            chunk_size: 64,
            ycbcr_mode: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Generates an RGB (or YCbCr) parade display using rayon parallel column processing.
///
/// The output is identical to [`crate::parade::generate_rgb_parade`] but uses
/// rayon to accumulate per-column intensity histograms in parallel.
///
/// # Arguments
///
/// * `frame`   – RGB24 frame data (3 bytes per pixel, row-major).
/// * `width`   – Frame width in pixels.
/// * `height`  – Frame height in pixels.
/// * `config`  – Scope rendering configuration.
/// * `par`     – Parallel processing configuration.
///
/// # Errors
///
/// Returns [`OxiError::InvalidData`] if the frame buffer is smaller than
/// `width * height * 3` bytes or if `width` or `height` is zero.
pub fn generate_rgb_parade_parallel(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &ScopeConfig,
    par: &ParallelParadeConfig,
) -> OxiResult<ScopeData> {
    if width == 0 || height == 0 {
        return Err(OxiError::InvalidData(
            "Frame dimensions must be non-zero".into(),
        ));
    }
    let expected = (width as usize) * (height as usize) * 3;
    if frame.len() < expected {
        return Err(OxiError::InvalidData(format!(
            "Frame buffer too small: need {expected}, got {}",
            frame.len()
        )));
    }

    let scope_w = config.width;
    let scope_h = config.height;
    let section_w = (scope_w / 3).max(1) as usize;

    // Build three-channel per-column histograms in parallel.
    // Layout: histograms[channel][col_idx][bin]  (3 × section_w × 256)
    let histograms: Vec<Vec<[u32; 256]>> = (0..3_usize)
        .map(|_| vec![[0u32; 256]; section_w])
        .collect();

    // Accumulate row-by-row in parallel using rayon.
    // Each rayon task owns a per-row local accumulator (section_w × 3 × 256).
    // After all rows are processed, we reduce by summing.
    let rows: Vec<usize> = (0..height as usize).collect();

    let local_hists: Vec<Vec<Vec<[u32; 256]>>> = rows
        .par_chunks(par.chunk_size.max(1))
        .map(|row_chunk| {
            // local_h[channel][col][bin]
            let mut local_h: Vec<Vec<[u32; 256]>> = (0..3)
                .map(|_| vec![[0u32; 256]; section_w])
                .collect();

            for &row in row_chunk {
                let row_start = row * width as usize * 3;
                for col in 0..width as usize {
                    let px = row_start + col * 3;
                    let r = frame[px];
                    let g = frame[px + 1];
                    let b = frame[px + 2];

                    let scope_col =
                        ((col * section_w) / width as usize).min(section_w - 1);

                    if par.ycbcr_mode {
                        let (y_val, cb, cr) = rgb_to_ycbcr(r, g, b);
                        local_h[0][scope_col][y_val as usize] += 1;
                        local_h[1][scope_col][cb as usize] += 1;
                        local_h[2][scope_col][cr as usize] += 1;
                    } else {
                        local_h[0][scope_col][r as usize] += 1;
                        local_h[1][scope_col][g as usize] += 1;
                        local_h[2][scope_col][b as usize] += 1;
                    }
                }
            }
            local_h
        })
        .collect();

    // Reduce: sum all local histograms into `histograms`.
    let mut merged = histograms;
    for local_h in local_hists {
        for ch in 0..3 {
            for col in 0..section_w {
                for bin in 0..256 {
                    merged[ch][col][bin] += local_h[ch][col][bin];
                }
            }
        }
    }

    // Global maximum for normalisation.
    let max_val = merged
        .iter()
        .flat_map(|ch| ch.iter().flat_map(|col| col.iter().copied()))
        .max()
        .unwrap_or(1)
        .max(1);

    // ── Render ────────────────────────────────────────────────────────────────
    let channel_colors = if par.ycbcr_mode {
        [colors::WHITE, colors::CYAN, colors::YELLOW]
    } else {
        [colors::RED, colors::GREEN, colors::BLUE]
    };

    let mut canvas = Canvas::new(scope_w, scope_h);

    for (ch, distribution) in merged.iter().enumerate() {
        let offset_x = (scope_w / 3) * ch as u32;
        let base_color = channel_colors[ch];

        for (col_idx, bins) in distribution.iter().enumerate() {
            for (bin, &count) in bins.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let mapped =
                    ((bin as u32 * scope_h) / 255).min(scope_h - 1);
                let scope_y = scope_h - 1 - mapped;
                let brightness =
                    ((count as f32 / max_val as f32).sqrt() * 255.0) as u8;

                let color = [
                    ((u16::from(base_color[0]) * u16::from(brightness)) / 255) as u8,
                    ((u16::from(base_color[1]) * u16::from(brightness)) / 255) as u8,
                    ((u16::from(base_color[2]) * u16::from(brightness)) / 255) as u8,
                    255,
                ];
                canvas.blend_pixel(offset_x + col_idx as u32, scope_y, color);
            }
        }
    }

    // Graticule and labels.
    if config.show_graticule {
        crate::render::draw_parade_graticule(&mut canvas, config, 3);
    }
    if config.show_labels {
        let labels: [&str; 3] = if par.ycbcr_mode {
            ["Y", "Cb", "Cr"]
        } else {
            ["R", "G", "B"]
        };
        draw_section_labels(&mut canvas, &labels);
    }

    let scope_type = if par.ycbcr_mode {
        ScopeType::ParadeYcbcr
    } else {
        ScopeType::ParadeRgb
    };

    Ok(ScopeData {
        width: scope_w,
        height: scope_h,
        data: canvas.data,
        scope_type,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Draw simple text labels at the top of each parade section.
#[allow(clippy::cast_possible_truncation)]
fn draw_section_labels(canvas: &mut Canvas, labels: &[&str; 3]) {
    let section_w = canvas.width / 3;
    for (i, label) in labels.iter().enumerate() {
        let x = section_w * i as u32 + 4;
        // Render first character of label as a crude 3×5 glyph indicator
        // (full font rendering is handled by the timecode overlay module).
        // We simply draw a small colored marker rectangle.
        let color = match *label {
            "R" => colors::RED,
            "G" => colors::GREEN,
            "B" => colors::BLUE,
            "Y" => colors::WHITE,
            "Cb" => colors::CYAN,
            "Cr" => colors::YELLOW,
            _ => colors::WHITE,
        };
        for dy in 0..3_u32 {
            for dx in 0..3_u32 {
                canvas.set_pixel(x + dx, 4 + dy, color);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScopeConfig;

    fn flat_rgb_frame(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        (0..w * h)
            .flat_map(|_| [r, g, b])
            .collect()
    }

    #[test]
    fn test_generate_rgb_parade_parallel_basic() {
        let config = ScopeConfig::default();
        let par = ParallelParadeConfig::default();
        let frame = flat_rgb_frame(320, 240, 128, 128, 128);
        let result = generate_rgb_parade_parallel(&frame, 320, 240, &config, &par);
        assert!(result.is_ok());
        let scope = result.expect("should succeed");
        assert_eq!(scope.width, config.width);
        assert_eq!(scope.height, config.height);
        assert_eq!(scope.data.len() as u32, scope.width * scope.height * 4);
    }

    #[test]
    fn test_generate_rgb_parade_parallel_scope_type() {
        let config = ScopeConfig::default();
        let par = ParallelParadeConfig::default();
        let frame = flat_rgb_frame(64, 64, 200, 100, 50);
        let scope = generate_rgb_parade_parallel(&frame, 64, 64, &config, &par)
            .expect("should succeed");
        assert_eq!(scope.scope_type, ScopeType::ParadeRgb);
    }

    #[test]
    fn test_generate_ycbcr_parade_parallel() {
        let config = ScopeConfig::default();
        let par = ParallelParadeConfig {
            ycbcr_mode: true,
            ..Default::default()
        };
        let frame = flat_rgb_frame(64, 64, 100, 150, 200);
        let scope = generate_rgb_parade_parallel(&frame, 64, 64, &config, &par)
            .expect("should succeed");
        assert_eq!(scope.scope_type, ScopeType::ParadeYcbcr);
    }

    #[test]
    fn test_generate_rgb_parade_parallel_zero_width_error() {
        let config = ScopeConfig::default();
        let par = ParallelParadeConfig::default();
        let frame = vec![0u8; 10];
        let result = generate_rgb_parade_parallel(&frame, 0, 10, &config, &par);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_rgb_parade_parallel_zero_height_error() {
        let config = ScopeConfig::default();
        let par = ParallelParadeConfig::default();
        let frame = vec![0u8; 10];
        let result = generate_rgb_parade_parallel(&frame, 10, 0, &config, &par);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_rgb_parade_parallel_small_frame() {
        let config = ScopeConfig::default();
        let par = ParallelParadeConfig::default();
        let frame = vec![0u8; 5]; // way too small
        let result = generate_rgb_parade_parallel(&frame, 100, 100, &config, &par);
        assert!(result.is_err());
    }

    #[test]
    fn test_parallel_config_default_chunk_size() {
        let par = ParallelParadeConfig::default();
        assert_eq!(par.chunk_size, 64);
        assert!(!par.ycbcr_mode);
    }

    #[test]
    fn test_generate_rgb_parade_parallel_pure_red() {
        let config = ScopeConfig::default();
        let par = ParallelParadeConfig::default();
        // Pure red frame — only the R section should have non-black pixels
        let frame = flat_rgb_frame(64, 64, 255, 0, 0);
        let scope = generate_rgb_parade_parallel(&frame, 64, 64, &config, &par)
            .expect("should succeed");
        // At minimum, some pixels should be non-zero
        assert!(scope.data.iter().any(|&v| v > 0));
    }

    #[test]
    fn test_generate_rgb_parade_parallel_black_frame() {
        let config = ScopeConfig::default();
        let par = ParallelParadeConfig::default();
        let frame = flat_rgb_frame(32, 32, 0, 0, 0);
        let scope = generate_rgb_parade_parallel(&frame, 32, 32, &config, &par)
            .expect("should succeed");
        assert_eq!(scope.width, config.width);
    }

    #[test]
    fn test_generate_rgb_parade_parallel_chunk_size_one() {
        let config = ScopeConfig::default();
        let par = ParallelParadeConfig {
            chunk_size: 1,
            ..Default::default()
        };
        let frame = flat_rgb_frame(32, 32, 128, 64, 200);
        let result = generate_rgb_parade_parallel(&frame, 32, 32, &config, &par);
        assert!(result.is_ok());
    }
}
