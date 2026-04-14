//! RPU metadata visualization and plotting utilities for debugging.
//!
//! Provides text-mode ASCII plots and structured reports for Dolby Vision RPU
//! metadata streams.  These utilities are intentionally dependency-free and
//! produce output suitable for integration in CI pipelines and log files.
//!
//! # Example
//!
//! ```rust
//! use oximedia_dolbyvision::{DolbyVisionRpu, Profile, Level1Metadata};
//! use oximedia_dolbyvision::rpu_visualize::{RpuPlotter, PlotConfig};
//!
//! let mut rpus = Vec::new();
//! for i in 0..24u16 {
//!     let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
//!     rpu.level1 = Some(Level1Metadata { min_pq: 100, max_pq: 2000 + i * 50, avg_pq: 1000 + i * 20 });
//!     rpus.push(rpu);
//! }
//!
//! let config = PlotConfig::default();
//! let plot = RpuPlotter::plot_l1_luminance(&rpus, &config);
//! assert!(plot.contains("max_pq"));
//! ```

#![forbid(unsafe_code)]

use crate::DolbyVisionRpu;

/// Configuration for ASCII plot rendering.
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Width of the plot in characters (columns).
    pub width: usize,
    /// Height of the plot in characters (rows).
    pub height: usize,
    /// Character used for drawing bars/points.
    pub bar_char: char,
    /// Character used for the plot background.
    pub bg_char: char,
    /// Whether to include a legend.
    pub show_legend: bool,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            width: 80,
            height: 20,
            bar_char: '█',
            bg_char: ' ',
            show_legend: true,
        }
    }
}

/// Statistics extracted from a sequence of Level-1 metadata entries.
#[derive(Debug, Clone)]
pub struct L1Statistics {
    /// Number of frames analysed.
    pub frame_count: usize,
    /// Minimum `min_pq` value across all frames.
    pub global_min_pq: u16,
    /// Maximum `max_pq` value across all frames.
    pub global_max_pq: u16,
    /// Mean `avg_pq` across all frames.
    pub mean_avg_pq: f64,
    /// Standard deviation of `avg_pq` across all frames.
    pub std_avg_pq: f64,
    /// p10 percentile of `avg_pq`.
    pub p10_avg_pq: u16,
    /// p50 percentile of `avg_pq`.
    pub p50_avg_pq: u16,
    /// p90 percentile of `avg_pq`.
    pub p90_avg_pq: u16,
}

/// ASCII-art RPU metadata plotter.
///
/// All methods are pure functions that return a `String` containing the rendered
/// plot.  No I/O is performed; the caller decides how to display or persist the
/// output.
pub struct RpuPlotter;

impl RpuPlotter {
    /// Generate an ASCII bar chart of Level-1 luminance values over time.
    ///
    /// Three lines are drawn for `min_pq`, `avg_pq`, and `max_pq` using
    /// different characters (`.`, `+`, `#`).  If an RPU has no L1 metadata
    /// the corresponding column is empty.
    ///
    /// # Arguments
    ///
    /// * `rpus` - Ordered slice of RPU entries (one per frame).
    /// * `config` - Plot configuration.
    ///
    /// # Returns
    ///
    /// A multi-line string representing the ASCII plot.
    #[must_use]
    pub fn plot_l1_luminance(rpus: &[DolbyVisionRpu], config: &PlotConfig) -> String {
        if rpus.is_empty() {
            return "(no frames to plot)\n".to_owned();
        }

        // Collect max_pq values (use 0 for frames without L1)
        let max_pq_vals: Vec<f64> = rpus
            .iter()
            .map(|r| r.level1.as_ref().map_or(0.0, |l| l.max_pq as f64))
            .collect();
        let avg_pq_vals: Vec<f64> = rpus
            .iter()
            .map(|r| r.level1.as_ref().map_or(0.0, |l| l.avg_pq as f64))
            .collect();
        let min_pq_vals: Vec<f64> = rpus
            .iter()
            .map(|r| r.level1.as_ref().map_or(0.0, |l| l.min_pq as f64))
            .collect();

        let global_max = max_pq_vals
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1.0);

        let w = config.width.max(10);
        let h = config.height.max(4);

        // Down-sample or up-sample to `w` columns
        let sample = |vals: &[f64]| -> Vec<f64> {
            (0..w)
                .map(|col| {
                    let idx = (col as f64 * (vals.len() as f64 - 1.0) / (w as f64 - 1.0))
                        .round() as usize;
                    vals.get(idx).copied().unwrap_or(0.0)
                })
                .collect()
        };

        let max_cols = sample(&max_pq_vals);
        let avg_cols = sample(&avg_pq_vals);
        let min_cols = sample(&min_pq_vals);

        // Build grid (height × width) — row 0 is top
        let mut grid = vec![vec![config.bg_char; w]; h];

        let to_row = |v: f64| -> usize {
            let norm = (v / global_max).clamp(0.0, 1.0);
            let row = ((1.0 - norm) * (h as f64 - 1.0)).round() as usize;
            row.min(h - 1)
        };

        for col in 0..w {
            let row_max = to_row(max_cols[col]);
            let row_avg = to_row(avg_cols[col]);
            let row_min = to_row(min_cols[col]);

            if grid[row_max][col] == config.bg_char {
                grid[row_max][col] = '#';
            }
            if grid[row_avg][col] == config.bg_char {
                grid[row_avg][col] = '+';
            }
            if grid[row_min][col] == config.bg_char {
                grid[row_min][col] = '.';
            }
        }

        let mut out = String::new();
        out.push_str(&format!(
            "┌─ L1 Luminance (PQ) — {} frames ─\n",
            rpus.len()
        ));

        for (row_idx, row) in grid.iter().enumerate() {
            let pq_label = global_max * (1.0 - row_idx as f64 / (h as f64 - 1.0));
            out.push_str(&format!("{:5.0} │", pq_label));
            for &ch in row {
                out.push(ch);
            }
            out.push('\n');
        }

        // X-axis
        out.push_str("      └");
        for _ in 0..w {
            out.push('─');
        }
        out.push('\n');
        out.push_str(&format!(
            "       frame 0{:>width$}\n",
            format!("frame {}", rpus.len() - 1),
            width = w - 7
        ));

        if config.show_legend {
            out.push_str("  Legend: # max_pq   + avg_pq   . min_pq\n");
        }

        out
    }

    /// Compute [`L1Statistics`] for a sequence of RPU entries.
    ///
    /// Frames without L1 metadata are skipped (they do not contribute to the
    /// statistics).
    ///
    /// # Returns
    ///
    /// `None` if no frames have L1 metadata.
    #[must_use]
    pub fn compute_l1_stats(rpus: &[DolbyVisionRpu]) -> Option<L1Statistics> {
        let l1_frames: Vec<_> = rpus.iter().filter_map(|r| r.level1.as_ref()).collect();
        if l1_frames.is_empty() {
            return None;
        }

        let global_min_pq = l1_frames.iter().map(|l| l.min_pq).min().unwrap_or(0);
        let global_max_pq = l1_frames.iter().map(|l| l.max_pq).max().unwrap_or(0);

        let avg_vals: Vec<f64> = l1_frames.iter().map(|l| l.avg_pq as f64).collect();
        let n = avg_vals.len() as f64;
        let mean = avg_vals.iter().sum::<f64>() / n;
        let variance = avg_vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let mut sorted_avg: Vec<u16> = l1_frames.iter().map(|l| l.avg_pq).collect();
        sorted_avg.sort_unstable();
        let p10 = sorted_avg[(sorted_avg.len() * 10 / 100).min(sorted_avg.len() - 1)];
        let p50 = sorted_avg[sorted_avg.len() / 2];
        let p90 = sorted_avg[(sorted_avg.len() * 90 / 100).min(sorted_avg.len() - 1)];

        Some(L1Statistics {
            frame_count: l1_frames.len(),
            global_min_pq,
            global_max_pq,
            mean_avg_pq: mean,
            std_avg_pq: std_dev,
            p10_avg_pq: p10,
            p50_avg_pq: p50,
            p90_avg_pq: p90,
        })
    }

    /// Render a human-readable text report for a slice of RPU entries.
    ///
    /// Includes profile distribution, L1 statistics summary, and per-frame
    /// table (limited to the first `max_rows` rows to keep output manageable).
    ///
    /// # Arguments
    ///
    /// * `rpus` - Ordered slice of RPU entries.
    /// * `max_rows` - Maximum number of per-frame rows to include in the table.
    #[must_use]
    pub fn text_report(rpus: &[DolbyVisionRpu], max_rows: usize) -> String {
        let mut out = String::new();

        out.push_str("=== Dolby Vision RPU Debug Report ===\n");
        out.push_str(&format!("Total frames: {}\n\n", rpus.len()));

        // Profile distribution
        let mut profile_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for rpu in rpus {
            let key = format!("{:?}", rpu.profile);
            *profile_counts.entry(key).or_insert(0) += 1;
        }
        out.push_str("--- Profile Distribution ---\n");
        let mut profile_list: Vec<_> = profile_counts.iter().collect();
        profile_list.sort_by_key(|(k, _)| k.as_str());
        for (profile, count) in &profile_list {
            out.push_str(&format!("  {:20} {:6} frames\n", profile, count));
        }
        out.push('\n');

        // L1 statistics
        if let Some(stats) = Self::compute_l1_stats(rpus) {
            out.push_str("--- L1 Luminance Statistics ---\n");
            out.push_str(&format!("  Frames with L1:   {}\n", stats.frame_count));
            out.push_str(&format!("  Global min_pq:    {}\n", stats.global_min_pq));
            out.push_str(&format!("  Global max_pq:    {}\n", stats.global_max_pq));
            out.push_str(&format!("  Mean avg_pq:      {:.1}\n", stats.mean_avg_pq));
            out.push_str(&format!("  Std  avg_pq:      {:.1}\n", stats.std_avg_pq));
            out.push_str(&format!("  p10  avg_pq:      {}\n", stats.p10_avg_pq));
            out.push_str(&format!("  p50  avg_pq:      {}\n", stats.p50_avg_pq));
            out.push_str(&format!("  p90  avg_pq:      {}\n", stats.p90_avg_pq));
            out.push('\n');
        }

        // Per-frame table header
        let rows = rpus.len().min(max_rows);
        if rows > 0 {
            out.push_str("--- Per-Frame Table (first ");
            out.push_str(&rows.to_string());
            out.push_str(" frames) ---\n");
            out.push_str(&format!(
                "{:>6}  {:>12}  {:>8}  {:>8}  {:>8}  {}\n",
                "frame", "profile", "min_pq", "avg_pq", "max_pq", "L2/L4/L5"
            ));
            out.push_str(&"-".repeat(70));
            out.push('\n');

            for (idx, rpu) in rpus.iter().take(rows).enumerate() {
                let (min_pq, avg_pq, max_pq) = rpu
                    .level1
                    .as_ref()
                    .map(|l| (l.min_pq, l.avg_pq, l.max_pq))
                    .map(|(a, b, c)| {
                        (a.to_string(), b.to_string(), c.to_string())
                    })
                    .unwrap_or_else(|| ("-".to_owned(), "-".to_owned(), "-".to_owned()));

                let flags = format!(
                    "{}{}{}",
                    if rpu.level2.is_some() { "L2 " } else { "" },
                    if rpu.level4.is_some() { "L4 " } else { "" },
                    if rpu.level5.is_some() { "L5 " } else { "" },
                );

                out.push_str(&format!(
                    "{:>6}  {:>12}  {:>8}  {:>8}  {:>8}  {}\n",
                    idx,
                    format!("{:?}", rpu.profile),
                    min_pq,
                    avg_pq,
                    max_pq,
                    flags.trim()
                ));
            }

            if rpus.len() > max_rows {
                out.push_str(&format!(
                    "  ... and {} more frames\n",
                    rpus.len() - max_rows
                ));
            }
        }

        out
    }

    /// Render an ASCII histogram of `avg_pq` values across a sequence of RPU entries.
    ///
    /// The histogram uses `bins` buckets spanning `[0, 4095]` (the full PQ range).
    ///
    /// # Arguments
    ///
    /// * `rpus` - RPU entries to histogram.
    /// * `bins` - Number of histogram bins (clamped to at least 1).
    /// * `bar_width` - Maximum bar width in characters.
    #[must_use]
    pub fn histogram_avg_pq(rpus: &[DolbyVisionRpu], bins: usize, bar_width: usize) -> String {
        let bins = bins.max(1);
        let bar_width = bar_width.max(1);

        let mut counts = vec![0usize; bins];
        let mut total = 0usize;

        for rpu in rpus {
            if let Some(l1) = &rpu.level1 {
                let bin = ((l1.avg_pq as usize * bins) / 4096).min(bins - 1);
                counts[bin] += 1;
                total += 1;
            }
        }

        if total == 0 {
            return "(no L1 metadata found)\n".to_owned();
        }

        let max_count = counts.iter().copied().max().unwrap_or(1).max(1);

        let mut out = String::new();
        out.push_str("avg_pq histogram\n");
        out.push_str("PQ range   count   bar\n");
        out.push_str(&"-".repeat(40));
        out.push('\n');

        for (i, &count) in counts.iter().enumerate() {
            let lo = i * 4096 / bins;
            let hi = (i + 1) * 4096 / bins - 1;
            let bar_len = (count * bar_width) / max_count;
            let bar: String = "█".repeat(bar_len);
            out.push_str(&format!(
                "{:4}-{:4}  {:6}  {}\n",
                lo, hi, count, bar
            ));
        }

        out
    }
}

/// Generate a compact single-line summary string for one RPU entry.
///
/// Useful for embedding in log lines.
#[must_use]
pub fn rpu_summary_line(frame_idx: usize, rpu: &DolbyVisionRpu) -> String {
    let l1 = rpu
        .level1
        .as_ref()
        .map(|l| format!("L1(min={} avg={} max={})", l.min_pq, l.avg_pq, l.max_pq))
        .unwrap_or_else(|| "L1(-)".to_owned());

    let extras = format!(
        "{}{}{}{}",
        if rpu.level2.is_some() { " L2" } else { "" },
        if rpu.level4.is_some() { " L4" } else { "" },
        if rpu.level5.is_some() { " L5" } else { "" },
        if rpu.level6.is_some() { " L6" } else { "" },
    );

    format!(
        "[frame {:06}] profile={:?} {}{}",
        frame_idx, rpu.profile, l1, extras
    )
}

/// Detect and report frames where luminance changes abruptly (potential scene cuts).
///
/// An abrupt change is flagged when `|curr.avg_pq - prev.avg_pq| >= threshold`.
///
/// # Returns
///
/// A `Vec` of `(frame_index, delta_avg_pq)` pairs for each flagged transition.
#[must_use]
pub fn detect_luminance_jumps(rpus: &[DolbyVisionRpu], threshold: u16) -> Vec<(usize, u16)> {
    let mut jumps = Vec::new();
    let mut prev_avg: Option<u16> = None;

    for (idx, rpu) in rpus.iter().enumerate() {
        if let Some(l1) = &rpu.level1 {
            if let Some(prev) = prev_avg {
                let delta = l1.avg_pq.abs_diff(prev);
                if delta >= threshold {
                    jumps.push((idx, delta));
                }
            }
            prev_avg = Some(l1.avg_pq);
        }
    }

    jumps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::Level1Metadata;
    use crate::Profile;

    fn make_rpu_with_l1(min: u16, avg: u16, max: u16) -> DolbyVisionRpu {
        let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
        rpu.level1 = Some(Level1Metadata {
            min_pq: min,
            avg_pq: avg,
            max_pq: max,
        });
        rpu
    }

    #[test]
    fn test_plot_l1_luminance_empty() {
        let plot = RpuPlotter::plot_l1_luminance(&[], &PlotConfig::default());
        assert!(plot.contains("no frames"));
    }

    #[test]
    fn test_plot_l1_luminance_single_frame() {
        let rpus = vec![make_rpu_with_l1(100, 1000, 3000)];
        let plot = RpuPlotter::plot_l1_luminance(&rpus, &PlotConfig::default());
        assert!(plot.contains("L1 Luminance"));
        assert!(plot.contains("max_pq"));
    }

    #[test]
    fn test_compute_l1_stats_empty() {
        let rpus: Vec<DolbyVisionRpu> = vec![DolbyVisionRpu::new(Profile::Profile8)];
        assert!(RpuPlotter::compute_l1_stats(&rpus).is_none());
    }

    #[test]
    fn test_compute_l1_stats_values() {
        let rpus: Vec<DolbyVisionRpu> = vec![
            make_rpu_with_l1(100, 1000, 3000),
            make_rpu_with_l1(200, 2000, 4000),
        ];
        let stats = RpuPlotter::compute_l1_stats(&rpus).expect("stats should exist");
        assert_eq!(stats.frame_count, 2);
        assert_eq!(stats.global_min_pq, 100);
        assert_eq!(stats.global_max_pq, 4000);
        assert!((stats.mean_avg_pq - 1500.0).abs() < 1.0);
    }

    #[test]
    fn test_text_report_contains_sections() {
        let rpus = vec![make_rpu_with_l1(0, 1000, 3000)];
        let report = RpuPlotter::text_report(&rpus, 10);
        assert!(report.contains("Profile Distribution"));
        assert!(report.contains("L1 Luminance Statistics"));
        assert!(report.contains("Per-Frame Table"));
    }

    #[test]
    fn test_histogram_avg_pq_no_l1() {
        let rpus = vec![DolbyVisionRpu::new(Profile::Profile8)];
        let hist = RpuPlotter::histogram_avg_pq(&rpus, 16, 40);
        assert!(hist.contains("no L1 metadata"));
    }

    #[test]
    fn test_histogram_avg_pq_with_data() {
        let rpus: Vec<_> = (0..16u16)
            .map(|i| make_rpu_with_l1(0, i * 256, 4000))
            .collect();
        let hist = RpuPlotter::histogram_avg_pq(&rpus, 16, 40);
        assert!(hist.contains("avg_pq histogram"));
    }

    #[test]
    fn test_rpu_summary_line() {
        let rpu = make_rpu_with_l1(100, 1000, 3500);
        let line = rpu_summary_line(42, &rpu);
        assert!(line.contains("frame 000042"));
        assert!(line.contains("avg=1000"));
    }

    #[test]
    fn test_detect_luminance_jumps() {
        let mut rpus: Vec<_> = (0..10)
            .map(|_| make_rpu_with_l1(0, 1000, 3000))
            .collect();
        // Sharp jump at frame 5: avg goes from 1000 to 3500 (+2500), then back to 1000 (-2500)
        rpus[5].level1 = Some(Level1Metadata {
            min_pq: 0,
            avg_pq: 3500,
            max_pq: 4000,
        });
        let jumps = detect_luminance_jumps(&rpus, 500);
        // Frame 5 (jump up from 1000→3500=2500) and frame 6 (jump down from 3500→1000=2500)
        assert_eq!(jumps.len(), 2);
        assert_eq!(jumps[0].0, 5);
        assert_eq!(jumps[1].0, 6);
    }
}
