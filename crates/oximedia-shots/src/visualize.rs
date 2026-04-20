//! Shot visualization and timeline generation.

use crate::types::{Scene, Shot, ShotType};
use std::io::Write;

/// Timeline visualizer for shots and scenes.
pub struct TimelineVisualizer {
    /// Width of the timeline in characters.
    width: usize,
    /// Height of each track in characters.
    track_height: usize,
}

impl TimelineVisualizer {
    /// Create a new timeline visualizer.
    #[must_use]
    pub const fn new(width: usize, track_height: usize) -> Self {
        Self {
            width,
            track_height,
        }
    }

    /// Generate ASCII timeline visualization.
    #[must_use]
    pub fn generate_ascii_timeline(&self, shots: &[Shot], scenes: &[Scene]) -> String {
        if shots.is_empty() {
            return String::from("No shots to visualize");
        }

        let mut output = String::new();

        // Calculate timeline scale
        // shots is non-empty (early-return guard above); last() is always Some.
        let max_time = if let Some(last_shot) = shots.last() {
            last_shot.end.to_seconds()
        } else {
            return String::from("No shots to visualize");
        };
        let time_scale = self.width as f64 / max_time;

        output.push_str("TIMELINE VISUALIZATION\n");
        output.push_str(&"=".repeat(self.width));
        output.push('\n');

        // Draw time markers
        output.push_str(&self.generate_time_markers(max_time));
        output.push('\n');

        // Draw shot track
        output.push_str("Shots: ");
        output.push_str(&self.generate_shot_track(shots, time_scale));
        output.push('\n');

        // Draw scene track
        if !scenes.is_empty() {
            output.push_str("Scenes:");
            output.push_str(&self.generate_scene_track(scenes, time_scale));
            output.push('\n');
        }

        // Draw legend
        output.push('\n');
        output.push_str(&self.generate_legend());

        output
    }

    /// Generate time markers.
    fn generate_time_markers(&self, max_time: f64) -> String {
        let mut markers = String::new();
        markers.push_str("Time:  ");

        let num_markers = 10;
        let marker_interval = max_time / num_markers as f64;

        for i in 0..=num_markers {
            let time = marker_interval * i as f64;
            let marker = format!("{:.1}s", time);
            markers.push_str(&marker);

            if i < num_markers {
                let spacing = (self.width / num_markers).saturating_sub(marker.len());
                markers.push_str(&" ".repeat(spacing));
            }
        }

        markers
    }

    /// Generate shot track.
    fn generate_shot_track(&self, shots: &[Shot], time_scale: f64) -> String {
        let mut track = vec![' '; self.width];

        for shot in shots {
            let start_pos = (shot.start.to_seconds() * time_scale) as usize;
            let end_pos = (shot.end.to_seconds() * time_scale) as usize;

            let char = self.shot_type_to_char(shot.shot_type);

            for pos in start_pos..end_pos.min(self.width) {
                track[pos] = char;
            }

            // Mark transitions
            if start_pos < self.width {
                track[start_pos] = '|';
            }
        }

        track.iter().collect()
    }

    /// Generate scene track.
    fn generate_scene_track(&self, scenes: &[Scene], time_scale: f64) -> String {
        let mut track = vec![' '; self.width];

        for (i, scene) in scenes.iter().enumerate() {
            let start_pos = (scene.start.to_seconds() * time_scale) as usize;
            let end_pos = (scene.end.to_seconds() * time_scale) as usize;

            let char = if i % 2 == 0 { '━' } else { '─' };

            for pos in start_pos..end_pos.min(self.width) {
                track[pos] = char;
            }

            // Mark scene boundaries
            if start_pos < self.width {
                track[start_pos] = '┃';
            }
        }

        track.iter().collect()
    }

    /// Convert shot type to visualization character.
    const fn shot_type_to_char(&self, shot_type: ShotType) -> char {
        match shot_type {
            ShotType::ExtremeCloseUp => 'E',
            ShotType::CloseUp => 'C',
            ShotType::MediumCloseUp => 'M',
            ShotType::MediumShot => 'm',
            ShotType::MediumLongShot => 'l',
            ShotType::LongShot => 'L',
            ShotType::ExtremeLongShot => 'X',
            ShotType::Unknown => '?',
        }
    }

    /// Generate legend.
    fn generate_legend(&self) -> String {
        let mut legend = String::new();

        legend.push_str("LEGEND:\n");
        legend.push_str("  Shot Types: E=ECU, C=CU, M=MCU, m=MS, l=MLS, L=LS, X=ELS, ?=Unknown\n");
        legend.push_str("  Transitions: | = Cut\n");
        legend.push_str("  Scenes: ┃ = Scene boundary, ━/─ = Scene\n");

        legend
    }

    /// Export timeline as SVG.
    ///
    /// # Errors
    ///
    /// Returns error if writing fails.
    pub fn export_svg<W: Write>(
        &self,
        shots: &[Shot],
        scenes: &[Scene],
        writer: &mut W,
    ) -> std::io::Result<()> {
        let max_time = shots.last().map_or(0.0, |s| s.end.to_seconds());
        let svg_width = 1000;
        let svg_height = 400;
        let time_scale = svg_width as f64 / max_time;

        // SVG header
        writeln!(writer, r#"<?xml version="1.0" encoding="UTF-8"?>"#)?;
        writeln!(
            writer,
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            svg_width, svg_height
        )?;

        // Background
        writeln!(
            writer,
            r#"  <rect width="{}" height="{}" fill="white"/>"#,
            svg_width, svg_height
        )?;

        // Draw scenes
        for (i, scene) in scenes.iter().enumerate() {
            let x = scene.start.to_seconds() * time_scale;
            let width = (scene.end.to_seconds() - scene.start.to_seconds()) * time_scale;
            let fill = if i % 2 == 0 { "#e0e0e0" } else { "#f0f0f0" };

            writeln!(
                writer,
                r#"  <rect x="{}" y="50" width="{}" height="100" fill="{}"/>"#,
                x, width, fill
            )?;
        }

        // Draw shots
        for shot in shots {
            let x = shot.start.to_seconds() * time_scale;
            let width = (shot.end.to_seconds() - shot.start.to_seconds()) * time_scale;
            let color = self.shot_type_to_color(shot.shot_type);

            writeln!(
                writer,
                r#"  <rect x="{}" y="100" width="{}" height="50" fill="{}" stroke="black" stroke-width="1"/>"#,
                x, width, color
            )?;
        }

        // SVG footer
        writeln!(writer, "</svg>")?;

        Ok(())
    }

    /// Get color for shot type.
    const fn shot_type_to_color(&self, shot_type: ShotType) -> &'static str {
        match shot_type {
            ShotType::ExtremeCloseUp => "#ff0000",
            ShotType::CloseUp => "#ff6600",
            ShotType::MediumCloseUp => "#ffcc00",
            ShotType::MediumShot => "#00ff00",
            ShotType::MediumLongShot => "#0099ff",
            ShotType::LongShot => "#0000ff",
            ShotType::ExtremeLongShot => "#6600ff",
            ShotType::Unknown => "#cccccc",
        }
    }
}

impl Default for TimelineVisualizer {
    fn default() -> Self {
        Self::new(80, 3)
    }
}

/// Shot distribution analyzer.
pub struct DistributionAnalyzer;

impl DistributionAnalyzer {
    /// Create a new distribution analyzer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Generate histogram of shot durations.
    #[must_use]
    pub fn duration_histogram(&self, shots: &[Shot], bins: usize) -> Vec<(f64, usize)> {
        if shots.is_empty() {
            return Vec::new();
        }

        let mut durations: Vec<f64> = shots.iter().map(|s| s.duration_seconds()).collect();
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_duration = durations[0];
        let max_duration = durations[durations.len() - 1];
        let bin_size = (max_duration - min_duration) / bins as f64;

        let mut histogram = vec![0; bins];

        for &duration in &durations {
            let bin_idx = if bin_size > 0.0 {
                ((duration - min_duration) / bin_size)
                    .floor()
                    .min(bins as f64 - 1.0) as usize
            } else {
                0
            };

            histogram[bin_idx] += 1;
        }

        histogram
            .into_iter()
            .enumerate()
            .map(|(i, count)| {
                let bin_start = min_duration + (i as f64 * bin_size);
                (bin_start, count)
            })
            .collect()
    }

    /// Generate ASCII bar chart.
    #[must_use]
    pub fn ascii_bar_chart(&self, shots: &[Shot], bins: usize) -> String {
        let histogram = self.duration_histogram(shots, bins);

        if histogram.is_empty() {
            return String::from("No data to display");
        }

        let mut chart = String::new();
        chart.push_str("SHOT DURATION DISTRIBUTION\n");
        chart.push_str("=========================\n\n");

        let max_count = histogram.iter().map(|(_, count)| *count).max().unwrap_or(0);
        let bar_width = 50;

        for (bin_start, count) in histogram {
            let bar_len = (count * bar_width).checked_div(max_count).unwrap_or(0);

            chart.push_str(&format!(
                "{:5.2}s: {} ({})\n",
                bin_start,
                "█".repeat(bar_len),
                count
            ));
        }

        chart
    }
}

impl Default for DistributionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CameraAngle, CompositionAnalysis, CoverageType, TransitionType};
    use oximedia_core::types::{Rational, Timestamp};

    #[test]
    fn test_visualizer_creation() {
        let _viz = TimelineVisualizer::new(80, 3);
    }

    #[test]
    fn test_ascii_timeline_empty() {
        let viz = TimelineVisualizer::default();
        let timeline = viz.generate_ascii_timeline(&[], &[]);
        assert!(timeline.contains("No shots"));
    }

    #[test]
    fn test_ascii_timeline_with_shots() {
        let viz = TimelineVisualizer::default();
        let shot = Shot {
            id: 1,
            start: Timestamp::new(0, Rational::new(1, 30)),
            end: Timestamp::new(60, Rational::new(1, 30)),
            shot_type: ShotType::MediumShot,
            angle: CameraAngle::EyeLevel,
            movements: Vec::new(),
            composition: CompositionAnalysis {
                rule_of_thirds: 0.5,
                symmetry: 0.5,
                balance: 0.5,
                leading_lines: 0.5,
                depth: 0.5,
            },
            coverage: CoverageType::Master,
            confidence: 0.8,
            transition: TransitionType::Cut,
        };

        let timeline = viz.generate_ascii_timeline(&[shot], &[]);
        assert!(timeline.contains("TIMELINE"));
    }

    #[test]
    fn test_distribution_analyzer() {
        let _analyzer = DistributionAnalyzer::new();
    }

    #[test]
    fn test_duration_histogram_empty() {
        let analyzer = DistributionAnalyzer::new();
        let histogram = analyzer.duration_histogram(&[], 10);
        assert!(histogram.is_empty());
    }
}
