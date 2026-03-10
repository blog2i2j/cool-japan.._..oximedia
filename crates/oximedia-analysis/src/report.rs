//! Analysis report generation.
//!
//! This module generates human-readable and machine-readable reports from
//! analysis results:
//! - **JSON Reports** - Structured data for programmatic use
//! - **HTML Reports** - Visual reports with charts and statistics
//! - **Timeline Visualization** - Scene markers and quality graphs
//!
//! # Example
//!
//! ```rust,no_run
//! use oximedia_analysis::{Analyzer, AnalysisConfig};
//!
//! let config = AnalysisConfig::default();
//! let analyzer = Analyzer::new(config);
//! // ... process frames ...
//! let results = analyzer.finalize();
//!
//! // Generate reports
//! let json_report = results.to_json()?;
//! let html_report = results.to_html()?;
//! ```

use crate::{AnalysisResult, AnalysisResults};

/// Generate HTML report from analysis results.
pub fn generate_html_report(results: &AnalysisResults) -> AnalysisResult<String> {
    let mut html = String::new();

    // HTML header
    html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
    html.push_str("    <meta charset=\"UTF-8\">\n");
    html.push_str(
        "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n",
    );
    html.push_str("    <title>OxiMedia Analysis Report</title>\n");
    html.push_str("    <style>\n");
    html.push_str(include_str!("report_style.css"));
    html.push_str("    </style>\n");
    html.push_str("</head>\n<body>\n");

    // Title and summary
    html.push_str("    <div class=\"container\">\n");
    html.push_str("        <h1>OxiMedia Analysis Report</h1>\n");

    // Overview section
    html.push_str("        <div class=\"section\">\n");
    html.push_str("            <h2>Overview</h2>\n");
    html.push_str("            <div class=\"stats\">\n");
    html.push_str(&format!(
        "                <div class=\"stat-item\"><strong>Total Frames:</strong> {}</div>\n",
        results.frame_count
    ));
    html.push_str(&format!(
        "                <div class=\"stat-item\"><strong>Frame Rate:</strong> {}/{} fps</div>\n",
        results.frame_rate.num, results.frame_rate.den
    ));
    let duration_secs =
        results.frame_count as f64 * results.frame_rate.den as f64 / results.frame_rate.num as f64;
    html.push_str(&format!(
        "                <div class=\"stat-item\"><strong>Duration:</strong> {duration_secs:.2}s</div>\n"
    ));
    html.push_str("            </div>\n");
    html.push_str("        </div>\n");

    // Scene detection
    if !results.scenes.is_empty() {
        html.push_str("        <div class=\"section\">\n");
        html.push_str("            <h2>Scene Detection</h2>\n");
        html.push_str(&format!(
            "            <p>Detected <strong>{}</strong> scenes</p>\n",
            results.scenes.len()
        ));
        html.push_str("            <table>\n");
        html.push_str("                <tr><th>Start Frame</th><th>End Frame</th><th>Duration</th><th>Type</th><th>Confidence</th></tr>\n");
        for scene in &results.scenes {
            html.push_str("                <tr>\n");
            html.push_str(&format!(
                "                    <td>{}</td>\n",
                scene.start_frame
            ));
            html.push_str(&format!(
                "                    <td>{}</td>\n",
                scene.end_frame
            ));
            html.push_str(&format!(
                "                    <td>{}</td>\n",
                scene.end_frame - scene.start_frame
            ));
            html.push_str(&format!(
                "                    <td>{:?}</td>\n",
                scene.change_type
            ));
            html.push_str(&format!(
                "                    <td>{:.2}</td>\n",
                scene.confidence
            ));
            html.push_str("                </tr>\n");
        }
        html.push_str("            </table>\n");
        html.push_str("        </div>\n");
    }

    // Black frames
    if !results.black_frames.is_empty() {
        html.push_str("        <div class=\"section\">\n");
        html.push_str("            <h2>Black Frames</h2>\n");
        html.push_str(&format!(
            "            <p>Detected <strong>{}</strong> black segments</p>\n",
            results.black_frames.len()
        ));
        html.push_str("            <table>\n");
        html.push_str("                <tr><th>Start Frame</th><th>End Frame</th><th>Duration</th><th>Avg Luminance</th><th>Black Ratio</th></tr>\n");
        for segment in &results.black_frames {
            html.push_str("                <tr>\n");
            html.push_str(&format!(
                "                    <td>{}</td>\n",
                segment.start_frame
            ));
            html.push_str(&format!(
                "                    <td>{}</td>\n",
                segment.end_frame
            ));
            html.push_str(&format!(
                "                    <td>{}</td>\n",
                segment.end_frame - segment.start_frame
            ));
            html.push_str(&format!(
                "                    <td>{:.1}</td>\n",
                segment.avg_luminance
            ));
            html.push_str(&format!(
                "                    <td>{:.1}%</td>\n",
                segment.black_pixel_ratio * 100.0
            ));
            html.push_str("                </tr>\n");
        }
        html.push_str("            </table>\n");
        html.push_str("        </div>\n");
    }

    // Quality assessment
    html.push_str("        <div class=\"section\">\n");
    html.push_str("            <h2>Quality Assessment</h2>\n");
    html.push_str("            <div class=\"stats\">\n");
    html.push_str(&format!(
        "                <div class=\"stat-item\"><strong>Average Quality Score:</strong> {:.2}/1.0</div>\n",
        results.quality_stats.average_score
    ));
    html.push_str(&format!(
        "                <div class=\"stat-item\"><strong>Blockiness:</strong> {:.3}</div>\n",
        results.quality_stats.avg_blockiness
    ));
    html.push_str(&format!(
        "                <div class=\"stat-item\"><strong>Blur:</strong> {:.3}</div>\n",
        results.quality_stats.avg_blur
    ));
    html.push_str(&format!(
        "                <div class=\"stat-item\"><strong>Noise:</strong> {:.3}</div>\n",
        results.quality_stats.avg_noise
    ));
    html.push_str("            </div>\n");
    html.push_str("        </div>\n");

    // Content classification
    if let Some(ref classification) = results.content_classification {
        html.push_str("        <div class=\"section\">\n");
        html.push_str("            <h2>Content Classification</h2>\n");
        html.push_str("            <div class=\"stats\">\n");
        html.push_str(&format!(
            "                <div class=\"stat-item\"><strong>Primary Type:</strong> {:?}</div>\n",
            classification.primary_type
        ));
        html.push_str(&format!(
            "                <div class=\"stat-item\"><strong>Confidence:</strong> {:.2}</div>\n",
            classification.confidence
        ));
        html.push_str(&format!(
            "                <div class=\"stat-item\"><strong>Temporal Activity:</strong> {:.2}</div>\n",
            classification.stats.avg_temporal_activity
        ));
        html.push_str(&format!(
            "                <div class=\"stat-item\"><strong>Spatial Complexity:</strong> {:.2}</div>\n",
            classification.stats.avg_spatial_complexity
        ));
        html.push_str("            </div>\n");
        html.push_str("        </div>\n");
    }

    // Motion analysis
    if let Some(ref motion) = results.motion_stats {
        html.push_str("        <div class=\"section\">\n");
        html.push_str("            <h2>Motion Analysis</h2>\n");
        html.push_str("            <div class=\"stats\">\n");
        html.push_str(&format!(
            "                <div class=\"stat-item\"><strong>Camera Motion:</strong> {:?}</div>\n",
            motion.camera_motion
        ));
        html.push_str(&format!(
            "                <div class=\"stat-item\"><strong>Average Motion:</strong> {:.2}</div>\n",
            motion.avg_motion
        ));
        html.push_str(&format!(
            "                <div class=\"stat-item\"><strong>Max Motion:</strong> {:.2}</div>\n",
            motion.max_motion
        ));
        html.push_str(&format!(
            "                <div class=\"stat-item\"><strong>Stability:</strong> {:.2}</div>\n",
            motion.stability
        ));
        html.push_str("            </div>\n");
        html.push_str("        </div>\n");
    }

    // Color analysis
    if let Some(ref color) = results.color_analysis {
        html.push_str("        <div class=\"section\">\n");
        html.push_str("            <h2>Color Analysis</h2>\n");
        html.push_str("            <div class=\"stats\">\n");
        html.push_str(&format!(
            "                <div class=\"stat-item\"><strong>Grading Style:</strong> {:?}</div>\n",
            color.grading_style
        ));
        html.push_str(&format!(
            "                <div class=\"stat-item\"><strong>Average Saturation:</strong> {:.2}</div>\n",
            color.avg_saturation
        ));
        html.push_str(&format!(
            "                <div class=\"stat-item\"><strong>Color Diversity:</strong> {:.2}</div>\n",
            color.color_diversity
        ));
        html.push_str("            </div>\n");

        if !color.dominant_colors.is_empty() {
            html.push_str("            <h3>Dominant Colors</h3>\n");
            html.push_str("            <div class=\"color-palette\">\n");
            for dc in &color.dominant_colors {
                let (r, g, b) = dc.rgb;
                html.push_str(&format!(
                    "                <div class=\"color-swatch\" style=\"background-color: rgb({}, {}, {});\" title=\"RGB({}, {}, {}) - {:.1}%\"></div>\n",
                    r, g, b, r, g, b, dc.percentage * 100.0
                ));
            }
            html.push_str("            </div>\n");
        }
        html.push_str("        </div>\n");
    }

    // Audio analysis
    if let Some(ref audio) = results.audio_analysis {
        html.push_str("        <div class=\"section\">\n");
        html.push_str("            <h2>Audio Analysis</h2>\n");
        html.push_str("            <div class=\"stats\">\n");
        html.push_str(&format!(
            "                <div class=\"stat-item\"><strong>Peak Level:</strong> {:.1} dBFS</div>\n",
            audio.peak_dbfs
        ));
        html.push_str(&format!(
            "                <div class=\"stat-item\"><strong>RMS Level:</strong> {:.1} dBFS</div>\n",
            audio.rms_dbfs
        ));
        html.push_str(&format!(
            "                <div class=\"stat-item\"><strong>Dynamic Range:</strong> {:.1} dB</div>\n",
            audio.dynamic_range_db
        ));
        if let Some(phase) = audio.phase_correlation {
            html.push_str(&format!(
                "                <div class=\"stat-item\"><strong>Phase Correlation:</strong> {phase:.2}</div>\n"
            ));
        }
        html.push_str("            </div>\n");

        if !audio.clipping_events.is_empty() {
            html.push_str(&format!(
                "            <p class=\"warning\">⚠ Detected {} clipping events</p>\n",
                audio.clipping_events.len()
            ));
        }

        if !audio.silence_segments.is_empty() {
            html.push_str(&format!(
                "            <p>Detected {} silence segments</p>\n",
                audio.silence_segments.len()
            ));
        }
        html.push_str("        </div>\n");
    }

    // Temporal analysis
    if let Some(ref temporal) = results.temporal_analysis {
        html.push_str("        <div class=\"section\">\n");
        html.push_str("            <h2>Temporal Analysis</h2>\n");
        html.push_str("            <div class=\"stats\">\n");
        html.push_str(&format!(
            "                <div class=\"stat-item\"><strong>Consistency:</strong> {:.2}</div>\n",
            temporal.consistency
        ));
        html.push_str(&format!(
            "                <div class=\"stat-item\"><strong>Temporal Noise:</strong> {:.2}</div>\n",
            temporal.temporal_noise
        ));
        html.push_str(&format!(
            "                <div class=\"stat-item\"><strong>Flicker Events:</strong> {}</div>\n",
            temporal.flicker_events.len()
        ));
        if let Some(telecine) = temporal.telecine {
            html.push_str(&format!(
                "                <div class=\"stat-item\"><strong>Telecine:</strong> {telecine:?}</div>\n"
            ));
        }
        html.push_str("            </div>\n");
        html.push_str("        </div>\n");
    }

    // Thumbnails
    if !results.thumbnails.is_empty() {
        html.push_str("        <div class=\"section\">\n");
        html.push_str("            <h2>Selected Thumbnails</h2>\n");
        html.push_str(&format!(
            "            <p>Selected {} representative frames</p>\n",
            results.thumbnails.len()
        ));
        html.push_str("            <table>\n");
        html.push_str("                <tr><th>Frame</th><th>Score</th><th>Luminance</th><th>Contrast</th><th>Sharpness</th></tr>\n");
        for thumb in &results.thumbnails {
            html.push_str("                <tr>\n");
            html.push_str(&format!("                    <td>{}</td>\n", thumb.frame));
            html.push_str(&format!(
                "                    <td>{:.2}</td>\n",
                thumb.score
            ));
            html.push_str(&format!(
                "                    <td>{:.1}</td>\n",
                thumb.avg_luminance
            ));
            html.push_str(&format!(
                "                    <td>{:.1}</td>\n",
                thumb.contrast
            ));
            html.push_str(&format!(
                "                    <td>{:.1}</td>\n",
                thumb.sharpness
            ));
            html.push_str("                </tr>\n");
        }
        html.push_str("            </table>\n");
        html.push_str("        </div>\n");
    }

    // Footer
    html.push_str("        <div class=\"footer\">\n");
    html.push_str("            <p>Generated by OxiMedia Analysis</p>\n");
    html.push_str("        </div>\n");
    html.push_str("    </div>\n");
    html.push_str("</body>\n</html>");

    Ok(html)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{quality::QualityStats, scene::Scene, scene::SceneChangeType};
    use oximedia_core::types::Rational;

    #[test]
    fn test_html_report_generation() {
        let results = AnalysisResults {
            frame_count: 1000,
            frame_rate: Rational::new(25, 1),
            scenes: vec![Scene {
                start_frame: 0,
                end_frame: 100,
                confidence: 0.95,
                change_type: SceneChangeType::Cut,
            }],
            black_frames: Vec::new(),
            quality_stats: QualityStats::default(),
            content_classification: None,
            thumbnails: Vec::new(),
            motion_stats: None,
            color_analysis: None,
            audio_analysis: None,
            temporal_analysis: None,
        };

        let html = generate_html_report(&results).expect("HTML report generation should succeed");
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("OxiMedia Analysis Report"));
        assert!(html.contains("1000")); // Frame count
        assert!(html.contains("Scene Detection"));
    }

    #[test]
    fn test_empty_report() {
        let results = AnalysisResults {
            frame_count: 0,
            frame_rate: Rational::new(25, 1),
            scenes: Vec::new(),
            black_frames: Vec::new(),
            quality_stats: QualityStats::default(),
            content_classification: None,
            thumbnails: Vec::new(),
            motion_stats: None,
            color_analysis: None,
            audio_analysis: None,
            temporal_analysis: None,
        };

        let html = generate_html_report(&results).expect("HTML report generation should succeed");
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("0")); // Frame count
    }
}
