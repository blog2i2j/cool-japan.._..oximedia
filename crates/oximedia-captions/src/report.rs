//! Quality report generation for caption files

use crate::error::Result;
use crate::types::CaptionTrack;
use crate::utils::Statistics;
use crate::validation::{ValidationReport, Validator};
use serde::{Deserialize, Serialize};

/// Comprehensive quality report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    /// File metadata
    pub metadata: ReportMetadata,
    /// Caption statistics
    pub statistics: CaptionStatistics,
    /// Validation report
    pub validation: ValidationReport,
    /// Timing analysis
    pub timing: TimingAnalysis,
    /// Readability analysis
    pub readability: ReadabilityAnalysis,
    /// Compliance status
    pub compliance: ComplianceStatus,
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// Report generation timestamp
    pub generated_at: chrono::DateTime<chrono::Utc>,
    /// File name (if applicable)
    pub file_name: Option<String>,
    /// Language
    pub language: String,
    /// Format
    pub format: Option<String>,
}

/// Caption statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionStatistics {
    /// Total captions
    pub total_captions: usize,
    /// Total words
    pub total_words: usize,
    /// Total characters
    pub total_characters: usize,
    /// Average words per caption
    pub avg_words_per_caption: f64,
    /// Average characters per caption
    pub avg_chars_per_caption: f64,
    /// Average caption duration (seconds)
    pub avg_duration_seconds: f64,
    /// Total duration (seconds)
    pub total_duration_seconds: f64,
    /// Shortest caption (seconds)
    pub shortest_caption: f64,
    /// Longest caption (seconds)
    pub longest_caption: f64,
}

/// Timing analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingAnalysis {
    /// Average reading speed (WPM)
    pub avg_reading_speed_wpm: f64,
    /// Maximum reading speed (WPM)
    pub max_reading_speed_wpm: f64,
    /// Captions exceeding recommended speed
    pub captions_too_fast: usize,
    /// Captions too short
    pub captions_too_short: usize,
    /// Overlapping captions
    pub overlapping_captions: usize,
    /// Average gap between captions (milliseconds)
    pub avg_gap_ms: i64,
    /// Minimum gap (milliseconds)
    pub min_gap_ms: i64,
}

/// Readability analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadabilityAnalysis {
    /// Average characters per line
    pub avg_chars_per_line: f64,
    /// Maximum characters per line
    pub max_chars_per_line: usize,
    /// Average lines per caption
    pub avg_lines_per_caption: f64,
    /// Maximum lines in a caption
    pub max_lines: usize,
    /// Captions exceeding character limit
    pub captions_too_long: usize,
    /// Captions with too many lines
    pub captions_too_many_lines: usize,
}

/// Compliance status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStatus {
    /// FCC compliance
    pub fcc_compliant: bool,
    /// FCC issues
    pub fcc_issues: Vec<String>,
    /// WCAG compliance (AA)
    pub wcag_aa_compliant: bool,
    /// WCAG issues
    pub wcag_issues: Vec<String>,
    /// EBU compliance
    pub ebu_compliant: bool,
    /// EBU issues
    pub ebu_issues: Vec<String>,
}

impl QualityReport {
    /// Generate a quality report for a caption track
    pub fn generate(track: &CaptionTrack) -> Result<Self> {
        let metadata = ReportMetadata {
            generated_at: chrono::Utc::now(),
            file_name: None,
            language: track.language.code.clone(),
            format: None,
        };

        let statistics = Self::calculate_statistics(track);
        let timing = Self::analyze_timing(track);
        let readability = Self::analyze_readability(track);

        // Run validation
        let mut validator = Validator::new();
        validator.add_rule(crate::validation::FccValidator::new());
        validator.add_rule(crate::validation::WcagValidator::new());
        validator.add_rule(crate::validation::OverlapDetector);
        let validation = validator.validate(track)?;

        let compliance = Self::check_compliance(track);

        Ok(Self {
            metadata,
            statistics,
            validation,
            timing,
            readability,
            compliance,
        })
    }

    fn calculate_statistics(track: &CaptionTrack) -> CaptionStatistics {
        let stats = Statistics::calculate(track);

        let total_chars: usize = track.captions.iter().map(|c| c.text.len()).sum();
        let total_words = track.total_words();
        let total_captions = track.count();

        let avg_words = if total_captions > 0 {
            total_words as f64 / total_captions as f64
        } else {
            0.0
        };

        let avg_chars = if total_captions > 0 {
            total_chars as f64 / total_captions as f64
        } else {
            0.0
        };

        let shortest = track
            .captions
            .iter()
            .map(|c| c.duration().as_secs())
            .min()
            .unwrap_or(0) as f64;

        let longest = track
            .captions
            .iter()
            .map(|c| c.duration().as_secs())
            .max()
            .unwrap_or(0) as f64;

        CaptionStatistics {
            total_captions,
            total_words,
            total_characters: total_chars,
            avg_words_per_caption: avg_words,
            avg_chars_per_caption: avg_chars,
            avg_duration_seconds: stats.avg_duration as f64 / 1000.0,
            total_duration_seconds: stats.total_duration.as_secs() as f64,
            shortest_caption: shortest,
            longest_caption: longest,
        }
    }

    fn analyze_timing(track: &CaptionTrack) -> TimingAnalysis {
        let stats = Statistics::calculate(track);

        let mut captions_too_fast = 0;
        let mut captions_too_short = 0;
        let mut overlapping = 0;
        let mut gaps = Vec::new();

        for (i, caption) in track.captions.iter().enumerate() {
            // Check reading speed
            if caption.reading_speed_wpm() > 180.0 {
                captions_too_fast += 1;
            }

            // Check duration
            if caption.duration().as_millis() < 1500 {
                captions_too_short += 1;
            }

            // Check for overlaps and gaps
            if i < track.captions.len() - 1 {
                let next = &track.captions[i + 1];
                if caption.end > next.start {
                    overlapping += 1;
                } else {
                    let gap = next.start.as_millis() - caption.end.as_millis();
                    gaps.push(gap);
                }
            }
        }

        let avg_gap = if gaps.is_empty() {
            0
        } else {
            gaps.iter().sum::<i64>() / gaps.len() as i64
        };

        let min_gap = gaps.iter().min().copied().unwrap_or(0);

        TimingAnalysis {
            avg_reading_speed_wpm: stats.avg_wpm,
            max_reading_speed_wpm: stats.max_wpm,
            captions_too_fast,
            captions_too_short,
            overlapping_captions: overlapping,
            avg_gap_ms: avg_gap,
            min_gap_ms: min_gap,
        }
    }

    fn analyze_readability(track: &CaptionTrack) -> ReadabilityAnalysis {
        let stats = Statistics::calculate(track);

        let mut captions_too_long = 0;
        let mut captions_too_many_lines = 0;

        for caption in &track.captions {
            if caption.max_chars_per_line() > 42 {
                captions_too_long += 1;
            }
            if caption.line_count() > 2 {
                captions_too_many_lines += 1;
            }
        }

        ReadabilityAnalysis {
            avg_chars_per_line: stats.avg_chars_per_line,
            max_chars_per_line: stats.max_chars_per_line,
            avg_lines_per_caption: stats.avg_lines,
            max_lines: stats.max_lines,
            captions_too_long,
            captions_too_many_lines,
        }
    }

    fn check_compliance(track: &CaptionTrack) -> ComplianceStatus {
        let mut fcc_issues = Vec::new();
        let mut wcag_issues = Vec::new();
        let mut ebu_issues = Vec::new();

        for caption in &track.captions {
            // FCC checks
            if caption.max_chars_per_line() > 32 {
                fcc_issues.push(format!(
                    "Caption at {} exceeds 32 chars per line",
                    caption.start
                ));
            }
            if caption.line_count() > 4 {
                fcc_issues.push(format!("Caption at {} exceeds 4 lines", caption.start));
            }
            if caption.reading_speed_wpm() > 180.0 {
                fcc_issues.push(format!(
                    "Caption at {} exceeds 180 WPM reading speed",
                    caption.start
                ));
            }

            // WCAG checks
            if let Some(bg_color) = caption.style.background_color {
                let contrast = caption.style.color.contrast_ratio(&bg_color);
                if contrast < 4.5 {
                    wcag_issues.push(format!(
                        "Caption at {} has insufficient contrast ratio: {:.2}:1",
                        caption.start, contrast
                    ));
                }
            }

            // EBU checks
            let cps = caption.character_count() as f64 / caption.duration().as_secs() as f64;
            if cps > 20.0 {
                ebu_issues.push(format!(
                    "Caption at {} exceeds 20 characters per second",
                    caption.start
                ));
            }
        }

        ComplianceStatus {
            fcc_compliant: fcc_issues.is_empty(),
            fcc_issues,
            wcag_aa_compliant: wcag_issues.is_empty(),
            wcag_issues,
            ebu_compliant: ebu_issues.is_empty(),
            ebu_issues,
        }
    }

    /// Generate a text summary of the report
    #[must_use]
    pub fn to_text_summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str("=== Caption Quality Report ===\n\n");

        summary.push_str(&format!("Generated: {}\n", self.metadata.generated_at));
        summary.push_str(&format!("Language: {}\n\n", self.metadata.language));

        summary.push_str("--- Statistics ---\n");
        summary.push_str(&format!(
            "Total Captions: {}\n",
            self.statistics.total_captions
        ));
        summary.push_str(&format!("Total Words: {}\n", self.statistics.total_words));
        summary.push_str(&format!(
            "Total Duration: {:.1}s\n",
            self.statistics.total_duration_seconds
        ));
        summary.push_str(&format!(
            "Average Duration: {:.2}s\n\n",
            self.statistics.avg_duration_seconds
        ));

        summary.push_str("--- Timing ---\n");
        summary.push_str(&format!(
            "Average Reading Speed: {:.1} WPM\n",
            self.timing.avg_reading_speed_wpm
        ));
        summary.push_str(&format!(
            "Maximum Reading Speed: {:.1} WPM\n",
            self.timing.max_reading_speed_wpm
        ));
        summary.push_str(&format!(
            "Captions Too Fast: {}\n",
            self.timing.captions_too_fast
        ));
        summary.push_str(&format!(
            "Captions Too Short: {}\n",
            self.timing.captions_too_short
        ));
        summary.push_str(&format!(
            "Overlapping Captions: {}\n\n",
            self.timing.overlapping_captions
        ));

        summary.push_str("--- Readability ---\n");
        summary.push_str(&format!(
            "Average Chars Per Line: {:.1}\n",
            self.readability.avg_chars_per_line
        ));
        summary.push_str(&format!(
            "Maximum Chars Per Line: {}\n",
            self.readability.max_chars_per_line
        ));
        summary.push_str(&format!(
            "Captions Too Long: {}\n\n",
            self.readability.captions_too_long
        ));

        summary.push_str("--- Compliance ---\n");
        summary.push_str(&format!(
            "FCC Compliant: {}\n",
            if self.compliance.fcc_compliant {
                "YES"
            } else {
                "NO"
            }
        ));
        summary.push_str(&format!(
            "WCAG AA Compliant: {}\n",
            if self.compliance.wcag_aa_compliant {
                "YES"
            } else {
                "NO"
            }
        ));
        summary.push_str(&format!(
            "EBU Compliant: {}\n\n",
            if self.compliance.ebu_compliant {
                "YES"
            } else {
                "NO"
            }
        ));

        summary.push_str("--- Validation ---\n");
        summary.push_str(&format!(
            "Errors: {}\n",
            self.validation.statistics.error_count
        ));
        summary.push_str(&format!(
            "Warnings: {}\n",
            self.validation.statistics.warning_count
        ));

        summary
    }

    /// Export report to JSON
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Export report to CSV
    #[must_use]
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();

        csv.push_str("Metric,Value\n");
        csv.push_str(&format!(
            "Total Captions,{}\n",
            self.statistics.total_captions
        ));
        csv.push_str(&format!("Total Words,{}\n", self.statistics.total_words));
        csv.push_str(&format!(
            "Avg Reading Speed (WPM),{:.1}\n",
            self.timing.avg_reading_speed_wpm
        ));
        csv.push_str(&format!(
            "Max Reading Speed (WPM),{:.1}\n",
            self.timing.max_reading_speed_wpm
        ));
        csv.push_str(&format!(
            "Captions Too Fast,{}\n",
            self.timing.captions_too_fast
        ));
        csv.push_str(&format!(
            "Overlapping Captions,{}\n",
            self.timing.overlapping_captions
        ));
        csv.push_str(&format!(
            "FCC Compliant,{}\n",
            if self.compliance.fcc_compliant {
                "YES"
            } else {
                "NO"
            }
        ));
        csv.push_str(&format!(
            "WCAG AA Compliant,{}\n",
            if self.compliance.wcag_aa_compliant {
                "YES"
            } else {
                "NO"
            }
        ));
        csv.push_str(&format!(
            "Validation Errors,{}\n",
            self.validation.statistics.error_count
        ));
        csv.push_str(&format!(
            "Validation Warnings,{}\n",
            self.validation.statistics.warning_count
        ));

        csv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Caption, Language, Timestamp};

    #[test]
    fn test_report_generation() {
        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(0),
                Timestamp::from_secs(5),
                "Test caption one".to_string(),
            ))
            .expect("operation should succeed in test");
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(10),
                Timestamp::from_secs(15),
                "Test caption two".to_string(),
            ))
            .expect("operation should succeed in test");

        let report = QualityReport::generate(&track).expect("generation should succeed");

        assert_eq!(report.statistics.total_captions, 2);
        assert_eq!(report.statistics.total_words, 6);
    }

    #[test]
    fn test_report_text_summary() {
        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(0),
                Timestamp::from_secs(5),
                "Test".to_string(),
            ))
            .expect("operation should succeed in test");

        let report = QualityReport::generate(&track).expect("generation should succeed");
        let summary = report.to_text_summary();

        assert!(summary.contains("Caption Quality Report"));
        assert!(summary.contains("Total Captions: 1"));
    }

    #[test]
    fn test_report_json_export() {
        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(0),
                Timestamp::from_secs(5),
                "Test".to_string(),
            ))
            .expect("operation should succeed in test");

        let report = QualityReport::generate(&track).expect("generation should succeed");
        let json = report.to_json().expect("JSON serialization should succeed");

        assert!(!json.is_empty());
        assert!(json.contains("statistics"));
    }
}
