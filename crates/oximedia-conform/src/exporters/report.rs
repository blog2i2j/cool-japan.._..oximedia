//! Report generation for conform sessions.

use crate::error::ConformResult;
use crate::types::{ClipMatch, ClipReference};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Match report containing all conform results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchReport {
    /// Successfully matched clips.
    pub matched: Vec<ClipMatch>,
    /// Clips with no matches found.
    pub missing: Vec<ClipReference>,
    /// Clips with ambiguous matches.
    pub ambiguous: Vec<AmbiguousMatch>,
    /// Statistics.
    pub stats: MatchStatistics,
}

/// Ambiguous match entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbiguousMatch {
    /// The clip with ambiguous matches.
    pub clip: ClipReference,
    /// All candidate matches.
    pub candidates: Vec<ClipMatch>,
}

/// Match statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchStatistics {
    /// Total clips to match.
    pub total_clips: usize,
    /// Successfully matched clips.
    pub matched_count: usize,
    /// Missing clips.
    pub missing_count: usize,
    /// Ambiguous clips.
    pub ambiguous_count: usize,
    /// Conform rate (0.0 - 1.0).
    pub conform_rate: f64,
}

impl MatchReport {
    /// Create a new match report.
    #[must_use]
    pub fn new(
        matched: Vec<ClipMatch>,
        missing: Vec<ClipReference>,
        ambiguous: Vec<AmbiguousMatch>,
    ) -> Self {
        let total_clips = matched.len() + missing.len() + ambiguous.len();
        let conform_rate = if total_clips == 0 {
            0.0
        } else {
            matched.len() as f64 / total_clips as f64
        };

        Self {
            stats: MatchStatistics {
                total_clips,
                matched_count: matched.len(),
                missing_count: missing.len(),
                ambiguous_count: ambiguous.len(),
                conform_rate,
            },
            matched,
            missing,
            ambiguous,
        }
    }

    /// Export report as JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> ConformResult<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Export report as HTML.
    #[must_use]
    pub fn to_html(&self) -> String {
        let mut html = String::new();
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>Conform Report</title>\n");
        html.push_str("<style>\n");
        html.push_str("body { font-family: Arial, sans-serif; margin: 20px; }\n");
        html.push_str("table { border-collapse: collapse; width: 100%; }\n");
        html.push_str("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
        html.push_str("th { background-color: #4CAF50; color: white; }\n");
        html.push_str(".matched { color: green; }\n");
        html.push_str(".missing { color: red; }\n");
        html.push_str(".ambiguous { color: orange; }\n");
        html.push_str("</style>\n</head>\n<body>\n");

        html.push_str("<h1>Conform Report</h1>\n");

        // Statistics
        html.push_str("<h2>Statistics</h2>\n");
        html.push_str("<ul>\n");
        html.push_str(&format!(
            "<li>Total Clips: {}</li>\n",
            self.stats.total_clips
        ));
        html.push_str(&format!(
            "<li class='matched'>Matched: {}</li>\n",
            self.stats.matched_count
        ));
        html.push_str(&format!(
            "<li class='missing'>Missing: {}</li>\n",
            self.stats.missing_count
        ));
        html.push_str(&format!(
            "<li class='ambiguous'>Ambiguous: {}</li>\n",
            self.stats.ambiguous_count
        ));
        html.push_str(&format!(
            "<li>Conform Rate: {:.1}%</li>\n",
            self.stats.conform_rate * 100.0
        ));
        html.push_str("</ul>\n");

        // Matched clips
        if !self.matched.is_empty() {
            html.push_str("<h2>Matched Clips</h2>\n");
            html.push_str("<table>\n");
            html.push_str("<tr><th>Clip ID</th><th>Source File</th><th>Match Score</th><th>Method</th></tr>\n");
            for m in &self.matched {
                html.push_str(&format!(
                    "<tr><td>{}</td><td>{}</td><td>{:.2}</td><td>{}</td></tr>\n",
                    m.clip.id,
                    m.clip.source_file.as_ref().unwrap_or(&"N/A".to_string()),
                    m.score,
                    m.method
                ));
            }
            html.push_str("</table>\n");
        }

        // Missing clips
        if !self.missing.is_empty() {
            html.push_str("<h2 class='missing'>Missing Clips</h2>\n");
            html.push_str("<table>\n");
            html.push_str("<tr><th>Clip ID</th><th>Source File</th></tr>\n");
            for clip in &self.missing {
                html.push_str(&format!(
                    "<tr><td>{}</td><td>{}</td></tr>\n",
                    clip.id,
                    clip.source_file.as_ref().unwrap_or(&"N/A".to_string())
                ));
            }
            html.push_str("</table>\n");
        }

        // Ambiguous clips
        if !self.ambiguous.is_empty() {
            html.push_str("<h2 class='ambiguous'>Ambiguous Matches</h2>\n");
            for amb in &self.ambiguous {
                html.push_str(&format!("<h3>Clip: {}</h3>\n", amb.clip.id));
                html.push_str("<table>\n");
                html.push_str("<tr><th>Candidate</th><th>Score</th><th>Method</th></tr>\n");
                for candidate in &amb.candidates {
                    html.push_str(&format!(
                        "<tr><td>{}</td><td>{:.2}</td><td>{}</td></tr>\n",
                        candidate.media.filename, candidate.score, candidate.method
                    ));
                }
                html.push_str("</table>\n");
            }
        }

        html.push_str("</body>\n</html>\n");
        html
    }

    /// Save report to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> ConformResult<()> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Save HTML report to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save_html<P: AsRef<Path>>(&self, path: P) -> ConformResult<()> {
        let html = self.to_html();
        std::fs::write(path, html)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_report() {
        let report = MatchReport::new(Vec::new(), Vec::new(), Vec::new());
        assert_eq!(report.stats.total_clips, 0);
        assert!((report.stats.conform_rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_report_statistics() {
        let report = MatchReport::new(vec![], vec![], vec![]);
        let json = report.to_json().expect("json should be valid");
        assert!(json.contains("total_clips"));
    }

    #[test]
    fn test_html_generation() {
        let report = MatchReport::new(Vec::new(), Vec::new(), Vec::new());
        let html = report.to_html();
        assert!(html.contains("<html>"));
        assert!(html.contains("Conform Report"));
    }
}
