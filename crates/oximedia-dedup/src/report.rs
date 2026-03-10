//! Duplicate detection reports and recommendations.
//!
//! This module provides:
//! - Duplicate group reporting
//! - Similarity scoring and ranking
//! - Storage savings estimation
//! - HTML and JSON export
//! - Deduplication recommendations

use crate::{DedupError, DedupResult};
use serde::{Deserialize, Serialize};

/// Duplicate detection report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateReport {
    /// Groups of duplicate files
    pub groups: Vec<DuplicateGroup>,

    /// Total number of duplicates
    pub total_duplicates: usize,

    /// Total wasted space in bytes
    pub wasted_space: u64,

    /// Report generation timestamp
    pub timestamp: i64,
}

impl DuplicateReport {
    /// Create a new empty report.
    #[must_use]
    pub fn new() -> Self {
        Self {
            groups: Vec::new(),
            total_duplicates: 0,
            wasted_space: 0,
            timestamp: current_timestamp(),
        }
    }

    /// Add a duplicate group.
    pub fn add_group(&mut self, group: DuplicateGroup) {
        if group.files.len() > 1 {
            self.total_duplicates += group.files.len() - 1;
            self.wasted_space += group.estimated_savings();
            self.groups.push(group);
        }
    }

    /// Add multiple groups.
    pub fn add_groups(&mut self, groups: Vec<DuplicateGroup>) {
        for group in groups {
            self.add_group(group);
        }
    }

    /// Sort groups by wasted space (descending).
    pub fn sort_by_savings(&mut self) {
        self.groups
            .sort_by(|a, b| b.estimated_savings().cmp(&a.estimated_savings()));
    }

    /// Sort groups by similarity score (descending).
    pub fn sort_by_similarity(&mut self) {
        self.groups.sort_by(|a, b| {
            b.max_similarity()
                .partial_cmp(&a.max_similarity())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Filter groups by minimum similarity.
    pub fn filter_by_similarity(&mut self, threshold: f64) {
        self.groups.retain(|g| g.max_similarity() >= threshold);
        self.recalculate_stats();
    }

    /// Recalculate statistics.
    fn recalculate_stats(&mut self) {
        self.total_duplicates = self.groups.iter().map(|g| g.files.len() - 1).sum();
        self.wasted_space = self.groups.iter().map(|g| g.estimated_savings()).sum();
    }

    /// Export to JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> DedupResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| DedupError::Hash(format!("JSON serialization failed: {e}")))
    }

    /// Export to JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn to_json_file(&self, path: impl AsRef<std::path::Path>) -> DedupResult<()> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Export to HTML report.
    #[must_use]
    pub fn to_html(&self) -> String {
        let mut html = String::from(
            r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OxiMedia Duplicate Detection Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
        }
        .summary {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .group {
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .file {
            padding: 8px;
            margin: 5px 0;
            background-color: #f9f9f9;
            border-left: 3px solid #4CAF50;
        }
        .score {
            display: inline-block;
            padding: 4px 8px;
            background-color: #2196F3;
            color: white;
            border-radius: 4px;
            font-size: 12px;
            margin-left: 10px;
        }
        .savings {
            color: #4CAF50;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>OxiMedia Duplicate Detection Report</h1>
"#,
        );

        // Summary section
        html.push_str(&format!(
            r#"
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Duplicate Groups:</strong> {}</p>
        <p><strong>Total Duplicate Files:</strong> {}</p>
        <p class="savings"><strong>Potential Storage Savings:</strong> {}</p>
        <p><strong>Generated:</strong> {}</p>
    </div>
"#,
            self.groups.len(),
            self.total_duplicates,
            format_bytes(self.wasted_space),
            format_timestamp(self.timestamp)
        ));

        // Duplicate groups
        html.push_str("    <h2>Duplicate Groups</h2>\n");

        for (i, group) in self.groups.iter().enumerate() {
            html.push_str(&format!(
                r#"
    <div class="group">
        <h3>Group {} <span class="score">Similarity: {:.1}%</span> <span class="savings">Savings: {}</span></h3>
"#,
                i + 1,
                group.max_similarity() * 100.0,
                format_bytes(group.estimated_savings())
            ));

            for file in &group.files {
                html.push_str(&format!(
                    r#"        <div class="file">{}</div>
"#,
                    html_escape(file)
                ));
            }

            if !group.scores.is_empty() {
                html.push_str("        <p><strong>Similarity Details:</strong></p>\n");
                html.push_str("        <ul>\n");
                for score in &group.scores {
                    html.push_str(&format!(
                        "            <li>{}: {:.1}%</li>\n",
                        score.method,
                        score.score * 100.0
                    ));
                }
                html.push_str("        </ul>\n");
            }

            html.push_str("    </div>\n");
        }

        html.push_str(
            r#"
</body>
</html>
"#,
        );

        html
    }

    /// Export to HTML file.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn to_html_file(&self, path: impl AsRef<std::path::Path>) -> DedupResult<()> {
        let html = self.to_html();
        std::fs::write(path, html)?;
        Ok(())
    }

    /// Get total number of groups.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Get recommendations for deduplication.
    #[must_use]
    pub fn get_recommendations(&self) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        for group in &self.groups {
            if let Some(rec) = group.recommend_action() {
                recommendations.push(rec);
            }
        }

        // Sort by priority (savings)
        recommendations.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        recommendations
    }
}

impl Default for DuplicateReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Group of duplicate files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroup {
    /// File paths in this group
    pub files: Vec<String>,

    /// Similarity scores
    pub scores: Vec<SimilarityScore>,
}

impl DuplicateGroup {
    /// Create a new duplicate group.
    #[must_use]
    pub fn new(files: Vec<String>) -> Self {
        Self {
            files,
            scores: Vec::new(),
        }
    }

    /// Add a similarity score.
    pub fn add_score(&mut self, score: SimilarityScore) {
        self.scores.push(score);
    }

    /// Get maximum similarity score.
    #[must_use]
    pub fn max_similarity(&self) -> f64 {
        self.scores.iter().map(|s| s.score).fold(0.0f64, f64::max)
    }

    /// Get average similarity score.
    #[must_use]
    pub fn avg_similarity(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.scores.iter().map(|s| s.score).sum();
        sum / self.scores.len() as f64
    }

    /// Estimate storage savings if duplicates are removed.
    #[must_use]
    pub fn estimated_savings(&self) -> u64 {
        if self.files.len() <= 1 {
            return 0;
        }

        // Calculate total size of all files
        let mut total_size = 0u64;
        for file in &self.files {
            if let Ok(metadata) = std::fs::metadata(file) {
                total_size += metadata.len();
            }
        }

        // Savings = total - largest file
        let mut largest = 0u64;
        for file in &self.files {
            if let Ok(metadata) = std::fs::metadata(file) {
                largest = largest.max(metadata.len());
            }
        }

        total_size.saturating_sub(largest)
    }

    /// Recommend which files to keep/delete.
    #[must_use]
    pub fn recommend_action(&self) -> Option<Recommendation> {
        if self.files.len() <= 1 {
            return None;
        }

        // Recommend keeping the file with the best quality indicators:
        // 1. Shortest path (likely original location)
        // 2. Largest file size (likely highest quality)
        // 3. Most recent modification time

        let mut best_file = None;
        let mut best_score = 0.0f64;

        for file in &self.files {
            let mut score = 0.0;

            // Shorter path is better
            let path_score = 1.0 / (file.len() as f64 + 1.0);
            score += path_score * 0.3;

            // Larger size is better
            if let Ok(metadata) = std::fs::metadata(file) {
                score += (metadata.len() as f64 / 1_000_000.0).min(1.0) * 0.4;

                // More recent modification is better
                if let Ok(modified) = metadata.modified() {
                    if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                        let age_days = (current_timestamp() - duration.as_secs() as i64) / 86400;
                        score += (1.0 / (age_days as f64 + 1.0)) * 0.3;
                    }
                }
            }

            if score > best_score {
                best_score = score;
                best_file = Some(file.clone());
            }
        }

        let keep_file = best_file?;
        let delete_files: Vec<String> = self
            .files
            .iter()
            .filter(|f| *f != &keep_file)
            .cloned()
            .collect();

        Some(Recommendation {
            action: RecommendationAction::DeleteDuplicates,
            keep_file,
            delete_files,
            reason: format!(
                "Keep the best quality file, remove {} duplicate(s)",
                self.files.len() - 1
            ),
            priority: self.estimated_savings() as f64,
        })
    }
}

/// Similarity score with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityScore {
    /// Detection method name
    pub method: String,

    /// Similarity score (0.0-1.0)
    pub score: f64,

    /// Additional metadata
    pub metadata: Vec<(String, String)>,
}

impl SimilarityScore {
    /// Create a new similarity score.
    #[must_use]
    pub fn new(method: String, score: f64) -> Self {
        Self {
            method,
            score,
            metadata: Vec::new(),
        }
    }

    /// Add metadata.
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.push((key, value));
    }
}

/// Deduplication recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Recommended action
    pub action: RecommendationAction,

    /// File to keep
    pub keep_file: String,

    /// Files to delete
    pub delete_files: Vec<String>,

    /// Reason for recommendation
    pub reason: String,

    /// Priority score (higher = more important)
    pub priority: f64,
}

/// Recommended action type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationAction {
    /// Delete duplicate files
    DeleteDuplicates,

    /// Create symbolic links
    CreateSymlinks,

    /// Move to archive
    Archive,

    /// Manual review needed
    ManualReview,
}

/// Get current Unix timestamp.
fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Format timestamp as readable string.
fn format_timestamp(timestamp: i64) -> String {
    // Simple formatting - in production, use chrono or similar
    let datetime = std::time::UNIX_EPOCH + std::time::Duration::from_secs(timestamp as u64);
    format!("{:?}", datetime)
}

/// Format bytes as human-readable string.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    const TB: u64 = 1024 * GB;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Escape HTML special characters.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_creation() {
        let report = DuplicateReport::new();
        assert_eq!(report.groups.len(), 0);
        assert_eq!(report.total_duplicates, 0);
    }

    #[test]
    fn test_add_group() {
        let mut report = DuplicateReport::new();

        let group = DuplicateGroup::new(vec!["file1.mp4".to_string(), "file2.mp4".to_string()]);

        report.add_group(group);

        assert_eq!(report.groups.len(), 1);
        assert_eq!(report.total_duplicates, 1);
    }

    #[test]
    fn test_duplicate_group() {
        let mut group = DuplicateGroup::new(vec![
            "file1.mp4".to_string(),
            "file2.mp4".to_string(),
            "file3.mp4".to_string(),
        ]);

        assert_eq!(group.files.len(), 3);

        group.add_score(SimilarityScore::new("hash".to_string(), 1.0));
        group.add_score(SimilarityScore::new("phash".to_string(), 0.95));

        assert_eq!(group.max_similarity(), 1.0);
        assert!((group.avg_similarity() - 0.975).abs() < 0.001);
    }

    #[test]
    fn test_similarity_score() {
        let mut score = SimilarityScore::new("test".to_string(), 0.95);
        assert_eq!(score.method, "test");
        assert_eq!(score.score, 0.95);

        score.add_metadata("key".to_string(), "value".to_string());
        assert_eq!(score.metadata.len(), 1);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 bytes");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_bytes(1024u64 * 1024 * 1024 * 1024), "1.00 TB");
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("test"), "test");
        assert_eq!(html_escape("<script>"), "&lt;script&gt;");
        assert_eq!(html_escape("a & b"), "a &amp; b");
        assert_eq!(html_escape("\"quoted\""), "&quot;quoted&quot;");
    }

    #[test]
    fn test_json_export() {
        let mut report = DuplicateReport::new();
        let group = DuplicateGroup::new(vec!["file1.mp4".to_string(), "file2.mp4".to_string()]);
        report.add_group(group);

        let json = report.to_json().expect("operation should succeed");
        assert!(json.contains("file1.mp4"));
        assert!(json.contains("file2.mp4"));
    }

    #[test]
    fn test_html_export() {
        let mut report = DuplicateReport::new();
        let group = DuplicateGroup::new(vec!["file1.mp4".to_string(), "file2.mp4".to_string()]);
        report.add_group(group);

        let html = report.to_html();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("file1.mp4"));
        assert!(html.contains("file2.mp4"));
    }

    #[test]
    fn test_sort_by_similarity() {
        let mut report = DuplicateReport::new();

        let mut group1 = DuplicateGroup::new(vec!["a".to_string(), "b".to_string()]);
        group1.add_score(SimilarityScore::new("test".to_string(), 0.9));

        let mut group2 = DuplicateGroup::new(vec!["c".to_string(), "d".to_string()]);
        group2.add_score(SimilarityScore::new("test".to_string(), 0.95));

        report.add_group(group1);
        report.add_group(group2);

        report.sort_by_similarity();

        assert_eq!(report.groups[0].max_similarity(), 0.95);
        assert_eq!(report.groups[1].max_similarity(), 0.9);
    }

    #[test]
    fn test_filter_by_similarity() {
        let mut report = DuplicateReport::new();

        let mut group1 = DuplicateGroup::new(vec!["a".to_string(), "b".to_string()]);
        group1.add_score(SimilarityScore::new("test".to_string(), 0.7));

        let mut group2 = DuplicateGroup::new(vec!["c".to_string(), "d".to_string()]);
        group2.add_score(SimilarityScore::new("test".to_string(), 0.95));

        report.add_group(group1);
        report.add_group(group2);

        report.filter_by_similarity(0.8);

        assert_eq!(report.groups.len(), 1);
        assert_eq!(report.groups[0].max_similarity(), 0.95);
    }
}
