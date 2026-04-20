//! Export review data to various formats.

use crate::{error::ReviewResult, SessionId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub mod csv;
pub mod edl;
pub mod pdf;

pub use csv::export_to_csv;
pub use edl::export_to_edl;
pub use pdf::export_to_pdf;

/// Export format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// PDF report.
    Pdf,
    /// CSV spreadsheet.
    Csv,
    /// EDL (Edit Decision List).
    Edl,
    /// JSON.
    Json,
    /// XML.
    Xml,
}

/// Export options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    /// Export format.
    pub format: ExportFormat,
    /// Include comments.
    pub include_comments: bool,
    /// Include drawings.
    pub include_drawings: bool,
    /// Include tasks.
    pub include_tasks: bool,
    /// Include approval history.
    pub include_approvals: bool,
    /// Include change requests.
    pub include_changes: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::Pdf,
            include_comments: true,
            include_drawings: true,
            include_tasks: true,
            include_approvals: true,
            include_changes: true,
        }
    }
}

impl ExportOptions {
    /// Create new export options.
    #[must_use]
    pub fn new(format: ExportFormat) -> Self {
        Self {
            format,
            ..Default::default()
        }
    }

    /// Set whether to include comments.
    #[must_use]
    pub fn include_comments(mut self, include: bool) -> Self {
        self.include_comments = include;
        self
    }

    /// Set whether to include drawings.
    #[must_use]
    pub fn include_drawings(mut self, include: bool) -> Self {
        self.include_drawings = include;
        self
    }

    /// Set whether to include tasks.
    #[must_use]
    pub fn include_tasks(mut self, include: bool) -> Self {
        self.include_tasks = include;
        self
    }
}

/// Export result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResult {
    /// Export ID.
    pub id: String,
    /// Session ID.
    pub session_id: SessionId,
    /// Export format.
    pub format: ExportFormat,
    /// Output file path.
    pub file_path: String,
    /// File size in bytes.
    pub file_size: u64,
    /// Export timestamp.
    pub exported_at: DateTime<Utc>,
}

/// Export session data.
///
/// # Errors
///
/// Returns error if export fails.
pub async fn export_session(
    session_id: SessionId,
    options: ExportOptions,
    output_path: &str,
) -> ReviewResult<ExportResult> {
    // In a real implementation, this would:
    // 1. Collect all session data based on options
    // 2. Format according to export format
    // 3. Write to file
    // 4. Return export result

    let result = ExportResult {
        id: uuid::Uuid::new_v4().to_string(),
        session_id,
        format: options.format,
        file_path: output_path.to_string(),
        file_size: 0,
        exported_at: Utc::now(),
    };

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_options_default() {
        let options = ExportOptions::default();
        assert_eq!(options.format, ExportFormat::Pdf);
        assert!(options.include_comments);
        assert!(options.include_drawings);
    }

    #[test]
    fn test_export_options_builder() {
        let options = ExportOptions::new(ExportFormat::Csv)
            .include_comments(false)
            .include_drawings(true);

        assert_eq!(options.format, ExportFormat::Csv);
        assert!(!options.include_comments);
        assert!(options.include_drawings);
    }

    #[tokio::test]
    async fn test_export_session() {
        let session_id = SessionId::new();
        let options = ExportOptions::new(ExportFormat::Pdf);

        let out_path = std::env::temp_dir()
            .join("oximedia-review-export-session.pdf")
            .to_string_lossy()
            .into_owned();
        let result = export_session(session_id, options, &out_path).await;
        assert!(result.is_ok());

        let export_result = result.expect("should succeed in test");
        assert_eq!(export_result.session_id, session_id);
        assert_eq!(export_result.format, ExportFormat::Pdf);
    }

    #[test]
    fn test_export_format_equality() {
        assert_eq!(ExportFormat::Pdf, ExportFormat::Pdf);
        assert_ne!(ExportFormat::Pdf, ExportFormat::Csv);
    }
}
