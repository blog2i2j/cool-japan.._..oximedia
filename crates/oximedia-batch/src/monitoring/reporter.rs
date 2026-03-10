//! Report generation for batch processing

use crate::database::Database;
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;
use std::sync::Arc;

/// Report format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// HTML format
    Html,
    /// Plain text format
    Text,
}

/// Batch processing report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchReport {
    /// Report title
    pub title: String,
    /// Generation timestamp
    pub generated_at: String,
    /// Summary statistics
    pub summary: ReportSummary,
    /// Job details
    pub jobs: Vec<JobReport>,
}

/// Report summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Total jobs
    pub total_jobs: u64,
    /// Completed jobs
    pub completed: u64,
    /// Failed jobs
    pub failed: u64,
    /// Cancelled jobs
    pub cancelled: u64,
    /// Total duration in seconds
    pub total_duration_secs: f64,
    /// Success rate percentage
    pub success_rate: f64,
}

/// Individual job report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobReport {
    /// Job ID
    pub id: String,
    /// Job name
    pub name: String,
    /// Status
    pub status: String,
    /// Created timestamp
    pub created_at: String,
    /// Started timestamp
    pub started_at: Option<String>,
    /// Completed timestamp
    pub completed_at: Option<String>,
    /// Duration in seconds
    pub duration_secs: Option<f64>,
    /// Error message if failed
    pub error: Option<String>,
}

/// Report generator
pub struct ReportGenerator {
    database: Arc<Database>,
}

impl ReportGenerator {
    /// Create a new report generator
    #[must_use]
    pub fn new(database: Arc<Database>) -> Self {
        Self { database }
    }

    /// Generate a batch report
    ///
    /// # Errors
    ///
    /// Returns an error if report generation fails
    pub fn generate(&self) -> Result<BatchReport> {
        let job_stats = self.database.get_statistics()?;
        let jobs = self.database.list_jobs()?;

        // Count cancelled jobs from the database directly
        let cancelled = self.database.count_jobs_by_status("Cancelled")?;

        // Calculate total duration from all job records
        let total_duration_secs = self.database.get_total_duration_secs()?;

        let summary = ReportSummary {
            total_jobs: job_stats.total,
            completed: job_stats.completed,
            failed: job_stats.failed,
            cancelled,
            total_duration_secs,
            success_rate: if job_stats.total > 0 {
                #[allow(clippy::cast_precision_loss)]
                let rate = (job_stats.completed as f64 / job_stats.total as f64) * 100.0;
                rate
            } else {
                0.0
            },
        };

        let job_reports: Vec<JobReport> = jobs
            .into_iter()
            .map(|job| {
                let status = self
                    .database
                    .get_job_status_string(&job.id)
                    .unwrap_or_else(|_| "Unknown".to_string());
                let started_at = self.database.get_job_started_at(&job.id).ok().flatten();
                let completed_at = self.database.get_job_completed_at(&job.id).ok().flatten();
                let duration_secs = self.database.get_job_duration_secs(&job.id).ok().flatten();
                let error = self.database.get_job_error(&job.id).ok().flatten();

                JobReport {
                    id: job.id.to_string(),
                    name: job.name.clone(),
                    status,
                    created_at: chrono::Utc::now().to_rfc3339(),
                    started_at,
                    completed_at,
                    duration_secs,
                    error,
                }
            })
            .collect();

        Ok(BatchReport {
            title: "Batch Processing Report".to_string(),
            generated_at: chrono::Utc::now().to_rfc3339(),
            summary,
            jobs: job_reports,
        })
    }

    /// Export report to JSON
    ///
    /// # Arguments
    ///
    /// * `report` - The report to export
    ///
    /// # Errors
    ///
    /// Returns an error if export fails
    pub fn export_json(&self, report: &BatchReport) -> Result<String> {
        Ok(serde_json::to_string_pretty(report)?)
    }

    /// Export report to CSV
    ///
    /// # Arguments
    ///
    /// * `report` - The report to export
    ///
    /// # Errors
    ///
    /// Returns an error if export fails
    pub fn export_csv(&self, report: &BatchReport) -> Result<String> {
        let mut csv = String::from("ID,Name,Status,Created,Started,Completed,Duration,Error\n");

        for job in &report.jobs {
            let _ = writeln!(
                csv,
                "{},{},{},{},{},{},{},{}",
                job.id,
                job.name,
                job.status,
                job.created_at,
                job.started_at.as_deref().unwrap_or(""),
                job.completed_at.as_deref().unwrap_or(""),
                job.duration_secs.map_or(String::new(), |d| d.to_string()),
                job.error.as_deref().unwrap_or("")
            );
        }

        Ok(csv)
    }

    /// Export report to HTML
    ///
    /// # Arguments
    ///
    /// * `report` - The report to export
    ///
    /// # Errors
    ///
    /// Returns an error if export fails
    pub fn export_html(&self, report: &BatchReport) -> Result<String> {
        let mut html = String::from("<html><head><title>Batch Report</title></head><body>");
        let _ = write!(html, "<h1>{}</h1>", report.title);
        let _ = write!(html, "<p>Generated: {}</p>", report.generated_at);
        html.push_str("<h2>Summary</h2>");
        html.push_str("<table border='1'>");
        let _ = write!(
            html,
            "<tr><td>Total Jobs</td><td>{}</td></tr>",
            report.summary.total_jobs
        );
        let _ = write!(
            html,
            "<tr><td>Completed</td><td>{}</td></tr>",
            report.summary.completed
        );
        let _ = write!(
            html,
            "<tr><td>Failed</td><td>{}</td></tr>",
            report.summary.failed
        );
        let _ = write!(
            html,
            "<tr><td>Success Rate</td><td>{:.2}%</td></tr>",
            report.summary.success_rate
        );
        html.push_str("</table>");
        html.push_str("<h2>Jobs</h2>");
        html.push_str("<table border='1'>");
        html.push_str("<tr><th>ID</th><th>Name</th><th>Status</th><th>Duration</th></tr>");

        for job in &report.jobs {
            let _ = write!(
                html,
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                job.id,
                job.name,
                job.status,
                job.duration_secs
                    .map_or(String::new(), |d| format!("{d:.2}s"))
            );
        }

        html.push_str("</table>");
        html.push_str("</body></html>");

        Ok(html)
    }

    /// Export report in specified format
    ///
    /// # Arguments
    ///
    /// * `report` - The report to export
    /// * `format` - Desired format
    ///
    /// # Errors
    ///
    /// Returns an error if export fails
    pub fn export(&self, report: &BatchReport, format: &ReportFormat) -> Result<String> {
        match format {
            ReportFormat::Json => self.export_json(report),
            ReportFormat::Csv => self.export_csv(report),
            ReportFormat::Html => self.export_html(report),
            ReportFormat::Text => Ok(format!("{report:?}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_report_generator_creation() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let database = Arc::new(Database::new(db_path).expect("failed to create database"));

        let generator = ReportGenerator::new(database);
        assert!(std::mem::size_of_val(&generator) > 0);
    }

    #[test]
    fn test_generate_report() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let database = Arc::new(Database::new(db_path).expect("failed to create database"));

        let generator = ReportGenerator::new(database);
        let report = generator.generate();

        assert!(report.is_ok());
        let report = report.expect("report should be valid");
        assert_eq!(report.summary.total_jobs, 0);
    }

    #[test]
    fn test_generate_report_cancelled_count() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let database = Arc::new(Database::new(db_path).expect("failed to create database"));

        let generator = ReportGenerator::new(Arc::clone(&database));
        let report = generator.generate().expect("failed to generate");
        // Empty database: no cancelled jobs
        assert_eq!(report.summary.cancelled, 0);
    }

    #[test]
    fn test_generate_report_total_duration() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let database = Arc::new(Database::new(db_path).expect("failed to create database"));

        let generator = ReportGenerator::new(Arc::clone(&database));
        let report = generator.generate().expect("failed to generate");
        // Empty database: total duration is 0
        assert_eq!(report.summary.total_duration_secs, 0.0);
    }

    #[test]
    fn test_export_json() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let database = Arc::new(Database::new(db_path).expect("failed to create database"));

        let generator = ReportGenerator::new(database);
        let report = generator.generate().expect("failed to generate");
        let json = generator.export_json(&report);

        assert!(json.is_ok());
        let json_str = json.expect("json should be valid");
        assert!(json_str.contains("title"));
    }

    #[test]
    fn test_export_csv() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let database = Arc::new(Database::new(db_path).expect("failed to create database"));

        let generator = ReportGenerator::new(database);
        let report = generator.generate().expect("failed to generate");
        let csv = generator.export_csv(&report);

        assert!(csv.is_ok());
        let csv_str = csv.expect("csv should be valid");
        assert!(csv_str.contains("ID,Name,Status"));
    }

    #[test]
    fn test_export_html() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let database = Arc::new(Database::new(db_path).expect("failed to create database"));

        let generator = ReportGenerator::new(database);
        let report = generator.generate().expect("failed to generate");
        let html = generator.export_html(&report);

        assert!(html.is_ok());
        let html_str = html.expect("html should be valid");
        assert!(html_str.contains("<html>"));
        assert!(html_str.contains("</html>"));
    }
}
