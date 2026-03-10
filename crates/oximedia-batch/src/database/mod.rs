//! Database persistence for batch processing

pub mod schema;

use crate::error::Result;
use crate::job::{BatchJob, BatchOperation, InputSpec, OutputSpec};
use crate::types::{JobId, JobState, Priority, RetryPolicy};
use chrono::Utc;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;

/// Database for job persistence
pub struct Database {
    pool: Pool<SqliteConnectionManager>,
}

impl Database {
    /// Create a new database connection
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the `SQLite` database file
    ///
    /// # Errors
    ///
    /// Returns an error if database initialization fails
    pub fn new(path: &str) -> Result<Self> {
        let manager = SqliteConnectionManager::file(path);
        let pool = Pool::new(manager)?;

        let db = Self { pool };

        // Initialize schema
        db.init_schema()?;

        Ok(db)
    }

    /// Initialize database schema
    fn init_schema(&self) -> Result<()> {
        let conn = self.pool.get()?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                operation TEXT NOT NULL,
                inputs TEXT,
                outputs TEXT,
                priority INTEGER NOT NULL,
                retry_policy TEXT,
                dependencies TEXT,
                schedule TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                status TEXT NOT NULL,
                error TEXT
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS job_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS job_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                input_file TEXT NOT NULL,
                output_files TEXT,
                success INTEGER NOT NULL,
                error TEXT,
                duration REAL,
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_logs_job_id ON job_logs(job_id)",
            [],
        )?;

        Ok(())
    }

    /// Save a job to the database
    ///
    /// # Arguments
    ///
    /// * `job` - The job to save
    ///
    /// # Errors
    ///
    /// Returns an error if saving fails
    pub fn save_job(&self, job: &BatchJob) -> Result<()> {
        let conn = self.pool.get()?;

        let operation_json = serde_json::to_string(&job.operation)?;
        let inputs_json = serde_json::to_string(&job.inputs)?;
        let outputs_json = serde_json::to_string(&job.outputs)?;
        let retry_json = serde_json::to_string(&job.retry)?;
        let dependencies_json = serde_json::to_string(&job.dependencies)?;
        let schedule_json = serde_json::to_string(&job.schedule)?;
        let metadata_json = serde_json::to_string(&job.metadata)?;

        conn.execute(
            "INSERT OR REPLACE INTO jobs (
                id, name, operation, inputs, outputs, priority,
                retry_policy, dependencies, schedule, metadata,
                created_at, status
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                job.id.as_str(),
                job.name,
                operation_json,
                inputs_json,
                outputs_json,
                job.priority as i32,
                retry_json,
                dependencies_json,
                schedule_json,
                metadata_json,
                Utc::now().to_rfc3339(),
                "Queued",
            ],
        )?;

        Ok(())
    }

    /// Update job status
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job
    /// * `status` - New status
    ///
    /// # Errors
    ///
    /// Returns an error if update fails
    pub fn update_job_status(&self, job_id: &JobId, status: JobState) -> Result<()> {
        let conn = self.pool.get()?;

        let status_str = match status {
            JobState::Queued => "Queued",
            JobState::Running => "Running",
            JobState::Completed => "Completed",
            JobState::Failed => "Failed",
            JobState::Cancelled => "Cancelled",
            JobState::Pending => "Pending",
        };

        match status {
            JobState::Running => {
                conn.execute(
                    "UPDATE jobs SET status = ?1, started_at = ?2 WHERE id = ?3",
                    params![status_str, Utc::now().to_rfc3339(), job_id.as_str()],
                )?;
            }
            JobState::Completed | JobState::Failed | JobState::Cancelled => {
                conn.execute(
                    "UPDATE jobs SET status = ?1, completed_at = ?2 WHERE id = ?3",
                    params![status_str, Utc::now().to_rfc3339(), job_id.as_str()],
                )?;
            }
            _ => {
                conn.execute(
                    "UPDATE jobs SET status = ?1 WHERE id = ?2",
                    params![status_str, job_id.as_str()],
                )?;
            }
        }

        Ok(())
    }

    /// Log job error
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job
    /// * `error` - Error message
    ///
    /// # Errors
    ///
    /// Returns an error if logging fails
    pub fn log_job_error(&self, job_id: &JobId, error: &str) -> Result<()> {
        let conn = self.pool.get()?;

        conn.execute(
            "UPDATE jobs SET error = ?1 WHERE id = ?2",
            params![error, job_id.as_str()],
        )?;

        conn.execute(
            "INSERT INTO job_logs (job_id, timestamp, level, message)
             VALUES (?1, ?2, ?3, ?4)",
            params![job_id.as_str(), Utc::now().to_rfc3339(), "ERROR", error],
        )?;

        Ok(())
    }

    /// Get job by ID
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job
    ///
    /// # Errors
    ///
    /// Returns an error if the job is not found
    pub fn get_job(&self, job_id: &JobId) -> Result<BatchJob> {
        let conn = self.pool.get()?;

        let mut stmt = conn.prepare(
            "SELECT id, name, operation, inputs, outputs, priority, retry_policy
             FROM jobs WHERE id = ?1",
        )?;

        let job = stmt.query_row(params![job_id.as_str()], |row| {
            let id: String = row.get(0)?;
            let name: String = row.get(1)?;
            let operation_json: String = row.get(2)?;
            let inputs_json: String = row.get(3)?;
            let outputs_json: String = row.get(4)?;
            let priority: i32 = row.get(5)?;
            let retry_json: String = row.get(6)?;

            let operation: BatchOperation = serde_json::from_str(&operation_json)
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
            let inputs: Vec<InputSpec> = serde_json::from_str(&inputs_json)
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
            let outputs: Vec<OutputSpec> = serde_json::from_str(&outputs_json)
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
            let retry: RetryPolicy = serde_json::from_str(&retry_json)
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

            let priority_enum = match priority {
                0 => Priority::Low,
                2 => Priority::High,
                _ => Priority::Normal,
            };

            let mut job = BatchJob::new(name, operation);
            job.id = JobId::from_string(id);
            job.inputs = inputs;
            job.outputs = outputs;
            job.priority = priority_enum;
            job.retry = retry;

            Ok(job)
        })?;

        Ok(job)
    }

    /// List all jobs
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub fn list_jobs(&self) -> Result<Vec<BatchJob>> {
        let conn = self.pool.get()?;

        let mut stmt = conn.prepare(
            "SELECT id, name, operation, inputs, outputs, priority, retry_policy
             FROM jobs ORDER BY created_at DESC",
        )?;

        let jobs = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let name: String = row.get(1)?;
                let operation_json: String = row.get(2)?;
                let inputs_json: String = row.get(3)?;
                let outputs_json: String = row.get(4)?;
                let priority: i32 = row.get(5)?;
                let retry_json: String = row.get(6)?;

                let operation: BatchOperation = serde_json::from_str(&operation_json)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
                let inputs: Vec<InputSpec> = serde_json::from_str(&inputs_json)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
                let outputs: Vec<OutputSpec> = serde_json::from_str(&outputs_json)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
                let retry: RetryPolicy = serde_json::from_str(&retry_json)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

                let priority_enum = match priority {
                    0 => Priority::Low,
                    2 => Priority::High,
                    _ => Priority::Normal,
                };

                let mut job = BatchJob::new(name, operation);
                job.id = JobId::from_string(id);
                job.inputs = inputs;
                job.outputs = outputs;
                job.priority = priority_enum;
                job.retry = retry;

                Ok(job)
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(jobs)
    }

    /// Get job statistics
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub fn get_statistics(&self) -> Result<JobStatistics> {
        let conn = self.pool.get()?;

        let total: i64 = conn.query_row("SELECT COUNT(*) FROM jobs", [], |row| row.get(0))?;

        let queued: i64 = conn.query_row(
            "SELECT COUNT(*) FROM jobs WHERE status = 'Queued'",
            [],
            |row| row.get(0),
        )?;

        let running: i64 = conn.query_row(
            "SELECT COUNT(*) FROM jobs WHERE status = 'Running'",
            [],
            |row| row.get(0),
        )?;

        let completed: i64 = conn.query_row(
            "SELECT COUNT(*) FROM jobs WHERE status = 'Completed'",
            [],
            |row| row.get(0),
        )?;

        let failed: i64 = conn.query_row(
            "SELECT COUNT(*) FROM jobs WHERE status = 'Failed'",
            [],
            |row| row.get(0),
        )?;

        #[allow(clippy::cast_sign_loss)]
        Ok(JobStatistics {
            total: total as u64,
            queued: queued as u64,
            running: running as u64,
            completed: completed as u64,
            failed: failed as u64,
        })
    }

    /// Count jobs by status string
    ///
    /// # Arguments
    ///
    /// * `status` - Status string to count
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub fn count_jobs_by_status(&self, status: &str) -> Result<u64> {
        let conn = self.pool.get()?;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM jobs WHERE status = ?1",
            params![status],
            |row| row.get(0),
        )?;
        #[allow(clippy::cast_sign_loss)]
        Ok(count as u64)
    }

    /// Get total duration in seconds across all completed jobs
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub fn get_total_duration_secs(&self) -> Result<f64> {
        let conn = self.pool.get()?;
        // Sum durations from job_results table where duration is stored
        let total: Option<f64> = conn
            .query_row(
                "SELECT SUM(duration) FROM job_results WHERE success = 1",
                [],
                |row| row.get(0),
            )
            .ok()
            .flatten();
        Ok(total.unwrap_or(0.0))
    }

    /// Get the status string for a job
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job
    ///
    /// # Errors
    ///
    /// Returns an error if the job is not found
    pub fn get_job_status_string(&self, job_id: &crate::types::JobId) -> Result<String> {
        let conn = self.pool.get()?;
        let status: String = conn.query_row(
            "SELECT status FROM jobs WHERE id = ?1",
            params![job_id.as_str()],
            |row| row.get(0),
        )?;
        Ok(status)
    }

    /// Get `started_at` timestamp for a job
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub fn get_job_started_at(&self, job_id: &crate::types::JobId) -> Result<Option<String>> {
        let conn = self.pool.get()?;
        let started_at: Option<String> = conn
            .query_row(
                "SELECT started_at FROM jobs WHERE id = ?1",
                params![job_id.as_str()],
                |row| row.get(0),
            )
            .ok()
            .flatten();
        Ok(started_at)
    }

    /// Get `completed_at` timestamp for a job
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub fn get_job_completed_at(&self, job_id: &crate::types::JobId) -> Result<Option<String>> {
        let conn = self.pool.get()?;
        let completed_at: Option<String> = conn
            .query_row(
                "SELECT completed_at FROM jobs WHERE id = ?1",
                params![job_id.as_str()],
                |row| row.get(0),
            )
            .ok()
            .flatten();
        Ok(completed_at)
    }

    /// Get duration in seconds for a job
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub fn get_job_duration_secs(&self, job_id: &crate::types::JobId) -> Result<Option<f64>> {
        let conn = self.pool.get()?;
        // Sum durations of all result records for this job
        let duration: Option<f64> = conn
            .query_row(
                "SELECT SUM(duration) FROM job_results WHERE job_id = ?1",
                params![job_id.as_str()],
                |row| row.get(0),
            )
            .ok()
            .flatten();
        Ok(duration)
    }

    /// Get error message for a job
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails
    pub fn get_job_error(&self, job_id: &crate::types::JobId) -> Result<Option<String>> {
        let conn = self.pool.get()?;
        let error: Option<String> = conn
            .query_row(
                "SELECT error FROM jobs WHERE id = ?1",
                params![job_id.as_str()],
                |row| row.get(0),
            )
            .ok()
            .flatten();
        Ok(error)
    }
}

/// Job statistics
#[derive(Debug, Clone)]
pub struct JobStatistics {
    /// Total number of jobs
    pub total: u64,
    /// Queued jobs
    pub queued: u64,
    /// Running jobs
    pub running: u64,
    /// Completed jobs
    pub completed: u64,
    /// Failed jobs
    pub failed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::FileOperation;
    use tempfile::NamedTempFile;

    #[test]
    fn test_database_creation() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");

        let result = Database::new(db_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_save_job() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let db = Database::new(db_path).expect("failed to create database");

        let job = BatchJob::new(
            "test-job".to_string(),
            BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        let result = db.save_job(&job);
        assert!(result.is_ok());
    }

    #[test]
    fn test_update_job_status() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let db = Database::new(db_path).expect("failed to create database");

        let job = BatchJob::new(
            "test-job".to_string(),
            BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        db.save_job(&job).expect("failed to save job");
        let result = db.update_job_status(&job.id, JobState::Running);
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_job() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let db = Database::new(db_path).expect("failed to create database");

        let job = BatchJob::new(
            "test-job".to_string(),
            BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        db.save_job(&job).expect("failed to save job");
        let retrieved = db.get_job(&job.id);
        assert!(retrieved.is_ok());
        assert_eq!(
            retrieved.expect("retrieved should be valid").name,
            "test-job"
        );
    }

    #[test]
    fn test_list_jobs() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let db = Database::new(db_path).expect("failed to create database");

        let job1 = BatchJob::new(
            "job1".to_string(),
            BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        let job2 = BatchJob::new(
            "job2".to_string(),
            BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        db.save_job(&job1).expect("failed to save job");
        db.save_job(&job2).expect("failed to save job");

        let jobs = db.list_jobs().expect("failed to list jobs");
        assert_eq!(jobs.len(), 2);
    }

    #[test]
    fn test_get_statistics() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let db = Database::new(db_path).expect("failed to create database");

        let stats = db.get_statistics();
        assert!(stats.is_ok());

        let stats = stats.expect("stats should be valid");
        assert_eq!(stats.total, 0);
    }
}
