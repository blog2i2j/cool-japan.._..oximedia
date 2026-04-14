//! Persistence layer for the encoding farm
//!
//! Provides SQLite-based persistent storage for:
//! - Jobs and their state
//! - Tasks and their assignments
//! - Worker registrations
//! - Job history and audit logs
//! - Metrics and statistics

mod schema;

use crate::{FarmError, JobId, JobState, JobType, Priority, Result, TaskId, TaskState, WorkerId};
use chrono::{DateTime, Utc};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{params, OptionalExtension};
use std::collections::HashMap;
use std::path::Path;

pub use schema::Schema;

/// Database connection pool
pub type DbPool = Pool<SqliteConnectionManager>;

/// Database manager for farm persistence
pub struct Database {
    pool: DbPool,
}

impl Database {
    /// Create a new database connection
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let manager = SqliteConnectionManager::file(path);
        let pool = Pool::new(manager).map_err(|e| {
            FarmError::Database(rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        })?;

        let db = Self { pool };
        db.initialize_schema()?;
        Ok(db)
    }

    /// Create an in-memory database (for testing)
    pub fn in_memory() -> Result<Self> {
        let manager = SqliteConnectionManager::memory();
        let pool = Pool::new(manager).map_err(|e| {
            FarmError::Database(rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        })?;

        let db = Self { pool };
        db.initialize_schema()?;
        Ok(db)
    }

    /// Initialize database schema
    fn initialize_schema(&self) -> Result<()> {
        let conn = self.pool.get().map_err(|e| {
            FarmError::Database(rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        })?;
        Schema::create_tables(&conn)?;
        Ok(())
    }

    /// Insert a new job
    pub fn insert_job(&self, job: &JobRecord) -> Result<()> {
        let conn = self.pool.get().map_err(|e| {
            FarmError::Database(rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        })?;

        conn.execute(
            "INSERT INTO jobs (id, job_type, state, priority, input_path, output_path,
             parameters, metadata, created_at, deadline)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                job.id.to_string(),
                job.job_type.to_string(),
                job.state.to_string(),
                i32::from(job.priority),
                job.input_path,
                job.output_path,
                serde_json::to_string(&job.parameters)?,
                serde_json::to_string(&job.metadata)?,
                job.created_at.timestamp(),
                job.deadline.map(|d| d.timestamp()),
            ],
        )?;

        Ok(())
    }

    /// Update job state
    pub fn update_job_state(&self, job_id: JobId, state: JobState) -> Result<()> {
        let conn = self.pool.get().map_err(|e| {
            FarmError::Database(rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        })?;

        let now = Utc::now().timestamp();
        let updated = match state {
            JobState::Running => conn.execute(
                "UPDATE jobs SET state = ?1, started_at = ?2 WHERE id = ?3",
                params![state.to_string(), now, job_id.to_string()],
            )?,
            JobState::Completed
            | JobState::CompletedWithWarnings
            | JobState::Failed
            | JobState::Cancelled => conn.execute(
                "UPDATE jobs SET state = ?1, completed_at = ?2 WHERE id = ?3",
                params![state.to_string(), now, job_id.to_string()],
            )?,
            _ => conn.execute(
                "UPDATE jobs SET state = ?1 WHERE id = ?2",
                params![state.to_string(), job_id.to_string()],
            )?,
        };

        if updated == 0 {
            return Err(FarmError::NotFound(format!("Job {job_id} not found")));
        }

        Ok(())
    }

    /// Get job by ID
    pub fn get_job(&self, job_id: JobId) -> Result<Option<JobRecord>> {
        let conn = self.pool.get().map_err(|e| {
            FarmError::Database(rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        })?;

        let job = conn
            .query_row(
                "SELECT id, job_type, state, priority, input_path, output_path,
                 parameters, metadata, created_at, started_at, completed_at, deadline
                 FROM jobs WHERE id = ?1",
                params![job_id.to_string()],
                |row| {
                    let parameters: String = row.get(6)?;
                    let metadata: String = row.get(7)?;
                    let created_at: i64 = row.get(8)?;
                    let started_at: Option<i64> = row.get(9)?;
                    let completed_at: Option<i64> = row.get(10)?;
                    let deadline: Option<i64> = row.get(11)?;

                    let id_str: String = row.get(0)?;
                    let id_uuid = uuid::Uuid::parse_str(&id_str).map_err(|_| {
                        rusqlite::Error::InvalidColumnType(
                            0,
                            id_str.clone(),
                            rusqlite::types::Type::Text,
                        )
                    })?;
                    let priority_i = row.get::<_, i32>(3)?;
                    let priority = Priority::try_from(priority_i).map_err(|_| {
                        rusqlite::Error::InvalidColumnType(
                            3,
                            "priority".to_string(),
                            rusqlite::types::Type::Integer,
                        )
                    })?;
                    let params_map: HashMap<String, serde_json::Value> =
                        serde_json::from_str(&parameters).map_err(|_| {
                            rusqlite::Error::InvalidColumnType(
                                6,
                                "parameters".to_string(),
                                rusqlite::types::Type::Text,
                            )
                        })?;
                    let meta_map: HashMap<String, String> = serde_json::from_str(&metadata)
                        .map_err(|_| {
                            rusqlite::Error::InvalidColumnType(
                                7,
                                "metadata".to_string(),
                                rusqlite::types::Type::Text,
                            )
                        })?;
                    let created = DateTime::from_timestamp(created_at, 0).ok_or_else(|| {
                        rusqlite::Error::InvalidColumnType(
                            8,
                            "created_at".to_string(),
                            rusqlite::types::Type::Integer,
                        )
                    })?;

                    Ok(JobRecord {
                        id: JobId::from_uuid(id_uuid),
                        job_type: parse_job_type(&row.get::<_, String>(1)?),
                        state: parse_job_state(&row.get::<_, String>(2)?),
                        priority,
                        input_path: row.get(4)?,
                        output_path: row.get(5)?,
                        parameters: params_map,
                        metadata: meta_map,
                        created_at: created,
                        started_at: started_at.and_then(|ts| DateTime::from_timestamp(ts, 0)),
                        completed_at: completed_at.and_then(|ts| DateTime::from_timestamp(ts, 0)),
                        deadline: deadline.and_then(|ts| DateTime::from_timestamp(ts, 0)),
                    })
                },
            )
            .optional()?;

        Ok(job)
    }

    /// List jobs with optional filters
    pub fn list_jobs(
        &self,
        state_filter: Option<JobState>,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Vec<JobRecord>> {
        let conn = self.pool.get().map_err(|e| {
            FarmError::Database(rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        })?;

        let mut query = "SELECT id, job_type, state, priority, input_path, output_path,
                         parameters, metadata, created_at, started_at, completed_at, deadline
                         FROM jobs"
            .to_string();

        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

        if let Some(state) = state_filter {
            query.push_str(" WHERE state = ?");
            params_vec.push(Box::new(state.to_string()));
        }

        query.push_str(" ORDER BY created_at DESC");

        if let Some(lim) = limit {
            query.push_str(" LIMIT ?");
            params_vec.push(Box::new(lim as i64));
        }

        if let Some(off) = offset {
            query.push_str(" OFFSET ?");
            params_vec.push(Box::new(off as i64));
        }

        let params_refs: Vec<&dyn rusqlite::ToSql> =
            params_vec.iter().map(std::convert::AsRef::as_ref).collect();

        let mut stmt = conn.prepare(&query)?;
        let jobs = stmt
            .query_map(&params_refs[..], |row| {
                let parameters: String = row.get(6)?;
                let metadata: String = row.get(7)?;
                let created_at: i64 = row.get(8)?;
                let started_at: Option<i64> = row.get(9)?;
                let completed_at: Option<i64> = row.get(10)?;
                let deadline: Option<i64> = row.get(11)?;

                let id_str: String = row.get(0)?;
                let id_uuid = uuid::Uuid::parse_str(&id_str).map_err(|_| {
                    rusqlite::Error::InvalidColumnType(
                        0,
                        id_str.clone(),
                        rusqlite::types::Type::Text,
                    )
                })?;
                let priority_i = row.get::<_, i32>(3)?;
                let priority = Priority::try_from(priority_i).map_err(|_| {
                    rusqlite::Error::InvalidColumnType(
                        3,
                        "priority".to_string(),
                        rusqlite::types::Type::Integer,
                    )
                })?;
                let params_map: HashMap<String, serde_json::Value> =
                    serde_json::from_str(&parameters).map_err(|_| {
                        rusqlite::Error::InvalidColumnType(
                            6,
                            "parameters".to_string(),
                            rusqlite::types::Type::Text,
                        )
                    })?;
                let meta_map: HashMap<String, String> =
                    serde_json::from_str(&metadata).map_err(|_| {
                        rusqlite::Error::InvalidColumnType(
                            7,
                            "metadata".to_string(),
                            rusqlite::types::Type::Text,
                        )
                    })?;
                let created = DateTime::from_timestamp(created_at, 0).ok_or_else(|| {
                    rusqlite::Error::InvalidColumnType(
                        8,
                        "created_at".to_string(),
                        rusqlite::types::Type::Integer,
                    )
                })?;

                Ok(JobRecord {
                    id: JobId::from_uuid(id_uuid),
                    job_type: parse_job_type(&row.get::<_, String>(1)?),
                    state: parse_job_state(&row.get::<_, String>(2)?),
                    priority,
                    input_path: row.get(4)?,
                    output_path: row.get(5)?,
                    parameters: params_map,
                    metadata: meta_map,
                    created_at: created,
                    started_at: started_at.and_then(|ts| DateTime::from_timestamp(ts, 0)),
                    completed_at: completed_at.and_then(|ts| DateTime::from_timestamp(ts, 0)),
                    deadline: deadline.and_then(|ts| DateTime::from_timestamp(ts, 0)),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(jobs)
    }

    /// Insert a new task
    pub fn insert_task(&self, task: &TaskRecord) -> Result<()> {
        let conn = self.pool.get().map_err(|e| {
            FarmError::Database(rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        })?;

        conn.execute(
            "INSERT INTO tasks (id, job_id, state, worker_id, task_type, payload,
             priority, created_at, assigned_at, retry_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                task.id.to_string(),
                task.job_id.to_string(),
                task.state.to_string(),
                task.worker_id
                    .as_ref()
                    .map(std::string::ToString::to_string),
                task.task_type,
                task.payload,
                i32::from(task.priority),
                task.created_at.timestamp(),
                task.assigned_at.map(|d| d.timestamp()),
                task.retry_count,
            ],
        )?;

        Ok(())
    }

    /// Update task state
    pub fn update_task_state(
        &self,
        task_id: TaskId,
        state: TaskState,
        worker_id: Option<WorkerId>,
    ) -> Result<()> {
        let conn = self.pool.get().map_err(|e| {
            FarmError::Database(rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        })?;

        let now = Utc::now().timestamp();

        conn.execute(
            "UPDATE tasks SET state = ?1, worker_id = ?2, assigned_at = ?3 WHERE id = ?4",
            params![
                state.to_string(),
                worker_id.map(|w| w.to_string()),
                if state == TaskState::Assigned {
                    Some(now)
                } else {
                    None
                },
                task_id.to_string()
            ],
        )?;

        Ok(())
    }

    /// Get pending tasks for a job
    pub fn get_pending_tasks(&self, limit: usize) -> Result<Vec<TaskRecord>> {
        let conn = self.pool.get().map_err(|e| {
            FarmError::Database(rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        })?;

        let mut stmt = conn.prepare(
            "SELECT id, job_id, state, worker_id, task_type, payload, priority,
             created_at, assigned_at, retry_count
             FROM tasks
             WHERE state = 'Pending'
             ORDER BY priority DESC, created_at ASC
             LIMIT ?1",
        )?;

        let tasks = stmt
            .query_map(params![limit as i64], |row| {
                let created_at: i64 = row.get(7)?;
                let assigned_at: Option<i64> = row.get(8)?;

                let id_str: String = row.get(0)?;
                let id_uuid = uuid::Uuid::parse_str(&id_str).map_err(|_| {
                    rusqlite::Error::InvalidColumnType(
                        0,
                        id_str.clone(),
                        rusqlite::types::Type::Text,
                    )
                })?;
                let job_id_str: String = row.get(1)?;
                let job_id_uuid = uuid::Uuid::parse_str(&job_id_str).map_err(|_| {
                    rusqlite::Error::InvalidColumnType(
                        1,
                        job_id_str.clone(),
                        rusqlite::types::Type::Text,
                    )
                })?;
                let priority_i = row.get::<_, i32>(6)?;
                let priority = Priority::try_from(priority_i).map_err(|_| {
                    rusqlite::Error::InvalidColumnType(
                        6,
                        "priority".to_string(),
                        rusqlite::types::Type::Integer,
                    )
                })?;
                let created = DateTime::from_timestamp(created_at, 0).ok_or_else(|| {
                    rusqlite::Error::InvalidColumnType(
                        7,
                        "created_at".to_string(),
                        rusqlite::types::Type::Integer,
                    )
                })?;

                Ok(TaskRecord {
                    id: TaskId::from_uuid(id_uuid),
                    job_id: JobId::from_uuid(job_id_uuid),
                    state: parse_task_state(&row.get::<_, String>(2)?),
                    worker_id: row.get::<_, Option<String>>(3)?.map(WorkerId::new),
                    task_type: row.get(4)?,
                    payload: row.get(5)?,
                    priority,
                    created_at: created,
                    assigned_at: assigned_at.and_then(|ts| DateTime::from_timestamp(ts, 0)),
                    retry_count: row.get(9)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(tasks)
    }

    /// Get tasks for a specific job
    pub fn get_job_tasks(&self, job_id: JobId) -> Result<Vec<TaskRecord>> {
        let conn = self.pool.get().map_err(|e| {
            FarmError::Database(rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        })?;

        let mut stmt = conn.prepare(
            "SELECT id, job_id, state, worker_id, task_type, payload, priority,
             created_at, assigned_at, retry_count
             FROM tasks
             WHERE job_id = ?1",
        )?;

        let tasks = stmt
            .query_map(params![job_id.to_string()], |row| {
                let created_at: i64 = row.get(7)?;
                let assigned_at: Option<i64> = row.get(8)?;

                let id_str: String = row.get(0)?;
                let id_uuid = uuid::Uuid::parse_str(&id_str).map_err(|_| {
                    rusqlite::Error::InvalidColumnType(
                        0,
                        id_str.clone(),
                        rusqlite::types::Type::Text,
                    )
                })?;
                let job_id_str: String = row.get(1)?;
                let job_id_uuid = uuid::Uuid::parse_str(&job_id_str).map_err(|_| {
                    rusqlite::Error::InvalidColumnType(
                        1,
                        job_id_str.clone(),
                        rusqlite::types::Type::Text,
                    )
                })?;
                let priority_i = row.get::<_, i32>(6)?;
                let priority = Priority::try_from(priority_i).map_err(|_| {
                    rusqlite::Error::InvalidColumnType(
                        6,
                        "priority".to_string(),
                        rusqlite::types::Type::Integer,
                    )
                })?;
                let created = DateTime::from_timestamp(created_at, 0).ok_or_else(|| {
                    rusqlite::Error::InvalidColumnType(
                        7,
                        "created_at".to_string(),
                        rusqlite::types::Type::Integer,
                    )
                })?;

                Ok(TaskRecord {
                    id: TaskId::from_uuid(id_uuid),
                    job_id: JobId::from_uuid(job_id_uuid),
                    state: parse_task_state(&row.get::<_, String>(2)?),
                    worker_id: row.get::<_, Option<String>>(3)?.map(WorkerId::new),
                    task_type: row.get(4)?,
                    payload: row.get(5)?,
                    priority,
                    created_at: created,
                    assigned_at: assigned_at.and_then(|ts| DateTime::from_timestamp(ts, 0)),
                    retry_count: row.get(9)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(tasks)
    }

    /// Increment task retry count
    pub fn increment_task_retry(&self, task_id: TaskId) -> Result<u32> {
        let conn = self.pool.get().map_err(|e| {
            FarmError::Database(rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        })?;

        conn.execute(
            "UPDATE tasks SET retry_count = retry_count + 1 WHERE id = ?1",
            params![task_id.to_string()],
        )?;

        let retry_count: u32 = conn.query_row(
            "SELECT retry_count FROM tasks WHERE id = ?1",
            params![task_id.to_string()],
            |row| row.get(0),
        )?;

        Ok(retry_count)
    }

    /// Get job statistics
    pub fn get_job_stats(&self) -> Result<JobStats> {
        let conn = self.pool.get().map_err(|e| {
            FarmError::Database(rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        })?;

        let total: i64 = conn.query_row("SELECT COUNT(*) FROM jobs", [], |row| row.get(0))?;
        let pending: i64 = conn.query_row(
            "SELECT COUNT(*) FROM jobs WHERE state = 'Pending'",
            [],
            |row| row.get(0),
        )?;
        let queued: i64 = conn.query_row(
            "SELECT COUNT(*) FROM jobs WHERE state = 'Queued'",
            [],
            |row| row.get(0),
        )?;
        let running: i64 = conn.query_row(
            "SELECT COUNT(*) FROM jobs WHERE state = 'Running'",
            [],
            |row| row.get(0),
        )?;
        let completed: i64 = conn.query_row(
            "SELECT COUNT(*) FROM jobs WHERE state = 'Completed'",
            [],
            |row| row.get(0),
        )?;
        let failed: i64 = conn.query_row(
            "SELECT COUNT(*) FROM jobs WHERE state = 'Failed'",
            [],
            |row| row.get(0),
        )?;

        Ok(JobStats {
            total: total as u64,
            pending: pending as u64,
            queued: queued as u64,
            running: running as u64,
            completed: completed as u64,
            failed: failed as u64,
        })
    }
}

/// Job record in the database
#[derive(Debug, Clone)]
pub struct JobRecord {
    pub id: JobId,
    pub job_type: JobType,
    pub state: JobState,
    pub priority: Priority,
    pub input_path: String,
    pub output_path: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub deadline: Option<DateTime<Utc>>,
}

/// Task record in the database
#[derive(Debug, Clone)]
pub struct TaskRecord {
    pub id: TaskId,
    pub job_id: JobId,
    pub state: TaskState,
    pub worker_id: Option<WorkerId>,
    pub task_type: String,
    pub payload: Vec<u8>,
    pub priority: Priority,
    pub created_at: DateTime<Utc>,
    pub assigned_at: Option<DateTime<Utc>>,
    pub retry_count: u32,
}

/// Job statistics
#[derive(Debug, Clone)]
pub struct JobStats {
    pub total: u64,
    pub pending: u64,
    pub queued: u64,
    pub running: u64,
    pub completed: u64,
    pub failed: u64,
}

// Helper functions for parsing
fn parse_job_type(s: &str) -> JobType {
    match s {
        "VideoTranscode" => JobType::VideoTranscode,
        "AudioTranscode" => JobType::AudioTranscode,
        "ThumbnailGeneration" => JobType::ThumbnailGeneration,
        "QcValidation" => JobType::QcValidation,
        "MediaAnalysis" => JobType::MediaAnalysis,
        "ContentFingerprinting" => JobType::ContentFingerprinting,
        "MultiOutputTranscode" => JobType::MultiOutputTranscode,
        _ => JobType::VideoTranscode,
    }
}

fn parse_job_state(s: &str) -> JobState {
    match s {
        "Pending" => JobState::Pending,
        "Queued" => JobState::Queued,
        "Running" => JobState::Running,
        "Completed" => JobState::Completed,
        "CompletedWithWarnings" => JobState::CompletedWithWarnings,
        "Failed" => JobState::Failed,
        "Cancelled" => JobState::Cancelled,
        "Paused" => JobState::Paused,
        _ => JobState::Pending,
    }
}

fn parse_task_state(s: &str) -> TaskState {
    match s {
        "Pending" => TaskState::Pending,
        "Assigned" => TaskState::Assigned,
        "Running" => TaskState::Running,
        "Completed" => TaskState::Completed,
        "Failed" => TaskState::Failed,
        _ => TaskState::Pending,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_creation() {
        let db = Database::in_memory().unwrap();
        let stats = db.get_job_stats().unwrap();
        assert_eq!(stats.total, 0);
    }

    #[test]
    fn test_job_insertion() {
        let db = Database::in_memory().unwrap();
        let job = JobRecord {
            id: JobId::new(),
            job_type: JobType::VideoTranscode,
            state: JobState::Pending,
            priority: Priority::Normal,
            input_path: "/input/test.mp4".to_string(),
            output_path: "/output/test.mp4".to_string(),
            parameters: HashMap::new(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            deadline: None,
        };

        db.insert_job(&job).unwrap();
        let retrieved = db.get_job(job.id).unwrap().unwrap();
        assert_eq!(retrieved.id, job.id);
        assert_eq!(retrieved.state, JobState::Pending);
    }

    #[test]
    fn test_job_state_update() {
        let db = Database::in_memory().unwrap();
        let job = JobRecord {
            id: JobId::new(),
            job_type: JobType::VideoTranscode,
            state: JobState::Pending,
            priority: Priority::Normal,
            input_path: "/input/test.mp4".to_string(),
            output_path: "/output/test.mp4".to_string(),
            parameters: HashMap::new(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            deadline: None,
        };

        db.insert_job(&job).unwrap();
        db.update_job_state(job.id, JobState::Running).unwrap();

        let retrieved = db.get_job(job.id).unwrap().unwrap();
        assert_eq!(retrieved.state, JobState::Running);
        assert!(retrieved.started_at.is_some());
    }

    #[test]
    fn test_task_insertion() {
        let db = Database::in_memory().unwrap();

        // Create a job first to satisfy foreign key constraint
        let job_id = JobId::new();
        let job = JobRecord {
            id: job_id,
            job_type: JobType::VideoTranscode,
            priority: Priority::Normal,
            state: JobState::Pending,
            input_path: "/test/input.mp4".to_string(),
            output_path: "/test/output.mp4".to_string(),
            parameters: HashMap::new(),
            metadata: HashMap::new(),
            deadline: None,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
        };
        db.insert_job(&job).unwrap();

        let task = TaskRecord {
            id: TaskId::new(),
            job_id,
            state: TaskState::Pending,
            worker_id: None,
            task_type: "transcode".to_string(),
            payload: vec![1, 2, 3],
            priority: Priority::Normal,
            created_at: Utc::now(),
            assigned_at: None,
            retry_count: 0,
        };

        db.insert_task(&task).unwrap();
        let tasks = db.get_pending_tasks(10).unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, task.id);
    }

    #[test]
    fn test_task_priority_ordering() {
        let db = Database::in_memory().unwrap();

        // Create a job first to satisfy foreign key constraint
        let job_id = JobId::new();
        let job = JobRecord {
            id: job_id,
            job_type: JobType::VideoTranscode,
            priority: Priority::Normal,
            state: JobState::Pending,
            input_path: "/test/input.mp4".to_string(),
            output_path: "/test/output.mp4".to_string(),
            parameters: HashMap::new(),
            metadata: HashMap::new(),
            deadline: None,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
        };
        db.insert_job(&job).unwrap();

        let task_low = TaskRecord {
            id: TaskId::new(),
            job_id,
            state: TaskState::Pending,
            worker_id: None,
            task_type: "transcode".to_string(),
            payload: vec![],
            priority: Priority::Low,
            created_at: Utc::now(),
            assigned_at: None,
            retry_count: 0,
        };

        let task_high = TaskRecord {
            id: TaskId::new(),
            job_id,
            state: TaskState::Pending,
            worker_id: None,
            task_type: "transcode".to_string(),
            payload: vec![],
            priority: Priority::High,
            created_at: Utc::now(),
            assigned_at: None,
            retry_count: 0,
        };

        db.insert_task(&task_low).unwrap();
        db.insert_task(&task_high).unwrap();

        let tasks = db.get_pending_tasks(10).unwrap();
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].priority, Priority::High);
        assert_eq!(tasks[1].priority, Priority::Low);
    }
    #[test]
    fn test_retry_count() {
        let db = Database::in_memory().unwrap();

        // Create a job first to satisfy foreign key constraint
        let job_id = JobId::new();
        let job = JobRecord {
            id: job_id,
            job_type: JobType::VideoTranscode,
            priority: Priority::Normal,
            state: JobState::Pending,
            input_path: "/test/input.mp4".to_string(),
            output_path: "/test/output.mp4".to_string(),
            parameters: HashMap::new(),
            metadata: HashMap::new(),
            deadline: None,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
        };
        db.insert_job(&job).unwrap();

        let task = TaskRecord {
            id: TaskId::new(),
            job_id,
            state: TaskState::Pending,
            worker_id: None,
            task_type: "transcode".to_string(),
            payload: vec![],
            priority: Priority::Normal,
            created_at: Utc::now(),
            assigned_at: None,
            retry_count: 0,
        };

        db.insert_task(&task).unwrap();
        let count1 = db.increment_task_retry(task.id).unwrap();
        assert_eq!(count1, 1);

        let count2 = db.increment_task_retry(task.id).unwrap();
        assert_eq!(count2, 2);
    }
}
