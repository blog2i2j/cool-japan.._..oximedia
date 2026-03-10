//! Worker implementation for job execution

use crate::database::Database;
use crate::queue::JobQueue;
use std::sync::Arc;

/// Worker for executing batch jobs
#[derive(Clone)]
pub struct Worker {
    id: usize,
    queue: Arc<JobQueue>,
    database: Arc<Database>,
}

impl Worker {
    /// Create a new worker
    #[must_use]
    pub fn new(id: usize, queue: Arc<JobQueue>, database: Arc<Database>) -> Self {
        Self {
            id,
            queue,
            database,
        }
    }

    /// Get worker ID
    #[must_use]
    pub const fn id(&self) -> usize {
        self.id
    }

    /// Get queue reference
    #[must_use]
    pub fn queue(&self) -> &Arc<JobQueue> {
        &self.queue
    }

    /// Get database reference
    #[must_use]
    pub fn database(&self) -> &Arc<Database> {
        &self.database
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_worker_creation() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");

        let database = Arc::new(Database::new(db_path).expect("failed to create database"));
        let queue = Arc::new(JobQueue::new());

        let worker = Worker::new(0, queue, database);
        assert_eq!(worker.id(), 0);
    }

    #[test]
    fn test_worker_clone() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");

        let database = Arc::new(Database::new(db_path).expect("failed to create database"));
        let queue = Arc::new(JobQueue::new());

        let worker1 = Worker::new(1, queue, database);
        let worker2 = worker1.clone();

        assert_eq!(worker1.id(), worker2.id());
    }
}
