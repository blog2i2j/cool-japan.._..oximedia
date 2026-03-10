//! Database schema definitions

use rusqlite::Connection;

pub struct Schema;

impl Schema {
    /// Create all database tables
    pub fn create_tables(conn: &Connection) -> Result<(), rusqlite::Error> {
        Self::create_jobs_table(conn)?;
        Self::create_tasks_table(conn)?;
        Self::create_workers_table(conn)?;
        Self::create_logs_table(conn)?;
        Self::create_metrics_table(conn)?;
        Ok(())
    }

    /// Create the jobs table
    fn create_jobs_table(conn: &Connection) -> Result<(), rusqlite::Error> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                state TEXT NOT NULL,
                priority INTEGER NOT NULL,
                input_path TEXT NOT NULL,
                output_path TEXT NOT NULL,
                parameters TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                started_at INTEGER,
                completed_at INTEGER,
                deadline INTEGER
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)",
            [],
        )?;

        Ok(())
    }

    /// Create the tasks table
    fn create_tasks_table(conn: &Connection) -> Result<(), rusqlite::Error> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                state TEXT NOT NULL,
                worker_id TEXT,
                task_type TEXT NOT NULL,
                payload BLOB NOT NULL,
                priority INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                assigned_at INTEGER,
                completed_at INTEGER,
                retry_count INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_job_id ON tasks(job_id)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_worker_id ON tasks(worker_id)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority)",
            [],
        )?;

        Ok(())
    }

    /// Create the workers table
    fn create_workers_table(conn: &Connection) -> Result<(), rusqlite::Error> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS workers (
                id TEXT PRIMARY KEY,
                hostname TEXT NOT NULL,
                state TEXT NOT NULL,
                capabilities TEXT NOT NULL,
                metadata TEXT NOT NULL,
                registered_at INTEGER NOT NULL,
                last_heartbeat INTEGER NOT NULL,
                active_tasks INTEGER NOT NULL DEFAULT 0
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_workers_state ON workers(state)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_workers_last_heartbeat ON workers(last_heartbeat)",
            [],
        )?;

        Ok(())
    }

    /// Create the logs table
    fn create_logs_table(conn: &Connection) -> Result<(), rusqlite::Error> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                job_id TEXT,
                task_id TEXT,
                worker_id TEXT,
                context TEXT
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_logs_job_id ON logs(job_id)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_logs_task_id ON logs(task_id)",
            [],
        )?;

        Ok(())
    }

    /// Create the metrics table
    fn create_metrics_table(conn: &Connection) -> Result<(), rusqlite::Error> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                labels TEXT NOT NULL
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name)",
            [],
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_creation() {
        let conn = Connection::open_in_memory().expect("operation should succeed");
        Schema::create_tables(&conn).expect("operation should succeed");

        // Verify tables exist
        let table_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
                [],
                |row| row.get(0),
            )
            .expect("operation should succeed");

        assert_eq!(table_count, 5); // jobs, tasks, workers, logs, metrics
    }

    #[test]
    fn test_jobs_table_exists() {
        let conn = Connection::open_in_memory().expect("operation should succeed");
        Schema::create_tables(&conn).expect("operation should succeed");

        let exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='jobs'",
                [],
                |row| row.get(0),
            )
            .expect("operation should succeed");

        assert_eq!(exists, 1);
    }

    #[test]
    fn test_tasks_table_exists() {
        let conn = Connection::open_in_memory().expect("operation should succeed");
        Schema::create_tables(&conn).expect("operation should succeed");

        let exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='tasks'",
                [],
                |row| row.get(0),
            )
            .expect("operation should succeed");

        assert_eq!(exists, 1);
    }

    #[test]
    fn test_workers_table_exists() {
        let conn = Connection::open_in_memory().expect("operation should succeed");
        Schema::create_tables(&conn).expect("operation should succeed");

        let exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='workers'",
                [],
                |row| row.get(0),
            )
            .expect("operation should succeed");

        assert_eq!(exists, 1);
    }
}
