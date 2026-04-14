//! Database schema definitions with optimized SQLite settings.
//!
//! Initializes all tables, indexes, and per-connection PRAGMAs for the farm
//! persistence layer.  Key optimizations:
//! - `PRAGMA journal_mode=WAL` — enables concurrent read/write access.
//! - `PRAGMA synchronous=NORMAL` — improves write throughput while keeping
//!   crash-safety sufficient for a media-processing workload.
//! - Composite index `idx_jobs_priority_created` for scheduler hot-path.
//! - Frequent queries wrapped in `conn.prepare_cached()` for statement reuse.

use rusqlite::Connection;

pub struct Schema;

impl Schema {
    /// Apply connection-level PRAGMAs for better concurrent performance.
    ///
    /// Must be called on every new connection before any queries are issued.
    pub fn apply_pragmas(conn: &Connection) -> Result<(), rusqlite::Error> {
        // WAL mode allows concurrent readers alongside a single writer.
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;
        // NORMAL is safe for a media-processing coordinator: WAL already
        // guards against most corruption scenarios.
        conn.execute_batch("PRAGMA synchronous=NORMAL;")?;
        Ok(())
    }

    /// Create all database tables and indexes.
    pub fn create_tables(conn: &Connection) -> Result<(), rusqlite::Error> {
        Self::apply_pragmas(conn)?;
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

        // Simple state index for queries like WHERE state='pending'
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state)",
            [],
        )?;

        // Composite priority + created_at index for the scheduler hot-path:
        // "give me the highest-priority job submitted first".
        // SQLite does not support DESC in `CREATE INDEX` in all versions,
        // so we use ASC and rely on the query planner to use it in both
        // directions.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority, created_at)",
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

    // ── CachedStatement helpers ───────────────────────────────────────────────

    /// Insert a job row using a cached prepared statement.
    ///
    /// Returns the number of rows inserted (1 on success).
    #[allow(clippy::too_many_arguments)]
    pub fn insert_job_cached(
        conn: &Connection,
        id: &str,
        job_type: &str,
        state: &str,
        priority: i64,
        input_path: &str,
        output_path: &str,
        parameters: &str,
        metadata: &str,
        created_at: i64,
    ) -> Result<usize, rusqlite::Error> {
        let mut stmt = conn.prepare_cached(
            "INSERT INTO jobs (id, job_type, state, priority, input_path, output_path, \
             parameters, metadata, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        )?;
        stmt.execute(rusqlite::params![
            id,
            job_type,
            state,
            priority,
            input_path,
            output_path,
            parameters,
            metadata,
            created_at,
        ])
    }

    /// Query pending jobs ordered by priority (highest first) then oldest first,
    /// using a cached prepared statement to avoid repeated parse overhead.
    pub fn query_pending_jobs_cached(
        conn: &Connection,
    ) -> Result<Vec<(String, String, i64, i64)>, rusqlite::Error> {
        let mut stmt = conn.prepare_cached(
            "SELECT id, state, priority, created_at \
             FROM jobs \
             WHERE state = 'pending' \
             ORDER BY priority DESC, created_at ASC \
             LIMIT 100",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })?;
        rows.collect()
    }

    /// Update a job's state using a cached prepared statement.
    pub fn update_job_state_cached(
        conn: &Connection,
        id: &str,
        new_state: &str,
    ) -> Result<usize, rusqlite::Error> {
        let mut stmt = conn.prepare_cached("UPDATE jobs SET state = ?1 WHERE id = ?2")?;
        stmt.execute(rusqlite::params![new_state, id])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_creation() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        Schema::create_tables(&conn).expect("create tables");

        // Verify tables exist
        let table_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
                [],
                |row| row.get(0),
            )
            .expect("count tables");

        assert_eq!(table_count, 5); // jobs, tasks, workers, logs, metrics
    }

    #[test]
    fn test_jobs_table_exists() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        Schema::create_tables(&conn).expect("create tables");

        let exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='jobs'",
                [],
                |row| row.get(0),
            )
            .expect("query");

        assert_eq!(exists, 1);
    }

    #[test]
    fn test_tasks_table_exists() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        Schema::create_tables(&conn).expect("create tables");

        let exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='tasks'",
                [],
                |row| row.get(0),
            )
            .expect("query");

        assert_eq!(exists, 1);
    }

    #[test]
    fn test_workers_table_exists() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        Schema::create_tables(&conn).expect("create tables");

        let exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='workers'",
                [],
                |row| row.get(0),
            )
            .expect("query");

        assert_eq!(exists, 1);
    }

    // ── New optimized-query tests (Task F) ─────────────────────────────────────

    #[test]
    fn test_pragmas_are_applied_without_error() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        // apply_pragmas should succeed even on an empty DB.
        Schema::apply_pragmas(&conn).expect("apply pragmas");
    }

    #[test]
    fn test_insert_and_query_pending_jobs_cached() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        Schema::create_tables(&conn).expect("create tables");

        // Insert two pending jobs with different priorities.
        Schema::insert_job_cached(
            &conn,
            "job-1",
            "transcode",
            "pending",
            5,
            "/in/a.mp4",
            "/out/a.mp4",
            "{}",
            "{}",
            1000,
        )
        .expect("insert job-1");
        Schema::insert_job_cached(
            &conn,
            "job-2",
            "transcode",
            "pending",
            10,
            "/in/b.mp4",
            "/out/b.mp4",
            "{}",
            "{}",
            900,
        )
        .expect("insert job-2");

        let rows = Schema::query_pending_jobs_cached(&conn).expect("query pending");
        assert_eq!(rows.len(), 2);
        // Highest priority (10) first — job-2.
        assert_eq!(rows[0].0, "job-2");
        assert_eq!(rows[0].2, 10);
    }

    #[test]
    fn test_update_job_state_cached() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        Schema::create_tables(&conn).expect("create tables");

        Schema::insert_job_cached(
            &conn,
            "job-x",
            "transcode",
            "pending",
            1,
            "/in/x.mp4",
            "/out/x.mp4",
            "{}",
            "{}",
            100,
        )
        .expect("insert");

        let updated =
            Schema::update_job_state_cached(&conn, "job-x", "running").expect("update state");
        assert_eq!(updated, 1);

        let state: String = conn
            .query_row("SELECT state FROM jobs WHERE id = 'job-x'", [], |row| {
                row.get(0)
            })
            .expect("query state");
        assert_eq!(state, "running");
    }

    #[test]
    fn test_composite_index_exists_on_jobs() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        Schema::create_tables(&conn).expect("create tables");

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master \
                 WHERE type='index' AND name='idx_jobs_priority'",
                [],
                |row| row.get(0),
            )
            .expect("query index");
        assert_eq!(count, 1, "composite priority+created_at index should exist");
    }

    #[test]
    fn test_query_only_returns_pending_jobs() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        Schema::create_tables(&conn).expect("create tables");

        // Insert one pending and one running job.
        Schema::insert_job_cached(
            &conn,
            "pending-job",
            "transcode",
            "pending",
            5,
            "/in/p.mp4",
            "/out/p.mp4",
            "{}",
            "{}",
            200,
        )
        .expect("insert pending");
        Schema::insert_job_cached(
            &conn,
            "running-job",
            "transcode",
            "running",
            5,
            "/in/r.mp4",
            "/out/r.mp4",
            "{}",
            "{}",
            100,
        )
        .expect("insert running");

        let rows = Schema::query_pending_jobs_cached(&conn).expect("query");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, "pending-job");
    }

    #[test]
    fn test_wal_pragma_executes_and_synchronous_pragma_executes() {
        // In-memory SQLite always reports "memory" for journal_mode regardless of WAL pragma
        // (in-memory DBs cannot use WAL), but the pragmas must execute without error and
        // the synchronous setting must be readable as a valid integer mode.
        let conn = Connection::open_in_memory().expect("in-memory db");
        Schema::apply_pragmas(&conn).expect("pragmas execute without error");

        // Verify journal_mode returns a non-empty string (memory, wal, delete, etc.)
        let journal_mode: String = conn
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .expect("query journal_mode");
        assert!(
            !journal_mode.is_empty(),
            "journal_mode should be a non-empty string"
        );

        // Verify synchronous pragma returns a valid integer (0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA)
        let sync_mode: i64 = conn
            .query_row("PRAGMA synchronous", [], |row| row.get(0))
            .expect("query synchronous");
        assert!(
            (0..=3).contains(&sync_mode),
            "synchronous pragma should be in range 0–3, got {sync_mode}"
        );
    }

    // ── Schema migration / idempotency tests ──────────────────────────────────

    #[test]
    fn test_create_tables_is_idempotent() {
        // Calling create_tables twice on the same connection must succeed without
        // error. All DDL statements use IF NOT EXISTS so this is expected to be
        // a no-op on the second call.
        let conn = Connection::open_in_memory().expect("in-memory db");
        Schema::create_tables(&conn).expect("first create_tables call");
        Schema::create_tables(&conn).expect("second create_tables call should be idempotent");

        // Verify that no extra tables were spuriously created.
        let table_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
                [],
                |row| row.get(0),
            )
            .expect("count tables after double-create");
        assert_eq!(
            table_count, 5,
            "idempotent create_tables must not duplicate tables"
        );
    }

    #[test]
    fn test_all_expected_table_names_exist() {
        let conn = Connection::open_in_memory().expect("in-memory db");
        Schema::create_tables(&conn).expect("create tables");

        let expected = ["jobs", "tasks", "workers", "logs", "metrics"];
        for name in expected {
            let exists: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                    rusqlite::params![name],
                    |row| row.get(0),
                )
                .expect("query table existence");
            assert_eq!(exists, 1, "table '{name}' should exist after create_tables");
        }
    }

    #[test]
    fn test_key_indexes_exist_after_double_create() {
        // Indexes must survive the idempotent double-create call.
        let conn = Connection::open_in_memory().expect("in-memory db");
        Schema::create_tables(&conn).expect("first create");
        Schema::create_tables(&conn).expect("second create");

        let expected_indexes = [
            "idx_jobs_state",
            "idx_jobs_priority",
            "idx_jobs_created_at",
            "idx_tasks_job_id",
            "idx_tasks_state",
            "idx_workers_state",
            "idx_workers_last_heartbeat",
            "idx_logs_timestamp",
            "idx_metrics_timestamp",
        ];
        for idx in expected_indexes {
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?1",
                    rusqlite::params![idx],
                    |row| row.get(0),
                )
                .expect("query index");
            assert_eq!(
                count, 1,
                "index '{idx}' should exist after double create_tables"
            );
        }
    }

    #[test]
    fn test_insert_and_query_work_after_idempotent_create() {
        // Ensure that DML still works correctly after the idempotent re-creation.
        let conn = Connection::open_in_memory().expect("in-memory db");
        Schema::create_tables(&conn).expect("first create");
        Schema::create_tables(&conn).expect("second create (idempotent)");

        // Insert a job and verify it is retrievable.
        Schema::insert_job_cached(
            &conn,
            "job-idem",
            "transcode",
            "pending",
            7,
            "/in/idem.mp4",
            "/out/idem.mp4",
            "{}",
            "{}",
            500,
        )
        .expect("insert after idempotent create");

        let rows = Schema::query_pending_jobs_cached(&conn).expect("query pending");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, "job-idem");
        assert_eq!(rows[0].2, 7, "priority should be preserved");
    }

    #[test]
    fn test_priority_ordering_with_multiple_jobs() {
        // Verify the scheduler hot-path ordering: highest priority first, then oldest.
        let conn = Connection::open_in_memory().expect("in-memory db");
        Schema::create_tables(&conn).expect("create tables");

        // Insert jobs with different priorities and creation times.
        let jobs = [
            ("j-low", 1i64, 1000i64),
            ("j-high", 10, 2000),
            ("j-mid", 5, 1500),
            ("j-hi2", 10, 1800), // same priority as j-high, older
        ];
        for (id, prio, ts) in jobs {
            Schema::insert_job_cached(
                &conn,
                id,
                "transcode",
                "pending",
                prio,
                "/in/x.mp4",
                "/out/x.mp4",
                "{}",
                "{}",
                ts,
            )
            .expect("insert");
        }

        let rows = Schema::query_pending_jobs_cached(&conn).expect("query pending");
        assert_eq!(rows.len(), 4);

        // First row: highest priority (10) and oldest created_at (1800) → j-hi2
        assert_eq!(
            rows[0].0, "j-hi2",
            "oldest among highest-priority should come first"
        );
        assert_eq!(rows[0].2, 10);
        // Second row: same priority, newer → j-high
        assert_eq!(rows[1].0, "j-high");
        // Third row: priority 5
        assert_eq!(rows[2].2, 5);
        // Last: priority 1
        assert_eq!(rows[3].2, 1);
    }

    #[test]
    fn test_update_job_state_affects_pending_query() {
        // After updating a job from 'pending' to 'running', it must not appear
        // in the pending-jobs query result.
        let conn = Connection::open_in_memory().expect("in-memory db");
        Schema::create_tables(&conn).expect("create tables");

        Schema::insert_job_cached(
            &conn,
            "j-a",
            "transcode",
            "pending",
            5,
            "/in/a.mp4",
            "/out/a.mp4",
            "{}",
            "{}",
            100,
        )
        .expect("insert j-a");
        Schema::insert_job_cached(
            &conn,
            "j-b",
            "transcode",
            "pending",
            3,
            "/in/b.mp4",
            "/out/b.mp4",
            "{}",
            "{}",
            200,
        )
        .expect("insert j-b");

        let before = Schema::query_pending_jobs_cached(&conn).expect("query before update");
        assert_eq!(before.len(), 2);

        Schema::update_job_state_cached(&conn, "j-a", "running").expect("update j-a");

        let after = Schema::query_pending_jobs_cached(&conn).expect("query after update");
        assert_eq!(after.len(), 1, "only j-b should remain pending");
        assert_eq!(after[0].0, "j-b");
    }
}
