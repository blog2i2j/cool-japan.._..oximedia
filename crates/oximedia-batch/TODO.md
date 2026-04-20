# oximedia-batch TODO

## Current Status
- 30+ modules providing batch processing: job queuing, worker pool, templates, watch folders, distributed processing, REST API, CLI
- Feature-gated: `sqlite` (database, API, CLI, execution, watch), `scripting` (Lua via mlua)
- Sub-directories: `api/`, `cli/`, `database/`, `execution/`, `examples/`, `metrics/`, `monitoring/`, `notifications/`, `operations/`, `presets/`, `queue/`, `script/`, `template/`, `utils/`, `watch/`
- `BatchEngine` is the main entry point wrapping `JobQueue`, `ExecutionEngine`, and `Database`

## Enhancements
- [ ] Add graceful shutdown support to `BatchEngine::stop()` that waits for in-progress jobs to complete
- [ ] Extend `retry_policy` with exponential backoff jitter to prevent thundering herd on retries
- [ ] Add job dependency chaining in `dep_graph` so jobs can declare predecessor requirements
- [ ] Implement `checkpointing` module with periodic state snapshots for crash recovery
- [ ] Extend `notifications` with webhook callback support on job state transitions
- [ ] Add `rate_limiter` integration with `execution` to enforce per-user or per-project job limits
- [x] Improve `priority_queue` with fair scheduling to prevent starvation of low-priority jobs
- [ ] Add `batch_schedule` support for cron-like recurring job schedules
- [ ] Extend `pipeline_validator` to check for circular dependencies in DAG workflows

## New Features
- [x] Add `job_migration` module for upgrading job schemas when template format changes
- [ ] Implement `cost_estimator` module predicting job duration and resource usage from historical data
- [ ] Add `dead_letter_queue` for permanently failed jobs with configurable retention
- [x] Implement `job_splitting` module to automatically partition large transcode jobs across workers
- [ ] Add `audit_log` module tracking who submitted/modified/cancelled each job
- [ ] Implement `cluster_discovery` module for auto-detecting batch workers on the network
- [x] Add `resource_reservation` module for pre-allocating GPU/CPU cores for high-priority jobs

## Performance
- [ ] Use connection pooling for SQLite database access in `database` module
- [x] Implement work-stealing scheduler in `execution` for better load balancing across workers
- [ ] Add memory-mapped file I/O for large batch input files in `operations`
- [ ] Cache template parsing results in `template` to avoid re-parsing on repeated job submissions
- [ ] Use zero-copy deserialization for job payloads in `queue` using serde `borrow`

## Testing
- [ ] Add integration test for full job lifecycle: submit, queue, execute, complete, query status
- [ ] Test `watch` folder monitoring with rapid file creation/deletion
- [ ] Add test for `conditional_dag` with branching execution paths
- [ ] Test `timeout_enforcer` correctly cancels jobs exceeding time limits
- [ ] Add stress test submitting 10,000 jobs concurrently to verify queue stability
- [ ] Test `task_group` parallel execution with mixed success/failure outcomes

## Documentation
- [ ] Document REST API endpoints in `api` module with request/response examples
- [ ] Add guide for creating custom `template` configurations for common transcode workflows
- [ ] Document `watch` folder setup including supported file patterns and polling intervals

## Wave 4 Progress (2026-04-18)
- [x] wasm-mio-fix: cfg-gate tokio deps for wasm32 target compatibility — Wave 4 Slice A
