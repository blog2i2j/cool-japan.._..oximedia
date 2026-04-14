# oximedia-workflow TODO

## Current Status
- 46 source files implementing comprehensive workflow orchestration engine
- Key features: DAG-based workflow definition, task dependencies and parallel execution, SQLite persistence (feature-gated), cron-style scheduling, REST API (axum), WebSocket real-time monitoring, WorkflowEngine main entry point, multi-pass encoding patterns, SLA tracking, cost tracking, approval gates, audit logging, notification system, retry policies, resource pools, workflow versioning/migration/snapshots/checkpoints
- Modules: api, approval_gate, audit_log, builder, cli, cost_tracking, dag, executor, monitoring, notification_system, patterns, persistence, queue, resource_pool, retry_policy, scheduler, sla/sla_tracking, state_machine, step_condition/step_conditions/step_result, task/task_dependency/task_graph/task_priority_queue/task_template, templates, triggers, validation (ComplexityAnalyzer), websocket, workflow/workflow_audit/workflow_checkpoint/workflow_log/workflow_metrics/workflow_migration/workflow_retry/workflow_snapshot/workflow_template/workflow_throttle/workflow_version
- Dependencies: tokio, axum, rusqlite (optional), serde, chrono, cron, uuid, clap, dashmap, parking_lot

## Enhancements
- [x] Add workflow branching/conditional paths in `dag` (if-else nodes that choose next task based on previous task output)
- [ ] Implement parallel fan-out/fan-in pattern in `executor` for tasks with shared dependencies
- [ ] Extend `retry_policy` with circuit breaker pattern (stop retrying after N consecutive failures across workflows)
- [ ] Add `workflow_migration` actual schema migration logic (currently likely scaffolding -- implement versioned DB migrations)
- [ ] Extend `triggers` with webhook triggers (HTTP POST starts workflow) beyond cron and file-watch
- [ ] Implement workflow pause/resume in `executor` with checkpoint serialization for long-running workflows
- [x] Add dynamic resource scaling in `resource_pool` based on queue depth (auto-allocate more workers under load)
- [x] Extend `notification_system` with Slack, email, and PagerDuty integration via webhook URLs
- [ ] Improve `cost_tracking` with actual cloud cost API integration (estimate compute cost per task based on duration and resource type)

## New Features
- [x] Implement `workflow_compose` module for composing smaller workflows into larger meta-workflows
- [ ] Add `workflow_import_export` for importing/exporting workflows as portable YAML/JSON bundles
- [x] Implement `workflow_diff` module for comparing two workflow versions and showing added/removed/changed tasks
- [ ] Add `workflow_simulation` dry-run mode that traces execution path without actually running tasks
- [x] Implement `workflow_marketplace` module for sharing and discovering reusable workflow templates
- [ ] Add `event_bus` module for publish/subscribe event-driven communication between workflow tasks
- [x] Implement `workflow_dashboard` data provider module that aggregates metrics for web UI consumption
- [x] Add `workflow_health_check` module for periodic validation of workflow engine health (DB connectivity, queue depth, stuck tasks)

## Performance
- [ ] Optimize `task_priority_queue` with binary heap instead of sorted Vec for O(log n) insert/extract
- [ ] Add connection pooling in `persistence` for SQLite (using r2d2 pool with configurable size)
- [ ] Implement batch task status updates in `executor` to reduce database write frequency
- [ ] Cache workflow DAG topology in `dag` after first computation to avoid recomputation on each execution
- [ ] Optimize `monitoring` metric collection with lock-free counters instead of mutex-guarded HashMap
- [ ] Add lazy deserialization in `persistence::load_workflow` to skip parsing task configs until accessed

## Testing
- [ ] Add integration test for full workflow lifecycle: create -> submit -> execute -> complete with SQLite persistence
- [ ] Test `dag` cycle detection with intentionally cyclic graphs and verify proper error reporting
- [ ] Add stress test for `queue` with 1000+ concurrent task submissions and verify ordering correctness
- [ ] Test `scheduler` cron trigger firing accuracy with mock clock
- [ ] Add `approval_gate` test verifying workflow blocks until approval is granted and resumes correctly
- [ ] Test `workflow_checkpoint` save/restore across process restarts (serialize state, reload, continue execution)
- [ ] Test `sla_tracking` with workflows that exceed SLA and verify breach notification

## Documentation
- [ ] Document workflow YAML/JSON schema with annotated examples for common media processing workflows
- [ ] Add REST API endpoint reference with request/response examples for `api` module
- [ ] Document task type catalog with required/optional parameters for each TaskType variant
