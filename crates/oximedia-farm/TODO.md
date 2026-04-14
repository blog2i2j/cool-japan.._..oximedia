# oximedia-farm TODO

## Current Status
- 34 source files covering coordinator, worker, communication, persistence, scheduling, fault tolerance
- gRPC communication with TLS support, SQLite persistence (feature-gated)
- Job types: VideoTranscode, AudioTranscode, ThumbnailGeneration, QcValidation, MediaAnalysis, ContentFingerprinting, MultiOutputTranscode
- Scheduling strategies, priority queue, load balancing, capacity planning
- Fault tolerance: circuit breaker, retry, checkpointing, task preemption
- Prometheus metrics, health monitoring, node affinity, worker pool
- Dependencies: tonic, prost, rusqlite, tokio, sysinfo, prometheus, dashmap, parking_lot

## Enhancements
- [x] Add job chaining/pipeline support in `dependency.rs` (output of job A feeds into job B)
- [x] Implement weighted scoring in `load_balancer.rs` combining CPU, GPU, memory, and network metrics
- [x] Enhance `scheduler/strategies.rs` with deadline-aware scheduling (priority boost as deadline approaches)
- [x] Add worker capability matching in `task_allocator.rs` (GPU jobs only to GPU-capable workers)
- [x] Implement progressive job status updates in `coordinator/job_queue.rs` with percentage and ETA
- [x] Enhance `farm_config.rs` with YAML/TOML configuration file loading
- [x] Add `worker_pool.rs` auto-scaling based on queue depth with min/max worker counts
- [x] Implement job output validation in `coordinator/mod.rs` before marking as completed
- [x] Add `render_stats.rs` historical render time prediction for capacity planning
- [x] Enhance `checkpoint.rs` with incremental checkpointing to reduce I/O overhead

## New Features
- [x] Implement web dashboard API endpoints for monitoring farm status
- [x] Add email/webhook notification system for job completion and failure events
- [x] Implement job templates in `job_template.rs` with parameterized encoding presets
- [x] Add multi-tenant support with per-tenant job quotas and resource isolation
- [x] Implement cost estimation per job based on resource usage and time
- [ ] Add S3/cloud object storage integration for input/output file management
- [x] Implement worker health scoring with automatic quarantine of failing workers
- [x] Add job dependency DAG visualization for complex encoding pipelines

## Performance
- [x] Optimize `persistence/schema.rs` SQLite queries with prepared statements and indexes
- [ ] Add connection pooling for SQLite using r2d2 (already a dependency, ensure usage)
- [ ] Implement batch heartbeat processing in coordinator to reduce per-heartbeat overhead
- [x] Use gRPC streaming for progress updates instead of individual RPCs
- [ ] Add in-memory caching layer in front of SQLite for frequently accessed job state
- [x] Optimize `priority_queue.rs` with a proper binary heap instead of sorted Vec

## Testing
- [x] Add end-to-end integration test: submit job -> assign to worker -> complete -> verify output
- [ ] Test `fault_tolerance/circuit_breaker.rs` state transitions (closed -> open -> half-open)
- [ ] Test `fault_tolerance/retry.rs` with exponential backoff and jitter verification
- [ ] Add `scheduler/strategies.rs` benchmarks with 10K+ jobs and 100+ workers
- [ ] Test `persistence/schema.rs` database migration and schema evolution
- [x] Test graceful shutdown: worker draining, in-flight task handoff
- [x] Add chaos testing: random worker disconnects during active jobs

## Documentation
- [ ] Document the coordinator-worker communication protocol (protobuf service definitions)
- [ ] Add deployment guide with Docker/Kubernetes configuration examples
- [ ] Document the scheduling algorithm decision tree and priority system
