# oximedia-renderfarm TODO

## Current Status
- 62 modules covering job management, worker pools, scheduling, cloud integration, cost tracking, tile rendering, and fault tolerance
- Key features: Coordinator, Scheduler, Worker management, Cloud bursting, Cost optimization, Multi-site support
- Dependencies: axum, tokio, rusqlite, prometheus, sysinfo, blake3, zstd, lz4

## Enhancements
- [ ] Add GPU resource tracking to `node_capability` (VRAM, CUDA/Vulkan compute units, GPU temperature)
- [ ] Extend `cost_optimizer` with spot instance pricing models and preemption handling for cloud workers
- [ ] Add weighted fair-share scheduling to `scheduler` alongside existing algorithms
- [ ] Implement render job dependency DAG validation in `job_dependency_graph` (cycle detection, unreachable node warnings)
- [ ] Add render progress ETA prediction in `progress` using historical completion rates per worker class
- [ ] Extend `elastic_scaling` with scale-down cooldown timers and min/max node constraints
- [ ] Add job preemption support in `render_job_queue` for higher-priority jobs arriving mid-render
- [ ] Implement chunk-level retry in `failure_recovery` instead of full-frame retry on transient errors

## New Features
- [x] Add a `render_template` module for reusable render configuration presets (resolution, codec, quality, frame range)
- [x] Add a `worker_benchmark` module to auto-profile worker performance and assign capability scores
- [ ] Implement a `render_cache` module for caching intermediate render outputs (e.g., lighting passes) across jobs
- [x] Add an `alert_rule` module with configurable alert thresholds (queue depth, idle workers, budget overrun)
- [ ] Implement a `resource_reservation` module for reserving worker capacity for scheduled high-priority jobs
- [ ] Add a `render_artifact` module for managing output files (checksums, storage locations, lifecycle policies)
- [ ] Implement `job_template` inheritance so child jobs inherit parent settings with overrides

## Performance
- [ ] Add connection pooling for the `api` axum handlers to reduce per-request overhead
- [ ] Implement batch insert for `render_log` entries instead of per-frame writes
- [ ] Add LRU eviction policy to `cache` module with configurable max memory usage
- [ ] Profile and optimize `tile_rendering` merge step for large frame resolutions (8K+)
- [ ] Use `crossbeam-channel` bounded channels in `frame_distribution` to apply backpressure when workers are saturated
- [ ] Implement zero-copy frame data transfer in `frame_merge` using memory-mapped files

## Testing
- [ ] Add integration tests for `multi_site` failover scenarios (primary site down, secondary takeover)
- [ ] Add load tests for `scheduler` with 1000+ concurrent jobs and 100+ workers
- [ ] Test `elastic_scaling` scale-up/scale-down timing under variable load
- [ ] Add property-based tests for `priority_queue` ordering guarantees
- [ ] Test `render_checkpoint` resume after simulated crash mid-frame

## Documentation
- [ ] Add architecture diagram showing Coordinator -> Scheduler -> Worker data flow
- [ ] Document the job lifecycle states in `job` module (Submitted -> Queued -> Running -> Complete/Failed)
- [ ] Add examples for `cloud` module showing hybrid on-prem + cloud bursting configuration
- [ ] Document `tile_rendering` strategy selection criteria (frame size, worker count, network bandwidth)
