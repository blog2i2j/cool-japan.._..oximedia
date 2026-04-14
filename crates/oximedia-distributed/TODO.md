# oximedia-distributed TODO

## Current Status
- 35 modules covering coordination, discovery, fault tolerance, consensus, sharding, and task management
- gRPC-based communication via tonic with protobuf serialization
- Features: backpressure, checkpointing, circuit breaker, Raft primitives, work stealing, replication
- Discovery methods: Static, mDNS, etcd, Consul
- Split strategies: Segment-based, Tile-based, GOP-based
- Dependencies: tonic, prost, tokio, dashmap, hickory-resolver, serde_json

## Enhancements
- [ ] Implement actual gRPC call in `DistributedEncoder::submit_job()` (currently returns Ok immediately)
- [ ] Implement actual gRPC call in `DistributedEncoder::job_status()` (currently returns `Pending`)
- [ ] Implement actual gRPC call in `DistributedEncoder::cancel_job()` (currently no-op)
- [ ] Add connection pooling and retry logic to coordinator client connections
- [x] Enhance `load_balancer.rs` with weighted round-robin based on worker capability scores
- [x] Add job dependency DAG support in `task_distribution.rs` for multi-step encoding pipelines
- [x] Implement graceful worker draining in `worker.rs` for rolling updates
- [ ] Add configurable backpressure thresholds in `backpressure.rs` per worker capacity
- [ ] Enhance `circuit_breaker.rs` with half-open state and configurable failure window
- [ ] Add job progress percentage tracking and ETA estimation in `job_tracker.rs`

## New Features
- [ ] Add WebSocket-based real-time job status notification channel
- [ ] Implement cross-region distributed encoding with geo-aware task placement
- [ ] Add S3/object-storage integration for distributed input/output file management
- [ ] Implement job preemption for higher-priority jobs in `task_priority_queue.rs`
- [ ] Add worker auto-scaling hooks (Kubernetes HPA integration)
- [ ] Implement distributed merge/concatenation of encoded segments after parallel encoding
- [ ] Add audit logging for all coordinator state changes in `snapshot_store.rs`

## Performance
- [ ] Implement zero-copy segment transfer between workers using shared memory or RDMA
- [ ] Add batch gRPC streaming for heartbeat and progress updates to reduce RPC overhead
- [ ] Optimize `shard_map.rs` consistent hashing with virtual nodes for better load distribution
- [ ] Profile and optimize `consensus.rs` Raft log replication latency
- [ ] Add connection multiplexing in `message_bus.rs` to reduce TCP connection overhead

## Testing
- [ ] Add integration tests simulating multi-worker cluster with tokio test utilities
- [ ] Test fault tolerance: kill workers mid-job and verify retry/reassignment
- [ ] Test `leader_election.rs` with simulated network partitions
- [ ] Add chaos testing for `replication.rs` with random message drops
- [ ] Test `work_stealing.rs` load balancing with heterogeneous worker speeds
- [ ] Add benchmarks for `task_queue.rs` throughput under high concurrency

## Documentation
- [ ] Document the gRPC service API (protobuf definitions) with usage examples
- [ ] Add deployment architecture diagram showing coordinator, workers, and discovery
- [ ] Document the Raft consensus protocol usage in `raft_primitives.rs`
