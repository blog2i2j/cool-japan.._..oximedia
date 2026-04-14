# oximedia-collab TODO

## Current Status
- 36 source files; CRDT-based real-time collaboration for multi-user video editing
- Key features: session management, CRDT sync, conflict resolution, edit locking (region/track/hierarchical), operational transformation, three-way merge, user presence, activity feed, audit trail, permissions, review links, notifications
- Dependencies: yrs (Yjs CRDT), tokio-tungstenite (WebSocket), dashmap, flate2/lz4 compression
- Supports up to 10+ concurrent editors with sub-second latency target

## Enhancements
- [x] Add role-based permission granularity beyond Owner/Editor/Viewer in `permission.rs` (e.g., Commenter, Reviewer)
- [x] Implement offline edit queue with automatic conflict resolution on reconnect in `sync.rs`
- [x] Add selective sync in `bandwidth_throttle.rs` to prioritize active timeline regions
- [x] Extend `conflict_resolve.rs` with visual diff presentation for conflicting edits
- [x] Improve `three_way_merge.rs` to handle overlapping timeline events with priority rules
- [x] Add session persistence/recovery in `session_manager.rs` for server restarts
- [x] Extend `audit_trail.rs` with export to structured log formats (JSON lines, OTLP)
- [x] Add rate limiting for sync operations in `sync.rs` to prevent flooding

## New Features
- [x] Implement real-time cursor/viewport sharing via `user_presence_map.rs` with smooth interpolation
- [x] Add comment threading with resolved/unresolved state in `comments.rs`
- [x] Implement session recording/playback for edit history replay
- [x] Add webhook notifications for external integrations in `notification.rs`
- [x] Implement collaborative markers/annotations with timestamp anchoring in `annotation.rs`
- [x] Add project branching/forking support in `snapshot_manager.rs`
- [x] Implement per-track locking granularity in `edit_lock.rs` with lock escalation
- [x] Add session analytics (edit frequency, active time, collaboration patterns) in `activity_feed.rs`

## Performance
- [x] Benchmark CRDT merge performance in `crdt.rs` with 10+ concurrent editors
- [x] Optimize `operation_log.rs` DAG traversal for large edit histories (>10K operations)
- [x] Add incremental state serialization in `changeset.rs` to reduce sync payload size
- [x] Profile WebSocket message throughput in `sync.rs` and optimize binary framing
- [x] Implement operation batching in `sync.rs` to reduce network round-trips

## Testing
- [x] Add concurrent editing stress test with simulated network partitions
- [x] Test `three_way_merge.rs` with complex overlapping timeline edits
- [x] Add property-based tests using `proptest` for CRDT convergence guarantees
- [x] Test `edit_lock.rs` deadlock detection under concurrent lock acquisition
- [x] Add latency measurement tests for sync round-trip under load
- [x] Test `snapshot_manager.rs` branch creation and fast-forward merge detection

## Documentation
- [ ] Document collaboration protocol and message format
- [ ] Add sequence diagrams for join/leave/sync flows
- [ ] Document CRDT data model and conflict resolution strategy
