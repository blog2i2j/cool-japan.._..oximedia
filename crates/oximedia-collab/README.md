# oximedia-collab

![Status: Stable](https://img.shields.io/badge/status-stable-green)

Real-time CRDT-based multi-user collaboration system for OxiMedia, supporting concurrent video editing with sub-second synchronization.

Part of the [oximedia](https://github.com/cool-japan/oximedia) workspace — a comprehensive pure-Rust media processing framework.

## Overview

`oximedia-collab` provides a comprehensive CRDT-based synchronization system supporting 10+ concurrent editors with sub-second latency. Built on Yjs (via `yrs`) for conflict-free merging, with WebSocket-based communication and offline-first architecture.

## Features

### CRDT Document Synchronization
- Yjs-based document synchronization via `yrs`
- Timeline operations: insert, delete, move clips and tracks
- Operational transformation with conflict resolution
- Version vector tracking for causality
- Lamport timestamps and vector clocks
- Three-way merge strategy
- Snapshot management for efficient sync
- Operation batching and causal order tracking

### Real-time Synchronization
- WebSocket-based communication protocol (tokio-tungstenite)
- Delta encoding and compression (gzip/lz4)
- Offline support with change queue management
- Reconnection strategies with exponential backoff
- Message batching and throttling
- Heartbeat/keep-alive mechanism
- Bandwidth monitoring and connection statistics

### Session Management
- Multi-user session coordination (up to 10 users per session)
- User presence tracking and active indicators
- Cursor and selection synchronization
- Session locking (who is editing what)
- Permission enforcement: Owner, Editor, Viewer roles
- Session metadata management and garbage collection

### Optimistic Locking
- Resource locking: clips, tracks, timeline, project
- Read/Write lock types
- Lock stealing with permission checks
- Timeout-based automatic release
- Deadlock detection and prevention
- RAII-style lock guards

### Shared History
- Per-user undo/redo stack
- Cross-user history and history branching
- Change attribution and history compaction
- History visualization (ASCII timeline, DOT graphs)
- Import/export functionality

### Awareness Protocol
- Yjs awareness implementation
- User state broadcasting
- Cursor position and selection range synchronization
- Viewport state sharing and user color assignment
- Ephemeral state management with heartbeat

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oximedia-collab = "0.1.3"
```

```rust
use oximedia_collab::{CollaborationServer, CollabConfig, User, UserRole};

#[tokio::main]
async fn main() {
    let config = CollabConfig::default();
    let server = CollaborationServer::new(config);

    // Create session
    let owner = User::new("Alice".to_string(), UserRole::Owner);
    let project_id = uuid::Uuid::new_v4();
    let session_id = server.create_session(project_id, owner).await?;

    // Join session
    let editor = User::new("Bob".to_string(), UserRole::Editor);
    server.join_session(session_id, editor).await?;

    // Start background tasks (GC, heartbeat)
    server.start_background_tasks().await;

    // ... use the session ...

    server.shutdown().await?;
}
```

## Configuration

```rust
CollabConfig {
    max_users_per_session: 10,
    lock_timeout_secs: 300,           // 5 minutes
    enable_compression: true,
    compression_threshold: 1024,       // 1KB
    history_limit: 1000,
    gc_interval_secs: 600,            // 10 minutes
    enable_offline: true,
    max_offline_queue: 10000,
}
```

## Architecture (34 source files, 698 public items)

| Module | Purpose |
|--------|---------|
| `crdt` | CRDT operations, merging, conflict resolution |
| `sync` | Network synchronization, WebSocket, reconnection |
| `lock` | Resource locking, deadlock prevention |
| `history` | Undo/redo, history management |
| `session` | Session coordination, user management |
| `awareness` | Presence tracking, cursor synchronization |
| `lib` | Public API, server management |

## Conflict Resolution Strategies

1. **Last-Write-Wins** — Timestamp-based resolution
2. **First-Write-Wins** — Original operation takes precedence
3. **User-ID-Wins** — Deterministic tiebreaker based on user ID
4. **Manual** — Requires explicit conflict resolution

## Performance

- **Latency**: Sub-second synchronization (typically <100ms)
- **Concurrency**: 10+ concurrent editors
- **Bandwidth**: Optimized with delta encoding and lz4/gzip compression
- **Memory**: Efficient with garbage collection and history limits

## Safety

- No unsafe code
- All shared state protected by `Arc<RwLock>`
- All errors handled via `Result` types

## License

Apache-2.0 — Copyright 2024-2026 COOLJAPAN OU (Team Kitasan)
