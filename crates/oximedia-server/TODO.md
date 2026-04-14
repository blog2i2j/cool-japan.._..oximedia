# oximedia-server TODO

## Current Status
- 43 modules (15 subdirectory modules) providing a full-featured RESTful media server built on axum with SQLx/SQLite, JWT authentication (Argon2 password hashing), HLS/DASH adaptive streaming, progressive download, WebSocket real-time updates, multi-part upload, batch operations, CDN integration (AWS/Azure/GCS feature-gated), RTMP ingest, DVR recording, admin API, webhooks, Prometheus metrics, rate limiting, circuit breaker, audit trail
- Core types: Server, AppState, Config, JobQueue
- API routes: auth, users, media CRUD, transcoding jobs, collections, search, stats, admin, streaming, WebSocket, webhooks, metrics
- Dependencies: axum, tower, sqlx, jsonwebtoken, argon2, prometheus, reqwest, and many more

## Enhancements
- [x] Replace `Mutex<JobQueue>` with `tokio::sync::RwLock` for concurrent read access to job status
- [x] Add request ID propagation through all handlers for end-to-end request tracing
- [x] Extend `rate_limit` with per-user and per-endpoint configurable rate limits (not just global)
- [ ] Implement `circuit_breaker` integration with transcoding backend to prevent cascade failures
- [ ] Add `api_versioning` header-based content negotiation (Accept-Version) alongside URL-based versioning
- [ ] Extend `auth_middleware` with OAuth2/OIDC provider integration for SSO
- [ ] Add `response_cache` ETags and conditional GET (If-None-Match) for media metadata endpoints
- [x] Implement `health_monitor` deep health checks (database connectivity, storage availability, transcode worker status)

## New Features
- [x] Add `graphql` module with GraphQL API alongside REST for flexible media queries
- [ ] Implement `media_processing_pipeline` for chaining operations (upload -> analyze -> transcode -> notify)
- [x] Add `live_ingest` module supporting SRT protocol alongside existing RTMP
- [ ] Implement `thumbnail_strip` module generating filmstrip-style thumbnail sprites for video scrubbing
- [ ] Add `media_proxy` module for proxying media requests to external storage with caching
- [ ] Implement `quota_management` per-user storage and bandwidth quotas with enforcement
- [ ] Add `event_bus` internal pub/sub system for decoupling handlers from side effects (analytics, webhooks)
- [ ] Implement `background_tasks` module with persistent task queue for long-running operations
- [ ] Add `content_delivery` module with edge caching configuration for multi-region deployment
- [ ] Implement `api_gateway` rate limiting, throttling, and request routing for microservice architecture

## Performance
- [x] Add database connection pool size tuning in `Config` with sensible defaults based on CPU count
- [ ] Implement response streaming for large media file downloads instead of loading into memory
- [ ] Add HTTP/2 server push for related resources (thumbnail with metadata response)
- [ ] Implement lazy deserialization in `handlers` for large JSON request bodies
- [x] Add query result pagination with cursor-based pagination for stable ordering in `list_media`
- [ ] Optimize `streaming::handlers` segment serving with memory-mapped file I/O

## Testing
- [ ] Add integration tests for the full upload -> transcode -> stream workflow
- [ ] Test `auth` token refresh flow with expired and valid refresh tokens
- [ ] Add load tests for concurrent HLS segment requests (100+ simultaneous viewers)
- [ ] Test `batch_ops` with mixed success/failure operations and verify partial completion handling
- [ ] Add tests for `webhooks` delivery retry logic with simulated endpoint failures
- [ ] Test `WebSocket` handler with connection lifecycle (connect, subscribe, receive events, disconnect)

## Documentation
- [ ] Add OpenAPI/Swagger specification for all REST endpoints
- [ ] Document the deployment architecture (reverse proxy, database, storage, CDN)
- [ ] Add authentication flow diagrams for JWT, API key, and OAuth2 paths
- [ ] Document the streaming pipeline: upload -> segment -> playlist generation -> CDN distribution
- [ ] Add operational runbook for common maintenance tasks (vacuum, backup, migration)
