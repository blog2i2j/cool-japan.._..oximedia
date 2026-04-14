# oximedia-server

![Status: Stable](https://img.shields.io/badge/status-stable-green)

Production-ready RESTful media server for OxiMedia with comprehensive media management, streaming, and transcoding capabilities.

Part of the [oximedia](https://github.com/cool-japan/oximedia) workspace — a comprehensive pure-Rust media processing framework.

Version: 0.1.3 — 2026-04-15

## Features

### RESTful API
- **File Upload**: Single and multi-part upload with chunked transfer
- **Media Management**: CRUD operations for media files
- **Transcoding**: Submit jobs for format conversion (patent-free codecs only)
- **Metadata**: Automatic extraction and flexible key-value storage
- **Thumbnails**: Automatic generation with sprite sheet support
- **Preview Generation**: Create preview clips for media files

### Media Library
- **SQLite Backend**: Lightweight, embedded database
- **Media Indexing**: Automatic scanning and cataloging
- **Full-Text Search**: Fast media search by filename and metadata
- **Collections**: Create playlists and organize media
- **Metadata Extraction**: Duration, codecs, resolution, bitrate

### Adaptive Bitrate Streaming
- **HLS (HTTP Live Streaming)**: Master/media playlists with multiple variants
- **DASH (Dynamic Adaptive Streaming)**: MPD manifests with init/media segments
- **Progressive Download**: Standard HTTP streaming
- **Range Requests**: Seek support for all streaming modes
- **Bandwidth Throttling**: Configurable rate limiting per stream
- **DVR**: DVR buffering and recording support
- **CDN Integration**: CDN-aware streaming configuration
- **RTMP Ingest**: RTMP stream ingest support

### Authentication & Security
- **JWT Tokens**: Stateless authentication with configurable expiration
- **API Keys**: Long-lived keys for programmatic access
- **Password Hashing**: Argon2 for secure password storage
- **Role-Based Access**: Admin, user, and guest roles
- **Rate Limiting**: Per-user request throttling
- **Audit Trail**: Comprehensive security audit logging
- **Access Log**: Structured access log with CLF formatting
- **Circuit Breaker**: Circuit breaker for downstream service protection

### Advanced Features
- **WebSocket**: Real-time updates for job progress and events
- **Multi-Part Upload**: Resumable uploads for large files
- **Thumbnail Sprites**: Seek preview images in a grid layout
- **Health Checks**: Readiness and liveness endpoints
- **CORS Support**: Configurable cross-origin resource sharing
- **API Versioning**: Semantic version registry with compatibility tracking
- **Connection Pooling**: Connection pool with statistics and monitoring
- **Response Caching**: TTL-based response cache with LRU eviction
- **Metrics**: Server metrics collection and reporting

## Quick Start

```rust
use oximedia_server::{Server, Config};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize configuration
    let config = Config::from_env();

    // Create server
    let server = Server::new(config).await?;

    // Start serving
    server.serve("0.0.0.0:8080").await?;

    Ok(())
}
```

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` — Register new user
- `POST /api/v1/auth/login` — Login and get JWT token
- `POST /api/v1/auth/refresh` — Refresh JWT token
- `POST /api/v1/auth/logout` — Logout

### User Management
- `GET /api/v1/users/me` — Get current user profile
- `POST /api/v1/users/me` — Update user profile
- `POST /api/v1/users/me/password` — Change password
- `GET /api/v1/users/me/api-keys` — List API keys

### Media Management
- `POST /api/v1/media/upload` — Upload media file
- `GET /api/v1/media` — List media files
- `GET /api/v1/media/:media_id` — Get media details
- `DELETE /api/v1/media/:media_id` — Delete media
- `GET /api/v1/media/:media_id/thumbnail` — Get thumbnail

### Streaming
- `GET /api/v1/stream/:media_id/master.m3u8` — HLS master playlist
- `GET /api/v1/stream/:media_id/manifest.mpd` — DASH manifest
- `GET /api/v1/stream/:media_id/progressive` — Progressive download

### Health Checks
- `GET /health` — Health check
- `GET /ready` — Readiness check

## Configuration

Environment variables:
- `DATABASE_URL` — Database connection string (default: `sqlite:oximedia.db`)
- `MEDIA_DIR` — Media storage directory (default: `media`)
- `JWT_SECRET` — JWT signing secret (auto-generated if not set)
- `MAX_UPLOAD_SIZE` — Maximum upload size in bytes (default: 5GB)
- `MAX_CONCURRENT_JOBS` — Concurrent transcoding jobs (default: 4)
- `RATE_LIMIT_PER_MINUTE` — Requests per minute (default: 60)

## Patent-Free Codecs Only

**Supported Codecs:**
- Video: AV1, VP9, VP8, Theora
- Audio: Opus, Vorbis, FLAC, PCM
- Containers: WebM, Ogg, Matroska

## API Overview

- `Server` — Main server instance with axum router
- `Config` — Server configuration with environment variable support
- `AppState` — Shared application state: database, auth, library
- `JobQueue` — In-memory transcoding job queue
- `StreamingServer` / `StreamingServerConfig` — Dedicated streaming server
- `ServerError` / `ServerResult` — Error and result types
- Modules: `access_log`, `api`, `api_versioning`, `audit_trail`, `auth`, `auth_middleware`, `cache`, `cdn`, `circuit_breaker`, `config`, `config_loader`, `conn_stats`, `connection_pool`, `dash`, `db`, `dvr`, `error`, `handlers`, `health_monitor`, `hls`, `library`, `metrics`, `middleware`, `models`, `rate_limit`, `record`, `request_log`, `request_router`, `request_validator`, `response_cache`, `rtmp`, `session`, `streaming`, `streaming_server`, `transcode`, `upload`, `websocket`, `ws_handler`

## License

Apache-2.0 — Copyright 2024-2026 COOLJAPAN OU (Team Kitasan)
