# oximedia-proxy

**Status: [Stable]** | Version: 0.1.6 | Tests: 567 | Updated: 2026-04-26

Proxy and offline editing workflow system for OxiMedia. Provides comprehensive proxy workflow management including generation, linking, conforming, and complete offline-to-online pipeline support.

Part of the [oximedia](https://github.com/cool-japan/oximedia) workspace — a comprehensive pure-Rust media processing framework.

## Features

- **Proxy Generation** - Quarter, half, and full resolution proxies with multiple codec options
- **Batch Processing** - Generate proxies for multiple files simultaneously
- **Proxy Linking** - Link proxies to high-resolution originals with SQLite database
- **Link Verification** - Validate proxy-original relationships
- **EDL Conforming** - Conform from CMX 3600 and other EDL formats
- **XML Conforming** - Final Cut Pro XML and Premiere Pro XML support
- **Frame-accurate relinking** - Preserve exact frame accuracy during conform
- **Offline/Online Workflow** - Complete offline-to-online-to-delivery pipeline
- **Smart Caching** - Intelligent proxy cache management with cleanup policies
- **Timecode Preservation** - Maintain accurate timecode across workflow
- **Metadata Sync** - Synchronize metadata between proxy and original
- **Sidecar Files** - Checksum and processing record management
- **Proxy Registry** - Central proxy registry with extensions
- **Proxy Scheduler** - Scheduled proxy generation
- **Proxy Pipeline** - Multi-stage proxy processing pipeline
- **Proxy Pool** - Proxy resource pool management
- **Proxy Quality** - Quality assessment for proxies
- **Proxy Manifest** - Proxy manifest generation
- **Proxy Index** - Proxy search index
- **Proxy Format** - Format compatibility checking
- **Proxy Aging** - Proxy lifecycle and aging management
- **Transcode Queue** - Priority-based transcode queue
- **Bandwidth Management** - Proxy bandwidth optimization
- **Validation** - Proxy validation and integrity checking
- **Format Compatibility** - Cross-format proxy compatibility
- **Resolution Management** - Multi-resolution proxy management
- **Offline Proxy** - Offline-specific proxy handling
- **Relink Proxy** - Proxy relinking workflows

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oximedia-proxy = "0.1.6"
```

```rust
use oximedia_proxy::{ProxyGenerator, ProxyPreset, ProxyLinkManager, ConformEngine, OfflineWorkflow};

// Generate a quarter-resolution proxy
let generator = ProxyGenerator::new();
let proxy_path = generator
    .generate("original.mov", "proxy.mp4", ProxyPreset::QuarterResH264)
    .await?;

// Link proxy to original
let mut manager = ProxyLinkManager::new("links.db").await?;
manager.link_proxy("proxy.mp4", "original.mov").await?;

// Conform from EDL
let engine = ConformEngine::new("links.db").await?;
let conformed = engine.conform_from_edl("edit.edl", "output.mov").await?;
```

## API Overview

**Core types:**
- `ProxyGenerator` / `ProxyPreset` — Proxy generation with quality presets
- `ProxyLinkManager` / `ProxyLink` — Database-backed proxy-original linking
- `ConformEngine` / `EdlConformer` — EDL/XML conforming
- `OfflineWorkflow` / `OnlineWorkflow` / `RoundtripWorkflow` — Complete workflow management
- `CacheManager` / `CacheStrategy` — Proxy cache management
- `ResolutionManager` / `ResolutionSwitcher` — Multi-resolution management
- `Quality` — Low/Medium/High quality presets with bitrate recommendations

**Modules:**
- `cache` — Cache management
- `conform` — EDL/XML conforming (EDL, mapper, timeline, XML)
- `examples` — Usage examples
- `format_compat` — Format compatibility
- `generate` — Proxy generation (encoder, optimizer, presets)
- `generation` — Generation pipeline
- `link` — Proxy linking (database, manager, statistics)
- `linking` — Linking utilities
- `media_link` — Media file linking
- `metadata` — Metadata synchronization
- `offline_edit`, `offline_proxy` — Offline editing support
- `proxy_aging` — Proxy lifecycle management
- `proxy_bandwidth` — Bandwidth optimization
- `proxy_cache` — Cache management
- `proxy_compare` — Proxy comparison
- `proxy_fingerprint` — Proxy fingerprinting
- `proxy_format` — Format management
- `proxy_index` — Search index
- `proxy_manifest` — Manifest generation
- `proxy_pipeline` — Processing pipeline
- `proxy_pool` — Resource pool
- `proxy_quality` — Quality assessment
- `proxy_registry_ext` — Registry extensions
- `proxy_scheduler` — Scheduled generation
- `proxy_status` — Status tracking
- `proxy_sync` — Synchronization
- `registry` — Central registry
- `relink_proxy` — Relinking workflows
- `render` — Render management
- `resolution` — Resolution management
- `sidecar` — Sidecar file management
- `smart_proxy` — Intelligent proxy selection
- `spec` — Proxy specifications
- `timecode` — Timecode verification
- `transcode_proxy`, `transcode_queue` — Transcoding
- `utils` — Utility functions
- `validation` — Validation (checker, validator)
- `workflow` — Workflow planning

## License

Apache-2.0 — Copyright 2024-2026 COOLJAPAN OU (Team Kitasan)
