# oximedia-drm

![Status: Stable](https://img.shields.io/badge/status-stable-green)
![Version: 0.1.3](https://img.shields.io/badge/version-0.1.3-blue)

DRM (Digital Rights Management) and encryption support for OxiMedia streaming, implementing CENC and supporting Widevine, PlayReady, FairPlay, and W3C Clear Key.

Part of the [oximedia](https://github.com/cool-japan/oximedia) workspace — a comprehensive pure-Rust media processing framework.

## Features

- **CENC** — Common Encryption (ISO 23001-7) implementation
- **Widevine** — Google Widevine DRM (feature-gated)
- **PlayReady** — Microsoft PlayReady DRM (feature-gated)
- **FairPlay Streaming** — Apple FairPlay Streaming (feature-gated)
- **W3C Clear Key** — Open DRM for testing and open platforms (feature-gated)
- **PSSH** — Protection System Specific Header generation and parsing
- **Key Management** — Content key generation and lifecycle management
- **Key Rotation** — Scheduled and event-driven key rotation with schedules
- **License Server** — License request/response handling
- **License Chain** — License chaining for multi-level rights
- **Device Authentication** — Device registry and authentication
- **Entitlement** — Entitlement management and validation
- **Playback Policy** — Time-based, concurrent stream, and geo-fence policies
- **Playback Rules** — Fine-grained playback rule engine
- **Geo-fencing** — Territory-based access control
- **Offline Playback** — Download and offline license support
- **Output Control** — Output protection level enforcement
- **Watermarking** — Forensic watermark embedding
- **Audit Trail** — DRM usage audit logging
- **Analytics** — DRM analytics collection
- **Session Tokens** — JWT-based session management
- **Access Grants** — Granular access grant management
- **Multi-key** — Multi-key encryption for adaptive streaming
- **Compliance** — Robustness rule compliance checking

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oximedia-drm = "0.1.3"
# Enable specific DRM systems:
oximedia-drm = { version = "0.1.3", features = ["all-drm"] }
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `clearkey` | W3C Clear Key support (default) |
| `widevine` | Google Widevine DRM |
| `playready` | Microsoft PlayReady DRM |
| `fairplay` | Apple FairPlay Streaming |
| `all-drm` | All DRM systems |

## API Overview

**Core types:**
- `DrmSystem` — Widevine / PlayReady / FairPlay / ClearKey
- `DrmError`, `Result` — Error types

**Encryption modules:**
- `cenc` — Common Encryption implementation
- `content_key` — Content key generation and management
- `key_management` — Key lifecycle management
- `key_rotation` — Key rotation triggering
- `key_rotation_schedule` — Key rotation scheduling
- `pssh` — PSSH box generation and parsing
- `multi_key` — Multi-key encryption

**License and policy modules:**
- `license_server` — License server integration
- `license_chain` — License chaining
- `policy` — Access policy definitions
- `policy_engine` — Policy evaluation engine
- `playback_policy` — Playback-specific policies
- `playback_rules` — Fine-grained playback rules
- `entitlement` — Entitlement management
- `access_grant` — Access grant types

**Device and session modules:**
- `device_auth` — Device authentication
- `device_registry` — Device registry
- `token` — Token management
- `session_token` — Session token (JWT)

**Restriction modules:**
- `geo_fence` — Geographic access restriction
- `offline` — Offline playback licenses
- `output_control` — Output protection enforcement

**Tracking modules:**
- `watermark_embed` — Forensic watermark embedding
- `audit_trail` — Audit logging
- `analytics` — DRM analytics
- `compliance` — Compliance checking

**DRM-specific modules (feature-gated):**
- `clearkey` — W3C Clear Key implementation
- `widevine` — Google Widevine
- `playready` — Microsoft PlayReady
- `fairplay` — Apple FairPlay Streaming

## License

Apache-2.0 — Copyright 2024-2026 COOLJAPAN OU (Team Kitasan)
