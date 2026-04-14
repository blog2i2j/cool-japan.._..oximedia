# oximedia-forensics

![Status: Stable](https://img.shields.io/badge/status-stable-green)
![Version: 0.1.3](https://img.shields.io/badge/version-0.1.3-blue)

Video and image forensics and tampering detection for OxiMedia, providing comprehensive tools for authenticity verification and forensic analysis.

Part of the [oximedia](https://github.com/cool-japan/oximedia) workspace — a comprehensive pure-Rust media processing framework.

## Features

- **Error Level Analysis (ELA)** — Detect image manipulation via compression error levels
- **Noise Pattern Analysis** — PRNU (Photo Response Non-Uniformity) sensor fingerprinting
- **Metadata Verification** — Verify EXIF, IPTC, and XMP metadata consistency
- **Copy-Move Detection** — Detect cloned regions within an image
- **Clone Detection** — Detect copy-pasted regions and duplicate content
- **Splicing Detection** — Detect image splicing and compositing
- **Steganalysis** — Detect hidden data in images
- **Compression History** — Analyze previous compression operations and JPEG quality
- **Shadow Analysis** — Shadow direction consistency analysis
- **Illumination Inconsistency** — Detect lighting discontinuities
- **Source Camera Identification** — Identify camera model from image fingerprint
- **Frequency Forensics** — DCT/FFT-based tampering detection
- **Geometric Analysis** — Perspective and distortion inconsistency
- **Format Forensics** — Container and codec format integrity analysis
- **Frame Forensics** — Video frame-level tampering detection
- **Time Forensics** — Timestamp and temporal metadata analysis
- **Blocking Artifacts** — Block artifact pattern analysis
- **Hash Registry** — Known-good hash registry for file integrity
- **Chain of Custody** — Provenance and custody tracking
- **Watermark Detection** — Detect embedded forensic watermarks
- **Authenticity Scoring** — Overall authenticity confidence scoring
- **Forensic Reporting** — Comprehensive forensic reports

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oximedia-forensics = "0.1.3"
# With computer vision features:
oximedia-forensics = { version = "0.1.3", features = ["cv"] }
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `cv` | Computer vision integration via oximedia-cv |

## API Overview

**Core types:**
- `ForensicsError`, `ForensicsResult` — Error types
- `ConfidenceLevel` — VeryLow / Low / Medium / High / VeryHigh

**ELA and noise modules:**
- `ela`, `ela_analysis` — Error Level Analysis
- `noise`, `noise_analysis` — Noise pattern and PRNU analysis
- `blocking` — Blocking artifact analysis

**Copy detection:**
- `copy_detect` — Copy-move detection
- `clone_detection` — Clone region detection
- `splicing` — Splicing/compositing detection

**Compression and format:**
- `compression` — JPEG artifact analysis
- `compression_history` — Multi-generation compression detection
- `format_forensics` — Container/codec integrity

**Metadata and timestamps:**
- `metadata` — Metadata extraction
- `metadata_forensics` — Metadata consistency verification
- `time_forensics` — Temporal metadata analysis

**Visual analysis:**
- `lighting` — Illumination inconsistency detection
- `shadow_analysis` — Shadow direction analysis
- `geometric` — Geometric inconsistency analysis
- `frequency_forensics` — DCT/FFT frequency analysis
- `pattern` — Pattern analysis

**Camera and sensor:**
- `source_camera` — Camera fingerprinting
- `fingerprint` — Image perceptual fingerprinting

**Steganography and watermarks:**
- `steganalysis` — Steganography detection
- `watermark_detect` — Watermark detection

**Video forensics:**
- `frame_forensics` — Per-frame video forensics
- `edit_history` — Video edit history analysis

**Provenance and reporting:**
- `authenticity` — Overall authenticity scoring
- `provenance` — Provenance tracking
- `chain_of_custody` — Custody chain management
- `file_integrity` — File integrity checking
- `hash_registry` — Known-good hash registry
- `report` — Forensic report generation
- `tampering` — Tampering summary

## License

Apache-2.0 — Copyright 2024-2026 COOLJAPAN OU (Team Kitasan)
