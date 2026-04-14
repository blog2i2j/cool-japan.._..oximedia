# oximedia-normalize

**Status: [Stable]** | Version: 0.1.3 | Updated: 2026-04-15

Professional broadcast loudness normalization for OxiMedia.

Part of the [oximedia](https://github.com/cool-japan/oximedia) workspace — a comprehensive pure-Rust media processing framework.

## Overview

`oximedia-normalize` provides comprehensive loudness normalization compliant with all major broadcast and streaming standards, including EBU R128, ATSC A/85, and streaming platform requirements.

## Features

### Broadcast Standards Support

- **EBU R128** - European Broadcasting Union (-23 LUFS ±1 LU, -1 dBTP max)
- **ATSC A/85** - US broadcast standard (-24 LKFS ±2 dB, -2 dBTP max)
- **Streaming Platforms** - Spotify, YouTube, Apple Music, Tidal, Netflix, etc.
- **ReplayGain** - Album and track gain (reference 89 dB SPL)

### Processing Modes

- **Two-pass Normalization** - Analyze first, then apply precise gain
- **One-pass Normalization** - Real-time with lookahead buffer
- **Linear Gain** - Simple gain adjustment to target loudness
- **Dynamic Normalization** - DRC for consistent loudness across content
- **True Peak Limiting** - Brick-wall limiter preventing clipping

### Advanced Features

- **Multi-pass Processing** - Iterative refinement for high-precision normalization
- **Batch Processing** - Process entire directories of audio files
- **Real-time Processing** - Low-latency normalization for live applications
- **Metadata Writing** - ReplayGain, R128, iTunes Sound Check tags
- **Compliance Checking** - Verify against all broadcast standards
- **Auto Gain Control (AGC)** - Automatic gain control for live content
- **DC Offset Removal** - Remove DC bias from audio
- **Dialogue Normalization** - Dialogue-specific normalization
- **Spectral Balance** - Frequency-aware normalization
- **Stem Loudness** - Multi-stem loudness management
- **Stereo Width** - Stereo width processing
- **Sidechain Processing** - Sidechain-based normalization
- **Voice Activity Detection** - VAD-aware normalization
- **Phase Correction** - Phase correction during normalization

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
oximedia-normalize = "0.1.3"
```

## Quick Start

### Two-pass Normalization

```rust
use oximedia_normalize::{Normalizer, NormalizerConfig};
use oximedia_metering::Standard;

// Configure for EBU R128 normalization
let config = NormalizerConfig::new(Standard::EbuR128, 48000.0, 2);
let mut normalizer = Normalizer::new(config)?;

// Pass 1: Analyze
normalizer.analyze_f32(audio_samples);
let analysis = normalizer.get_analysis();
println!("Current: {:.1} LUFS, Target: {:.1} LUFS",
         analysis.integrated_lufs,
         analysis.target_lufs);

// Pass 2: Normalize
let mut output = vec![0.0f32; audio_samples.len()];
normalizer.process_f32(audio_samples, &mut output)?;
```

### Real-time Normalization

```rust
use oximedia_normalize::{RealtimeNormalizer, RealtimeConfig};
use oximedia_metering::Standard;

let config = RealtimeConfig::new(Standard::Spotify, 48000.0, 2);
let mut normalizer = RealtimeNormalizer::new(config)?;

// Process audio chunks
loop {
    let chunk = get_next_audio_chunk();
    let mut output = vec![0.0f32; chunk.len()];
    normalizer.process_chunk(&chunk, &mut output)?;
    send_to_output(&output);
}
```

### Batch Processing

```rust
use oximedia_normalize::{BatchProcessor, BatchConfig};
use oximedia_metering::Standard;

let config = BatchConfig::new(Standard::Spotify);
let processor = BatchProcessor::new(config);

// Process entire directory
let results = processor.process_directory(
    Path::new("input/"),
    Path::new("output/")
)?;

// Generate report
let report = BatchProcessor::generate_report(&results);
println!("{}", report.format());
```

## Architecture

### Core Modules

- **`analyzer`** - Two-pass loudness analysis using ITU-R BS.1770-4
- **`processor`** - Normalization processing with gain, limiting, and DRC
- **`limiter`, `limiter_chain`** - True peak limiter with lookahead buffering
- **`drc`** - Broadcast-quality dynamic range compressor
- **`targets`, `loudness_target`** - Target loudness standards and presets
- **`replaygain`** - ReplayGain calculation and tagging
- **`metadata`** - Loudness metadata writing (ID3v2, Vorbis, APE, MP4)
- **`batch`** - Batch file processing
- **`realtime`** - Real-time normalization with low latency
- **`multipass`** - Multi-pass processing controller
- **`agc`, `auto_gain`** - Automatic gain control
- **`broadcast_standard`** - Broadcast standard definitions
- **`compliance_checker`** - Standards compliance verification
- **`dc_offset`** - DC offset removal
- **`dialogue_norm`** - Dialogue normalization
- **`dynamic_range`** - Dynamic range processing
- **`ebu_r128`** - EBU R128 implementation
- **`fade_normalization`** - Fade-aware normalization
- **`format_loudness`** - Format-specific loudness settings
- **`gain_schedule`** - Gain scheduling
- **`loudness_history`** - Loudness history tracking
- **`metering_bridge`** - Integration with oximedia-metering
- **`multi_channel_loud`** - Multi-channel loudness
- **`noise_profile`** - Noise profile analysis
- **`normalize_report`** - Normalization reporting
- **`peak_limit`** - Peak limiting
- **`phase_correction`** - Phase correction
- **`sidechain`** - Sidechain processing
- **`spectral_balance`** - Spectral balance normalization
- **`stem_loudness`** - Stem-level loudness management
- **`stereo_width`** - Stereo width processing
- **`target_loudness`** - Target loudness configuration
- **`true_peak_limiter`** - True peak brick-wall limiter
- **`voice_activity`** - Voice activity detection

### Processing Pipeline

```
Input Audio
    ↓
K-weighting Filter (ITU-R BS.1770-4)
    ↓
Loudness Analysis (Gating, Integration)
    ↓
Gain Calculation
    ↓
Gain Application
    ↓
Dynamic Range Compression (optional)
    ↓
True Peak Limiting (optional)
    ↓
Output Audio
```

## Loudness Targets

### Broadcast

| Standard | Target LUFS | Max Peak | Tolerance |
|----------|-------------|----------|-----------|
| EBU R128 | -23.0 | -1.0 dBTP | ±1.0 LU |
| ATSC A/85 | -24.0 | -2.0 dBTP | ±2.0 dB |
| BBC iPlayer | -23.0 | -1.0 dBTP | ±1.0 LU |

### Streaming Platforms

| Platform | Target LUFS | Max Peak |
|----------|-------------|----------|
| Spotify | -14.0 | -1.0 dBTP |
| YouTube | -14.0 | -1.0 dBTP |
| Apple Music | -16.0 | -1.0 dBTP |
| Tidal | -14.0 | -1.0 dBTP |
| Netflix (Drama) | -27.0 | -2.0 dBTP |
| Amazon Prime | -24.0 | -2.0 dBTP |

## Technical Details

### Loudness Measurement

- **ITU-R BS.1770-4** compliant K-weighting filter
- **Absolute gate** at -70 LKFS
- **Relative gate** at -10 LU below ungated loudness
- **True peak detection** via 4x oversampling with sinc interpolation

### True Peak Limiting

- Lookahead buffer (configurable, default 5-10ms)
- 4x oversampling for accurate peak detection
- Attack/release envelope shaping
- Zero artifacts brick-wall limiting

### Dynamic Range Compression

- Configurable threshold, ratio, attack, release
- Soft knee for smooth compression
- Automatic makeup gain
- Broadcast-style envelope following

## License

Apache-2.0 — Copyright 2024-2026 COOLJAPAN OU (Team Kitasan)
