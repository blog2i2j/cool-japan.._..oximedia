# oximedia-cli

Command-line interface for the OxiMedia multimedia framework.

## Overview

`oximedia-cli` provides a command-line tool for working with media files using only royalty-free codecs.

## Installation

```bash
cargo install --path oximedia-cli
```

Or build from source:

```bash
cargo build --release -p oximedia-cli
```

## Commands

### Probe

Analyze media files and display format information:

```bash
# Basic probe
oximedia probe -i video.mkv

# Verbose output with technical details
oximedia probe -i video.mkv -V

# Show stream information
oximedia probe -i video.mkv --streams
```

### Info

Display supported formats and codecs:

```bash
oximedia info
```

Output:
```
Supported Containers:
  ✓ Matroska (.mkv)
  ✓ WebM (.webm)
  ✓ Ogg (.ogg, .opus, .oga)
  ✓ FLAC (.flac)
  ✓ WAV (.wav)

Supported Video Codecs (Green List):
  ✓ AV1 (Primary codec, best compression)
  ✓ VP9 (Excellent quality/size ratio)
  ✓ VP8 (Legacy support)
  ✓ Theora (Legacy support)

Supported Audio Codecs (Green List):
  ✓ Opus (Primary codec, versatile)
  ✓ Vorbis (High quality)
  ✓ FLAC (Lossless)
  ✓ PCM (Uncompressed)

Rejected Codecs (Patent-Encumbered):
  ✗ H.264/AVC
  ✗ H.265/HEVC
  ✗ AAC
  ✗ AC-3/E-AC-3
  ✗ DTS
```

### Transcode

Convert media files between formats:

```bash
# Basic transcode to VP9
oximedia transcode -i input.mkv -o output.webm --codec vp9

# With bitrate control
oximedia transcode -i input.mkv -o output.webm --codec vp9 --bitrate 2M

# With quality control (CRF)
oximedia transcode -i input.mkv -o output.webm --codec av1 --crf 30

# With scaling
oximedia transcode -i input.mkv -o output.webm --scale 1280:720

# Two-pass encoding
oximedia transcode -i input.mkv -o output.webm --codec vp9 --two-pass

# Seek and duration
oximedia transcode -i input.mkv -o output.webm --ss 00:01:00 -t 30
```

### Extract

Extract frames from video:

```bash
# Extract all frames as PNG
oximedia extract video.mkv frames_%04d.png

# Extract first 100 frames
oximedia extract video.mkv frames_%04d.png -n 100

# Extract every 30th frame (1 fps from 30fps video)
oximedia extract video.mkv frames_%04d.png --every 30

# Extract as JPEG with quality
oximedia extract video.mkv frames_%04d.jpg --format jpg --quality 85

# Start from specific time
oximedia extract video.mkv frames_%04d.png --ss 00:05:00
```

### Batch

Process multiple files:

```bash
# Batch transcode with config file
oximedia batch input_dir/ output_dir/ config.toml

# Parallel processing with 4 jobs
oximedia batch input_dir/ output_dir/ config.toml -j 4

# Dry run (show what would be done)
oximedia batch input_dir/ output_dir/ config.toml --dry-run

# Continue on errors
oximedia batch input_dir/ output_dir/ config.toml --continue-on-error
```

## FFmpeg-Compatible Options

The CLI supports FFmpeg-style options for familiarity:

| OxiMedia | FFmpeg | Description |
|----------|--------|-------------|
| `-i` | `-i` | Input file |
| `-o` | (positional) | Output file |
| `--codec` | `-c:v` | Video codec |
| `--audio-codec` | `-c:a` | Audio codec |
| `--bitrate` | `-b:v` | Video bitrate |
| `--audio-bitrate` | `-b:a` | Audio bitrate |
| `--video-filter` | `-vf` | Video filter chain |
| `--ss` | `-ss` | Start time (seek) |
| `-t` | `-t` | Duration |
| `-r` | `-r` | Frame rate |
| `-y` | `-y` | Overwrite output |

## Global Options

| Option | Description |
|--------|-------------|
| `-v`, `--verbose` | Increase verbosity (can stack: -v, -vv, -vvv) |
| `-q`, `--quiet` | Suppress all output except errors |
| `--no-color` | Disable colored output |

## Examples

```bash
# Probe a file
oximedia probe -i video.mkv

# Convert MP4 to WebM (VP9 + Opus)
oximedia transcode -i video.mp4 -o video.webm --codec vp9 --audio-codec opus

# High quality AV1 encoding
oximedia transcode -i video.mkv -o video.webm --codec av1 --crf 20 --preset slow

# Extract thumbnails (1 per minute)
oximedia extract video.mkv thumb_%03d.jpg --every 1800 --format jpg

# Batch convert all MKV files to WebM
oximedia batch videos/ output/ convert.toml -j 8
```

## Config File Format (TOML)

For batch processing:

```toml
[output]
format = "webm"
video_codec = "vp9"
audio_codec = "opus"

[video]
bitrate = "2M"
crf = 30
preset = "medium"

[audio]
bitrate = "128k"

[processing]
scale = "1920:-1"
```

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Unsupported format |
| 5 | Patent-encumbered codec detected |

## Policy

- Only supports patent-free codecs (Green List)
- Rejects patent-encumbered codecs with clear error messages
- Apache 2.0 license

## License

Apache-2.0

Version: 0.1.4 — 2026-04-20 — 19 tests
