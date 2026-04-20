# OxiMedia WASM

WebAssembly bindings for OxiMedia - Patent-free multimedia processing in the browser.

## Features

- **Format Probing**: Detect container formats (WebM, Matroska, Ogg, FLAC, WAV, MP4)
- **Container Demuxing**: Extract compressed packets from media files
- **Zero-Copy**: Efficient buffer management using JavaScript `ArrayBuffer`
- **Patent-Free**: Only supports royalty-free codecs (AV1, VP9, VP8, Opus, Vorbis, FLAC)
- **Browser-Native**: No file system dependencies, works entirely in-memory

## Installation

### From npm

```bash
npm install @cooljapan/oximedia
```

Available as three packages:
- `@cooljapan/oximedia` (bundler — for webpack, vite, rollup, etc.)
- `@cooljapan/oximedia-web` (browser — `<script type="module">`)
- `@cooljapan/oximedia-node` (Node.js — `require()`)

### From Source

#### Prerequisites

- [Rust](https://rustup.rs/) (1.75 or later)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)

Install wasm-pack:

```bash
cargo install wasm-pack
```

### Building

Build for different targets:

```bash
# Build for all targets (web, node, bundler)
./build.sh

# Build for web only (development mode with debug symbols)
./build-dev.sh

# Build for specific target
wasm-pack build --target web --out-dir pkg
```

Available targets:
- `web`: For use in browsers via `<script type="module">`
- `nodejs`: For use in Node.js
- `bundler`: For use with webpack, rollup, parcel, etc.

## Usage

### In the Browser

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>OxiMedia WASM Demo</title>
</head>
<body>
    <input type="file" id="fileInput" accept="video/*,audio/*">
    <pre id="output"></pre>

    <script type="module">
        import init, * as oximedia from './pkg-web/oximedia_wasm.js';

        async function run() {
            // Initialize the WASM module
            await init();

            console.log('OxiMedia version:', oximedia.version());

            // Handle file input
            document.getElementById('fileInput').addEventListener('change', async (e) => {
                const file = e.target.files[0];
                const arrayBuffer = await file.arrayBuffer();
                const data = new Uint8Array(arrayBuffer);

                // Probe format
                try {
                    const result = oximedia.probe_format(data);
                    console.log('Format:', result.format());
                    console.log('Confidence:', result.confidence());
                    console.log('Description:', result.description());

                    // Create demuxer
                    const demuxer = new oximedia.WasmDemuxer(data);
                    const probe = demuxer.probe();

                    // Get streams
                    const streams = demuxer.streams();
                    let output = `Format: ${probe.format()}\n`;
                    output += `Streams: ${streams.length}\n\n`;

                    for (const stream of streams) {
                        output += `Stream ${stream.index()}:\n`;
                        output += `  Codec: ${stream.codec()}\n`;
                        output += `  Type: ${stream.media_type()}\n`;
                        const params = stream.codec_params();
                        if (params.has_video_params()) {
                            output += `  Resolution: ${params.width()}x${params.height()}\n`;
                        }
                        if (params.has_audio_params()) {
                            output += `  Sample Rate: ${params.sample_rate()} Hz\n`;
                            output += `  Channels: ${params.channels()}\n`;
                        }
                        output += '\n';
                    }

                    // Read some packets
                    let packetCount = 0;
                    while (packetCount < 10) {
                        const packet = demuxer.read_packet();
                        if (!packet) break;
                        output += `Packet ${packetCount}: `;
                        output += `stream=${packet.stream_index()}, `;
                        output += `size=${packet.size()} bytes, `;
                        output += `keyframe=${packet.is_keyframe()}\n`;
                        packetCount++;
                    }

                    document.getElementById('output').textContent = output;
                } catch (e) {
                    console.error('Error:', e);
                    document.getElementById('output').textContent = 'Error: ' + e;
                }
            });
        }

        run();
    </script>
</body>
</html>
```

### With a Bundler (webpack, vite, etc.)

```javascript
import init, * as oximedia from 'oximedia-wasm';

async function processMedia(data) {
    // Initialize WASM module
    await init();

    // Probe format
    const result = oximedia.probe_format(data);
    console.log('Detected format:', result.format());

    // Create demuxer
    const demuxer = new oximedia.WasmDemuxer(data);
    demuxer.probe();

    // Get streams
    const streams = demuxer.streams();
    console.log('Found', streams.length, 'streams');

    // Read packets
    let packet;
    while ((packet = demuxer.read_packet()) !== null) {
        console.log('Packet:', packet.stream_index(), packet.size());
    }
}
```

### In Node.js

```javascript
const fs = require('fs');
const { probe_format, WasmDemuxer } = require('./pkg-node/oximedia_wasm');

// Read file
const data = fs.readFileSync('video.webm');

// Probe format
const result = probe_format(data);
console.log('Format:', result.format());

// Demux
const demuxer = new WasmDemuxer(data);
demuxer.probe();

const streams = demuxer.streams();
console.log('Streams:', streams.length);
```

## API Reference

### Functions

#### `probe_format(data: Uint8Array): WasmProbeResult`

Detects the container format from raw bytes.

**Parameters:**
- `data`: At least the first 12 bytes of the file

**Returns:** `WasmProbeResult` with format and confidence

**Throws:** Exception if format cannot be detected

### Classes

#### `WasmDemuxer`

Synchronous demuxer for extracting packets from containers.

**Constructor:**
- `new WasmDemuxer(data: Uint8Array)`

**Methods:**
- `probe(): WasmProbeResult` - Detect format and parse headers
- `streams(): WasmStreamInfo[]` - Get stream information
- `read_packet(): WasmPacket | null` - Read next packet (null at EOF)
- `size(): number` - Total size in bytes
- `position(): number` - Current position in bytes
- `is_eof(): boolean` - Check if all packets have been read

#### `WasmProbeResult`

Result of format probing.

**Methods:**
- `format(): string` - Container format name
- `confidence(): number` - Confidence score (0.0 to 1.0)
- `description(): string` - Human-readable description
- `is_video_container(): boolean` - Check if video container
- `is_audio_only(): boolean` - Check if audio-only container

#### `WasmStreamInfo`

Information about a media stream.

**Methods:**
- `index(): number` - Stream index
- `codec(): string` - Codec name
- `media_type(): string` - Media type ("Video", "Audio", "Subtitle")
- `is_video(): boolean` - Check if video stream
- `is_audio(): boolean` - Check if audio stream
- `duration_seconds(): number | undefined` - Duration in seconds
- `timebase_num(): number` - Timebase numerator
- `timebase_den(): number` - Timebase denominator
- `codec_params(): WasmCodecParams` - Codec parameters
- `metadata(): WasmMetadata` - Stream metadata

#### `WasmPacket`

Compressed media packet.

**Methods:**
- `stream_index(): number` - Stream index
- `size(): number` - Packet size in bytes
- `data(): Uint8Array` - Packet data
- `is_keyframe(): boolean` - Check if keyframe
- `is_corrupt(): boolean` - Check if potentially corrupt
- `pts(): number` - Presentation timestamp
- `dts(): number | undefined` - Decode timestamp
- `duration(): number | undefined` - Packet duration

## Supported Formats

### Containers
- Matroska (.mkv)
- WebM (.webm)
- Ogg (.ogg, .opus, .oga)
- FLAC (.flac)
- WAV (.wav)
- MP4 (.mp4) - AV1/VP9 only

### Video Codecs
- AV1
- VP9
- VP8
- Theora

### Audio Codecs
- Opus
- Vorbis
- FLAC
- PCM

## License

Apache-2.0

Version: 0.1.4 — 2026-04-20 — 505 tests

## See Also

- [OxiMedia](https://github.com/cool-japan/oximedia) - The main Rust library
- [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/) - Rust/WASM interop
- [wasm-pack](https://rustwasm.github.io/wasm-pack/) - WASM build tool
