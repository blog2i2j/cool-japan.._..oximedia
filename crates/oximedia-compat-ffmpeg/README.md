# oximedia-compat-ffmpeg

FFmpeg CLI argument compatibility layer for [OxiMedia](https://github.com/cool-japan/oximedia).

## Overview

This crate parses FFmpeg-style command-line arguments and translates them into native OxiMedia
transcode operations. It provides a familiar interface for users migrating from FFmpeg, while
automatically mapping patent-encumbered codecs (e.g., `libx264`, `aac`) to OxiMedia's patent-free
alternatives (AV1, Opus).

### Key Components

- **Argument Parser** -- Parses FFmpeg-style flags (`-i`, `-c:v`, `-c:a`, `-crf`, `-vf`, `-af`, `-map`, etc.)
- **Codec Map** -- 80+ codec name mappings with semantic categories (direct match, patent-substituted, copy)
- **Filter Lexer** -- Parses FFmpeg filter graph syntax (`-vf`, `-af`, `-filter_complex`)
- **Stream Specifier** -- Handles FFmpeg stream selection notation (`0:v`, `0:a:1`, etc.)
- **Translator** -- Converts parsed arguments into `TranscodeJob` structs for OxiMedia's transcoding engine
- **Diagnostics** -- Structured warnings and errors formatted in FFmpeg style

## Usage

```rust
use oximedia_compat_ffmpeg::parse_and_translate;

let args: Vec<String> = vec![
    "-i".into(), "input.mkv".into(),
    "-c:v".into(), "libaom-av1".into(),
    "-crf".into(), "28".into(),
    "-c:a".into(), "libopus".into(),
    "output.webm".into(),
];

let result = parse_and_translate(&args);
for diag in &result.diagnostics {
    eprintln!("{}", diag.format_ffmpeg_style("oximedia-ff"));
}
```

## License

Apache-2.0

Copyright COOLJAPAN OU (Team Kitasan)
