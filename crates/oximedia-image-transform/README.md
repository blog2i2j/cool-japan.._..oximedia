# oximedia-image-transform

![Version: 0.1.4](https://img.shields.io/badge/version-0.1.4-blue)
![Tests: 501](https://img.shields.io/badge/tests-501-brightgreen)
![Updated: 2026-04-20](https://img.shields.io/badge/updated-2026--04--20-blue)

Cloudflare Images-compatible URL image transformation for the [OxiMedia](https://github.com/cool-japan/oximedia) Sovereign Media Framework.

## Overview

`oximedia-image-transform` provides a complete, pure-Rust implementation of Cloudflare Images' URL-based image transformation API. Parse transformation URLs, negotiate output formats via HTTP Accept headers, build processing pipelines, and validate requests against security policies -- all without any C/Fortran dependencies.

## Features

- **URL Parsing** -- parse `/cdn-cgi/image/` paths, query strings, and comma-separated transform strings
- **Content Negotiation** -- Accept-header-based format selection (AVIF > WebP > JPEG/PNG)
- **Processing Pipeline** -- ordered step-based transformation (decode, trim, resize, rotate, color, sharpen/blur, border/pad, encode)
- **Security Validation** -- SSRF prevention, path traversal detection, dimension limits
- **Cache Key Generation** -- deterministic FNV-1a based cache keys

## Supported Parameters

| Parameter | Short | Description |
|-----------|-------|-------------|
| `width` | `w` | Target width in pixels |
| `height` | `h` | Target height in pixels |
| `quality` | `q` | Output quality (1-100) |
| `format` | `f` | Output format: `auto`, `avif`, `webp`, `jpeg`, `png`, `gif`, `baseline`, `json` |
| `fit` | -- | Resize mode: `scale-down`, `contain`, `cover`, `crop`, `pad`, `fill` |
| `gravity` | `g` | Crop anchor: `auto`, `center`, `top`, `bottom`, `left`, `right`, `face`, `0.5x0.5` |
| `sharpen` | -- | Sharpen amount (0.0-10.0) |
| `blur` | -- | Gaussian blur radius (0.0-250.0) |
| `brightness` | -- | Brightness adjustment (-1.0 to 1.0) |
| `contrast` | -- | Contrast adjustment (-1.0 to 1.0) |
| `gamma` | -- | Gamma correction (0.1-10.0) |
| `rotate` | -- | Rotation: `0`, `90`, `180`, `270`, `auto` |
| `dpr` | -- | Device pixel ratio (1.0-4.0) |
| `trim` | -- | Edge trimming in pixels |
| `background` | `bg` | Background color (CSS hex) |
| `border` | -- | Border: `width:color` or `t,r,b,l:color` |
| `padding` | `pad` | Fractional padding (0.0-1.0) |
| `metadata` | -- | Metadata handling: `keep`, `copyright`, `none` |
| `anim` | -- | Animate GIFs: `true` / `false` |
| `compression` | -- | Compression strategy: `fast`, `default`, `slow` |
| `onerror` | -- | Fallback URL on error |

## URL Formats

```text
# CDN path format (Cloudflare-compatible)
/cdn-cgi/image/width=800,height=600,format=auto/path/to/image.jpg

# Short aliases
/cdn-cgi/image/w=800,h=600,f=webp,fit=cover/photo.jpg

# Query string format
?width=800&height=600&quality=85&format=auto
```

## Usage

```rust
use oximedia_image_transform::parser::parse_cdn_url;
use oximedia_image_transform::transform::{OutputFormat, FitMode};
use oximedia_image_transform::negotiation::negotiate_format;
use oximedia_image_transform::security::{validate_request, SecurityConfig};

// Parse a Cloudflare-style URL
let req = parse_cdn_url("/cdn-cgi/image/w=800,f=auto,fit=cover/photo.jpg").unwrap();
assert_eq!(req.params.width, Some(800));
assert_eq!(req.params.fit, FitMode::Cover);

// Negotiate output format from Accept header
let format = negotiate_format(
    "image/avif,image/webp;q=0.9,image/jpeg;q=0.8",
    req.params.format,
);
assert_eq!(format, OutputFormat::Avif);

// Validate against security policy
let config = SecurityConfig::default();
assert!(validate_request(&req.source_path, &req.params, &config).is_ok());
```

## Security

Built-in protections against:

- **SSRF** -- detects private/reserved IPs (RFC 1918, RFC 4193, CGNAT RFC 6598, link-local, documentation ranges RFC 5737)
- **Path Traversal** -- blocks `../`, encoded variants (`%2e%2e`, `%2f`, `%5c`), null bytes, backslashes, tilde home references
- **Resource Exhaustion** -- configurable dimension limits (default 12000x12000), file size limits (100 MB input, 50 MB output)

## Processing Pipeline

The processor module builds an ordered pipeline from transform parameters:

1. Decode
2. Trim (edge removal)
3. Resize (bilinear interpolation)
4. Rotate (90/180/270/auto)
5. Color adjustments (brightness, contrast, gamma)
6. Sharpen (unsharp mask) / Blur (separable Gaussian)
7. Border / Padding
8. Background (alpha flattening)
9. Encode

## Architecture

| Module | Description |
|--------|-------------|
| `transform` | Strongly-typed parameter structs and enums |
| `parser` | URL parsing and cache key generation |
| `negotiation` | Accept header parsing and format selection |
| `processor` | Image processing pipeline and pixel operations |
| `security` | SSRF prevention and input validation |

## License

Apache-2.0

Copyright (c) COOLJAPAN OU (Team Kitasan)
