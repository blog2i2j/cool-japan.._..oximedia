//! WebAssembly bindings for `oximedia-convert`.
//!
//! Provides format detection from magic bytes, codec/format/profile listing,
//! pixel format conversion, and conversion validation -- all synchronous,
//! returning JSON strings for easy JavaScript consumption.

use wasm_bindgen::prelude::*;

use oximedia_convert::format_detector::{FormatDetector, MediaFormat};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Detect a `MediaFormat` from raw header bytes using the format detector.
fn detect_media_format(header: &[u8]) -> MediaFormat {
    let detector = FormatDetector::new();
    detector.detect_from_header(header)
}

// ---------------------------------------------------------------------------
// Public WASM API
// ---------------------------------------------------------------------------

/// Detect media format from raw header bytes (magic bytes).
///
/// Returns a JSON object:
/// ```json
/// {"format":"webm","mime":"video/webm","is_video":true,"is_audio":false,"is_image":false}
/// ```
#[wasm_bindgen]
pub fn wasm_detect_format(header_bytes: &[u8]) -> Result<String, JsValue> {
    if header_bytes.len() < 4 {
        return Err(crate::utils::js_err(
            "Need at least 4 bytes for format detection",
        ));
    }

    let format = detect_media_format(header_bytes);

    Ok(format!(
        r#"{{"format":"{}","mime":"{}","extension":"{}","is_video":{},"is_audio":{},"is_image":{}}}"#,
        format.extension(),
        format.mime_type(),
        format.extension(),
        format.is_video(),
        format.is_audio(),
        format.is_image(),
    ))
}

/// List all supported codecs (video and audio) as a JSON array.
///
/// Each element: `{"name":"av1","type":"video","patent_free":true,"description":"..."}`
#[wasm_bindgen]
pub fn wasm_list_convert_codecs() -> Result<String, JsValue> {
    let codecs = [
        ("av1", "video", true, "AV1 - AOMedia Video 1 (royalty-free)"),
        ("vp9", "video", true, "VP9 - Google VP9 (royalty-free)"),
        ("vp8", "video", true, "VP8 - Google VP8 (royalty-free)"),
        ("opus", "audio", true, "Opus - IETF Opus (royalty-free)"),
        (
            "vorbis",
            "audio",
            true,
            "Vorbis - Xiph.org Vorbis (royalty-free)",
        ),
        ("flac", "audio", true, "FLAC - Free Lossless Audio Codec"),
        ("pcm", "audio", true, "PCM - Uncompressed audio"),
    ];

    let entries: Vec<String> = codecs
        .iter()
        .map(|(name, ctype, patent_free, desc)| {
            format!(
                r#"{{"name":"{}","type":"{}","patent_free":{},"description":"{}"}}"#,
                name, ctype, patent_free, desc,
            )
        })
        .collect();

    Ok(format!("[{}]", entries.join(",")))
}

/// List all supported container formats as a JSON array.
///
/// Each element: `{"name":"webm","extension":"webm","mime":"video/webm","type":"video"}`
#[wasm_bindgen]
pub fn wasm_list_formats() -> Result<String, JsValue> {
    let formats = [
        ("WebM", "webm", "video/webm", "video"),
        ("Matroska", "mkv", "video/x-matroska", "video"),
        ("Ogg", "ogg", "audio/ogg", "audio/video"),
        ("WAV", "wav", "audio/wav", "audio"),
        ("FLAC", "flac", "audio/flac", "audio"),
        ("MP4", "mp4", "video/mp4", "video"),
    ];

    let entries: Vec<String> = formats
        .iter()
        .map(|(name, ext, mime, ftype)| {
            format!(
                r#"{{"name":"{}","extension":"{}","mime":"{}","type":"{}"}}"#,
                name, ext, mime, ftype,
            )
        })
        .collect();

    Ok(format!("[{}]", entries.join(",")))
}

/// List available conversion profiles as a JSON array.
///
/// Each element: `{"name":"Web Optimized","id":"web_optimized","description":"..."}`
#[wasm_bindgen]
pub fn wasm_list_profiles() -> Result<String, JsValue> {
    let profiles = [
        (
            "web_optimized",
            "Web Optimized",
            "MP4 optimized for web playback",
        ),
        (
            "streaming",
            "Streaming",
            "HLS/DASH adaptive streaming variants",
        ),
        (
            "archive",
            "Archive",
            "Lossless format for long-term preservation",
        ),
        ("email", "Email", "Highly compressed for email sharing"),
        ("mobile", "Mobile", "Optimized for mobile devices"),
        ("youtube", "YouTube", "Optimized for YouTube upload"),
        ("instagram", "Instagram", "Instagram-compliant format"),
        ("tiktok", "TikTok", "TikTok-compliant vertical format"),
        ("broadcast", "Broadcast", "Broadcast-compliant format"),
        ("audio_mp3", "Audio MP3", "Extract audio as MP3"),
        ("audio_flac", "Audio FLAC", "Extract audio as lossless FLAC"),
        ("audio_aac", "Audio AAC", "Extract audio as AAC"),
    ];

    let entries: Vec<String> = profiles
        .iter()
        .map(|(id, name, desc)| {
            format!(
                r#"{{"id":"{}","name":"{}","description":"{}"}}"#,
                id, name, desc,
            )
        })
        .collect();

    Ok(format!("[{}]", entries.join(",")))
}

/// Convert between simple pixel formats in-memory.
///
/// Supported format strings: "rgb24", "rgba32", "bgr24", "bgra32", "gray8".
///
/// Returns the converted pixel buffer.
#[wasm_bindgen]
pub fn wasm_pixel_convert(
    data: &[u8],
    width: u32,
    height: u32,
    from_format: &str,
    to_format: &str,
) -> Result<Vec<u8>, JsValue> {
    let pixel_count = (width as usize) * (height as usize);
    if pixel_count == 0 {
        return Err(crate::utils::js_err("Width and height must be > 0"));
    }

    let from = from_format.to_lowercase();
    let to = to_format.to_lowercase();

    if from == to {
        return Ok(data.to_vec());
    }

    // Validate input buffer size.
    let from_bpp = bytes_per_pixel(&from)?;
    let expected_size = pixel_count * from_bpp;
    if data.len() < expected_size {
        return Err(crate::utils::js_err(&format!(
            "Input buffer too small: need {expected_size} bytes for {width}x{height} {from_format}, got {}",
            data.len()
        )));
    }

    // Perform conversion.
    match (from.as_str(), to.as_str()) {
        // RGB <-> RGBA
        ("rgb24", "rgba32") => Ok(rgb_to_rgba(data, pixel_count)),
        ("rgba32", "rgb24") => Ok(rgba_to_rgb(data, pixel_count)),
        // BGR <-> RGB
        ("bgr24", "rgb24") | ("rgb24", "bgr24") => {
            let mut out = data[..pixel_count * 3].to_vec();
            swap_channels_3(&mut out, pixel_count);
            Ok(out)
        }
        // BGR <-> RGBA
        ("bgr24", "rgba32") => {
            let mut rgb = data[..pixel_count * 3].to_vec();
            swap_channels_3(&mut rgb, pixel_count);
            Ok(rgb_to_rgba(&rgb, pixel_count))
        }
        ("rgba32", "bgr24") => {
            let mut rgb = rgba_to_rgb(data, pixel_count);
            swap_channels_3(&mut rgb, pixel_count);
            Ok(rgb)
        }
        // BGRA <-> RGBA
        ("bgra32", "rgba32") | ("rgba32", "bgra32") => {
            let mut out = data[..pixel_count * 4].to_vec();
            swap_channels_4(&mut out, pixel_count);
            Ok(out)
        }
        // BGR <-> BGRA
        ("bgr24", "bgra32") => Ok(rgb_to_rgba(data, pixel_count)),
        ("bgra32", "bgr24") => Ok(rgba_to_rgb(data, pixel_count)),
        // RGB <-> BGRA
        ("rgb24", "bgra32") => {
            let rgba = rgb_to_rgba(data, pixel_count);
            let mut out = rgba;
            swap_channels_4(&mut out, pixel_count);
            Ok(out)
        }
        ("bgra32", "rgb24") => {
            let mut rgba = data[..pixel_count * 4].to_vec();
            swap_channels_4(&mut rgba, pixel_count);
            Ok(rgba_to_rgb(&rgba, pixel_count))
        }
        // To grayscale
        ("rgb24", "gray8") => Ok(rgb_to_gray(data, pixel_count)),
        ("bgr24", "gray8") => {
            let mut rgb = data[..pixel_count * 3].to_vec();
            swap_channels_3(&mut rgb, pixel_count);
            Ok(rgb_to_gray(&rgb, pixel_count))
        }
        ("rgba32", "gray8") => {
            let rgb = rgba_to_rgb(data, pixel_count);
            Ok(rgb_to_gray(&rgb, pixel_count))
        }
        ("bgra32", "gray8") => {
            let mut rgba = data[..pixel_count * 4].to_vec();
            swap_channels_4(&mut rgba, pixel_count);
            let rgb = rgba_to_rgb(&rgba, pixel_count);
            Ok(rgb_to_gray(&rgb, pixel_count))
        }
        // From grayscale
        ("gray8", "rgb24") | ("gray8", "bgr24") => Ok(gray_to_rgb(data, pixel_count)),
        ("gray8", "rgba32") => {
            let rgb = gray_to_rgb(data, pixel_count);
            Ok(rgb_to_rgba(&rgb, pixel_count))
        }
        ("gray8", "bgra32") => {
            let rgb = gray_to_rgb(data, pixel_count);
            let mut rgba = rgb_to_rgba(&rgb, pixel_count);
            swap_channels_4(&mut rgba, pixel_count);
            Ok(rgba)
        }
        _ => Err(crate::utils::js_err(&format!(
            "Unsupported conversion: '{from_format}' -> '{to_format}'. Supported: rgb24, rgba32, bgr24, bgra32, gray8"
        ))),
    }
}

/// Validate whether a conversion between the given formats and codecs is
/// supported.
///
/// Returns JSON: `{"supported":true,"warnings":[],"recommended_video":"av1","recommended_audio":"opus"}`
#[wasm_bindgen]
pub fn wasm_validate_conversion(
    input_format: &str,
    output_format: &str,
    video_codec: &str,
    audio_codec: &str,
) -> Result<String, JsValue> {
    let mut warnings: Vec<String> = Vec::new();
    let mut supported = true;

    // Validate output container.
    let valid_containers = ["webm", "mkv", "ogg", "wav", "flac", "mp4"];
    let out_lower = output_format.to_lowercase();
    if !valid_containers.contains(&out_lower.as_str()) {
        warnings.push(format!(
            "Output format '{output_format}' is not directly supported"
        ));
        supported = false;
    }

    // Validate video codec.
    let valid_video = ["av1", "vp9", "vp8", "copy", "none", ""];
    let vc_lower = video_codec.to_lowercase();
    if !valid_video.contains(&vc_lower.as_str()) {
        warnings.push(format!(
            "Video codec '{video_codec}' is patented or not supported. Use av1, vp9, or vp8"
        ));
        supported = false;
    }

    // Validate audio codec.
    let valid_audio = ["opus", "vorbis", "flac", "pcm", "copy", "none", ""];
    let ac_lower = audio_codec.to_lowercase();
    if !valid_audio.contains(&ac_lower.as_str()) {
        warnings.push(format!(
            "Audio codec '{audio_codec}' is patented or not supported. Use opus, vorbis, flac, or pcm"
        ));
        supported = false;
    }

    // Container-codec compatibility checks.
    if out_lower == "webm"
        && !["av1", "vp9", "vp8", "copy", "none", ""].contains(&vc_lower.as_str())
    {
        warnings.push("WebM only supports VP8, VP9, or AV1 video".to_string());
    }
    if out_lower == "webm" && !["opus", "vorbis", "copy", "none", ""].contains(&ac_lower.as_str()) {
        warnings.push("WebM only supports Opus or Vorbis audio".to_string());
    }
    if out_lower == "ogg" && !vc_lower.is_empty() && vc_lower != "none" && vc_lower != "copy" {
        warnings.push("Ogg is primarily an audio container; video support is limited".to_string());
    }

    // Recommend codecs for the target container.
    let rec_video = match out_lower.as_str() {
        "webm" => "vp9",
        "mkv" | "mp4" => "av1",
        _ => "none",
    };
    let rec_audio = match out_lower.as_str() {
        "webm" | "mkv" | "mp4" | "ogg" => "opus",
        "wav" => "pcm",
        "flac" => "flac",
        _ => "opus",
    };

    let warnings_json: Vec<String> = warnings.iter().map(|w| format!(r#""{}""#, w)).collect();

    Ok(format!(
        r#"{{"supported":{},"warnings":[{}],"recommended_video":"{}","recommended_audio":"{}","input_format":"{}","output_format":"{}"}}"#,
        supported,
        warnings_json.join(","),
        rec_video,
        rec_audio,
        input_format,
        output_format,
    ))
}

/// Recommend codecs for a given container and content type.
///
/// `content_type` can be "video", "audio", or "mixed".
///
/// Returns JSON: `{"container":"webm","video_codec":"vp9","audio_codec":"opus","profile":"web_optimized"}`
#[wasm_bindgen]
pub fn wasm_recommend_codec(container: &str, content_type: &str) -> Result<String, JsValue> {
    let ct_lower = container.to_lowercase();
    let type_lower = content_type.to_lowercase();

    let (video_codec, audio_codec, profile) = match ct_lower.as_str() {
        "webm" => match type_lower.as_str() {
            "audio" => ("none", "opus", "web_optimized"),
            _ => ("vp9", "opus", "web_optimized"),
        },
        "mkv" => match type_lower.as_str() {
            "audio" => ("none", "flac", "archive"),
            _ => ("av1", "opus", "archive"),
        },
        "mp4" => match type_lower.as_str() {
            "audio" => ("none", "opus", "web_optimized"),
            _ => ("av1", "opus", "web_optimized"),
        },
        "ogg" => ("none", "vorbis", "web_optimized"),
        "wav" => ("none", "pcm", "archive"),
        "flac" => ("none", "flac", "archive"),
        _ => {
            return Err(crate::utils::js_err(&format!(
                "Unknown container: '{container}'. Supported: webm, mkv, mp4, ogg, wav, flac"
            )));
        }
    };

    Ok(format!(
        r#"{{"container":"{}","video_codec":"{}","audio_codec":"{}","profile":"{}","content_type":"{}"}}"#,
        ct_lower, video_codec, audio_codec, profile, type_lower,
    ))
}

// ---------------------------------------------------------------------------
// Pixel conversion helpers (pure Rust, no dependencies)
// ---------------------------------------------------------------------------

/// Get bytes per pixel for a format string.
fn bytes_per_pixel(format: &str) -> Result<usize, JsValue> {
    match format {
        "rgb24" | "bgr24" => Ok(3),
        "rgba32" | "bgra32" => Ok(4),
        "gray8" => Ok(1),
        _ => Err(crate::utils::js_err(&format!(
            "Unknown pixel format: '{format}'"
        ))),
    }
}

/// RGB -> RGBA (alpha = 255).
fn rgb_to_rgba(rgb: &[u8], pixel_count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixel_count * 4);
    for i in 0..pixel_count {
        let base = i * 3;
        if base + 2 < rgb.len() {
            out.push(rgb[base]);
            out.push(rgb[base + 1]);
            out.push(rgb[base + 2]);
            out.push(255);
        }
    }
    out
}

/// RGBA -> RGB (drop alpha).
fn rgba_to_rgb(rgba: &[u8], pixel_count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixel_count * 3);
    for i in 0..pixel_count {
        let base = i * 4;
        if base + 2 < rgba.len() {
            out.push(rgba[base]);
            out.push(rgba[base + 1]);
            out.push(rgba[base + 2]);
        }
    }
    out
}

/// Swap first and third channels in a 3-byte-per-pixel buffer (RGB <-> BGR).
fn swap_channels_3(data: &mut [u8], pixel_count: usize) {
    for i in 0..pixel_count {
        let base = i * 3;
        if base + 2 < data.len() {
            data.swap(base, base + 2);
        }
    }
}

/// Swap first and third channels in a 4-byte-per-pixel buffer (RGBA <-> BGRA).
fn swap_channels_4(data: &mut [u8], pixel_count: usize) {
    for i in 0..pixel_count {
        let base = i * 4;
        if base + 2 < data.len() {
            data.swap(base, base + 2);
        }
    }
}

/// RGB -> Grayscale using BT.601 luma coefficients.
fn rgb_to_gray(rgb: &[u8], pixel_count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixel_count);
    for i in 0..pixel_count {
        let base = i * 3;
        if base + 2 < rgb.len() {
            let r = rgb[base] as f64;
            let g = rgb[base + 1] as f64;
            let b = rgb[base + 2] as f64;
            let luma = (0.299 * r + 0.587 * g + 0.114 * b)
                .round()
                .min(255.0)
                .max(0.0) as u8;
            out.push(luma);
        }
    }
    out
}

/// Grayscale -> RGB (replicate channel).
fn gray_to_rgb(gray: &[u8], pixel_count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixel_count * 3);
    for i in 0..pixel_count {
        if i < gray.len() {
            out.push(gray[i]);
            out.push(gray[i]);
            out.push(gray[i]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format_webm() {
        // WebM/Matroska magic: 0x1A 0x45 0xDF 0xA3
        let header = [0x1A, 0x45, 0xDF, 0xA3, 0x00, 0x00, 0x00, 0x00];
        let result = wasm_detect_format(&header);
        assert!(result.is_ok());
        let json = result.expect("should detect");
        // Should detect as mkv or webm (Matroska family)
        assert!(json.contains("\"format\":") && (json.contains("mkv") || json.contains("webm")));
    }

    #[test]
    fn test_detect_format_too_small() {
        let header = [0x00, 0x01];
        let result = wasm_detect_format(&header);
        assert!(result.is_err());
    }

    #[test]
    fn test_list_codecs_json() {
        let result = wasm_list_convert_codecs();
        assert!(result.is_ok());
        let json = result.expect("should list");
        assert!(json.contains("av1"));
        assert!(json.contains("opus"));
        assert!(json.contains("vorbis"));
    }

    #[test]
    fn test_list_formats_json() {
        let result = wasm_list_formats();
        assert!(result.is_ok());
        let json = result.expect("should list");
        assert!(json.contains("webm"));
        assert!(json.contains("mkv"));
    }

    #[test]
    fn test_list_profiles_json() {
        let result = wasm_list_profiles();
        assert!(result.is_ok());
        let json = result.expect("should list");
        assert!(json.contains("web_optimized"));
        assert!(json.contains("archive"));
    }

    #[test]
    fn test_pixel_convert_rgb_to_rgba() {
        let rgb = vec![255, 0, 0, 0, 255, 0]; // 2 pixels: red, green
        let result = wasm_pixel_convert(&rgb, 2, 1, "rgb24", "rgba32");
        assert!(result.is_ok());
        let rgba = result.expect("should convert");
        assert_eq!(rgba.len(), 8);
        assert_eq!(&rgba[0..4], &[255, 0, 0, 255]); // red + alpha
        assert_eq!(&rgba[4..8], &[0, 255, 0, 255]); // green + alpha
    }

    #[test]
    fn test_pixel_convert_rgba_to_rgb() {
        let rgba = vec![255, 0, 0, 128, 0, 255, 0, 200];
        let result = wasm_pixel_convert(&rgba, 2, 1, "rgba32", "rgb24");
        assert!(result.is_ok());
        let rgb = result.expect("should convert");
        assert_eq!(rgb.len(), 6);
        assert_eq!(&rgb[0..3], &[255, 0, 0]);
        assert_eq!(&rgb[3..6], &[0, 255, 0]);
    }

    #[test]
    fn test_pixel_convert_rgb_to_gray() {
        // Pure white pixel
        let rgb = vec![255, 255, 255];
        let result = wasm_pixel_convert(&rgb, 1, 1, "rgb24", "gray8");
        assert!(result.is_ok());
        let gray = result.expect("should convert");
        assert_eq!(gray.len(), 1);
        assert_eq!(gray[0], 255);
    }

    #[test]
    fn test_pixel_convert_same_format() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let result = wasm_pixel_convert(&data, 2, 1, "rgb24", "rgb24");
        assert!(result.is_ok());
        assert_eq!(result.expect("same format"), data);
    }

    #[test]
    fn test_validate_conversion_supported() {
        let result = wasm_validate_conversion("mp4", "webm", "vp9", "opus");
        assert!(result.is_ok());
        let json = result.expect("should validate");
        assert!(json.contains("\"supported\":true"));
    }

    #[test]
    fn test_validate_conversion_unsupported_codec() {
        let result = wasm_validate_conversion("mp4", "webm", "h264", "aac");
        assert!(result.is_ok());
        let json = result.expect("should validate");
        assert!(json.contains("\"supported\":false"));
    }

    #[test]
    fn test_recommend_codec_webm() {
        let result = wasm_recommend_codec("webm", "video");
        assert!(result.is_ok());
        let json = result.expect("should recommend");
        assert!(json.contains("\"video_codec\":\"vp9\""));
        assert!(json.contains("\"audio_codec\":\"opus\""));
    }

    #[test]
    fn test_recommend_codec_unknown() {
        let result = wasm_recommend_codec("xyz", "video");
        assert!(result.is_err());
    }
}
