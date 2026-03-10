//! WASM image operations: format detection, header parsing, filtering, histograms.
//!
//! All functions operate on in-memory byte arrays -- no file system access.
//! Results are returned as JSON strings or raw byte arrays.

use wasm_bindgen::prelude::*;

use oximedia_image::format_detect::FormatDetector;
use oximedia_image::histogram_ops::{Histogram, RgbHistogram};

// ---------------------------------------------------------------------------
// Format detection
// ---------------------------------------------------------------------------

/// Detect image format from raw file bytes (magic number inspection).
///
/// Returns a JSON string with `format`, `extension`, `hdr`, and `lossless` fields.
///
/// # Errors
///
/// Returns an error if the input data is too short for detection.
#[wasm_bindgen]
pub fn wasm_detect_image_format(data: &[u8]) -> Result<String, JsValue> {
    if data.len() < 4 {
        return Err(crate::utils::js_err(
            "Data too short for format detection (need >= 4 bytes)",
        ));
    }

    let fmt = FormatDetector::detect(data);

    let result = serde_json::json!({
        "format": fmt.name(),
        "extension": fmt.extension(),
        "hdr": fmt.is_hdr(),
        "lossless": fmt.is_lossless(),
    });

    serde_json::to_string(&result)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization error: {}", e)))
}

// ---------------------------------------------------------------------------
// DPX header parsing
// ---------------------------------------------------------------------------

/// Parse a DPX file header from raw bytes and return metadata as JSON.
///
/// Expects at least 2048 bytes (the DPX file header size). Returns width,
/// height, bit depth, color space, orientation, and file metadata.
///
/// # Errors
///
/// Returns an error if the data is too short or the magic number is invalid.
#[wasm_bindgen]
pub fn wasm_read_dpx(data: &[u8]) -> Result<String, JsValue> {
    if data.len() < 2048 {
        return Err(crate::utils::js_err(
            "DPX header requires at least 2048 bytes",
        ));
    }

    // Check magic: "SDPX" (big-endian) or "XPDS" (little-endian)
    let is_be = &data[0..4] == b"SDPX";
    let is_le = &data[0..4] == b"XPDS";

    if !is_be && !is_le {
        return Err(crate::utils::js_err("Invalid DPX magic number"));
    }

    let read_u32 = |offset: usize| -> u32 {
        let bytes = [
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ];
        if is_be {
            u32::from_be_bytes(bytes)
        } else {
            u32::from_le_bytes(bytes)
        }
    };

    let read_u16 = |offset: usize| -> u16 {
        let bytes = [data[offset], data[offset + 1]];
        if is_be {
            u16::from_be_bytes(bytes)
        } else {
            u16::from_le_bytes(bytes)
        }
    };

    let read_string = |offset: usize, max_len: usize| -> String {
        let end = (offset + max_len).min(data.len());
        let slice = &data[offset..end];
        let null_pos = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
        String::from_utf8_lossy(&slice[..null_pos])
            .trim()
            .to_string()
    };

    // File header fields
    let image_offset = read_u32(4);
    let version = read_string(8, 8);
    let file_size = read_u32(16);
    let filename = read_string(36, 100);
    let timestamp = read_string(136, 24);
    let creator = read_string(160, 100);
    let project = read_string(260, 200);

    // Image header (offset 768)
    let orientation = read_u16(768);
    let element_count = read_u16(770);
    let pixels_per_line = read_u32(772); // width
    let lines_per_element = read_u32(776); // height

    // First image element (offset 780)
    let bit_depth_raw = data.get(803).copied().unwrap_or(0);

    let descriptor = data.get(800).copied().unwrap_or(0);
    let colorimetric = descriptor;

    let color_desc = match descriptor {
        0 => "User-defined",
        6 => "Luma (Y)",
        50 => "RGB",
        51 => "RGBA",
        100 => "CbYCrY (4:2:2)",
        102 => "CbYACrYA (4:2:2:4)",
        _ => "Unknown",
    };

    let result = serde_json::json!({
        "format": "DPX",
        "version": version,
        "endianness": if is_be { "big" } else { "little" },
        "width": pixels_per_line,
        "height": lines_per_element,
        "bit_depth": bit_depth_raw,
        "element_count": element_count,
        "orientation": orientation,
        "color_descriptor": color_desc,
        "colorimetric": colorimetric,
        "image_offset": image_offset,
        "file_size": file_size,
        "filename": filename,
        "timestamp": timestamp,
        "creator": creator,
        "project": project,
    });

    serde_json::to_string(&result)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization error: {}", e)))
}

// ---------------------------------------------------------------------------
// EXR header parsing
// ---------------------------------------------------------------------------

/// Parse an OpenEXR file header from raw bytes and return metadata as JSON.
///
/// Returns width, height, channel info, compression, and line order.
///
/// # Errors
///
/// Returns an error if the data is too short or the magic number is invalid.
#[wasm_bindgen]
pub fn wasm_read_exr(data: &[u8]) -> Result<String, JsValue> {
    if data.len() < 12 {
        return Err(crate::utils::js_err(
            "EXR header requires at least 12 bytes",
        ));
    }

    // Magic: 0x76 0x2F 0x31 0x01
    if data[0] != 0x76 || data[1] != 0x2F || data[2] != 0x31 || data[3] != 0x01 {
        return Err(crate::utils::js_err("Invalid EXR magic number"));
    }

    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let version_number = version & 0xFF;
    let is_tiled = (version & 0x200) != 0;
    let has_long_names = (version & 0x400) != 0;
    let is_multipart = (version & 0x1000) != 0;

    // Parse attribute headers
    let mut offset = 8;
    let mut channels = Vec::new();
    let mut compression = "unknown";
    let mut data_window = [0i32; 4]; // xMin, yMin, xMax, yMax
    let mut display_window = [0i32; 4];
    let mut line_order_str = "unknown";
    let mut found_data_window = false;

    // Read attributes until null name
    while offset < data.len() {
        // Read attribute name (null-terminated)
        let name_start = offset;
        while offset < data.len() && data[offset] != 0 {
            offset += 1;
        }
        if offset >= data.len() {
            break;
        }
        let name = String::from_utf8_lossy(&data[name_start..offset]).to_string();
        offset += 1; // skip null

        if name.is_empty() {
            break; // end of header
        }

        // Read type name (null-terminated)
        let type_start = offset;
        while offset < data.len() && data[offset] != 0 {
            offset += 1;
        }
        if offset >= data.len() {
            break;
        }
        let _attr_type = String::from_utf8_lossy(&data[type_start..offset]).to_string();
        offset += 1;

        // Read size
        if offset + 4 > data.len() {
            break;
        }
        let size = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        if offset + size > data.len() {
            break;
        }

        let attr_data = &data[offset..offset + size];

        match name.as_str() {
            "channels" => {
                // Parse channel list
                let mut ch_offset = 0;
                while ch_offset < attr_data.len() {
                    let ch_name_start = ch_offset;
                    while ch_offset < attr_data.len() && attr_data[ch_offset] != 0 {
                        ch_offset += 1;
                    }
                    if ch_offset >= attr_data.len() {
                        break;
                    }
                    let ch_name =
                        String::from_utf8_lossy(&attr_data[ch_name_start..ch_offset]).to_string();
                    ch_offset += 1;

                    if ch_name.is_empty() {
                        break;
                    }

                    if ch_offset + 16 <= attr_data.len() {
                        let pixel_type = u32::from_le_bytes([
                            attr_data[ch_offset],
                            attr_data[ch_offset + 1],
                            attr_data[ch_offset + 2],
                            attr_data[ch_offset + 3],
                        ]);
                        let type_name = match pixel_type {
                            0 => "uint32",
                            1 => "half",
                            2 => "float",
                            _ => "unknown",
                        };
                        channels.push(serde_json::json!({
                            "name": ch_name,
                            "type": type_name,
                        }));
                        ch_offset += 16; // type(4) + pLinear(1) + reserved(3) + xSampling(4) + ySampling(4)
                    } else {
                        break;
                    }
                }
            }
            "compression" => {
                if !attr_data.is_empty() {
                    compression = match attr_data[0] {
                        0 => "none",
                        1 => "rle",
                        2 => "zips",
                        3 => "zip",
                        4 => "piz",
                        5 => "pxr24",
                        6 => "b44",
                        7 => "b44a",
                        8 => "dwaa",
                        9 => "dwab",
                        _ => "unknown",
                    };
                }
            }
            "dataWindow" => {
                if attr_data.len() >= 16 {
                    for i in 0..4 {
                        data_window[i] = i32::from_le_bytes([
                            attr_data[i * 4],
                            attr_data[i * 4 + 1],
                            attr_data[i * 4 + 2],
                            attr_data[i * 4 + 3],
                        ]);
                    }
                    found_data_window = true;
                }
            }
            "displayWindow" => {
                if attr_data.len() >= 16 {
                    for i in 0..4 {
                        display_window[i] = i32::from_le_bytes([
                            attr_data[i * 4],
                            attr_data[i * 4 + 1],
                            attr_data[i * 4 + 2],
                            attr_data[i * 4 + 3],
                        ]);
                    }
                }
            }
            "lineOrder" => {
                if !attr_data.is_empty() {
                    line_order_str = match attr_data[0] {
                        0 => "increasing_y",
                        1 => "decreasing_y",
                        2 => "random_y",
                        _ => "unknown",
                    };
                }
            }
            _ => {}
        }

        offset += size;
    }

    let width = if found_data_window {
        (data_window[2] - data_window[0] + 1).max(0) as u32
    } else {
        0
    };
    let height = if found_data_window {
        (data_window[3] - data_window[1] + 1).max(0) as u32
    } else {
        0
    };

    let result = serde_json::json!({
        "format": "OpenEXR",
        "version": version_number,
        "tiled": is_tiled,
        "long_names": has_long_names,
        "multipart": is_multipart,
        "width": width,
        "height": height,
        "channels": channels,
        "channel_count": channels.len(),
        "compression": compression,
        "line_order": line_order_str,
        "data_window": {
            "x_min": data_window[0],
            "y_min": data_window[1],
            "x_max": data_window[2],
            "y_max": data_window[3],
        },
        "display_window": {
            "x_min": display_window[0],
            "y_min": display_window[1],
            "x_max": display_window[2],
            "y_max": display_window[3],
        },
    });

    serde_json::to_string(&result)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization error: {}", e)))
}

// ---------------------------------------------------------------------------
// Image filtering
// ---------------------------------------------------------------------------

/// Apply a pixel filter to raw 8-bit image data.
///
/// Supported filter types: `blur`, `sharpen`, `brightness`, `contrast`, `gamma`.
/// The `strength` parameter controls filter intensity (interpretation depends on type).
///
/// # Errors
///
/// Returns an error for unknown filter types or invalid parameters.
#[wasm_bindgen]
pub fn wasm_apply_image_filter(
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    filter_type: &str,
    strength: f64,
) -> Result<Vec<u8>, JsValue> {
    let w = width as usize;
    let h = height as usize;
    let ch = channels as usize;
    let expected = w * h * ch;

    if data.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "Data too small: need {} bytes for {}x{}x{}, got {}",
            expected,
            w,
            h,
            ch,
            data.len()
        )));
    }

    let pixel_data = &data[..expected];
    let mut output = pixel_data.to_vec();

    match filter_type {
        "blur" => {
            let radius = (strength as u32).max(1);
            if ch == 1 {
                output = oximedia_image::filter::box_blur(pixel_data, w, h, radius);
            } else {
                let pixel_count = w * h;
                for c in 0..ch {
                    let mut channel = vec![0u8; pixel_count];
                    for i in 0..pixel_count {
                        channel[i] = pixel_data[i * ch + c];
                    }
                    let blurred = oximedia_image::filter::box_blur(&channel, w, h, radius);
                    for i in 0..pixel_count {
                        output[i * ch + c] = blurred[i];
                    }
                }
            }
        }
        "sharpen" => {
            if ch == 1 {
                let sharpened = oximedia_image::filter::sharpen(pixel_data, w, h);
                for i in 0..output.len() {
                    let orig = pixel_data[i] as f64;
                    let sharp = sharpened[i] as f64;
                    output[i] = (orig + (sharp - orig) * strength).clamp(0.0, 255.0) as u8;
                }
            } else {
                let pixel_count = w * h;
                for c in 0..ch {
                    let mut channel = vec![0u8; pixel_count];
                    for i in 0..pixel_count {
                        channel[i] = pixel_data[i * ch + c];
                    }
                    let sharpened = oximedia_image::filter::sharpen(&channel, w, h);
                    for i in 0..pixel_count {
                        let orig = pixel_data[i * ch + c] as f64;
                        let sharp = sharpened[i] as f64;
                        output[i * ch + c] =
                            (orig + (sharp - orig) * strength).clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
        "brightness" => {
            let offset = (strength * 255.0) as i32;
            for byte in &mut output {
                *byte = ((*byte as i32) + offset).clamp(0, 255) as u8;
            }
        }
        "contrast" => {
            for byte in &mut output {
                let val = ((*byte as f64 - 128.0) * strength + 128.0).clamp(0.0, 255.0);
                *byte = val as u8;
            }
        }
        "gamma" => {
            if strength <= 0.0 {
                return Err(crate::utils::js_err("Gamma must be > 0"));
            }
            // Gamma > 1 compresses the tonal range toward the shadows (darkens),
            // gamma < 1 expands it toward the highlights (brightens).
            for byte in &mut output {
                let norm = *byte as f64 / 255.0;
                *byte = (norm.powf(strength) * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
        other => {
            return Err(crate::utils::js_err(&format!(
                "Unknown filter type '{}'. Supported: blur, sharpen, brightness, contrast, gamma",
                other
            )));
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Histogram
// ---------------------------------------------------------------------------

/// Compute a histogram from raw 8-bit image data.
///
/// `mode` can be `"rgb"` (per-channel) or `"luma"` (BT.601 luminance).
/// Returns JSON with `bins` (array of 256 counts per channel).
///
/// # Errors
///
/// Returns an error for invalid mode or insufficient data.
#[wasm_bindgen]
pub fn wasm_image_histogram(
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    mode: &str,
) -> Result<String, JsValue> {
    let w = width as usize;
    let h = height as usize;
    let ch = channels as usize;
    let expected = w * h * ch;

    if data.len() < expected {
        return Err(crate::utils::js_err(&format!(
            "Data too small: need {} bytes, got {}",
            expected,
            data.len()
        )));
    }

    let pixel_count = w * h;

    match mode {
        "luma" => {
            let mut hist = Histogram::new();
            for i in 0..pixel_count {
                let luma = if ch >= 3 {
                    let r = data[i * ch] as f64;
                    let g = data[i * ch + 1] as f64;
                    let b = data[i * ch + 2] as f64;
                    (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0) as u8
                } else {
                    data[i * ch]
                };
                hist.accumulate(luma);
            }

            let result = serde_json::json!({
                "mode": "luma",
                "channels": 1,
                "bins": [hist.bins.to_vec()],
                "statistics": {
                    "mean": hist.mean(),
                    "std_dev": hist.std_dev(),
                    "median": hist.median(),
                    "mode": hist.mode(),
                },
            });

            serde_json::to_string(&result)
                .map_err(|e| crate::utils::js_err(&format!("JSON error: {}", e)))
        }
        "rgb" | "per-channel" => {
            if ch >= 3 {
                let mut rgb = RgbHistogram::new();
                for i in 0..pixel_count {
                    let base = i * ch;
                    rgb.accumulate_rgb(data[base], data[base + 1], data[base + 2]);
                }

                let result = serde_json::json!({
                    "mode": "rgb",
                    "channels": 3,
                    "bins": [
                        rgb.red.bins.to_vec(),
                        rgb.green.bins.to_vec(),
                        rgb.blue.bins.to_vec(),
                    ],
                    "statistics": {
                        "r": { "mean": rgb.red.mean(), "std_dev": rgb.red.std_dev(), "median": rgb.red.median() },
                        "g": { "mean": rgb.green.mean(), "std_dev": rgb.green.std_dev(), "median": rgb.green.median() },
                        "b": { "mean": rgb.blue.mean(), "std_dev": rgb.blue.std_dev(), "median": rgb.blue.median() },
                    },
                });

                serde_json::to_string(&result)
                    .map_err(|e| crate::utils::js_err(&format!("JSON error: {}", e)))
            } else {
                let mut hist = Histogram::new();
                hist.accumulate_slice(&data[..pixel_count]);
                let result = serde_json::json!({
                    "mode": "grayscale",
                    "channels": 1,
                    "bins": [hist.bins.to_vec()],
                    "statistics": {
                        "mean": hist.mean(),
                        "std_dev": hist.std_dev(),
                        "median": hist.median(),
                    },
                });
                serde_json::to_string(&result)
                    .map_err(|e| crate::utils::js_err(&format!("JSON error: {}", e)))
            }
        }
        other => Err(crate::utils::js_err(&format!(
            "Unknown histogram mode '{}'. Valid: rgb, luma, per-channel",
            other
        ))),
    }
}

// ---------------------------------------------------------------------------
// Pixel depth conversion
// ---------------------------------------------------------------------------

/// Convert pixel data between bit depths (e.g. 8-bit to 16-bit or vice versa).
///
/// # Errors
///
/// Returns an error for unsupported depth combinations.
#[wasm_bindgen]
pub fn wasm_convert_pixel_depth(
    data: &[u8],
    from_depth: u32,
    to_depth: u32,
) -> Result<Vec<u8>, JsValue> {
    match (from_depth, to_depth) {
        (8, 16) => {
            let mut out = Vec::with_capacity(data.len() * 2);
            for &b in data {
                let val = u16::from(b) * 257; // 0-255 -> 0-65535
                out.extend_from_slice(&val.to_le_bytes());
            }
            Ok(out)
        }
        (16, 8) => {
            if data.len() % 2 != 0 {
                return Err(crate::utils::js_err(
                    "16-bit data must have even byte length",
                ));
            }
            let mut out = Vec::with_capacity(data.len() / 2);
            for chunk in data.chunks_exact(2) {
                let val = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push((val >> 8) as u8);
            }
            Ok(out)
        }
        (8, 32) => {
            let mut out = Vec::with_capacity(data.len() * 4);
            for &b in data {
                let val = b as f32 / 255.0;
                out.extend_from_slice(&val.to_le_bytes());
            }
            Ok(out)
        }
        (32, 8) => {
            if data.len() % 4 != 0 {
                return Err(crate::utils::js_err(
                    "32-bit float data must be 4-byte aligned",
                ));
            }
            let mut out = Vec::with_capacity(data.len() / 4);
            for chunk in data.chunks_exact(4) {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push((val * 255.0).clamp(0.0, 255.0) as u8);
            }
            Ok(out)
        }
        (f, t) if f == t => Ok(data.to_vec()),
        (f, t) => Err(crate::utils::js_err(&format!(
            "Unsupported depth conversion: {}-bit to {}-bit. Supported: 8<->16, 8<->32",
            f, t
        ))),
    }
}

// ---------------------------------------------------------------------------
// Generic image info
// ---------------------------------------------------------------------------

/// Attempt to detect format and extract basic metadata from raw image bytes.
///
/// This is a convenience function that combines format detection with
/// header parsing when possible.
///
/// # Errors
///
/// Returns an error if the data is too short for detection.
#[wasm_bindgen]
pub fn wasm_image_info(data: &[u8]) -> Result<String, JsValue> {
    if data.len() < 4 {
        return Err(crate::utils::js_err("Data too short (need >= 4 bytes)"));
    }

    let fmt = FormatDetector::detect(data);

    // Try format-specific parsing for more detail
    match fmt {
        oximedia_image::format_detect::ImageFormat::Dpx => {
            if data.len() >= 2048 {
                return wasm_read_dpx(data);
            }
        }
        oximedia_image::format_detect::ImageFormat::Exr => {
            if data.len() >= 12 {
                return wasm_read_exr(data);
            }
        }
        _ => {}
    }

    // Fallback: basic format info
    let result = serde_json::json!({
        "format": fmt.name(),
        "extension": fmt.extension(),
        "hdr": fmt.is_hdr(),
        "lossless": fmt.is_lossless(),
        "detail": "Format-specific metadata requires more header bytes or is not supported for this format",
    });

    serde_json::to_string(&result).map_err(|e| crate::utils::js_err(&format!("JSON error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_png_format() {
        let header = [
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x00,
        ];
        let result = wasm_detect_image_format(&header).expect("should succeed");
        assert!(result.contains("PNG"));
    }

    #[test]
    fn test_detect_dpx_format() {
        let mut header = vec![0u8; 12];
        header[..4].copy_from_slice(b"SDPX");
        let result = wasm_detect_image_format(&header).expect("should succeed");
        assert!(result.contains("DPX"));
    }

    #[test]
    fn test_detect_exr_format() {
        let header = [
            0x76, 0x2F, 0x31, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let result = wasm_detect_image_format(&header).expect("should succeed");
        assert!(result.contains("OpenEXR"));
    }

    #[test]
    fn test_detect_too_short() {
        let header = [0x89, 0x50];
        assert!(wasm_detect_image_format(&header).is_err());
    }

    #[test]
    fn test_brightness_filter() {
        let data = vec![128u8; 12]; // 2x2 RGB
        let result =
            wasm_apply_image_filter(&data, 2, 2, 3, "brightness", 0.1).expect("should succeed");
        assert_eq!(result.len(), 12);
        // 128 + 25.5 = ~153
        assert!(result[0] > 128);
    }

    #[test]
    fn test_gamma_filter() {
        let data = vec![128u8; 4]; // 2x2 grayscale
        let result = wasm_apply_image_filter(&data, 2, 2, 1, "gamma", 2.2).expect("should succeed");
        assert_eq!(result.len(), 4);
        // gamma > 1 darkens midtones
        assert!(result[0] < 128);
    }

    #[test]
    fn test_unknown_filter() {
        let data = vec![128u8; 4];
        assert!(wasm_apply_image_filter(&data, 2, 2, 1, "magic", 1.0).is_err());
    }

    #[test]
    fn test_histogram_luma() {
        let data = vec![128u8; 12]; // 2x2 RGB all 128
        let result = wasm_image_histogram(&data, 2, 2, 3, "luma").expect("should succeed");
        assert!(result.contains("luma"));
        // BT.601 luma of (128,128,128): 0.299*128 + 0.587*128 + 0.114*128.
        // The IEEE-754 sum may truncate to 127 or round to 128; accept both.
        assert!(
            result.contains("127") || result.contains("128"),
            "expected luma near 128, got: {}",
            &result[..result.len().min(200)]
        );
    }

    #[test]
    fn test_convert_depth_8_to_16() {
        let data = vec![0u8, 255];
        let result = wasm_convert_pixel_depth(&data, 8, 16).expect("should succeed");
        assert_eq!(result.len(), 4);
        let last = u16::from_le_bytes([result[2], result[3]]);
        assert_eq!(last, 65535);
    }

    #[test]
    fn test_convert_depth_same() {
        let data = vec![42u8; 10];
        let result = wasm_convert_pixel_depth(&data, 8, 8).expect("should succeed");
        assert_eq!(result, data);
    }

    #[test]
    fn test_image_info_unknown() {
        let data = vec![0u8; 16];
        let result = wasm_image_info(&data).expect("should succeed");
        assert!(result.contains("Unknown"));
    }
}
