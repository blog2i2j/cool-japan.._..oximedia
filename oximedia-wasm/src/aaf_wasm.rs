//! WebAssembly bindings for `oximedia-aaf` AAF file operations.
//!
//! All functions accept AAF data bytes and return JSON strings
//! with structure information. Designed for synchronous browser usage.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn escape_json_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn json_string_array(items: &[String]) -> String {
    let entries: Vec<String> = items
        .iter()
        .map(|s| format!("\"{}\"", escape_json_string(s)))
        .collect();
    format!("[{}]", entries.join(","))
}

// ---------------------------------------------------------------------------
// AAF Header Parsing
// ---------------------------------------------------------------------------

/// Parse an AAF file header and return structure info as JSON.
///
/// `data` should contain the AAF file bytes.
///
/// Returns JSON:
/// ```json
/// {
///   "valid": true,
///   "data_size": 1024,
///   "has_structured_storage": true,
///   "magic_bytes": "D0CF11E0"
/// }
/// ```
///
/// # Errors
///
/// Returns an error if the data cannot be parsed.
#[wasm_bindgen]
pub fn wasm_parse_aaf_header(data: &[u8]) -> Result<String, JsValue> {
    if data.len() < 8 {
        return Err(crate::utils::js_err(
            "AAF data too small (need at least 8 bytes)",
        ));
    }

    // Check for Structured Storage (OLE/COM) magic: D0 CF 11 E0 A1 B1 1A E1
    let has_ss = data.len() >= 8
        && data[0] == 0xD0
        && data[1] == 0xCF
        && data[2] == 0x11
        && data[3] == 0xE0
        && data[4] == 0xA1
        && data[5] == 0xB1
        && data[6] == 0x1A
        && data[7] == 0xE1;

    let magic = if data.len() >= 4 {
        format!(
            "{:02X}{:02X}{:02X}{:02X}",
            data[0], data[1], data[2], data[3]
        )
    } else {
        "N/A".to_string()
    };

    // Extract sector size from header (offset 30, 2 bytes LE)
    let sector_size = if has_ss && data.len() >= 32 {
        let power = u16::from_le_bytes([data[30], data[31]]);
        1u32 << power
    } else {
        0
    };

    Ok(format!(
        "{{\"valid\":{},\"data_size\":{},\"has_structured_storage\":{},\
         \"magic_bytes\":\"{}\",\"sector_size\":{}}}",
        has_ss,
        data.len(),
        has_ss,
        escape_json_string(&magic),
        sector_size,
    ))
}

/// Validate AAF data and return issues as JSON.
///
/// Returns JSON:
/// ```json
/// {
///   "valid": true,
///   "issues": [],
///   "data_size": 1024
/// }
/// ```
///
/// # Errors
///
/// Returns an error if basic parsing fails.
#[wasm_bindgen]
pub fn wasm_validate_aaf(data: &[u8]) -> Result<String, JsValue> {
    let mut issues = Vec::new();

    if data.len() < 512 {
        issues.push("File too small to be a valid AAF (minimum 512 bytes)".to_string());
    }

    // Check Structured Storage magic
    let has_magic =
        data.len() >= 8 && data[0] == 0xD0 && data[1] == 0xCF && data[2] == 0x11 && data[3] == 0xE0;

    if !has_magic {
        issues.push("Missing Structured Storage (OLE) magic bytes".to_string());
    }

    // Check for full magic sequence
    if data.len() >= 8 && has_magic {
        if data[4] != 0xA1 || data[5] != 0xB1 || data[6] != 0x1A || data[7] != 0xE1 {
            issues.push("Incomplete Structured Storage magic signature".to_string());
        }
    }

    // Check minor version at offset 24
    if data.len() >= 26 && has_magic {
        let minor_ver = u16::from_le_bytes([data[24], data[25]]);
        if minor_ver > 0x003E {
            issues.push(format!("Unusual minor version: 0x{:04X}", minor_ver));
        }
    }

    // Check major version at offset 26
    if data.len() >= 28 && has_magic {
        let major_ver = u16::from_le_bytes([data[26], data[27]]);
        if major_ver != 3 && major_ver != 4 {
            issues.push(format!(
                "Unexpected major version: {} (expected 3 or 4)",
                major_ver
            ));
        }
    }

    let valid = issues.is_empty();
    let issues_json = json_string_array(&issues);

    Ok(format!(
        "{{\"valid\":{},\"issues\":{},\"data_size\":{}}}",
        valid,
        issues_json,
        data.len(),
    ))
}

/// List tracks from AAF data as JSON.
///
/// Since full AAF parsing in WASM requires structured storage traversal,
/// this performs a heuristic scan for track-related data definitions.
///
/// Returns JSON array of track descriptions.
///
/// # Errors
///
/// Returns an error if the data is too small to analyze.
#[wasm_bindgen]
pub fn wasm_aaf_track_list(data: &[u8]) -> Result<String, JsValue> {
    if data.len() < 512 {
        return Err(crate::utils::js_err(
            "AAF data too small to extract track information",
        ));
    }

    let mut tracks = Vec::new();

    // Heuristic: scan for known AAF data definition UUIDs
    // Picture data definition: 01030202-0100-0000-060e-2b3404010101
    // Sound data definition: 01030202-0200-0000-060e-2b3404010101

    let has_video = find_aaf_data_def(data, &[0x01, 0x03, 0x02, 0x02, 0x01]);
    let has_audio = find_aaf_data_def(data, &[0x01, 0x03, 0x02, 0x02, 0x02]);

    let mut idx = 0u32;

    if has_video {
        tracks.push(format!(
            "{{\"index\":{},\"type\":\"Video\",\"detected\":true}}",
            idx
        ));
        idx += 1;
    }

    if has_audio {
        tracks.push(format!(
            "{{\"index\":{},\"type\":\"Audio\",\"detected\":true}}",
            idx
        ));
        idx += 1;
    }

    // Check for timecode data def pattern
    let has_timecode = find_aaf_data_def(data, &[0x01, 0x03, 0x02, 0x01]);
    if has_timecode {
        tracks.push(format!(
            "{{\"index\":{},\"type\":\"Timecode\",\"detected\":true}}",
            idx
        ));
    }

    // If no tracks found heuristically, report empty
    if tracks.is_empty() {
        tracks.push("{\"index\":0,\"type\":\"Unknown\",\"detected\":false}".to_string());
    }

    Ok(format!("[{}]", tracks.join(",")))
}

/// Check if a byte pattern (AAF data definition prefix) exists in the data.
fn find_aaf_data_def(data: &[u8], pattern: &[u8]) -> bool {
    if pattern.is_empty() || pattern.len() > data.len() {
        return false;
    }
    data.windows(pattern.len()).any(|w| w == pattern)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_header() {
        // Structured Storage magic + enough padding
        let mut data = vec![0u8; 512];
        data[0] = 0xD0;
        data[1] = 0xCF;
        data[2] = 0x11;
        data[3] = 0xE0;
        data[4] = 0xA1;
        data[5] = 0xB1;
        data[6] = 0x1A;
        data[7] = 0xE1;
        data[30] = 9; // sector size = 2^9 = 512

        let result = wasm_parse_aaf_header(&data);
        assert!(result.is_ok());
        let json = result.expect("should parse");
        assert!(json.contains("\"valid\":true"));
        assert!(json.contains("\"sector_size\":512"));
    }

    #[test]
    fn test_parse_invalid_header() {
        let data = vec![0u8; 512];
        let result = wasm_parse_aaf_header(&data);
        assert!(result.is_ok());
        let json = result.expect("should parse");
        assert!(json.contains("\"valid\":false"));
    }

    #[test]
    fn test_validate_too_small() {
        let data = vec![0u8; 100];
        let result = wasm_validate_aaf(&data);
        assert!(result.is_ok());
        let json = result.expect("should validate");
        assert!(json.contains("too small"));
    }

    #[test]
    fn test_track_list_too_small() {
        let data = vec![0u8; 10];
        let result = wasm_aaf_track_list(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_track_list_with_video_pattern() {
        let mut data = vec![0u8; 1024];
        // Insert video data def pattern
        data[100] = 0x01;
        data[101] = 0x03;
        data[102] = 0x02;
        data[103] = 0x02;
        data[104] = 0x01;

        let result = wasm_aaf_track_list(&data);
        assert!(result.is_ok());
        let json = result.expect("should find tracks");
        assert!(json.contains("Video"));
    }
}
