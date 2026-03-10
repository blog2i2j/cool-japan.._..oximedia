//! WebAssembly bindings for `oximedia-imf` IMF package operations.
//!
//! All functions accept XML/data bytes and return JSON strings
//! with IMF analysis results. Designed for synchronous browser usage.

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
// CPL Validation
// ---------------------------------------------------------------------------

/// Validate an IMF CPL (Composition Playlist) from XML data.
///
/// `xml_data` should contain the CPL XML as UTF-8 bytes.
///
/// Returns JSON:
/// ```json
/// {
///   "valid": true,
///   "errors": [],
///   "warnings": [],
///   "element_count": 42
/// }
/// ```
///
/// # Errors
///
/// Returns an error if the XML cannot be parsed.
#[wasm_bindgen]
pub fn wasm_validate_imf_cpl(xml_data: &[u8]) -> Result<String, JsValue> {
    let xml_str = std::str::from_utf8(xml_data)
        .map_err(|e| crate::utils::js_err(&format!("Invalid UTF-8 in CPL data: {e}")))?;

    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // Basic structural checks on the CPL XML
    if !xml_str.contains("CompositionPlaylist") {
        errors.push("Missing CompositionPlaylist root element".to_string());
    }

    if !xml_str.contains("Id") {
        errors.push("Missing Id element in CPL".to_string());
    }

    if !xml_str.contains("EditRate") {
        warnings.push("No EditRate found in CPL".to_string());
    }

    if !xml_str.contains("SegmentList") && !xml_str.contains("Segment") {
        warnings.push("No segments found in CPL".to_string());
    }

    // Check for required SMPTE namespace
    if !xml_str.contains("smpte") && !xml_str.contains("2067") {
        warnings.push("No SMPTE namespace reference found".to_string());
    }

    let valid = errors.is_empty();
    let errors_json = json_string_array(&errors);
    let warnings_json = json_string_array(&warnings);

    Ok(format!(
        "{{\"valid\":{},\"errors\":{},\"warnings\":{},\"data_size\":{}}}",
        valid,
        errors_json,
        warnings_json,
        xml_data.len(),
    ))
}

/// Parse an IMF CPL and return composition info as JSON.
///
/// `xml_data` should contain the CPL XML as UTF-8 bytes.
///
/// Returns JSON:
/// ```json
/// {
///   "title": "My Composition",
///   "edit_rate": "24/1",
///   "duration_frames": 1440,
///   "segment_count": 1,
///   "sequence_count": 2
/// }
/// ```
///
/// # Errors
///
/// Returns an error if the XML cannot be parsed.
#[wasm_bindgen]
pub fn wasm_parse_imf_cpl(xml_data: &[u8]) -> Result<String, JsValue> {
    let xml_str = std::str::from_utf8(xml_data)
        .map_err(|e| crate::utils::js_err(&format!("Invalid UTF-8 in CPL data: {e}")))?;

    // Extract title
    let title = extract_xml_element(xml_str, "ContentTitle")
        .or_else(|| extract_xml_element(xml_str, "Title"))
        .unwrap_or_else(|| "Untitled".to_string());

    // Extract edit rate
    let edit_rate = extract_xml_element(xml_str, "EditRate")
        .unwrap_or_else(|| "24 1".to_string())
        .replace(' ', "/");

    // Count segments
    let segment_count = xml_str.matches("<Segment>").count() + xml_str.matches("<Segment ").count();

    // Count sequences
    let sequence_count = xml_str.matches("Sequence>").count() / 2; // open+close tags

    Ok(format!(
        "{{\"title\":\"{}\",\"edit_rate\":\"{}\",\"segment_count\":{},\
         \"sequence_count\":{},\"data_size\":{}}}",
        escape_json_string(&title),
        escape_json_string(&edit_rate),
        segment_count,
        sequence_count,
        xml_data.len(),
    ))
}

/// Extract track information from an IMF CPL.
///
/// Returns JSON array of track descriptions.
///
/// # Errors
///
/// Returns an error if the XML cannot be parsed.
#[wasm_bindgen]
pub fn wasm_imf_track_info(xml_data: &[u8]) -> Result<String, JsValue> {
    let xml_str = std::str::from_utf8(xml_data)
        .map_err(|e| crate::utils::js_err(&format!("Invalid UTF-8 in CPL data: {e}")))?;

    let mut tracks = Vec::new();
    let mut idx = 0u32;

    // Detect video sequences
    if xml_str.contains("MainImageSequence") {
        tracks.push(format!(
            "{{\"index\":{},\"type\":\"MainImage\",\"found\":true}}",
            idx
        ));
        idx += 1;
    }

    // Detect audio sequences
    let audio_count = xml_str.matches("MainAudioSequence").count() / 2;
    for _ in 0..audio_count
        .max(1)
        .min(if xml_str.contains("MainAudioSequence") {
            audio_count
        } else {
            0
        })
    {
        tracks.push(format!(
            "{{\"index\":{},\"type\":\"MainAudio\",\"found\":true}}",
            idx
        ));
        idx += 1;
    }

    // Detect subtitle sequences
    if xml_str.contains("SubtitlesSequence") {
        tracks.push(format!(
            "{{\"index\":{},\"type\":\"Subtitles\",\"found\":true}}",
            idx
        ));
        idx += 1;
    }

    // Detect marker sequences
    if xml_str.contains("MarkerSequence") {
        tracks.push(format!(
            "{{\"index\":{},\"type\":\"Marker\",\"found\":true}}",
            idx
        ));
    }

    Ok(format!("[{}]", tracks.join(",")))
}

// ---------------------------------------------------------------------------
// XML helpers
// ---------------------------------------------------------------------------

/// Extract the text content of a simple XML element.
fn extract_xml_element(xml: &str, tag: &str) -> Option<String> {
    let open_tag = format!("<{}", tag);
    let close_tag = format!("</{}>", tag);

    if let Some(open_pos) = xml.find(&open_tag) {
        let after_open = &xml[open_pos..];
        // Find the end of the opening tag
        if let Some(gt_pos) = after_open.find('>') {
            let content_start = open_pos + gt_pos + 1;
            if let Some(close_pos) = xml[content_start..].find(&close_tag) {
                let content = &xml[content_start..content_start + close_pos];
                return Some(content.trim().to_string());
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_valid_cpl() {
        let cpl =
            b"<CompositionPlaylist><Id>test</Id><EditRate>24 1</EditRate></CompositionPlaylist>";
        let result = wasm_validate_imf_cpl(cpl);
        assert!(result.is_ok());
        let json = result.expect("should validate");
        assert!(json.contains("\"valid\":true"));
    }

    #[test]
    fn test_validate_invalid_cpl() {
        let cpl = b"<SomeOtherXml>content</SomeOtherXml>";
        let result = wasm_validate_imf_cpl(cpl);
        assert!(result.is_ok());
        let json = result.expect("should report errors");
        assert!(json.contains("\"valid\":false"));
    }

    #[test]
    fn test_parse_cpl_title() {
        let cpl = b"<CompositionPlaylist><ContentTitle>My Movie</ContentTitle><EditRate>25 1</EditRate></CompositionPlaylist>";
        let result = wasm_parse_imf_cpl(cpl);
        assert!(result.is_ok());
        let json = result.expect("should parse");
        assert!(json.contains("My Movie"));
        assert!(json.contains("25/1"));
    }

    #[test]
    fn test_track_info_video_audio() {
        let cpl = b"<CompositionPlaylist><MainImageSequence></MainImageSequence><MainAudioSequence></MainAudioSequence></CompositionPlaylist>";
        let result = wasm_imf_track_info(cpl);
        assert!(result.is_ok());
        let json = result.expect("should find tracks");
        assert!(json.contains("MainImage"));
        assert!(json.contains("MainAudio"));
    }

    #[test]
    fn test_extract_xml_element() {
        let xml = "<root><Title>Hello World</Title></root>";
        let title = extract_xml_element(xml, "Title");
        assert_eq!(title, Some("Hello World".to_string()));

        let missing = extract_xml_element(xml, "Missing");
        assert!(missing.is_none());
    }
}
