//! WebAssembly bindings for `oximedia-captions` caption processing.
//!
//! Provides caption parsing, format conversion, and validation for
//! browser-based caption editing and display.

use wasm_bindgen::prelude::*;

use oximedia_captions::{export::Exporter, import::Importer, validation, CaptionFormat};

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

fn parse_format(s: &str) -> Result<CaptionFormat, JsValue> {
    match s.to_lowercase().as_str() {
        "srt" => Ok(CaptionFormat::Srt),
        "vtt" | "webvtt" => Ok(CaptionFormat::WebVtt),
        "ass" => Ok(CaptionFormat::Ass),
        "ssa" => Ok(CaptionFormat::Ssa),
        "ttml" => Ok(CaptionFormat::Ttml),
        "dfxp" => Ok(CaptionFormat::Dfxp),
        "scc" => Ok(CaptionFormat::Scc),
        "stl" | "ebu-stl" => Ok(CaptionFormat::EbuStl),
        "itt" => Ok(CaptionFormat::ITt),
        "cea608" | "cea-608" => Ok(CaptionFormat::Cea608),
        "cea708" | "cea-708" => Ok(CaptionFormat::Cea708),
        other => Err(crate::utils::js_err(&format!(
            "Unknown caption format: {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Parse captions
// ---------------------------------------------------------------------------

/// Parse caption data and return a JSON representation.
///
/// `data` is the caption file content as bytes.
/// `format` is "srt", "vtt", "ass", "ttml", "scc", etc. or "auto".
///
/// Returns JSON:
/// ```json
/// {
///   "language": "en",
///   "count": 42,
///   "captions": [
///     {
///       "start_ms": 1000,
///       "end_ms": 3000,
///       "text": "Hello world",
///       "start_tc": "00:00:01.000",
///       "end_tc": "00:00:03.000"
///     },
///     ...
///   ]
/// }
/// ```
///
/// # Errors
///
/// Returns an error if parsing fails.
#[wasm_bindgen]
pub fn wasm_parse_captions(data: &[u8], format: &str) -> Result<String, JsValue> {
    if data.is_empty() {
        return Err(crate::utils::js_err("Empty caption data"));
    }

    let track = if format == "auto" {
        Importer::import_auto(data)
            .map_err(|e| crate::utils::js_err(&format!("Parse failed: {e}")))?
    } else {
        let fmt = parse_format(format)?;
        Importer::import(data, fmt)
            .map_err(|e| crate::utils::js_err(&format!("Parse failed: {e}")))?
    };

    let captions_json: Vec<String> = track
        .captions
        .iter()
        .map(|cap| {
            let (sh, sm, ss, sms) = cap.start.as_hmsm();
            let (eh, em, es, ems) = cap.end.as_hmsm();
            format!(
                "{{\"start_ms\":{},\"end_ms\":{},\"text\":\"{}\",\
                 \"start_tc\":\"{:02}:{:02}:{:02}.{:03}\",\
                 \"end_tc\":\"{:02}:{:02}:{:02}.{:03}\"}}",
                cap.start.as_millis(),
                cap.end.as_millis(),
                escape_json_string(&cap.text),
                sh,
                sm,
                ss,
                sms,
                eh,
                em,
                es,
                ems,
            )
        })
        .collect();

    Ok(format!(
        "{{\"language\":\"{}\",\"count\":{},\"captions\":[{}]}}",
        escape_json_string(&track.language.code),
        track.captions.len(),
        captions_json.join(","),
    ))
}

// ---------------------------------------------------------------------------
// Convert captions
// ---------------------------------------------------------------------------

/// Convert caption data from one format to another.
///
/// `data` is the caption file content as bytes.
/// `from_format` is the source format (or "auto").
/// `to_format` is the target format.
///
/// Returns the converted caption content as a string.
///
/// # Errors
///
/// Returns an error if conversion fails.
#[wasm_bindgen]
pub fn wasm_convert_captions(
    data: &[u8],
    from_format: &str,
    to_format: &str,
) -> Result<String, JsValue> {
    if data.is_empty() {
        return Err(crate::utils::js_err("Empty caption data"));
    }

    let track = if from_format == "auto" {
        Importer::import_auto(data)
            .map_err(|e| crate::utils::js_err(&format!("Import failed: {e}")))?
    } else {
        let fmt = parse_format(from_format)?;
        Importer::import(data, fmt)
            .map_err(|e| crate::utils::js_err(&format!("Import failed: {e}")))?
    };

    let target = parse_format(to_format)?;
    let output = Exporter::export(&track, target)
        .map_err(|e| crate::utils::js_err(&format!("Export failed: {e}")))?;

    String::from_utf8(output)
        .map_err(|e| crate::utils::js_err(&format!("Output is not valid UTF-8: {e}")))
}

// ---------------------------------------------------------------------------
// Validate captions
// ---------------------------------------------------------------------------

/// Validate caption data against a standard.
///
/// `data` is the caption file content as bytes.
/// `standard` is the standard name: "fcc", "wcag", "cea608", "cea708".
///
/// Returns JSON:
/// ```json
/// {
///   "passed": true,
///   "standard": "fcc",
///   "total_captions": 42,
///   "errors": 0,
///   "warnings": 2,
///   "avg_reading_speed": 145.5,
///   "max_chars_per_line": 37,
///   "issues": [
///     {"severity": "Warning", "message": "...", "rule": "..."},
///     ...
///   ]
/// }
/// ```
///
/// # Errors
///
/// Returns an error if parsing or validation fails.
#[wasm_bindgen]
pub fn wasm_validate_captions(data: &[u8], standard: &str) -> Result<String, JsValue> {
    if data.is_empty() {
        return Err(crate::utils::js_err("Empty caption data"));
    }

    let track = Importer::import_auto(data)
        .map_err(|e| crate::utils::js_err(&format!("Parse failed: {e}")))?;

    let validator = validation::Validator::new();
    let report = validator
        .validate(&track)
        .map_err(|e| crate::utils::js_err(&format!("Validation failed: {e}")))?;

    let issues_json: Vec<String> = report
        .issues
        .iter()
        .map(|issue| {
            format!(
                "{{\"severity\":\"{:?}\",\"message\":\"{}\",\"rule\":\"{}\"}}",
                issue.severity,
                escape_json_string(&issue.message),
                escape_json_string(&issue.rule),
            )
        })
        .collect();

    Ok(format!(
        "{{\"passed\":{},\"standard\":\"{}\",\"total_captions\":{},\
         \"errors\":{},\"warnings\":{},\"avg_reading_speed\":{:.1},\
         \"max_chars_per_line\":{},\"issues\":[{}]}}",
        report.passed(),
        escape_json_string(standard),
        report.statistics.total_captions,
        report.statistics.error_count,
        report.statistics.warning_count,
        report.statistics.avg_reading_speed,
        report.statistics.max_chars_per_line,
        issues_json.join(","),
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_format_known() {
        assert!(parse_format("srt").is_ok());
        assert!(parse_format("vtt").is_ok());
        assert!(parse_format("ass").is_ok());
        assert!(parse_format("ttml").is_ok());
    }

    #[test]
    fn test_parse_format_unknown() {
        assert!(parse_format("xyz").is_err());
    }

    #[test]
    fn test_parse_captions_empty() {
        let result = wasm_parse_captions(&[], "auto");
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_captions_empty() {
        let result = wasm_convert_captions(&[], "auto", "srt");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_captions_empty() {
        let result = wasm_validate_captions(&[], "fcc");
        assert!(result.is_err());
    }
}
