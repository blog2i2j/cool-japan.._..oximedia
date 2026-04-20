//! Integration tests for the ffprobe-compatible output formatter.

use oximedia_compat_ffmpeg::ffprobe::{ProbeFormat, ProbeOutput, ProbeStream};
use oximedia_compat_ffmpeg::ffprobe_output::{format_probe_result, FfprobeOutputFormat};

fn make_output() -> ProbeOutput {
    let mut stream = ProbeStream::new_video("av1", 1920, 1080, "16:9", 24.0);
    stream.bit_rate = Some(4_000_000);
    stream.duration = Some("120.000000".to_string());
    let format = ProbeFormat::new("test.mkv", "matroska,webm", 60_000_000, 120.0);
    ProbeOutput {
        format: Some(format),
        streams: vec![stream],
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_json_is_valid_structure() {
    let out = make_output();
    let json = format_probe_result(&out, FfprobeOutputFormat::Json).expect("json");
    // Must start and end with braces (JSON object).
    assert!(json.trim_start().starts_with('{'), "should be JSON object");
    assert!(json.trim_end().ends_with('}'), "should close JSON object");
}

#[test]
fn test_json_contains_streams_key() {
    let out = make_output();
    let json = format_probe_result(&out, FfprobeOutputFormat::Json).expect("json");
    assert!(json.contains("\"streams\""), "should have streams key");
}

#[test]
fn test_json_contains_codec_name() {
    let out = make_output();
    let json = format_probe_result(&out, FfprobeOutputFormat::Json).expect("json");
    assert!(
        json.contains("\"codec_name\""),
        "should have codec_name key"
    );
    assert!(json.contains("\"av1\""), "should contain av1 value");
}

#[test]
fn test_json_contains_format_key() {
    let out = make_output();
    let json = format_probe_result(&out, FfprobeOutputFormat::Json).expect("json");
    assert!(json.contains("\"format\""), "should have format key");
    assert!(
        json.contains("\"matroska,webm\""),
        "should contain format name"
    );
}

#[test]
fn test_json_roundtrip_via_serde() {
    let out = make_output();
    let json_str = format_probe_result(&out, FfprobeOutputFormat::Json).expect("json");
    // Parse with serde_json to confirm it's valid JSON.
    let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("should be valid JSON");
    assert!(parsed.is_object(), "top-level should be an object");
    // Verify streams array exists and has one element.
    let streams = parsed.get("streams").expect("streams key");
    assert!(streams.is_array(), "streams should be an array");
    assert_eq!(streams.as_array().unwrap().len(), 1);
    // Verify codec_name in first stream.
    let codec_name = streams[0]
        .get("codec_name")
        .and_then(|v| v.as_str())
        .expect("codec_name");
    assert_eq!(codec_name, "av1");
}

// ─────────────────────────────────────────────────────────────────────────────
// XML tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_xml_starts_with_declaration() {
    let out = make_output();
    let xml = format_probe_result(&out, FfprobeOutputFormat::Xml).expect("xml");
    assert!(
        xml.starts_with("<?xml"),
        "should start with XML declaration"
    );
}

#[test]
fn test_xml_has_ffprobe_root() {
    let out = make_output();
    let xml = format_probe_result(&out, FfprobeOutputFormat::Xml).expect("xml");
    assert!(xml.contains("<ffprobe"), "should have <ffprobe> open tag");
    assert!(
        xml.contains("</ffprobe>"),
        "should have </ffprobe> close tag"
    );
}

#[test]
fn test_xml_contains_stream_element() {
    let out = make_output();
    let xml = format_probe_result(&out, FfprobeOutputFormat::Xml).expect("xml");
    assert!(xml.contains("<stream "), "should have <stream> element");
    assert!(
        xml.contains("codec_name=\"av1\""),
        "should have codec_name attr"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// CSV tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_csv_first_line_starts_with_stream() {
    let out = make_output();
    let csv = format_probe_result(&out, FfprobeOutputFormat::Csv).expect("csv");
    assert!(
        csv.starts_with("stream,"),
        "CSV stream row should start with 'stream,'"
    );
}

#[test]
fn test_csv_contains_format_row() {
    let out = make_output();
    let csv = format_probe_result(&out, FfprobeOutputFormat::Csv).expect("csv");
    assert!(csv.contains("format,"), "CSV should have a format row");
}

// ─────────────────────────────────────────────────────────────────────────────
// Default / flat tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_default_contains_codec_key() {
    let out = make_output();
    let flat = format_probe_result(&out, FfprobeOutputFormat::Default).expect("default");
    assert!(
        flat.contains("codec_name"),
        "flat output should contain codec_name"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// FfprobeOutputFormat enum tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_format_from_str_all_variants() {
    assert_eq!(
        FfprobeOutputFormat::from_str("json"),
        Some(FfprobeOutputFormat::Json)
    );
    assert_eq!(
        FfprobeOutputFormat::from_str("xml"),
        Some(FfprobeOutputFormat::Xml)
    );
    assert_eq!(
        FfprobeOutputFormat::from_str("csv"),
        Some(FfprobeOutputFormat::Csv)
    );
    assert_eq!(
        FfprobeOutputFormat::from_str("flat"),
        Some(FfprobeOutputFormat::Default)
    );
    assert_eq!(
        FfprobeOutputFormat::from_str("default"),
        Some(FfprobeOutputFormat::Default)
    );
    assert_eq!(FfprobeOutputFormat::from_str("unknown"), None);
}
