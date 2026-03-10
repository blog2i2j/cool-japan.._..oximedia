//! Subtitle parsing and conversion for the browser.

use oximedia_subtitle::{AssParser, SrtParser, Subtitle, WebVttParser};
use wasm_bindgen::prelude::*;

// ─── JSON serialisation helper ────────────────────────────────────────────────

/// Serialise a slice of subtitles to a JSON array string.
fn subtitles_to_json(subs: &[Subtitle]) -> Result<String, JsValue> {
    let mut json = String::from('[');
    for (i, sub) in subs.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        let id_json = match &sub.id {
            Some(id) => format!("\"{}\"", id.replace('"', "\\\"")),
            None => "null".to_string(),
        };
        let text_json = sub
            .text
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r");
        json.push_str(&format!(
            "{{\"id\":{id_json},\"start_ms\":{},\"end_ms\":{},\"text\":\"{}\"}}",
            sub.start_time, sub.end_time, text_json
        ));
    }
    json.push(']');
    Ok(json)
}

// ─── Parsing functions ────────────────────────────────────────────────────────

/// Parse SRT subtitle text and return a JSON array of cues.
///
/// JSON schema: `[{ "id": string|null, "start_ms": number, "end_ms": number, "text": string }]`
///
/// # Errors
///
/// Returns an error if the text is not valid SRT.
#[wasm_bindgen]
pub fn wasm_parse_srt(text: &str) -> Result<String, JsValue> {
    let subs = SrtParser::parse(text)
        .map_err(|e| crate::utils::js_err(&format!("SRT parse error: {e}")))?;
    subtitles_to_json(&subs)
}

/// Parse WebVTT subtitle text and return a JSON array of cues.
///
/// JSON schema: `[{ "id": string|null, "start_ms": number, "end_ms": number, "text": string }]`
///
/// # Errors
///
/// Returns an error if the text is not valid WebVTT.
#[wasm_bindgen]
pub fn wasm_parse_vtt(text: &str) -> Result<String, JsValue> {
    let subs = WebVttParser::parse(text)
        .map_err(|e| crate::utils::js_err(&format!("WebVTT parse error: {e}")))?;
    subtitles_to_json(&subs)
}

/// Parse ASS/SSA subtitle text and return a JSON array of cues.
///
/// JSON schema: `[{ "id": string|null, "start_ms": number, "end_ms": number, "text": string }]`
///
/// # Errors
///
/// Returns an error if the text is not valid ASS/SSA.
#[wasm_bindgen]
pub fn wasm_parse_ass(text: &str) -> Result<String, JsValue> {
    let subs = AssParser::parse(text)
        .map_err(|e| crate::utils::js_err(&format!("ASS parse error: {e}")))?;
    subtitles_to_json(&subs)
}

// ─── Format conversion ────────────────────────────────────────────────────────

/// Convert subtitles between formats.
///
/// `from_format`: `"srt"`, `"vtt"`, or `"ass"`
/// `to_format`:   `"srt"`, `"vtt"`, or `"ass"`
///
/// # Errors
///
/// Returns an error if the source format cannot be parsed or the target format
/// is unknown.
#[wasm_bindgen]
pub fn wasm_convert_subtitles(
    text: &str,
    from_format: &str,
    to_format: &str,
) -> Result<String, JsValue> {
    let subs = parse_by_format(text, from_format)?;
    serialize_to_format(&subs, to_format)
}

/// Shift subtitle timing by `offset_ms` milliseconds.
///
/// Subtitles whose start time would go negative are clamped to zero.
///
/// # Errors
///
/// Returns an error if the source text cannot be parsed.
#[wasm_bindgen]
pub fn wasm_shift_subtitles(text: &str, format: &str, offset_ms: i64) -> Result<String, JsValue> {
    let mut subs = parse_by_format(text, format)?;
    for sub in &mut subs {
        sub.start_time = (sub.start_time + offset_ms).max(0);
        sub.end_time = (sub.end_time + offset_ms).max(0);
    }
    serialize_to_format(&subs, format)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Parse subtitle text using the specified format name.
fn parse_by_format(text: &str, format: &str) -> Result<Vec<Subtitle>, JsValue> {
    match format.to_lowercase().as_str() {
        "srt" => SrtParser::parse(text)
            .map_err(|e| crate::utils::js_err(&format!("SRT parse error: {e}"))),
        "vtt" | "webvtt" => WebVttParser::parse(text)
            .map_err(|e| crate::utils::js_err(&format!("WebVTT parse error: {e}"))),
        "ass" | "ssa" => AssParser::parse(text)
            .map_err(|e| crate::utils::js_err(&format!("ASS parse error: {e}"))),
        other => Err(crate::utils::js_err(&format!(
            "Unknown source format: \"{other}\". Use \"srt\", \"vtt\", or \"ass\"."
        ))),
    }
}

/// Serialise a list of subtitles to the target format string.
fn serialize_to_format(subs: &[Subtitle], format: &str) -> Result<String, JsValue> {
    match format.to_lowercase().as_str() {
        "srt" => Ok(subs_to_srt(subs)),
        "vtt" | "webvtt" => Ok(subs_to_vtt(subs)),
        "ass" | "ssa" => Ok(subs_to_ass(subs)),
        other => Err(crate::utils::js_err(&format!(
            "Unknown target format: \"{other}\". Use \"srt\", \"vtt\", or \"ass\"."
        ))),
    }
}

// ─── Format serialisers ───────────────────────────────────────────────────────

/// Format milliseconds as `HH:MM:SS,mmm` (SRT).
fn ms_to_srt_time(ms: i64) -> String {
    let ms = ms.max(0) as u64;
    let millis = ms % 1000;
    let total_s = ms / 1000;
    let secs = total_s % 60;
    let total_m = total_s / 60;
    let mins = total_m % 60;
    let hours = total_m / 60;
    format!("{hours:02}:{mins:02}:{secs:02},{millis:03}")
}

/// Format milliseconds as `HH:MM:SS.mmm` (VTT/ASS).
fn ms_to_vtt_time(ms: i64) -> String {
    let ms = ms.max(0) as u64;
    let millis = ms % 1000;
    let total_s = ms / 1000;
    let secs = total_s % 60;
    let total_m = total_s / 60;
    let mins = total_m % 60;
    let hours = total_m / 60;
    format!("{hours:02}:{mins:02}:{secs:02}.{millis:03}")
}

/// Format milliseconds as `H:MM:SS.cc` (ASS centiseconds).
fn ms_to_ass_time(ms: i64) -> String {
    let ms = ms.max(0) as u64;
    let cs = (ms % 1000) / 10;
    let total_s = ms / 1000;
    let secs = total_s % 60;
    let total_m = total_s / 60;
    let mins = total_m % 60;
    let hours = total_m / 60;
    format!("{hours}:{mins:02}:{secs:02}.{cs:02}")
}

/// Serialise subtitles to SRT format.
fn subs_to_srt(subs: &[Subtitle]) -> String {
    let mut out = String::new();
    for (i, sub) in subs.iter().enumerate() {
        let seq = sub
            .id
            .as_deref()
            .unwrap_or("")
            .parse::<usize>()
            .unwrap_or(i + 1);
        out.push_str(&format!(
            "{seq}\n{} --> {}\n{}\n\n",
            ms_to_srt_time(sub.start_time),
            ms_to_srt_time(sub.end_time),
            sub.text
        ));
    }
    out
}

/// Serialise subtitles to WebVTT format.
fn subs_to_vtt(subs: &[Subtitle]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for sub in subs {
        if let Some(ref id) = sub.id {
            out.push_str(id);
            out.push('\n');
        }
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            ms_to_vtt_time(sub.start_time),
            ms_to_vtt_time(sub.end_time),
            sub.text
        ));
    }
    out
}

/// Build a minimal [`Subtitle`] for testing without importing internals.
#[cfg(test)]
fn make_sub(id: Option<&str>, start_ms: i64, end_ms: i64, text: &str) -> Subtitle {
    let sub = Subtitle::new(start_ms, end_ms, text.to_string());
    match id {
        Some(s) => sub.with_id(s),
        None => sub,
    }
}

/// Serialise subtitles to a minimal ASS/SSA format.
fn subs_to_ass(subs: &[Subtitle]) -> String {
    let mut out = String::new();
    out.push_str("[Script Info]\n");
    out.push_str("ScriptType: v4.00+\n");
    out.push_str("PlayResX: 1920\n");
    out.push_str("PlayResY: 1080\n\n");

    out.push_str("[V4+ Styles]\n");
    out.push_str("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n");
    out.push_str("Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1\n\n");

    out.push_str("[Events]\n");
    out.push_str(
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
    );
    for sub in subs {
        let text = sub.text.replace('\n', "\\N");
        out.push_str(&format!(
            "Dialogue: 0,{},{},Default,,0,0,0,,{text}\n",
            ms_to_ass_time(sub.start_time),
            ms_to_ass_time(sub.end_time),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Time-formatting helpers ───────────────────────────────────────────

    #[test]
    fn test_ms_to_srt_time_zero() {
        assert_eq!(ms_to_srt_time(0), "00:00:00,000");
    }

    #[test]
    fn test_ms_to_srt_time_one_hour() {
        // 3 600 000 ms = 1 hour exactly
        assert_eq!(ms_to_srt_time(3_600_000), "01:00:00,000");
    }

    #[test]
    fn test_ms_to_srt_time_complex() {
        // 1h 2m 3s 456ms
        let ms = 3_600_000 + 2 * 60_000 + 3_000 + 456;
        assert_eq!(ms_to_srt_time(ms), "01:02:03,456");
    }

    #[test]
    fn test_ms_to_srt_time_negative_clamps_to_zero() {
        // Negative input should be treated as 0.
        assert_eq!(ms_to_srt_time(-5000), "00:00:00,000");
    }

    #[test]
    fn test_ms_to_vtt_time_uses_dot_separator() {
        let ts = ms_to_vtt_time(1_234);
        assert!(
            ts.contains('.'),
            "VTT timestamp should use '.' separator, got: {ts}"
        );
        assert_eq!(ts, "00:00:01.234");
    }

    #[test]
    fn test_ms_to_ass_time_centiseconds() {
        // 1500 ms = 1s 50cs → "0:00:01.50"
        let ts = ms_to_ass_time(1_500);
        assert_eq!(ts, "0:00:01.50");
    }

    #[test]
    fn test_ms_to_ass_time_zero() {
        assert_eq!(ms_to_ass_time(0), "0:00:00.00");
    }

    // ─── subtitles_to_json ─────────────────────────────────────────────────

    #[test]
    fn test_subtitles_to_json_empty() {
        let json = subtitles_to_json(&[]).expect("subtitle conversion should succeed");
        assert_eq!(json, "[]");
    }

    #[test]
    fn test_subtitles_to_json_one_entry() {
        let subs = vec![make_sub(Some("1"), 1000, 3000, "Hello")];
        let json = subtitles_to_json(&subs).expect("subtitle conversion should succeed");
        assert!(json.contains("\"id\":\"1\""), "id missing: {json}");
        assert!(
            json.contains("\"start_ms\":1000"),
            "start_ms missing: {json}"
        );
        assert!(json.contains("\"end_ms\":3000"), "end_ms missing: {json}");
        assert!(json.contains("\"text\":\"Hello\""), "text missing: {json}");
    }

    #[test]
    fn test_subtitles_to_json_null_id() {
        let subs = vec![make_sub(None, 0, 500, "No id")];
        let json = subtitles_to_json(&subs).expect("subtitle conversion should succeed");
        assert!(json.contains("\"id\":null"), "expected null id: {json}");
    }

    #[test]
    fn test_subtitles_to_json_escapes_quotes_in_text() {
        let subs = vec![make_sub(None, 0, 1000, r#"Say "hello""#)];
        let json = subtitles_to_json(&subs).expect("subtitle conversion should succeed");
        assert!(
            json.contains(r#"Say \"hello\""#),
            "inner quotes not escaped: {json}"
        );
    }

    // ─── subs_to_srt ──────────────────────────────────────────────────────

    #[test]
    fn test_subs_to_srt_format() {
        let subs = vec![make_sub(Some("1"), 1000, 3000, "Hello world")];
        let srt = subs_to_srt(&subs);
        assert!(srt.contains("00:00:01,000 --> 00:00:03,000"));
        assert!(srt.contains("Hello world"));
    }

    #[test]
    fn test_subs_to_srt_sequence_numbers() {
        let subs = vec![
            make_sub(None, 0, 1000, "One"),
            make_sub(None, 1000, 2000, "Two"),
        ];
        let srt = subs_to_srt(&subs);
        // Sequence 1 appears at the very start or after a blank line.
        assert!(
            srt.starts_with("1\n") || srt.contains("\n1\n"),
            "seq 1 missing: {srt}"
        );
        // Sequence 2 is always preceded by a newline.
        assert!(srt.contains("\n2\n"), "seq 2 missing: {srt}");
    }

    // ─── subs_to_vtt ──────────────────────────────────────────────────────

    #[test]
    fn test_subs_to_vtt_header() {
        let vtt = subs_to_vtt(&[]);
        assert!(vtt.starts_with("WEBVTT"), "expected WEBVTT header: {vtt}");
    }

    #[test]
    fn test_subs_to_vtt_dot_timestamp() {
        let subs = vec![make_sub(Some("cue1"), 500, 1500, "VTT cue")];
        let vtt = subs_to_vtt(&subs);
        assert!(vtt.contains("00:00:00.500 --> 00:00:01.500"));
        assert!(vtt.contains("VTT cue"));
    }

    // ─── subs_to_ass ──────────────────────────────────────────────────────

    #[test]
    fn test_subs_to_ass_script_info_header() {
        let ass = subs_to_ass(&[]);
        assert!(ass.contains("[Script Info]"));
        assert!(ass.contains("ScriptType: v4.00+"));
    }

    #[test]
    fn test_subs_to_ass_dialogue_line() {
        let subs = vec![make_sub(None, 0, 2000, "ASS subtitle")];
        let ass = subs_to_ass(&subs);
        assert!(ass.contains("Dialogue:"), "Dialogue line missing: {ass}");
        assert!(ass.contains("ASS subtitle"), "text missing: {ass}");
    }

    #[test]
    fn test_subs_to_ass_newlines_replaced() {
        let subs = vec![make_sub(None, 0, 1000, "Line1\nLine2")];
        let ass = subs_to_ass(&subs);
        assert!(
            ass.contains("Line1\\NLine2"),
            "soft-newline not replaced: {ass}"
        );
    }

    // ─── format routing (supported formats only — error paths use JsValue) ─

    #[test]
    fn test_serialize_to_srt_non_empty() {
        let subs = vec![make_sub(Some("1"), 0, 1000, "test")];
        let result = serialize_to_format(&subs, "srt");
        assert!(result.is_ok(), "srt serialisation must succeed: {result:?}");
        assert!(result
            .expect("subtitle format conversion should succeed")
            .contains("00:00:00,000 --> 00:00:01,000"));
    }

    #[test]
    fn test_serialize_to_vtt_non_empty() {
        let subs = vec![make_sub(None, 500, 2000, "vtt test")];
        let result = serialize_to_format(&subs, "vtt");
        assert!(result.is_ok(), "vtt serialisation must succeed: {result:?}");
        assert!(result
            .expect("subtitle format conversion should succeed")
            .starts_with("WEBVTT"));
    }

    #[test]
    fn test_serialize_to_ass_non_empty() {
        let subs = vec![make_sub(None, 0, 1500, "ass test")];
        let result = serialize_to_format(&subs, "ass");
        assert!(result.is_ok(), "ass serialisation must succeed: {result:?}");
        assert!(result
            .expect("subtitle format conversion should succeed")
            .contains("[Script Info]"));
    }

    // ─── shift logic ──────────────────────────────────────────────────────

    #[test]
    fn test_shift_positive() {
        let mut sub = make_sub(None, 1000, 3000, "x");
        sub.start_time = (sub.start_time + 500).max(0);
        sub.end_time = (sub.end_time + 500).max(0);
        assert_eq!(sub.start_time, 1500);
        assert_eq!(sub.end_time, 3500);
    }

    #[test]
    fn test_shift_negative_clamps_to_zero() {
        let mut sub = make_sub(None, 200, 1000, "x");
        let offset: i64 = -500;
        sub.start_time = (sub.start_time + offset).max(0);
        sub.end_time = (sub.end_time + offset).max(0);
        // start_time: 200 - 500 = -300 → clamped to 0
        assert_eq!(sub.start_time, 0);
        // end_time: 1000 - 500 = 500
        assert_eq!(sub.end_time, 500);
    }
}
