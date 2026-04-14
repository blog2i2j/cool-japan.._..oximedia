//! TranscodeProfile import and export in JSON format.
//!
//! This module provides a self-contained, dependency-free implementation for
//! serialising and deserialising [`TranscodeProfileExport`] to/from a compact
//! JSON object.  No external JSON crate is required.
//!
//! # JSON format
//!
//! ```json
//! {
//!   "name": "youtube_1080p",
//!   "video_codec": "vp9",
//!   "audio_codec": "opus",
//!   "video_bitrate": "5000k",
//!   "audio_bitrate": "128k",
//!   "preset": "good"
//! }
//! ```

use std::io::Write as _;

/// A portable, serialisable representation of a transcode profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscodeProfileExport {
    /// Human-readable profile name.
    pub name: String,
    /// Video codec identifier (e.g. `"vp9"`, `"av1"`).
    pub video_codec: String,
    /// Audio codec identifier (e.g. `"opus"`, `"flac"`).
    pub audio_codec: String,
    /// Video bitrate string (e.g. `"5000k"`, `"2M"`).
    pub video_bitrate: String,
    /// Audio bitrate string (e.g. `"128k"`).
    pub audio_bitrate: String,
    /// Encoder preset string (e.g. `"good"`, `"best"`, `"realtime"`).
    pub preset: String,
}

impl TranscodeProfileExport {
    /// Creates a new profile export record.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        video_codec: impl Into<String>,
        audio_codec: impl Into<String>,
        video_bitrate: impl Into<String>,
        audio_bitrate: impl Into<String>,
        preset: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            video_codec: video_codec.into(),
            audio_codec: audio_codec.into(),
            video_bitrate: video_bitrate.into(),
            audio_bitrate: audio_bitrate.into(),
            preset: preset.into(),
        }
    }

    /// Serialises this profile to a JSON string.
    ///
    /// All string values are JSON-escaped.  The output is a single-object JSON
    /// document with six keys in a deterministic order.
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            "{{\n  \"name\": {},\n  \"video_codec\": {},\n  \"audio_codec\": {},\n  \"video_bitrate\": {},\n  \"audio_bitrate\": {},\n  \"preset\": {}\n}}",
            json_string(&self.name),
            json_string(&self.video_codec),
            json_string(&self.audio_codec),
            json_string(&self.video_bitrate),
            json_string(&self.audio_bitrate),
            json_string(&self.preset),
        )
    }

    /// Parses a [`TranscodeProfileExport`] from a JSON string.
    ///
    /// Accepts any ordering of the six required keys and ignores unknown keys.
    /// Returns an error string if any required key is missing or the JSON is
    /// malformed.
    pub fn from_json(s: &str) -> Result<Self, String> {
        import_profile_from_json(s)
    }
}

/// Writes a profile to a file at `path` in JSON format.
///
/// # Errors
///
/// Returns an [`std::io::Error`] if the file cannot be created or written.
pub fn export_profile_to_file(
    profile: &TranscodeProfileExport,
    path: &str,
) -> std::io::Result<()> {
    let json = profile.to_json();
    let mut file = std::fs::File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

/// Parses a [`TranscodeProfileExport`] from a JSON string.
///
/// This is a deliberately minimal parser that only understands flat
/// `"key": "value"` JSON objects — sufficient for the six-field profile
/// format.  It does not handle nested objects, arrays, numbers, booleans,
/// or `null`.
///
/// # Errors
///
/// Returns an error string if a required field is missing or input is malformed.
pub fn import_profile_from_json(json: &str) -> Result<TranscodeProfileExport, String> {
    let mut name = None::<String>;
    let mut video_codec = None::<String>;
    let mut audio_codec = None::<String>;
    let mut video_bitrate = None::<String>;
    let mut audio_bitrate = None::<String>;
    let mut preset = None::<String>;

    // Iterate over `"key": "value"` pairs.
    // Strategy: scan for quoted tokens in key position, then value position.
    let chars: Vec<char> = json.chars().collect();
    let len = chars.len();
    let mut i = 0usize;

    while i < len {
        // Skip whitespace and structural characters until we find a '"' (key).
        if chars[i] != '"' {
            i += 1;
            continue;
        }

        // Parse key.
        let key = match parse_quoted_string(&chars, i) {
            Some((k, end)) => {
                i = end;
                k
            }
            None => return Err(format!("Malformed JSON key at position {i}")),
        };

        // Skip whitespace and the mandatory ':'.
        while i < len && (chars[i] == ' ' || chars[i] == '\t' || chars[i] == '\n' || chars[i] == '\r') {
            i += 1;
        }
        if i >= len || chars[i] != ':' {
            return Err(format!("Expected ':' after key \"{}\" at position {}", key, i));
        }
        i += 1; // consume ':'

        // Skip whitespace.
        while i < len && (chars[i] == ' ' || chars[i] == '\t' || chars[i] == '\n' || chars[i] == '\r') {
            i += 1;
        }

        // Parse value (must be a quoted string for our format).
        if i >= len || chars[i] != '"' {
            // Non-string value — skip until next ',' or '}'.
            while i < len && chars[i] != ',' && chars[i] != '}' {
                i += 1;
            }
            continue;
        }

        let value = match parse_quoted_string(&chars, i) {
            Some((v, end)) => {
                i = end;
                v
            }
            None => return Err(format!("Malformed JSON value for key \"{}\" at position {}", key, i)),
        };

        match key.as_str() {
            "name" => name = Some(value),
            "video_codec" => video_codec = Some(value),
            "audio_codec" => audio_codec = Some(value),
            "video_bitrate" => video_bitrate = Some(value),
            "audio_bitrate" => audio_bitrate = Some(value),
            "preset" => preset = Some(value),
            _ => {} // ignore unknown keys
        }
    }

    Ok(TranscodeProfileExport {
        name: name.ok_or("missing field: name")?,
        video_codec: video_codec.ok_or("missing field: video_codec")?,
        audio_codec: audio_codec.ok_or("missing field: audio_codec")?,
        video_bitrate: video_bitrate.ok_or("missing field: video_bitrate")?,
        audio_bitrate: audio_bitrate.ok_or("missing field: audio_bitrate")?,
        preset: preset.ok_or("missing field: preset")?,
    })
}

// ─── internal helpers ─────────────────────────────────────────────────────────

/// Wraps `s` in JSON double quotes, escaping `\`, `"`, and control characters.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Parses a JSON-quoted string starting at `chars[start]` (which must be `"`).
///
/// Returns `(string_value, index_after_closing_quote)` or `None` on error.
fn parse_quoted_string(chars: &[char], start: usize) -> Option<(String, usize)> {
    if chars.get(start) != Some(&'"') {
        return None;
    }
    let mut out = String::new();
    let mut i = start + 1;
    while i < chars.len() {
        match chars[i] {
            '"' => {
                return Some((out, i + 1));
            }
            '\\' => {
                i += 1;
                match chars.get(i) {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('/') => out.push('/'),
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some('u') => {
                        // \uXXXX
                        let hex: String = chars
                            .get(i + 1..i + 5)
                            .unwrap_or(&[])
                            .iter()
                            .collect();
                        if let Ok(code) = u32::from_str_radix(&hex, 16) {
                            if let Some(c) = char::from_u32(code) {
                                out.push(c);
                            }
                        }
                        i += 4;
                    }
                    Some(&c) => out.push(c),
                    None => return None,
                }
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    None // unterminated string
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_profile() -> TranscodeProfileExport {
        TranscodeProfileExport::new(
            "youtube_1080p",
            "vp9",
            "opus",
            "5000k",
            "128k",
            "good",
        )
    }

    // ── to_json ────────────────────────────────────────────────────────────

    #[test]
    fn test_to_json_contains_all_keys() {
        let json = sample_profile().to_json();
        assert!(json.contains("\"name\""), "missing name key");
        assert!(json.contains("\"video_codec\""), "missing video_codec key");
        assert!(json.contains("\"audio_codec\""), "missing audio_codec key");
        assert!(json.contains("\"video_bitrate\""), "missing video_bitrate key");
        assert!(json.contains("\"audio_bitrate\""), "missing audio_bitrate key");
        assert!(json.contains("\"preset\""), "missing preset key");
    }

    #[test]
    fn test_to_json_values_present() {
        let json = sample_profile().to_json();
        assert!(json.contains("\"youtube_1080p\""));
        assert!(json.contains("\"vp9\""));
        assert!(json.contains("\"opus\""));
        assert!(json.contains("\"5000k\""));
        assert!(json.contains("\"128k\""));
        assert!(json.contains("\"good\""));
    }

    #[test]
    fn test_to_json_starts_and_ends_with_braces() {
        let json = sample_profile().to_json();
        let trimmed = json.trim();
        assert!(trimmed.starts_with('{'));
        assert!(trimmed.ends_with('}'));
    }

    // ── from_json / round-trip ─────────────────────────────────────────────

    #[test]
    fn test_round_trip() {
        let original = sample_profile();
        let json = original.to_json();
        let parsed = TranscodeProfileExport::from_json(&json)
            .expect("round-trip parse failed");
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_from_json_missing_field_returns_error() {
        let bad_json = r#"{"name":"test","video_codec":"vp9","audio_codec":"opus","video_bitrate":"1M","audio_bitrate":"64k"}"#;
        let result = TranscodeProfileExport::from_json(bad_json);
        assert!(result.is_err(), "expected error for missing 'preset' field");
    }

    #[test]
    fn test_from_json_ignores_extra_keys() {
        let json = r#"{
          "name": "test",
          "video_codec": "av1",
          "audio_codec": "flac",
          "video_bitrate": "8M",
          "audio_bitrate": "256k",
          "preset": "best",
          "unknown_key": "ignored"
        }"#;
        let parsed = TranscodeProfileExport::from_json(json).expect("parse failed");
        assert_eq!(parsed.video_codec, "av1");
        assert_eq!(parsed.preset, "best");
    }

    #[test]
    fn test_json_escape_special_chars() {
        let profile = TranscodeProfileExport::new(
            "test\"quote",
            "vp9",
            "opus",
            "1M",
            "64k",
            "good",
        );
        let json = profile.to_json();
        let parsed = TranscodeProfileExport::from_json(&json).expect("parse with escaped chars");
        assert_eq!(parsed.name, "test\"quote");
    }

    // ── export_profile_to_file ─────────────────────────────────────────────

    #[test]
    fn test_export_and_reimport_via_file() {
        let profile = sample_profile();
        let tmp_path = std::env::temp_dir().join("oximedia_profile_test.json");
        let path_str = tmp_path.to_string_lossy().into_owned();

        export_profile_to_file(&profile, &path_str)
            .expect("export to file failed");

        let contents = std::fs::read_to_string(&tmp_path).expect("read file");
        let reimported = TranscodeProfileExport::from_json(&contents)
            .expect("reimport failed");

        assert_eq!(reimported, profile);
        let _ = std::fs::remove_file(&tmp_path);
    }
}
