//! JSON serialization and deserialization for `Metadata`.
//!
//! Provides a hand-written JSON encoder and decoder that avoids a dependency on
//! `serde_json`, using only the already-imported `serde` derive macros together
//! with careful string building and parsing.
//!
//! # Supported value types
//!
//! | `MetadataValue` variant | JSON representation               |
//! |-------------------------|-----------------------------------|
//! | `Text(s)`               | `{"type":"text","value":"..."}`   |
//! | `TextList(v)`           | `{"type":"text_list","value":[…]}`|
//! | `Binary(b)`             | `{"type":"binary","value":"<hex>"}` (lowercase hex) |
//! | `Integer(n)`            | `{"type":"integer","value":N}`    |
//! | `Float(f)`              | `{"type":"float","value":F}`      |
//! | `Boolean(b)`            | `{"type":"boolean","value":true/false}` |
//! | `DateTime(s)`           | `{"type":"datetime","value":"…"}` |
//! | `Picture(_)`            | `{"type":"picture","value":null}` (data omitted) |
//! | `Pictures(_)`           | `{"type":"pictures","value":null}`|
//!
//! Pictures are intentionally omitted from the JSON representation to keep
//! payloads manageable. Binary data is encoded as a lowercase hex string.
//!
//! # Example
//!
//! ```
//! use oximedia_metadata::{Metadata, MetadataFormat, MetadataValue};
//! use oximedia_metadata::metadata_json::{to_json, from_json};
//!
//! let mut m = Metadata::new(MetadataFormat::Id3v2);
//! m.insert("TIT2".to_string(), MetadataValue::Text("My Song".to_string()));
//! m.insert("TLEN".to_string(), MetadataValue::Integer(220000));
//!
//! let json = to_json(&m).unwrap();
//! let m2 = from_json(&json).unwrap();
//!
//! assert_eq!(
//!     m2.get("TIT2").and_then(MetadataValue::as_text),
//!     Some("My Song")
//! );
//! ```

use crate::{Error, Metadata, MetadataFormat, MetadataValue};
use std::collections::HashMap;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Escape a string for inclusion inside a JSON string literal.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r"\\"),
            '\n' => out.push_str(r"\n"),
            '\r' => out.push_str(r"\r"),
            '\t' => out.push_str(r"\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!(r"\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Encode binary data as a lowercase hex string.
fn to_hex(data: &[u8]) -> String {
    data.iter().map(|b| format!("{b:02x}")).collect()
}

/// Decode a lowercase hex string into bytes.
fn from_hex(s: &str) -> Result<Vec<u8>, Error> {
    if s.len() % 2 != 0 {
        return Err(Error::ParseError("hex string has odd length".to_string()));
    }
    (0..s.len() / 2)
        .map(|i| {
            u8::from_str_radix(&s[i * 2..i * 2 + 2], 16)
                .map_err(|_| Error::ParseError(format!("invalid hex byte at position {}", i * 2)))
        })
        .collect()
}

// ─── Serialization ────────────────────────────────────────────────────────────

/// Serialize a `MetadataValue` to a JSON object fragment.
fn value_to_json(value: &MetadataValue) -> String {
    match value {
        MetadataValue::Text(s) => {
            format!(r#"{{"type":"text","value":"{}"}}"#, json_escape(s))
        }
        MetadataValue::TextList(list) => {
            let items: Vec<String> = list
                .iter()
                .map(|s| format!(r#""{}""#, json_escape(s)))
                .collect();
            format!(r#"{{"type":"text_list","value":[{}]}}"#, items.join(","))
        }
        MetadataValue::Binary(data) => {
            format!(r#"{{"type":"binary","value":"{}"}}"#, to_hex(data))
        }
        MetadataValue::Integer(n) => {
            format!(r#"{{"type":"integer","value":{n}}}"#)
        }
        MetadataValue::Float(f) => {
            // Use enough precision to round-trip f64
            format!(r#"{{"type":"float","value":{f:.17e}}}"#)
        }
        MetadataValue::Boolean(b) => {
            format!(r#"{{"type":"boolean","value":{b}}}"#)
        }
        MetadataValue::DateTime(dt) => {
            format!(r#"{{"type":"datetime","value":"{}"}}"#, json_escape(dt))
        }
        MetadataValue::Picture(_) => {
            r#"{"type":"picture","value":null}"#.to_string()
        }
        MetadataValue::Pictures(_) => {
            r#"{"type":"pictures","value":null}"#.to_string()
        }
    }
}

/// Serialize a `MetadataFormat` to a JSON string.
fn format_to_json(format: MetadataFormat) -> &'static str {
    match format {
        MetadataFormat::Id3v2 => "id3v2",
        MetadataFormat::VorbisComments => "vorbis",
        MetadataFormat::Apev2 => "apev2",
        MetadataFormat::iTunes => "itunes",
        MetadataFormat::Xmp => "xmp",
        MetadataFormat::Exif => "exif",
        MetadataFormat::Iptc => "iptc",
        MetadataFormat::QuickTime => "quicktime",
        MetadataFormat::Matroska => "matroska",
    }
}

/// Serialize a `Metadata` container to a JSON string.
///
/// # Errors
///
/// Returns an error if any field key contains characters that cannot be
/// safely embedded in JSON (this is not expected in practice).
pub fn to_json(metadata: &Metadata) -> Result<String, Error> {
    let mut out = String::new();
    out.push_str(r#"{"format":""#);
    out.push_str(format_to_json(metadata.format()));
    out.push_str(r#"","fields":{"#);

    let mut first = true;
    // Collect and sort keys for stable output
    let mut keys: Vec<&String> = metadata.fields().keys().collect();
    keys.sort();

    for key in keys {
        if !first {
            out.push(',');
        }
        first = false;
        out.push('"');
        out.push_str(&json_escape(key));
        out.push_str("\":");
        let value = metadata
            .fields()
            .get(key)
            .ok_or_else(|| Error::WriteError("missing field during iteration".to_string()))?;
        out.push_str(&value_to_json(value));
    }

    out.push_str("}}");
    Ok(out)
}

// ─── Deserialization ──────────────────────────────────────────────────────────

/// Minimal JSON tokenizer / parser state.
struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src: src.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let b = self.src.get(self.pos).copied();
        if b.is_some() {
            self.pos += 1;
        }
        b
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, ch: u8) -> Result<(), Error> {
        self.skip_ws();
        match self.advance() {
            Some(b) if b == ch => Ok(()),
            Some(b) => Err(Error::ParseError(format!(
                "expected '{}' but got '{}' at position {}",
                ch as char, b as char, self.pos
            ))),
            None => Err(Error::ParseError(format!(
                "expected '{}' but reached end of input",
                ch as char
            ))),
        }
    }

    fn parse_string(&mut self) -> Result<String, Error> {
        self.skip_ws();
        self.expect(b'"')?;
        let mut s = String::new();
        loop {
            match self.advance() {
                None => return Err(Error::ParseError("unterminated string".to_string())),
                Some(b'"') => break,
                Some(b'\\') => {
                    match self.advance() {
                        Some(b'"') => s.push('"'),
                        Some(b'\\') => s.push('\\'),
                        Some(b'n') => s.push('\n'),
                        Some(b'r') => s.push('\r'),
                        Some(b't') => s.push('\t'),
                        Some(b'u') => {
                            // 4 hex digits
                            let mut code_buf = [0u8; 4];
                            for slot in &mut code_buf {
                                *slot = self.advance().ok_or_else(|| {
                                    Error::ParseError("unterminated \\u escape".to_string())
                                })?;
                            }
                            let hex_str = std::str::from_utf8(&code_buf).map_err(|_| {
                                Error::ParseError("invalid unicode escape".to_string())
                            })?;
                            let codepoint = u32::from_str_radix(hex_str, 16).map_err(|_| {
                                Error::ParseError(format!("invalid \\u{hex_str}"))
                            })?;
                            let ch = char::from_u32(codepoint).ok_or_else(|| {
                                Error::ParseError(format!("invalid codepoint {codepoint}"))
                            })?;
                            s.push(ch);
                        }
                        Some(b) => s.push(b as char),
                        None => {
                            return Err(Error::ParseError("unterminated escape".to_string()))
                        }
                    }
                }
                Some(b) => s.push(b as char),
            }
        }
        Ok(s)
    }

    fn parse_number_raw(&mut self) -> Result<String, Error> {
        self.skip_ws();
        let start = self.pos;
        // Consume optional sign
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(b) = self.peek() {
            if b.is_ascii_digit()
                || b == b'.'
                || b == b'e'
                || b == b'E'
                || b == b'+'
                || b == b'-'
            {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(Error::ParseError("expected number".to_string()));
        }
        std::str::from_utf8(&self.src[start..self.pos])
            .map(str::to_string)
            .map_err(|_| Error::ParseError("invalid UTF-8 in number".to_string()))
    }

    fn parse_bool(&mut self) -> Result<bool, Error> {
        self.skip_ws();
        if self.src[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(true)
        } else if self.src[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(false)
        } else {
            Err(Error::ParseError("expected boolean".to_string()))
        }
    }

    fn skip_null(&mut self) -> Result<(), Error> {
        self.skip_ws();
        if self.src[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(())
        } else {
            Err(Error::ParseError("expected null".to_string()))
        }
    }

    /// Parse `"type":"<type>","value":<value>` inside an already-opened `{`.
    fn parse_metadata_value(&mut self) -> Result<MetadataValue, Error> {
        // Expect "type" key
        self.skip_ws();
        let key1 = self.parse_string()?;
        if key1 != "type" {
            return Err(Error::ParseError(format!(
                "expected 'type' key, got '{key1}'"
            )));
        }
        self.expect(b':')?;
        let type_str = self.parse_string()?;
        self.expect(b',')?;

        // Expect "value" key
        self.skip_ws();
        let key2 = self.parse_string()?;
        if key2 != "value" {
            return Err(Error::ParseError(format!(
                "expected 'value' key, got '{key2}'"
            )));
        }
        self.expect(b':')?;
        self.skip_ws();

        let value = match type_str.as_str() {
            "text" => MetadataValue::Text(self.parse_string()?),
            "text_list" => {
                self.expect(b'[')?;
                let mut list = Vec::new();
                self.skip_ws();
                if self.peek() != Some(b']') {
                    loop {
                        list.push(self.parse_string()?);
                        self.skip_ws();
                        match self.peek() {
                            Some(b',') => {
                                self.pos += 1;
                            }
                            _ => break,
                        }
                    }
                }
                self.expect(b']')?;
                MetadataValue::TextList(list)
            }
            "binary" => {
                let hex = self.parse_string()?;
                MetadataValue::Binary(from_hex(&hex)?)
            }
            "integer" => {
                let raw = self.parse_number_raw()?;
                let n: i64 = raw.parse().map_err(|_| {
                    Error::ParseError(format!("invalid integer: {raw}"))
                })?;
                MetadataValue::Integer(n)
            }
            "float" => {
                let raw = self.parse_number_raw()?;
                let f: f64 = raw.parse().map_err(|_| {
                    Error::ParseError(format!("invalid float: {raw}"))
                })?;
                MetadataValue::Float(f)
            }
            "boolean" => MetadataValue::Boolean(self.parse_bool()?),
            "datetime" => MetadataValue::DateTime(self.parse_string()?),
            "picture" | "pictures" => {
                self.skip_null()?;
                // Pictures are not round-trippable; return empty binary
                MetadataValue::Binary(Vec::new())
            }
            other => {
                return Err(Error::ParseError(format!(
                    "unknown metadata value type: {other}"
                )));
            }
        };

        Ok(value)
    }
}

/// Parse a format string into a `MetadataFormat`.
fn parse_format(s: &str) -> Result<MetadataFormat, Error> {
    match s {
        "id3v2" => Ok(MetadataFormat::Id3v2),
        "vorbis" => Ok(MetadataFormat::VorbisComments),
        "apev2" => Ok(MetadataFormat::Apev2),
        "itunes" => Ok(MetadataFormat::iTunes),
        "xmp" => Ok(MetadataFormat::Xmp),
        "exif" => Ok(MetadataFormat::Exif),
        "iptc" => Ok(MetadataFormat::Iptc),
        "quicktime" => Ok(MetadataFormat::QuickTime),
        "matroska" => Ok(MetadataFormat::Matroska),
        other => Err(Error::ParseError(format!("unknown format: {other}"))),
    }
}

/// Deserialize a `Metadata` container from a JSON string produced by [`to_json`].
///
/// # Errors
///
/// Returns an error if the JSON is malformed or contains unknown types.
pub fn from_json(json: &str) -> Result<Metadata, Error> {
    let mut parser = Parser::new(json);
    parser.expect(b'{')?;

    let mut format: Option<MetadataFormat> = None;
    let mut fields: HashMap<String, MetadataValue> = HashMap::new();

    loop {
        parser.skip_ws();
        let key = parser.parse_string()?;
        parser.expect(b':')?;

        match key.as_str() {
            "format" => {
                let fmt_str = parser.parse_string()?;
                format = Some(parse_format(&fmt_str)?);
            }
            "fields" => {
                parser.expect(b'{')?;
                parser.skip_ws();
                if parser.peek() != Some(b'}') {
                    loop {
                        let field_key = parser.parse_string()?;
                        parser.expect(b':')?;
                        parser.expect(b'{')?;
                        let value = parser.parse_metadata_value()?;
                        parser.expect(b'}')?;
                        fields.insert(field_key, value);

                        parser.skip_ws();
                        match parser.peek() {
                            Some(b',') => {
                                parser.pos += 1;
                            }
                            _ => break,
                        }
                    }
                }
                parser.expect(b'}')?;
            }
            other => {
                return Err(Error::ParseError(format!(
                    "unexpected top-level key: {other}"
                )));
            }
        }

        parser.skip_ws();
        match parser.peek() {
            Some(b',') => {
                parser.pos += 1;
            }
            _ => break,
        }
    }

    parser.expect(b'}')?;

    let format = format
        .ok_or_else(|| Error::ParseError("missing 'format' field".to_string()))?;
    let mut metadata = Metadata::new(format);
    for (k, v) in fields {
        metadata.insert(k, v);
    }
    Ok(metadata)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Metadata, MetadataFormat, MetadataValue};

    fn round_trip(m: &Metadata) -> Metadata {
        let json = to_json(m).expect("to_json should succeed in test");
        from_json(&json).expect("from_json should succeed in test")
    }

    #[test]
    fn test_serialize_text_field() {
        let mut m = Metadata::new(MetadataFormat::Id3v2);
        m.insert("TIT2".to_string(), MetadataValue::Text("My Song".to_string()));
        let json = to_json(&m).expect("should succeed in test");
        assert!(json.contains(r#""type":"text""#));
        assert!(json.contains("My Song"));
    }

    #[test]
    fn test_round_trip_text() {
        let mut m = Metadata::new(MetadataFormat::Id3v2);
        m.insert("TIT2".to_string(), MetadataValue::Text("My Song".to_string()));
        let m2 = round_trip(&m);
        assert_eq!(
            m2.get("TIT2").and_then(MetadataValue::as_text),
            Some("My Song")
        );
    }

    #[test]
    fn test_round_trip_integer() {
        let mut m = Metadata::new(MetadataFormat::Matroska);
        m.insert("duration".to_string(), MetadataValue::Integer(180_000));
        let m2 = round_trip(&m);
        assert_eq!(m2.get("duration").and_then(MetadataValue::as_integer), Some(180_000));
    }

    #[test]
    fn test_round_trip_float() {
        let mut m = Metadata::new(MetadataFormat::Exif);
        m.insert("latitude".to_string(), MetadataValue::Float(48.8566));
        let m2 = round_trip(&m);
        let f = m2.get("latitude").and_then(MetadataValue::as_float).expect("should succeed in test");
        assert!((f - 48.8566_f64).abs() < 1e-10);
    }

    #[test]
    fn test_round_trip_boolean() {
        let mut m = Metadata::new(MetadataFormat::iTunes);
        m.insert("compilation".to_string(), MetadataValue::Boolean(true));
        let m2 = round_trip(&m);
        assert_eq!(m2.get("compilation").and_then(MetadataValue::as_boolean), Some(true));
    }

    #[test]
    fn test_round_trip_binary() {
        let mut m = Metadata::new(MetadataFormat::Apev2);
        m.insert("cover".to_string(), MetadataValue::Binary(vec![0xDE, 0xAD, 0xBE, 0xEF]));
        let m2 = round_trip(&m);
        assert_eq!(
            m2.get("cover").and_then(MetadataValue::as_binary),
            Some(&[0xDE_u8, 0xAD, 0xBE, 0xEF][..])
        );
    }

    #[test]
    fn test_round_trip_text_list() {
        let mut m = Metadata::new(MetadataFormat::VorbisComments);
        m.insert(
            "GENRE".to_string(),
            MetadataValue::TextList(vec!["Rock".to_string(), "Blues".to_string()]),
        );
        let m2 = round_trip(&m);
        match m2.get("GENRE") {
            Some(MetadataValue::TextList(list)) => {
                assert_eq!(list, &["Rock", "Blues"]);
            }
            other => panic!("expected TextList, got {other:?}"),
        }
    }

    #[test]
    fn test_round_trip_datetime() {
        let mut m = Metadata::new(MetadataFormat::Xmp);
        m.insert("CreateDate".to_string(), MetadataValue::DateTime("2026-03-11T12:00:00Z".to_string()));
        let m2 = round_trip(&m);
        assert_eq!(
            m2.get("CreateDate").and_then(MetadataValue::as_datetime),
            Some("2026-03-11T12:00:00Z")
        );
    }

    #[test]
    fn test_round_trip_format_preserved() {
        let mut m = Metadata::new(MetadataFormat::QuickTime);
        m.insert("title".to_string(), MetadataValue::Text("Film".to_string()));
        let m2 = round_trip(&m);
        assert_eq!(m2.format(), MetadataFormat::QuickTime);
    }

    #[test]
    fn test_round_trip_empty_metadata() {
        let m = Metadata::new(MetadataFormat::Iptc);
        let m2 = round_trip(&m);
        assert_eq!(m2.fields().len(), 0);
        assert_eq!(m2.format(), MetadataFormat::Iptc);
    }

    #[test]
    fn test_json_escape_special_chars() {
        let mut m = Metadata::new(MetadataFormat::Id3v2);
        m.insert(
            "comment".to_string(),
            MetadataValue::Text("Hello \"World\"\nLine2".to_string()),
        );
        let json = to_json(&m).expect("should succeed in test");
        // Should not contain unescaped newline or quote
        assert!(json.contains(r#"\""#));
        assert!(json.contains(r"\n"));
        let m2 = from_json(&json).expect("should succeed in test");
        assert_eq!(
            m2.get("comment").and_then(MetadataValue::as_text),
            Some("Hello \"World\"\nLine2")
        );
    }

    #[test]
    fn test_from_json_invalid_format() {
        let json = r#"{"format":"unknown","fields":{}}"#;
        assert!(from_json(json).is_err());
    }

    #[test]
    fn test_from_json_missing_format() {
        let json = r#"{"fields":{}}"#;
        assert!(from_json(json).is_err());
    }

    #[test]
    fn test_multiple_fields_round_trip() {
        let mut m = Metadata::new(MetadataFormat::Id3v2);
        m.insert("TIT2".to_string(), MetadataValue::Text("Song".to_string()));
        m.insert("TPE1".to_string(), MetadataValue::Text("Artist".to_string()));
        m.insert("TRCK".to_string(), MetadataValue::Integer(7));
        let m2 = round_trip(&m);
        assert_eq!(m2.fields().len(), 3);
        assert_eq!(m2.get("TRCK").and_then(MetadataValue::as_integer), Some(7));
    }
}
