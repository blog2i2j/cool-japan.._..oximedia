//! Annotation export: serialize annotation collections to JSON and CSV.
//!
//! [`AnnotationExporter`] converts a slice of [`crate::annotations::Annotation`]
//! to portable string representations.  Both formats are produced without
//! external crate dependencies — the JSON emitter is a hand-written serialiser
//! that produces valid RFC 8259 output, and the CSV follows RFC 4180.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_review::annotation_export::AnnotationExporter;
//! use oximedia_review::annotations::{Annotation, AnnotationType};
//!
//! let annotations = vec![
//!     Annotation {
//!         id: 1,
//!         media_id: "clip-01".to_string(),
//!         timestamp_ms: 1000,
//!         duration_ms: 0,
//!         annotation_type: AnnotationType::Note,
//!         text: "Check colour grade".to_string(),
//!         author: "alice".to_string(),
//!         resolved: false,
//!     },
//! ];
//!
//! let json = AnnotationExporter::to_json(&annotations);
//! assert!(json.contains("clip-01"));
//!
//! let csv = AnnotationExporter::to_csv(&annotations);
//! assert!(csv.contains("alice"));
//! ```

use crate::annotations::{Annotation, AnnotationType};

// ─── AnnotationExporter ───────────────────────────────────────────────────────

/// Export a collection of annotations to common portable formats.
pub struct AnnotationExporter;

impl AnnotationExporter {
    /// Serialize a slice of annotations to a JSON array string.
    ///
    /// The output is a JSON array `[...]` where each element is an object with
    /// the following keys: `id`, `media_id`, `timestamp_ms`, `duration_ms`,
    /// `type`, `text`, `author`, `resolved`.
    ///
    /// All string values are JSON-escaped (backslash, double-quote, control
    /// characters).
    #[must_use]
    pub fn to_json(annotations: &[Annotation]) -> String {
        let mut out = String::from("[\n");
        for (i, a) in annotations.iter().enumerate() {
            let type_str = annotation_type_str(a.annotation_type);
            let comma = if i + 1 < annotations.len() { "," } else { "" };
            out.push_str(&format!(
                "  {{\n    \"id\": {},\n    \"media_id\": {},\n    \"timestamp_ms\": {},\n    \"duration_ms\": {},\n    \"type\": {},\n    \"text\": {},\n    \"author\": {},\n    \"resolved\": {}\n  }}{}\n",
                a.id,
                json_string(&a.media_id),
                a.timestamp_ms,
                a.duration_ms,
                json_string(type_str),
                json_string(&a.text),
                json_string(&a.author),
                if a.resolved { "true" } else { "false" },
                comma,
            ));
        }
        out.push(']');
        out
    }

    /// Serialize a slice of annotations to a CSV string (RFC 4180).
    ///
    /// The first row is a header: `id,media_id,timestamp_ms,duration_ms,type,text,author,resolved`.
    /// String fields are quoted and internal double-quotes are escaped as `""`.
    #[must_use]
    pub fn to_csv(annotations: &[Annotation]) -> String {
        let mut out =
            String::from("id,media_id,timestamp_ms,duration_ms,type,text,author,resolved\r\n");
        for a in annotations {
            let type_str = annotation_type_str(a.annotation_type);
            out.push_str(&format!(
                "{},{},{},{},{},{},{},{}\r\n",
                a.id,
                csv_field(&a.media_id),
                a.timestamp_ms,
                a.duration_ms,
                csv_field(type_str),
                csv_field(&a.text),
                csv_field(&a.author),
                if a.resolved { "true" } else { "false" },
            ));
        }
        out
    }

    /// Export to JSON and write to a `String`; same as `to_json` but provided
    /// for API symmetry with a potential async file-write variant.
    #[must_use]
    pub fn to_json_string(annotations: &[Annotation]) -> String {
        Self::to_json(annotations)
    }

    /// Export to CSV and write to a `String`; same as `to_csv` but provided
    /// for API symmetry.
    #[must_use]
    pub fn to_csv_string(annotations: &[Annotation]) -> String {
        Self::to_csv(annotations)
    }

    /// Count annotations by type.
    ///
    /// Returns a `Vec<(&str, usize)>` of `(type_name, count)` sorted by type name.
    #[must_use]
    pub fn count_by_type(annotations: &[Annotation]) -> Vec<(&'static str, usize)> {
        let types = [
            AnnotationType::Note,
            AnnotationType::Question,
            AnnotationType::Approval,
            AnnotationType::Rejection,
            AnnotationType::Correction,
            AnnotationType::Highlight,
        ];
        let mut counts: Vec<(&'static str, usize)> = types
            .iter()
            .map(|&t| {
                let count = annotations
                    .iter()
                    .filter(|a| a.annotation_type == t)
                    .count();
                (annotation_type_str(t), count)
            })
            .collect();
        counts.sort_by_key(|&(name, _)| name);
        counts
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Return a stable string label for an annotation type.
fn annotation_type_str(t: AnnotationType) -> &'static str {
    match t {
        AnnotationType::Note => "Note",
        AnnotationType::Question => "Question",
        AnnotationType::Approval => "Approval",
        AnnotationType::Rejection => "Rejection",
        AnnotationType::Correction => "Correction",
        AnnotationType::Highlight => "Highlight",
    }
}

/// Escape a string for inclusion as a JSON string value (with surrounding `"`).
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

/// Wrap a CSV field in double-quotes and escape internal double-quotes as `""`.
fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        let escaped = s.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        s.to_string()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotations::{Annotation, AnnotationType};

    fn make_annotation(
        id: u64,
        media_id: &str,
        t: AnnotationType,
        text: &str,
        author: &str,
    ) -> Annotation {
        Annotation {
            id,
            media_id: media_id.to_string(),
            timestamp_ms: id * 1000,
            duration_ms: 500,
            annotation_type: t,
            text: text.to_string(),
            author: author.to_string(),
            resolved: false,
        }
    }

    // ── to_json ───────────────────────────────────────────────────────────────

    #[test]
    fn json_empty_is_empty_array() {
        let json = AnnotationExporter::to_json(&[]);
        // The hand-rolled JSON always opens with "[\n" then closes with "]"
        assert!(json.contains('['));
        assert!(json.contains(']'));
        // After trimming whitespace the result should be "[]"
        let stripped: String = json.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(stripped, "[]");
    }

    #[test]
    fn json_contains_id_and_media() {
        let a = make_annotation(42, "clip-77", AnnotationType::Note, "check this", "bob");
        let json = AnnotationExporter::to_json(&[a]);
        assert!(json.contains("42"), "id missing: {json}");
        assert!(json.contains("clip-77"), "media_id missing: {json}");
    }

    #[test]
    fn json_escapes_quotes_in_text() {
        let mut a = make_annotation(1, "m1", AnnotationType::Note, r#"He said "hello""#, "alice");
        a.text = r#"He said "hello""#.to_string();
        let json = AnnotationExporter::to_json(&[a]);
        assert!(json.contains(r#"\""#), "quote not escaped: {json}");
    }

    #[test]
    fn json_multiple_annotations() {
        let annotations = vec![
            make_annotation(1, "m1", AnnotationType::Note, "first", "a"),
            make_annotation(2, "m2", AnnotationType::Question, "second", "b"),
        ];
        let json = AnnotationExporter::to_json(&annotations);
        assert!(json.contains("\"id\": 1"));
        assert!(json.contains("\"id\": 2"));
        // Should be a valid array: starts with '[' and ends with ']'
        assert!(json.trim_start().starts_with('['));
        assert!(json.trim_end().ends_with(']'));
    }

    #[test]
    fn json_resolved_field() {
        let mut a = make_annotation(1, "m", AnnotationType::Approval, "ok", "x");
        a.resolved = true;
        let json = AnnotationExporter::to_json(&[a]);
        assert!(json.contains("\"resolved\": true"));
    }

    // ── to_csv ────────────────────────────────────────────────────────────────

    #[test]
    fn csv_header_present() {
        let csv = AnnotationExporter::to_csv(&[]);
        assert!(csv.starts_with("id,media_id,timestamp_ms"));
    }

    #[test]
    fn csv_row_count_matches_annotation_count() {
        let annotations = vec![
            make_annotation(1, "m1", AnnotationType::Note, "a", "x"),
            make_annotation(2, "m2", AnnotationType::Correction, "b", "y"),
        ];
        let csv = AnnotationExporter::to_csv(&annotations);
        // 1 header + 2 data rows = 3 non-empty lines (CRLF terminated)
        let rows: Vec<&str> = csv.split("\r\n").filter(|l| !l.is_empty()).collect();
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn csv_fields_with_commas_are_quoted() {
        let mut a = make_annotation(1, "m", AnnotationType::Note, "hello, world", "author");
        a.text = "hello, world".to_string();
        let csv = AnnotationExporter::to_csv(&[a]);
        assert!(csv.contains("\"hello, world\""), "csv={csv}");
    }

    #[test]
    fn csv_double_quotes_are_escaped() {
        let mut a = make_annotation(1, "m", AnnotationType::Note, "say \"hi\"", "author");
        a.text = "say \"hi\"".to_string();
        let csv = AnnotationExporter::to_csv(&[a]);
        assert!(csv.contains("\"\""), "csv={csv}");
    }

    // ── count_by_type ─────────────────────────────────────────────────────────

    #[test]
    fn count_by_type_empty() {
        let counts = AnnotationExporter::count_by_type(&[]);
        let total: usize = counts.iter().map(|(_, c)| c).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn count_by_type_correct_counts() {
        let annotations = vec![
            make_annotation(1, "m", AnnotationType::Note, "a", "x"),
            make_annotation(2, "m", AnnotationType::Note, "b", "x"),
            make_annotation(3, "m", AnnotationType::Question, "c", "x"),
        ];
        let counts = AnnotationExporter::count_by_type(&annotations);
        let note_count = counts.iter().find(|&&(t, _)| t == "Note").map(|&(_, c)| c);
        let question_count = counts
            .iter()
            .find(|&&(t, _)| t == "Question")
            .map(|&(_, c)| c);
        assert_eq!(note_count, Some(2));
        assert_eq!(question_count, Some(1));
    }
}
