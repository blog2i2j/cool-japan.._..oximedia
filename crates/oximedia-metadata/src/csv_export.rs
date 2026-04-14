//! Bulk CSV export of metadata collections.
//!
//! [`MetadataCsvExporter`] serialises a slice of `(id, &MetadataMap)` pairs
//! into RFC 4180-compatible CSV.
//!
//! The first column is always `id`; subsequent columns are the **union** of
//! all field names found across all records, sorted alphabetically for
//! deterministic output.  Missing fields are exported as empty cells.

use crate::inherit::MetadataMap;
use std::collections::BTreeSet;

/// Exports metadata collections to CSV format.
pub struct MetadataCsvExporter;

impl MetadataCsvExporter {
    /// Export `items` as a CSV string.
    ///
    /// - First row: header (`id,<sorted field names...>`).
    /// - Subsequent rows: one row per item.
    /// - Values are quoted if they contain a comma, double-quote, or newline.
    pub fn export(items: &[(u64, &MetadataMap)]) -> String {
        // Collect the union of all field names.
        let fields: BTreeSet<&str> = items
            .iter()
            .flat_map(|(_, m)| m.keys().map(String::as_str))
            .collect();

        let fields: Vec<&str> = fields.into_iter().collect();

        let mut out = String::new();

        // Header row
        out.push_str("id");
        for f in &fields {
            out.push(',');
            out.push_str(&csv_quote(f));
        }
        out.push('\n');

        // Data rows
        for (id, map) in items {
            out.push_str(&id.to_string());
            for f in &fields {
                out.push(',');
                let value = map.get(*f).map(String::as_str).unwrap_or("");
                out.push_str(&csv_quote(value));
            }
            out.push('\n');
        }

        out
    }
}

/// Quote a CSV cell value if it contains special characters.
fn csv_quote(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        // Escape any double-quotes by doubling them.
        let escaped = value.replace('"', "\"\"");
        format!("\"{}\"", escaped)
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn map(pairs: &[(&str, &str)]) -> MetadataMap {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn test_empty_export() {
        let csv = MetadataCsvExporter::export(&[]);
        // Only header with just "id"
        assert_eq!(csv, "id\n");
    }

    #[test]
    fn test_single_item() {
        let m = map(&[("title", "My Song"), ("artist", "Artist")]);
        let csv = MetadataCsvExporter::export(&[(1, &m)]);
        // Header should be sorted: artist, title
        let lines: Vec<&str> = csv.trim_end().split('\n').collect();
        assert_eq!(lines[0], "id,artist,title");
        assert_eq!(lines[1], "1,Artist,My Song");
    }

    #[test]
    fn test_missing_fields_are_empty() {
        let m1 = map(&[("title", "A"), ("artist", "X")]);
        let m2 = map(&[("title", "B")]);
        let csv = MetadataCsvExporter::export(&[(1, &m1), (2, &m2)]);
        let lines: Vec<&str> = csv.trim_end().split('\n').collect();
        // Row 2 has empty artist (columns are sorted: artist, title)
        assert_eq!(lines[2], "2,,B");
    }

    #[test]
    fn test_comma_in_value_is_quoted() {
        let m = map(&[("title", "Hello, World")]);
        let csv = MetadataCsvExporter::export(&[(1, &m)]);
        assert!(csv.contains("\"Hello, World\""));
    }

    #[test]
    fn test_double_quote_escaped() {
        let m = map(&[("title", "Say \"Hi\"")]);
        let csv = MetadataCsvExporter::export(&[(1, &m)]);
        assert!(csv.contains("\"Say \"\"Hi\"\"\""));
    }

    #[test]
    fn test_multiple_items_union_of_fields() {
        let m1 = map(&[("a", "1")]);
        let m2 = map(&[("b", "2")]);
        let csv = MetadataCsvExporter::export(&[(1, &m1), (2, &m2)]);
        let lines: Vec<&str> = csv.trim_end().split('\n').collect();
        assert_eq!(lines[0], "id,a,b");
        assert_eq!(lines[1], "1,1,");
        assert_eq!(lines[2], "2,,2");
    }
}
