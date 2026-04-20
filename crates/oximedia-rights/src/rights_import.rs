//! Bulk rights data import from CSV and JSON formats.
//!
//! Provides parsers that convert tabular or structured rights data into
//! [`ImportedRight`] records that can then be inserted into any rights store.
//! Both formats share the same output type so downstream handling is uniform.
//!
//! # CSV column order
//! ```text
//! record_id, asset_id, holder, active, granted_at, expires_at, notes
//! ```
//! - `active`: `"true"` / `"1"` / `"yes"` = active, anything else = inactive
//! - `expires_at`: empty or `"0"` = no expiry; otherwise Unix seconds
//! - `notes`: optional, may be empty

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{Result, RightsError};

// ── ImportedRight ─────────────────────────────────────────────────────────────

/// A single rights record as produced by an import operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImportedRight {
    /// Unique record identifier.
    pub record_id: String,
    /// Asset this right applies to.
    pub asset_id: String,
    /// Rights holder name or ID.
    pub holder: String,
    /// Whether the right is currently active.
    pub active: bool,
    /// Unix timestamp when the right was granted.
    pub granted_at: u64,
    /// Optional expiry (Unix seconds).
    pub expires_at: Option<u64>,
    /// Free-text notes.
    pub notes: String,
    /// Arbitrary extra fields (from JSON imports with additional keys).
    pub extra: HashMap<String, String>,
}

impl ImportedRight {
    /// Create a minimal active record with no expiry.
    #[must_use]
    pub fn new(
        record_id: impl Into<String>,
        asset_id: impl Into<String>,
        holder: impl Into<String>,
        granted_at: u64,
    ) -> Self {
        Self {
            record_id: record_id.into(),
            asset_id: asset_id.into(),
            holder: holder.into(),
            active: true,
            granted_at,
            expires_at: None,
            notes: String::new(),
            extra: HashMap::new(),
        }
    }
}

// ── ImportError ───────────────────────────────────────────────────────────────

/// Details of a single failed row during import.
#[derive(Debug, Clone)]
pub struct ImportError {
    /// Zero-based index of the row / item that failed.
    pub index: usize,
    /// Human-readable description of what went wrong.
    pub message: String,
}

// ── ImportResult ──────────────────────────────────────────────────────────────

/// The outcome of a bulk import operation.
#[derive(Debug)]
pub struct ImportResult {
    /// Successfully parsed records.
    pub records: Vec<ImportedRight>,
    /// Rows that failed to parse.
    pub errors: Vec<ImportError>,
}

impl ImportResult {
    /// Number of successfully imported records.
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.records.len()
    }

    /// Number of rows that could not be parsed.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Whether there were no parse errors.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.errors.is_empty()
    }
}

// ── CSV helpers ───────────────────────────────────────────────────────────────

/// Parse a boolean-ish string for the `active` column.
fn parse_bool(s: &str) -> bool {
    matches!(s.trim().to_lowercase().as_str(), "true" | "1" | "yes")
}

/// Parse an optional Unix-seconds field (empty / "0" → None).
fn parse_optional_ts(s: &str) -> Option<u64> {
    let trimmed = s.trim();
    if trimmed.is_empty() || trimmed == "0" {
        return None;
    }
    trimmed.parse::<u64>().ok()
}

// ── RightsImporter ────────────────────────────────────────────────────────────

/// Parser for bulk rights import operations.
///
/// Supports two input formats:
/// - **CSV** (comma-separated, no header quoting needed for basic fields)
/// - **JSON** (array of objects, keys match [`ImportedRight`] fields)
#[derive(Debug, Default)]
pub struct RightsImporter {
    /// Whether to skip rows with validation errors (default: `true`).
    pub skip_errors: bool,
}

impl RightsImporter {
    /// Create a new importer that tolerates errors in individual rows.
    #[must_use]
    pub fn new() -> Self {
        Self { skip_errors: true }
    }

    /// Create a strict importer that aborts on the first error.
    #[must_use]
    pub fn strict() -> Self {
        Self { skip_errors: false }
    }

    /// Import from a CSV string.
    ///
    /// The CSV **must** contain a header row with at minimum:
    /// `record_id`, `asset_id`, `holder`, `active`, `granted_at`
    ///
    /// Optional additional columns: `expires_at`, `notes`.
    /// Any other columns are stored in `extra`.
    ///
    /// # Errors
    /// Returns `RightsError::Serialization` only when operating in strict mode
    /// and a row fails.
    pub fn import_csv(&self, csv: &str) -> Result<ImportResult> {
        let mut lines = csv.lines();
        let header_line = match lines.next() {
            Some(h) => h,
            None => {
                return Ok(ImportResult {
                    records: vec![],
                    errors: vec![],
                })
            }
        };

        let headers: Vec<&str> = header_line.split(',').map(str::trim).collect();
        let col = |name: &str| -> Option<usize> {
            headers.iter().position(|&h| h.eq_ignore_ascii_case(name))
        };

        let idx_record_id = col("record_id");
        let idx_asset_id = col("asset_id");
        let idx_holder = col("holder");
        let idx_active = col("active");
        let idx_granted_at = col("granted_at");
        let idx_expires_at = col("expires_at");
        let idx_notes = col("notes");

        let required = [
            ("record_id", idx_record_id),
            ("asset_id", idx_asset_id),
            ("holder", idx_holder),
            ("granted_at", idx_granted_at),
        ];
        for (name, idx) in &required {
            if idx.is_none() {
                return Err(RightsError::Serialization(format!(
                    "Missing required CSV column: {name}"
                )));
            }
        }

        let idx_record_id = idx_record_id.ok_or_else(|| {
            RightsError::Serialization("Missing required CSV column: record_id".into())
        })?;
        let idx_asset_id = idx_asset_id.ok_or_else(|| {
            RightsError::Serialization("Missing required CSV column: asset_id".into())
        })?;
        let idx_holder = idx_holder.ok_or_else(|| {
            RightsError::Serialization("Missing required CSV column: holder".into())
        })?;
        let idx_granted_at = idx_granted_at.ok_or_else(|| {
            RightsError::Serialization("Missing required CSV column: granted_at".into())
        })?;

        let mut records = Vec::new();
        let mut errors = Vec::new();

        for (row_idx, line) in lines.enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();

            let get = |idx: usize| -> &str { fields.get(idx).copied().unwrap_or("") };

            let granted_at = match get(idx_granted_at).parse::<u64>() {
                Ok(v) => v,
                Err(_) => {
                    let err = ImportError {
                        index: row_idx,
                        message: format!("Row {row_idx}: invalid granted_at"),
                    };
                    if !self.skip_errors {
                        return Err(RightsError::Serialization(err.message));
                    }
                    errors.push(err);
                    continue;
                }
            };

            let record_id = get(idx_record_id).to_string();
            if record_id.is_empty() {
                let err = ImportError {
                    index: row_idx,
                    message: format!("Row {row_idx}: empty record_id"),
                };
                if !self.skip_errors {
                    return Err(RightsError::Serialization(err.message));
                }
                errors.push(err);
                continue;
            }

            let active = idx_active.map_or(true, |i| parse_bool(get(i)));
            let expires_at = idx_expires_at.and_then(|i| parse_optional_ts(get(i)));
            let notes = idx_notes.map_or("", |i| get(i)).to_string();

            // Extra columns
            let mut extra = HashMap::new();
            for (col_idx, &header) in headers.iter().enumerate() {
                let skip = [
                    "record_id",
                    "asset_id",
                    "holder",
                    "active",
                    "granted_at",
                    "expires_at",
                    "notes",
                ];
                if skip.iter().any(|s| s.eq_ignore_ascii_case(header)) {
                    continue;
                }
                if let Some(&val) = fields.get(col_idx) {
                    if !val.is_empty() {
                        extra.insert(header.to_string(), val.to_string());
                    }
                }
            }

            records.push(ImportedRight {
                record_id,
                asset_id: get(idx_asset_id).to_string(),
                holder: get(idx_holder).to_string(),
                active,
                granted_at,
                expires_at,
                notes,
                extra,
            });
        }

        Ok(ImportResult { records, errors })
    }

    /// Import from a JSON string containing an array of objects.
    ///
    /// Each object must have at minimum: `record_id`, `asset_id`, `holder`,
    /// `granted_at`. Other recognised keys: `active`, `expires_at`, `notes`.
    /// Any unrecognised keys are stored in `extra`.
    ///
    /// # Errors
    /// Returns `RightsError::Serialization` if the top-level structure is not
    /// a JSON array, or in strict mode when an item is malformed.
    pub fn import_json(&self, json: &str) -> Result<ImportResult> {
        let array: Vec<serde_json::Value> = serde_json::from_str(json)
            .map_err(|e| RightsError::Serialization(format!("JSON parse error: {e}")))?;

        let mut records = Vec::new();
        let mut errors = Vec::new();

        for (idx, item) in array.iter().enumerate() {
            match self.parse_json_item(item, idx) {
                Ok(record) => records.push(record),
                Err(msg) => {
                    if !self.skip_errors {
                        return Err(RightsError::Serialization(msg));
                    }
                    errors.push(ImportError {
                        index: idx,
                        message: msg,
                    });
                }
            }
        }

        Ok(ImportResult { records, errors })
    }

    fn parse_json_item(
        &self,
        item: &serde_json::Value,
        idx: usize,
    ) -> std::result::Result<ImportedRight, String> {
        let obj = item
            .as_object()
            .ok_or_else(|| format!("Item {idx} is not an object"))?;

        let get_str = |key: &str| -> std::result::Result<String, String> {
            obj.get(key)
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .ok_or_else(|| format!("Item {idx}: missing or non-string field '{key}'"))
        };

        let record_id = get_str("record_id")?;
        let asset_id = get_str("asset_id")?;
        let holder = get_str("holder")?;

        let granted_at = obj
            .get("granted_at")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| format!("Item {idx}: missing or invalid 'granted_at'"))?;

        let active = obj.get("active").and_then(|v| v.as_bool()).unwrap_or(true);

        let expires_at = obj
            .get("expires_at")
            .and_then(|v| v.as_u64())
            .and_then(|v| if v == 0 { None } else { Some(v) });

        let notes = obj
            .get("notes")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let known_keys = [
            "record_id",
            "asset_id",
            "holder",
            "active",
            "granted_at",
            "expires_at",
            "notes",
        ];
        let mut extra = HashMap::new();
        for (k, v) in obj {
            if !known_keys.contains(&k.as_str()) {
                if let Some(s) = v.as_str() {
                    extra.insert(k.clone(), s.to_string());
                } else {
                    extra.insert(k.clone(), v.to_string());
                }
            }
        }

        Ok(ImportedRight {
            record_id,
            asset_id,
            holder,
            active,
            granted_at,
            expires_at,
            notes,
            extra,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CSV import ──

    const SAMPLE_CSV: &str = "\
record_id,asset_id,holder,active,granted_at,expires_at,notes
r1,asset-A,Alice,true,1000,5000,First record
r2,asset-B,Bob,false,2000,,
r3,asset-C,Carol,1,3000,9999,Third";

    #[test]
    fn test_csv_import_count() {
        let importer = RightsImporter::new();
        let result = importer.import_csv(SAMPLE_CSV).expect("csv import");
        assert_eq!(result.success_count(), 3);
        assert!(result.is_clean());
    }

    #[test]
    fn test_csv_import_fields_r1() {
        let importer = RightsImporter::new();
        let result = importer.import_csv(SAMPLE_CSV).expect("csv import");
        let r1 = result
            .records
            .iter()
            .find(|r| r.record_id == "r1")
            .expect("r1");
        assert_eq!(r1.asset_id, "asset-A");
        assert_eq!(r1.holder, "Alice");
        assert!(r1.active);
        assert_eq!(r1.granted_at, 1000);
        assert_eq!(r1.expires_at, Some(5000));
        assert_eq!(r1.notes, "First record");
    }

    #[test]
    fn test_csv_import_inactive_r2() {
        let importer = RightsImporter::new();
        let result = importer.import_csv(SAMPLE_CSV).expect("csv import");
        let r2 = result
            .records
            .iter()
            .find(|r| r.record_id == "r2")
            .expect("r2");
        assert!(!r2.active);
        assert!(r2.expires_at.is_none());
    }

    #[test]
    fn test_csv_import_active_truthy_1() {
        let importer = RightsImporter::new();
        let result = importer.import_csv(SAMPLE_CSV).expect("csv import");
        let r3 = result
            .records
            .iter()
            .find(|r| r.record_id == "r3")
            .expect("r3");
        assert!(r3.active);
    }

    #[test]
    fn test_csv_import_missing_required_column() {
        let csv = "asset_id,holder,granted_at\nasset-A,Alice,1000\n";
        let importer = RightsImporter::new();
        let result = importer.import_csv(csv);
        assert!(result.is_err());
    }

    #[test]
    fn test_csv_import_invalid_granted_at() {
        let csv = "record_id,asset_id,holder,granted_at\nr1,a,Alice,not_a_number\n";
        let importer = RightsImporter::new(); // skip_errors = true
        let result = importer.import_csv(csv).expect("import");
        assert_eq!(result.success_count(), 0);
        assert_eq!(result.error_count(), 1);
    }

    #[test]
    fn test_csv_import_strict_aborts_on_error() {
        let csv = "record_id,asset_id,holder,granted_at\nr1,a,Alice,bad_ts\n";
        let importer = RightsImporter::strict();
        assert!(importer.import_csv(csv).is_err());
    }

    #[test]
    fn test_csv_import_empty_string() {
        let importer = RightsImporter::new();
        let result = importer.import_csv("").expect("empty csv");
        assert_eq!(result.success_count(), 0);
    }

    #[test]
    fn test_csv_import_header_only() {
        let csv = "record_id,asset_id,holder,granted_at\n";
        let importer = RightsImporter::new();
        let result = importer.import_csv(csv).expect("header only csv");
        assert_eq!(result.success_count(), 0);
    }

    // ── JSON import ──

    const SAMPLE_JSON: &str = r#"[
      {"record_id": "r1", "asset_id": "asset-A", "holder": "Alice", "granted_at": 1000, "expires_at": 5000, "notes": "Note A"},
      {"record_id": "r2", "asset_id": "asset-B", "holder": "Bob",   "granted_at": 2000, "active": false},
      {"record_id": "r3", "asset_id": "asset-C", "holder": "Carol", "granted_at": 3000, "custom_field": "extra_val"}
    ]"#;

    #[test]
    fn test_json_import_count() {
        let importer = RightsImporter::new();
        let result = importer.import_json(SAMPLE_JSON).expect("json import");
        assert_eq!(result.success_count(), 3);
        assert!(result.is_clean());
    }

    #[test]
    fn test_json_import_fields_r1() {
        let importer = RightsImporter::new();
        let result = importer.import_json(SAMPLE_JSON).expect("json import");
        let r1 = result
            .records
            .iter()
            .find(|r| r.record_id == "r1")
            .expect("r1");
        assert_eq!(r1.asset_id, "asset-A");
        assert_eq!(r1.expires_at, Some(5000));
        assert_eq!(r1.notes, "Note A");
        assert!(r1.active);
    }

    #[test]
    fn test_json_import_inactive_r2() {
        let importer = RightsImporter::new();
        let result = importer.import_json(SAMPLE_JSON).expect("json import");
        let r2 = result
            .records
            .iter()
            .find(|r| r.record_id == "r2")
            .expect("r2");
        assert!(!r2.active);
    }

    #[test]
    fn test_json_import_extra_fields() {
        let importer = RightsImporter::new();
        let result = importer.import_json(SAMPLE_JSON).expect("json import");
        let r3 = result
            .records
            .iter()
            .find(|r| r.record_id == "r3")
            .expect("r3");
        assert_eq!(
            r3.extra.get("custom_field").map(String::as_str),
            Some("extra_val")
        );
    }

    #[test]
    fn test_json_import_missing_field() {
        let json = r#"[{"asset_id": "a", "holder": "h", "granted_at": 1000}]"#;
        let importer = RightsImporter::new(); // skip_errors
        let result = importer.import_json(json).expect("json import");
        assert_eq!(result.success_count(), 0);
        assert_eq!(result.error_count(), 1);
    }

    #[test]
    fn test_json_import_strict_aborts() {
        let json = r#"[{"asset_id": "a", "holder": "h", "granted_at": 1000}]"#;
        let importer = RightsImporter::strict();
        assert!(importer.import_json(json).is_err());
    }

    #[test]
    fn test_json_import_invalid_top_level() {
        let json = r#"{"not": "an array"}"#;
        let importer = RightsImporter::new();
        assert!(importer.import_json(json).is_err());
    }

    #[test]
    fn test_json_import_empty_array() {
        let importer = RightsImporter::new();
        let result = importer.import_json("[]").expect("empty json");
        assert_eq!(result.success_count(), 0);
        assert!(result.is_clean());
    }

    #[test]
    fn test_json_import_expires_at_zero_becomes_none() {
        let json =
            r#"[{"record_id":"r","asset_id":"a","holder":"h","granted_at":1000,"expires_at":0}]"#;
        let importer = RightsImporter::new();
        let result = importer.import_json(json).expect("import");
        assert!(result.records[0].expires_at.is_none());
    }
}
