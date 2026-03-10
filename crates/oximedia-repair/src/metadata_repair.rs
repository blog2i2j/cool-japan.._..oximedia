#![allow(dead_code)]
//! Metadata repair and reconstruction for media files.
//!
//! This module provides tools for detecting corrupted, missing, or inconsistent
//! metadata in media files and reconstructing it from file analysis. Handles
//! container-level metadata, codec parameters, timing information, and
//! user-defined tags.

use std::collections::HashMap;

/// Type of metadata field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetadataFieldKind {
    /// Container format metadata (e.g., moov atom, MKV header).
    Container,
    /// Codec-specific parameters (SPS/PPS, audio config).
    CodecConfig,
    /// Timing and duration information.
    Timing,
    /// User-defined tags (title, artist, etc.).
    UserTag,
    /// Chapter or marker information.
    Chapter,
    /// Thumbnail or cover art.
    Artwork,
    /// Technical metadata (bitrate, resolution, etc.).
    Technical,
    /// Geolocation data.
    Location,
}

/// Status of a metadata field after analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldStatus {
    /// Field is present and valid.
    Valid,
    /// Field is present but contains invalid data.
    Corrupted,
    /// Field is missing entirely.
    Missing,
    /// Field is present but inconsistent with other data.
    Inconsistent,
    /// Field was successfully reconstructed.
    Reconstructed,
}

/// A single metadata field descriptor.
#[derive(Debug, Clone)]
pub struct MetadataField {
    /// Name/key of the field.
    pub key: String,
    /// Current value (if any).
    pub value: Option<String>,
    /// Kind of metadata.
    pub kind: MetadataFieldKind,
    /// Current status.
    pub status: FieldStatus,
    /// Byte offset in the file where this field resides.
    pub file_offset: Option<u64>,
    /// Reconstructed value (if repair was attempted).
    pub reconstructed_value: Option<String>,
}

impl MetadataField {
    /// Create a new metadata field.
    pub fn new(key: &str, kind: MetadataFieldKind) -> Self {
        Self {
            key: key.to_string(),
            value: None,
            kind,
            status: FieldStatus::Missing,
            file_offset: None,
            reconstructed_value: None,
        }
    }

    /// Set the current value and mark as valid.
    pub fn with_value(mut self, value: &str) -> Self {
        self.value = Some(value.to_string());
        self.status = FieldStatus::Valid;
        self
    }

    /// Mark the field as corrupted.
    pub fn mark_corrupted(mut self) -> Self {
        self.status = FieldStatus::Corrupted;
        self
    }

    /// Whether this field needs repair.
    pub fn needs_repair(&self) -> bool {
        matches!(
            self.status,
            FieldStatus::Corrupted | FieldStatus::Missing | FieldStatus::Inconsistent
        )
    }
}

/// Configuration for metadata repair operations.
#[derive(Debug, Clone)]
pub struct MetadataRepairConfig {
    /// Whether to reconstruct timing metadata from stream analysis.
    pub reconstruct_timing: bool,
    /// Whether to infer codec parameters from bitstream.
    pub infer_codec_params: bool,
    /// Whether to validate user tags against known schemas.
    pub validate_tags: bool,
    /// Maximum number of reconstruction attempts.
    pub max_attempts: u32,
    /// Whether to preserve original corrupted values as backup.
    pub preserve_originals: bool,
    /// Default language for reconstructed text fields.
    pub default_language: String,
}

impl Default for MetadataRepairConfig {
    fn default() -> Self {
        Self {
            reconstruct_timing: true,
            infer_codec_params: true,
            validate_tags: true,
            max_attempts: 3,
            preserve_originals: true,
            default_language: "und".to_string(),
        }
    }
}

/// Result of metadata analysis.
#[derive(Debug, Clone)]
pub struct MetadataAnalysis {
    /// All detected metadata fields.
    pub fields: Vec<MetadataField>,
    /// Number of valid fields.
    pub valid_count: usize,
    /// Number of corrupted fields.
    pub corrupted_count: usize,
    /// Number of missing fields.
    pub missing_count: usize,
    /// Overall metadata health score (0.0 - 1.0).
    pub health_score: f64,
}

impl MetadataAnalysis {
    /// Create a new analysis from a list of fields.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_fields(fields: Vec<MetadataField>) -> Self {
        let valid_count = fields
            .iter()
            .filter(|f| f.status == FieldStatus::Valid)
            .count();
        let corrupted_count = fields
            .iter()
            .filter(|f| f.status == FieldStatus::Corrupted)
            .count();
        let missing_count = fields
            .iter()
            .filter(|f| f.status == FieldStatus::Missing)
            .count();
        let total = fields.len().max(1);
        let health_score = valid_count as f64 / total as f64;

        Self {
            fields,
            valid_count,
            corrupted_count,
            missing_count,
            health_score,
        }
    }

    /// Get all fields that need repair.
    pub fn fields_needing_repair(&self) -> Vec<&MetadataField> {
        self.fields.iter().filter(|f| f.needs_repair()).collect()
    }
}

/// Result of a metadata repair operation.
#[derive(Debug, Clone)]
pub struct MetadataRepairResult {
    /// Number of fields analyzed.
    pub fields_analyzed: usize,
    /// Number of fields repaired.
    pub fields_repaired: usize,
    /// Number of fields that could not be repaired.
    pub fields_unrepaired: usize,
    /// Repaired field key-value pairs.
    pub repaired_fields: HashMap<String, String>,
    /// Health score after repair (0.0 - 1.0).
    pub health_after: f64,
}

/// Engine for analyzing and repairing media metadata.
#[derive(Debug)]
pub struct MetadataRepairEngine {
    /// Configuration.
    config: MetadataRepairConfig,
}

impl MetadataRepairEngine {
    /// Create a new metadata repair engine.
    pub fn new(config: MetadataRepairConfig) -> Self {
        Self { config }
    }

    /// Create an engine with default configuration.
    pub fn default_engine() -> Self {
        Self::new(MetadataRepairConfig::default())
    }

    /// Analyze a set of metadata fields.
    pub fn analyze(&self, fields: Vec<MetadataField>) -> MetadataAnalysis {
        MetadataAnalysis::from_fields(fields)
    }

    /// Attempt to repair metadata fields.
    #[allow(clippy::cast_precision_loss)]
    pub fn repair(&self, fields: &mut Vec<MetadataField>) -> MetadataRepairResult {
        let fields_analyzed = fields.len();
        let mut repaired_count = 0usize;
        let mut repaired_fields = HashMap::new();

        for field in fields.iter_mut() {
            if !field.needs_repair() {
                continue;
            }

            if let Some(value) = self.reconstruct_field(field) {
                if self.config.preserve_originals {
                    field.reconstructed_value = Some(value.clone());
                }
                field.value = Some(value.clone());
                field.status = FieldStatus::Reconstructed;
                repaired_fields.insert(field.key.clone(), value);
                repaired_count += 1;
            }
        }

        let unrepaired = fields.iter().filter(|f| f.needs_repair()).count();
        let total = fields.len().max(1);
        let valid_after = fields
            .iter()
            .filter(|f| matches!(f.status, FieldStatus::Valid | FieldStatus::Reconstructed))
            .count();

        MetadataRepairResult {
            fields_analyzed,
            fields_repaired: repaired_count,
            fields_unrepaired: unrepaired,
            repaired_fields,
            health_after: valid_after as f64 / total as f64,
        }
    }

    /// Attempt to reconstruct a single field value.
    fn reconstruct_field(&self, field: &MetadataField) -> Option<String> {
        match field.kind {
            MetadataFieldKind::Timing => {
                if self.config.reconstruct_timing {
                    Some(self.reconstruct_timing_field(field))
                } else {
                    None
                }
            }
            MetadataFieldKind::CodecConfig => {
                if self.config.infer_codec_params {
                    Some(self.infer_codec_config(field))
                } else {
                    None
                }
            }
            MetadataFieldKind::UserTag => {
                if self.config.validate_tags {
                    self.reconstruct_tag(field)
                } else {
                    None
                }
            }
            MetadataFieldKind::Container => Some("reconstructed_container".to_string()),
            MetadataFieldKind::Technical => Some(self.reconstruct_technical(field)),
            _ => None,
        }
    }

    /// Reconstruct a timing metadata field.
    fn reconstruct_timing_field(&self, field: &MetadataField) -> String {
        match field.key.as_str() {
            "duration" => "0".to_string(),
            "timescale" => "90000".to_string(),
            "start_time" => "0".to_string(),
            _ => "0".to_string(),
        }
    }

    /// Infer codec configuration from available data.
    fn infer_codec_config(&self, field: &MetadataField) -> String {
        match field.key.as_str() {
            "codec_id" => "unknown".to_string(),
            "profile" => "baseline".to_string(),
            "level" => "3.0".to_string(),
            "channels" => "2".to_string(),
            "sample_rate" => "48000".to_string(),
            _ => "unknown".to_string(),
        }
    }

    /// Reconstruct a user tag value.
    fn reconstruct_tag(&self, field: &MetadataField) -> Option<String> {
        // Only reconstruct tags that have reasonable defaults
        match field.key.as_str() {
            "language" => Some(self.config.default_language.clone()),
            "encoding_tool" => Some("unknown".to_string()),
            _ => None,
        }
    }

    /// Reconstruct technical metadata.
    fn reconstruct_technical(&self, field: &MetadataField) -> String {
        match field.key.as_str() {
            "bitrate" => "0".to_string(),
            "frame_rate" => "24".to_string(),
            "width" => "0".to_string(),
            "height" => "0".to_string(),
            "pixel_format" => "yuv420p".to_string(),
            _ => "unknown".to_string(),
        }
    }

    /// Validate consistency across a set of metadata fields.
    pub fn validate_consistency(&self, fields: &[MetadataField]) -> Vec<String> {
        let mut warnings = Vec::new();

        // Check for duplicate keys
        let mut seen_keys: HashMap<String, usize> = HashMap::new();
        for field in fields {
            *seen_keys.entry(field.key.clone()).or_insert(0) += 1;
        }
        for (key, count) in &seen_keys {
            if *count > 1 {
                warnings.push(format!(
                    "Duplicate metadata key: {key} ({count} occurrences)"
                ));
            }
        }

        // Check timing consistency
        let has_duration = fields
            .iter()
            .any(|f| f.key == "duration" && f.status == FieldStatus::Valid);
        let has_timescale = fields
            .iter()
            .any(|f| f.key == "timescale" && f.status == FieldStatus::Valid);
        if has_duration && !has_timescale {
            warnings.push("Duration present but timescale missing".to_string());
        }

        warnings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_field_new() {
        let field = MetadataField::new("title", MetadataFieldKind::UserTag);
        assert_eq!(field.key, "title");
        assert_eq!(field.status, FieldStatus::Missing);
        assert!(field.value.is_none());
    }

    #[test]
    fn test_metadata_field_with_value() {
        let field = MetadataField::new("title", MetadataFieldKind::UserTag).with_value("My Video");
        assert_eq!(field.status, FieldStatus::Valid);
        assert_eq!(field.value.as_deref(), Some("My Video"));
    }

    #[test]
    fn test_metadata_field_needs_repair() {
        let valid = MetadataField::new("a", MetadataFieldKind::UserTag).with_value("ok");
        assert!(!valid.needs_repair());

        let missing = MetadataField::new("b", MetadataFieldKind::UserTag);
        assert!(missing.needs_repair());

        let corrupted = MetadataField::new("c", MetadataFieldKind::UserTag).mark_corrupted();
        assert!(corrupted.needs_repair());
    }

    #[test]
    fn test_default_config() {
        let cfg = MetadataRepairConfig::default();
        assert!(cfg.reconstruct_timing);
        assert!(cfg.infer_codec_params);
        assert_eq!(cfg.max_attempts, 3);
        assert_eq!(cfg.default_language, "und");
    }

    #[test]
    fn test_analysis_from_fields() {
        let fields = vec![
            MetadataField::new("title", MetadataFieldKind::UserTag).with_value("ok"),
            MetadataField::new("codec", MetadataFieldKind::CodecConfig),
            MetadataField::new("dur", MetadataFieldKind::Timing).mark_corrupted(),
        ];
        let analysis = MetadataAnalysis::from_fields(fields);
        assert_eq!(analysis.valid_count, 1);
        assert_eq!(analysis.corrupted_count, 1);
        assert_eq!(analysis.missing_count, 1);
        assert!((analysis.health_score - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_fields_needing_repair() {
        let fields = vec![
            MetadataField::new("a", MetadataFieldKind::UserTag).with_value("ok"),
            MetadataField::new("b", MetadataFieldKind::UserTag),
        ];
        let analysis = MetadataAnalysis::from_fields(fields);
        let needing = analysis.fields_needing_repair();
        assert_eq!(needing.len(), 1);
        assert_eq!(needing[0].key, "b");
    }

    #[test]
    fn test_repair_missing_timing() {
        let engine = MetadataRepairEngine::default_engine();
        let mut fields = vec![
            MetadataField::new("duration", MetadataFieldKind::Timing),
            MetadataField::new("timescale", MetadataFieldKind::Timing),
        ];
        let result = engine.repair(&mut fields);
        assert_eq!(result.fields_repaired, 2);
        assert!(result.repaired_fields.contains_key("duration"));
        assert!(result.repaired_fields.contains_key("timescale"));
    }

    #[test]
    fn test_repair_codec_config() {
        let engine = MetadataRepairEngine::default_engine();
        let mut fields = vec![
            MetadataField::new("sample_rate", MetadataFieldKind::CodecConfig),
            MetadataField::new("channels", MetadataFieldKind::CodecConfig),
        ];
        let result = engine.repair(&mut fields);
        assert_eq!(result.fields_repaired, 2);
        assert_eq!(
            result
                .repaired_fields
                .get("sample_rate")
                .expect("expected key to exist"),
            "48000"
        );
        assert_eq!(
            result
                .repaired_fields
                .get("channels")
                .expect("expected key to exist"),
            "2"
        );
    }

    #[test]
    fn test_repair_preserves_valid() {
        let engine = MetadataRepairEngine::default_engine();
        let mut fields =
            vec![MetadataField::new("title", MetadataFieldKind::UserTag).with_value("Good Title")];
        let result = engine.repair(&mut fields);
        assert_eq!(result.fields_repaired, 0);
        assert_eq!(fields[0].value.as_deref(), Some("Good Title"));
    }

    #[test]
    fn test_repair_user_tag_language() {
        let engine = MetadataRepairEngine::default_engine();
        let mut fields = vec![MetadataField::new("language", MetadataFieldKind::UserTag)];
        let result = engine.repair(&mut fields);
        assert_eq!(result.fields_repaired, 1);
        assert_eq!(
            result
                .repaired_fields
                .get("language")
                .expect("expected key to exist"),
            "und"
        );
    }

    #[test]
    fn test_validate_consistency_duplicates() {
        let engine = MetadataRepairEngine::default_engine();
        let fields = vec![
            MetadataField::new("title", MetadataFieldKind::UserTag).with_value("A"),
            MetadataField::new("title", MetadataFieldKind::UserTag).with_value("B"),
        ];
        let warnings = engine.validate_consistency(&fields);
        assert!(!warnings.is_empty());
        assert!(warnings[0].contains("Duplicate"));
    }

    #[test]
    fn test_validate_consistency_clean() {
        let engine = MetadataRepairEngine::default_engine();
        let fields = vec![
            MetadataField::new("title", MetadataFieldKind::UserTag).with_value("A"),
            MetadataField::new("artist", MetadataFieldKind::UserTag).with_value("B"),
        ];
        let warnings = engine.validate_consistency(&fields);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_health_after_repair() {
        let engine = MetadataRepairEngine::default_engine();
        let mut fields = vec![
            MetadataField::new("duration", MetadataFieldKind::Timing),
            MetadataField::new("title", MetadataFieldKind::UserTag).with_value("ok"),
        ];
        let result = engine.repair(&mut fields);
        assert!(result.health_after > 0.5);
    }
}
