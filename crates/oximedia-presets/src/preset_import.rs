//! Preset import utilities.
//!
//! Supports importing preset bundles from different serialization formats
//! (JSON, TOML, CSV summary) into the preset manager.

#![allow(dead_code)]

use crate::preset_manager::{ManagedPreset, ManagedPresetCategory, PresetManager};

/// Supported import formats for preset bundles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportFormat {
    /// JSON preset bundle (array of preset objects).
    Json,
    /// TOML preset bundle.
    Toml,
    /// Comma-separated value summary (id, name, category, description).
    Csv,
}

impl std::fmt::Display for ImportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Json => "JSON",
            Self::Toml => "TOML",
            Self::Csv => "CSV",
        };
        write!(f, "{}", s)
    }
}

/// Outcome of a single preset import operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportOutcome {
    /// Preset was successfully added as a new entry.
    Added,
    /// Preset replaced an existing entry with the same ID.
    Updated,
    /// Preset was skipped (e.g. already up-to-date or policy rejection).
    Skipped,
}

/// Aggregated result of importing a batch of presets.
#[derive(Debug, Default)]
pub struct ImportResult {
    /// Number of presets successfully added.
    pub added: usize,
    /// Number of presets that replaced an existing entry.
    pub updated: usize,
    /// Number of presets that were skipped.
    pub skipped: usize,
    /// Errors encountered during import (preset id → message).
    pub errors: Vec<(String, String)>,
}

impl ImportResult {
    /// Total number of presets processed (added + updated + skipped + errors).
    #[must_use]
    pub fn total_processed(&self) -> usize {
        self.added + self.updated + self.skipped + self.errors.len()
    }

    /// Returns `true` if there were no errors.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Policy controlling how duplicate IDs are handled during import.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplicatePolicy {
    /// Overwrite existing presets with the same ID.
    Overwrite,
    /// Skip presets that already exist.
    Skip,
    /// Append a numeric suffix to the ID to avoid the conflict.
    Rename,
}

/// Importer that reads preset data and populates a [`PresetManager`].
#[derive(Debug)]
pub struct PresetImporter {
    format: ImportFormat,
    duplicate_policy: DuplicatePolicy,
}

impl PresetImporter {
    /// Create a new importer for the specified format.
    #[must_use]
    pub fn new(format: ImportFormat) -> Self {
        Self {
            format,
            duplicate_policy: DuplicatePolicy::Overwrite,
        }
    }

    /// Set the duplicate-handling policy.
    #[must_use]
    pub fn with_duplicate_policy(mut self, policy: DuplicatePolicy) -> Self {
        self.duplicate_policy = policy;
        self
    }

    /// Format this importer targets.
    #[must_use]
    pub fn format(&self) -> ImportFormat {
        self.format
    }

    /// Import presets from a CSV string into `manager`.
    ///
    /// Each line (after the header) is expected to have the form:
    /// `id,name,category,description`
    pub fn import_csv(&self, csv: &str, manager: &mut PresetManager) -> ImportResult {
        let mut result = ImportResult::default();

        for (line_num, line) in csv.lines().enumerate() {
            // Skip the header row.
            if line_num == 0 {
                continue;
            }
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.splitn(4, ',').collect();
            if parts.len() < 2 {
                result.errors.push((
                    format!("line:{}", line_num + 1),
                    "too few columns".to_string(),
                ));
                continue;
            }

            let id = parts[0].trim();
            let name = parts[1].trim();
            let category_str = if parts.len() > 2 { parts[2].trim() } else { "" };
            let description = if parts.len() > 3 { parts[3].trim() } else { "" };

            let category = parse_category(category_str);
            let preset = ManagedPreset::new(id, name, category).with_description(description);

            self.apply_preset(preset, manager, &mut result);
        }

        result
    }

    /// Import a slice of already-parsed `ManagedPreset` values.
    pub fn import_presets(
        &self,
        presets: Vec<ManagedPreset>,
        manager: &mut PresetManager,
    ) -> ImportResult {
        let mut result = ImportResult::default();
        for preset in presets {
            self.apply_preset(preset, manager, &mut result);
        }
        result
    }

    fn apply_preset(
        &self,
        mut preset: ManagedPreset,
        manager: &mut PresetManager,
        result: &mut ImportResult,
    ) {
        if manager.contains(&preset.id) {
            match self.duplicate_policy {
                DuplicatePolicy::Skip => {
                    result.skipped += 1;
                    return;
                }
                DuplicatePolicy::Overwrite => {
                    manager.insert(preset);
                    result.updated += 1;
                }
                DuplicatePolicy::Rename => {
                    let base = preset.id.clone();
                    let mut suffix = 1u32;
                    loop {
                        let new_id = format!("{}-{}", base, suffix);
                        if !manager.contains(&new_id) {
                            preset.id = new_id;
                            break;
                        }
                        suffix += 1;
                    }
                    manager.insert(preset);
                    result.added += 1;
                }
            }
        } else {
            manager.insert(preset);
            result.added += 1;
        }
    }
}

/// Parse a category string into a [`ManagedPresetCategory`].
fn parse_category(s: &str) -> ManagedPresetCategory {
    match s.to_lowercase().as_str() {
        "platform" => ManagedPresetCategory::Platform,
        "broadcast" => ManagedPresetCategory::Broadcast,
        "streaming" => ManagedPresetCategory::Streaming,
        "archive" => ManagedPresetCategory::Archive,
        "mobile" => ManagedPresetCategory::Mobile,
        "web" => ManagedPresetCategory::Web,
        "social" => ManagedPresetCategory::Social,
        _ => ManagedPresetCategory::Custom,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csv_data() -> &'static str {
        "id,name,category,description\n\
         yt-1080p,YouTube 1080p,platform,Full HD upload\n\
         hls-720p,HLS 720p,streaming,HLS rung\n\
         my-custom,Custom Preset,custom,User preset\n"
    }

    #[test]
    fn test_import_csv_basic() {
        let importer = PresetImporter::new(ImportFormat::Csv);
        let mut manager = PresetManager::new();
        let result = importer.import_csv(csv_data(), &mut manager);
        assert_eq!(result.added, 3);
        assert_eq!(result.updated, 0);
        assert_eq!(result.skipped, 0);
        assert!(result.is_ok());
        assert_eq!(manager.count(), 3);
    }

    #[test]
    fn test_import_csv_header_skipped() {
        let importer = PresetImporter::new(ImportFormat::Csv);
        let mut manager = PresetManager::new();
        importer.import_csv(csv_data(), &mut manager);
        // "id" should not be a preset
        assert!(!manager.contains("id"));
    }

    #[test]
    fn test_import_csv_categories_parsed() {
        let importer = PresetImporter::new(ImportFormat::Csv);
        let mut manager = PresetManager::new();
        importer.import_csv(csv_data(), &mut manager);
        assert_eq!(
            manager
                .get("yt-1080p")
                .expect("get should succeed")
                .category,
            ManagedPresetCategory::Platform
        );
        assert_eq!(
            manager
                .get("hls-720p")
                .expect("get should succeed")
                .category,
            ManagedPresetCategory::Streaming
        );
    }

    #[test]
    fn test_duplicate_policy_overwrite() {
        let importer = PresetImporter::new(ImportFormat::Csv)
            .with_duplicate_policy(DuplicatePolicy::Overwrite);
        let mut manager = PresetManager::new();
        importer.import_csv(csv_data(), &mut manager);
        let result = importer.import_csv(csv_data(), &mut manager);
        assert_eq!(result.updated, 3);
        assert_eq!(result.added, 0);
    }

    #[test]
    fn test_duplicate_policy_skip() {
        let importer =
            PresetImporter::new(ImportFormat::Csv).with_duplicate_policy(DuplicatePolicy::Skip);
        let mut manager = PresetManager::new();
        importer.import_csv(csv_data(), &mut manager);
        let result = importer.import_csv(csv_data(), &mut manager);
        assert_eq!(result.skipped, 3);
        assert_eq!(result.added, 0);
    }

    #[test]
    fn test_duplicate_policy_rename() {
        let importer =
            PresetImporter::new(ImportFormat::Csv).with_duplicate_policy(DuplicatePolicy::Rename);
        let mut manager = PresetManager::new();
        importer.import_csv(csv_data(), &mut manager);
        let result = importer.import_csv(csv_data(), &mut manager);
        assert_eq!(result.added, 3);
        // Renamed variants should exist
        assert!(manager.contains("yt-1080p-1"));
    }

    #[test]
    fn test_import_presets_slice() {
        let importer = PresetImporter::new(ImportFormat::Json);
        let mut manager = PresetManager::new();
        let presets = vec![
            ManagedPreset::new("a", "Preset A", ManagedPresetCategory::Custom),
            ManagedPreset::new("b", "Preset B", ManagedPresetCategory::Platform),
        ];
        let result = importer.import_presets(presets, &mut manager);
        assert_eq!(result.added, 2);
        assert_eq!(manager.count(), 2);
    }

    #[test]
    fn test_import_csv_bad_line_recorded_as_error() {
        let bad_csv = "id,name,category,description\nbadline\n";
        let importer = PresetImporter::new(ImportFormat::Csv);
        let mut manager = PresetManager::new();
        let result = importer.import_csv(bad_csv, &mut manager);
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_import_result_total_processed() {
        let mut r = ImportResult::default();
        r.added = 2;
        r.updated = 1;
        r.skipped = 1;
        assert_eq!(r.total_processed(), 4);
    }

    #[test]
    fn test_import_format_display() {
        assert_eq!(ImportFormat::Json.to_string(), "JSON");
        assert_eq!(ImportFormat::Toml.to_string(), "TOML");
        assert_eq!(ImportFormat::Csv.to_string(), "CSV");
    }

    #[test]
    fn test_importer_format_accessor() {
        let importer = PresetImporter::new(ImportFormat::Toml);
        assert_eq!(importer.format(), ImportFormat::Toml);
    }

    #[test]
    fn test_empty_csv_no_errors() {
        let importer = PresetImporter::new(ImportFormat::Csv);
        let mut manager = PresetManager::new();
        let result = importer.import_csv("id,name,category,description\n", &mut manager);
        assert_eq!(result.total_processed(), 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_category_fallback_to_custom() {
        let cat = parse_category("unknown_type");
        assert_eq!(cat, ManagedPresetCategory::Custom);
    }
}
