//! Foley asset management: track categories, cue sheets, and sync markers.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// Foley track category.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FoleyCategory {
    /// Footsteps on various surfaces.
    Footsteps,
    /// Clothing movement and rustle.
    Clothing,
    /// Props handling and interactions.
    Props,
    /// Water effects (pouring, splashing).
    Water,
    /// Impacts and falls.
    Impact,
    /// Custom category.
    Custom(String),
}

impl FoleyCategory {
    /// Return a display name for the category.
    pub fn display_name(&self) -> String {
        match self {
            Self::Footsteps => "Footsteps".to_string(),
            Self::Clothing => "Clothing".to_string(),
            Self::Props => "Props".to_string(),
            Self::Water => "Water".to_string(),
            Self::Impact => "Impact".to_string(),
            Self::Custom(s) => s.clone(),
        }
    }
}

/// A sync marker indicating where a Foley cue should align to picture.
#[derive(Debug, Clone)]
pub struct FoleySyncMarker {
    /// Marker label (e.g., "footstep-1").
    pub label: String,
    /// Position in samples.
    pub position_samples: u64,
    /// Sample rate.
    pub sample_rate: u32,
    /// Optional description of the visual event.
    pub description: Option<String>,
}

impl FoleySyncMarker {
    /// Create a new sync marker.
    pub fn new(label: &str, position_samples: u64, sample_rate: u32) -> Self {
        Self {
            label: label.to_string(),
            position_samples,
            sample_rate,
            description: None,
        }
    }

    /// Position in seconds.
    pub fn position_secs(&self) -> f64 {
        self.position_samples as f64 / self.sample_rate as f64
    }

    /// Add a description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }
}

/// A Foley asset stored in the library.
#[derive(Debug, Clone)]
pub struct FoleyAsset {
    /// Unique asset ID.
    pub asset_id: String,
    /// Human-readable name.
    pub name: String,
    /// Category.
    pub category: FoleyCategory,
    /// File path to the audio file.
    pub file_path: String,
    /// Duration in samples.
    pub duration_samples: u64,
    /// Sample rate.
    pub sample_rate: u32,
    /// Tags for searchability.
    pub tags: Vec<String>,
}

impl FoleyAsset {
    /// Create a new Foley asset.
    pub fn new(
        asset_id: &str,
        name: &str,
        category: FoleyCategory,
        file_path: &str,
        duration_samples: u64,
        sample_rate: u32,
    ) -> Self {
        Self {
            asset_id: asset_id.to_string(),
            name: name.to_string(),
            category,
            file_path: file_path.to_string(),
            duration_samples,
            sample_rate,
            tags: Vec::new(),
        }
    }

    /// Duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.duration_samples as f64 / self.sample_rate as f64
    }

    /// Add a tag.
    pub fn add_tag(&mut self, tag: &str) {
        self.tags.push(tag.to_string());
    }
}

/// A single cue in a Foley cue sheet.
#[derive(Debug, Clone)]
pub struct FoleySheetEntry {
    /// Cue number.
    pub cue_number: u32,
    /// Asset ID to use.
    pub asset_id: String,
    /// Asset name (denormalized for readability).
    pub asset_name: String,
    /// Category of this cue.
    pub category: FoleyCategory,
    /// Sync markers for this cue.
    pub sync_markers: Vec<FoleySyncMarker>,
    /// Whether this cue has been recorded.
    pub is_recorded: bool,
    /// Whether this cue has been approved.
    pub is_approved: bool,
    /// Optional notes.
    pub notes: Option<String>,
}

impl FoleySheetEntry {
    /// Create a new cue sheet entry.
    pub fn new(cue_number: u32, asset_id: &str, asset_name: &str, category: FoleyCategory) -> Self {
        Self {
            cue_number,
            asset_id: asset_id.to_string(),
            asset_name: asset_name.to_string(),
            category,
            sync_markers: Vec::new(),
            is_recorded: false,
            is_approved: false,
            notes: None,
        }
    }

    /// Add a sync marker.
    pub fn add_sync_marker(&mut self, marker: FoleySyncMarker) {
        self.sync_markers.push(marker);
    }

    /// Mark as recorded.
    pub fn mark_recorded(&mut self) {
        self.is_recorded = true;
    }

    /// Mark as approved.
    pub fn mark_approved(&mut self) {
        self.is_approved = true;
    }

    /// Number of sync markers.
    pub fn sync_marker_count(&self) -> usize {
        self.sync_markers.len()
    }
}

/// A Foley cue sheet for a full reel or scene.
#[derive(Debug, Default)]
pub struct FoleyCueSheet {
    /// Project name.
    pub project: String,
    /// Reel or scene identifier.
    pub scene: String,
    /// All cue entries.
    entries: Vec<FoleySheetEntry>,
}

impl FoleyCueSheet {
    /// Create a new cue sheet.
    pub fn new(project: &str, scene: &str) -> Self {
        Self {
            project: project.to_string(),
            scene: scene.to_string(),
            entries: Vec::new(),
        }
    }

    /// Add a cue entry.
    pub fn add_entry(&mut self, entry: FoleySheetEntry) {
        self.entries.push(entry);
    }

    /// Total number of cues.
    pub fn cue_count(&self) -> usize {
        self.entries.len()
    }

    /// Number of recorded cues.
    pub fn recorded_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_recorded).count()
    }

    /// Number of approved cues.
    pub fn approved_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_approved).count()
    }

    /// Cues grouped by category.
    pub fn grouped_by_category(&self) -> HashMap<String, Vec<&FoleySheetEntry>> {
        let mut map: HashMap<String, Vec<&FoleySheetEntry>> = HashMap::new();
        for entry in &self.entries {
            map.entry(entry.category.display_name())
                .or_default()
                .push(entry);
        }
        map
    }

    /// Completion percentage.
    pub fn completion_percent(&self) -> f64 {
        if self.entries.is_empty() {
            return 100.0;
        }
        self.approved_count() as f64 / self.entries.len() as f64 * 100.0
    }
}

/// Foley asset library.
#[derive(Debug, Default)]
pub struct FoleyLibrary {
    assets: HashMap<String, FoleyAsset>,
}

impl FoleyLibrary {
    /// Create a new empty library.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an asset.
    pub fn add_asset(&mut self, asset: FoleyAsset) {
        self.assets.insert(asset.asset_id.clone(), asset);
    }

    /// Get an asset by ID.
    pub fn get_asset(&self, asset_id: &str) -> Option<&FoleyAsset> {
        self.assets.get(asset_id)
    }

    /// Search by category.
    pub fn search_by_category(&self, category: &FoleyCategory) -> Vec<&FoleyAsset> {
        self.assets
            .values()
            .filter(|a| &a.category == category)
            .collect()
    }

    /// Search by tag.
    pub fn search_by_tag(&self, tag: &str) -> Vec<&FoleyAsset> {
        self.assets
            .values()
            .filter(|a| a.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Total asset count.
    pub fn count(&self) -> usize {
        self.assets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_asset(id: &str, cat: FoleyCategory) -> FoleyAsset {
        FoleyAsset::new(id, id, cat, &format!("/foley/{id}.wav"), 48000, 48000)
    }

    #[test]
    fn test_foley_category_display_name() {
        assert_eq!(FoleyCategory::Footsteps.display_name(), "Footsteps");
        assert_eq!(FoleyCategory::Clothing.display_name(), "Clothing");
        assert_eq!(FoleyCategory::Custom("FX".to_string()).display_name(), "FX");
    }

    #[test]
    fn test_sync_marker_position_secs() {
        let marker = FoleySyncMarker::new("m1", 48000, 48000);
        assert!((marker.position_secs() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_sync_marker_with_description() {
        let marker = FoleySyncMarker::new("m2", 0, 48000).with_description("Character enters room");
        assert_eq!(marker.description.as_deref(), Some("Character enters room"));
    }

    #[test]
    fn test_foley_asset_duration_secs() {
        let asset = make_asset("step1", FoleyCategory::Footsteps);
        assert!((asset.duration_secs() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_foley_asset_add_tag() {
        let mut asset = make_asset("step1", FoleyCategory::Footsteps);
        asset.add_tag("wood");
        asset.add_tag("dry");
        assert_eq!(asset.tags.len(), 2);
    }

    #[test]
    fn test_cue_sheet_entry_new() {
        let entry = FoleySheetEntry::new(1, "asset-1", "Footstep", FoleyCategory::Footsteps);
        assert_eq!(entry.cue_number, 1);
        assert!(!entry.is_recorded);
        assert!(!entry.is_approved);
    }

    #[test]
    fn test_cue_sheet_entry_mark_recorded_and_approved() {
        let mut entry = FoleySheetEntry::new(1, "asset-1", "Footstep", FoleyCategory::Footsteps);
        entry.mark_recorded();
        assert!(entry.is_recorded);
        entry.mark_approved();
        assert!(entry.is_approved);
    }

    #[test]
    fn test_cue_sheet_entry_add_sync_marker() {
        let mut entry = FoleySheetEntry::new(1, "asset-1", "Footstep", FoleyCategory::Footsteps);
        entry.add_sync_marker(FoleySyncMarker::new("s1", 1000, 48000));
        assert_eq!(entry.sync_marker_count(), 1);
    }

    #[test]
    fn test_cue_sheet_cue_count() {
        let mut sheet = FoleyCueSheet::new("Film", "Scene 1");
        sheet.add_entry(FoleySheetEntry::new(
            1,
            "a1",
            "Step",
            FoleyCategory::Footsteps,
        ));
        sheet.add_entry(FoleySheetEntry::new(
            2,
            "a2",
            "Cloth",
            FoleyCategory::Clothing,
        ));
        assert_eq!(sheet.cue_count(), 2);
    }

    #[test]
    fn test_cue_sheet_recorded_count() {
        let mut sheet = FoleyCueSheet::new("Film", "Scene 1");
        let mut e1 = FoleySheetEntry::new(1, "a1", "Step", FoleyCategory::Footsteps);
        e1.mark_recorded();
        sheet.add_entry(e1);
        sheet.add_entry(FoleySheetEntry::new(2, "a2", "Prop", FoleyCategory::Props));
        assert_eq!(sheet.recorded_count(), 1);
    }

    #[test]
    fn test_cue_sheet_approved_count() {
        let mut sheet = FoleyCueSheet::new("Film", "Scene 1");
        let mut e1 = FoleySheetEntry::new(1, "a1", "Step", FoleyCategory::Footsteps);
        e1.mark_approved();
        sheet.add_entry(e1);
        sheet.add_entry(FoleySheetEntry::new(2, "a2", "Prop", FoleyCategory::Props));
        assert_eq!(sheet.approved_count(), 1);
    }

    #[test]
    fn test_cue_sheet_completion_percent_empty() {
        let sheet = FoleyCueSheet::new("Film", "Scene 1");
        assert!((sheet.completion_percent() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cue_sheet_completion_percent() {
        let mut sheet = FoleyCueSheet::new("Film", "Scene 1");
        let mut e1 = FoleySheetEntry::new(1, "a1", "Step", FoleyCategory::Footsteps);
        e1.mark_approved();
        sheet.add_entry(e1);
        sheet.add_entry(FoleySheetEntry::new(2, "a2", "Prop", FoleyCategory::Props));
        assert!((sheet.completion_percent() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_cue_sheet_grouped_by_category() {
        let mut sheet = FoleyCueSheet::new("Film", "Scene 1");
        sheet.add_entry(FoleySheetEntry::new(
            1,
            "a1",
            "S1",
            FoleyCategory::Footsteps,
        ));
        sheet.add_entry(FoleySheetEntry::new(
            2,
            "a2",
            "S2",
            FoleyCategory::Footsteps,
        ));
        sheet.add_entry(FoleySheetEntry::new(3, "a3", "W1", FoleyCategory::Water));
        let groups = sheet.grouped_by_category();
        assert_eq!(groups["Footsteps"].len(), 2);
        assert_eq!(groups["Water"].len(), 1);
    }

    #[test]
    fn test_foley_library_add_and_get() {
        let mut lib = FoleyLibrary::new();
        lib.add_asset(make_asset("step1", FoleyCategory::Footsteps));
        let asset = lib.get_asset("step1");
        assert!(asset.is_some());
        assert_eq!(asset.expect("asset should be valid").asset_id, "step1");
    }

    #[test]
    fn test_foley_library_search_by_category() {
        let mut lib = FoleyLibrary::new();
        lib.add_asset(make_asset("step1", FoleyCategory::Footsteps));
        lib.add_asset(make_asset("step2", FoleyCategory::Footsteps));
        lib.add_asset(make_asset("cloth1", FoleyCategory::Clothing));
        let footsteps = lib.search_by_category(&FoleyCategory::Footsteps);
        assert_eq!(footsteps.len(), 2);
    }

    #[test]
    fn test_foley_library_search_by_tag() {
        let mut lib = FoleyLibrary::new();
        let mut asset = make_asset("step1", FoleyCategory::Footsteps);
        asset.add_tag("wood");
        lib.add_asset(asset);
        let results = lib.search_by_tag("wood");
        assert_eq!(results.len(), 1);
        let no_results = lib.search_by_tag("metal");
        assert!(no_results.is_empty());
    }

    #[test]
    fn test_foley_library_count() {
        let mut lib = FoleyLibrary::new();
        lib.add_asset(make_asset("a1", FoleyCategory::Props));
        lib.add_asset(make_asset("a2", FoleyCategory::Impact));
        assert_eq!(lib.count(), 2);
    }
}
