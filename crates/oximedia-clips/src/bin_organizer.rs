#![allow(dead_code)]
//! Bin organization and automatic clip sorting.
//!
//! This module provides smart bin organization features that automatically
//! sort, categorize, and arrange clips into bins based on various criteria
//! such as date, camera, scene, codec, resolution, or custom rules.

use std::collections::HashMap;

/// Criteria used for organizing clips into bins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrganizeCriteria {
    /// Organize by recording date.
    ByDate,
    /// Organize by camera / source name.
    ByCamera,
    /// Organize by scene number.
    ByScene,
    /// Organize by codec type.
    ByCodec,
    /// Organize by resolution.
    ByResolution,
    /// Organize by frame rate.
    ByFrameRate,
    /// Organize by rating.
    ByRating,
    /// Organize by keyword.
    ByKeyword,
    /// Organize by media type (video, audio, image).
    ByMediaType,
    /// Organize by file extension.
    ByExtension,
}

/// Media type classification for a clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClipMediaType {
    /// Video clip.
    Video,
    /// Audio-only clip.
    Audio,
    /// Still image.
    Image,
    /// Graphics or title card.
    Graphics,
    /// Unknown media type.
    Unknown,
}

/// Lightweight clip descriptor used for bin organization.
#[derive(Debug, Clone)]
pub struct ClipDescriptor {
    /// Unique clip identifier.
    pub clip_id: u64,
    /// Display name of the clip.
    pub name: String,
    /// Recording date as ISO string (e.g., "2025-03-15").
    pub date: Option<String>,
    /// Camera or source name.
    pub camera: Option<String>,
    /// Scene number or label.
    pub scene: Option<String>,
    /// Codec name (e.g., "H.264", "ProRes").
    pub codec: Option<String>,
    /// Width in pixels.
    pub width: Option<u32>,
    /// Height in pixels.
    pub height: Option<u32>,
    /// Frame rate.
    pub frame_rate: Option<f64>,
    /// Rating (1-5, or 0 for unrated).
    pub rating: u8,
    /// Keywords/tags.
    pub keywords: Vec<String>,
    /// Media type.
    pub media_type: ClipMediaType,
    /// File extension.
    pub extension: Option<String>,
}

impl ClipDescriptor {
    /// Creates a new clip descriptor with an ID and name.
    #[must_use]
    pub fn new(clip_id: u64, name: impl Into<String>) -> Self {
        Self {
            clip_id,
            name: name.into(),
            date: None,
            camera: None,
            scene: None,
            codec: None,
            width: None,
            height: None,
            frame_rate: None,
            rating: 0,
            keywords: Vec::new(),
            media_type: ClipMediaType::Unknown,
            extension: None,
        }
    }

    /// Returns a resolution string like "1920x1080" if available.
    #[must_use]
    pub fn resolution_label(&self) -> Option<String> {
        match (self.width, self.height) {
            (Some(w), Some(h)) => Some(format!("{w}x{h}")),
            _ => None,
        }
    }

    /// Returns the frame rate label (e.g., "23.976", "30.0").
    #[must_use]
    pub fn frame_rate_label(&self) -> Option<String> {
        self.frame_rate.map(|fps| format!("{fps:.3}"))
    }
}

/// A bin that holds organized clips.
#[derive(Debug, Clone)]
pub struct OrganizedBin {
    /// Bin name.
    pub name: String,
    /// Criteria that created this bin.
    pub criteria: OrganizeCriteria,
    /// Clip IDs in this bin.
    pub clip_ids: Vec<u64>,
    /// Sub-bins for hierarchical organization.
    pub sub_bins: Vec<OrganizedBin>,
}

impl OrganizedBin {
    /// Creates a new organized bin.
    #[must_use]
    pub fn new(name: impl Into<String>, criteria: OrganizeCriteria) -> Self {
        Self {
            name: name.into(),
            criteria,
            clip_ids: Vec::new(),
            sub_bins: Vec::new(),
        }
    }

    /// Adds a clip ID to this bin.
    pub fn add_clip(&mut self, clip_id: u64) {
        self.clip_ids.push(clip_id);
    }

    /// Returns the number of clips in this bin (not counting sub-bins).
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.clip_ids.len()
    }

    /// Returns the total clip count including sub-bins.
    #[must_use]
    pub fn total_clip_count(&self) -> usize {
        let sub_count: usize = self.sub_bins.iter().map(|b| b.total_clip_count()).sum();
        self.clip_ids.len() + sub_count
    }

    /// Returns true if this bin is empty (no clips and no sub-bins).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clip_ids.is_empty() && self.sub_bins.is_empty()
    }
}

/// Rule-based bin assignment.
#[derive(Debug, Clone)]
pub struct BinRule {
    /// Name of the target bin.
    pub bin_name: String,
    /// Criteria to match.
    pub criteria: OrganizeCriteria,
    /// Match value (e.g., specific date, camera name, etc.).
    pub match_value: String,
    /// Whether the match is case-sensitive.
    pub case_sensitive: bool,
}

impl BinRule {
    /// Creates a new bin rule.
    #[must_use]
    pub fn new(
        bin_name: impl Into<String>,
        criteria: OrganizeCriteria,
        match_value: impl Into<String>,
    ) -> Self {
        Self {
            bin_name: bin_name.into(),
            criteria,
            match_value: match_value.into(),
            case_sensitive: false,
        }
    }

    /// Tests whether a clip matches this rule.
    #[must_use]
    pub fn matches(&self, clip: &ClipDescriptor) -> bool {
        let clip_value = self.extract_value(clip);
        if self.case_sensitive {
            clip_value == self.match_value
        } else {
            clip_value.to_lowercase() == self.match_value.to_lowercase()
        }
    }

    /// Extracts the relevant value from a clip based on criteria.
    fn extract_value(&self, clip: &ClipDescriptor) -> String {
        match self.criteria {
            OrganizeCriteria::ByDate => clip.date.clone().unwrap_or_default(),
            OrganizeCriteria::ByCamera => clip.camera.clone().unwrap_or_default(),
            OrganizeCriteria::ByScene => clip.scene.clone().unwrap_or_default(),
            OrganizeCriteria::ByCodec => clip.codec.clone().unwrap_or_default(),
            OrganizeCriteria::ByResolution => clip.resolution_label().unwrap_or_default(),
            OrganizeCriteria::ByFrameRate => clip.frame_rate_label().unwrap_or_default(),
            OrganizeCriteria::ByRating => format!("{}", clip.rating),
            OrganizeCriteria::ByKeyword => clip.keywords.join(","),
            OrganizeCriteria::ByMediaType => format!("{:?}", clip.media_type),
            OrganizeCriteria::ByExtension => clip.extension.clone().unwrap_or_default(),
        }
    }
}

/// Bin organizer that sorts clips into bins.
#[derive(Debug)]
pub struct BinOrganizer {
    /// Custom rules for bin assignment.
    rules: Vec<BinRule>,
    /// Default criteria when no rules match.
    default_criteria: OrganizeCriteria,
}

impl BinOrganizer {
    /// Creates a new bin organizer with default date-based criteria.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            default_criteria: OrganizeCriteria::ByDate,
        }
    }

    /// Creates a new organizer with the specified default criteria.
    #[must_use]
    pub fn with_criteria(criteria: OrganizeCriteria) -> Self {
        Self {
            rules: Vec::new(),
            default_criteria: criteria,
        }
    }

    /// Adds a rule for bin assignment.
    pub fn add_rule(&mut self, rule: BinRule) {
        self.rules.push(rule);
    }

    /// Returns the number of rules.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Organizes clips into bins by the default criteria.
    #[must_use]
    pub fn organize(&self, clips: &[ClipDescriptor]) -> Vec<OrganizedBin> {
        self.organize_by(clips, self.default_criteria)
    }

    /// Organizes clips into bins by the specified criteria.
    #[must_use]
    pub fn organize_by(
        &self,
        clips: &[ClipDescriptor],
        criteria: OrganizeCriteria,
    ) -> Vec<OrganizedBin> {
        let mut groups: HashMap<String, Vec<u64>> = HashMap::new();

        for clip in clips {
            let key = self.get_grouping_key(clip, criteria);
            groups.entry(key).or_default().push(clip.clip_id);
        }

        let mut bins: Vec<OrganizedBin> = groups
            .into_iter()
            .map(|(name, ids)| {
                let mut bin = OrganizedBin::new(name, criteria);
                bin.clip_ids = ids;
                bin
            })
            .collect();

        bins.sort_by(|a, b| a.name.cmp(&b.name));
        bins
    }

    /// Applies custom rules to clips and returns assigned bins.
    #[must_use]
    pub fn apply_rules(&self, clips: &[ClipDescriptor]) -> Vec<OrganizedBin> {
        let mut bin_map: HashMap<String, OrganizedBin> = HashMap::new();

        for clip in clips {
            for rule in &self.rules {
                if rule.matches(clip) {
                    let bin = bin_map
                        .entry(rule.bin_name.clone())
                        .or_insert_with(|| OrganizedBin::new(&rule.bin_name, rule.criteria));
                    bin.add_clip(clip.clip_id);
                }
            }
        }

        let mut bins: Vec<OrganizedBin> = bin_map.into_values().collect();
        bins.sort_by(|a, b| a.name.cmp(&b.name));
        bins
    }

    /// Gets the grouping key for a clip based on criteria.
    fn get_grouping_key(&self, clip: &ClipDescriptor, criteria: OrganizeCriteria) -> String {
        match criteria {
            OrganizeCriteria::ByDate => clip.date.clone().unwrap_or_else(|| "Unknown Date".into()),
            OrganizeCriteria::ByCamera => clip
                .camera
                .clone()
                .unwrap_or_else(|| "Unknown Camera".into()),
            OrganizeCriteria::ByScene => clip.scene.clone().unwrap_or_else(|| "No Scene".into()),
            OrganizeCriteria::ByCodec => {
                clip.codec.clone().unwrap_or_else(|| "Unknown Codec".into())
            }
            OrganizeCriteria::ByResolution => clip
                .resolution_label()
                .unwrap_or_else(|| "Unknown Resolution".into()),
            OrganizeCriteria::ByFrameRate => clip
                .frame_rate_label()
                .unwrap_or_else(|| "Unknown FPS".into()),
            OrganizeCriteria::ByRating => {
                if clip.rating == 0 {
                    "Unrated".into()
                } else {
                    format!("{} Stars", clip.rating)
                }
            }
            OrganizeCriteria::ByKeyword => {
                if clip.keywords.is_empty() {
                    "No Keywords".into()
                } else {
                    clip.keywords.first().cloned().unwrap_or_default()
                }
            }
            OrganizeCriteria::ByMediaType => format!("{:?}", clip.media_type),
            OrganizeCriteria::ByExtension => {
                clip.extension.clone().unwrap_or_else(|| "Unknown".into())
            }
        }
    }
}

impl Default for BinOrganizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clip(id: u64, name: &str) -> ClipDescriptor {
        ClipDescriptor::new(id, name)
    }

    fn make_video_clip(id: u64, name: &str, date: &str, camera: &str) -> ClipDescriptor {
        let mut clip = ClipDescriptor::new(id, name);
        clip.date = Some(date.to_string());
        clip.camera = Some(camera.to_string());
        clip.media_type = ClipMediaType::Video;
        clip.width = Some(1920);
        clip.height = Some(1080);
        clip.frame_rate = Some(24.0);
        clip.codec = Some("H.264".to_string());
        clip.extension = Some("mov".to_string());
        clip
    }

    #[test]
    fn test_clip_descriptor_new() {
        let clip = make_clip(1, "Test Clip");
        assert_eq!(clip.clip_id, 1);
        assert_eq!(clip.name, "Test Clip");
        assert_eq!(clip.rating, 0);
    }

    #[test]
    fn test_resolution_label() {
        let mut clip = make_clip(1, "Clip");
        assert!(clip.resolution_label().is_none());
        clip.width = Some(1920);
        clip.height = Some(1080);
        assert_eq!(
            clip.resolution_label()
                .expect("resolution_label should succeed"),
            "1920x1080"
        );
    }

    #[test]
    fn test_frame_rate_label() {
        let mut clip = make_clip(1, "Clip");
        assert!(clip.frame_rate_label().is_none());
        clip.frame_rate = Some(23.976);
        assert_eq!(
            clip.frame_rate_label()
                .expect("frame_rate_label should succeed"),
            "23.976"
        );
    }

    #[test]
    fn test_organized_bin_clip_count() {
        let mut bin = OrganizedBin::new("Day 1", OrganizeCriteria::ByDate);
        assert_eq!(bin.clip_count(), 0);
        assert!(bin.is_empty());
        bin.add_clip(1);
        bin.add_clip(2);
        assert_eq!(bin.clip_count(), 2);
        assert!(!bin.is_empty());
    }

    #[test]
    fn test_organized_bin_total_clip_count() {
        let mut bin = OrganizedBin::new("Root", OrganizeCriteria::ByDate);
        bin.add_clip(1);
        let mut sub = OrganizedBin::new("Sub", OrganizeCriteria::ByCamera);
        sub.add_clip(2);
        sub.add_clip(3);
        bin.sub_bins.push(sub);
        assert_eq!(bin.total_clip_count(), 3);
    }

    #[test]
    fn test_bin_rule_matches() {
        let rule = BinRule::new("Cam A", OrganizeCriteria::ByCamera, "Camera A");
        let mut clip = make_clip(1, "Test");
        clip.camera = Some("Camera A".to_string());
        assert!(rule.matches(&clip));
    }

    #[test]
    fn test_bin_rule_case_insensitive() {
        let rule = BinRule::new("Cam A", OrganizeCriteria::ByCamera, "camera a");
        let mut clip = make_clip(1, "Test");
        clip.camera = Some("Camera A".to_string());
        assert!(rule.matches(&clip));
    }

    #[test]
    fn test_bin_rule_no_match() {
        let rule = BinRule::new("Cam A", OrganizeCriteria::ByCamera, "Camera B");
        let mut clip = make_clip(1, "Test");
        clip.camera = Some("Camera A".to_string());
        assert!(!rule.matches(&clip));
    }

    #[test]
    fn test_organize_by_date() {
        let clips = vec![
            make_video_clip(1, "Clip1", "2025-03-01", "Cam A"),
            make_video_clip(2, "Clip2", "2025-03-01", "Cam B"),
            make_video_clip(3, "Clip3", "2025-03-02", "Cam A"),
        ];
        let organizer = BinOrganizer::with_criteria(OrganizeCriteria::ByDate);
        let bins = organizer.organize(&clips);
        assert_eq!(bins.len(), 2);
        // Sorted alphabetically
        assert_eq!(bins[0].name, "2025-03-01");
        assert_eq!(bins[0].clip_count(), 2);
        assert_eq!(bins[1].name, "2025-03-02");
        assert_eq!(bins[1].clip_count(), 1);
    }

    #[test]
    fn test_organize_by_camera() {
        let clips = vec![
            make_video_clip(1, "Clip1", "2025-03-01", "Cam A"),
            make_video_clip(2, "Clip2", "2025-03-01", "Cam B"),
            make_video_clip(3, "Clip3", "2025-03-02", "Cam A"),
        ];
        let organizer = BinOrganizer::with_criteria(OrganizeCriteria::ByCamera);
        let bins = organizer.organize(&clips);
        assert_eq!(bins.len(), 2);
    }

    #[test]
    fn test_organize_by_resolution() {
        let mut clips = vec![
            make_video_clip(1, "Clip1", "2025-03-01", "Cam A"),
            make_video_clip(2, "Clip2", "2025-03-01", "Cam B"),
        ];
        clips[1].width = Some(1280);
        clips[1].height = Some(720);
        let organizer = BinOrganizer::with_criteria(OrganizeCriteria::ByResolution);
        let bins = organizer.organize(&clips);
        assert_eq!(bins.len(), 2);
    }

    #[test]
    fn test_organize_empty() {
        let organizer = BinOrganizer::new();
        let bins = organizer.organize(&[]);
        assert!(bins.is_empty());
    }

    #[test]
    fn test_apply_rules() {
        let clips = vec![
            make_video_clip(1, "Clip1", "2025-03-01", "Camera A"),
            make_video_clip(2, "Clip2", "2025-03-01", "Camera B"),
            make_video_clip(3, "Clip3", "2025-03-02", "Camera A"),
        ];
        let mut organizer = BinOrganizer::new();
        organizer.add_rule(BinRule::new(
            "A Cam",
            OrganizeCriteria::ByCamera,
            "Camera A",
        ));
        assert_eq!(organizer.rule_count(), 1);

        let bins = organizer.apply_rules(&clips);
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].name, "A Cam");
        assert_eq!(bins[0].clip_count(), 2);
    }

    #[test]
    fn test_organize_by_rating() {
        let mut clips = vec![
            make_clip(1, "Clip1"),
            make_clip(2, "Clip2"),
            make_clip(3, "Clip3"),
        ];
        clips[0].rating = 5;
        clips[1].rating = 3;
        clips[2].rating = 5;
        let organizer = BinOrganizer::with_criteria(OrganizeCriteria::ByRating);
        let bins = organizer.organize(&clips);
        assert_eq!(bins.len(), 2);
    }
}
