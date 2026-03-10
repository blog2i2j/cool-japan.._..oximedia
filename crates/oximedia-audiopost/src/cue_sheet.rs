#![allow(dead_code)]
//! Cue sheet management for audio post-production.
//!
//! Models cue sheets that track music, sound effects, and dialogue placements
//! across a timeline. Supports import/export for interchange between DAWs
//! and editorial systems.

use std::fmt;

/// Type of cue (music, SFX, dialogue, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CueType {
    /// Music cue.
    Music,
    /// Sound effect.
    SoundEffect,
    /// Dialogue.
    Dialogue,
    /// Foley.
    Foley,
    /// Ambience / atmosphere.
    Ambience,
    /// Voice-over / narration.
    Narration,
}

impl fmt::Display for CueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Music => write!(f, "MX"),
            Self::SoundEffect => write!(f, "SFX"),
            Self::Dialogue => write!(f, "DX"),
            Self::Foley => write!(f, "FLY"),
            Self::Ambience => write!(f, "AMB"),
            Self::Narration => write!(f, "VO"),
        }
    }
}

/// A single cue entry on the cue sheet.
#[derive(Debug, Clone, PartialEq)]
pub struct CueEntry {
    /// Cue identifier (e.g. "1M1", "SFX-042").
    pub cue_id: String,
    /// Human-readable description.
    pub description: String,
    /// Cue type.
    pub cue_type: CueType,
    /// Start time in seconds.
    pub start_s: f64,
    /// End time in seconds.
    pub end_s: f64,
    /// Optional reel or scene reference.
    pub scene: Option<String>,
    /// Optional notes.
    pub notes: Option<String>,
}

impl CueEntry {
    /// Create a new cue entry.
    #[must_use]
    pub fn new(
        cue_id: &str,
        description: &str,
        cue_type: CueType,
        start_s: f64,
        end_s: f64,
    ) -> Self {
        Self {
            cue_id: cue_id.to_string(),
            description: description.to_string(),
            cue_type,
            start_s,
            end_s,
            scene: None,
            notes: None,
        }
    }

    /// Set optional scene reference.
    #[must_use]
    pub fn with_scene(mut self, scene: &str) -> Self {
        self.scene = Some(scene.to_string());
        self
    }

    /// Set optional notes.
    #[must_use]
    pub fn with_notes(mut self, notes: &str) -> Self {
        self.notes = Some(notes.to_string());
        self
    }

    /// Duration in seconds.
    #[must_use]
    pub fn duration_s(&self) -> f64 {
        (self.end_s - self.start_s).max(0.0)
    }

    /// Check if this cue overlaps with another.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start_s < other.end_s && other.start_s < self.end_s
    }

    /// Format timecode as HH:MM:SS.mmm.
    fn format_time(secs: f64) -> String {
        let total = secs.max(0.0);
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let h = (total / 3600.0) as u32;
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let m = ((total % 3600.0) / 60.0) as u32;
        let s = total % 60.0;
        format!("{h:02}:{m:02}:{s:06.3}")
    }
}

impl fmt::Display for CueEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{typ}] {id}: {desc} ({start} - {end})",
            typ = self.cue_type,
            id = self.cue_id,
            desc = self.description,
            start = Self::format_time(self.start_s),
            end = Self::format_time(self.end_s),
        )
    }
}

/// A complete cue sheet containing ordered cue entries.
#[derive(Debug, Clone)]
pub struct CueSheet {
    /// Project title.
    pub title: String,
    /// Cue entries sorted by start time.
    entries: Vec<CueEntry>,
    /// Frame rate for timecode display.
    pub frame_rate: f64,
}

impl CueSheet {
    /// Create a new empty cue sheet.
    #[must_use]
    pub fn new(title: &str, frame_rate: f64) -> Self {
        Self {
            title: title.to_string(),
            entries: Vec::new(),
            frame_rate,
        }
    }

    /// Add a cue and keep entries sorted by start time.
    pub fn add_cue(&mut self, entry: CueEntry) {
        let pos = self
            .entries
            .binary_search_by(|e| {
                e.start_s
                    .partial_cmp(&entry.start_s)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|p| p);
        self.entries.insert(pos, entry);
    }

    /// Remove a cue by its ID. Returns `true` if found and removed.
    pub fn remove_cue(&mut self, cue_id: &str) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.cue_id != cue_id);
        self.entries.len() < before
    }

    /// Number of cue entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the sheet is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get a cue by its ID.
    #[must_use]
    pub fn get_cue(&self, cue_id: &str) -> Option<&CueEntry> {
        self.entries.iter().find(|e| e.cue_id == cue_id)
    }

    /// Return all entries.
    #[must_use]
    pub fn entries(&self) -> &[CueEntry] {
        &self.entries
    }

    /// Filter entries by cue type.
    #[must_use]
    pub fn entries_by_type(&self, cue_type: CueType) -> Vec<&CueEntry> {
        self.entries
            .iter()
            .filter(|e| e.cue_type == cue_type)
            .collect()
    }

    /// Total duration covered (from first cue start to last cue end).
    #[must_use]
    pub fn total_duration_s(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let start = self.entries.first().map_or(0.0, |e| e.start_s);
        let end = self
            .entries
            .iter()
            .map(|e| e.end_s)
            .reduce(f64::max)
            .unwrap_or(0.0);
        (end - start).max(0.0)
    }

    /// Find all overlapping cue pairs.
    #[must_use]
    pub fn find_overlaps(&self) -> Vec<(&CueEntry, &CueEntry)> {
        let mut overlaps = Vec::new();
        for i in 0..self.entries.len() {
            for j in (i + 1)..self.entries.len() {
                if self.entries[i].overlaps(&self.entries[j]) {
                    overlaps.push((&self.entries[i], &self.entries[j]));
                }
            }
        }
        overlaps
    }
}

/// Cue sheet export format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Tab-separated values.
    Tsv,
    /// Comma-separated values.
    Csv,
    /// Simple text report.
    Text,
}

/// Exports cue sheets to various formats.
#[derive(Debug)]
pub struct CueSheetExporter;

impl CueSheetExporter {
    /// Export a cue sheet to the requested format.
    #[must_use]
    pub fn export(sheet: &CueSheet, format: ExportFormat) -> String {
        match format {
            ExportFormat::Csv => Self::to_csv(sheet),
            ExportFormat::Tsv => Self::to_tsv(sheet),
            ExportFormat::Text => Self::to_text(sheet),
        }
    }

    /// Export as CSV.
    fn to_csv(sheet: &CueSheet) -> String {
        let mut out = String::from("CueID,Type,Description,Start,End,Scene,Notes\n");
        for e in &sheet.entries {
            out.push_str(&format!(
                "{},{},{},{:.3},{:.3},{},{}\n",
                e.cue_id,
                e.cue_type,
                e.description,
                e.start_s,
                e.end_s,
                e.scene.as_deref().unwrap_or(""),
                e.notes.as_deref().unwrap_or(""),
            ));
        }
        out
    }

    /// Export as TSV.
    fn to_tsv(sheet: &CueSheet) -> String {
        let mut out = String::from("CueID\tType\tDescription\tStart\tEnd\tScene\tNotes\n");
        for e in &sheet.entries {
            out.push_str(&format!(
                "{}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{}\n",
                e.cue_id,
                e.cue_type,
                e.description,
                e.start_s,
                e.end_s,
                e.scene.as_deref().unwrap_or(""),
                e.notes.as_deref().unwrap_or(""),
            ));
        }
        out
    }

    /// Export as plain text report.
    fn to_text(sheet: &CueSheet) -> String {
        let mut out = format!("Cue Sheet: {}\n", sheet.title);
        out.push_str(&format!("Frame Rate: {:.2} fps\n", sheet.frame_rate));
        out.push_str(&format!("Total Cues: {}\n", sheet.len()));
        out.push_str("---\n");
        for e in &sheet.entries {
            out.push_str(&format!("{e}\n"));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cue_type_display() {
        assert_eq!(format!("{}", CueType::Music), "MX");
        assert_eq!(format!("{}", CueType::SoundEffect), "SFX");
        assert_eq!(format!("{}", CueType::Dialogue), "DX");
    }

    #[test]
    fn test_cue_entry_creation() {
        let cue = CueEntry::new("1M1", "Main title music", CueType::Music, 0.0, 30.0);
        assert_eq!(cue.cue_id, "1M1");
        assert!((cue.duration_s() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cue_entry_with_scene() {
        let cue = CueEntry::new("SFX-01", "Gunshot", CueType::SoundEffect, 10.0, 10.5)
            .with_scene("Scene 3")
            .with_notes("Close range");
        assert_eq!(cue.scene.as_deref(), Some("Scene 3"));
        assert_eq!(cue.notes.as_deref(), Some("Close range"));
    }

    #[test]
    fn test_cue_overlap_positive() {
        let a = CueEntry::new("A", "a", CueType::Music, 0.0, 10.0);
        let b = CueEntry::new("B", "b", CueType::Music, 5.0, 15.0);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_cue_overlap_negative() {
        let a = CueEntry::new("A", "a", CueType::Music, 0.0, 5.0);
        let b = CueEntry::new("B", "b", CueType::Music, 5.0, 10.0);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_cue_sheet_creation() {
        let sheet = CueSheet::new("My Film", 24.0);
        assert!(sheet.is_empty());
        assert_eq!(sheet.len(), 0);
    }

    #[test]
    fn test_cue_sheet_add_sorted() {
        let mut sheet = CueSheet::new("Film", 24.0);
        sheet.add_cue(CueEntry::new("B", "later", CueType::Music, 10.0, 20.0));
        sheet.add_cue(CueEntry::new("A", "first", CueType::Music, 0.0, 5.0));
        assert_eq!(sheet.entries()[0].cue_id, "A");
        assert_eq!(sheet.entries()[1].cue_id, "B");
    }

    #[test]
    fn test_cue_sheet_remove() {
        let mut sheet = CueSheet::new("Film", 24.0);
        sheet.add_cue(CueEntry::new("A", "a", CueType::Music, 0.0, 5.0));
        assert!(sheet.remove_cue("A"));
        assert!(sheet.is_empty());
    }

    #[test]
    fn test_cue_sheet_get_cue() {
        let mut sheet = CueSheet::new("Film", 24.0);
        sheet.add_cue(CueEntry::new(
            "FX-01",
            "Boom",
            CueType::SoundEffect,
            3.0,
            4.0,
        ));
        let found = sheet.get_cue("FX-01");
        assert!(found.is_some());
        assert_eq!(found.expect("found should be valid").description, "Boom");
    }

    #[test]
    fn test_entries_by_type() {
        let mut sheet = CueSheet::new("Film", 24.0);
        sheet.add_cue(CueEntry::new("1M1", "Title", CueType::Music, 0.0, 30.0));
        sheet.add_cue(CueEntry::new(
            "SFX-01",
            "Boom",
            CueType::SoundEffect,
            5.0,
            5.5,
        ));
        sheet.add_cue(CueEntry::new("2M1", "End", CueType::Music, 90.0, 120.0));
        let music = sheet.entries_by_type(CueType::Music);
        assert_eq!(music.len(), 2);
    }

    #[test]
    fn test_total_duration() {
        let mut sheet = CueSheet::new("Film", 24.0);
        sheet.add_cue(CueEntry::new("A", "a", CueType::Music, 10.0, 20.0));
        sheet.add_cue(CueEntry::new("B", "b", CueType::Music, 50.0, 80.0));
        assert!((sheet.total_duration_s() - 70.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_find_overlaps() {
        let mut sheet = CueSheet::new("Film", 24.0);
        sheet.add_cue(CueEntry::new("A", "a", CueType::Music, 0.0, 10.0));
        sheet.add_cue(CueEntry::new("B", "b", CueType::SoundEffect, 8.0, 12.0));
        sheet.add_cue(CueEntry::new("C", "c", CueType::Dialogue, 20.0, 30.0));
        let overlaps = sheet.find_overlaps();
        assert_eq!(overlaps.len(), 1);
    }

    #[test]
    fn test_export_csv() {
        let mut sheet = CueSheet::new("Film", 24.0);
        sheet.add_cue(CueEntry::new("1M1", "Theme", CueType::Music, 0.0, 30.0));
        let csv = CueSheetExporter::export(&sheet, ExportFormat::Csv);
        assert!(csv.contains("CueID,Type"));
        assert!(csv.contains("1M1"));
    }

    #[test]
    fn test_export_tsv() {
        let mut sheet = CueSheet::new("Film", 24.0);
        sheet.add_cue(CueEntry::new(
            "SFX-01",
            "Bang",
            CueType::SoundEffect,
            5.0,
            5.5,
        ));
        let tsv = CueSheetExporter::export(&sheet, ExportFormat::Tsv);
        assert!(tsv.contains("CueID\tType"));
        assert!(tsv.contains("SFX-01"));
    }

    #[test]
    fn test_export_text() {
        let mut sheet = CueSheet::new("My Film", 23.976);
        sheet.add_cue(CueEntry::new("1M1", "Theme", CueType::Music, 0.0, 30.0));
        let text = CueSheetExporter::export(&sheet, ExportFormat::Text);
        assert!(text.contains("Cue Sheet: My Film"));
        assert!(text.contains("Total Cues: 1"));
    }

    #[test]
    fn test_cue_entry_display() {
        let cue = CueEntry::new("1M1", "Main theme", CueType::Music, 61.5, 120.0);
        let display = format!("{cue}");
        assert!(display.contains("[MX]"));
        assert!(display.contains("1M1"));
    }
}
