#![allow(dead_code)]
//! Conform manifest generation and tracking.
//!
//! Generates detailed manifests that record every decision made during a
//! conform session — source file mapping, timecode transforms, clip matching
//! confidence, and final output file locations.

use std::collections::HashMap;

/// Status of a single clip in the manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipStatus {
    /// Clip was matched and conformed successfully.
    Matched,
    /// Clip was matched but with low confidence.
    LowConfidence,
    /// Clip could not be matched to any source.
    Unmatched,
    /// Clip was explicitly skipped by the operator.
    Skipped,
    /// Clip match is ambiguous (multiple candidates).
    Ambiguous,
}

impl std::fmt::Display for ClipStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Matched => write!(f, "MATCHED"),
            Self::LowConfidence => write!(f, "LOW_CONFIDENCE"),
            Self::Unmatched => write!(f, "UNMATCHED"),
            Self::Skipped => write!(f, "SKIPPED"),
            Self::Ambiguous => write!(f, "AMBIGUOUS"),
        }
    }
}

/// A single clip entry in the conform manifest.
#[derive(Debug, Clone)]
pub struct ManifestClip {
    /// Unique clip identifier from the EDL/timeline.
    pub clip_id: String,
    /// Source reel name or tape name.
    pub reel_name: String,
    /// Timeline in-point as a timecode string.
    pub timeline_in: String,
    /// Timeline out-point as a timecode string.
    pub timeline_out: String,
    /// Source in-point as a timecode string.
    pub source_in: String,
    /// Source out-point as a timecode string.
    pub source_out: String,
    /// Path to the matched source file (if found).
    pub source_path: Option<String>,
    /// Matching confidence (0.0 to 1.0).
    pub confidence: f64,
    /// Status of this clip.
    pub status: ClipStatus,
    /// Additional notes or warnings.
    pub notes: Vec<String>,
}

impl ManifestClip {
    /// Create a new manifest clip entry.
    #[must_use]
    pub fn new(clip_id: String, reel_name: String) -> Self {
        Self {
            clip_id,
            reel_name,
            timeline_in: String::new(),
            timeline_out: String::new(),
            source_in: String::new(),
            source_out: String::new(),
            source_path: None,
            confidence: 0.0,
            status: ClipStatus::Unmatched,
            notes: Vec::new(),
        }
    }

    /// Whether this clip was successfully matched.
    #[must_use]
    pub fn is_matched(&self) -> bool {
        matches!(self.status, ClipStatus::Matched | ClipStatus::LowConfidence)
    }

    /// Add a note to this clip.
    pub fn add_note(&mut self, note: String) {
        self.notes.push(note);
    }
}

/// Summary statistics for a conform manifest.
#[derive(Debug, Clone)]
pub struct ManifestSummary {
    /// Total number of clips in the manifest.
    pub total_clips: usize,
    /// Number of successfully matched clips.
    pub matched_clips: usize,
    /// Number of unmatched clips.
    pub unmatched_clips: usize,
    /// Number of low-confidence matches.
    pub low_confidence_clips: usize,
    /// Number of ambiguous matches.
    pub ambiguous_clips: usize,
    /// Number of skipped clips.
    pub skipped_clips: usize,
    /// Average confidence across matched clips.
    pub avg_confidence: f64,
}

impl ManifestSummary {
    /// Match rate as a percentage (0.0 to 100.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn match_rate_percent(&self) -> f64 {
        if self.total_clips == 0 {
            return 0.0;
        }
        (self.matched_clips as f64 / self.total_clips as f64) * 100.0
    }

    /// Whether the conform is fully matched.
    #[must_use]
    pub fn is_fully_matched(&self) -> bool {
        self.unmatched_clips == 0 && self.ambiguous_clips == 0
    }
}

/// A complete conform manifest.
#[derive(Debug, Clone)]
pub struct ConformManifest {
    /// Session name.
    pub session_name: String,
    /// Creation timestamp (ISO 8601 string).
    pub created_at: String,
    /// Source EDL/timeline file path.
    pub source_file: String,
    /// Output directory path.
    pub output_dir: String,
    /// All clip entries.
    pub clips: Vec<ManifestClip>,
    /// Free-form metadata.
    pub metadata: HashMap<String, String>,
}

impl ConformManifest {
    /// Create a new empty manifest.
    #[must_use]
    pub fn new(session_name: String, source_file: String, output_dir: String) -> Self {
        Self {
            session_name,
            created_at: String::new(),
            source_file,
            output_dir,
            clips: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a clip entry.
    pub fn add_clip(&mut self, clip: ManifestClip) {
        self.clips.push(clip);
    }

    /// Set a metadata key-value pair.
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Compute summary statistics from the current clip list.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn summary(&self) -> ManifestSummary {
        let total = self.clips.len();
        let mut matched = 0;
        let mut unmatched = 0;
        let mut low_conf = 0;
        let mut ambiguous = 0;
        let mut skipped = 0;
        let mut conf_sum = 0.0_f64;
        let mut conf_count = 0;

        for clip in &self.clips {
            match clip.status {
                ClipStatus::Matched => {
                    matched += 1;
                    conf_sum += clip.confidence;
                    conf_count += 1;
                }
                ClipStatus::LowConfidence => {
                    matched += 1;
                    low_conf += 1;
                    conf_sum += clip.confidence;
                    conf_count += 1;
                }
                ClipStatus::Unmatched => unmatched += 1,
                ClipStatus::Ambiguous => ambiguous += 1,
                ClipStatus::Skipped => skipped += 1,
            }
        }

        let avg_confidence = if conf_count > 0 {
            conf_sum / f64::from(conf_count)
        } else {
            0.0
        };

        ManifestSummary {
            total_clips: total,
            matched_clips: matched,
            unmatched_clips: unmatched,
            low_confidence_clips: low_conf,
            ambiguous_clips: ambiguous,
            skipped_clips: skipped,
            avg_confidence,
        }
    }

    /// Find clips matching a given reel name.
    #[must_use]
    pub fn clips_for_reel(&self, reel_name: &str) -> Vec<&ManifestClip> {
        self.clips
            .iter()
            .filter(|c| c.reel_name == reel_name)
            .collect()
    }

    /// Find all unmatched clips.
    #[must_use]
    pub fn unmatched_clips(&self) -> Vec<&ManifestClip> {
        self.clips
            .iter()
            .filter(|c| c.status == ClipStatus::Unmatched)
            .collect()
    }

    /// Serialise the manifest to a simple text report.
    #[must_use]
    pub fn to_text_report(&self) -> String {
        let summary = self.summary();
        let mut out = String::new();
        out.push_str(&format!("Conform Manifest: {}\n", self.session_name));
        out.push_str(&format!("Source: {}\n", self.source_file));
        out.push_str(&format!("Output: {}\n", self.output_dir));
        out.push_str(&format!(
            "Clips: {} total, {} matched ({:.1}%)\n",
            summary.total_clips,
            summary.matched_clips,
            summary.match_rate_percent()
        ));
        out.push_str(&format!("Avg confidence: {:.2}\n", summary.avg_confidence));
        out.push('\n');
        for (i, clip) in self.clips.iter().enumerate() {
            out.push_str(&format!(
                "  [{i}] {}: {} -> {} [{}] conf={:.2}\n",
                clip.clip_id, clip.source_in, clip.source_out, clip.status, clip.confidence,
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clip(id: &str, reel: &str, status: ClipStatus, confidence: f64) -> ManifestClip {
        ManifestClip {
            clip_id: id.to_string(),
            reel_name: reel.to_string(),
            timeline_in: "01:00:00:00".to_string(),
            timeline_out: "01:00:10:00".to_string(),
            source_in: "00:10:00:00".to_string(),
            source_out: "00:10:10:00".to_string(),
            source_path: Some("/media/reel1.mxf".to_string()),
            confidence,
            status,
            notes: Vec::new(),
        }
    }

    #[test]
    fn test_new_manifest() {
        let m = ConformManifest::new(
            "Test".to_string(),
            "test.edl".to_string(),
            "/out".to_string(),
        );
        assert_eq!(m.clips.len(), 0);
        assert_eq!(m.session_name, "Test");
    }

    #[test]
    fn test_add_clip() {
        let mut m = ConformManifest::new(
            "Test".to_string(),
            "test.edl".to_string(),
            "/out".to_string(),
        );
        m.add_clip(make_clip("001", "R1", ClipStatus::Matched, 0.95));
        assert_eq!(m.clips.len(), 1);
    }

    #[test]
    fn test_summary_all_matched() {
        let mut m = ConformManifest::new(
            "Test".to_string(),
            "test.edl".to_string(),
            "/out".to_string(),
        );
        m.add_clip(make_clip("001", "R1", ClipStatus::Matched, 0.95));
        m.add_clip(make_clip("002", "R1", ClipStatus::Matched, 0.90));
        let s = m.summary();
        assert_eq!(s.total_clips, 2);
        assert_eq!(s.matched_clips, 2);
        assert!(s.is_fully_matched());
        assert!((s.match_rate_percent() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_summary_with_unmatched() {
        let mut m = ConformManifest::new(
            "Test".to_string(),
            "test.edl".to_string(),
            "/out".to_string(),
        );
        m.add_clip(make_clip("001", "R1", ClipStatus::Matched, 0.95));
        m.add_clip(make_clip("002", "R2", ClipStatus::Unmatched, 0.0));
        let s = m.summary();
        assert_eq!(s.unmatched_clips, 1);
        assert!(!s.is_fully_matched());
    }

    #[test]
    fn test_summary_avg_confidence() {
        let mut m = ConformManifest::new(
            "Test".to_string(),
            "test.edl".to_string(),
            "/out".to_string(),
        );
        m.add_clip(make_clip("001", "R1", ClipStatus::Matched, 0.80));
        m.add_clip(make_clip("002", "R1", ClipStatus::Matched, 1.00));
        let s = m.summary();
        assert!((s.avg_confidence - 0.90).abs() < 0.01);
    }

    #[test]
    fn test_clips_for_reel() {
        let mut m = ConformManifest::new(
            "Test".to_string(),
            "test.edl".to_string(),
            "/out".to_string(),
        );
        m.add_clip(make_clip("001", "R1", ClipStatus::Matched, 0.95));
        m.add_clip(make_clip("002", "R2", ClipStatus::Matched, 0.90));
        m.add_clip(make_clip("003", "R1", ClipStatus::Matched, 0.85));
        let r1_clips = m.clips_for_reel("R1");
        assert_eq!(r1_clips.len(), 2);
    }

    #[test]
    fn test_unmatched_clips() {
        let mut m = ConformManifest::new(
            "Test".to_string(),
            "test.edl".to_string(),
            "/out".to_string(),
        );
        m.add_clip(make_clip("001", "R1", ClipStatus::Matched, 0.95));
        m.add_clip(make_clip("002", "R2", ClipStatus::Unmatched, 0.0));
        let unmatched = m.unmatched_clips();
        assert_eq!(unmatched.len(), 1);
        assert_eq!(unmatched[0].clip_id, "002");
    }

    #[test]
    fn test_clip_status_display() {
        assert_eq!(format!("{}", ClipStatus::Matched), "MATCHED");
        assert_eq!(format!("{}", ClipStatus::Unmatched), "UNMATCHED");
        assert_eq!(format!("{}", ClipStatus::LowConfidence), "LOW_CONFIDENCE");
    }

    #[test]
    fn test_clip_is_matched() {
        let c1 = make_clip("001", "R1", ClipStatus::Matched, 0.95);
        assert!(c1.is_matched());
        let c2 = make_clip("002", "R1", ClipStatus::LowConfidence, 0.4);
        assert!(c2.is_matched());
        let c3 = make_clip("003", "R1", ClipStatus::Unmatched, 0.0);
        assert!(!c3.is_matched());
    }

    #[test]
    fn test_set_metadata() {
        let mut m = ConformManifest::new(
            "Test".to_string(),
            "test.edl".to_string(),
            "/out".to_string(),
        );
        m.set_metadata("operator".to_string(), "Jane".to_string());
        assert_eq!(
            m.metadata.get("operator").expect("get should succeed"),
            "Jane"
        );
    }

    #[test]
    fn test_text_report() {
        let mut m = ConformManifest::new(
            "My Conform".to_string(),
            "project.edl".to_string(),
            "/output".to_string(),
        );
        m.add_clip(make_clip("001", "R1", ClipStatus::Matched, 0.95));
        let report = m.to_text_report();
        assert!(report.contains("My Conform"));
        assert!(report.contains("100.0%"));
    }

    #[test]
    fn test_empty_manifest_summary() {
        let m = ConformManifest::new(
            "Empty".to_string(),
            "test.edl".to_string(),
            "/out".to_string(),
        );
        let s = m.summary();
        assert_eq!(s.total_clips, 0);
        assert!((s.match_rate_percent()).abs() < 0.01);
    }

    #[test]
    fn test_add_note() {
        let mut clip = ManifestClip::new("001".to_string(), "R1".to_string());
        clip.add_note("Source file missing audio track".to_string());
        assert_eq!(clip.notes.len(), 1);
        assert!(clip.notes[0].contains("audio track"));
    }
}
