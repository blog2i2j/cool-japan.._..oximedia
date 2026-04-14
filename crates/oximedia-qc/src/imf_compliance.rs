//! IMF (Interoperable Master Format) compliance checking.
//!
//! Implements SMPTE ST 2067-2 and ST 2084 (PQ) compliance rules via the
//! `ImfComplianceChecker` which implements the [`crate::rules::QcRule`] trait.
//!
//! ## Standards Covered
//!
//! - **SMPTE ST 2067-2**: IMF Core Constraints — CPL structure, PKL hash integrity,
//!   audio layout, track file format constraints.
//! - **SMPTE ST 2084**: Perceptual Quantizer (PQ) EOTF — valid signal range (0–10000 cd/m²),
//!   mastering display peak luminance, maxCLL / maxFALL.

#![allow(dead_code)]

use crate::rules::{CheckResult, QcContext, QcRule, RuleCategory, Severity};
use oximedia_core::OxiResult;

// ---------------------------------------------------------------------------
// IMF-specific types
// ---------------------------------------------------------------------------

/// Valid audio channel count options per ST 2067-2 §9.
const IMF_VALID_AUDIO_CHANNEL_COUNTS: &[u32] = &[1, 2, 4, 6, 8, 16, 24];

/// Minimum number of CPL edit units required for a valid package.
const IMF_MIN_CPL_EDIT_UNITS: u64 = 1;

/// Maximum allowed PQ signal value (corresponds to 10 000 cd/m²).
const PQ_MAX_SIGNAL: f64 = 1.0;

/// Minimum valid PQ signal value.
const PQ_MIN_SIGNAL: f64 = 0.0;

/// Minimum mastering display peak luminance in cd/m² per ST 2084.
const PQ_MIN_PEAK_LUMINANCE: f64 = 1000.0;

/// Maximum mastering display peak luminance in cd/m² per ST 2084.
const PQ_MAX_PEAK_LUMINANCE: f64 = 10000.0;

// ---------------------------------------------------------------------------
// CPL (Composition Play List) structure
// ---------------------------------------------------------------------------

/// Represents a validated CPL structure as parsed from IMF metadata.
#[derive(Debug, Clone)]
pub struct CplStructure {
    /// Unique identifier for this CPL.
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// Total edit unit count.
    pub edit_unit_count: u64,
    /// Frame rate numerator.
    pub frame_rate_numerator: u32,
    /// Frame rate denominator.
    pub frame_rate_denominator: u32,
    /// Whether a video track file reference is present.
    pub has_video_track: bool,
    /// Whether at least one audio track file reference is present.
    pub has_audio_track: bool,
    /// Audio channel count (sum across all audio tracks).
    pub audio_channel_count: u32,
}

impl CplStructure {
    /// Creates a new CPL structure with the given parameters.
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        edit_unit_count: u64,
        frame_rate_numerator: u32,
        frame_rate_denominator: u32,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            edit_unit_count,
            frame_rate_numerator,
            frame_rate_denominator,
            has_video_track: false,
            has_audio_track: false,
            audio_channel_count: 0,
        }
    }

    /// Returns the rational frame rate as a floating-point value.
    pub fn frame_rate(&self) -> f64 {
        if self.frame_rate_denominator == 0 {
            return 0.0;
        }
        self.frame_rate_numerator as f64 / self.frame_rate_denominator as f64
    }
}

// ---------------------------------------------------------------------------
// PKL (Packing List) hash entry
// ---------------------------------------------------------------------------

/// Represents a single PKL hash entry for a track file.
#[derive(Debug, Clone)]
pub struct PklHashEntry {
    /// Track file UUID.
    pub uuid: String,
    /// Expected hash value (hex-encoded SHA-256 or MD5).
    pub expected_hash: String,
    /// Computed hash value (None if not yet computed / file missing).
    pub computed_hash: Option<String>,
}

impl PklHashEntry {
    /// Creates a new hash entry.
    pub fn new(
        uuid: impl Into<String>,
        expected_hash: impl Into<String>,
        computed_hash: Option<String>,
    ) -> Self {
        Self {
            uuid: uuid.into(),
            expected_hash: expected_hash.into(),
            computed_hash,
        }
    }

    /// Returns `true` if the computed hash matches the expected hash.
    pub fn is_valid(&self) -> bool {
        match &self.computed_hash {
            Some(computed) => computed.eq_ignore_ascii_case(&self.expected_hash),
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// PQ signal metadata
// ---------------------------------------------------------------------------

/// PQ (SMPTE ST 2084) signal metadata for a content stream.
#[derive(Debug, Clone)]
pub struct PqSignalMetadata {
    /// Minimum PQ signal value in the stream (normalized 0.0–1.0).
    pub min_signal: f64,
    /// Maximum PQ signal value in the stream (normalized 0.0–1.0).
    pub max_signal: f64,
    /// Mastering display peak luminance in cd/m².
    pub mastering_peak_luminance: f64,
    /// Maximum Content Light Level (MaxCLL) in cd/m².
    pub max_cll: f64,
    /// Maximum Frame-Average Light Level (MaxFALL) in cd/m².
    pub max_fall: f64,
}

impl PqSignalMetadata {
    /// Creates a new PQ metadata instance.
    pub fn new(
        min_signal: f64,
        max_signal: f64,
        mastering_peak_luminance: f64,
        max_cll: f64,
        max_fall: f64,
    ) -> Self {
        Self {
            min_signal,
            max_signal,
            mastering_peak_luminance,
            max_cll,
            max_fall,
        }
    }
}

// ---------------------------------------------------------------------------
// IMF compliance input
// ---------------------------------------------------------------------------

/// Input data for IMF compliance checking.
///
/// Populate this struct from parsed CPL/PKL XML before running the checker.
#[derive(Debug, Clone, Default)]
pub struct ImfComplianceInput {
    /// Parsed CPL structures in this IMP.
    pub cpls: Vec<CplStructure>,
    /// PKL hash entries for all track files.
    pub pkl_hashes: Vec<PklHashEntry>,
    /// PQ metadata, if the content uses PQ transfer function.
    pub pq_metadata: Option<PqSignalMetadata>,
    /// Whether the IMP claims PQ (ST 2084) HDR content.
    pub is_pq_content: bool,
}

impl ImfComplianceInput {
    /// Creates an empty input set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a CPL structure.
    pub fn add_cpl(&mut self, cpl: CplStructure) {
        self.cpls.push(cpl);
    }

    /// Adds a PKL hash entry.
    pub fn add_pkl_hash(&mut self, entry: PklHashEntry) {
        self.pkl_hashes.push(entry);
    }
}

// ---------------------------------------------------------------------------
// Compliance checker
// ---------------------------------------------------------------------------

/// Performs IMF compliance checks per SMPTE ST 2067-2 and ST 2084.
///
/// Can be used standalone via [`check_imf`](ImfComplianceChecker::check_imf) or
/// as a [`QcRule`] plug-in that attaches its results to a [`QcContext`].
#[derive(Debug, Clone)]
pub struct ImfComplianceChecker {
    /// IMF data to validate (set before running checks).
    pub input: ImfComplianceInput,
    /// Minimum required frame rate for video tracks.
    pub min_frame_rate: f64,
    /// Maximum allowed frame rate for video tracks.
    pub max_frame_rate: f64,
}

impl ImfComplianceChecker {
    /// Creates a new checker with the given input and default settings.
    pub fn new(input: ImfComplianceInput) -> Self {
        Self {
            input,
            min_frame_rate: 23.976,
            max_frame_rate: 60.0,
        }
    }

    /// Sets the allowed frame rate range.
    pub fn with_frame_rate_range(mut self, min: f64, max: f64) -> Self {
        self.min_frame_rate = min;
        self.max_frame_rate = max;
        self
    }

    /// Runs all IMF compliance checks and returns a list of [`CheckResult`] values.
    pub fn check_imf(&self) -> Vec<CheckResult> {
        let mut results: Vec<CheckResult> = Vec::new();

        self.check_cpl_structure(&mut results);
        self.check_pkl_hashes(&mut results);
        self.check_audio_layout(&mut results);

        if self.input.is_pq_content {
            self.check_pq_signal(&mut results);
        }

        results
    }

    // -----------------------------------------------------------------------
    // CPL structure checks (ST 2067-2 §8)
    // -----------------------------------------------------------------------
    fn check_cpl_structure(&self, results: &mut Vec<CheckResult>) {
        if self.input.cpls.is_empty() {
            results.push(
                CheckResult::fail(
                    "IMF-CPL-001",
                    Severity::Error,
                    "IMP contains no CPL documents; at least one CPL is required (ST 2067-2 §8.1)",
                )
                .with_recommendation("Add a valid CPL XML document to the IMP."),
            );
            return;
        }

        for cpl in &self.input.cpls {
            // Edit unit count
            if cpl.edit_unit_count < IMF_MIN_CPL_EDIT_UNITS {
                results.push(
                    CheckResult::fail(
                        "IMF-CPL-002",
                        Severity::Error,
                        format!(
                            "CPL '{}' has {} edit units; at least {} required",
                            cpl.id, cpl.edit_unit_count, IMF_MIN_CPL_EDIT_UNITS
                        ),
                    )
                    .with_recommendation("Ensure the CPL references at least one edit unit."),
                );
            }

            // Frame rate validity
            let fps = cpl.frame_rate();
            if fps < self.min_frame_rate || fps > self.max_frame_rate {
                results.push(
                    CheckResult::fail(
                        "IMF-CPL-003",
                        Severity::Error,
                        format!(
                            "CPL '{}' frame rate {}/{} ({:.3} fps) is outside allowed range \
                             [{:.3}, {:.3}] fps",
                            cpl.id,
                            cpl.frame_rate_numerator,
                            cpl.frame_rate_denominator,
                            fps,
                            self.min_frame_rate,
                            self.max_frame_rate
                        ),
                    )
                    .with_recommendation(
                        "Use a standard frame rate (23.976, 24, 25, 29.97, 30, 50, 59.94, 60).",
                    ),
                );
            }

            // Video track presence
            if !cpl.has_video_track {
                results.push(
                    CheckResult::fail(
                        "IMF-CPL-004",
                        Severity::Warning,
                        format!("CPL '{}' contains no video track file references", cpl.id),
                    )
                    .with_recommendation("Add a main image track (MXF OP-1a) to the CPL."),
                );
            }

            // CPL title must not be empty
            if cpl.title.trim().is_empty() {
                results.push(
                    CheckResult::fail(
                        "IMF-CPL-005",
                        Severity::Warning,
                        format!("CPL '{}' has an empty ContentTitle element", cpl.id),
                    )
                    .with_recommendation(
                        "Populate the ContentTitle element with a descriptive title.",
                    ),
                );
            }
        }

        // Aggregate pass if no failures were added for CPL structure
        if self.input.cpls.iter().all(|c| {
            c.edit_unit_count >= IMF_MIN_CPL_EDIT_UNITS && c.has_video_track && {
                let fps = c.frame_rate();
                fps >= self.min_frame_rate && fps <= self.max_frame_rate
            }
        }) {
            results.push(CheckResult::pass("IMF-CPL-OK"));
        }
    }

    // -----------------------------------------------------------------------
    // PKL hash integrity checks (ST 2067-2 §9)
    // -----------------------------------------------------------------------
    fn check_pkl_hashes(&self, results: &mut Vec<CheckResult>) {
        if self.input.pkl_hashes.is_empty() {
            results.push(
                CheckResult::fail(
                    "IMF-PKL-001",
                    Severity::Warning,
                    "PKL contains no hash entries; track file integrity cannot be verified",
                )
                .with_recommendation("Ensure the PKL lists hash values for all track files."),
            );
            return;
        }

        let mut all_valid = true;
        for entry in &self.input.pkl_hashes {
            if entry.computed_hash.is_none() {
                results.push(
                    CheckResult::fail(
                        "IMF-PKL-002",
                        Severity::Error,
                        format!(
                            "Track file '{}' listed in PKL is missing or its hash could not be computed",
                            entry.uuid
                        ),
                    )
                    .with_recommendation("Verify that all track files referenced in the PKL are present."),
                );
                all_valid = false;
            } else if !entry.is_valid() {
                results.push(
                    CheckResult::fail(
                        "IMF-PKL-003",
                        Severity::Error,
                        format!(
                            "Hash mismatch for track file '{}': expected '{}', got '{}'",
                            entry.uuid,
                            entry.expected_hash,
                            entry.computed_hash.as_deref().unwrap_or("<none>")
                        ),
                    )
                    .with_recommendation(
                        "Re-deliver the affected track file or update the PKL hash.",
                    ),
                );
                all_valid = false;
            }
        }

        if all_valid {
            results.push(CheckResult::pass("IMF-PKL-OK"));
        }
    }

    // -----------------------------------------------------------------------
    // Audio layout checks (ST 2067-2 §9, ST 2067-8)
    // -----------------------------------------------------------------------
    fn check_audio_layout(&self, results: &mut Vec<CheckResult>) {
        for cpl in &self.input.cpls {
            if !cpl.has_audio_track {
                results.push(
                    CheckResult::fail(
                        "IMF-AUDIO-001",
                        Severity::Warning,
                        format!("CPL '{}' contains no audio track file references", cpl.id),
                    )
                    .with_recommendation("Add main audio track files conforming to ST 2067-8."),
                );
                continue;
            }

            let ch = cpl.audio_channel_count;
            if !IMF_VALID_AUDIO_CHANNEL_COUNTS.contains(&ch) {
                results.push(
                    CheckResult::fail(
                        "IMF-AUDIO-002",
                        Severity::Error,
                        format!(
                            "CPL '{}' has {ch} audio channels; \
                             ST 2067-2 permits {:?} channels only",
                            cpl.id, IMF_VALID_AUDIO_CHANNEL_COUNTS
                        ),
                    )
                    .with_recommendation(
                        "Adjust the audio track channel layout to a permitted count.",
                    ),
                );
            } else {
                results.push(CheckResult::pass("IMF-AUDIO-OK"));
            }
        }
    }

    // -----------------------------------------------------------------------
    // PQ signal range checks (ST 2084)
    // -----------------------------------------------------------------------
    fn check_pq_signal(&self, results: &mut Vec<CheckResult>) {
        let meta = match &self.input.pq_metadata {
            Some(m) => m,
            None => {
                results.push(
                    CheckResult::fail(
                        "IMF-PQ-001",
                        Severity::Error,
                        "Content is declared as PQ HDR but no PQ signal metadata was provided",
                    )
                    .with_recommendation(
                        "Include SMPTE ST 2086 mastering display metadata in the IMF package.",
                    ),
                );
                return;
            }
        };

        // Signal range
        if meta.min_signal < PQ_MIN_SIGNAL || meta.max_signal > PQ_MAX_SIGNAL {
            results.push(
                CheckResult::fail(
                    "IMF-PQ-002",
                    Severity::Error,
                    format!(
                        "PQ signal range [{:.4}, {:.4}] is outside the valid ST 2084 range \
                         [{PQ_MIN_SIGNAL:.1}, {PQ_MAX_SIGNAL:.1}]",
                        meta.min_signal, meta.max_signal
                    ),
                )
                .with_recommendation("Verify that the PQ EOTF is applied correctly and signal values are normalized."),
            );
        }

        // Mastering peak luminance
        if meta.mastering_peak_luminance < PQ_MIN_PEAK_LUMINANCE
            || meta.mastering_peak_luminance > PQ_MAX_PEAK_LUMINANCE
        {
            results.push(
                CheckResult::fail(
                    "IMF-PQ-003",
                    Severity::Error,
                    format!(
                        "Mastering display peak luminance {:.0} cd/m² is outside the \
                         permitted range [{PQ_MIN_PEAK_LUMINANCE:.0}, {PQ_MAX_PEAK_LUMINANCE:.0}] cd/m²",
                        meta.mastering_peak_luminance
                    ),
                )
                .with_recommendation("Use a mastering display with a peak luminance between 1000 and 10 000 cd/m²."),
            );
        }

        // MaxCLL must not exceed mastering peak
        if meta.max_cll > meta.mastering_peak_luminance {
            results.push(
                CheckResult::fail(
                    "IMF-PQ-004",
                    Severity::Warning,
                    format!(
                        "MaxCLL ({:.0} cd/m²) exceeds mastering display peak luminance ({:.0} cd/m²)",
                        meta.max_cll, meta.mastering_peak_luminance
                    ),
                )
                .with_recommendation("Review HDR mastering metadata; MaxCLL should not exceed display peak luminance."),
            );
        }

        // MaxFALL must not exceed MaxCLL
        if meta.max_fall > meta.max_cll {
            results.push(
                CheckResult::fail(
                    "IMF-PQ-005",
                    Severity::Warning,
                    format!(
                        "MaxFALL ({:.0} cd/m²) exceeds MaxCLL ({:.0} cd/m²)",
                        meta.max_fall, meta.max_cll
                    ),
                )
                .with_recommendation(
                    "MaxFALL (frame-average) cannot exceed MaxCLL (single pixel maximum).",
                ),
            );
        }

        // All PQ checks passed
        let pq_passed = meta.min_signal >= PQ_MIN_SIGNAL
            && meta.max_signal <= PQ_MAX_SIGNAL
            && meta.mastering_peak_luminance >= PQ_MIN_PEAK_LUMINANCE
            && meta.mastering_peak_luminance <= PQ_MAX_PEAK_LUMINANCE
            && meta.max_cll <= meta.mastering_peak_luminance
            && meta.max_fall <= meta.max_cll;

        if pq_passed {
            results.push(CheckResult::pass("IMF-PQ-OK"));
        }
    }
}

// ---------------------------------------------------------------------------
// QcRule implementation
// ---------------------------------------------------------------------------

impl QcRule for ImfComplianceChecker {
    fn name(&self) -> &str {
        "imf_compliance"
    }

    fn category(&self) -> RuleCategory {
        RuleCategory::Compliance
    }

    fn description(&self) -> &str {
        "IMF compliance checks per SMPTE ST 2067-2 and ST 2084 (PQ)"
    }

    fn check(&self, _context: &QcContext) -> OxiResult<Vec<CheckResult>> {
        Ok(self.check_imf())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_cpl() -> CplStructure {
        let mut cpl = CplStructure::new("cpl-001", "Test Feature Film", 100, 24000, 1001);
        cpl.has_video_track = true;
        cpl.has_audio_track = true;
        cpl.audio_channel_count = 6;
        cpl
    }

    fn valid_pkl() -> Vec<PklHashEntry> {
        vec![
            PklHashEntry::new("uuid-001", "abc123", Some("abc123".to_string())),
            PklHashEntry::new("uuid-002", "def456", Some("def456".to_string())),
        ]
    }

    fn valid_pq() -> PqSignalMetadata {
        PqSignalMetadata::new(0.0, 0.95, 4000.0, 4000.0, 400.0)
    }

    #[test]
    fn test_cpl_structure_valid() {
        let mut input = ImfComplianceInput::new();
        input.add_cpl(valid_cpl());
        for e in valid_pkl() {
            input.add_pkl_hash(e);
        }
        let checker = ImfComplianceChecker::new(input);
        let results = checker.check_imf();
        let failures: Vec<_> = results.iter().filter(|r| !r.passed).collect();
        assert!(failures.is_empty(), "Expected no failures: {:?}", failures);
    }

    #[test]
    fn test_no_cpls_produces_error() {
        let input = ImfComplianceInput::new();
        let checker = ImfComplianceChecker::new(input);
        let results = checker.check_imf();
        assert!(results
            .iter()
            .any(|r| !r.passed && r.rule_name == "IMF-CPL-001"));
    }

    #[test]
    fn test_cpl_zero_edit_units() {
        let mut cpl = valid_cpl();
        cpl.edit_unit_count = 0;
        let mut input = ImfComplianceInput::new();
        input.add_cpl(cpl);
        let checker = ImfComplianceChecker::new(input);
        let results = checker.check_imf();
        assert!(results
            .iter()
            .any(|r| !r.passed && r.rule_name == "IMF-CPL-002"));
    }

    #[test]
    fn test_cpl_invalid_frame_rate() {
        let mut cpl = valid_cpl();
        cpl.frame_rate_numerator = 1;
        cpl.frame_rate_denominator = 1; // 1 fps - way too slow
        let mut input = ImfComplianceInput::new();
        input.add_cpl(cpl);
        let checker = ImfComplianceChecker::new(input);
        let results = checker.check_imf();
        assert!(results
            .iter()
            .any(|r| !r.passed && r.rule_name == "IMF-CPL-003"));
    }

    #[test]
    fn test_pkl_hash_mismatch() {
        let mut input = ImfComplianceInput::new();
        input.add_cpl(valid_cpl());
        input.add_pkl_hash(PklHashEntry::new(
            "uuid-001",
            "abc123",
            Some("WRONG".to_string()),
        ));
        let checker = ImfComplianceChecker::new(input);
        let results = checker.check_imf();
        assert!(results
            .iter()
            .any(|r| !r.passed && r.rule_name == "IMF-PKL-003"));
    }

    #[test]
    fn test_pkl_hash_missing_file() {
        let mut input = ImfComplianceInput::new();
        input.add_cpl(valid_cpl());
        input.add_pkl_hash(PklHashEntry::new("uuid-001", "abc123", None));
        let checker = ImfComplianceChecker::new(input);
        let results = checker.check_imf();
        assert!(results
            .iter()
            .any(|r| !r.passed && r.rule_name == "IMF-PKL-002"));
    }

    #[test]
    fn test_invalid_audio_channel_count() {
        let mut cpl = valid_cpl();
        cpl.audio_channel_count = 7; // Not in valid list
        let mut input = ImfComplianceInput::new();
        input.add_cpl(cpl);
        for e in valid_pkl() {
            input.add_pkl_hash(e);
        }
        let checker = ImfComplianceChecker::new(input);
        let results = checker.check_imf();
        assert!(results
            .iter()
            .any(|r| !r.passed && r.rule_name == "IMF-AUDIO-002"));
    }

    #[test]
    fn test_pq_signal_out_of_range() {
        let mut input = ImfComplianceInput::new();
        input.add_cpl(valid_cpl());
        for e in valid_pkl() {
            input.add_pkl_hash(e);
        }
        input.is_pq_content = true;
        input.pq_metadata = Some(PqSignalMetadata::new(
            -0.1, // below 0
            1.0, 4000.0, 4000.0, 400.0,
        ));
        let checker = ImfComplianceChecker::new(input);
        let results = checker.check_imf();
        assert!(results
            .iter()
            .any(|r| !r.passed && r.rule_name == "IMF-PQ-002"));
    }

    #[test]
    fn test_pq_max_cll_exceeds_peak() {
        let mut input = ImfComplianceInput::new();
        input.add_cpl(valid_cpl());
        for e in valid_pkl() {
            input.add_pkl_hash(e);
        }
        input.is_pq_content = true;
        input.pq_metadata = Some(PqSignalMetadata::new(
            0.0, 0.95, 1000.0, 1500.0, // MaxCLL > mastering peak
            400.0,
        ));
        let checker = ImfComplianceChecker::new(input);
        let results = checker.check_imf();
        assert!(results
            .iter()
            .any(|r| !r.passed && r.rule_name == "IMF-PQ-004"));
    }

    #[test]
    fn test_pq_max_fall_exceeds_max_cll() {
        let mut input = ImfComplianceInput::new();
        input.add_cpl(valid_cpl());
        for e in valid_pkl() {
            input.add_pkl_hash(e);
        }
        input.is_pq_content = true;
        input.pq_metadata = Some(PqSignalMetadata::new(
            0.0, 0.9, 4000.0, 800.0, 1200.0, // MaxFALL > MaxCLL
        ));
        let checker = ImfComplianceChecker::new(input);
        let results = checker.check_imf();
        assert!(results
            .iter()
            .any(|r| !r.passed && r.rule_name == "IMF-PQ-005"));
    }

    #[test]
    fn test_imf_rule_trait() {
        let input = ImfComplianceInput::new();
        let checker = ImfComplianceChecker::new(input);
        assert_eq!(checker.name(), "imf_compliance");
        assert_eq!(checker.category(), RuleCategory::Compliance);
    }

    #[test]
    fn test_pkl_hash_case_insensitive_match() {
        let entry = PklHashEntry::new("uuid-001", "ABC123", Some("abc123".to_string()));
        assert!(entry.is_valid());
    }

    #[test]
    fn test_pq_no_metadata_produces_error() {
        let mut input = ImfComplianceInput::new();
        input.add_cpl(valid_cpl());
        for e in valid_pkl() {
            input.add_pkl_hash(e);
        }
        input.is_pq_content = true;
        // pq_metadata is None
        let checker = ImfComplianceChecker::new(input);
        let results = checker.check_imf();
        assert!(results
            .iter()
            .any(|r| !r.passed && r.rule_name == "IMF-PQ-001"));
    }
}
