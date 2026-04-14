//! QC extensions: new checker types requested in TODO.md (0.1.3 session).
//!
//! Provides:
//! - `DolbyVisionQc::check_l1_metadata` – L1 metadata validation
//! - `HdrQc::check_peak_luminance` – per-profile peak luminance gating
//! - `TemporalQcChecker::detect_frozen_frames` – freeze segment detection
//! - `ClosedCaptionQcChecker::check_timing` – overlap / backwards-timing detection
//! - `BitrateQcChecker::check_cbr_compliance` – CBR tolerance enforcement
//! - `ColorGamutQc::check_sdr_out_of_range` – SDR out-of-range pixel ratio
//! - `FormatQcChecker::check_container_codec_compatibility` – container/codec compat
//! - `ComplianceReport::to_csv` – CSV export
//! - `QcScheduler` + `QcDatabase` – FIFO scheduling and in-memory persistence

use std::collections::{HashMap, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// Shared primitive types
// ─────────────────────────────────────────────────────────────────────────────

/// Severity of a QC issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QcIssueSeverity {
    /// Informational finding.
    Info,
    /// Warning — may need attention.
    Warning,
    /// Error — must be fixed before delivery.
    Error,
}

impl QcIssueSeverity {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

impl std::fmt::Display for QcIssueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// A single QC issue found during validation.
#[derive(Debug, Clone)]
pub struct QcIssue {
    /// Short code identifying the check (e.g. `"DV-L1-RANGE"`).
    pub code: String,
    /// Severity of the issue.
    pub severity: QcIssueSeverity,
    /// Optional timecode / frame index where the issue occurs.
    pub timecode: Option<String>,
    /// Human-readable description.
    pub description: String,
}

impl QcIssue {
    /// Create a new error-severity issue.
    pub fn error(code: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            severity: QcIssueSeverity::Error,
            timecode: None,
            description: description.into(),
        }
    }

    /// Create a new warning-severity issue.
    pub fn warning(code: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            severity: QcIssueSeverity::Warning,
            timecode: None,
            description: description.into(),
        }
    }

    /// Attach a timecode string.
    #[must_use]
    pub fn with_timecode(mut self, tc: impl Into<String>) -> Self {
        self.timecode = Some(tc.into());
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Dolby Vision QC – L1 metadata validation
// ─────────────────────────────────────────────────────────────────────────────

/// Dolby Vision Level-1 metadata block (per scene / per frame).
#[derive(Debug, Clone)]
pub struct DvL1Metadata {
    /// Minimum PQ value (0–4095).
    pub min_pq: u16,
    /// Mid PQ value (0–4095).
    pub mid_pq: u16,
    /// Maximum PQ value (0–4095).
    pub max_pq: u16,
}

impl DvL1Metadata {
    /// Construct an L1 metadata block.
    #[must_use]
    pub fn new(min_pq: u16, mid_pq: u16, max_pq: u16) -> Self {
        Self {
            min_pq,
            mid_pq,
            max_pq,
        }
    }
}

/// Dolby Vision quality control checker.
pub struct DolbyVisionQc;

impl DolbyVisionQc {
    /// Validate a Dolby Vision Level-1 metadata block.
    ///
    /// Rules enforced:
    /// - All values must be in [0, 4095].
    /// - `min_pq < mid_pq < max_pq` must hold (strict ordering).
    #[must_use]
    pub fn check_l1_metadata(l1: &DvL1Metadata) -> Vec<QcIssue> {
        let mut issues = Vec::new();

        // Range check
        for (label, value) in [
            ("min_pq", l1.min_pq),
            ("mid_pq", l1.mid_pq),
            ("max_pq", l1.max_pq),
        ] {
            if value > 4095 {
                issues.push(QcIssue::error(
                    "DV-L1-RANGE",
                    format!("{label} value {value} exceeds maximum of 4095"),
                ));
            }
        }

        // Ordering check: min_pq < mid_pq
        if l1.min_pq >= l1.mid_pq {
            issues.push(QcIssue::error(
                "DV-L1-ORDER",
                format!(
                    "min_pq ({}) must be strictly less than mid_pq ({})",
                    l1.min_pq, l1.mid_pq
                ),
            ));
        }

        // Ordering check: mid_pq < max_pq
        if l1.mid_pq >= l1.max_pq {
            issues.push(QcIssue::error(
                "DV-L1-ORDER",
                format!(
                    "mid_pq ({}) must be strictly less than max_pq ({})",
                    l1.mid_pq, l1.max_pq
                ),
            ));
        }

        issues
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. HDR QC – peak luminance check
// ─────────────────────────────────────────────────────────────────────────────

/// HDR delivery profile for peak luminance gating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrProfile {
    /// HDR10 – maximum peak luminance 10,000 nits.
    Hdr10,
    /// HLG (Hybrid Log-Gamma) – maximum peak luminance 1,000 nits.
    Hlg,
    /// Dolby Vision – maximum peak luminance 10,000 nits.
    DolbyVision,
    /// HDR10+ – maximum peak luminance 10,000 nits.
    Hdr10Plus,
}

impl HdrProfile {
    /// Maximum allowed peak luminance in nits for this profile.
    #[must_use]
    pub fn max_luminance_nits(self) -> f32 {
        match self {
            Self::Hdr10 | Self::DolbyVision | Self::Hdr10Plus => 10_000.0,
            Self::Hlg => 1_000.0,
        }
    }
}

/// HDR quality control helper (standalone functions).
pub struct HdrQc;

impl HdrQc {
    /// Check whether a peak luminance value exceeds the profile limit.
    ///
    /// Returns `Some(QcIssue)` if `nits` exceeds the profile maximum, otherwise `None`.
    #[must_use]
    pub fn check_peak_luminance(nits: f32, profile: HdrProfile) -> Option<QcIssue> {
        let max = profile.max_luminance_nits();
        if nits > max {
            Some(QcIssue::error(
                "HDR-PEAK-LUMA",
                format!(
                    "Peak luminance {nits:.1} nits exceeds {profile:?} maximum of {max:.0} nits"
                ),
            ))
        } else {
            None
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Temporal QC – frozen frame detection
// ─────────────────────────────────────────────────────────────────────────────

/// A perceptual hash of a single video frame (32-byte opaque blob).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHash(pub [u8; 32]);

impl FrameHash {
    /// Construct from raw bytes.
    #[must_use]
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Returns true when two hashes are identical (frozen frame criterion).
    #[must_use]
    pub fn matches(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

/// A contiguous range of video frames that appear frozen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrozenSegment {
    /// Index of the first frozen frame (inclusive).
    pub start_frame: u32,
    /// Index of the last frozen frame (inclusive).
    pub end_frame: u32,
    /// Duration in frames.
    pub duration_frames: u32,
}

impl FrozenSegment {
    /// Construct a frozen segment.
    #[must_use]
    pub fn new(start_frame: u32, end_frame: u32) -> Self {
        let duration_frames = end_frame.saturating_sub(start_frame) + 1;
        Self {
            start_frame,
            end_frame,
            duration_frames,
        }
    }
}

/// Temporal QC checker for freeze-frame, PTS, and A/V sync analysis.
pub struct TemporalQcChecker;

impl TemporalQcChecker {
    /// Detect frozen frame segments in a sequence of frame hashes.
    ///
    /// A freeze is detected when `min_duration_frames` or more consecutive
    /// frames share the same hash value.  Returns the list of frozen segments.
    #[must_use]
    pub fn detect_frozen_frames(
        frames: &[FrameHash],
        min_duration_frames: u32,
    ) -> Vec<FrozenSegment> {
        if frames.is_empty() || min_duration_frames == 0 {
            return Vec::new();
        }

        let mut segments = Vec::new();
        let mut run_start: usize = 0;
        let mut run_len: u32 = 1;

        for i in 1..frames.len() {
            if frames[i].matches(&frames[run_start]) {
                run_len += 1;
            } else {
                if run_len >= min_duration_frames {
                    segments.push(FrozenSegment::new(
                        run_start as u32,
                        (run_start + run_len as usize - 1) as u32,
                    ));
                }
                run_start = i;
                run_len = 1;
            }
        }

        // Flush final run
        if run_len >= min_duration_frames {
            segments.push(FrozenSegment::new(
                run_start as u32,
                (run_start + run_len as usize - 1) as u32,
            ));
        }

        segments
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Closed caption QC – timing overlap / backwards-timing detection
// ─────────────────────────────────────────────────────────────────────────────

/// A single closed caption cue with timing in seconds.
#[derive(Debug, Clone)]
pub struct CaptionCue {
    /// Presentation start time in seconds.
    pub start_secs: f64,
    /// Presentation end time in seconds.
    pub end_secs: f64,
    /// Caption text content.
    pub text: String,
}

impl CaptionCue {
    /// Create a new cue.
    #[must_use]
    pub fn new(start_secs: f64, end_secs: f64, text: impl Into<String>) -> Self {
        Self {
            start_secs,
            end_secs,
            text: text.into(),
        }
    }
}

/// Closed caption QC checker for timing issues.
pub struct ClosedCaptionQcChecker;

impl ClosedCaptionQcChecker {
    /// Check a list of caption cues for timing issues.
    ///
    /// Detects:
    /// - Backwards timings (`start >= end`).
    /// - Overlaps between consecutive cues.
    #[must_use]
    pub fn check_timing(cues: &[CaptionCue]) -> Vec<QcIssue> {
        let mut issues = Vec::new();

        for (i, cue) in cues.iter().enumerate() {
            // Backwards / zero-duration cue
            if cue.start_secs >= cue.end_secs {
                issues.push(
                    QcIssue::error(
                        "CC-TIMING-BACKWARDS",
                        format!(
                            "Cue {} has start ({:.3}s) >= end ({:.3}s)",
                            i, cue.start_secs, cue.end_secs
                        ),
                    )
                    .with_timecode(format!("{:.3}s", cue.start_secs)),
                );
            }

            // Overlap with next cue
            if let Some(next) = cues.get(i + 1) {
                if cue.end_secs > next.start_secs {
                    issues.push(
                        QcIssue::warning(
                            "CC-TIMING-OVERLAP",
                            format!(
                                "Cue {} (ends {:.3}s) overlaps cue {} (starts {:.3}s)",
                                i,
                                cue.end_secs,
                                i + 1,
                                next.start_secs
                            ),
                        )
                        .with_timecode(format!("{:.3}s", next.start_secs)),
                    );
                }
            }
        }

        issues
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Bitrate QC – CBR compliance checker
// ─────────────────────────────────────────────────────────────────────────────

/// Bitrate QC checker (simplified standalone variant for CBR enforcement).
pub struct BitrateQcChecker;

impl BitrateQcChecker {
    /// Check whether a set of per-segment bitrates comply with a CBR target.
    ///
    /// Each segment's bitrate must fall within `target ± tolerance_pct%`.
    /// Returns one `QcIssue` per out-of-tolerance segment.
    #[must_use]
    pub fn check_cbr_compliance(bitrates: &[u32], target: u32, tolerance_pct: f32) -> Vec<QcIssue> {
        if bitrates.is_empty() {
            return Vec::new();
        }

        let tol = target as f32 * tolerance_pct / 100.0;
        let lower = target as f32 - tol;
        let upper = target as f32 + tol;

        bitrates
            .iter()
            .enumerate()
            .filter_map(|(i, &br)| {
                let br_f = br as f32;
                if br_f < lower || br_f > upper {
                    Some(
                        QcIssue::warning(
                            "BR-CBR",
                            format!(
                                "Segment {i}: bitrate {br} kbps outside CBR window [{lower:.0}, {upper:.0}] kbps"
                            ),
                        )
                        .with_timecode(format!("segment-{i}")),
                    )
                } else {
                    None
                }
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Color gamut QC – SDR out-of-range pixel ratio
// ─────────────────────────────────────────────────────────────────────────────

/// Color gamut quality control for SDR content.
pub struct ColorGamutQc;

impl ColorGamutQc {
    /// Compute the fraction of pixels in an 8-bit RGB frame that have any
    /// channel value above 235 (the SDR legal white level).
    ///
    /// `frame` must be a packed R8G8B8 (3 bytes per pixel) buffer of
    /// exactly `w * h * 3` bytes.  Returns a ratio in [0.0, 1.0].
    #[must_use]
    pub fn check_sdr_out_of_range(frame: &[u8], w: u32, h: u32) -> f32 {
        let expected = (w as usize) * (h as usize) * 3;
        if frame.len() < expected || expected == 0 {
            return 0.0;
        }

        let total_pixels = (w as usize) * (h as usize);
        let out_of_range = frame[..expected]
            .chunks_exact(3)
            .filter(|px| px[0] > 235 || px[1] > 235 || px[2] > 235)
            .count();

        out_of_range as f32 / total_pixels as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Format QC – container/codec compatibility check
// ─────────────────────────────────────────────────────────────────────────────

/// Format QC checker for container/codec compatibility rules.
pub struct FormatQcChecker;

impl FormatQcChecker {
    /// Verify that a container and codec combination is compatible.
    ///
    /// Rules:
    /// - `webm` → video codec must be one of: vp8, vp9, av1.
    /// - `mkv` → any codec is allowed.
    /// - `mp4` → video codec must be one of: h264, hevc, aac (when aac is used as
    ///   video codec field, which is unusual but we match the spec exactly).
    ///
    /// Returns `Ok(())` on success or `Err(description)` on incompatibility.
    pub fn check_container_codec_compatibility(container: &str, codec: &str) -> Result<(), String> {
        let container_lc = container.to_lowercase();
        let codec_lc = codec.to_lowercase();

        match container_lc.as_str() {
            "webm" => {
                let allowed = ["vp8", "vp9", "av1"];
                if allowed.contains(&codec_lc.as_str()) {
                    Ok(())
                } else {
                    Err(format!(
                        "Container 'webm' does not support codec '{codec}'. \
                         Allowed: vp8, vp9, av1"
                    ))
                }
            }
            "mkv" | "matroska" => {
                // MKV supports virtually any codec
                Ok(())
            }
            "mp4" | "m4v" => {
                let allowed = ["h264", "hevc", "aac", "h265", "avc"];
                if allowed.contains(&codec_lc.as_str()) {
                    Ok(())
                } else {
                    Err(format!(
                        "Container 'mp4' does not support codec '{codec}'. \
                         Allowed: h264, hevc, aac"
                    ))
                }
            }
            _ => {
                // Unknown container: allow anything but emit a warning via Ok(())
                Ok(())
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Compliance report – CSV export
// ─────────────────────────────────────────────────────────────────────────────

/// Compliance report with CSV export capability.
pub struct ComplianceReport;

impl ComplianceReport {
    /// Serialize a list of `QcIssue` values to CSV.
    ///
    /// Columns: `severity,code,timecode,description`
    ///
    /// The header row is always included.  Fields containing commas or double-
    /// quotes are properly quoted per RFC 4180.
    #[must_use]
    pub fn to_csv(issues: &[QcIssue]) -> String {
        let mut out = String::from("severity,code,timecode,description\n");
        for issue in issues {
            let tc = issue.timecode.as_deref().unwrap_or("");
            out.push_str(&format!(
                "{},{},{},{}\n",
                csv_field(issue.severity.label()),
                csv_field(&issue.code),
                csv_field(tc),
                csv_field(&issue.description),
            ));
        }
        out
    }
}

/// Quote a CSV field if it contains commas, double-quotes, or newlines.
fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. QC scheduling – FIFO queue
// ─────────────────────────────────────────────────────────────────────────────

/// A QC check preset identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QcPreset {
    /// Fast basic checks only.
    Basic,
    /// Streaming-platform checks.
    Streaming,
    /// Full broadcast compliance suite.
    Broadcast,
    /// All available checks.
    Comprehensive,
}

/// A pending QC job in the scheduler queue.
#[derive(Debug, Clone)]
pub struct QcSchedulerJob {
    /// Media asset identifier.
    pub media_id: u64,
    /// Preset to apply for this job.
    pub preset: QcPreset,
}

/// Simple FIFO QC scheduler.
#[derive(Debug, Default)]
pub struct QcScheduler {
    queue: VecDeque<QcSchedulerJob>,
}

impl QcScheduler {
    /// Create a new empty scheduler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }

    /// Add a job to the back of the FIFO queue.
    pub fn add_job(&mut self, media_id: u64, preset: QcPreset) {
        self.queue.push_back(QcSchedulerJob { media_id, preset });
    }

    /// Take the next job from the front of the queue (FIFO).
    ///
    /// Returns `None` when the queue is empty.
    pub fn next_job(&mut self) -> Option<(u64, QcPreset)> {
        self.queue.pop_front().map(|j| (j.media_id, j.preset))
    }

    /// Returns the number of pending jobs.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Returns true if no jobs are queued.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. QC database – in-memory persistence
// ─────────────────────────────────────────────────────────────────────────────

/// Status of a QC report result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QcStatus {
    /// All checks passed.
    Passed,
    /// At least one warning was found.
    Warning,
    /// At least one error was found.
    Failed,
}

/// A minimal QC report record for storage purposes.
#[derive(Debug, Clone)]
pub struct QcReport {
    /// Unique report ID assigned by the database.
    pub id: u64,
    /// Status of the validation run.
    pub status: QcStatus,
    /// Issues found during validation.
    pub issues: Vec<QcIssue>,
}

impl QcReport {
    /// Build a report from a list of issues.
    #[must_use]
    pub fn from_issues(issues: Vec<QcIssue>) -> Self {
        let status = if issues.iter().any(|i| i.severity == QcIssueSeverity::Error) {
            QcStatus::Failed
        } else if issues
            .iter()
            .any(|i| i.severity == QcIssueSeverity::Warning)
        {
            QcStatus::Warning
        } else {
            QcStatus::Passed
        };
        Self {
            id: 0,
            status,
            issues,
        }
    }
}

/// In-memory QC result database.
#[derive(Debug, Default)]
pub struct QcDatabase {
    next_id: u64,
    reports: HashMap<u64, QcReport>,
    status_index: HashMap<QcStatus, Vec<u64>>,
}

impl QcDatabase {
    /// Create a new empty database.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: 1,
            reports: HashMap::new(),
            status_index: HashMap::new(),
        }
    }

    /// Store a QC report and return its assigned ID.
    pub fn save_result(&mut self, result: &QcReport) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let mut stored = result.clone();
        stored.id = id;

        self.status_index.entry(stored.status).or_default().push(id);

        self.reports.insert(id, stored);
        id
    }

    /// Query all report IDs with the given status.
    #[must_use]
    pub fn query_by_status(&self, status: QcStatus) -> Vec<u64> {
        self.status_index.get(&status).cloned().unwrap_or_default()
    }

    /// Retrieve a stored report by ID.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&QcReport> {
        self.reports.get(&id)
    }

    /// Total number of stored reports.
    #[must_use]
    pub fn len(&self) -> usize {
        self.reports.len()
    }

    /// Returns true when no reports have been stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.reports.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- DolbyVisionQc ---

    #[test]
    fn test_dv_l1_valid_metadata() {
        let l1 = DvL1Metadata::new(100, 2000, 4000);
        let issues = DolbyVisionQc::check_l1_metadata(&l1);
        assert!(issues.is_empty(), "Expected no issues, got {issues:?}");
    }

    #[test]
    fn test_dv_l1_out_of_range() {
        let l1 = DvL1Metadata::new(100, 2000, 5000); // max_pq > 4095
        let issues = DolbyVisionQc::check_l1_metadata(&l1);
        assert!(
            issues.iter().any(|i| i.code == "DV-L1-RANGE"),
            "Should report range error"
        );
    }

    #[test]
    fn test_dv_l1_bad_ordering_min_ge_mid() {
        let l1 = DvL1Metadata::new(2000, 1000, 4000); // min > mid
        let issues = DolbyVisionQc::check_l1_metadata(&l1);
        assert!(
            issues.iter().any(|i| i.code == "DV-L1-ORDER"),
            "Should report ordering error"
        );
    }

    #[test]
    fn test_dv_l1_bad_ordering_mid_ge_max() {
        let l1 = DvL1Metadata::new(100, 4000, 3000); // mid > max
        let issues = DolbyVisionQc::check_l1_metadata(&l1);
        assert!(
            issues.iter().any(|i| i.code == "DV-L1-ORDER"),
            "Should report ordering error"
        );
    }

    // --- HdrQc ---

    #[test]
    fn test_hdr_peak_luminance_hdr10_ok() {
        assert!(HdrQc::check_peak_luminance(5000.0, HdrProfile::Hdr10).is_none());
    }

    #[test]
    fn test_hdr_peak_luminance_hdr10_exceeded() {
        let issue = HdrQc::check_peak_luminance(12_000.0, HdrProfile::Hdr10);
        assert!(issue.is_some());
        assert_eq!(
            issue.as_ref().map(|i| i.code.as_str()),
            Some("HDR-PEAK-LUMA")
        );
    }

    #[test]
    fn test_hdr_peak_luminance_hlg_limit() {
        assert!(HdrQc::check_peak_luminance(1000.0, HdrProfile::Hlg).is_none());
        assert!(HdrQc::check_peak_luminance(1001.0, HdrProfile::Hlg).is_some());
    }

    // --- TemporalQcChecker – frozen frames ---

    #[test]
    fn test_detect_frozen_frames_finds_5_frame_freeze() {
        let frozen_hash = FrameHash::new([0xAA; 32]);
        let different = FrameHash::new([0xBB; 32]);

        let frames = vec![
            different.clone(),
            frozen_hash.clone(),
            frozen_hash.clone(),
            frozen_hash.clone(),
            frozen_hash.clone(),
            frozen_hash.clone(),
            different.clone(),
        ];

        let segments = TemporalQcChecker::detect_frozen_frames(&frames, 3);
        assert_eq!(segments.len(), 1, "Should find exactly one frozen segment");
        assert_eq!(segments[0].start_frame, 1);
        assert_eq!(segments[0].end_frame, 5);
        assert_eq!(segments[0].duration_frames, 5);
    }

    #[test]
    fn test_detect_frozen_frames_no_freeze_below_min() {
        let h1 = FrameHash::new([0x01; 32]);
        let h2 = FrameHash::new([0x02; 32]);
        let frames = vec![h1.clone(), h1.clone(), h2.clone()];
        let segments = TemporalQcChecker::detect_frozen_frames(&frames, 3);
        assert!(segments.is_empty());
    }

    #[test]
    fn test_detect_frozen_frames_empty_input() {
        let segments = TemporalQcChecker::detect_frozen_frames(&[], 2);
        assert!(segments.is_empty());
    }

    // --- ClosedCaptionQcChecker ---

    #[test]
    fn test_caption_timing_valid() {
        let cues = vec![
            CaptionCue::new(0.0, 2.0, "First."),
            CaptionCue::new(3.0, 5.0, "Second."),
        ];
        let issues = ClosedCaptionQcChecker::check_timing(&cues);
        assert!(issues.is_empty(), "Expected no timing issues");
    }

    #[test]
    fn test_caption_timing_overlap() {
        let cues = vec![
            CaptionCue::new(0.0, 3.0, "First."),
            CaptionCue::new(2.0, 5.0, "Second."), // starts during first
        ];
        let issues = ClosedCaptionQcChecker::check_timing(&cues);
        assert!(
            issues.iter().any(|i| i.code == "CC-TIMING-OVERLAP"),
            "Should detect overlap"
        );
    }

    #[test]
    fn test_caption_timing_backwards() {
        let cues = vec![CaptionCue::new(5.0, 2.0, "Bad cue")]; // end < start
        let issues = ClosedCaptionQcChecker::check_timing(&cues);
        assert!(
            issues.iter().any(|i| i.code == "CC-TIMING-BACKWARDS"),
            "Should detect backwards timing"
        );
    }

    // --- BitrateQcChecker ---

    #[test]
    fn test_cbr_compliance_all_ok() {
        let bitrates = vec![1000u32, 1020, 980, 1010, 990];
        let issues = BitrateQcChecker::check_cbr_compliance(&bitrates, 1000, 5.0);
        assert!(issues.is_empty(), "All within 5% tolerance");
    }

    #[test]
    fn test_cbr_compliance_violation() {
        let bitrates = vec![1000u32, 1500, 1000]; // 1500 is 50% over target 1000
        let issues = BitrateQcChecker::check_cbr_compliance(&bitrates, 1000, 10.0);
        assert_eq!(issues.len(), 1, "Should flag one out-of-tolerance segment");
        assert_eq!(issues[0].code, "BR-CBR");
    }

    // --- ColorGamutQc ---

    #[test]
    fn test_sdr_out_of_range_zero_for_valid_frame() {
        // All pixels have values <= 235
        let frame = vec![100u8, 200u8, 235u8, 50, 50, 50];
        let ratio = ColorGamutQc::check_sdr_out_of_range(&frame, 2, 1);
        assert_eq!(ratio, 0.0, "No pixels should be out of range");
    }

    #[test]
    fn test_sdr_out_of_range_all_pixels_over_limit() {
        // All pixels have at least one channel > 235
        let frame = vec![255u8, 255u8, 255u8, 240, 240, 240];
        let ratio = ColorGamutQc::check_sdr_out_of_range(&frame, 2, 1);
        assert!(
            (ratio - 1.0).abs() < f32::EPSILON,
            "All pixels out of range"
        );
    }

    #[test]
    fn test_sdr_out_of_range_half_pixels() {
        // First pixel in range, second out of range
        let frame = vec![100u8, 100u8, 100u8, 250u8, 100u8, 100u8];
        let ratio = ColorGamutQc::check_sdr_out_of_range(&frame, 2, 1);
        assert!(
            (ratio - 0.5).abs() < f32::EPSILON,
            "Half of pixels out of range"
        );
    }

    // --- FormatQcChecker ---

    #[test]
    fn test_webm_vp9_ok() {
        assert!(FormatQcChecker::check_container_codec_compatibility("webm", "vp9").is_ok());
    }

    #[test]
    fn test_mp4_vp9_incompatible() {
        let result = FormatQcChecker::check_container_codec_compatibility("mp4", "vp9");
        assert!(result.is_err(), "mp4 does not support vp9");
        assert!(result.err().unwrap().contains("mp4"));
    }

    #[test]
    fn test_mkv_allows_any_codec() {
        assert!(FormatQcChecker::check_container_codec_compatibility("mkv", "vp9").is_ok());
        assert!(FormatQcChecker::check_container_codec_compatibility("mkv", "h264").is_ok());
        assert!(FormatQcChecker::check_container_codec_compatibility("mkv", "av1").is_ok());
    }

    #[test]
    fn test_mp4_h264_ok() {
        assert!(FormatQcChecker::check_container_codec_compatibility("mp4", "h264").is_ok());
    }

    #[test]
    fn test_webm_h264_incompatible() {
        let result = FormatQcChecker::check_container_codec_compatibility("webm", "h264");
        assert!(result.is_err());
    }

    // --- ComplianceReport::to_csv ---

    #[test]
    fn test_to_csv_empty() {
        let csv = ComplianceReport::to_csv(&[]);
        assert!(csv.starts_with("severity,code,timecode,description\n"));
        assert_eq!(csv.lines().count(), 1, "Header only");
    }

    #[test]
    fn test_to_csv_with_issues() {
        let issues = vec![
            QcIssue::error("VID-001", "Luma clipping").with_timecode("00:01:23:00"),
            QcIssue::warning("AUD-002", "High level"),
        ];
        let csv = ComplianceReport::to_csv(&issues);
        assert!(csv.contains("error,VID-001,00:01:23:00,Luma clipping"));
        assert!(csv.contains("warning,AUD-002,,High level"));
    }

    #[test]
    fn test_to_csv_escapes_commas() {
        let issues = vec![QcIssue::error("X", "Comma, here")];
        let csv = ComplianceReport::to_csv(&issues);
        assert!(csv.contains("\"Comma, here\""));
    }

    // --- QcScheduler ---

    #[test]
    fn test_scheduler_fifo_order() {
        let mut sched = QcScheduler::new();
        sched.add_job(1, QcPreset::Basic);
        sched.add_job(2, QcPreset::Streaming);
        sched.add_job(3, QcPreset::Broadcast);

        assert_eq!(sched.pending_count(), 3);

        let (id1, preset1) = sched.next_job().expect("should have a job");
        assert_eq!(id1, 1);
        assert_eq!(preset1, QcPreset::Basic);

        let (id2, _) = sched.next_job().expect("should have a job");
        assert_eq!(id2, 2);

        assert_eq!(sched.pending_count(), 1);
    }

    #[test]
    fn test_scheduler_empty_returns_none() {
        let mut sched = QcScheduler::new();
        assert!(sched.next_job().is_none());
    }

    // --- QcDatabase ---

    #[test]
    fn test_database_save_and_query() {
        let mut db = QcDatabase::new();
        let report = QcReport::from_issues(vec![QcIssue::error("E1", "err")]);

        let id = db.save_result(&report);
        assert!(id > 0);

        let ids = db.query_by_status(QcStatus::Failed);
        assert!(ids.contains(&id));
    }

    #[test]
    fn test_database_query_by_status_passed() {
        let mut db = QcDatabase::new();
        let report = QcReport::from_issues(vec![]);

        let id = db.save_result(&report);
        let ids = db.query_by_status(QcStatus::Passed);
        assert!(ids.contains(&id));
    }

    #[test]
    fn test_database_len_increments() {
        let mut db = QcDatabase::new();
        assert_eq!(db.len(), 0);
        db.save_result(&QcReport::from_issues(vec![]));
        assert_eq!(db.len(), 1);
        db.save_result(&QcReport::from_issues(vec![]));
        assert_eq!(db.len(), 2);
    }
}
