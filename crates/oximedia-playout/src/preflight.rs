//! Content pre-flight validation before broadcast playout.
//!
//! Validates media files against a configurable set of broadcast standards
//! before they enter the playout queue: file existence, duration, codec
//! compliance, bitrate floors, and audio sample-rate requirements.

use crate::content::{AudioMetadata, ContentItem, VideoMetadata};
use std::path::Path;
use std::time::SystemTime;

// ─── Check specification ──────────────────────────────────────────────────────

/// Rules that every content item must satisfy before being cleared for playout.
#[derive(Debug, Clone)]
pub struct PreflightCheck {
    /// Minimum allowed clip duration in seconds.
    pub min_duration_secs: f64,
    /// Maximum allowed clip duration in seconds.
    pub max_duration_secs: f64,
    /// Whether a video stream is required.
    pub required_video: bool,
    /// Whether an audio stream is required.
    pub required_audio: bool,
    /// Whitelisted video codecs (empty = any allowed).
    pub allowed_video_codecs: Vec<String>,
    /// Whitelisted audio codecs (empty = any allowed).
    pub allowed_audio_codecs: Vec<String>,
    /// Minimum video bitrate in kbps (0 = no minimum).
    pub min_video_bitrate_kbps: u32,
    /// Minimum audio sample rate in Hz (0 = no minimum).
    pub min_audio_sample_rate: u32,
    /// Whether to verify the file exists on disk.
    pub check_file_exists: bool,
}

impl PreflightCheck {
    /// Typical broadcast channel pre-flight settings.
    pub fn default_broadcast_check() -> Self {
        Self {
            min_duration_secs: 1.0,
            max_duration_secs: 14_400.0, // 4 hours
            required_video: true,
            required_audio: true,
            allowed_video_codecs: vec!["av1".to_string(), "vp9".to_string(), "ffv1".to_string()],
            allowed_audio_codecs: vec!["opus".to_string(), "flac".to_string(), "pcm".to_string()],
            min_video_bitrate_kbps: 1_000,
            min_audio_sample_rate: 48_000,
            check_file_exists: true,
        }
    }

    /// Permissive check (useful for unit tests / development).
    pub fn permissive() -> Self {
        Self {
            min_duration_secs: 0.0,
            max_duration_secs: f64::MAX,
            required_video: false,
            required_audio: false,
            allowed_video_codecs: Vec::new(),
            allowed_audio_codecs: Vec::new(),
            min_video_bitrate_kbps: 0,
            min_audio_sample_rate: 0,
            check_file_exists: false,
        }
    }
}

impl Default for PreflightCheck {
    fn default() -> Self {
        Self::default_broadcast_check()
    }
}

// ─── Issue taxonomy ───────────────────────────────────────────────────────────

/// A single validation problem found during pre-flight.
#[derive(Debug, Clone, PartialEq)]
pub enum PreflightIssue {
    /// The content file is not accessible on disk.
    FileMissing { path: String },
    /// Clip duration falls outside the configured window.
    DurationOutOfRange { actual: f64, min: f64, max: f64 },
    /// No video stream detected / declared.
    MissingVideoStream,
    /// No audio stream detected / declared.
    MissingAudioStream,
    /// Video codec is not in the allowed list.
    UnsupportedVideoCodec { codec: String, allowed: Vec<String> },
    /// Audio codec is not in the allowed list.
    UnsupportedAudioCodec { codec: String, allowed: Vec<String> },
    /// Video bitrate is below the minimum threshold.
    BitrateTooLow { actual_kbps: u32, min_kbps: u32 },
    /// Audio sample rate is below the minimum threshold.
    AudioSampleRateTooLow { actual: u32, min: u32 },
    /// File appears to be corrupt or unreadable.
    CorruptFile { reason: String },
}

impl std::fmt::Display for PreflightIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileMissing { path } => write!(f, "File missing: {path}"),
            Self::DurationOutOfRange { actual, min, max } => {
                write!(f, "Duration {actual:.2}s out of range [{min:.2}, {max:.2}]")
            }
            Self::MissingVideoStream => write!(f, "No video stream"),
            Self::MissingAudioStream => write!(f, "No audio stream"),
            Self::UnsupportedVideoCodec { codec, allowed } => {
                write!(f, "Video codec '{codec}' not in allowed list {allowed:?}")
            }
            Self::UnsupportedAudioCodec { codec, allowed } => {
                write!(f, "Audio codec '{codec}' not in allowed list {allowed:?}")
            }
            Self::BitrateTooLow {
                actual_kbps,
                min_kbps,
            } => {
                write!(
                    f,
                    "Video bitrate {actual_kbps} kbps < minimum {min_kbps} kbps"
                )
            }
            Self::AudioSampleRateTooLow { actual, min } => {
                write!(f, "Audio sample rate {actual} Hz < minimum {min} Hz")
            }
            Self::CorruptFile { reason } => write!(f, "Corrupt file: {reason}"),
        }
    }
}

// ─── Metadata helper ─────────────────────────────────────────────────────────

/// Lightweight metadata view used by the checker.  Either extracted from a
/// `ContentItem` or supplied directly.
#[derive(Debug, Clone, Default)]
pub struct ContentMetadata {
    /// Duration in seconds.
    pub duration_secs: Option<f64>,
    /// Whether a video stream is present.
    pub has_video: bool,
    /// Whether an audio stream is present.
    pub has_audio: bool,
    /// Video codec name (lower-case).
    pub video_codec: Option<String>,
    /// Audio codec name (lower-case).
    pub audio_codec: Option<String>,
    /// Video bitrate in kbps.
    pub video_bitrate_kbps: Option<u32>,
    /// Audio sample rate in Hz.
    pub audio_sample_rate: Option<u32>,
}

impl ContentMetadata {
    /// Build from a `ContentItem`.
    pub fn from_content_item(item: &ContentItem) -> Self {
        let duration_secs = if item.duration_ms > 0 {
            Some(item.duration_ms as f64 / 1_000.0)
        } else {
            None
        };

        let has_video = item.video_metadata.is_some();
        let has_audio = item.audio_metadata.is_some();

        let video_codec = item
            .video_metadata
            .as_ref()
            .map(|v: &VideoMetadata| v.codec.to_lowercase());
        let video_bitrate_kbps = item.video_metadata.as_ref().map(|v| v.bitrate_kbps);

        let audio_codec = item
            .audio_metadata
            .as_ref()
            .map(|a: &AudioMetadata| a.codec.to_lowercase());
        let audio_sample_rate = item.audio_metadata.as_ref().map(|a| a.sample_rate);

        Self {
            duration_secs,
            has_video,
            has_audio,
            video_codec,
            audio_codec,
            video_bitrate_kbps,
            audio_sample_rate,
        }
    }
}

// ─── Result ──────────────────────────────────────────────────────────────────

/// Outcome of a single content pre-flight check.
#[derive(Debug, Clone)]
pub struct PreflightResult {
    /// Caller-provided content identifier.
    pub content_id: String,
    /// `true` if no issues were found.
    pub passed: bool,
    /// All detected issues (empty iff `passed == true`).
    pub issues: Vec<PreflightIssue>,
    /// Wall-clock time at which the check was performed.
    pub checked_at: SystemTime,
}

impl PreflightResult {
    fn new(content_id: impl Into<String>) -> Self {
        Self {
            content_id: content_id.into(),
            passed: true,
            issues: Vec::new(),
            checked_at: SystemTime::now(),
        }
    }

    fn add_issue(&mut self, issue: PreflightIssue) {
        self.passed = false;
        self.issues.push(issue);
    }
}

// ─── Checker ─────────────────────────────────────────────────────────────────

/// Validates content items against a `PreflightCheck` specification.
pub struct PreflightChecker {
    check: PreflightCheck,
}

impl PreflightChecker {
    /// Create a new checker with the given check specification.
    pub fn new(check: PreflightCheck) -> Self {
        Self { check }
    }

    /// Create a checker using typical broadcast defaults.
    pub fn broadcast() -> Self {
        Self::new(PreflightCheck::default_broadcast_check())
    }

    /// Validate a single content item at `path`.
    ///
    /// `metadata` is optional: if provided it is used directly; otherwise the
    /// checker will attempt to infer properties from the file extension.
    pub fn check_content(
        &self,
        id: &str,
        path: &Path,
        metadata: Option<&ContentMetadata>,
    ) -> PreflightResult {
        let mut result = PreflightResult::new(id);

        // ── 1. File existence ──────────────────────────────────────────
        if self.check.check_file_exists && !path.exists() {
            result.add_issue(PreflightIssue::FileMissing {
                path: path.to_string_lossy().into_owned(),
            });
            // No point continuing without a file
            return result;
        }

        // ── 2. Resolve metadata ────────────────────────────────────────
        let inferred: ContentMetadata;
        let meta: &ContentMetadata = if let Some(m) = metadata {
            m
        } else {
            inferred = infer_metadata_from_extension(path);
            &inferred
        };

        // ── 3. Duration ────────────────────────────────────────────────
        if let Some(dur) = meta.duration_secs {
            if dur < self.check.min_duration_secs || dur > self.check.max_duration_secs {
                result.add_issue(PreflightIssue::DurationOutOfRange {
                    actual: dur,
                    min: self.check.min_duration_secs,
                    max: self.check.max_duration_secs,
                });
            }
        }

        // ── 4. Stream presence ─────────────────────────────────────────
        if self.check.required_video && !meta.has_video {
            result.add_issue(PreflightIssue::MissingVideoStream);
        }

        if self.check.required_audio && !meta.has_audio {
            result.add_issue(PreflightIssue::MissingAudioStream);
        }

        // ── 5. Codec whitelisting ──────────────────────────────────────
        if !self.check.allowed_video_codecs.is_empty() {
            if let Some(codec) = &meta.video_codec {
                if !self.check.allowed_video_codecs.contains(codec) {
                    result.add_issue(PreflightIssue::UnsupportedVideoCodec {
                        codec: codec.clone(),
                        allowed: self.check.allowed_video_codecs.clone(),
                    });
                }
            }
        }

        if !self.check.allowed_audio_codecs.is_empty() {
            if let Some(codec) = &meta.audio_codec {
                if !self.check.allowed_audio_codecs.contains(codec) {
                    result.add_issue(PreflightIssue::UnsupportedAudioCodec {
                        codec: codec.clone(),
                        allowed: self.check.allowed_audio_codecs.clone(),
                    });
                }
            }
        }

        // ── 6. Bitrate floor ───────────────────────────────────────────
        if self.check.min_video_bitrate_kbps > 0 {
            if let Some(kbps) = meta.video_bitrate_kbps {
                if kbps < self.check.min_video_bitrate_kbps {
                    result.add_issue(PreflightIssue::BitrateTooLow {
                        actual_kbps: kbps,
                        min_kbps: self.check.min_video_bitrate_kbps,
                    });
                }
            }
        }

        // ── 7. Audio sample rate ───────────────────────────────────────
        if self.check.min_audio_sample_rate > 0 {
            if let Some(rate) = meta.audio_sample_rate {
                if rate < self.check.min_audio_sample_rate {
                    result.add_issue(PreflightIssue::AudioSampleRateTooLow {
                        actual: rate,
                        min: self.check.min_audio_sample_rate,
                    });
                }
            }
        }

        result
    }

    /// Bulk-check an entire playlist of content items.
    pub fn check_playlist(&self, playlist: &[ContentItem]) -> Vec<PreflightResult> {
        playlist
            .iter()
            .map(|item| {
                let meta = ContentMetadata::from_content_item(item);
                self.check_content(&item.id.to_string(), &item.file_path, Some(&meta))
            })
            .collect()
    }

    /// Return all issues across all results.
    pub fn summarise(results: &[PreflightResult]) -> Vec<String> {
        results
            .iter()
            .filter(|r| !r.passed)
            .flat_map(|r| r.issues.iter().map(|i| format!("[{}] {i}", r.content_id)))
            .collect()
    }
}

// ─── Extension-based metadata inference ──────────────────────────────────────

/// Guess codec/stream information from the file extension.
///
/// This is a lightweight heuristic for use in tests or when no external
/// demuxer is available.
fn infer_metadata_from_extension(path: &Path) -> ContentMetadata {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "av1" => ContentMetadata {
            has_video: true,
            video_codec: Some("av1".to_string()),
            ..Default::default()
        },
        "vp9" => ContentMetadata {
            has_video: true,
            video_codec: Some("vp9".to_string()),
            ..Default::default()
        },
        "ffv1" => ContentMetadata {
            has_video: true,
            video_codec: Some("ffv1".to_string()),
            ..Default::default()
        },
        "opus" => ContentMetadata {
            has_audio: true,
            audio_codec: Some("opus".to_string()),
            ..Default::default()
        },
        "flac" => ContentMetadata {
            has_audio: true,
            audio_codec: Some("flac".to_string()),
            ..Default::default()
        },
        "pcm" => ContentMetadata {
            has_audio: true,
            audio_codec: Some("pcm".to_string()),
            ..Default::default()
        },
        "mxf" | "mp4" | "mov" => ContentMetadata {
            has_video: true,
            has_audio: true,
            video_codec: Some("av1".to_string()),
            audio_codec: Some("opus".to_string()),
            ..Default::default()
        },
        _ => ContentMetadata::default(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_meta(
        duration: Option<f64>,
        has_video: bool,
        has_audio: bool,
        vcod: Option<&str>,
        acod: Option<&str>,
        vbr: Option<u32>,
        asr: Option<u32>,
    ) -> ContentMetadata {
        ContentMetadata {
            duration_secs: duration,
            has_video,
            has_audio,
            video_codec: vcod.map(|s| s.to_string()),
            audio_codec: acod.map(|s| s.to_string()),
            video_bitrate_kbps: vbr,
            audio_sample_rate: asr,
        }
    }

    /// Return a broadcast-standards checker with file-existence check disabled
    /// (tests don't create real files on disk).
    fn broadcast_checker() -> PreflightChecker {
        let mut check = PreflightCheck::default_broadcast_check();
        check.check_file_exists = false;
        PreflightChecker::new(check)
    }

    fn tmp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(name)
    }

    // 1. Permissive check on non-existent path passes (check_file_exists=false)
    #[test]
    fn test_permissive_missing_file_ok() {
        let checker = PreflightChecker::new(PreflightCheck::permissive());
        let path = tmp_path("no_such_file_oximedia.av1");
        let r = checker.check_content("c1", &path, None);
        assert!(r.passed, "Permissive should pass: {:?}", r.issues);
    }

    // 2. Broadcast check: valid metadata → pass
    #[test]
    fn test_broadcast_valid_passes() {
        let checker = broadcast_checker();
        let meta = make_meta(
            Some(30.0),
            true,
            true,
            Some("av1"),
            Some("opus"),
            Some(5_000),
            Some(48_000),
        );
        let r = checker.check_content("ok", &tmp_path("x.av1"), Some(&meta));
        assert!(r.passed, "Issues: {:?}", r.issues);
    }

    // 3. Missing video stream
    #[test]
    fn test_missing_video_stream() {
        let checker = broadcast_checker();
        let meta = make_meta(
            Some(30.0),
            false,
            true,
            None,
            Some("opus"),
            None,
            Some(48_000),
        );
        let r = checker.check_content("no_video", &tmp_path("x.opus"), Some(&meta));
        assert!(!r.passed);
        assert!(r
            .issues
            .iter()
            .any(|i| *i == PreflightIssue::MissingVideoStream));
    }

    // 4. Missing audio stream
    #[test]
    fn test_missing_audio_stream() {
        let checker = broadcast_checker();
        let meta = make_meta(
            Some(30.0),
            true,
            false,
            Some("av1"),
            None,
            Some(5_000),
            None,
        );
        let r = checker.check_content("no_audio", &tmp_path("x.av1"), Some(&meta));
        assert!(!r.passed);
        assert!(r
            .issues
            .iter()
            .any(|i| *i == PreflightIssue::MissingAudioStream));
    }

    // 5. Unsupported video codec
    #[test]
    fn test_unsupported_video_codec() {
        let checker = broadcast_checker();
        let meta = make_meta(
            Some(30.0),
            true,
            true,
            Some("h264"),
            Some("opus"),
            Some(5_000),
            Some(48_000),
        );
        let r = checker.check_content("bad_vcod", &tmp_path("x.mp4"), Some(&meta));
        assert!(!r.passed);
        assert!(r
            .issues
            .iter()
            .any(|i| matches!(i, PreflightIssue::UnsupportedVideoCodec { .. })));
    }

    // 6. Unsupported audio codec
    #[test]
    fn test_unsupported_audio_codec() {
        let checker = broadcast_checker();
        let meta = make_meta(
            Some(30.0),
            true,
            true,
            Some("av1"),
            Some("aac"),
            Some(5_000),
            Some(48_000),
        );
        let r = checker.check_content("bad_acod", &tmp_path("x.av1"), Some(&meta));
        assert!(!r.passed);
        assert!(r
            .issues
            .iter()
            .any(|i| matches!(i, PreflightIssue::UnsupportedAudioCodec { .. })));
    }

    // 7. Duration too short
    #[test]
    fn test_duration_too_short() {
        let checker = broadcast_checker();
        let meta = make_meta(
            Some(0.5),
            true,
            true,
            Some("av1"),
            Some("opus"),
            Some(5_000),
            Some(48_000),
        );
        let r = checker.check_content("short", &tmp_path("x.av1"), Some(&meta));
        assert!(!r.passed);
        assert!(r
            .issues
            .iter()
            .any(|i| matches!(i, PreflightIssue::DurationOutOfRange { .. })));
    }

    // 8. Duration too long
    #[test]
    fn test_duration_too_long() {
        let checker = broadcast_checker();
        let meta = make_meta(
            Some(50_000.0),
            true,
            true,
            Some("av1"),
            Some("opus"),
            Some(5_000),
            Some(48_000),
        );
        let r = checker.check_content("long", &tmp_path("x.av1"), Some(&meta));
        assert!(!r.passed);
        assert!(r
            .issues
            .iter()
            .any(|i| matches!(i, PreflightIssue::DurationOutOfRange { .. })));
    }

    // 9. Bitrate too low
    #[test]
    fn test_bitrate_too_low() {
        let checker = broadcast_checker();
        let meta = make_meta(
            Some(30.0),
            true,
            true,
            Some("av1"),
            Some("opus"),
            Some(100),
            Some(48_000),
        );
        let r = checker.check_content("low_br", &tmp_path("x.av1"), Some(&meta));
        assert!(!r.passed);
        assert!(r
            .issues
            .iter()
            .any(|i| matches!(i, PreflightIssue::BitrateTooLow { .. })));
    }

    // 10. Audio sample rate too low
    #[test]
    fn test_audio_sample_rate_too_low() {
        let checker = broadcast_checker();
        let meta = make_meta(
            Some(30.0),
            true,
            true,
            Some("av1"),
            Some("opus"),
            Some(5_000),
            Some(22_050),
        );
        let r = checker.check_content("low_sr", &tmp_path("x.av1"), Some(&meta));
        assert!(!r.passed);
        assert!(r
            .issues
            .iter()
            .any(|i| matches!(i, PreflightIssue::AudioSampleRateTooLow { .. })));
    }

    // 11. File missing check (uses a checker with check_file_exists=true)
    #[test]
    fn test_file_missing() {
        let mut check = PreflightCheck::default_broadcast_check();
        check.check_file_exists = true;
        let checker = PreflightChecker::new(check);
        let path = tmp_path("definitely_no_such_file_xyz_oximedia.av1");
        // Make sure it really doesn't exist
        let _ = std::fs::remove_file(&path);
        let r = checker.check_content("missing", &path, None);
        assert!(!r.passed);
        assert!(r
            .issues
            .iter()
            .any(|i| matches!(i, PreflightIssue::FileMissing { .. })));
    }

    // 12. Extension inference: .flac → audio only
    #[test]
    fn test_extension_inference_flac() {
        let meta = infer_metadata_from_extension(&tmp_path("track.flac"));
        assert!(meta.has_audio);
        assert!(!meta.has_video);
        assert_eq!(meta.audio_codec.as_deref(), Some("flac"));
    }

    // 13. check_playlist returns one result per item
    #[test]
    fn test_check_playlist_count() {
        use crate::content::{AvailabilityStatus, ContentType, QcStatus};
        use chrono::Utc;
        use uuid::Uuid;

        let checker = PreflightChecker::new(PreflightCheck::permissive());
        let items: Vec<ContentItem> = (0..3)
            .map(|i| ContentItem {
                id: Uuid::new_v4(),
                title: format!("item{i}"),
                file_path: tmp_path(&format!("item{i}.av1")),
                file_size: 0,
                content_type: ContentType::Video,
                duration_ms: 30_000,
                video_metadata: None,
                audio_metadata: None,
                qc_status: QcStatus::NotChecked,
                qc_issues: Vec::new(),
                availability: AvailabilityStatus::Available,
                proxy_path: None,
                thumbnail_paths: Vec::new(),
                metadata: Default::default(),
                created_at: Utc::now(),
                modified_at: Utc::now(),
            })
            .collect();

        let results = checker.check_playlist(&items);
        assert_eq!(results.len(), 3);
    }

    // 14. summarise lists only failing items
    #[test]
    fn test_summarise_failing_only() {
        let checker = broadcast_checker();
        let good_meta = make_meta(
            Some(30.0),
            true,
            true,
            Some("av1"),
            Some("opus"),
            Some(5_000),
            Some(48_000),
        );
        let bad_meta = make_meta(
            Some(30.0),
            false,
            true,
            None,
            Some("opus"),
            None,
            Some(48_000),
        );

        let r_good = checker.check_content("good", &tmp_path("g.av1"), Some(&good_meta));
        let r_bad = checker.check_content("bad", &tmp_path("b.av1"), Some(&bad_meta));

        let summary = PreflightChecker::summarise(&[r_good, r_bad]);
        assert_eq!(summary.len(), 1);
        assert!(summary[0].contains("bad"));
    }

    // 15. PreflightIssue Display is human-readable
    #[test]
    fn test_issue_display() {
        let issue = PreflightIssue::BitrateTooLow {
            actual_kbps: 500,
            min_kbps: 1_000,
        };
        let s = issue.to_string();
        assert!(s.contains("500"));
        assert!(s.contains("1000"));
    }
}
