//! Sign language track metadata — language code, interpreter info, and coverage windows.
//!
//! Provides structured metadata for sign language video tracks embedded in or
//! referenced by a media asset.  Supports multiple sign languages, multiple
//! interpreters, partial coverage windows, and quality grading.

use std::collections::HashMap;
use std::fmt;

use crate::{AccessError, AccessResult};

// ─── Types ────────────────────────────────────────────────────────────────────

/// Recognised sign language codes (ISO 639-3 where available).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SignLanguageCode {
    /// American Sign Language (`ase`).
    Asl,
    /// British Sign Language (`bfi`).
    Bsl,
    /// Japanese Sign Language (`jsl`).
    Jsl,
    /// Auslan — Australian Sign Language (`asf`).
    Auslan,
    /// International Sign (not ISO-standardised).
    InternationalSign,
    /// Arbitrary IETF/ISO tag for less common sign languages.
    Other(String),
}

impl SignLanguageCode {
    /// Return the ISO 639-3 code (or a descriptive tag for `Other`).
    #[must_use]
    pub fn iso_code(&self) -> &str {
        match self {
            Self::Asl => "ase",
            Self::Bsl => "bfi",
            Self::Jsl => "jsl",
            Self::Auslan => "asf",
            Self::InternationalSign => "ils",
            Self::Other(tag) => tag.as_str(),
        }
    }

    /// Human-readable name.
    #[must_use]
    pub fn display_name(&self) -> &str {
        match self {
            Self::Asl => "American Sign Language",
            Self::Bsl => "British Sign Language",
            Self::Jsl => "Japanese Sign Language",
            Self::Auslan => "Auslan",
            Self::InternationalSign => "International Sign",
            Self::Other(tag) => tag.as_str(),
        }
    }
}

impl fmt::Display for SignLanguageCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.display_name(), self.iso_code())
    }
}

/// Positioning of the sign-language window within the video frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignerPosition {
    /// Lower-right corner (most common for broadcast).
    BottomRight,
    /// Lower-left corner.
    BottomLeft,
    /// Upper-right corner.
    TopRight,
    /// Upper-left corner.
    TopLeft,
    /// Full-screen signer (replaces main video).
    FullScreen,
    /// Custom or unspecified position.
    Custom,
}

impl SignerPosition {
    /// Human-readable description.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::BottomRight => "Bottom-right",
            Self::BottomLeft => "Bottom-left",
            Self::TopRight => "Top-right",
            Self::TopLeft => "Top-left",
            Self::FullScreen => "Full-screen",
            Self::Custom => "Custom",
        }
    }
}

impl fmt::Display for SignerPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.description())
    }
}

/// Quality grade of the sign language interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InterpretationQuality {
    /// Machine-generated (avatar / synthesis).
    Automated = 0,
    /// Student or trainee interpreter.
    Trainee = 1,
    /// Professional interpreter.
    Professional = 2,
    /// Certified broadcast-level interpreter.
    Broadcast = 3,
    /// Native deaf signer.
    NativeSigner = 4,
}

impl InterpretationQuality {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Automated => "Automated",
            Self::Trainee => "Trainee",
            Self::Professional => "Professional",
            Self::Broadcast => "Broadcast",
            Self::NativeSigner => "Native Signer",
        }
    }
}

impl fmt::Display for InterpretationQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Information about a sign language interpreter.
#[derive(Debug, Clone)]
pub struct InterpreterInfo {
    /// Optional display name (may be withheld for privacy).
    pub name: Option<String>,
    /// Interpreter's qualification / quality grade.
    pub quality: InterpretationQuality,
    /// Organisation or agency the interpreter belongs to.
    pub organisation: Option<String>,
    /// Contact or identifier for rights/credits purposes.
    pub identifier: Option<String>,
}

impl InterpreterInfo {
    /// Create a new interpreter record.
    #[must_use]
    pub fn new(quality: InterpretationQuality) -> Self {
        Self {
            name: None,
            quality,
            organisation: None,
            identifier: None,
        }
    }

    /// Set the interpreter's name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the interpreter's organisation.
    #[must_use]
    pub fn with_organisation(mut self, org: impl Into<String>) -> Self {
        self.organisation = Some(org.into());
        self
    }

    /// Set a unique identifier.
    #[must_use]
    pub fn with_identifier(mut self, id: impl Into<String>) -> Self {
        self.identifier = Some(id.into());
        self
    }
}

/// A half-open time window `[start_ms, end_ms)` in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoverageWindow {
    /// Start time in milliseconds (inclusive).
    pub start_ms: u64,
    /// End time in milliseconds (exclusive).
    pub end_ms: u64,
}

impl CoverageWindow {
    /// Create a new coverage window.
    ///
    /// # Errors
    /// Returns [`AccessError::InvalidTiming`] when `end_ms <= start_ms`.
    pub fn new(start_ms: u64, end_ms: u64) -> AccessResult<Self> {
        if end_ms <= start_ms {
            return Err(AccessError::InvalidTiming(format!(
                "CoverageWindow: end_ms ({end_ms}) must exceed start_ms ({start_ms})"
            )));
        }
        Ok(Self { start_ms, end_ms })
    }

    /// Duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms - self.start_ms
    }

    /// Whether `ms` is within this window.
    #[must_use]
    pub fn contains(&self, ms: u64) -> bool {
        ms >= self.start_ms && ms < self.end_ms
    }

    /// Whether this window overlaps with `other`.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start_ms < other.end_ms && other.start_ms < self.end_ms
    }
}

impl fmt::Display for CoverageWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{} ms, {} ms)", self.start_ms, self.end_ms)
    }
}

/// Metadata for a single sign language track.
#[derive(Debug, Clone)]
pub struct SignLanguageTrack {
    /// Track identifier (e.g. `"sl-bsl-001"`).
    pub track_id: String,
    /// Sign language used.
    pub sign_language: SignLanguageCode,
    /// Interpreter details.
    pub interpreter: Option<InterpreterInfo>,
    /// Signer placement on screen.
    pub position: SignerPosition,
    /// Percentage size of the signer window relative to the frame (e.g. `25.0` = 25%).
    pub window_size_pct: f32,
    /// Time windows during which the sign language track is active.
    coverage_windows: Vec<CoverageWindow>,
    /// Total video duration for coverage calculations.
    pub video_duration_ms: Option<u64>,
    /// Additional key-value annotations.
    annotations: HashMap<String, String>,
}

impl SignLanguageTrack {
    /// Create a new sign language track with no coverage windows yet.
    ///
    /// # Errors
    /// Returns [`AccessError::Other`] when `window_size_pct` is outside `(0.0, 100.0]`.
    pub fn new(
        track_id: impl Into<String>,
        sign_language: SignLanguageCode,
        position: SignerPosition,
        window_size_pct: f32,
    ) -> AccessResult<Self> {
        if !(0.0..=100.0).contains(&window_size_pct) || window_size_pct == 0.0 {
            return Err(AccessError::Other(format!(
                "window_size_pct must be in (0, 100], got {window_size_pct}"
            )));
        }
        Ok(Self {
            track_id: track_id.into(),
            sign_language,
            interpreter: None,
            position,
            window_size_pct,
            coverage_windows: Vec::new(),
            video_duration_ms: None,
            annotations: HashMap::new(),
        })
    }

    /// Attach interpreter information.
    #[must_use]
    pub fn with_interpreter(mut self, interpreter: InterpreterInfo) -> Self {
        self.interpreter = Some(interpreter);
        self
    }

    /// Set the total video duration for coverage ratio calculations.
    #[must_use]
    pub fn with_video_duration_ms(mut self, duration_ms: u64) -> Self {
        self.video_duration_ms = Some(duration_ms);
        self
    }

    /// Add a coverage window.
    ///
    /// # Errors
    /// Returns [`AccessError::SyncError`] when the new window overlaps an existing one.
    pub fn add_coverage_window(&mut self, window: CoverageWindow) -> AccessResult<()> {
        for existing in &self.coverage_windows {
            if existing.overlaps(&window) {
                return Err(AccessError::SyncError(format!(
                    "Coverage window {} overlaps with existing {} in track '{}'",
                    window, existing, self.track_id
                )));
            }
        }
        self.coverage_windows.push(window);
        // Keep sorted by start_ms for deterministic gap queries.
        self.coverage_windows.sort_by_key(|w| w.start_ms);
        Ok(())
    }

    /// Remove all coverage windows.
    pub fn clear_coverage(&mut self) {
        self.coverage_windows.clear();
    }

    /// Return a reference to coverage windows ordered by start time.
    #[must_use]
    pub fn coverage_windows(&self) -> &[CoverageWindow] {
        &self.coverage_windows
    }

    /// Total covered duration in milliseconds.
    #[must_use]
    pub fn covered_ms(&self) -> u64 {
        self.coverage_windows.iter().map(|w| w.duration_ms()).sum()
    }

    /// Whether the sign language track is active at `timecode_ms`.
    #[must_use]
    pub fn active_at(&self, timecode_ms: u64) -> bool {
        self.coverage_windows.iter().any(|w| w.contains(timecode_ms))
    }

    /// Coverage fraction `[0.0, 1.0]` relative to `video_duration_ms`.
    ///
    /// Returns `None` when `video_duration_ms` is not set or zero.
    #[must_use]
    pub fn coverage_fraction(&self) -> Option<f64> {
        let total = self.video_duration_ms?;
        if total == 0 {
            return None;
        }
        Some(self.covered_ms() as f64 / total as f64)
    }

    /// Set an arbitrary annotation key-value pair.
    pub fn annotate(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.annotations.insert(key.into(), value.into());
    }

    /// Retrieve an annotation by key.
    #[must_use]
    pub fn annotation(&self, key: &str) -> Option<&str> {
        self.annotations.get(key).map(String::as_str)
    }

    /// Whether this track provides full coverage of the video.
    ///
    /// Returns `false` when `video_duration_ms` is not set.
    #[must_use]
    pub fn is_full_coverage(&self) -> bool {
        self.coverage_fraction().map(|f| f >= 1.0).unwrap_or(false)
    }
}

/// Registry of sign language tracks for a media asset.
#[derive(Debug, Clone, Default)]
pub struct SignLanguageRegistry {
    tracks: Vec<SignLanguageTrack>,
}

impl SignLanguageRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self { tracks: Vec::new() }
    }

    /// Register a track.
    pub fn register(&mut self, track: SignLanguageTrack) {
        self.tracks.push(track);
    }

    /// Return the number of registered tracks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Return all tracks for a given sign language ISO code.
    #[must_use]
    pub fn tracks_for_language(&self, iso_code: &str) -> Vec<&SignLanguageTrack> {
        self.tracks
            .iter()
            .filter(|t| t.sign_language.iso_code() == iso_code)
            .collect()
    }

    /// Return the best-quality track for a sign language (highest `InterpretationQuality`).
    #[must_use]
    pub fn best_quality_track(&self, iso_code: &str) -> Option<&SignLanguageTrack> {
        self.tracks
            .iter()
            .filter(|t| t.sign_language.iso_code() == iso_code)
            .filter(|t| t.interpreter.is_some())
            .max_by_key(|t| t.interpreter.as_ref().map(|i| i.quality).unwrap_or(InterpretationQuality::Automated))
    }

    /// Return distinct ISO codes of all registered sign languages.
    #[must_use]
    pub fn sign_languages(&self) -> Vec<&str> {
        let mut codes: Vec<&str> = self
            .tracks
            .iter()
            .map(|t| t.sign_language.iso_code())
            .collect();
        codes.sort_unstable();
        codes.dedup();
        codes
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_track(id: &str, lang: SignLanguageCode) -> SignLanguageTrack {
        SignLanguageTrack::new(id, lang, SignerPosition::BottomRight, 25.0)
            .expect("valid track")
    }

    #[test]
    fn test_sign_language_code_iso() {
        assert_eq!(SignLanguageCode::Asl.iso_code(), "ase");
        assert_eq!(SignLanguageCode::Bsl.iso_code(), "bfi");
        assert_eq!(SignLanguageCode::Jsl.iso_code(), "jsl");
        assert_eq!(SignLanguageCode::Other("xyz".to_string()).iso_code(), "xyz");
    }

    #[test]
    fn test_coverage_window_invalid() {
        assert!(CoverageWindow::new(5000, 1000).is_err());
        assert!(CoverageWindow::new(1000, 1000).is_err());
    }

    #[test]
    fn test_coverage_window_overlap_detection() {
        let a = CoverageWindow::new(0, 3000).unwrap();
        let b = CoverageWindow::new(2000, 5000).unwrap();
        let c = CoverageWindow::new(3000, 6000).unwrap();
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_track_add_overlapping_windows_rejected() {
        let mut track = make_track("t1", SignLanguageCode::Bsl);
        track
            .add_coverage_window(CoverageWindow::new(0, 5000).unwrap())
            .unwrap();
        let result = track.add_coverage_window(CoverageWindow::new(4000, 8000).unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn test_track_covered_ms_and_fraction() {
        let mut track = make_track("t1", SignLanguageCode::Bsl)
            .with_video_duration_ms(20_000);
        track
            .add_coverage_window(CoverageWindow::new(0, 5_000).unwrap())
            .unwrap();
        track
            .add_coverage_window(CoverageWindow::new(10_000, 15_000).unwrap())
            .unwrap();
        assert_eq!(track.covered_ms(), 10_000);
        let frac = track.coverage_fraction().expect("fraction");
        assert!((frac - 0.5).abs() < 1e-9, "frac={frac}");
    }

    #[test]
    fn test_track_active_at() {
        let mut track = make_track("t1", SignLanguageCode::Asl);
        track
            .add_coverage_window(CoverageWindow::new(1_000, 4_000).unwrap())
            .unwrap();
        assert!(track.active_at(2_000));
        assert!(!track.active_at(500));
        assert!(!track.active_at(4_000));
    }

    #[test]
    fn test_track_window_size_validation() {
        assert!(SignLanguageTrack::new("t", SignLanguageCode::Bsl, SignerPosition::BottomRight, 0.0).is_err());
        assert!(SignLanguageTrack::new("t", SignLanguageCode::Bsl, SignerPosition::BottomRight, 101.0).is_err());
        assert!(SignLanguageTrack::new("t", SignLanguageCode::Bsl, SignerPosition::BottomRight, 100.0).is_ok());
    }

    #[test]
    fn test_annotations() {
        let mut track = make_track("t1", SignLanguageCode::Bsl);
        track.annotate("production-id", "P001");
        assert_eq!(track.annotation("production-id"), Some("P001"));
        assert_eq!(track.annotation("missing"), None);
    }

    #[test]
    fn test_registry_best_quality() {
        let mut reg = SignLanguageRegistry::new();
        let trainee = make_track("trainee", SignLanguageCode::Bsl)
            .with_interpreter(InterpreterInfo::new(InterpretationQuality::Trainee));
        let broadcast = make_track("broadcast", SignLanguageCode::Bsl)
            .with_interpreter(InterpreterInfo::new(InterpretationQuality::Broadcast));
        reg.register(trainee);
        reg.register(broadcast);

        let best = reg.best_quality_track("bfi").expect("track found");
        assert_eq!(best.track_id, "broadcast");
    }

    #[test]
    fn test_registry_sign_languages() {
        let mut reg = SignLanguageRegistry::new();
        reg.register(make_track("a", SignLanguageCode::Asl));
        reg.register(make_track("b", SignLanguageCode::Bsl));
        reg.register(make_track("c", SignLanguageCode::Asl));

        let codes = reg.sign_languages();
        assert_eq!(codes.len(), 2);
        assert!(codes.contains(&"ase"));
        assert!(codes.contains(&"bfi"));
    }

    #[test]
    fn test_interpreter_info_builder() {
        let info = InterpreterInfo::new(InterpretationQuality::Professional)
            .with_name("Jane Doe")
            .with_organisation("AccessMedia Ltd")
            .with_identifier("IM-001");
        assert_eq!(info.name.as_deref(), Some("Jane Doe"));
        assert_eq!(info.organisation.as_deref(), Some("AccessMedia Ltd"));
        assert_eq!(info.quality, InterpretationQuality::Professional);
    }

    #[test]
    fn test_is_full_coverage() {
        let mut track = make_track("t1", SignLanguageCode::Bsl)
            .with_video_duration_ms(5_000);
        assert!(!track.is_full_coverage());
        track
            .add_coverage_window(CoverageWindow::new(0, 5_000).unwrap())
            .unwrap();
        assert!(track.is_full_coverage());
    }
}
