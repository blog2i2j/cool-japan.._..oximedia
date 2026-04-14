//! Conversion profile matching: automatic profile selection based on media
//! properties, and compatibility scoring.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Describes basic media properties used for profile matching.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MediaSpec {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Frames per second (0.0 for audio-only)
    pub fps: f64,
    /// Audio sample rate in Hz (0 for video-only)
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u8,
    /// Whether the source has HDR metadata
    pub hdr: bool,
    /// Whether the source has alpha channel
    pub has_alpha: bool,
    /// Approximate bitrate in kbps
    pub bitrate_kbps: u32,
}

impl MediaSpec {
    /// Create a typical HD video spec.
    #[must_use]
    pub fn hd_video() -> Self {
        Self {
            width: 1920,
            height: 1080,
            fps: 30.0,
            sample_rate: 48_000,
            channels: 2,
            hdr: false,
            has_alpha: false,
            bitrate_kbps: 8_000,
        }
    }

    /// Returns the total pixel count.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Aspect ratio width/height, or 1.0 if height is 0.
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        if self.height == 0 {
            return 1.0;
        }
        f64::from(self.width) / f64::from(self.height)
    }
}

/// Represents a named conversion profile with constraints.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConversionProfile {
    /// Unique profile name
    pub name: String,
    /// Maximum width supported
    pub max_width: u32,
    /// Maximum height supported
    pub max_height: u32,
    /// Maximum fps supported
    pub max_fps: f64,
    /// Whether HDR is supported
    pub supports_hdr: bool,
    /// Whether alpha is supported
    pub supports_alpha: bool,
    /// Target bitrate in kbps
    pub target_bitrate_kbps: u32,
    /// Output format tag (e.g. "mp4")
    pub output_format: String,
}

impl ConversionProfile {
    /// Score how well this profile matches a `MediaSpec` (0.0–1.0).
    /// Higher is better.
    #[must_use]
    pub fn compatibility_score(&self, spec: &MediaSpec) -> f64 {
        let mut score = 1.0_f64;

        // Resolution penalty
        if spec.width > self.max_width || spec.height > self.max_height {
            score -= 0.3;
        }

        // FPS penalty
        if spec.fps > self.max_fps {
            score -= 0.2;
        }

        // HDR support
        if spec.hdr && !self.supports_hdr {
            score -= 0.25;
        }

        // Alpha support
        if spec.has_alpha && !self.supports_alpha {
            score -= 0.15;
        }

        // Bitrate distance penalty (normalised, max 0.1)
        let br_diff =
            (i64::from(spec.bitrate_kbps) - i64::from(self.target_bitrate_kbps)).unsigned_abs();
        let br_penalty = (br_diff as f64 / 50_000.0).min(0.1);
        score -= br_penalty;

        score.clamp(0.0, 1.0)
    }

    /// Whether this profile can handle the given spec (no hard-fail criteria).
    #[must_use]
    pub fn is_compatible(&self, spec: &MediaSpec) -> bool {
        self.compatibility_score(spec) > 0.0
    }
}

/// Selects the best-matching profile from a registry for a given media spec.
#[derive(Debug, Default)]
pub struct ProfileMatcher {
    profiles: Vec<ConversionProfile>,
}

impl ProfileMatcher {
    /// Create an empty matcher.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a profile.
    pub fn register(&mut self, profile: ConversionProfile) {
        self.profiles.push(profile);
    }

    /// Return all profiles sorted by compatibility score (best first).
    #[must_use]
    pub fn ranked_profiles<'a>(&'a self, spec: &MediaSpec) -> Vec<(&'a ConversionProfile, f64)> {
        let mut ranked: Vec<_> = self
            .profiles
            .iter()
            .map(|p| (p, p.compatibility_score(spec)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Return the best-matching profile name, if any.
    #[must_use]
    pub fn best_match(&self, spec: &MediaSpec) -> Option<&ConversionProfile> {
        self.profiles.iter().max_by(|a, b| {
            a.compatibility_score(spec)
                .partial_cmp(&b.compatibility_score(spec))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Number of registered profiles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Whether no profiles are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Return profiles compatible with the spec (score > 0).
    #[must_use]
    pub fn compatible_profiles<'a>(&'a self, spec: &MediaSpec) -> Vec<&'a ConversionProfile> {
        self.profiles
            .iter()
            .filter(|p| p.is_compatible(spec))
            .collect()
    }
}

/// Hint that influences automatic profile selection.
///
/// Callers can supply one or more hints to guide the `AutoProfileSelector`
/// toward a particular optimisation goal without fully specifying a profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionHint {
    /// Prefer the smallest output file (maximise compression).
    MinimiseSize,
    /// Prefer the highest fidelity / least lossy output.
    MaximiseFidelity,
    /// Target browser-compatible output for web delivery.
    WebCompatible,
    /// Target mobile device playback (lower resolution / bitrate).
    MobileTarget,
    /// Preserve all source characteristics (HDR, alpha, high bitrate).
    ArchiveQuality,
    /// Optimise for fast encoding speed over quality.
    FastEncode,
}

/// Selection context aggregating a media spec and optional hints.
///
/// The `AutoProfileSelector` uses this to find the best profile from a
/// registered set, combining compatibility scoring with caller intent.
#[derive(Debug, Clone)]
pub struct SelectionContext {
    /// Observed or declared properties of the source media.
    pub spec: MediaSpec,
    /// Zero or more caller-supplied hints.
    pub hints: Vec<SelectionHint>,
    /// Optional preferred output format extension (e.g. "webm").
    pub preferred_format: Option<String>,
}

impl SelectionContext {
    /// Build a context from a spec only (no hints, no preferred format).
    #[must_use]
    pub fn from_spec(spec: MediaSpec) -> Self {
        Self {
            spec,
            hints: Vec::new(),
            preferred_format: None,
        }
    }

    /// Add a selection hint.
    #[must_use]
    pub fn with_hint(mut self, hint: SelectionHint) -> Self {
        self.hints.push(hint);
        self
    }

    /// Set the preferred output format.
    #[must_use]
    pub fn with_format(mut self, fmt: impl Into<String>) -> Self {
        self.preferred_format = Some(fmt.into());
        self
    }

    /// Returns `true` if the given hint is present.
    #[must_use]
    pub fn has_hint(&self, hint: &SelectionHint) -> bool {
        self.hints.contains(hint)
    }
}

/// Result of automatic profile selection.
#[derive(Debug, Clone)]
pub struct ProfileSelection {
    /// The selected profile.
    pub profile: ConversionProfile,
    /// Final combined score (0.0–1.0, higher is better).
    pub score: f64,
    /// Human-readable explanation of why this profile was chosen.
    pub rationale: String,
    /// Whether any hints caused an adjustment to the score.
    pub hint_adjusted: bool,
}

/// Automatic profile selector combining compatibility scoring with caller hints.
///
/// # Algorithm
///
/// For each registered profile the selector computes:
///
/// 1. A *compatibility score* via [`ConversionProfile::compatibility_score`].
/// 2. A *hint bonus* (±0.0–0.3) added when the profile aligns with or
///    contradicts a caller-supplied [`SelectionHint`].
/// 3. A *format bonus* (+0.1) when the profile's `output_format` matches the
///    caller's `preferred_format`.
///
/// The profile with the highest combined score is returned.
///
/// # Tie-breaking
///
/// When two profiles share the same final score, the one whose name sorts
/// lexicographically first is chosen for deterministic output.
#[derive(Debug, Default)]
pub struct AutoProfileSelector {
    profiles: Vec<ConversionProfile>,
}

impl AutoProfileSelector {
    /// Create a new empty selector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a selector pre-loaded with all profiles from [`ProfileLibrary`].
    #[must_use]
    pub fn with_standard_profiles() -> Self {
        let mut s = Self::new();
        s.register(ProfileLibrary::web_streaming());
        s.register(ProfileLibrary::mobile());
        s.register(ProfileLibrary::archive());
        s.register(ProfileLibrary::broadcast_hd());
        s.register(ProfileLibrary::social_media());
        s.register(ProfileLibrary::screen_recording());
        s
    }

    /// Register a profile for selection.
    pub fn register(&mut self, profile: ConversionProfile) {
        self.profiles.push(profile);
    }

    /// Number of registered profiles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Returns `true` when no profiles are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Select the best profile for the given context.
    ///
    /// Returns `None` when no profiles have been registered.
    #[must_use]
    pub fn select(&self, ctx: &SelectionContext) -> Option<ProfileSelection> {
        if self.profiles.is_empty() {
            return None;
        }

        struct Candidate<'a> {
            profile: &'a ConversionProfile,
            score: f64,
            rationale: String,
            hint_adjusted: bool,
        }

        let mut best: Option<Candidate<'_>> = None;

        for profile in &self.profiles {
            let base = profile.compatibility_score(&ctx.spec);
            let (hint_delta, adjusted) = self.hint_bonus(profile, ctx);
            let fmt_bonus = self.format_bonus(profile, ctx);
            let total = (base + hint_delta + fmt_bonus).clamp(0.0, 1.0);
            let rationale = self.build_rationale(profile, base, hint_delta, fmt_bonus, ctx);

            let replace = match &best {
                None => true,
                Some(prev) => {
                    total > prev.score || (total == prev.score && profile.name < prev.profile.name)
                }
            };

            if replace {
                best = Some(Candidate {
                    profile,
                    score: total,
                    rationale,
                    hint_adjusted: adjusted,
                });
            }
        }

        best.map(|c| ProfileSelection {
            profile: c.profile.clone(),
            score: c.score,
            rationale: c.rationale,
            hint_adjusted: c.hint_adjusted,
        })
    }

    /// Compute a hint-derived score adjustment for `profile` given `ctx`.
    ///
    /// Returns `(delta, was_adjusted)`.
    fn hint_bonus(&self, profile: &ConversionProfile, ctx: &SelectionContext) -> (f64, bool) {
        let mut delta = 0.0_f64;
        let mut adjusted = false;

        for hint in &ctx.hints {
            match hint {
                SelectionHint::MinimiseSize => {
                    let ratio =
                        profile.target_bitrate_kbps as f64 / ctx.spec.bitrate_kbps.max(1) as f64;
                    if ratio < 0.5 {
                        delta += 0.15;
                        adjusted = true;
                    } else if ratio > 2.0 {
                        delta -= 0.1;
                        adjusted = true;
                    }
                }
                SelectionHint::MaximiseFidelity => {
                    if profile.target_bitrate_kbps >= 30_000 {
                        delta += 0.15;
                        adjusted = true;
                    }
                    if profile.supports_hdr && ctx.spec.hdr {
                        delta += 0.05;
                        adjusted = true;
                    }
                }
                SelectionHint::WebCompatible => {
                    if profile.output_format == "mp4" || profile.output_format == "webm" {
                        delta += 0.1;
                        adjusted = true;
                    }
                }
                SelectionHint::MobileTarget => {
                    if profile.max_width <= 1280 && profile.max_height <= 720 {
                        delta += 0.2;
                        adjusted = true;
                    } else {
                        delta -= 0.05;
                        adjusted = true;
                    }
                }
                SelectionHint::ArchiveQuality => {
                    if profile.supports_hdr && profile.supports_alpha {
                        delta += 0.2;
                        adjusted = true;
                    }
                    if profile.max_width >= 3840 {
                        delta += 0.05;
                        adjusted = true;
                    }
                }
                SelectionHint::FastEncode => {
                    if profile.max_width <= 1920 && profile.target_bitrate_kbps <= 10_000 {
                        delta += 0.1;
                        adjusted = true;
                    }
                }
            }
        }

        (delta, adjusted)
    }

    /// Format-match bonus: +0.1 when `profile.output_format` matches `ctx.preferred_format`.
    fn format_bonus(&self, profile: &ConversionProfile, ctx: &SelectionContext) -> f64 {
        if let Some(ref fmt) = ctx.preferred_format {
            if profile.output_format.to_lowercase() == fmt.to_lowercase() {
                return 0.1;
            }
        }
        0.0
    }

    /// Build a human-readable rationale string.
    fn build_rationale(
        &self,
        profile: &ConversionProfile,
        base: f64,
        hint_delta: f64,
        format_bonus: f64,
        ctx: &SelectionContext,
    ) -> String {
        let total = (base + hint_delta + format_bonus).clamp(0.0, 1.0);
        let mut parts = vec![format!(
            "Profile '{}' selected (combined score {:.2} = base {:.2}",
            profile.name, total, base,
        )];

        if hint_delta.abs() > f64::EPSILON {
            parts.push(format!(", hint adjustment {hint_delta:+.2}"));
        }
        if format_bonus > f64::EPSILON {
            parts.push(format!(", format bonus {format_bonus:+.2}"));
        }
        parts.push(")".to_string());

        if !ctx.hints.is_empty() {
            let names: Vec<&str> = ctx
                .hints
                .iter()
                .map(|h| match h {
                    SelectionHint::MinimiseSize => "MinimiseSize",
                    SelectionHint::MaximiseFidelity => "MaximiseFidelity",
                    SelectionHint::WebCompatible => "WebCompatible",
                    SelectionHint::MobileTarget => "MobileTarget",
                    SelectionHint::ArchiveQuality => "ArchiveQuality",
                    SelectionHint::FastEncode => "FastEncode",
                })
                .collect();
            parts.push(format!("; hints=[{}]", names.join(", ")));
        }
        if let Some(ref fmt) = ctx.preferred_format {
            parts.push(format!("; preferred_format={fmt}"));
        }

        parts.concat()
    }

    /// Return all profiles with their selection scores, sorted best-first.
    #[must_use]
    pub fn ranked(&self, ctx: &SelectionContext) -> Vec<(f64, &ConversionProfile)> {
        let mut ranked: Vec<(f64, &ConversionProfile)> = self
            .profiles
            .iter()
            .map(|p| {
                let base = p.compatibility_score(&ctx.spec);
                let (bonus, _) = self.hint_bonus(p, ctx);
                let fmt = self.format_bonus(p, ctx);
                ((base + bonus + fmt).clamp(0.0, 1.0), p)
            })
            .collect();

        ranked.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.name.cmp(&b.1.name))
        });
        ranked
    }
}

/// Helper to build common profiles.
pub struct ProfileLibrary;

impl ProfileLibrary {
    /// Standard web streaming profile.
    #[must_use]
    pub fn web_streaming() -> ConversionProfile {
        ConversionProfile {
            name: "web-streaming".into(),
            max_width: 1920,
            max_height: 1080,
            max_fps: 60.0,
            supports_hdr: false,
            supports_alpha: false,
            target_bitrate_kbps: 4_000,
            output_format: "mp4".into(),
        }
    }

    /// Archive preservation profile.
    #[must_use]
    pub fn archive() -> ConversionProfile {
        ConversionProfile {
            name: "archive".into(),
            max_width: 7680,
            max_height: 4320,
            max_fps: 120.0,
            supports_hdr: true,
            supports_alpha: true,
            target_bitrate_kbps: 50_000,
            output_format: "mkv".into(),
        }
    }

    /// Mobile-optimised profile.
    #[must_use]
    pub fn mobile() -> ConversionProfile {
        ConversionProfile {
            name: "mobile".into(),
            max_width: 1280,
            max_height: 720,
            max_fps: 30.0,
            supports_hdr: false,
            supports_alpha: false,
            target_bitrate_kbps: 1_500,
            output_format: "mp4".into(),
        }
    }

    /// Broadcast HD profile (high bitrate, HDR support).
    #[must_use]
    pub fn broadcast_hd() -> ConversionProfile {
        ConversionProfile {
            name: "broadcast-hd".into(),
            max_width: 1920,
            max_height: 1080,
            max_fps: 60.0,
            supports_hdr: true,
            supports_alpha: false,
            target_bitrate_kbps: 20_000,
            output_format: "mkv".into(),
        }
    }

    /// Social-media optimised profile (1080p, web-friendly, moderate bitrate).
    #[must_use]
    pub fn social_media() -> ConversionProfile {
        ConversionProfile {
            name: "social-media".into(),
            max_width: 1920,
            max_height: 1080,
            max_fps: 60.0,
            supports_hdr: false,
            supports_alpha: false,
            target_bitrate_kbps: 6_000,
            output_format: "mp4".into(),
        }
    }

    /// Screen recording profile (monitor-sized, lossless-friendly, WebM output).
    #[must_use]
    pub fn screen_recording() -> ConversionProfile {
        ConversionProfile {
            name: "screen-recording".into(),
            max_width: 3840,
            max_height: 2160,
            max_fps: 60.0,
            supports_hdr: false,
            supports_alpha: false,
            target_bitrate_kbps: 8_000,
            output_format: "webm".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── AutoProfileSelector tests ─────────────────────────────────────────────

    fn mobile_spec() -> MediaSpec {
        MediaSpec {
            width: 720,
            height: 480,
            fps: 30.0,
            sample_rate: 44_100,
            channels: 2,
            hdr: false,
            has_alpha: false,
            bitrate_kbps: 1_200,
        }
    }

    fn hdr_4k_spec() -> MediaSpec {
        MediaSpec {
            width: 3840,
            height: 2160,
            fps: 60.0,
            sample_rate: 48_000,
            channels: 6,
            hdr: true,
            has_alpha: false,
            bitrate_kbps: 40_000,
        }
    }

    #[test]
    fn auto_selector_empty_returns_none() {
        let sel = AutoProfileSelector::new();
        let ctx = SelectionContext::from_spec(MediaSpec::hd_video());
        assert!(sel.select(&ctx).is_none());
    }

    #[test]
    fn auto_selector_single_profile_always_selected() {
        let mut sel = AutoProfileSelector::new();
        sel.register(ProfileLibrary::web_streaming());
        let ctx = SelectionContext::from_spec(MediaSpec::hd_video());
        let result = sel.select(&ctx).expect("should select the only profile");
        assert_eq!(result.profile.name, "web-streaming");
    }

    #[test]
    fn auto_selector_standard_profiles_loaded() {
        let sel = AutoProfileSelector::with_standard_profiles();
        assert!(sel.len() >= 5, "should have at least 5 standard profiles");
    }

    #[test]
    fn auto_selector_mobile_hint_picks_mobile_profile() {
        let sel = AutoProfileSelector::with_standard_profiles();
        let ctx = SelectionContext::from_spec(mobile_spec()).with_hint(SelectionHint::MobileTarget);
        let result = sel.select(&ctx).expect("should select a profile");
        assert_eq!(result.profile.name, "mobile");
    }

    #[test]
    fn auto_selector_archive_hint_picks_archive() {
        let sel = AutoProfileSelector::with_standard_profiles();
        let ctx =
            SelectionContext::from_spec(hdr_4k_spec()).with_hint(SelectionHint::ArchiveQuality);
        let result = sel.select(&ctx).expect("should select a profile");
        assert_eq!(result.profile.name, "archive");
    }

    #[test]
    fn auto_selector_web_hint_picks_web_compatible() {
        let sel = AutoProfileSelector::with_standard_profiles();
        let ctx = SelectionContext::from_spec(MediaSpec::hd_video())
            .with_hint(SelectionHint::WebCompatible);
        let result = sel.select(&ctx).expect("should select a profile");
        let fmt = &result.profile.output_format;
        assert!(
            fmt == "mp4" || fmt == "webm",
            "web-compatible profile should produce mp4 or webm, got {fmt}"
        );
    }

    #[test]
    fn auto_selector_format_bonus_applied() {
        let sel = AutoProfileSelector::with_standard_profiles();
        let ctx = SelectionContext::from_spec(MediaSpec::hd_video()).with_format("webm");
        let result = sel.select(&ctx).expect("should select a profile");
        assert_eq!(result.profile.output_format, "webm");
    }

    #[test]
    fn auto_selector_score_in_range() {
        let sel = AutoProfileSelector::with_standard_profiles();
        let ctx = SelectionContext::from_spec(MediaSpec::hd_video());
        let result = sel.select(&ctx).expect("should select a profile");
        assert!(
            result.score >= 0.0 && result.score <= 1.0,
            "score={} out of [0,1]",
            result.score
        );
    }

    #[test]
    fn auto_selector_rationale_is_non_empty() {
        let sel = AutoProfileSelector::with_standard_profiles();
        let ctx = SelectionContext::from_spec(MediaSpec::hd_video())
            .with_hint(SelectionHint::MaximiseFidelity);
        let result = sel.select(&ctx).expect("should select a profile");
        assert!(!result.rationale.is_empty());
        assert!(
            result.rationale.contains(&result.profile.name),
            "rationale should mention profile name"
        );
    }

    #[test]
    fn auto_selector_ranked_sorted_descending() {
        let sel = AutoProfileSelector::with_standard_profiles();
        let ctx = SelectionContext::from_spec(MediaSpec::hd_video());
        let ranked = sel.ranked(&ctx);
        assert!(!ranked.is_empty());
        for w in ranked.windows(2) {
            assert!(w[0].0 >= w[1].0, "ranked list must be in descending order");
        }
    }

    #[test]
    fn auto_selector_ranked_count_matches_registered() {
        let sel = AutoProfileSelector::with_standard_profiles();
        let ctx = SelectionContext::from_spec(MediaSpec::hd_video());
        assert_eq!(sel.ranked(&ctx).len(), sel.len());
    }

    #[test]
    fn auto_selector_minimise_size_avoids_archive() {
        let sel = AutoProfileSelector::with_standard_profiles();
        let ctx = SelectionContext::from_spec(MediaSpec::hd_video())
            .with_hint(SelectionHint::MinimiseSize);
        let result = sel.select(&ctx).expect("should select a profile");
        // archive has 50000 kbps bitrate → should not win when minimising size
        assert_ne!(
            result.profile.name, "archive",
            "MinimiseSize hint should not select archive (highest bitrate)"
        );
    }

    #[test]
    fn selection_context_has_hint() {
        let ctx = SelectionContext::from_spec(MediaSpec::hd_video())
            .with_hint(SelectionHint::WebCompatible)
            .with_hint(SelectionHint::FastEncode);
        assert!(ctx.has_hint(&SelectionHint::WebCompatible));
        assert!(ctx.has_hint(&SelectionHint::FastEncode));
        assert!(!ctx.has_hint(&SelectionHint::MobileTarget));
    }

    #[test]
    fn auto_selector_is_empty_and_len() {
        let mut sel = AutoProfileSelector::new();
        assert!(sel.is_empty());
        sel.register(ProfileLibrary::mobile());
        assert!(!sel.is_empty());
        assert_eq!(sel.len(), 1);
    }

    #[test]
    fn auto_selector_hint_adjusted_flag_set() {
        let sel = AutoProfileSelector::with_standard_profiles();
        let ctx = SelectionContext::from_spec(mobile_spec()).with_hint(SelectionHint::MobileTarget);
        let result = sel.select(&ctx).expect("should select a profile");
        assert!(result.hint_adjusted, "hint should have adjusted the score");
    }

    // ── Existing ProfileMatcher tests ─────────────────────────────────────────

    #[test]
    fn test_media_spec_pixel_count() {
        let spec = MediaSpec::hd_video();
        assert_eq!(spec.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_media_spec_aspect_ratio() {
        let spec = MediaSpec::hd_video();
        let ar = spec.aspect_ratio();
        assert!((ar - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_media_spec_zero_height() {
        let mut spec = MediaSpec::hd_video();
        spec.height = 0;
        assert!((spec.aspect_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_profile_perfect_match() {
        let profile = ProfileLibrary::web_streaming();
        let spec = MediaSpec::hd_video();
        let score = profile.compatibility_score(&spec);
        assert!(score > 0.8, "score = {score}");
    }

    #[test]
    fn test_profile_oversized_resolution_penalty() {
        let profile = ProfileLibrary::mobile();
        let spec = MediaSpec::hd_video(); // 1920x1080 > mobile max 1280x720
        let score = profile.compatibility_score(&spec);
        assert!(score < 0.75);
    }

    #[test]
    fn test_profile_hdr_penalty() {
        let mut spec = MediaSpec::hd_video();
        spec.hdr = true;
        let profile = ProfileLibrary::web_streaming(); // no HDR support
        let score = profile.compatibility_score(&spec);
        assert!(score < 0.8);
    }

    #[test]
    fn test_profile_alpha_penalty() {
        let mut spec = MediaSpec::hd_video();
        spec.has_alpha = true;
        let profile = ProfileLibrary::web_streaming();
        let score = profile.compatibility_score(&spec);
        assert!(score < 0.9);
    }

    #[test]
    fn test_profile_is_compatible() {
        let profile = ProfileLibrary::archive();
        assert!(profile.is_compatible(&MediaSpec::hd_video()));
    }

    #[test]
    fn test_matcher_best_match() {
        let mut matcher = ProfileMatcher::new();
        matcher.register(ProfileLibrary::web_streaming());
        matcher.register(ProfileLibrary::mobile());
        matcher.register(ProfileLibrary::archive());
        let best = matcher
            .best_match(&MediaSpec::hd_video())
            .expect("should find best match");
        // Archive should win for HD source with moderate bitrate
        // (mobile is penalised for oversized resolution)
        assert_ne!(best.name, "mobile");
    }

    #[test]
    fn test_matcher_ranked_profiles_order() {
        let mut matcher = ProfileMatcher::new();
        matcher.register(ProfileLibrary::web_streaming());
        matcher.register(ProfileLibrary::mobile());
        let ranked = matcher.ranked_profiles(&MediaSpec::hd_video());
        assert!(ranked[0].1 >= ranked[1].1);
    }

    #[test]
    fn test_matcher_compatible_profiles() {
        let mut matcher = ProfileMatcher::new();
        matcher.register(ProfileLibrary::archive());
        let compat = matcher.compatible_profiles(&MediaSpec::hd_video());
        assert!(!compat.is_empty());
    }

    #[test]
    fn test_matcher_empty() {
        let matcher = ProfileMatcher::new();
        assert!(matcher.is_empty());
        assert!(matcher.best_match(&MediaSpec::hd_video()).is_none());
    }

    #[test]
    fn test_matcher_len() {
        let mut matcher = ProfileMatcher::new();
        matcher.register(ProfileLibrary::mobile());
        assert_eq!(matcher.len(), 1);
    }

    #[test]
    fn test_profile_score_clamped_to_zero() {
        // Force all penalties to fire at once
        let profile = ProfileLibrary::mobile();
        let spec = MediaSpec {
            width: 7680,
            height: 4320,
            fps: 120.0,
            sample_rate: 48_000,
            channels: 2,
            hdr: true,
            has_alpha: true,
            bitrate_kbps: 100_000,
        };
        let score = profile.compatibility_score(&spec);
        assert!(score >= 0.0, "score must be >= 0, got {score}");
    }
}
