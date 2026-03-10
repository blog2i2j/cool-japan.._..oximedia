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
}

#[cfg(test)]
mod tests {
    use super::*;

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
            .expect("should find best profile");
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
