/// Conversion profile management for `OxiMedia`.
///
/// Provides pre-built and user-defined conversion profiles that bundle
/// codec, bitrate, resolution and frame-rate settings into a single named unit.
///
/// A named set of conversion parameters used to drive the conversion pipeline.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct ConversionProfile {
    /// Human-readable profile name (e.g., "web-720p").
    pub name: String,
    /// Target video codec identifier (e.g., "av1", "vp9").
    pub video_codec: String,
    /// Target video bitrate in kilobits per second.
    pub video_bitrate_kbps: u32,
    /// Target audio codec identifier (e.g., "opus", "vorbis").
    pub audio_codec: String,
    /// Target audio bitrate in kilobits per second.
    pub audio_bitrate_kbps: u32,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Frame-rate numerator.
    pub fps_num: u32,
    /// Frame-rate denominator.
    pub fps_den: u32,
    /// Encoder speed preset (e.g., "fast", "medium", "slow").
    pub preset: String,
}

impl ConversionProfile {
    /// Creates a new profile with sane defaults (1080p, AV1, Opus).
    #[allow(dead_code)]
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            video_codec: "av1".to_string(),
            audio_codec: "opus".to_string(),
            video_bitrate_kbps: 4_000,
            audio_bitrate_kbps: 128,
            width: 1920,
            height: 1080,
            fps_num: 30,
            fps_den: 1,
            preset: "medium".to_string(),
        }
    }

    /// Returns a profile optimised for 720p web delivery (VP9 / Opus).
    #[allow(dead_code)]
    #[must_use]
    pub fn for_web_720p() -> Self {
        Self {
            name: "web-720p".to_string(),
            video_codec: "vp9".to_string(),
            video_bitrate_kbps: 2_500,
            audio_codec: "opus".to_string(),
            audio_bitrate_kbps: 128,
            width: 1280,
            height: 720,
            fps_num: 30,
            fps_den: 1,
            preset: "fast".to_string(),
        }
    }

    /// Returns a profile optimised for 1080p web delivery (AV1 / Opus).
    #[allow(dead_code)]
    #[must_use]
    pub fn for_web_1080p() -> Self {
        Self {
            name: "web-1080p".to_string(),
            video_codec: "av1".to_string(),
            video_bitrate_kbps: 4_500,
            audio_codec: "opus".to_string(),
            audio_bitrate_kbps: 192,
            width: 1920,
            height: 1080,
            fps_num: 30,
            fps_den: 1,
            preset: "medium".to_string(),
        }
    }

    /// Returns a broadcast-grade profile (AV1 / Opus, high bitrate, 1080i-ish).
    #[allow(dead_code)]
    #[must_use]
    pub fn for_broadcast() -> Self {
        Self {
            name: "broadcast".to_string(),
            video_codec: "av1".to_string(),
            video_bitrate_kbps: 15_000,
            audio_codec: "flac".to_string(),
            audio_bitrate_kbps: 320,
            width: 1920,
            height: 1080,
            fps_num: 50,
            fps_den: 1,
            preset: "slow".to_string(),
        }
    }

    /// Returns a lossless archive profile (AV1 lossless / FLAC).
    #[allow(dead_code)]
    #[must_use]
    pub fn for_archive() -> Self {
        Self {
            name: "archive".to_string(),
            video_codec: "av1".to_string(),
            video_bitrate_kbps: 50_000,
            audio_codec: "flac".to_string(),
            audio_bitrate_kbps: 1_411,
            width: 3840,
            height: 2160,
            fps_num: 60,
            fps_den: 1,
            preset: "slow".to_string(),
        }
    }

    /// Returns the total number of pixels per frame (`width × height`).
    #[allow(dead_code)]
    #[must_use]
    pub fn pixel_count(&self) -> u32 {
        self.width.saturating_mul(self.height)
    }

    /// Returns the frame rate as a floating-point number.
    ///
    /// Returns `0.0` if `fps_den` is zero (avoids divide-by-zero).
    #[allow(dead_code)]
    #[must_use]
    pub fn fps(&self) -> f64 {
        if self.fps_den == 0 {
            0.0
        } else {
            f64::from(self.fps_num) / f64::from(self.fps_den)
        }
    }

    /// Returns the total bitrate (video + audio) in kilobits per second.
    #[allow(dead_code)]
    #[must_use]
    pub fn total_bitrate_kbps(&self) -> u32 {
        self.video_bitrate_kbps
            .saturating_add(self.audio_bitrate_kbps)
    }

    /// Returns `true` if the output is a 4K resolution (width ≥ 3840).
    #[allow(dead_code)]
    #[must_use]
    pub fn is_4k(&self) -> bool {
        self.width >= 3840
    }

    /// Returns the aspect ratio as a `(numerator, denominator)` pair, simplified.
    ///
    /// Falls back to `(self.width, self.height)` if GCD cannot be computed.
    #[allow(dead_code)]
    #[must_use]
    pub fn aspect_ratio(&self) -> (u32, u32) {
        let g = gcd(self.width, self.height);
        if g == 0 {
            (self.width, self.height)
        } else {
            (self.width / g, self.height / g)
        }
    }
}

/// Greatest common divisor via Euclidean algorithm.
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ── ProfileRegistry ───────────────────────────────────────────────────────────

/// A collection of named [`ConversionProfile`]s.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct ProfileRegistry {
    /// Stored profiles in insertion order.
    profiles: Vec<ConversionProfile>,
}

impl ProfileRegistry {
    /// Creates a new, empty registry.
    #[allow(dead_code)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            profiles: Vec::new(),
        }
    }

    /// Creates a registry pre-loaded with the built-in profiles.
    #[allow(dead_code)]
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();
        reg.add(ConversionProfile::for_web_720p());
        reg.add(ConversionProfile::for_web_1080p());
        reg.add(ConversionProfile::for_broadcast());
        reg.add(ConversionProfile::for_archive());
        reg
    }

    /// Adds a profile to the registry.  If a profile with the same name
    /// already exists it is replaced.
    #[allow(dead_code)]
    pub fn add(&mut self, p: ConversionProfile) {
        if let Some(existing) = self.profiles.iter_mut().find(|x| x.name == p.name) {
            *existing = p;
        } else {
            self.profiles.push(p);
        }
    }

    /// Finds a profile by name (case-insensitive).
    #[allow(dead_code)]
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<&ConversionProfile> {
        let lower = name.to_lowercase();
        self.profiles
            .iter()
            .find(|p| p.name.to_lowercase() == lower)
    }

    /// Returns a list of all profile names in the registry.
    #[allow(dead_code)]
    #[must_use]
    pub fn list_names(&self) -> Vec<&str> {
        self.profiles.iter().map(|p| p.name.as_str()).collect()
    }

    /// Returns the number of profiles in the registry.
    #[allow(dead_code)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Returns `true` if the registry contains no profiles.
    #[allow(dead_code)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Removes the profile with the given name.  Returns `true` if found.
    #[allow(dead_code)]
    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.profiles.len();
        self.profiles.retain(|p| p.name != name);
        self.profiles.len() < before
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── ConversionProfile ─────────────────────────────────────────────────────

    #[test]
    fn profile_new_defaults() {
        let p = ConversionProfile::new("custom");
        assert_eq!(p.name, "custom");
        assert_eq!(p.video_codec, "av1");
        assert_eq!(p.audio_codec, "opus");
        assert_eq!(p.width, 1920);
        assert_eq!(p.height, 1080);
    }

    #[test]
    fn profile_for_web_720p() {
        let p = ConversionProfile::for_web_720p();
        assert_eq!(p.name, "web-720p");
        assert_eq!(p.width, 1280);
        assert_eq!(p.height, 720);
        assert_eq!(p.video_codec, "vp9");
    }

    #[test]
    fn profile_for_web_1080p() {
        let p = ConversionProfile::for_web_1080p();
        assert_eq!(p.name, "web-1080p");
        assert_eq!(p.width, 1920);
        assert_eq!(p.height, 1080);
    }

    #[test]
    fn profile_for_broadcast_has_high_bitrate() {
        let p = ConversionProfile::for_broadcast();
        assert!(p.video_bitrate_kbps >= 10_000);
        assert_eq!(p.fps(), 50.0);
    }

    #[test]
    fn profile_for_archive_is_4k() {
        let p = ConversionProfile::for_archive();
        assert!(p.is_4k());
        assert_eq!(p.width, 3840);
    }

    #[test]
    fn profile_pixel_count() {
        let p = ConversionProfile::for_web_720p();
        assert_eq!(p.pixel_count(), 1280 * 720);
    }

    #[test]
    fn profile_fps_ratio() {
        let mut p = ConversionProfile::new("test");
        p.fps_num = 24;
        p.fps_den = 1;
        assert!((p.fps() - 24.0).abs() < 1e-9);
    }

    #[test]
    fn profile_fps_zero_denominator() {
        let mut p = ConversionProfile::new("test");
        p.fps_num = 30;
        p.fps_den = 0;
        assert_eq!(p.fps(), 0.0);
    }

    #[test]
    fn profile_total_bitrate() {
        let mut p = ConversionProfile::new("test");
        p.video_bitrate_kbps = 4_000;
        p.audio_bitrate_kbps = 192;
        assert_eq!(p.total_bitrate_kbps(), 4_192);
    }

    #[test]
    fn profile_aspect_ratio_16_9() {
        let p = ConversionProfile::for_web_1080p(); // 1920x1080
        assert_eq!(p.aspect_ratio(), (16, 9));
    }

    // ── ProfileRegistry ───────────────────────────────────────────────────────

    #[test]
    fn registry_new_is_empty() {
        let r = ProfileRegistry::new();
        assert!(r.is_empty());
    }

    #[test]
    fn registry_add_and_find() {
        let mut r = ProfileRegistry::new();
        r.add(ConversionProfile::for_web_720p());
        assert!(r.find_by_name("web-720p").is_some());
    }

    #[test]
    fn registry_find_case_insensitive() {
        let mut r = ProfileRegistry::new();
        r.add(ConversionProfile::for_web_720p());
        assert!(r.find_by_name("WEB-720P").is_some());
    }

    #[test]
    fn registry_add_replaces_existing() {
        let mut r = ProfileRegistry::new();
        let mut p = ConversionProfile::for_web_720p();
        r.add(p.clone());
        p.video_bitrate_kbps = 9_999;
        r.add(p);
        assert_eq!(r.len(), 1);
        assert_eq!(
            r.find_by_name("web-720p")
                .expect("web-720p profile should exist")
                .video_bitrate_kbps,
            9_999
        );
    }

    #[test]
    fn registry_list_names() {
        let r = ProfileRegistry::with_defaults();
        let names = r.list_names();
        assert!(names.contains(&"web-720p"));
        assert!(names.contains(&"broadcast"));
    }

    #[test]
    fn registry_remove() {
        let mut r = ProfileRegistry::with_defaults();
        assert!(r.remove("archive"));
        assert!(r.find_by_name("archive").is_none());
        assert!(!r.remove("archive")); // already gone
    }
}
