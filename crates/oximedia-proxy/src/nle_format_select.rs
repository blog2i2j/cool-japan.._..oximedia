#![allow(dead_code)]
//! Automatic proxy format selection based on NLE detection.
//!
//! Different Non-Linear Editors (NLEs) have different preferred proxy formats:
//! - Adobe Premiere Pro prefers ProRes or H.264
//! - DaVinci Resolve prefers DNxHR or ProRes
//! - Final Cut Pro X prefers ProRes Proxy
//! - Avid Media Composer requires DNxHD/DNxHR
//!
//! This module detects the target NLE (by name, project file extension, or
//! environment hints) and recommends the optimal proxy codec, container,
//! resolution, and bitrate.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// NLE identification
// ---------------------------------------------------------------------------

/// Known NLE applications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Nle {
    /// Adobe Premiere Pro.
    PremierePro,
    /// DaVinci Resolve (free or Studio).
    DaVinciResolve,
    /// Apple Final Cut Pro X.
    FinalCutPro,
    /// Avid Media Composer.
    AvidMediaComposer,
    /// Vegas Pro.
    VegasPro,
    /// HitFilm / FXHome.
    HitFilm,
    /// Unknown or generic NLE.
    Generic,
}

impl Nle {
    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::PremierePro => "Adobe Premiere Pro",
            Self::DaVinciResolve => "DaVinci Resolve",
            Self::FinalCutPro => "Final Cut Pro",
            Self::AvidMediaComposer => "Avid Media Composer",
            Self::VegasPro => "Vegas Pro",
            Self::HitFilm => "HitFilm",
            Self::Generic => "Generic",
        }
    }

    /// Project file extensions associated with this NLE.
    pub fn project_extensions(&self) -> &[&str] {
        match self {
            Self::PremierePro => &[".prproj"],
            Self::DaVinciResolve => &[".drp", ".dra"],
            Self::FinalCutPro => &[".fcpxml", ".fcpbundle"],
            Self::AvidMediaComposer => &[".avp", ".avs"],
            Self::VegasPro => &[".veg", ".vf"],
            Self::HitFilm => &[".hfp"],
            Self::Generic => &[],
        }
    }
}

/// Detect the NLE from a project file path.
///
/// Examines the file extension to identify the NLE. Returns [`Nle::Generic`]
/// if no match is found.
pub fn detect_nle_from_project(path: &str) -> Nle {
    let lower = path.to_lowercase();
    let nles = [
        Nle::PremierePro,
        Nle::DaVinciResolve,
        Nle::FinalCutPro,
        Nle::AvidMediaComposer,
        Nle::VegasPro,
        Nle::HitFilm,
    ];
    for nle in &nles {
        for ext in nle.project_extensions() {
            if lower.ends_with(ext) {
                return *nle;
            }
        }
    }
    Nle::Generic
}

/// Detect the NLE from an application name string.
///
/// Performs case-insensitive substring matching.
pub fn detect_nle_from_app_name(name: &str) -> Nle {
    let lower = name.to_lowercase();
    if lower.contains("premiere") {
        Nle::PremierePro
    } else if lower.contains("resolve") || lower.contains("davinci") {
        Nle::DaVinciResolve
    } else if lower.contains("final cut") || lower.contains("fcpx") {
        Nle::FinalCutPro
    } else if lower.contains("avid") || lower.contains("media composer") {
        Nle::AvidMediaComposer
    } else if lower.contains("vegas") {
        Nle::VegasPro
    } else if lower.contains("hitfilm") {
        Nle::HitFilm
    } else {
        Nle::Generic
    }
}

// ---------------------------------------------------------------------------
// Proxy format recommendation
// ---------------------------------------------------------------------------

/// Recommended proxy codec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyCodecRecommendation {
    /// Primary codec name.
    pub codec: String,
    /// Alternative codec if primary is not available.
    pub fallback_codec: Option<String>,
    /// Recommended container format (e.g. "mov", "mxf", "mp4").
    pub container: String,
    /// Recommended bitrate in kbps.
    pub bitrate_kbps: u32,
    /// Recommended resolution (width, height).
    pub resolution: (u32, u32),
    /// Whether the codec is patent-free.
    pub patent_free: bool,
}

impl ProxyCodecRecommendation {
    /// Create a new recommendation.
    pub fn new(
        codec: impl Into<String>,
        container: impl Into<String>,
        bitrate_kbps: u32,
        resolution: (u32, u32),
    ) -> Self {
        Self {
            codec: codec.into(),
            fallback_codec: None,
            container: container.into(),
            bitrate_kbps,
            resolution,
            patent_free: false,
        }
    }

    /// Set fallback codec.
    pub fn with_fallback(mut self, codec: impl Into<String>) -> Self {
        self.fallback_codec = Some(codec.into());
        self
    }

    /// Mark as patent-free.
    pub fn with_patent_free(mut self, free: bool) -> Self {
        self.patent_free = free;
        self
    }

    /// Estimated proxy file size for a given duration in seconds.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn estimated_size_bytes(&self, duration_secs: f64) -> u64 {
        // bitrate_kbps * 1000 / 8 * duration_secs
        (self.bitrate_kbps as f64 * 125.0 * duration_secs) as u64
    }
}

/// Source resolution category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceResolution {
    /// SD (up to 720p).
    Sd,
    /// HD (720p to 1080p).
    Hd,
    /// UHD / 4K.
    Uhd,
    /// 8K or higher.
    HigherThan4K,
}

impl SourceResolution {
    /// Classify a width into a resolution category.
    pub fn from_width(width: u32) -> Self {
        if width >= 7680 {
            Self::HigherThan4K
        } else if width >= 3840 {
            Self::Uhd
        } else if width >= 1280 {
            Self::Hd
        } else {
            Self::Sd
        }
    }

    /// Recommended proxy resolution for this source category.
    pub fn proxy_resolution(&self) -> (u32, u32) {
        match self {
            Self::Sd => (640, 360),
            Self::Hd => (960, 540),
            Self::Uhd => (1920, 1080),
            Self::HigherThan4K => (1920, 1080),
        }
    }
}

// ---------------------------------------------------------------------------
// NLE format selector
// ---------------------------------------------------------------------------

/// Selects the optimal proxy format for a given NLE and source resolution.
pub struct NleFormatSelector {
    /// Custom codec overrides per NLE.
    overrides: HashMap<Nle, ProxyCodecRecommendation>,
    /// Whether to prefer patent-free codecs when possible.
    prefer_patent_free: bool,
}

impl NleFormatSelector {
    /// Create a new selector with default recommendations.
    pub fn new() -> Self {
        Self {
            overrides: HashMap::new(),
            prefer_patent_free: false,
        }
    }

    /// Enable patent-free codec preference.
    pub fn with_patent_free_preference(mut self) -> Self {
        self.prefer_patent_free = true;
        self
    }

    /// Register a custom codec override for an NLE.
    pub fn set_override(&mut self, nle: Nle, recommendation: ProxyCodecRecommendation) {
        self.overrides.insert(nle, recommendation);
    }

    /// Get the recommended proxy format for a given NLE and source width.
    pub fn recommend(&self, nle: Nle, source_width: u32) -> ProxyCodecRecommendation {
        // Check for overrides first
        if let Some(custom) = self.overrides.get(&nle) {
            return custom.clone();
        }

        let res_cat = SourceResolution::from_width(source_width);
        let proxy_res = res_cat.proxy_resolution();

        if self.prefer_patent_free {
            return self.patent_free_recommendation(proxy_res);
        }

        match nle {
            Nle::PremierePro => ProxyCodecRecommendation::new("prores_proxy", "mov", 4_500, proxy_res)
                .with_fallback("h264"),

            Nle::DaVinciResolve => ProxyCodecRecommendation::new("dnxhr_lb", "mxf", 3_600, proxy_res)
                .with_fallback("prores_proxy"),

            Nle::FinalCutPro => ProxyCodecRecommendation::new("prores_proxy", "mov", 4_500, proxy_res),

            Nle::AvidMediaComposer => ProxyCodecRecommendation::new("dnxhd", "mxf", 3_600, proxy_res)
                .with_fallback("dnxhr_lb"),

            Nle::VegasPro => ProxyCodecRecommendation::new("h264", "mp4", 5_000, proxy_res)
                .with_fallback("prores_proxy"),

            Nle::HitFilm => ProxyCodecRecommendation::new("h264", "mp4", 5_000, proxy_res),

            Nle::Generic => self.patent_free_recommendation(proxy_res),
        }
    }

    /// Patent-free codec recommendation (VP9 in WebM).
    fn patent_free_recommendation(&self, resolution: (u32, u32)) -> ProxyCodecRecommendation {
        ProxyCodecRecommendation::new("vp9", "webm", 4_000, resolution)
            .with_fallback("av1")
            .with_patent_free(true)
    }

    /// Recommend based on project file path (auto-detect NLE).
    pub fn recommend_from_project(&self, project_path: &str, source_width: u32) -> ProxyCodecRecommendation {
        let nle = detect_nle_from_project(project_path);
        self.recommend(nle, source_width)
    }

    /// Recommend based on application name (auto-detect NLE).
    pub fn recommend_from_app(&self, app_name: &str, source_width: u32) -> ProxyCodecRecommendation {
        let nle = detect_nle_from_app_name(app_name);
        self.recommend(nle, source_width)
    }

    /// Get all supported NLEs and their primary codecs.
    pub fn supported_nles(&self) -> Vec<(Nle, String)> {
        let nles = [
            Nle::PremierePro,
            Nle::DaVinciResolve,
            Nle::FinalCutPro,
            Nle::AvidMediaComposer,
            Nle::VegasPro,
            Nle::HitFilm,
            Nle::Generic,
        ];
        nles.iter()
            .map(|nle| (*nle, self.recommend(*nle, 1920).codec))
            .collect()
    }
}

impl Default for NleFormatSelector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_nle_premiere() {
        assert_eq!(detect_nle_from_project("project.prproj"), Nle::PremierePro);
    }

    #[test]
    fn test_detect_nle_resolve() {
        assert_eq!(detect_nle_from_project("timeline.drp"), Nle::DaVinciResolve);
        assert_eq!(detect_nle_from_project("archive.dra"), Nle::DaVinciResolve);
    }

    #[test]
    fn test_detect_nle_fcp() {
        assert_eq!(detect_nle_from_project("edit.fcpxml"), Nle::FinalCutPro);
        assert_eq!(detect_nle_from_project("lib.fcpbundle"), Nle::FinalCutPro);
    }

    #[test]
    fn test_detect_nle_avid() {
        assert_eq!(detect_nle_from_project("seq.avp"), Nle::AvidMediaComposer);
    }

    #[test]
    fn test_detect_nle_vegas() {
        assert_eq!(detect_nle_from_project("show.veg"), Nle::VegasPro);
    }

    #[test]
    fn test_detect_nle_hitfilm() {
        assert_eq!(detect_nle_from_project("comp.hfp"), Nle::HitFilm);
    }

    #[test]
    fn test_detect_nle_unknown() {
        assert_eq!(detect_nle_from_project("random.xyz"), Nle::Generic);
    }

    #[test]
    fn test_detect_from_app_name() {
        assert_eq!(detect_nle_from_app_name("Adobe Premiere Pro 2024"), Nle::PremierePro);
        assert_eq!(detect_nle_from_app_name("DaVinci Resolve Studio"), Nle::DaVinciResolve);
        assert_eq!(detect_nle_from_app_name("Final Cut Pro"), Nle::FinalCutPro);
        assert_eq!(detect_nle_from_app_name("Avid Media Composer"), Nle::AvidMediaComposer);
        assert_eq!(detect_nle_from_app_name("VEGAS Pro 21"), Nle::VegasPro);
        assert_eq!(detect_nle_from_app_name("HitFilm Express"), Nle::HitFilm);
        assert_eq!(detect_nle_from_app_name("Unknown Editor"), Nle::Generic);
    }

    #[test]
    fn test_recommend_premiere() {
        let selector = NleFormatSelector::new();
        let rec = selector.recommend(Nle::PremierePro, 3840);
        assert_eq!(rec.codec, "prores_proxy");
        assert_eq!(rec.container, "mov");
        assert_eq!(rec.resolution, (1920, 1080)); // 4K source -> 1080p proxy
    }

    #[test]
    fn test_recommend_resolve() {
        let selector = NleFormatSelector::new();
        let rec = selector.recommend(Nle::DaVinciResolve, 1920);
        assert_eq!(rec.codec, "dnxhr_lb");
        assert_eq!(rec.container, "mxf");
        assert_eq!(rec.resolution, (960, 540)); // HD source -> 540p proxy
    }

    #[test]
    fn test_recommend_avid() {
        let selector = NleFormatSelector::new();
        let rec = selector.recommend(Nle::AvidMediaComposer, 1920);
        assert_eq!(rec.codec, "dnxhd");
        assert_eq!(rec.fallback_codec.as_deref(), Some("dnxhr_lb"));
    }

    #[test]
    fn test_recommend_fcp() {
        let selector = NleFormatSelector::new();
        let rec = selector.recommend(Nle::FinalCutPro, 3840);
        assert_eq!(rec.codec, "prores_proxy");
        assert_eq!(rec.container, "mov");
    }

    #[test]
    fn test_recommend_patent_free() {
        let selector = NleFormatSelector::new().with_patent_free_preference();
        let rec = selector.recommend(Nle::PremierePro, 1920);
        assert_eq!(rec.codec, "vp9");
        assert!(rec.patent_free);
        assert_eq!(rec.fallback_codec.as_deref(), Some("av1"));
    }

    #[test]
    fn test_recommend_generic() {
        let selector = NleFormatSelector::new();
        let rec = selector.recommend(Nle::Generic, 1920);
        // Generic uses patent-free
        assert_eq!(rec.codec, "vp9");
        assert!(rec.patent_free);
    }

    #[test]
    fn test_recommend_from_project() {
        let selector = NleFormatSelector::new();
        let rec = selector.recommend_from_project("edit.prproj", 3840);
        assert_eq!(rec.codec, "prores_proxy");
    }

    #[test]
    fn test_recommend_from_app() {
        let selector = NleFormatSelector::new();
        let rec = selector.recommend_from_app("DaVinci Resolve", 1920);
        assert_eq!(rec.codec, "dnxhr_lb");
    }

    #[test]
    fn test_custom_override() {
        let mut selector = NleFormatSelector::new();
        let custom = ProxyCodecRecommendation::new("custom_codec", "mkv", 2_000, (1280, 720));
        selector.set_override(Nle::PremierePro, custom);

        let rec = selector.recommend(Nle::PremierePro, 3840);
        assert_eq!(rec.codec, "custom_codec");
        assert_eq!(rec.container, "mkv");
    }

    #[test]
    fn test_source_resolution_classification() {
        assert_eq!(SourceResolution::from_width(640), SourceResolution::Sd);
        assert_eq!(SourceResolution::from_width(1920), SourceResolution::Hd);
        assert_eq!(SourceResolution::from_width(3840), SourceResolution::Uhd);
        assert_eq!(SourceResolution::from_width(7680), SourceResolution::HigherThan4K);
    }

    #[test]
    fn test_proxy_resolution_mapping() {
        assert_eq!(SourceResolution::Sd.proxy_resolution(), (640, 360));
        assert_eq!(SourceResolution::Hd.proxy_resolution(), (960, 540));
        assert_eq!(SourceResolution::Uhd.proxy_resolution(), (1920, 1080));
    }

    #[test]
    fn test_estimated_size() {
        let rec = ProxyCodecRecommendation::new("h264", "mp4", 8_000, (1920, 1080));
        // 8000 kbps * 125 bytes/kbps * 60 secs = 60_000_000 bytes
        let size = rec.estimated_size_bytes(60.0);
        assert_eq!(size, 60_000_000);
    }

    #[test]
    fn test_supported_nles() {
        let selector = NleFormatSelector::new();
        let nles = selector.supported_nles();
        assert_eq!(nles.len(), 7);
        assert!(nles.iter().any(|(nle, _)| *nle == Nle::PremierePro));
    }

    #[test]
    fn test_nle_name() {
        assert_eq!(Nle::PremierePro.name(), "Adobe Premiere Pro");
        assert_eq!(Nle::DaVinciResolve.name(), "DaVinci Resolve");
        assert_eq!(Nle::FinalCutPro.name(), "Final Cut Pro");
    }

    #[test]
    fn test_nle_extensions() {
        assert!(Nle::PremierePro.project_extensions().contains(&".prproj"));
        assert!(Nle::FinalCutPro.project_extensions().contains(&".fcpxml"));
        assert!(Nle::Generic.project_extensions().is_empty());
    }

    #[test]
    fn test_default_selector() {
        let selector = NleFormatSelector::default();
        let rec = selector.recommend(Nle::Generic, 1920);
        assert_eq!(rec.codec, "vp9");
    }

    #[test]
    fn test_case_insensitive_detection() {
        assert_eq!(detect_nle_from_project("Project.PRPROJ"), Nle::PremierePro);
        assert_eq!(detect_nle_from_app_name("PREMIERE pro"), Nle::PremierePro);
    }
}
