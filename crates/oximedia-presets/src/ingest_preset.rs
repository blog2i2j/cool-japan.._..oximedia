//! Ingest preset definitions for capture and acquisition workflows.
#![allow(dead_code)]

/// Quality tier for ingest operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IngestQuality {
    /// Proxy / offline-edit quality (low bitrate).
    Proxy,
    /// Standard definition ingest.
    Sd,
    /// High definition ingest.
    Hd,
    /// Ultra-high definition ingest.
    Uhd,
    /// Lossless / visually lossless master quality.
    Lossless,
    /// RAW sensor data capture.
    Raw,
}

impl IngestQuality {
    /// Nominal target bitrate in Mbit/s for this quality tier.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn bitrate_mbps(&self) -> f64 {
        match self {
            IngestQuality::Proxy => 5.0,
            IngestQuality::Sd => 25.0,
            IngestQuality::Hd => 50.0,
            IngestQuality::Uhd => 200.0,
            IngestQuality::Lossless => 0.0, // lossless is variable
            IngestQuality::Raw => 0.0,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            IngestQuality::Proxy => "Proxy",
            IngestQuality::Sd => "SD",
            IngestQuality::Hd => "HD",
            IngestQuality::Uhd => "UHD",
            IngestQuality::Lossless => "Lossless",
            IngestQuality::Raw => "RAW",
        }
    }
}

/// A single ingest preset describing how to capture or transcode on ingest.
#[derive(Debug, Clone)]
pub struct IngestPreset {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Quality tier.
    pub quality: IngestQuality,
    /// Target video codec (e.g. `"h264"`, `"prores"`, `"dnxhd"`).
    pub video_codec: String,
    /// Target audio codec.
    pub audio_codec: String,
    /// Container format.
    pub container: String,
    /// Maximum frame width accepted.
    pub max_width: u32,
    /// Maximum frame height accepted.
    pub max_height: u32,
    /// Whether colour space metadata should be preserved verbatim.
    pub preserve_color_space: bool,
}

impl IngestPreset {
    /// Create a new ingest preset.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, quality: IngestQuality) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            quality,
            video_codec: "h264".to_string(),
            audio_codec: "aac".to_string(),
            container: "mp4".to_string(),
            max_width: 3840,
            max_height: 2160,
            preserve_color_space: true,
        }
    }

    /// Returns `true` when this preset captures without lossy compression.
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        matches!(self.quality, IngestQuality::Lossless | IngestQuality::Raw)
    }

    /// Returns `true` when the preset targets ultra-high-definition content.
    #[must_use]
    pub fn is_uhd(&self) -> bool {
        matches!(self.quality, IngestQuality::Uhd)
    }

    /// Nominal ingest bitrate in Mbit/s (delegates to quality tier).
    #[must_use]
    pub fn bitrate_mbps(&self) -> f64 {
        self.quality.bitrate_mbps()
    }
}

/// Library of ingest presets, keyed by ID.
#[derive(Debug, Default)]
pub struct IngestPresetLibrary {
    presets: Vec<IngestPreset>,
}

impl IngestPresetLibrary {
    /// Create an empty library.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a preset to the library.
    pub fn add(&mut self, preset: IngestPreset) {
        self.presets.push(preset);
    }

    /// Find a preset by ID, returning a reference if found.
    #[must_use]
    pub fn find(&self, id: &str) -> Option<&IngestPreset> {
        self.presets.iter().find(|p| p.id == id)
    }

    /// Return the preset with the highest nominal bitrate (or lossless).
    ///
    /// Lossless / RAW presets are always ranked higher than any
    /// lossy preset regardless of bitrate.
    #[must_use]
    pub fn best_quality(&self) -> Option<&IngestPreset> {
        self.presets.iter().max_by(|a, b| {
            // Lossless > everything else
            let a_is_lossless = a.is_lossless();
            let b_is_lossless = b.is_lossless();
            match (a_is_lossless, b_is_lossless) {
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                _ => a
                    .quality
                    .bitrate_mbps()
                    .partial_cmp(&b.quality.bitrate_mbps())
                    .unwrap_or(std::cmp::Ordering::Equal),
            }
        })
    }

    /// Number of presets in the library.
    #[must_use]
    pub fn len(&self) -> usize {
        self.presets.len()
    }

    /// Returns `true` when the library contains no presets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.presets.is_empty()
    }

    /// Return all presets matching a quality tier.
    #[must_use]
    pub fn by_quality(&self, quality: IngestQuality) -> Vec<&IngestPreset> {
        self.presets
            .iter()
            .filter(|p| p.quality == quality)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingest_quality_bitrate_proxy() {
        assert!((IngestQuality::Proxy.bitrate_mbps() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ingest_quality_bitrate_hd() {
        assert!((IngestQuality::Hd.bitrate_mbps() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ingest_quality_bitrate_lossless_is_zero() {
        assert_eq!(IngestQuality::Lossless.bitrate_mbps(), 0.0);
    }

    #[test]
    fn test_ingest_quality_labels() {
        assert_eq!(IngestQuality::Proxy.label(), "Proxy");
        assert_eq!(IngestQuality::Lossless.label(), "Lossless");
        assert_eq!(IngestQuality::Raw.label(), "RAW");
    }

    #[test]
    fn test_preset_new_defaults() {
        let p = IngestPreset::new("test", "Test Preset", IngestQuality::Hd);
        assert_eq!(p.id, "test");
        assert_eq!(p.quality, IngestQuality::Hd);
        assert_eq!(p.container, "mp4");
    }

    #[test]
    fn test_preset_is_lossless_false_for_hd() {
        let p = IngestPreset::new("hd", "HD", IngestQuality::Hd);
        assert!(!p.is_lossless());
    }

    #[test]
    fn test_preset_is_lossless_true_for_lossless() {
        let p = IngestPreset::new("ll", "Lossless", IngestQuality::Lossless);
        assert!(p.is_lossless());
    }

    #[test]
    fn test_preset_is_lossless_true_for_raw() {
        let p = IngestPreset::new("raw", "RAW", IngestQuality::Raw);
        assert!(p.is_lossless());
    }

    #[test]
    fn test_preset_is_uhd() {
        let p = IngestPreset::new("uhd", "UHD", IngestQuality::Uhd);
        assert!(p.is_uhd());
        let p2 = IngestPreset::new("hd", "HD", IngestQuality::Hd);
        assert!(!p2.is_uhd());
    }

    #[test]
    fn test_library_add_and_find() {
        let mut lib = IngestPresetLibrary::new();
        let p = IngestPreset::new("id-1", "Preset 1", IngestQuality::Hd);
        lib.add(p);
        assert!(lib.find("id-1").is_some());
        assert!(lib.find("nonexistent").is_none());
    }

    #[test]
    fn test_library_find_returns_correct_preset() {
        let mut lib = IngestPresetLibrary::new();
        lib.add(IngestPreset::new("a", "A", IngestQuality::Proxy));
        lib.add(IngestPreset::new("b", "B", IngestQuality::Hd));
        let found = lib.find("b").expect("found should be valid");
        assert_eq!(found.quality, IngestQuality::Hd);
    }

    #[test]
    fn test_library_best_quality_prefers_lossless() {
        let mut lib = IngestPresetLibrary::new();
        lib.add(IngestPreset::new("hd", "HD", IngestQuality::Hd));
        lib.add(IngestPreset::new("uhd", "UHD", IngestQuality::Uhd));
        lib.add(IngestPreset::new("ll", "LL", IngestQuality::Lossless));
        let best = lib.best_quality().expect("best should be valid");
        assert!(best.is_lossless());
    }

    #[test]
    fn test_library_best_quality_without_lossless() {
        let mut lib = IngestPresetLibrary::new();
        lib.add(IngestPreset::new("proxy", "Proxy", IngestQuality::Proxy));
        lib.add(IngestPreset::new("uhd", "UHD", IngestQuality::Uhd));
        let best = lib.best_quality().expect("best should be valid");
        assert_eq!(best.quality, IngestQuality::Uhd);
    }

    #[test]
    fn test_library_best_quality_empty_returns_none() {
        let lib = IngestPresetLibrary::new();
        assert!(lib.best_quality().is_none());
    }

    #[test]
    fn test_library_len_and_is_empty() {
        let mut lib = IngestPresetLibrary::new();
        assert!(lib.is_empty());
        lib.add(IngestPreset::new("x", "X", IngestQuality::Sd));
        assert_eq!(lib.len(), 1);
        assert!(!lib.is_empty());
    }

    #[test]
    fn test_library_by_quality_filter() {
        let mut lib = IngestPresetLibrary::new();
        lib.add(IngestPreset::new("p1", "Proxy 1", IngestQuality::Proxy));
        lib.add(IngestPreset::new("p2", "Proxy 2", IngestQuality::Proxy));
        lib.add(IngestPreset::new("hd", "HD", IngestQuality::Hd));
        let proxies = lib.by_quality(IngestQuality::Proxy);
        assert_eq!(proxies.len(), 2);
        let hds = lib.by_quality(IngestQuality::Hd);
        assert_eq!(hds.len(), 1);
    }
}
