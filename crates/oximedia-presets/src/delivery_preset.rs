//! Delivery preset definitions for final output and distribution.
#![allow(dead_code)]

/// The target delivery environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeliveryTarget {
    /// Linear broadcast (SDI, ATSC, DVB, ISDB).
    Broadcast,
    /// OTT / adaptive-bitrate streaming (HLS, DASH).
    Streaming,
    /// Digital cinema package (DCP / DCDM).
    FilmDcp,
    /// Small-screen / limited-bandwidth first delivery.
    MobileFirst,
    /// Physical disc authoring (Blu-ray, DVD).
    Disc,
    /// Social media / web upload.
    Social,
}

impl DeliveryTarget {
    /// Maximum loudness in LKFS/LUFS accepted by this delivery target.
    ///
    /// Returns `None` for targets without a mandated loudness ceiling.
    #[must_use]
    pub fn max_lkfs(&self) -> Option<f32> {
        match self {
            DeliveryTarget::Broadcast => Some(-24.0),
            DeliveryTarget::Streaming => Some(-14.0),
            DeliveryTarget::FilmDcp => Some(-31.0),
            DeliveryTarget::MobileFirst => Some(-16.0),
            DeliveryTarget::Disc => Some(-23.0),
            DeliveryTarget::Social => None,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            DeliveryTarget::Broadcast => "Broadcast",
            DeliveryTarget::Streaming => "Streaming",
            DeliveryTarget::FilmDcp => "Film DCP",
            DeliveryTarget::MobileFirst => "Mobile First",
            DeliveryTarget::Disc => "Disc",
            DeliveryTarget::Social => "Social",
        }
    }
}

/// A delivery preset coupling a target with codec and quality parameters.
#[derive(Debug, Clone)]
pub struct DeliveryPreset {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Delivery target.
    pub target: DeliveryTarget,
    /// Video codec (e.g. `"h264"`, `"hevc"`, `"av1"`).
    pub video_codec: String,
    /// Audio codec (e.g. `"aac"`, `"ac3"`, `"eac3"`).
    pub audio_codec: String,
    /// Target video bitrate in kbps.
    pub video_kbps: u32,
    /// Whether the preset carries HDR metadata.
    pub hdr: bool,
    /// Whether the preset carries Dolby Atmos audio.
    pub atmos: bool,
}

impl DeliveryPreset {
    /// Create a new delivery preset with sensible defaults.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, target: DeliveryTarget) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            target,
            video_codec: "h264".to_string(),
            audio_codec: "aac".to_string(),
            video_kbps: 5_000,
            hdr: false,
            atmos: false,
        }
    }

    /// Returns `true` when the preset carries any HDR metadata.
    #[must_use]
    pub fn is_hdr(&self) -> bool {
        self.hdr
    }

    /// Returns `true` when Dolby Atmos immersive audio is included.
    #[must_use]
    pub fn is_atmos(&self) -> bool {
        self.atmos
    }

    /// Loudness ceiling for this preset's delivery target.
    #[must_use]
    pub fn max_lkfs(&self) -> Option<f32> {
        self.target.max_lkfs()
    }
}

/// Selects a delivery preset from a collection given a target environment.
#[derive(Debug, Default)]
pub struct DeliveryPresetSelector {
    presets: Vec<DeliveryPreset>,
}

impl DeliveryPresetSelector {
    /// Create an empty selector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a preset.
    pub fn add(&mut self, preset: DeliveryPreset) {
        self.presets.push(preset);
    }

    /// Select the first preset matching the given target.
    ///
    /// When `prefer_hdr` is `true`, an HDR preset is preferred over SDR if
    /// one is available for the target.
    #[must_use]
    pub fn select(&self, target: DeliveryTarget, prefer_hdr: bool) -> Option<&DeliveryPreset> {
        let candidates: Vec<&DeliveryPreset> =
            self.presets.iter().filter(|p| p.target == target).collect();

        if candidates.is_empty() {
            return None;
        }

        if prefer_hdr {
            // Prefer HDR preset if available
            if let Some(hdr) = candidates.iter().find(|p| p.hdr) {
                return Some(hdr);
            }
        }

        candidates.into_iter().next()
    }

    /// Total number of presets registered.
    #[must_use]
    pub fn count(&self) -> usize {
        self.presets.len()
    }

    /// Return all presets for a given delivery target.
    #[must_use]
    pub fn all_for_target(&self, target: DeliveryTarget) -> Vec<&DeliveryPreset> {
        self.presets.iter().filter(|p| p.target == target).collect()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_max_lkfs() {
        assert_eq!(DeliveryTarget::Broadcast.max_lkfs(), Some(-24.0));
    }

    #[test]
    fn test_streaming_max_lkfs() {
        assert_eq!(DeliveryTarget::Streaming.max_lkfs(), Some(-14.0));
    }

    #[test]
    fn test_film_dcp_max_lkfs() {
        assert_eq!(DeliveryTarget::FilmDcp.max_lkfs(), Some(-31.0));
    }

    #[test]
    fn test_social_no_lkfs_ceiling() {
        assert!(DeliveryTarget::Social.max_lkfs().is_none());
    }

    #[test]
    fn test_delivery_target_labels() {
        assert_eq!(DeliveryTarget::Broadcast.label(), "Broadcast");
        assert_eq!(DeliveryTarget::FilmDcp.label(), "Film DCP");
        assert_eq!(DeliveryTarget::MobileFirst.label(), "Mobile First");
    }

    #[test]
    fn test_preset_new_defaults() {
        let p = DeliveryPreset::new("bc", "Broadcast HD", DeliveryTarget::Broadcast);
        assert_eq!(p.target, DeliveryTarget::Broadcast);
        assert!(!p.is_hdr());
        assert!(!p.is_atmos());
    }

    #[test]
    fn test_preset_is_hdr_true() {
        let mut p = DeliveryPreset::new("hdr", "HDR", DeliveryTarget::Streaming);
        p.hdr = true;
        assert!(p.is_hdr());
    }

    #[test]
    fn test_preset_max_lkfs_delegates_to_target() {
        let p = DeliveryPreset::new("disc", "Disc", DeliveryTarget::Disc);
        assert_eq!(p.max_lkfs(), Some(-23.0));
    }

    #[test]
    fn test_selector_add_and_count() {
        let mut sel = DeliveryPresetSelector::new();
        assert_eq!(sel.count(), 0);
        sel.add(DeliveryPreset::new("a", "A", DeliveryTarget::Broadcast));
        assert_eq!(sel.count(), 1);
    }

    #[test]
    fn test_selector_select_returns_matching_target() {
        let mut sel = DeliveryPresetSelector::new();
        sel.add(DeliveryPreset::new(
            "bc",
            "Broadcast",
            DeliveryTarget::Broadcast,
        ));
        sel.add(DeliveryPreset::new(
            "str",
            "Streaming",
            DeliveryTarget::Streaming,
        ));
        let result = sel.select(DeliveryTarget::Broadcast, false);
        assert!(result.is_some());
        assert_eq!(
            result.expect("test expectation failed").target,
            DeliveryTarget::Broadcast
        );
    }

    #[test]
    fn test_selector_select_no_match_returns_none() {
        let mut sel = DeliveryPresetSelector::new();
        sel.add(DeliveryPreset::new(
            "bc",
            "Broadcast",
            DeliveryTarget::Broadcast,
        ));
        let result = sel.select(DeliveryTarget::FilmDcp, false);
        assert!(result.is_none());
    }

    #[test]
    fn test_selector_prefers_hdr() {
        let mut sel = DeliveryPresetSelector::new();
        let mut sdr = DeliveryPreset::new("sdr", "SDR", DeliveryTarget::Streaming);
        sdr.hdr = false;
        let mut hdr = DeliveryPreset::new("hdr", "HDR", DeliveryTarget::Streaming);
        hdr.hdr = true;
        sel.add(sdr);
        sel.add(hdr);
        let result = sel
            .select(DeliveryTarget::Streaming, true)
            .expect("result should be valid");
        assert!(result.is_hdr());
    }

    #[test]
    fn test_selector_all_for_target() {
        let mut sel = DeliveryPresetSelector::new();
        sel.add(DeliveryPreset::new("a", "A", DeliveryTarget::Social));
        sel.add(DeliveryPreset::new("b", "B", DeliveryTarget::Social));
        sel.add(DeliveryPreset::new("c", "C", DeliveryTarget::Broadcast));
        let social = sel.all_for_target(DeliveryTarget::Social);
        assert_eq!(social.len(), 2);
    }

    #[test]
    fn test_disc_target_lkfs() {
        let target = DeliveryTarget::Disc;
        assert_eq!(target.max_lkfs(), Some(-23.0));
    }

    #[test]
    fn test_mobile_first_lkfs() {
        assert_eq!(DeliveryTarget::MobileFirst.max_lkfs(), Some(-16.0));
    }

    #[test]
    fn test_selector_empty_select_returns_none() {
        let sel = DeliveryPresetSelector::new();
        assert!(sel.select(DeliveryTarget::Broadcast, false).is_none());
    }
}
