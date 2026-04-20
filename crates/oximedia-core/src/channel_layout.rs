//! Audio channel layout definitions and utilities.
//!
//! Provides channel layout descriptors for common speaker configurations
//! including Dolby Atmos height-channel formats, and utilities for
//! interleaving / indexing individual channels.

#![allow(dead_code)]

/// An individual audio channel (speaker position).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AudioChannel {
    /// Front left.
    FrontLeft,
    /// Front right.
    FrontRight,
    /// Front centre.
    FrontCenter,
    /// Low-frequency effects (subwoofer).
    LowFrequency,
    /// Back / surround left.
    BackLeft,
    /// Back / surround right.
    BackRight,
    /// Side left.
    SideLeft,
    /// Side right.
    SideRight,
    /// Front wide left.
    FrontWideLeft,
    /// Front wide right.
    FrontWideRight,
    /// Top front left (Atmos height channel).
    TopFrontLeft,
    /// Top front right (Atmos height channel).
    TopFrontRight,
    /// Top middle left (Atmos height channel).
    TopMiddleLeft,
    /// Top middle right (Atmos height channel).
    TopMiddleRight,
    /// Top back left (Atmos height channel).
    TopBackLeft,
    /// Top back right (Atmos height channel).
    TopBackRight,
    /// Back center (mono surround / quad-with-center configurations).
    BackCenter,
}

impl AudioChannel {
    /// Returns a short label for the channel.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::FrontLeft => "FL",
            Self::FrontRight => "FR",
            Self::FrontCenter => "FC",
            Self::LowFrequency => "LFE",
            Self::BackLeft => "BL",
            Self::BackRight => "BR",
            Self::SideLeft => "SL",
            Self::SideRight => "SR",
            Self::FrontWideLeft => "FWL",
            Self::FrontWideRight => "FWR",
            Self::TopFrontLeft => "TFL",
            Self::TopFrontRight => "TFR",
            Self::TopMiddleLeft => "TML",
            Self::TopMiddleRight => "TMR",
            Self::TopBackLeft => "TBL",
            Self::TopBackRight => "TBR",
            Self::BackCenter => "BC",
        }
    }

    /// Returns `true` if this channel is part of the surround field.
    ///
    /// Surround channels include all back, side, and overhead positions.
    #[must_use]
    pub fn is_surround(self) -> bool {
        matches!(
            self,
            Self::BackLeft
                | Self::BackRight
                | Self::SideLeft
                | Self::SideRight
                | Self::FrontWideLeft
                | Self::FrontWideRight
                | Self::TopFrontLeft
                | Self::TopFrontRight
                | Self::TopMiddleLeft
                | Self::TopMiddleRight
                | Self::TopBackLeft
                | Self::TopBackRight
                | Self::BackCenter
        )
    }

    /// Returns `true` if this is an overhead (height) channel used in Atmos/spatial audio.
    #[must_use]
    pub fn is_height_channel(self) -> bool {
        matches!(
            self,
            Self::TopFrontLeft
                | Self::TopFrontRight
                | Self::TopMiddleLeft
                | Self::TopMiddleRight
                | Self::TopBackLeft
                | Self::TopBackRight
        )
    }
}

/// A named audio channel layout.
///
/// Standard presets cover mono through 7.1.4 Atmos configurations.
/// The `Custom` variant allows arbitrary channel lists via [`ChannelLayout::custom`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelLayoutKind {
    /// Mono (1 channel: FC).
    Mono,
    /// Stereo (2 channels: FL, FR).
    Stereo,
    /// Binaural stereo — identical wire format to `Stereo` but semantically
    /// intended for HRTF-rendered headphone playback.
    Binaural,
    /// 2.1 (3 channels: FL, FR, LFE).
    Surround21,
    /// 3.0 (3 channels: FL, FR, FC).
    Surround30,
    /// Quad (4 channels: FL, FR, BL, BR).
    Quad,
    /// 4.0 (4 channels: FL, FR, FC, BC) — quad with front centre and back centre.
    Surround40,
    /// 5.1 (6 channels: FL, FR, FC, LFE, BL, BR).
    Surround51,
    /// 7.1 (8 channels: FL, FR, FC, LFE, BL, BR, SL, SR).
    Surround71,
    /// 7.1.4 (11 channels: FL, FR, FC, LFE, BL, BR, SL, SR, TFL, TFR, TBL/TBR shared pair omitted).
    Surround714,
    /// 9.1.6 (16 channels).
    Surround916,
    /// 5.1.4 Dolby Atmos (10 channels: FL FR FC LFE BL BR TFL TFR TBL TBR).
    Atmos514,
    /// 7.1.4 Dolby Atmos (12 channels: FL FR FC LFE BL BR SL SR TFL TFR TBL TBR).
    Atmos714,
    /// 9.1.6 Dolby Atmos bed (16 channels).
    DolbyAtmosBed9_1_6,
    /// Custom layout (channels described separately).
    Custom,
}

impl ChannelLayoutKind {
    /// Returns the number of channels for standard layouts.
    ///
    /// Returns `0` for `Custom`.
    #[must_use]
    pub fn channel_count(self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo | Self::Binaural => 2,
            Self::Surround21 | Self::Surround30 => 3,
            Self::Quad | Self::Surround40 => 4,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
            Self::Surround714 => 11,
            Self::Surround916 | Self::DolbyAtmosBed9_1_6 => 16,
            Self::Atmos514 => 10,
            Self::Atmos714 => 12,
            Self::Custom => 0,
        }
    }

    /// Returns the ordered channel list for standard layouts.
    ///
    /// Returns an empty slice for `Custom`.
    #[must_use]
    pub fn channels(self) -> &'static [AudioChannel] {
        match self {
            Self::Mono => &[AudioChannel::FrontCenter],
            Self::Stereo | Self::Binaural => &[AudioChannel::FrontLeft, AudioChannel::FrontRight],
            Self::Surround21 => &[
                AudioChannel::FrontLeft,
                AudioChannel::FrontRight,
                AudioChannel::LowFrequency,
            ],
            Self::Surround30 => &[
                AudioChannel::FrontLeft,
                AudioChannel::FrontRight,
                AudioChannel::FrontCenter,
            ],
            Self::Quad => &[
                AudioChannel::FrontLeft,
                AudioChannel::FrontRight,
                AudioChannel::BackLeft,
                AudioChannel::BackRight,
            ],
            Self::Surround40 => &[
                AudioChannel::FrontLeft,
                AudioChannel::FrontRight,
                AudioChannel::FrontCenter,
                AudioChannel::BackCenter,
            ],
            Self::Surround51 => &[
                AudioChannel::FrontLeft,
                AudioChannel::FrontRight,
                AudioChannel::FrontCenter,
                AudioChannel::LowFrequency,
                AudioChannel::BackLeft,
                AudioChannel::BackRight,
            ],
            Self::Surround71 => &[
                AudioChannel::FrontLeft,
                AudioChannel::FrontRight,
                AudioChannel::FrontCenter,
                AudioChannel::LowFrequency,
                AudioChannel::BackLeft,
                AudioChannel::BackRight,
                AudioChannel::SideLeft,
                AudioChannel::SideRight,
            ],
            Self::Surround714 => &[
                AudioChannel::FrontLeft,
                AudioChannel::FrontRight,
                AudioChannel::FrontCenter,
                AudioChannel::LowFrequency,
                AudioChannel::BackLeft,
                AudioChannel::BackRight,
                AudioChannel::SideLeft,
                AudioChannel::SideRight,
                AudioChannel::TopFrontLeft,
                AudioChannel::TopFrontRight,
                AudioChannel::TopBackLeft,
            ],
            Self::Surround916 | Self::DolbyAtmosBed9_1_6 => &[
                AudioChannel::FrontLeft,
                AudioChannel::FrontRight,
                AudioChannel::FrontCenter,
                AudioChannel::LowFrequency,
                AudioChannel::BackLeft,
                AudioChannel::BackRight,
                AudioChannel::SideLeft,
                AudioChannel::SideRight,
                AudioChannel::FrontWideLeft,
                AudioChannel::FrontWideRight,
                AudioChannel::TopFrontLeft,
                AudioChannel::TopFrontRight,
                AudioChannel::TopMiddleLeft,
                AudioChannel::TopMiddleRight,
                AudioChannel::TopBackLeft,
                AudioChannel::TopBackRight,
            ],
            Self::Atmos514 => &[
                AudioChannel::FrontLeft,
                AudioChannel::FrontRight,
                AudioChannel::FrontCenter,
                AudioChannel::LowFrequency,
                AudioChannel::BackLeft,
                AudioChannel::BackRight,
                AudioChannel::TopFrontLeft,
                AudioChannel::TopFrontRight,
                AudioChannel::TopBackLeft,
                AudioChannel::TopBackRight,
            ],
            Self::Atmos714 => &[
                AudioChannel::FrontLeft,
                AudioChannel::FrontRight,
                AudioChannel::FrontCenter,
                AudioChannel::LowFrequency,
                AudioChannel::BackLeft,
                AudioChannel::BackRight,
                AudioChannel::SideLeft,
                AudioChannel::SideRight,
                AudioChannel::TopFrontLeft,
                AudioChannel::TopFrontRight,
                AudioChannel::TopBackLeft,
                AudioChannel::TopBackRight,
            ],
            Self::Custom => &[],
        }
    }

    /// Returns `true` if the layout has a low-frequency effects channel.
    #[must_use]
    pub fn has_lfe(self) -> bool {
        self.channels().contains(&AudioChannel::LowFrequency)
    }

    /// Returns `true` if this layout contains Atmos height channels.
    #[must_use]
    pub fn has_height_channels(self) -> bool {
        self.channels().iter().any(|c| c.is_height_channel())
    }

    /// Returns the number of height (overhead) channels in this layout.
    #[must_use]
    pub fn height_channel_count(self) -> usize {
        self.channels()
            .iter()
            .filter(|c| c.is_height_channel())
            .count()
    }

    /// Returns the bed channel count (total minus height channels).
    ///
    /// For non-Atmos layouts this equals `channel_count`. For `Atmos514`
    /// this is 6 (the 5.1 bed), for `Atmos714` this is 8.
    #[must_use]
    pub fn bed_channel_count(self) -> usize {
        self.channel_count()
            .saturating_sub(self.height_channel_count())
    }

    /// Returns a human-readable name for this layout kind.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Mono => "Mono",
            Self::Stereo => "Stereo",
            Self::Binaural => "Binaural",
            Self::Surround21 => "2.1",
            Self::Surround30 => "3.0",
            Self::Quad => "Quad",
            Self::Surround40 => "4.0",
            Self::Surround51 => "5.1",
            Self::Surround71 => "7.1",
            Self::Surround714 => "7.1.4",
            Self::Surround916 => "9.1.6",
            Self::Atmos514 => "5.1.4 Atmos",
            Self::Atmos714 => "7.1.4 Atmos",
            Self::DolbyAtmosBed9_1_6 => "9.1.6 Atmos Bed",
            Self::Custom => "Custom",
        }
    }
}

/// A concrete channel layout (possibly custom).
#[derive(Debug, Clone)]
pub struct ChannelLayout {
    /// Kind of layout.
    pub kind: ChannelLayoutKind,
    /// Custom channel list (only used when `kind == Custom`).
    custom_channels: Vec<AudioChannel>,
}

impl ChannelLayout {
    /// Creates a standard layout.
    #[must_use]
    pub fn standard(kind: ChannelLayoutKind) -> Self {
        Self {
            kind,
            custom_channels: Vec::new(),
        }
    }

    /// Creates a custom layout from an explicit channel list.
    #[must_use]
    pub fn custom(channels: Vec<AudioChannel>) -> Self {
        Self {
            kind: ChannelLayoutKind::Custom,
            custom_channels: channels,
        }
    }

    /// Returns a slice of the channels in this layout.
    #[must_use]
    pub fn channels(&self) -> &[AudioChannel] {
        if self.kind == ChannelLayoutKind::Custom {
            &self.custom_channels
        } else {
            self.kind.channels()
        }
    }

    /// Returns the total number of channels.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channels().len()
    }

    /// Returns the index of `ch` in this layout, or `None` if not present.
    #[must_use]
    pub fn index_of(&self, ch: AudioChannel) -> Option<usize> {
        self.channels().iter().position(|&c| c == ch)
    }

    /// Returns `true` if this layout contains the given channel.
    #[must_use]
    pub fn contains(&self, ch: AudioChannel) -> bool {
        self.channels().contains(&ch)
    }

    /// Returns a human-readable description of the layout.
    #[must_use]
    pub fn description(&self) -> String {
        self.channels()
            .iter()
            .map(|c| c.label())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Returns `true` if this layout has a low-frequency effects channel.
    #[must_use]
    pub fn has_lfe(&self) -> bool {
        self.contains(AudioChannel::LowFrequency)
    }

    /// Returns `true` if this layout contains any height channels.
    #[must_use]
    pub fn has_height_channels(&self) -> bool {
        self.channels().iter().any(|c| c.is_height_channel())
    }

    /// Returns the number of overhead height channels in this layout.
    #[must_use]
    pub fn height_channel_count(&self) -> usize {
        self.channels()
            .iter()
            .filter(|c| c.is_height_channel())
            .count()
    }

    /// Returns the human-readable name for the layout kind.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.kind.name()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. AudioChannel::label
    #[test]
    fn test_channel_labels() {
        assert_eq!(AudioChannel::FrontLeft.label(), "FL");
        assert_eq!(AudioChannel::LowFrequency.label(), "LFE");
        assert_eq!(AudioChannel::SideRight.label(), "SR");
        assert_eq!(AudioChannel::TopBackLeft.label(), "TBL");
        assert_eq!(AudioChannel::TopBackRight.label(), "TBR");
        assert_eq!(AudioChannel::BackCenter.label(), "BC");
    }

    // 2. AudioChannel::is_surround
    #[test]
    fn test_is_surround_true() {
        assert!(AudioChannel::BackLeft.is_surround());
        assert!(AudioChannel::SideRight.is_surround());
        assert!(AudioChannel::TopFrontLeft.is_surround());
        assert!(AudioChannel::TopBackLeft.is_surround());
        assert!(AudioChannel::TopBackRight.is_surround());
        assert!(AudioChannel::BackCenter.is_surround());
    }

    #[test]
    fn test_is_surround_false() {
        assert!(!AudioChannel::FrontLeft.is_surround());
        assert!(!AudioChannel::LowFrequency.is_surround());
        assert!(!AudioChannel::FrontCenter.is_surround());
    }

    // 3. AudioChannel::is_height_channel
    #[test]
    fn test_is_height_channel() {
        assert!(AudioChannel::TopFrontLeft.is_height_channel());
        assert!(AudioChannel::TopFrontRight.is_height_channel());
        assert!(AudioChannel::TopBackLeft.is_height_channel());
        assert!(AudioChannel::TopBackRight.is_height_channel());
        assert!(!AudioChannel::FrontLeft.is_height_channel());
        assert!(!AudioChannel::BackLeft.is_height_channel());
        assert!(!AudioChannel::SideLeft.is_height_channel());
        assert!(!AudioChannel::LowFrequency.is_height_channel());
    }

    // 4. ChannelLayoutKind::channel_count
    #[test]
    fn test_layout_kind_channel_count() {
        assert_eq!(ChannelLayoutKind::Mono.channel_count(), 1);
        assert_eq!(ChannelLayoutKind::Stereo.channel_count(), 2);
        assert_eq!(ChannelLayoutKind::Binaural.channel_count(), 2);
        assert_eq!(ChannelLayoutKind::Surround21.channel_count(), 3);
        assert_eq!(ChannelLayoutKind::Surround30.channel_count(), 3);
        assert_eq!(ChannelLayoutKind::Quad.channel_count(), 4);
        assert_eq!(ChannelLayoutKind::Surround40.channel_count(), 4);
        assert_eq!(ChannelLayoutKind::Surround51.channel_count(), 6);
        assert_eq!(ChannelLayoutKind::Surround71.channel_count(), 8);
        assert_eq!(ChannelLayoutKind::Surround714.channel_count(), 11);
        assert_eq!(ChannelLayoutKind::Surround916.channel_count(), 16);
        assert_eq!(ChannelLayoutKind::Atmos514.channel_count(), 10);
        assert_eq!(ChannelLayoutKind::Atmos714.channel_count(), 12);
        assert_eq!(ChannelLayoutKind::DolbyAtmosBed9_1_6.channel_count(), 16);
        assert_eq!(ChannelLayoutKind::Custom.channel_count(), 0);
    }

    // 5. ChannelLayoutKind::has_lfe
    #[test]
    fn test_has_lfe() {
        assert!(ChannelLayoutKind::Surround21.has_lfe());
        assert!(ChannelLayoutKind::Surround51.has_lfe());
        assert!(ChannelLayoutKind::Surround71.has_lfe());
        assert!(ChannelLayoutKind::Surround714.has_lfe());
        assert!(ChannelLayoutKind::Surround916.has_lfe());
        assert!(ChannelLayoutKind::Atmos514.has_lfe());
        assert!(ChannelLayoutKind::Atmos714.has_lfe());
        assert!(ChannelLayoutKind::DolbyAtmosBed9_1_6.has_lfe());
        assert!(!ChannelLayoutKind::Stereo.has_lfe());
        assert!(!ChannelLayoutKind::Mono.has_lfe());
        assert!(!ChannelLayoutKind::Quad.has_lfe());
        assert!(!ChannelLayoutKind::Surround30.has_lfe());
        assert!(!ChannelLayoutKind::Surround40.has_lfe());
    }

    // 6. ChannelLayoutKind::has_height_channels
    #[test]
    fn test_layout_kind_has_height_channels() {
        assert!(ChannelLayoutKind::Atmos514.has_height_channels());
        assert!(ChannelLayoutKind::Atmos714.has_height_channels());
        assert!(ChannelLayoutKind::Surround714.has_height_channels());
        assert!(ChannelLayoutKind::Surround916.has_height_channels());
        assert!(ChannelLayoutKind::DolbyAtmosBed9_1_6.has_height_channels());
        assert!(!ChannelLayoutKind::Surround71.has_height_channels());
        assert!(!ChannelLayoutKind::Surround51.has_height_channels());
        assert!(!ChannelLayoutKind::Stereo.has_height_channels());
    }

    // 7. ChannelLayoutKind::height_channel_count
    #[test]
    fn test_height_channel_count() {
        assert_eq!(ChannelLayoutKind::Atmos514.height_channel_count(), 4);
        assert_eq!(ChannelLayoutKind::Atmos714.height_channel_count(), 4);
        assert_eq!(ChannelLayoutKind::Surround714.height_channel_count(), 3);
        assert_eq!(ChannelLayoutKind::Surround916.height_channel_count(), 6);
        assert_eq!(
            ChannelLayoutKind::DolbyAtmosBed9_1_6.height_channel_count(),
            6
        );
        assert_eq!(ChannelLayoutKind::Surround71.height_channel_count(), 0);
        assert_eq!(ChannelLayoutKind::Stereo.height_channel_count(), 0);
    }

    // 8. ChannelLayoutKind::bed_channel_count
    #[test]
    fn test_bed_channel_count() {
        assert_eq!(ChannelLayoutKind::Atmos514.bed_channel_count(), 6);
        assert_eq!(ChannelLayoutKind::Atmos714.bed_channel_count(), 8);
        assert_eq!(ChannelLayoutKind::Surround714.bed_channel_count(), 8);
        assert_eq!(ChannelLayoutKind::Surround916.bed_channel_count(), 10);
        assert_eq!(
            ChannelLayoutKind::DolbyAtmosBed9_1_6.bed_channel_count(),
            10
        );
        assert_eq!(ChannelLayoutKind::Surround71.bed_channel_count(), 8);
        assert_eq!(ChannelLayoutKind::Stereo.bed_channel_count(), 2);
    }

    // 9. ChannelLayout::standard – channel_count matches kind
    #[test]
    fn test_standard_layout_count() {
        let layout = ChannelLayout::standard(ChannelLayoutKind::Surround51);
        assert_eq!(layout.channel_count(), 6);
    }

    // 10. ChannelLayout::standard – contains check (Stereo)
    #[test]
    fn test_standard_layout_contains() {
        let layout = ChannelLayout::standard(ChannelLayoutKind::Stereo);
        assert!(layout.contains(AudioChannel::FrontLeft));
        assert!(layout.contains(AudioChannel::FrontRight));
        assert!(!layout.contains(AudioChannel::FrontCenter));
    }

    // 11. ChannelLayout::index_of – found
    #[test]
    fn test_index_of_found() {
        let layout = ChannelLayout::standard(ChannelLayoutKind::Surround51);
        // 5.1 order: FL FR FC LFE BL BR
        assert_eq!(layout.index_of(AudioChannel::FrontLeft), Some(0));
        assert_eq!(layout.index_of(AudioChannel::LowFrequency), Some(3));
        assert_eq!(layout.index_of(AudioChannel::BackRight), Some(5));
    }

    // 12. ChannelLayout::index_of – not found
    #[test]
    fn test_index_of_not_found() {
        let layout = ChannelLayout::standard(ChannelLayoutKind::Stereo);
        assert_eq!(layout.index_of(AudioChannel::FrontCenter), None);
    }

    // 13. ChannelLayout::custom
    #[test]
    fn test_custom_layout() {
        let layout = ChannelLayout::custom(vec![
            AudioChannel::FrontLeft,
            AudioChannel::FrontRight,
            AudioChannel::TopFrontLeft,
            AudioChannel::TopFrontRight,
        ]);
        assert_eq!(layout.channel_count(), 4);
        assert_eq!(layout.kind, ChannelLayoutKind::Custom);
        assert!(layout.contains(AudioChannel::TopFrontLeft));
    }

    // 14. ChannelLayout::description (Stereo)
    #[test]
    fn test_description_stereo() {
        let layout = ChannelLayout::standard(ChannelLayoutKind::Stereo);
        assert_eq!(layout.description(), "FL, FR");
    }

    // 15. ChannelLayout::description (Mono)
    #[test]
    fn test_description_mono() {
        let layout = ChannelLayout::standard(ChannelLayoutKind::Mono);
        assert_eq!(layout.description(), "FC");
    }

    // 16. 7.1 has side channels
    #[test]
    fn test_surround71_side_channels() {
        let layout = ChannelLayout::standard(ChannelLayoutKind::Surround71);
        assert!(layout.contains(AudioChannel::SideLeft));
        assert!(layout.contains(AudioChannel::SideRight));
    }

    // 17. 2.1 has LFE
    #[test]
    fn test_surround21_has_lfe() {
        let layout = ChannelLayout::standard(ChannelLayoutKind::Surround21);
        assert!(layout.contains(AudioChannel::LowFrequency));
        assert_eq!(layout.channel_count(), 3);
    }

    // 18. Custom layout index_of
    #[test]
    fn test_custom_index_of() {
        let layout = ChannelLayout::custom(vec![AudioChannel::SideLeft, AudioChannel::SideRight]);
        assert_eq!(layout.index_of(AudioChannel::SideRight), Some(1));
        assert_eq!(layout.index_of(AudioChannel::FrontLeft), None);
    }

    // 19. Binaural is stereo wire format
    #[test]
    fn test_binaural_channel_count() {
        let layout = ChannelLayout::standard(ChannelLayoutKind::Binaural);
        assert_eq!(layout.channel_count(), 2);
        assert!(layout.contains(AudioChannel::FrontLeft));
        assert!(layout.contains(AudioChannel::FrontRight));
    }

    // 20. Quad layout
    #[test]
    fn test_quad_layout() {
        let layout = ChannelLayout::standard(ChannelLayoutKind::Quad);
        assert_eq!(layout.channel_count(), 4);
        assert!(layout.contains(AudioChannel::FrontLeft));
        assert!(layout.contains(AudioChannel::BackLeft));
        assert!(!layout.contains(AudioChannel::FrontCenter));
        assert!(!layout.has_lfe());
    }

    // 21. 4.0 layout has BackCenter
    #[test]
    fn test_surround40_layout() {
        let layout = ChannelLayout::standard(ChannelLayoutKind::Surround40);
        assert_eq!(layout.channel_count(), 4);
        assert!(layout.contains(AudioChannel::FrontCenter));
        assert!(layout.contains(AudioChannel::BackCenter));
        assert!(!layout.has_lfe());
    }

    // 22. 3.0 layout
    #[test]
    fn test_surround30_layout() {
        let layout = ChannelLayout::standard(ChannelLayoutKind::Surround30);
        assert_eq!(layout.channel_count(), 3);
        assert!(layout.contains(AudioChannel::FrontCenter));
        assert!(!layout.has_lfe());
    }

    // 23. Atmos 5.1.4 layout
    #[test]
    fn test_atmos514_layout() {
        let layout = ChannelLayout::standard(ChannelLayoutKind::Atmos514);
        assert_eq!(layout.channel_count(), 10);
        assert!(layout.has_lfe());
        assert!(layout.has_height_channels());
        assert_eq!(layout.height_channel_count(), 4);
        assert!(layout.contains(AudioChannel::TopFrontLeft));
        assert!(layout.contains(AudioChannel::TopFrontRight));
        assert!(layout.contains(AudioChannel::TopBackLeft));
        assert!(layout.contains(AudioChannel::TopBackRight));
    }

    // 24. Atmos 7.1.4 layout
    #[test]
    fn test_atmos714_layout() {
        let layout = ChannelLayout::standard(ChannelLayoutKind::Atmos714);
        assert_eq!(layout.channel_count(), 12);
        assert!(layout.has_lfe());
        assert!(layout.has_height_channels());
        assert_eq!(layout.height_channel_count(), 4);
        assert!(layout.contains(AudioChannel::SideLeft));
        assert!(layout.contains(AudioChannel::SideRight));
        assert!(layout.contains(AudioChannel::TopBackLeft));
        assert!(layout.contains(AudioChannel::TopBackRight));
    }

    // 25. Atmos index_of height channel
    #[test]
    fn test_atmos514_index_of_height() {
        let layout = ChannelLayout::standard(ChannelLayoutKind::Atmos514);
        // 5.1.4 order: FL FR FC LFE BL BR TFL TFR TBL TBR
        assert_eq!(layout.index_of(AudioChannel::TopFrontLeft), Some(6));
        assert_eq!(layout.index_of(AudioChannel::TopBackRight), Some(9));
    }

    // 26. layout name()
    #[test]
    fn test_layout_name() {
        assert_eq!(ChannelLayoutKind::Atmos514.name(), "5.1.4 Atmos");
        assert_eq!(ChannelLayoutKind::Atmos714.name(), "7.1.4 Atmos");
        assert_eq!(ChannelLayoutKind::Surround714.name(), "7.1.4");
        assert_eq!(ChannelLayoutKind::Surround916.name(), "9.1.6");
        assert_eq!(
            ChannelLayoutKind::DolbyAtmosBed9_1_6.name(),
            "9.1.6 Atmos Bed"
        );
        assert_eq!(ChannelLayoutKind::Surround71.name(), "7.1");
        assert_eq!(ChannelLayoutKind::Binaural.name(), "Binaural");
        assert_eq!(ChannelLayoutKind::Quad.name(), "Quad");
        assert_eq!(
            ChannelLayout::standard(ChannelLayoutKind::Stereo).name(),
            "Stereo"
        );
    }

    // 27. Atmos 7.1.4 bed count
    #[test]
    fn test_atmos714_bed_count() {
        assert_eq!(ChannelLayoutKind::Atmos714.bed_channel_count(), 8);
        assert_eq!(ChannelLayoutKind::Atmos514.bed_channel_count(), 6);
    }

    // 28. TopBackLeft/TopBackRight are surround and height
    #[test]
    fn test_top_back_channels_classification() {
        assert!(AudioChannel::TopBackLeft.is_surround());
        assert!(AudioChannel::TopBackRight.is_surround());
        assert!(AudioChannel::TopBackLeft.is_height_channel());
        assert!(AudioChannel::TopBackRight.is_height_channel());
    }

    #[test]
    fn test_surround916_layout() {
        let layout = ChannelLayout::standard(ChannelLayoutKind::Surround916);
        assert_eq!(layout.channel_count(), 16);
        assert!(layout.contains(AudioChannel::FrontWideLeft));
        assert!(layout.contains(AudioChannel::FrontWideRight));
        assert!(layout.contains(AudioChannel::TopMiddleLeft));
        assert!(layout.contains(AudioChannel::TopMiddleRight));
    }

    #[test]
    fn test_dolby_atmos_bed_9_1_6_layout() {
        let layout = ChannelLayout::standard(ChannelLayoutKind::DolbyAtmosBed9_1_6);
        assert_eq!(layout.channel_count(), 16);
        assert!(layout.has_height_channels());
        assert_eq!(layout.height_channel_count(), 6);
        assert_eq!(layout.name(), "9.1.6 Atmos Bed");
    }

    // 29. BackCenter is surround but not height
    #[test]
    fn test_back_center_classification() {
        assert!(AudioChannel::BackCenter.is_surround());
        assert!(!AudioChannel::BackCenter.is_height_channel());
    }

    // 30. ChannelLayout::has_height_channels (custom)
    #[test]
    fn test_custom_has_height_channels() {
        let layout = ChannelLayout::custom(vec![
            AudioChannel::FrontLeft,
            AudioChannel::FrontRight,
            AudioChannel::TopFrontLeft,
        ]);
        assert!(layout.has_height_channels());
        assert_eq!(layout.height_channel_count(), 1);

        let flat = ChannelLayout::custom(vec![AudioChannel::FrontLeft, AudioChannel::FrontRight]);
        assert!(!flat.has_height_channels());
    }
}
