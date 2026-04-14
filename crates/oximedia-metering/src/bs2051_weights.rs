//! ITU-R BS.2051 channel weights for immersive audio layouts.
//!
//! Implements NHK 22.2 channel weights per ITU-R BS.2051 for immersive audio
//! loudness measurement.  The 22.2 layout contains 24 channels: 9 top-layer,
//! 10 mid-layer, 3 bottom-layer, and 2 LFE channels.
//!
//! Per BS.2051 the loudness contribution of each channel is weighted according
//! to its positional layer:
//!
//! | Layer     | Weight |
//! |-----------|--------|
//! | Top       | 0.707  |
//! | Mid       | 1.000  |
//! | Bottom    | 0.707  |
//! | LFE       | 0.000  |

/// Channel layer classification for BS.2051 / NHK 22.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bs2051ChannelGroup {
    /// Top-layer speaker (above ear height).
    TopLayer,
    /// Mid-layer speaker (ear height / horizontal plane).
    MidLayer,
    /// Bottom-layer speaker (below ear height).
    BottomLayer,
    /// Low-frequency effects channel (not included in loudness sum).
    Lfe,
}

impl Bs2051ChannelGroup {
    /// Return the loudness weight for this channel group as specified in
    /// ITU-R BS.2051.
    ///
    /// * `TopLayer`    → 0.707 (≈ −3 dB, cosine of ≈ 45° elevation)
    /// * `MidLayer`    → 1.000 (reference level)
    /// * `BottomLayer` → 0.707 (≈ −3 dB)
    /// * `Lfe`         → 0.000 (excluded from loudness measurement)
    #[must_use]
    pub fn weight(self) -> f32 {
        match self {
            Self::TopLayer => 0.707,
            Self::MidLayer => 1.000,
            Self::BottomLayer => 0.707,
            Self::Lfe => 0.000,
        }
    }
}

/// BS.2051 channel weight table for a given loudspeaker layout.
///
/// Each entry maps a channel index to its [`Bs2051ChannelGroup`] (and
/// therefore its loudness weight).
#[derive(Clone, Debug)]
pub struct Bs2051Weights {
    /// Per-channel group assignments (indexed 0..N).
    pub channel_groups: Vec<Bs2051ChannelGroup>,
}

impl Bs2051Weights {
    /// Create a weight table for the NHK 22.2 loudspeaker layout.
    ///
    /// Channel ordering follows ITU-R BS.2051-3, Table 1:
    ///
    /// **Top layer (9 channels, indices 0–8):**
    /// TpFC, TpFL, TpFR, TpSiL, TpSiR, TpBL, TpBR, TpBC, TpC
    /// Weight: 0.707
    ///
    /// **Mid layer (10 channels, indices 9–18):**
    /// FC, FL, FR, FLc, FRc, BL, BR, BC, SiL, SiR
    /// Weight: 1.000
    ///
    /// **Bottom layer (3 channels, indices 19–21):**
    /// BtFL, BtFR, BtFC
    /// Weight: 0.707
    ///
    /// **LFE (2 channels, indices 22–23):**
    /// LFE1, LFE2
    /// Weight: 0.000
    #[must_use]
    pub fn nhk_22_2() -> Self {
        let mut groups = Vec::with_capacity(24);

        // 9 top-layer channels
        for _ in 0..9 {
            groups.push(Bs2051ChannelGroup::TopLayer);
        }
        // 10 mid-layer channels
        for _ in 0..10 {
            groups.push(Bs2051ChannelGroup::MidLayer);
        }
        // 3 bottom-layer channels
        for _ in 0..3 {
            groups.push(Bs2051ChannelGroup::BottomLayer);
        }
        // 2 LFE channels
        for _ in 0..2 {
            groups.push(Bs2051ChannelGroup::Lfe);
        }

        Self {
            channel_groups: groups,
        }
    }

    /// Create a custom weight table from an explicit list of channel groups.
    #[must_use]
    pub fn from_groups(channel_groups: Vec<Bs2051ChannelGroup>) -> Self {
        Self { channel_groups }
    }

    /// Return the weight for a specific channel index.
    ///
    /// Returns `None` if `channel` is out of range.
    #[must_use]
    pub fn weight_for(&self, channel: usize) -> Option<f32> {
        self.channel_groups.get(channel).map(|g| g.weight())
    }

    /// Number of channels in this layout.
    #[must_use]
    pub fn num_channels(&self) -> usize {
        self.channel_groups.len()
    }
}

/// Compute an approximate integrated LKFS value for a single mono channel
/// using the RMS of its samples.
///
/// This is a simplified, gate-free RMS-to-LKFS conversion used internally
/// by [`compute_integrated_loudness_bs2051`].  For broadcast-grade gated
/// measurements use the full `LoudnessMeter` pipeline.
///
/// Returns `f32::NEG_INFINITY` if the channel is silent.
fn rms_to_lkfs(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mean_sq: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
    if mean_sq <= 0.0 {
        f32::NEG_INFINITY
    } else {
        // LKFS = -0.691 + 10 * log10(mean_sq)  (ITU-R BS.1770 simplified)
        -0.691 + 10.0 * mean_sq.log10()
    }
}

/// Compute a BS.2051-weighted integrated loudness across all channels.
///
/// Each channel's contribution is weighted by its layer weight before
/// summation.  The final result is expressed in LKFS (dB relative to
/// full-scale).
///
/// # Arguments
///
/// * `channels` - Slice of per-channel sample vectors.  The number of channel
///   vectors must match `layout.num_channels()`; extra channels are ignored
///   and missing channels are treated as silent.
/// * `layout` - [`Bs2051Weights`] describing the channel layout and per-group
///   weights.
///
/// # Returns
///
/// Weighted integrated loudness in LKFS.
/// Returns `f32::NEG_INFINITY` if all channels are silent or the layout has
/// no channels.
#[must_use]
pub fn compute_integrated_loudness_bs2051(
    channels: &[Vec<f32>],
    layout: &Bs2051Weights,
) -> f32 {
    if layout.num_channels() == 0 {
        return f32::NEG_INFINITY;
    }

    let mut weighted_sum = 0.0_f32;
    let mut weight_total = 0.0_f32;

    for (ch_idx, group) in layout.channel_groups.iter().enumerate() {
        let weight = group.weight();
        if weight == 0.0 {
            // LFE channels are excluded from loudness measurement.
            continue;
        }

        // Retrieve per-channel mean-square directly; treat missing channels as silent.
        let mean_sq = if let Some(samples) = channels.get(ch_idx) {
            if samples.is_empty() {
                0.0_f32
            } else {
                samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32
            }
        } else {
            0.0_f32
        };

        if mean_sq > 0.0 {
            // Accumulate weighted mean-square power per BS.1770-4 eq. (1): Σ Gᵢ·⟨zᵢ²⟩
            weighted_sum += weight * mean_sq;
            weight_total = weight_total.max(weight); // track that at least one channel contributed
        }
    }

    if weighted_sum <= 0.0 || weight_total == 0.0 {
        return f32::NEG_INFINITY;
    }

    // Per ITU-R BS.1770-4 eq. (1): Lk = -0.691 + 10·log10(Σ Gᵢ·⟨z²ᵢ⟩)
    -0.691 + 10.0 * weighted_sum.log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Bs2051ChannelGroup ────────────────────────────────────────────────────

    #[test]
    fn test_channel_group_weights() {
        assert!((Bs2051ChannelGroup::TopLayer.weight() - 0.707).abs() < 1e-6);
        assert!((Bs2051ChannelGroup::MidLayer.weight() - 1.000).abs() < 1e-6);
        assert!((Bs2051ChannelGroup::BottomLayer.weight() - 0.707).abs() < 1e-6);
        assert_eq!(Bs2051ChannelGroup::Lfe.weight(), 0.0);
    }

    // ── Bs2051Weights::nhk_22_2 ───────────────────────────────────────────────

    #[test]
    fn test_nhk_22_2_channel_count() {
        let weights = Bs2051Weights::nhk_22_2();
        assert_eq!(weights.num_channels(), 24);
    }

    #[test]
    fn test_nhk_22_2_top_layer_channels() {
        let weights = Bs2051Weights::nhk_22_2();
        // Channels 0–8 are top-layer (9 channels).
        for i in 0..9 {
            assert_eq!(
                weights.channel_groups[i],
                Bs2051ChannelGroup::TopLayer,
                "channel {i} should be TopLayer"
            );
        }
    }

    #[test]
    fn test_nhk_22_2_mid_layer_channels() {
        let weights = Bs2051Weights::nhk_22_2();
        // Channels 9–18 are mid-layer (10 channels).
        for i in 9..19 {
            assert_eq!(
                weights.channel_groups[i],
                Bs2051ChannelGroup::MidLayer,
                "channel {i} should be MidLayer"
            );
        }
    }

    #[test]
    fn test_nhk_22_2_bottom_layer_channels() {
        let weights = Bs2051Weights::nhk_22_2();
        // Channels 19–21 are bottom-layer (3 channels).
        for i in 19..22 {
            assert_eq!(
                weights.channel_groups[i],
                Bs2051ChannelGroup::BottomLayer,
                "channel {i} should be BottomLayer"
            );
        }
    }

    #[test]
    fn test_nhk_22_2_lfe_channels() {
        let weights = Bs2051Weights::nhk_22_2();
        // Channels 22–23 are LFE (2 channels).
        assert_eq!(weights.channel_groups[22], Bs2051ChannelGroup::Lfe);
        assert_eq!(weights.channel_groups[23], Bs2051ChannelGroup::Lfe);
    }

    #[test]
    fn test_weight_for_out_of_range() {
        let weights = Bs2051Weights::nhk_22_2();
        assert!(weights.weight_for(24).is_none());
        assert!(weights.weight_for(100).is_none());
    }

    #[test]
    fn test_weight_for_in_range() {
        let weights = Bs2051Weights::nhk_22_2();
        // Channel 0 → TopLayer → 0.707
        let w = weights.weight_for(0).expect("channel 0 should exist");
        assert!((w - 0.707).abs() < 1e-6);
        // Channel 9 → MidLayer → 1.0
        let w9 = weights.weight_for(9).expect("channel 9 should exist");
        assert!((w9 - 1.0).abs() < 1e-6);
    }

    // ── compute_integrated_loudness_bs2051 ────────────────────────────────────

    #[test]
    fn test_loudness_empty_layout() {
        let layout = Bs2051Weights::from_groups(vec![]);
        let result = compute_integrated_loudness_bs2051(&[], &layout);
        assert_eq!(result, f32::NEG_INFINITY);
    }

    #[test]
    fn test_loudness_silent_channels() {
        let layout = Bs2051Weights::nhk_22_2();
        let silent: Vec<Vec<f32>> = (0..24).map(|_| vec![0.0f32; 1000]).collect();
        let result = compute_integrated_loudness_bs2051(&silent, &layout);
        assert_eq!(result, f32::NEG_INFINITY);
    }

    #[test]
    fn test_loudness_mid_layer_only_matches_rms_lkfs() {
        // Single mid-layer channel at known RMS level should produce ~that LKFS.
        let layout = Bs2051Weights::from_groups(vec![Bs2051ChannelGroup::MidLayer]);
        // Sine at amplitude 0.5: RMS = 0.5 / sqrt(2)
        let samples: Vec<f32> = (0..48000)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin())
            .collect();
        let expected_lkfs = rms_to_lkfs(&samples);
        let result = compute_integrated_loudness_bs2051(&[samples], &layout);
        assert!(
            (result - expected_lkfs).abs() < 0.5,
            "Expected ≈ {expected_lkfs:.2} LKFS, got {result:.2}"
        );
    }

    #[test]
    fn test_lfe_channels_excluded() {
        // An LFE-only layout should produce NEG_INFINITY even with loud signals.
        let layout = Bs2051Weights::from_groups(vec![
            Bs2051ChannelGroup::Lfe,
            Bs2051ChannelGroup::Lfe,
        ]);
        let loud: Vec<Vec<f32>> = (0..2).map(|_| vec![1.0f32; 1000]).collect();
        let result = compute_integrated_loudness_bs2051(&loud, &layout);
        assert_eq!(result, f32::NEG_INFINITY);
    }

    #[test]
    fn test_loudness_finite_for_non_silent_signal() {
        let layout = Bs2051Weights::nhk_22_2();
        // Put a non-silent signal in every non-LFE channel.
        let mut channels: Vec<Vec<f32>> = vec![vec![0.0f32; 100]; 24];
        for (i, ch) in channels.iter_mut().enumerate() {
            if i < 22 {
                // Non-LFE
                for s in ch.iter_mut() {
                    *s = 0.3;
                }
            }
        }
        let result = compute_integrated_loudness_bs2051(&channels, &layout);
        assert!(result.is_finite(), "Expected finite LKFS, got {result}");
    }

    #[test]
    fn test_loudness_top_weighted_less_than_mid() {
        // Pure top-layer signal should be softer than same RMS on mid-layer
        // because top weight (0.707) < mid weight (1.0).
        let top_layout =
            Bs2051Weights::from_groups(vec![Bs2051ChannelGroup::TopLayer]);
        let mid_layout =
            Bs2051Weights::from_groups(vec![Bs2051ChannelGroup::MidLayer]);

        let signal = vec![0.5f32; 48000];
        let top_loudness =
            compute_integrated_loudness_bs2051(&[signal.clone()], &top_layout);
        let mid_loudness =
            compute_integrated_loudness_bs2051(&[signal], &mid_layout);

        assert!(
            top_loudness < mid_loudness,
            "Top ({top_loudness:.2}) should be quieter than mid ({mid_loudness:.2})"
        );
    }
}
