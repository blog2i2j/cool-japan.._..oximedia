//! SIP, AFL, and PFL solo modes with signal routing.
//!
//! This module implements a [`SoloRouter`] that reroutes signal paths according
//! to three standard console solo modes:
//!
//! - **SIP (Solo-In-Place)**: Mutes all non-soloed channels in the main mix bus.
//!   Solo-safe channels are never muted.
//! - **AFL (After-Fader Listen)**: Taps the post-fader/pan signal of soloed
//!   channels and routes it to a dedicated monitor bus.  The main mix is
//!   unaffected.
//! - **PFL (Pre-Fader Listen)**: Taps the pre-fader signal of soloed channels
//!   and routes it to the monitor bus.  The main mix is unaffected.
//!
//! The [`SoloRouter`] maintains per-channel solo state, solo-safe flags, and
//! exposes methods to compute gain multipliers and generate monitor bus
//! contributions for each channel.

use std::collections::HashSet;

use crate::channel::ChannelId;

// ---------------------------------------------------------------------------
// SoloMode
// ---------------------------------------------------------------------------

/// Console solo mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SoloMode {
    /// Solo-In-Place: non-soloed (non-safe) channels are muted / dimmed.
    Sip,
    /// After-Fader Listen: post-fader signal routed to monitor bus.
    Afl,
    /// Pre-Fader Listen: pre-fader signal routed to monitor bus.
    Pfl,
}

// ---------------------------------------------------------------------------
// SoloRouter
// ---------------------------------------------------------------------------

/// Routes signals through the solo system.
///
/// The router tracks which channels are soloed and which are solo-safe, then
/// computes per-channel main-mix gain multipliers and monitor bus
/// contributions.
#[derive(Debug, Clone)]
pub struct SoloRouter {
    /// Active solo mode.
    mode: SoloMode,
    /// Channels currently in solo.
    soloed: HashSet<ChannelId>,
    /// Channels marked solo-safe (never muted in SIP mode).
    solo_safe: HashSet<ChannelId>,
    /// Dim level applied to non-soloed channels in SIP mode.
    /// 0.0 = full mute, 1.0 = no attenuation.
    sip_dim_db: f32,
    /// Linear gain derived from `sip_dim_db`.
    sip_dim_linear: f32,
    /// Monitor bus output gain (applied to AFL/PFL taps).
    monitor_gain: f32,
    /// Whether exclusive solo is enabled — soloing a channel clears all others.
    exclusive: bool,
}

impl SoloRouter {
    /// Create a new router with the given mode and full mute in SIP.
    #[must_use]
    pub fn new(mode: SoloMode) -> Self {
        Self {
            mode,
            soloed: HashSet::new(),
            solo_safe: HashSet::new(),
            sip_dim_db: f32::NEG_INFINITY,
            sip_dim_linear: 0.0,
            monitor_gain: 1.0,
            exclusive: false,
        }
    }

    // -- mode ---------------------------------------------------------------

    /// Set the solo mode.
    pub fn set_mode(&mut self, mode: SoloMode) {
        self.mode = mode;
    }

    /// Current solo mode.
    #[must_use]
    pub fn mode(&self) -> SoloMode {
        self.mode
    }

    // -- exclusive ----------------------------------------------------------

    /// Enable or disable exclusive solo.
    pub fn set_exclusive(&mut self, exclusive: bool) {
        self.exclusive = exclusive;
    }

    /// Whether exclusive solo is enabled.
    #[must_use]
    pub fn exclusive(&self) -> bool {
        self.exclusive
    }

    // -- solo state ---------------------------------------------------------

    /// Solo a channel.  In exclusive mode, all other solos are cleared first.
    pub fn solo(&mut self, channel_id: ChannelId) {
        if self.exclusive {
            self.soloed.clear();
        }
        self.soloed.insert(channel_id);
    }

    /// Unsolo a channel.
    pub fn unsolo(&mut self, channel_id: ChannelId) {
        self.soloed.remove(&channel_id);
    }

    /// Toggle solo state.
    pub fn toggle_solo(&mut self, channel_id: ChannelId) {
        if self.soloed.contains(&channel_id) {
            self.soloed.remove(&channel_id);
        } else {
            if self.exclusive {
                self.soloed.clear();
            }
            self.soloed.insert(channel_id);
        }
    }

    /// Clear all solos.
    pub fn clear_solos(&mut self) {
        self.soloed.clear();
    }

    /// Whether any channel is currently soloed.
    #[must_use]
    pub fn any_soloed(&self) -> bool {
        !self.soloed.is_empty()
    }

    /// Whether a specific channel is soloed.
    #[must_use]
    pub fn is_soloed(&self, channel_id: ChannelId) -> bool {
        self.soloed.contains(&channel_id)
    }

    /// Number of currently soloed channels.
    #[must_use]
    pub fn soloed_count(&self) -> usize {
        self.soloed.len()
    }

    /// Iterator over currently soloed channel IDs.
    pub fn soloed_channels(&self) -> impl Iterator<Item = &ChannelId> {
        self.soloed.iter()
    }

    // -- solo-safe ----------------------------------------------------------

    /// Mark or unmark a channel as solo-safe.
    pub fn set_solo_safe(&mut self, channel_id: ChannelId, safe: bool) {
        if safe {
            self.solo_safe.insert(channel_id);
        } else {
            self.solo_safe.remove(&channel_id);
        }
    }

    /// Whether a channel is solo-safe.
    #[must_use]
    pub fn is_solo_safe(&self, channel_id: ChannelId) -> bool {
        self.solo_safe.contains(&channel_id)
    }

    // -- SIP dim level ------------------------------------------------------

    /// Set SIP dim level in dB (0 dB = no attenuation, −∞ dB = full mute).
    ///
    /// The value is clamped to `−120.0..=0.0`.
    pub fn set_sip_dim_db(&mut self, db: f32) {
        self.sip_dim_db = db.clamp(-120.0, 0.0);
        self.sip_dim_linear = if self.sip_dim_db <= -120.0 {
            0.0
        } else {
            10.0_f32.powf(self.sip_dim_db / 20.0)
        };
    }

    /// Set SIP dim level as a linear gain (0.0 = full mute, 1.0 = no attenuation).
    pub fn set_sip_dim_linear(&mut self, gain: f32) {
        self.sip_dim_linear = gain.clamp(0.0, 1.0);
        self.sip_dim_db = if self.sip_dim_linear <= 1e-12 {
            f32::NEG_INFINITY
        } else {
            20.0 * self.sip_dim_linear.log10()
        };
    }

    /// Current SIP dim level as linear gain.
    #[must_use]
    pub fn sip_dim_linear(&self) -> f32 {
        self.sip_dim_linear
    }

    // -- monitor gain -------------------------------------------------------

    /// Set monitor bus output gain.
    pub fn set_monitor_gain(&mut self, gain: f32) {
        self.monitor_gain = gain.clamp(0.0, 4.0);
    }

    /// Current monitor bus output gain.
    #[must_use]
    pub fn monitor_gain(&self) -> f32 {
        self.monitor_gain
    }

    // -- gain computation ---------------------------------------------------

    /// Main-mix gain multiplier for a channel.
    ///
    /// | Condition            | SIP           | AFL / PFL |
    /// |----------------------|---------------|-----------|
    /// | No solos active      | 1.0           | 1.0       |
    /// | Channel is soloed    | 1.0           | 1.0       |
    /// | Channel is solo-safe | 1.0           | 1.0       |
    /// | Otherwise            | `sip_dim_lin` | 1.0       |
    #[must_use]
    pub fn main_mix_gain(&self, channel_id: ChannelId) -> f32 {
        if !self.any_soloed() {
            return 1.0;
        }
        match self.mode {
            SoloMode::Sip => {
                if self.soloed.contains(&channel_id) || self.solo_safe.contains(&channel_id) {
                    1.0
                } else {
                    self.sip_dim_linear
                }
            }
            SoloMode::Afl | SoloMode::Pfl => 1.0,
        }
    }

    /// Monitor bus gain for a channel.
    ///
    /// Returns `monitor_gain` for soloed channels in AFL/PFL mode, 0.0 otherwise.
    #[must_use]
    pub fn monitor_bus_gain(&self, channel_id: ChannelId) -> f32 {
        if !self.soloed.contains(&channel_id) {
            return 0.0;
        }
        match self.mode {
            SoloMode::Sip => 0.0,
            SoloMode::Afl | SoloMode::Pfl => self.monitor_gain,
        }
    }

    /// Whether the monitor bus should use the pre-fader signal.
    #[must_use]
    pub fn monitor_uses_pre_fader(&self) -> bool {
        self.mode == SoloMode::Pfl
    }

    // -- buffer processing --------------------------------------------------

    /// Route a channel through the solo system.
    ///
    /// # Arguments
    ///
    /// * `channel_id` — channel being processed.
    /// * `pre_fader` — mono pre-fader signal.
    /// * `post_fader_left` — post-fader left channel.
    /// * `post_fader_right` — post-fader right channel.
    ///
    /// # Returns
    ///
    /// A [`SoloRouteResult`] containing:
    /// * The main mix left/right buffers (possibly attenuated by SIP dim).
    /// * The monitor bus left/right contribution.
    #[must_use]
    pub fn route(
        &self,
        channel_id: ChannelId,
        pre_fader: &[f32],
        post_fader_left: &[f32],
        post_fader_right: &[f32],
    ) -> SoloRouteResult {
        let n = post_fader_left.len().min(post_fader_right.len());
        let main_gain = self.main_mix_gain(channel_id);
        let mon_gain = self.monitor_bus_gain(channel_id);

        // Main mix (with SIP dim if applicable).
        let main_left: Vec<f32> = post_fader_left.iter().take(n).map(|&s| s * main_gain).collect();
        let main_right: Vec<f32> = post_fader_right.iter().take(n).map(|&s| s * main_gain).collect();

        // Monitor bus.
        let (monitor_left, monitor_right) = if mon_gain.abs() < f32::EPSILON {
            (vec![0.0_f32; n], vec![0.0_f32; n])
        } else if self.monitor_uses_pre_fader() {
            // PFL: mono pre-fader to both channels.
            let scaled: Vec<f32> = pre_fader.iter().take(n).map(|&s| s * mon_gain).collect();
            (scaled.clone(), scaled)
        } else {
            // AFL: post-fader stereo.
            let l: Vec<f32> = post_fader_left.iter().take(n).map(|&s| s * mon_gain).collect();
            let r: Vec<f32> = post_fader_right.iter().take(n).map(|&s| s * mon_gain).collect();
            (l, r)
        };

        SoloRouteResult {
            main_left,
            main_right,
            monitor_left,
            monitor_right,
        }
    }

    /// Apply SIP muting in-place to a stereo pair.
    pub fn apply_sip_in_place(&self, channel_id: ChannelId, left: &mut [f32], right: &mut [f32]) {
        let gain = self.main_mix_gain(channel_id);
        if (gain - 1.0).abs() > f32::EPSILON {
            for s in left.iter_mut() {
                *s *= gain;
            }
            for s in right.iter_mut() {
                *s *= gain;
            }
        }
    }
}

impl Default for SoloRouter {
    fn default() -> Self {
        Self::new(SoloMode::Sip)
    }
}

// ---------------------------------------------------------------------------
// SoloRouteResult
// ---------------------------------------------------------------------------

/// Output from [`SoloRouter::route`].
#[derive(Debug, Clone)]
pub struct SoloRouteResult {
    /// Main mix left channel (possibly SIP-dimmed).
    pub main_left: Vec<f32>,
    /// Main mix right channel (possibly SIP-dimmed).
    pub main_right: Vec<f32>,
    /// Monitor bus left channel (AFL/PFL contribution).
    pub monitor_left: Vec<f32>,
    /// Monitor bus right channel (AFL/PFL contribution).
    pub monitor_right: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn ch() -> ChannelId {
        ChannelId(Uuid::new_v4())
    }

    // -- basic state --------------------------------------------------------

    #[test]
    fn test_default_mode_is_sip() {
        let router = SoloRouter::default();
        assert_eq!(router.mode(), SoloMode::Sip);
        assert!(!router.any_soloed());
    }

    #[test]
    fn test_solo_unsolo_toggle() {
        let mut router = SoloRouter::new(SoloMode::Sip);
        let c = ch();
        router.solo(c);
        assert!(router.is_soloed(c));
        assert_eq!(router.soloed_count(), 1);
        router.unsolo(c);
        assert!(!router.is_soloed(c));
        // toggle
        router.toggle_solo(c);
        assert!(router.is_soloed(c));
        router.toggle_solo(c);
        assert!(!router.is_soloed(c));
    }

    #[test]
    fn test_clear_solos() {
        let mut router = SoloRouter::new(SoloMode::Sip);
        router.solo(ch());
        router.solo(ch());
        router.clear_solos();
        assert!(!router.any_soloed());
    }

    // -- SIP ----------------------------------------------------------------

    #[test]
    fn test_sip_mutes_non_soloed() {
        let mut router = SoloRouter::new(SoloMode::Sip);
        let soloed = ch();
        let other = ch();
        router.solo(soloed);

        assert!((router.main_mix_gain(soloed) - 1.0).abs() < f32::EPSILON);
        assert!((router.main_mix_gain(other) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sip_dim_level() {
        let mut router = SoloRouter::new(SoloMode::Sip);
        router.set_sip_dim_linear(0.25);
        let soloed = ch();
        let other = ch();
        router.solo(soloed);

        assert!((router.main_mix_gain(soloed) - 1.0).abs() < f32::EPSILON);
        assert!((router.main_mix_gain(other) - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_sip_solo_safe_not_muted() {
        let mut router = SoloRouter::new(SoloMode::Sip);
        let soloed = ch();
        let safe = ch();
        let normal = ch();
        router.solo(soloed);
        router.set_solo_safe(safe, true);

        assert!((router.main_mix_gain(soloed) - 1.0).abs() < f32::EPSILON);
        assert!((router.main_mix_gain(safe) - 1.0).abs() < f32::EPSILON);
        assert!((router.main_mix_gain(normal) - 0.0).abs() < f32::EPSILON);
    }

    // -- AFL ----------------------------------------------------------------

    #[test]
    fn test_afl_does_not_affect_main_mix() {
        let mut router = SoloRouter::new(SoloMode::Afl);
        let soloed = ch();
        let other = ch();
        router.solo(soloed);

        assert!((router.main_mix_gain(soloed) - 1.0).abs() < f32::EPSILON);
        assert!((router.main_mix_gain(other) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_afl_monitor_uses_post_fader() {
        let mut router = SoloRouter::new(SoloMode::Afl);
        let c = ch();
        router.solo(c);

        assert!(!router.monitor_uses_pre_fader());
        assert!((router.monitor_bus_gain(c) - 1.0).abs() < f32::EPSILON);

        let pre = vec![0.9_f32; 16];
        let post_l = vec![0.5_f32; 16];
        let post_r = vec![0.3_f32; 16];
        let result = router.route(c, &pre, &post_l, &post_r);

        // Monitor should use post-fader values.
        for &m in &result.monitor_left {
            assert!((m - 0.5).abs() < 1e-5, "AFL monitor L should be post-fader");
        }
        for &m in &result.monitor_right {
            assert!((m - 0.3).abs() < 1e-5, "AFL monitor R should be post-fader");
        }
    }

    // -- PFL ----------------------------------------------------------------

    #[test]
    fn test_pfl_monitor_uses_pre_fader() {
        let mut router = SoloRouter::new(SoloMode::Pfl);
        let c = ch();
        router.solo(c);

        assert!(router.monitor_uses_pre_fader());
        assert!((router.monitor_bus_gain(c) - 1.0).abs() < f32::EPSILON);

        let pre = vec![0.8_f32; 16];
        let post_l = vec![0.4_f32; 16];
        let post_r = vec![0.2_f32; 16];
        let result = router.route(c, &pre, &post_l, &post_r);

        // Monitor should use pre-fader (mono to both L/R).
        for &m in &result.monitor_left {
            assert!((m - 0.8).abs() < 1e-5, "PFL monitor L should be pre-fader");
        }
        assert_eq!(
            result.monitor_left, result.monitor_right,
            "PFL monitor should be mono (L==R)"
        );
    }

    // -- no-solo passthrough ------------------------------------------------

    #[test]
    fn test_no_solo_unity_gain() {
        let router = SoloRouter::new(SoloMode::Sip);
        let c = ch();
        assert!((router.main_mix_gain(c) - 1.0).abs() < f32::EPSILON);
        assert!((router.monitor_bus_gain(c) - 0.0).abs() < f32::EPSILON);
    }

    // -- exclusive solo -----------------------------------------------------

    #[test]
    fn test_exclusive_solo_clears_others() {
        let mut router = SoloRouter::new(SoloMode::Sip);
        router.set_exclusive(true);
        let c1 = ch();
        let c2 = ch();
        router.solo(c1);
        assert!(router.is_soloed(c1));
        router.solo(c2);
        assert!(!router.is_soloed(c1), "exclusive should have cleared c1");
        assert!(router.is_soloed(c2));
        assert_eq!(router.soloed_count(), 1);
    }

    // -- route result structure ---------------------------------------------

    #[test]
    fn test_route_sip_dims_main() {
        let mut router = SoloRouter::new(SoloMode::Sip);
        let soloed = ch();
        let muted = ch();
        router.solo(soloed);

        let pre = vec![1.0_f32; 8];
        let post_l = vec![0.6_f32; 8];
        let post_r = vec![0.6_f32; 8];

        let result = router.route(muted, &pre, &post_l, &post_r);
        for &s in &result.main_left {
            assert!(s.abs() < f32::EPSILON, "SIP should mute non-soloed channel main");
        }
        for &s in &result.monitor_left {
            assert!(
                s.abs() < f32::EPSILON,
                "SIP should not produce monitor output"
            );
        }
    }

    // -- solo-safe toggle ---------------------------------------------------

    #[test]
    fn test_solo_safe_toggle() {
        let mut router = SoloRouter::new(SoloMode::Sip);
        let c = ch();
        router.set_solo_safe(c, true);
        assert!(router.is_solo_safe(c));
        router.set_solo_safe(c, false);
        assert!(!router.is_solo_safe(c));
    }

    // -- monitor gain -------------------------------------------------------

    #[test]
    fn test_custom_monitor_gain() {
        let mut router = SoloRouter::new(SoloMode::Afl);
        router.set_monitor_gain(0.5);
        let c = ch();
        router.solo(c);
        assert!((router.monitor_bus_gain(c) - 0.5).abs() < f32::EPSILON);
    }

    // -- sip dim dB ---------------------------------------------------------

    #[test]
    fn test_sip_dim_db_conversion() {
        let mut router = SoloRouter::new(SoloMode::Sip);
        // −6 dB ≈ 0.501
        router.set_sip_dim_db(-6.0);
        let lin = router.sip_dim_linear();
        assert!(
            (lin - 0.501).abs() < 0.01,
            "−6 dB should be ~0.501, got {lin}"
        );
    }

    // -- non-soloed no monitor ----------------------------------------------

    #[test]
    fn test_non_soloed_no_monitor_output() {
        let mut router = SoloRouter::new(SoloMode::Pfl);
        let soloed = ch();
        let other = ch();
        router.solo(soloed);

        let pre = vec![1.0_f32; 8];
        let post_l = vec![0.5_f32; 8];
        let post_r = vec![0.5_f32; 8];

        let result = router.route(other, &pre, &post_l, &post_r);
        for &s in &result.monitor_left {
            assert!(
                s.abs() < f32::EPSILON,
                "Non-soloed channel should produce no monitor output"
            );
        }
    }
}
