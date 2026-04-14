//! Cue monitoring — PFL/AFL cue bus, cue mix level, headphone output routing.
//!
//! In a professional mixing console a *cue* (or "pre-fader listen / PFL" /
//! "after-fader listen / AFL") bus lets the engineer listen to individual
//! channels without affecting the main mix.  This module provides:
//!
//! - A `CueBus` that accumulates stereo audio from cued sources.
//! - `CueMode` selection per source (PFL, AFL, or disabled).
//! - A `CueMixLevel` to control the monitor / headphone output volume.
//! - `HeadphoneRoute` to select between the cue bus and the main mix.
//! - A `CueMonitor` that ties all the above together.
//!
//! # Signal Flow
//!
//! ```text
//! Channel strip:
//!   ┌── Pre-fader tap  ──► CueBus (PFL mode)
//!   └── Post-fader tap ──► CueBus (AFL mode)
//!                              │
//!                           CueMix level
//!                              │
//!                         Headphone out
//! ```
//!
//! # Example
//!
//! ```rust
//! use oximedia_mixer::cue_monitor::{CueMonitor, CueMonitorConfig, CueMode};
//!
//! let mut cue = CueMonitor::new(CueMonitorConfig {
//!     sample_rate: 48000,
//!     ..Default::default()
//! });
//! cue.enable_cue("vocals".to_string(), CueMode::Pfl);
//! let output = cue.process_block(
//!     &[("vocals".to_string(), (vec![0.5; 64], vec![0.5; 64]))],
//!     64,
//! );
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ── CueMode ──────────────────────────────────────────────────────────────────

/// Determines at which point in the signal chain the cue tap is taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CueMode {
    /// Pre-Fader Listen — tap taken before the channel fader.
    #[default]
    Pfl,
    /// After-Fader Listen — tap taken after the channel fader.
    Afl,
}

impl std::fmt::Display for CueMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CueMode::Pfl => write!(f, "PFL"),
            CueMode::Afl => write!(f, "AFL"),
        }
    }
}

// ── HeadphoneRoute ───────────────────────────────────────────────────────────

/// Selects what is routed to the headphone / monitor output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum HeadphoneRoute {
    /// Route the cue bus to headphones.
    #[default]
    CueBus,
    /// Route the main mix output to headphones.
    MainMix,
    /// Route a blend of cue bus + main mix.
    Blend,
}

impl std::fmt::Display for HeadphoneRoute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HeadphoneRoute::CueBus => write!(f, "CueBus"),
            HeadphoneRoute::MainMix => write!(f, "MainMix"),
            HeadphoneRoute::Blend => write!(f, "Blend"),
        }
    }
}

// ── CueMonitorConfig ─────────────────────────────────────────────────────────

/// Configuration for a `CueMonitor`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CueMonitorConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Linear gain for the cue bus output (0.0 – 4.0).
    pub cue_level: f32,
    /// Where headphone output comes from.
    pub headphone_route: HeadphoneRoute,
    /// Blend ratio between cue and main mix when `HeadphoneRoute::Blend`.
    /// 0.0 = full main, 1.0 = full cue.
    pub blend_ratio: f32,
}

impl Default for CueMonitorConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            cue_level: 1.0,
            headphone_route: HeadphoneRoute::CueBus,
            blend_ratio: 0.5,
        }
    }
}

// ── CueSource ────────────────────────────────────────────────────────────────

/// Cue source entry — a named source (channel or bus) with its cue mode.
#[derive(Debug, Clone)]
struct CueSource {
    name: String,
    mode: CueMode,
}

// ── CueBusMetrics ────────────────────────────────────────────────────────────

/// Metering data from the most recent process block.
#[derive(Debug, Clone, Default)]
pub struct CueBusMetrics {
    /// Peak absolute level on the left channel (linear).
    pub peak_left: f32,
    /// Peak absolute level on the right channel (linear).
    pub peak_right: f32,
    /// Whether any source is currently cued.
    pub any_active: bool,
}

// ── CueMonitor ───────────────────────────────────────────────────────────────

/// Cue monitor — manages cue sources, accumulates the cue bus, and provides
/// headphone output routing.
pub struct CueMonitor {
    config: CueMonitorConfig,
    /// Active cue sources indexed by source name.
    sources: HashMap<String, CueSource>,
    /// Insertion order for deterministic iteration.
    order: Vec<String>,
    /// Metrics from the last process block.
    metrics: CueBusMetrics,
}

impl CueMonitor {
    /// Create a new `CueMonitor` with the given configuration.
    pub fn new(config: CueMonitorConfig) -> Self {
        Self {
            config,
            sources: HashMap::new(),
            order: Vec::new(),
            metrics: CueBusMetrics::default(),
        }
    }

    /// Return the current configuration.
    pub fn config(&self) -> &CueMonitorConfig {
        &self.config
    }

    // ── Source management ─────────────────────────────────────────────────────

    /// Enable cue monitoring for a named source with the given mode.
    pub fn enable_cue(&mut self, name: String, mode: CueMode) {
        if !self.sources.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.sources.insert(name.clone(), CueSource { name, mode });
    }

    /// Disable cue monitoring for a named source.
    pub fn disable_cue(&mut self, name: &str) {
        self.sources.remove(name);
        self.order.retain(|n| n != name);
    }

    /// Returns true if the named source is currently cued.
    pub fn is_cued(&self, name: &str) -> bool {
        self.sources.contains_key(name)
    }

    /// Returns the `CueMode` for a source, or `None` if not cued.
    pub fn cue_mode(&self, name: &str) -> Option<CueMode> {
        self.sources.get(name).map(|s| s.mode)
    }

    /// Returns the number of active cue sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    // ── Level and routing control ─────────────────────────────────────────────

    /// Set the cue bus output level (linear gain, clamped to 0.0 – 4.0).
    pub fn set_cue_level(&mut self, level: f32) {
        self.config.cue_level = level.clamp(0.0, 4.0);
    }

    /// Get the current cue bus output level.
    pub fn cue_level(&self) -> f32 {
        self.config.cue_level
    }

    /// Set the headphone routing.
    pub fn set_headphone_route(&mut self, route: HeadphoneRoute) {
        self.config.headphone_route = route;
    }

    /// Set the blend ratio when `HeadphoneRoute::Blend` is active.
    /// 0.0 = full main, 1.0 = full cue.
    pub fn set_blend_ratio(&mut self, ratio: f32) {
        self.config.blend_ratio = ratio.clamp(0.0, 1.0);
    }

    // ── Processing ────────────────────────────────────────────────────────────

    /// Process one block of audio.
    ///
    /// `sources` maps source names to their stereo buffers
    /// `(left_samples, right_samples)`.  For PFL sources the caller should
    /// provide the pre-fader buffer; for AFL sources the post-fader buffer.
    ///
    /// `main_mix` is the stereo output of the main bus, used when
    /// `HeadphoneRoute` is `MainMix` or `Blend`.
    ///
    /// Returns the stereo headphone output buffer `(left, right)`.
    pub fn process_block(
        &mut self,
        sources: &[(String, (Vec<f32>, Vec<f32>))],
        num_frames: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        self.process_with_main(sources, None, num_frames)
    }

    /// Like `process_block` but also accepts a main mix buffer for blend/main
    /// headphone routing.
    pub fn process_with_main(
        &mut self,
        sources: &[(String, (Vec<f32>, Vec<f32>))],
        main_mix: Option<&(Vec<f32>, Vec<f32>)>,
        num_frames: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        // Build a lookup from source name → buffer.
        let src_map: HashMap<&str, &(Vec<f32>, Vec<f32>)> =
            sources.iter().map(|(n, b)| (n.as_str(), b)).collect();

        // Accumulate cue bus.
        let mut cue_l = vec![0.0_f32; num_frames];
        let mut cue_r = vec![0.0_f32; num_frames];
        let mut any_active = false;

        for name in &self.order {
            if let Some(src) = self.sources.get(name) {
                if let Some(buf) = src_map.get(src.name.as_str()) {
                    any_active = true;
                    let ll = buf.0.len().min(num_frames);
                    let rl = buf.1.len().min(num_frames);
                    for i in 0..ll {
                        cue_l[i] += buf.0[i];
                    }
                    for i in 0..rl {
                        cue_r[i] += buf.1[i];
                    }
                }
            }
        }

        // Apply cue level.
        let cue_gain = self.config.cue_level;
        if cue_gain != 1.0 {
            for s in cue_l.iter_mut() {
                *s *= cue_gain;
            }
            for s in cue_r.iter_mut() {
                *s *= cue_gain;
            }
        }

        // Update metering.
        self.metrics = CueBusMetrics {
            peak_left: cue_l.iter().copied().fold(0.0_f32, |a, x| a.max(x.abs())),
            peak_right: cue_r.iter().copied().fold(0.0_f32, |a, x| a.max(x.abs())),
            any_active,
        };

        // Produce headphone output.
        let (hp_l, hp_r) = match self.config.headphone_route {
            HeadphoneRoute::CueBus => (cue_l, cue_r),
            HeadphoneRoute::MainMix => {
                if let Some(main) = main_mix {
                    let l_len = main.0.len().min(num_frames);
                    let r_len = main.1.len().min(num_frames);
                    let mut ml = vec![0.0_f32; num_frames];
                    let mut mr = vec![0.0_f32; num_frames];
                    for i in 0..l_len {
                        ml[i] = main.0[i];
                    }
                    for i in 0..r_len {
                        mr[i] = main.1[i];
                    }
                    (ml, mr)
                } else {
                    (vec![0.0_f32; num_frames], vec![0.0_f32; num_frames])
                }
            }
            HeadphoneRoute::Blend => {
                let blend = self.config.blend_ratio.clamp(0.0, 1.0);
                let main_gain = 1.0 - blend;
                let mut bl = cue_l.clone();
                let mut br = cue_r.clone();
                if let Some(main) = main_mix {
                    let l_len = main.0.len().min(num_frames);
                    let r_len = main.1.len().min(num_frames);
                    for i in 0..l_len {
                        bl[i] = cue_l[i] * blend + main.0[i] * main_gain;
                    }
                    for i in 0..r_len {
                        br[i] = cue_r[i] * blend + main.1[i] * main_gain;
                    }
                } else {
                    // No main mix: scale cue by blend ratio only.
                    for s in bl.iter_mut() {
                        *s *= blend;
                    }
                    for s in br.iter_mut() {
                        *s *= blend;
                    }
                }
                (bl, br)
            }
        };

        (hp_l, hp_r)
    }

    /// Return the metering data from the most recent process block.
    pub fn metrics(&self) -> &CueBusMetrics {
        &self.metrics
    }

    /// Returns true if any source is currently cued.
    pub fn any_active(&self) -> bool {
        !self.sources.is_empty()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_buf(frames: usize, v: f32) -> (Vec<f32>, Vec<f32>) {
        (vec![v; frames], vec![v; frames])
    }

    #[test]
    fn test_enable_disable_cue() {
        let mut cue = CueMonitor::new(CueMonitorConfig::default());
        assert!(!cue.is_cued("ch1"));
        cue.enable_cue("ch1".into(), CueMode::Pfl);
        assert!(cue.is_cued("ch1"));
        assert_eq!(cue.source_count(), 1);
        cue.disable_cue("ch1");
        assert!(!cue.is_cued("ch1"));
        assert_eq!(cue.source_count(), 0);
    }

    #[test]
    fn test_cue_mode_pfl_afl() {
        let mut cue = CueMonitor::new(CueMonitorConfig::default());
        cue.enable_cue("ch1".into(), CueMode::Pfl);
        cue.enable_cue("ch2".into(), CueMode::Afl);
        assert_eq!(cue.cue_mode("ch1"), Some(CueMode::Pfl));
        assert_eq!(cue.cue_mode("ch2"), Some(CueMode::Afl));
        assert_eq!(cue.cue_mode("ch99"), None);
    }

    #[test]
    fn test_process_block_accumulates_sources() {
        let mut cue = CueMonitor::new(CueMonitorConfig::default());
        cue.enable_cue("ch1".into(), CueMode::Pfl);
        cue.enable_cue("ch2".into(), CueMode::Pfl);

        let srcs = vec![
            ("ch1".into(), flat_buf(64, 0.3)),
            ("ch2".into(), flat_buf(64, 0.4)),
        ];
        let (l, _) = cue.process_block(&srcs, 64);
        // 0.3 + 0.4 = 0.7
        assert!((l[0] - 0.7).abs() < 1e-5, "expected 0.7, got {}", l[0]);
    }

    #[test]
    fn test_cue_level_scales_output() {
        let mut cue = CueMonitor::new(CueMonitorConfig::default());
        cue.enable_cue("ch1".into(), CueMode::Afl);
        cue.set_cue_level(0.5);

        let srcs = vec![("ch1".into(), flat_buf(64, 1.0))];
        let (l, _) = cue.process_block(&srcs, 64);
        assert!((l[0] - 0.5).abs() < 1e-5, "expected 0.5, got {}", l[0]);
    }

    #[test]
    fn test_headphone_route_main_mix() {
        let mut cue = CueMonitor::new(CueMonitorConfig {
            headphone_route: HeadphoneRoute::MainMix,
            ..Default::default()
        });
        cue.enable_cue("ch1".into(), CueMode::Pfl);

        let srcs = vec![("ch1".into(), flat_buf(64, 0.9))];
        let main = flat_buf(64, 0.2);
        let (l, _) = cue.process_with_main(&srcs, Some(&main), 64);
        // Should output main mix value, not cue.
        assert!((l[0] - 0.2).abs() < 1e-5, "expected 0.2, got {}", l[0]);
    }

    #[test]
    fn test_headphone_route_blend() {
        let mut cue = CueMonitor::new(CueMonitorConfig {
            headphone_route: HeadphoneRoute::Blend,
            blend_ratio: 0.5,
            ..Default::default()
        });
        cue.enable_cue("ch1".into(), CueMode::Afl);

        let srcs = vec![("ch1".into(), flat_buf(64, 0.8))];
        let main = flat_buf(64, 0.4);
        let (l, _) = cue.process_with_main(&srcs, Some(&main), 64);
        // blend=0.5: 0.8*0.5 + 0.4*0.5 = 0.6
        assert!((l[0] - 0.6).abs() < 1e-4, "expected 0.6, got {}", l[0]);
    }

    #[test]
    fn test_metrics_updated_after_process() {
        let mut cue = CueMonitor::new(CueMonitorConfig::default());
        cue.enable_cue("ch1".into(), CueMode::Pfl);

        let srcs = vec![("ch1".into(), flat_buf(64, 0.6))];
        cue.process_block(&srcs, 64);
        let m = cue.metrics();
        assert!((m.peak_left - 0.6).abs() < 1e-5);
        assert!(m.any_active);
    }

    #[test]
    fn test_no_sources_produces_silence() {
        let mut cue = CueMonitor::new(CueMonitorConfig::default());
        let (l, r) = cue.process_block(&[], 64);
        assert!(l.iter().all(|&x| x == 0.0));
        assert!(r.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_enable_overwrites_mode() {
        let mut cue = CueMonitor::new(CueMonitorConfig::default());
        cue.enable_cue("ch1".into(), CueMode::Pfl);
        cue.enable_cue("ch1".into(), CueMode::Afl);
        assert_eq!(cue.cue_mode("ch1"), Some(CueMode::Afl));
        assert_eq!(cue.source_count(), 1);
    }

    #[test]
    fn test_disable_uncued_source_is_noop() {
        let mut cue = CueMonitor::new(CueMonitorConfig::default());
        // Disabling something that was never cued should not panic or error.
        cue.disable_cue("nonexistent");
        assert_eq!(cue.source_count(), 0);
    }
}
