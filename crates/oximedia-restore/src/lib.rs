//! Professional audio restoration tools for `OxiMedia`.
//!
//! `oximedia-restore` provides comprehensive audio restoration capabilities for
//! recovering and enhancing degraded audio recordings.
//!
//! # Features
//!
//! - **Click/Pop Removal** - Remove vinyl clicks and digital glitches
//! - **Hum Removal** - Remove 50Hz/60Hz hum and harmonics
//! - **Noise Reduction** - Spectral subtraction, gating, and Wiener filtering
//! - **Declipping** - Restore clipped audio peaks
//! - **Dehiss** - Remove tape hiss and background noise
//! - **Decrackle** - Remove crackle from old recordings
//! - **Azimuth Correction** - Correct tape azimuth errors
//! - **Wow/Flutter Removal** - Remove tape speed variations
//! - **DC Offset Removal** - Remove DC bias
//! - **Phase Correction** - Correct phase issues
//! - **Breath Removal** - Detect and attenuate breaths in podcast/voiceover
//! - **Loudness Normalisation** - EBU R128 / ITU-R BS.1770-4 integrated loudness
//!
//! # Example
//!
//! ```
//! use oximedia_restore::presets::VinylRestoration;
//! use oximedia_restore::RestoreChain;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a restoration chain for vinyl
//! let mut chain = RestoreChain::new();
//! chain.add_preset(VinylRestoration::default());
//!
//! // Process samples
//! let samples = vec![0.0; 44100]; // 1 second at 44.1 kHz
//! let restored = chain.process(&samples, 44100)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Restoration Presets
//!
//! Pre-configured restoration chains for common scenarios:
//!
//! - **Vinyl Restoration** - Click removal, decrackle, hum removal
//! - **Tape Restoration** - Azimuth, wow/flutter, hiss removal
//! - **Broadcast Cleanup** - Declipping, noise reduction, DC removal
//! - **Archival** - Full restoration chain for preservation

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod azimuth;
pub mod breath_removal;
pub mod click;
pub mod clip;
pub mod crackle;
pub mod dc;
pub mod declip;
pub mod error;
pub mod flutter_repair;
pub mod hiss;
pub mod hum;
pub mod loudness_normalization;
pub mod noise;
pub mod phase;
pub mod room_correction;
pub mod utils;
pub mod wow;

pub mod audio_restoration_report;
pub mod banding_reduce;
pub mod batch;
pub mod bit_depth;
pub mod color_bleed;
pub mod color_restore;
pub mod deband;
pub mod declicker;
pub mod deflicker;
pub mod dropout_fix;
pub mod dynamic_eq;
pub mod film_grain;
pub mod frame_interp;
pub mod grain_add;
pub mod grain_restore;
pub mod harmonic_enhance;
pub mod harmonic_reconstruct;
pub mod noise_profile_match;
pub mod overlap_add;
pub mod pitch_correct;
pub mod pitch_correction;
pub mod presets;
pub mod restore_plan;
pub mod restore_report;
pub mod restore_undo;
pub mod reverb_reduction;
pub mod scan_line;
pub mod spectral_repair;
pub mod spectral_sub;
pub mod stereo_field_repair;
pub mod stereo_width;
pub mod tape_dropout_repair;
pub mod tape_speed_correct;
pub mod telecine_detect;
pub mod time_stretcher;
pub mod transient_repair;
pub mod upscale;
pub mod vintage;
pub mod vinyl_surface_noise;

// Re-exports
pub use error::{RestoreError, RestoreResult};

use azimuth::AzimuthCorrector;
use breath_removal::BreathRemover;
use click::{ClickDetector, ClickRemover};
use clip::{BasicDeclipper, ClipDetector};
use crackle::{CrackleDetector, CrackleRemover};
use dc::DcRemover;
use hiss::HissRemover;
use hum::HumRemover;
use loudness_normalization::LoudnessNormalizer;
use noise::{NoiseGate, SpectralSubtraction, WienerFilter};
use phase::PhaseCorrector;
use wow::WowFlutterCorrector;

// ---------------------------------------------------------------------------
// Step ordering constants — used by the validation logic
// ---------------------------------------------------------------------------

/// Priority bucket assigned to each restoration step variant.
///
/// Steps with a *lower* priority number must come *before* steps with a higher
/// number.  Steps in the same bucket may appear in any order relative to each
/// other.  The validation only enforces the logical ordering constraints
/// documented in the EBU/SMPTE recommended processing chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum StepPriority {
    /// Must be first — removes DC bias that would corrupt all subsequent analysis.
    DcRemoval = 0,
    /// Clipping repair depends on knowing the signal envelope, which DC distorts.
    Declipping = 1,
    /// Azimuth errors affect stereo alignment; fix before processing individual channels.
    AzimuthCorrection = 2,
    /// Wow/flutter correction must precede spectral analysis.
    SpeedCorrection = 3,
    /// Impulsive noise (clicks, crackle) — remove before broadband reduction.
    ImpulsiveNoise = 4,
    /// Harmonic noise (hum) — narrowband, remove before broadband.
    HarmonicNoise = 5,
    /// Broadband noise reduction (Wiener, spectral subtraction, gate, hiss).
    BroadbandNoise = 6,
    /// Breath removal — works on what remains after noise reduction.
    BreathRemoval = 7,
    /// Loudness normalisation — must be last so the target level is preserved.
    LoudnessNorm = 8,
    /// Stereo field / phase — always last for stereo-pair operations.
    PhaseCorrection = 9,
}

fn step_priority(step: &RestorationStep) -> StepPriority {
    match step {
        RestorationStep::DcRemoval(_) => StepPriority::DcRemoval,
        RestorationStep::Declipping { .. } => StepPriority::Declipping,
        RestorationStep::AzimuthCorrection(_) => StepPriority::AzimuthCorrection,
        RestorationStep::WowFlutterCorrection(_) => StepPriority::SpeedCorrection,
        RestorationStep::ClickRemoval { .. } => StepPriority::ImpulsiveNoise,
        RestorationStep::CrackleRemoval { .. } => StepPriority::ImpulsiveNoise,
        RestorationStep::HumRemoval(_) => StepPriority::HarmonicNoise,
        RestorationStep::NoiseReduction(_) => StepPriority::BroadbandNoise,
        RestorationStep::WienerFilter(_) => StepPriority::BroadbandNoise,
        RestorationStep::NoiseGate(_) => StepPriority::BroadbandNoise,
        RestorationStep::HissRemoval(_) => StepPriority::BroadbandNoise,
        RestorationStep::BreathRemoval(_) => StepPriority::BreathRemoval,
        RestorationStep::LoudnessNormalization(_) => StepPriority::LoudnessNorm,
        RestorationStep::PhaseCorrection(_) => StepPriority::PhaseCorrection,
    }
}

// ---------------------------------------------------------------------------
// Restoration step
// ---------------------------------------------------------------------------

/// Restoration step in a processing chain.
#[derive(Debug)]
pub enum RestorationStep {
    /// Remove DC offset.
    DcRemoval(DcRemover),
    /// Detect and remove clicks/pops.
    ClickRemoval {
        /// Click detector.
        detector: ClickDetector,
        /// Click remover.
        remover: ClickRemover,
    },
    /// Remove hum and harmonics.
    HumRemoval(HumRemover),
    /// Spectral noise reduction.
    NoiseReduction(SpectralSubtraction),
    /// Wiener filtering.
    WienerFilter(WienerFilter),
    /// Noise gate.
    NoiseGate(NoiseGate),
    /// Declipping.
    Declipping {
        /// Clipping detector.
        detector: ClipDetector,
        /// Declipper.
        declipper: BasicDeclipper,
    },
    /// Hiss removal.
    HissRemoval(HissRemover),
    /// Crackle removal.
    CrackleRemoval {
        /// Crackle detector.
        detector: CrackleDetector,
        /// Crackle remover.
        remover: CrackleRemover,
    },
    /// Azimuth correction (stereo only).
    AzimuthCorrection(AzimuthCorrector),
    /// Wow/flutter correction.
    WowFlutterCorrection(WowFlutterCorrector),
    /// Phase correction (stereo only).
    PhaseCorrection(PhaseCorrector),
    /// Breath removal (podcast/voiceover).
    BreathRemoval(BreathRemover),
    /// EBU R128 loudness normalisation.
    LoudnessNormalization(LoudnessNormalizer),
}

// ---------------------------------------------------------------------------
// Wrapped step with bypass toggle
// ---------------------------------------------------------------------------

/// A restoration step together with an enabled/disabled toggle.
///
/// Disabled steps are preserved in the chain but skipped during processing,
/// making it easy to A/B compare with and without a particular step.
#[derive(Debug)]
pub struct ChainStep {
    /// The processing step.
    pub step: RestorationStep,
    /// When `false` the step is skipped (bypassed) without being removed.
    pub enabled: bool,
}

impl ChainStep {
    /// Create an enabled step.
    pub fn enabled(step: RestorationStep) -> Self {
        Self {
            step,
            enabled: true,
        }
    }

    /// Create a disabled (bypassed) step.
    pub fn disabled(step: RestorationStep) -> Self {
        Self {
            step,
            enabled: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Multichannel layout
// ---------------------------------------------------------------------------

/// Surround channel assignment for multichannel processing.
///
/// Channels are stored in the conventional order used by both the ITU-R BS.775
/// standard (5.1) and the Dolby/SMPTE 7.1 extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurroundChannel {
    /// Front left.
    FrontLeft = 0,
    /// Front right.
    FrontRight = 1,
    /// Centre.
    Centre = 2,
    /// LFE / subwoofer.
    Lfe = 3,
    /// Surround left (5.1) / rear-side left (7.1).
    SurroundLeft = 4,
    /// Surround right (5.1) / rear-side right (7.1).
    SurroundRight = 5,
    /// Rear left (7.1 only).
    RearLeft = 6,
    /// Rear right (7.1 only).
    RearRight = 7,
}

/// Layout descriptor for a multichannel buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultichannelLayout {
    /// 5.1 surround (6 channels): L, R, C, LFE, Ls, Rs.
    Surround51,
    /// 7.1 surround (8 channels): L, R, C, LFE, Ls, Rs, Lrs, Rrs.
    Surround71,
}

impl MultichannelLayout {
    /// Return the number of channels in this layout.
    #[must_use]
    pub fn channel_count(self) -> usize {
        match self {
            Self::Surround51 => 6,
            Self::Surround71 => 8,
        }
    }

    /// Return the channel assignment for a given index.
    #[must_use]
    pub fn channel_at(self, idx: usize) -> Option<SurroundChannel> {
        use SurroundChannel::{
            Centre, FrontLeft, FrontRight, Lfe, RearLeft, RearRight, SurroundLeft, SurroundRight,
        };
        let map_51 = [
            FrontLeft,
            FrontRight,
            Centre,
            Lfe,
            SurroundLeft,
            SurroundRight,
        ];
        let map_71 = [
            FrontLeft,
            FrontRight,
            Centre,
            Lfe,
            SurroundLeft,
            SurroundRight,
            RearLeft,
            RearRight,
        ];
        match self {
            Self::Surround51 => map_51.get(idx).copied(),
            Self::Surround71 => map_71.get(idx).copied(),
        }
    }

    /// Returns `true` if the channel at `idx` is the LFE channel.
    #[must_use]
    pub fn is_lfe(self, idx: usize) -> bool {
        self.channel_at(idx) == Some(SurroundChannel::Lfe)
    }
}

// ---------------------------------------------------------------------------
// Step ordering validation errors
// ---------------------------------------------------------------------------

/// Details of a step-ordering violation detected by [`RestoreChain::validate_order`].
#[derive(Debug, Clone)]
pub struct OrderingViolation {
    /// Zero-based index of the *earlier* (out-of-order) step.
    pub earlier_index: usize,
    /// Zero-based index of the *later* step that must precede the earlier one.
    pub later_index: usize,
    /// Human-readable description of why the ordering is problematic.
    pub description: String,
}

// ---------------------------------------------------------------------------
// RestoreChain
// ---------------------------------------------------------------------------

/// Audio restoration processing chain.
///
/// # Per-step bypass
///
/// Each step is stored as a [`ChainStep`] with an `enabled` flag.  Use
/// [`RestoreChain::set_step_enabled`] to toggle individual steps without
/// removing them from the chain.
///
/// # Step ordering validation
///
/// Call [`RestoreChain::validate_order`] to check for logical ordering
/// violations.  The chain will still process even if violations exist, but
/// quality may be reduced.
#[derive(Debug)]
pub struct RestoreChain {
    steps: Vec<ChainStep>,
}

impl RestoreChain {
    /// Create a new empty restoration chain.
    #[must_use]
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    // -----------------------------------------------------------------------
    // Step management
    // -----------------------------------------------------------------------

    /// Add a restoration step to the chain (enabled by default).
    pub fn add_step(&mut self, step: RestorationStep) {
        self.steps.push(ChainStep::enabled(step));
    }

    /// Add a restoration step in a disabled (bypassed) state.
    pub fn add_step_disabled(&mut self, step: RestorationStep) {
        self.steps.push(ChainStep::disabled(step));
    }

    /// Add a preset to the chain (all steps enabled).
    pub fn add_preset(&mut self, preset: impl Into<Vec<RestorationStep>>) {
        for step in preset.into() {
            self.steps.push(ChainStep::enabled(step));
        }
    }

    /// Enable or disable the step at `index`.
    ///
    /// # Errors
    ///
    /// Returns [`RestoreError::InvalidParameter`] if `index` is out of bounds.
    pub fn set_step_enabled(&mut self, index: usize, enabled: bool) -> RestoreResult<()> {
        match self.steps.get_mut(index) {
            Some(s) => {
                s.enabled = enabled;
                Ok(())
            }
            None => Err(RestoreError::InvalidParameter(format!(
                "step index {index} out of range (chain has {} steps)",
                self.steps.len()
            ))),
        }
    }

    /// Returns `true` if the step at `index` is enabled.
    ///
    /// Returns `None` if `index` is out of bounds.
    #[must_use]
    pub fn is_step_enabled(&self, index: usize) -> Option<bool> {
        self.steps.get(index).map(|s| s.enabled)
    }

    // -----------------------------------------------------------------------
    // Ordering validation
    // -----------------------------------------------------------------------

    /// Validate that the step ordering follows recommended signal-chain logic.
    ///
    /// Returns a (possibly empty) list of [`OrderingViolation`]s.  The chain
    /// can still be executed with violations present, but audio quality may
    /// suffer.
    ///
    /// Only **enabled** steps are checked — disabled (bypassed) steps do not
    /// affect the logical ordering of the active processing path.
    #[must_use]
    pub fn validate_order(&self) -> Vec<OrderingViolation> {
        let enabled: Vec<(usize, &ChainStep)> = self
            .steps
            .iter()
            .enumerate()
            .filter(|(_, s)| s.enabled)
            .collect();

        let mut violations = Vec::new();

        // Scan every pair (i, j) where i < j.  If priority[i] > priority[j]
        // the earlier step should have come after the later step.
        for (a, (ai, as_)) in enabled.iter().enumerate() {
            for (bi, bs) in enabled.iter().skip(a + 1) {
                let pa = step_priority(&as_.step);
                let pb = step_priority(&bs.step);
                if pa > pb {
                    violations.push(OrderingViolation {
                        earlier_index: *ai,
                        later_index: *bi,
                        description: format!(
                            "Step {ai} ({pa:?}) should come after step {bi} ({pb:?})"
                        ),
                    });
                }
            }
        }

        violations
    }

    // -----------------------------------------------------------------------
    // Mono processing
    // -----------------------------------------------------------------------

    /// Process mono audio samples through all enabled steps.
    pub fn process(&mut self, samples: &[f32], sample_rate: u32) -> RestoreResult<Vec<f32>> {
        let mut output = samples.to_vec();

        for chain_step in &mut self.steps {
            if !chain_step.enabled {
                continue;
            }
            output = Self::process_step_mono(&mut chain_step.step, &output, sample_rate)?;
        }

        Ok(output)
    }

    fn process_step_mono(
        step: &mut RestorationStep,
        samples: &[f32],
        sample_rate: u32,
    ) -> RestoreResult<Vec<f32>> {
        match step {
            RestorationStep::DcRemoval(remover) => remover.process(samples),
            RestorationStep::ClickRemoval { detector, remover } => {
                let clicks = detector.detect(samples)?;
                remover.remove(samples, &clicks)
            }
            RestorationStep::HumRemoval(remover) => remover.process(samples),
            RestorationStep::NoiseReduction(reducer) => reducer.process(samples),
            RestorationStep::WienerFilter(filter) => filter.process(samples),
            RestorationStep::NoiseGate(gate) => gate.process(samples),
            RestorationStep::Declipping {
                detector,
                declipper,
            } => {
                let regions = detector.detect(samples)?;
                declipper.restore(samples, &regions)
            }
            RestorationStep::HissRemoval(remover) => remover.process(samples, sample_rate),
            RestorationStep::CrackleRemoval { detector, remover } => {
                let crackles = detector.detect(samples)?;
                remover.remove(samples, &crackles)
            }
            RestorationStep::WowFlutterCorrection(corrector) => corrector.correct(samples),
            RestorationStep::BreathRemoval(remover) => {
                let mut buf = samples.to_vec();
                remover.process(&mut buf, sample_rate)?;
                Ok(buf)
            }
            RestorationStep::LoudnessNormalization(normalizer) => normalizer.process(samples),
            // Stereo-only steps are skipped for mono
            RestorationStep::AzimuthCorrection(_) | RestorationStep::PhaseCorrection(_) => {
                Ok(samples.to_vec())
            }
        }
    }

    // -----------------------------------------------------------------------
    // Stereo processing
    // -----------------------------------------------------------------------

    /// Process stereo audio samples through all enabled steps.
    pub fn process_stereo(
        &mut self,
        left: &[f32],
        right: &[f32],
        sample_rate: u32,
    ) -> RestoreResult<(Vec<f32>, Vec<f32>)> {
        let mut out_left = left.to_vec();
        let mut out_right = right.to_vec();

        for chain_step in &mut self.steps {
            if !chain_step.enabled {
                continue;
            }
            (out_left, out_right) =
                Self::process_step_stereo(&mut chain_step.step, out_left, out_right, sample_rate)?;
        }

        Ok((out_left, out_right))
    }

    fn process_step_stereo(
        step: &mut RestorationStep,
        mut left: Vec<f32>,
        mut right: Vec<f32>,
        sample_rate: u32,
    ) -> RestoreResult<(Vec<f32>, Vec<f32>)> {
        match step {
            RestorationStep::DcRemoval(remover) => {
                left = remover.process(&left)?;
                let mut remover_r = remover.clone();
                right = remover_r.process(&right)?;
            }
            RestorationStep::ClickRemoval { detector, remover } => {
                let clicks_l = detector.detect(&left)?;
                left = remover.remove(&left, &clicks_l)?;

                let clicks_r = detector.detect(&right)?;
                right = remover.remove(&right, &clicks_r)?;
            }
            RestorationStep::HumRemoval(remover) => {
                left = remover.process(&left)?;
                let mut remover_r = remover.clone();
                right = remover_r.process(&right)?;
            }
            RestorationStep::NoiseReduction(reducer) => {
                left = reducer.process(&left)?;
                right = reducer.process(&right)?;
            }
            RestorationStep::WienerFilter(filter) => {
                left = filter.process(&left)?;
                right = filter.process(&right)?;
            }
            RestorationStep::NoiseGate(gate) => {
                left = gate.process(&left)?;
                let mut gate_r = gate.clone();
                right = gate_r.process(&right)?;
            }
            RestorationStep::Declipping {
                detector,
                declipper,
            } => {
                let regions_l = detector.detect(&left)?;
                left = declipper.restore(&left, &regions_l)?;

                let regions_r = detector.detect(&right)?;
                right = declipper.restore(&right, &regions_r)?;
            }
            RestorationStep::HissRemoval(remover) => {
                left = remover.process(&left, sample_rate)?;
                right = remover.process(&right, sample_rate)?;
            }
            RestorationStep::CrackleRemoval { detector, remover } => {
                let crackles_l = detector.detect(&left)?;
                left = remover.remove(&left, &crackles_l)?;

                let crackles_r = detector.detect(&right)?;
                right = remover.remove(&right, &crackles_r)?;
            }
            RestorationStep::AzimuthCorrection(corrector) => {
                (left, right) = corrector.correct(&left, &right)?;
            }
            RestorationStep::WowFlutterCorrection(corrector) => {
                left = corrector.correct(&left)?;
                right = corrector.correct(&right)?;
            }
            RestorationStep::PhaseCorrection(corrector) => {
                (left, right) = corrector.correct(&left, &right)?;
            }
            RestorationStep::BreathRemoval(remover) => {
                remover.process(&mut left, sample_rate)?;
                remover.process(&mut right, sample_rate)?;
            }
            RestorationStep::LoudnessNormalization(normalizer) => {
                left = normalizer.process(&left)?;
                right = normalizer.process(&right)?;
            }
        }
        Ok((left, right))
    }

    // -----------------------------------------------------------------------
    // Multichannel processing
    // -----------------------------------------------------------------------

    /// Process surround sound audio (5.1 or 7.1) through all enabled steps.
    ///
    /// `channels` is a slice of per-channel sample buffers.  The length of
    /// `channels` must match the channel count of `layout`
    /// (6 for [`MultichannelLayout::Surround51`], 8 for
    /// [`MultichannelLayout::Surround71`]).
    ///
    /// Processing rules:
    /// - The **LFE** channel bypasses all spectral-analysis steps (hiss, noise
    ///   reduction, Wiener, hum) since these operate in the full-bandwidth domain
    ///   and would corrupt the sub-bass channel.
    /// - **Stereo-pair steps** (azimuth, phase) are applied to the front L/R
    ///   pair only.
    /// - All other steps run independently on each channel.
    ///
    /// # Errors
    ///
    /// Returns [`RestoreError::InvalidParameter`] when `channels.len()` does not
    /// match the expected channel count for `layout`.
    pub fn process_multichannel(
        &mut self,
        channels: Vec<Vec<f32>>,
        layout: MultichannelLayout,
        sample_rate: u32,
    ) -> RestoreResult<Vec<Vec<f32>>> {
        let expected = layout.channel_count();
        if channels.len() != expected {
            return Err(RestoreError::InvalidParameter(format!(
                "expected {expected} channels for {layout:?}, got {}",
                channels.len()
            )));
        }

        // Verify all channels have the same length
        let first_len = channels.first().map_or(0, Vec::len);
        for (i, ch) in channels.iter().enumerate() {
            if ch.len() != first_len {
                return Err(RestoreError::InvalidParameter(format!(
                    "channel {i} has {} samples, expected {first_len}",
                    ch.len()
                )));
            }
        }

        let mut out_channels: Vec<Vec<f32>> = channels;

        for chain_step in &mut self.steps {
            if !chain_step.enabled {
                continue;
            }
            out_channels = Self::process_step_multichannel(
                &mut chain_step.step,
                out_channels,
                &layout,
                sample_rate,
            )?;
        }

        Ok(out_channels)
    }

    fn process_step_multichannel(
        step: &mut RestorationStep,
        mut channels: Vec<Vec<f32>>,
        layout: &MultichannelLayout,
        sample_rate: u32,
    ) -> RestoreResult<Vec<Vec<f32>>> {
        // Identify LFE index and front L/R indices for this layout
        let lfe_idx = (0..layout.channel_count()).find(|&i| layout.is_lfe(i));
        let front_l = 0_usize; // FrontLeft is always index 0
        let front_r = 1_usize; // FrontRight is always index 1

        /// Returns true when the step should skip the LFE channel.
        fn skip_lfe_for_step(s: &RestorationStep) -> bool {
            matches!(
                s,
                RestorationStep::HissRemoval(_)
                    | RestorationStep::NoiseReduction(_)
                    | RestorationStep::WienerFilter(_)
                    | RestorationStep::HumRemoval(_)
                    | RestorationStep::NoiseGate(_)
            )
        }

        match step {
            // Stereo-pair steps: apply to front L/R only
            RestorationStep::AzimuthCorrection(corrector) => {
                let left = channels[front_l].clone();
                let right = channels[front_r].clone();
                let (new_l, new_r) = corrector.correct(&left, &right)?;
                channels[front_l] = new_l;
                channels[front_r] = new_r;
            }
            RestorationStep::PhaseCorrection(corrector) => {
                let left = channels[front_l].clone();
                let right = channels[front_r].clone();
                let (new_l, new_r) = corrector.correct(&left, &right)?;
                channels[front_l] = new_l;
                channels[front_r] = new_r;
            }
            // All other steps: apply per-channel (optionally skipping LFE)
            other => {
                let skip_lfe = skip_lfe_for_step(other);

                for ch_idx in 0..channels.len() {
                    if skip_lfe && Some(ch_idx) == lfe_idx {
                        continue;
                    }

                    // We need a temporary borrow; clone the channel buffer,
                    // process it, then write back.
                    let ch_buf = channels[ch_idx].clone();
                    let processed = Self::process_step_mono(other, &ch_buf, sample_rate)?;
                    channels[ch_idx] = processed;
                }
            }
        }

        Ok(channels)
    }

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------

    /// Clear all steps from the chain.
    pub fn clear(&mut self) {
        self.steps.clear();
    }

    /// Get total number of steps in the chain (including disabled steps).
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if chain is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Count of currently enabled (active) steps.
    #[must_use]
    pub fn enabled_count(&self) -> usize {
        self.steps.iter().filter(|s| s.enabled).count()
    }
}

impl Default for RestoreChain {
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

    fn make_dc_step() -> RestorationStep {
        RestorationStep::DcRemoval(DcRemover::new(10.0, 44100))
    }

    // ------ Basic chain operations ------------------------------------------

    #[test]
    fn test_restore_chain() {
        let mut chain = RestoreChain::new();
        assert!(chain.is_empty());

        chain.add_step(make_dc_step());
        assert_eq!(chain.len(), 1);

        let samples = vec![0.5; 1000];
        let result = chain
            .process(&samples, 44100)
            .expect("should succeed in test");
        assert_eq!(result.len(), samples.len());
    }

    #[test]
    fn test_stereo_processing() {
        let mut chain = RestoreChain::new();
        chain.add_step(make_dc_step());

        let left = vec![0.5; 1000];
        let right = vec![0.5; 1000];

        let (out_l, out_r) = chain
            .process_stereo(&left, &right, 44100)
            .expect("should succeed in test");
        assert_eq!(out_l.len(), left.len());
        assert_eq!(out_r.len(), right.len());
    }

    #[test]
    fn test_clear() {
        let mut chain = RestoreChain::new();
        chain.add_step(make_dc_step());
        assert!(!chain.is_empty());

        chain.clear();
        assert!(chain.is_empty());
    }

    // ------ Per-step bypass toggle ------------------------------------------

    #[test]
    fn test_step_bypass_toggle_disabled_skips_processing() {
        let mut chain = RestoreChain::new();
        chain.add_step(RestorationStep::DcRemoval(DcRemover::new(10.0, 44100)));

        // Disable the step — output should equal the input identically.
        chain
            .set_step_enabled(0, false)
            .expect("index should be valid");
        assert_eq!(chain.is_step_enabled(0), Some(false));

        let samples: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let output = chain.process(&samples, 44100).expect("process ok");
        assert_eq!(
            output, samples,
            "bypassed step should leave samples unchanged"
        );
    }

    #[test]
    fn test_step_bypass_toggle_re_enable() {
        let mut chain = RestoreChain::new();
        chain.add_step(make_dc_step());
        chain.set_step_enabled(0, false).expect("valid");
        chain.set_step_enabled(0, true).expect("valid");
        assert_eq!(chain.is_step_enabled(0), Some(true));
    }

    #[test]
    fn test_set_step_enabled_out_of_range() {
        let mut chain = RestoreChain::new();
        let result = chain.set_step_enabled(99, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_step_enabled_out_of_range() {
        let chain = RestoreChain::new();
        assert_eq!(chain.is_step_enabled(0), None);
    }

    #[test]
    fn test_enabled_count() {
        let mut chain = RestoreChain::new();
        chain.add_step(make_dc_step());
        chain.add_step(make_dc_step());
        assert_eq!(chain.enabled_count(), 2);

        chain.set_step_enabled(0, false).expect("valid");
        assert_eq!(chain.enabled_count(), 1);
    }

    #[test]
    fn test_add_step_disabled() {
        let mut chain = RestoreChain::new();
        chain.add_step_disabled(make_dc_step());
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.enabled_count(), 0);
        assert_eq!(chain.is_step_enabled(0), Some(false));
    }

    // ------ Step ordering validation ----------------------------------------

    #[test]
    fn test_validate_order_correct_chain() {
        let mut chain = RestoreChain::new();
        // Correct order: DC → Declipping → Hum → Noise → Loudness
        chain.add_step(RestorationStep::DcRemoval(DcRemover::new(10.0, 44100)));
        chain.add_step(RestorationStep::Declipping {
            detector: clip::ClipDetector::new(clip::ClipDetectorConfig::default()),
            declipper: clip::BasicDeclipper::new(clip::DeclipConfig::default()),
        });
        chain.add_step(RestorationStep::HumRemoval(HumRemover::new_standard(
            50.0, 44100, 3, 10.0,
        )));
        chain.add_step(RestorationStep::LoudnessNormalization(
            LoudnessNormalizer::new(
                loudness_normalization::LoudnessNormalizerConfig::default(),
                44100,
            ),
        ));

        let violations = chain.validate_order();
        assert!(
            violations.is_empty(),
            "correctly ordered chain should have no violations; got: {violations:?}"
        );
    }

    #[test]
    fn test_validate_order_detects_violation() {
        let mut chain = RestoreChain::new();
        // Wrong order: Loudness before DC removal
        chain.add_step(RestorationStep::LoudnessNormalization(
            LoudnessNormalizer::new(
                loudness_normalization::LoudnessNormalizerConfig::default(),
                44100,
            ),
        ));
        chain.add_step(RestorationStep::DcRemoval(DcRemover::new(10.0, 44100)));

        let violations = chain.validate_order();
        assert!(!violations.is_empty(), "should detect out-of-order steps");
    }

    #[test]
    fn test_validate_order_bypassed_steps_ignored() {
        let mut chain = RestoreChain::new();
        // Add loudness first but disabled, then DC removal enabled — no violation
        chain.add_step(RestorationStep::LoudnessNormalization(
            LoudnessNormalizer::new(
                loudness_normalization::LoudnessNormalizerConfig::default(),
                44100,
            ),
        ));
        chain.set_step_enabled(0, false).expect("valid");
        chain.add_step(RestorationStep::DcRemoval(DcRemover::new(10.0, 44100)));

        let violations = chain.validate_order();
        assert!(
            violations.is_empty(),
            "disabled steps should be excluded from ordering check"
        );
    }

    // ------ Multichannel processing -----------------------------------------

    #[test]
    fn test_process_multichannel_51_wrong_channel_count() {
        let mut chain = RestoreChain::new();
        chain.add_step(make_dc_step());

        // Provide 5 channels instead of 6
        let bad_channels: Vec<Vec<f32>> = (0..5).map(|_| vec![0.0f32; 256]).collect();
        let result =
            chain.process_multichannel(bad_channels, MultichannelLayout::Surround51, 48000);
        assert!(result.is_err(), "should fail with wrong channel count");
    }

    #[test]
    fn test_process_multichannel_51_dc_removal() {
        let mut chain = RestoreChain::new();
        chain.add_step(RestorationStep::DcRemoval(DcRemover::new(10.0, 48000)));

        let n = 512_usize;
        let channels: Vec<Vec<f32>> = (0..6).map(|ch| vec![0.1 * ch as f32; n]).collect();

        let output = chain
            .process_multichannel(channels, MultichannelLayout::Surround51, 48000)
            .expect("5.1 DC removal should succeed");

        assert_eq!(output.len(), 6, "should still have 6 channels");
        for ch in &output {
            assert_eq!(ch.len(), n, "each channel should have {n} samples");
        }
    }

    #[test]
    fn test_process_multichannel_71() {
        let mut chain = RestoreChain::new();
        chain.add_step(RestorationStep::DcRemoval(DcRemover::new(10.0, 48000)));

        let n = 512_usize;
        let channels: Vec<Vec<f32>> = (0..8).map(|_| vec![0.0f32; n]).collect();

        let output = chain
            .process_multichannel(channels, MultichannelLayout::Surround71, 48000)
            .expect("7.1 processing should succeed");

        assert_eq!(output.len(), 8);
        for ch in &output {
            assert_eq!(ch.len(), n);
        }
    }

    #[test]
    fn test_process_multichannel_mismatched_channel_lengths() {
        let mut chain = RestoreChain::new();
        chain.add_step(make_dc_step());

        let mut channels: Vec<Vec<f32>> = (0..6).map(|_| vec![0.0f32; 256]).collect();
        channels[2] = vec![0.0f32; 128]; // different length

        let result = chain.process_multichannel(channels, MultichannelLayout::Surround51, 48000);
        assert!(result.is_err(), "mismatched channel lengths should fail");
    }

    #[test]
    fn test_multichannel_layout_lfe_detection() {
        let layout = MultichannelLayout::Surround51;
        // In 5.1, channel 3 is LFE
        assert!(layout.is_lfe(3));
        assert!(!layout.is_lfe(0));
        assert!(!layout.is_lfe(5));
    }

    #[test]
    fn test_multichannel_layout_channel_count() {
        assert_eq!(MultichannelLayout::Surround51.channel_count(), 6);
        assert_eq!(MultichannelLayout::Surround71.channel_count(), 8);
    }

    // ------ New step types --------------------------------------------------

    #[test]
    fn test_breath_removal_step() {
        let mut chain = RestoreChain::new();
        chain.add_step(RestorationStep::BreathRemoval(BreathRemover::default()));

        let samples = vec![0.1f32; 4096];
        let output = chain.process(&samples, 44100).expect("should succeed");
        assert_eq!(output.len(), samples.len());
    }

    #[test]
    fn test_loudness_normalization_step() {
        let mut chain = RestoreChain::new();
        chain.add_step(RestorationStep::LoudnessNormalization(
            LoudnessNormalizer::new(
                loudness_normalization::LoudnessNormalizerConfig::default(),
                44100,
            ),
        ));

        let samples = vec![0.5f32; 44100 * 2]; // 2 seconds
        let output = chain.process(&samples, 44100).expect("should succeed");
        assert_eq!(output.len(), samples.len());
    }
}
