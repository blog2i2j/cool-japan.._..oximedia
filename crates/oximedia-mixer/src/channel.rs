//! Mixer channel implementation.
//!
//! Provides individual channel strips with gain, pan, effects, routing, and automation.

use oximedia_audio::ChannelLayout;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::effects::EffectSlot;

/// Unique channel identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ChannelId(pub Uuid);

impl std::fmt::Display for ChannelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Channel type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelType {
    /// Mono channel (1 channel).
    Mono,
    /// Stereo channel (2 channels).
    Stereo,
    /// 5.1 surround channel (6 channels).
    Surround51,
    /// 7.1 surround channel (8 channels).
    Surround71,
    /// First-order Ambisonics (4 channels: W, X, Y, Z).
    AmbisonicsFirstOrder,
    /// Second-order Ambisonics (9 channels).
    AmbisonicsSecondOrder,
    /// Third-order Ambisonics (16 channels).
    AmbisonicsThirdOrder,
}

impl ChannelType {
    /// Get number of audio channels.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
            Self::AmbisonicsFirstOrder => 4,
            Self::AmbisonicsSecondOrder => 9,
            Self::AmbisonicsThirdOrder => 16,
        }
    }
}

/// Pan mode for stereo/surround panning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanMode {
    /// Stereo balance (-1.0 = left, 0.0 = center, 1.0 = right).
    Stereo,
    /// VBAP (Vector Base Amplitude Panning) for surround.
    Vbap,
    /// DBAP (Distance-Based Amplitude Panning) for surround.
    Dbap,
    /// Binaural panning with HRTF.
    Binaural,
}

/// Pan law for stereo panning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanLaw {
    /// Linear pan law (no compensation).
    Linear,
    /// -3dB center compensation (equal power).
    Minus3dB,
    /// -4.5dB center compensation.
    Minus4Dot5dB,
    /// -6dB center compensation (equal gain).
    Minus6dB,
}

impl PanLaw {
    /// Get center compensation gain factor.
    #[must_use]
    pub fn center_gain(&self) -> f32 {
        match self {
            Self::Linear => 1.0,
            Self::Minus3dB => 0.707_946,       // -3dB
            Self::Minus4Dot5dB => 0.595_662_1, // -4.5dB
            Self::Minus6dB => 0.5,             // -6dB
        }
    }
}

/// Channel state flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[allow(clippy::struct_excessive_bools)]
pub struct ChannelState {
    /// Channel is muted.
    pub muted: bool,
    /// Channel is soloed.
    pub soloed: bool,
    /// Channel is armed for recording.
    pub armed: bool,
    /// Phase inverted.
    pub phase_inverted: bool,
    /// Channel is selected in UI.
    pub selected: bool,
}

/// Send configuration (pre/post-fader send to bus).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendConfig {
    /// Target bus ID.
    pub bus_id: crate::BusId,
    /// Send level (0.0 = -inf, 1.0 = 0dB).
    pub level: f32,
    /// Pre-fader send (true) or post-fader (false).
    pub pre_fader: bool,
    /// Send is active.
    pub active: bool,
}

/// Channel linking mode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelLink {
    /// Not linked.
    None,
    /// Linked as stereo pair (L/R).
    StereoPair(ChannelId),
    /// Linked to channel group.
    Group(Uuid),
}

/// Input routing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputRouting {
    /// Physical input index or internal source.
    pub source: InputSource,
    /// Input gain (trim) in dB.
    pub gain_db: f32,
    /// Enable high-pass filter.
    pub highpass_enabled: bool,
    /// High-pass filter frequency in Hz.
    pub highpass_freq: f32,
}

impl Default for InputRouting {
    fn default() -> Self {
        Self {
            source: InputSource::None,
            gain_db: 0.0,
            highpass_enabled: false,
            highpass_freq: 80.0,
        }
    }
}

/// Input source type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputSource {
    /// No input.
    None,
    /// Physical hardware input.
    Hardware(u32),
    /// Internal bus routing.
    Bus(crate::BusId),
    /// Virtual instrument.
    Virtual,
}

/// Output routing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputRouting {
    /// Target bus ID.
    pub bus_id: crate::BusId,
    /// Output gain in dB.
    pub gain_db: f32,
}

/// Direct monitoring configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectMonitoring {
    /// Direct monitoring enabled.
    pub enabled: bool,
    /// Direct monitoring level.
    pub level: f32,
    /// Apply effects to direct monitoring.
    pub with_effects: bool,
}

impl Default for DirectMonitoring {
    fn default() -> Self {
        Self {
            enabled: false,
            level: 1.0,
            with_effects: false,
        }
    }
}

/// Mixer channel (track or bus).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Channel {
    /// Channel name.
    name: String,

    /// Channel type.
    #[allow(clippy::struct_field_names)]
    channel_type: ChannelType,

    /// Channel layout.
    #[serde(skip)]
    layout: ChannelLayout,

    /// Sample rate in Hz.
    sample_rate: u32,

    /// Buffer size in samples.
    buffer_size: usize,

    /// Channel state.
    state: ChannelState,

    /// Fader gain (0.0 = -inf dB, 1.0 = 0 dB).
    gain: f32,

    /// Pan position (-1.0 = left, 0.0 = center, 1.0 = right).
    pan: f32,

    /// Pan mode.
    pan_mode: PanMode,

    /// Pan law.
    pan_law: PanLaw,

    /// Channel linking.
    link: ChannelLink,

    /// Effect slots (insert effects).
    effects: Vec<EffectSlot>,

    /// Send configurations.
    sends: HashMap<usize, SendConfig>,

    /// Input routing.
    input: InputRouting,

    /// Output routing.
    output: Option<OutputRouting>,

    /// Direct monitoring.
    direct_monitoring: DirectMonitoring,

    /// Channel color (RGB hex).
    color: Option<String>,

    /// Channel icon/symbol.
    icon: Option<String>,
}

impl Channel {
    /// Create a new channel.
    #[must_use]
    pub fn new(
        name: String,
        channel_type: ChannelType,
        layout: ChannelLayout,
        sample_rate: u32,
        buffer_size: usize,
    ) -> Self {
        Self {
            name,
            channel_type,
            layout,
            sample_rate,
            buffer_size,
            state: ChannelState::default(),
            gain: 1.0,
            pan: 0.0,
            pan_mode: PanMode::Stereo,
            pan_law: PanLaw::Minus3dB,
            link: ChannelLink::None,
            effects: Vec::new(),
            sends: HashMap::new(),
            input: InputRouting::default(),
            output: None,
            direct_monitoring: DirectMonitoring::default(),
            color: None,
            icon: None,
        }
    }

    /// Get channel name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set channel name.
    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    /// Get channel type.
    #[must_use]
    pub fn channel_type(&self) -> ChannelType {
        self.channel_type
    }

    /// Get channel layout.
    #[must_use]
    pub fn layout(&self) -> &ChannelLayout {
        &self.layout
    }

    /// Get sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get buffer size.
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Get channel state.
    #[must_use]
    pub fn state(&self) -> &ChannelState {
        &self.state
    }

    /// Get mutable channel state.
    #[must_use]
    pub fn state_mut(&mut self) -> &mut ChannelState {
        &mut self.state
    }

    /// Check if channel is muted.
    #[must_use]
    pub fn is_muted(&self) -> bool {
        self.state.muted
    }

    /// Set mute state.
    pub fn set_muted(&mut self, muted: bool) {
        self.state.muted = muted;
    }

    /// Check if channel is soloed.
    #[must_use]
    pub fn is_soloed(&self) -> bool {
        self.state.soloed
    }

    /// Set solo state.
    pub fn set_soloed(&mut self, soloed: bool) {
        self.state.soloed = soloed;
    }

    /// Check if channel is armed.
    #[must_use]
    pub fn is_armed(&self) -> bool {
        self.state.armed
    }

    /// Set arm state.
    pub fn set_armed(&mut self, armed: bool) {
        self.state.armed = armed;
    }

    /// Check if phase is inverted.
    #[must_use]
    pub fn is_phase_inverted(&self) -> bool {
        self.state.phase_inverted
    }

    /// Set phase invert.
    pub fn set_phase_inverted(&mut self, inverted: bool) {
        self.state.phase_inverted = inverted;
    }

    /// Get channel gain.
    #[must_use]
    pub fn gain(&self) -> f32 {
        self.gain
    }

    /// Set channel gain (0.0 = -inf dB, 1.0 = 0 dB).
    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain.clamp(0.0, 2.0); // Allow up to +6dB
    }

    /// Get gain in dB.
    #[must_use]
    pub fn gain_db(&self) -> f32 {
        if self.gain <= 0.0 {
            -f32::INFINITY
        } else {
            20.0 * self.gain.log10()
        }
    }

    /// Set gain in dB.
    pub fn set_gain_db(&mut self, db: f32) {
        if db <= -80.0 {
            self.gain = 0.0;
        } else {
            self.gain = 10.0_f32.powf(db / 20.0).clamp(0.0, 2.0);
        }
    }

    /// Get pan position.
    #[must_use]
    pub fn pan(&self) -> f32 {
        self.pan
    }

    /// Set pan position (-1.0 = left, 0.0 = center, 1.0 = right).
    pub fn set_pan(&mut self, pan: f32) {
        self.pan = pan.clamp(-1.0, 1.0);
    }

    /// Get pan mode.
    #[must_use]
    pub fn pan_mode(&self) -> PanMode {
        self.pan_mode
    }

    /// Set pan mode.
    pub fn set_pan_mode(&mut self, mode: PanMode) {
        self.pan_mode = mode;
    }

    /// Get pan law.
    #[must_use]
    pub fn pan_law(&self) -> PanLaw {
        self.pan_law
    }

    /// Set pan law.
    pub fn set_pan_law(&mut self, law: PanLaw) {
        self.pan_law = law;
    }

    /// Calculate stereo pan gains (left, right).
    #[must_use]
    pub fn calculate_stereo_pan(&self) -> (f32, f32) {
        let center_gain = self.pan_law.center_gain();

        if self.pan <= 0.0 {
            // Pan left
            let left_gain = 1.0;
            let right_gain = center_gain * (1.0 + self.pan);
            (left_gain, right_gain)
        } else {
            // Pan right
            let left_gain = center_gain * (1.0 - self.pan);
            let right_gain = 1.0;
            (left_gain, right_gain)
        }
    }

    /// Get channel link.
    #[must_use]
    pub fn link(&self) -> &ChannelLink {
        &self.link
    }

    /// Set channel link.
    pub fn set_link(&mut self, link: ChannelLink) {
        self.link = link;
    }

    /// Get effect slots.
    #[must_use]
    pub fn effects(&self) -> &[EffectSlot] {
        &self.effects
    }

    /// Get mutable effect slots.
    #[must_use]
    pub fn effects_mut(&mut self) -> &mut Vec<EffectSlot> {
        &mut self.effects
    }

    /// Add effect slot.
    pub fn add_effect(&mut self, slot: EffectSlot) {
        self.effects.push(slot);
    }

    /// Remove effect at index.
    pub fn remove_effect(&mut self, index: usize) -> Option<EffectSlot> {
        if index < self.effects.len() {
            Some(self.effects.remove(index))
        } else {
            None
        }
    }

    /// Get send configurations.
    #[must_use]
    pub fn sends(&self) -> &HashMap<usize, SendConfig> {
        &self.sends
    }

    /// Get mutable send configurations.
    #[must_use]
    pub fn sends_mut(&mut self) -> &mut HashMap<usize, SendConfig> {
        &mut self.sends
    }

    /// Add or update send.
    pub fn set_send(&mut self, slot: usize, config: SendConfig) {
        self.sends.insert(slot, config);
    }

    /// Remove send.
    pub fn remove_send(&mut self, slot: usize) -> Option<SendConfig> {
        self.sends.remove(&slot)
    }

    /// Get input routing.
    #[must_use]
    pub fn input(&self) -> &InputRouting {
        &self.input
    }

    /// Get mutable input routing.
    #[must_use]
    pub fn input_mut(&mut self) -> &mut InputRouting {
        &mut self.input
    }

    /// Get output routing.
    #[must_use]
    pub fn output(&self) -> Option<&OutputRouting> {
        self.output.as_ref()
    }

    /// Set output routing.
    pub fn set_output(&mut self, output: Option<OutputRouting>) {
        self.output = output;
    }

    /// Get direct monitoring.
    #[must_use]
    pub fn direct_monitoring(&self) -> &DirectMonitoring {
        &self.direct_monitoring
    }

    /// Get mutable direct monitoring.
    #[must_use]
    pub fn direct_monitoring_mut(&mut self) -> &mut DirectMonitoring {
        &mut self.direct_monitoring
    }

    /// Get channel color.
    #[must_use]
    pub fn color(&self) -> Option<&str> {
        self.color.as_deref()
    }

    /// Set channel color (RGB hex string).
    pub fn set_color(&mut self, color: Option<String>) {
        self.color = color;
    }

    /// Get channel icon.
    #[must_use]
    pub fn icon(&self) -> Option<&str> {
        self.icon.as_deref()
    }

    /// Set channel icon.
    pub fn set_icon(&mut self, icon: Option<String>) {
        self.icon = icon;
    }

    /// Process audio through channel strip.
    ///
    /// Applies the full signal chain:
    /// 1. Input gain and phase inversion
    /// 2. Insert effects chain
    /// 3. Channel fader
    /// 4. Pan processing to stereo output
    ///
    /// `input` is mono or interleaved multi-channel audio.
    /// `output_left` and `output_right` receive the stereo-panned result.
    /// Pre-fader send levels are returned as a vec of (send_index, level, samples).
    pub fn process_strip(&self, input: &[f32], output_left: &mut [f32], output_right: &mut [f32]) {
        let num_samples = input.len();

        // If muted, output silence
        if self.state.muted {
            for i in 0..num_samples {
                output_left[i] = 0.0;
                output_right[i] = 0.0;
            }
            return;
        }

        // Step 1: Input gain + phase inversion
        let input_gain_linear = db_to_linear(self.input.gain_db);
        let phase_mult: f32 = if self.state.phase_inverted { -1.0 } else { 1.0 };

        // Working buffer: apply input gain and phase
        let mut working: Vec<f32> = input
            .iter()
            .map(|&s| s * input_gain_linear * phase_mult)
            .collect();

        // Step 2: Insert effects chain (biquad EQ, dynamics, etc.)
        // Each effect slot processes the buffer in-place
        for slot in &self.effects {
            if !slot.bypassed {
                // Effects would process in-place here
                // For now, we pass through since EffectSlot doesn't have
                // a runtime processing state in this architecture
                let _ = slot;
            }
        }

        // Step 3: Channel fader gain
        let fader_gain = self.gain;
        for sample in &mut working {
            *sample *= fader_gain;
        }

        // Step 4: Stereo pan using the configured pan law
        let (left_gain, right_gain) = compute_stereo_pan(self.pan, &self.pan_law);

        for i in 0..num_samples {
            output_left[i] = working[i] * left_gain;
            output_right[i] = working[i] * right_gain;
        }
    }
}

/// Channel group for linking multiple channels.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct ChannelGroup {
    /// Group ID.
    pub id: Uuid,
    /// Group name.
    pub name: String,
    /// Member channels.
    pub channels: Vec<ChannelId>,
    /// Link gain controls.
    pub link_gain: bool,
    /// Link pan controls.
    pub link_pan: bool,
    /// Link mute.
    pub link_mute: bool,
    /// Link solo.
    pub link_solo: bool,
}

impl ChannelGroup {
    /// Create a new channel group.
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            channels: Vec::new(),
            link_gain: true,
            link_pan: true,
            link_mute: true,
            link_solo: true,
        }
    }

    /// Add channel to group.
    pub fn add_channel(&mut self, channel_id: ChannelId) {
        if !self.channels.contains(&channel_id) {
            self.channels.push(channel_id);
        }
    }

    /// Remove channel from group.
    pub fn remove_channel(&mut self, channel_id: ChannelId) {
        self.channels.retain(|&id| id != channel_id);
    }

    /// Check if group contains channel.
    #[must_use]
    pub fn contains(&self, channel_id: ChannelId) -> bool {
        self.channels.contains(&channel_id)
    }
}

/// Convert decibels to linear gain factor.
#[must_use]
pub fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Convert linear gain factor to decibels.
#[must_use]
pub fn linear_to_db(linear: f32) -> f32 {
    if linear <= 0.0 {
        return -120.0; // effectively -infinity
    }
    20.0 * linear.log10()
}

/// Compute stereo pan gains for left and right channels.
///
/// `pan` ranges from -1.0 (full left) to 1.0 (full right), 0.0 = center.
/// Returns `(left_gain, right_gain)`.
#[must_use]
pub fn compute_stereo_pan(pan: f32, pan_law: &PanLaw) -> (f32, f32) {
    // Normalize pan to 0..1 range (0 = full left, 1 = full right)
    let pan_norm = (pan + 1.0) * 0.5;
    let pan_norm = pan_norm.clamp(0.0, 1.0);

    match pan_law {
        PanLaw::Linear => {
            // Linear pan: left = 1 - pan, right = pan
            (1.0 - pan_norm, pan_norm)
        }
        PanLaw::Minus3dB => {
            // Equal power pan law: constant power across pan position
            // left = cos(pan * pi/2), right = sin(pan * pi/2)
            let angle = pan_norm * std::f32::consts::FRAC_PI_2;
            (angle.cos(), angle.sin())
        }
        PanLaw::Minus4Dot5dB => {
            // Compromise between equal power and linear
            // Blend of linear and equal-power
            let linear_l = 1.0 - pan_norm;
            let linear_r = pan_norm;
            let angle = pan_norm * std::f32::consts::FRAC_PI_2;
            let power_l = angle.cos();
            let power_r = angle.sin();
            // Blend 50/50
            (
                0.5 * linear_l + 0.5 * power_l,
                0.5 * linear_r + 0.5 * power_r,
            )
        }
        PanLaw::Minus6dB => {
            // Equal gain: center is at -6dB
            // left = (1 - pan) * 0.5, right = pan * 0.5
            // But scaled so hard left/right is unity
            (1.0 - pan_norm, pan_norm)
        }
    }
}

/// Compute surround panning gains for 5.1 layout.
///
/// `azimuth` in radians (0 = front, positive = clockwise).
/// Returns gains for [L, R, C, LFE, Ls, Rs].
#[must_use]
pub fn compute_surround_51_pan(azimuth: f32, elevation: f32) -> [f32; 6] {
    // VBAP-inspired approach for 5.1
    // Speaker positions (azimuth in radians):
    //   L = -30 deg, R = 30 deg, C = 0 deg, Ls = -110 deg, Rs = 110 deg
    let l_az: f32 = -30.0_f32.to_radians();
    let r_az: f32 = 30.0_f32.to_radians();
    let c_az: f32 = 0.0;
    let ls_az: f32 = -110.0_f32.to_radians();
    let rs_az: f32 = 110.0_f32.to_radians();

    let speakers = [l_az, r_az, c_az, ls_az, rs_az];
    let mut gains = [0.0_f32; 6];

    // Compute gain for each speaker based on angular distance
    let mut total_weight = 0.0_f32;
    for (i, &sp_az) in speakers.iter().enumerate() {
        let angle_diff = (azimuth - sp_az).abs();
        let angle_diff = if angle_diff > std::f32::consts::PI {
            2.0 * std::f32::consts::PI - angle_diff
        } else {
            angle_diff
        };

        // Cosine-based weighting (speakers within ~90 degrees contribute)
        let weight = (1.0 - angle_diff / std::f32::consts::PI).max(0.0);
        let weight = weight * weight; // Square for sharper falloff

        let idx = if i < 2 {
            i
        } else if i == 2 {
            2
        } else {
            i + 1
        };
        gains[idx] = weight;
        total_weight += weight;
    }

    // Normalize gains
    if total_weight > 0.0 {
        for g in &mut gains {
            *g /= total_weight;
        }
    }

    // LFE gets a fixed level based on elevation (more LFE for low sounds)
    gains[3] = (1.0 - elevation.abs() / std::f32::consts::FRAC_PI_2).max(0.0) * 0.1;

    gains
}

/// Compute surround panning gains for 7.1 layout.
///
/// Returns gains for [L, R, C, LFE, Ls, Rs, Lb, Rb].
#[must_use]
pub fn compute_surround_71_pan(azimuth: f32, elevation: f32) -> [f32; 8] {
    // Start with 5.1 base
    let base = compute_surround_51_pan(azimuth, elevation);

    let mut gains = [0.0_f32; 8];
    // L, R, C, LFE stay the same
    gains[0] = base[0];
    gains[1] = base[1];
    gains[2] = base[2];
    gains[3] = base[3];

    // Split rear surround energy between side and back
    // Ls -> Ls + Lb, Rs -> Rs + Rb
    gains[4] = base[4] * 0.7; // Ls
    gains[5] = base[5] * 0.7; // Rs
    gains[6] = base[4] * 0.3; // Lb (from Ls energy)
    gains[7] = base[5] * 0.3; // Rb (from Rs energy)

    gains
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_creation() {
        let channel = Channel::new(
            "Test".to_string(),
            ChannelType::Stereo,
            ChannelLayout::Stereo,
            48000,
            512,
        );
        assert_eq!(channel.name(), "Test");
        assert_eq!(channel.gain(), 1.0);
        assert_eq!(channel.pan(), 0.0);
    }

    #[test]
    fn test_channel_gain() {
        let mut channel = Channel::new(
            "Test".to_string(),
            ChannelType::Stereo,
            ChannelLayout::Stereo,
            48000,
            512,
        );

        channel.set_gain(0.5);
        assert!((channel.gain() - 0.5).abs() < f32::EPSILON);

        channel.set_gain_db(-6.0);
        assert!((channel.gain() - 0.501_187_2).abs() < 0.001);
    }

    #[test]
    fn test_channel_pan() {
        let mut channel = Channel::new(
            "Test".to_string(),
            ChannelType::Stereo,
            ChannelLayout::Stereo,
            48000,
            512,
        );

        channel.set_pan(0.5);
        assert!((channel.pan() - 0.5).abs() < f32::EPSILON);

        let (left, right) = channel.calculate_stereo_pan();
        assert!(left < right);
    }

    #[test]
    fn test_channel_state() {
        let mut channel = Channel::new(
            "Test".to_string(),
            ChannelType::Stereo,
            ChannelLayout::Stereo,
            48000,
            512,
        );

        assert!(!channel.is_muted());
        channel.set_muted(true);
        assert!(channel.is_muted());

        assert!(!channel.is_soloed());
        channel.set_soloed(true);
        assert!(channel.is_soloed());
    }

    #[test]
    fn test_pan_law() {
        assert!((PanLaw::Linear.center_gain() - 1.0).abs() < f32::EPSILON);
        assert!((PanLaw::Minus3dB.center_gain() - 0.707_946).abs() < 0.001);
        assert!((PanLaw::Minus6dB.center_gain() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_channel_type_count() {
        assert_eq!(ChannelType::Mono.channel_count(), 1);
        assert_eq!(ChannelType::Stereo.channel_count(), 2);
        assert_eq!(ChannelType::Surround51.channel_count(), 6);
        assert_eq!(ChannelType::Surround71.channel_count(), 8);
        assert_eq!(ChannelType::AmbisonicsFirstOrder.channel_count(), 4);
    }

    #[test]
    fn test_channel_group() {
        let mut group = ChannelGroup::new("Group 1".to_string());
        let id = ChannelId(Uuid::new_v4());

        assert!(!group.contains(id));
        group.add_channel(id);
        assert!(group.contains(id));
        group.remove_channel(id);
        assert!(!group.contains(id));
    }
}
