//! Professional audio mixer with automation for `OxiMedia`.
//!
//! This crate provides a complete digital audio mixing console with:
//!
//! - **Multi-channel mixing** - Support for 100+ channels with flexible routing
//! - **Channel types** - Mono, Stereo, 5.1, 7.1, and Ambisonics
//! - **Effect processing** - Dynamics, EQ, reverb, delay, modulation, distortion
//! - **Automation system** - Full parameter automation with multiple modes
//! - **Bus architecture** - Master, group, and auxiliary buses
//! - **Professional metering** - Peak, RMS, VU, LUFS, phase correlation
//! - **Session management** - Save/load mixer state with undo/redo
//!
//! # Architecture
//!
//! The mixer follows a professional DAW-style architecture:
//!
//! ```text
//! Input → Channel → Effects → Fader → Pan → Sends → Bus → Master Out
//! ```
//!
//! ## Channels
//!
//! Each channel provides:
//! - Input gain and phase inversion
//! - Insert effect chain (up to 8 slots)
//! - Channel fader with gain control
//! - Pan control (stereo, surround, binaural)
//! - Solo/Mute/Arm states
//! - Pre/post-fader sends to buses
//! - Direct monitoring output
//! - Channel linking for stereo pairs
//!
//! ## Buses
//!
//! Multiple bus types:
//! - **Master Bus** - Final stereo mixdown output
//! - **Group Buses** - Submix multiple channels together
//! - **Auxiliary Buses** - Effect sends/returns (reverb, delay, etc.)
//! - **Matrix Buses** - Advanced routing and monitoring
//!
//! ## Automation
//!
//! Full parameter automation with:
//! - **Read Mode** - Play back recorded automation
//! - **Write Mode** - Record all parameter changes
//! - **Touch Mode** - Record only when touching controls
//! - **Latch Mode** - Continue last value after release
//! - **Trim Mode** - Apply relative changes to existing automation
//!
//! ## Effects
//!
//! Professional effect categories:
//! - **Dynamics** - Compressor, Limiter, Gate, Expander, De-esser
//! - **EQ** - Parametric, Graphic, Shelving, High/Low Pass
//! - **Time-based** - Reverb, Delay, Echo, Chorus, Flanger
//! - **Modulation** - Phaser, Vibrato, Tremolo, Ring Modulator
//! - **Distortion** - Saturation, Overdrive, Bit Crusher, Wave Shaper
//!
//! ## Metering
//!
//! Professional-grade metering:
//! - **Peak Meters** - Sample-accurate peak detection
//! - **RMS Meters** - Average level measurement
//! - **VU Meters** - IEC 60268-10 standard (300ms ballistics)
//! - **LUFS Meters** - EBU R128 loudness metering
//! - **Phase Correlation** - Stereo compatibility checking
//! - **Spectrum Analyzer** - Real-time frequency analysis
//!
//! # Real-time Performance
//!
//! The mixer is optimized for low-latency operation:
//! - Lock-free audio processing path
//! - SIMD optimizations for DSP
//! - Memory-efficient buffer management
//! - Zero-copy audio routing where possible
//! - Target latency: <10ms for 48kHz/512 samples
//!
//! # Example
//!
//! ```rust
//! use oximedia_mixer::{AudioMixer, MixerConfig, ChannelType, MixerResult};
//! use oximedia_audio::ChannelLayout;
//!
//! fn example() -> MixerResult<()> {
//!     // Create mixer with default configuration
//!     let config = MixerConfig {
//!         sample_rate: 48000,
//!         buffer_size: 512,
//!         max_channels: 64,
//!         ..Default::default()
//!     };
//!
//!     let mut mixer = AudioMixer::new(config);
//!
//!     // Add a stereo channel
//!     let channel_id = mixer.add_channel(
//!         "Vocals".to_string(),
//!         ChannelType::Stereo,
//!         ChannelLayout::Stereo,
//!     )?;
//!
//!     // Set channel gain (0.0 = -inf dB, 1.0 = 0 dB)
//!     mixer.set_channel_gain(channel_id, 0.8)?;
//!
//!     // Pan center
//!     mixer.set_channel_pan(channel_id, 0.0)?;
//!
//!     // Process audio
//!     // let output = mixer.process(&input_frame)?;
//!     Ok(())
//! }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod automation;
pub mod automation_lane;
pub mod aux_send;
pub mod bus;
pub mod channel;
pub mod channel_strip;
pub mod crossfade;
pub mod delay_line;
pub mod dynamics;
pub mod effects;
pub mod effects_chain;
pub mod eq_band;
pub mod group_bus;
pub mod insert_chain;
pub mod limiter;
pub mod matrix_mixer;
pub mod meter_bridge;
pub mod metering;
pub mod monitor_mix;
pub mod pan_matrix;
pub mod routing;
pub mod scene_recall;
pub mod send_return;
pub mod session;
pub mod sidechain;
pub mod snapshot;
pub mod vca;

use std::collections::HashMap;

use oximedia_audio::{AudioFrame, ChannelLayout};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub use automation::{
    AutomationCurve, AutomationData, AutomationMode, AutomationParameter, AutomationPoint,
};
pub use bus::{Bus, BusConfig, BusId, BusType};
pub use channel::{Channel, ChannelId, ChannelType, PanMode};
pub use effects::{Effect, EffectCategory, EffectId, EffectSlot};
pub use metering::{Meter, MeterType, MeteringData};
pub use session::{MixerSession, SessionData};

/// Audio mixer error types.
#[derive(Debug, thiserror::Error)]
pub enum MixerError {
    /// Channel not found.
    #[error("Channel not found: {0}")]
    ChannelNotFound(ChannelId),

    /// Bus not found.
    #[error("Bus not found: {0}")]
    BusNotFound(BusId),

    /// Effect not found.
    #[error("Effect not found: {0}")]
    EffectNotFound(EffectId),

    /// Invalid parameter value.
    #[error("Invalid parameter value: {0}")]
    InvalidParameter(String),

    /// Maximum channels exceeded.
    #[error("Maximum channels exceeded: {0}")]
    MaxChannelsExceeded(usize),

    /// Audio processing error.
    #[error("Audio processing error: {0}")]
    ProcessingError(String),

    /// Session error.
    #[error("Session error: {0}")]
    SessionError(String),
}

/// Result type for mixer operations.
pub type MixerResult<T> = Result<T, MixerError>;

/// Mixer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixerConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,

    /// Buffer size in samples.
    pub buffer_size: usize,

    /// Maximum number of channels.
    pub max_channels: usize,

    /// Maximum number of buses.
    pub max_buses: usize,

    /// Maximum number of effects per channel.
    pub max_effects_per_channel: usize,

    /// Enable automation.
    pub enable_automation: bool,

    /// Enable metering.
    pub enable_metering: bool,

    /// Metering update rate in Hz.
    pub metering_rate: u32,
}

impl Default for MixerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            buffer_size: 512,
            max_channels: 128,
            max_buses: 32,
            max_effects_per_channel: 8,
            enable_automation: true,
            enable_metering: true,
            metering_rate: 30,
        }
    }
}

/// Professional audio mixer.
#[derive(Debug)]
pub struct AudioMixer {
    config: MixerConfig,
    channels: HashMap<ChannelId, Channel>,
    buses: HashMap<BusId, Bus>,
    master_bus: Bus,
    session: MixerSession,
    sample_count: u64,
}

impl AudioMixer {
    /// Create a new audio mixer.
    #[must_use]
    pub fn new(config: MixerConfig) -> Self {
        let master_bus = Bus::new(
            "Master".to_string(),
            BusType::Master,
            ChannelLayout::Stereo,
            config.sample_rate,
            config.buffer_size,
        );

        Self {
            config,
            channels: HashMap::new(),
            buses: HashMap::new(),
            master_bus,
            session: MixerSession::new(),
            sample_count: 0,
        }
    }

    /// Get mixer configuration.
    #[must_use]
    pub fn config(&self) -> &MixerConfig {
        &self.config
    }

    /// Get current sample count.
    #[must_use]
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Get current time in seconds.
    #[must_use]
    pub fn time_seconds(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        {
            self.sample_count as f64 / f64::from(self.config.sample_rate)
        }
    }

    /// Add a new channel.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::MaxChannelsExceeded` if the maximum number of channels is reached.
    pub fn add_channel(
        &mut self,
        name: String,
        channel_type: ChannelType,
        layout: ChannelLayout,
    ) -> MixerResult<ChannelId> {
        if self.channels.len() >= self.config.max_channels {
            return Err(MixerError::MaxChannelsExceeded(self.config.max_channels));
        }

        let id = ChannelId(Uuid::new_v4());
        let channel = Channel::new(
            name,
            channel_type,
            layout,
            self.config.sample_rate,
            self.config.buffer_size,
        );

        self.channels.insert(id, channel);
        Ok(id)
    }

    /// Remove a channel.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ChannelNotFound` if the channel does not exist.
    pub fn remove_channel(&mut self, id: ChannelId) -> MixerResult<()> {
        self.channels
            .remove(&id)
            .ok_or(MixerError::ChannelNotFound(id))?;
        Ok(())
    }

    /// Get a channel.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ChannelNotFound` if the channel does not exist.
    pub fn get_channel(&self, id: ChannelId) -> MixerResult<&Channel> {
        self.channels
            .get(&id)
            .ok_or(MixerError::ChannelNotFound(id))
    }

    /// Get a mutable channel.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ChannelNotFound` if the channel does not exist.
    pub fn get_channel_mut(&mut self, id: ChannelId) -> MixerResult<&mut Channel> {
        self.channels
            .get_mut(&id)
            .ok_or(MixerError::ChannelNotFound(id))
    }

    /// Get all channels.
    #[must_use]
    pub fn channels(&self) -> &HashMap<ChannelId, Channel> {
        &self.channels
    }

    /// Set channel gain (0.0 = -inf dB, 1.0 = 0 dB).
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ChannelNotFound` if the channel does not exist.
    pub fn set_channel_gain(&mut self, id: ChannelId, gain: f32) -> MixerResult<()> {
        let channel = self.get_channel_mut(id)?;
        channel.set_gain(gain);
        Ok(())
    }

    /// Set channel pan (-1.0 = full left, 0.0 = center, 1.0 = full right).
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ChannelNotFound` if the channel does not exist.
    pub fn set_channel_pan(&mut self, id: ChannelId, pan: f32) -> MixerResult<()> {
        let channel = self.get_channel_mut(id)?;
        channel.set_pan(pan);
        Ok(())
    }

    /// Process audio for one buffer period.
    ///
    /// The full DSP pipeline:
    /// 1. Extract f32 samples from the input frame's raw byte buffer
    /// 2. For each channel: input gain -> effects -> fader -> pan
    /// 3. Sum all channel outputs into master stereo bus
    /// 4. Apply master bus soft clipping to prevent digital overs
    /// 5. Pack the result back into an `AudioFrame`
    ///
    /// # Errors
    ///
    /// Returns `MixerError::ProcessingError` if audio processing fails.
    pub fn process(&mut self, frame: &AudioFrame) -> MixerResult<AudioFrame> {
        let buffer_size = self.config.buffer_size;

        // Master bus stereo accumulators
        let mut master_left = vec![0.0_f32; buffer_size];
        let mut master_right = vec![0.0_f32; buffer_size];

        // Extract f32 samples from the raw byte data in the input frame.
        // AudioBuffer stores raw bytes; for F32 format each sample is 4 bytes LE.
        let input_samples = extract_f32_samples(frame, buffer_size);

        // Process each channel through its strip
        let channel_ids: Vec<ChannelId> = self.channels.keys().copied().collect();

        for channel_id in &channel_ids {
            if let Some(channel) = self.channels.get(channel_id) {
                let mut ch_left = vec![0.0_f32; buffer_size];
                let mut ch_right = vec![0.0_f32; buffer_size];

                // Process through the channel strip
                channel.process_strip(&input_samples, &mut ch_left, &mut ch_right);

                // Sum into master bus
                for i in 0..buffer_size {
                    master_left[i] += ch_left[i];
                    master_right[i] += ch_right[i];
                }
            }
        }

        // Apply master bus soft clipping to prevent digital overs
        for i in 0..buffer_size {
            master_left[i] = soft_clip(master_left[i]);
            master_right[i] = soft_clip(master_right[i]);
        }

        self.sample_count += buffer_size as u64;

        // Create output frame with interleaved stereo packed as raw bytes
        let mut output = AudioFrame::new(
            oximedia_core::SampleFormat::F32,
            self.config.sample_rate,
            ChannelLayout::Stereo,
        );

        // Pack interleaved stereo f32 samples into bytes (little-endian)
        let mut raw_bytes = Vec::with_capacity(buffer_size * 2 * 4);
        for i in 0..buffer_size {
            raw_bytes.extend_from_slice(&master_left[i].to_le_bytes());
            raw_bytes.extend_from_slice(&master_right[i].to_le_bytes());
        }
        output.samples = oximedia_audio::AudioBuffer::Interleaved(bytes::Bytes::from(raw_bytes));

        Ok(output)
    }

    /// Get mixer session.
    #[must_use]
    pub fn session(&self) -> &MixerSession {
        &self.session
    }

    /// Get mutable mixer session.
    #[must_use]
    pub fn session_mut(&mut self) -> &mut MixerSession {
        &mut self.session
    }

    /// Add a new bus.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::MaxChannelsExceeded` if the maximum number of buses is reached.
    pub fn add_bus(
        &mut self,
        name: String,
        bus_type: BusType,
        layout: ChannelLayout,
    ) -> MixerResult<BusId> {
        if self.buses.len() >= self.config.max_buses {
            return Err(MixerError::MaxChannelsExceeded(self.config.max_buses));
        }

        let id = BusId(Uuid::new_v4());
        let bus = Bus::new(
            name,
            bus_type,
            layout,
            self.config.sample_rate,
            self.config.buffer_size,
        );

        self.buses.insert(id, bus);
        Ok(id)
    }

    /// Get a bus.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::BusNotFound` if the bus does not exist.
    pub fn get_bus(&self, id: BusId) -> MixerResult<&Bus> {
        self.buses.get(&id).ok_or(MixerError::BusNotFound(id))
    }

    /// Get master bus.
    #[must_use]
    pub fn master_bus(&self) -> &Bus {
        &self.master_bus
    }

    /// Get mutable master bus.
    #[must_use]
    pub fn master_bus_mut(&mut self) -> &mut Bus {
        &mut self.master_bus
    }
}

/// Extract f32 samples from an `AudioFrame`.
///
/// Interprets the raw bytes in the frame as little-endian f32 values.
/// Returns a mono buffer of at most `max_samples` samples.
fn extract_f32_samples(frame: &AudioFrame, max_samples: usize) -> Vec<f32> {
    let raw_bytes = match &frame.samples {
        oximedia_audio::AudioBuffer::Interleaved(data) => data.as_ref(),
        oximedia_audio::AudioBuffer::Planar(planes) => {
            if let Some(first) = planes.first() {
                first.as_ref()
            } else {
                return vec![0.0; max_samples];
            }
        }
    };

    // Each f32 sample is 4 bytes
    let num_f32_samples = raw_bytes.len() / 4;
    let count = num_f32_samples.min(max_samples);

    let mut samples = Vec::with_capacity(count);
    for i in 0..count {
        let offset = i * 4;
        if offset + 4 <= raw_bytes.len() {
            let bytes: [u8; 4] = [
                raw_bytes[offset],
                raw_bytes[offset + 1],
                raw_bytes[offset + 2],
                raw_bytes[offset + 3],
            ];
            samples.push(f32::from_le_bytes(bytes));
        }
    }

    // Pad with zeros if input is shorter than buffer_size
    samples.resize(max_samples, 0.0);
    samples
}

/// Soft clipping function using tanh-like saturation.
///
/// Maps input linearly near zero and smoothly saturates towards +/-1.0.
/// This prevents hard digital clipping artifacts.
fn soft_clip(x: f32) -> f32 {
    if x.abs() < 0.5 {
        x // Linear region for small signals
    } else if x > 0.0 {
        // Soft saturation for positive values
        let t = (x - 0.5) * 2.0;
        0.5 + 0.5 * (1.0 - (-t).exp()) / (1.0 + (-t).exp())
    } else {
        // Soft saturation for negative values
        let t = (-x - 0.5) * 2.0;
        -(0.5 + 0.5 * (1.0 - (-t).exp()) / (1.0 + (-t).exp()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixer_creation() {
        let config = MixerConfig::default();
        let mixer = AudioMixer::new(config);
        assert_eq!(mixer.channels().len(), 0);
    }

    #[test]
    fn test_add_channel() {
        let config = MixerConfig::default();
        let mut mixer = AudioMixer::new(config);

        let id = mixer
            .add_channel(
                "Test".to_string(),
                ChannelType::Stereo,
                ChannelLayout::Stereo,
            )
            .expect("test expectation failed");

        assert!(mixer.get_channel(id).is_ok());
    }

    #[test]
    fn test_channel_gain() {
        let config = MixerConfig::default();
        let mut mixer = AudioMixer::new(config);

        let id = mixer
            .add_channel(
                "Test".to_string(),
                ChannelType::Stereo,
                ChannelLayout::Stereo,
            )
            .expect("test expectation failed");

        mixer
            .set_channel_gain(id, 0.5)
            .expect("set_channel_gain should succeed");
        let channel = mixer.get_channel(id).expect("channel should be valid");
        assert!((channel.gain() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_max_channels() {
        let config = MixerConfig {
            max_channels: 2,
            ..Default::default()
        };
        let mut mixer = AudioMixer::new(config);

        mixer
            .add_channel(
                "Channel 1".to_string(),
                ChannelType::Stereo,
                ChannelLayout::Stereo,
            )
            .expect("test expectation failed");
        mixer
            .add_channel(
                "Channel 2".to_string(),
                ChannelType::Stereo,
                ChannelLayout::Stereo,
            )
            .expect("test expectation failed");

        let result = mixer.add_channel(
            "Channel 3".to_string(),
            ChannelType::Stereo,
            ChannelLayout::Stereo,
        );

        assert!(result.is_err());
    }
}
