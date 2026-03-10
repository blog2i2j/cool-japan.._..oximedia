//! Spatial audio processing with Ambisonics and binaural rendering.
//!
//! This module provides comprehensive 3D spatial audio capabilities:
//!
//! # Ambisonics
//!
//! Ambisonics is a scene-based spatial audio format that captures a full 360° sound field.
//! It's resolution-independent and can be decoded to any speaker configuration.
//!
//! - **First-order** (4 channels: W, X, Y, Z) - Basic 3D soundfield
//! - **Second-order** (9 channels) - Higher spatial resolution
//! - **Third-order** (16 channels) - Very high spatial resolution
//!
//! ```rust
//! use oximedia_audio::spatial::{AmbisonicEncoder, AmbisonicOrder, SphericalCoord};
//!
//! let encoder = AmbisonicEncoder::new(AmbisonicOrder::First);
//! let direction = SphericalCoord::new(0.0, 0.0, 1.0); // Front
//! let coefficients = encoder.compute_coefficients(direction);
//! ```
//!
//! # Binaural Rendering
//!
//! Binaural audio uses HRTF (Head-Related Transfer Functions) to create 3D audio
//! for headphones, simulating how sound waves interact with the head and ears.
//!
//! ```rust
//! use oximedia_audio::spatial::{BinauralRenderer, SourcePosition};
//!
//! let mut renderer = BinauralRenderer::new(44100)?;
//! let position = SourcePosition::from_spherical(0.0, 0.0, 2.0);
//!
//! let input = vec![0.0; 512];
//! let mut output_left = vec![0.0; 512];
//! let mut output_right = vec![0.0; 512];
//!
//! renderer.render(&input, &position, &mut output_left, &mut output_right)?;
//! ```
//!
//! # Panning
//!
//! Various panning algorithms for positioning audio sources:
//!
//! - **Stereo panning** - Constant power, linear, -4.5dB, -6dB laws
//! - **VBAP** - Vector Base Amplitude Panning for arbitrary speaker layouts
//! - **DBAP** - Distance-Based Amplitude Panning
//!
//! # Spatial Reverb
//!
//! Room simulation with early reflections and late reverberation:
//!
//! ```rust
//! use oximedia_audio::spatial::{SpatialReverb, ReverbPreset};
//!
//! let mut reverb = SpatialReverb::new(44100);
//! ReverbPreset::MediumRoom.apply(&mut reverb);
//!
//! let input = vec![0.0; 512];
//! let mut output = vec![0.0; 512];
//! reverb.process(&input, &mut output)?;
//! ```
//!
//! # ITU-R BS.2051 Compliance
//!
//! This implementation follows ITU-R BS.2051 recommendations for advanced sound systems,
//! including proper speaker layouts and Ambisonics normalization (N3D/ACN).

pub mod ambisonics;
pub mod binaural;
pub mod hrtf_data;
pub mod panning;
pub mod reverb;

// Re-export main types
pub use ambisonics::{
    AmbisonicDecoder, AmbisonicEncoder, AmbisonicOrder, AmbisonicProcessor, AmbisonicRotator,
    RotationAngles, SpeakerConfig, SphericalCoord,
};

pub use binaural::{
    BinauralPreset, BinauralRenderer, DistanceModel, DopplerProcessor, ListenerOrientation,
    SourcePosition,
};

pub use hrtf_data::{HrirMeasurement, HrtfDatabase, HrtfManager};

pub use panning::{DbapPanner, PanLaw, SpeakerLayout, SpeakerPosition, StereoPanner, VbapPanner};

pub use reverb::{
    EarlyReflection, EarlyReflectionsProcessor, LateReverbProcessor, ReverbPreset, SpatialReverb,
};

use crate::{AudioError, AudioResult};

/// Main spatial audio processor combining all spatial processing capabilities
pub struct SpatialAudioProcessor {
    /// Sample rate
    sample_rate: u32,
    /// Ambisonic processor (optional)
    ambisonic_processor: Option<AmbisonicProcessor>,
    /// Binaural renderer (optional)
    binaural_renderer: Option<BinauralRenderer>,
    /// Spatial reverb (optional)
    spatial_reverb: Option<SpatialReverb>,
    /// Stereo panner (always available)
    stereo_panner: StereoPanner,
}

impl SpatialAudioProcessor {
    /// Create a new spatial audio processor
    pub fn new(sample_rate: u32) -> AudioResult<Self> {
        Ok(Self {
            sample_rate,
            ambisonic_processor: None,
            binaural_renderer: None,
            spatial_reverb: None,
            stereo_panner: StereoPanner::new(PanLaw::ConstantPower),
        })
    }

    /// Enable Ambisonics processing
    pub fn enable_ambisonics(&mut self, order: AmbisonicOrder, speaker_config: SpeakerConfig) {
        self.ambisonic_processor = Some(AmbisonicProcessor::new(order, speaker_config));
    }

    /// Enable binaural rendering
    pub fn enable_binaural(&mut self) -> AudioResult<()> {
        self.binaural_renderer = Some(BinauralRenderer::new(self.sample_rate)?);
        Ok(())
    }

    /// Enable spatial reverb
    pub fn enable_reverb(&mut self) {
        self.spatial_reverb = Some(SpatialReverb::new(self.sample_rate));
    }

    /// Disable Ambisonics processing
    pub fn disable_ambisonics(&mut self) {
        self.ambisonic_processor = None;
    }

    /// Disable binaural rendering
    pub fn disable_binaural(&mut self) {
        self.binaural_renderer = None;
    }

    /// Disable spatial reverb
    pub fn disable_reverb(&mut self) {
        self.spatial_reverb = None;
    }

    /// Get ambisonic processor reference
    pub fn ambisonic_processor(&self) -> Option<&AmbisonicProcessor> {
        self.ambisonic_processor.as_ref()
    }

    /// Get ambisonic processor mutable reference
    pub fn ambisonic_processor_mut(&mut self) -> Option<&mut AmbisonicProcessor> {
        self.ambisonic_processor.as_mut()
    }

    /// Get binaural renderer reference
    pub fn binaural_renderer(&self) -> Option<&BinauralRenderer> {
        self.binaural_renderer.as_ref()
    }

    /// Get binaural renderer mutable reference
    pub fn binaural_renderer_mut(&mut self) -> Option<&mut BinauralRenderer> {
        self.binaural_renderer.as_mut()
    }

    /// Get spatial reverb reference
    pub fn spatial_reverb(&self) -> Option<&SpatialReverb> {
        self.spatial_reverb.as_ref()
    }

    /// Get spatial reverb mutable reference
    pub fn spatial_reverb_mut(&mut self) -> Option<&mut SpatialReverb> {
        self.spatial_reverb.as_mut()
    }

    /// Get stereo panner reference
    pub fn stereo_panner(&self) -> &StereoPanner {
        &self.stereo_panner
    }

    /// Get stereo panner mutable reference
    pub fn stereo_panner_mut(&mut self) -> &mut StereoPanner {
        &mut self.stereo_panner
    }

    /// Process mono source to stereo with simple panning
    pub fn process_stereo(
        &mut self,
        input: &[f32],
        pan_position: f32,
        output_left: &mut [f32],
        output_right: &mut [f32],
    ) -> AudioResult<()> {
        self.stereo_panner.set_position(pan_position);
        self.stereo_panner
            .process_mono(input, output_left, output_right)?;

        // Apply reverb if enabled
        if let Some(reverb) = &mut self.spatial_reverb {
            let mut reverb_left = vec![0.0; output_left.len()];
            let mut reverb_right = vec![0.0; output_right.len()];

            reverb.process_stereo(
                output_left,
                output_right,
                &mut reverb_left,
                &mut reverb_right,
            )?;

            // Mix with original (50/50 for simplicity)
            for i in 0..output_left.len() {
                output_left[i] = output_left[i] * 0.7 + reverb_left[i] * 0.3;
                output_right[i] = output_right[i] * 0.7 + reverb_right[i] * 0.3;
            }
        }

        Ok(())
    }

    /// Process mono source to binaural stereo
    pub fn process_binaural(
        &mut self,
        input: &[f32],
        position: SourcePosition,
        output_left: &mut [f32],
        output_right: &mut [f32],
    ) -> AudioResult<()> {
        if let Some(renderer) = &mut self.binaural_renderer {
            renderer.render(input, &position, output_left, output_right)?;

            // Apply reverb if enabled
            if let Some(reverb) = &mut self.spatial_reverb {
                let mut reverb_left = vec![0.0; output_left.len()];
                let mut reverb_right = vec![0.0; output_right.len()];

                reverb.process_stereo(
                    output_left,
                    output_right,
                    &mut reverb_left,
                    &mut reverb_right,
                )?;

                for i in 0..output_left.len() {
                    output_left[i] = output_left[i] * 0.8 + reverb_left[i] * 0.2;
                    output_right[i] = output_right[i] * 0.8 + reverb_right[i] * 0.2;
                }
            }

            Ok(())
        } else {
            Err(AudioError::Internal(
                "Binaural rendering not enabled".to_string(),
            ))
        }
    }

    /// Process mono source through Ambisonics to speaker array
    pub fn process_ambisonics(
        &mut self,
        input: &[f32],
        direction: SphericalCoord,
        output: &mut [Vec<f32>],
    ) -> AudioResult<()> {
        if let Some(processor) = &mut self.ambisonic_processor {
            processor.process_source(input, direction, output)?;

            // Apply reverb if enabled (per channel)
            if let Some(reverb) = &mut self.spatial_reverb {
                for channel_output in output.iter_mut() {
                    let mut reverb_output = vec![0.0; channel_output.len()];
                    reverb.process(channel_output, &mut reverb_output)?;

                    for i in 0..channel_output.len() {
                        channel_output[i] = channel_output[i] * 0.7 + reverb_output[i] * 0.3;
                    }
                }
            }

            Ok(())
        } else {
            Err(AudioError::Internal("Ambisonics not enabled".to_string()))
        }
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Check if Ambisonics is enabled
    pub fn is_ambisonics_enabled(&self) -> bool {
        self.ambisonic_processor.is_some()
    }

    /// Check if binaural rendering is enabled
    pub fn is_binaural_enabled(&self) -> bool {
        self.binaural_renderer.is_some()
    }

    /// Check if spatial reverb is enabled
    pub fn is_reverb_enabled(&self) -> bool {
        self.spatial_reverb.is_some()
    }
}

/// Preset configurations for common use cases
pub struct SpatialPreset;

impl SpatialPreset {
    /// Stereo music production preset
    pub fn stereo_music(sample_rate: u32) -> AudioResult<SpatialAudioProcessor> {
        let mut processor = SpatialAudioProcessor::new(sample_rate)?;
        processor.enable_reverb();
        if let Some(reverb) = processor.spatial_reverb_mut() {
            ReverbPreset::MediumRoom.apply(reverb);
        }
        Ok(processor)
    }

    /// Binaural headphone preset
    pub fn binaural_headphones(sample_rate: u32) -> AudioResult<SpatialAudioProcessor> {
        let mut processor = SpatialAudioProcessor::new(sample_rate)?;
        processor.enable_binaural()?;
        processor.enable_reverb();
        if let Some(reverb) = processor.spatial_reverb_mut() {
            ReverbPreset::SmallRoom.apply(reverb);
        }
        Ok(processor)
    }

    /// 5.1 surround preset
    pub fn surround_5_1(sample_rate: u32) -> AudioResult<SpatialAudioProcessor> {
        let mut processor = SpatialAudioProcessor::new(sample_rate)?;
        processor.enable_ambisonics(AmbisonicOrder::First, SpeakerConfig::surround_5_1());
        processor.enable_reverb();
        if let Some(reverb) = processor.spatial_reverb_mut() {
            ReverbPreset::LargeHall.apply(reverb);
        }
        Ok(processor)
    }

    /// VR/AR spatial audio preset
    pub fn vr_spatial(sample_rate: u32) -> AudioResult<SpatialAudioProcessor> {
        let mut processor = SpatialAudioProcessor::new(sample_rate)?;
        processor.enable_binaural()?;
        processor.enable_reverb();

        // Enable Doppler for VR
        if let Some(binaural) = processor.binaural_renderer_mut() {
            binaural.set_doppler_enabled(true);
            binaural.set_distance_model(DistanceModel::Inverse {
                min: 0.5,
                rolloff: 1.0,
            });
        }

        if let Some(reverb) = processor.spatial_reverb_mut() {
            ReverbPreset::MediumRoom.apply(reverb);
        }

        Ok(processor)
    }

    /// Game audio preset
    pub fn game_audio(sample_rate: u32) -> AudioResult<SpatialAudioProcessor> {
        let mut processor = SpatialAudioProcessor::new(sample_rate)?;
        processor.enable_binaural()?;
        processor.enable_reverb();

        if let Some(binaural) = processor.binaural_renderer_mut() {
            binaural.set_distance_model(DistanceModel::Exponential {
                min: 1.0,
                rolloff: 2.0,
            });
        }

        if let Some(reverb) = processor.spatial_reverb_mut() {
            reverb.set_rt60(1.5);
            reverb.set_damping(0.4);
            reverb.set_mix(0.3);
        }

        Ok(processor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_audio_processor_creation() {
        let processor = SpatialAudioProcessor::new(44100);
        assert!(processor.is_ok());

        let processor = processor.expect("should succeed");
        assert_eq!(processor.sample_rate(), 44100);
        assert!(!processor.is_ambisonics_enabled());
        assert!(!processor.is_binaural_enabled());
        assert!(!processor.is_reverb_enabled());
    }

    #[test]
    fn test_enable_features() {
        let mut processor = SpatialAudioProcessor::new(44100).expect("should succeed");

        processor.enable_ambisonics(AmbisonicOrder::First, SpeakerConfig::stereo());
        assert!(processor.is_ambisonics_enabled());

        processor.enable_binaural().expect("should succeed");
        assert!(processor.is_binaural_enabled());

        processor.enable_reverb();
        assert!(processor.is_reverb_enabled());
    }

    #[test]
    fn test_disable_features() {
        let mut processor = SpatialAudioProcessor::new(44100).expect("should succeed");

        processor.enable_ambisonics(AmbisonicOrder::First, SpeakerConfig::stereo());
        processor.disable_ambisonics();
        assert!(!processor.is_ambisonics_enabled());
    }

    #[test]
    fn test_process_stereo() {
        let mut processor = SpatialAudioProcessor::new(44100).expect("should succeed");

        let input = vec![1.0; 100];
        let mut left = vec![0.0; 100];
        let mut right = vec![0.0; 100];

        let result = processor.process_stereo(&input, 0.0, &mut left, &mut right);
        assert!(result.is_ok());

        assert!(left.iter().any(|&x| x != 0.0));
        assert!(right.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_process_binaural() {
        let mut processor = SpatialAudioProcessor::new(44100).expect("should succeed");
        processor.enable_binaural().expect("should succeed");

        let input = vec![1.0; 100];
        let mut left = vec![0.0; 100];
        let mut right = vec![0.0; 100];

        let position = SourcePosition::from_spherical(0.0, 0.0, 2.0);

        let result = processor.process_binaural(&input, position, &mut left, &mut right);
        assert!(result.is_ok());
    }

    #[test]
    fn test_spatial_presets() {
        let processor = SpatialPreset::stereo_music(44100);
        assert!(processor.is_ok());

        let processor = SpatialPreset::binaural_headphones(44100);
        assert!(processor.is_ok());

        let processor = SpatialPreset::surround_5_1(44100);
        assert!(processor.is_ok());

        let processor = SpatialPreset::vr_spatial(44100);
        assert!(processor.is_ok());

        let processor = SpatialPreset::game_audio(44100);
        assert!(processor.is_ok());
    }
}
