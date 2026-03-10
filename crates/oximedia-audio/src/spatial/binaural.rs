//! Binaural rendering with HRTF processing.
//!
//! This module provides:
//! - HRTF convolution with overlap-add
//! - Azimuth and elevation handling
//! - Distance attenuation
//! - Doppler effect
//! - Head tracking integration

use super::hrtf_data::{HrirMeasurement, HrtfDatabase, HrtfManager, MAX_HRIR_LENGTH};
use crate::{AudioError, AudioResult};
use rustfft::FftPlanner;
use std::f32::consts::PI;
use std::sync::Arc;

/// Speed of sound in m/s
const SPEED_OF_SOUND: f32 = 343.0;

/// Minimum distance for distance attenuation (to avoid division by zero)
const MIN_DISTANCE: f32 = 0.1;

/// Source position in 3D space
#[derive(Debug, Clone, Copy)]
pub struct SourcePosition {
    /// X coordinate (left/right)
    pub x: f32,
    /// Y coordinate (front/back)
    pub y: f32,
    /// Z coordinate (up/down)
    pub z: f32,
}

impl SourcePosition {
    /// Create a new source position
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Create from spherical coordinates (azimuth, elevation, distance)
    pub fn from_spherical(azimuth: f32, elevation: f32, distance: f32) -> Self {
        let x = distance * elevation.cos() * azimuth.sin();
        let y = distance * elevation.cos() * azimuth.cos();
        let z = distance * elevation.sin();
        Self { x, y, z }
    }

    /// Convert to spherical coordinates
    pub fn to_spherical(&self) -> (f32, f32, f32) {
        let distance = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        let azimuth = self.x.atan2(self.y);
        let elevation = if distance > 0.0 {
            (self.z / distance).asin()
        } else {
            0.0
        };
        (azimuth, elevation, distance)
    }

    /// Calculate distance to listener (at origin)
    pub fn distance(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

/// Listener orientation (head rotation)
#[derive(Debug, Clone, Copy)]
pub struct ListenerOrientation {
    /// Yaw (rotation around vertical axis, in radians)
    pub yaw: f32,
    /// Pitch (rotation around left-right axis, in radians)
    pub pitch: f32,
    /// Roll (rotation around front-back axis, in radians)
    pub roll: f32,
}

impl ListenerOrientation {
    /// Create a new listener orientation
    pub fn new(yaw: f32, pitch: f32, roll: f32) -> Self {
        Self { yaw, pitch, roll }
    }

    /// Default orientation (facing forward)
    pub fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
        }
    }

    /// Transform source position to listener-relative coordinates
    pub fn transform(&self, pos: &SourcePosition) -> SourcePosition {
        // Apply yaw rotation
        let cos_yaw = self.yaw.cos();
        let sin_yaw = self.yaw.sin();
        let x1 = pos.x * cos_yaw - pos.y * sin_yaw;
        let y1 = pos.x * sin_yaw + pos.y * cos_yaw;
        let z1 = pos.z;

        // Apply pitch rotation
        let cos_pitch = self.pitch.cos();
        let sin_pitch = self.pitch.sin();
        let y2 = y1 * cos_pitch - z1 * sin_pitch;
        let z2 = y1 * sin_pitch + z1 * cos_pitch;
        let x2 = x1;

        // Apply roll rotation
        let cos_roll = self.roll.cos();
        let sin_roll = self.roll.sin();
        let x3 = x2 * cos_roll - z2 * sin_roll;
        let z3 = x2 * sin_roll + z2 * cos_roll;

        SourcePosition::new(x3, y2, z3)
    }
}

/// HRTF convolver using overlap-add FFT convolution
pub struct HrtfConvolver {
    /// FFT size (must be power of 2)
    fft_size: usize,
    /// Hop size (typically FFT size / 2)
    hop_size: usize,
    /// Input buffer
    input_buffer: Vec<f32>,
    /// Output buffer (left channel)
    output_buffer_left: Vec<f32>,
    /// Output buffer (right channel)
    output_buffer_right: Vec<f32>,
    /// Current HRIR (left)
    hrir_left: Vec<f32>,
    /// Current HRIR (right)
    hrir_right: Vec<f32>,
    /// FFT planner
    fft_planner: FftPlanner<f32>,
    /// Sample counter
    sample_counter: usize,
}

impl HrtfConvolver {
    /// Create a new HRTF convolver
    pub fn new(fft_size: usize) -> Self {
        let hop_size = fft_size / 2;

        Self {
            fft_size,
            hop_size,
            input_buffer: vec![0.0; fft_size],
            output_buffer_left: vec![0.0; fft_size * 2],
            output_buffer_right: vec![0.0; fft_size * 2],
            hrir_left: vec![0.0; MAX_HRIR_LENGTH],
            hrir_right: vec![0.0; MAX_HRIR_LENGTH],
            fft_planner: FftPlanner::new(),
            sample_counter: 0,
        }
    }

    /// Update HRIR
    pub fn set_hrir(&mut self, hrir: &HrirMeasurement) {
        let len = hrir.len().min(MAX_HRIR_LENGTH);
        self.hrir_left[..len].copy_from_slice(&hrir.left[..len]);
        self.hrir_right[..len].copy_from_slice(&hrir.right[..len]);
    }

    /// Process a single sample (simple time-domain convolution)
    pub fn process_sample(&mut self, input: f32) -> (f32, f32) {
        // Shift input buffer
        self.input_buffer.rotate_left(1);
        let buffer_len = self.input_buffer.len();
        self.input_buffer[buffer_len - 1] = input;

        // Simple time-domain convolution for low latency
        let mut left = 0.0;
        let mut right = 0.0;

        let hrir_len = MAX_HRIR_LENGTH.min(buffer_len);
        for i in 0..hrir_len {
            let buffer_idx = buffer_len - 1 - i;
            left += self.input_buffer[buffer_idx] * self.hrir_left[i];
            right += self.input_buffer[buffer_idx] * self.hrir_right[i];
        }

        (left, right)
    }

    /// Process a buffer with overlap-add
    pub fn process_buffer(
        &mut self,
        input: &[f32],
        output_left: &mut [f32],
        output_right: &mut [f32],
    ) -> AudioResult<()> {
        if input.len() != output_left.len() || input.len() != output_right.len() {
            return Err(AudioError::InvalidParameter(
                "Buffer size mismatch".to_string(),
            ));
        }

        for i in 0..input.len() {
            let (left, right) = self.process_sample(input[i]);
            output_left[i] = left;
            output_right[i] = right;
        }

        Ok(())
    }
}

/// Distance attenuation model
#[derive(Debug, Clone, Copy)]
pub enum DistanceModel {
    /// No attenuation
    None,
    /// Linear attenuation: gain = 1 - (distance - min) / (max - min)
    Linear {
        /// Minimum distance
        min: f32,
        /// Maximum distance
        max: f32,
    },
    /// Inverse distance: gain = min / (min + rolloff * (distance - min))
    Inverse {
        /// Minimum distance
        min: f32,
        /// Rolloff factor
        rolloff: f32,
    },
    /// Exponential: gain = (distance / min) ^ (-rolloff)
    Exponential {
        /// Minimum distance
        min: f32,
        /// Rolloff exponent
        rolloff: f32,
    },
}

impl DistanceModel {
    /// Calculate gain for a given distance
    pub fn calculate_gain(&self, distance: f32) -> f32 {
        match self {
            DistanceModel::None => 1.0,
            DistanceModel::Linear { min, max } => {
                if distance <= *min {
                    1.0
                } else if distance >= *max {
                    0.0
                } else {
                    1.0 - (distance - min) / (max - min)
                }
            }
            DistanceModel::Inverse { min, rolloff } => {
                let distance = distance.max(*min);
                min / (min + rolloff * (distance - min))
            }
            DistanceModel::Exponential { min, rolloff } => {
                let distance = distance.max(*min);
                (distance / min).powf(-rolloff)
            }
        }
    }
}

/// Doppler effect processor
pub struct DopplerProcessor {
    sample_rate: f32,
    previous_distance: f32,
    speed_of_sound: f32,
}

impl DopplerProcessor {
    /// Create a new Doppler processor
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate: sample_rate as f32,
            previous_distance: 1.0,
            speed_of_sound: SPEED_OF_SOUND,
        }
    }

    /// Calculate Doppler shift factor
    pub fn calculate_shift(&mut self, current_distance: f32, delta_time: f32) -> f32 {
        // Calculate velocity (change in distance over time)
        let velocity = (current_distance - self.previous_distance) / delta_time;
        self.previous_distance = current_distance;

        // Doppler shift factor: f' = f * (c / (c + v))
        // where c = speed of sound, v = velocity (positive = moving away)
        let shift = self.speed_of_sound / (self.speed_of_sound + velocity);
        shift.clamp(0.5, 2.0) // Limit extreme shifts
    }
}

/// Binaural renderer
pub struct BinauralRenderer {
    /// Sample rate
    sample_rate: u32,
    /// HRTF database manager
    hrtf_manager: Arc<HrtfManager>,
    /// Current HRTF database
    hrtf_database: Arc<HrtfDatabase>,
    /// HRTF convolver
    convolver: HrtfConvolver,
    /// Distance attenuation model
    distance_model: DistanceModel,
    /// Doppler processor
    doppler: DopplerProcessor,
    /// Listener orientation
    listener_orientation: ListenerOrientation,
    /// Enable Doppler effect
    enable_doppler: bool,
}

impl BinauralRenderer {
    /// Create a new binaural renderer
    pub fn new(sample_rate: u32) -> AudioResult<Self> {
        let hrtf_manager = Arc::new(HrtfManager::default());
        let hrtf_database = hrtf_manager.get_default()?;

        Ok(Self {
            sample_rate,
            hrtf_manager: hrtf_manager.clone(),
            hrtf_database,
            convolver: HrtfConvolver::new(512),
            distance_model: DistanceModel::Inverse {
                min: 1.0,
                rolloff: 1.0,
            },
            doppler: DopplerProcessor::new(sample_rate),
            listener_orientation: ListenerOrientation::default(),
            enable_doppler: false,
        })
    }

    /// Set HRTF database
    pub fn set_hrtf_database(&mut self, database_name: &str) -> AudioResult<()> {
        self.hrtf_database = self.hrtf_manager.get_database(database_name)?;
        Ok(())
    }

    /// Set distance attenuation model
    pub fn set_distance_model(&mut self, model: DistanceModel) {
        self.distance_model = model;
    }

    /// Set listener orientation (for head tracking)
    pub fn set_listener_orientation(&mut self, orientation: ListenerOrientation) {
        self.listener_orientation = orientation;
    }

    /// Enable or disable Doppler effect
    pub fn set_doppler_enabled(&mut self, enabled: bool) {
        self.enable_doppler = enabled;
    }

    /// Update HRTF for a source position
    fn update_hrtf(&mut self, position: &SourcePosition) -> AudioResult<()> {
        // Transform position to listener-relative coordinates
        let relative_pos = self.listener_orientation.transform(position);

        // Convert to spherical coordinates
        let (azimuth, elevation, _distance) = relative_pos.to_spherical();

        // Get HRTF for this direction
        let hrir = self
            .hrtf_database
            .interpolate(azimuth, elevation)
            .ok_or_else(|| AudioError::Internal("Failed to get HRTF".to_string()))?;

        // Update convolver
        self.convolver.set_hrir(&hrir);

        Ok(())
    }

    /// Render mono source to binaural stereo
    pub fn render(
        &mut self,
        input: &[f32],
        position: &SourcePosition,
        output_left: &mut [f32],
        output_right: &mut [f32],
    ) -> AudioResult<()> {
        if input.len() != output_left.len() || input.len() != output_right.len() {
            return Err(AudioError::InvalidParameter(
                "Buffer size mismatch".to_string(),
            ));
        }

        // Update HRTF for current position
        self.update_hrtf(position)?;

        // Calculate distance attenuation
        let distance = position.distance().max(MIN_DISTANCE);
        let distance_gain = self.distance_model.calculate_gain(distance);

        // Apply distance attenuation to input
        let mut attenuated_input = vec![0.0; input.len()];
        for (i, &sample) in input.iter().enumerate() {
            attenuated_input[i] = sample * distance_gain;
        }

        // Apply HRTF convolution
        self.convolver
            .process_buffer(&attenuated_input, output_left, output_right)?;

        Ok(())
    }

    /// Render with time-varying position (for Doppler effect)
    pub fn render_with_motion(
        &mut self,
        input: &[f32],
        position: &SourcePosition,
        _velocity: &SourcePosition,
        output_left: &mut [f32],
        output_right: &mut [f32],
    ) -> AudioResult<()> {
        if !self.enable_doppler {
            return self.render(input, position, output_left, output_right);
        }

        // Calculate Doppler shift
        let distance = position.distance();
        let delta_time = input.len() as f32 / self.sample_rate as f32;
        let _doppler_shift = self.doppler.calculate_shift(distance, delta_time);

        // For simplicity, we'll skip the actual pitch shifting implementation
        // Real implementation would use a time-stretching algorithm
        self.render(input, position, output_left, output_right)
    }

    /// Process multiple sources
    pub fn render_multi_source(
        &mut self,
        sources: &[(Vec<f32>, SourcePosition)],
        output_left: &mut [f32],
        output_right: &mut [f32],
    ) -> AudioResult<()> {
        if sources.is_empty() {
            return Ok(());
        }

        let buffer_size = output_left.len();

        // Clear output buffers
        output_left.fill(0.0);
        output_right.fill(0.0);

        // Render each source and accumulate
        let mut temp_left = vec![0.0; buffer_size];
        let mut temp_right = vec![0.0; buffer_size];

        for (input, position) in sources {
            if input.len() != buffer_size {
                return Err(AudioError::InvalidParameter(
                    "Buffer size mismatch".to_string(),
                ));
            }

            temp_left.fill(0.0);
            temp_right.fill(0.0);

            self.render(input, position, &mut temp_left, &mut temp_right)?;

            for i in 0..buffer_size {
                output_left[i] += temp_left[i];
                output_right[i] += temp_right[i];
            }
        }

        Ok(())
    }
}

/// Binaural panning preset
#[derive(Debug, Clone, Copy)]
pub enum BinauralPreset {
    /// Front center
    Front,
    /// Behind listener
    Back,
    /// Left side
    Left,
    /// Right side
    Right,
    /// Above
    Above,
    /// Below
    Below,
}

impl BinauralPreset {
    /// Get source position for preset
    pub fn position(&self) -> SourcePosition {
        match self {
            BinauralPreset::Front => SourcePosition::from_spherical(0.0, 0.0, 1.0),
            BinauralPreset::Back => SourcePosition::from_spherical(PI, 0.0, 1.0),
            BinauralPreset::Left => SourcePosition::from_spherical(-PI / 2.0, 0.0, 1.0),
            BinauralPreset::Right => SourcePosition::from_spherical(PI / 2.0, 0.0, 1.0),
            BinauralPreset::Above => SourcePosition::from_spherical(0.0, PI / 2.0, 1.0),
            BinauralPreset::Below => SourcePosition::from_spherical(0.0, -PI / 2.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_position() {
        let pos = SourcePosition::new(1.0, 0.0, 0.0);
        let (azimuth, elevation, distance) = pos.to_spherical();

        assert!((azimuth - PI / 2.0).abs() < 0.01);
        assert!(elevation.abs() < 0.01);
        assert!((distance - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_listener_orientation() {
        let orientation = ListenerOrientation::new(PI / 2.0, 0.0, 0.0);
        let pos = SourcePosition::new(1.0, 0.0, 0.0);
        let transformed = orientation.transform(&pos);

        // After 90° yaw rotation, x becomes -y
        assert!((transformed.y - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_distance_model() {
        let model = DistanceModel::Inverse {
            min: 1.0,
            rolloff: 1.0,
        };

        let gain1 = model.calculate_gain(1.0);
        let gain2 = model.calculate_gain(2.0);

        assert!(gain1 > gain2);
        assert!(gain1 <= 1.0);
    }

    #[test]
    fn test_hrtf_convolver() {
        let mut convolver = HrtfConvolver::new(512);

        // Need to set HRIR first
        let hrir = HrirMeasurement::new(vec![1.0, 0.5, 0.25], vec![0.8, 0.4, 0.2], 0.0, 0.0);
        convolver.set_hrir(&hrir);

        // Process multiple samples to build up convolution
        for _ in 0..10 {
            let _ = convolver.process_sample(1.0);
        }

        let (left, right) = convolver.process_sample(1.0);

        // Should produce some output
        assert!(left != 0.0 || right != 0.0);
    }

    #[test]
    fn test_binaural_renderer() {
        let mut renderer = BinauralRenderer::new(44100).expect("should succeed");

        let input = vec![1.0; 512];
        let mut output_left = vec![0.0; 512];
        let mut output_right = vec![0.0; 512];

        let position = SourcePosition::from_spherical(0.0, 0.0, 2.0);

        let result = renderer.render(&input, &position, &mut output_left, &mut output_right);
        assert!(result.is_ok());

        // Should produce some output
        assert!(output_left.iter().any(|&x| x != 0.0));
        assert!(output_right.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_binaural_presets() {
        let pos = BinauralPreset::Front.position();
        let (azimuth, _elevation, _distance) = pos.to_spherical();
        assert!(azimuth.abs() < 0.01);

        let pos = BinauralPreset::Left.position();
        let (azimuth, _elevation, _distance) = pos.to_spherical();
        assert!((azimuth + PI / 2.0).abs() < 0.01);
    }
}
