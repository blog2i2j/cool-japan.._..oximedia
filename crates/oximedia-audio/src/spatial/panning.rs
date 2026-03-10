//! Panning algorithms for spatial audio.
//!
//! This module provides various panning techniques including:
//! - Stereo panning (constant power, linear)
//! - VBAP (Vector Base Amplitude Panning)
//! - DBAP (Distance-Based Amplitude Panning)
//! - Surround panning (5.1, 7.1, etc.)

use crate::{AudioError, AudioResult};
use std::f32::consts::PI;

/// Stereo panning law
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PanLaw {
    /// Constant power (-3dB at center)
    ConstantPower,
    /// Linear (no compensation)
    Linear,
    /// -4.5dB at center
    Minus4_5dB,
    /// -6dB at center
    Minus6dB,
}

/// Stereo panner with configurable pan law
#[derive(Debug, Clone)]
pub struct StereoPanner {
    /// Current pan position (-1.0 = left, 0.0 = center, 1.0 = right)
    position: f32,
    /// Panning law
    law: PanLaw,
}

impl StereoPanner {
    /// Create a new stereo panner
    pub fn new(law: PanLaw) -> Self {
        Self { position: 0.0, law }
    }

    /// Set pan position (-1.0 to 1.0)
    pub fn set_position(&mut self, position: f32) {
        self.position = position.clamp(-1.0, 1.0);
    }

    /// Get current pan position
    pub fn position(&self) -> f32 {
        self.position
    }

    /// Calculate stereo gains for current position
    pub fn calculate_gains(&self) -> (f32, f32) {
        // Normalize position to 0.0-1.0 range
        let norm_pos = (self.position + 1.0) / 2.0;

        match self.law {
            PanLaw::ConstantPower => {
                // Constant power panning (-3dB at center)
                let angle = norm_pos * PI / 2.0;
                let left = angle.cos();
                let right = angle.sin();
                (left, right)
            }
            PanLaw::Linear => {
                // Linear panning (no compensation)
                let left = 1.0 - norm_pos;
                let right = norm_pos;
                (left, right)
            }
            PanLaw::Minus4_5dB => {
                // -4.5dB at center
                let left = (1.0 - norm_pos).sqrt();
                let right = norm_pos.sqrt();
                let compensation = 2.0_f32.sqrt();
                (left * compensation, right * compensation)
            }
            PanLaw::Minus6dB => {
                // -6dB at center (simple linear)
                let left = 1.0 - norm_pos;
                let right = norm_pos;
                (left, right)
            }
        }
    }

    /// Process mono to stereo
    pub fn process_mono(
        &self,
        input: &[f32],
        output_left: &mut [f32],
        output_right: &mut [f32],
    ) -> AudioResult<()> {
        if input.len() != output_left.len() || input.len() != output_right.len() {
            return Err(AudioError::InvalidParameter(
                "Buffer size mismatch".to_string(),
            ));
        }

        let (left_gain, right_gain) = self.calculate_gains();

        for i in 0..input.len() {
            output_left[i] = input[i] * left_gain;
            output_right[i] = input[i] * right_gain;
        }

        Ok(())
    }
}

/// 3D speaker position
#[derive(Debug, Clone, Copy)]
pub struct SpeakerPosition {
    /// X coordinate (left/right)
    pub x: f32,
    /// Y coordinate (front/back)
    pub y: f32,
    /// Z coordinate (up/down)
    pub z: f32,
}

impl SpeakerPosition {
    /// Create from Cartesian coordinates
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

    /// Convert to spherical coordinates (azimuth, elevation, distance)
    pub fn to_spherical(&self) -> (f32, f32, f32) {
        let distance = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        let azimuth = self.x.atan2(self.y);
        let elevation = (self.z / distance).asin();
        (azimuth, elevation, distance)
    }

    /// Calculate distance to another position
    pub fn distance_to(&self, other: &SpeakerPosition) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Normalize to unit vector
    pub fn normalize(&self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if len > 0.0 {
            Self {
                x: self.x / len,
                y: self.y / len,
                z: self.z / len,
            }
        } else {
            *self
        }
    }

    /// Dot product with another position
    pub fn dot(&self, other: &SpeakerPosition) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

/// Standard speaker layouts
pub struct SpeakerLayout;

impl SpeakerLayout {
    /// Stereo layout (L, R)
    pub fn stereo() -> Vec<SpeakerPosition> {
        vec![
            SpeakerPosition::from_spherical(-30.0_f32.to_radians(), 0.0, 1.0), // Left
            SpeakerPosition::from_spherical(30.0_f32.to_radians(), 0.0, 1.0),  // Right
        ]
    }

    /// 5.1 surround layout (L, R, C, LFE, LS, RS)
    pub fn surround_5_1() -> Vec<SpeakerPosition> {
        vec![
            SpeakerPosition::from_spherical(-30.0_f32.to_radians(), 0.0, 1.0), // L
            SpeakerPosition::from_spherical(30.0_f32.to_radians(), 0.0, 1.0),  // R
            SpeakerPosition::from_spherical(0.0, 0.0, 1.0),                    // C
            SpeakerPosition::from_spherical(0.0, -15.0_f32.to_radians(), 1.0), // LFE
            SpeakerPosition::from_spherical(-110.0_f32.to_radians(), 0.0, 1.0), // LS
            SpeakerPosition::from_spherical(110.0_f32.to_radians(), 0.0, 1.0), // RS
        ]
    }

    /// 7.1 surround layout (L, R, C, LFE, LS, RS, LB, RB)
    pub fn surround_7_1() -> Vec<SpeakerPosition> {
        vec![
            SpeakerPosition::from_spherical(-30.0_f32.to_radians(), 0.0, 1.0), // L
            SpeakerPosition::from_spherical(30.0_f32.to_radians(), 0.0, 1.0),  // R
            SpeakerPosition::from_spherical(0.0, 0.0, 1.0),                    // C
            SpeakerPosition::from_spherical(0.0, -15.0_f32.to_radians(), 1.0), // LFE
            SpeakerPosition::from_spherical(-90.0_f32.to_radians(), 0.0, 1.0), // LS
            SpeakerPosition::from_spherical(90.0_f32.to_radians(), 0.0, 1.0),  // RS
            SpeakerPosition::from_spherical(-150.0_f32.to_radians(), 0.0, 1.0), // LB
            SpeakerPosition::from_spherical(150.0_f32.to_radians(), 0.0, 1.0), // RB
        ]
    }
}

/// VBAP (Vector Base Amplitude Panning) implementation
pub struct VbapPanner {
    /// Speaker positions
    speakers: Vec<SpeakerPosition>,
    /// Precomputed speaker triplets for 3D panning
    triplets: Vec<(usize, usize, usize)>,
}

impl VbapPanner {
    /// Create a new VBAP panner with speaker layout
    pub fn new(speakers: Vec<SpeakerPosition>) -> AudioResult<Self> {
        if speakers.len() < 2 {
            return Err(AudioError::InvalidParameter(
                "Need at least 2 speakers".to_string(),
            ));
        }

        let triplets = Self::compute_triplets(&speakers);

        Ok(Self { speakers, triplets })
    }

    /// Compute valid speaker triplets for 3D VBAP
    fn compute_triplets(speakers: &[SpeakerPosition]) -> Vec<(usize, usize, usize)> {
        let mut triplets = Vec::new();
        let n = speakers.len();

        // For simplicity, use all combinations of 3 speakers
        // In a real implementation, only use triplets that form valid triangles
        if n >= 3 {
            for i in 0..n {
                for j in (i + 1)..n {
                    for k in (j + 1)..n {
                        triplets.push((i, j, k));
                    }
                }
            }
        }

        triplets
    }

    /// Calculate gains for a sound source at given position
    pub fn calculate_gains(&self, source_pos: &SpeakerPosition) -> Vec<f32> {
        let mut gains = vec![0.0; self.speakers.len()];

        if self.speakers.len() == 2 {
            // Simple stereo panning
            let (azimuth, _, _) = source_pos.to_spherical();
            let pan = (azimuth / (60.0_f32.to_radians())).clamp(-1.0, 1.0);
            let normalized = (pan + 1.0) / 2.0;
            gains[0] = (1.0 - normalized).sqrt();
            gains[1] = normalized.sqrt();
        } else {
            // Find best triplet and calculate gains
            let normalized_source = source_pos.normalize();

            let mut best_triplet = None;
            let mut best_gains = vec![0.0; 3];

            for &(i, j, k) in &self.triplets {
                if let Some(triplet_gains) = self.calculate_triplet_gains(
                    &normalized_source,
                    &self.speakers[i],
                    &self.speakers[j],
                    &self.speakers[k],
                ) {
                    best_triplet = Some((i, j, k));
                    best_gains = triplet_gains;
                    break;
                }
            }

            if let Some((i, j, k)) = best_triplet {
                gains[i] = best_gains[0];
                gains[j] = best_gains[1];
                gains[k] = best_gains[2];
            }
        }

        // Normalize gains
        let sum: f32 = gains.iter().map(|&g| g * g).sum();
        if sum > 0.0 {
            let norm = sum.sqrt();
            for gain in &mut gains {
                *gain /= norm;
            }
        }

        gains
    }

    /// Calculate gains for a speaker triplet
    fn calculate_triplet_gains(
        &self,
        source: &SpeakerPosition,
        s1: &SpeakerPosition,
        s2: &SpeakerPosition,
        s3: &SpeakerPosition,
    ) -> Option<Vec<f32>> {
        // Solve: source = g1*s1 + g2*s2 + g3*s3
        // Using Cramer's rule

        let det = s1.x * (s2.y * s3.z - s2.z * s3.y) - s1.y * (s2.x * s3.z - s2.z * s3.x)
            + s1.z * (s2.x * s3.y - s2.y * s3.x);

        if det.abs() < 1e-6 {
            return None;
        }

        let g1 = (source.x * (s2.y * s3.z - s2.z * s3.y) - source.y * (s2.x * s3.z - s2.z * s3.x)
            + source.z * (s2.x * s3.y - s2.y * s3.x))
            / det;

        let g2 = (s1.x * (source.y * s3.z - source.z * s3.y)
            - s1.y * (source.x * s3.z - source.z * s3.x)
            + s1.z * (source.x * s3.y - source.y * s3.x))
            / det;

        let g3 = (s1.x * (s2.y * source.z - s2.z * source.y)
            - s1.y * (s2.x * source.z - s2.z * source.x)
            + s1.z * (s2.x * source.y - s2.y * source.x))
            / det;

        // Check if gains are valid (all non-negative)
        if g1 >= 0.0 && g2 >= 0.0 && g3 >= 0.0 {
            Some(vec![g1, g2, g3])
        } else {
            None
        }
    }

    /// Process audio with VBAP
    pub fn process(
        &self,
        input: &[f32],
        source_pos: &SpeakerPosition,
        outputs: &mut [Vec<f32>],
    ) -> AudioResult<()> {
        if outputs.len() != self.speakers.len() {
            return Err(AudioError::InvalidParameter(
                "Output count must match speaker count".to_string(),
            ));
        }

        for output in outputs.iter() {
            if output.len() != input.len() {
                return Err(AudioError::InvalidParameter(
                    "Buffer size mismatch".to_string(),
                ));
            }
        }

        let gains = self.calculate_gains(source_pos);

        for (speaker_idx, output) in outputs.iter_mut().enumerate() {
            let gain = gains[speaker_idx];
            for (i, &sample) in input.iter().enumerate() {
                output[i] = sample * gain;
            }
        }

        Ok(())
    }
}

/// DBAP (Distance-Based Amplitude Panning) implementation
pub struct DbapPanner {
    /// Speaker positions
    speakers: Vec<SpeakerPosition>,
    /// Rolloff exponent (typically 1.0 to 2.0)
    rolloff: f32,
}

impl DbapPanner {
    /// Create a new DBAP panner
    pub fn new(speakers: Vec<SpeakerPosition>, rolloff: f32) -> AudioResult<Self> {
        if speakers.is_empty() {
            return Err(AudioError::InvalidParameter(
                "Need at least 1 speaker".to_string(),
            ));
        }

        Ok(Self { speakers, rolloff })
    }

    /// Calculate gains based on inverse distance
    pub fn calculate_gains(&self, source_pos: &SpeakerPosition) -> Vec<f32> {
        let mut gains = Vec::with_capacity(self.speakers.len());
        let mut total_weight = 0.0;

        // Calculate inverse distance weights
        for speaker in &self.speakers {
            let distance = source_pos.distance_to(speaker).max(0.1); // Avoid division by zero
            let weight = 1.0 / distance.powf(self.rolloff);
            gains.push(weight);
            total_weight += weight * weight;
        }

        // Normalize to maintain constant power
        let norm = total_weight.sqrt();
        if norm > 0.0 {
            for gain in &mut gains {
                *gain /= norm;
            }
        }

        gains
    }

    /// Process audio with DBAP
    pub fn process(
        &self,
        input: &[f32],
        source_pos: &SpeakerPosition,
        outputs: &mut [Vec<f32>],
    ) -> AudioResult<()> {
        if outputs.len() != self.speakers.len() {
            return Err(AudioError::InvalidParameter(
                "Output count must match speaker count".to_string(),
            ));
        }

        for output in outputs.iter() {
            if output.len() != input.len() {
                return Err(AudioError::InvalidParameter(
                    "Buffer size mismatch".to_string(),
                ));
            }
        }

        let gains = self.calculate_gains(source_pos);

        for (speaker_idx, output) in outputs.iter_mut().enumerate() {
            let gain = gains[speaker_idx];
            for (i, &sample) in input.iter().enumerate() {
                output[i] = sample * gain;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stereo_panner() {
        let mut panner = StereoPanner::new(PanLaw::ConstantPower);
        panner.set_position(0.0);

        let (left, right) = panner.calculate_gains();
        assert!((left - 0.707).abs() < 0.01);
        assert!((right - 0.707).abs() < 0.01);

        panner.set_position(-1.0);
        let (left, right) = panner.calculate_gains();
        assert!(left > 0.99);
        assert!(right < 0.01);
    }

    #[test]
    fn test_speaker_position() {
        let pos = SpeakerPosition::new(1.0, 0.0, 0.0);
        let (azimuth, elevation, distance) = pos.to_spherical();

        assert!((azimuth - PI / 2.0).abs() < 0.01);
        assert!(elevation.abs() < 0.01);
        assert!((distance - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_vbap_stereo() {
        let speakers = SpeakerLayout::stereo();
        let panner = VbapPanner::new(speakers).expect("should succeed");

        let source = SpeakerPosition::from_spherical(0.0, 0.0, 1.0);
        let gains = panner.calculate_gains(&source);

        assert_eq!(gains.len(), 2);
        assert!(gains[0] > 0.0);
        assert!(gains[1] > 0.0);
    }

    #[test]
    fn test_dbap() {
        let speakers = SpeakerLayout::stereo();
        let panner = DbapPanner::new(speakers, 1.0).expect("should succeed");

        let source = SpeakerPosition::from_spherical(0.0, 0.0, 1.0);
        let gains = panner.calculate_gains(&source);

        assert_eq!(gains.len(), 2);
        assert!(gains[0] > 0.0);
        assert!(gains[1] > 0.0);
    }

    #[test]
    fn test_process_mono_to_stereo() {
        let panner = StereoPanner::new(PanLaw::ConstantPower);
        let input = vec![1.0; 100];
        let mut left = vec![0.0; 100];
        let mut right = vec![0.0; 100];

        let result = panner.process_mono(&input, &mut left, &mut right);
        assert!(result.is_ok());

        assert!(left[0] > 0.0);
        assert!(right[0] > 0.0);
    }
}
