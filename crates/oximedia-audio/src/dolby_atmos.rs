//! Dolby Atmos object metadata parsing for spatial audio rendering.
//!
//! This module parses and represents Dolby Atmos audio object metadata as used
//! in immersive audio production. The metadata drives spatial audio rendering
//! by specifying the 3D position, size, and properties of each audio object.
//!
//! # Dolby Atmos Overview
//!
//! Dolby Atmos uses **audio objects** rather than fixed channel positions.
//! Each object has:
//! - A 3D position (x, y, z) in normalized space [0, 1]³
//! - A size (width, height, depth) for spread rendering
//! - A gain/volume value
//! - Optional automation data for moving objects
//!
//! # Metadata Format
//!
//! Atmos metadata is typically carried as:
//! 1. **ADMS (Audio Definition Model Serialization)** in MXF/BWF files
//! 2. **Dolby Metadata** in Dolby Digital Plus (E-AC-3) bitstreams
//! 3. **IAB (Immersive Audio Bitstream)** SMPTE ST 2098-2 format
//!
//! This module parses the binary metadata block format used in broadcast
//! workflows, as well as a structured builder API for software renderers.
//!
//! # Standards
//!
//! - SMPTE ST 2098-2: Immersive Audio Bitstream
//! - Dolby Atmos Authoring Specification v1.2
//! - ITU-R BS.2051-3: Advanced sound systems

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// Error type for Dolby Atmos metadata operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AtmosError {
    /// Insufficient data for parsing.
    #[error("Need more data: expected {expected} bytes, got {got}")]
    NeedMoreData {
        /// Expected byte count.
        expected: usize,
        /// Actual byte count received.
        got: usize,
    },

    /// Invalid metadata version.
    #[error("Unsupported metadata version: {0}")]
    UnsupportedVersion(u8),

    /// Object ID out of range.
    #[error("Object ID {0} out of range (max {1})")]
    ObjectIdOutOfRange(u8, u8),

    /// Invalid 3D coordinates.
    #[error("Invalid coordinates: ({x}, {y}, {z})")]
    InvalidCoordinates {
        /// X coordinate.
        x: f32,
        /// Y coordinate.
        y: f32,
        /// Z coordinate.
        z: f32,
    },

    /// Metadata block CRC mismatch.
    #[error("CRC mismatch: expected {expected:#04x}, got {computed:#04x}")]
    CrcMismatch {
        /// Expected CRC value.
        expected: u8,
        /// Computed CRC value.
        computed: u8,
    },
}

/// Type alias for Atmos metadata results.
pub type AtmosResult<T> = Result<T, AtmosError>;

/// 3D position of an audio object in normalized room coordinates.
///
/// The coordinate system follows Dolby convention:
/// - x: left (0.0) to right (1.0)
/// - y: bottom (0.0) to top (1.0)
/// - z: front (0.0) to back (1.0)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectPosition {
    /// Horizontal position (left=0.0, center=0.5, right=1.0).
    pub x: f32,
    /// Vertical position (bottom=0.0, center=0.5, top=1.0).
    pub y: f32,
    /// Depth position (front=0.0, center=0.5, back=1.0).
    pub z: f32,
}

impl ObjectPosition {
    /// Create a new 3D position, clamped to [0, 1]³.
    #[must_use]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            x: x.clamp(0.0, 1.0),
            y: y.clamp(0.0, 1.0),
            z: z.clamp(0.0, 1.0),
        }
    }

    /// Center of room position.
    #[must_use]
    pub fn center() -> Self {
        Self::new(0.5, 0.5, 0.5)
    }

    /// Front center (default speaker position).
    #[must_use]
    pub fn front_center() -> Self {
        Self::new(0.5, 0.5, 0.0)
    }

    /// Convert to Cartesian coordinates in range [-1, 1]³.
    #[must_use]
    pub fn to_cartesian(self) -> (f32, f32, f32) {
        (
            self.x * 2.0 - 1.0, // [-1, 1]
            self.y * 2.0 - 1.0,
            self.z * 2.0 - 1.0,
        )
    }

    /// Compute distance from the origin (center).
    #[must_use]
    pub fn distance_from_center(self) -> f32 {
        let (cx, cy, cz) = self.to_cartesian();
        (cx * cx + cy * cy + cz * cz).sqrt()
    }

    /// Interpolate linearly between two positions.
    #[must_use]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self::new(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t,
        )
    }

    /// Parse a position from 3 bytes (Q8 fixed-point each, range 0–255 → 0.0–1.0).
    #[must_use]
    pub fn from_bytes(data: &[u8]) -> Self {
        let x = data.first().copied().unwrap_or(128) as f32 / 255.0;
        let y = data.get(1).copied().unwrap_or(128) as f32 / 255.0;
        let z = data.get(2).copied().unwrap_or(0) as f32 / 255.0;
        Self::new(x, y, z)
    }

    /// Serialize to 3 bytes.
    #[must_use]
    pub fn to_bytes(self) -> [u8; 3] {
        [
            (self.x * 255.0).round() as u8,
            (self.y * 255.0).round() as u8,
            (self.z * 255.0).round() as u8,
        ]
    }
}

impl Default for ObjectPosition {
    fn default() -> Self {
        Self::front_center()
    }
}

/// Object size for spread rendering.
///
/// A point source has size (0, 0, 0). Larger values spread the sound
/// over a wider area, useful for ambient sounds or large virtual sources.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectSize {
    /// Width in [0, 1] (0 = point source, 1 = full width).
    pub width: f32,
    /// Height in [0, 1].
    pub height: f32,
    /// Depth in [0, 1].
    pub depth: f32,
}

impl ObjectSize {
    /// Create a new size, clamped to [0, 1]³.
    #[must_use]
    pub fn new(width: f32, height: f32, depth: f32) -> Self {
        Self {
            width: width.clamp(0.0, 1.0),
            height: height.clamp(0.0, 1.0),
            depth: depth.clamp(0.0, 1.0),
        }
    }

    /// Point source (zero size).
    #[must_use]
    pub fn point() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Full-room ambient.
    #[must_use]
    pub fn ambient() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }
}

impl Default for ObjectSize {
    fn default() -> Self {
        Self::point()
    }
}

/// Rendering intent for an audio object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderingIntent {
    /// Direct rendering to speakers.
    Direct,
    /// Binaural rendering for headphones.
    Binaural,
    /// Diffuse ambient rendering.
    Diffuse,
    /// Bed-based rendering (maps to fixed channel layout).
    Bed,
}

impl RenderingIntent {
    /// Parse from 2-bit field.
    fn from_bits(bits: u8) -> Self {
        match bits & 0x03 {
            0 => Self::Direct,
            1 => Self::Binaural,
            2 => Self::Diffuse,
            3 => Self::Bed,
            _ => Self::Direct,
        }
    }

    /// Serialize to 2 bits.
    fn to_bits(self) -> u8 {
        match self {
            Self::Direct => 0,
            Self::Binaural => 1,
            Self::Diffuse => 2,
            Self::Bed => 3,
        }
    }
}

/// Automation keyframe for a moving audio object.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutomationPoint {
    /// Timestamp in seconds.
    pub time_secs: f64,
    /// Position at this keyframe.
    pub position: ObjectPosition,
    /// Gain at this keyframe (linear, 0.0–4.0, 1.0 = 0 dB).
    pub gain: f32,
    /// Interpolation type to the next keyframe.
    pub interpolation: AutomationInterpolation,
}

/// Interpolation type for automation curves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutomationInterpolation {
    /// Step (hold until next keyframe).
    Step,
    /// Linear interpolation.
    Linear,
    /// Smooth cubic spline.
    Smooth,
}

impl AutomationInterpolation {
    fn from_bits(bits: u8) -> Self {
        match bits & 0x03 {
            0 => Self::Step,
            1 => Self::Linear,
            2 => Self::Smooth,
            _ => Self::Linear,
        }
    }
}

/// A single Dolby Atmos audio object and its rendering metadata.
#[derive(Debug, Clone)]
pub struct AtmosObject {
    /// Object identifier (0–127 for IAB, 0–255 for Atmos).
    pub id: u8,
    /// Channel/track index this object reads from.
    pub channel_index: u8,
    /// Current 3D position.
    pub position: ObjectPosition,
    /// Object spatial size.
    pub size: ObjectSize,
    /// Gain/volume (linear, 1.0 = 0 dB).
    pub gain: f32,
    /// Rendering intent.
    pub rendering_intent: RenderingIntent,
    /// Whether the object snaps to nearest bed channel.
    pub snap_to_speaker: bool,
    /// Whether the object is enabled.
    pub enabled: bool,
    /// Optional automation track.
    pub automation: Vec<AutomationPoint>,
    /// Arbitrary metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

impl AtmosObject {
    /// Create a new object at the front center.
    #[must_use]
    pub fn new(id: u8, channel_index: u8) -> Self {
        Self {
            id,
            channel_index,
            position: ObjectPosition::front_center(),
            size: ObjectSize::point(),
            gain: 1.0,
            rendering_intent: RenderingIntent::Direct,
            snap_to_speaker: false,
            enabled: true,
            automation: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the 3D position.
    #[must_use]
    pub fn with_position(mut self, position: ObjectPosition) -> Self {
        self.position = position;
        self
    }

    /// Set the spatial size.
    #[must_use]
    pub fn with_size(mut self, size: ObjectSize) -> Self {
        self.size = size;
        self
    }

    /// Set the gain in dB.
    #[must_use]
    pub fn with_gain_db(mut self, gain_db: f32) -> Self {
        self.gain = 10.0_f32.powf(gain_db / 20.0);
        self
    }

    /// Get the gain in dB.
    #[must_use]
    pub fn gain_db(&self) -> f32 {
        if self.gain > 0.0 {
            20.0 * self.gain.log10()
        } else {
            f32::NEG_INFINITY
        }
    }

    /// Add an automation keyframe.
    pub fn add_automation_point(&mut self, point: AutomationPoint) {
        // Insert in time order
        let pos = self
            .automation
            .binary_search_by(|p| {
                p.time_secs
                    .partial_cmp(&point.time_secs)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|e| e);
        self.automation.insert(pos, point);
    }

    /// Evaluate position at a given time using automation keyframes.
    ///
    /// Returns the static position if no automation is defined.
    #[must_use]
    pub fn position_at(&self, time_secs: f64) -> ObjectPosition {
        if self.automation.is_empty() {
            return self.position;
        }

        // Find surrounding keyframes
        let after_pos = self
            .automation
            .iter()
            .position(|p| p.time_secs > time_secs);

        match after_pos {
            None => self.automation.last().map(|p| p.position).unwrap_or(self.position),
            Some(0) => self.automation[0].position,
            Some(i) => {
                let before = &self.automation[i - 1];
                let after = &self.automation[i];
                let t = (time_secs - before.time_secs) / (after.time_secs - before.time_secs);
                let t = t.clamp(0.0, 1.0) as f32;

                match before.interpolation {
                    AutomationInterpolation::Step => before.position,
                    AutomationInterpolation::Linear => before.position.lerp(after.position, t),
                    AutomationInterpolation::Smooth => {
                        // Smooth step: 3t² - 2t³
                        let smooth_t = t * t * (3.0 - 2.0 * t);
                        before.position.lerp(after.position, smooth_t)
                    }
                }
            }
        }
    }

    /// Evaluate gain at a given time.
    #[must_use]
    pub fn gain_at(&self, time_secs: f64) -> f32 {
        if self.automation.is_empty() {
            return self.gain;
        }

        let after_pos = self
            .automation
            .iter()
            .position(|p| p.time_secs > time_secs);

        match after_pos {
            None => self.automation.last().map(|p| p.gain).unwrap_or(self.gain),
            Some(0) => self.automation[0].gain,
            Some(i) => {
                let before = &self.automation[i - 1];
                let after = &self.automation[i];
                let t = ((time_secs - before.time_secs) / (after.time_secs - before.time_secs))
                    .clamp(0.0, 1.0) as f32;
                before.gain + (after.gain - before.gain) * t
            }
        }
    }
}

/// Dolby Atmos program (a complete immersive audio scene).
#[derive(Debug, Clone)]
pub struct AtmosProgram {
    /// Program ID.
    pub id: u32,
    /// Program name.
    pub name: String,
    /// Audio objects in this program.
    pub objects: Vec<AtmosObject>,
    /// Bed channel layout (fixed-position channels).
    pub bed_layout: BedLayout,
    /// Frame rate (for timecode synchronization).
    pub frame_rate: f32,
    /// Sample rate.
    pub sample_rate: u32,
    /// Total duration in seconds.
    pub duration_secs: f64,
}

/// Bed channel layout (fixed-position channels in the Atmos mix).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BedLayout {
    /// 5.1 bed (L, R, C, LFE, Ls, Rs).
    FivePointOne,
    /// 7.1 bed (L, R, C, LFE, Lss, Rss, Lrs, Rrs).
    SevenPointOne,
    /// 7.1.2 bed with overhead pairs.
    SevenPointOnePointTwo,
    /// 7.1.4 bed with four overhead channels.
    SevenPointOnePointFour,
    /// Custom bed layout.
    Custom(u8),
}

impl BedLayout {
    /// Number of bed channels.
    #[must_use]
    pub fn channel_count(self) -> u8 {
        match self {
            Self::FivePointOne => 6,
            Self::SevenPointOne => 8,
            Self::SevenPointOnePointTwo => 10,
            Self::SevenPointOnePointFour => 12,
            Self::Custom(n) => n,
        }
    }
}

impl AtmosProgram {
    /// Create a new Atmos program.
    #[must_use]
    pub fn new(id: u32, name: impl Into<String>, sample_rate: u32) -> Self {
        Self {
            id,
            name: name.into(),
            objects: Vec::new(),
            bed_layout: BedLayout::SevenPointOne,
            frame_rate: 29.97,
            sample_rate,
            duration_secs: 0.0,
        }
    }

    /// Add an audio object.
    pub fn add_object(&mut self, object: AtmosObject) {
        self.objects.push(object);
    }

    /// Get object by ID.
    #[must_use]
    pub fn get_object(&self, id: u8) -> Option<&AtmosObject> {
        self.objects.iter().find(|o| o.id == id)
    }

    /// Get mutable object by ID.
    pub fn get_object_mut(&mut self, id: u8) -> Option<&mut AtmosObject> {
        self.objects.iter_mut().find(|o| o.id == id)
    }

    /// Total number of channels (bed + objects).
    #[must_use]
    pub fn total_channels(&self) -> usize {
        self.bed_layout.channel_count() as usize + self.objects.len()
    }
}

/// Metadata block parser for Dolby Atmos binary format.
///
/// Parses the binary IAB (Immersive Audio Bitstream) frame structure
/// as specified in SMPTE ST 2098-2.
pub struct AtmosMetadataParser;

/// Header of an IAB metadata frame.
#[derive(Debug, Clone)]
pub struct IabFrameHeader {
    /// IAB version (should be 1).
    pub version: u8,
    /// Number of audio samples in this frame.
    pub num_samples: u32,
    /// Number of audio bed definitions.
    pub num_bed_definitions: u8,
    /// Number of audio objects.
    pub num_objects: u8,
    /// Number of user data elements.
    pub num_user_data: u8,
}

impl AtmosMetadataParser {
    /// Parse an IAB frame header from bytes.
    ///
    /// IAB frame layout (simplified):
    /// - byte 0: version
    /// - bytes 1-4: num_samples (big-endian u32)
    /// - byte 5: num_bed_definitions
    /// - byte 6: num_objects
    /// - byte 7: num_user_data
    ///
    /// # Errors
    ///
    /// Returns error if data is too short or version is unsupported.
    pub fn parse_iab_header(data: &[u8]) -> AtmosResult<IabFrameHeader> {
        const HEADER_SIZE: usize = 8;
        if data.len() < HEADER_SIZE {
            return Err(AtmosError::NeedMoreData {
                expected: HEADER_SIZE,
                got: data.len(),
            });
        }

        let version = data[0];
        if version != 1 {
            return Err(AtmosError::UnsupportedVersion(version));
        }

        let num_samples = u32::from_be_bytes([data[1], data[2], data[3], data[4]]);
        let num_bed_definitions = data[5];
        let num_objects = data[6];
        let num_user_data = data[7];

        Ok(IabFrameHeader {
            version,
            num_samples,
            num_bed_definitions,
            num_objects,
            num_user_data,
        })
    }

    /// Parse object metadata from an IAB element.
    ///
    /// Each object element contains:
    /// - byte 0: object_id
    /// - byte 1: channel_index
    /// - bytes 2-4: position (x, y, z as Q8)
    /// - bytes 5-7: size (width, height, depth as Q8)
    /// - byte 8: gain (Q8, 0=0 dB, 255=+12 dB range)
    /// - byte 9: flags (enabled, snap, rendering_intent bits)
    ///
    /// # Errors
    ///
    /// Returns error if data is too short.
    pub fn parse_object_element(data: &[u8]) -> AtmosResult<AtmosObject> {
        const ELEMENT_SIZE: usize = 10;
        if data.len() < ELEMENT_SIZE {
            return Err(AtmosError::NeedMoreData {
                expected: ELEMENT_SIZE,
                got: data.len(),
            });
        }

        let id = data[0];
        let channel_index = data[1];
        let position = ObjectPosition::from_bytes(&data[2..5]);
        let size = ObjectSize::new(
            data[5] as f32 / 255.0,
            data[6] as f32 / 255.0,
            data[7] as f32 / 255.0,
        );
        // Gain: Q8, 0=silent, 128=0 dB (unity), 255=+6 dB
        let gain = if data[8] == 0 {
            0.0
        } else {
            10.0_f32.powf((data[8] as f32 / 128.0 - 1.0) * 6.0 / 20.0)
        };

        let flags = data[9];
        let enabled = (flags & 0x80) != 0;
        let snap_to_speaker = (flags & 0x40) != 0;
        let rendering_intent = RenderingIntent::from_bits((flags >> 4) & 0x03);

        let mut object = AtmosObject::new(id, channel_index);
        object.position = position;
        object.size = size;
        object.gain = gain;
        object.enabled = enabled;
        object.snap_to_speaker = snap_to_speaker;
        object.rendering_intent = rendering_intent;

        Ok(object)
    }

    /// Serialize an object to its binary element representation.
    #[must_use]
    pub fn serialize_object(obj: &AtmosObject) -> Vec<u8> {
        let mut data = Vec::with_capacity(10);
        data.push(obj.id);
        data.push(obj.channel_index);
        data.extend_from_slice(&obj.position.to_bytes());
        data.push((obj.size.width * 255.0).round() as u8);
        data.push((obj.size.height * 255.0).round() as u8);
        data.push((obj.size.depth * 255.0).round() as u8);

        // Gain: unity = 128
        let gain_encoded = if obj.gain <= 0.0 {
            0u8
        } else {
            let db = 20.0 * obj.gain.log10();
            ((db / 6.0 + 1.0) * 128.0).clamp(0.0, 255.0).round() as u8
        };
        data.push(gain_encoded);

        let mut flags = 0u8;
        if obj.enabled {
            flags |= 0x80;
        }
        if obj.snap_to_speaker {
            flags |= 0x40;
        }
        flags |= (obj.rendering_intent.to_bits() & 0x03) << 4;
        data.push(flags);

        data
    }

    /// Parse a complete Atmos metadata block.
    ///
    /// # Errors
    ///
    /// Returns error if parsing fails.
    pub fn parse_metadata_block(data: &[u8]) -> AtmosResult<Vec<AtmosObject>> {
        let header = Self::parse_iab_header(data)?;
        let mut objects = Vec::with_capacity(header.num_objects as usize);
        let element_size = 10;
        let offset = 8; // header size

        for i in 0..header.num_objects as usize {
            let start = offset + i * element_size;
            let end = start + element_size;
            if end > data.len() {
                break;
            }
            let object = Self::parse_object_element(&data[start..end])?;
            objects.push(object);
        }

        Ok(objects)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_position_new() {
        let pos = ObjectPosition::new(0.5, 0.5, 0.0);
        assert_eq!(pos.x, 0.5);
        assert_eq!(pos.y, 0.5);
        assert_eq!(pos.z, 0.0);
    }

    #[test]
    fn test_object_position_clamped() {
        let pos = ObjectPosition::new(2.0, -1.0, 0.5);
        assert_eq!(pos.x, 1.0);
        assert_eq!(pos.y, 0.0);
        assert_eq!(pos.z, 0.5);
    }

    #[test]
    fn test_object_position_to_cartesian() {
        let pos = ObjectPosition::new(1.0, 0.0, 0.5);
        let (x, y, z) = pos.to_cartesian();
        assert!((x - 1.0).abs() < 1e-5);
        assert!((y + 1.0).abs() < 1e-5);
        assert!(z.abs() < 1e-5);
    }

    #[test]
    fn test_object_position_lerp() {
        let a = ObjectPosition::new(0.0, 0.0, 0.0);
        let b = ObjectPosition::new(1.0, 1.0, 1.0);
        let mid = a.lerp(b, 0.5);
        assert!((mid.x - 0.5).abs() < 1e-5);
        assert!((mid.y - 0.5).abs() < 1e-5);
        assert!((mid.z - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_position_bytes_roundtrip() {
        let pos = ObjectPosition::new(0.5, 0.25, 0.75);
        let bytes = pos.to_bytes();
        let parsed = ObjectPosition::from_bytes(&bytes);
        // Allow for quantization error (1/255 ≈ 0.004)
        assert!((pos.x - parsed.x).abs() < 0.005);
        assert!((pos.y - parsed.y).abs() < 0.005);
        assert!((pos.z - parsed.z).abs() < 0.005);
    }

    #[test]
    fn test_object_size_defaults() {
        let size = ObjectSize::default();
        assert_eq!(size, ObjectSize::point());
        assert_eq!(size.width, 0.0);
    }

    #[test]
    fn test_atmos_object_gain_db() {
        let obj = AtmosObject::new(0, 0).with_gain_db(0.0);
        assert!((obj.gain - 1.0).abs() < 1e-5);
        assert!((obj.gain_db() - 0.0).abs() < 1e-3);
    }

    #[test]
    fn test_atmos_object_gain_db_positive() {
        let obj = AtmosObject::new(0, 0).with_gain_db(6.0);
        assert!((obj.gain_db() - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_atmos_object_automation_lerp() {
        let mut obj = AtmosObject::new(0, 0);
        obj.add_automation_point(AutomationPoint {
            time_secs: 0.0,
            position: ObjectPosition::new(0.0, 0.5, 0.0),
            gain: 1.0,
            interpolation: AutomationInterpolation::Linear,
        });
        obj.add_automation_point(AutomationPoint {
            time_secs: 1.0,
            position: ObjectPosition::new(1.0, 0.5, 0.0),
            gain: 1.0,
            interpolation: AutomationInterpolation::Linear,
        });

        let pos = obj.position_at(0.5);
        assert!((pos.x - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_atmos_object_automation_step() {
        let mut obj = AtmosObject::new(0, 0);
        obj.add_automation_point(AutomationPoint {
            time_secs: 0.0,
            position: ObjectPosition::new(0.0, 0.5, 0.0),
            gain: 1.0,
            interpolation: AutomationInterpolation::Step,
        });
        obj.add_automation_point(AutomationPoint {
            time_secs: 1.0,
            position: ObjectPosition::new(1.0, 0.5, 0.0),
            gain: 1.0,
            interpolation: AutomationInterpolation::Step,
        });

        let pos = obj.position_at(0.5);
        assert!((pos.x - 0.0).abs() < 0.01); // Step: holds at first keyframe
    }

    #[test]
    fn test_parse_iab_header_valid() {
        let mut data = [0u8; 8];
        data[0] = 1; // version 1
        let num_samples: u32 = 1024;
        data[1..5].copy_from_slice(&num_samples.to_be_bytes());
        data[5] = 1; // 1 bed
        data[6] = 3; // 3 objects
        data[7] = 0; // no user data

        let header = AtmosMetadataParser::parse_iab_header(&data).unwrap();
        assert_eq!(header.version, 1);
        assert_eq!(header.num_samples, 1024);
        assert_eq!(header.num_bed_definitions, 1);
        assert_eq!(header.num_objects, 3);
    }

    #[test]
    fn test_parse_iab_header_bad_version() {
        let data = [2u8, 0, 0, 4, 0, 0, 0, 0]; // version 2
        assert!(matches!(
            AtmosMetadataParser::parse_iab_header(&data),
            Err(AtmosError::UnsupportedVersion(2))
        ));
    }

    #[test]
    fn test_parse_iab_header_too_short() {
        let data = [1u8, 0, 0];
        assert!(matches!(
            AtmosMetadataParser::parse_iab_header(&data),
            Err(AtmosError::NeedMoreData { .. })
        ));
    }

    #[test]
    fn test_parse_object_element_valid() {
        let data = [
            5u8,  // object_id
            2,    // channel_index
            128, 128, 0, // position: center-center-front
            0, 0, 0, // size: point
            128, // gain: unity
            0x80, // flags: enabled
        ];

        let obj = AtmosMetadataParser::parse_object_element(&data).unwrap();
        assert_eq!(obj.id, 5);
        assert_eq!(obj.channel_index, 2);
        assert!(obj.enabled);
        assert!(!obj.snap_to_speaker);
    }

    #[test]
    fn test_serialize_parse_object_roundtrip() {
        let mut original = AtmosObject::new(3, 5);
        original.position = ObjectPosition::new(0.25, 0.75, 0.5);
        original.size = ObjectSize::new(0.1, 0.2, 0.0);
        original.enabled = true;
        original.rendering_intent = RenderingIntent::Binaural;

        let serialized = AtmosMetadataParser::serialize_object(&original);
        assert_eq!(serialized.len(), 10);

        let parsed = AtmosMetadataParser::parse_object_element(&serialized).unwrap();
        assert_eq!(parsed.id, original.id);
        assert_eq!(parsed.channel_index, original.channel_index);
        assert!(parsed.enabled);
        assert_eq!(parsed.rendering_intent, RenderingIntent::Binaural);
        // Position should be within quantization error
        assert!((parsed.position.x - original.position.x).abs() < 0.005);
    }

    #[test]
    fn test_parse_metadata_block() {
        // Build: 8-byte header + 2 object elements (10 bytes each)
        let mut data = Vec::new();
        data.push(1u8); // version
        data.extend_from_slice(&1024u32.to_be_bytes()); // num_samples
        data.push(0); // bed_definitions
        data.push(2); // 2 objects
        data.push(0); // user_data

        // Object 0
        data.extend_from_slice(&[0, 0, 128, 128, 0, 0, 0, 0, 128, 0x80]);
        // Object 1
        data.extend_from_slice(&[1, 1, 200, 100, 50, 0, 0, 0, 128, 0x80]);

        let objects = AtmosMetadataParser::parse_metadata_block(&data).unwrap();
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].id, 0);
        assert_eq!(objects[1].id, 1);
    }

    #[test]
    fn test_atmos_program_operations() {
        let mut program = AtmosProgram::new(1, "Test Mix", 48000);
        let obj = AtmosObject::new(0, 0);
        program.add_object(obj);

        assert_eq!(program.objects.len(), 1);
        assert!(program.get_object(0).is_some());
        assert!(program.get_object(99).is_none());
    }

    #[test]
    fn test_bed_layout_channels() {
        assert_eq!(BedLayout::FivePointOne.channel_count(), 6);
        assert_eq!(BedLayout::SevenPointOne.channel_count(), 8);
        assert_eq!(BedLayout::SevenPointOnePointFour.channel_count(), 12);
    }

    #[test]
    fn test_rendering_intent_roundtrip() {
        for intent in &[
            RenderingIntent::Direct,
            RenderingIntent::Binaural,
            RenderingIntent::Diffuse,
            RenderingIntent::Bed,
        ] {
            let bits = intent.to_bits();
            let recovered = RenderingIntent::from_bits(bits);
            assert_eq!(recovered, *intent);
        }
    }

    #[test]
    fn test_automation_smooth_interpolation() {
        let mut obj = AtmosObject::new(0, 0);
        obj.add_automation_point(AutomationPoint {
            time_secs: 0.0,
            position: ObjectPosition::new(0.0, 0.5, 0.0),
            gain: 0.0,
            interpolation: AutomationInterpolation::Smooth,
        });
        obj.add_automation_point(AutomationPoint {
            time_secs: 1.0,
            position: ObjectPosition::new(1.0, 0.5, 0.0),
            gain: 1.0,
            interpolation: AutomationInterpolation::Smooth,
        });

        // At t=0.5, smooth step should give 0.5 (symmetry)
        let pos = obj.position_at(0.5);
        assert!((pos.x - 0.5).abs() < 0.01, "smooth interp at t=0.5: {}", pos.x);

        // At t=0: should be 0
        let pos0 = obj.position_at(0.0);
        assert!((pos0.x - 0.0).abs() < 0.01);

        // At t=1: should be 1
        let pos1 = obj.position_at(1.0);
        assert!((pos1.x - 1.0).abs() < 0.01, "pos at t=1.0: {}", pos1.x);
    }
}
