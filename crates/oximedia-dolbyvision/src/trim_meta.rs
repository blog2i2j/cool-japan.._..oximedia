//! Dolby Vision trim metadata levels L2 through L5.
//!
//! Trim metadata allows Dolby Vision to describe how to adapt HDR content for
//! different target display capabilities.  Each level targets a different nit
//! range and provides progressively more detailed tone-mapping hints.
//!
//! # Level Overview
//!
//! | Level | Purpose | Target |
//! |-------|---------|--------|
//! | L2 | Per-shot trim | 100 nit SDR reference |
//! | L3 | Mid-tone detail | 600 nit mid-range HDR |
//! | L4 | Global dimming | 1000 nit HDR reference |
//! | L5 | Active area | Any (letterbox / pillarbox metadata) |

use crate::DolbyVisionError;

/// Dolby Vision trim metadata container for levels L2–L5.
///
/// Each level targets a different display category.  Call [`TrimMetadata::new`]
/// with the desired level, then configure the instance with the builder methods
/// before serialising via [`TrimMetadata::to_bytes`].
#[derive(Debug, Clone)]
pub struct TrimMetadata {
    /// Metadata level: 2, 3, 4, or 5.
    pub level: u8,
    /// Target display peak luminance in nits.
    pub target_nits: f32,
    /// Lift (shadow) adjustment in the trim pass [-1.0, 1.0].
    pub lift: f32,
    /// Gain (highlight) adjustment in the trim pass [-1.0, 1.0].
    pub gain: f32,
    /// Gamma exponent offset in the trim pass [-1.0, 1.0].
    pub gamma: f32,
    /// Chroma weight adjustment [0.0, 2.0].
    pub chroma_weight: f32,
    /// Saturation gain adjustment [0.0, 2.0].
    pub saturation_gain: f32,
}

impl TrimMetadata {
    /// Create a new trim metadata block for the given level.
    ///
    /// # Errors
    ///
    /// Returns [`DolbyVisionError::InvalidPayload`] if `level` is not 2, 3, 4,
    /// or 5.
    pub fn new(level: u8) -> Result<Self, DolbyVisionError> {
        let default_target_nits = Self::default_nits_for_level(level)?;
        Ok(Self {
            level,
            target_nits: default_target_nits,
            lift: 0.0,
            gain: 0.0,
            gamma: 0.0,
            chroma_weight: 1.0,
            saturation_gain: 1.0,
        })
    }

    /// Set the target display peak luminance in nits.
    ///
    /// Typical values: 100 (L2), 600 (L3), 1000 (L4), 4000 (L5).
    /// Negative or zero values are rejected.
    ///
    /// # Errors
    ///
    /// Returns [`DolbyVisionError::InvalidPayload`] if `nits` is not positive.
    pub fn set_target_nits(&mut self, nits: f32) -> Result<(), DolbyVisionError> {
        if nits <= 0.0 {
            return Err(DolbyVisionError::InvalidPayload(format!(
                "target_nits must be positive, got {nits}"
            )));
        }
        self.target_nits = nits;
        Ok(())
    }

    /// Set the shadow (lift) adjustment.
    ///
    /// # Errors
    ///
    /// Returns [`DolbyVisionError::InvalidPayload`] when `lift` is outside
    /// `[-1.0, 1.0]`.
    pub fn set_lift(&mut self, lift: f32) -> Result<(), DolbyVisionError> {
        if !(-1.0..=1.0).contains(&lift) {
            return Err(DolbyVisionError::InvalidPayload(format!(
                "lift {lift} is outside [-1.0, 1.0]"
            )));
        }
        self.lift = lift;
        Ok(())
    }

    /// Set the highlight (gain) adjustment.
    ///
    /// # Errors
    ///
    /// Returns [`DolbyVisionError::InvalidPayload`] when `gain` is outside
    /// `[-1.0, 1.0]`.
    pub fn set_gain(&mut self, gain: f32) -> Result<(), DolbyVisionError> {
        if !(-1.0..=1.0).contains(&gain) {
            return Err(DolbyVisionError::InvalidPayload(format!(
                "gain {gain} is outside [-1.0, 1.0]"
            )));
        }
        self.gain = gain;
        Ok(())
    }

    /// Serialise the trim metadata block to bytes.
    ///
    /// The binary layout is:
    ///
    /// ```text
    /// [0]     level (u8)
    /// [1..5]  target_nits as little-endian f32
    /// [5..9]  lift as little-endian f32
    /// [9..13] gain as little-endian f32
    /// [13..17] gamma as little-endian f32
    /// [17..21] chroma_weight as little-endian f32
    /// [21..25] saturation_gain as little-endian f32
    /// ```
    ///
    /// Total: 25 bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(25);
        buf.push(self.level);
        buf.extend_from_slice(&self.target_nits.to_le_bytes());
        buf.extend_from_slice(&self.lift.to_le_bytes());
        buf.extend_from_slice(&self.gain.to_le_bytes());
        buf.extend_from_slice(&self.gamma.to_le_bytes());
        buf.extend_from_slice(&self.chroma_weight.to_le_bytes());
        buf.extend_from_slice(&self.saturation_gain.to_le_bytes());
        buf
    }

    /// Deserialise a trim metadata block from bytes produced by [`to_bytes`].
    ///
    /// # Errors
    ///
    /// Returns [`DolbyVisionError::InvalidPayload`] if the slice is shorter
    /// than 25 bytes or the embedded level value is not 2–5.
    pub fn from_bytes(data: &[u8]) -> Result<Self, DolbyVisionError> {
        if data.len() < 25 {
            return Err(DolbyVisionError::InvalidPayload(format!(
                "trim metadata requires 25 bytes, got {}",
                data.len()
            )));
        }
        let level = data[0];
        Self::default_nits_for_level(level)?; // validate level
        let read_f32 = |off: usize| -> f32 {
            f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        };
        Ok(Self {
            level,
            target_nits: read_f32(1),
            lift: read_f32(5),
            gain: read_f32(9),
            gamma: read_f32(13),
            chroma_weight: read_f32(17),
            saturation_gain: read_f32(21),
        })
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn default_nits_for_level(level: u8) -> Result<f32, DolbyVisionError> {
        match level {
            2 => Ok(100.0),
            3 => Ok(600.0),
            4 => Ok(1000.0),
            5 => Ok(4000.0),
            other => Err(DolbyVisionError::InvalidPayload(format!(
                "trim level must be 2, 3, 4 or 5; got {other}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid_levels() {
        for level in 2u8..=5 {
            let t = TrimMetadata::new(level).expect("valid level");
            assert_eq!(t.level, level);
            assert!(t.target_nits > 0.0);
        }
    }

    #[test]
    fn test_new_invalid_level() {
        assert!(TrimMetadata::new(1).is_err());
        assert!(TrimMetadata::new(6).is_err());
        assert!(TrimMetadata::new(0).is_err());
    }

    #[test]
    fn test_set_target_nits() {
        let mut t = TrimMetadata::new(2).expect("ok");
        assert!(t.set_target_nits(500.0).is_ok());
        assert_eq!(t.target_nits, 500.0);
        assert!(t.set_target_nits(0.0).is_err());
        assert!(t.set_target_nits(-10.0).is_err());
    }

    #[test]
    fn test_roundtrip() {
        let mut t = TrimMetadata::new(3).expect("ok");
        t.set_target_nits(800.0).expect("ok");
        t.lift = 0.1;
        t.gain = -0.2;
        t.gamma = 0.05;
        t.chroma_weight = 1.2;
        t.saturation_gain = 0.9;

        let bytes = t.to_bytes();
        assert_eq!(bytes.len(), 25);

        let restored = TrimMetadata::from_bytes(&bytes).expect("roundtrip");
        assert_eq!(restored.level, 3);
        assert!((restored.target_nits - 800.0).abs() < 1e-4);
        assert!((restored.lift - 0.1).abs() < 1e-6);
        assert!((restored.gain - (-0.2)).abs() < 1e-6);
    }

    #[test]
    fn test_to_bytes_length() {
        let t = TrimMetadata::new(4).expect("ok");
        assert_eq!(t.to_bytes().len(), 25);
    }
}
