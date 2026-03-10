//! Multi-angle synchronization for multicam production.
//!
//! Provides synchronization methods, per-angle offsets, and aggregate sync results.

#![allow(dead_code)]

/// Method used to synchronize a camera angle to the reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMethod {
    /// Synchronization via embedded timecode (LTC/VITC/SMPTE).
    Timecode,
    /// Synchronization via audio clap/clapper-board detection.
    AudioClap,
    /// Synchronization via musical beat alignment.
    BeatSync,
    /// Manually supplied frame offset.
    ManualOffset,
}

impl SyncMethod {
    /// Returns `true` for methods that do not require human intervention.
    #[must_use]
    pub fn is_automatic(&self) -> bool {
        match self {
            SyncMethod::Timecode | SyncMethod::AudioClap | SyncMethod::BeatSync => true,
            SyncMethod::ManualOffset => false,
        }
    }
}

/// Frame offset for a single camera angle relative to the reference angle.
#[derive(Debug, Clone)]
pub struct AngleOffset {
    /// Identifier of the camera angle.
    pub angle_id: u8,
    /// Signed frame offset (positive = angle is ahead of reference).
    pub offset_frames: i64,
    /// Synchronization method that produced this offset.
    pub sync_method: SyncMethod,
    /// Confidence score in [0.0, 1.0].
    pub confidence: f32,
}

impl AngleOffset {
    /// Returns `true` when `confidence >= threshold`.
    #[must_use]
    pub fn is_reliable(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }

    /// Returns `true` when the offset is exactly zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.offset_frames == 0
    }
}

/// Aggregate synchronization result across all angles.
#[derive(Debug, Clone)]
pub struct AngleSyncResult {
    /// Per-angle offsets.
    pub offsets: Vec<AngleOffset>,
    /// The angle chosen as the sync reference (offset is always 0 for this angle).
    pub reference_angle: u8,
}

impl AngleSyncResult {
    /// Looks up the offset for `angle_id`. Returns `None` if not present.
    #[must_use]
    pub fn offset_for(&self, angle_id: u8) -> Option<&AngleOffset> {
        self.offsets.iter().find(|o| o.angle_id == angle_id)
    }

    /// Returns `true` when every angle meets the reliability `threshold`.
    #[must_use]
    pub fn all_reliable(&self, threshold: f32) -> bool {
        self.offsets.iter().all(|o| o.is_reliable(threshold))
    }

    /// Returns the maximum absolute offset in frames across all angles.
    #[must_use]
    pub fn max_offset_frames(&self) -> i64 {
        self.offsets
            .iter()
            .map(|o| o.offset_frames.abs())
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- SyncMethod ---

    #[test]
    fn test_timecode_is_automatic() {
        assert!(SyncMethod::Timecode.is_automatic());
    }

    #[test]
    fn test_audio_clap_is_automatic() {
        assert!(SyncMethod::AudioClap.is_automatic());
    }

    #[test]
    fn test_beat_sync_is_automatic() {
        assert!(SyncMethod::BeatSync.is_automatic());
    }

    #[test]
    fn test_manual_offset_not_automatic() {
        assert!(!SyncMethod::ManualOffset.is_automatic());
    }

    // --- AngleOffset ---

    #[test]
    fn test_is_reliable_above_threshold() {
        let offset = AngleOffset {
            angle_id: 1,
            offset_frames: 3,
            sync_method: SyncMethod::AudioClap,
            confidence: 0.9,
        };
        assert!(offset.is_reliable(0.8));
    }

    #[test]
    fn test_is_reliable_below_threshold() {
        let offset = AngleOffset {
            angle_id: 1,
            offset_frames: 3,
            sync_method: SyncMethod::AudioClap,
            confidence: 0.5,
        };
        assert!(!offset.is_reliable(0.8));
    }

    #[test]
    fn test_is_reliable_exact_threshold() {
        let offset = AngleOffset {
            angle_id: 2,
            offset_frames: 0,
            sync_method: SyncMethod::Timecode,
            confidence: 0.75,
        };
        assert!(offset.is_reliable(0.75));
    }

    #[test]
    fn test_is_zero_true() {
        let offset = AngleOffset {
            angle_id: 0,
            offset_frames: 0,
            sync_method: SyncMethod::Timecode,
            confidence: 1.0,
        };
        assert!(offset.is_zero());
    }

    #[test]
    fn test_is_zero_false_positive() {
        let offset = AngleOffset {
            angle_id: 0,
            offset_frames: 5,
            sync_method: SyncMethod::ManualOffset,
            confidence: 1.0,
        };
        assert!(!offset.is_zero());
    }

    #[test]
    fn test_is_zero_false_negative() {
        let offset = AngleOffset {
            angle_id: 0,
            offset_frames: -3,
            sync_method: SyncMethod::BeatSync,
            confidence: 0.6,
        };
        assert!(!offset.is_zero());
    }

    // --- AngleSyncResult ---

    fn make_result() -> AngleSyncResult {
        AngleSyncResult {
            reference_angle: 0,
            offsets: vec![
                AngleOffset {
                    angle_id: 0,
                    offset_frames: 0,
                    sync_method: SyncMethod::Timecode,
                    confidence: 1.0,
                },
                AngleOffset {
                    angle_id: 1,
                    offset_frames: 12,
                    sync_method: SyncMethod::AudioClap,
                    confidence: 0.85,
                },
                AngleOffset {
                    angle_id: 2,
                    offset_frames: -5,
                    sync_method: SyncMethod::BeatSync,
                    confidence: 0.6,
                },
            ],
        }
    }

    #[test]
    fn test_offset_for_found() {
        let result = make_result();
        let o = result
            .offset_for(1)
            .expect("multicam test operation should succeed");
        assert_eq!(o.offset_frames, 12);
    }

    #[test]
    fn test_offset_for_not_found() {
        let result = make_result();
        assert!(result.offset_for(99).is_none());
    }

    #[test]
    fn test_all_reliable_true() {
        let result = make_result();
        assert!(result.all_reliable(0.5));
    }

    #[test]
    fn test_all_reliable_false() {
        let result = make_result();
        // angle 2 has confidence 0.6, threshold 0.7 should fail
        assert!(!result.all_reliable(0.7));
    }

    #[test]
    fn test_max_offset_frames() {
        let result = make_result();
        // abs values: 0, 12, 5 → max is 12
        assert_eq!(result.max_offset_frames(), 12);
    }

    #[test]
    fn test_max_offset_empty() {
        let result = AngleSyncResult {
            reference_angle: 0,
            offsets: vec![],
        };
        assert_eq!(result.max_offset_frames(), 0);
    }
}
