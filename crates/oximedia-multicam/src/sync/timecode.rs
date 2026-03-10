//! Timecode-based synchronization for multi-camera production.
//!
//! This module provides synchronization using SMPTE timecode (LTC/VITC).

use super::{SyncConfig, SyncMethod, SyncOffset, SyncResult, Synchronizer};
use crate::{AngleId, Result};
use oximedia_timecode::{FrameRate, Timecode};

/// Timecode synchronizer
#[derive(Debug)]
pub struct TimecodeSync {
    /// Timecode sequences for each angle
    timecodes: Vec<Vec<TimecodeEntry>>,
    /// Frame rate for each angle
    frame_rates: Vec<FrameRate>,
}

/// Timecode entry with frame number
#[derive(Debug, Clone)]
pub struct TimecodeEntry {
    /// Frame number in the stream
    pub frame: u64,
    /// Timecode value
    pub timecode: Timecode,
}

impl TimecodeEntry {
    /// Create a new timecode entry
    #[must_use]
    pub fn new(frame: u64, timecode: Timecode) -> Self {
        Self { frame, timecode }
    }
}

impl TimecodeSync {
    /// Create a new timecode synchronizer
    #[must_use]
    pub fn new(angle_count: usize) -> Self {
        Self {
            timecodes: vec![Vec::new(); angle_count],
            frame_rates: vec![FrameRate::Fps25; angle_count],
        }
    }

    /// Add timecode sequence for an angle
    pub fn add_timecodes(
        &mut self,
        angle: AngleId,
        entries: Vec<TimecodeEntry>,
        frame_rate: FrameRate,
    ) {
        if angle < self.timecodes.len() {
            self.timecodes[angle] = entries;
            self.frame_rates[angle] = frame_rate;
        }
    }

    /// Find offset using timecode matching
    ///
    /// # Errors
    ///
    /// Returns an error if synchronization fails
    pub fn find_offset(&self, angle_a: AngleId, angle_b: AngleId) -> Result<SyncOffset> {
        if angle_a >= self.timecodes.len() || angle_b >= self.timecodes.len() {
            return Err(crate::MultiCamError::AngleNotFound(angle_a.max(angle_b)));
        }

        let tc_a = &self.timecodes[angle_a];
        let tc_b = &self.timecodes[angle_b];

        if tc_a.is_empty() || tc_b.is_empty() {
            return Err(crate::MultiCamError::InsufficientData(
                "No timecode data available".to_string(),
            ));
        }

        // Find matching timecode values
        let offset = self.match_timecodes(tc_a, tc_b)?;

        Ok(SyncOffset::new(angle_b, offset.0, offset.1, offset.2))
    }

    /// Match timecode sequences
    fn match_timecodes(
        &self,
        tc_a: &[TimecodeEntry],
        tc_b: &[TimecodeEntry],
    ) -> Result<(i64, f64, f64)> {
        // Find first matching timecode
        for entry_a in tc_a {
            for entry_b in tc_b {
                if self.timecodes_match(&entry_a.timecode, &entry_b.timecode) {
                    // Calculate frame offset
                    let offset = entry_b.frame as i64 - entry_a.frame as i64;
                    return Ok((offset, 0.0, 1.0)); // Perfect confidence for timecode match
                }
            }
        }

        Err(crate::MultiCamError::SyncFailed(
            "No matching timecodes found".to_string(),
        ))
    }

    /// Check if two timecodes match
    fn timecodes_match(&self, tc_a: &Timecode, tc_b: &Timecode) -> bool {
        tc_a.hours == tc_b.hours
            && tc_a.minutes == tc_b.minutes
            && tc_a.seconds == tc_b.seconds
            && tc_a.frames == tc_b.frames
    }

    /// Find timecode at specific frame
    #[must_use]
    pub fn find_timecode_at_frame(&self, angle: AngleId, frame: u64) -> Option<Timecode> {
        if angle >= self.timecodes.len() {
            return None;
        }

        self.timecodes[angle]
            .iter()
            .find(|entry| entry.frame == frame)
            .map(|entry| entry.timecode)
    }

    /// Validate timecode continuity
    #[must_use]
    pub fn validate_continuity(&self, angle: AngleId) -> bool {
        if angle >= self.timecodes.len() {
            return false;
        }

        let entries = &self.timecodes[angle];
        if entries.len() < 2 {
            return true;
        }

        for i in 1..entries.len() {
            let prev = &entries[i - 1];
            let curr = &entries[i];

            // Check frame numbers are sequential
            if curr.frame != prev.frame + 1 {
                continue;
            }

            // Check timecode increments correctly
            let mut expected = prev.timecode;
            if expected.increment().is_err() {
                return false;
            }

            if !self.timecodes_match(&expected, &curr.timecode) {
                return false;
            }
        }

        true
    }

    /// Detect timecode breaks
    #[must_use]
    pub fn detect_breaks(&self, angle: AngleId) -> Vec<u64> {
        if angle >= self.timecodes.len() {
            return Vec::new();
        }

        let entries = &self.timecodes[angle];
        let mut breaks = Vec::new();

        for i in 1..entries.len() {
            let prev = &entries[i - 1];
            let curr = &entries[i];

            // Check if frame numbers are sequential
            if curr.frame != prev.frame + 1 {
                continue;
            }

            // Check if timecode increments correctly
            let mut expected = prev.timecode;
            if expected.increment().is_ok() && !self.timecodes_match(&expected, &curr.timecode) {
                breaks.push(curr.frame);
            }
        }

        breaks
    }

    /// Interpolate missing timecodes
    pub fn interpolate_timecodes(&mut self, angle: AngleId) {
        if angle >= self.timecodes.len() {
            return;
        }

        let entries = &self.timecodes[angle];
        if entries.len() < 2 {
            return;
        }

        let mut interpolated = Vec::new();
        let _frame_rate = self.frame_rates[angle];

        for i in 0..entries.len() {
            interpolated.push(entries[i].clone());

            if i + 1 < entries.len() {
                let curr = &entries[i];
                let next = &entries[i + 1];
                let gap = next.frame - curr.frame;

                // Interpolate if there's a gap
                if gap > 1 {
                    let mut tc = curr.timecode;
                    for frame in (curr.frame + 1)..next.frame {
                        if tc.increment().is_ok() {
                            interpolated.push(TimecodeEntry::new(frame, tc));
                        }
                    }
                }
            }
        }

        self.timecodes[angle] = interpolated;
    }

    /// Convert timecode to frame number
    #[must_use]
    pub fn timecode_to_frame(&self, timecode: &Timecode) -> u64 {
        timecode.to_frames()
    }

    /// Sync using jam sync (copy timecode from one angle to another)
    pub fn jam_sync(&mut self, source_angle: AngleId, dest_angle: AngleId, offset_frames: i64) {
        if source_angle >= self.timecodes.len() || dest_angle >= self.timecodes.len() {
            return;
        }

        let source_entries = self.timecodes[source_angle].clone();
        let mut dest_entries = Vec::new();

        for entry in source_entries {
            let new_frame = (entry.frame as i64 + offset_frames).max(0) as u64;
            dest_entries.push(TimecodeEntry::new(new_frame, entry.timecode));
        }

        self.timecodes[dest_angle] = dest_entries;
    }
}

impl Synchronizer for TimecodeSync {
    fn synchronize(&self, _config: &SyncConfig) -> Result<SyncResult> {
        if self.timecodes.is_empty() {
            return Err(crate::MultiCamError::InsufficientData(
                "No timecode data available".to_string(),
            ));
        }

        let mut offsets = Vec::new();
        let reference_angle = 0;

        // Calculate offsets relative to first angle
        for angle in 1..self.timecodes.len() {
            let offset = self.find_offset(reference_angle, angle)?;
            offsets.push(offset);
        }

        // Add zero offset for reference angle
        offsets.insert(0, SyncOffset::new(reference_angle, 0, 0.0, 1.0));

        // Calculate average confidence (timecode sync is very reliable)
        let confidence = 1.0;

        Ok(SyncResult {
            reference_angle,
            offsets,
            confidence,
            method: SyncMethod::Timecode,
        })
    }

    fn method(&self) -> SyncMethod {
        SyncMethod::Timecode
    }

    fn is_reliable(&self) -> bool {
        !self.timecodes.is_empty()
            && self.timecodes.iter().all(|tc| !tc.is_empty())
            && self
                .timecodes
                .iter()
                .enumerate()
                .all(|(i, _)| self.validate_continuity(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timecode_sync_creation() {
        let sync = TimecodeSync::new(3);
        assert_eq!(sync.timecodes.len(), 3);
        assert_eq!(sync.frame_rates.len(), 3);
    }

    #[test]
    fn test_timecode_entry() {
        let tc = Timecode::new(1, 0, 0, 0, FrameRate::Fps25)
            .expect("multicam test operation should succeed");
        let entry = TimecodeEntry::new(100, tc);
        assert_eq!(entry.frame, 100);
        assert_eq!(entry.timecode.hours, 1);
    }

    #[test]
    fn test_find_timecode_at_frame() {
        let mut sync = TimecodeSync::new(1);
        let tc = Timecode::new(1, 0, 0, 0, FrameRate::Fps25)
            .expect("multicam test operation should succeed");
        let entries = vec![TimecodeEntry::new(100, tc)];
        sync.add_timecodes(0, entries, FrameRate::Fps25);

        let found = sync.find_timecode_at_frame(0, 100);
        assert!(found.is_some());
        assert_eq!(
            found.expect("multicam test operation should succeed").hours,
            1
        );
    }

    #[test]
    fn test_detect_breaks() {
        let mut sync = TimecodeSync::new(1);
        let tc1 = Timecode::new(1, 0, 0, 0, FrameRate::Fps25)
            .expect("multicam test operation should succeed");
        let tc2 = Timecode::new(1, 0, 0, 1, FrameRate::Fps25)
            .expect("multicam test operation should succeed");
        let tc3 = Timecode::new(1, 0, 0, 5, FrameRate::Fps25)
            .expect("multicam test operation should succeed"); // Jump!

        let entries = vec![
            TimecodeEntry::new(0, tc1),
            TimecodeEntry::new(1, tc2),
            TimecodeEntry::new(2, tc3),
        ];
        sync.add_timecodes(0, entries, FrameRate::Fps25);

        let breaks = sync.detect_breaks(0);
        assert!(!breaks.is_empty());
    }
}
