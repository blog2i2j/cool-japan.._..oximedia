#![allow(dead_code)]
//! Subtitle-to-timecode synchronization utilities.
//!
//! Provides tools for aligning subtitle cue timestamps with SMPTE timecodes,
//! applying linear and offset corrections, and detecting drift between subtitle
//! streams and the master timecode.

use crate::{FrameRate, Timecode, TimecodeError};

/// A single subtitle cue with in/out timecodes.
#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleCue {
    /// Cue identifier (sequential number or label).
    pub id: u32,
    /// Start timecode.
    pub tc_in: Timecode,
    /// End timecode.
    pub tc_out: Timecode,
    /// Text content of the cue.
    pub text: String,
}

impl SubtitleCue {
    /// Create a new subtitle cue.
    pub fn new(id: u32, tc_in: Timecode, tc_out: Timecode, text: String) -> Self {
        Self {
            id,
            tc_in,
            tc_out,
            text,
        }
    }

    /// Duration of this cue in frames.
    pub fn duration_frames(&self) -> u64 {
        let f_in = self.tc_in.to_frames();
        let f_out = self.tc_out.to_frames();
        f_out.saturating_sub(f_in)
    }

    /// Duration of this cue in seconds (approximate for non-integer rates).
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_secs(&self) -> f64 {
        let fps = self.tc_in.frame_rate.fps as f64;
        self.duration_frames() as f64 / fps
    }

    /// Check if this cue overlaps another cue.
    pub fn overlaps(&self, other: &SubtitleCue) -> bool {
        let a_in = self.tc_in.to_frames();
        let a_out = self.tc_out.to_frames();
        let b_in = other.tc_in.to_frames();
        let b_out = other.tc_out.to_frames();
        a_in < b_out && b_in < a_out
    }
}

/// Offset correction: shift all cues by a fixed number of frames.
pub fn apply_frame_offset(
    cues: &[SubtitleCue],
    offset: i64,
    rate: FrameRate,
) -> Result<Vec<SubtitleCue>, TimecodeError> {
    let mut result = Vec::with_capacity(cues.len());
    for cue in cues {
        let f_in = cue.tc_in.to_frames() as i64 + offset;
        let f_out = cue.tc_out.to_frames() as i64 + offset;
        if f_in < 0 || f_out < 0 {
            return Err(TimecodeError::InvalidFrames);
        }
        let new_in = Timecode::from_frames(f_in as u64, rate)?;
        let new_out = Timecode::from_frames(f_out as u64, rate)?;
        result.push(SubtitleCue::new(cue.id, new_in, new_out, cue.text.clone()));
    }
    Ok(result)
}

/// Linear time-stretch: scale all cue times relative to `anchor_frame` by
/// `factor`.
///
/// A factor of 1.0 is identity. Factor > 1.0 stretches, < 1.0 compresses.
///
/// # Errors
///
/// Returns an error if any resulting timecode is invalid.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn apply_linear_stretch(
    cues: &[SubtitleCue],
    anchor_frame: u64,
    factor: f64,
    rate: FrameRate,
) -> Result<Vec<SubtitleCue>, TimecodeError> {
    let mut result = Vec::with_capacity(cues.len());
    let anchor = anchor_frame as f64;
    for cue in cues {
        let f_in = cue.tc_in.to_frames() as f64;
        let f_out = cue.tc_out.to_frames() as f64;
        let new_in = anchor + (f_in - anchor) * factor;
        let new_out = anchor + (f_out - anchor) * factor;
        if new_in < 0.0 || new_out < 0.0 {
            return Err(TimecodeError::InvalidFrames);
        }
        let tc_in = Timecode::from_frames(new_in.round() as u64, rate)?;
        let tc_out = Timecode::from_frames(new_out.round() as u64, rate)?;
        result.push(SubtitleCue::new(cue.id, tc_in, tc_out, cue.text.clone()));
    }
    Ok(result)
}

/// Compute the average drift (in frames) between subtitle cue-in times and a
/// set of expected reference timecodes.
///
/// Both slices must have the same length. Returns `None` if empty.
#[allow(clippy::cast_precision_loss)]
pub fn compute_average_drift(
    cues: &[SubtitleCue],
    reference_in_frames: &[u64],
) -> Option<f64> {
    if cues.is_empty() || cues.len() != reference_in_frames.len() {
        return None;
    }
    let total: f64 = cues
        .iter()
        .zip(reference_in_frames.iter())
        .map(|(cue, &ref_f)| cue.tc_in.to_frames() as f64 - ref_f as f64)
        .sum();
    Some(total / cues.len() as f64)
}

/// Sort a list of cues by their in-timecode.
pub fn sort_cues_by_in(cues: &mut [SubtitleCue]) {
    cues.sort_by_key(|c| c.tc_in.to_frames());
}

/// Detect overlapping cues and return pairs of indices.
pub fn find_overlaps(cues: &[SubtitleCue]) -> Vec<(usize, usize)> {
    let mut overlaps = Vec::new();
    for i in 0..cues.len() {
        for j in (i + 1)..cues.len() {
            if cues[i].overlaps(&cues[j]) {
                overlaps.push((i, j));
            }
        }
    }
    overlaps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tc(h: u8, m: u8, s: u8, f: u8) -> Timecode {
        Timecode::new(h, m, s, f, FrameRate::Fps25).expect("valid timecode")
    }

    fn make_cue(id: u32, s_in: u8, f_in: u8, s_out: u8, f_out: u8) -> SubtitleCue {
        SubtitleCue::new(id, tc(0, 0, s_in, f_in), tc(0, 0, s_out, f_out), format!("cue {id}"))
    }

    #[test]
    fn test_cue_duration_frames() {
        let cue = make_cue(1, 0, 0, 1, 0);
        assert_eq!(cue.duration_frames(), 25);
    }

    #[test]
    fn test_cue_duration_secs() {
        let cue = make_cue(1, 0, 0, 2, 0);
        let d = cue.duration_secs();
        assert!((d - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_cue_overlap() {
        let a = make_cue(1, 0, 0, 2, 0); // 0..50
        let b = make_cue(2, 1, 0, 3, 0); // 25..75
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_cue_no_overlap() {
        let a = make_cue(1, 0, 0, 1, 0); // 0..25
        let b = make_cue(2, 1, 0, 2, 0); // 25..50
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_apply_frame_offset_positive() {
        let cues = vec![make_cue(1, 0, 0, 1, 0)];
        let shifted = apply_frame_offset(&cues, 10, FrameRate::Fps25).expect("frame offset should succeed");
        assert_eq!(shifted[0].tc_in.to_frames(), 10);
        assert_eq!(shifted[0].tc_out.to_frames(), 35);
    }

    #[test]
    fn test_apply_frame_offset_negative_clamp() {
        let cues = vec![make_cue(1, 0, 0, 1, 0)];
        let result = apply_frame_offset(&cues, -100, FrameRate::Fps25);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_stretch_identity() {
        let cues = vec![make_cue(1, 1, 0, 2, 0)];
        let stretched = apply_linear_stretch(&cues, 0, 1.0, FrameRate::Fps25).expect("linear stretch should succeed");
        assert_eq!(stretched[0].tc_in.to_frames(), cues[0].tc_in.to_frames());
        assert_eq!(stretched[0].tc_out.to_frames(), cues[0].tc_out.to_frames());
    }

    #[test]
    fn test_linear_stretch_double() {
        let cues = vec![make_cue(1, 1, 0, 2, 0)]; // frames 25..50
        let stretched = apply_linear_stretch(&cues, 0, 2.0, FrameRate::Fps25).expect("linear stretch should succeed");
        assert_eq!(stretched[0].tc_in.to_frames(), 50);
        assert_eq!(stretched[0].tc_out.to_frames(), 100);
    }

    #[test]
    fn test_compute_average_drift_zero() {
        let cues = vec![make_cue(1, 1, 0, 2, 0)];
        let refs = vec![25];
        let drift = compute_average_drift(&cues, &refs).expect("drift computation should succeed");
        assert!((drift).abs() < 1e-9);
    }

    #[test]
    fn test_compute_average_drift_some() {
        let cues = vec![make_cue(1, 1, 0, 2, 0), make_cue(2, 2, 0, 3, 0)];
        let refs = vec![20, 45];
        let drift = compute_average_drift(&cues, &refs).expect("drift computation should succeed");
        // cue1: 25-20=5, cue2: 50-45=5, avg=5
        assert!((drift - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_sort_cues_by_in() {
        let mut cues = vec![make_cue(2, 2, 0, 3, 0), make_cue(1, 0, 0, 1, 0)];
        sort_cues_by_in(&mut cues);
        assert_eq!(cues[0].id, 1);
        assert_eq!(cues[1].id, 2);
    }

    #[test]
    fn test_find_overlaps() {
        let cues = vec![
            make_cue(1, 0, 0, 2, 0),
            make_cue(2, 1, 0, 3, 0),
            make_cue(3, 5, 0, 6, 0),
        ];
        let overlaps = find_overlaps(&cues);
        assert_eq!(overlaps.len(), 1);
        assert_eq!(overlaps[0], (0, 1));
    }

    #[test]
    fn test_compute_average_drift_empty() {
        let drift = compute_average_drift(&[], &[]);
        assert!(drift.is_none());
    }
}
