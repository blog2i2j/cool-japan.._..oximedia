#![allow(dead_code)]
//! Timecode format conversion utilities.
//!
//! Converts timecodes between different frame rates, between wall-clock time
//! and timecode, and between SMPTE string representations and frame numbers.

use crate::{FrameRate, Timecode, TimecodeError};

/// Strategy for converting timecodes between different frame rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvertStrategy {
    /// Preserve the wall-clock time as closely as possible.
    PreserveTime,
    /// Preserve the frame number (snap to nearest frame in target rate).
    PreserveFrame,
    /// Preserve the HH:MM:SS:FF display string (may change actual time).
    PreserveDisplay,
}

/// Result of a timecode conversion.
#[derive(Debug, Clone)]
pub struct ConvertResult {
    /// The converted timecode
    pub timecode: Timecode,
    /// The rounding error in seconds (positive means output is later)
    pub rounding_error_secs: f64,
    /// Whether the conversion was exact (no rounding)
    pub exact: bool,
}

/// Converts a timecode from one frame rate to another.
///
/// # Errors
///
/// Returns an error if the target timecode is invalid (e.g., exceeds 24h).
#[allow(clippy::cast_precision_loss)]
pub fn convert_frame_rate(
    tc: &Timecode,
    target_rate: FrameRate,
    strategy: ConvertStrategy,
) -> Result<ConvertResult, TimecodeError> {
    match strategy {
        ConvertStrategy::PreserveTime => convert_preserve_time(tc, target_rate),
        ConvertStrategy::PreserveFrame => convert_preserve_frame(tc, target_rate),
        ConvertStrategy::PreserveDisplay => convert_preserve_display(tc, target_rate),
    }
}

/// Converts preserving wall-clock time.
#[allow(clippy::cast_precision_loss)]
fn convert_preserve_time(
    tc: &Timecode,
    target_rate: FrameRate,
) -> Result<ConvertResult, TimecodeError> {
    let source_fps = if tc.frame_rate.drop_frame {
        29.97
    } else {
        tc.frame_rate.fps as f64
    };
    let src_frames = tc.to_frames();
    let time_secs = src_frames as f64 / source_fps;

    let target_fps = target_rate.as_float();
    let target_frames = (time_secs * target_fps).round() as u64;

    let result_tc = Timecode::from_frames(target_frames, target_rate)?;
    let result_time = target_frames as f64 / target_fps;
    let error = result_time - time_secs;

    Ok(ConvertResult {
        timecode: result_tc,
        rounding_error_secs: error,
        exact: error.abs() < 1e-9,
    })
}

/// Converts preserving the frame number (modulo target fps).
#[allow(clippy::cast_precision_loss)]
fn convert_preserve_frame(
    tc: &Timecode,
    target_rate: FrameRate,
) -> Result<ConvertResult, TimecodeError> {
    let src_frames = tc.to_frames();
    let result_tc = Timecode::from_frames(src_frames, target_rate)?;
    let source_fps = if tc.frame_rate.drop_frame {
        29.97
    } else {
        tc.frame_rate.fps as f64
    };
    let target_fps = target_rate.as_float();
    let error = src_frames as f64 * (1.0 / target_fps - 1.0 / source_fps);

    Ok(ConvertResult {
        timecode: result_tc,
        rounding_error_secs: error,
        exact: (source_fps - target_fps).abs() < 1e-9,
    })
}

/// Converts preserving the HH:MM:SS:FF display.
#[allow(clippy::cast_precision_loss)]
fn convert_preserve_display(
    tc: &Timecode,
    target_rate: FrameRate,
) -> Result<ConvertResult, TimecodeError> {
    let target_fps = target_rate.frames_per_second() as u8;
    let frames = if tc.frames >= target_fps {
        target_fps - 1
    } else {
        tc.frames
    };
    let result_tc = Timecode::new(tc.hours, tc.minutes, tc.seconds, frames, target_rate)?;
    let source_fps = if tc.frame_rate.drop_frame {
        29.97
    } else {
        tc.frame_rate.fps as f64
    };
    let tfps = target_rate.as_float();
    let src_time = tc.to_frames() as f64 / source_fps;
    let dst_time = result_tc.to_frames() as f64 / tfps;

    Ok(ConvertResult {
        timecode: result_tc,
        rounding_error_secs: dst_time - src_time,
        exact: false,
    })
}

/// Converts a wall-clock duration in seconds to a timecode.
///
/// # Errors
///
/// Returns an error if the duration exceeds 24 hours.
#[allow(clippy::cast_precision_loss)]
pub fn seconds_to_timecode(secs: f64, rate: FrameRate) -> Result<Timecode, TimecodeError> {
    if secs < 0.0 {
        return Err(TimecodeError::InvalidConfiguration);
    }
    let fps = rate.as_float();
    let total_frames = (secs * fps).round() as u64;
    Timecode::from_frames(total_frames, rate)
}

/// Converts a timecode to wall-clock seconds.
#[allow(clippy::cast_precision_loss)]
pub fn timecode_to_seconds(tc: &Timecode) -> f64 {
    let fps = if tc.frame_rate.drop_frame {
        29.97
    } else {
        tc.frame_rate.fps as f64
    };
    tc.to_frames() as f64 / fps
}

/// Parses a SMPTE timecode string like "01:02:03:04" or "01:02:03;04".
///
/// The separator between seconds and frames determines drop-frame vs non-drop:
/// - `:` for non-drop frame
/// - `;` for drop frame
///
/// # Errors
///
/// Returns an error if the string format is invalid.
pub fn parse_smpte_string(s: &str, rate: FrameRate) -> Result<Timecode, TimecodeError> {
    let s = s.trim();
    if s.len() < 11 {
        return Err(TimecodeError::InvalidConfiguration);
    }
    let parts: Vec<&str> = s.split([':', ';']).collect();
    if parts.len() != 4 {
        return Err(TimecodeError::InvalidConfiguration);
    }
    let hours: u8 = parts[0].parse().map_err(|_| TimecodeError::InvalidHours)?;
    let minutes: u8 = parts[1]
        .parse()
        .map_err(|_| TimecodeError::InvalidMinutes)?;
    let seconds: u8 = parts[2]
        .parse()
        .map_err(|_| TimecodeError::InvalidSeconds)?;
    let frames: u8 = parts[3].parse().map_err(|_| TimecodeError::InvalidFrames)?;

    Timecode::new(hours, minutes, seconds, frames, rate)
}

/// Formats a frame count as an SMPTE timecode string.
///
/// # Errors
///
/// Returns an error if the frame count produces an invalid timecode.
pub fn frames_to_smpte_string(frames: u64, rate: FrameRate) -> Result<String, TimecodeError> {
    let tc = Timecode::from_frames(frames, rate)?;
    Ok(tc.to_string())
}

/// Converts a timecode to a total millisecond value.
#[allow(clippy::cast_precision_loss)]
pub fn timecode_to_millis(tc: &Timecode) -> u64 {
    let secs = timecode_to_seconds(tc);
    (secs * 1000.0).round() as u64
}

/// Converts milliseconds to a timecode.
///
/// # Errors
///
/// Returns an error if the milliseconds value exceeds 24 hours.
pub fn millis_to_timecode(ms: u64, rate: FrameRate) -> Result<Timecode, TimecodeError> {
    #[allow(clippy::cast_precision_loss)]
    let secs = ms as f64 / 1000.0;
    seconds_to_timecode(secs, rate)
}

/// Computes the number of real-time samples (at a given audio sample rate)
/// that correspond to a timecode offset.
#[allow(clippy::cast_precision_loss)]
pub fn timecode_to_audio_samples(tc: &Timecode, sample_rate: u32) -> u64 {
    let secs = timecode_to_seconds(tc);
    (secs * sample_rate as f64).round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seconds_to_timecode_25fps() {
        let tc = seconds_to_timecode(3661.0, FrameRate::Fps25)
            .expect("seconds to timecode should succeed");
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 1);
        assert_eq!(tc.seconds, 1);
        assert_eq!(tc.frames, 0);
    }

    #[test]
    fn test_timecode_to_seconds() {
        let tc = Timecode::new(1, 0, 0, 0, FrameRate::Fps25).expect("valid timecode");
        let secs = timecode_to_seconds(&tc);
        assert!((secs - 3600.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_smpte_ndf() {
        let tc = parse_smpte_string("01:02:03:04", FrameRate::Fps25).expect("should succeed");
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 2);
        assert_eq!(tc.seconds, 3);
        assert_eq!(tc.frames, 4);
    }

    #[test]
    fn test_parse_smpte_invalid() {
        assert!(parse_smpte_string("bad", FrameRate::Fps25).is_err());
    }

    #[test]
    fn test_frames_to_smpte_string() {
        let s =
            frames_to_smpte_string(25, FrameRate::Fps25).expect("frames to SMPTE should succeed");
        assert_eq!(s, "00:00:01:00");
    }

    #[test]
    fn test_millis_roundtrip() {
        let tc = Timecode::new(0, 1, 30, 0, FrameRate::Fps25).expect("valid timecode");
        let ms = timecode_to_millis(&tc);
        let tc2 =
            millis_to_timecode(ms, FrameRate::Fps25).expect("millis to timecode should succeed");
        assert_eq!(tc.hours, tc2.hours);
        assert_eq!(tc.minutes, tc2.minutes);
        assert_eq!(tc.seconds, tc2.seconds);
    }

    #[test]
    fn test_convert_preserve_time_same_rate() {
        let tc = Timecode::new(1, 0, 0, 0, FrameRate::Fps25).expect("valid timecode");
        let result = convert_frame_rate(&tc, FrameRate::Fps25, ConvertStrategy::PreserveTime)
            .expect("conversion should succeed");
        assert!(result.rounding_error_secs.abs() < 0.001);
    }

    #[test]
    fn test_convert_preserve_time_25_to_30() {
        let tc = Timecode::new(0, 0, 1, 0, FrameRate::Fps25).expect("valid timecode");
        let result = convert_frame_rate(&tc, FrameRate::Fps30, ConvertStrategy::PreserveTime)
            .expect("conversion should succeed");
        assert_eq!(result.timecode.seconds, 1);
        assert_eq!(result.timecode.frames, 0);
    }

    #[test]
    fn test_convert_preserve_display() {
        let tc = Timecode::new(1, 2, 3, 10, FrameRate::Fps30).expect("valid timecode");
        let result = convert_frame_rate(&tc, FrameRate::Fps25, ConvertStrategy::PreserveDisplay)
            .expect("conversion should succeed");
        assert_eq!(result.timecode.hours, 1);
        assert_eq!(result.timecode.minutes, 2);
        assert_eq!(result.timecode.seconds, 3);
        assert_eq!(result.timecode.frames, 10);
    }

    #[test]
    fn test_convert_preserve_frame() {
        let tc = Timecode::new(0, 0, 0, 10, FrameRate::Fps25).expect("valid timecode");
        let result = convert_frame_rate(&tc, FrameRate::Fps30, ConvertStrategy::PreserveFrame)
            .expect("conversion should succeed");
        assert_eq!(result.timecode.frames, 10);
    }

    #[test]
    fn test_audio_samples() {
        let tc = Timecode::new(0, 0, 1, 0, FrameRate::Fps25).expect("valid timecode");
        let samples = timecode_to_audio_samples(&tc, 48000);
        assert_eq!(samples, 48000);
    }

    #[test]
    fn test_negative_seconds_error() {
        assert!(seconds_to_timecode(-1.0, FrameRate::Fps25).is_err());
    }
}
