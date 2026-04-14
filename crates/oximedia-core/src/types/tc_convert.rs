//! Frame number ↔ timecode conversion utilities.
//!
//! Converts between absolute frame counts and `(hours, minutes, seconds, frame)`
//! components at a given frame rate.  Both integer and fractional (NTSC) frame
//! rates are supported.

/// Convert an absolute frame number to `(hours, minutes, seconds, frame)`.
///
/// Uses integer arithmetic based on `floor(fps)`.
///
/// # Arguments
///
/// * `frame_num` - Zero-based absolute frame number.
/// * `fps`       - Frame rate (frames per second, must be > 0).
///
/// # Returns
///
/// `(hours, minutes, seconds, frame)` as `(u8, u8, u8, u32)`.
/// Returns `(0, 0, 0, 0)` if `fps <= 0`.
#[must_use]
pub fn frame_to_tc(frame_num: u64, fps: f32) -> (u8, u8, u8, u32) {
    let fps_int = fps.floor() as u64;
    if fps_int == 0 {
        return (0, 0, 0, 0);
    }

    let frame = (frame_num % fps_int) as u32;
    let total_secs = frame_num / fps_int;
    let seconds = (total_secs % 60) as u8;
    let total_mins = total_secs / 60;
    let minutes = (total_mins % 60) as u8;
    let hours = ((total_mins / 60) % 24) as u8;

    (hours, minutes, seconds, frame)
}

/// Convert `(hours, minutes, seconds, frame)` to an absolute frame number.
///
/// Uses integer arithmetic based on `floor(fps)`.
///
/// # Arguments
///
/// * `h`   - Hours (0–23).
/// * `m`   - Minutes (0–59).
/// * `s`   - Seconds (0–59).
/// * `f`   - Frame within the second (0-indexed).
/// * `fps` - Frame rate (frames per second, must be > 0).
///
/// # Returns
///
/// Absolute zero-based frame number, or 0 if `fps <= 0`.
#[must_use]
pub fn tc_to_frame(h: u8, m: u8, s: u8, f: u32, fps: f32) -> u64 {
    let fps_int = fps.floor() as u64;
    if fps_int == 0 {
        return 0;
    }

    let total_secs = h as u64 * 3600 + m as u64 * 60 + s as u64;
    total_secs * fps_int + f as u64
}

/// Convert a frame number to total seconds (floating-point).
///
/// Useful for synchronising with wall-clock time.
#[must_use]
pub fn frame_to_seconds(frame_num: u64, fps: f32) -> f64 {
    if fps <= 0.0 {
        return 0.0;
    }
    frame_num as f64 / fps as f64
}

/// Convert seconds to the nearest frame number at the given frame rate.
#[must_use]
pub fn seconds_to_frame(seconds: f64, fps: f32) -> u64 {
    if fps <= 0.0 || seconds < 0.0 {
        return 0;
    }
    (seconds * fps as f64).round() as u64
}

/// Format `(h, m, s, f)` as a timecode string `"HH:MM:SS:FF"`.
#[must_use]
pub fn format_tc_ndf(h: u8, m: u8, s: u8, f: u32) -> String {
    format!("{h:02}:{m:02}:{s:02}:{f:02}")
}

/// Format `(h, m, s, f)` as a drop-frame timecode string `"HH:MM:SS;FF"`.
#[must_use]
pub fn format_tc_df(h: u8, m: u8, s: u8, f: u32) -> String {
    format!("{h:02}:{m:02}:{s:02};{f:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_to_tc_zero() {
        assert_eq!(frame_to_tc(0, 24.0), (0, 0, 0, 0));
    }

    #[test]
    fn test_frame_to_tc_one_second() {
        assert_eq!(frame_to_tc(24, 24.0), (0, 0, 1, 0));
    }

    #[test]
    fn test_frame_to_tc_one_minute() {
        assert_eq!(frame_to_tc(24 * 60, 24.0), (0, 1, 0, 0));
    }

    #[test]
    fn test_frame_to_tc_one_hour() {
        assert_eq!(frame_to_tc(24 * 3600, 24.0), (1, 0, 0, 0));
    }

    #[test]
    fn test_frame_to_tc_mid_frame() {
        // 1 hour + 2 min + 3 sec + 5 frames @ 25 fps
        let f = 25 * (3600 + 120 + 3) + 5;
        assert_eq!(frame_to_tc(f, 25.0), (1, 2, 3, 5));
    }

    #[test]
    fn test_tc_to_frame_zero() {
        assert_eq!(tc_to_frame(0, 0, 0, 0, 24.0), 0);
    }

    #[test]
    fn test_tc_to_frame_one_second() {
        assert_eq!(tc_to_frame(0, 0, 1, 0, 25.0), 25);
    }

    #[test]
    fn test_tc_to_frame_one_hour() {
        assert_eq!(tc_to_frame(1, 0, 0, 0, 30.0), 30 * 3600);
    }

    #[test]
    fn test_roundtrip_frame_to_tc_to_frame() {
        for fps in [24.0_f32, 25.0, 30.0, 50.0, 60.0] {
            for frame_num in [0_u64, 1, 100, 86399, 86400, 86401] {
                let (h, m, s, f) = frame_to_tc(frame_num, fps);
                let back = tc_to_frame(h, m, s, f, fps);
                assert_eq!(back, frame_num, "fps={fps} frame_num={frame_num}");
            }
        }
    }

    #[test]
    fn test_frame_to_tc_zero_fps() {
        assert_eq!(frame_to_tc(100, 0.0), (0, 0, 0, 0));
    }

    #[test]
    fn test_tc_to_frame_zero_fps() {
        assert_eq!(tc_to_frame(1, 0, 0, 0, 0.0), 0);
    }

    #[test]
    fn test_frame_to_seconds() {
        assert!((frame_to_seconds(24, 24.0) - 1.0).abs() < 1e-9);
        assert!((frame_to_seconds(0, 25.0) - 0.0).abs() < 1e-9);
        assert!((frame_to_seconds(75, 25.0) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_seconds_to_frame() {
        assert_eq!(seconds_to_frame(1.0, 24.0), 24);
        assert_eq!(seconds_to_frame(0.0, 25.0), 0);
        assert_eq!(seconds_to_frame(3.0, 30.0), 90);
    }

    #[test]
    fn test_seconds_to_frame_negative() {
        assert_eq!(seconds_to_frame(-1.0, 24.0), 0);
    }

    #[test]
    fn test_format_tc_ndf() {
        assert_eq!(format_tc_ndf(1, 2, 3, 15), "01:02:03:15");
    }

    #[test]
    fn test_format_tc_df() {
        assert_eq!(format_tc_df(0, 59, 59, 29), "00:59:59;29");
    }

    #[test]
    fn test_ntsc_29_97_approximation() {
        // 29.97 fps: floor = 29
        let (h, m, s, f) = frame_to_tc(29, 29.97);
        assert_eq!((h, m, s, f), (0, 0, 1, 0));
    }
}
