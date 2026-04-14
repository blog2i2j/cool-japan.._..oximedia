//! Bitrate estimation and file-size utilities.
//!
//! Provides functions to convert between file size, duration, and bitrate,
//! which is useful for streaming profiles, transcode planning, and storage
//! quota calculations.

/// Calculate the bitrate (in bits per second) required to fit a file of
/// `size_bytes` into `duration_secs` of playback.
///
/// # Arguments
///
/// * `size_bytes`    - File or stream size in bytes.
/// * `duration_secs` - Playback duration in seconds (must be > 0).
///
/// # Returns
///
/// Bitrate in bits per second, or 0 if `duration_secs <= 0`.
#[must_use]
pub fn bitrate_for_file(size_bytes: u64, duration_secs: f64) -> u64 {
    if duration_secs <= 0.0 {
        return 0;
    }
    let bits = size_bytes as f64 * 8.0;
    (bits / duration_secs).round() as u64
}

/// Calculate the expected file size (in bytes) for a given bitrate and duration.
///
/// # Arguments
///
/// * `bitrate_kbps`  - Bitrate in **kilobits per second** (kbps).
/// * `duration_secs` - Duration in seconds (must be > 0).
///
/// # Returns
///
/// File size in bytes, or 0 if `duration_secs <= 0` or `bitrate_kbps == 0`.
#[must_use]
pub fn file_size_for_bitrate(bitrate_kbps: u64, duration_secs: f64) -> u64 {
    if duration_secs <= 0.0 || bitrate_kbps == 0 {
        return 0;
    }
    let bits = bitrate_kbps as f64 * 1000.0 * duration_secs;
    (bits / 8.0).round() as u64
}

/// Convert bitrate in bits per second to kilobits per second.
#[must_use]
pub fn bps_to_kbps(bps: u64) -> u64 {
    bps / 1000
}

/// Convert bitrate in kilobits per second to bits per second.
#[must_use]
pub fn kbps_to_bps(kbps: u64) -> u64 {
    kbps * 1000
}

/// Convert bitrate in bits per second to megabits per second.
#[must_use]
pub fn bps_to_mbps(bps: u64) -> f64 {
    bps as f64 / 1_000_000.0
}

/// Calculate the average video bitrate given total bitrate, audio bitrate, and
/// number of audio tracks.
///
/// `video_bitrate = total_bitrate - audio_bitrate * audio_tracks`
///
/// Returns 0 if audio would exceed total.
#[must_use]
pub fn video_bitrate(total_bps: u64, audio_bps_per_track: u64, audio_tracks: u32) -> u64 {
    let total_audio = audio_bps_per_track.saturating_mul(audio_tracks as u64);
    total_bps.saturating_sub(total_audio)
}

/// Encode resolution metadata alongside bitrate for per-title encoding ladders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitrateRung {
    /// Horizontal resolution in pixels.
    pub width: u32,
    /// Vertical resolution in pixels.
    pub height: u32,
    /// Target bitrate in kbps.
    pub bitrate_kbps: u64,
}

impl BitrateRung {
    /// Create a new bitrate ladder rung.
    #[must_use]
    pub const fn new(width: u32, height: u32, bitrate_kbps: u64) -> Self {
        Self {
            width,
            height,
            bitrate_kbps,
        }
    }

    /// Estimated file size for `duration_secs` at this rung.
    #[must_use]
    pub fn estimated_size_bytes(&self, duration_secs: f64) -> u64 {
        file_size_for_bitrate(self.bitrate_kbps, duration_secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitrate_for_file_basic() {
        // 1 MB file at 8 seconds → 1_000_000 * 8 / 8 = 1_000_000 bps = 1 Mbps
        let bps = bitrate_for_file(1_000_000, 8.0);
        assert_eq!(bps, 1_000_000);
    }

    #[test]
    fn test_bitrate_for_file_zero_duration() {
        assert_eq!(bitrate_for_file(1_000_000, 0.0), 0);
        assert_eq!(bitrate_for_file(1_000_000, -1.0), 0);
    }

    #[test]
    fn test_bitrate_for_file_small_file() {
        // 100 bytes over 1 second → 800 bps
        assert_eq!(bitrate_for_file(100, 1.0), 800);
    }

    #[test]
    fn test_file_size_for_bitrate_basic() {
        // 1000 kbps * 8 seconds = 8_000_000 bits = 1_000_000 bytes
        let bytes = file_size_for_bitrate(1000, 8.0);
        assert_eq!(bytes, 1_000_000);
    }

    #[test]
    fn test_file_size_for_bitrate_zero_duration() {
        assert_eq!(file_size_for_bitrate(1000, 0.0), 0);
    }

    #[test]
    fn test_file_size_for_bitrate_zero_bitrate() {
        assert_eq!(file_size_for_bitrate(0, 60.0), 0);
    }

    #[test]
    fn test_bps_to_kbps() {
        assert_eq!(bps_to_kbps(5_000_000), 5_000);
        assert_eq!(bps_to_kbps(0), 0);
    }

    #[test]
    fn test_kbps_to_bps() {
        assert_eq!(kbps_to_bps(5_000), 5_000_000);
    }

    #[test]
    fn test_bps_to_mbps() {
        assert!((bps_to_mbps(5_000_000) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_video_bitrate() {
        // 8 Mbps total, 192 kbps audio × 2 tracks = 384 kbps audio
        let total = 8_000_000_u64;
        let audio = 192_000_u64;
        let vb = video_bitrate(total, audio, 2);
        assert_eq!(vb, total - 2 * audio);
    }

    #[test]
    fn test_video_bitrate_audio_exceeds_total() {
        // Should saturate at 0
        assert_eq!(video_bitrate(100, 200, 1), 0);
    }

    #[test]
    fn test_bitrate_rung_estimated_size() {
        let rung = BitrateRung::new(1920, 1080, 4000);
        // 4000 kbps * 10s = 40_000_000 bits = 5_000_000 bytes
        assert_eq!(rung.estimated_size_bytes(10.0), 5_000_000);
    }

    #[test]
    fn test_roundtrip_bitrate_and_size() {
        let original_bytes = 50_000_000_u64; // 50 MB
        let duration = 60.0; // 1 minute
        let bps = bitrate_for_file(original_bytes, duration);
        let kbps = bps_to_kbps(bps);
        let recovered = file_size_for_bitrate(kbps, duration);
        // Allow up to 1% error from rounding
        let diff = (recovered as i64 - original_bytes as i64).unsigned_abs();
        assert!(
            diff < original_bytes / 100,
            "original={original_bytes} recovered={recovered} bps={bps}"
        );
    }

    // ── Additional tests ──────────────────────────────────────────────────────

    #[test]
    fn test_bitrate_for_file_large_file() {
        // 10 GB file over 2 hours = 7200 seconds
        let bps = bitrate_for_file(10_000_000_000_u64, 7200.0);
        // 10e9 * 8 / 7200 ≈ 11_111_111 bps ≈ 11.1 Mbps
        assert!((bps as f64 - 11_111_111.0).abs() < 100.0);
    }

    #[test]
    fn test_file_size_for_bitrate_streaming_profile() {
        // Typical OTT: 4500 kbps for 30 minutes
        let bytes = file_size_for_bitrate(4500, 1800.0);
        // 4500 kbps * 1000 * 1800 / 8 = 1_012_500_000
        assert_eq!(bytes, 1_012_500_000);
    }

    #[test]
    fn test_bps_to_mbps_zero() {
        assert!((bps_to_mbps(0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bps_to_mbps_10mbps() {
        assert!((bps_to_mbps(10_000_000) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_kbps_roundtrip() {
        let original_kbps: u64 = 48_000;
        let bps = kbps_to_bps(original_kbps);
        let recovered = bps_to_kbps(bps);
        assert_eq!(recovered, original_kbps);
    }

    #[test]
    fn test_video_bitrate_8mbps_with_stereo_opus() {
        // 8 Mbps total, 128 kbps stereo Opus
        let vb = video_bitrate(8_000_000, 128_000, 1);
        assert_eq!(vb, 7_872_000);
    }

    #[test]
    fn test_video_bitrate_multi_audio_tracks() {
        // 10 Mbps total, 5 audio tracks at 192 kbps each = 960 kbps audio
        let vb = video_bitrate(10_000_000, 192_000, 5);
        assert_eq!(vb, 10_000_000 - 5 * 192_000);
    }

    #[test]
    fn test_bitrate_rung_new_fields() {
        let rung = BitrateRung::new(3840, 2160, 15_000); // 4K @ 15 Mbps
        assert_eq!(rung.width, 3840);
        assert_eq!(rung.height, 2160);
        assert_eq!(rung.bitrate_kbps, 15_000);
    }

    #[test]
    fn test_bitrate_rung_estimated_size_two_hours() {
        let rung = BitrateRung::new(1920, 1080, 8000); // 8 Mbps
                                                       // 8000 kbps * 1000 * 7200 / 8 = 7_200_000_000 bytes
        assert_eq!(rung.estimated_size_bytes(7200.0), 7_200_000_000);
    }

    #[test]
    fn test_bitrate_for_file_negative_duration_returns_zero() {
        assert_eq!(bitrate_for_file(100_000, -5.0), 0);
        assert_eq!(bitrate_for_file(0, 10.0), 0);
    }
}
