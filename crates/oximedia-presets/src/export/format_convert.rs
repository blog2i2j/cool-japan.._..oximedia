//! Format conversion helpers for preset import/export.
//!
//! Provides utilities for converting preset parameters between different
//! representation formats (e.g., kbps ↔ bps, percentage strings, codec names).

#![allow(dead_code)]

/// Convert a bitrate value in bits-per-second to kbps (as a f64).
pub fn bps_to_kbps(bps: u64) -> f64 {
    bps as f64 / 1000.0
}

/// Convert a bitrate value in kbps to bits-per-second (as a u64).
///
/// Fractional kbps values are rounded to the nearest bps.
pub fn kbps_to_bps(kbps: f64) -> u64 {
    (kbps * 1000.0).round() as u64
}

/// Convert a bitrate value in bits-per-second to Mbps (as a f64).
pub fn bps_to_mbps(bps: u64) -> f64 {
    bps as f64 / 1_000_000.0
}

/// Convert a bitrate value in Mbps to bits-per-second.
pub fn mbps_to_bps(mbps: f64) -> u64 {
    (mbps * 1_000_000.0).round() as u64
}

/// Format a bitrate as a human-readable string (auto-scales to kbps or Mbps).
pub fn format_bitrate(bps: u64) -> String {
    if bps >= 1_000_000 {
        format!("{:.2} Mbps", bps_to_mbps(bps))
    } else {
        format!("{:.1} kbps", bps_to_kbps(bps))
    }
}

/// Codec name normalisation table.
const CODEC_ALIASES: &[(&str, &str)] = &[
    ("h264", "libx264"),
    ("avc", "libx264"),
    ("hevc", "libx265"),
    ("h265", "libx265"),
    ("vp9", "libvpx-vp9"),
    ("av1", "libaom-av1"),
    ("aac", "aac"),
    ("mp3", "libmp3lame"),
    ("opus", "libopus"),
    ("vorbis", "libvorbis"),
];

/// Normalise a codec name to its canonical FFmpeg library name.
///
/// Returns the input unchanged if no alias is found.
pub fn normalise_codec_name(name: &str) -> &str {
    let lower = name.to_lowercase();
    for &(alias, canonical) in CODEC_ALIASES {
        if alias == lower.as_str() {
            return canonical;
        }
    }
    name
}

/// Friendly codec name (inverse lookup from FFmpeg library to short name).
pub fn friendly_codec_name(ffmpeg_name: &str) -> &str {
    for &(friendly, canonical) in CODEC_ALIASES {
        if canonical == ffmpeg_name {
            return friendly;
        }
    }
    ffmpeg_name
}

/// Parse a resolution string like "1920x1080" into `(width, height)`.
pub fn parse_resolution(s: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() != 2 {
        return None;
    }
    let width: u32 = parts[0].trim().parse().ok()?;
    let height: u32 = parts[1].trim().parse().ok()?;
    Some((width, height))
}

/// Format a resolution pair as a string like "1920x1080".
pub fn format_resolution(width: u32, height: u32) -> String {
    format!("{}x{}", width, height)
}

/// Parse a percentage string like "85%" into a `f32` in the range 0.0–1.0.
pub fn parse_percentage(s: &str) -> Option<f32> {
    let trimmed = s.trim().trim_end_matches('%');
    let pct: f32 = trimmed.parse().ok()?;
    Some(pct / 100.0)
}

/// Format a fraction (0.0–1.0) as a percentage string like "85.0%".
pub fn format_percentage(fraction: f32) -> String {
    format!("{:.1}%", fraction * 100.0)
}

/// Convert a CRF value to an approximate quality percentage (0–100).
///
/// For H.264/H.265 where CRF ranges 0–51 (0 = lossless, 51 = worst).
pub fn crf_to_quality_percent(crf: u32) -> f32 {
    let crf_clamped = crf.min(51) as f32;
    100.0 * (1.0 - crf_clamped / 51.0)
}

/// Convert a quality percentage (0–100) to an approximate CRF value.
pub fn quality_percent_to_crf(quality: f32) -> u32 {
    let q = quality.clamp(0.0, 100.0);
    ((1.0 - q / 100.0) * 51.0).round() as u32
}

/// Pixel aspect ratio as a reduced fraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixelAspectRatio {
    /// Numerator.
    pub num: u32,
    /// Denominator.
    pub den: u32,
}

impl PixelAspectRatio {
    /// Create a new PAR, reducing the fraction.
    pub fn new(num: u32, den: u32) -> Option<Self> {
        if den == 0 {
            return None;
        }
        let g = gcd(num, den);
        Some(Self {
            num: num / g,
            den: den / g,
        })
    }

    /// Display as "num:den".
    pub fn to_string_repr(&self) -> String {
        format!("{}:{}", self.num, self.den)
    }

    /// Compute the floating-point ratio.
    pub fn ratio(&self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    if a == 0 {
        1
    } else {
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bps_to_kbps() {
        assert!((bps_to_kbps(5_000_000) - 5000.0).abs() < 1e-6);
    }

    #[test]
    fn test_kbps_to_bps() {
        assert_eq!(kbps_to_bps(5000.0), 5_000_000);
    }

    #[test]
    fn test_bps_to_mbps() {
        assert!((bps_to_mbps(5_000_000) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_mbps_to_bps() {
        assert_eq!(mbps_to_bps(5.0), 5_000_000);
    }

    #[test]
    fn test_format_bitrate_mbps() {
        let s = format_bitrate(5_000_000);
        assert!(s.contains("Mbps"));
    }

    #[test]
    fn test_format_bitrate_kbps() {
        let s = format_bitrate(192_000);
        assert!(s.contains("kbps"));
    }

    #[test]
    fn test_normalise_codec_name_h264() {
        assert_eq!(normalise_codec_name("h264"), "libx264");
        assert_eq!(normalise_codec_name("avc"), "libx264");
    }

    #[test]
    fn test_normalise_codec_name_unknown() {
        assert_eq!(normalise_codec_name("my-codec"), "my-codec");
    }

    #[test]
    fn test_friendly_codec_name() {
        assert_eq!(friendly_codec_name("libx264"), "h264");
        assert_eq!(friendly_codec_name("libx265"), "hevc");
    }

    #[test]
    fn test_parse_resolution_valid() {
        let (w, h) = parse_resolution("1920x1080").expect("test expectation failed");
        assert_eq!(w, 1920);
        assert_eq!(h, 1080);
    }

    #[test]
    fn test_parse_resolution_invalid() {
        assert!(parse_resolution("1920").is_none());
        assert!(parse_resolution("AxB").is_none());
    }

    #[test]
    fn test_format_resolution() {
        assert_eq!(format_resolution(1920, 1080), "1920x1080");
    }

    #[test]
    fn test_parse_percentage() {
        let v = parse_percentage("85%").expect("v should be valid");
        assert!((v - 0.85).abs() < 1e-5);
    }

    #[test]
    fn test_parse_percentage_no_sign() {
        let v = parse_percentage("50").expect("v should be valid");
        assert!((v - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_format_percentage() {
        assert_eq!(format_percentage(0.85), "85.0%");
        assert_eq!(format_percentage(1.0), "100.0%");
    }

    #[test]
    fn test_crf_to_quality() {
        assert!((crf_to_quality_percent(0) - 100.0).abs() < 1e-3);
        assert!((crf_to_quality_percent(51) - 0.0).abs() < 1e-3);
        let mid = crf_to_quality_percent(23);
        assert!(mid > 0.0 && mid < 100.0);
    }

    #[test]
    fn test_quality_to_crf_roundtrip() {
        let crf = 23u32;
        let quality = crf_to_quality_percent(crf);
        let back = quality_percent_to_crf(quality);
        assert!((crf as i32 - back as i32).abs() <= 1);
    }

    #[test]
    fn test_pixel_aspect_ratio_new() {
        let par = PixelAspectRatio::new(16, 9).expect("par should be valid");
        assert_eq!(par.num, 16);
        assert_eq!(par.den, 9);
        assert_eq!(par.to_string_repr(), "16:9");
    }

    #[test]
    fn test_pixel_aspect_ratio_reduction() {
        let par = PixelAspectRatio::new(4, 2).expect("par should be valid");
        assert_eq!(par.num, 2);
        assert_eq!(par.den, 1);
    }

    #[test]
    fn test_pixel_aspect_ratio_zero_den() {
        assert!(PixelAspectRatio::new(16, 0).is_none());
    }

    #[test]
    fn test_pixel_aspect_ratio_ratio() {
        let par = PixelAspectRatio::new(16, 9).expect("par should be valid");
        assert!((par.ratio() - 16.0 / 9.0).abs() < 1e-9);
    }
}
