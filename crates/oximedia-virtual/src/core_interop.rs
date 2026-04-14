#![allow(dead_code)]
//! Integration with `oximedia-core` frame types.
//!
//! This module bridges the virtual production data structures to the canonical
//! `oximedia-core` frame types (`VideoFrameInfo`, `Timestamp`, `PixelFormat`,
//! etc.) so that virtual production outputs can interoperate directly with the
//! rest of the OxiMedia pipeline.
//!
//! # Design
//!
//! Virtual production internally uses its own lightweight pixel and colour
//! structures (defined in `led`, `color`, `icvfx` etc.) tuned for real-time
//! throughput.  When video needs to be handed off to encoding, streaming, or
//! compositing layers that use `oximedia-core` types, callers use the
//! conversion helpers here.

use oximedia_core::frame_info::{ColorPrimaries, FrameType, VideoFrameInfo};
use oximedia_core::types::{PixelFormat, Rational, Timestamp};

// ─────────────────────────────────────────────────────────────────
// VP Frame ↔ VideoFrameInfo
// ─────────────────────────────────────────────────────────────────

/// A virtual production rendered frame with full oximedia-core metadata.
///
/// Wraps the raw RGBA pixel data produced by the LED wall renderer together
/// with the `VideoFrameInfo` descriptor required by downstream pipeline stages.
#[derive(Debug, Clone)]
pub struct VpVideoFrame {
    /// Raw RGBA pixel data (4 bytes per pixel, row-major).
    pub data: Vec<u8>,
    /// Frame metadata compatible with `oximedia-core`.
    pub info: VideoFrameInfo,
    /// Presentation timestamp.
    pub pts: Timestamp,
    /// Frame sequence number (0-based).
    pub sequence: u64,
}

impl VpVideoFrame {
    /// Create a new virtual-production frame.
    #[must_use]
    pub fn new(data: Vec<u8>, info: VideoFrameInfo, pts: Timestamp, sequence: u64) -> Self {
        Self {
            data,
            info,
            pts,
            sequence,
        }
    }

    /// Construct a frame descriptor for an RGBA LED wall output at the given
    /// resolution and frame rate.
    ///
    /// * `width` / `height` — Output dimensions in pixels.
    /// * `fps` — Frame rate as a [`Rational`].
    /// * `pts` — Presentation timestamp for this frame.
    /// * `sequence` — Zero-based frame index.
    /// * `data` — Raw RGBA pixel data (must be `width * height * 4` bytes).
    #[must_use]
    pub fn led_output(
        data: Vec<u8>,
        width: u32,
        height: u32,
        fps: Rational,
        sequence: u64,
    ) -> Self {
        // Compute pts from sequence number and fps.
        let pts = frame_to_timestamp(sequence, fps);
        let dur = if fps.num != 0 {
            fps.den as i64
        } else {
            0
        };
        let info = VideoFrameInfo::new(
            pts.pts,
            pts.pts,
            dur,
            width,
            height,
            FrameType::Intra,
            ColorPrimaries::Bt709,
            0,
        );
        Self::new(data, info, pts, sequence)
    }

    /// Returns the number of bytes per pixel (always 4 for RGBA VP output).
    #[must_use]
    pub const fn bytes_per_pixel() -> usize {
        4
    }

    /// Expected data length in bytes.
    #[must_use]
    pub fn expected_data_len(&self) -> usize {
        self.info.width as usize * self.info.height as usize * Self::bytes_per_pixel()
    }

    /// Returns `true` if `data.len()` matches the expected size.
    #[must_use]
    pub fn is_data_valid(&self) -> bool {
        self.data.len() == self.expected_data_len()
    }

    /// Get a pixel value at `(x, y)`.
    ///
    /// Returns `None` if the coordinates are out of bounds or data is invalid.
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.info.width || y >= self.info.height {
            return None;
        }
        let idx = (y as usize * self.info.width as usize + x as usize) * 4;
        if idx + 4 > self.data.len() {
            return None;
        }
        Some([
            self.data[idx],
            self.data[idx + 1],
            self.data[idx + 2],
            self.data[idx + 3],
        ])
    }
}

// ─────────────────────────────────────────────────────────────────
// Timestamp helpers
// ─────────────────────────────────────────────────────────────────

/// Convert a frame sequence index to a [`Timestamp`] given a frame rate.
///
/// The timestamp is expressed in the timebase `1/fps_den` units (i.e., `fps_den`
/// ticks per fps_den denominator units).  For 24/1 fps, frame 12 → pts = 12,
/// timebase = 1/1.
#[must_use]
pub fn frame_to_timestamp(frame: u64, fps: Rational) -> Timestamp {
    if fps.num == 0 {
        return Timestamp::new(0, Rational::new(1, 1));
    }
    // Use timebase 1/fps_num so each frame maps to fps_den ticks.
    // e.g. 24fps → timebase 1/24, pts_val = frame * 1
    // e.g. 30000/1001fps → timebase 1/30000, pts_val = frame * 1001
    let timebase = Rational::new(1, fps.num);
    let pts_val = frame as i64 * fps.den as i64;
    Timestamp::new(pts_val, timebase)
}

/// Convert a [`Timestamp`] back to the nearest frame index at the given fps.
#[must_use]
pub fn timestamp_to_frame(pts: Timestamp, fps: Rational) -> u64 {
    if fps.den == 0 || pts.timebase.den == 0 {
        return 0;
    }
    let pts_secs = pts.to_seconds();
    let fps_f = fps.num as f64 / fps.den as f64;
    (pts_secs * fps_f).round() as u64
}

// ─────────────────────────────────────────────────────────────────
// PixelFormat helpers
// ─────────────────────────────────────────────────────────────────

/// Returns `true` if the given [`PixelFormat`] can hold HDR content (10-bit+).
#[must_use]
pub const fn is_hdr_pixel_format(fmt: PixelFormat) -> bool {
    matches!(
        fmt,
        PixelFormat::Yuv420p10le
            | PixelFormat::Yuv420p12le
            | PixelFormat::Gray16
            | PixelFormat::P010
    )
}

/// Returns the approximate bytes-per-pixel for the common packed formats.
/// Returns `None` for planar or exotic formats (consult `PixelFormat::bytes_per_pixel_approx()`
/// for those).
#[must_use]
pub const fn packed_bytes_per_pixel(fmt: PixelFormat) -> Option<usize> {
    match fmt {
        PixelFormat::Rgba32 => Some(4),
        PixelFormat::Rgb24 => Some(3),
        PixelFormat::Gray8 => Some(1),
        PixelFormat::Gray16 => Some(2),
        _ => None,
    }
}

/// Create a [`VideoFrameInfo`] suitable for an RGBA32 VP output frame.
#[must_use]
pub fn rgba_frame_info(width: u32, height: u32, fps: Rational, frame_idx: u64) -> VideoFrameInfo {
    let pts = frame_to_timestamp(frame_idx, fps);
    let dur = if fps.num != 0 { fps.den as i64 } else { 0 };
    VideoFrameInfo::new(
        pts.pts,
        pts.pts,
        dur,
        width,
        height,
        FrameType::Intra,
        ColorPrimaries::Bt709,
        0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_led_output_frame_info() {
        let fps = Rational::new(24, 1);
        let data = vec![0u8; 4 * 4 * 4]; // 4x4 RGBA
        let frame = VpVideoFrame::led_output(data, 4, 4, fps, 0);
        assert_eq!(frame.info.width, 4);
        assert_eq!(frame.info.height, 4);
        assert!(frame.is_data_valid());
    }

    #[test]
    fn test_pixel_out_of_bounds() {
        let fps = Rational::new(24, 1);
        let data = vec![255u8; 4 * 4 * 4];
        let frame = VpVideoFrame::led_output(data, 4, 4, fps, 0);
        assert!(frame.pixel(3, 3).is_some());
        assert!(frame.pixel(4, 0).is_none());
    }

    #[test]
    fn test_frame_to_timestamp_24fps() {
        let fps = Rational::new(24, 1);
        let ts = frame_to_timestamp(24, fps);
        // 24 frames at 24fps = 1 second
        assert!((ts.to_seconds() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_timestamp_to_frame_roundtrip() {
        let fps = Rational::new(24, 1);
        for f in [0u64, 1, 12, 24, 100] {
            let ts = frame_to_timestamp(f, fps);
            let back = timestamp_to_frame(ts, fps);
            assert_eq!(back, f, "round-trip failed at frame {f}");
        }
    }

    #[test]
    fn test_is_hdr_pixel_format() {
        assert!(is_hdr_pixel_format(PixelFormat::Yuv420p10le));
        assert!(!is_hdr_pixel_format(PixelFormat::Rgba32));
    }

    #[test]
    fn test_packed_bytes_per_pixel() {
        assert_eq!(packed_bytes_per_pixel(PixelFormat::Rgba32), Some(4));
        assert_eq!(packed_bytes_per_pixel(PixelFormat::Rgb24), Some(3));
        assert_eq!(packed_bytes_per_pixel(PixelFormat::Gray8), Some(1));
        assert!(packed_bytes_per_pixel(PixelFormat::Yuv420p).is_none());
    }

    #[test]
    fn test_rgba_frame_info() {
        let fps = Rational::new(30, 1);
        let info = rgba_frame_info(1920, 1080, fps, 0);
        assert_eq!(info.width, 1920);
        assert_eq!(info.height, 1080);
    }
}
