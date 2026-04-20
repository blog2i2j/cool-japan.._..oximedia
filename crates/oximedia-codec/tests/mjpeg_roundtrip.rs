//! MJPEG encode–decode round-trip integration tests.
//!
//! All frames are synthesized in-test; no binary fixtures are used.
//! Each test exercises the full pipeline:
//!   VideoFrame (RGB24) → MjpegEncoder → EncodedPacket
//!                      → MjpegDecoder  → VideoFrame (RGB24) → PSNR check
//!
//! The `oximedia-image` JPEG baseline codec is spec-compliant: DQT is written
//! in zigzag order, AC coefficients are stored in zigzag order, and the decoder
//! correctly dequantizes and remaps to natural order before IDCT.  PSNR
//! thresholds reflect empirically measured fidelity for the adversarial test
//! pattern (modulo-256 wrap-around discontinuities at 128×96 RGB24):
//!   Q50 ≈ 27 dB, Q85 ≈ 33 dB, Q95 ≈ 40 dB.
//! These are intentionally conservative (a few dB below measured) to give
//! head-room for minor platform floating-point differences.

use oximedia_codec::frame::{FrameType, Plane, VideoFrame};
use oximedia_codec::mjpeg::{MjpegConfig, MjpegDecoder, MjpegEncoder};
use oximedia_codec::traits::{VideoDecoder, VideoEncoder};
use oximedia_core::{PixelFormat, Rational, Timestamp};

// ── Frame factory ─────────────────────────────────────────────────────────────

/// Build a 128×96 RGB24 test frame.
///
/// Each pixel (row, col):
///   R = ((row * 2 + col * 3) & 0xFF) as u8
///   G = ((row * 3 + col * 2) & 0xFF) as u8
///   B = ((row + col * 5)     & 0xFF) as u8
fn make_rgb24_frame_128x96() -> VideoFrame {
    const WIDTH: u32 = 128;
    const HEIGHT: u32 = 96;

    let w = WIDTH as usize;
    let h = HEIGHT as usize;
    let mut rgb = vec![0u8; w * h * 3];

    for row in 0..h {
        for col in 0..w {
            let idx = (row * w + col) * 3;
            rgb[idx] = ((row * 2 + col * 3) & 0xFF) as u8; // R
            rgb[idx + 1] = ((row * 3 + col * 2) & 0xFF) as u8; // G
            rgb[idx + 2] = ((row + col * 5) & 0xFF) as u8; // B
        }
    }

    let mut frame = VideoFrame::new(PixelFormat::Rgb24, WIDTH, HEIGHT);
    frame.planes = vec![Plane::with_dimensions(rgb, w * 3, WIDTH, HEIGHT)];
    frame.timestamp = Timestamp::new(0, Rational::new(1, 30000));
    frame.frame_type = FrameType::Key;
    frame
}

// ── PSNR helper ───────────────────────────────────────────────────────────────

/// Compute PSNR (dB) between two equal-length u8 slices.
///
/// Returns `f64::INFINITY` when the slices are bit-exact.
fn compute_psnr(original: &[u8], decoded: &[u8]) -> f64 {
    assert_eq!(
        original.len(),
        decoded.len(),
        "PSNR: slice lengths differ ({} vs {})",
        original.len(),
        decoded.len()
    );
    let mse: f64 = original
        .iter()
        .zip(decoded.iter())
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d
        })
        .sum::<f64>()
        / original.len() as f64;
    if mse < 1e-10 {
        return f64::INFINITY;
    }
    20.0 * (255.0_f64).log10() - 10.0 * mse.log10()
}

// ── Roundtrip helper ──────────────────────────────────────────────────────────

/// Encode an RGB24 frame at the given quality and return the decoded RGB24 pixel data.
///
/// Pipeline: `frame` (RGB24) → `MjpegEncoder(quality)` → `MjpegDecoder` (RGB24 output)
/// → packed RGB `Vec<u8>`.
fn encode_then_decode_rgb(frame: &VideoFrame, quality: u8) -> Vec<u8> {
    let config = MjpegConfig::new(frame.width, frame.height)
        .expect("valid MJPEG config")
        .with_quality(quality)
        .with_pixel_format(PixelFormat::Rgb24);

    let mut encoder = MjpegEncoder::new(config).expect("MjpegEncoder::new");

    encoder.send_frame(frame).expect("encoder::send_frame");

    let pkt = encoder
        .receive_packet()
        .expect("encoder::receive_packet")
        .expect("expected encoded packet");

    // MjpegDecoder defaults to Rgb24 output.
    let mut decoder = MjpegDecoder::new(frame.width, frame.height);

    decoder
        .send_packet(&pkt.data, pkt.pts)
        .expect("decoder::send_packet");

    let decoded = decoder
        .receive_frame()
        .expect("decoder::receive_frame")
        .expect("expected decoded frame");

    assert_eq!(
        decoded.format,
        PixelFormat::Rgb24,
        "MJPEG decoder default output must be RGB24"
    );
    assert!(
        !decoded.planes.is_empty(),
        "decoded frame must have at least one plane"
    );
    assert_eq!(decoded.width, frame.width, "decoded width must match input");
    assert_eq!(
        decoded.height, frame.height,
        "decoded height must match input"
    );

    decoded.planes[0].data.clone()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Encode a 128×96 RGB24 frame at quality 85, decode, and verify round-trip
/// fidelity.  The adversarial modulo-256 test pattern achieves ≥ 32 dB at Q85.
#[test]
fn test_mjpeg_roundtrip_q85() {
    let frame = make_rgb24_frame_128x96();

    let original_rgb = frame.planes[0].data.clone();
    let decoded_rgb = encode_then_decode_rgb(&frame, 85);

    let psnr = compute_psnr(&original_rgb, &decoded_rgb);

    assert!(
        psnr >= 32.0,
        "MJPEG Q85 RGB PSNR should be ≥ 32 dB, got {psnr:.2} dB"
    );
}

/// Higher JPEG quality must produce higher (or equal) PSNR.
///
/// Tests at Q50, Q85, Q95 and asserts:
///   - absolute minimums: Q50 ≥ 26 dB, Q85 ≥ 32 dB, Q95 ≥ 39 dB
///     (empirically measured with 128×96 adversarial wrap pattern, with 1 dB margin)
///   - monotonic ordering: PSNR(50) ≤ PSNR(85) ≤ PSNR(95)
#[test]
fn test_mjpeg_psnr_monotonic() {
    let frame = make_rgb24_frame_128x96();
    let original_rgb = frame.planes[0].data.clone();

    let psnr_50 = compute_psnr(&original_rgb, &encode_then_decode_rgb(&frame, 50));
    let psnr_85 = compute_psnr(&original_rgb, &encode_then_decode_rgb(&frame, 85));
    let psnr_95 = compute_psnr(&original_rgb, &encode_then_decode_rgb(&frame, 95));

    assert!(
        psnr_50 >= 26.0,
        "Q50 PSNR must be ≥ 26 dB, got {psnr_50:.2} dB"
    );
    assert!(
        psnr_85 >= 32.0,
        "Q85 PSNR must be ≥ 32 dB, got {psnr_85:.2} dB"
    );
    assert!(
        psnr_95 >= 39.0,
        "Q95 PSNR must be ≥ 39 dB, got {psnr_95:.2} dB"
    );
    assert!(
        psnr_50 <= psnr_85,
        "Q85 PSNR ({psnr_85:.2} dB) must be ≥ Q50 PSNR ({psnr_50:.2} dB)"
    );
    assert!(
        psnr_85 <= psnr_95,
        "Q95 PSNR ({psnr_95:.2} dB) must be ≥ Q85 PSNR ({psnr_85:.2} dB)"
    );
}
