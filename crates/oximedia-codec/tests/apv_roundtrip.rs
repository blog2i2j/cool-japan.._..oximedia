//! APV encode–decode round-trip integration tests.
//!
//! All frames are synthesized in-test; no binary fixtures are used.
//! Each test exercises the full pipeline:
//!   VideoFrame (YUV420p) → ApvEncoder → EncodedPacket
//!                        → ApvDecoder  → VideoFrame (YUV420p) → PSNR check
//!
//! APV is a DCT/QP-based lossy intra-frame codec with no lossless mode.
//! PSNR thresholds reflect observed implementation behaviour: at QP=22 the
//! implementation achieves roughly 22–24 dB on synthetic gradients.  The
//! threshold is set conservatively below this to give headroom without being
//! trivial.  See `test_encoder_decoder_roundtrip_psnr` in `decoder.rs` for
//! the reference baseline (QP=10 → >25 dB on a 64×64 gradient).

use oximedia_codec::apv::{ApvConfig, ApvDecoder, ApvEncoder};
use oximedia_codec::frame::{FrameType, Plane, VideoFrame};
use oximedia_codec::traits::{VideoDecoder, VideoEncoder};
use oximedia_core::{PixelFormat, Rational, Timestamp};

// ── Frame factory ─────────────────────────────────────────────────────────────

/// Build a 96×64 YUV420p test frame.
///
/// Luma  : `y[row * width + col] = ((row * 3 + col * 5) & 0xFF) as u8`
/// Chroma: constant 128 (neutral grey)
///
/// YUV420 is used because it is APV's default chroma format; the frame
/// geometry therefore requires no chroma subsampling conversion.
fn make_yuv420p_frame_96x64() -> VideoFrame {
    const WIDTH: u32 = 96;
    const HEIGHT: u32 = 64;

    let w = WIDTH as usize;
    let h = HEIGHT as usize;
    let chroma_w = (w + 1) / 2;
    let chroma_h = (h + 1) / 2;

    // Luma — synthetic gradient pattern
    let mut y_data = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            y_data[row * w + col] = ((row * 3 + col * 5) & 0xFF) as u8;
        }
    }

    // Chroma — neutral grey (128 = no colour cast)
    let cb_data = vec![128u8; chroma_w * chroma_h];
    let cr_data = vec![128u8; chroma_w * chroma_h];

    let mut frame = VideoFrame::new(PixelFormat::Yuv420p, WIDTH, HEIGHT);
    frame.planes = vec![
        Plane::with_dimensions(y_data, w, WIDTH, HEIGHT),
        Plane::with_dimensions(cb_data, chroma_w, (WIDTH + 1) / 2, (HEIGHT + 1) / 2),
        Plane::with_dimensions(cr_data, chroma_w, (WIDTH + 1) / 2, (HEIGHT + 1) / 2),
    ];
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

/// Encode a YUV420p frame at the given QP and return the decoded luma plane.
///
/// Pipeline: `frame` (YUV420p) → `ApvEncoder(qp)` → `ApvDecoder` (YUV420p output)
/// → luma `Vec<u8>`.
fn encode_then_decode_luma(frame: &VideoFrame, qp: u8) -> Vec<u8> {
    let config = ApvConfig::new(frame.width, frame.height)
        .expect("valid APV config")
        .with_qp(qp);

    let mut encoder = ApvEncoder::new(config).expect("ApvEncoder::new");

    encoder.send_frame(frame).expect("encoder::send_frame");

    let pkt = encoder
        .receive_packet()
        .expect("encoder::receive_packet")
        .expect("expected encoded packet");

    // ApvDecoder::new() takes zero arguments; default output is Yuv420p.
    let mut decoder = ApvDecoder::new();

    decoder
        .send_packet(&pkt.data, pkt.pts)
        .expect("decoder::send_packet");

    let decoded = decoder
        .receive_frame()
        .expect("decoder::receive_frame")
        .expect("expected decoded frame");

    assert_eq!(
        decoded.format,
        PixelFormat::Yuv420p,
        "APV decoder default output must be YUV420p"
    );
    assert!(
        !decoded.planes.is_empty(),
        "decoded frame must have at least the Y plane"
    );
    assert_eq!(decoded.width, frame.width, "decoded width must match input");
    assert_eq!(
        decoded.height, frame.height,
        "decoded height must match input"
    );

    decoded.planes[0].data.clone()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Encode a 96×64 YUV420p frame with default QP (22), decode, and verify
/// luma PSNR ≥ 20 dB.
///
/// APV at QP=22 produces ~22–24 dB on synthetic gradients (observed).
/// A 20 dB floor is conservative — about 2–4 dB below typical output —
/// ensuring the test catches gross codec failures without being brittle to
/// minor implementation variations.
#[test]
fn test_apv_roundtrip_default() {
    let frame = make_yuv420p_frame_96x64();

    let original_y = frame.planes[0].data.clone();
    let decoded_y = encode_then_decode_luma(&frame, 22);

    let psnr = compute_psnr(&original_y, &decoded_y);

    assert!(
        psnr >= 20.0,
        "APV QP=22 luma PSNR should be ≥ 20 dB, got {psnr:.2} dB"
    );
}
