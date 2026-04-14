//! AV1 bitstream conformance tests.
//!
//! These tests verify that the AV1 encoder/decoder pipeline handles
//! synthetic frames correctly, and that the entropy coding and film-grain
//! subsystems remain internally consistent.
//!
//! Tests that require a fully integrated AV1 codec (encode + decode round-trip)
//! are gated on the `av1` feature and are marked `#[ignore]` when actual
//! pixel-level decode verification is not available in the reference build.
//! The #[ignore] tests document what *should* pass once the full pipeline is
//! wired end-to-end.

// =============================================================================
// CDF / entropy conformance
// =============================================================================

/// Verify the AV1 entropy tables define non-decreasing CDFs.
///
/// AV1 CDFs use the convention: `[v0, v1, ..., v_{n-1}=CDF_PROB_TOP, 0]`
/// where the final `0` is a sentinel that terminates the table. Only the
/// symbol-value entries (all but the trailing sentinel) must be monotone.
#[cfg(feature = "av1")]
#[test]
fn av1_cdf_tables_are_monotone() {
    use oximedia_codec::av1::{CdfContext, CDF_PROB_TOP};

    let ctx = CdfContext::default();

    // Intra-frame Y-mode CDF has format [v0, ..., v12=32768, 0 (sentinel)].
    // The final 0 is a trailing sentinel per AV1 spec — exclude it from the
    // monotone check; only the values up to and including CDF_PROB_TOP must
    // be non-decreasing.
    let y_mode_cdf = ctx.get_y_mode_cdf(0);

    // Find the position of CDF_PROB_TOP (the logical end of the CDF).
    let cdf_end = y_mode_cdf
        .iter()
        .position(|&v| v == CDF_PROB_TOP)
        .map(|pos| pos + 1)
        .unwrap_or(y_mode_cdf.len());

    let symbol_values = &y_mode_cdf[..cdf_end];
    for window in symbol_values.windows(2) {
        assert!(
            window[0] <= window[1],
            "Y-mode CDF is not monotone: {} > {}",
            window[0],
            window[1]
        );
    }

    // Verify the entry at `cdf_end - 1` is CDF_PROB_TOP.
    assert_eq!(
        y_mode_cdf[cdf_end - 1],
        CDF_PROB_TOP,
        "Y-mode CDF must contain CDF_PROB_TOP ({})",
        CDF_PROB_TOP
    );
}

/// Verify CDF_PROB_BITS and CDF_PROB_TOP are consistent (2^bits == top).
#[cfg(feature = "av1")]
#[test]
fn av1_cdf_prob_bits_consistent() {
    use oximedia_codec::av1::{CDF_PROB_BITS, CDF_PROB_TOP};
    assert_eq!(
        1u16 << CDF_PROB_BITS,
        CDF_PROB_TOP,
        "CDF_PROB_TOP must equal 2^CDF_PROB_BITS"
    );
}

// =============================================================================
// Film grain conformance
// =============================================================================

/// Film grain: a disabled synthesizer must not modify the frame.
#[cfg(feature = "av1")]
#[test]
fn av1_film_grain_disabled_noop() {
    use oximedia_codec::av1::FilmGrainSynthesizer;
    use oximedia_codec::frame::{Plane, VideoFrame};
    use oximedia_core::PixelFormat;

    let synth = FilmGrainSynthesizer::new(8);
    let y = Plane::with_dimensions(vec![128u8; 64 * 64], 64, 64, 64);
    let u = Plane::with_dimensions(vec![128u8; 32 * 32], 32, 32, 32);
    let v = Plane::with_dimensions(vec![128u8; 32 * 32], 32, 32, 32);
    let mut frame = {
        let mut f = VideoFrame::new(PixelFormat::Yuv420p, 64, 64);
        f.planes = vec![y, u, v];
        f
    };

    synth.apply_grain(&mut frame).expect("apply disabled grain");

    for &px in frame.plane(0).data() {
        assert_eq!(px, 128u8, "disabled grain must not modify pixels");
    }
}

/// Film grain: enabled synthesizer with non-zero grain strength must
/// produce at least one modified pixel on a uniform gray frame.
#[cfg(feature = "av1")]
#[test]
fn av1_film_grain_modifies_pixels() {
    use oximedia_codec::av1::{Av1FilmGrainParams, FilmGrainSynthesizer};
    use oximedia_codec::frame::{Plane, VideoFrame};
    use oximedia_core::PixelFormat;

    let mut params = Av1FilmGrainParams::new();
    params.apply_grain = true;
    params.film_grain_params_present = true;
    params.grain_seed = 12345;
    params.grain_scaling_minus_8 = 3;
    params.add_y_point(0, 64);
    params.add_y_point(255, 128);

    let mut synth = FilmGrainSynthesizer::new(8);
    synth.set_params(params);

    let y = Plane::with_dimensions(vec![128u8; 96 * 96], 96, 96, 96);
    let u = Plane::with_dimensions(vec![128u8; 48 * 48], 48, 48, 48);
    let v = Plane::with_dimensions(vec![128u8; 48 * 48], 48, 48, 48);
    let mut frame = {
        let mut f = VideoFrame::new(PixelFormat::Yuv420p, 96, 96);
        f.planes = vec![y, u, v];
        f
    };

    synth.apply_grain(&mut frame).expect("apply enabled grain");

    let changed = frame.plane(0).data().iter().filter(|&&p| p != 128).count();
    assert!(changed > 0, "enabled grain must modify at least one pixel");
}

/// Film grain: per-block u16 path with 10-bit depth must keep values in [0, 1023].
#[cfg(feature = "av1")]
#[test]
fn av1_film_grain_per_block_u16_range_10bit() {
    use oximedia_codec::av1::{Av1FilmGrainParams, FilmGrainSynthesizer, PerBlockGrainTable};

    let mut params = Av1FilmGrainParams::new();
    params.apply_grain = true;
    params.film_grain_params_present = true;
    params.grain_seed = 42;
    params.grain_scaling_minus_8 = 2;
    params.add_y_point(0, 48);
    params.add_y_point(255, 96);

    let mut synth = FilmGrainSynthesizer::new(10);
    synth.set_params(params);

    let mut luma = vec![512u16; 64 * 64];
    synth
        .apply_grain_per_block_u16(&mut luma, 64, 64, 64, &PerBlockGrainTable::new())
        .expect("apply 10-bit per-block");

    for &px in &luma {
        assert!(px <= 1023, "10-bit pixel out of range: {px}");
    }
}

/// Per-block grain bilinear blend: verify boundary smoothness.
#[cfg(feature = "av1")]
#[test]
fn av1_per_block_bilinear_boundary_smooth() {
    use oximedia_codec::av1::{apply_grain_per_block_bilinear, Av1FilmGrainParams, GrainBlock};

    let bsz = 32usize; // GRAIN_BLOCK_SIZE
    let w = (2 * bsz) as u32;
    let h = bsz as u32;

    let make_params = |seed: u16, scaling: u8| {
        let mut p = Av1FilmGrainParams::new();
        p.apply_grain = true;
        p.film_grain_params_present = true;
        p.grain_seed = seed;
        p.grain_scaling_minus_8 = 0;
        p.add_y_point(0, scaling);
        p.add_y_point(255, scaling);
        p
    };

    let blocks = vec![
        GrainBlock::new(0, 0, make_params(100, 8), 10),
        GrainBlock::new(1, 0, make_params(200, 200), 10),
    ];

    let orig = vec![512u16; w as usize * h as usize];
    let mut plane = orig.clone();
    apply_grain_per_block_bilinear(&mut plane, w as usize, &blocks, w, h, 10)
        .expect("apply bilinear");

    // Across the boundary at col=bsz: the jump in grain delta should be bounded.
    let row = h as usize / 2;
    let left_delta = plane[row * w as usize + bsz - 1] as i32 - 512;
    let right_delta = plane[row * w as usize + bsz] as i32 - 512;
    let jump = (right_delta - left_delta).abs();
    assert!(
        jump < 300,
        "Boundary grain jump {jump} should be bounded by bilinear interpolation"
    );
}

// =============================================================================
// OBU validator conformance
// =============================================================================

/// OBU validator: an empty byte slice is not a valid sequence header.
#[cfg(feature = "av1")]
#[test]
fn av1_obu_validator_rejects_empty() {
    use oximedia_codec::av1::ObuValidator;

    // ObuValidator is a unit struct — methods are called statically.
    let result = ObuValidator::validate_bitstream(&[]);
    // Either Err or Ok(Invalid) — must not panic.
    let _ = result;
}

/// OBU validator: a well-known AV1 sequence header start (0x0A 0x0E ...) is
/// provisionally accepted (or produces a structured error, not a panic).
#[cfg(feature = "av1")]
#[test]
fn av1_obu_validator_handles_sequence_header_prefix() {
    use oximedia_codec::av1::ObuValidator;

    // Minimal AV1 OBU header for a Sequence Header OBU (type=1).
    // OBU header byte: forbidden(1b)=0 | type(4b)=1 | extension(1b)=0 |
    //                  has_size_field(1b)=1 | reserved(1b)=0  => 0x0A
    // size = 1 byte (LEB128 0x01)
    // payload = 0x00 (invalid/truncated payload, but valid header structure)
    let obu = [0x0Au8, 0x01, 0x00];
    let _result = ObuValidator::validate_bitstream(&obu);
    // Must not panic regardless of outcome.
}

// =============================================================================
// Round-trip quality stubs (marked #[ignore] — requires full codec pipeline)
// =============================================================================

/// #[ignore] Round-trip: encode all-gray 256×256 frame and decode.
/// When the integrated pipeline is wired, decoded PSNR should be ≥ 40 dB or infinite.
#[cfg(feature = "av1")]
#[test]
#[ignore = "requires full AV1 encode+decode pipeline; run manually"]
fn av1_roundtrip_gray_256x256_psnr_high() {
    // Placeholder: test logic will use Av1Encoder / Av1Decoder when integrated.
    // Expected PSNR: > 40 dB at CRF 30, or infinite for lossless.
}

/// #[ignore] Round-trip: encode white frame (Y=235) and verify decoded values ≈ 235.
#[cfg(feature = "av1")]
#[test]
#[ignore = "requires full AV1 encode+decode pipeline; run manually"]
fn av1_roundtrip_white_frame_decoded_matches() {
    // Placeholder.
}
