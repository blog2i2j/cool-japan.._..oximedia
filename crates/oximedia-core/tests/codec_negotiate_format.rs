use oximedia_core::codec_negotiation::{FormatConversionResult, FormatNegotiator};
use oximedia_core::types::{PixelFormat, SampleFormat};

#[test]
fn test_negotiate_direct_match() {
    let negotiator = FormatNegotiator::<PixelFormat> {
        decoder_produces: &[PixelFormat::Yuv420p],
        encoder_accepts: &[PixelFormat::Yuv420p],
    };
    assert_eq!(
        negotiator.negotiate(),
        FormatConversionResult::Direct(PixelFormat::Yuv420p)
    );
}

#[test]
fn test_negotiate_convert_within_yuv() {
    let negotiator = FormatNegotiator::<PixelFormat> {
        decoder_produces: &[PixelFormat::Yuv422p],
        encoder_accepts: &[PixelFormat::Yuv420p],
    };
    match negotiator.negotiate() {
        FormatConversionResult::Convert { from, to, cost } => {
            assert_eq!(from, PixelFormat::Yuv422p);
            assert_eq!(to, PixelFormat::Yuv420p);
            assert!(cost >= 1 && cost <= 4);
        }
        other => panic!("expected Convert, got {:?}", other),
    }
}

#[test]
fn test_negotiate_incompatible_when_no_path() {
    // Empty encoder_accepts → no path
    let negotiator = FormatNegotiator::<PixelFormat> {
        decoder_produces: &[PixelFormat::Yuv420p],
        encoder_accepts: &[],
    };
    assert_eq!(negotiator.negotiate(), FormatConversionResult::Incompatible);
}

#[test]
fn test_negotiate_picks_lowest_cost() {
    // Yuv420p decoder; encoder accepts [Yuv422p, Nv12]
    // Nv12 is YUV420 family (cost 1 from Yuv420p)
    // Yuv422p is different subsampling (cost 2)
    // Should pick Nv12 (lowest cost = 1)
    let negotiator = FormatNegotiator::<PixelFormat> {
        decoder_produces: &[PixelFormat::Yuv420p],
        encoder_accepts: &[PixelFormat::Yuv422p, PixelFormat::Nv12],
    };
    match negotiator.negotiate() {
        FormatConversionResult::Convert { to, cost, .. } => {
            assert_eq!(to, PixelFormat::Nv12);
            assert_eq!(cost, 1);
        }
        other => panic!("expected Convert to Nv12, got {:?}", other),
    }
}

#[test]
fn test_negotiate_rgb_to_yuv_cost_3() {
    // RGB → YUV cross-family = cost 3
    let negotiator = FormatNegotiator::<PixelFormat> {
        decoder_produces: &[PixelFormat::Rgb24],
        encoder_accepts: &[PixelFormat::Yuv420p],
    };
    match negotiator.negotiate() {
        FormatConversionResult::Convert { cost, .. } => {
            assert_eq!(cost, 3);
        }
        other => panic!("expected Convert, got {:?}", other),
    }
}

#[test]
fn test_negotiate_sample_direct_match() {
    let neg = FormatNegotiator::<SampleFormat> {
        decoder_produces: &[SampleFormat::F32],
        encoder_accepts: &[SampleFormat::F32],
    };
    assert_eq!(
        neg.negotiate(),
        FormatConversionResult::Direct(SampleFormat::F32)
    );
}

#[test]
fn test_negotiate_sample_interleaved_to_planar_cost_1() {
    // S16 → S16p: same encoding, interleaved↔planar = cost 1
    let neg = FormatNegotiator::<SampleFormat> {
        decoder_produces: &[SampleFormat::S16],
        encoder_accepts: &[SampleFormat::S16p],
    };
    match neg.negotiate() {
        FormatConversionResult::Convert { from, to, cost } => {
            assert_eq!(from, SampleFormat::S16);
            assert_eq!(to, SampleFormat::S16p);
            assert_eq!(cost, 1);
        }
        other => panic!("expected Convert, got {:?}", other),
    }
}

#[test]
fn test_negotiate_sample_int_to_float_cost_3() {
    let neg = FormatNegotiator::<SampleFormat> {
        decoder_produces: &[SampleFormat::S16],
        encoder_accepts: &[SampleFormat::F32],
    };
    match neg.negotiate() {
        FormatConversionResult::Convert { cost, .. } => {
            assert_eq!(cost, 3);
        }
        other => panic!("expected Convert, got {:?}", other),
    }
}

#[test]
fn test_negotiate_gray_same_family_cost_1() {
    // Gray8 → Gray16: same family = cost 1
    let neg = FormatNegotiator::<PixelFormat> {
        decoder_produces: &[PixelFormat::Gray8],
        encoder_accepts: &[PixelFormat::Gray16],
    };
    match neg.negotiate() {
        FormatConversionResult::Convert { cost, .. } => {
            assert_eq!(cost, 1);
        }
        other => panic!("expected Convert, got {:?}", other),
    }
}

#[test]
fn test_negotiate_prefers_direct_over_convert() {
    // decoder produces [Yuv422p, Yuv420p]; encoder accepts [Yuv420p, Nv12]
    // Direct match on Yuv420p should win
    let neg = FormatNegotiator::<PixelFormat> {
        decoder_produces: &[PixelFormat::Yuv422p, PixelFormat::Yuv420p],
        encoder_accepts: &[PixelFormat::Yuv420p, PixelFormat::Nv12],
    };
    // Direct match found (Yuv420p in both)
    assert_eq!(
        neg.negotiate(),
        FormatConversionResult::Direct(PixelFormat::Yuv420p)
    );
}
