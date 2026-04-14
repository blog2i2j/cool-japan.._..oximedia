//! Property-based fuzz tests for PNG, GIF, and WebP decoders.
//!
//! These tests use proptest to generate random or semi-random byte sequences
//! and assert that each decoder handles them gracefully (returns `Err` for
//! invalid input rather than panicking, accessing out-of-bounds memory, or
//! hanging indefinitely).
//!
//! The tests do NOT check for correct output when given valid data — that is
//! covered by the codec round-trip tests in `codec_quality.rs`.

use proptest::prelude::*;

// =============================================================================
// PNG decoder fuzz tests
// =============================================================================

// Completely random bytes must not panic the PNG decoder.
proptest! {
    #[test]
    fn png_decoder_no_panic_random_bytes(data in prop::collection::vec(any::<u8>(), 0..512)) {
        use oximedia_codec::png::PngDecoder;
        // Decoder may return Err — that is fine.  It must not panic.
        let _ = PngDecoder::new(&data).and_then(|d| d.decode());
    }
}

// PNG magic header (8 bytes) followed by random body must not panic.
proptest! {
    #[test]
    fn png_decoder_no_panic_magic_plus_random(body in prop::collection::vec(any::<u8>(), 0..512)) {
        use oximedia_codec::png::PngDecoder;
        // PNG signature: 0x89 0x50 0x4E 0x47 0x0D 0x0A 0x1A 0x0A
        let mut data = vec![0x89u8, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        data.extend_from_slice(&body);
        let _ = PngDecoder::new(&data).and_then(|d| d.decode());
    }
}

// A small valid 1x1 PNG followed by garbage appended bytes must not panic.
proptest! {
    #[test]
    fn png_decoder_no_panic_valid_plus_garbage(
        garbage in prop::collection::vec(any::<u8>(), 0..256)
    ) {
        use oximedia_codec::png::PngDecoder;
        // Minimal valid 1x1 8-bit grayscale PNG.
        let mut data: Vec<u8> = vec![
            // PNG signature
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            // IHDR chunk: length=13
            0x00, 0x00, 0x00, 0x0D,
            0x49, 0x48, 0x44, 0x52, // "IHDR"
            0x00, 0x00, 0x00, 0x01, // width=1
            0x00, 0x00, 0x00, 0x01, // height=1
            0x08,                   // bit depth=8
            0x00,                   // colour type=0 (grayscale)
            0x00,                   // compression method=0
            0x00,                   // filter method=0
            0x00,                   // interlace method=0
            0x3A, 0x7E, 0x9B, 0x55, // CRC
            // IDAT chunk: length=11 (deflate compressed: filter=0, pixel=0xFF)
            0x00, 0x00, 0x00, 0x0B,
            0x49, 0x44, 0x41, 0x54, // "IDAT"
            0x08, 0xD7, 0x63, 0xF8, 0xFF, 0x00, 0x02, 0x00, 0x01,
            0xE2, 0x21, 0xBC, 0x33, // CRC
            // IEND chunk
            0x00, 0x00, 0x00, 0x00,
            0x49, 0x45, 0x4E, 0x44, // "IEND"
            0xAE, 0x42, 0x60, 0x82, // CRC
        ];
        data.extend_from_slice(&garbage);
        let _ = PngDecoder::new(&data).and_then(|d| d.decode());
    }
}

// =============================================================================
// GIF decoder fuzz tests
// =============================================================================

// Completely random bytes must not panic the GIF decoder.
proptest! {
    #[test]
    fn gif_decoder_no_panic_random_bytes(data in prop::collection::vec(any::<u8>(), 0..512)) {
        use oximedia_codec::gif::GifDecoder;
        // GifDecoder::new may return Err — that is fine. It must not panic.
        let _ = GifDecoder::new(&data);
    }
}

// GIF89a header (6 bytes) followed by random body must not panic.
proptest! {
    #[test]
    fn gif_decoder_no_panic_header_plus_random(body in prop::collection::vec(any::<u8>(), 0..512)) {
        use oximedia_codec::gif::GifDecoder;
        // GIF89a signature
        let mut data = b"GIF89a".to_vec();
        data.extend_from_slice(&body);
        let _ = GifDecoder::new(&data);
    }
}

// GIF87a header variant followed by random body must not panic.
proptest! {
    #[test]
    fn gif_decoder_no_panic_87a_header_plus_random(body in prop::collection::vec(any::<u8>(), 0..256)) {
        use oximedia_codec::gif::GifDecoder;
        let mut data = b"GIF87a".to_vec();
        data.extend_from_slice(&body);
        let _ = GifDecoder::new(&data);
    }
}

// =============================================================================
// WebP decoder fuzz tests
// =============================================================================

// Completely random bytes must not panic the WebP VP8L decoder.
proptest! {
    #[test]
    fn webp_vp8l_decoder_no_panic_random_bytes(data in prop::collection::vec(any::<u8>(), 0..512)) {
        use oximedia_codec::webp::Vp8lDecoder;
        let mut dec = Vp8lDecoder::new();
        let _ = dec.decode(&data);
    }
}

// RIFF/WEBP header (12 bytes) followed by random body must not panic.
proptest! {
    #[test]
    fn webp_decoder_no_panic_riff_header_plus_random(body in prop::collection::vec(any::<u8>(), 0..512)) {
        use oximedia_codec::webp::Vp8lDecoder;
        // RIFF....WEBP header structure.
        let total_size = body.len() as u32 + 4;
        let mut data = Vec::with_capacity(12 + body.len());
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&total_size.to_le_bytes());
        data.extend_from_slice(b"WEBP");
        data.extend_from_slice(&body);
        let mut dec = Vp8lDecoder::new();
        let _ = dec.decode(&data);
    }
}

// VP8L chunk signature followed by random body must not panic.
proptest! {
    #[test]
    fn webp_vp8l_chunk_plus_random_no_panic(body in prop::collection::vec(any::<u8>(), 0..256)) {
        use oximedia_codec::webp::Vp8lDecoder;
        // VP8L chunk header: "VP8L" + 4-byte length.
        let chunk_len = body.len() as u32;
        let mut data = Vec::with_capacity(8 + body.len());
        data.extend_from_slice(b"VP8L");
        data.extend_from_slice(&chunk_len.to_le_bytes());
        data.extend_from_slice(&body);
        let mut dec = Vp8lDecoder::new();
        let _ = dec.decode(&data);
    }
}
