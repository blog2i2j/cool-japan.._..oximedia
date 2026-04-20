//! Integration tests verifying that `JxlStreamingDecoder` handles truncated or
//! malformed ISOBMFF streams without panicking.
//!
//! The tests here are deliberately adversarial: they feed partial or garbled
//! byte sequences and assert only that no panic occurs.  The decoder is
//! allowed to return errors, yield zero frames, or return `None` immediately —
//! all are acceptable outcomes.

use oximedia_codec::jpegxl::JxlStreamingDecoder;
use std::io::Cursor;

// ── Helper ─────────────────────────────────────────────────────────────────────

/// Build a minimal ftyp box with correct 4cc so the decoder enters ISOBMFF mode,
/// but with a `size` field that promises more bytes than actually follow.
fn truncated_ftyp_isobmff() -> Vec<u8> {
    let mut data: Vec<u8> = Vec::new();
    // size = 24 (ftyp header 8 + 16 payload), but we only write 12 bytes.
    data.extend_from_slice(&24u32.to_be_bytes()); // size
    data.extend_from_slice(b"ftyp"); // fourcc
    data.extend_from_slice(b"jxl "); // major brand → triggers ISOBMFF detection
                                     // Deliberately truncated: we stop here (12 bytes total, not 24).
    data
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[test]
fn streaming_truncated_isobmff_does_not_panic() {
    let data = truncated_ftyp_isobmff();

    let decoder_result = JxlStreamingDecoder::new(Cursor::new(data));
    match decoder_result {
        Err(_) => {
            // Constructor caught the truncation — acceptable.
        }
        Ok(mut decoder) => {
            // Constructor succeeded; the iterator should surface the error or
            // simply terminate — but must not panic.
            let first = decoder.next();
            match first {
                Some(Err(_)) => {
                    // Error yielded — iterator should then terminate.
                    assert!(
                        decoder.next().is_none(),
                        "iterator must return None after a terminal error"
                    );
                }
                Some(Ok(_)) | None => {
                    // Either graceful early termination or an unexpected success —
                    // neither causes a panic, so the test passes.
                }
            }
        }
    }
}

#[test]
fn streaming_empty_stream_does_not_panic() {
    // Completely empty input — should be detected as native format (no ftyp).
    let decoder_result = JxlStreamingDecoder::new(Cursor::new(Vec::<u8>::new()));
    match decoder_result {
        Err(_) => {}
        Ok(mut decoder) => {
            // Must not panic; may return None or an error.
            let _ = decoder.next();
        }
    }
}

#[test]
fn streaming_malformed_jxlp_payload_too_short_does_not_panic() {
    // Build a valid ftyp, then a jxlp box with a payload that is only 2 bytes
    // (< 4 bytes required for the index field).
    let mut data: Vec<u8> = Vec::new();

    // ftyp box (24 bytes total)
    let ftyp_payload = b"jxl \x00\x00\x00\x00jxl isom";
    let ftyp_size = 8u32 + ftyp_payload.len() as u32;
    data.extend_from_slice(&ftyp_size.to_be_bytes());
    data.extend_from_slice(b"ftyp");
    data.extend_from_slice(ftyp_payload);

    // jxlp box with a 2-byte payload (truncated index field)
    let jxlp_payload = b"\xAB"; // only 1 byte
    let jxlp_size = 8u32 + jxlp_payload.len() as u32;
    data.extend_from_slice(&jxlp_size.to_be_bytes());
    data.extend_from_slice(b"jxlp");
    data.extend_from_slice(jxlp_payload);

    let decoder_result = JxlStreamingDecoder::new(Cursor::new(data));
    match decoder_result {
        Err(_) => {}
        Ok(decoder) => {
            let results: Vec<_> = decoder.collect();
            // Every item that exists should be an error — no panics.
            for r in &results {
                assert!(r.is_err(), "short jxlp must yield an error, not a frame");
            }
        }
    }
}
