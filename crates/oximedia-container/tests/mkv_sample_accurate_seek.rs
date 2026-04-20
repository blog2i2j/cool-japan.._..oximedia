//! Integration test for sample-accurate seeking in Matroska/WebM.
//!
//! Builds a minimal valid WebM stream with 5 SimpleBlock frames at
//! PTS = 0, 1, 2, 3, 4 (timecode-scale units), then calls
//! `seek_sample_accurate(2)` and verifies that the next `read_packet`
//! returns exactly PTS == 2.

use bytes::Bytes;
use oximedia_container::demux::MatroskaDemuxer;
use oximedia_container::Demuxer;
use oximedia_core::OxiError;
use oximedia_io::MemorySource;

// ---------------------------------------------------------------------------
// Synthetic MKV builder
// ---------------------------------------------------------------------------

/// Encode a variable-length integer (VINT) as used in EBML sizes.
fn ebml_vint_size(mut n: u64) -> Vec<u8> {
    // Find the minimum width (1–8 bytes).
    let width = if n < 0x7F {
        1usize
    } else if n < 0x3FFF {
        2
    } else if n < 0x1F_FFFF {
        3
    } else if n < 0x0FFF_FFFF {
        4
    } else {
        8
    };
    let marker = 1u64 << (7 * width);
    n |= marker;
    let bytes = n.to_be_bytes();
    bytes[8 - width..].to_vec()
}

/// Encode an unsigned integer into the minimum number of bytes (big-endian).
fn uint_bytes(v: u64) -> Vec<u8> {
    if v == 0 {
        return vec![0];
    }
    let bytes = v.to_be_bytes();
    let leading = bytes.iter().take_while(|&&b| b == 0).count();
    bytes[leading..].to_vec()
}

/// Build an EBML element: header + data.
fn ebml_elem(id: &[u8], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(id);
    out.extend_from_slice(&ebml_vint_size(data.len() as u64));
    out.extend_from_slice(data);
    out
}

/// Build an EBML uint element.
fn ebml_uint(id: &[u8], value: u64) -> Vec<u8> {
    ebml_elem(id, &uint_bytes(value))
}

/// Build an EBML float element (f64, big-endian 8 bytes).
fn ebml_float(id: &[u8], value: f64) -> Vec<u8> {
    ebml_elem(id, &value.to_bits().to_be_bytes())
}

/// Build an EBML string element.
fn ebml_string(id: &[u8], value: &str) -> Vec<u8> {
    ebml_elem(id, value.as_bytes())
}

/// Build a SimpleBlock element with the given timecode and frame payload.
///
/// SimpleBlock layout (after element header):
///   VINT track-number | i16 timecode | u8 flags | payload
///
/// Flags: bit 7 = keyframe.
fn simple_block(track: u64, timecode: i16, payload: &[u8]) -> Vec<u8> {
    // Encode track as a VINT (1-byte for track 1..126)
    let track_vint = ebml_vint_size(track);
    let tc_bytes = timecode.to_be_bytes();
    let flags: u8 = 0x80; // keyframe
    let mut block_data = Vec::new();
    block_data.extend_from_slice(&track_vint);
    block_data.extend_from_slice(&tc_bytes);
    block_data.push(flags);
    block_data.extend_from_slice(payload);
    // SimpleBlock ID = 0xA3
    ebml_elem(&[0xA3], &block_data)
}

/// Build a minimal WebM stream with `frame_count` SimpleBlock frames.
///
/// All frames are in a single cluster at timestamp 0.
/// Frame PTS[i] = i (in timecode units of 1 ms by default).
fn build_webm(frame_count: usize) -> Vec<u8> {
    // -----------------------------------------------------------------------
    // EBML header (DocType = "webm")
    // -----------------------------------------------------------------------
    let ebml_version = ebml_uint(&[0x42, 0x86], 1); // EBMLVersion
    let ebml_read_version = ebml_uint(&[0x42, 0xF7], 1); // EBMLReadVersion
    let ebml_max_id_len = ebml_uint(&[0x42, 0xF2], 4); // EBMLMaxIDLength
    let ebml_max_size_len = ebml_uint(&[0x42, 0xF3], 8); // EBMLMaxSizeLength
    let doc_type = ebml_string(&[0x42, 0x82], "webm"); // DocType
    let doc_type_version = ebml_uint(&[0x42, 0x87], 4); // DocTypeVersion
    let doc_type_read_version = ebml_uint(&[0x42, 0x85], 2); // DocTypeReadVersion

    let mut ebml_body = Vec::new();
    ebml_body.extend_from_slice(&ebml_version);
    ebml_body.extend_from_slice(&ebml_read_version);
    ebml_body.extend_from_slice(&ebml_max_id_len);
    ebml_body.extend_from_slice(&ebml_max_size_len);
    ebml_body.extend_from_slice(&doc_type);
    ebml_body.extend_from_slice(&doc_type_version);
    ebml_body.extend_from_slice(&doc_type_read_version);
    let ebml_header = ebml_elem(&[0x1A, 0x45, 0xDF, 0xA3], &ebml_body);

    // -----------------------------------------------------------------------
    // Segment children
    // -----------------------------------------------------------------------

    // Info element
    let timecode_scale = ebml_uint(&[0x2A, 0xD7, 0xB1], 1_000_000); // 1 ms
    let duration_float = ebml_float(&[0x44, 0x89], frame_count as f64);
    let info_body: Vec<u8> = [timecode_scale, duration_float].concat();
    let info_elem = ebml_elem(&[0x15, 0x49, 0xA9, 0x66], &info_body);

    // TrackEntry
    let track_number = ebml_uint(&[0xD7], 1);
    let track_uid = ebml_uint(&[0x73, 0xC5], 1);
    let track_type = ebml_uint(&[0x83], 1); // video
    let codec_id = ebml_string(&[0x86], "V_VP9");
    // VideoSettings: width=320, height=240
    let pixel_width = ebml_uint(&[0xB0], 320);
    let pixel_height = ebml_uint(&[0xBA], 240);
    let video_settings = ebml_elem(&[0xE0], &[pixel_width, pixel_height].concat());

    let mut track_entry_body = Vec::new();
    track_entry_body.extend_from_slice(&track_number);
    track_entry_body.extend_from_slice(&track_uid);
    track_entry_body.extend_from_slice(&track_type);
    track_entry_body.extend_from_slice(&codec_id);
    track_entry_body.extend_from_slice(&video_settings);
    let track_entry = ebml_elem(&[0xAE], &track_entry_body);
    let tracks_elem = ebml_elem(&[0x16, 0x54, 0xAE, 0x6B], &track_entry);

    // Cluster: timestamp = 0, then SimpleBlocks at timecodes 0..frame_count-1
    let cluster_ts = ebml_uint(&[0xE7], 0);
    let mut cluster_body = cluster_ts;
    for i in 0..frame_count {
        // Use a 4-byte dummy payload so packets are non-empty
        let payload = [(i + 1) as u8; 4];
        // timecode is relative to cluster timestamp (0), so equals the frame index
        let block = simple_block(1, i as i16, &payload);
        cluster_body.extend_from_slice(&block);
    }
    // Cluster ID = 0x1F43B675, with unbounded size
    let mut cluster_elem = Vec::new();
    cluster_elem.extend_from_slice(&[0x1F, 0x43, 0xB6, 0x75]);
    cluster_elem.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    cluster_elem.extend_from_slice(&cluster_body);

    // -----------------------------------------------------------------------
    // Segment (unbounded size)
    // -----------------------------------------------------------------------
    let mut segment_body = Vec::new();
    segment_body.extend_from_slice(&info_elem);
    segment_body.extend_from_slice(&tracks_elem);
    segment_body.extend_from_slice(&cluster_elem);

    let mut segment = Vec::new();
    segment.extend_from_slice(&[0x18, 0x53, 0x80, 0x67]); // Segment ID
    segment.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]); // unknown size
    segment.extend_from_slice(&segment_body);

    // -----------------------------------------------------------------------
    // Assemble
    // -----------------------------------------------------------------------
    let mut out = Vec::new();
    out.extend_from_slice(&ebml_header);
    out.extend_from_slice(&segment);
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_seek_sample_accurate_exact_match() {
    let data = build_webm(5);
    let source = MemorySource::new(Bytes::from(data));
    let mut demuxer = MatroskaDemuxer::new(source);

    demuxer.probe().await.expect("probe should succeed");

    // Seek to PTS = 2
    demuxer
        .seek_sample_accurate(2)
        .await
        .expect("seek_sample_accurate should succeed");

    // The next packet should have PTS == 2 exactly.
    // The skip_until mechanism ensures we skip pts < 2 and stop filtering at pts == 2.
    let packet = demuxer
        .read_packet()
        .await
        .expect("read_packet after seek should succeed");

    let pts = packet.timestamp.pts;
    assert_eq!(
        pts, 2,
        "After seek_sample_accurate(2), first packet PTS should be exactly 2, got {pts}"
    );
}

#[tokio::test]
async fn test_seek_sample_accurate_to_zero() {
    let data = build_webm(5);
    let source = MemorySource::new(Bytes::from(data));
    let mut demuxer = MatroskaDemuxer::new(source);

    demuxer.probe().await.expect("probe should succeed");

    demuxer
        .seek_sample_accurate(0)
        .await
        .expect("seek to 0 should succeed");

    let packet = demuxer
        .read_packet()
        .await
        .expect("read_packet after seek to 0 should succeed");

    assert_eq!(
        packet.timestamp.pts, 0,
        "seek_sample_accurate(0) should return first frame (PTS=0)"
    );
}

#[tokio::test]
async fn test_seek_sample_accurate_sequential_reads() {
    let data = build_webm(5);
    let source = MemorySource::new(Bytes::from(data));
    let mut demuxer = MatroskaDemuxer::new(source);

    demuxer.probe().await.expect("probe should succeed");

    // Seek to PTS = 2 and verify we can read subsequent packets in order
    demuxer
        .seek_sample_accurate(2)
        .await
        .expect("seek should succeed");

    let mut pts_values = Vec::new();
    loop {
        match demuxer.read_packet().await {
            Ok(pkt) => {
                pts_values.push(pkt.timestamp.pts);
            }
            Err(OxiError::Eof) => break,
            Err(e) => panic!("Unexpected error: {e:?}"),
        }
    }

    assert!(
        !pts_values.is_empty(),
        "Should read at least one packet after seek"
    );
    // All returned PTS values should be >= 2
    for &pts in &pts_values {
        assert!(
            pts >= 2,
            "All packets after seek_sample_accurate(2) should have PTS >= 2, got {pts}"
        );
    }
    // PTS values should be monotonically non-decreasing
    for window in pts_values.windows(2) {
        assert!(
            window[1] >= window[0],
            "PTS should be non-decreasing: {} then {}",
            window[0],
            window[1]
        );
    }
}

#[tokio::test]
async fn test_seek_sample_accurate_before_headers_returns_error() {
    let data = build_webm(3);
    let source = MemorySource::new(Bytes::from(data));
    let mut demuxer = MatroskaDemuxer::new(source);

    // Do NOT call probe() — headers are not parsed
    let result = demuxer.seek_sample_accurate(1).await;
    assert!(
        result.is_err(),
        "seek_sample_accurate before probe should return error"
    );
}
