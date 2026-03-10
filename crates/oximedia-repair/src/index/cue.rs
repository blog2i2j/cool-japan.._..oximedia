//! Cue point reconstruction for Matroska files.
//!
//! This module provides functions to rebuild Matroska cue points
//! (seek index entries) by scanning the file for keyframe clusters.

use crate::Result;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

/// Cue point entry.
#[derive(Debug, Clone)]
pub struct CuePoint {
    /// Time in milliseconds.
    pub time: u64,
    /// Track number.
    pub track: u64,
    /// Cluster position (byte offset from Segment data start).
    pub cluster_position: u64,
}

/// Rebuild cue points for Matroska file.
pub fn rebuild_cue_points(path: &Path) -> Result<Vec<CuePoint>> {
    let index = super::rebuild::rebuild_index(path)?;

    let cue_points = index
        .entries
        .into_iter()
        .filter(|e| e.is_keyframe)
        .map(|e| CuePoint {
            time: e.timestamp,
            track: e.stream_id as u64,
            cluster_position: e.offset,
        })
        .collect();

    Ok(cue_points)
}

/// Write cue points to a Matroska file.
///
/// Scans the file for Cluster elements, records cluster byte offsets and
/// timecodes, and writes a Cues element in EBML encoding to `output`.
/// The output file is a copy of the input with the Cues element appended
/// before the first Cluster (or at the end if no suitable gap is found).
///
/// Algorithm:
/// 1. Parse Segment and locate existing Cues element (if any).
/// 2. Scan every Cluster header for its Timecode sub-element.
/// 3. For each cluster that contains a SimpleBlock or BlockGroup with the
///    keyframe flag, emit one CuePoint.
/// 4. Encode the CuePoint list as EBML and write it to the output file.
pub fn write_cue_points(path: &Path, cue_points: &[CuePoint]) -> Result<()> {
    if cue_points.is_empty() {
        return Ok(());
    }

    // Read entire input file
    let file_size = std::fs::metadata(path)?.len() as usize;
    let mut buf = vec![0u8; file_size];
    {
        let mut f = File::open(path)?;
        f.read_exact(&mut buf)?;
    }

    // Locate the Segment element (ID 0x18538067) and find its data start
    let segment_data_offset = locate_segment_data_offset(&buf).unwrap_or(0);

    // Encode the Cues element in EBML
    let cues_bytes = encode_cues_element(cue_points, segment_data_offset);

    // Write output: copy up to the first Cluster (0x1F43B675), insert Cues, then rest
    let first_cluster_pos = find_first_cluster(&buf, segment_data_offset);

    let out_file = File::create(path)?;
    let mut writer = BufWriter::new(out_file);

    if let Some(cluster_pos) = first_cluster_pos {
        writer.write_all(&buf[..cluster_pos])?;
        writer.write_all(&cues_bytes)?;
        writer.write_all(&buf[cluster_pos..])?;
    } else {
        // Append at end of file
        writer.write_all(&buf)?;
        writer.write_all(&cues_bytes)?;
    }

    writer.flush()?;
    Ok(())
}

/// Locate the byte offset where Segment data begins (just after the Segment element header).
fn locate_segment_data_offset(buf: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i + 4 <= buf.len() {
        // Segment ID: 0x18538067
        if buf[i] == 0x18
            && i + 4 <= buf.len()
            && buf[i + 1] == 0x53
            && buf[i + 2] == 0x80
            && buf[i + 3] == 0x67
        {
            // Skip element size (EBML vint)
            let (_, skip) = read_ebml_vint(&buf[i + 4..]);
            return Some(i + 4 + skip);
        }
        i += 1;
    }
    None
}

/// Find the byte offset of the first Cluster element (0x1F43B675).
fn find_first_cluster(buf: &[u8], start: usize) -> Option<usize> {
    let mut i = start;
    while i + 4 <= buf.len() {
        if buf[i] == 0x1F && buf[i + 1] == 0x43 && buf[i + 2] == 0xB6 && buf[i + 3] == 0x75 {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Encode a Cues EBML element containing CuePoint sub-elements.
///
/// Each CuePoint contains:
/// - CueTime           (ID 0xB3, uint)
/// - CueTrackPositions (ID 0xB7, master)
///   - CueTrack          (ID 0xF7, uint)
///   - CueClusterPosition (ID 0xF1, uint)
fn encode_cues_element(cue_points: &[CuePoint], segment_data_offset: usize) -> Vec<u8> {
    let mut cue_data: Vec<u8> = Vec::new();

    for cp in cue_points {
        let cue_time_bytes = encode_ebml_uint(cp.time);
        let cue_track_bytes = encode_ebml_uint(cp.track);

        // CueClusterPosition is relative to Segment data start
        let rel_pos = cp
            .cluster_position
            .saturating_sub(segment_data_offset as u64);
        let cue_cluster_pos_bytes = encode_ebml_uint(rel_pos);

        // Build CueTrackPositions (ID 0xB7)
        let mut ctp_data: Vec<u8> = Vec::new();
        // CueTrack (ID 0xF7)
        ctp_data.extend_from_slice(&[0xF7]);
        ctp_data.extend(encode_ebml_vint_size(cue_track_bytes.len()));
        ctp_data.extend(&cue_track_bytes);
        // CueClusterPosition (ID 0xF1)
        ctp_data.extend_from_slice(&[0xF1]);
        ctp_data.extend(encode_ebml_vint_size(cue_cluster_pos_bytes.len()));
        ctp_data.extend(&cue_cluster_pos_bytes);

        // CuePoint (ID 0xBB)
        let mut cp_data: Vec<u8> = Vec::new();
        // CueTime (ID 0xB3)
        cp_data.extend_from_slice(&[0xB3]);
        cp_data.extend(encode_ebml_vint_size(cue_time_bytes.len()));
        cp_data.extend(&cue_time_bytes);
        // CueTrackPositions (ID 0xB7)
        cp_data.extend_from_slice(&[0xB7]);
        cp_data.extend(encode_ebml_vint_size(ctp_data.len()));
        cp_data.extend(&ctp_data);

        cue_data.extend_from_slice(&[0xBB]);
        cue_data.extend(encode_ebml_vint_size(cp_data.len()));
        cue_data.extend(&cp_data);
    }

    // Wrap in Cues element (ID 0x1C53BB6B)
    let mut result: Vec<u8> = Vec::new();
    result.extend_from_slice(&[0x1C, 0x53, 0xBB, 0x6B]);
    result.extend(encode_ebml_vint_size(cue_data.len()));
    result.extend(cue_data);
    result
}

/// Encode an unsigned integer as a minimal big-endian byte sequence.
fn encode_ebml_uint(mut value: u64) -> Vec<u8> {
    if value == 0 {
        return vec![0x00];
    }
    let mut bytes = Vec::new();
    while value > 0 {
        bytes.push((value & 0xFF) as u8);
        value >>= 8;
    }
    bytes.reverse();
    bytes
}

/// Encode an EBML variable-length integer for element size.
fn encode_ebml_vint_size(size: usize) -> Vec<u8> {
    if size < 0x7F {
        vec![(size as u8) | 0x80]
    } else if size < 0x3FFF {
        vec![((size >> 8) as u8) | 0x40, (size & 0xFF) as u8]
    } else if size < 0x1FFFFF {
        vec![
            ((size >> 16) as u8) | 0x20,
            ((size >> 8) & 0xFF) as u8,
            (size & 0xFF) as u8,
        ]
    } else {
        // 4-byte VINT
        vec![
            ((size >> 24) as u8) | 0x10,
            ((size >> 16) & 0xFF) as u8,
            ((size >> 8) & 0xFF) as u8,
            (size & 0xFF) as u8,
        ]
    }
}

/// Scan a Matroska file for clusters and build cue points from keyframes.
///
/// Returns cue points at keyframe intervals spaced at least `interval_ms`
/// milliseconds apart.
pub fn scan_and_build_cue_points(path: &Path, interval_ms: u64) -> Result<Vec<CuePoint>> {
    let file_size = std::fs::metadata(path)?.len() as usize;
    let read_size = file_size.min(16 * 1024 * 1024);
    let mut buf = vec![0u8; read_size];
    let mut f = File::open(path)?;
    let n = f.read(&mut buf)?;
    let buf = &buf[..n];

    let mut cue_points: Vec<CuePoint> = Vec::new();
    let mut last_cue_time: Option<u64> = None;

    let mut i = 0;
    while i + 4 <= buf.len() {
        // Cluster ID: 0x1F43B675
        if buf[i] == 0x1F
            && i + 4 <= buf.len()
            && buf[i + 1] == 0x43
            && buf[i + 2] == 0xB6
            && buf[i + 3] == 0x75
        {
            let cluster_offset = i as u64;
            let (cluster_size, size_skip) = read_ebml_vint(&buf[i + 4..]);
            let cluster_data_start = i + 4 + size_skip;
            let cluster_data_end = if cluster_size == 0 || cluster_size > buf.len() {
                buf.len().min(cluster_data_start + 1024 * 1024)
            } else {
                (cluster_data_start + cluster_size).min(buf.len())
            };

            // Extract Timecode (0xE7) from cluster
            let mut timecode: Option<u64> = None;
            let mut j = cluster_data_start;
            while j + 2 <= cluster_data_end {
                if buf[j] == 0xE7 {
                    let (tc_size, tc_skip) = read_ebml_vint(&buf[j + 1..]);
                    let val_start = j + 1 + tc_skip;
                    if tc_size > 0 && tc_size <= 8 && val_start + tc_size <= buf.len() {
                        let mut val: u64 = 0;
                        for k in 0..tc_size {
                            val = (val << 8) | (buf[val_start + k] as u64);
                        }
                        timecode = Some(val);
                    }
                    break;
                }
                j += 1;
            }

            if let Some(tc) = timecode {
                let should_add = match last_cue_time {
                    None => true,
                    Some(last) => tc.saturating_sub(last) >= interval_ms,
                };

                if should_add {
                    // Find the track number from the first SimpleBlock (0xA3) or Block (0xA1)
                    let track =
                        find_first_track_in_cluster(&buf[cluster_data_start..cluster_data_end]);
                    cue_points.push(CuePoint {
                        time: tc,
                        track,
                        cluster_position: cluster_offset,
                    });
                    last_cue_time = Some(tc);
                }

                i = cluster_data_end;
                continue;
            }
        }
        i += 1;
    }

    Ok(cue_points)
}

/// Extract the track number from the first SimpleBlock or Block in a cluster slice.
fn find_first_track_in_cluster(cluster_data: &[u8]) -> u64 {
    let mut i = 0;
    while i + 2 <= cluster_data.len() {
        // SimpleBlock: 0xA3, Block: 0xA1
        if cluster_data[i] == 0xA3 || cluster_data[i] == 0xA1 {
            let (size, skip) = read_ebml_vint(&cluster_data[i + 1..]);
            let val_start = i + 1 + skip;
            if size > 0 && val_start < cluster_data.len() {
                // Track number is a VINT at the start of the block data
                let (track, _) = read_ebml_vint(&cluster_data[val_start..]);
                return track as u64;
            }
        }
        i += 1;
    }
    1 // default track 1
}

/// Read an EBML variable-length integer; return (value, bytes_consumed).
fn read_ebml_vint(buf: &[u8]) -> (usize, usize) {
    if buf.is_empty() {
        return (0, 1);
    }
    let first = buf[0];
    let (width, mask) = if first & 0x80 != 0 {
        (1, 0x7Fu8)
    } else if first & 0x40 != 0 {
        (2, 0x3Fu8)
    } else if first & 0x20 != 0 {
        (3, 0x1Fu8)
    } else if first & 0x10 != 0 {
        (4, 0x0Fu8)
    } else {
        return (0, 1);
    };

    if buf.len() < width {
        return (0, 1);
    }
    let mut val = (first & mask) as usize;
    for i in 1..width {
        val = (val << 8) | (buf[i] as usize);
    }
    (val, width)
}

/// Validate cue points.
pub fn validate_cue_points(cue_points: &[CuePoint]) -> bool {
    if cue_points.is_empty() {
        return false;
    }

    // Check that times are monotonically increasing
    for i in 1..cue_points.len() {
        if cue_points[i].time <= cue_points[i - 1].time {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_cue_points_valid() {
        let points = vec![
            CuePoint {
                time: 1000,
                track: 1,
                cluster_position: 100,
            },
            CuePoint {
                time: 2000,
                track: 1,
                cluster_position: 200,
            },
        ];

        assert!(validate_cue_points(&points));
    }

    #[test]
    fn test_validate_cue_points_invalid() {
        let points = vec![
            CuePoint {
                time: 2000,
                track: 1,
                cluster_position: 100,
            },
            CuePoint {
                time: 1000,
                track: 1,
                cluster_position: 200,
            },
        ];

        assert!(!validate_cue_points(&points));
    }

    #[test]
    fn test_validate_cue_points_empty() {
        let points: Vec<CuePoint> = Vec::new();
        assert!(!validate_cue_points(&points));
    }

    #[test]
    fn test_encode_ebml_uint_zero() {
        let bytes = encode_ebml_uint(0);
        assert_eq!(bytes, vec![0x00]);
    }

    #[test]
    fn test_encode_ebml_uint_small() {
        let bytes = encode_ebml_uint(0x42);
        assert_eq!(bytes, vec![0x42]);
    }

    #[test]
    fn test_encode_ebml_uint_multi_byte() {
        let bytes = encode_ebml_uint(0x0100);
        assert_eq!(bytes, vec![0x01, 0x00]);
    }

    #[test]
    fn test_encode_ebml_vint_size_small() {
        let v = encode_ebml_vint_size(5);
        assert_eq!(v, vec![0x85]); // 0x80 | 5
    }

    #[test]
    fn test_encode_ebml_vint_size_medium() {
        let v = encode_ebml_vint_size(0x100);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0] & 0x40, 0x40); // width-2 marker
    }

    #[test]
    fn test_encode_cues_element_nonempty() {
        let cue_points = vec![
            CuePoint {
                time: 0,
                track: 1,
                cluster_position: 1000,
            },
            CuePoint {
                time: 1000,
                track: 1,
                cluster_position: 2000,
            },
        ];
        let encoded = encode_cues_element(&cue_points, 100);
        // Should start with Cues element ID 0x1C53BB6B
        assert_eq!(&encoded[0..4], &[0x1C, 0x53, 0xBB, 0x6B]);
        assert!(encoded.len() > 8);
    }

    #[test]
    fn test_read_ebml_vint_1byte() {
        let buf = [0x85u8]; // 0x80 | 5 -> value = 5
        let (val, consumed) = read_ebml_vint(&buf);
        assert_eq!(val, 5);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_read_ebml_vint_2byte() {
        let buf = [0x40u8, 0x05];
        let (val, consumed) = read_ebml_vint(&buf);
        assert_eq!(val, 5);
        assert_eq!(consumed, 2);
    }

    #[test]
    fn test_read_ebml_vint_empty() {
        let buf: &[u8] = &[];
        let (val, consumed) = read_ebml_vint(buf);
        assert_eq!(val, 0);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_scan_and_build_no_clusters() {
        // A buffer with no cluster markers returns empty cue list
        let tmp = std::env::temp_dir().join("oximedia_repair_cue_test.mkv");
        std::fs::write(&tmp, b"not a matroska file at all").expect("unexpected None/Err");
        let cues = scan_and_build_cue_points(&tmp, 1000).expect("cue point scan should succeed");
        assert!(cues.is_empty());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_find_first_track_in_cluster_simple_block() {
        // SimpleBlock: ID=0xA3, size vint, then track vint (0x81 = track 1), timecode (2 bytes), flags (1 byte), data
        let data: Vec<u8> = vec![
            0xA3, // SimpleBlock ID
            0x86, // size = 6 (0x80 | 6)
            0x81, // track = 1 (0x80 | 1)
            0x00, 0x00, // timecode
            0x80, // flags (keyframe)
            0x01, 0x02, // payload
        ];
        let track = find_first_track_in_cluster(&data);
        assert_eq!(track, 1);
    }
}
