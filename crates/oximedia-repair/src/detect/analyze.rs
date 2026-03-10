//! File analysis and diagnostics.
//!
//! This module provides detailed analysis of media files to identify
//! structural issues, missing components, and potential problems.

use crate::{Issue, IssueType, Result, Severity};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Statistics about a timestamp sequence.
#[derive(Debug, Clone)]
pub struct TimestampStats {
    /// Number of timestamps analyzed.
    pub count: usize,
    /// Minimum timestamp value.
    pub min: i64,
    /// Maximum timestamp value.
    pub max: i64,
    /// Mean inter-frame delta (ms).
    pub mean_delta: f64,
    /// Variance of inter-frame deltas.
    pub variance: f64,
    /// Number of negative timestamps.
    pub negative_count: usize,
    /// Number of duplicate timestamps.
    pub duplicate_count: usize,
    /// Number of non-monotonic timestamp pairs.
    pub non_monotonic_count: usize,
    /// Maximum gap between consecutive timestamps (ms).
    pub max_gap: i64,
    /// Number of gaps larger than 3× mean delta.
    pub large_gap_count: usize,
}

/// Statistics about an index table.
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Number of index entries found.
    pub entry_count: usize,
    /// Number of entries with invalid byte offsets.
    pub invalid_offset_count: usize,
    /// Number of missing expected keyframe entries.
    pub missing_keyframe_count: usize,
    /// Whether the index covers the full file.
    pub covers_full_file: bool,
    /// Number of entries where offset exceeds file size.
    pub out_of_bounds_count: usize,
    /// Number of duplicate offset entries.
    pub duplicate_offset_count: usize,
}

/// Result of metadata validation.
#[derive(Debug, Clone)]
pub struct MetadataValidation {
    /// Whether the metadata was successfully parsed.
    pub parseable: bool,
    /// Number of invalid UTF-8 string fields found.
    pub invalid_utf8_count: usize,
    /// Number of truncated string fields found.
    pub truncated_string_count: usize,
    /// Whether duration field is present and nonzero.
    pub has_valid_duration: bool,
    /// Whether any timestamps in metadata are negative.
    pub has_negative_timestamps: bool,
    /// Whether codec information is present.
    pub has_codec_info: bool,
}

/// Analyze a media file for structural issues.
///
/// This function performs a comprehensive analysis of the file structure
/// to identify issues such as:
/// - Missing or corrupted indices
/// - Invalid metadata
/// - Timestamp inconsistencies
/// - Missing keyframes
pub fn analyze_file(path: &Path) -> Result<Vec<Issue>> {
    let mut issues = Vec::new();

    // Check if file exists and is readable
    if !path.exists() {
        return Ok(issues);
    }

    let metadata = std::fs::metadata(path)?;
    let file_size = metadata.len();

    // Check for suspiciously small files
    if file_size < 1024 {
        issues.push(Issue {
            issue_type: IssueType::Truncated,
            severity: Severity::High,
            description: format!("File is very small ({} bytes), likely truncated", file_size),
            location: Some(0),
            fixable: false,
        });
    }

    // Analyze container structure
    issues.extend(analyze_container_structure(path)?);

    // Analyze timestamps
    issues.extend(analyze_timestamps(path)?);

    // Analyze indices
    issues.extend(analyze_indices(path)?);

    // Analyze metadata
    issues.extend(analyze_metadata(path)?);

    Ok(issues)
}

/// Analyze container structure for issues.
fn analyze_container_structure(path: &Path) -> Result<Vec<Issue>> {
    let mut issues = Vec::new();

    let file = std::fs::File::open(path)?;
    let metadata = file.metadata()?;
    let size = metadata.len();

    // Check if file size is a power of 2 (possible truncation)
    if size > 0 && size.is_power_of_two() && size < 1024 * 1024 {
        issues.push(Issue {
            issue_type: IssueType::Truncated,
            severity: Severity::Medium,
            description: "File size is suspiciously round (power of 2), may be truncated"
                .to_string(),
            location: None,
            fixable: true,
        });
    }

    Ok(issues)
}

/// Analyze timestamps in a media file for gaps, duplicates, and non-monotonic ordering.
///
/// Scans the file for timestamp-bearing structures (PTS/DTS in MPEG-TS packets,
/// timecode atoms in MP4, SimpleBlock timestamps in Matroska) and computes
/// statistics about the distribution and validity of the timestamp sequence.
pub fn analyze_timestamps(path: &Path) -> Result<Vec<Issue>> {
    let mut issues = Vec::new();
    let mut file = File::open(path)?;
    let file_size = file.metadata()?.len();

    if file_size == 0 {
        return Ok(issues);
    }

    // Read a representative sample for timestamp scanning
    let read_size = file_size.min(4 * 1024 * 1024) as usize; // up to 4 MiB
    let mut buf = vec![0u8; read_size];
    file.seek(SeekFrom::Start(0))?;
    let n = file.read(&mut buf)?;
    let buf = &buf[..n];

    // Collect raw timestamps from multiple container formats
    let mut timestamps: Vec<i64> = Vec::new();

    // --- MPEG-TS packet scanning ---
    // MPEG-TS packets are exactly 188 bytes starting with 0x47
    if buf.len() >= 188 {
        timestamps.extend(extract_mpegts_timestamps(buf));
    }

    // --- Matroska SimpleBlock timestamp scanning ---
    // Look for EBML cluster timecodes (element ID 0xE7 = Timecode)
    timestamps.extend(extract_matroska_cluster_timecodes(buf));

    // --- MP4/MOV DTS scanning from stts atom ---
    // Look for 'stts' box signature to check sample delta values
    timestamps.extend(extract_mp4_sample_timestamps(buf));

    // If we found no format-specific timestamps, produce no issues
    if timestamps.is_empty() {
        return Ok(issues);
    }

    let stats = compute_timestamp_stats(&timestamps);

    // Report negative timestamps
    if stats.negative_count > 0 {
        issues.push(Issue {
            issue_type: IssueType::InvalidTimestamps,
            severity: Severity::Medium,
            description: format!(
                "Found {} negative timestamp(s); first is {} ms",
                stats.negative_count,
                timestamps.iter().find(|&&t| t < 0).copied().unwrap_or(0)
            ),
            location: None,
            fixable: true,
        });
    }

    // Report non-monotonic timestamps
    if stats.non_monotonic_count > 0 {
        let severity = if stats.non_monotonic_count > timestamps.len() / 10 {
            Severity::High
        } else {
            Severity::Medium
        };
        issues.push(Issue {
            issue_type: IssueType::InvalidTimestamps,
            severity,
            description: format!(
                "Timestamp sequence has {} non-monotonic step(s) out of {} samples",
                stats.non_monotonic_count, stats.count
            ),
            location: None,
            fixable: true,
        });
    }

    // Report duplicate timestamps
    if stats.duplicate_count > 0 {
        issues.push(Issue {
            issue_type: IssueType::InvalidTimestamps,
            severity: Severity::Low,
            description: format!(
                "Found {} duplicate timestamp value(s) (same PTS for different frames)",
                stats.duplicate_count
            ),
            location: None,
            fixable: true,
        });
    }

    // Report large gaps (> 3× mean delta)
    if stats.large_gap_count > 0 && stats.mean_delta > 0.0 {
        issues.push(Issue {
            issue_type: IssueType::InvalidTimestamps,
            severity: Severity::Medium,
            description: format!(
                "Found {} timestamp gap(s) larger than 3× mean frame delta ({:.1} ms); max gap is {} ms",
                stats.large_gap_count, stats.mean_delta, stats.max_gap
            ),
            location: None,
            fixable: true,
        });
    }

    // Report high timestamp variance (jitter)
    if stats.variance > 1.0 && stats.mean_delta > 0.0 {
        let cv = stats.variance.sqrt() / stats.mean_delta; // coefficient of variation
        if cv > 0.5 {
            issues.push(Issue {
                issue_type: IssueType::InvalidTimestamps,
                severity: Severity::Low,
                description: format!(
                    "High timestamp jitter detected (CV={:.2}, σ={:.1} ms, mean={:.1} ms)",
                    cv,
                    stats.variance.sqrt(),
                    stats.mean_delta
                ),
                location: None,
                fixable: false,
            });
        }
    }

    Ok(issues)
}

/// Extract PTS values from MPEG-TS packets in a buffer.
fn extract_mpegts_timestamps(buf: &[u8]) -> Vec<i64> {
    let mut timestamps = Vec::new();
    let mut i = 0;

    while i + 188 <= buf.len() {
        if buf[i] != 0x47 {
            i += 1;
            continue;
        }
        // Validate next sync byte
        if i + 188 < buf.len() && buf[i + 188] != 0x47 {
            i += 1;
            continue;
        }

        let pkt = &buf[i..i + 188];
        let has_payload = (pkt[3] & 0x10) != 0;
        let has_adaptation = (pkt[3] & 0x20) != 0;

        let payload_start = if has_adaptation && pkt.len() > 4 {
            let adapt_len = pkt[4] as usize;
            5 + adapt_len
        } else {
            4
        };

        if has_payload && payload_start + 14 <= pkt.len() {
            let pes = &pkt[payload_start..];
            // PES start code 0x000001
            if pes.len() >= 14 && pes[0] == 0x00 && pes[1] == 0x00 && pes[2] == 0x01 {
                let pts_dts_flags = (pes[7] >> 6) & 0x03;
                if pts_dts_flags >= 2 && pes.len() >= 14 {
                    // PTS present
                    let pts = parse_pes_timestamp(&pes[9..14]);
                    // Convert 90kHz clock to ms
                    timestamps.push(pts / 90);
                }
            }
        }

        i += 188;
    }

    timestamps
}

/// Parse a 5-byte PES timestamp field into a 33-bit value.
fn parse_pes_timestamp(bytes: &[u8]) -> i64 {
    if bytes.len() < 5 {
        return 0;
    }
    let b0 = bytes[0] as i64;
    let b1 = bytes[1] as i64;
    let b2 = bytes[2] as i64;
    let b3 = bytes[3] as i64;
    let b4 = bytes[4] as i64;

    ((b0 & 0x0E) << 29) | (b1 << 22) | ((b2 & 0xFE) << 14) | (b3 << 7) | ((b4 & 0xFE) >> 1)
}

/// Extract cluster timecodes from a Matroska buffer.
fn extract_matroska_cluster_timecodes(buf: &[u8]) -> Vec<i64> {
    let mut timestamps = Vec::new();
    // Element ID for Timecode (within Cluster) is 0xE7
    // Element ID for Cluster is 0x1F43B675
    let mut i = 0;
    while i + 6 <= buf.len() {
        // Look for Cluster element ID
        if buf[i] == 0x1F && buf[i + 1] == 0x43 && buf[i + 2] == 0xB6 && buf[i + 3] == 0x75 {
            // Skip cluster size (EBML variable-length integer)
            let (_, skip) = read_ebml_vint(&buf[i + 4..]);
            let cluster_data_start = i + 4 + skip;

            // Within the cluster, look for Timecode (0xE7)
            if cluster_data_start + 3 <= buf.len() {
                let j = cluster_data_start;
                if j < buf.len() && buf[j] == 0xE7 {
                    let (tc_size, tc_skip) = read_ebml_vint(&buf[j + 1..]);
                    let val_start = j + 1 + tc_skip;
                    if tc_size <= 8 && val_start + tc_size <= buf.len() {
                        let mut val: i64 = 0;
                        for k in 0..tc_size {
                            val = (val << 8) | (buf[val_start + k] as i64);
                        }
                        timestamps.push(val);
                    }
                }
            }
            i += 4;
        } else {
            i += 1;
        }
    }
    timestamps
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

/// Extract timestamps from an MP4 stts atom.
fn extract_mp4_sample_timestamps(buf: &[u8]) -> Vec<i64> {
    let mut timestamps = Vec::new();
    // Search for 'stts' box
    let mut i = 0;
    while i + 16 <= buf.len() {
        if &buf[i + 4..i + 8] == b"stts" {
            // stts: version(1), flags(3), entry_count(4), [count(4), delta(4)]...
            let entry_count_start = i + 12;
            if entry_count_start + 4 > buf.len() {
                break;
            }
            let entry_count = u32::from_be_bytes([
                buf[entry_count_start],
                buf[entry_count_start + 1],
                buf[entry_count_start + 2],
                buf[entry_count_start + 3],
            ]) as usize;

            let max_entries = entry_count.min(1000); // limit scan
            let mut current_ts: i64 = 0;
            for e in 0..max_entries {
                let offset = entry_count_start + 4 + e * 8;
                if offset + 8 > buf.len() {
                    break;
                }
                let sample_count = u32::from_be_bytes([
                    buf[offset],
                    buf[offset + 1],
                    buf[offset + 2],
                    buf[offset + 3],
                ]) as usize;
                let sample_delta = u32::from_be_bytes([
                    buf[offset + 4],
                    buf[offset + 5],
                    buf[offset + 6],
                    buf[offset + 7],
                ]) as i64;

                let take = sample_count.min(100);
                for _ in 0..take {
                    timestamps.push(current_ts);
                    current_ts += sample_delta;
                }
            }
            break;
        }
        i += 1;
    }
    timestamps
}

/// Compute statistics over a sorted timestamp sequence.
fn compute_timestamp_stats(timestamps: &[i64]) -> TimestampStats {
    if timestamps.is_empty() {
        return TimestampStats {
            count: 0,
            min: 0,
            max: 0,
            mean_delta: 0.0,
            variance: 0.0,
            negative_count: 0,
            duplicate_count: 0,
            non_monotonic_count: 0,
            max_gap: 0,
            large_gap_count: 0,
        };
    }

    let count = timestamps.len();
    let min = timestamps.iter().copied().min().unwrap_or(0);
    let max = timestamps.iter().copied().max().unwrap_or(0);
    let negative_count = timestamps.iter().filter(|&&t| t < 0).count();

    let mut deltas: Vec<i64> = Vec::with_capacity(count.saturating_sub(1));
    let mut duplicate_count = 0usize;
    let mut non_monotonic_count = 0usize;
    let mut max_gap: i64 = 0;

    for i in 1..count {
        let delta = timestamps[i] - timestamps[i - 1];
        if delta == 0 {
            duplicate_count += 1;
        } else if delta < 0 {
            non_monotonic_count += 1;
        }
        let abs_delta = delta.abs();
        if abs_delta > max_gap {
            max_gap = abs_delta;
        }
        deltas.push(delta);
    }

    // Compute mean and variance of positive deltas
    let positive_deltas: Vec<f64> = deltas
        .iter()
        .filter(|&&d| d > 0)
        .map(|&d| d as f64)
        .collect();
    let mean_delta = if positive_deltas.is_empty() {
        0.0
    } else {
        positive_deltas.iter().sum::<f64>() / positive_deltas.len() as f64
    };

    let variance = if positive_deltas.len() > 1 {
        let sq_diff: f64 = positive_deltas
            .iter()
            .map(|&d| (d - mean_delta).powi(2))
            .sum();
        sq_diff / (positive_deltas.len() - 1) as f64
    } else {
        0.0
    };

    let large_gap_count = if mean_delta > 0.0 {
        deltas
            .iter()
            .filter(|&&d| d as f64 > mean_delta * 3.0)
            .count()
    } else {
        0
    };

    TimestampStats {
        count,
        min,
        max,
        mean_delta,
        variance,
        negative_count,
        duplicate_count,
        non_monotonic_count,
        max_gap,
        large_gap_count,
    }
}

/// Analyze index entries in a media file for validity and completeness.
///
/// Reads the file and attempts to locate index structures for any supported
/// container format (AVI idx1, MP4 stco/co64, Matroska Cues).  For each
/// index entry it checks:
/// - The byte offset is within the file bounds.
/// - The byte offset is not a duplicate of another entry.
/// - There are no large unexplained gaps in coverage.
/// - Keyframe index entries are present at reasonable intervals.
pub fn analyze_indices(path: &Path) -> Result<Vec<Issue>> {
    let mut issues = Vec::new();
    let mut file = File::open(path)?;
    let file_size = file.metadata()?.len();

    if file_size < 16 {
        return Ok(issues);
    }

    // Read header to detect format
    let mut header = [0u8; 16];
    file.read_exact(&mut header)?;

    // Dispatch to format-specific index analyzers
    let stats = if &header[0..4] == b"RIFF" && header.len() >= 12 && &header[8..12] == b"AVI " {
        analyze_avi_index(&mut file, file_size)?
    } else if header.len() >= 8 && &header[4..8] == b"ftyp" {
        analyze_mp4_index(&mut file, file_size)?
    } else if header.len() >= 4 && header[0..4] == [0x1A, 0x45, 0xDF, 0xA3] {
        analyze_matroska_index(&mut file, file_size)?
    } else {
        // Unknown format – nothing to say about the index
        return Ok(issues);
    };

    if stats.entry_count == 0 {
        issues.push(Issue {
            issue_type: IssueType::MissingIndex,
            severity: Severity::High,
            description: "No seek index found in file; seeking will be slow or impossible"
                .to_string(),
            location: None,
            fixable: true,
        });
        return Ok(issues);
    }

    if stats.out_of_bounds_count > 0 {
        issues.push(Issue {
            issue_type: IssueType::MissingIndex,
            severity: Severity::High,
            description: format!(
                "Index contains {} entr{} with byte offset beyond end of file (file size = {})",
                stats.out_of_bounds_count,
                if stats.out_of_bounds_count == 1 {
                    "y"
                } else {
                    "ies"
                },
                file_size
            ),
            location: None,
            fixable: true,
        });
    }

    if stats.invalid_offset_count > 0 {
        issues.push(Issue {
            issue_type: IssueType::MissingIndex,
            severity: Severity::Medium,
            description: format!(
                "{} index entr{} with zero or invalid byte offset",
                stats.invalid_offset_count,
                if stats.invalid_offset_count == 1 {
                    "y"
                } else {
                    "ies"
                }
            ),
            location: None,
            fixable: true,
        });
    }

    if stats.duplicate_offset_count > 0 {
        issues.push(Issue {
            issue_type: IssueType::MissingIndex,
            severity: Severity::Low,
            description: format!(
                "{} duplicate byte offset{} in index (possible index corruption)",
                stats.duplicate_offset_count,
                if stats.duplicate_offset_count == 1 {
                    ""
                } else {
                    "s"
                }
            ),
            location: None,
            fixable: true,
        });
    }

    if !stats.covers_full_file {
        issues.push(Issue {
            issue_type: IssueType::MissingIndex,
            severity: Severity::Medium,
            description: "Index does not appear to cover the full file; tail may not be seekable"
                .to_string(),
            location: None,
            fixable: true,
        });
    }

    if stats.missing_keyframe_count > 0 {
        issues.push(Issue {
            issue_type: IssueType::MissingKeyframes,
            severity: Severity::Medium,
            description: format!(
                "Index is missing approximately {} keyframe entr{} at regular intervals",
                stats.missing_keyframe_count,
                if stats.missing_keyframe_count == 1 {
                    "y"
                } else {
                    "ies"
                }
            ),
            location: None,
            fixable: true,
        });
    }

    Ok(issues)
}

/// Analyze the AVI idx1 index.
fn analyze_avi_index(file: &mut File, file_size: u64) -> Result<IndexStats> {
    // Scan for idx1 chunk
    let mut offsets: Vec<u64> = Vec::new();
    let mut pos = 12u64; // skip RIFF header

    file.seek(SeekFrom::Start(12))?;

    let mut buf8 = [0u8; 8];
    loop {
        if pos + 8 > file_size {
            break;
        }
        file.seek(SeekFrom::Start(pos))?;
        if file.read_exact(&mut buf8).is_err() {
            break;
        }
        let chunk_id = &buf8[0..4];
        let chunk_size = u32::from_le_bytes([buf8[4], buf8[5], buf8[6], buf8[7]]) as u64;

        if chunk_id == b"idx1" {
            // Found idx1; read up to 64k entries
            let entry_count = (chunk_size / 16).min(65536) as usize;
            let mut entry_buf = vec![0u8; entry_count * 16];
            if file.read_exact(&mut entry_buf).is_ok() {
                for i in 0..entry_count {
                    let base = i * 16;
                    let off = u32::from_le_bytes([
                        entry_buf[base + 8],
                        entry_buf[base + 9],
                        entry_buf[base + 10],
                        entry_buf[base + 11],
                    ]) as u64;
                    offsets.push(off);
                }
            }
            break;
        }

        // Skip to next chunk (word-aligned)
        let aligned = (chunk_size + 1) & !1;
        pos += 8 + aligned;
        if pos >= file_size {
            break;
        }
    }

    compute_index_stats(&offsets, file_size)
}

/// Analyze the MP4 stco / co64 chunk offset table.
fn analyze_mp4_index(file: &mut File, file_size: u64) -> Result<IndexStats> {
    let read_size = file_size.min(2 * 1024 * 1024) as usize;
    let mut buf = vec![0u8; read_size];
    file.seek(SeekFrom::Start(0))?;
    let n = file.read(&mut buf)?;
    let buf = &buf[..n];

    let mut offsets: Vec<u64> = Vec::new();

    // Search for 'stco' or 'co64' atoms
    let mut i = 0;
    while i + 16 <= buf.len() {
        if &buf[i + 4..i + 8] == b"stco" {
            let entry_count =
                u32::from_be_bytes([buf[i + 12], buf[i + 13], buf[i + 14], buf[i + 15]]) as usize;
            let max_entries = entry_count.min(65536);
            for e in 0..max_entries {
                let o = i + 16 + e * 4;
                if o + 4 > buf.len() {
                    break;
                }
                let off = u32::from_be_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]) as u64;
                offsets.push(off);
            }
            break;
        } else if &buf[i + 4..i + 8] == b"co64" {
            let entry_count =
                u32::from_be_bytes([buf[i + 12], buf[i + 13], buf[i + 14], buf[i + 15]]) as usize;
            let max_entries = entry_count.min(65536);
            for e in 0..max_entries {
                let o = i + 16 + e * 8;
                if o + 8 > buf.len() {
                    break;
                }
                let off = u64::from_be_bytes([
                    buf[o],
                    buf[o + 1],
                    buf[o + 2],
                    buf[o + 3],
                    buf[o + 4],
                    buf[o + 5],
                    buf[o + 6],
                    buf[o + 7],
                ]);
                offsets.push(off);
            }
            break;
        }
        i += 1;
    }

    compute_index_stats(&offsets, file_size)
}

/// Analyze Matroska Cues index.
fn analyze_matroska_index(file: &mut File, file_size: u64) -> Result<IndexStats> {
    let read_size = file_size.min(2 * 1024 * 1024) as usize;
    let mut buf = vec![0u8; read_size];
    file.seek(SeekFrom::Start(0))?;
    let n = file.read(&mut buf)?;
    let buf = &buf[..n];

    let mut offsets: Vec<u64> = Vec::new();

    // Cues element ID is 0x1C53BB6B
    // CueClusterPosition element ID is 0xF1
    let mut i = 0;
    while i + 4 <= buf.len() {
        if buf[i] == 0x1C && buf[i + 1] == 0x53 && buf[i + 2] == 0xBB && buf[i + 3] == 0x6B {
            // Found Cues element – scan for CueClusterPosition (0xF1)
            let end = (i + 65536).min(buf.len());
            let mut j = i + 4;
            while j + 2 <= end {
                if buf[j] == 0xF1 {
                    let (size, skip) = read_ebml_vint(&buf[j + 1..]);
                    let val_start = j + 1 + skip;
                    if size <= 8 && val_start + size <= buf.len() {
                        let mut val: u64 = 0;
                        for k in 0..size {
                            val = (val << 8) | (buf[val_start + k] as u64);
                        }
                        offsets.push(val);
                    }
                    j += 1 + skip + size;
                } else {
                    j += 1;
                }
            }
            break;
        }
        i += 1;
    }

    compute_index_stats(&offsets, file_size)
}

/// Compute IndexStats from a list of byte offsets and the file size.
fn compute_index_stats(offsets: &[u64], file_size: u64) -> Result<IndexStats> {
    if offsets.is_empty() {
        return Ok(IndexStats {
            entry_count: 0,
            invalid_offset_count: 0,
            missing_keyframe_count: 0,
            covers_full_file: false,
            out_of_bounds_count: 0,
            duplicate_offset_count: 0,
        });
    }

    let entry_count = offsets.len();
    let out_of_bounds_count = offsets.iter().filter(|&&o| o >= file_size).count();
    let invalid_offset_count = offsets.iter().filter(|&&o| o == 0).count();

    // Detect duplicates
    let mut sorted = offsets.to_vec();
    sorted.sort_unstable();
    let duplicate_offset_count = sorted.windows(2).filter(|w| w[0] == w[1]).count();

    // Check coverage: highest valid offset should be within 5% of file end
    let max_valid_offset = offsets.iter().filter(|&&o| o < file_size).max().copied();
    let covers_full_file = max_valid_offset
        .map(|o| o as f64 >= file_size as f64 * 0.95)
        .unwrap_or(false);

    // Estimate missing keyframes: for videos we expect a keyframe at least
    // every 10 seconds.  If total duration is > 60 s, check density.
    // Heuristic: one index entry per 500 KiB is normal; flag if sparser.
    let bytes_per_entry = file_size / entry_count.max(1) as u64;
    let missing_keyframe_count = if bytes_per_entry > 5 * 1024 * 1024 {
        // Rough estimate: one per 5 MiB is very sparse
        ((file_size / (5 * 1024 * 1024)) as usize).saturating_sub(entry_count)
    } else {
        0
    };

    Ok(IndexStats {
        entry_count,
        invalid_offset_count,
        missing_keyframe_count,
        covers_full_file,
        out_of_bounds_count,
        duplicate_offset_count,
    })
}

/// Analyze metadata fields in a media file for corruption and validity.
///
/// Reads the file and attempts to parse metadata from:
/// - MP4 udta/ilst/moov atoms
/// - Matroska Tags and Info elements
/// - AVI INFO chunk
///
/// Checks for: invalid UTF-8, truncated strings, zero/negative duration,
/// negative creation timestamps, and missing essential codec information.
pub fn analyze_metadata(path: &Path) -> Result<Vec<Issue>> {
    let mut issues = Vec::new();
    let mut file = File::open(path)?;
    let file_size = file.metadata()?.len();

    if file_size < 16 {
        return Ok(issues);
    }

    let mut header = [0u8; 16];
    file.read_exact(&mut header)?;

    let validation = if &header[0..4] == b"RIFF" && header.len() >= 12 && &header[8..12] == b"AVI "
    {
        validate_avi_metadata(&mut file, file_size)?
    } else if header.len() >= 8 && &header[4..8] == b"ftyp" {
        validate_mp4_metadata(&mut file, file_size)?
    } else if header.len() >= 4 && header[0..4] == [0x1A, 0x45, 0xDF, 0xA3] {
        validate_matroska_metadata(&mut file, file_size)?
    } else {
        return Ok(issues);
    };

    if !validation.parseable {
        issues.push(Issue {
            issue_type: IssueType::CorruptMetadata,
            severity: Severity::Medium,
            description: "Metadata section could not be parsed; structure may be corrupted"
                .to_string(),
            location: None,
            fixable: true,
        });
        return Ok(issues);
    }

    if validation.invalid_utf8_count > 0 {
        issues.push(Issue {
            issue_type: IssueType::CorruptMetadata,
            severity: Severity::Medium,
            description: format!(
                "{} metadata string field{} contain invalid UTF-8 sequences",
                validation.invalid_utf8_count,
                if validation.invalid_utf8_count == 1 {
                    ""
                } else {
                    "s"
                }
            ),
            location: None,
            fixable: true,
        });
    }

    if validation.truncated_string_count > 0 {
        issues.push(Issue {
            issue_type: IssueType::CorruptMetadata,
            severity: Severity::Low,
            description: format!(
                "{} metadata string field{} appear truncated (null-terminated before stated length)",
                validation.truncated_string_count,
                if validation.truncated_string_count == 1 { "" } else { "s" }
            ),
            location: None,
            fixable: true,
        });
    }

    if !validation.has_valid_duration {
        issues.push(Issue {
            issue_type: IssueType::CorruptMetadata,
            severity: Severity::Medium,
            description: "Metadata reports zero or missing duration; seekbar and progress display \
                          will be incorrect"
                .to_string(),
            location: None,
            fixable: true,
        });
    }

    if validation.has_negative_timestamps {
        issues.push(Issue {
            issue_type: IssueType::CorruptMetadata,
            severity: Severity::Low,
            description:
                "Metadata contains a negative creation/modification timestamp (Unix epoch < 0)"
                    .to_string(),
            location: None,
            fixable: true,
        });
    }

    if !validation.has_codec_info {
        issues.push(Issue {
            issue_type: IssueType::CorruptMetadata,
            severity: Severity::Low,
            description:
                "Codec information is absent from metadata; player codec detection may fail"
                    .to_string(),
            location: None,
            fixable: false,
        });
    }

    Ok(issues)
}

/// Validate AVI INFO chunk metadata.
fn validate_avi_metadata(file: &mut File, file_size: u64) -> Result<MetadataValidation> {
    let read_size = file_size.min(128 * 1024) as usize;
    let mut buf = vec![0u8; read_size];
    file.seek(SeekFrom::Start(0))?;
    let n = file.read(&mut buf)?;
    let buf = &buf[..n];

    let mut invalid_utf8_count = 0;
    let mut truncated_string_count = 0;
    let mut has_valid_duration = false;
    let mut has_codec_info = false;

    // Scan for INFO LIST chunk
    let mut i = 12usize;
    while i + 12 <= buf.len() {
        if &buf[i..i + 4] == b"LIST" {
            let list_size =
                u32::from_le_bytes([buf[i + 4], buf[i + 5], buf[i + 6], buf[i + 7]]) as usize;
            if i + 8 + 4 <= buf.len() && &buf[i + 8..i + 12] == b"INFO" {
                // Scan string chunks inside INFO
                let mut j = i + 12;
                let end = (i + 8 + list_size).min(buf.len());
                while j + 8 <= end {
                    let chunk_size =
                        u32::from_le_bytes([buf[j + 4], buf[j + 5], buf[j + 6], buf[j + 7]])
                            as usize;
                    let data_start = j + 8;
                    let data_end = (data_start + chunk_size).min(buf.len());
                    if data_start < data_end {
                        let string_bytes = &buf[data_start..data_end];
                        let text = std::str::from_utf8(string_bytes);
                        if text.is_err() {
                            invalid_utf8_count += 1;
                        } else {
                            // Check for premature null termination
                            let nul_pos = string_bytes.iter().position(|&b| b == 0);
                            if let Some(pos) = nul_pos {
                                if pos + 1 < string_bytes.len() {
                                    truncated_string_count += 1;
                                }
                            }
                        }
                    }
                    let aligned = (chunk_size + 1) & !1;
                    j += 8 + aligned;
                }
            }
            let aligned = (list_size + 1) & !1;
            i += 8 + aligned;
        } else {
            i += 1;
        }
    }

    // Look for avih chunk for duration and codec
    let mut k = 12usize;
    while k + 8 <= buf.len() {
        if &buf[k..k + 4] == b"avih" {
            // avih: DWORD MicroSecPerFrame, DWORD MaxBytesPerSec, ..., DWORD TotalFrames
            if k + 8 + 40 <= buf.len() {
                let usec_per_frame =
                    u32::from_le_bytes([buf[k + 8], buf[k + 9], buf[k + 10], buf[k + 11]]);
                let total_frames = u32::from_le_bytes([
                    buf[k + 8 + 16],
                    buf[k + 8 + 17],
                    buf[k + 8 + 18],
                    buf[k + 8 + 19],
                ]);
                has_valid_duration = usec_per_frame > 0 && total_frames > 0;
            }
            // Look for strh codec type
            has_codec_info = buf[k..].windows(4).any(|w| w == b"vids" || w == b"auds");
            break;
        }
        k += 1;
    }

    Ok(MetadataValidation {
        parseable: true,
        invalid_utf8_count,
        truncated_string_count,
        has_valid_duration,
        has_negative_timestamps: false,
        has_codec_info,
    })
}

/// Validate MP4 metadata in moov/udta/mvhd atoms.
fn validate_mp4_metadata(file: &mut File, file_size: u64) -> Result<MetadataValidation> {
    let read_size = file_size.min(2 * 1024 * 1024) as usize;
    let mut buf = vec![0u8; read_size];
    file.seek(SeekFrom::Start(0))?;
    let n = file.read(&mut buf)?;
    let buf = &buf[..n];

    let mut invalid_utf8_count = 0;
    let mut truncated_string_count = 0;
    let mut has_valid_duration = false;
    let mut has_negative_timestamps = false;
    let mut has_codec_info = false;

    // Scan for mvhd (Movie Header Box) for duration/timestamps
    let mut i = 0;
    while i + 8 <= buf.len() {
        if i + 8 <= buf.len() && &buf[i + 4..i + 8] == b"mvhd" {
            // mvhd version 0: size(4) type(4) version(1) flags(3) creation_time(4)
            //   modification_time(4) timescale(4) duration(4)
            // mvhd version 1: size(4) type(4) version(1) flags(3) creation_time(8)
            //   modification_time(8) timescale(4) duration(8)
            if i + 32 <= buf.len() {
                let version = buf[i + 8];
                if version == 0 {
                    let timescale =
                        u32::from_be_bytes([buf[i + 20], buf[i + 21], buf[i + 22], buf[i + 23]]);
                    let duration =
                        u32::from_be_bytes([buf[i + 24], buf[i + 25], buf[i + 26], buf[i + 27]]);
                    has_valid_duration = timescale > 0 && duration > 0;
                } else if version == 1 && i + 44 <= buf.len() {
                    let timescale =
                        u32::from_be_bytes([buf[i + 28], buf[i + 29], buf[i + 30], buf[i + 31]]);
                    let duration = u64::from_be_bytes([
                        buf[i + 32],
                        buf[i + 33],
                        buf[i + 34],
                        buf[i + 35],
                        buf[i + 36],
                        buf[i + 37],
                        buf[i + 38],
                        buf[i + 39],
                    ]);
                    has_valid_duration = timescale > 0 && duration > 0;
                }
            }
        }

        // Scan for 'stsd' (sample description) for codec info
        if i + 8 <= buf.len() && &buf[i + 4..i + 8] == b"stsd" {
            has_codec_info = true;
        }

        // Scan for ilst string metadata (iTunes/MP4 tags)
        if i + 8 <= buf.len() && &buf[i + 4..i + 8] == b"\xA9nam"
            || (i + 8 <= buf.len() && &buf[i + 4..i + 8] == b"\xA9art")
        {
            // data atom inside
            if i + 24 <= buf.len() {
                let data_bytes = &buf[i + 24..];
                let text_end = data_bytes
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(data_bytes.len().min(256));
                let text_bytes = &data_bytes[..text_end];
                if std::str::from_utf8(text_bytes).is_err() {
                    invalid_utf8_count += 1;
                }
                // Check for embedded nulls before end
                if let Some(nul) = text_bytes.iter().position(|&b| b == 0) {
                    if nul + 1 < text_bytes.len() {
                        truncated_string_count += 1;
                    }
                }
            }
        }

        // Check for negative timestamps in 'elst' (edit list)
        if i + 8 <= buf.len() && &buf[i + 4..i + 8] == b"elst" {
            if i + 16 <= buf.len() {
                let media_time =
                    i32::from_be_bytes([buf[i + 12], buf[i + 13], buf[i + 14], buf[i + 15]]);
                if media_time < 0 {
                    has_negative_timestamps = true;
                }
            }
        }

        i += 1;
    }

    Ok(MetadataValidation {
        parseable: true,
        invalid_utf8_count,
        truncated_string_count,
        has_valid_duration,
        has_negative_timestamps,
        has_codec_info,
    })
}

/// Validate Matroska metadata (Info and Tags elements).
fn validate_matroska_metadata(file: &mut File, file_size: u64) -> Result<MetadataValidation> {
    let read_size = file_size.min(2 * 1024 * 1024) as usize;
    let mut buf = vec![0u8; read_size];
    file.seek(SeekFrom::Start(0))?;
    let n = file.read(&mut buf)?;
    let buf = &buf[..n];

    let mut invalid_utf8_count = 0;
    let mut truncated_string_count = 0;
    let mut has_valid_duration = false;
    let mut has_codec_info = false;

    // Segment Info element ID: 0x1549A966
    // Duration element within Info: 0x4489
    // CodecID element: 0x86
    // TagString element: 0x4487
    let mut i = 0;
    while i + 4 <= buf.len() {
        // Segment Info
        if buf[i] == 0x15
            && i + 4 <= buf.len()
            && buf[i + 1] == 0x49
            && buf[i + 2] == 0xA9
            && buf[i + 3] == 0x66
        {
            // Scan next 1024 bytes for Duration (0x4489)
            let end = (i + 1024).min(buf.len());
            let mut j = i + 4;
            while j + 2 <= end {
                if buf[j] == 0x44 && j + 2 <= end && buf[j + 1] == 0x89 {
                    let (size, skip) = read_ebml_vint(&buf[j + 2..]);
                    let val_start = j + 2 + skip;
                    if size == 4 && val_start + 4 <= buf.len() {
                        let dur = f32::from_be_bytes([
                            buf[val_start],
                            buf[val_start + 1],
                            buf[val_start + 2],
                            buf[val_start + 3],
                        ]);
                        has_valid_duration = dur > 0.0;
                    } else if size == 8 && val_start + 8 <= buf.len() {
                        let dur = f64::from_be_bytes([
                            buf[val_start],
                            buf[val_start + 1],
                            buf[val_start + 2],
                            buf[val_start + 3],
                            buf[val_start + 4],
                            buf[val_start + 5],
                            buf[val_start + 6],
                            buf[val_start + 7],
                        ]);
                        has_valid_duration = dur > 0.0;
                    }
                    break;
                }
                j += 1;
            }
        }

        // CodecID (0x86) – present in TrackEntry
        if buf[i] == 0x86 && i + 2 <= buf.len() {
            let (size, skip) = read_ebml_vint(&buf[i + 1..]);
            let val_start = i + 1 + skip;
            if size > 0 && val_start + size <= buf.len() {
                let codec_bytes = &buf[val_start..val_start + size];
                if std::str::from_utf8(codec_bytes).is_ok() {
                    has_codec_info = true;
                }
            }
        }

        // TagString (0x4487) – UTF-8 tag values
        if buf[i] == 0x44 && i + 2 <= buf.len() && buf[i + 1] == 0x87 {
            let (size, skip) = read_ebml_vint(&buf[i + 2..]);
            let val_start = i + 2 + skip;
            if size > 0 && val_start + size <= buf.len() {
                let string_bytes = &buf[val_start..val_start + size];
                if std::str::from_utf8(string_bytes).is_err() {
                    invalid_utf8_count += 1;
                } else {
                    let nul_pos = string_bytes.iter().position(|&b| b == 0);
                    if let Some(pos) = nul_pos {
                        if pos + 1 < string_bytes.len() {
                            truncated_string_count += 1;
                        }
                    }
                }
            }
        }

        i += 1;
    }

    Ok(MetadataValidation {
        parseable: true,
        invalid_utf8_count,
        truncated_string_count,
        has_valid_duration,
        has_negative_timestamps: false,
        has_codec_info,
    })
}

/// Calculate file entropy to detect corruption.
///
/// High entropy may indicate encrypted or compressed data,
/// very low entropy may indicate padding or zeros.
pub fn calculate_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut counts = [0u32; 256];
    for &byte in data {
        counts[byte as usize] += 1;
    }

    let len = data.len() as f64;
    let mut entropy = 0.0;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / len;
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// Detect repeated patterns that might indicate corruption.
pub fn detect_patterns(data: &[u8]) -> Vec<(usize, usize, usize)> {
    let mut patterns = Vec::new();

    // Look for repeated sequences of at least 16 bytes
    const MIN_PATTERN_LEN: usize = 16;
    const MIN_PATTERN_COUNT: usize = 3;

    for pattern_len in MIN_PATTERN_LEN..=256 {
        if pattern_len > data.len() / MIN_PATTERN_COUNT {
            break;
        }

        let mut i = 0;
        while i + pattern_len <= data.len() {
            let pattern = &data[i..i + pattern_len];
            let mut count = 1;
            let mut j = i + pattern_len;

            while j + pattern_len <= data.len() {
                if &data[j..j + pattern_len] == pattern {
                    count += 1;
                    j += pattern_len;
                } else {
                    j += 1;
                }
            }

            if count >= MIN_PATTERN_COUNT {
                patterns.push((i, pattern_len, count));
                i += pattern_len * count;
            } else {
                i += 1;
            }
        }
    }

    patterns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_entropy_uniform() {
        let data = vec![0u8; 256];
        let entropy = calculate_entropy(&data);
        assert!(entropy < 0.1);
    }

    #[test]
    fn test_calculate_entropy_random() {
        let data: Vec<u8> = (0..=255).collect();
        let entropy = calculate_entropy(&data);
        assert!(entropy > 7.0);
    }

    #[test]
    fn test_calculate_entropy_empty() {
        let data: Vec<u8> = Vec::new();
        let entropy = calculate_entropy(&data);
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_detect_patterns() {
        let mut data = Vec::new();
        let pattern = b"REPEATINGPATTERN";
        for _ in 0..5 {
            data.extend_from_slice(pattern);
        }

        let patterns = detect_patterns(&data);
        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_detect_patterns_none() {
        let data: Vec<u8> = (0..=255).collect();
        let patterns = detect_patterns(&data);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_compute_timestamp_stats_basic() {
        let timestamps = vec![0i64, 33, 66, 99, 132];
        let stats = compute_timestamp_stats(&timestamps);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.negative_count, 0);
        assert_eq!(stats.duplicate_count, 0);
        assert_eq!(stats.non_monotonic_count, 0);
        assert!((stats.mean_delta - 33.0).abs() < 0.1);
    }

    #[test]
    fn test_compute_timestamp_stats_negative() {
        let timestamps = vec![-100i64, 0, 33, 66];
        let stats = compute_timestamp_stats(&timestamps);
        assert_eq!(stats.negative_count, 1);
    }

    #[test]
    fn test_compute_timestamp_stats_non_monotonic() {
        let timestamps = vec![0i64, 100, 50, 150];
        let stats = compute_timestamp_stats(&timestamps);
        assert_eq!(stats.non_monotonic_count, 1);
    }

    #[test]
    fn test_compute_timestamp_stats_duplicates() {
        let timestamps = vec![0i64, 33, 33, 66];
        let stats = compute_timestamp_stats(&timestamps);
        assert_eq!(stats.duplicate_count, 1);
    }

    #[test]
    fn test_compute_timestamp_stats_large_gap() {
        // mean delta = 33, gap of 1000 should be flagged
        let timestamps = vec![0i64, 33, 66, 1066, 1099];
        let stats = compute_timestamp_stats(&timestamps);
        assert!(stats.large_gap_count >= 1);
        assert_eq!(stats.max_gap, 1000);
    }

    #[test]
    fn test_compute_index_stats_empty() {
        let stats = compute_index_stats(&[], 1000).expect("index stats should succeed");
        assert_eq!(stats.entry_count, 0);
        assert!(!stats.covers_full_file);
    }

    #[test]
    fn test_compute_index_stats_valid() {
        let offsets = vec![100u64, 200, 300, 400, 500];
        let stats = compute_index_stats(&offsets, 1000).expect("index stats should succeed");
        assert_eq!(stats.entry_count, 5);
        assert_eq!(stats.out_of_bounds_count, 0);
        // 500 is 50% of 1000, not >= 95%, so covers_full_file should be false
        assert!(!stats.covers_full_file);
    }

    #[test]
    fn test_compute_index_stats_out_of_bounds() {
        let offsets = vec![100u64, 200, 5000]; // 5000 > file_size=1000
        let stats = compute_index_stats(&offsets, 1000).expect("index stats should succeed");
        assert_eq!(stats.out_of_bounds_count, 1);
    }

    #[test]
    fn test_compute_index_stats_duplicates() {
        let offsets = vec![100u64, 100, 200, 300];
        let stats = compute_index_stats(&offsets, 1000).expect("index stats should succeed");
        assert_eq!(stats.duplicate_offset_count, 1);
    }

    #[test]
    fn test_read_ebml_vint_1byte() {
        let buf = [0x82u8]; // 0x82 = 1000_0010 -> width 1, value = 0x82 & 0x7F = 2
        let (val, consumed) = read_ebml_vint(&buf);
        assert_eq!(val, 2);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_read_ebml_vint_2byte() {
        let buf = [0x40u8, 0x05]; // 0x40 = 0100_0000 -> width 2, value = (0x40 & 0x3F) << 8 | 0x05 = 5
        let (val, consumed) = read_ebml_vint(&buf);
        assert_eq!(val, 5);
        assert_eq!(consumed, 2);
    }

    #[test]
    fn test_analyze_timestamps_empty_buffer() {
        let timestamps: Vec<i64> = Vec::new();
        let stats = compute_timestamp_stats(&timestamps);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean_delta, 0.0);
    }

    #[test]
    fn test_parse_pes_timestamp_zeros() {
        let bytes = [0u8; 5];
        let ts = parse_pes_timestamp(&bytes);
        assert_eq!(ts, 0);
    }
}
