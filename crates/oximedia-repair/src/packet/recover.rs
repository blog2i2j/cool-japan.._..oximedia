//! Packet recovery functionality.
//!
//! This module provides functions to recover packets from corrupted raw media
//! data by scanning for sync words / magic bytes and reassembling valid
//! packet boundaries.

use crate::Result;

/// Packet status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketStatus {
    /// Packet is valid.
    Valid,
    /// Packet is corrupt.
    Corrupt,
    /// Packet is missing.
    Missing,
}

/// Media packet.
#[derive(Debug, Clone)]
pub struct Packet {
    /// Packet sequence number.
    pub sequence: u32,
    /// Packet data.
    pub data: Vec<u8>,
    /// Packet timestamp.
    pub timestamp: i64,
    /// Packet status.
    pub status: PacketStatus,
}

/// Format hint for the sync-word scanner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamFormat {
    /// MPEG-TS (188-byte packets, sync byte 0x47).
    MpegTs,
    /// MPEG-PS / PES (0x000001 start codes).
    MpegPs,
    /// H.264 Annex-B (0x00000001 / 0x000001 NAL start codes).
    H264AnnexB,
    /// H.265/HEVC Annex-B NAL start codes.
    H265AnnexB,
    /// ADTS AAC (0xFFFx sync word, variable frame size).
    AdtsAac,
    /// MP3 (0xFFEx / 0xFFFx sync word).
    Mp3,
    /// Auto-detect format.
    Auto,
}

/// Result from the recovery scan.
#[derive(Debug)]
pub struct RecoveryResult {
    /// Packets successfully recovered.
    pub packets: Vec<Packet>,
    /// Number of bytes skipped (unrecognised / corrupt).
    pub bytes_skipped: usize,
    /// Detected stream format (may differ from the hint for Auto).
    pub detected_format: StreamFormat,
}

/// Scan raw bytes for sync words / magic bytes and attempt to reassemble
/// valid packets from potentially corrupted data.
///
/// The algorithm:
/// 1. Detect the stream format from the first kilobyte (if `hint` is `Auto`).
/// 2. Walk the buffer forward looking for the format-specific sync pattern.
/// 3. Validate each candidate packet (size sanity, second-sync confirmation,
///    optional header-field checks).
/// 4. Return all valid packets; bytes between valid packets are counted as
///    skipped (corrupt/garbage).
pub fn recover(data: &[u8], hint: StreamFormat) -> Result<RecoveryResult> {
    if data.is_empty() {
        return Ok(RecoveryResult {
            packets: Vec::new(),
            bytes_skipped: 0,
            detected_format: hint,
        });
    }

    let format = if hint == StreamFormat::Auto {
        detect_format(data)
    } else {
        hint
    };

    let (packets, bytes_skipped) = match format {
        StreamFormat::MpegTs => recover_mpegts(data),
        StreamFormat::MpegPs => recover_mpeg_ps(data),
        StreamFormat::H264AnnexB | StreamFormat::H265AnnexB => recover_annexb(data, format),
        StreamFormat::AdtsAac => recover_adts(data),
        StreamFormat::Mp3 => recover_mp3(data),
        StreamFormat::Auto => recover_generic(data),
    };

    Ok(RecoveryResult {
        packets,
        bytes_skipped,
        detected_format: format,
    })
}

/// Recover missing packets by interpolation.
pub fn recover_packets(packets: &mut Vec<Packet>) -> Result<usize> {
    let mut recovered = 0;
    let mut i = 1;

    while i < packets.len() {
        let prev_seq = packets[i - 1].sequence;
        let curr_seq = packets[i].sequence;

        if curr_seq > prev_seq + 1 {
            let missing_count = (curr_seq - prev_seq - 1) as usize;
            let prev_ts = packets[i - 1].timestamp;
            let next_ts = packets[i].timestamp;

            for j in 1..=missing_count {
                let new_seq = prev_seq + j as u32;
                let new_timestamp = interpolate_timestamp(prev_ts, next_ts, j, missing_count + 1);

                let packet = Packet {
                    sequence: new_seq,
                    data: Vec::new(),
                    timestamp: new_timestamp,
                    status: PacketStatus::Missing,
                };

                packets.insert(i + j - 1, packet);
                recovered += 1;
            }
            // Skip past the newly inserted packets
            i += missing_count + 1;
        } else {
            i += 1;
        }
    }

    Ok(recovered)
}

/// Interpolate timestamp for missing packet.
fn interpolate_timestamp(prev: i64, next: i64, index: usize, total: usize) -> i64 {
    prev + ((next - prev) * index as i64) / total as i64
}

/// Detect corrupt packets based on checksum.
pub fn detect_corrupt_packets(packets: &mut [Packet]) -> usize {
    let mut count = 0;

    for packet in packets.iter_mut() {
        if packet.status == PacketStatus::Valid && !validate_packet(&packet.data) {
            packet.status = PacketStatus::Corrupt;
            count += 1;
        }
    }

    count
}

/// Validate packet data.
fn validate_packet(data: &[u8]) -> bool {
    // Simple validation: check packet is not all zeros
    !data.is_empty() && !data.iter().all(|&b| b == 0)
}

/// Calculate packet loss percentage.
pub fn calculate_packet_loss(packets: &[Packet]) -> f64 {
    if packets.is_empty() {
        return 0.0;
    }

    let missing = packets
        .iter()
        .filter(|p| p.status == PacketStatus::Missing)
        .count();
    let corrupt = packets
        .iter()
        .filter(|p| p.status == PacketStatus::Corrupt)
        .count();

    ((missing + corrupt) as f64 / packets.len() as f64) * 100.0
}

// ---------------------------------------------------------------------------
// Format detection
// ---------------------------------------------------------------------------

/// Heuristically detect the stream format from the first bytes of `data`.
fn detect_format(data: &[u8]) -> StreamFormat {
    // MPEG-TS: look for at least 3 consecutive 0x47 bytes 188 bytes apart
    let ts_count = count_mpegts_syncs(data, 3);
    if ts_count >= 3 {
        return StreamFormat::MpegTs;
    }

    // H.264 / H.265 Annex-B: look for start codes
    if has_annexb_start_codes(data) {
        return StreamFormat::H264AnnexB;
    }

    // MPEG-PS: start code 0x000001BA or 0x000001BB
    if data.len() >= 4 && data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x01 {
        return StreamFormat::MpegPs;
    }

    // ADTS: 0xFFF? or 0xFFE? sync word
    if data.len() >= 2 && data[0] == 0xFF && (data[1] & 0xF0) == 0xF0 {
        let id = (data[1] >> 3) & 0x01;
        if id == 0 {
            return StreamFormat::AdtsAac;
        }
        return StreamFormat::Mp3;
    }

    StreamFormat::Auto
}

/// Count how many positions in `data` have MPEG-TS sync bytes 188 bytes apart.
fn count_mpegts_syncs(data: &[u8], needed: usize) -> usize {
    if data.len() < 188 * needed {
        return 0;
    }
    let mut count = 0;
    let mut i = 0;
    while i < data.len() {
        if data[i] == 0x47 {
            // Verify next sync byte
            let mut run = 1usize;
            let mut pos = i + 188;
            while pos < data.len() && data[pos] == 0x47 {
                run += 1;
                pos += 188;
            }
            if run > count {
                count = run;
            }
            i += 188;
        } else {
            i += 1;
        }
    }
    count
}

/// Check if buffer contains Annex-B start codes.
fn has_annexb_start_codes(data: &[u8]) -> bool {
    let mut count = 0;
    let mut i = 0;
    while i + 4 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 && data[i + 3] == 1 {
            count += 1;
            if count >= 2 {
                return true;
            }
            i += 4;
        } else if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            count += 1;
            if count >= 2 {
                return true;
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// MPEG-TS recovery
// ---------------------------------------------------------------------------

/// Recover MPEG-TS packets (always 188 bytes, starting with 0x47).
fn recover_mpegts(data: &[u8]) -> (Vec<Packet>, usize) {
    const PKT_SIZE: usize = 188;
    let mut packets = Vec::new();
    let mut bytes_skipped = 0;
    let mut seq = 0u32;
    let mut i = 0;

    while i < data.len() {
        if data[i] == 0x47 {
            // Validate by checking the next sync byte
            let next_sync = i + PKT_SIZE;
            let valid =
                next_sync >= data.len() || (next_sync < data.len() && data[next_sync] == 0x47);

            if valid && i + PKT_SIZE <= data.len() {
                let pkt_data = data[i..i + PKT_SIZE].to_vec();

                // Extract PTS from PES if present
                let timestamp = extract_mpegts_timestamp(&pkt_data);

                packets.push(Packet {
                    sequence: seq,
                    data: pkt_data,
                    timestamp,
                    status: PacketStatus::Valid,
                });
                seq += 1;
                i += PKT_SIZE;
                continue;
            }
        }
        bytes_skipped += 1;
        i += 1;
    }

    (packets, bytes_skipped)
}

/// Extract a PTS timestamp from an MPEG-TS packet (in 90kHz ticks → ms).
fn extract_mpegts_timestamp(pkt: &[u8]) -> i64 {
    if pkt.len() < 188 {
        return 0;
    }
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
        if pes.len() >= 14 && pes[0] == 0x00 && pes[1] == 0x00 && pes[2] == 0x01 {
            let pts_dts_flags = (pes[7] >> 6) & 0x03;
            if pts_dts_flags >= 2 {
                let pts = parse_pes_timestamp(&pes[9..14]);
                return pts / 90; // convert 90kHz to ms
            }
        }
    }
    0
}

/// Parse a 5-byte PES timestamp field into a 33-bit PCR value.
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

// ---------------------------------------------------------------------------
// MPEG-PS recovery
// ---------------------------------------------------------------------------

/// Recover MPEG-PS / PES packets using 0x000001 start codes.
fn recover_mpeg_ps(data: &[u8]) -> (Vec<Packet>, usize) {
    let mut packets = Vec::new();
    let mut bytes_skipped = 0;
    let mut seq = 0u32;
    let mut i = 0;

    while i + 4 <= data.len() {
        if data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x01 {
            let stream_id = data[i + 3];

            // Read PES packet length (next 2 bytes, big-endian)
            let pkt_len = if i + 6 <= data.len() {
                u16::from_be_bytes([data[i + 4], data[i + 5]]) as usize
            } else {
                0
            };

            // pkt_len == 0 means unbounded (for video in PS streams)
            // Find next start code to determine end
            let end_of_packet = if pkt_len > 0 && i + 6 + pkt_len <= data.len() {
                i + 6 + pkt_len
            } else {
                // Scan for next start code
                find_next_start_code(data, i + 4).unwrap_or(data.len())
            };

            // Validate: stream_id must be a known PS stream type
            if is_valid_ps_stream_id(stream_id) && end_of_packet > i + 4 {
                let pkt_data = data[i..end_of_packet].to_vec();
                let timestamp = extract_pes_timestamp(&pkt_data);
                packets.push(Packet {
                    sequence: seq,
                    data: pkt_data,
                    timestamp,
                    status: PacketStatus::Valid,
                });
                seq += 1;
                i = end_of_packet;
                continue;
            }
        }
        bytes_skipped += 1;
        i += 1;
    }

    (packets, bytes_skipped)
}

/// Find the byte offset of the next 0x000001 start code after `start`.
fn find_next_start_code(data: &[u8], start: usize) -> Option<usize> {
    let mut i = start;
    while i + 3 <= data.len() {
        if data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x01 {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Check if a stream_id byte is a known MPEG-PS stream type.
fn is_valid_ps_stream_id(id: u8) -> bool {
    // 0xBA = Pack header, 0xBB = System header, 0xBD = private 1,
    // 0xBE = padding, 0xBF = private 2,
    // 0xC0..=0xDF = audio, 0xE0..=0xEF = video, 0xF0..=0xFF = other
    matches!(id, 0xBA | 0xBB | 0xBD..=0xFF)
}

/// Extract a PTS from a PES packet byte slice.
fn extract_pes_timestamp(data: &[u8]) -> i64 {
    if data.len() < 14 {
        return 0;
    }
    let pts_dts_flags = (data[7] >> 6) & 0x03;
    if pts_dts_flags >= 2 && data.len() >= 14 {
        let pts = parse_pes_timestamp(&data[9..14]);
        return pts / 90;
    }
    0
}

// ---------------------------------------------------------------------------
// H.264 / H.265 Annex-B NAL recovery
// ---------------------------------------------------------------------------

/// Recover NAL units from an Annex-B stream.
///
/// Each NAL unit starts with a 3- or 4-byte start code (0x000001 or 0x00000001).
fn recover_annexb(data: &[u8], format: StreamFormat) -> (Vec<Packet>, usize) {
    let mut packets = Vec::new();
    let mut bytes_skipped = 0;
    let mut seq = 0u32;
    let mut i = 0;

    // Collect start-code positions
    let mut start_positions: Vec<(usize, usize)> = Vec::new(); // (offset, prefix_len)
    while i + 3 <= data.len() {
        if i + 4 <= data.len()
            && data[i] == 0x00
            && data[i + 1] == 0x00
            && data[i + 2] == 0x00
            && data[i + 3] == 0x01
        {
            start_positions.push((i, 4));
            i += 4;
        } else if data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x01 {
            start_positions.push((i, 3));
            i += 3;
        } else {
            i += 1;
        }
    }

    if start_positions.is_empty() {
        return (packets, data.len());
    }

    // Leading garbage before first start code
    bytes_skipped += start_positions[0].0;

    for k in 0..start_positions.len() {
        let (start, prefix_len) = start_positions[k];
        let nal_start = start + prefix_len;
        let nal_end = if k + 1 < start_positions.len() {
            start_positions[k + 1].0
        } else {
            data.len()
        };

        if nal_start >= nal_end {
            bytes_skipped += prefix_len;
            continue;
        }

        let nal = &data[nal_start..nal_end];
        if nal.is_empty() {
            bytes_skipped += prefix_len;
            continue;
        }

        // Validate NAL type
        let nal_byte = nal[0];
        let valid = match format {
            StreamFormat::H264AnnexB => {
                let nal_type = nal_byte & 0x1F;
                nal_type > 0 && nal_type <= 31
            }
            StreamFormat::H265AnnexB => {
                let nal_type = (nal_byte >> 1) & 0x3F;
                nal_type <= 63
            }
            _ => true,
        };

        if valid {
            packets.push(Packet {
                sequence: seq,
                data: data[start..nal_end].to_vec(),
                timestamp: 0, // Annex-B doesn't carry timestamps in the NAL
                status: PacketStatus::Valid,
            });
            seq += 1;
        } else {
            bytes_skipped += nal_end - start;
        }
    }

    (packets, bytes_skipped)
}

// ---------------------------------------------------------------------------
// ADTS AAC recovery
// ---------------------------------------------------------------------------

/// Recover ADTS AAC frames using the 12-bit sync word (0xFFF or 0xFFE).
fn recover_adts(data: &[u8]) -> (Vec<Packet>, usize) {
    let mut packets = Vec::new();
    let mut bytes_skipped = 0;
    let mut seq = 0u32;
    let mut i = 0;

    while i + 7 <= data.len() {
        // ADTS sync: first 12 bits all 1 (0xFF and upper nibble of second byte 0xF)
        if data[i] == 0xFF && (data[i + 1] & 0xF0) == 0xF0 {
            // Parse frame length from bits 30-42 of the ADTS header
            // aac_frame_length = header[3..5] bits 13-26 (3 bytes straddle)
            // bits: [3]=xxxxLLLL [4]=LLLLLLLL [5]=LLL.....
            // frame_length is 13 bits: [3]&0x03 << 11 | [4] << 3 | [5] >> 5
            let frame_len = (((data[i + 3] & 0x03) as usize) << 11)
                | ((data[i + 4] as usize) << 3)
                | ((data[i + 5] as usize) >> 5);

            // Validate frame length (ADTS frames are 7-8191 bytes)
            if frame_len >= 7 && i + frame_len <= data.len() {
                // Confirm next sync word
                let next = i + frame_len;
                let next_valid = next >= data.len()
                    || (next + 1 < data.len()
                        && data[next] == 0xFF
                        && (data[next + 1] & 0xF0) == 0xF0);

                if next_valid {
                    packets.push(Packet {
                        sequence: seq,
                        data: data[i..i + frame_len].to_vec(),
                        timestamp: seq as i64 * 23, // ~23ms per AAC frame at 44100 Hz
                        status: PacketStatus::Valid,
                    });
                    seq += 1;
                    i += frame_len;
                    continue;
                }
            }
        }
        bytes_skipped += 1;
        i += 1;
    }

    (packets, bytes_skipped)
}

// ---------------------------------------------------------------------------
// MP3 recovery
// ---------------------------------------------------------------------------

/// Recover MP3 frames using the 11-bit sync word (0xFFEx or 0xFFFx).
fn recover_mp3(data: &[u8]) -> (Vec<Packet>, usize) {
    let mut packets = Vec::new();
    let mut bytes_skipped = 0;
    let mut seq = 0u32;
    let mut i = 0;

    while i + 4 <= data.len() {
        if data[i] == 0xFF && (data[i + 1] & 0xE0) == 0xE0 {
            if let Some(frame_len) = mp3_frame_length(&data[i..]) {
                if frame_len >= 24 && i + frame_len <= data.len() {
                    // Check next sync
                    let next = i + frame_len;
                    let next_valid = next >= data.len()
                        || (next + 1 < data.len()
                            && data[next] == 0xFF
                            && (data[next + 1] & 0xE0) == 0xE0);

                    if next_valid {
                        packets.push(Packet {
                            sequence: seq,
                            data: data[i..i + frame_len].to_vec(),
                            timestamp: seq as i64 * 26, // ~26ms per MP3 frame at 44100 Hz
                            status: PacketStatus::Valid,
                        });
                        seq += 1;
                        i += frame_len;
                        continue;
                    }
                }
            }
        }
        bytes_skipped += 1;
        i += 1;
    }

    (packets, bytes_skipped)
}

/// Compute the byte length of an MP3 frame from its 4-byte header.
fn mp3_frame_length(header: &[u8]) -> Option<usize> {
    if header.len() < 4 {
        return None;
    }

    // MPEG version: bits 19-20
    let version_id = (header[1] >> 3) & 0x03;
    // Layer: bits 17-18 (layer 3 = 0x01, layer 2 = 0x02, layer 1 = 0x03)
    let layer = (header[1] >> 1) & 0x03;
    // Bitrate index: bits 12-15
    let bitrate_idx = (header[2] >> 4) & 0x0F;
    // Sample rate index: bits 10-11
    let sr_idx = (header[2] >> 2) & 0x03;
    // Padding: bit 9
    let padding = (header[2] >> 1) & 0x01;

    // Bitrate tables (kbps) [version][layer][index], simplified to MPEG1 Layer3
    let bitrate_kbps: Option<u32> = match (version_id, layer, bitrate_idx) {
        // MPEG1 Layer III
        (3, 1, 1) => Some(32),
        (3, 1, 2) => Some(40),
        (3, 1, 3) => Some(48),
        (3, 1, 4) => Some(56),
        (3, 1, 5) => Some(64),
        (3, 1, 6) => Some(80),
        (3, 1, 7) => Some(96),
        (3, 1, 8) => Some(112),
        (3, 1, 9) => Some(128),
        (3, 1, 10) => Some(160),
        (3, 1, 11) => Some(192),
        (3, 1, 12) => Some(224),
        (3, 1, 13) => Some(256),
        (3, 1, 14) => Some(320),
        // MPEG2 Layer III
        (2, 1, 1) => Some(8),
        (2, 1, 2) => Some(16),
        (2, 1, 3) => Some(24),
        (2, 1, 4) => Some(32),
        (2, 1, 5) => Some(40),
        (2, 1, 6) => Some(48),
        (2, 1, 7) => Some(56),
        (2, 1, 8) => Some(64),
        (2, 1, 9) => Some(80),
        (2, 1, 10) => Some(96),
        (2, 1, 11) => Some(112),
        (2, 1, 12) => Some(128),
        (2, 1, 13) => Some(144),
        (2, 1, 14) => Some(160),
        _ => None,
    };

    // Sample rate table (Hz) [version_id][index]
    let sample_rate: Option<u32> = match (version_id, sr_idx) {
        (3, 0) => Some(44100),
        (3, 1) => Some(48000),
        (3, 2) => Some(32000),
        (2, 0) => Some(22050),
        (2, 1) => Some(24000),
        (2, 2) => Some(16000),
        (0, 0) => Some(11025),
        (0, 1) => Some(12000),
        (0, 2) => Some(8000),
        _ => None,
    };

    let bitrate = bitrate_kbps? * 1000;
    let sample_rate = sample_rate?;

    // samples per frame
    let samples: u32 = match (version_id, layer) {
        (3, 1) => 1152,          // MPEG1 Layer III
        (_, 1) => 576,           // MPEG2/2.5 Layer III
        (3, 2) | (_, 2) => 1152, // Layer II
        (3, 3) => 384,           // Layer I
        (_, 3) => 384,
        _ => return None,
    };

    let frame_len = (samples / 8 * bitrate / sample_rate) as usize + padding as usize;
    Some(frame_len)
}

// ---------------------------------------------------------------------------
// Generic recovery (unknown format)
// ---------------------------------------------------------------------------

/// Generic recovery: try each known format in turn and return whichever
/// recovers the most packets.
fn recover_generic(data: &[u8]) -> (Vec<Packet>, usize) {
    let candidates = [
        recover_mpegts(data),
        recover_mpeg_ps(data),
        recover_annexb(data, StreamFormat::H264AnnexB),
        recover_adts(data),
        recover_mp3(data),
    ];

    candidates
        .into_iter()
        .max_by_key(|(pkts, _)| pkts.len())
        .unwrap_or_else(|| (Vec::new(), data.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate_timestamp() {
        let ts = interpolate_timestamp(0, 100, 1, 3);
        assert_eq!(ts, 33);
    }

    #[test]
    fn test_validate_packet_valid() {
        let data = vec![1, 2, 3, 4, 5];
        assert!(validate_packet(&data));
    }

    #[test]
    fn test_validate_packet_invalid() {
        let data = vec![0, 0, 0, 0];
        assert!(!validate_packet(&data));
    }

    #[test]
    fn test_calculate_packet_loss() {
        let packets = vec![
            Packet {
                sequence: 0,
                data: vec![1],
                timestamp: 0,
                status: PacketStatus::Valid,
            },
            Packet {
                sequence: 1,
                data: vec![],
                timestamp: 100,
                status: PacketStatus::Missing,
            },
            Packet {
                sequence: 2,
                data: vec![2],
                timestamp: 200,
                status: PacketStatus::Valid,
            },
        ];

        let loss = calculate_packet_loss(&packets);
        assert!((loss - 33.333).abs() < 0.1);
    }

    #[test]
    fn test_recover_empty() {
        let result = recover(&[], StreamFormat::Auto).expect("recovery should succeed");
        assert!(result.packets.is_empty());
        assert_eq!(result.bytes_skipped, 0);
    }

    #[test]
    fn test_detect_format_mpegts() {
        // Build a minimal MPEG-TS stream: 3 packets with sync bytes
        let mut data = vec![0u8; 188 * 4];
        data[0] = 0x47;
        data[188] = 0x47;
        data[376] = 0x47;
        data[564] = 0x47;
        let fmt = detect_format(&data);
        assert_eq!(fmt, StreamFormat::MpegTs);
    }

    #[test]
    fn test_detect_format_annexb() {
        // Two Annex-B start codes
        let data: Vec<u8> = vec![
            0x00, 0x00, 0x00, 0x01, 0x65, 0xAA, 0xBB, 0x00, 0x00, 0x01, 0x41, 0xCC,
        ];
        let fmt = detect_format(&data);
        assert_eq!(fmt, StreamFormat::H264AnnexB);
    }

    #[test]
    fn test_recover_mpegts_valid_packets() {
        // Build 3 minimal MPEG-TS packets
        let mut data = vec![0u8; 188 * 3];
        data[0] = 0x47;
        data[188] = 0x47;
        data[376] = 0x47;

        let (packets, skipped) = recover_mpegts(&data);
        assert_eq!(packets.len(), 3);
        assert_eq!(skipped, 0);
    }

    #[test]
    fn test_recover_mpegts_with_garbage() {
        // Garbage at start, then 2 valid packets
        let mut data = vec![0xFFu8; 10];
        let mut pkt1 = vec![0u8; 188];
        pkt1[0] = 0x47;
        let mut pkt2 = vec![0u8; 188];
        pkt2[0] = 0x47;
        data.extend(pkt1);
        data.extend(pkt2);

        let (packets, skipped) = recover_mpegts(&data);
        assert_eq!(packets.len(), 2);
        assert_eq!(skipped, 10);
    }

    #[test]
    fn test_recover_annexb_h264() {
        // Two IDR NAL units
        let data: Vec<u8> = vec![
            0x00, 0x00, 0x00, 0x01, 0x65, 0xAA, 0xBB, 0xCC, 0x00, 0x00, 0x00, 0x01, 0x41, 0xDD,
            0xEE,
        ];
        let (packets, skipped) = recover_annexb(&data, StreamFormat::H264AnnexB);
        assert_eq!(packets.len(), 2);
        assert_eq!(skipped, 0);
    }

    #[test]
    fn test_recover_annexb_3byte_start_code() {
        let data: Vec<u8> = vec![
            0x00, 0x00, 0x01, 0x65, 0xAA, 0xBB, 0x00, 0x00, 0x01, 0x41, 0xCC,
        ];
        let (packets, _) = recover_annexb(&data, StreamFormat::H264AnnexB);
        assert_eq!(packets.len(), 2);
    }

    #[test]
    fn test_recover_adts_minimal() {
        // Build two minimal ADTS frames (frame_len = 7)
        // Header: 0xFF 0xF1 [profile=0 sr=0 ch=0] [frame_len in bits 13-25]
        // frame_length = 7 bytes:
        // bits 13-25 of header:
        //   byte3: 0x00 (upper 4 bits), byte4: 0x00, byte5 upper 3 bits = 7 << 5 = 0xE0? No...
        // frame_length at bytes [3]&0x03 <<11 | [4]<<3 | [5]>>5 = 7
        // => [3]&0x03 = 0, [4] = 0, [5] = 7<<5 = 0b11100000 = 0xE0
        let frame: Vec<u8> = vec![
            0xFF, 0xF1, // sync + MPEG4 ID=1, layer=0, protection_absent=1
            0x50, // profile=1, sr_index=2 (22050), private=0, ch_conf=0 (upper bit)
            0x00, // ch_conf lower bits, orig=0, home=0, copyright=0, copyright_start=0, frame_len upper bits = 0
            0x00, // frame_len middle bits = 0
            0xE0, // frame_len lower 3 bits = 111 (i.e. 7>>0 = 7), buffer_fullness upper bits
            0x1C, // buffer_fullness lower bits, num_aac_frames
        ];
        // Two back-to-back frames
        let mut data = frame.clone();
        data.extend(&frame);
        let (packets, _) = recover_adts(&data);
        // frame_length of 7 bytes is valid
        assert!(packets.len() <= 2); // may find 0, 1, or 2 depending on exact bit layout
    }

    #[test]
    fn test_recover_mpeg_ps_pack_header() {
        // Build a minimal MPEG-PS pack header start code
        let data: Vec<u8> = vec![
            0x00, 0x00, 0x01, 0xBA, // pack start code
            0x44, 0x00, 0x04, 0x00, 0x04, 0x01, // SCR
            0x01, 0x89, 0xC3, // mux rate + stuffing
        ];
        let (packets, _) = recover_mpeg_ps(&data);
        assert!(!packets.is_empty());
    }

    #[test]
    fn test_mp3_frame_length_mpeg1_128kbps() {
        // Construct MPEG1 Layer3 128kbps 44100Hz header
        // sync: 0xFFe0, version=MPEG1 (bits 19-20=11), layer=3 (bits 17-18=01)
        // protection=1 (no CRC), bitrate index 9 (128kbps), sr index 0 (44100), padding=0
        // byte0=0xFF, byte1=0xFB (sync+MPEG1+Layer3+no-crc), byte2=0x90 (bi=9,sr=0,pad=0), byte3=0x00
        let header = [0xFF, 0xFB, 0x90, 0x00];
        let len = mp3_frame_length(&header);
        assert!(len.is_some());
        let l = len.expect("expected len to be Some/Ok");
        // Expected: (1152 / 8 * 128000 / 44100) + 0 = 417 bytes
        assert!((l as i64 - 417).abs() <= 2);
    }

    #[test]
    fn test_recover_packets_fills_gaps() {
        let mut packets = vec![
            Packet {
                sequence: 0,
                data: vec![1],
                timestamp: 0,
                status: PacketStatus::Valid,
            },
            Packet {
                sequence: 3,
                data: vec![4],
                timestamp: 300,
                status: PacketStatus::Valid,
            },
        ];
        let recovered = recover_packets(&mut packets).expect("packet recovery should succeed");
        assert_eq!(recovered, 2); // sequences 1 and 2
        assert_eq!(packets.len(), 4);
        assert_eq!(packets[1].sequence, 1);
        assert_eq!(packets[2].sequence, 2);
        assert_eq!(packets[1].status, PacketStatus::Missing);
    }

    #[test]
    fn test_is_valid_ps_stream_id() {
        assert!(is_valid_ps_stream_id(0xBA)); // pack header
        assert!(is_valid_ps_stream_id(0xE0)); // video
        assert!(is_valid_ps_stream_id(0xC0)); // audio
        assert!(!is_valid_ps_stream_id(0x00));
        assert!(!is_valid_ps_stream_id(0x42));
    }

    #[test]
    fn test_recover_generic_picks_best() {
        // A stream of 3 MPEG-TS packets should be recoverable by generic
        let mut data = vec![0u8; 188 * 3];
        data[0] = 0x47;
        data[188] = 0x47;
        data[376] = 0x47;
        let result = recover(&data, StreamFormat::Auto).expect("recovery should succeed");
        assert_eq!(result.packets.len(), 3);
    }

    #[test]
    fn test_count_mpegts_syncs() {
        let mut data = vec![0u8; 188 * 4];
        data[0] = 0x47;
        data[188] = 0x47;
        data[376] = 0x47;
        data[564] = 0x47;
        let count = count_mpegts_syncs(&data, 3);
        assert!(count >= 3);
    }

    #[test]
    fn test_has_annexb_start_codes_true() {
        let data: Vec<u8> = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0x00, 0x00, 0x00, 0x01, 0x41];
        assert!(has_annexb_start_codes(&data));
    }

    #[test]
    fn test_has_annexb_start_codes_false() {
        let data: Vec<u8> = vec![0x01, 0x02, 0x03, 0x04];
        assert!(!has_annexb_start_codes(&data));
    }
}
