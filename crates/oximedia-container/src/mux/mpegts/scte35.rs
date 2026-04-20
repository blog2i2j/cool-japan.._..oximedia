//! SCTE-35 splice_info_section emitter for MPEG-TS.
//!
//! Constructs well-formed SCTE-35 byte payloads that can be inserted into an
//! MPEG-TS stream on a dedicated splice PID.  All emitted sections carry a
//! valid MPEG-2 CRC-32 computed over the preceding bytes.
//!
//! # Supported commands
//!
//! - [`emit_time_signal`] — `0x06 time_signal` with an optional 33-bit PTS.
//! - [`emit_splice_null`] — `0x00 splice_null` (keep-alive / heartbeat).
//! - [`emit_splice_insert`] — `0x05 splice_insert` (immediate out-of-network break).
//!
//! # Example
//!
//! ```ignore
//! use std::time::Duration;
//! use oximedia_container::mux::mpegts::scte35::emit_time_signal;
//!
//! // Emit a time_signal section for PTS = 5 seconds.
//! let bytes = emit_time_signal(Some(Duration::from_secs(5)));
//! // bytes is a valid splice_info_section ready to be wrapped in a TS packet.
//! ```

use std::time::Duration;

// ─── MPEG-2 CRC-32 ──────────────────────────────────────────────────────────

/// MPEG-2 CRC-32 polynomial (0x04C11DB7).
const CRC32_POLY: u32 = 0x04C1_1DB7;

/// Computes the MPEG-2 CRC-32 for `data`.
///
/// Initialised to `0xFFFFFFFF`; the final value is returned without post-inversion.
fn compute_crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= u32::from(byte) << 24;
        for _ in 0..8 {
            if crc & 0x8000_0000 != 0 {
                crc = (crc << 1) ^ CRC32_POLY;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

// ─── Section header helpers ──────────────────────────────────────────────────

/// Constant tier value: all tiers (0x0FFF).
const TIER_ALL: u16 = 0x0FFF;

/// SCTE-35 table ID.
const TABLE_ID: u8 = 0xFC;

/// Writes a skeleton `splice_info_section` header up to (but not including)
/// the splice command bytes.
///
/// Layout (bytes):
/// ```text
///  0      table_id (0xFC)
///  1-2    section_syntax_indicator(0) | private_indicator(0) | reserved(11) | section_length(12)
///  3      protocol_version (0x00)
///  4      encrypted_packet(0) | encryption_algorithm(0) | pts_adjustment[32]
///  5-8    pts_adjustment[31:0] (all zero)
///  9      cw_index (0x00)
/// 10-11   tier(12) | splice_command_length(12)
/// 12      splice_command_type
/// ```
///
/// The `section_length` field is left as a placeholder (zeroed) and must be
/// back-patched by the caller using [`backpatch_section_length`].
fn write_section_header(buf: &mut Vec<u8>, command_type: u8, command_len: u16) {
    buf.push(TABLE_ID);
    // section_syntax_indicator=0, private_indicator=0, reserved=11b (0x30 mask)
    // section_length placeholder – back-patched after building the full section
    buf.push(0x30); // upper nibble reserved bits; section_length will be or'd in
    buf.push(0x00); // lower byte of section_length placeholder
    buf.push(0x00); // protocol_version
                    // encrypted_packet=0, encryption_algorithm=0b000000, pts_adjustment bit32=0
    buf.extend_from_slice(&[0x00; 5]); // pts_adjustment = 0
    buf.push(0x00); // cw_index
                    // tier(12) | splice_command_length(12)
    let tier = TIER_ALL;
    buf.push(((tier >> 4) & 0xFF) as u8);
    buf.push((((tier & 0x0F) << 4) | ((command_len >> 8) & 0x0F)) as u8);
    buf.push((command_len & 0xFF) as u8);
    buf.push(command_type);
}

/// Back-patches the `section_length` field at bytes [1..=2] of `buf`.
///
/// `section_length` counts all bytes from byte 3 onwards (i.e., total length
/// minus the 3-byte header) and must include the 4-byte CRC that has not yet
/// been appended when this is called.
fn backpatch_section_length(buf: &mut Vec<u8>) {
    // section_length = total payload after byte[2], plus 4 (CRC not yet written)
    let section_length = buf.len() - 3 + 4;
    buf[1] = 0x30 | ((section_length >> 8) as u8 & 0x0F);
    buf[2] = (section_length & 0xFF) as u8;
}

/// Appends the MPEG-2 CRC-32 of `buf` to `buf` as 4 big-endian bytes.
fn append_crc(buf: &mut Vec<u8>) {
    let crc = compute_crc32(buf);
    buf.extend_from_slice(&crc.to_be_bytes());
}

// ─── Time signal helper ──────────────────────────────────────────────────────

/// Encodes a 5-byte `splice_time` for use in `time_signal` and `splice_insert`
/// commands.
///
/// When `pts` is `Some(d)`, `time_specified_flag=1` and the 33-bit PTS is
/// computed from the Duration in 90 kHz ticks (clamped to 33 bits).
/// When `pts` is `None`, `time_specified_flag=0` and 4 reserved bytes follow.
fn encode_splice_time(pts: Option<Duration>) -> [u8; 5] {
    match pts {
        Some(d) => {
            // 90 kHz ticks; 33-bit field → clamp to 2^33-1
            let ticks = (d.as_secs_f64() * 90_000.0) as u64 & 0x1_FFFF_FFFF;
            [
                0x80 | ((ticks >> 32) as u8 & 0x01), // time_specified_flag=1 | reserved(6) | pts[32]
                ((ticks >> 24) & 0xFF) as u8,
                ((ticks >> 16) & 0xFF) as u8,
                ((ticks >> 8) & 0xFF) as u8,
                (ticks & 0xFF) as u8,
            ]
        }
        None => [
            0x7F, // time_specified_flag=0, reserved=0x7F
            0xFF, 0xFF, 0xFF, 0xFF,
        ],
    }
}

// ─── Public emitters ─────────────────────────────────────────────────────────

/// Emits a SCTE-35 `time_signal` `splice_info_section` (command type `0x06`).
///
/// The returned bytes form a complete, CRC-verified section suitable for
/// wrapping in a private TS packet (PUSI=1, adaptation_field not required).
///
/// # Arguments
///
/// - `pts` — Optional presentation time for the signal.  If `None` the
///   `time_specified_flag` is set to 0 (no PTS carried in the section).
///
/// # Example
///
/// ```ignore
/// use std::time::Duration;
/// use oximedia_container::mux::mpegts::scte35::emit_time_signal;
///
/// let bytes = emit_time_signal(Some(Duration::from_secs(30)));
/// assert_eq!(bytes[0], 0xFC); // table_id
/// ```
#[must_use]
pub fn emit_time_signal(pts: Option<Duration>) -> Vec<u8> {
    let splice_time = encode_splice_time(pts);
    let command_len = splice_time.len() as u16;

    let mut buf = Vec::with_capacity(24);
    write_section_header(&mut buf, 0x06, command_len);
    buf.extend_from_slice(&splice_time);

    // descriptor_loop_length = 0
    buf.extend_from_slice(&[0x00, 0x00]);

    backpatch_section_length(&mut buf);
    append_crc(&mut buf);

    buf
}

/// Emits a SCTE-35 `splice_null` section (command type `0x00`).
///
/// A `splice_null` is used as a heartbeat to keep the splice channel alive
/// without signalling an actual event.  It carries no command payload.
#[must_use]
pub fn emit_splice_null() -> Vec<u8> {
    let mut buf = Vec::with_capacity(20);
    write_section_header(&mut buf, 0x00, 0x000);

    // descriptor_loop_length = 0
    buf.extend_from_slice(&[0x00, 0x00]);

    backpatch_section_length(&mut buf);
    append_crc(&mut buf);

    buf
}

/// Configuration for a `splice_insert` command.
///
/// Represents an out-of-network (ad break start) or in-network (return to
/// programme) event signalled at a given PTS.
#[derive(Debug, Clone)]
pub struct SpliceInsertConfig {
    /// Unique 32-bit event identifier.
    pub event_id: u32,
    /// `true` = start of ad break (out of network); `false` = return to programme.
    pub out_of_network: bool,
    /// Duration of the break.  When `Some` the `duration_flag` is set.
    pub duration: Option<Duration>,
    /// `true` to auto-return to network after the break.
    pub auto_return: bool,
    /// Programme splice PTS.  `None` = immediate.
    pub splice_pts: Option<Duration>,
    /// Unique programme identifier (16 bits).
    pub unique_program_id: u16,
}

/// Emits a SCTE-35 `splice_insert` section (command type `0x05`).
///
/// # Arguments
///
/// - `cfg` — Insert configuration; see [`SpliceInsertConfig`].
///
/// # Example
///
/// ```ignore
/// use std::time::Duration;
/// use oximedia_container::mux::mpegts::scte35::{emit_splice_insert, SpliceInsertConfig};
///
/// let section = emit_splice_insert(&SpliceInsertConfig {
///     event_id: 42,
///     out_of_network: true,
///     duration: Some(Duration::from_secs(30)),
///     auto_return: true,
///     splice_pts: Some(Duration::from_secs(5)),
///     unique_program_id: 1,
/// });
/// assert_eq!(section[0], 0xFC);
/// ```
#[must_use]
pub fn emit_splice_insert(cfg: &SpliceInsertConfig) -> Vec<u8> {
    // Build command bytes first so we know the length.
    let mut cmd = Vec::with_capacity(16);

    // event_id (4 bytes)
    cmd.extend_from_slice(&cfg.event_id.to_be_bytes());

    // event_cancel_indicator = 0, reserved = 0x7F
    cmd.push(0x7F); // event_cancel=0 | reserved

    let immediate = cfg.splice_pts.is_none();
    let duration_flag = cfg.duration.is_some();
    let flags = (if cfg.out_of_network { 0x80 } else { 0x00 })
        | 0x40 // program_splice_flag = 1
        | (if duration_flag { 0x20 } else { 0x00 })
        | (if immediate { 0x10 } else { 0x00 });
    cmd.push(flags);

    // splice_time (only if !immediate)
    if !immediate {
        let splice_time = encode_splice_time(cfg.splice_pts);
        cmd.extend_from_slice(&splice_time);
    }

    // break_duration (only if duration_flag)
    if let Some(dur) = cfg.duration {
        let ticks = (dur.as_secs_f64() * 90_000.0) as u64 & 0x1_FFFF_FFFF;
        let auto_return_flag: u8 = if cfg.auto_return { 0x80 } else { 0x00 };
        cmd.push(auto_return_flag | 0x7E | ((ticks >> 32) as u8 & 0x01));
        cmd.extend_from_slice(&((ticks & 0xFFFF_FFFF) as u32).to_be_bytes());
    }

    // unique_program_id (2 bytes)
    cmd.extend_from_slice(&cfg.unique_program_id.to_be_bytes());

    // avail_num = 0, avails_expected = 1
    cmd.push(0x00);
    cmd.push(0x01);

    let command_len = cmd.len() as u16;

    let mut buf = Vec::with_capacity(32);
    write_section_header(&mut buf, 0x05, command_len);
    buf.extend_from_slice(&cmd);

    // descriptor_loop_length = 0
    buf.extend_from_slice(&[0x00, 0x00]);

    backpatch_section_length(&mut buf);
    append_crc(&mut buf);

    buf
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demux::mpegts::scte35::{Scte35Config, Scte35Parser};

    fn parse_section(bytes: &[u8]) -> crate::demux::mpegts::scte35::SpliceInfoSection {
        let mut parser = Scte35Parser::new(Scte35Config::default());
        parser.parse(bytes).expect("should parse emitted section")
    }

    // ── emit_splice_null ────────────────────────────────────────────────

    #[test]
    fn test_emit_null_table_id() {
        let bytes = emit_splice_null();
        assert_eq!(bytes[0], TABLE_ID, "table_id must be 0xFC");
    }

    #[test]
    fn test_emit_null_crc_valid() {
        let bytes = emit_splice_null();
        let section = parse_section(&bytes);
        assert_eq!(
            section.splice_command,
            crate::demux::mpegts::scte35::SpliceCommand::Null
        );
    }

    #[test]
    fn test_emit_null_minimum_length() {
        let bytes = emit_splice_null();
        // At minimum: 3 header bytes + 10 fixed bytes + 0 cmd bytes + 2 desc_loop + 4 CRC = 19
        assert!(
            bytes.len() >= 19,
            "section too short: {} bytes",
            bytes.len()
        );
    }

    // ── emit_time_signal ────────────────────────────────────────────────

    #[test]
    fn test_emit_time_signal_no_pts() {
        let bytes = emit_time_signal(None);
        let section = parse_section(&bytes);
        if let crate::demux::mpegts::scte35::SpliceCommand::TimeSignal(t) = section.splice_command {
            assert!(!t.time_specified, "time_specified_flag should be 0");
            assert!(t.pts_time.is_none());
        } else {
            panic!("expected TimeSignal, got {:?}", section.splice_command);
        }
    }

    #[test]
    fn test_emit_time_signal_with_pts_roundtrip() {
        // 5 seconds at 90 kHz = 450,000 ticks
        let pts = Duration::from_secs(5);
        let bytes = emit_time_signal(Some(pts));
        let section = parse_section(&bytes);

        if let crate::demux::mpegts::scte35::SpliceCommand::TimeSignal(t) = section.splice_command {
            assert!(t.time_specified, "time_specified_flag should be 1");
            let ticks = t.pts_time.expect("pts_time must be Some");
            assert_eq!(ticks, 450_000, "expected 450 000, got {ticks}");
        } else {
            panic!("expected TimeSignal, got {:?}", section.splice_command);
        }
    }

    #[test]
    fn test_emit_time_signal_pts_30s() {
        let bytes = emit_time_signal(Some(Duration::from_secs(30)));
        let section = parse_section(&bytes);
        if let crate::demux::mpegts::scte35::SpliceCommand::TimeSignal(t) = section.splice_command {
            let ticks = t.pts_time.expect("pts_time must be Some");
            assert_eq!(ticks, 2_700_000, "30 s × 90 000 Hz = 2 700 000 ticks");
        } else {
            panic!("expected TimeSignal");
        }
    }

    #[test]
    fn test_emit_time_signal_crc_field_matches() {
        let bytes = emit_time_signal(Some(Duration::from_millis(100)));
        // The last 4 bytes are the CRC stored in the section.
        let stored_crc = u32::from_be_bytes([
            bytes[bytes.len() - 4],
            bytes[bytes.len() - 3],
            bytes[bytes.len() - 2],
            bytes[bytes.len() - 1],
        ]);
        let computed = compute_crc32(&bytes[..bytes.len() - 4]);
        assert_eq!(stored_crc, computed, "stored CRC must match recomputed CRC");
    }

    // ── emit_splice_insert ──────────────────────────────────────────────

    #[test]
    fn test_emit_splice_insert_immediate_out_of_network() {
        let cfg = SpliceInsertConfig {
            event_id: 0xDEAD_BEEF,
            out_of_network: true,
            duration: None,
            auto_return: false,
            splice_pts: None, // immediate
            unique_program_id: 7,
        };
        let bytes = emit_splice_insert(&cfg);
        let section = parse_section(&bytes);

        if let crate::demux::mpegts::scte35::SpliceCommand::Insert(ins) = section.splice_command {
            assert_eq!(ins.event_id, 0xDEAD_BEEF);
            assert!(ins.out_of_network);
            assert!(ins.program_splice);
            assert!(ins.immediate);
            assert!(ins.duration.is_none());
            assert_eq!(ins.unique_program_id, 7);
        } else {
            panic!("expected Insert, got {:?}", section.splice_command);
        }
    }

    #[test]
    fn test_emit_splice_insert_with_duration_and_pts() {
        let cfg = SpliceInsertConfig {
            event_id: 1,
            out_of_network: true,
            duration: Some(Duration::from_secs(30)),
            auto_return: true,
            splice_pts: Some(Duration::from_secs(10)),
            unique_program_id: 100,
        };
        let bytes = emit_splice_insert(&cfg);
        let section = parse_section(&bytes);

        if let crate::demux::mpegts::scte35::SpliceCommand::Insert(ins) = section.splice_command {
            assert_eq!(ins.event_id, 1);
            assert!(ins.out_of_network);
            assert!(!ins.immediate);
            let dur = ins.duration.expect("duration must be present");
            assert!(dur.auto_return, "auto_return should be set");
            // 30 s × 90 000 Hz = 2 700 000 ticks
            assert_eq!(dur.duration, 2_700_000);
        } else {
            panic!("expected Insert, got {:?}", section.splice_command);
        }
    }

    #[test]
    fn test_emit_splice_insert_crc_valid() {
        let cfg = SpliceInsertConfig {
            event_id: 42,
            out_of_network: false,
            duration: None,
            auto_return: false,
            splice_pts: None,
            unique_program_id: 0,
        };
        let bytes = emit_splice_insert(&cfg);
        let stored = u32::from_be_bytes([
            bytes[bytes.len() - 4],
            bytes[bytes.len() - 3],
            bytes[bytes.len() - 2],
            bytes[bytes.len() - 1],
        ]);
        let computed = compute_crc32(&bytes[..bytes.len() - 4]);
        assert_eq!(stored, computed);
    }

    // ── compute_crc32 ───────────────────────────────────────────────────

    #[test]
    fn test_crc32_empty_input() {
        assert_eq!(compute_crc32(&[]), 0xFFFF_FFFF);
    }

    #[test]
    fn test_crc32_known_value() {
        // MPEG-2 CRC-32 of "123456789" = 0x0376E6E7
        assert_eq!(compute_crc32(b"123456789"), 0x0376_E6E7);
    }

    // ── encode_splice_time ──────────────────────────────────────────────

    #[test]
    fn test_encode_splice_time_none() {
        let bytes = encode_splice_time(None);
        assert_eq!(bytes[0], 0x7F, "time_specified_flag should be 0");
    }

    #[test]
    fn test_encode_splice_time_1s() {
        let bytes = encode_splice_time(Some(Duration::from_secs(1)));
        assert_eq!(bytes[0] & 0x80, 0x80, "time_specified_flag should be 1");
        // Reconstruct PTS from 5 bytes
        let pts = (u64::from(bytes[0] & 0x01) << 32)
            | (u64::from(bytes[1]) << 24)
            | (u64::from(bytes[2]) << 16)
            | (u64::from(bytes[3]) << 8)
            | u64::from(bytes[4]);
        assert_eq!(pts, 90_000, "1 s × 90 000 Hz = 90 000 ticks");
    }
}
