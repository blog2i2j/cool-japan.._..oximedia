//! SCTE-35 splice_info_section parser for MPEG-TS.
//!
//! Parses SCTE-35 (ANSI/SCTE 35) digital program insertion messages
//! used for ad insertion, content replacement, and other signalling
//! in broadcast and OTT streams.
//!
//! SCTE-35 is the ANSI/SCTE standard for signalling ad insertion cue points
//! in MPEG-TS streams.  A `splice_info_section` is carried on a dedicated PID
//! (conventional default: auto-detected from PMT, configurable via [`Scte35Config`]).
//!
//! ## Event Types
//!
//! | Command | Use Case |
//! |---------|----------|
//! | `SpliceNull` | Keep-alive with no action |
//! | `SpliceInsert` | Mark start/end of an ad break |
//! | `TimeSignal` | Carry a PTS timestamp for downstream processing |
//! | `BandwidthReservation` | Reserve bandwidth for future cue messages |
//! | `SpliceSchedule` | Pre-announce future splices by wall-clock time |
//! | `PrivateCommand` | Vendor-specific extensions |
//!
//! ## PID Convention
//!
//! By convention, SCTE-35 sections are discovered via the Program Map Table (PMT)
//! rather than a fixed PID.  [`Scte35Config::with_pid`] allows overriding the PID
//! per-stream when it is known in advance.
//!
//! ## CRC Validation
//!
//! All parsed sections are validated against the MPEG-2 CRC32 polynomial
//! (`0x04C11DB7`) when [`Scte35Config::without_crc_check`] is not applied.
//!
//! ## Example
//!
//! ```no_run
//! use oximedia_container::demux::mpegts::scte35::parse_splice_info_section;
//!
//! fn handle_packet(data: &[u8]) {
//!     match parse_splice_info_section(data) {
//!         Ok(event) => println!("SCTE-35: {:?}", event.splice_command),
//!         Err(e) => eprintln!("parse error: {e}"),
//!     }
//! }
//! ```
//!
//! # Supported splice commands
//!
//! - `splice_null` (0x00)
//! - `splice_schedule` (0x04) — header only
//! - `splice_insert` (0x05) — full parse
//! - `time_signal` (0x06) — full parse
//! - `bandwidth_reservation` (0x07)
//! - `private_command` (0xFF) — opaque payload

use oximedia_core::{OxiError, OxiResult};

/// Default PID for SCTE-35 data in MPEG-TS.
pub const SCTE35_DEFAULT_PID: u16 = 0x1FFF;

/// SCTE-35 table ID.
const SCTE35_TABLE_ID: u8 = 0xFC;

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for the SCTE-35 parser.
#[derive(Debug, Clone)]
pub struct Scte35Config {
    /// PID to monitor for SCTE-35 messages.
    /// Default: 0x1FFF (null PID, meaning auto-detect from PMT).
    pub pid: u16,
    /// Whether to validate CRC-32.
    pub verify_crc: bool,
}

impl Default for Scte35Config {
    fn default() -> Self {
        Self {
            pid: SCTE35_DEFAULT_PID,
            verify_crc: true,
        }
    }
}

impl Scte35Config {
    /// Creates a new config with a specific PID.
    #[must_use]
    pub const fn with_pid(mut self, pid: u16) -> Self {
        self.pid = pid;
        self
    }

    /// Disables CRC verification.
    #[must_use]
    pub const fn without_crc_check(mut self) -> Self {
        self.verify_crc = false;
        self
    }
}

// ─── Splice Time ────────────────────────────────────────────────────────────

/// A splice time value (33-bit PTS).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpliceTime {
    /// Whether a PTS time is specified.
    pub time_specified: bool,
    /// PTS value (33 bits) in 90 kHz ticks. Valid only if `time_specified` is true.
    pub pts_time: Option<u64>,
}

impl SpliceTime {
    /// Immediate splice (no time specified).
    pub const IMMEDIATE: Self = Self {
        time_specified: false,
        pts_time: None,
    };

    /// Creates a splice time with a specific PTS.
    #[must_use]
    pub fn at_pts(pts: u64) -> Self {
        Self {
            time_specified: true,
            pts_time: Some(pts),
        }
    }

    /// Returns the PTS time in seconds (90 kHz clock).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn seconds(&self) -> Option<f64> {
        self.pts_time.map(|pts| pts as f64 / 90000.0)
    }
}

// ─── Break Duration ─────────────────────────────────────────────────────────

/// Duration of a splice break.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BreakDuration {
    /// If true, return to network feed after break.
    pub auto_return: bool,
    /// Duration in 90 kHz ticks.
    pub duration: u64,
}

impl BreakDuration {
    /// Returns the duration in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn seconds(&self) -> f64 {
        self.duration as f64 / 90000.0
    }
}

// ─── Splice Commands ────────────────────────────────────────────────────────

/// Parsed SCTE-35 splice command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpliceCommand {
    /// Null command (heartbeat / keep-alive).
    Null,
    /// Splice schedule (planned events, header-level info only).
    Schedule {
        /// Number of events in the schedule.
        event_count: u8,
    },
    /// Splice insert — the primary ad insertion command.
    Insert(SpliceInsert),
    /// Time signal — signals a time for descriptor-based actions.
    TimeSignal(SpliceTime),
    /// Bandwidth reservation.
    BandwidthReservation,
    /// Private command.
    Private {
        /// Private command identifier (32 bits).
        identifier: u32,
        /// Opaque payload.
        data: Vec<u8>,
    },
    /// Unknown or unsupported command type.
    Unknown {
        /// Command type byte.
        command_type: u8,
        /// Raw command bytes.
        data: Vec<u8>,
    },
}

/// A splice_insert command with full detail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpliceInsert {
    /// Unique event ID.
    pub event_id: u32,
    /// If true, this cancels a previously signalled event.
    pub event_cancel: bool,
    /// If true, this is an out-of-network splice (start of break).
    pub out_of_network: bool,
    /// If true, this is a program-level splice; otherwise component-level.
    pub program_splice: bool,
    /// Duration of the break, if signalled.
    pub duration: Option<BreakDuration>,
    /// If true, the splice is immediate (no time specified).
    pub immediate: bool,
    /// Splice time, if not immediate.
    pub splice_time: Option<SpliceTime>,
    /// Unique program ID for the splice.
    pub unique_program_id: u16,
    /// Avail number (0-based index within a multi-break avail).
    pub avail_num: u8,
    /// Total number of avails.
    pub avails_expected: u8,
}

// ─── Descriptors ────────────────────────────────────────────────────────────

/// A SCTE-35 splice descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpliceDescriptor {
    /// Descriptor tag (e.g., 0x00 = avail_descriptor).
    pub tag: u8,
    /// Descriptor length (excluding tag and length fields).
    pub length: u8,
    /// CUEI identifier (should be 0x43554549 for "CUEI").
    pub identifier: u32,
    /// Raw descriptor data (after identifier).
    pub data: Vec<u8>,
}

// ─── Splice Info Section ────────────────────────────────────────────────────

/// A complete parsed SCTE-35 splice_info_section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpliceInfoSection {
    /// Protocol version (should be 0).
    pub protocol_version: u8,
    /// Whether the section is encrypted.
    pub encrypted: bool,
    /// Encryption algorithm (0 = not encrypted).
    pub encryption_algorithm: u8,
    /// PTS adjustment value (33 bits, 90 kHz).
    pub pts_adjustment: u64,
    /// Unique identifier for the content provider.
    pub cw_index: u8,
    /// Tier (12 bits): authorization tier.
    pub tier: u16,
    /// The parsed splice command.
    pub splice_command: SpliceCommand,
    /// Splice descriptors.
    pub descriptors: Vec<SpliceDescriptor>,
    /// CRC-32 from the section (for verification).
    pub crc32: u32,
}

// ─── Parser ─────────────────────────────────────────────────────────────────

/// SCTE-35 parser that parses `splice_info_section` from raw bytes.
#[derive(Debug, Clone)]
pub struct Scte35Parser {
    config: Scte35Config,
    /// Number of sections parsed successfully.
    parse_count: u64,
}

impl Scte35Parser {
    /// Creates a new parser with the given configuration.
    #[must_use]
    pub fn new(config: Scte35Config) -> Self {
        Self {
            config,
            parse_count: 0,
        }
    }

    /// Creates a parser with default configuration.
    #[must_use]
    pub fn default_parser() -> Self {
        Self::new(Scte35Config::default())
    }

    /// Returns the configured PID.
    #[must_use]
    pub const fn pid(&self) -> u16 {
        self.config.pid
    }

    /// Returns the number of sections parsed so far.
    #[must_use]
    pub const fn parse_count(&self) -> u64 {
        self.parse_count
    }

    /// Parses a SCTE-35 splice_info_section from raw bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the section is malformed, too short,
    /// or fails CRC verification (when enabled).
    pub fn parse(&mut self, data: &[u8]) -> OxiResult<SpliceInfoSection> {
        if data.len() < 15 {
            return Err(OxiError::InvalidData(
                "SCTE-35 section too short".to_string(),
            ));
        }

        // Table ID must be 0xFC
        if data[0] != SCTE35_TABLE_ID {
            return Err(OxiError::InvalidData(format!(
                "Invalid SCTE-35 table ID: expected 0xFC, got 0x{:02X}",
                data[0]
            )));
        }

        // Section syntax indicator (bit 7 of data[1]) should be 0
        let section_length = (((u16::from(data[1]) & 0x0F) << 8) | u16::from(data[2])) as usize;

        if data.len() < section_length + 3 {
            return Err(OxiError::InvalidData(format!(
                "SCTE-35 section length mismatch: declared {}, available {}",
                section_length + 3,
                data.len()
            )));
        }

        let section_data = &data[..section_length + 3];

        // CRC-32 verification
        if self.config.verify_crc && section_data.len() >= 4 {
            let crc_offset = section_data.len() - 4;
            let expected_crc = u32::from_be_bytes([
                section_data[crc_offset],
                section_data[crc_offset + 1],
                section_data[crc_offset + 2],
                section_data[crc_offset + 3],
            ]);
            let computed_crc = compute_crc32(&section_data[..crc_offset]);
            if computed_crc != expected_crc {
                return Err(OxiError::InvalidData(format!(
                    "SCTE-35 CRC mismatch: expected 0x{expected_crc:08X}, computed 0x{computed_crc:08X}"
                )));
            }
        }

        let protocol_version = data[3];
        let encrypted = (data[4] & 0x80) != 0;
        let encryption_algorithm = (data[4] >> 1) & 0x3F;

        // PTS adjustment: 33 bits starting at bit 0 of data[4] (1 bit) + data[5..9] (32 bits)
        let pts_adjustment = (u64::from(data[4] & 0x01) << 32)
            | (u64::from(data[5]) << 24)
            | (u64::from(data[6]) << 16)
            | (u64::from(data[7]) << 8)
            | u64::from(data[8]);

        let cw_index = data[9];

        // Tier: 12 bits from data[10..11]
        let tier = ((u16::from(data[10]) << 4) | (u16::from(data[11]) >> 4)) & 0x0FFF;

        // Splice command length: 12 bits from data[11..12]
        let splice_command_length =
            (((u16::from(data[11]) & 0x0F) << 8) | u16::from(data[12])) as usize;

        let splice_command_type = data[13];
        let cmd_start = 14;
        let cmd_end = if splice_command_length == 0xFFF {
            // Unknown length — use remaining data minus CRC and descriptors
            section_data.len().saturating_sub(4)
        } else {
            (cmd_start + splice_command_length).min(section_data.len().saturating_sub(4))
        };

        let cmd_data = if cmd_end > cmd_start && cmd_end <= section_data.len() {
            &section_data[cmd_start..cmd_end]
        } else {
            &[]
        };

        let splice_command = self.parse_command(splice_command_type, cmd_data)?;

        // Parse descriptors
        let desc_start = cmd_end;
        let descriptors = if desc_start + 2 <= section_data.len().saturating_sub(4) {
            let desc_loop_length =
                u16::from_be_bytes([section_data[desc_start], section_data[desc_start + 1]])
                    as usize;
            self.parse_descriptors(&section_data[desc_start + 2..], desc_loop_length)
        } else {
            Vec::new()
        };

        // CRC
        let crc_offset = section_data.len().saturating_sub(4);
        let crc32 = if crc_offset + 4 <= section_data.len() {
            u32::from_be_bytes([
                section_data[crc_offset],
                section_data[crc_offset + 1],
                section_data[crc_offset + 2],
                section_data[crc_offset + 3],
            ])
        } else {
            0
        };

        self.parse_count += 1;

        Ok(SpliceInfoSection {
            protocol_version,
            encrypted,
            encryption_algorithm,
            pts_adjustment,
            cw_index,
            tier,
            splice_command,
            descriptors,
            crc32,
        })
    }

    fn parse_command(&self, command_type: u8, data: &[u8]) -> OxiResult<SpliceCommand> {
        match command_type {
            0x00 => Ok(SpliceCommand::Null),
            0x04 => {
                let count = data.first().copied().unwrap_or_default();
                Ok(SpliceCommand::Schedule { event_count: count })
            }
            0x05 => self.parse_splice_insert(data),
            0x06 => {
                let time = self.parse_splice_time(data);
                Ok(SpliceCommand::TimeSignal(time))
            }
            0x07 => Ok(SpliceCommand::BandwidthReservation),
            0xFF => {
                let identifier = if data.len() >= 4 {
                    u32::from_be_bytes([data[0], data[1], data[2], data[3]])
                } else {
                    0
                };
                let payload = if data.len() > 4 {
                    data[4..].to_vec()
                } else {
                    Vec::new()
                };
                Ok(SpliceCommand::Private {
                    identifier,
                    data: payload,
                })
            }
            _ => Ok(SpliceCommand::Unknown {
                command_type,
                data: data.to_vec(),
            }),
        }
    }

    fn parse_splice_insert(&self, data: &[u8]) -> OxiResult<SpliceCommand> {
        if data.len() < 5 {
            return Err(OxiError::InvalidData("splice_insert too short".to_string()));
        }

        let event_id = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let event_cancel = (data[4] & 0x80) != 0;

        if event_cancel {
            return Ok(SpliceCommand::Insert(SpliceInsert {
                event_id,
                event_cancel: true,
                out_of_network: false,
                program_splice: false,
                duration: None,
                immediate: false,
                splice_time: None,
                unique_program_id: 0,
                avail_num: 0,
                avails_expected: 0,
            }));
        }

        if data.len() < 10 {
            return Err(OxiError::InvalidData(
                "splice_insert body too short".to_string(),
            ));
        }

        let out_of_network = (data[5] & 0x80) != 0;
        let program_splice = (data[5] & 0x40) != 0;
        let duration_flag = (data[5] & 0x20) != 0;
        let immediate = (data[5] & 0x10) != 0;

        let mut offset = 6;

        // Parse splice_time if program_splice && !immediate
        let splice_time = if program_splice && !immediate {
            if offset < data.len() {
                let time = self.parse_splice_time(&data[offset..]);
                let time_len = if time.time_specified { 5 } else { 1 };
                offset += time_len;
                Some(time)
            } else {
                None
            }
        } else {
            None
        };

        // Parse break_duration if duration_flag
        let duration = if duration_flag && offset + 5 <= data.len() {
            let auto_return = (data[offset] & 0x80) != 0;
            let dur = (u64::from(data[offset] & 0x01) << 32)
                | (u64::from(data[offset + 1]) << 24)
                | (u64::from(data[offset + 2]) << 16)
                | (u64::from(data[offset + 3]) << 8)
                | u64::from(data[offset + 4]);
            offset += 5;
            Some(BreakDuration {
                auto_return,
                duration: dur,
            })
        } else {
            None
        };

        // unique_program_id, avail_num, avails_expected
        let unique_program_id = if offset + 2 <= data.len() {
            let v = u16::from_be_bytes([data[offset], data[offset + 1]]);
            offset += 2;
            v
        } else {
            0
        };

        let avail_num = if offset < data.len() {
            let v = data[offset];
            offset += 1;
            v
        } else {
            0
        };

        let avails_expected = if offset < data.len() { data[offset] } else { 0 };

        Ok(SpliceCommand::Insert(SpliceInsert {
            event_id,
            event_cancel: false,
            out_of_network,
            program_splice,
            duration,
            immediate,
            splice_time,
            unique_program_id,
            avail_num,
            avails_expected,
        }))
    }

    fn parse_splice_time(&self, data: &[u8]) -> SpliceTime {
        if data.is_empty() {
            return SpliceTime::IMMEDIATE;
        }

        let time_specified = (data[0] & 0x80) != 0;
        if time_specified && data.len() >= 5 {
            let pts = (u64::from(data[0] & 0x01) << 32)
                | (u64::from(data[1]) << 24)
                | (u64::from(data[2]) << 16)
                | (u64::from(data[3]) << 8)
                | u64::from(data[4]);
            SpliceTime::at_pts(pts)
        } else {
            SpliceTime::IMMEDIATE
        }
    }

    fn parse_descriptors(&self, data: &[u8], loop_length: usize) -> Vec<SpliceDescriptor> {
        let mut descriptors = Vec::new();
        let mut offset = 0;
        let end = loop_length.min(data.len());

        while offset + 6 <= end {
            let tag = data[offset];
            let length = data[offset + 1];
            let total_len = 2 + length as usize;

            if offset + total_len > end {
                break;
            }

            let identifier = if length >= 4 {
                u32::from_be_bytes([
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                ])
            } else {
                0
            };

            let desc_data = if length > 4 {
                data[offset + 6..offset + total_len].to_vec()
            } else {
                Vec::new()
            };

            descriptors.push(SpliceDescriptor {
                tag,
                length,
                identifier,
                data: desc_data,
            });

            offset += total_len;
        }

        descriptors
    }
}

// ─── CRC-32 ─────────────────────────────────────────────────────────────────

/// MPEG-2 CRC-32 polynomial.
const CRC32_POLY: u32 = 0x04C1_1DB7;

/// Computes the MPEG-2 CRC-32 for a byte slice.
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

// ─── Convenience free function ───────────────────────────────────────────────

/// Parses a raw SCTE-35 `splice_info_section` byte slice using the default parser
/// configuration (CRC verification enabled, auto-detect PID).
///
/// This is a convenience wrapper around [`Scte35Parser::default_parser`] followed by
/// [`Scte35Parser::parse`].
///
/// # Errors
///
/// Returns an error if the section is malformed, too short, or fails CRC verification.
pub fn parse_splice_info_section(data: &[u8]) -> OxiResult<SpliceInfoSection> {
    Scte35Parser::default_parser().parse(data)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_minimal_scte35_section(command_type: u8, command_data: &[u8]) -> Vec<u8> {
        let mut section = Vec::new();

        // Table ID
        section.push(SCTE35_TABLE_ID);

        // Section length placeholder (will be filled)
        section.push(0x00);
        section.push(0x00);

        // protocol_version
        section.push(0x00);

        // encrypted_packet(1) | encryption_algorithm(6) | pts_adjustment(33)
        section.extend_from_slice(&[0x00, 0x00, 0x00, 0x00, 0x00]); // no encryption, pts_adj=0

        // cw_index
        section.push(0x00);

        // tier(12) + splice_command_length(12)
        let cmd_len = command_data.len() as u16;
        let tier: u16 = 0x0FFF; // all tiers
        section.push(((tier >> 4) & 0xFF) as u8);
        section.push((((tier & 0x0F) << 4) | ((cmd_len >> 8) & 0x0F)) as u8);
        section.push((cmd_len & 0xFF) as u8);

        // splice_command_type
        section.push(command_type);

        // command data
        section.extend_from_slice(command_data);

        // descriptor_loop_length = 0
        section.extend_from_slice(&[0x00, 0x00]);

        // Fix section_length (everything after byte 2, minus 3 header bytes, plus 4 CRC)
        let section_length = section.len() - 3 + 4;
        section[1] = ((section_length >> 8) & 0x0F) as u8;
        section[2] = (section_length & 0xFF) as u8;

        // CRC-32
        let crc = compute_crc32(&section);
        section.extend_from_slice(&crc.to_be_bytes());

        section
    }

    #[test]
    fn test_scte35_config_default() {
        let cfg = Scte35Config::default();
        assert_eq!(cfg.pid, SCTE35_DEFAULT_PID);
        assert!(cfg.verify_crc);
    }

    #[test]
    fn test_scte35_config_with_pid() {
        let cfg = Scte35Config::default().with_pid(0x100);
        assert_eq!(cfg.pid, 0x100);
    }

    #[test]
    fn test_scte35_config_without_crc() {
        let cfg = Scte35Config::default().without_crc_check();
        assert!(!cfg.verify_crc);
    }

    #[test]
    fn test_splice_time_immediate() {
        let t = SpliceTime::IMMEDIATE;
        assert!(!t.time_specified);
        assert!(t.pts_time.is_none());
        assert!(t.seconds().is_none());
    }

    #[test]
    fn test_splice_time_at_pts() {
        let t = SpliceTime::at_pts(90000);
        assert!(t.time_specified);
        assert_eq!(t.pts_time, Some(90000));
        let secs = t.seconds().expect("should have seconds");
        assert!((secs - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_break_duration_seconds() {
        let bd = BreakDuration {
            auto_return: true,
            duration: 2_700_000, // 30 seconds at 90kHz
        };
        assert!((bd.seconds() - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_null_command() {
        let data = build_minimal_scte35_section(0x00, &[]);
        let mut parser = Scte35Parser::default_parser();
        let section = parser.parse(&data).expect("should parse");
        assert_eq!(section.splice_command, SpliceCommand::Null);
        assert_eq!(section.protocol_version, 0);
        assert!(!section.encrypted);
        assert_eq!(parser.parse_count(), 1);
    }

    #[test]
    fn test_parse_bandwidth_reservation() {
        let data = build_minimal_scte35_section(0x07, &[]);
        let mut parser = Scte35Parser::default_parser();
        let section = parser.parse(&data).expect("should parse");
        assert_eq!(section.splice_command, SpliceCommand::BandwidthReservation);
    }

    #[test]
    fn test_parse_time_signal() {
        // Time signal with PTS = 90000 (1 second)
        let cmd = [0x80 | 0x00, 0x00, 0x01, 0x5F, 0x90]; // time_specified=1, pts=90000
        let data = build_minimal_scte35_section(0x06, &cmd);
        let mut parser = Scte35Parser::default_parser();
        let section = parser.parse(&data).expect("should parse");
        if let SpliceCommand::TimeSignal(time) = &section.splice_command {
            assert!(time.time_specified);
            assert_eq!(time.pts_time, Some(90000));
        } else {
            panic!("Expected TimeSignal");
        }
    }

    #[test]
    fn test_parse_splice_insert_cancel() {
        // splice_insert with cancel flag
        let mut cmd = Vec::new();
        cmd.extend_from_slice(&0x12345678u32.to_be_bytes()); // event_id
        cmd.push(0x80); // event_cancel=1
        let data = build_minimal_scte35_section(0x05, &cmd);
        let mut parser = Scte35Parser::default_parser();
        let section = parser.parse(&data).expect("should parse");
        if let SpliceCommand::Insert(insert) = &section.splice_command {
            assert_eq!(insert.event_id, 0x12345678);
            assert!(insert.event_cancel);
        } else {
            panic!("Expected Insert");
        }
    }

    #[test]
    fn test_parse_splice_insert_immediate() {
        let mut cmd = Vec::new();
        cmd.extend_from_slice(&1u32.to_be_bytes()); // event_id
        cmd.push(0x00); // not cancelled
        cmd.push(0xD0); // out_of_network=1, program_splice=1, duration=0, immediate=1
                        // unique_program_id, avail_num, avails_expected
        cmd.extend_from_slice(&100u16.to_be_bytes());
        cmd.push(0);
        cmd.push(1);
        let data = build_minimal_scte35_section(0x05, &cmd);
        let mut parser = Scte35Parser::default_parser();
        let section = parser.parse(&data).expect("should parse");
        if let SpliceCommand::Insert(insert) = &section.splice_command {
            assert_eq!(insert.event_id, 1);
            assert!(!insert.event_cancel);
            assert!(insert.out_of_network);
            assert!(insert.program_splice);
            assert!(insert.immediate);
            assert!(insert.splice_time.is_none());
            assert_eq!(insert.unique_program_id, 100);
            assert_eq!(insert.avails_expected, 1);
        } else {
            panic!("Expected Insert");
        }
    }

    #[test]
    fn test_parse_private_command() {
        let mut cmd = Vec::new();
        cmd.extend_from_slice(b"CUEI"); // identifier
        cmd.extend_from_slice(&[0x01, 0x02, 0x03]); // private data
        let data = build_minimal_scte35_section(0xFF, &cmd);
        let mut parser = Scte35Parser::default_parser();
        let section = parser.parse(&data).expect("should parse");
        if let SpliceCommand::Private { identifier, data } = &section.splice_command {
            assert_eq!(*identifier, 0x43554549); // "CUEI"
            assert_eq!(data, &[0x01, 0x02, 0x03]);
        } else {
            panic!("Expected Private");
        }
    }

    #[test]
    fn test_parse_unknown_command() {
        let data = build_minimal_scte35_section(0x42, &[0xAA, 0xBB]);
        let mut parser = Scte35Parser::default_parser();
        let section = parser.parse(&data).expect("should parse");
        if let SpliceCommand::Unknown { command_type, data } = &section.splice_command {
            assert_eq!(*command_type, 0x42);
            assert_eq!(data, &[0xAA, 0xBB]);
        } else {
            panic!("Expected Unknown");
        }
    }

    #[test]
    fn test_parse_too_short() {
        let mut parser = Scte35Parser::default_parser();
        let result = parser.parse(&[0xFC, 0x00]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_wrong_table_id() {
        let mut data = build_minimal_scte35_section(0x00, &[]);
        data[0] = 0x00; // wrong table ID
        let mut parser = Scte35Parser::default_parser();
        let result = parser.parse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_crc_failure() {
        let mut data = build_minimal_scte35_section(0x00, &[]);
        // Corrupt the CRC
        let len = data.len();
        data[len - 1] ^= 0xFF;
        let mut parser = Scte35Parser::new(Scte35Config::default());
        let result = parser.parse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_no_crc_check() {
        let mut data = build_minimal_scte35_section(0x00, &[]);
        let len = data.len();
        data[len - 1] ^= 0xFF; // corrupt CRC
        let mut parser = Scte35Parser::new(Scte35Config::default().without_crc_check());
        let result = parser.parse(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compute_crc32_empty() {
        let crc = compute_crc32(&[]);
        assert_eq!(crc, 0xFFFF_FFFF);
    }

    #[test]
    fn test_compute_crc32_known() {
        // CRC of "123456789" with MPEG-2 polynomial
        let crc = compute_crc32(b"123456789");
        assert_eq!(crc, 0x0376_E6E7);
    }

    #[test]
    fn test_tier_parsing() {
        let data = build_minimal_scte35_section(0x00, &[]);
        let mut parser = Scte35Parser::default_parser();
        let section = parser.parse(&data).expect("should parse");
        assert_eq!(section.tier, 0x0FFF);
    }

    #[test]
    fn test_schedule_command() {
        let data = build_minimal_scte35_section(0x04, &[3]); // 3 events
        let mut parser = Scte35Parser::default_parser();
        let section = parser.parse(&data).expect("should parse");
        if let SpliceCommand::Schedule { event_count } = &section.splice_command {
            assert_eq!(*event_count, 3);
        } else {
            panic!("Expected Schedule");
        }
    }

    #[test]
    fn test_parser_count_increments() {
        let mut parser = Scte35Parser::default_parser();
        assert_eq!(parser.parse_count(), 0);

        let data = build_minimal_scte35_section(0x00, &[]);
        let _ = parser.parse(&data);
        assert_eq!(parser.parse_count(), 1);
        let _ = parser.parse(&data);
        assert_eq!(parser.parse_count(), 2);
    }

    #[test]
    fn test_splice_descriptor_parsing() {
        // Build a section with a descriptor
        let mut section = Vec::new();
        section.push(SCTE35_TABLE_ID);
        section.push(0x00);
        section.push(0x00);
        section.push(0x00); // protocol_version
        section.extend_from_slice(&[0x00; 5]); // encrypted + pts_adj
        section.push(0x00); // cw_index
        let tier: u16 = 0x0FFF;
        let cmd_len: u16 = 0;
        section.push(((tier >> 4) & 0xFF) as u8);
        section.push((((tier & 0x0F) << 4) | ((cmd_len >> 8) & 0x0F)) as u8);
        section.push((cmd_len & 0xFF) as u8);
        section.push(0x00); // null command

        // Descriptor loop: one descriptor
        let desc_data = [0x00, 0x08, 0x43, 0x55, 0x45, 0x49, 0xAA, 0xBB, 0xCC, 0xDD];
        let desc_len = desc_data.len() as u16;
        section.extend_from_slice(&desc_len.to_be_bytes());
        section.extend_from_slice(&desc_data);

        // Fix section length
        let section_length = section.len() - 3 + 4;
        section[1] = ((section_length >> 8) & 0x0F) as u8;
        section[2] = (section_length & 0xFF) as u8;

        // CRC
        let crc = compute_crc32(&section);
        section.extend_from_slice(&crc.to_be_bytes());

        let mut parser = Scte35Parser::default_parser();
        let result = parser.parse(&section).expect("should parse");
        assert_eq!(result.descriptors.len(), 1);
        assert_eq!(result.descriptors[0].tag, 0x00);
        assert_eq!(result.descriptors[0].identifier, 0x43554549); // "CUEI"
        assert_eq!(result.descriptors[0].data, vec![0xAA, 0xBB, 0xCC, 0xDD]);
    }
}
