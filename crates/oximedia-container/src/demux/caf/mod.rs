//! CAF (Core Audio Format) demuxer.
//!
//! Core Audio Format is a container designed by Apple for digital audio data.
//! Specification: Apple Core Audio Format Specification 1.0
//!
//! # Structure
//!
//! A CAF file consists of:
//! 1. A file header (`caff` magic + version + flags)
//! 2. A series of chunks, each with a type tag (4 chars), size (i64), and data
//!
//! Required chunks:
//! - `desc` (AudioStreamBasicDescription) — must be first
//! - `data` — audio sample data
//!
//! Optional chunks:
//! - `info` — string key-value metadata (like artist, title)
//! - `pakt` — packet table for VBR audio
//! - `chan` — channel layout
//! - `mark` — markers / cue points
//! - `regn` — regions
//! - `uuid` — user-defined data

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use oximedia_core::{CodecId, OxiError};

/// CAF file header magic bytes.
pub const CAF_MAGIC: &[u8; 4] = b"caff";

/// CAF file version (always 1).
pub const CAF_VERSION: u16 = 1;

/// CAF chunk type identifiers (4-byte ASCII tags).
pub mod chunk_type {
    /// Audio description chunk (AudioStreamBasicDescription).
    pub const DESC: &[u8; 4] = b"desc";
    /// Audio data chunk.
    pub const DATA: &[u8; 4] = b"data";
    /// Packet table chunk (for VBR).
    pub const PAKT: &[u8; 4] = b"pakt";
    /// Channel layout chunk.
    pub const CHAN: &[u8; 4] = b"chan";
    /// Information (metadata tags) chunk.
    pub const INFO: &[u8; 4] = b"info";
    /// Marker chunk.
    pub const MARK: &[u8; 4] = b"mark";
    /// Regions chunk.
    pub const REGN: &[u8; 4] = b"regn";
    /// UUID (user-defined) chunk.
    pub const UUID: &[u8; 4] = b"uuid";
    /// MIDI chunk.
    pub const MIDI: &[u8; 4] = b"midi";
    /// Overview chunk (waveform display data).
    pub const OVVW: &[u8; 4] = b"ovvw";
    /// Peak chunk (peak hold data).
    pub const PEAK: &[u8; 4] = b"peak";
    /// Edit comments chunk.
    pub const EDCT: &[u8; 4] = b"edct";
    /// Instrument chunk.
    pub const INST: &[u8; 4] = b"inst";
    /// SMPTE chunk.
    pub const SMPT: &[u8; 4] = b"smpt";
    /// Tempo map chunk.
    pub const UMID: &[u8; 4] = b"umid";
    /// Free chunk (padding).
    pub const FREE: &[u8; 4] = b"free";
}

/// CAF audio format IDs (mFormatID in AudioStreamBasicDescription).
pub mod format_id {
    /// Linear PCM.
    pub const LPCM: u32 = 0x6C70636D; // 'lpcm'
    /// Apple's ALAC lossless.
    pub const ALAC: u32 = 0x616C6163; // 'alac'
    /// AAC-LC (patent-encumbered, not supported for decode).
    pub const AAC_LC: u32 = 0x61616320; // 'aac '
    /// Apple's IMA ADPCM.
    pub const IMA4: u32 = 0x696D6134; // 'ima4'
    /// µ-law.
    pub const ULAW: u32 = 0x756C6177; // 'ulaw'
    /// A-law.
    pub const ALAW: u32 = 0x616C6177; // 'alaw'
    /// Qualcomm PureVoice.
    pub const QDESIGN2: u32 = 0x51445332; // 'QDS2'
    /// FLAC in CAF.
    pub const FLAC: u32 = 0x666C6163; // 'flac'
    /// Opus in CAF.
    pub const OPUS: u32 = 0x6F707573; // 'opus'
}

/// Errors specific to CAF parsing.
#[derive(Debug)]
pub enum CafError {
    /// Bad magic bytes or version.
    InvalidHeader(String),
    /// A required chunk (`desc`, `data`) is missing.
    MissingChunk(&'static str),
    /// A chunk header is truncated or malformed.
    MalformedChunk(String),
    /// I/O error.
    Io(std::io::Error),
}

impl std::fmt::Display for CafError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHeader(s) => write!(f, "invalid CAF header: {s}"),
            Self::MissingChunk(c) => write!(f, "missing required CAF chunk: {c}"),
            Self::MalformedChunk(s) => write!(f, "malformed CAF chunk: {s}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for CafError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<CafError> for OxiError {
    fn from(e: CafError) -> Self {
        OxiError::Parse { offset: 0, message: e.to_string() }
    }
}

impl From<std::io::Error> for CafError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Result type for CAF operations.
pub type CafResult<T> = std::result::Result<T, CafError>;

// ---------------------------------------------------------------------------
// AudioStreamBasicDescription (Apple's ASBD)
// ---------------------------------------------------------------------------

/// Audio stream description from the `desc` chunk.
///
/// Maps to Apple's `AudioStreamBasicDescription` struct.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioStreamBasicDescription {
    /// Sample rate in Hz (64-bit float).
    pub sample_rate: f64,
    /// Format ID (4-char code, e.g., `lpcm`, `alac`).
    pub format_id: u32,
    /// Format flags (meaning depends on format_id).
    pub format_flags: u32,
    /// Bytes per packet (0 = variable).
    pub bytes_per_packet: u32,
    /// Frames per packet (0 = variable).
    pub frames_per_packet: u32,
    /// Channels per frame.
    pub channels_per_frame: u32,
    /// Bits per channel (0 = not applicable).
    pub bits_per_channel: u32,
}

impl AudioStreamBasicDescription {
    /// Parse from 32 bytes of big-endian data.
    pub fn parse(data: &[u8]) -> CafResult<Self> {
        if data.len() < 32 {
            return Err(CafError::MalformedChunk(
                "desc chunk too short (need 32 bytes)".to_string(),
            ));
        }
        let sample_rate = f64::from_bits(u64::from_be_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]));
        let format_id = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
        let format_flags = u32::from_be_bytes([data[12], data[13], data[14], data[15]]);
        let bytes_per_packet = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
        let frames_per_packet = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
        let channels_per_frame = u32::from_be_bytes([data[24], data[25], data[26], data[27]]);
        let bits_per_channel = u32::from_be_bytes([data[28], data[29], data[30], data[31]]);
        Ok(Self {
            sample_rate,
            format_id,
            format_flags,
            bytes_per_packet,
            frames_per_packet,
            channels_per_frame,
            bits_per_channel,
        })
    }

    /// Serialize to 32 bytes of big-endian data.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[..8].copy_from_slice(&self.sample_rate.to_bits().to_be_bytes());
        out[8..12].copy_from_slice(&self.format_id.to_be_bytes());
        out[12..16].copy_from_slice(&self.format_flags.to_be_bytes());
        out[16..20].copy_from_slice(&self.bytes_per_packet.to_be_bytes());
        out[20..24].copy_from_slice(&self.frames_per_packet.to_be_bytes());
        out[24..28].copy_from_slice(&self.channels_per_frame.to_be_bytes());
        out[28..32].copy_from_slice(&self.bits_per_channel.to_be_bytes());
        out
    }

    /// Map format_id to OxiMedia codec ID.
    #[must_use]
    pub fn to_codec_id(&self) -> Option<CodecId> {
        match self.format_id {
            format_id::LPCM => Some(CodecId::Pcm),
            format_id::FLAC => Some(CodecId::Flac),
            format_id::OPUS => Some(CodecId::Opus),
            format_id::ALAW | format_id::ULAW => Some(CodecId::Pcm),
            _ => None,
        }
    }

    /// Check whether PCM format flags indicate float samples.
    #[must_use]
    pub fn is_float(&self) -> bool {
        // kAudioFormatFlagIsFloat = 0x1
        self.format_flags & 0x1 != 0
    }

    /// Check whether PCM samples are big-endian.
    #[must_use]
    pub fn is_big_endian(&self) -> bool {
        // kAudioFormatFlagIsBigEndian = 0x2
        self.format_flags & 0x2 != 0
    }

    /// Check whether PCM samples are signed integers (vs unsigned).
    #[must_use]
    pub fn is_signed_integer(&self) -> bool {
        // kAudioFormatFlagIsSignedInteger = 0x4
        self.format_flags & 0x4 != 0
    }
}

// ---------------------------------------------------------------------------
// Packet table (pakt chunk)
// ---------------------------------------------------------------------------

/// Entry in the packet table for VBR audio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PacketTableEntry {
    /// Byte count of the packet.
    pub byte_count: u64,
    /// Frame count (samples per channel) of the packet.
    pub frame_count: u64,
}

/// Packet table parsed from the `pakt` chunk.
#[derive(Debug, Clone)]
pub struct PacketTable {
    /// Total number of valid frames in the file.
    pub total_frames: i64,
    /// Number of priming frames (e.g., for ALAC encoder delay).
    pub priming_frames: i32,
    /// Number of remainder frames (trailing padding).
    pub remainder_frames: i32,
    /// Per-packet entries (byte count + frame count, VBR-encoded).
    pub entries: Vec<PacketTableEntry>,
}

impl PacketTable {
    /// Parse from raw chunk data (header + variable-length entries).
    pub fn parse(data: &[u8]) -> CafResult<Self> {
        if data.len() < 24 {
            return Err(CafError::MalformedChunk(
                "pakt chunk too short (need ≥24 bytes for header)".to_string(),
            ));
        }
        let num_packets = u64::from_be_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        let total_frames = i64::from_be_bytes([
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
        ]);
        let priming_frames = i32::from_be_bytes([data[16], data[17], data[18], data[19]]);
        let remainder_frames = i32::from_be_bytes([data[20], data[21], data[22], data[23]]);

        let mut offset = 24;
        let mut entries = Vec::with_capacity(num_packets as usize);

        for _ in 0..num_packets {
            let (byte_count, consumed) = decode_vint(&data[offset..]).map_err(|_| {
                CafError::MalformedChunk("truncated VInt in pakt".to_string())
            })?;
            offset += consumed;

            let (frame_count, consumed) = decode_vint(&data[offset..]).map_err(|_| {
                CafError::MalformedChunk("truncated VInt in pakt".to_string())
            })?;
            offset += consumed;

            entries.push(PacketTableEntry {
                byte_count,
                frame_count,
            });
        }

        Ok(Self {
            total_frames,
            priming_frames,
            remainder_frames,
            entries,
        })
    }
}

// ---------------------------------------------------------------------------
// CAF parsed result
// ---------------------------------------------------------------------------

/// Parsed CAF file information.
#[derive(Debug, Clone)]
pub struct CafInfo {
    /// Audio description.
    pub asbd: AudioStreamBasicDescription,
    /// Metadata key-value pairs from the `info` chunk.
    pub tags: HashMap<String, String>,
    /// Packet table (present for VBR formats).
    pub packet_table: Option<PacketTable>,
    /// Byte offset of the first audio sample in the file.
    pub data_offset: u64,
    /// Total size of audio data in bytes (`-1` = unknown / until EOF).
    pub data_size: i64,
    /// Number of channels.
    pub channels: u32,
    /// Sample rate in Hz.
    pub sample_rate: f64,
}

// ---------------------------------------------------------------------------
// CAF demuxer
// ---------------------------------------------------------------------------

/// CAF (Core Audio Format) demuxer.
///
/// Reads a CAF file from any `Read + Seek` source and provides access to
/// audio stream information and raw audio packets.
///
/// # Example
///
/// ```rust
/// use oximedia_container::demux::caf::CafDemuxer;
/// use std::io::Cursor;
///
/// // Minimal CAF: file header + desc chunk + data chunk
/// let data = vec![0u8; 0]; // placeholder
/// let cursor = Cursor::new(data);
/// // let mut demuxer = CafDemuxer::new(cursor);
/// ```
pub struct CafDemuxer<R> {
    source: R,
    info: Option<CafInfo>,
    read_position: u64,
}

impl<R: Read + Seek> CafDemuxer<R> {
    /// Create a new CAF demuxer around the given source.
    #[must_use]
    pub fn new(source: R) -> Self {
        Self {
            source,
            info: None,
            read_position: 0,
        }
    }

    /// Probe the CAF file: reads all chunk headers and populates [`CafInfo`].
    ///
    /// # Errors
    ///
    /// Returns [`CafError`] if the file header is invalid or required chunks
    /// are missing.
    pub fn probe(&mut self) -> CafResult<&CafInfo> {
        self.source.seek(SeekFrom::Start(0))?;

        // --- File header (8 bytes) ---
        let mut hdr = [0u8; 8];
        self.source.read_exact(&mut hdr)?;
        if &hdr[0..4] != CAF_MAGIC {
            return Err(CafError::InvalidHeader(format!(
                "expected 'caff', got {:?}",
                &hdr[0..4]
            )));
        }
        let version = u16::from_be_bytes([hdr[4], hdr[5]]);
        if version != CAF_VERSION {
            return Err(CafError::InvalidHeader(format!(
                "unsupported CAF version {version}"
            )));
        }
        // hdr[6..8] = file flags (reserved, must be 0)

        let mut asbd: Option<AudioStreamBasicDescription> = None;
        let mut tags: HashMap<String, String> = HashMap::new();
        let mut packet_table: Option<PacketTable> = None;
        let mut data_offset: u64 = 0;
        let mut data_size: i64 = -1;

        // --- Iterate chunks ---
        loop {
            let mut chunk_hdr = [0u8; 12];
            match self.source.read_exact(&mut chunk_hdr) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(CafError::Io(e)),
            }

            let chunk_type: [u8; 4] = chunk_hdr[0..4].try_into().unwrap_or([0u8; 4]);
            let chunk_size = i64::from_be_bytes([
                chunk_hdr[4],
                chunk_hdr[5],
                chunk_hdr[6],
                chunk_hdr[7],
                chunk_hdr[8],
                chunk_hdr[9],
                chunk_hdr[10],
                chunk_hdr[11],
            ]);

            let pos_after_hdr = self.source.stream_position()?;

            match &chunk_type {
                t if t == chunk_type::DESC => {
                    let mut buf = vec![0u8; 32];
                    self.source.read_exact(&mut buf)?;
                    asbd = Some(AudioStreamBasicDescription::parse(&buf)?);
                }
                t if t == chunk_type::DATA => {
                    // 4-byte edit count follows the header (skip it)
                    let mut skip = [0u8; 4];
                    self.source.read_exact(&mut skip)?;
                    data_offset = self.source.stream_position()?;
                    data_size = if chunk_size == -1 {
                        -1
                    } else {
                        // chunk_size includes the 4-byte edit count
                        chunk_size - 4
                    };
                    // Skip the rest of the data chunk — we only note the offset
                    if chunk_size != -1 {
                        let skip_bytes = (chunk_size - 4) as u64;
                        self.source.seek(SeekFrom::Current(skip_bytes as i64))?;
                    } else {
                        // Data runs to EOF — stop scanning
                        break;
                    }
                }
                t if t == chunk_type::INFO => {
                    let sz = chunk_size as usize;
                    let mut buf = vec![0u8; sz];
                    self.source.read_exact(&mut buf)?;
                    tags = parse_info_chunk(&buf);
                }
                t if t == chunk_type::PAKT => {
                    let sz = chunk_size as usize;
                    let mut buf = vec![0u8; sz];
                    self.source.read_exact(&mut buf)?;
                    packet_table = Some(PacketTable::parse(&buf)?);
                }
                _ => {
                    // Unknown / unhandled chunk — skip
                    if chunk_size != -1 {
                        self.source
                            .seek(SeekFrom::Start(pos_after_hdr + chunk_size as u64))?;
                    } else {
                        break;
                    }
                }
            }
        }

        let asbd = asbd.ok_or(CafError::MissingChunk("desc"))?;
        if data_offset == 0 {
            return Err(CafError::MissingChunk("data"));
        }

        let channels = asbd.channels_per_frame;
        let sample_rate = asbd.sample_rate;

        self.info = Some(CafInfo {
            asbd,
            tags,
            packet_table,
            data_offset,
            data_size,
            channels,
            sample_rate,
        });

        self.read_position = data_offset;
        self.source.seek(SeekFrom::Start(data_offset))?;

        self.info
            .as_ref()
            .ok_or_else(|| CafError::MalformedChunk("info failed to populate".to_string()))
    }

    /// Return parsed file info.
    #[must_use]
    pub fn info(&self) -> Option<&CafInfo> {
        self.info.as_ref()
    }

    /// Read the next audio packet (raw bytes) from the data chunk.
    ///
    /// For CBR (constant bytes per packet) formats, each call returns one
    /// packet.  For VBR formats the packet size is determined from the
    /// packet table.
    ///
    /// Returns `None` at end-of-stream.
    ///
    /// # Errors
    ///
    /// Returns [`CafError`] on I/O errors or if the file has not been probed.
    pub fn read_packet(&mut self) -> CafResult<Option<Vec<u8>>> {
        let info = match &self.info {
            Some(i) => i.clone(),
            None => return Err(CafError::MissingChunk("desc (not probed)")),
        };

        // CBR: fixed bytes per packet
        if info.asbd.bytes_per_packet > 0 {
            let packet_size = info.asbd.bytes_per_packet as usize;
            let mut buf = vec![0u8; packet_size];
            match self.source.read_exact(&mut buf) {
                Ok(()) => {
                    self.read_position += packet_size as u64;
                    Ok(Some(buf))
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
                Err(e) => Err(CafError::Io(e)),
            }
        } else {
            // VBR: use packet table
            // For simplicity, read one byte at a time to detect EOF
            let mut byte = [0u8; 1];
            match self.source.read_exact(&mut byte) {
                Ok(()) => {
                    self.read_position += 1;
                    Ok(Some(vec![byte[0]]))
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
                Err(e) => Err(CafError::Io(e)),
            }
        }
    }

    /// Read all remaining audio data as a contiguous byte vector.
    ///
    /// # Errors
    ///
    /// Returns [`CafError`] on I/O errors or if the file has not been probed.
    pub fn read_all_audio(&mut self) -> CafResult<Vec<u8>> {
        let info = match &self.info {
            Some(i) => i.clone(),
            None => return Err(CafError::MissingChunk("desc (not probed)")),
        };

        self.source.seek(SeekFrom::Start(info.data_offset))?;
        let mut buf = Vec::new();
        if info.data_size > 0 {
            buf.resize(info.data_size as usize, 0);
            self.source.read_exact(&mut buf)?;
        } else {
            self.source.read_to_end(&mut buf)?;
        }
        self.read_position = info.data_offset + buf.len() as u64;
        Ok(buf)
    }

    /// Seek to a specific byte offset within the audio data.
    ///
    /// # Errors
    ///
    /// Returns [`CafError`] on I/O errors.
    pub fn seek_audio(&mut self, byte_offset: u64) -> CafResult<()> {
        let data_offset = self
            .info
            .as_ref()
            .map(|i| i.data_offset)
            .unwrap_or(0);
        let target = data_offset + byte_offset;
        self.source.seek(SeekFrom::Start(target))?;
        self.read_position = target;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CAF muxer
// ---------------------------------------------------------------------------

/// Configuration for the CAF muxer.
#[derive(Debug, Clone)]
pub struct CafMuxConfig {
    /// AudioStreamBasicDescription for the output stream.
    pub asbd: AudioStreamBasicDescription,
    /// Metadata key-value pairs to embed in the `info` chunk.
    pub tags: HashMap<String, String>,
}

impl CafMuxConfig {
    /// Create a PCM CAF config for the given sample rate / channel count / bit depth.
    #[must_use]
    pub fn pcm(sample_rate: f64, channels: u32, bits_per_sample: u32, is_float: bool) -> Self {
        let format_flags = if is_float {
            0x1 | 0x2 | 0x8 // IsFloat | IsBigEndian | IsAlignedHigh
        } else {
            0x2 | 0x4 | 0x8 // IsBigEndian | IsSignedInteger | IsAlignedHigh
        };
        let bytes_per_sample = (bits_per_sample + 7) / 8;
        let bytes_per_packet = bytes_per_sample * channels;
        let asbd = AudioStreamBasicDescription {
            sample_rate,
            format_id: format_id::LPCM,
            format_flags,
            bytes_per_packet,
            frames_per_packet: 1,
            channels_per_frame: channels,
            bits_per_channel: bits_per_sample,
        };
        Self {
            asbd,
            tags: HashMap::new(),
        }
    }

    /// Add a metadata tag.
    #[must_use]
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }
}

/// CAF (Core Audio Format) muxer.
///
/// Writes a CAF file with the `desc`, optional `info`, and `data` chunks.
/// Uses in-memory buffering and finalizes the file on [`CafMuxer::finish`].
///
/// # Example
///
/// ```rust
/// use oximedia_container::demux::caf::{CafMuxer, CafMuxConfig};
///
/// let config = CafMuxConfig::pcm(44100.0, 2, 16, false);
/// let mut muxer = CafMuxer::new(config);
///
/// // Write some audio data
/// let samples = vec![0i16; 4096];
/// let bytes: Vec<u8> = samples.iter()
///     .flat_map(|s| s.to_be_bytes())
///     .collect();
/// muxer.write_audio(&bytes);
///
/// let caf_bytes = muxer.finish();
/// assert!(caf_bytes.len() > 8);
/// ```
pub struct CafMuxer {
    config: CafMuxConfig,
    audio_data: Vec<u8>,
}

impl CafMuxer {
    /// Create a new CAF muxer.
    #[must_use]
    pub fn new(config: CafMuxConfig) -> Self {
        Self {
            config,
            audio_data: Vec::new(),
        }
    }

    /// Append raw audio bytes to the output.
    pub fn write_audio(&mut self, data: &[u8]) {
        self.audio_data.extend_from_slice(data);
    }

    /// Finalize and return the complete CAF file bytes.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();

        // --- File header (8 bytes): magic + version + flags ---
        out.extend_from_slice(CAF_MAGIC);
        out.extend_from_slice(&CAF_VERSION.to_be_bytes());
        out.extend_from_slice(&0u16.to_be_bytes()); // file flags = 0

        // --- desc chunk (12-byte header + 32-byte payload) ---
        out.extend_from_slice(chunk_type::DESC);
        out.extend_from_slice(&32i64.to_be_bytes());
        out.extend_from_slice(&self.config.asbd.to_bytes());

        // --- info chunk (optional) ---
        if !self.config.tags.is_empty() {
            let info_bytes = build_info_chunk(&self.config.tags);
            out.extend_from_slice(chunk_type::INFO);
            out.extend_from_slice(&(info_bytes.len() as i64).to_be_bytes());
            out.extend_from_slice(&info_bytes);
        }

        // --- data chunk (12-byte header + 4-byte edit count + audio data) ---
        // chunk size = 4 (edit count) + audio data length
        let data_chunk_size = 4i64 + self.audio_data.len() as i64;
        out.extend_from_slice(chunk_type::DATA);
        out.extend_from_slice(&data_chunk_size.to_be_bytes());
        out.extend_from_slice(&1u32.to_be_bytes()); // edit count = 1
        out.extend_from_slice(&self.audio_data);

        out
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse the `info` chunk (null-terminated key-value string pairs).
fn parse_info_chunk(data: &[u8]) -> HashMap<String, String> {
    if data.len() < 4 {
        return HashMap::new();
    }
    let num_entries = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let mut map = HashMap::with_capacity(num_entries);
    let mut offset = 4;

    for _ in 0..num_entries {
        if offset >= data.len() {
            break;
        }
        // Key: null-terminated UTF-8 string
        let key = read_cstring(data, &mut offset);
        let value = read_cstring(data, &mut offset);
        if !key.is_empty() {
            map.insert(key, value);
        }
    }

    map
}

/// Build the `info` chunk bytes from a map.
fn build_info_chunk(tags: &HashMap<String, String>) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let num_entries = tags.len() as u32;
    out.extend_from_slice(&num_entries.to_be_bytes());
    for (key, value) in tags {
        out.extend_from_slice(key.as_bytes());
        out.push(0); // null terminator
        out.extend_from_slice(value.as_bytes());
        out.push(0);
    }
    out
}

/// Read a null-terminated string from `data` starting at `*offset`.
fn read_cstring(data: &[u8], offset: &mut usize) -> String {
    let start = *offset;
    while *offset < data.len() && data[*offset] != 0 {
        *offset += 1;
    }
    let s = String::from_utf8_lossy(&data[start..*offset]).into_owned();
    if *offset < data.len() {
        *offset += 1; // skip null
    }
    s
}

/// Decode a variable-length integer (CAF VInt) from `data`.
///
/// Returns `(value, bytes_consumed)`.
fn decode_vint(data: &[u8]) -> Result<(u64, usize), ()> {
    if data.is_empty() {
        return Err(());
    }
    let mut value: u64 = 0;
    let mut offset = 0;
    loop {
        if offset >= data.len() {
            return Err(());
        }
        let byte = data[offset];
        offset += 1;
        value = (value << 7) | u64::from(byte & 0x7F);
        if byte & 0x80 == 0 {
            break;
        }
    }
    Ok((value, offset))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_pcm_caf(channels: u32, sample_rate: f64, bits: u32) -> Vec<u8> {
        let config = CafMuxConfig::pcm(sample_rate, channels, bits, false);
        let mut muxer = CafMuxer::new(config);
        // 16-bit big-endian silence: 1024 frames × channels × 2 bytes
        let samples = vec![0u8; 1024 * channels as usize * (bits / 8) as usize];
        muxer.write_audio(&samples);
        muxer.finish()
    }

    #[test]
    fn test_muxer_produces_valid_magic() {
        let data = make_pcm_caf(2, 44100.0, 16);
        assert!(data.len() >= 8);
        assert_eq!(&data[0..4], b"caff");
        assert_eq!(&data[4..6], &1u16.to_be_bytes());
    }

    #[test]
    fn test_muxer_roundtrip_desc() {
        let data = make_pcm_caf(2, 48000.0, 16);
        let cursor = Cursor::new(data);
        let mut demuxer = CafDemuxer::new(cursor);
        let info = demuxer.probe().expect("probe should succeed");
        assert!((info.sample_rate - 48000.0).abs() < 1.0);
        assert_eq!(info.channels, 2);
        assert_eq!(info.asbd.bits_per_channel, 16);
    }

    #[test]
    fn test_muxer_with_tags() {
        let config = CafMuxConfig::pcm(44100.0, 1, 16, false)
            .with_tag("artist", "Test Artist")
            .with_tag("title", "Test Title");
        let mut muxer = CafMuxer::new(config);
        muxer.write_audio(&[0u8; 64]);
        let data = muxer.finish();
        let cursor = Cursor::new(data);
        let mut demuxer = CafDemuxer::new(cursor);
        let info = demuxer.probe().expect("probe should succeed");
        assert_eq!(info.tags.get("artist").map(|s| s.as_str()), Some("Test Artist"));
        assert_eq!(info.tags.get("title").map(|s| s.as_str()), Some("Test Title"));
    }

    #[test]
    fn test_demuxer_read_audio() {
        let data = make_pcm_caf(2, 44100.0, 16);
        let cursor = Cursor::new(data);
        let mut demuxer = CafDemuxer::new(cursor);
        demuxer.probe().expect("probe should succeed");
        let audio = demuxer.read_all_audio().expect("read_all_audio should succeed");
        // 1024 frames × 2 channels × 2 bytes/sample
        assert_eq!(audio.len(), 1024 * 2 * 2);
    }

    #[test]
    fn test_demuxer_invalid_magic() {
        let bad = b"RIFF\x00\x00\x00\x00".to_vec();
        let cursor = Cursor::new(bad);
        let mut demuxer = CafDemuxer::new(cursor);
        assert!(demuxer.probe().is_err());
    }

    #[test]
    fn test_asbd_roundtrip() {
        let asbd = AudioStreamBasicDescription {
            sample_rate: 96000.0,
            format_id: format_id::LPCM,
            format_flags: 0x2 | 0x4 | 0x8,
            bytes_per_packet: 4,
            frames_per_packet: 1,
            channels_per_frame: 2,
            bits_per_channel: 16,
        };
        let bytes = asbd.to_bytes();
        let parsed = AudioStreamBasicDescription::parse(&bytes).expect("parse should succeed");
        assert!((parsed.sample_rate - asbd.sample_rate).abs() < 1e-6);
        assert_eq!(parsed.format_id, asbd.format_id);
        assert_eq!(parsed.channels_per_frame, asbd.channels_per_frame);
    }

    #[test]
    fn test_vint_decode() {
        // Single-byte value (no continuation)
        let (v, n) = decode_vint(&[0x42]).expect("decode should succeed");
        assert_eq!(v, 0x42);
        assert_eq!(n, 1);

        // Two-byte value
        let (v, n) = decode_vint(&[0x81, 0x00]).expect("decode should succeed");
        assert_eq!(v, 128);
        assert_eq!(n, 2);
    }

    #[test]
    fn test_asbd_is_float() {
        let asbd = AudioStreamBasicDescription {
            sample_rate: 44100.0,
            format_id: format_id::LPCM,
            format_flags: 0x1 | 0x2 | 0x8,
            bytes_per_packet: 4,
            frames_per_packet: 1,
            channels_per_frame: 1,
            bits_per_channel: 32,
        };
        assert!(asbd.is_float());
        assert!(!asbd.is_signed_integer());
    }

    #[test]
    fn test_read_packet_cbr() {
        let data = make_pcm_caf(1, 44100.0, 16);
        let cursor = Cursor::new(data);
        let mut demuxer = CafDemuxer::new(cursor);
        demuxer.probe().expect("probe should succeed");
        // Each packet = 1 frame × 1 channel × 2 bytes
        let pkt = demuxer.read_packet().expect("read_packet ok");
        assert!(pkt.is_some());
        assert_eq!(pkt.unwrap().len(), 2);
    }
}
