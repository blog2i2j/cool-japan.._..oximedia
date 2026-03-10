//! Output modules for various broadcast formats
//!
//! Supports SDI, NDI, RTMP, SRT, and IP multicast outputs with
//! frame-accurate synchronization and multiple simultaneous outputs.

use crate::playback::FrameBuffer;
use crate::{AudioFormat, PlayoutError, Result, VideoFormat};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::io::{BufWriter, Write};
use std::net::IpAddr;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Output type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputType {
    /// SDI output via Blackmagic Decklink
    SDI,
    /// NDI (Network Device Interface)
    NDI,
    /// RTMP streaming
    RTMP,
    /// SRT (Secure Reliable Transport)
    SRT,
    /// SMPTE ST 2110 (uncompressed IP)
    ST2110,
    /// SMPTE ST 2022 (compressed IP)
    ST2022,
    /// Local file output
    File,
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output type
    pub output_type: OutputType,

    /// Output name/identifier
    pub name: String,

    /// Video format
    pub video_format: VideoFormat,

    /// Audio format
    pub audio_format: AudioFormat,

    /// Output-specific settings
    pub settings: OutputSettings,

    /// Enable/disable flag
    pub enabled: bool,

    /// Priority (for multiple outputs)
    pub priority: u32,
}

/// Output-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputSettings {
    SDI(SDISettings),
    NDI(NDISettings),
    RTMP(RTMPSettings),
    SRT(SRTSettings),
    ST2110(ST2110Settings),
    ST2022(ST2022Settings),
    File(FileSettings),
}

/// SDI output settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDISettings {
    /// Device ID (e.g., 0 for first card)
    pub device_id: u32,

    /// Output connector (e.g., "SDI", "HDMI")
    pub connector: String,

    /// Video connection (e.g., "single_link", "dual_link", "quad_link")
    pub video_connection: String,

    /// Enable genlock
    pub genlock: bool,

    /// Reference source
    pub reference_source: String,

    /// Timecode source
    pub timecode_source: String,

    /// Low latency mode
    pub low_latency: bool,
}

impl Default for SDISettings {
    fn default() -> Self {
        Self {
            device_id: 0,
            connector: "SDI".to_string(),
            video_connection: "single_link".to_string(),
            genlock: false,
            reference_source: "internal".to_string(),
            timecode_source: "internal".to_string(),
            low_latency: true,
        }
    }
}

/// NDI output settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NDISettings {
    /// NDI source name
    pub source_name: String,

    /// NDI groups
    pub groups: Vec<String>,

    /// Clock synchronization
    pub clock_sync: bool,

    /// Enable audio
    pub enable_audio: bool,

    /// Enable metadata
    pub enable_metadata: bool,

    /// Bandwidth (in Mbps, 0 = unlimited)
    pub bandwidth_mbps: u32,
}

impl Default for NDISettings {
    fn default() -> Self {
        Self {
            source_name: "OxiMedia Playout".to_string(),
            groups: vec!["public".to_string()],
            clock_sync: true,
            enable_audio: true,
            enable_metadata: true,
            bandwidth_mbps: 0,
        }
    }
}

/// RTMP output settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RTMPSettings {
    /// RTMP URL (e.g., "rtmp://server/app")
    pub url: String,

    /// Stream key
    pub stream_key: String,

    /// Video bitrate in kbps
    pub video_bitrate_kbps: u32,

    /// Audio bitrate in kbps
    pub audio_bitrate_kbps: u32,

    /// Video codec
    pub video_codec: String,

    /// Audio codec
    pub audio_codec: String,

    /// Keyframe interval (GOP size)
    pub keyframe_interval: u32,

    /// Enable low latency
    pub low_latency: bool,

    /// Connection timeout in seconds
    pub timeout_seconds: u32,
}

impl Default for RTMPSettings {
    fn default() -> Self {
        Self {
            url: "rtmp://localhost/live".to_string(),
            stream_key: "stream".to_string(),
            video_bitrate_kbps: 5000,
            audio_bitrate_kbps: 128,
            video_codec: "h264".to_string(),
            audio_codec: "aac".to_string(),
            keyframe_interval: 50,
            low_latency: false,
            timeout_seconds: 30,
        }
    }
}

/// SRT output settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SRTSettings {
    /// SRT address
    pub address: String,

    /// Port
    pub port: u16,

    /// Mode (caller, listener, rendezvous)
    pub mode: SRTMode,

    /// Latency in milliseconds
    pub latency_ms: u32,

    /// Encryption passphrase (empty = no encryption)
    pub passphrase: String,

    /// Maximum bandwidth in bps (0 = unlimited)
    pub max_bandwidth_bps: u64,

    /// Enable forward error correction
    pub fec: bool,

    /// Overhead percentage for FEC
    pub overhead_percent: u32,
}

/// SRT connection mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SRTMode {
    Caller,
    Listener,
    Rendezvous,
}

impl Default for SRTSettings {
    fn default() -> Self {
        Self {
            address: "127.0.0.1".to_string(),
            port: 9000,
            mode: SRTMode::Caller,
            latency_ms: 120,
            passphrase: String::new(),
            max_bandwidth_bps: 0,
            fec: false,
            overhead_percent: 25,
        }
    }
}

/// SMPTE ST 2110 settings (uncompressed IP video)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ST2110Settings {
    /// Multicast address for video
    pub video_address: IpAddr,

    /// Video port
    pub video_port: u16,

    /// Multicast address for audio
    pub audio_address: IpAddr,

    /// Audio port
    pub audio_port: u16,

    /// Multicast address for ancillary data
    pub anc_address: Option<IpAddr>,

    /// Ancillary data port
    pub anc_port: Option<u16>,

    /// PTP domain
    pub ptp_domain: u8,

    /// Enable PTP synchronization
    pub ptp_sync: bool,

    /// Packet time (microseconds)
    pub packet_time_us: u32,

    /// Network interface
    pub interface: String,
}

impl Default for ST2110Settings {
    fn default() -> Self {
        Self {
            video_address: "239.0.0.1"
                .parse()
                .expect("invariant: valid multicast IP literal"),
            video_port: 5000,
            audio_address: "239.0.0.2"
                .parse()
                .expect("invariant: valid multicast IP literal"),
            audio_port: 5002,
            anc_address: None,
            anc_port: None,
            ptp_domain: 127,
            ptp_sync: true,
            packet_time_us: 125,
            interface: "eth0".to_string(),
        }
    }
}

/// SMPTE ST 2022 settings (compressed IP video with FEC)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ST2022Settings {
    /// Multicast address
    pub multicast_address: IpAddr,

    /// Port
    pub port: u16,

    /// Enable FEC
    pub fec_enabled: bool,

    /// FEC column address
    pub fec_col_address: Option<IpAddr>,

    /// FEC column port
    pub fec_col_port: Option<u16>,

    /// FEC row address
    pub fec_row_address: Option<IpAddr>,

    /// FEC row port
    pub fec_row_port: Option<u16>,

    /// FEC columns (L)
    pub fec_l: u16,

    /// FEC rows (D)
    pub fec_d: u16,

    /// Network interface
    pub interface: String,
}

impl Default for ST2022Settings {
    fn default() -> Self {
        Self {
            multicast_address: "239.1.0.1"
                .parse()
                .expect("invariant: valid multicast IP literal"),
            port: 5004,
            fec_enabled: true,
            fec_col_address: Some(
                "239.1.0.2"
                    .parse()
                    .expect("invariant: valid multicast IP literal"),
            ),
            fec_col_port: Some(5006),
            fec_row_address: Some(
                "239.1.0.3"
                    .parse()
                    .expect("invariant: valid multicast IP literal"),
            ),
            fec_row_port: Some(5008),
            fec_l: 4,
            fec_d: 4,
            interface: "eth0".to_string(),
        }
    }
}

/// File output settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSettings {
    /// Output file path
    pub path: String,

    /// Container format
    pub format: String,

    /// Video codec
    pub video_codec: String,

    /// Audio codec
    pub audio_codec: String,

    /// Video bitrate in kbps
    pub video_bitrate_kbps: u32,

    /// Audio bitrate in kbps
    pub audio_bitrate_kbps: u32,
}

impl Default for FileSettings {
    fn default() -> Self {
        Self {
            path: "/tmp/output.mxf".to_string(),
            format: "mxf".to_string(),
            video_codec: "mpeg2video".to_string(),
            audio_codec: "pcm_s24le".to_string(),
            video_bitrate_kbps: 50000,
            audio_bitrate_kbps: 1536,
        }
    }
}

/// Output statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutputStats {
    /// Total frames sent
    pub frames_sent: u64,

    /// Frames dropped
    pub frames_dropped: u64,

    /// Bytes sent
    pub bytes_sent: u64,

    /// Current bitrate (bps)
    pub current_bitrate_bps: u64,

    /// Average bitrate (bps)
    pub avg_bitrate_bps: u64,

    /// Network errors
    pub network_errors: u64,

    /// Buffer underruns
    pub buffer_underruns: u64,

    /// Connection status
    pub connected: bool,

    /// Last error message
    pub last_error: Option<String>,
}

/// Output state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputState {
    Stopped,
    Starting,
    Running,
    Error,
}

/// RTMP streaming state
#[allow(dead_code)]
struct RtmpState {
    url: String,
    stream_key: String,
    sequence_number: u32,
}

#[allow(dead_code)]
impl RtmpState {
    fn new(url: String, stream_key: String) -> Self {
        Self {
            url,
            stream_key,
            sequence_number: 0,
        }
    }

    /// Build an FLV video tag for RTMP transport.
    ///
    /// FLV tag layout:
    ///   [0]     Tag type (0x09 = video)
    ///   [1..=3] Data size (big-endian u24)
    ///   [4..=6] Timestamp (ms, big-endian u24)
    ///   [7]     Timestamp extension (high byte)
    ///   [8..=10] Stream ID (always 0, big-endian u24)
    ///   [11..]  Payload (video data)
    ///   last 4  Previous tag size (big-endian u32)
    fn build_flv_video_tag(data: &[u8], timestamp_ms: u64) -> Vec<u8> {
        let data_size = data.len() as u32;
        let ts = timestamp_ms as u32;
        let ts_ext = ((timestamp_ms >> 24) & 0xFF) as u8;

        let mut tag = Vec::with_capacity(11 + data.len() + 4);
        // Tag type: video
        tag.push(0x09);
        // Data size (u24 big-endian)
        tag.push(((data_size >> 16) & 0xFF) as u8);
        tag.push(((data_size >> 8) & 0xFF) as u8);
        tag.push((data_size & 0xFF) as u8);
        // Timestamp (u24 big-endian lower 24 bits)
        tag.push(((ts >> 16) & 0xFF) as u8);
        tag.push(((ts >> 8) & 0xFF) as u8);
        tag.push((ts & 0xFF) as u8);
        // Timestamp extension (high byte)
        tag.push(ts_ext);
        // Stream ID (u24, always 0)
        tag.push(0x00);
        tag.push(0x00);
        tag.push(0x00);
        // Payload
        tag.extend_from_slice(data);
        // Previous tag size (u32 big-endian): header (11) + data
        let prev_tag_size = 11u32 + data_size;
        tag.push(((prev_tag_size >> 24) & 0xFF) as u8);
        tag.push(((prev_tag_size >> 16) & 0xFF) as u8);
        tag.push(((prev_tag_size >> 8) & 0xFF) as u8);
        tag.push((prev_tag_size & 0xFF) as u8);
        tag
    }

    /// Send a video frame as an RTMP FLV message (simulation).
    fn send_frame(&mut self, data: &[u8], timestamp_ms: u64) -> Result<()> {
        let tag = Self::build_flv_video_tag(data, timestamp_ms);
        tracing::debug!(
            "RTMP [{}]/[{}] seq={} ts={}ms flv_tag_size={}",
            self.url,
            self.stream_key,
            self.sequence_number,
            timestamp_ms,
            tag.len()
        );
        self.sequence_number = self.sequence_number.wrapping_add(1);
        Ok(())
    }
}

/// SRT streaming state
#[allow(dead_code)]
struct SrtState {
    address: String,
    port: u16,
    socket_id: u32,
    seq_num: u32,
}

#[allow(dead_code)]
impl SrtState {
    fn new(address: String, port: u16) -> Self {
        // socket_id: derive a stable pseudo-identifier from the address/port
        let socket_id = {
            let mut h: u32 = 0x811c9dc5;
            for b in address.bytes() {
                h = h.wrapping_mul(0x01000193) ^ b as u32;
            }
            h = h.wrapping_mul(0x01000193) ^ (port as u32);
            h
        };
        Self {
            address,
            port,
            socket_id,
            seq_num: 0,
        }
    }

    /// Chunk data into SRT data packets (max 1316 bytes payload each).
    ///
    /// Each chunk is prefixed with a minimal 16-byte SRT data packet header:
    ///   [0..=3]  Sequence number (big-endian u32, MSB=0 → data packet)
    ///   [4..=7]  Message number + PP/O/KK flags (big-endian u32)
    ///   [8..=11] Timestamp (microseconds, big-endian u32)
    ///   [12..=15] Destination socket ID (big-endian u32)
    fn send_frame(&mut self, data: &[u8], timestamp_ms: u64) -> Result<()> {
        const MAX_PAYLOAD: usize = 1316;
        let timestamp_us = (timestamp_ms * 1000) as u32;
        let total_chunks = data.chunks(MAX_PAYLOAD).count();

        for (chunk_idx, chunk) in data.chunks(MAX_PAYLOAD).enumerate() {
            let is_first = chunk_idx == 0;
            let is_last = chunk_idx == total_chunks - 1;

            // PP bits: 10=first, 00=middle, 01=last, 11=single
            let pp: u8 = match (is_first, is_last) {
                (true, true) => 0b11,
                (true, false) => 0b10,
                (false, true) => 0b01,
                _ => 0b00,
            };

            let seq = self.seq_num & 0x7FFF_FFFF; // clear MSB (data packet)
            let msg_number: u32 = (self.seq_num & 0x003F_FFFF) | ((pp as u32) << 30);

            let mut header = [0u8; 16];
            header[0..4].copy_from_slice(&seq.to_be_bytes());
            header[4..8].copy_from_slice(&msg_number.to_be_bytes());
            header[8..12].copy_from_slice(&timestamp_us.to_be_bytes());
            header[12..16].copy_from_slice(&self.socket_id.to_be_bytes());

            tracing::debug!(
                "SRT [{}:{}] seq={} chunk={}/{} payload={}B",
                self.address,
                self.port,
                self.seq_num,
                chunk_idx + 1,
                total_chunks,
                chunk.len()
            );
            let _ = (header, chunk); // would be sent over UDP socket
            self.seq_num = self.seq_num.wrapping_add(1);
        }
        Ok(())
    }
}

/// File output state using a buffered writer.
#[allow(dead_code)]
struct FileOutputState {
    writer: BufWriter<std::fs::File>,
    path: String,
    frames_written: u64,
}

#[allow(dead_code)]
impl FileOutputState {
    fn new(path: &str) -> Result<Self> {
        let file = std::fs::File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            path: path.to_string(),
            frames_written: 0,
        })
    }

    /// Write a raw binary frame to the output file.
    fn write_frame(&mut self, data: &[u8]) -> Result<()> {
        // Write a simple frame header: magic (4 bytes) + frame_number (8 bytes) + data_len (4 bytes)
        let magic: u32 = 0x4F584D46; // "OXMF"
        let frame_number = self.frames_written;
        let data_len = data.len() as u32;

        self.writer.write_all(&magic.to_be_bytes())?;
        self.writer.write_all(&frame_number.to_be_bytes())?;
        self.writer.write_all(&data_len.to_be_bytes())?;
        self.writer.write_all(data)?;
        self.frames_written += 1;
        Ok(())
    }
}

/// ST 2110 RTP packetization state (RFC 4175 uncompressed video).
#[allow(dead_code)]
struct St2110State {
    settings: ST2110Settings,
    /// RTP sequence number (wraps at 16-bit boundary)
    rtp_seq: u16,
    /// RTP SSRC (Synchronization Source identifier)
    ssrc: u32,
}

#[allow(dead_code)]
impl St2110State {
    fn new(settings: ST2110Settings) -> Self {
        // Use video_port as part of SSRC for a stable identifier
        let ssrc = 0x21100000u32 | (settings.video_port as u32);
        Self {
            settings,
            rtp_seq: 0,
            ssrc,
        }
    }

    /// Packetize a video frame as RFC 4175 RTP packets.
    ///
    /// RTP header (12 bytes):
    ///   V=2, P=0, X=0, CC=0, M=0/1, PT=96
    ///   Sequence number (16-bit)
    ///   Timestamp (32-bit, 90 kHz clock)
    ///   SSRC (32-bit)
    ///
    /// RFC 4175 payload header (6 bytes per line):
    ///   Line number (16-bit) | F (field) bit
    ///   Offset (16-bit) | C (continuation) bit
    ///   Length (16-bit)
    fn send_frame(
        &mut self,
        data: &[u8],
        timestamp_ms: u64,
        width: u32,
        height: u32,
    ) -> Result<()> {
        // 90 kHz RTP clock
        let rtp_timestamp = ((timestamp_ms * 90_000) / 1_000) as u32;
        // Bytes per pixel: 2 (YCbCr 4:2:2 packed, 8-bit → 2 bytes/pixel)
        let bytes_per_pixel: u32 = 2;
        let bytes_per_line = width * bytes_per_pixel;
        // Max RTP payload after headers (1 line segment per packet for simplicity)
        let max_line_payload: u32 = 1400;
        let total_lines = height;

        for line in 0..total_lines {
            let mut offset: u32 = 0;
            while offset < bytes_per_line {
                let seg_len = (bytes_per_line - offset).min(max_line_payload);
                let is_last_seg = (offset + seg_len) >= bytes_per_line;
                let is_last_line = line == (total_lines - 1);
                let marker = is_last_seg && is_last_line;

                // RTP header
                let mut rtp_header = [0u8; 12];
                rtp_header[0] = 0x80; // V=2
                rtp_header[1] = if marker { 0x80 | 96 } else { 96 }; // M + PT=96
                rtp_header[2..4].copy_from_slice(&self.rtp_seq.to_be_bytes());
                rtp_header[4..8].copy_from_slice(&rtp_timestamp.to_be_bytes());
                rtp_header[8..12].copy_from_slice(&self.ssrc.to_be_bytes());

                // RFC 4175 payload header (6 bytes, single line segment)
                let line_num = line as u16;
                let continuation: u16 = 0; // no continuation
                let mut rfc4175_hdr = [0u8; 6];
                rfc4175_hdr[0..2].copy_from_slice(&line_num.to_be_bytes());
                rfc4175_hdr[2..4]
                    .copy_from_slice(&(offset as u16 | (continuation << 15)).to_be_bytes());
                rfc4175_hdr[4..6].copy_from_slice(&(seg_len as u16).to_be_bytes());

                // Pixel data slice
                let data_start = (line * bytes_per_line + offset) as usize;
                let data_end = (data_start + seg_len as usize).min(data.len());
                let _pixel_data = &data[data_start..data_end];

                tracing::trace!(
                    "ST2110 RTP seq={} line={} off={} len={} marker={}",
                    self.rtp_seq,
                    line,
                    offset,
                    seg_len,
                    marker
                );
                let _ = (rtp_header, rfc4175_hdr); // would be sent via UDP
                self.rtp_seq = self.rtp_seq.wrapping_add(1);
                offset += seg_len;
            }
        }
        Ok(())
    }
}

/// ST 2022 FEC state for compressed IP video.
///
/// Implements column/row interleaving per SMPTE ST 2022-1.
/// Media packets are arranged in an L×D matrix; column FEC covers each
/// column (L packets), row FEC covers each row (D packets).
#[allow(dead_code)]
struct St2022State {
    settings: ST2022Settings,
    /// RTP sequence number
    rtp_seq: u16,
    /// SSRC
    ssrc: u32,
    /// Ring buffer of recent media packets for FEC computation (rows × cols)
    media_matrix: Vec<Vec<u8>>,
    /// Position within the matrix
    matrix_pos: usize,
}

#[allow(dead_code)]
impl St2022State {
    fn new(settings: ST2022Settings) -> Self {
        let l = settings.fec_l as usize;
        let d = settings.fec_d as usize;
        let ssrc = 0x20220000u32 | (settings.port as u32);
        Self {
            settings,
            rtp_seq: 0,
            ssrc,
            media_matrix: vec![vec![0u8; l]; d],
            matrix_pos: 0,
        }
    }

    /// XOR a source packet into a FEC packet accumulator.
    fn xor_into(fec: &mut Vec<u8>, src: &[u8]) {
        if fec.len() < src.len() {
            fec.resize(src.len(), 0);
        }
        for (f, s) in fec.iter_mut().zip(src.iter()) {
            *f ^= s;
        }
    }

    /// Send a compressed frame with FEC column/row packets.
    fn send_frame(&mut self, data: &[u8], timestamp_ms: u64) -> Result<()> {
        let rtp_timestamp = ((timestamp_ms * 90_000) / 1_000) as u32;
        let l = self.settings.fec_l as usize;
        let d = self.settings.fec_d as usize;
        let fec_enabled = self.settings.fec_enabled;

        // Split data into L*D chunks for matrix placement
        let chunk_size = (data.len() / (l * d)).max(1);
        let mut col_fec: Vec<Vec<u8>> = vec![Vec::new(); l];
        let mut row_fec: Vec<Vec<u8>> = vec![Vec::new(); d];

        for row in 0..d {
            for col in 0..l {
                let start = (row * l + col) * chunk_size;
                let end = (start + chunk_size).min(data.len());
                let chunk = if start < data.len() {
                    &data[start..end]
                } else {
                    &[]
                };

                // Build and "send" RTP media packet
                let mut rtp_header = [0u8; 12];
                rtp_header[0] = 0x80;
                rtp_header[1] = 97; // PT=97 for compressed media
                rtp_header[2..4].copy_from_slice(&self.rtp_seq.to_be_bytes());
                rtp_header[4..8].copy_from_slice(&rtp_timestamp.to_be_bytes());
                rtp_header[8..12].copy_from_slice(&self.ssrc.to_be_bytes());

                tracing::trace!(
                    "ST2022 media seq={} row={} col={} payload={}B",
                    self.rtp_seq,
                    row,
                    col,
                    chunk.len()
                );
                let _ = (rtp_header, chunk);
                self.rtp_seq = self.rtp_seq.wrapping_add(1);

                // Accumulate FEC
                if fec_enabled {
                    Self::xor_into(&mut col_fec[col], chunk);
                    Self::xor_into(&mut row_fec[row], chunk);
                }
            }
        }

        // Send column FEC packets
        if fec_enabled && self.settings.fec_col_address.is_some() {
            for (col, fec_data) in col_fec.iter().enumerate() {
                tracing::trace!("ST2022 col-FEC col={} fec_size={}B", col, fec_data.len());
            }
        }

        // Send row FEC packets
        if fec_enabled && self.settings.fec_row_address.is_some() {
            for (row, fec_data) in row_fec.iter().enumerate() {
                tracing::trace!("ST2022 row-FEC row={} fec_size={}B", row, fec_data.len());
            }
        }

        Ok(())
    }
}

/// Internal output state
struct InternalState {
    state: OutputState,
    stats: OutputStats,
    frame_count: u64,
    /// RTMP streaming state (present when output type is RTMP)
    #[allow(dead_code)]
    rtmp: Option<RtmpState>,
    /// SRT streaming state
    #[allow(dead_code)]
    srt: Option<SrtState>,
    /// File output state
    #[allow(dead_code)]
    file_output: Option<FileOutputState>,
    /// ST 2110 packetizer state
    #[allow(dead_code)]
    st2110: Option<St2110State>,
    /// ST 2022 FEC state
    #[allow(dead_code)]
    st2022: Option<St2022State>,
}

/// Base output interface
pub struct Output {
    config: OutputConfig,
    state: Arc<RwLock<InternalState>>,
    frame_tx: mpsc::Sender<FrameBuffer>,
    #[allow(dead_code)]
    frame_rx: Arc<RwLock<mpsc::Receiver<FrameBuffer>>>,
}

impl Output {
    /// Create a new output
    pub fn new(config: OutputConfig) -> Self {
        let (frame_tx, frame_rx) = mpsc::channel(100);

        let state = InternalState {
            state: OutputState::Stopped,
            stats: OutputStats::default(),
            frame_count: 0,
            rtmp: None,
            srt: None,
            file_output: None,
            st2110: None,
            st2022: None,
        };

        Self {
            config,
            state: Arc::new(RwLock::new(state)),
            frame_tx,
            frame_rx: Arc::new(RwLock::new(frame_rx)),
        }
    }

    /// Start the output
    pub async fn start(&self) -> Result<()> {
        {
            let mut state = self.state.write();

            if state.state != OutputState::Stopped {
                return Err(PlayoutError::Output("Output already running".to_string()));
            }

            state.state = OutputState::Starting;
        }

        // Initialize output based on type
        match &self.config.settings {
            OutputSettings::SDI(settings) => self.start_sdi(settings).await?,
            OutputSettings::NDI(settings) => self.start_ndi(settings).await?,
            OutputSettings::RTMP(settings) => self.start_rtmp(settings).await?,
            OutputSettings::SRT(settings) => self.start_srt(settings).await?,
            OutputSettings::ST2110(settings) => self.start_st2110(settings).await?,
            OutputSettings::ST2022(settings) => self.start_st2022(settings).await?,
            OutputSettings::File(settings) => self.start_file(settings).await?,
        }

        {
            let mut state = self.state.write();
            state.state = OutputState::Running;
            state.stats.connected = true;
        }

        Ok(())
    }

    /// Stop the output
    pub async fn stop(&self) -> Result<()> {
        let mut state = self.state.write();
        state.state = OutputState::Stopped;
        state.stats.connected = false;
        Ok(())
    }

    /// Send a frame to the output
    pub async fn send_frame(&self, frame: FrameBuffer) -> Result<()> {
        self.frame_tx
            .send(frame)
            .await
            .map_err(|e| PlayoutError::Output(format!("Failed to send frame: {e}")))?;

        let mut state = self.state.write();
        state.frame_count += 1;
        state.stats.frames_sent += 1;

        Ok(())
    }

    /// Get output statistics
    pub fn get_stats(&self) -> OutputStats {
        self.state.read().stats.clone()
    }

    /// Get output state
    pub fn get_state(&self) -> OutputState {
        self.state.read().state
    }

    /// Get configuration
    pub fn config(&self) -> &OutputConfig {
        &self.config
    }

    /// Start SDI output.
    ///
    /// SDI output requires hardware (e.g. Decklink cards) which is not
    /// available in the pure-Rust default build.  When a hardware
    /// integration feature is enabled in the future, this method will
    /// initialise the device.  For now it logs the configuration and
    /// succeeds immediately so the rest of the output pipeline can be
    /// exercised.
    async fn start_sdi(&self, settings: &SDISettings) -> Result<()> {
        tracing::info!(
            "SDI output configured: device={}, connector={}, genlock={}",
            settings.device_id,
            settings.connector,
            settings.genlock,
        );
        Ok(())
    }

    /// Start NDI output.
    ///
    /// NDI (Network Device Interface) requires the proprietary NDI SDK
    /// which is not bundled with the pure-Rust default build.  When a
    /// dedicated `ndi` feature is enabled in the future, this method
    /// will create an NDI sender.  For now it logs the source name and
    /// succeeds immediately.
    async fn start_ndi(&self, settings: &NDISettings) -> Result<()> {
        tracing::info!(
            "NDI output configured: source='{}', groups={:?}, clock_sync={}",
            settings.source_name,
            settings.groups,
            settings.clock_sync,
        );
        Ok(())
    }

    /// Start RTMP output
    async fn start_rtmp(&self, settings: &RTMPSettings) -> Result<()> {
        tracing::info!("Starting RTMP output: {}", settings.url);
        let rtmp_state = RtmpState::new(settings.url.clone(), settings.stream_key.clone());
        self.state.write().rtmp = Some(rtmp_state);
        Ok(())
    }

    /// Start SRT output
    async fn start_srt(&self, settings: &SRTSettings) -> Result<()> {
        tracing::info!(
            "Starting SRT output: {}:{}",
            settings.address,
            settings.port
        );
        let srt_state = SrtState::new(settings.address.clone(), settings.port);
        self.state.write().srt = Some(srt_state);
        Ok(())
    }

    /// Start ST 2110 output
    async fn start_st2110(&self, settings: &ST2110Settings) -> Result<()> {
        tracing::info!(
            "Starting ST 2110 output: {}:{}",
            settings.video_address,
            settings.video_port
        );
        let st2110_state = St2110State::new(settings.clone());
        self.state.write().st2110 = Some(st2110_state);
        Ok(())
    }

    /// Start ST 2022 output
    async fn start_st2022(&self, settings: &ST2022Settings) -> Result<()> {
        tracing::info!(
            "Starting ST 2022 output: {}:{}",
            settings.multicast_address,
            settings.port
        );
        let st2022_state = St2022State::new(settings.clone());
        self.state.write().st2022 = Some(st2022_state);
        Ok(())
    }

    /// Start file output
    async fn start_file(&self, settings: &FileSettings) -> Result<()> {
        tracing::info!("Starting file output: {}", settings.path);
        let file_state = FileOutputState::new(&settings.path)?;
        self.state.write().file_output = Some(file_state);
        Ok(())
    }

    /// Update statistics
    #[allow(dead_code)]
    fn update_stats(&self, bytes: u64) {
        let mut state = self.state.write();
        state.stats.bytes_sent += bytes;

        // Calculate bitrate using the configured video format's frame rate
        if state.stats.frames_sent > 0 {
            let frame_rate = self.config.video_format.fps();
            state.stats.avg_bitrate_bps =
                (state.stats.bytes_sent * 8 * frame_rate as u64) / state.stats.frames_sent;
        }
    }

    /// Record error
    #[allow(dead_code)]
    fn record_error(&self, error: String) {
        let mut state = self.state.write();
        state.stats.network_errors += 1;
        state.stats.last_error = Some(error);
        state.state = OutputState::Error;
    }
}

/// Output manager for handling multiple outputs
pub struct OutputManager {
    outputs: Arc<RwLock<Vec<Arc<Output>>>>,
}

impl OutputManager {
    /// Create a new output manager
    pub fn new() -> Self {
        Self {
            outputs: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Add an output
    pub fn add_output(&self, config: OutputConfig) -> Arc<Output> {
        let output = Arc::new(Output::new(config));
        self.outputs.write().push(Arc::clone(&output));
        output
    }

    /// Remove an output
    pub fn remove_output(&self, name: &str) -> Result<()> {
        let mut outputs = self.outputs.write();
        let original_len = outputs.len();
        outputs.retain(|output| output.config().name != name);

        if outputs.len() < original_len {
            Ok(())
        } else {
            Err(PlayoutError::Output(format!("Output not found: {name}")))
        }
    }

    /// Get output by name
    pub fn get_output(&self, name: &str) -> Option<Arc<Output>> {
        self.outputs
            .read()
            .iter()
            .find(|output| output.config().name == name)
            .map(Arc::clone)
    }

    /// Start all enabled outputs
    pub async fn start_all(&self) -> Result<()> {
        let outputs = {
            let outputs = self.outputs.read();
            outputs
                .iter()
                .filter(|o| o.config().enabled)
                .cloned()
                .collect::<Vec<_>>()
        };
        for output in outputs {
            output.start().await?;
        }
        Ok(())
    }

    /// Stop all outputs
    pub async fn stop_all(&self) -> Result<()> {
        let outputs = {
            let outputs = self.outputs.read();
            outputs.iter().cloned().collect::<Vec<_>>()
        };
        for output in outputs {
            output.stop().await?;
        }
        Ok(())
    }

    /// Send frame to all active outputs
    pub async fn broadcast_frame(&self, frame: FrameBuffer) -> Result<()> {
        let outputs = {
            let outputs = self.outputs.read();
            outputs
                .iter()
                .filter(|o| o.get_state() == OutputState::Running)
                .cloned()
                .collect::<Vec<_>>()
        };
        for output in outputs {
            // Clone frame for each output
            output.send_frame(frame.clone()).await?;
        }
        Ok(())
    }

    /// Get statistics for all outputs
    pub fn get_all_stats(&self) -> Vec<(String, OutputStats)> {
        self.outputs
            .read()
            .iter()
            .map(|output| (output.config().name.clone(), output.get_stats()))
            .collect()
    }

    /// List all outputs
    pub fn list_outputs(&self) -> Vec<String> {
        self.outputs
            .read()
            .iter()
            .map(|output| output.config().name.clone())
            .collect()
    }

    /// Clear all outputs
    pub fn clear(&self) {
        self.outputs.write().clear();
    }
}

impl Default for OutputManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdi_settings_default() {
        let settings = SDISettings::default();
        assert_eq!(settings.device_id, 0);
        assert_eq!(settings.connector, "SDI");
    }

    #[test]
    fn test_ndi_settings_default() {
        let settings = NDISettings::default();
        assert_eq!(settings.source_name, "OxiMedia Playout");
        assert!(settings.clock_sync);
    }

    #[test]
    fn test_rtmp_settings_default() {
        let settings = RTMPSettings::default();
        assert_eq!(settings.video_codec, "h264");
        assert_eq!(settings.audio_codec, "aac");
    }

    #[test]
    fn test_srt_settings_default() {
        let settings = SRTSettings::default();
        assert_eq!(settings.port, 9000);
        assert_eq!(settings.mode, SRTMode::Caller);
    }

    #[test]
    fn test_output_config() {
        let config = OutputConfig {
            output_type: OutputType::RTMP,
            name: "Test Output".to_string(),
            video_format: VideoFormat::HD1080p25,
            audio_format: AudioFormat::default(),
            settings: OutputSettings::RTMP(RTMPSettings::default()),
            enabled: true,
            priority: 100,
        };

        assert_eq!(config.output_type, OutputType::RTMP);
        assert!(config.enabled);
    }

    #[test]
    fn test_output_manager() {
        let manager = OutputManager::new();
        assert_eq!(manager.list_outputs().len(), 0);

        let config = OutputConfig {
            output_type: OutputType::RTMP,
            name: "Test".to_string(),
            video_format: VideoFormat::HD1080p25,
            audio_format: AudioFormat::default(),
            settings: OutputSettings::RTMP(RTMPSettings::default()),
            enabled: true,
            priority: 100,
        };

        manager.add_output(config);
        assert_eq!(manager.list_outputs().len(), 1);
    }

    #[test]
    fn test_output_stats() {
        let stats = OutputStats::default();
        assert_eq!(stats.frames_sent, 0);
        assert!(!stats.connected);
    }

    #[tokio::test]
    async fn test_output_lifecycle() {
        let config = OutputConfig {
            output_type: OutputType::File,
            name: "Test".to_string(),
            video_format: VideoFormat::HD1080p25,
            audio_format: AudioFormat::default(),
            settings: OutputSettings::File(FileSettings::default()),
            enabled: true,
            priority: 100,
        };

        let output = Output::new(config);
        assert_eq!(output.get_state(), OutputState::Stopped);
    }
}
