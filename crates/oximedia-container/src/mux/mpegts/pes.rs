//! PES (Packetized Elementary Stream) packet construction.
//!
//! This module provides functionality for constructing PES packets
//! from compressed media data.

use oximedia_core::{CodecId, MediaType, OxiResult};

/// PES start code prefix (3 bytes).
const PES_START_CODE_PREFIX: &[u8] = &[0x00, 0x00, 0x01];

/// Maximum PES packet payload size (64 KB - reasonable limit).
const MAX_PES_PAYLOAD_SIZE: usize = 65535;

/// Stream ID for video streams.
const STREAM_ID_VIDEO: u8 = 0xE0;

/// Stream ID for audio streams.
const STREAM_ID_AUDIO: u8 = 0xC0;

/// Stream ID for private stream 1 (used for subtitles, etc.).
const STREAM_ID_PRIVATE_1: u8 = 0xBD;

/// PES packet builder.
#[derive(Clone, Copy)]
pub struct PesPacketBuilder {
    /// Stream ID.
    stream_id: u8,
    /// Presentation timestamp (33-bit).
    pts: Option<i64>,
    /// Decode timestamp (33-bit).
    dts: Option<i64>,
}

impl PesPacketBuilder {
    /// Creates a new PES packet builder.
    ///
    /// # Arguments
    ///
    /// * `codec` - Codec ID to determine stream type
    /// * `stream_index` - Stream index for ID assignment
    #[must_use]
    pub fn new(codec: CodecId, stream_index: usize) -> Self {
        let stream_id = Self::assign_stream_id(codec, stream_index);

        Self {
            stream_id,
            pts: None,
            dts: None,
        }
    }

    /// Assigns a stream ID based on codec and stream index.
    #[allow(clippy::cast_possible_truncation)]
    fn assign_stream_id(codec: CodecId, stream_index: usize) -> u8 {
        match codec.media_type() {
            MediaType::Video => STREAM_ID_VIDEO.saturating_add(stream_index as u8),
            MediaType::Audio => STREAM_ID_AUDIO.saturating_add(stream_index as u8),
            MediaType::Subtitle | MediaType::Data | MediaType::Attachment => STREAM_ID_PRIVATE_1,
        }
    }

    /// Sets the presentation timestamp.
    #[must_use]
    pub const fn with_pts(mut self, pts: i64) -> Self {
        self.pts = Some(pts);
        self
    }

    /// Sets the decode timestamp.
    #[must_use]
    pub const fn with_dts(mut self, dts: i64) -> Self {
        self.dts = Some(dts);
        self
    }

    /// Builds a PES packet from payload data.
    ///
    /// # Arguments
    ///
    /// * `payload` - Elementary stream payload data
    ///
    /// # Errors
    ///
    /// Returns an error if the payload is too large.
    pub fn build(&self, payload: &[u8]) -> OxiResult<Vec<u8>> {
        if payload.len() > MAX_PES_PAYLOAD_SIZE {
            return Err(oximedia_core::OxiError::InvalidData(format!(
                "PES payload too large: {} bytes (max {})",
                payload.len(),
                MAX_PES_PAYLOAD_SIZE
            )));
        }

        let mut pes_packet = Vec::new();

        // Start code prefix (3 bytes)
        pes_packet.extend_from_slice(PES_START_CODE_PREFIX);

        // Stream ID (1 byte)
        pes_packet.push(self.stream_id);

        // Determine header flags and size
        let has_presentation_ts = self.pts.is_some();
        let has_decode_ts = self.dts.is_some();

        let pts_dts_flags = if has_presentation_ts && has_decode_ts {
            0x03 // Both PTS and DTS
        } else if has_presentation_ts {
            0x02 // PTS only
        } else {
            0x00 // Neither
        };

        let header_data_length = if has_presentation_ts && has_decode_ts {
            10 // 5 bytes PTS + 5 bytes DTS
        } else if has_presentation_ts {
            5 // 5 bytes PTS
        } else {
            0
        };

        // Calculate PES packet length (or 0 for unbounded)
        // PES packet length = header_fields (3) + header_data_length + payload_length
        #[allow(clippy::cast_possible_truncation)]
        let pes_packet_length: u16 = if payload.len() + 3 + header_data_length > 65535 {
            0 // Unbounded (for large packets)
        } else {
            (3 + header_data_length + payload.len()) as u16
        };

        // PES packet length (2 bytes)
        #[allow(clippy::cast_possible_truncation)]
        {
            pes_packet.push((pes_packet_length >> 8) as u8);
            pes_packet.push((pes_packet_length & 0xFF) as u8);
        }

        // PES header flags (2 bytes)
        pes_packet.push(0x80); // '10' marker bits + scrambling/priority/alignment flags
        pes_packet.push(pts_dts_flags << 6); // PTS_DTS_flags + other flags

        // PES header data length (1 byte)
        #[allow(clippy::cast_possible_truncation)]
        pes_packet.push(header_data_length as u8);

        // PTS/DTS
        if let Some(pts) = self.pts {
            Self::write_timestamp(
                &mut pes_packet,
                pts,
                if has_decode_ts { 0x03 } else { 0x02 },
            );
        }

        if let Some(dts) = self.dts {
            Self::write_timestamp(&mut pes_packet, dts, 0x01);
        }

        // Payload
        pes_packet.extend_from_slice(payload);

        Ok(pes_packet)
    }

    /// Writes a 33-bit timestamp in PES format.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Output buffer
    /// * `timestamp` - 33-bit timestamp value
    /// * `prefix` - Prefix bits (0x02 for PTS, 0x01 for DTS, 0x03 for PTS with DTS)
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn write_timestamp(buffer: &mut Vec<u8>, timestamp: i64, prefix: u8) {
        let ts = timestamp as u64;

        buffer.push(((prefix << 4) | (((ts >> 29) & 0x0E) as u8)) | 0x01);
        buffer.push((ts >> 22) as u8);
        buffer.push((((ts >> 14) & 0xFE) as u8) | 0x01);
        buffer.push((ts >> 7) as u8);
        buffer.push(((ts << 1) as u8) | 0x01);
    }

    /// Splits large payloads into multiple PES packets.
    ///
    /// # Arguments
    ///
    /// * `payload` - Elementary stream payload data
    /// * `chunk_size` - Maximum size for each chunk
    ///
    /// # Errors
    ///
    /// Returns an error if packet construction fails.
    #[allow(dead_code)]
    pub fn build_chunked(&self, payload: &[u8], chunk_size: usize) -> OxiResult<Vec<Vec<u8>>> {
        let mut packets = Vec::new();
        let mut offset = 0;

        while offset < payload.len() {
            let chunk_end = std::cmp::min(offset + chunk_size, payload.len());
            let chunk = &payload[offset..chunk_end];

            // Only include PTS/DTS in the first packet
            let builder = if offset == 0 {
                Self {
                    stream_id: self.stream_id,
                    pts: self.pts,
                    dts: self.dts,
                }
            } else {
                Self {
                    stream_id: self.stream_id,
                    pts: None,
                    dts: None,
                }
            };

            packets.push(builder.build(chunk)?);
            offset = chunk_end;
        }

        Ok(packets)
    }
}

/// Calculates the PES header size for given timestamps.
///
/// # Arguments
///
/// * `has_pts` - Whether PTS is present
/// * `has_dts` - Whether DTS is present
#[must_use]
#[allow(dead_code)]
pub const fn pes_header_size(has_presentation_ts: bool, has_decode_ts: bool) -> usize {
    // Start code (3) + stream_id (1) + packet_length (2) + flags (3)
    let base_size = 9;

    let timestamp_size = if has_presentation_ts && has_decode_ts {
        10 // PTS (5) + DTS (5)
    } else if has_presentation_ts {
        5 // PTS (5)
    } else {
        0
    };

    base_size + timestamp_size
}

/// Calculates stuffing bytes needed to align PES packet.
///
/// # Arguments
///
/// * `current_size` - Current packet size
/// * `alignment` - Desired alignment (e.g., 188 for TS packet)
#[allow(dead_code)]
#[must_use]
pub const fn calculate_stuffing(current_size: usize, alignment: usize) -> usize {
    let remainder = current_size % alignment;
    if remainder == 0 {
        0
    } else {
        alignment - remainder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assign_stream_id() {
        assert_eq!(
            PesPacketBuilder::assign_stream_id(CodecId::Av1, 0),
            STREAM_ID_VIDEO
        );
        assert_eq!(
            PesPacketBuilder::assign_stream_id(CodecId::Av1, 1),
            STREAM_ID_VIDEO + 1
        );
        assert_eq!(
            PesPacketBuilder::assign_stream_id(CodecId::Opus, 0),
            STREAM_ID_AUDIO
        );
    }

    #[test]
    fn test_build_pes_packet_no_timestamps() {
        let builder = PesPacketBuilder::new(CodecId::Av1, 0);
        let payload = vec![0x01, 0x02, 0x03, 0x04];

        let pes_packet = builder.build(&payload).expect("operation should succeed");

        // Check start code
        assert_eq!(&pes_packet[0..3], PES_START_CODE_PREFIX);

        // Check stream ID
        assert_eq!(pes_packet[3], STREAM_ID_VIDEO);

        // Check packet length (payload + 3 header bytes)
        let packet_length = (u16::from(pes_packet[4]) << 8) | u16::from(pes_packet[5]);
        assert_eq!(packet_length, 7); // 3 (header) + 4 (payload)

        // Check payload
        assert_eq!(&pes_packet[9..], &payload);
    }

    #[test]
    fn test_build_pes_packet_with_pts() {
        let builder = PesPacketBuilder::new(CodecId::Av1, 0).with_pts(90000);
        let payload = vec![0x01, 0x02, 0x03];

        let pes_packet = builder.build(&payload).expect("operation should succeed");

        // Check flags
        assert_eq!(pes_packet[7] & 0xC0, 0x80); // PTS only

        // Check header data length
        assert_eq!(pes_packet[8], 5); // 5 bytes for PTS

        // PTS should be at offset 9-13
        assert!(pes_packet.len() >= 14);
    }

    #[test]
    fn test_build_pes_packet_with_pts_dts() {
        let builder = PesPacketBuilder::new(CodecId::Av1, 0)
            .with_pts(90000)
            .with_dts(89000);
        let payload = vec![0x01, 0x02, 0x03];

        let pes_packet = builder.build(&payload).expect("operation should succeed");

        // Check flags
        assert_eq!(pes_packet[7] & 0xC0, 0xC0); // Both PTS and DTS

        // Check header data length
        assert_eq!(pes_packet[8], 10); // 10 bytes for PTS + DTS
    }

    #[test]
    fn test_pes_header_size() {
        assert_eq!(pes_header_size(false, false), 9);
        assert_eq!(pes_header_size(true, false), 14); // 9 + 5
        assert_eq!(pes_header_size(true, true), 19); // 9 + 10
    }

    #[test]
    fn test_calculate_stuffing() {
        assert_eq!(calculate_stuffing(188, 188), 0);
        assert_eq!(calculate_stuffing(180, 188), 8);
        assert_eq!(calculate_stuffing(100, 188), 88);
        assert_eq!(calculate_stuffing(0, 188), 0);
    }

    #[test]
    fn test_build_chunked() {
        let builder = PesPacketBuilder::new(CodecId::Av1, 0).with_pts(90000);
        let payload = vec![0u8; 1000];

        let packets = builder
            .build_chunked(&payload, 300)
            .expect("operation should succeed");

        // Should create 4 chunks: 300, 300, 300, 100
        assert_eq!(packets.len(), 4);

        // Only first packet should have PTS
        assert_eq!(packets[0][8], 5); // Header data length = 5 (PTS only)
        assert_eq!(packets[1][8], 0); // No PTS/DTS
    }

    #[test]
    fn test_build_pes_packet_too_large() {
        let builder = PesPacketBuilder::new(CodecId::Av1, 0);
        let payload = vec![0u8; MAX_PES_PAYLOAD_SIZE + 1];

        let result = builder.build(&payload);
        assert!(result.is_err());
    }
}
