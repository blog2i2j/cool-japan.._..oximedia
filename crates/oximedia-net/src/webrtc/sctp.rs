//! SCTP (Stream Control Transmission Protocol) over DTLS.
//!
//! This module implements a simplified SCTP layer for WebRTC data channels.

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use crate::error::{NetError, NetResult};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// SCTP chunk type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ChunkType {
    /// Data chunk.
    Data = 0,
    /// Init chunk.
    Init = 1,
    /// Init ACK chunk.
    InitAck = 2,
    /// SACK chunk.
    Sack = 3,
    /// Heartbeat chunk.
    Heartbeat = 4,
    /// Heartbeat ACK chunk.
    HeartbeatAck = 5,
    /// Abort chunk.
    Abort = 6,
    /// Shutdown chunk.
    Shutdown = 7,
    /// Shutdown ACK chunk.
    ShutdownAck = 8,
    /// Error chunk.
    Error = 9,
    /// Cookie Echo chunk.
    CookieEcho = 10,
    /// Cookie ACK chunk.
    CookieAck = 11,
    /// Shutdown Complete chunk.
    ShutdownComplete = 14,
}

impl ChunkType {
    /// Parses from byte value.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Data),
            1 => Some(Self::Init),
            2 => Some(Self::InitAck),
            3 => Some(Self::Sack),
            4 => Some(Self::Heartbeat),
            5 => Some(Self::HeartbeatAck),
            6 => Some(Self::Abort),
            7 => Some(Self::Shutdown),
            8 => Some(Self::ShutdownAck),
            9 => Some(Self::Error),
            10 => Some(Self::CookieEcho),
            11 => Some(Self::CookieAck),
            14 => Some(Self::ShutdownComplete),
            _ => None,
        }
    }
}

/// SCTP chunk.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Chunk type.
    pub chunk_type: u8,
    /// Chunk flags.
    pub flags: u8,
    /// Chunk data.
    pub data: Bytes,
}

impl Chunk {
    /// Creates a new chunk.
    #[must_use]
    pub fn new(chunk_type: ChunkType, flags: u8, data: impl Into<Bytes>) -> Self {
        Self {
            chunk_type: chunk_type as u8,
            flags,
            data: data.into(),
        }
    }

    /// Encodes the chunk.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::new();

        buf.put_u8(self.chunk_type);
        buf.put_u8(self.flags);
        buf.put_u16((self.data.len() + 4) as u16);
        buf.put(self.data.clone());

        // Padding to 4-byte boundary
        let padding = (4 - ((self.data.len() + 4) % 4)) % 4;
        for _ in 0..padding {
            buf.put_u8(0);
        }

        buf.freeze()
    }

    /// Parses a chunk.
    pub fn parse(data: &mut Bytes) -> NetResult<Self> {
        if data.remaining() < 4 {
            return Err(NetError::parse(0, "Chunk too short"));
        }

        let chunk_type = data.get_u8();
        let flags = data.get_u8();
        let length = data.get_u16() as usize;

        if length < 4 {
            return Err(NetError::parse(0, "Invalid chunk length"));
        }

        let data_len = length - 4;
        if data.remaining() < data_len {
            return Err(NetError::parse(0, "Incomplete chunk data"));
        }

        let chunk_data = data.copy_to_bytes(data_len);

        // Skip padding
        let padding = (4 - (length % 4)) % 4;
        data.advance(padding.min(data.remaining()));

        Ok(Self {
            chunk_type,
            flags,
            data: chunk_data,
        })
    }
}

/// SCTP packet.
#[derive(Debug, Clone)]
pub struct Packet {
    /// Source port.
    pub source_port: u16,
    /// Destination port.
    pub dest_port: u16,
    /// Verification tag.
    pub verification_tag: u32,
    /// Chunks.
    pub chunks: Vec<Chunk>,
}

impl Packet {
    /// Creates a new packet.
    #[must_use]
    pub fn new(source_port: u16, dest_port: u16, verification_tag: u32) -> Self {
        Self {
            source_port,
            dest_port,
            verification_tag,
            chunks: Vec::new(),
        }
    }

    /// Adds a chunk.
    #[must_use]
    pub fn with_chunk(mut self, chunk: Chunk) -> Self {
        self.chunks.push(chunk);
        self
    }

    /// Encodes the packet.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::new();

        buf.put_u16(self.source_port);
        buf.put_u16(self.dest_port);
        buf.put_u32(self.verification_tag);
        buf.put_u32(0); // Checksum placeholder

        // Encode chunks
        for chunk in &self.chunks {
            buf.put(chunk.encode());
        }

        // Calculate checksum (CRC32c)
        let checksum = calculate_crc32c(&buf);
        buf[8..12].copy_from_slice(&checksum.to_le_bytes());

        buf.freeze()
    }

    /// Parses a packet.
    pub fn parse(data: &[u8]) -> NetResult<Self> {
        if data.len() < 12 {
            return Err(NetError::parse(0, "Packet too short"));
        }

        let mut cursor = Bytes::copy_from_slice(data);

        let source_port = cursor.get_u16();
        let dest_port = cursor.get_u16();
        let verification_tag = cursor.get_u32();
        let received_checksum = cursor.get_u32();

        // Verify CRC32c checksum.
        // The checksum field in the header must be zeroed before calculating.
        let mut check_data = data.to_vec();
        check_data[8] = 0;
        check_data[9] = 0;
        check_data[10] = 0;
        check_data[11] = 0;
        let expected = calculate_crc32c(&check_data);
        if received_checksum != 0 && received_checksum != expected {
            return Err(NetError::parse(0, "SCTP checksum mismatch"));
        }

        let mut chunks = Vec::new();
        while cursor.remaining() >= 4 {
            let chunk = Chunk::parse(&mut cursor)?;
            chunks.push(chunk);
        }

        Ok(Self {
            source_port,
            dest_port,
            verification_tag,
            chunks,
        })
    }
}

/// SCTP association state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssociationState {
    /// Closed.
    Closed,
    /// Cookie wait.
    CookieWait,
    /// Cookie echoed.
    CookieEchoed,
    /// Established.
    Established,
    /// Shutdown pending.
    ShutdownPending,
    /// Shutdown sent.
    ShutdownSent,
    /// Shutdown received.
    ShutdownReceived,
    /// Shutdown ACK sent.
    ShutdownAckSent,
}

/// SCTP association.
pub struct Association {
    /// Local port.
    local_port: u16,
    /// Remote port.
    remote_port: u16,
    /// Local verification tag.
    local_tag: u32,
    /// Remote verification tag.
    remote_tag: Arc<Mutex<u32>>,
    /// Association state.
    state: Arc<Mutex<AssociationState>>,
    /// Stream data.
    streams: Arc<Mutex<HashMap<u16, Vec<Bytes>>>>,
    /// Next TSN to send.
    next_tsn: Arc<Mutex<u32>>,
}

impl Association {
    /// Creates a new association.
    #[must_use]
    pub fn new(local_port: u16, remote_port: u16) -> Self {
        use rand::RngExt;
        let local_tag = rand::rng().random::<u32>();

        Self {
            local_port,
            remote_port,
            local_tag,
            remote_tag: Arc::new(Mutex::new(0)),
            state: Arc::new(Mutex::new(AssociationState::Closed)),
            streams: Arc::new(Mutex::new(HashMap::new())),
            next_tsn: Arc::new(Mutex::new(1)),
        }
    }

    /// Initiates the association.
    #[must_use]
    pub fn init(&self) -> Packet {
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = AssociationState::CookieWait;

        let mut init_data = BytesMut::new();
        init_data.put_u32(self.local_tag); // Initiate tag
        init_data.put_u32(65535); // A_rwnd
        init_data.put_u16(65535); // Number of outbound streams
        init_data.put_u16(65535); // Number of inbound streams
        init_data.put_u32(*self.next_tsn.lock().unwrap_or_else(|e| e.into_inner())); // Initial TSN

        let chunk = Chunk::new(ChunkType::Init, 0, init_data.freeze());

        Packet::new(self.local_port, self.remote_port, 0).with_chunk(chunk)
    }

    /// Handles received packet.
    pub fn handle_packet(&self, packet: Packet) -> NetResult<Option<Packet>> {
        for chunk in &packet.chunks {
            match ChunkType::from_u8(chunk.chunk_type) {
                Some(ChunkType::Init) => {
                    return Ok(Some(self.handle_init(chunk)?));
                }
                Some(ChunkType::InitAck) => {
                    self.handle_init_ack(chunk)?;
                }
                Some(ChunkType::CookieEcho) => {
                    return Ok(Some(self.handle_cookie_echo()?));
                }
                Some(ChunkType::CookieAck) => {
                    self.handle_cookie_ack()?;
                }
                Some(ChunkType::Data) => {
                    self.handle_data(chunk)?;
                }
                Some(ChunkType::Sack) => {
                    // Handle SACK
                }
                _ => {
                    // Unknown chunk
                }
            }
        }

        Ok(None)
    }

    /// Handles INIT chunk.
    fn handle_init(&self, chunk: &Chunk) -> NetResult<Packet> {
        if chunk.data.len() < 16 {
            return Err(NetError::parse(0, "Invalid INIT chunk"));
        }

        let mut cursor = chunk.data.clone();
        let remote_tag = cursor.get_u32();
        self.set_remote_tag(remote_tag);

        // Create INIT-ACK
        let mut init_ack_data = BytesMut::new();
        init_ack_data.put_u32(self.local_tag);
        init_ack_data.put_u32(65535); // A_rwnd
        init_ack_data.put_u16(65535); // Outbound streams
        init_ack_data.put_u16(65535); // Inbound streams
        init_ack_data.put_u32(*self.next_tsn.lock().unwrap_or_else(|e| e.into_inner()));

        let chunk = Chunk::new(ChunkType::InitAck, 0, init_ack_data.freeze());

        *self.remote_tag.lock().unwrap_or_else(|e| e.into_inner()) = remote_tag;
        Ok(Packet::new(self.local_port, self.remote_port, remote_tag).with_chunk(chunk))
    }

    /// Handles INIT-ACK chunk.
    fn handle_init_ack(&self, chunk: &Chunk) -> NetResult<()> {
        if chunk.data.len() < 16 {
            return Err(NetError::parse(0, "Invalid INIT-ACK chunk"));
        }

        let mut cursor = chunk.data.clone();
        let remote_tag = cursor.get_u32();
        self.set_remote_tag(remote_tag);

        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = AssociationState::CookieEchoed;
        Ok(())
    }

    /// Handles COOKIE-ECHO chunk.
    fn handle_cookie_echo(&self) -> NetResult<Packet> {
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = AssociationState::Established;

        let chunk = Chunk::new(ChunkType::CookieAck, 0, Bytes::new());
        let remote_tag = *self.remote_tag.lock().unwrap_or_else(|e| e.into_inner());
        Ok(Packet::new(self.local_port, self.remote_port, remote_tag).with_chunk(chunk))
    }

    /// Handles COOKIE-ACK chunk.
    fn handle_cookie_ack(&self) -> NetResult<()> {
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = AssociationState::Established;
        Ok(())
    }

    /// Handles DATA chunk.
    fn handle_data(&self, chunk: &Chunk) -> NetResult<()> {
        if chunk.data.len() < 12 {
            return Err(NetError::parse(0, "Invalid DATA chunk"));
        }

        let mut cursor = chunk.data.clone();
        let _tsn = cursor.get_u32();
        let stream_id = cursor.get_u16();
        let _ssn = cursor.get_u16();
        let _ppid = cursor.get_u32();

        let user_data = cursor.copy_to_bytes(cursor.remaining());

        // Store data
        let mut streams = self.streams.lock().unwrap_or_else(|e| e.into_inner());
        streams.entry(stream_id).or_default().push(user_data);

        Ok(())
    }

    /// Sends data on a stream.
    #[must_use]
    pub fn send_data(&self, stream_id: u16, data: impl Into<Bytes>) -> Packet {
        let data = data.into();

        let mut tsn = self.next_tsn.lock().unwrap_or_else(|e| e.into_inner());
        let current_tsn = *tsn;
        *tsn = tsn.wrapping_add(1);
        drop(tsn);

        let mut chunk_data = BytesMut::new();
        chunk_data.put_u32(current_tsn); // TSN
        chunk_data.put_u16(stream_id); // Stream ID
        chunk_data.put_u16(0); // Stream sequence number
        chunk_data.put_u32(51); // PPID (WebRTC String)
        chunk_data.put(data);

        let chunk = Chunk::new(ChunkType::Data, 0x03, chunk_data.freeze()); // Flags: B=1, E=1

        let remote_tag = *self.remote_tag.lock().unwrap_or_else(|e| e.into_inner());
        Packet::new(self.local_port, self.remote_port, remote_tag).with_chunk(chunk)
    }

    /// Receives data from a stream.
    #[must_use]
    pub fn recv_data(&self, stream_id: u16) -> Option<Bytes> {
        let mut streams = self.streams.lock().unwrap_or_else(|e| e.into_inner());
        streams.get_mut(&stream_id).and_then(|queue| {
            if queue.is_empty() {
                None
            } else {
                Some(queue.remove(0))
            }
        })
    }

    /// Gets the association state.
    #[must_use]
    pub fn state(&self) -> AssociationState {
        *self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Sets the remote verification tag.
    fn set_remote_tag(&self, tag: u32) {
        *self.remote_tag.lock().unwrap_or_else(|e| e.into_inner()) = tag;
    }
}

/// Calculates CRC32c checksum.
fn calculate_crc32c(data: &[u8]) -> u32 {
    // Simplified CRC32c calculation
    // In production, use a proper CRC32c implementation
    let mut crc = 0xFFFF_FFFFu32;

    for byte in data {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0x82F6_3B78
            } else {
                crc >> 1
            };
        }
    }

    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_type() {
        assert_eq!(ChunkType::from_u8(0), Some(ChunkType::Data));
        assert_eq!(ChunkType::from_u8(1), Some(ChunkType::Init));
    }

    #[test]
    fn test_chunk_encode_decode() {
        let chunk = Chunk::new(ChunkType::Data, 0, vec![1, 2, 3, 4]);
        let encoded = chunk.encode();
        assert!(encoded.len() >= 8); // 4 byte header + 4 byte data
    }

    #[test]
    fn test_packet_new() {
        let packet = Packet::new(5000, 5001, 12345);
        assert_eq!(packet.source_port, 5000);
        assert_eq!(packet.dest_port, 5001);
        assert_eq!(packet.verification_tag, 12345);
    }

    #[test]
    fn test_association_new() {
        let assoc = Association::new(5000, 5001);
        assert_eq!(assoc.state(), AssociationState::Closed);
    }

    #[test]
    fn test_association_init() {
        let assoc = Association::new(5000, 5001);
        let packet = assoc.init();
        assert_eq!(assoc.state(), AssociationState::CookieWait);
        assert!(!packet.chunks.is_empty());
    }
}
