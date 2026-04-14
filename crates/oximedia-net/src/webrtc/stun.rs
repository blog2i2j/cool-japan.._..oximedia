//! STUN (Session Traversal Utilities for NAT) protocol implementation.
//!
//! This module implements STUN messages for ICE candidate gathering
//! and connectivity checks.

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use crate::error::{NetError, NetResult};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use hmac::{Hmac, KeyInit, Mac};
use sha1::Sha1;
use std::net::SocketAddr;

/// STUN message type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    /// Binding request.
    BindingRequest,
    /// Binding success response.
    BindingResponse,
    /// Binding error response.
    BindingError,
    /// Binding indication.
    BindingIndication,
}

impl MessageType {
    /// Returns the message type value.
    #[must_use]
    pub const fn value(&self) -> u16 {
        match self {
            Self::BindingRequest => 0x0001,
            Self::BindingResponse => 0x0101,
            Self::BindingError => 0x0111,
            Self::BindingIndication => 0x0011,
        }
    }

    /// Parses from value.
    #[must_use]
    pub const fn from_value(value: u16) -> Option<Self> {
        match value {
            0x0001 => Some(Self::BindingRequest),
            0x0101 => Some(Self::BindingResponse),
            0x0111 => Some(Self::BindingError),
            0x0011 => Some(Self::BindingIndication),
            _ => None,
        }
    }
}

/// STUN attribute type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeType {
    /// Mapped address.
    MappedAddress,
    /// XOR mapped address.
    XorMappedAddress,
    /// Username.
    Username,
    /// Message integrity.
    MessageIntegrity,
    /// Fingerprint.
    Fingerprint,
    /// Error code.
    ErrorCode,
    /// Realm.
    Realm,
    /// Nonce.
    Nonce,
    /// Unknown attributes.
    UnknownAttributes,
    /// Software.
    Software,
    /// Alternate server.
    AlternateServer,
    /// Priority.
    Priority,
    /// Use candidate.
    UseCandidate,
    /// Ice controlled.
    IceControlled,
    /// Ice controlling.
    IceControlling,
}

impl AttributeType {
    /// Returns the attribute type value.
    #[must_use]
    pub const fn value(&self) -> u16 {
        match self {
            Self::MappedAddress => 0x0001,
            Self::XorMappedAddress => 0x0020,
            Self::Username => 0x0006,
            Self::MessageIntegrity => 0x0008,
            Self::Fingerprint => 0x8028,
            Self::ErrorCode => 0x0009,
            Self::Realm => 0x0014,
            Self::Nonce => 0x0015,
            Self::UnknownAttributes => 0x000A,
            Self::Software => 0x8022,
            Self::AlternateServer => 0x8023,
            Self::Priority => 0x0024,
            Self::UseCandidate => 0x0025,
            Self::IceControlled => 0x8029,
            Self::IceControlling => 0x802A,
        }
    }

    /// Parses from value.
    #[must_use]
    pub const fn from_value(value: u16) -> Option<Self> {
        match value {
            0x0001 => Some(Self::MappedAddress),
            0x0020 => Some(Self::XorMappedAddress),
            0x0006 => Some(Self::Username),
            0x0008 => Some(Self::MessageIntegrity),
            0x8028 => Some(Self::Fingerprint),
            0x0009 => Some(Self::ErrorCode),
            0x0014 => Some(Self::Realm),
            0x0015 => Some(Self::Nonce),
            0x000A => Some(Self::UnknownAttributes),
            0x8022 => Some(Self::Software),
            0x8023 => Some(Self::AlternateServer),
            0x0024 => Some(Self::Priority),
            0x0025 => Some(Self::UseCandidate),
            0x8029 => Some(Self::IceControlled),
            0x802A => Some(Self::IceControlling),
            _ => None,
        }
    }
}

/// STUN attribute.
#[derive(Debug, Clone)]
pub struct Attribute {
    /// Attribute type.
    pub attr_type: u16,
    /// Attribute value.
    pub value: Bytes,
}

impl Attribute {
    /// Creates a new attribute.
    #[must_use]
    pub fn new(attr_type: AttributeType, value: impl Into<Bytes>) -> Self {
        Self {
            attr_type: attr_type.value(),
            value: value.into(),
        }
    }

    /// Creates a XOR mapped address attribute.
    #[must_use]
    pub fn xor_mapped_address(addr: SocketAddr, transaction_id: &[u8; 12]) -> Self {
        let mut buf = BytesMut::new();

        match addr {
            SocketAddr::V4(v4) => {
                buf.put_u8(0); // Reserved
                buf.put_u8(0x01); // IPv4

                // XOR port with magic cookie (first 2 bytes)
                let port = v4.port() ^ 0x2112;
                buf.put_u16(port);

                // XOR address with magic cookie
                let ip_bytes = v4.ip().octets();
                let magic = 0x2112A442u32.to_be_bytes();
                for (i, byte) in ip_bytes.iter().enumerate() {
                    buf.put_u8(byte ^ magic[i]);
                }
            }
            SocketAddr::V6(v6) => {
                buf.put_u8(0); // Reserved
                buf.put_u8(0x02); // IPv6

                // XOR port with magic cookie (first 2 bytes)
                let port = v6.port() ^ 0x2112;
                buf.put_u16(port);

                // XOR address with magic cookie + transaction ID
                let ip_bytes = v6.ip().octets();
                let magic = 0x2112A442u32.to_be_bytes();
                let mut xor_mask = Vec::new();
                xor_mask.extend_from_slice(&magic);
                xor_mask.extend_from_slice(transaction_id);

                for (i, byte) in ip_bytes.iter().enumerate() {
                    buf.put_u8(byte ^ xor_mask[i]);
                }
            }
        }

        Self {
            attr_type: AttributeType::XorMappedAddress.value(),
            value: buf.freeze(),
        }
    }

    /// Creates a username attribute.
    #[must_use]
    pub fn username(username: impl AsRef<str>) -> Self {
        Self::new(
            AttributeType::Username,
            Bytes::from(username.as_ref().to_string()),
        )
    }

    /// Creates a priority attribute.
    #[must_use]
    pub fn priority(priority: u32) -> Self {
        let mut buf = BytesMut::new();
        buf.put_u32(priority);
        Self::new(AttributeType::Priority, buf.freeze())
    }

    /// Creates a use-candidate attribute.
    #[must_use]
    pub fn use_candidate() -> Self {
        Self::new(AttributeType::UseCandidate, Bytes::new())
    }

    /// Creates an ICE-controlling attribute.
    #[must_use]
    pub fn ice_controlling(tie_breaker: u64) -> Self {
        let mut buf = BytesMut::new();
        buf.put_u64(tie_breaker);
        Self::new(AttributeType::IceControlling, buf.freeze())
    }

    /// Creates an ICE-controlled attribute.
    #[must_use]
    pub fn ice_controlled(tie_breaker: u64) -> Self {
        let mut buf = BytesMut::new();
        buf.put_u64(tie_breaker);
        Self::new(AttributeType::IceControlled, buf.freeze())
    }

    /// Encodes the attribute to bytes.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::new();
        buf.put_u16(self.attr_type);
        buf.put_u16(self.value.len() as u16);
        buf.put(self.value.clone());

        // Add padding to 4-byte boundary
        let padding = (4 - (self.value.len() % 4)) % 4;
        for _ in 0..padding {
            buf.put_u8(0);
        }

        buf.freeze()
    }

    /// Parses XOR mapped address.
    pub fn parse_xor_mapped_address(&self, transaction_id: &[u8; 12]) -> NetResult<SocketAddr> {
        if self.value.len() < 4 {
            return Err(NetError::parse(0, "XOR-MAPPED-ADDRESS too short"));
        }

        let mut cursor = self.value.clone();
        cursor.advance(1); // Skip reserved byte
        let family = cursor.get_u8();
        let port = cursor.get_u16() ^ 0x2112;

        match family {
            0x01 => {
                // IPv4
                if cursor.remaining() < 4 {
                    return Err(NetError::parse(0, "Invalid IPv4 address"));
                }
                let magic = 0x2112A442u32.to_be_bytes();
                let mut ip = [0u8; 4];
                for (i, byte) in ip.iter_mut().enumerate() {
                    *byte = cursor.get_u8() ^ magic[i];
                }
                Ok(SocketAddr::new(std::net::IpAddr::V4(ip.into()), port))
            }
            0x02 => {
                // IPv6
                if cursor.remaining() < 16 {
                    return Err(NetError::parse(0, "Invalid IPv6 address"));
                }
                let magic = 0x2112A442u32.to_be_bytes();
                let mut xor_mask = Vec::new();
                xor_mask.extend_from_slice(&magic);
                xor_mask.extend_from_slice(transaction_id);

                let mut ip = [0u8; 16];
                for (i, byte) in ip.iter_mut().enumerate() {
                    *byte = cursor.get_u8() ^ xor_mask[i];
                }
                Ok(SocketAddr::new(std::net::IpAddr::V6(ip.into()), port))
            }
            _ => Err(NetError::parse(0, "Unknown address family")),
        }
    }
}

/// STUN message.
#[derive(Debug, Clone)]
pub struct Message {
    /// Message type.
    pub message_type: MessageType,
    /// Transaction ID (96 bits).
    pub transaction_id: [u8; 12],
    /// Attributes.
    pub attributes: Vec<Attribute>,
}

impl Message {
    /// Magic cookie value.
    pub const MAGIC_COOKIE: u32 = 0x2112A442;

    /// Creates a new message.
    #[must_use]
    pub fn new(message_type: MessageType) -> Self {
        let mut transaction_id = [0u8; 12];
        use rand::RngExt;
        rand::rng().fill(&mut transaction_id);

        Self {
            message_type,
            transaction_id,
            attributes: Vec::new(),
        }
    }

    /// Creates a binding request.
    #[must_use]
    pub fn binding_request() -> Self {
        Self::new(MessageType::BindingRequest)
    }

    /// Creates a binding response.
    #[must_use]
    pub fn binding_response(transaction_id: [u8; 12]) -> Self {
        Self {
            message_type: MessageType::BindingResponse,
            transaction_id,
            attributes: Vec::new(),
        }
    }

    /// Adds an attribute.
    #[must_use]
    pub fn with_attribute(mut self, attr: Attribute) -> Self {
        self.attributes.push(attr);
        self
    }

    /// Gets an attribute by type.
    #[must_use]
    pub fn get_attribute(&self, attr_type: AttributeType) -> Option<&Attribute> {
        self.attributes
            .iter()
            .find(|a| a.attr_type == attr_type.value())
    }

    /// Encodes the message to bytes.
    #[must_use]
    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::new();

        // Encode attributes first to get length
        let mut attrs_buf = BytesMut::new();
        for attr in &self.attributes {
            attrs_buf.put(attr.encode());
        }

        // Message type
        buf.put_u16(self.message_type.value());

        // Message length (does not include 20-byte header)
        buf.put_u16(attrs_buf.len() as u16);

        // Magic cookie
        buf.put_u32(Self::MAGIC_COOKIE);

        // Transaction ID
        buf.put_slice(&self.transaction_id);

        // Attributes
        buf.put(attrs_buf);

        buf.freeze()
    }

    /// Encodes with message integrity.
    pub fn encode_with_integrity(&self, password: &str) -> Bytes {
        // Create message with all attributes except MESSAGE-INTEGRITY
        let mut msg = self.clone();
        msg.attributes
            .retain(|a| a.attr_type != AttributeType::MessageIntegrity.value());

        // Encode message
        let encoded = msg.encode();

        // Calculate HMAC-SHA1
        let mut mac = Hmac::<Sha1>::new_from_slice(password.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(&encoded);
        let result = mac.finalize();
        let hmac_bytes = result.into_bytes();

        // Add MESSAGE-INTEGRITY attribute
        let mut buf = BytesMut::from(encoded.as_ref());

        // Update length to include MESSAGE-INTEGRITY
        let new_length = buf.len() - 20 + 4 + 20; // +4 for attr header, +20 for HMAC
        buf[2..4].copy_from_slice(&(new_length as u16).to_be_bytes());

        // Add MESSAGE-INTEGRITY
        buf.put_u16(AttributeType::MessageIntegrity.value());
        buf.put_u16(20); // HMAC-SHA1 is 20 bytes
        buf.put_slice(&hmac_bytes);

        buf.freeze()
    }

    /// Parses a STUN message.
    pub fn parse(data: &[u8]) -> NetResult<Self> {
        if data.len() < 20 {
            return Err(NetError::parse(0, "Message too short"));
        }

        let mut cursor = Bytes::copy_from_slice(data);

        // Parse header
        let message_type = cursor.get_u16();
        let message_type = MessageType::from_value(message_type)
            .ok_or_else(|| NetError::parse(0, "Unknown message type"))?;

        let length = cursor.get_u16() as usize;
        let magic = cursor.get_u32();

        if magic != Self::MAGIC_COOKIE {
            return Err(NetError::parse(4, "Invalid magic cookie"));
        }

        let mut transaction_id = [0u8; 12];
        cursor.copy_to_slice(&mut transaction_id);

        // Parse attributes
        let mut attributes = Vec::new();
        let mut parsed = 0;

        while parsed < length && cursor.remaining() >= 4 {
            let attr_type = cursor.get_u16();
            let attr_length = cursor.get_u16() as usize;

            if cursor.remaining() < attr_length {
                return Err(NetError::parse(parsed as u64, "Attribute value too short"));
            }

            let value = cursor.copy_to_bytes(attr_length);

            attributes.push(Attribute { attr_type, value });

            // Skip padding
            let padding = (4 - (attr_length % 4)) % 4;
            cursor.advance(padding.min(cursor.remaining()));

            parsed += 4 + attr_length + padding;
        }

        Ok(Self {
            message_type,
            transaction_id,
            attributes,
        })
    }

    /// Verifies message integrity.
    pub fn verify_integrity(&self, password: &str) -> bool {
        let integrity_attr = self.get_attribute(AttributeType::MessageIntegrity);
        if let Some(attr) = integrity_attr {
            if attr.value.len() != 20 {
                return false;
            }

            // Create message without MESSAGE-INTEGRITY
            let mut msg = self.clone();
            msg.attributes
                .retain(|a| a.attr_type != AttributeType::MessageIntegrity.value());

            let encoded = msg.encode();

            // Calculate HMAC
            let mut mac = Hmac::<Sha1>::new_from_slice(password.as_bytes())
                .expect("HMAC can take key of any size");
            mac.update(&encoded);

            mac.verify_slice(&attr.value).is_ok()
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_type() {
        assert_eq!(MessageType::BindingRequest.value(), 0x0001);
        assert_eq!(
            MessageType::from_value(0x0001),
            Some(MessageType::BindingRequest)
        );
    }

    #[test]
    fn test_binding_request() {
        let msg = Message::binding_request();
        assert_eq!(msg.message_type, MessageType::BindingRequest);
    }

    #[test]
    fn test_encode_decode() {
        let msg = Message::binding_request().with_attribute(Attribute::username("test:user"));

        let encoded = msg.encode();
        let decoded = Message::parse(&encoded).expect("should succeed in test");

        assert_eq!(decoded.message_type, msg.message_type);
        assert_eq!(decoded.transaction_id, msg.transaction_id);
        assert_eq!(decoded.attributes.len(), 1);
    }

    #[test]
    fn test_xor_mapped_address_v4() {
        let addr: SocketAddr = "192.168.1.100:5000"
            .parse()
            .expect("should succeed in test");
        let transaction_id = [0u8; 12];
        let attr = Attribute::xor_mapped_address(addr, &transaction_id);

        let encoded = attr.encode();
        assert!(encoded.len() >= 8);
    }

    #[test]
    fn test_priority_attribute() {
        let attr = Attribute::priority(12345);
        assert_eq!(attr.attr_type, AttributeType::Priority.value());
    }
}
