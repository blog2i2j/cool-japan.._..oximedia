//! WebRTC Data Channel implementation.
//!
//! This module provides data channel functionality over SCTP/DTLS.

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use super::dtls::DtlsConnection;
use super::sctp::{Association, AssociationState};
use crate::error::{NetError, NetResult};
use bytes::Bytes;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// Data channel state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataChannelState {
    /// Connecting.
    Connecting,
    /// Open.
    Open,
    /// Closing.
    Closing,
    /// Closed.
    Closed,
}

/// Data channel configuration.
#[derive(Debug, Clone)]
pub struct DataChannelConfig {
    /// Channel label.
    pub label: String,
    /// Ordered delivery.
    pub ordered: bool,
    /// Max packet lifetime (ms).
    pub max_packet_lifetime: Option<u32>,
    /// Max retransmits.
    pub max_retransmits: Option<u32>,
    /// Protocol.
    pub protocol: String,
    /// Negotiated.
    pub negotiated: bool,
    /// Stream ID.
    pub id: Option<u16>,
}

impl Default for DataChannelConfig {
    fn default() -> Self {
        Self {
            label: String::new(),
            ordered: true,
            max_packet_lifetime: None,
            max_retransmits: None,
            protocol: String::new(),
            negotiated: false,
            id: None,
        }
    }
}

impl DataChannelConfig {
    /// Creates a new configuration.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            ..Default::default()
        }
    }

    /// Sets ordered delivery.
    #[must_use]
    pub const fn ordered(mut self, ordered: bool) -> Self {
        self.ordered = ordered;
        self
    }

    /// Sets max packet lifetime.
    #[must_use]
    pub const fn max_packet_lifetime(mut self, lifetime: u32) -> Self {
        self.max_packet_lifetime = Some(lifetime);
        self
    }

    /// Sets max retransmits.
    #[must_use]
    pub const fn max_retransmits(mut self, retransmits: u32) -> Self {
        self.max_retransmits = Some(retransmits);
        self
    }

    /// Sets protocol.
    #[must_use]
    pub fn protocol(mut self, protocol: impl Into<String>) -> Self {
        self.protocol = protocol.into();
        self
    }

    /// Sets stream ID.
    #[must_use]
    pub const fn id(mut self, id: u16) -> Self {
        self.id = Some(id);
        self
    }
}

/// Data channel message type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageType {
    /// Text message.
    Text,
    /// Binary message.
    Binary,
}

/// Data channel message.
#[derive(Debug, Clone)]
pub struct Message {
    /// Message type.
    pub message_type: MessageType,
    /// Message data.
    pub data: Bytes,
}

impl Message {
    /// Creates a text message.
    #[must_use]
    pub fn text(data: impl Into<String>) -> Self {
        Self {
            message_type: MessageType::Text,
            data: Bytes::from(data.into()),
        }
    }

    /// Creates a binary message.
    #[must_use]
    pub fn binary(data: impl Into<Bytes>) -> Self {
        Self {
            message_type: MessageType::Binary,
            data: data.into(),
        }
    }

    /// Returns the message as a string (if text).
    #[must_use]
    pub fn as_text(&self) -> Option<String> {
        if self.message_type == MessageType::Text {
            String::from_utf8(self.data.to_vec()).ok()
        } else {
            None
        }
    }

    /// Returns the message data.
    #[must_use]
    pub fn as_bytes(&self) -> &Bytes {
        &self.data
    }
}

/// Data channel.
pub struct DataChannel {
    /// Configuration.
    config: DataChannelConfig,
    /// Stream ID.
    stream_id: u16,
    /// State.
    state: Arc<Mutex<DataChannelState>>,
    /// SCTP association.
    association: Arc<Association>,
    /// DTLS connection.
    dtls: Arc<DtlsConnection>,
    /// Message receiver.
    rx: Arc<Mutex<mpsc::UnboundedReceiver<Message>>>,
    /// Message sender.
    tx: mpsc::UnboundedSender<Message>,
}

impl DataChannel {
    /// Creates a new data channel.
    #[must_use]
    pub fn new(
        config: DataChannelConfig,
        stream_id: u16,
        association: Arc<Association>,
        dtls: Arc<DtlsConnection>,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();

        Self {
            config,
            stream_id,
            state: Arc::new(Mutex::new(DataChannelState::Connecting)),
            association,
            dtls,
            rx: Arc::new(Mutex::new(rx)),
            tx,
        }
    }

    /// Opens the data channel.
    pub async fn open(&self) -> NetResult<()> {
        // Wait for SCTP association to be established
        while self.association.state() != AssociationState::Established {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = DataChannelState::Open;
        Ok(())
    }

    /// Sends a text message.
    pub async fn send_text(&self, text: impl AsRef<str>) -> NetResult<()> {
        self.send_message(Message::text(text.as_ref())).await
    }

    /// Sends a binary message.
    pub async fn send_binary(&self, data: impl Into<Bytes>) -> NetResult<()> {
        self.send_message(Message::binary(data)).await
    }

    /// Sends a message.
    pub async fn send_message(&self, message: Message) -> NetResult<()> {
        let state = *self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state != DataChannelState::Open {
            return Err(NetError::invalid_state("Data channel not open"));
        }

        // Encode message with PPID
        let data = message.data;

        // Send via SCTP
        let packet = self.association.send_data(self.stream_id, data);
        let encoded = packet.encode();

        // Send via DTLS
        self.dtls.send(&encoded).await?;

        Ok(())
    }

    /// Receives a message.
    pub async fn recv_message(&self) -> NetResult<Message> {
        let mut rx = self.rx.lock().unwrap_or_else(|e| e.into_inner());
        rx.recv()
            .await
            .ok_or_else(|| NetError::connection("Channel closed"))
    }

    /// Polls for received data from SCTP.
    pub fn poll_recv(&self) -> Option<Message> {
        if let Some(data) = self.association.recv_data(self.stream_id) {
            // Determine message type (simplified - would check PPID)
            Some(Message::binary(data))
        } else {
            None
        }
    }

    /// Closes the data channel.
    ///
    /// Sends a zero-length data chunk on the stream to signal closure, then
    /// transitions to the Closed state.
    pub async fn close(&self) -> NetResult<()> {
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = DataChannelState::Closing;

        // Send a zero-length SCTP data chunk to signal stream closure to the peer.
        // The packet is routed through DTLS as with normal data messages.
        let close_packet = self
            .association
            .send_data(self.stream_id, bytes::Bytes::new());
        let encoded = close_packet.encode();
        // Send the close signal; ignore errors since we transition to Closed regardless.
        let _ = self.dtls.send(&encoded).await;

        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = DataChannelState::Closed;
        Ok(())
    }

    /// Gets the channel state.
    #[must_use]
    pub fn state(&self) -> DataChannelState {
        *self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Gets the channel label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.config.label
    }

    /// Gets the stream ID.
    #[must_use]
    pub const fn stream_id(&self) -> u16 {
        self.stream_id
    }

    /// Returns true if the channel is ordered.
    #[must_use]
    pub fn is_ordered(&self) -> bool {
        self.config.ordered
    }

    /// Gets the protocol.
    #[must_use]
    pub fn protocol(&self) -> &str {
        &self.config.protocol
    }
}

/// Data channel manager.
pub struct DataChannelManager {
    /// SCTP association.
    association: Arc<Association>,
    /// DTLS connection.
    dtls: Arc<DtlsConnection>,
    /// Data channels.
    channels: Arc<Mutex<Vec<Arc<DataChannel>>>>,
    /// Next stream ID.
    next_stream_id: Arc<Mutex<u16>>,
}

impl DataChannelManager {
    /// Creates a new manager.
    #[must_use]
    pub fn new(association: Arc<Association>, dtls: Arc<DtlsConnection>) -> Self {
        Self {
            association,
            dtls,
            channels: Arc::new(Mutex::new(Vec::new())),
            next_stream_id: Arc::new(Mutex::new(0)),
        }
    }

    /// Creates a new data channel.
    pub async fn create_channel(&self, config: DataChannelConfig) -> NetResult<Arc<DataChannel>> {
        let stream_id = if let Some(id) = config.id {
            id
        } else {
            let mut next_id = self
                .next_stream_id
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let id = *next_id;
            *next_id += 2; // Increment by 2 (odd for client, even for server)
            id
        };

        let channel = Arc::new(DataChannel::new(
            config,
            stream_id,
            self.association.clone(),
            self.dtls.clone(),
        ));

        self.channels
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(channel.clone());

        channel.open().await?;

        Ok(channel)
    }

    /// Gets all channels.
    #[must_use]
    pub fn channels(&self) -> Vec<Arc<DataChannel>> {
        self.channels
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Finds a channel by stream ID.
    #[must_use]
    pub fn find_channel(&self, stream_id: u16) -> Option<Arc<DataChannel>> {
        self.channels
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .find(|ch| ch.stream_id() == stream_id)
            .cloned()
    }

    /// Processes incoming SCTP packet.
    pub async fn process_packet(&self, data: &[u8]) -> NetResult<()> {
        let packet = super::sctp::Packet::parse(data)?;

        if let Some(response) = self.association.handle_packet(packet)? {
            let encoded = response.encode();
            self.dtls.send(&encoded).await?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_channel_config() {
        let config = DataChannelConfig::new("test")
            .ordered(true)
            .max_retransmits(3);

        assert_eq!(config.label, "test");
        assert!(config.ordered);
        assert_eq!(config.max_retransmits, Some(3));
    }

    #[test]
    fn test_message_text() {
        let msg = Message::text("Hello");
        assert_eq!(msg.message_type, MessageType::Text);
        assert_eq!(msg.as_text().expect("should succeed in test"), "Hello");
    }

    #[test]
    fn test_message_binary() {
        let data = vec![1, 2, 3, 4];
        let msg = Message::binary(data.clone());
        assert_eq!(msg.message_type, MessageType::Binary);
        assert_eq!(msg.as_bytes().as_ref(), &data);
    }

    #[test]
    fn test_data_channel_state() {
        assert_ne!(DataChannelState::Open, DataChannelState::Closed);
        assert_eq!(DataChannelState::Open, DataChannelState::Open);
    }
}
