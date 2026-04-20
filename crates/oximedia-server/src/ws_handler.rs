//! WebSocket frame parsing, ping/pong, connection lifecycle, and broadcast.
//!
//! This module provides types and logic for managing WebSocket connections,
//! parsing frames, handling control frames, and broadcasting messages to
//! multiple subscribers.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// WebSocket frame opcode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    /// Continuation frame
    Continuation = 0x0,
    /// Text frame (UTF-8)
    Text = 0x1,
    /// Binary frame
    Binary = 0x2,
    /// Connection close frame
    Close = 0x8,
    /// Ping frame
    Ping = 0x9,
    /// Pong frame
    Pong = 0xA,
}

impl Opcode {
    /// Returns true if this is a control frame (close, ping, pong)
    pub fn is_control(self) -> bool {
        matches!(self, Opcode::Close | Opcode::Ping | Opcode::Pong)
    }

    /// Returns true if this is a data frame (text, binary, continuation)
    pub fn is_data(self) -> bool {
        !self.is_control()
    }

    /// Attempts to create an opcode from a raw nibble value
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x0 => Some(Opcode::Continuation),
            0x1 => Some(Opcode::Text),
            0x2 => Some(Opcode::Binary),
            0x8 => Some(Opcode::Close),
            0x9 => Some(Opcode::Ping),
            0xA => Some(Opcode::Pong),
            _ => None,
        }
    }
}

/// A WebSocket frame
#[derive(Debug, Clone)]
pub struct WsFrame {
    /// Whether this is the final fragment
    pub fin: bool,
    /// Frame opcode
    pub opcode: Opcode,
    /// Payload data
    pub payload: Vec<u8>,
    /// Whether the payload is masked (client → server)
    pub masked: bool,
    /// Masking key (4 bytes, only meaningful when `masked == true`)
    pub mask_key: Option<[u8; 4]>,
}

impl WsFrame {
    /// Creates a new text frame
    pub fn text(data: impl Into<String>) -> Self {
        let bytes = data.into().into_bytes();
        Self {
            fin: true,
            opcode: Opcode::Text,
            payload: bytes,
            masked: false,
            mask_key: None,
        }
    }

    /// Creates a new binary frame
    pub fn binary(data: Vec<u8>) -> Self {
        Self {
            fin: true,
            opcode: Opcode::Binary,
            payload: data,
            masked: false,
            mask_key: None,
        }
    }

    /// Creates a ping frame with optional application data
    pub fn ping(data: Vec<u8>) -> Self {
        Self {
            fin: true,
            opcode: Opcode::Ping,
            payload: data,
            masked: false,
            mask_key: None,
        }
    }

    /// Creates a pong frame echoing the provided data
    pub fn pong(data: Vec<u8>) -> Self {
        Self {
            fin: true,
            opcode: Opcode::Pong,
            payload: data,
            masked: false,
            mask_key: None,
        }
    }

    /// Creates a close frame with optional status code and reason
    pub fn close(code: u16, reason: impl Into<String>) -> Self {
        let reason_bytes = reason.into().into_bytes();
        let mut payload = Vec::with_capacity(2 + reason_bytes.len());
        payload.push((code >> 8) as u8);
        payload.push((code & 0xFF) as u8);
        payload.extend_from_slice(&reason_bytes);
        Self {
            fin: true,
            opcode: Opcode::Close,
            payload,
            masked: false,
            mask_key: None,
        }
    }

    /// Applies (or removes) the XOR mask to/from the payload in-place
    pub fn apply_mask(&mut self) {
        if let Some(key) = self.mask_key {
            for (i, byte) in self.payload.iter_mut().enumerate() {
                *byte ^= key[i % 4];
            }
        }
    }

    /// Returns the payload as a UTF-8 string if this is a text frame
    pub fn as_text(&self) -> Option<&str> {
        if self.opcode == Opcode::Text {
            std::str::from_utf8(&self.payload).ok()
        } else {
            None
        }
    }

    /// Returns the close status code (first 2 bytes of payload), if present
    pub fn close_code(&self) -> Option<u16> {
        if self.opcode == Opcode::Close && self.payload.len() >= 2 {
            Some(((self.payload[0] as u16) << 8) | (self.payload[1] as u16))
        } else {
            None
        }
    }

    /// Encodes the frame into a byte buffer (server-to-client, unmasked)
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let first_byte = if self.fin { 0x80 } else { 0x00 } | (self.opcode as u8);
        buf.push(first_byte);

        let len = self.payload.len();
        if len <= 125 {
            buf.push(len as u8);
        } else if len <= 65535 {
            buf.push(126);
            buf.push((len >> 8) as u8);
            buf.push((len & 0xFF) as u8);
        } else {
            buf.push(127);
            for shift in (0..8).rev() {
                buf.push(((len >> (shift * 8)) & 0xFF) as u8);
            }
        }
        buf.extend_from_slice(&self.payload);
        buf
    }
}

/// WebSocket connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Handshake in progress
    Connecting,
    /// Fully open
    Open,
    /// Close handshake initiated
    Closing,
    /// Connection is closed
    Closed,
}

/// Statistics for a WebSocket connection
#[derive(Debug, Clone)]
pub struct ConnectionStats {
    /// Number of messages sent
    pub messages_sent: u64,
    /// Number of messages received
    pub messages_received: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Number of ping/pong round trips
    pub ping_pong_count: u64,
    /// Last ping round-trip time
    pub last_rtt: Option<Duration>,
    /// Connection uptime
    pub connected_at: Instant,
}

impl ConnectionStats {
    fn new() -> Self {
        Self {
            messages_sent: 0,
            messages_received: 0,
            bytes_sent: 0,
            bytes_received: 0,
            ping_pong_count: 0,
            last_rtt: None,
            connected_at: Instant::now(),
        }
    }

    /// Returns uptime as a Duration
    pub fn uptime(&self) -> Duration {
        self.connected_at.elapsed()
    }
}

/// A single managed WebSocket connection
pub struct WsConnection {
    /// Unique connection ID
    pub id: String,
    /// Current connection state
    pub state: ConnectionState,
    /// Per-connection statistics
    pub stats: ConnectionStats,
    /// Pending outbound frames
    pub outbound: Vec<WsFrame>,
    /// Subscribed topic channels
    pub subscriptions: Vec<String>,
    /// Pending ping timestamp (for RTT measurement)
    pending_ping: Option<Instant>,
}

impl WsConnection {
    /// Creates a new connection with the given ID
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            state: ConnectionState::Connecting,
            stats: ConnectionStats::new(),
            outbound: Vec::new(),
            subscriptions: Vec::new(),
            pending_ping: None,
        }
    }

    /// Marks the connection as open
    pub fn open(&mut self) {
        self.state = ConnectionState::Open;
    }

    /// Enqueues a frame for sending and updates stats
    pub fn send(&mut self, frame: WsFrame) {
        let len = frame.payload.len() as u64;
        self.outbound.push(frame);
        self.stats.messages_sent += 1;
        self.stats.bytes_sent += len;
    }

    /// Records a received frame and updates stats
    pub fn receive(&mut self, frame: &WsFrame) {
        self.stats.messages_received += 1;
        self.stats.bytes_received += frame.payload.len() as u64;
    }

    /// Sends a ping and records the time
    pub fn send_ping(&mut self, data: Vec<u8>) {
        self.pending_ping = Some(Instant::now());
        self.send(WsFrame::ping(data));
    }

    /// Records a pong reception and computes RTT
    pub fn receive_pong(&mut self) {
        if let Some(sent_at) = self.pending_ping.take() {
            self.stats.last_rtt = Some(sent_at.elapsed());
            self.stats.ping_pong_count += 1;
        }
    }

    /// Initiates the close handshake
    pub fn close(&mut self, code: u16, reason: &str) {
        self.state = ConnectionState::Closing;
        self.send(WsFrame::close(code, reason));
    }

    /// Subscribes to a topic channel
    pub fn subscribe(&mut self, topic: impl Into<String>) {
        let t = topic.into();
        if !self.subscriptions.contains(&t) {
            self.subscriptions.push(t);
        }
    }

    /// Unsubscribes from a topic channel
    pub fn unsubscribe(&mut self, topic: &str) {
        self.subscriptions.retain(|t| t != topic);
    }

    /// Returns true if the connection is subscribed to the given topic
    pub fn is_subscribed(&self, topic: &str) -> bool {
        self.subscriptions.iter().any(|t| t == topic)
    }
}

/// Broadcast hub for sending messages to groups of connections
pub struct BroadcastHub {
    connections: Arc<Mutex<HashMap<String, WsConnection>>>,
}

impl BroadcastHub {
    /// Creates a new empty broadcast hub
    pub fn new() -> Self {
        Self {
            connections: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Registers a new connection
    pub fn register(&self, conn: WsConnection) {
        let mut map = self.connections.lock().unwrap_or_else(|e| e.into_inner());
        map.insert(conn.id.clone(), conn);
    }

    /// Removes a connection by ID
    pub fn remove(&self, id: &str) {
        let mut map = self.connections.lock().unwrap_or_else(|e| e.into_inner());
        map.remove(id);
    }

    /// Returns the number of active connections
    pub fn connection_count(&self) -> usize {
        self.connections
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Broadcasts a frame to all connections subscribed to a topic.
    /// Returns the count of connections that received the frame.
    pub fn broadcast_to_topic(&self, topic: &str, frame: WsFrame) -> usize {
        let mut map = self.connections.lock().unwrap_or_else(|e| e.into_inner());
        let mut count = 0;
        for conn in map.values_mut() {
            if conn.state == ConnectionState::Open && conn.is_subscribed(topic) {
                conn.send(frame.clone());
                count += 1;
            }
        }
        count
    }

    /// Broadcasts a frame to all open connections.
    /// Returns the count of connections that received the frame.
    pub fn broadcast_all(&self, frame: WsFrame) -> usize {
        let mut map = self.connections.lock().unwrap_or_else(|e| e.into_inner());
        let mut count = 0;
        for conn in map.values_mut() {
            if conn.state == ConnectionState::Open {
                conn.send(frame.clone());
                count += 1;
            }
        }
        count
    }

    /// Sends a frame to a specific connection by ID
    pub fn send_to(&self, id: &str, frame: WsFrame) -> bool {
        let mut map = self.connections.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(conn) = map.get_mut(id) {
            if conn.state == ConnectionState::Open {
                conn.send(frame);
                return true;
            }
        }
        false
    }

    /// Returns IDs of all connections subscribed to a topic
    pub fn subscribers(&self, topic: &str) -> Vec<String> {
        let map = self.connections.lock().unwrap_or_else(|e| e.into_inner());
        map.values()
            .filter(|c| c.is_subscribed(topic))
            .map(|c| c.id.clone())
            .collect()
    }
}

impl Default for BroadcastHub {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opcode_is_control() {
        assert!(Opcode::Close.is_control());
        assert!(Opcode::Ping.is_control());
        assert!(Opcode::Pong.is_control());
        assert!(!Opcode::Text.is_control());
        assert!(!Opcode::Binary.is_control());
    }

    #[test]
    fn test_opcode_from_u8() {
        assert_eq!(Opcode::from_u8(0x1), Some(Opcode::Text));
        assert_eq!(Opcode::from_u8(0x8), Some(Opcode::Close));
        assert_eq!(Opcode::from_u8(0xF), None);
    }

    #[test]
    fn test_ws_frame_text() {
        let frame = WsFrame::text("hello");
        assert_eq!(frame.opcode, Opcode::Text);
        assert!(frame.fin);
        assert_eq!(frame.as_text(), Some("hello"));
    }

    #[test]
    fn test_ws_frame_binary() {
        let frame = WsFrame::binary(vec![1, 2, 3]);
        assert_eq!(frame.opcode, Opcode::Binary);
        assert_eq!(frame.payload, vec![1, 2, 3]);
    }

    #[test]
    fn test_ws_frame_ping_pong() {
        let ping = WsFrame::ping(b"ping-data".to_vec());
        let pong = WsFrame::pong(b"ping-data".to_vec());
        assert_eq!(ping.opcode, Opcode::Ping);
        assert_eq!(pong.opcode, Opcode::Pong);
    }

    #[test]
    fn test_ws_frame_close_code() {
        let frame = WsFrame::close(1000, "Normal closure");
        assert_eq!(frame.opcode, Opcode::Close);
        assert_eq!(frame.close_code(), Some(1000));
    }

    #[test]
    fn test_ws_frame_mask_unmask() {
        let payload = b"Hello".to_vec();
        let mut frame = WsFrame::binary(payload.clone());
        frame.mask_key = Some([0xDE, 0xAD, 0xBE, 0xEF]);
        frame.masked = true;
        frame.apply_mask();
        // payload should now be XOR'd
        assert_ne!(frame.payload, payload);
        // applying mask again should restore original
        frame.apply_mask();
        assert_eq!(frame.payload, payload);
    }

    #[test]
    fn test_ws_frame_encode_small() {
        let frame = WsFrame::text("hi");
        let encoded = frame.encode();
        // First byte: fin(1) + opcode(1) = 0x81; second byte: length 2
        assert_eq!(encoded[0], 0x81);
        assert_eq!(encoded[1], 2);
        assert_eq!(&encoded[2..], b"hi");
    }

    #[test]
    fn test_ws_connection_lifecycle() {
        let mut conn = WsConnection::new("conn-1");
        assert_eq!(conn.state, ConnectionState::Connecting);
        conn.open();
        assert_eq!(conn.state, ConnectionState::Open);
        conn.close(1000, "bye");
        assert_eq!(conn.state, ConnectionState::Closing);
    }

    #[test]
    fn test_ws_connection_send_receive_stats() {
        let mut conn = WsConnection::new("conn-2");
        conn.open();
        conn.send(WsFrame::text("hello world"));
        assert_eq!(conn.stats.messages_sent, 1);
        assert_eq!(conn.stats.bytes_sent, 11);
        let frame = WsFrame::binary(vec![0u8; 5]);
        conn.receive(&frame);
        assert_eq!(conn.stats.messages_received, 1);
        assert_eq!(conn.stats.bytes_received, 5);
    }

    #[test]
    fn test_ws_connection_ping_pong_rtt() {
        let mut conn = WsConnection::new("conn-3");
        conn.open();
        conn.send_ping(b"test".to_vec());
        assert!(conn.pending_ping.is_some());
        conn.receive_pong();
        assert!(conn.pending_ping.is_none());
        assert!(conn.stats.last_rtt.is_some());
        assert_eq!(conn.stats.ping_pong_count, 1);
    }

    #[test]
    fn test_ws_connection_subscriptions() {
        let mut conn = WsConnection::new("conn-4");
        conn.subscribe("jobs");
        conn.subscribe("alerts");
        assert!(conn.is_subscribed("jobs"));
        assert!(conn.is_subscribed("alerts"));
        assert!(!conn.is_subscribed("events"));
        conn.unsubscribe("jobs");
        assert!(!conn.is_subscribed("jobs"));
    }

    #[test]
    fn test_broadcast_hub_registration() {
        let hub = BroadcastHub::new();
        let mut c = WsConnection::new("c1");
        c.open();
        hub.register(c);
        assert_eq!(hub.connection_count(), 1);
        hub.remove("c1");
        assert_eq!(hub.connection_count(), 0);
    }

    #[test]
    fn test_broadcast_hub_topic_broadcast() {
        let hub = BroadcastHub::new();
        for i in 0..5 {
            let mut c = WsConnection::new(format!("c{i}"));
            c.open();
            if i % 2 == 0 {
                c.subscribe("topic-a");
            }
            hub.register(c);
        }
        let count = hub.broadcast_to_topic("topic-a", WsFrame::text("hello"));
        assert_eq!(count, 3); // c0, c2, c4
    }

    #[test]
    fn test_broadcast_hub_broadcast_all() {
        let hub = BroadcastHub::new();
        for i in 0..4 {
            let mut c = WsConnection::new(format!("c{i}"));
            c.open();
            hub.register(c);
        }
        let count = hub.broadcast_all(WsFrame::text("broadcast"));
        assert_eq!(count, 4);
    }

    #[test]
    fn test_broadcast_hub_send_to() {
        let hub = BroadcastHub::new();
        let mut c = WsConnection::new("target");
        c.open();
        hub.register(c);
        let ok = hub.send_to("target", WsFrame::text("direct message"));
        assert!(ok);
        let not_ok = hub.send_to("missing", WsFrame::text("nope"));
        assert!(!not_ok);
    }

    #[test]
    fn test_broadcast_hub_subscribers() {
        let hub = BroadcastHub::new();
        for i in 0..3 {
            let mut c = WsConnection::new(format!("sub{i}"));
            c.open();
            c.subscribe("news");
            hub.register(c);
        }
        let mut subs = hub.subscribers("news");
        subs.sort();
        assert_eq!(subs, vec!["sub0", "sub1", "sub2"]);
    }

    #[test]
    fn test_connection_stats_uptime() {
        let stats = ConnectionStats::new();
        // uptime() returns a Duration, which is always non-negative
        let _uptime = stats.uptime();
    }
}
