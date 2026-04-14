//! QUIC datagram mode for ultra-low-latency media transport.
//!
//! QUIC datagrams (RFC 9221) provide unreliable, connectionless delivery
//! within an established QUIC connection.  Unlike QUIC streams, datagrams
//! are not retransmitted on loss and have no per-datagram ordering guarantee,
//! making them suitable for real-time media where latency matters more than
//! reliability.
//!
//! This module provides:
//!
//! - [`QuicDatagramConfig`] — maximum datagram size and congestion window
//! - [`QuicDatagram`] — a single datagram unit with sequence number and priority
//! - [`QuicDatagramSender`] — generates sequenced datagrams from raw media bytes
//! - [`should_fragment`] — predicate to check if data exceeds the max datagram size
//! - [`fragment`] — split a byte buffer into max-size chunks
//!
//! # Fragmentation
//!
//! QUIC datagrams are limited by the connection's maximum datagram size
//! (derived from the PMTU).  Media payloads that exceed this limit must be
//! fragmented before transmission.  The [`fragment`] function performs a
//! simple sequential split; reassembly is the responsibility of the receiver
//! using the `sequence` and order within the stream.
//!
//! # Example
//!
//! ```
//! use oximedia_net::quic_datagram::{
//!     fragment, should_fragment, QuicDatagramConfig, QuicDatagramSender,
//! };
//!
//! let config = QuicDatagramConfig {
//!     max_datagram_size: 1200,
//!     congestion_window: 65536,
//! };
//!
//! let large_payload = vec![0u8; 4800]; // 4× the max size
//! assert!(should_fragment(&large_payload, &config));
//!
//! let chunks = fragment(&large_payload, config.max_datagram_size);
//! assert_eq!(chunks.len(), 4);
//!
//! let mut sender = QuicDatagramSender::new(config);
//! let dg = sender.next_datagram(chunks[0].clone(), 1);
//! assert_eq!(dg.sequence, 0);
//! assert_eq!(dg.stream_id, 1);
//! ```

// ─── QuicDatagramConfig ───────────────────────────────────────────────────────

/// Configuration for QUIC datagram mode (RFC 9221).
#[derive(Debug, Clone)]
pub struct QuicDatagramConfig {
    /// Maximum size of a single datagram payload in bytes.
    ///
    /// Should be set to (PMTU − QUIC/UDP overhead), typically 1200 bytes.
    pub max_datagram_size: usize,
    /// Initial congestion window size in bytes.
    ///
    /// Controls how many bytes may be in-flight across all datagrams.
    pub congestion_window: u32,
}

impl Default for QuicDatagramConfig {
    fn default() -> Self {
        Self {
            max_datagram_size: 1200,
            congestion_window: 65536,
        }
    }
}

impl QuicDatagramConfig {
    /// Returns the maximum number of full datagrams that fit in the
    /// congestion window.
    #[must_use]
    pub fn max_datagrams_in_flight(&self) -> u32 {
        if self.max_datagram_size == 0 {
            return 0;
        }
        self.congestion_window / self.max_datagram_size as u32
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `max_datagram_size` is zero.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_datagram_size == 0 {
            return Err("max_datagram_size must be > 0".to_owned());
        }
        Ok(())
    }
}

// ─── QuicDatagram ─────────────────────────────────────────────────────────────

/// A single QUIC datagram payload unit.
///
/// Wraps raw media bytes with metadata required for sender pacing,
/// receiver deduplication, and priority-based scheduling.
#[derive(Debug, Clone)]
pub struct QuicDatagram {
    /// Application-level stream identifier.
    ///
    /// Used to group related datagrams (e.g., all video datagrams from one
    /// encoder share the same `stream_id`).
    pub stream_id: u64,
    /// Monotonically increasing sequence number within `stream_id`.
    ///
    /// The receiver uses this to detect gaps (dropped datagrams) and to
    /// order fragments when reassembling large payloads.
    pub sequence: u64,
    /// Raw payload bytes.
    pub data: Vec<u8>,
    /// Delivery priority (0 = highest, 255 = lowest).
    ///
    /// Higher-priority datagrams should be scheduled before lower-priority
    /// ones when the congestion window is constrained.
    pub priority: u8,
}

impl QuicDatagram {
    /// Returns the payload size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the datagram payload is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ─── QuicDatagramSender ───────────────────────────────────────────────────────

/// Sends sequenced QUIC datagrams for a single logical media stream.
///
/// Maintains a per-sender sequence counter that starts at zero and
/// increments by one with each call to [`Self::next_datagram`].
///
/// The sender does **not** perform actual I/O; it generates [`QuicDatagram`]
/// values that the caller then transmits via the underlying QUIC stack.
///
/// # Example
///
/// ```
/// use oximedia_net::quic_datagram::{QuicDatagramConfig, QuicDatagramSender};
///
/// let mut sender = QuicDatagramSender::new(QuicDatagramConfig::default());
/// let d0 = sender.next_datagram(vec![1, 2, 3], 42);
/// let d1 = sender.next_datagram(vec![4, 5, 6], 42);
/// assert_eq!(d0.sequence, 0);
/// assert_eq!(d1.sequence, 1);
/// ```
pub struct QuicDatagramSender {
    /// Sender configuration.
    pub config: QuicDatagramConfig,
    /// Next sequence number to assign.
    pub sequence: u64,
}

impl QuicDatagramSender {
    /// Create a new sender with the given configuration.
    ///
    /// The sequence counter starts at 0.
    #[must_use]
    pub fn new(config: QuicDatagramConfig) -> Self {
        Self {
            config,
            sequence: 0,
        }
    }

    /// Create a new sender with default configuration and the given stream ID.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(QuicDatagramConfig::default())
    }

    /// Generate the next datagram for the given `data` payload and `stream_id`.
    ///
    /// Increments the internal sequence counter after constructing the
    /// datagram.  Priority defaults to `128` (mid-range).
    ///
    /// If the payload exceeds [`QuicDatagramConfig::max_datagram_size`] the
    /// datagram is still created — callers should use [`should_fragment`]
    /// before calling this to decide whether to split first.
    #[must_use]
    pub fn next_datagram(&mut self, data: Vec<u8>, stream_id: u64) -> QuicDatagram {
        self.next_datagram_with_priority(data, stream_id, 128)
    }

    /// Generate the next datagram with an explicit priority.
    #[must_use]
    pub fn next_datagram_with_priority(
        &mut self,
        data: Vec<u8>,
        stream_id: u64,
        priority: u8,
    ) -> QuicDatagram {
        let seq = self.sequence;
        self.sequence = self.sequence.wrapping_add(1);
        QuicDatagram {
            stream_id,
            sequence: seq,
            data,
            priority,
        }
    }

    /// Returns the total number of datagrams sent so far.
    #[must_use]
    pub fn datagrams_sent(&self) -> u64 {
        self.sequence
    }

    /// Reset the sequence counter to zero.
    pub fn reset(&mut self) {
        self.sequence = 0;
    }
}

// ─── Free functions ───────────────────────────────────────────────────────────

/// Returns `true` if `data` is larger than `config.max_datagram_size` and
/// therefore needs to be fragmented before transmission.
///
/// A zero-length payload never needs fragmentation.
#[must_use]
pub fn should_fragment(data: &[u8], config: &QuicDatagramConfig) -> bool {
    config.max_datagram_size > 0 && data.len() > config.max_datagram_size
}

/// Split `data` into consecutive chunks of at most `max_size` bytes.
///
/// If `data` is empty an empty `Vec` is returned.
/// If `max_size` is zero a single chunk containing all data is returned to
/// avoid an infinite loop.
///
/// The last chunk may be shorter than `max_size`.
#[must_use]
pub fn fragment(data: &[u8], max_size: usize) -> Vec<Vec<u8>> {
    if data.is_empty() {
        return Vec::new();
    }
    if max_size == 0 {
        return vec![data.to_vec()];
    }
    data.chunks(max_size).map(|chunk| chunk.to_vec()).collect()
}

/// Reassemble a sequence of fragments back into a contiguous buffer.
///
/// Fragments are concatenated in the order provided.  An empty slice of
/// fragments returns an empty `Vec`.
#[must_use]
pub fn reassemble(fragments: &[Vec<u8>]) -> Vec<u8> {
    let total: usize = fragments.iter().map(Vec::len).sum();
    let mut buf = Vec::with_capacity(total);
    for frag in fragments {
        buf.extend_from_slice(frag);
    }
    buf
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> QuicDatagramConfig {
        QuicDatagramConfig {
            max_datagram_size: 100,
            congestion_window: 1000,
        }
    }

    // 1. Default config max_datagram_size is 1200
    #[test]
    fn test_default_max_datagram_size() {
        let cfg = QuicDatagramConfig::default();
        assert_eq!(cfg.max_datagram_size, 1200);
    }

    // 2. Default config congestion_window is 65536
    #[test]
    fn test_default_congestion_window() {
        let cfg = QuicDatagramConfig::default();
        assert_eq!(cfg.congestion_window, 65536);
    }

    // 3. max_datagrams_in_flight correct
    #[test]
    fn test_max_datagrams_in_flight() {
        let cfg = small_config(); // 1000 / 100 = 10
        assert_eq!(cfg.max_datagrams_in_flight(), 10);
    }

    // 4. validate rejects zero max_datagram_size
    #[test]
    fn test_validate_zero_size() {
        let cfg = QuicDatagramConfig {
            max_datagram_size: 0,
            congestion_window: 1000,
        };
        assert!(cfg.validate().is_err());
    }

    // 5. validate accepts valid config
    #[test]
    fn test_validate_valid() {
        assert!(QuicDatagramConfig::default().validate().is_ok());
    }

    // 6. QuicDatagramSender sequence starts at 0
    #[test]
    fn test_sender_sequence_starts_at_zero() {
        let sender = QuicDatagramSender::new(QuicDatagramConfig::default());
        assert_eq!(sender.sequence, 0);
    }

    // 7. next_datagram increments sequence
    #[test]
    fn test_next_datagram_increments_sequence() {
        let mut sender = QuicDatagramSender::new(QuicDatagramConfig::default());
        let d0 = sender.next_datagram(vec![1, 2, 3], 10);
        let d1 = sender.next_datagram(vec![4, 5, 6], 10);
        assert_eq!(d0.sequence, 0);
        assert_eq!(d1.sequence, 1);
    }

    // 8. next_datagram sets stream_id correctly
    #[test]
    fn test_next_datagram_stream_id() {
        let mut sender = QuicDatagramSender::new(QuicDatagramConfig::default());
        let dg = sender.next_datagram(vec![0u8; 50], 42);
        assert_eq!(dg.stream_id, 42);
    }

    // 9. next_datagram default priority is 128
    #[test]
    fn test_next_datagram_default_priority() {
        let mut sender = QuicDatagramSender::new(QuicDatagramConfig::default());
        let dg = sender.next_datagram(vec![0], 1);
        assert_eq!(dg.priority, 128);
    }

    // 10. next_datagram_with_priority sets priority
    #[test]
    fn test_next_datagram_custom_priority() {
        let mut sender = QuicDatagramSender::new(QuicDatagramConfig::default());
        let dg = sender.next_datagram_with_priority(vec![0], 1, 0);
        assert_eq!(dg.priority, 0);
    }

    // 11. datagrams_sent tracks count
    #[test]
    fn test_datagrams_sent() {
        let mut sender = QuicDatagramSender::new(QuicDatagramConfig::default());
        for _ in 0..5 {
            sender.next_datagram(vec![], 1);
        }
        assert_eq!(sender.datagrams_sent(), 5);
    }

    // 12. reset brings sequence back to 0
    #[test]
    fn test_reset_sequence() {
        let mut sender = QuicDatagramSender::new(QuicDatagramConfig::default());
        sender.next_datagram(vec![], 1);
        sender.next_datagram(vec![], 1);
        sender.reset();
        assert_eq!(sender.sequence, 0);
    }

    // 13. should_fragment returns false for small data
    #[test]
    fn test_should_fragment_small_data() {
        let cfg = small_config();
        let data = vec![0u8; 50]; // less than 100
        assert!(!should_fragment(&data, &cfg));
    }

    // 14. should_fragment returns false for exactly max_size data
    #[test]
    fn test_should_fragment_exact_max() {
        let cfg = small_config();
        let data = vec![0u8; 100]; // exactly 100
        assert!(!should_fragment(&data, &cfg));
    }

    // 15. should_fragment returns true for data exceeding max_datagram_size
    #[test]
    fn test_should_fragment_large_data() {
        let cfg = small_config();
        let data = vec![0u8; 101];
        assert!(should_fragment(&data, &cfg));
    }

    // 16. fragment splits correctly
    #[test]
    fn test_fragment_splits_into_chunks() {
        let data = vec![0u8; 250];
        let chunks = fragment(&data, 100);
        assert_eq!(chunks.len(), 3); // 100 + 100 + 50
        assert_eq!(chunks[0].len(), 100);
        assert_eq!(chunks[1].len(), 100);
        assert_eq!(chunks[2].len(), 50);
    }

    // 17. fragment with exact divisor
    #[test]
    fn test_fragment_exact_divisor() {
        let data = vec![1u8; 400];
        let chunks = fragment(&data, 100);
        assert_eq!(chunks.len(), 4);
        assert!(chunks.iter().all(|c| c.len() == 100));
    }

    // 18. fragment of empty data returns empty vec
    #[test]
    fn test_fragment_empty_data() {
        let chunks = fragment(&[], 100);
        assert!(chunks.is_empty());
    }

    // 19. reassemble reconstructs original data
    #[test]
    fn test_reassemble_roundtrip() {
        let original: Vec<u8> = (0..=255u8).collect();
        let chunks = fragment(&original, 64);
        let recovered = reassemble(&chunks);
        assert_eq!(recovered, original);
    }

    // 20. QuicDatagram size and is_empty
    #[test]
    fn test_quic_datagram_size_is_empty() {
        let dg_empty = QuicDatagram {
            stream_id: 0,
            sequence: 0,
            data: vec![],
            priority: 128,
        };
        assert!(dg_empty.is_empty());
        assert_eq!(dg_empty.size(), 0);

        let dg_full = QuicDatagram {
            stream_id: 0,
            sequence: 1,
            data: vec![1, 2, 3],
            priority: 64,
        };
        assert!(!dg_full.is_empty());
        assert_eq!(dg_full.size(), 3);
    }

    // 21. with_defaults creates sender with default config
    #[test]
    fn test_with_defaults() {
        let sender = QuicDatagramSender::with_defaults();
        assert_eq!(sender.config.max_datagram_size, 1200);
        assert_eq!(sender.sequence, 0);
    }

    // 22. Multiple stream IDs are independent
    #[test]
    fn test_multiple_stream_ids() {
        let mut sender = QuicDatagramSender::new(QuicDatagramConfig::default());
        let d_video = sender.next_datagram(vec![0], 1);
        let d_audio = sender.next_datagram(vec![0], 2);
        // Both datagrams from the same sender but different stream IDs
        assert_eq!(d_video.stream_id, 1);
        assert_eq!(d_audio.stream_id, 2);
        // Sequence is global to this sender
        assert_eq!(d_video.sequence, 0);
        assert_eq!(d_audio.sequence, 1);
    }

    // 23. should_fragment returns false for empty data (no fragmentation needed)
    #[test]
    fn test_should_fragment_empty_is_false() {
        let cfg = small_config();
        assert!(!should_fragment(&[], &cfg));
    }
}
