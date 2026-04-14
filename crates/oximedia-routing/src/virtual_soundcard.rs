#![allow(dead_code)]
//! Virtual soundcard module for OS-level audio routing.
//!
//! Provides an abstraction layer over platform audio loopback interfaces
//! (WASAPI on Windows, CoreAudio on macOS, ALSA on Linux).  Rather than
//! directly linking to platform APIs (which would break the pure-Rust policy
//! on other targets), this module models the *routing* state — which virtual
//! device is connected to which application stream — and exposes a unified
//! interface that higher-level code (or a thin platform shim) can drive.
//!
//! # Design
//!
//! - [`VirtualSoundcard`] — represents a single loopback audio device with a
//!   fixed channel count and sample rate.
//! - [`SoundcardRegistry`] — collection of virtual soundcards; manages creation,
//!   removal and name-based lookup.
//! - [`AppStream`] — an application-side audio stream (source or sink) that can
//!   be connected to a virtual soundcard.
//! - [`SoundcardConnection`] — binding between an [`AppStream`] and a
//!   [`VirtualSoundcard`] port range.
//! - [`SoundcardError`] — errors returned by the registry and soundcard
//!   operations.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// SoundcardDirection
// ---------------------------------------------------------------------------

/// Whether a virtual soundcard port acts as a capture (input) or playback
/// (output) device from the application's perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoundcardDirection {
    /// The virtual soundcard exposes audio **to** applications (record).
    Capture,
    /// The virtual soundcard receives audio **from** applications (playback).
    Playback,
    /// Bidirectional — the device exposes both capture and playback ports.
    Duplex,
}

impl fmt::Display for SoundcardDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capture => write!(f, "capture"),
            Self::Playback => write!(f, "playback"),
            Self::Duplex => write!(f, "duplex"),
        }
    }
}

// ---------------------------------------------------------------------------
// SoundcardBackend
// ---------------------------------------------------------------------------

/// The OS-level audio subsystem this virtual device is intended to model.
///
/// This is purely informational metadata — the actual system calls are
/// performed by a platform shim outside this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoundcardBackend {
    /// Windows Audio Session API loopback.
    Wasapi,
    /// macOS CoreAudio loopback (e.g. via BlackHole or Loopback).
    CoreAudio,
    /// Linux ALSA loopback (`snd-aloop` kernel module).
    AlsaLoopback,
    /// JACK Audio Connection Kit virtual port.
    Jack,
    /// Platform-agnostic in-process loopback (for tests and embedded use).
    InProcess,
}

impl fmt::Display for SoundcardBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wasapi => write!(f, "WASAPI"),
            Self::CoreAudio => write!(f, "CoreAudio"),
            Self::AlsaLoopback => write!(f, "ALSA loopback"),
            Self::Jack => write!(f, "JACK"),
            Self::InProcess => write!(f, "in-process"),
        }
    }
}

// ---------------------------------------------------------------------------
// VirtualSoundcard
// ---------------------------------------------------------------------------

/// A virtual loopback audio device.
///
/// Models a pair of capture/playback ports with a fixed channel count, sample
/// rate and bit depth.  The device can be either active (OS driver is loaded)
/// or inactive (defined but not yet instantiated).
#[derive(Debug, Clone)]
pub struct VirtualSoundcard {
    /// Unique identifier for this soundcard within the registry.
    pub id: String,
    /// Human-readable device name (as seen by applications).
    pub name: String,
    /// Number of audio channels.
    pub channels: u8,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Bit depth (e.g. 16, 24, 32).
    pub bit_depth: u8,
    /// Direction(s) this device exposes.
    pub direction: SoundcardDirection,
    /// Underlying OS audio subsystem.
    pub backend: SoundcardBackend,
    /// Whether the OS driver is currently instantiated.
    pub active: bool,
    /// Arbitrary key/value metadata (e.g. OS device index, buffer size hint).
    pub metadata: HashMap<String, String>,
}

impl VirtualSoundcard {
    /// Creates a new, inactive virtual soundcard.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        channels: u8,
        sample_rate: u32,
        bit_depth: u8,
        direction: SoundcardDirection,
        backend: SoundcardBackend,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            channels,
            sample_rate,
            bit_depth,
            direction,
            backend,
            active: false,
            metadata: HashMap::new(),
        }
    }

    /// Marks the device as active.
    pub fn activate(&mut self) {
        self.active = true;
    }

    /// Marks the device as inactive.
    pub fn deactivate(&mut self) {
        self.active = false;
    }

    /// Returns `true` if the device supports capture (application can record
    /// from it).
    pub fn supports_capture(&self) -> bool {
        matches!(
            self.direction,
            SoundcardDirection::Capture | SoundcardDirection::Duplex
        )
    }

    /// Returns `true` if the device supports playback (application can play
    /// into it).
    pub fn supports_playback(&self) -> bool {
        matches!(
            self.direction,
            SoundcardDirection::Playback | SoundcardDirection::Duplex
        )
    }

    /// Inserts a metadata key/value pair.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Retrieves a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }
}

impl fmt::Display for VirtualSoundcard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = if self.active { "active" } else { "inactive" };
        write!(
            f,
            "VirtualSoundcard('{}', {}ch, {}Hz, {}bit, {}, {}, {})",
            self.name,
            self.channels,
            self.sample_rate,
            self.bit_depth,
            self.direction,
            self.backend,
            state
        )
    }
}

// ---------------------------------------------------------------------------
// AppStream
// ---------------------------------------------------------------------------

/// Represents an application-side audio stream.
///
/// An `AppStream` is the peer that connects to a [`VirtualSoundcard`] to
/// either send or receive audio.  Typical examples: a DAW track output, a
/// browser tab, a conferencing app microphone capture.
#[derive(Debug, Clone)]
pub struct AppStream {
    /// Unique identifier for this stream.
    pub id: String,
    /// Human-readable application or stream name.
    pub app_name: String,
    /// Number of audio channels the stream expects.
    pub channels: u8,
    /// Sample rate the stream is running at.
    pub sample_rate: u32,
    /// Whether this stream is a source (sends audio) or sink (receives audio).
    pub is_source: bool,
}

impl AppStream {
    /// Creates a new application stream.
    pub fn new(
        id: impl Into<String>,
        app_name: impl Into<String>,
        channels: u8,
        sample_rate: u32,
        is_source: bool,
    ) -> Self {
        Self {
            id: id.into(),
            app_name: app_name.into(),
            channels,
            sample_rate,
            is_source,
        }
    }
}

impl fmt::Display for AppStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.is_source { "source" } else { "sink" };
        write!(
            f,
            "AppStream('{}', {}ch, {}Hz, {})",
            self.app_name, self.channels, self.sample_rate, kind
        )
    }
}

// ---------------------------------------------------------------------------
// SoundcardConnection
// ---------------------------------------------------------------------------

/// A binding between an [`AppStream`] and a [`VirtualSoundcard`].
///
/// A connection is valid only when the channel counts and sample rates are
/// compatible (identical, or can be converted by the platform shim).
#[derive(Debug, Clone)]
pub struct SoundcardConnection {
    /// Unique connection identifier.
    pub id: String,
    /// The soundcard being connected to.
    pub soundcard_id: String,
    /// The application stream being connected.
    pub stream_id: String,
    /// First channel index on the soundcard that this stream occupies (zero-based).
    pub start_channel: u8,
    /// Whether this connection is currently active.
    pub active: bool,
}

impl SoundcardConnection {
    /// Creates a new active connection.
    pub fn new(
        id: impl Into<String>,
        soundcard_id: impl Into<String>,
        stream_id: impl Into<String>,
        start_channel: u8,
    ) -> Self {
        Self {
            id: id.into(),
            soundcard_id: soundcard_id.into(),
            stream_id: stream_id.into(),
            start_channel,
            active: true,
        }
    }
}

// ---------------------------------------------------------------------------
// SoundcardError
// ---------------------------------------------------------------------------

/// Errors returned by [`SoundcardRegistry`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SoundcardError {
    /// A soundcard with the given id already exists.
    DuplicateSoundcardId(String),
    /// No soundcard with the given id exists.
    SoundcardNotFound(String),
    /// A stream with the given id already exists.
    DuplicateStreamId(String),
    /// No stream with the given id exists.
    StreamNotFound(String),
    /// A connection with the given id already exists.
    DuplicateConnectionId(String),
    /// No connection with the given id exists.
    ConnectionNotFound(String),
    /// The stream's channel count would overflow the soundcard's channel range.
    ChannelRangeExceeded {
        /// First channel requested.
        start: u8,
        /// Channels needed.
        needed: u8,
        /// Maximum channels available on the soundcard.
        available: u8,
    },
    /// The stream's sample rate does not match the soundcard's sample rate.
    SampleRateMismatch {
        /// Sample rate of the stream.
        stream_rate: u32,
        /// Sample rate of the soundcard.
        soundcard_rate: u32,
    },
    /// The soundcard direction does not support the requested operation.
    DirectionMismatch,
    /// The soundcard is not active and cannot accept connections.
    SoundcardNotActive(String),
}

impl fmt::Display for SoundcardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateSoundcardId(id) => write!(f, "soundcard id already registered: {id}"),
            Self::SoundcardNotFound(id) => write!(f, "soundcard not found: {id}"),
            Self::DuplicateStreamId(id) => write!(f, "stream id already registered: {id}"),
            Self::StreamNotFound(id) => write!(f, "stream not found: {id}"),
            Self::DuplicateConnectionId(id) => write!(f, "connection id already registered: {id}"),
            Self::ConnectionNotFound(id) => write!(f, "connection not found: {id}"),
            Self::ChannelRangeExceeded {
                start,
                needed,
                available,
            } => write!(
                f,
                "channel range [{start}..{}] exceeds soundcard capacity ({})",
                start + needed - 1,
                available
            ),
            Self::SampleRateMismatch {
                stream_rate,
                soundcard_rate,
            } => write!(
                f,
                "stream sample rate {stream_rate} Hz does not match soundcard {soundcard_rate} Hz"
            ),
            Self::DirectionMismatch => {
                write!(
                    f,
                    "soundcard direction does not support the requested operation"
                )
            }
            Self::SoundcardNotActive(id) => {
                write!(f, "soundcard '{id}' is not active")
            }
        }
    }
}

impl std::error::Error for SoundcardError {}

// ---------------------------------------------------------------------------
// SoundcardRegistry
// ---------------------------------------------------------------------------

/// Registry of virtual soundcards, application streams, and their connections.
///
/// This is the central authority for OS-level audio routing.  It validates
/// that connections are compatible before registering them, and cascades
/// removal so there are no dangling references.
#[derive(Debug, Default)]
pub struct SoundcardRegistry {
    soundcards: HashMap<String, VirtualSoundcard>,
    streams: HashMap<String, AppStream>,
    connections: HashMap<String, SoundcardConnection>,
    /// Index: soundcard_id → set of connection ids
    soundcard_connections: HashMap<String, Vec<String>>,
}

impl SoundcardRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    // -----------------------------------------------------------------------
    // Soundcard management
    // -----------------------------------------------------------------------

    /// Registers a new virtual soundcard.
    pub fn register_soundcard(&mut self, card: VirtualSoundcard) -> Result<(), SoundcardError> {
        if self.soundcards.contains_key(&card.id) {
            return Err(SoundcardError::DuplicateSoundcardId(card.id.clone()));
        }
        let id = card.id.clone();
        self.soundcards.insert(id.clone(), card);
        self.soundcard_connections.entry(id).or_default();
        Ok(())
    }

    /// Removes a soundcard and all connections to it.
    pub fn remove_soundcard(
        &mut self,
        soundcard_id: &str,
    ) -> Result<VirtualSoundcard, SoundcardError> {
        let card = self
            .soundcards
            .remove(soundcard_id)
            .ok_or_else(|| SoundcardError::SoundcardNotFound(soundcard_id.to_string()))?;

        if let Some(conn_ids) = self.soundcard_connections.remove(soundcard_id) {
            for conn_id in &conn_ids {
                self.connections.remove(conn_id);
            }
        }

        Ok(card)
    }

    /// Returns a reference to the soundcard with the given id.
    pub fn get_soundcard(&self, soundcard_id: &str) -> Option<&VirtualSoundcard> {
        self.soundcards.get(soundcard_id)
    }

    /// Returns a mutable reference to the soundcard with the given id.
    pub fn get_soundcard_mut(&mut self, soundcard_id: &str) -> Option<&mut VirtualSoundcard> {
        self.soundcards.get_mut(soundcard_id)
    }

    /// Returns all registered soundcards.
    pub fn all_soundcards(&self) -> Vec<&VirtualSoundcard> {
        self.soundcards.values().collect()
    }

    /// Returns all active soundcards.
    pub fn active_soundcards(&self) -> Vec<&VirtualSoundcard> {
        self.soundcards.values().filter(|c| c.active).collect()
    }

    /// Looks up a soundcard by human-readable name (first match).
    pub fn find_soundcard_by_name(&self, name: &str) -> Option<&VirtualSoundcard> {
        self.soundcards.values().find(|c| c.name == name)
    }

    // -----------------------------------------------------------------------
    // App stream management
    // -----------------------------------------------------------------------

    /// Registers an application stream.
    pub fn register_stream(&mut self, stream: AppStream) -> Result<(), SoundcardError> {
        if self.streams.contains_key(&stream.id) {
            return Err(SoundcardError::DuplicateStreamId(stream.id.clone()));
        }
        self.streams.insert(stream.id.clone(), stream);
        Ok(())
    }

    /// Removes an application stream and all connections from it.
    pub fn remove_stream(&mut self, stream_id: &str) -> Result<AppStream, SoundcardError> {
        let stream = self
            .streams
            .remove(stream_id)
            .ok_or_else(|| SoundcardError::StreamNotFound(stream_id.to_string()))?;

        // Remove all connections referencing this stream
        let to_remove: Vec<String> = self
            .connections
            .values()
            .filter(|c| c.stream_id == stream_id)
            .map(|c| c.id.clone())
            .collect();

        for conn_id in &to_remove {
            if let Some(conn) = self.connections.remove(conn_id) {
                if let Some(ids) = self.soundcard_connections.get_mut(&conn.soundcard_id) {
                    ids.retain(|id| id != conn_id);
                }
            }
        }

        Ok(stream)
    }

    /// Returns a reference to the stream with the given id.
    pub fn get_stream(&self, stream_id: &str) -> Option<&AppStream> {
        self.streams.get(stream_id)
    }

    /// Returns all registered streams.
    pub fn all_streams(&self) -> Vec<&AppStream> {
        self.streams.values().collect()
    }

    // -----------------------------------------------------------------------
    // Connections
    // -----------------------------------------------------------------------

    /// Creates a connection between an application stream and a virtual
    /// soundcard, validating compatibility first.
    pub fn connect(&mut self, connection: SoundcardConnection) -> Result<(), SoundcardError> {
        if self.connections.contains_key(&connection.id) {
            return Err(SoundcardError::DuplicateConnectionId(connection.id.clone()));
        }

        let card = self
            .soundcards
            .get(&connection.soundcard_id)
            .ok_or_else(|| SoundcardError::SoundcardNotFound(connection.soundcard_id.clone()))?;

        if !card.active {
            return Err(SoundcardError::SoundcardNotActive(
                connection.soundcard_id.clone(),
            ));
        }

        let stream = self
            .streams
            .get(&connection.stream_id)
            .ok_or_else(|| SoundcardError::StreamNotFound(connection.stream_id.clone()))?;

        // Validate sample rate compatibility
        if stream.sample_rate != card.sample_rate {
            return Err(SoundcardError::SampleRateMismatch {
                stream_rate: stream.sample_rate,
                soundcard_rate: card.sample_rate,
            });
        }

        // Validate channel range
        let end_channel = connection
            .start_channel
            .checked_add(stream.channels)
            .ok_or(SoundcardError::ChannelRangeExceeded {
                start: connection.start_channel,
                needed: stream.channels,
                available: card.channels,
            })?;

        if end_channel > card.channels {
            return Err(SoundcardError::ChannelRangeExceeded {
                start: connection.start_channel,
                needed: stream.channels,
                available: card.channels,
            });
        }

        // Validate direction compatibility
        if stream.is_source && !card.supports_capture() {
            return Err(SoundcardError::DirectionMismatch);
        }
        if !stream.is_source && !card.supports_playback() {
            return Err(SoundcardError::DirectionMismatch);
        }

        let conn_id = connection.id.clone();
        let card_id = connection.soundcard_id.clone();
        self.connections.insert(conn_id.clone(), connection);
        self.soundcard_connections
            .entry(card_id)
            .or_default()
            .push(conn_id);

        Ok(())
    }

    /// Removes a connection by id.
    pub fn disconnect(&mut self, connection_id: &str) -> Result<(), SoundcardError> {
        let conn = self
            .connections
            .remove(connection_id)
            .ok_or_else(|| SoundcardError::ConnectionNotFound(connection_id.to_string()))?;

        if let Some(ids) = self.soundcard_connections.get_mut(&conn.soundcard_id) {
            ids.retain(|id| id != connection_id);
        }

        Ok(())
    }

    /// Returns all connections for a given soundcard.
    pub fn connections_for_soundcard(&self, soundcard_id: &str) -> Vec<&SoundcardConnection> {
        let ids = match self.soundcard_connections.get(soundcard_id) {
            Some(v) => v,
            None => return Vec::new(),
        };
        ids.iter()
            .filter_map(|id| self.connections.get(id))
            .collect()
    }

    /// Returns all connections for a given stream.
    pub fn connections_for_stream(&self, stream_id: &str) -> Vec<&SoundcardConnection> {
        self.connections
            .values()
            .filter(|c| c.stream_id == stream_id)
            .collect()
    }

    /// Returns the total number of registered connections.
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Returns the total number of registered soundcards.
    pub fn soundcard_count(&self) -> usize {
        self.soundcards.len()
    }

    /// Returns the total number of registered streams.
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_card(
        id: &str,
        name: &str,
        channels: u8,
        rate: u32,
        dir: SoundcardDirection,
    ) -> VirtualSoundcard {
        VirtualSoundcard::new(
            id,
            name,
            channels,
            rate,
            24,
            dir,
            SoundcardBackend::InProcess,
        )
    }

    fn make_active_card(
        id: &str,
        channels: u8,
        rate: u32,
        dir: SoundcardDirection,
    ) -> VirtualSoundcard {
        let mut card = make_card(id, id, channels, rate, dir);
        card.activate();
        card
    }

    fn make_stream(id: &str, channels: u8, rate: u32, is_source: bool) -> AppStream {
        AppStream::new(id, id, channels, rate, is_source)
    }

    fn make_connection(id: &str, card_id: &str, stream_id: &str, start: u8) -> SoundcardConnection {
        SoundcardConnection::new(id, card_id, stream_id, start)
    }

    #[test]
    fn test_register_and_retrieve_soundcard() {
        let mut reg = SoundcardRegistry::new();
        let card = make_card("card-1", "Loopback A", 2, 48000, SoundcardDirection::Duplex);
        reg.register_soundcard(card).expect("register failed");
        let retrieved = reg.get_soundcard("card-1");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Loopback A");
    }

    #[test]
    fn test_duplicate_soundcard_id_returns_error() {
        let mut reg = SoundcardRegistry::new();
        reg.register_soundcard(make_card("dup", "A", 2, 48000, SoundcardDirection::Duplex))
            .expect("first ok");
        let result =
            reg.register_soundcard(make_card("dup", "B", 2, 48000, SoundcardDirection::Duplex));
        assert!(matches!(
            result,
            Err(SoundcardError::DuplicateSoundcardId(_))
        ));
    }

    #[test]
    fn test_activate_and_deactivate() {
        let mut reg = SoundcardRegistry::new();
        reg.register_soundcard(make_card(
            "act",
            "Act",
            2,
            48000,
            SoundcardDirection::Duplex,
        ))
        .expect("ok");
        let card = reg.get_soundcard_mut("act").expect("should exist");
        assert!(!card.active);
        card.activate();
        assert!(card.active);
        card.deactivate();
        assert!(!card.active);
    }

    #[test]
    fn test_connect_stream_to_soundcard() {
        let mut reg = SoundcardRegistry::new();
        reg.register_soundcard(make_active_card(
            "card-a",
            2,
            48000,
            SoundcardDirection::Duplex,
        ))
        .expect("card ok");
        reg.register_stream(make_stream("stream-a", 2, 48000, true))
            .expect("stream ok");
        let conn = make_connection("conn-1", "card-a", "stream-a", 0);
        reg.connect(conn).expect("connect failed");
        assert_eq!(reg.connection_count(), 1);
    }

    #[test]
    fn test_connect_inactive_soundcard_fails() {
        let mut reg = SoundcardRegistry::new();
        reg.register_soundcard(make_card(
            "card-inactive",
            "I",
            2,
            48000,
            SoundcardDirection::Duplex,
        ))
        .expect("ok");
        reg.register_stream(make_stream("stream-b", 2, 48000, true))
            .expect("ok");
        let conn = make_connection("conn-2", "card-inactive", "stream-b", 0);
        let result = reg.connect(conn);
        assert!(matches!(result, Err(SoundcardError::SoundcardNotActive(_))));
    }

    #[test]
    fn test_sample_rate_mismatch_fails() {
        let mut reg = SoundcardRegistry::new();
        reg.register_soundcard(make_active_card(
            "card-r",
            2,
            48000,
            SoundcardDirection::Duplex,
        ))
        .expect("ok");
        reg.register_stream(make_stream("stream-r", 2, 44100, true))
            .expect("ok");
        let conn = make_connection("conn-r", "card-r", "stream-r", 0);
        let result = reg.connect(conn);
        assert!(matches!(
            result,
            Err(SoundcardError::SampleRateMismatch { .. })
        ));
    }

    #[test]
    fn test_channel_overflow_fails() {
        let mut reg = SoundcardRegistry::new();
        reg.register_soundcard(make_active_card(
            "card-ch",
            2,
            48000,
            SoundcardDirection::Duplex,
        ))
        .expect("ok");
        // 2-channel stream starting at channel 1 would need channels [1, 2], but card only has [0, 1]
        reg.register_stream(make_stream("stream-ch", 2, 48000, true))
            .expect("ok");
        let conn = make_connection("conn-ch", "card-ch", "stream-ch", 1);
        let result = reg.connect(conn);
        assert!(matches!(
            result,
            Err(SoundcardError::ChannelRangeExceeded { .. })
        ));
    }

    #[test]
    fn test_direction_mismatch_fails() {
        let mut reg = SoundcardRegistry::new();
        // Playback-only card, but stream is a source (capture)
        reg.register_soundcard(make_active_card(
            "card-play",
            2,
            48000,
            SoundcardDirection::Playback,
        ))
        .expect("ok");
        reg.register_stream(make_stream("stream-src", 2, 48000, true))
            .expect("ok");
        let conn = make_connection("conn-dir", "card-play", "stream-src", 0);
        let result = reg.connect(conn);
        assert!(matches!(result, Err(SoundcardError::DirectionMismatch)));
    }

    #[test]
    fn test_disconnect() {
        let mut reg = SoundcardRegistry::new();
        reg.register_soundcard(make_active_card(
            "card-dc",
            2,
            48000,
            SoundcardDirection::Duplex,
        ))
        .expect("ok");
        reg.register_stream(make_stream("stream-dc", 2, 48000, true))
            .expect("ok");
        reg.connect(make_connection("conn-dc", "card-dc", "stream-dc", 0))
            .expect("ok");
        assert_eq!(reg.connection_count(), 1);
        reg.disconnect("conn-dc").expect("disconnect failed");
        assert_eq!(reg.connection_count(), 0);
    }

    #[test]
    fn test_remove_soundcard_cascades_connections() {
        let mut reg = SoundcardRegistry::new();
        reg.register_soundcard(make_active_card(
            "card-cas",
            2,
            48000,
            SoundcardDirection::Duplex,
        ))
        .expect("ok");
        reg.register_stream(make_stream("stream-cas", 2, 48000, true))
            .expect("ok");
        reg.connect(make_connection("conn-cas", "card-cas", "stream-cas", 0))
            .expect("ok");
        assert_eq!(reg.connection_count(), 1);
        reg.remove_soundcard("card-cas").expect("remove failed");
        assert_eq!(reg.connection_count(), 0);
    }

    #[test]
    fn test_find_soundcard_by_name() {
        let mut reg = SoundcardRegistry::new();
        reg.register_soundcard(make_card(
            "card-name",
            "My Loopback",
            8,
            48000,
            SoundcardDirection::Duplex,
        ))
        .expect("ok");
        let found = reg.find_soundcard_by_name("My Loopback");
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "card-name");
        assert!(reg.find_soundcard_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_display_implementations() {
        let card = make_card("c", "Display Card", 4, 48000, SoundcardDirection::Duplex);
        let s = format!("{card}");
        assert!(s.contains("Display Card"));
        assert!(s.contains("4ch"));

        let stream = make_stream("s", 2, 48000, true);
        let s = format!("{stream}");
        assert!(s.contains("source"));
    }

    #[test]
    fn test_metadata_on_soundcard() {
        let mut card = make_card("meta", "Meta Card", 2, 48000, SoundcardDirection::Capture);
        card.set_metadata("os_device_index", "3");
        assert_eq!(card.get_metadata("os_device_index"), Some("3"));
        assert!(card.get_metadata("missing").is_none());
    }
}
