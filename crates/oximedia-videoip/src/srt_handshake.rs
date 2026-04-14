//! SRT (Secure Reliable Transport) handshake protocol implementation.
//!
//! This module implements the SRT handshake state machine as specified in the
//! [SRT Protocol Technical Overview](https://haivision.github.io/srt-rfc/draft-sharabayko-srt.html)
//! and the [SRT RFC draft](https://datatracker.ietf.org/doc/html/draft-sharabayko-srt-01).
//!
//! # Handshake Phases
//!
//! SRT uses a **two-phase** handshake on top of UDP:
//!
//! ```text
//! Caller                                 Listener
//!   │                                       │
//!   │──── Induction (HSREQ) ────────────>  │  Phase 1: Induction
//!   │<─── Induction (HSRSP, cookie) ──────  │
//!   │                                       │
//!   │──── Conclusion (HSREQ + cookie) ──>  │  Phase 2: Conclusion
//!   │<─── Conclusion (HSRSP, done) ───────  │
//!   │                                       │
//!   │           (data flow)                 │
//! ```
//!
//! # Extension Fields
//!
//! The conclusion phase carries optional **handshake extension blocks**:
//!
//! - `HSREQ` — SRT version, latency, and flags negotiation
//! - `HSRSP` — Peer latency confirmation and flags acknowledgement
//! - `KMREQ` / `KMRSP` — Key Material (encryption key wrapping)
//! - `SID`   — Stream Identifier routing string
//!
//! # Rejection Reasons
//!
//! If the listener rejects the connection it sends a `REJ_*` code in the
//! `type` field of the handshake packet.  Common codes are defined in
//! [`RejectionReason`].

use std::fmt;

// ─── Constants ────────────────────────────────────────────────────────────────

/// Magic number in every SRT handshake packet (0x0000_0004 in the wire format).
pub const SRT_MAGIC: u32 = 0x0000_0004;

/// SRT protocol version 1.4 (major=1, minor=4, patch=0 encoded as 0x00010400).
pub const SRT_VERSION_1_4: u32 = 0x0001_0400;

/// SRT protocol version 1.5 (major=1, minor=5, patch=0 encoded as 0x00010500).
pub const SRT_VERSION_1_5: u32 = 0x0001_0500;

/// Maximum SRT stream ID length (512 bytes per spec §3.2.1.3).
pub const MAX_STREAM_ID_LEN: usize = 512;

/// Minimum latency in milliseconds accepted by the spec (20 ms).
pub const MIN_LATENCY_MS: u32 = 20;

/// Maximum latency in milliseconds accepted by the spec (30 000 ms).
pub const MAX_LATENCY_MS: u32 = 30_000;

// ─── Handshake Type ───────────────────────────────────────────────────────────

/// The `type` field of an SRT handshake packet, which encodes both the
/// handshake phase and (on rejection) the rejection reason.
///
/// Values ≤ `0x0000_FFFF` are phase identifiers; values ≥ `0x1000_0000` are
/// rejection codes (bit 28 set).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeType {
    /// Caller → Listener: first induction request.
    Induction,
    /// Listener → Caller: induction response (carries a SYN cookie).
    WaveaHand,
    /// Caller → Listener: conclusion with extension blocks.
    Conclusion,
    /// Listener → Caller: final conclusion acknowledgement.
    Agreement,
    /// Rejection from the listener (carries a [`RejectionReason`]).
    Rejection(RejectionReason),
}

impl HandshakeType {
    /// The raw `u32` wire value for this type.
    #[must_use]
    pub fn to_wire(self) -> u32 {
        match self {
            Self::Induction => 1,
            Self::WaveaHand => 0,
            Self::Conclusion => 0xFFFF_FFFF,
            Self::Agreement => 0xFFFF_FFFE,
            Self::Rejection(r) => 0x1000_0000 | (r as u32),
        }
    }

    /// Decode a raw `u32` wire value into a `HandshakeType`.
    ///
    /// Returns `None` if the value does not map to any known type.
    #[must_use]
    pub fn from_wire(v: u32) -> Option<Self> {
        match v {
            1 => Some(Self::Induction),
            0 => Some(Self::WaveaHand),
            0xFFFF_FFFF => Some(Self::Conclusion),
            0xFFFF_FFFE => Some(Self::Agreement),
            v if v & 0x1000_0000 != 0 => {
                let code = v & !0x1000_0000;
                Some(Self::Rejection(RejectionReason::from_code(code)))
            }
            _ => None,
        }
    }
}

impl fmt::Display for HandshakeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Induction => write!(f, "INDUCTION"),
            Self::WaveaHand => write!(f, "WAVEAHAND"),
            Self::Conclusion => write!(f, "CONCLUSION"),
            Self::Agreement => write!(f, "AGREEMENT"),
            Self::Rejection(r) => write!(f, "REJECTION({r})"),
        }
    }
}

// ─── Rejection Reason ─────────────────────────────────────────────────────────

/// SRT rejection reason codes (carried in the `type` field on a `REJ_*`
/// handshake response from the listener).
///
/// Codes 0–999 are system-defined; 1000+ are application-defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum RejectionReason {
    /// Unknown / unspecified reason.
    Unknown = 0,
    /// System error (OS-level failure).
    System = 1,
    /// Peer reject due to internal error.
    Peer = 2,
    /// Resource exhausted (no socket slots available).
    Resource = 3,
    /// Forbidden by security policy.
    Forbidden = 4,
    /// Incompatible SRT version.
    Version = 5,
    /// Passphrase mismatch.
    Passphrase = 6,
    /// Media type mismatch.
    MediaType = 7,
    /// Bad request (malformed handshake).
    BadRequest = 8,
    /// Unauthorized / authentication failure.
    Unauthorized = 9,
    /// Overloaded — listener is too busy.
    Overloaded = 10,
    /// Conflict — stream ID already in use.
    Conflict = 11,
    /// IP address not allowed (geo-blocking).
    GeoBlocked = 12,
    /// Stream has already been closed.
    ClosedSession = 13,
    /// Timeout during connection setup.
    Timeout = 14,
    /// Application-defined rejection (≥ 1000).
    ApplicationDefined = 1000,
}

impl RejectionReason {
    /// Convert a raw code into a `RejectionReason`.
    #[must_use]
    pub fn from_code(code: u32) -> Self {
        match code {
            0 => Self::Unknown,
            1 => Self::System,
            2 => Self::Peer,
            3 => Self::Resource,
            4 => Self::Forbidden,
            5 => Self::Version,
            6 => Self::Passphrase,
            7 => Self::MediaType,
            8 => Self::BadRequest,
            9 => Self::Unauthorized,
            10 => Self::Overloaded,
            11 => Self::Conflict,
            12 => Self::GeoBlocked,
            13 => Self::ClosedSession,
            14 => Self::Timeout,
            v if v >= 1000 => Self::ApplicationDefined,
            _ => Self::Unknown,
        }
    }
}

impl fmt::Display for RejectionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Unknown => "UNKNOWN",
            Self::System => "SYSTEM",
            Self::Peer => "PEER",
            Self::Resource => "RESOURCE",
            Self::Forbidden => "FORBIDDEN",
            Self::Version => "VERSION",
            Self::Passphrase => "PASSPHRASE",
            Self::MediaType => "MEDIA_TYPE",
            Self::BadRequest => "BAD_REQUEST",
            Self::Unauthorized => "UNAUTHORIZED",
            Self::Overloaded => "OVERLOADED",
            Self::Conflict => "CONFLICT",
            Self::GeoBlocked => "GEO_BLOCKED",
            Self::ClosedSession => "CLOSED_SESSION",
            Self::Timeout => "TIMEOUT",
            Self::ApplicationDefined => "APP_DEFINED",
        };
        write!(f, "{s}")
    }
}

// ─── Extension Flags (HSREQ) ─────────────────────────────────────────────────

/// Bitfield flags carried in the `HSREQ` / `HSRSP` extension blocks.
///
/// Each flag is a single bit in the 32-bit flags field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SrtFlags(pub u32);

impl SrtFlags {
    // Bit 0 — TSBPD sender (Timestamp-Based Packet Delivery, sender side)
    /// TSBPD sender mode enabled.
    pub const TSBPD_SND: u32 = 1 << 0;
    /// TSBPD receiver mode enabled.
    pub const TSBPD_RCV: u32 = 1 << 1;
    /// Haicrypt (legacy HaiVision encryption) disabled.
    pub const HAICRYPT_OFF: u32 = 1 << 2;
    /// Timestamp drift correction enabled.
    pub const TLPKT_DROP: u32 = 1 << 3;
    /// `NAK` Feedback enabled.
    pub const NAK_REPORT: u32 = 1 << 4;
    /// Receiver re-order tolerance mode.
    pub const REXMIT_FLGS: u32 = 1 << 5;
    /// Stream-ID extension block included.
    pub const STREAM_ID: u32 = 1 << 6;

    /// Construct flags from a raw bitmask.
    #[must_use]
    pub const fn new(bits: u32) -> Self {
        Self(bits)
    }

    /// Test whether a given bit flag is set.
    #[must_use]
    pub fn has(self, flag: u32) -> bool {
        self.0 & flag != 0
    }

    /// Set a flag bit.
    pub fn set(&mut self, flag: u32) {
        self.0 |= flag;
    }

    /// Clear a flag bit.
    pub fn clear(&mut self, flag: u32) {
        self.0 &= !flag;
    }

    /// Return the raw bitmask.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }
}

// ─── HSREQ Extension Block ────────────────────────────────────────────────────

/// `HSREQ` handshake extension — SRT capability negotiation block.
///
/// This block is sent by the caller in the `Conclusion` phase and echoed
/// (with peer adjustments) by the listener in the `Agreement` / `Conclusion`
/// response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HsreqBlock {
    /// Caller's SRT version (e.g. [`SRT_VERSION_1_4`]).
    pub srt_version: u32,
    /// Capability flags (sender side).
    pub srt_flags: SrtFlags,
    /// Timestamp-based packet delivery latency requested by the sender (ms).
    ///
    /// This is the *sender* TsbpdDelay in milliseconds.  Stored as a `u16`
    /// pair `[recv_latency, sender_latency]` in the wire format; here we
    /// store them separately for clarity.
    pub recv_tsbpd_delay_ms: u16,
    /// Sender TSBPD delay (ms).
    pub snd_tsbpd_delay_ms: u16,
}

impl HsreqBlock {
    /// Construct a `HSREQ` block with SRT 1.4 defaults.
    #[must_use]
    pub fn new(recv_latency_ms: u16, snd_latency_ms: u16) -> Self {
        Self {
            srt_version: SRT_VERSION_1_4,
            srt_flags: SrtFlags::new(SrtFlags::TSBPD_SND | SrtFlags::TSBPD_RCV | SrtFlags::NAK_REPORT),
            recv_tsbpd_delay_ms: recv_latency_ms,
            snd_tsbpd_delay_ms: snd_latency_ms,
        }
    }

    /// Validate the block: latency must be within [`MIN_LATENCY_MS`] ..
    /// [`MAX_LATENCY_MS`], and the version must be ≥ 1.4.
    ///
    /// Returns an error string on failure.
    pub fn validate(&self) -> Result<(), String> {
        if self.srt_version < SRT_VERSION_1_4 {
            return Err(format!(
                "unsupported SRT version 0x{:08X} (minimum 0x{:08X})",
                self.srt_version, SRT_VERSION_1_4
            ));
        }
        for (name, val) in [
            ("recv_tsbpd_delay_ms", u32::from(self.recv_tsbpd_delay_ms)),
            ("snd_tsbpd_delay_ms", u32::from(self.snd_tsbpd_delay_ms)),
        ] {
            if val < MIN_LATENCY_MS || val > MAX_LATENCY_MS {
                return Err(format!(
                    "{name} = {val} ms is out of [{MIN_LATENCY_MS}, {MAX_LATENCY_MS}] range"
                ));
            }
        }
        Ok(())
    }
}

// ─── HSRSP Extension Block ────────────────────────────────────────────────────

/// `HSRSP` handshake extension — listener's response to `HSREQ`.
///
/// Contains the negotiated (min of both sides) latency values and the
/// flags as acknowledged by the listener.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HsrspBlock {
    /// Listener's SRT version.
    pub srt_version: u32,
    /// Acknowledged flags.
    pub srt_flags: SrtFlags,
    /// Negotiated receiver latency (ms) — `min(caller_recv, listener_snd)`.
    pub recv_tsbpd_delay_ms: u16,
    /// Negotiated sender latency (ms) — `min(caller_snd, listener_recv)`.
    pub snd_tsbpd_delay_ms: u16,
}

impl HsrspBlock {
    /// Negotiate an `HSRSP` from a listener `HsreqBlock` given the listener's
    /// own latency preferences.
    ///
    /// Per the SRT spec the negotiated latency is `max(caller_latency,
    /// listener_latency)` to ensure both sides can satisfy their buffer
    /// requirements.
    #[must_use]
    pub fn negotiate(
        caller_req: &HsreqBlock,
        listener_recv_latency_ms: u16,
        listener_snd_latency_ms: u16,
        listener_version: u32,
    ) -> Self {
        // Negotiated latency = max of what either side needs.
        let recv = caller_req.recv_tsbpd_delay_ms.max(listener_recv_latency_ms);
        let snd  = caller_req.snd_tsbpd_delay_ms.max(listener_snd_latency_ms);

        // Negotiate version: pick minimum supported.
        let version = listener_version.min(caller_req.srt_version);

        // Intersect flags: only enable features both sides support.
        let flags = SrtFlags::new(caller_req.srt_flags.bits() & {
            // Listener always supports the baseline flags in this model.
            SrtFlags::TSBPD_SND | SrtFlags::TSBPD_RCV | SrtFlags::NAK_REPORT
        });

        Self {
            srt_version: version,
            srt_flags: flags,
            recv_tsbpd_delay_ms: recv,
            snd_tsbpd_delay_ms: snd,
        }
    }
}

// ─── Stream ID Extension Block ────────────────────────────────────────────────

/// `SID` extension block — application-level stream routing identifier.
///
/// The stream ID is an arbitrary UTF-8 string up to [`MAX_STREAM_ID_LEN`]
/// bytes, used to route an incoming connection to the correct application
/// handler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamIdBlock {
    /// Stream identifier string.
    pub stream_id: String,
}

impl StreamIdBlock {
    /// Construct a new stream-ID block after validating length.
    pub fn new(stream_id: impl Into<String>) -> Result<Self, String> {
        let s = stream_id.into();
        if s.len() > MAX_STREAM_ID_LEN {
            return Err(format!(
                "stream ID length {} exceeds maximum {MAX_STREAM_ID_LEN}",
                s.len()
            ));
        }
        Ok(Self { stream_id: s })
    }
}

// ─── Core Handshake Packet ────────────────────────────────────────────────────

/// The core SRT handshake packet (control type `0x0000`), excluding extension
/// blocks.
///
/// Field layout follows the SRT RFC §3.2.1 wire format.  All multi-byte
/// fields are big-endian on the wire; here we use host-byte-order types.
#[derive(Debug, Clone)]
pub struct HandshakePacket {
    /// UDT/SRT version — must be [`SRT_MAGIC`] (4).
    pub udt_version: u32,
    /// Encryption field: 0 = no encryption, 2 = AES-128, 3 = AES-192, 4 = AES-256.
    pub encryption_field: u16,
    /// Extension field (bitfield of which extension blocks are present).
    pub extension_field: u16,
    /// Random initial sequence number proposed by the initiating side.
    pub initial_packet_seq_no: u32,
    /// Maximum Segment Size in bytes (including UDP + SRT headers).
    pub mss: u32,
    /// Maximum flow window size in packets.
    pub max_flow_window_size: u32,
    /// Handshake type / phase.
    pub handshake_type: HandshakeType,
    /// SRT socket ID of the sender.
    pub srt_socket_id: u32,
    /// SYN cookie (0 in induction request; set by listener in response).
    pub syn_cookie: u32,
    /// Peer's IPv4 or IPv6 address (16 bytes; IPv4 uses first 4, rest zero).
    pub peer_addr: [u8; 16],
}

impl HandshakePacket {
    /// Construct a new induction request (caller side, phase 1).
    #[must_use]
    pub fn induction_request(socket_id: u32, initial_seq: u32, mss: u32) -> Self {
        Self {
            udt_version: SRT_MAGIC,
            encryption_field: 0,
            extension_field: 0x4A17, // SRT magic extension field for induction
            initial_packet_seq_no: initial_seq,
            mss,
            max_flow_window_size: 8192,
            handshake_type: HandshakeType::Induction,
            srt_socket_id: socket_id,
            syn_cookie: 0,
            peer_addr: [0u8; 16],
        }
    }

    /// Construct a listener's induction response (phase 1 reply).
    #[must_use]
    pub fn induction_response(
        socket_id: u32,
        initial_seq: u32,
        mss: u32,
        syn_cookie: u32,
        peer_addr_v4: [u8; 4],
    ) -> Self {
        let mut peer_addr = [0u8; 16];
        peer_addr[..4].copy_from_slice(&peer_addr_v4);
        Self {
            udt_version: SRT_MAGIC,
            encryption_field: 0,
            extension_field: 0x4A17,
            initial_packet_seq_no: initial_seq,
            mss,
            max_flow_window_size: 8192,
            handshake_type: HandshakeType::WaveaHand,
            srt_socket_id: socket_id,
            syn_cookie,
            peer_addr,
        }
    }

    /// Construct a conclusion request (caller side, phase 2).
    ///
    /// The `extension_field` bits indicate which extension blocks are
    /// included in the payload that follows the core header:
    /// - bit 0: `HSREQ`
    /// - bit 1: `KMREQ`
    /// - bit 2: `SID`
    #[must_use]
    pub fn conclusion_request(
        socket_id: u32,
        initial_seq: u32,
        mss: u32,
        syn_cookie: u32,
        extension_field: u16,
    ) -> Self {
        Self {
            udt_version: SRT_MAGIC,
            encryption_field: 0,
            extension_field,
            initial_packet_seq_no: initial_seq,
            mss,
            max_flow_window_size: 8192,
            handshake_type: HandshakeType::Conclusion,
            srt_socket_id: socket_id,
            syn_cookie,
            peer_addr: [0u8; 16],
        }
    }

    /// Construct a rejection packet from the listener.
    #[must_use]
    pub fn rejection(socket_id: u32, reason: RejectionReason) -> Self {
        Self {
            udt_version: SRT_MAGIC,
            encryption_field: 0,
            extension_field: 0,
            initial_packet_seq_no: 0,
            mss: 0,
            max_flow_window_size: 0,
            handshake_type: HandshakeType::Rejection(reason),
            srt_socket_id: socket_id,
            syn_cookie: 0,
            peer_addr: [0u8; 16],
        }
    }

    /// Validate the invariants of this packet:
    /// - `udt_version` must be [`SRT_MAGIC`]
    /// - `mss` must be ≥ 76 (minimum SRT MTU)
    /// - `max_flow_window_size` must be > 0 (unless rejected/waveahand)
    pub fn validate(&self) -> Result<(), String> {
        if self.udt_version != SRT_MAGIC {
            return Err(format!(
                "invalid UDT version 0x{:08X} (expected 0x{:08X})",
                self.udt_version, SRT_MAGIC
            ));
        }
        // Rejection and WaveaHand packets may have zero MSS.
        let ignore_mss = matches!(
            self.handshake_type,
            HandshakeType::Rejection(_) | HandshakeType::WaveaHand
        );
        if !ignore_mss && self.mss < 76 {
            return Err(format!("MSS {} is below minimum 76", self.mss));
        }
        Ok(())
    }
}

// ─── Handshake State Machine ──────────────────────────────────────────────────

/// Phase of a caller-side SRT handshake state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallerPhase {
    /// Handshake not yet started.
    Idle,
    /// Induction request sent; awaiting listener cookie.
    AwaitingInduction,
    /// Conclusion request sent; awaiting final agreement.
    AwaitingConclusion,
    /// Handshake complete — data transfer may begin.
    Connected,
    /// Handshake failed (rejected or timed out).
    Failed(RejectionReason),
}

/// Caller-side SRT handshake state machine.
///
/// Drive by calling [`CallerHandshake::start`] to generate the induction
/// packet, then [`CallerHandshake::on_induction_response`] when the listener
/// cookie arrives, then [`CallerHandshake::on_conclusion_response`] to
/// finalize.
#[derive(Debug)]
pub struct CallerHandshake {
    /// Local socket ID.
    pub socket_id: u32,
    /// Initial sequence number (randomly chosen; caller picks this).
    pub initial_seq: u32,
    /// MSS for this connection.
    pub mss: u32,
    /// SRT capability request block.
    pub hsreq: HsreqBlock,
    /// Optional stream ID.
    pub stream_id: Option<StreamIdBlock>,
    /// Current phase.
    pub phase: CallerPhase,
    /// SYN cookie received from the listener in phase 1.
    pub syn_cookie: u32,
}

impl CallerHandshake {
    /// Create a new caller handshake with the given parameters.
    #[must_use]
    pub fn new(
        socket_id: u32,
        initial_seq: u32,
        mss: u32,
        recv_latency_ms: u16,
        snd_latency_ms: u16,
    ) -> Self {
        Self {
            socket_id,
            initial_seq,
            mss,
            hsreq: HsreqBlock::new(recv_latency_ms, snd_latency_ms),
            stream_id: None,
            phase: CallerPhase::Idle,
            syn_cookie: 0,
        }
    }

    /// Attach a stream ID to this handshake.
    pub fn with_stream_id(mut self, sid: StreamIdBlock) -> Self {
        self.stream_id = Some(sid);
        self
    }

    /// Generate the induction (phase 1) packet.
    ///
    /// Transitions the state machine from `Idle` → `AwaitingInduction`.
    pub fn start(&mut self) -> Result<HandshakePacket, String> {
        if self.phase != CallerPhase::Idle {
            return Err(format!("cannot start in phase {:?}", self.phase));
        }
        self.phase = CallerPhase::AwaitingInduction;
        Ok(HandshakePacket::induction_request(
            self.socket_id,
            self.initial_seq,
            self.mss,
        ))
    }

    /// Process the listener's induction response and generate the conclusion
    /// packet.
    ///
    /// Validates the response, stores the SYN cookie, then transitions
    /// `AwaitingInduction` → `AwaitingConclusion`.
    pub fn on_induction_response(
        &mut self,
        response: &HandshakePacket,
    ) -> Result<HandshakePacket, String> {
        if self.phase != CallerPhase::AwaitingInduction {
            return Err(format!("unexpected induction response in phase {:?}", self.phase));
        }
        if response.handshake_type != HandshakeType::WaveaHand {
            return Err(format!(
                "expected WAVEAHAND, got {}",
                response.handshake_type
            ));
        }
        response.validate()?;

        self.syn_cookie = response.syn_cookie;
        self.phase = CallerPhase::AwaitingConclusion;

        // Build conclusion extension field.
        let mut ext_field: u16 = 1; // bit 0 = HSREQ present
        if self.stream_id.is_some() {
            ext_field |= 1 << 2; // bit 2 = SID present
        }

        Ok(HandshakePacket::conclusion_request(
            self.socket_id,
            self.initial_seq,
            self.mss,
            self.syn_cookie,
            ext_field,
        ))
    }

    /// Process the listener's conclusion response (agreement / rejection).
    ///
    /// On success transitions to `Connected`; on rejection sets `Failed`.
    pub fn on_conclusion_response(
        &mut self,
        response: &HandshakePacket,
    ) -> Result<Option<HsrspBlock>, String> {
        if self.phase != CallerPhase::AwaitingConclusion {
            return Err(format!(
                "unexpected conclusion response in phase {:?}",
                self.phase
            ));
        }
        match response.handshake_type {
            HandshakeType::Rejection(r) => {
                self.phase = CallerPhase::Failed(r);
                Err(format!("connection rejected: {r}"))
            }
            HandshakeType::Agreement => {
                self.phase = CallerPhase::Connected;
                Ok(None)
            }
            HandshakeType::Conclusion => {
                // Two-way conclusion: listener sends its own HSRSP.
                self.phase = CallerPhase::Connected;
                Ok(None)
            }
            other => {
                Err(format!("unexpected handshake type in conclusion response: {other}"))
            }
        }
    }

    /// Returns `true` if the handshake has completed successfully.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.phase == CallerPhase::Connected
    }
}

// ─── Listener-side Handshake ──────────────────────────────────────────────────

/// Phase of a listener-side SRT handshake state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListenerPhase {
    /// Waiting for an induction request.
    AwaitingInduction,
    /// Induction response sent; waiting for caller's conclusion.
    AwaitingConclusion,
    /// Handshake complete.
    Connected,
    /// Handshake failed.
    Rejected(RejectionReason),
}

/// Listener-side SRT handshake state machine.
///
/// The listener is stateless until it issues a SYN cookie, but we track the
/// phase here for correctness.
#[derive(Debug)]
pub struct ListenerHandshake {
    /// Listener's local socket ID.
    pub socket_id: u32,
    /// MSS for this connection.
    pub mss: u32,
    /// Listener's receive latency preference (ms).
    pub recv_latency_ms: u16,
    /// Listener's send latency preference (ms).
    pub snd_latency_ms: u16,
    /// Listener's SRT version.
    pub srt_version: u32,
    /// Issued SYN cookie (computed from caller address + timestamp).
    pub syn_cookie: u32,
    /// Current phase.
    pub phase: ListenerPhase,
    /// Negotiated HSRSP (available after conclusion).
    pub negotiated: Option<HsrspBlock>,
}

impl ListenerHandshake {
    /// Create a new listener handshake.
    #[must_use]
    pub fn new(
        socket_id: u32,
        mss: u32,
        recv_latency_ms: u16,
        snd_latency_ms: u16,
    ) -> Self {
        Self {
            socket_id,
            mss,
            recv_latency_ms,
            snd_latency_ms,
            srt_version: SRT_VERSION_1_4,
            syn_cookie: 0,
            phase: ListenerPhase::AwaitingInduction,
            negotiated: None,
        }
    }

    /// Process a caller's induction request and generate the induction response.
    ///
    /// `cookie` should be derived from the caller's IP+port + current epoch
    /// (in production).  Here the caller supplies it directly for testability.
    pub fn on_induction_request(
        &mut self,
        request: &HandshakePacket,
        cookie: u32,
        caller_ipv4: [u8; 4],
    ) -> Result<HandshakePacket, String> {
        if self.phase != ListenerPhase::AwaitingInduction {
            return Err(format!("unexpected induction in phase {:?}", self.phase));
        }
        if request.handshake_type != HandshakeType::Induction {
            return Err(format!(
                "expected INDUCTION, got {}",
                request.handshake_type
            ));
        }
        request.validate()?;

        self.syn_cookie = cookie;
        self.phase = ListenerPhase::AwaitingConclusion;

        Ok(HandshakePacket::induction_response(
            self.socket_id,
            request.initial_packet_seq_no,
            self.mss.min(request.mss),
            cookie,
            caller_ipv4,
        ))
    }

    /// Process a caller's conclusion request and generate the agreement.
    ///
    /// Validates the SYN cookie and version, performs latency negotiation,
    /// and emits either an `AGREEMENT` or a rejection packet.
    pub fn on_conclusion_request(
        &mut self,
        request: &HandshakePacket,
        caller_hsreq: &HsreqBlock,
    ) -> Result<HandshakePacket, String> {
        if self.phase != ListenerPhase::AwaitingConclusion {
            return Err(format!("unexpected conclusion in phase {:?}", self.phase));
        }
        if request.handshake_type != HandshakeType::Conclusion {
            return Err(format!(
                "expected CONCLUSION, got {}",
                request.handshake_type
            ));
        }

        // Cookie validation.
        if request.syn_cookie != self.syn_cookie {
            self.phase = ListenerPhase::Rejected(RejectionReason::BadRequest);
            return Ok(HandshakePacket::rejection(
                self.socket_id,
                RejectionReason::BadRequest,
            ));
        }

        // Version check.
        if caller_hsreq.srt_version < SRT_VERSION_1_4 {
            self.phase = ListenerPhase::Rejected(RejectionReason::Version);
            return Ok(HandshakePacket::rejection(
                self.socket_id,
                RejectionReason::Version,
            ));
        }

        // Validate caller's extension block.
        if let Err(e) = caller_hsreq.validate() {
            self.phase = ListenerPhase::Rejected(RejectionReason::BadRequest);
            return Err(format!("caller HSREQ invalid: {e}"));
        }

        // Negotiate.
        let hsrsp = HsrspBlock::negotiate(
            caller_hsreq,
            self.recv_latency_ms,
            self.snd_latency_ms,
            self.srt_version,
        );
        self.negotiated = Some(hsrsp);
        self.phase = ListenerPhase::Connected;

        // Build the agreement packet.
        let mut agreement = HandshakePacket::conclusion_request(
            self.socket_id,
            request.initial_packet_seq_no,
            self.mss.min(request.mss),
            self.syn_cookie,
            0,
        );
        agreement.handshake_type = HandshakeType::Agreement;
        Ok(agreement)
    }

    /// Returns `true` if the handshake completed successfully.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.phase == ListenerPhase::Connected
    }

    /// Reject the current handshake with the given reason.
    ///
    /// Returns the rejection packet to send to the caller.
    pub fn reject(&mut self, reason: RejectionReason) -> HandshakePacket {
        self.phase = ListenerPhase::Rejected(reason);
        HandshakePacket::rejection(self.socket_id, reason)
    }
}

// ─── SYN Cookie Utility ───────────────────────────────────────────────────────

/// Compute a simple SYN cookie from the caller's IP address and a secret
/// epoch value.
///
/// In a real implementation the epoch would be derived from the system clock
/// modulo a short window (e.g. 64 seconds) so that stale cookies expire.
/// Here we use a deterministic but collision-resistant hash suitable for
/// testing.
///
/// The formula: `FNV-1a(ip_bytes ++ epoch_bytes)` truncated to 32 bits.
#[must_use]
pub fn compute_syn_cookie(caller_ipv4: [u8; 4], epoch: u32) -> u32 {
    const FNV_OFFSET: u32 = 2_166_136_261;
    const FNV_PRIME: u32 = 16_777_619;

    let mut hash = FNV_OFFSET;
    for byte in caller_ipv4.iter().chain(epoch.to_le_bytes().iter()) {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── HandshakeType round-trip ────────────────────────────────────────────

    #[test]
    fn test_handshake_type_wire_roundtrip() {
        let types = [
            HandshakeType::Induction,
            HandshakeType::WaveaHand,
            HandshakeType::Conclusion,
            HandshakeType::Agreement,
            HandshakeType::Rejection(RejectionReason::Passphrase),
        ];
        for t in types {
            let wire = t.to_wire();
            let decoded = HandshakeType::from_wire(wire).expect("should decode");
            assert_eq!(t, decoded, "roundtrip failed for {t}");
        }
    }

    #[test]
    fn test_rejection_type_carries_reason() {
        let t = HandshakeType::Rejection(RejectionReason::Forbidden);
        let wire = t.to_wire();
        let decoded = HandshakeType::from_wire(wire).expect("should decode rejection");
        assert_eq!(decoded, HandshakeType::Rejection(RejectionReason::Forbidden));
    }

    #[test]
    fn test_unknown_wire_value_returns_none() {
        // 0x0000_0002 is not a valid handshake type.
        assert!(HandshakeType::from_wire(2).is_none());
    }

    // ── RejectionReason ─────────────────────────────────────────────────────

    #[test]
    fn test_rejection_reason_from_code() {
        assert_eq!(RejectionReason::from_code(6), RejectionReason::Passphrase);
        assert_eq!(RejectionReason::from_code(9), RejectionReason::Unauthorized);
        assert_eq!(RejectionReason::from_code(1000), RejectionReason::ApplicationDefined);
        assert_eq!(RejectionReason::from_code(9999), RejectionReason::ApplicationDefined);
    }

    #[test]
    fn test_rejection_reason_display() {
        assert_eq!(RejectionReason::Passphrase.to_string(), "PASSPHRASE");
        assert_eq!(RejectionReason::Version.to_string(), "VERSION");
    }

    // ── SrtFlags ────────────────────────────────────────────────────────────

    #[test]
    fn test_srt_flags_set_and_has() {
        let mut flags = SrtFlags::new(0);
        assert!(!flags.has(SrtFlags::TSBPD_SND));
        flags.set(SrtFlags::TSBPD_SND);
        assert!(flags.has(SrtFlags::TSBPD_SND));
        flags.clear(SrtFlags::TSBPD_SND);
        assert!(!flags.has(SrtFlags::TSBPD_SND));
    }

    #[test]
    fn test_srt_flags_multiple_bits() {
        let flags = SrtFlags::new(SrtFlags::TSBPD_SND | SrtFlags::NAK_REPORT);
        assert!(flags.has(SrtFlags::TSBPD_SND));
        assert!(flags.has(SrtFlags::NAK_REPORT));
        assert!(!flags.has(SrtFlags::HAICRYPT_OFF));
    }

    // ── HsreqBlock ──────────────────────────────────────────────────────────

    #[test]
    fn test_hsreq_validate_ok() {
        let hsreq = HsreqBlock::new(120, 120);
        assert!(hsreq.validate().is_ok());
    }

    #[test]
    fn test_hsreq_validate_latency_too_low() {
        let mut hsreq = HsreqBlock::new(10, 120); // 10 ms < MIN_LATENCY_MS=20
        hsreq.recv_tsbpd_delay_ms = 10;
        assert!(hsreq.validate().is_err());
    }

    #[test]
    fn test_hsreq_validate_old_version() {
        let mut hsreq = HsreqBlock::new(120, 120);
        hsreq.srt_version = 0x0001_0300; // 1.3 < minimum 1.4
        assert!(hsreq.validate().is_err());
    }

    // ── HsrspBlock negotiation ───────────────────────────────────────────────

    #[test]
    fn test_hsrsp_negotiation_takes_max_latency() {
        let caller_req = HsreqBlock::new(200, 100);
        // Listener wants 300 ms recv, 150 ms snd.
        let hsrsp = HsrspBlock::negotiate(&caller_req, 300, 150, SRT_VERSION_1_4);
        // recv = max(200, 300) = 300; snd = max(100, 150) = 150
        assert_eq!(hsrsp.recv_tsbpd_delay_ms, 300);
        assert_eq!(hsrsp.snd_tsbpd_delay_ms, 150);
    }

    #[test]
    fn test_hsrsp_negotiation_uses_min_version() {
        let caller_req = HsreqBlock {
            srt_version: SRT_VERSION_1_5,
            ..HsreqBlock::new(120, 120)
        };
        let hsrsp = HsrspBlock::negotiate(&caller_req, 120, 120, SRT_VERSION_1_4);
        // Should pick min(1.5, 1.4) = 1.4
        assert_eq!(hsrsp.srt_version, SRT_VERSION_1_4);
    }

    // ── StreamIdBlock ────────────────────────────────────────────────────────

    #[test]
    fn test_stream_id_valid() {
        let block = StreamIdBlock::new("my-camera-feed").expect("should construct");
        assert_eq!(block.stream_id, "my-camera-feed");
    }

    #[test]
    fn test_stream_id_too_long() {
        let too_long = "x".repeat(MAX_STREAM_ID_LEN + 1);
        assert!(StreamIdBlock::new(too_long).is_err());
    }

    // ── HandshakePacket ──────────────────────────────────────────────────────

    #[test]
    fn test_induction_request_validates() {
        let pkt = HandshakePacket::induction_request(42, 1000, 1500);
        assert!(pkt.validate().is_ok());
        assert_eq!(pkt.handshake_type, HandshakeType::Induction);
        assert_eq!(pkt.udt_version, SRT_MAGIC);
    }

    #[test]
    fn test_induction_response_carries_cookie() {
        let pkt = HandshakePacket::induction_response(99, 1000, 1500, 0xDEAD_BEEF, [192, 168, 1, 1]);
        assert_eq!(pkt.syn_cookie, 0xDEAD_BEEF);
        assert_eq!(pkt.peer_addr[..4], [192, 168, 1, 1]);
        assert_eq!(pkt.handshake_type, HandshakeType::WaveaHand);
    }

    #[test]
    fn test_rejection_packet_valid_type() {
        let pkt = HandshakePacket::rejection(7, RejectionReason::Overloaded);
        assert_eq!(pkt.handshake_type, HandshakeType::Rejection(RejectionReason::Overloaded));
    }

    // ── Full caller/listener handshake round-trip ────────────────────────────

    #[test]
    fn test_full_handshake_roundtrip() {
        // Caller setup
        let mut caller = CallerHandshake::new(
            0xAABB_CCDD, // socket_id
            12345,       // initial_seq
            1500,        // mss
            120,         // recv_latency_ms
            120,         // snd_latency_ms
        );

        // Listener setup
        let mut listener = ListenerHandshake::new(
            0x1122_3344, // socket_id
            1500,        // mss
            200,         // recv_latency_ms (listener wants 200 ms)
            150,         // snd_latency_ms
        );

        // Phase 1: caller → listener (induction request)
        let induction_req = caller.start().expect("start should succeed");
        assert_eq!(caller.phase, CallerPhase::AwaitingInduction);

        // Listener responds with cookie
        let cookie = compute_syn_cookie([10, 0, 0, 1], 42);
        let induction_resp = listener
            .on_induction_request(&induction_req, cookie, [10, 0, 0, 1])
            .expect("listener should accept induction");
        assert_eq!(listener.phase, ListenerPhase::AwaitingConclusion);
        assert_eq!(induction_resp.syn_cookie, cookie);

        // Phase 2: caller processes cookie, sends conclusion
        let conclusion_req = caller
            .on_induction_response(&induction_resp)
            .expect("caller should accept induction response");
        assert_eq!(caller.phase, CallerPhase::AwaitingConclusion);
        assert_eq!(conclusion_req.syn_cookie, cookie);

        // Listener processes conclusion and sends agreement
        let hsreq = HsreqBlock::new(120, 120);
        let agreement = listener
            .on_conclusion_request(&conclusion_req, &hsreq)
            .expect("listener should send agreement");
        assert_eq!(agreement.handshake_type, HandshakeType::Agreement);
        assert!(listener.is_connected());

        // Caller finalises on agreement
        caller
            .on_conclusion_response(&agreement)
            .expect("caller should accept agreement");
        assert!(caller.is_connected());
    }

    #[test]
    fn test_handshake_rejection_on_bad_cookie() {
        let mut listener = ListenerHandshake::new(1, 1500, 120, 120);

        // Fake an induction so listener is in AwaitingConclusion.
        let induction_req = HandshakePacket::induction_request(2, 1000, 1500);
        let real_cookie = 0xCAFE_BABE;
        listener
            .on_induction_request(&induction_req, real_cookie, [127, 0, 0, 1])
            .expect("induction ok");

        // Caller sends conclusion with WRONG cookie.
        let mut bad_conclusion = HandshakePacket::conclusion_request(2, 1000, 1500, 0xDEAD_BEEF, 1);
        bad_conclusion.handshake_type = HandshakeType::Conclusion;
        let hsreq = HsreqBlock::new(120, 120);
        let rejection = listener
            .on_conclusion_request(&bad_conclusion, &hsreq)
            .expect("should return rejection packet");
        assert_eq!(
            rejection.handshake_type,
            HandshakeType::Rejection(RejectionReason::BadRequest)
        );
        assert_eq!(
            listener.phase,
            ListenerPhase::Rejected(RejectionReason::BadRequest)
        );
    }

    // ── SYN cookie utility ───────────────────────────────────────────────────

    #[test]
    fn test_syn_cookie_deterministic() {
        let ip = [192, 168, 1, 100];
        let c1 = compute_syn_cookie(ip, 100);
        let c2 = compute_syn_cookie(ip, 100);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_syn_cookie_different_ip_different_result() {
        let c1 = compute_syn_cookie([10, 0, 0, 1], 50);
        let c2 = compute_syn_cookie([10, 0, 0, 2], 50);
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_syn_cookie_different_epoch_different_result() {
        let ip = [172, 16, 0, 1];
        let c1 = compute_syn_cookie(ip, 0);
        let c2 = compute_syn_cookie(ip, 1);
        assert_ne!(c1, c2);
    }

    // ── Display ──────────────────────────────────────────────────────────────

    #[test]
    fn test_handshake_type_display() {
        assert_eq!(HandshakeType::Induction.to_string(), "INDUCTION");
        assert_eq!(HandshakeType::WaveaHand.to_string(), "WAVEAHAND");
        assert_eq!(HandshakeType::Conclusion.to_string(), "CONCLUSION");
        assert_eq!(HandshakeType::Agreement.to_string(), "AGREEMENT");
        let rej = HandshakeType::Rejection(RejectionReason::Forbidden);
        assert_eq!(rej.to_string(), "REJECTION(FORBIDDEN)");
    }
}
