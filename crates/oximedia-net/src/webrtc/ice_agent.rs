//! ICE (Interactive Connectivity Establishment) agent.
//!
//! This module implements the ICE agent which handles:
//! - Candidate gathering
//! - Connectivity checks
//! - Pair nomination

#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use super::ice::{CandidateType, IceCandidate, IceServer, TransportProtocol};
use super::stun::{Attribute, AttributeType, Message, MessageType};
use crate::error::{NetError, NetResult};
use std::net::{IpAddr, SocketAddr, UdpSocket};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::net::UdpSocket as TokioUdpSocket;

/// ICE agent role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IceRole {
    /// Controlling (initiator).
    Controlling,
    /// Controlled (responder).
    Controlled,
}

/// ICE connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IceConnectionState {
    /// New connection.
    New,
    /// Gathering candidates.
    Gathering,
    /// Checking connectivity.
    Checking,
    /// Connected.
    Connected,
    /// Completed all checks.
    Completed,
    /// Failed to connect.
    Failed,
    /// Disconnected.
    Disconnected,
    /// Closed.
    Closed,
}

/// ICE gathering state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IceGatheringState {
    /// New.
    New,
    /// Gathering.
    Gathering,
    /// Complete.
    Complete,
}

/// Candidate pair state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidatePairState {
    /// Waiting to check.
    Waiting,
    /// In progress.
    InProgress,
    /// Succeeded.
    Succeeded,
    /// Failed.
    Failed,
    /// Frozen (not ready to check).
    Frozen,
}

/// Candidate pair for connectivity checks.
#[derive(Debug, Clone)]
pub struct CandidatePair {
    /// Local candidate.
    pub local: IceCandidate,
    /// Remote candidate.
    pub remote: IceCandidate,
    /// Pair state.
    pub state: CandidatePairState,
    /// Pair priority.
    pub priority: u64,
    /// Nominated flag.
    pub nominated: bool,
    /// Last check time.
    pub last_check: Option<Instant>,
    /// Number of checks sent.
    pub checks_sent: u32,
}

impl CandidatePair {
    /// Creates a new candidate pair.
    #[must_use]
    pub fn new(local: IceCandidate, remote: IceCandidate, controlling: bool) -> Self {
        let priority = Self::calculate_priority(&local, &remote, controlling);
        Self {
            local,
            remote,
            state: CandidatePairState::Waiting,
            priority,
            nominated: false,
            last_check: None,
            checks_sent: 0,
        }
    }

    /// Calculates pair priority according to RFC 5245.
    #[must_use]
    pub fn calculate_priority(
        local: &IceCandidate,
        remote: &IceCandidate,
        controlling: bool,
    ) -> u64 {
        let g = if controlling {
            local.priority
        } else {
            remote.priority
        };
        let d = if controlling {
            remote.priority
        } else {
            local.priority
        };

        let g = u64::from(g);
        let d = u64::from(d);

        (1_u64 << 32).wrapping_mul(g.min(d)) + 2 * g.max(d) + if g > d { 1 } else { 0 }
    }

    /// Returns the local socket address.
    pub fn local_addr(&self) -> NetResult<SocketAddr> {
        let ip: IpAddr = self
            .local
            .address
            .parse()
            .map_err(|_| NetError::parse(0, "Invalid local address"))?;
        Ok(SocketAddr::new(ip, self.local.port))
    }

    /// Returns the remote socket address.
    pub fn remote_addr(&self) -> NetResult<SocketAddr> {
        let ip: IpAddr = self
            .remote
            .address
            .parse()
            .map_err(|_| NetError::parse(0, "Invalid remote address"))?;
        Ok(SocketAddr::new(ip, self.remote.port))
    }
}

/// ICE agent configuration.
#[derive(Debug, Clone)]
pub struct IceAgentConfig {
    /// ICE servers (STUN/TURN).
    pub ice_servers: Vec<IceServer>,
    /// Local ICE ufrag.
    pub local_ufrag: String,
    /// Local ICE password.
    pub local_pwd: String,
    /// Remote ICE ufrag.
    pub remote_ufrag: Option<String>,
    /// Remote ICE password.
    pub remote_pwd: Option<String>,
    /// Controlling role.
    pub controlling: bool,
    /// Tie breaker value.
    pub tie_breaker: u64,
}

impl Default for IceAgentConfig {
    fn default() -> Self {
        use rand::RngExt;
        let mut rng = rand::rng();

        Self {
            ice_servers: Vec::new(),
            local_ufrag: generate_ice_string(8),
            local_pwd: generate_ice_string(24),
            remote_ufrag: None,
            remote_pwd: None,
            controlling: true,
            tie_breaker: rng.random::<u64>(),
        }
    }
}

/// ICE agent.
pub struct IceAgent {
    /// Configuration.
    config: IceAgentConfig,
    /// Connection state.
    state: Arc<Mutex<IceConnectionState>>,
    /// Gathering state.
    gathering_state: Arc<Mutex<IceGatheringState>>,
    /// Local candidates.
    local_candidates: Arc<Mutex<Vec<IceCandidate>>>,
    /// Remote candidates.
    remote_candidates: Arc<Mutex<Vec<IceCandidate>>>,
    /// Candidate pairs.
    pairs: Arc<Mutex<Vec<CandidatePair>>>,
    /// Selected pair.
    selected_pair: Arc<Mutex<Option<CandidatePair>>>,
    /// UDP socket for connectivity checks.
    socket: Arc<Mutex<Option<Arc<TokioUdpSocket>>>>,
}

impl IceAgent {
    /// Creates a new ICE agent.
    #[must_use]
    pub fn new(config: IceAgentConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(IceConnectionState::New)),
            gathering_state: Arc::new(Mutex::new(IceGatheringState::New)),
            local_candidates: Arc::new(Mutex::new(Vec::new())),
            remote_candidates: Arc::new(Mutex::new(Vec::new())),
            pairs: Arc::new(Mutex::new(Vec::new())),
            selected_pair: Arc::new(Mutex::new(None)),
            socket: Arc::new(Mutex::new(None)),
        }
    }

    /// Gathers local candidates.
    pub async fn gather_candidates(&self) -> NetResult<Vec<IceCandidate>> {
        *self
            .gathering_state
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = IceGatheringState::Gathering;

        let mut candidates = Vec::new();

        // Gather host candidates
        candidates.extend(self.gather_host_candidates()?);

        // Gather server reflexive candidates from STUN servers
        for server in &self.config.ice_servers {
            if !server.is_turn() {
                if let Some(srflx) = self.gather_srflx_candidate(server).await? {
                    candidates.push(srflx);
                }
            }
        }

        // Store candidates
        *self
            .local_candidates
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = candidates.clone();
        *self
            .gathering_state
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = IceGatheringState::Complete;

        Ok(candidates)
    }

    /// Gathers host candidates from local interfaces.
    fn gather_host_candidates(&self) -> NetResult<Vec<IceCandidate>> {
        let mut candidates = Vec::new();

        // Get local addresses
        let socket = UdpSocket::bind("0.0.0.0:0")
            .map_err(|e| NetError::connection(format!("Failed to bind socket: {e}")))?;

        let local_addr = socket
            .local_addr()
            .map_err(|e| NetError::connection(format!("Failed to get local address: {e}")))?;

        // Create host candidate
        let foundation = super::ice::compute_foundation(
            CandidateType::Host,
            &local_addr.ip().to_string(),
            TransportProtocol::Udp,
            None,
        );

        let candidate =
            IceCandidate::host(foundation, local_addr.ip().to_string(), local_addr.port());

        candidates.push(candidate);

        // Store socket for later use
        drop(socket); // Close the sync socket

        Ok(candidates)
    }

    /// Gathers server reflexive candidate from STUN server.
    async fn gather_srflx_candidate(&self, server: &IceServer) -> NetResult<Option<IceCandidate>> {
        if server.urls.is_empty() {
            return Ok(None);
        }

        // Parse STUN server URL (simple parsing)
        let url = &server.urls[0];
        let addr = url
            .strip_prefix("stun:")
            .or_else(|| url.strip_prefix("stun://"))
            .ok_or_else(|| NetError::invalid_url("Invalid STUN URL"))?;

        let server_addr: SocketAddr = tokio::net::lookup_host(addr)
            .await
            .map_err(|e| NetError::connection(format!("Failed to resolve STUN server: {e}")))?
            .next()
            .ok_or_else(|| NetError::connection("No addresses found for STUN server"))?;

        // Create socket
        let socket = TokioUdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| NetError::connection(format!("Failed to bind socket: {e}")))?;

        let local_addr = socket
            .local_addr()
            .map_err(|e| NetError::connection(format!("Failed to get local address: {e}")))?;

        // Send binding request
        let request = Message::binding_request();
        let encoded = request.encode();

        socket
            .send_to(&encoded, server_addr)
            .await
            .map_err(|e| NetError::connection(format!("Failed to send STUN request: {e}")))?;

        // Wait for response (with timeout)
        let mut buf = [0u8; 2048];
        let result = tokio::time::timeout(Duration::from_secs(5), socket.recv_from(&mut buf)).await;

        match result {
            Ok(Ok((len, _))) => {
                let response = Message::parse(&buf[..len])?;

                if response.message_type == MessageType::BindingResponse {
                    // Extract XOR-MAPPED-ADDRESS
                    if let Some(attr) = response.get_attribute(AttributeType::XorMappedAddress) {
                        let mapped_addr =
                            attr.parse_xor_mapped_address(&response.transaction_id)?;

                        let foundation = super::ice::compute_foundation(
                            CandidateType::ServerReflexive,
                            &mapped_addr.ip().to_string(),
                            TransportProtocol::Udp,
                            Some(addr),
                        );

                        let candidate = IceCandidate::server_reflexive(
                            foundation,
                            mapped_addr.ip().to_string(),
                            mapped_addr.port(),
                            local_addr.ip().to_string(),
                            local_addr.port(),
                        );

                        return Ok(Some(candidate));
                    }
                }
            }
            Ok(Err(e)) => {
                return Err(NetError::connection(format!(
                    "Failed to receive STUN response: {e}"
                )));
            }
            Err(_) => {
                return Err(NetError::timeout("STUN request timeout"));
            }
        }

        Ok(None)
    }

    /// Adds remote candidates.
    pub fn add_remote_candidate(&self, candidate: IceCandidate) {
        let mut remote = self
            .remote_candidates
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        remote.push(candidate);

        // Create candidate pairs
        self.create_pairs();
    }

    /// Sets remote ICE parameters.
    pub fn set_remote_params(&mut self, ufrag: String, pwd: String) {
        self.config.remote_ufrag = Some(ufrag);
        self.config.remote_pwd = Some(pwd);
    }

    /// Creates candidate pairs.
    fn create_pairs(&self) {
        let local = self
            .local_candidates
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let remote = self
            .remote_candidates
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut pairs = self.pairs.lock().unwrap_or_else(|e| e.into_inner());

        pairs.clear();

        for local_cand in local.iter() {
            for remote_cand in remote.iter() {
                let pair = CandidatePair::new(
                    local_cand.clone(),
                    remote_cand.clone(),
                    self.config.controlling,
                );
                pairs.push(pair);
            }
        }

        // Sort by priority (highest first)
        pairs.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Performs connectivity checks.
    pub async fn check_connectivity(&self) -> NetResult<()> {
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = IceConnectionState::Checking;

        // Get pairs to check
        let pairs_to_check = {
            let pairs = self.pairs.lock().unwrap_or_else(|e| e.into_inner());
            pairs.clone()
        };

        if pairs_to_check.is_empty() {
            *self.state.lock().unwrap_or_else(|e| e.into_inner()) = IceConnectionState::Failed;
            return Err(NetError::connection("No candidate pairs to check"));
        }

        // Check pairs in priority order
        for pair in &pairs_to_check {
            if let Ok(true) = self.check_pair(pair).await {
                // Found a working pair
                *self.selected_pair.lock().unwrap_or_else(|e| e.into_inner()) = Some(pair.clone());
                *self.state.lock().unwrap_or_else(|e| e.into_inner()) =
                    IceConnectionState::Connected;
                return Ok(());
            }
        }

        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = IceConnectionState::Failed;
        Err(NetError::connection("No valid candidate pairs found"))
    }

    /// Checks a single candidate pair.
    async fn check_pair(&self, pair: &CandidatePair) -> NetResult<bool> {
        let local_addr = pair.local_addr()?;
        let remote_addr = pair.remote_addr()?;

        // Create or reuse socket
        let socket = {
            let mut sock = self.socket.lock().unwrap_or_else(|e| e.into_inner());
            match sock.as_ref() {
                Some(s) => s.clone(),
                None => {
                    let new_sock = Arc::new(
                        TokioUdpSocket::bind(local_addr)
                            .await
                            .map_err(|e| NetError::connection(format!("Failed to bind: {e}")))?,
                    );
                    *sock = Some(new_sock.clone());
                    new_sock
                }
            }
        };

        // Build STUN binding request
        let username = format!(
            "{}:{}",
            self.config.remote_ufrag.as_ref().unwrap_or(&String::new()),
            self.config.local_ufrag
        );

        let mut request = Message::binding_request()
            .with_attribute(Attribute::username(&username))
            .with_attribute(Attribute::priority(pair.local.priority));

        if self.config.controlling {
            request = request.with_attribute(Attribute::ice_controlling(self.config.tie_breaker));
            if pair.nominated {
                request = request.with_attribute(Attribute::use_candidate());
            }
        } else {
            request = request.with_attribute(Attribute::ice_controlled(self.config.tie_breaker));
        }

        // Encode with message integrity
        let pwd = self
            .config
            .remote_pwd
            .as_ref()
            .unwrap_or(&self.config.local_pwd);
        let encoded = request.encode_with_integrity(pwd);

        // Send request
        socket
            .send_to(&encoded, remote_addr)
            .await
            .map_err(|e| NetError::connection(format!("Failed to send: {e}")))?;

        // Wait for response
        let mut buf = [0u8; 2048];
        let result =
            tokio::time::timeout(Duration::from_millis(500), socket.recv_from(&mut buf)).await;

        match result {
            Ok(Ok((len, _))) => {
                let response = Message::parse(&buf[..len])?;
                Ok(response.message_type == MessageType::BindingResponse)
            }
            _ => Ok(false),
        }
    }

    /// Gets the connection state.
    #[must_use]
    pub fn state(&self) -> IceConnectionState {
        *self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Gets the gathering state.
    #[must_use]
    pub fn gathering_state(&self) -> IceGatheringState {
        *self
            .gathering_state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Gets local candidates.
    #[must_use]
    pub fn local_candidates(&self) -> Vec<IceCandidate> {
        self.local_candidates
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Gets the selected pair.
    #[must_use]
    pub fn selected_pair(&self) -> Option<CandidatePair> {
        self.selected_pair
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Gets the socket for data transmission.
    #[must_use]
    pub fn socket(&self) -> Option<Arc<TokioUdpSocket>> {
        self.socket
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

/// Generates a random ICE string.
#[must_use]
fn generate_ice_string(length: usize) -> String {
    use rand::RngExt;
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut rng = rand::rng();

    (0..length)
        .map(|_| {
            let idx = rng.random_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ice_role() {
        assert_eq!(IceRole::Controlling, IceRole::Controlling);
        assert_ne!(IceRole::Controlling, IceRole::Controlled);
    }

    #[test]
    fn test_ice_config_default() {
        let config = IceAgentConfig::default();
        assert!(!config.local_ufrag.is_empty());
        assert!(!config.local_pwd.is_empty());
        assert!(config.controlling);
    }

    #[test]
    fn test_candidate_pair_priority() {
        let local = IceCandidate::host("1", "192.168.1.1", 5000).with_priority(100);
        let remote = IceCandidate::host("2", "192.168.1.2", 5001).with_priority(200);

        let pair = CandidatePair::new(local, remote, true);
        assert!(pair.priority > 0);
    }

    #[test]
    fn test_generate_ice_string() {
        let s = generate_ice_string(16);
        assert_eq!(s.len(), 16);
        assert!(s.chars().all(|c| c.is_alphanumeric()));
    }
}
