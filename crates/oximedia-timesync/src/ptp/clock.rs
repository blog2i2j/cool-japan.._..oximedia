//! PTP clock implementations (Ordinary Clock and Boundary Clock).

use super::bmca::{recommend_state, PortState};
use super::dataset::{CurrentDataSet, DefaultDataSet, ParentDataSet, TimePropertiesDataSet};
use super::message::{
    AnnounceMessage, DelayReqMessage, DelayRespMessage, Flags, FollowUpMessage, Header,
    MessageType, SyncMessage,
};
use super::{ClockIdentity, CommunicationMode, Domain, PortIdentity, PtpTimestamp};
use crate::error::{TimeSyncError, TimeSyncResult};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::UdpSocket;
use tracing::{debug, info};

/// PTP multicast address for event messages (port 319).
const PTP_MULTICAST_EVENT: &str = "224.0.1.129:319";

/// PTP Ordinary Clock (OC) implementation.
///
/// An ordinary clock has a single PTP port and can operate as master or slave.
pub struct OrdinaryClock {
    /// Default dataset
    default_ds: DefaultDataSet,
    /// Current dataset
    current_ds: CurrentDataSet,
    /// Parent dataset
    parent_ds: ParentDataSet,
    /// Time properties dataset
    time_props_ds: TimePropertiesDataSet,
    /// Port state
    port_state: PortState,
    /// Current master (if slave)
    current_master: Option<PortIdentity>,
    /// Sequence ID counter
    sequence_id: u16,
    /// Socket for communication
    socket: Option<Arc<UdpSocket>>,
    /// Communication mode
    comm_mode: CommunicationMode,
    /// Sync interval (log2 seconds, e.g., 0 = 1s, -1 = 0.5s)
    sync_interval: i8,
    /// Delay request interval
    #[allow(dead_code)]
    delay_req_interval: i8,
    /// Announce interval
    announce_interval: i8,
    /// Received announce messages
    #[allow(dead_code)]
    received_announces: HashMap<PortIdentity, AnnounceMessage>,
    // ── Two-step slave state ────────────────────────────────────────────
    /// Reception timestamp of the last Sync message (t2).
    sync_receive_time: Option<PtpTimestamp>,
    /// Sequence ID of the Sync we are waiting for Follow_Up on.
    pending_sync_seq: Option<u16>,
    /// Origin timestamp from the last Sync (t1) — only valid in one-step mode.
    sync_origin: Option<PtpTimestamp>,
    /// Precise origin timestamp from Follow_Up (t1 in two-step).
    follow_up_origin: Option<PtpTimestamp>,
    /// Transmission timestamp of the last Delay_Req (t3).
    delay_req_send_time: Option<PtpTimestamp>,
    /// Sequence ID of the Delay_Req we are waiting for a response on.
    pending_delay_seq: Option<u16>,
    /// Address of the last Sync source (for sending Delay_Req back).
    sync_source_addr: Option<SocketAddr>,
}

impl OrdinaryClock {
    /// Create a new ordinary clock.
    #[must_use]
    pub fn new(clock_identity: ClockIdentity, domain: Domain) -> Self {
        let mut default_ds = DefaultDataSet::new(clock_identity);
        default_ds.domain_number = domain.0;

        Self {
            current_ds: CurrentDataSet::default(),
            parent_ds: ParentDataSet::from_local(&default_ds),
            time_props_ds: TimePropertiesDataSet::default(),
            default_ds,
            port_state: PortState::Initializing,
            current_master: None,
            sequence_id: 0,
            socket: None,
            comm_mode: CommunicationMode::Multicast,
            sync_interval: 0, // 1 second
            delay_req_interval: 0,
            announce_interval: 1, // 2 seconds
            received_announces: HashMap::new(),
            sync_receive_time: None,
            pending_sync_seq: None,
            sync_origin: None,
            follow_up_origin: None,
            delay_req_send_time: None,
            pending_delay_seq: None,
            sync_source_addr: None,
        }
    }

    /// Set communication mode.
    pub fn set_communication_mode(&mut self, mode: CommunicationMode) {
        self.comm_mode = mode;
    }

    /// Set as slave-only clock.
    pub fn set_slave_only(&mut self) {
        self.default_ds.set_slave_only();
    }

    /// Set as grandmaster-capable clock.
    pub fn set_grandmaster_capable(&mut self, clock_class: u8, accuracy: u8) {
        self.default_ds
            .set_grandmaster_capable(clock_class, accuracy);
    }

    /// Bind to a socket.
    pub async fn bind(&mut self, addr: SocketAddr) -> TimeSyncResult<()> {
        let socket = UdpSocket::bind(addr).await?;
        self.socket = Some(Arc::new(socket));
        info!("PTP clock bound to {}", addr);
        Ok(())
    }

    /// Get current port state.
    #[must_use]
    pub fn port_state(&self) -> PortState {
        self.port_state
    }

    /// Get current clock offset from master (nanoseconds).
    #[must_use]
    pub fn offset_from_master(&self) -> i64 {
        self.current_ds.offset_from_master
    }

    /// Get mean path delay (nanoseconds).
    #[must_use]
    pub fn mean_path_delay(&self) -> i64 {
        self.current_ds.mean_path_delay
    }

    /// Handle received PTP message.
    pub async fn handle_message(&mut self, data: &[u8], src: SocketAddr) -> TimeSyncResult<()> {
        if data.len() < 34 {
            return Err(TimeSyncError::InvalidPacket("Packet too short".to_string()));
        }

        let mut buf = data;
        let header = Header::deserialize(&mut buf)?;

        // Check domain
        if header.domain.0 != self.default_ds.domain_number {
            debug!("Ignoring message from different domain");
            return Ok(());
        }

        match header.message_type {
            MessageType::Sync => {
                let sync = SyncMessage::deserialize(data)?;
                self.handle_sync(sync, src).await?;
            }
            MessageType::FollowUp => {
                let follow_up = FollowUpMessage::deserialize(data)?;
                self.handle_follow_up(follow_up).await?;
            }
            MessageType::DelayReq => {
                let delay_req = DelayReqMessage::deserialize(data)?;
                self.handle_delay_req(delay_req, src).await?;
            }
            MessageType::DelayResp => {
                let delay_resp = DelayRespMessage::deserialize(data)?;
                self.handle_delay_resp(delay_resp).await?;
            }
            MessageType::Announce => {
                let announce = AnnounceMessage::deserialize(data)?;
                self.handle_announce(announce).await?;
            }
            _ => {
                debug!("Unhandled message type: {:?}", header.message_type);
            }
        }

        Ok(())
    }

    /// Send a sync message (master only).
    pub async fn send_sync(&mut self) -> TimeSyncResult<()> {
        if self.port_state != PortState::Master {
            return Ok(());
        }

        let socket = self
            .socket
            .as_ref()
            .ok_or_else(|| TimeSyncError::InvalidConfig("Socket not bound".to_string()))?;

        let timestamp = PtpTimestamp::now();
        self.sequence_id = self.sequence_id.wrapping_add(1);

        let port_id = PortIdentity::new(self.default_ds.clock_identity, 1);

        let mut flags = Flags::default();
        flags.two_step = self.default_ds.two_step_flag;

        let header = Header {
            message_type: MessageType::Sync,
            version: 2,
            message_length: 44,
            domain: Domain(self.default_ds.domain_number),
            flags,
            correction_field: 0,
            source_port_identity: port_id,
            sequence_id: self.sequence_id,
            control: 0,
            log_message_interval: self.sync_interval,
        };

        let sync = SyncMessage {
            header,
            origin_timestamp: timestamp,
        };

        let data = sync.serialize()?;

        let dest = match self.comm_mode {
            CommunicationMode::Multicast => PTP_MULTICAST_EVENT
                .parse()
                .map_err(|e| TimeSyncError::InvalidConfig(format!("multicast parse: {e}")))?,
            CommunicationMode::Unicast(addr) => addr,
        };

        socket.send_to(&data, dest).await?;

        // Send follow-up if two-step
        if self.default_ds.two_step_flag {
            self.send_follow_up(timestamp, self.sequence_id).await?;
        }

        Ok(())
    }

    /// Send announce message.
    pub async fn send_announce(&mut self) -> TimeSyncResult<()> {
        if self.port_state != PortState::Master {
            return Ok(());
        }

        let socket = self
            .socket
            .as_ref()
            .ok_or_else(|| TimeSyncError::InvalidConfig("Socket not bound".to_string()))?;

        let timestamp = PtpTimestamp::now();
        self.sequence_id = self.sequence_id.wrapping_add(1);

        let port_id = PortIdentity::new(self.default_ds.clock_identity, 1);

        let header = Header {
            message_type: MessageType::Announce,
            version: 2,
            message_length: 64,
            domain: Domain(self.default_ds.domain_number),
            flags: Flags::default(),
            correction_field: 0,
            source_port_identity: port_id,
            sequence_id: self.sequence_id,
            control: 5,
            log_message_interval: self.announce_interval,
        };

        let announce = AnnounceMessage {
            header,
            origin_timestamp: timestamp,
            current_utc_offset: self.time_props_ds.current_utc_offset,
            grandmaster_priority1: self.default_ds.priority1,
            grandmaster_clock_quality: self.default_ds.clock_quality,
            grandmaster_priority2: self.default_ds.priority2,
            grandmaster_identity: self.default_ds.clock_identity,
            steps_removed: 0,
            time_source: self.time_props_ds.time_source as u8,
        };

        let data = announce.serialize()?;

        let dest = match self.comm_mode {
            CommunicationMode::Multicast => PTP_MULTICAST_EVENT
                .parse()
                .map_err(|e| TimeSyncError::InvalidConfig(format!("multicast parse: {e}")))?,
            CommunicationMode::Unicast(addr) => addr,
        };

        socket.send_to(&data, dest).await?;
        Ok(())
    }

    async fn send_follow_up(&self, timestamp: PtpTimestamp, seq_id: u16) -> TimeSyncResult<()> {
        let socket = self
            .socket
            .as_ref()
            .ok_or_else(|| TimeSyncError::InvalidConfig("Socket not bound".to_string()))?;

        let port_id = PortIdentity::new(self.default_ds.clock_identity, 1);

        let header = Header {
            message_type: MessageType::FollowUp,
            version: 2,
            message_length: 44,
            domain: Domain(self.default_ds.domain_number),
            flags: Flags::default(),
            correction_field: 0,
            source_port_identity: port_id,
            sequence_id: seq_id,
            control: 2,
            log_message_interval: self.sync_interval,
        };

        let follow_up = FollowUpMessage {
            header,
            precise_origin_timestamp: timestamp,
        };

        let data = follow_up.serialize()?;

        let dest = match self.comm_mode {
            CommunicationMode::Multicast => PTP_MULTICAST_EVENT
                .parse()
                .map_err(|e| TimeSyncError::InvalidConfig(format!("multicast parse: {e}")))?,
            CommunicationMode::Unicast(addr) => addr,
        };

        socket.send_to(&data, dest).await?;
        Ok(())
    }

    async fn handle_sync(&mut self, sync: SyncMessage, src: SocketAddr) -> TimeSyncResult<()> {
        if self.port_state != PortState::Slave {
            return Ok(());
        }

        // Record the reception timestamp (t2)
        let t2 = PtpTimestamp::now();
        debug!(
            "Received Sync seq={} from {:?}",
            sync.header.sequence_id, src
        );

        self.sync_receive_time = Some(t2);
        self.pending_sync_seq = Some(sync.header.sequence_id);
        self.sync_source_addr = Some(src);

        if sync.header.flags.two_step {
            // Two-step mode: wait for Follow_Up to get precise t1
            self.sync_origin = None;
        } else {
            // One-step mode: origin_timestamp in Sync IS the precise t1
            self.sync_origin = Some(sync.origin_timestamp);
            // We can proceed directly to sending Delay_Req
            self.send_delay_req(src).await?;
        }

        Ok(())
    }

    async fn handle_follow_up(&mut self, follow_up: FollowUpMessage) -> TimeSyncResult<()> {
        if self.port_state != PortState::Slave {
            return Ok(());
        }

        // Verify this Follow_Up matches the pending Sync
        let expected_seq = match self.pending_sync_seq {
            Some(seq) => seq,
            None => {
                debug!("Follow_Up received without pending Sync, ignoring");
                return Ok(());
            }
        };

        if follow_up.header.sequence_id != expected_seq {
            debug!(
                "Follow_Up seq {} does not match pending Sync seq {}",
                follow_up.header.sequence_id, expected_seq
            );
            return Ok(());
        }

        debug!(
            "Received Follow_Up seq={}, precise_origin={}s {}ns",
            follow_up.header.sequence_id,
            follow_up.precise_origin_timestamp.seconds,
            follow_up.precise_origin_timestamp.nanoseconds,
        );

        // Store the precise origin timestamp (t1) from the Follow_Up
        self.follow_up_origin = Some(follow_up.precise_origin_timestamp);

        // Now send a Delay_Req to measure the path delay
        if let Some(src) = self.sync_source_addr {
            self.send_delay_req(src).await?;
        }

        Ok(())
    }

    async fn handle_delay_req(
        &mut self,
        delay_req: DelayReqMessage,
        src: SocketAddr,
    ) -> TimeSyncResult<()> {
        if self.port_state != PortState::Master {
            return Ok(());
        }

        let socket = self
            .socket
            .as_ref()
            .ok_or_else(|| TimeSyncError::InvalidConfig("Socket not bound".to_string()))?;

        let receive_timestamp = PtpTimestamp::now();
        let port_id = PortIdentity::new(self.default_ds.clock_identity, 1);

        let header = Header {
            message_type: MessageType::DelayResp,
            version: 2,
            message_length: 54,
            domain: Domain(self.default_ds.domain_number),
            flags: Flags::default(),
            correction_field: 0,
            source_port_identity: port_id,
            sequence_id: delay_req.header.sequence_id,
            control: 3,
            log_message_interval: 0x7F,
        };

        let delay_resp = DelayRespMessage {
            header,
            receive_timestamp,
            requesting_port_identity: delay_req.header.source_port_identity,
        };

        let data = delay_resp.serialize()?;
        socket.send_to(&data, src).await?;

        Ok(())
    }

    /// Send a Delay_Req message to the master.
    async fn send_delay_req(&mut self, master_addr: SocketAddr) -> TimeSyncResult<()> {
        let socket = self
            .socket
            .as_ref()
            .ok_or_else(|| TimeSyncError::InvalidConfig("Socket not bound".to_string()))?;

        let t3 = PtpTimestamp::now();
        self.sequence_id = self.sequence_id.wrapping_add(1);
        let seq_id = self.sequence_id;

        let port_id = PortIdentity::new(self.default_ds.clock_identity, 1);

        let header = Header {
            message_type: MessageType::DelayReq,
            version: 2,
            message_length: 44,
            domain: Domain(self.default_ds.domain_number),
            flags: Flags::default(),
            correction_field: 0,
            source_port_identity: port_id,
            sequence_id: seq_id,
            control: 1,
            log_message_interval: 0x7F,
        };

        let delay_req = DelayReqMessage {
            header,
            origin_timestamp: t3,
        };

        let data = delay_req.serialize()?;
        socket.send_to(&data, master_addr).await?;

        self.delay_req_send_time = Some(t3);
        self.pending_delay_seq = Some(seq_id);

        debug!("Sent Delay_Req seq={} to {}", seq_id, master_addr);
        Ok(())
    }

    async fn handle_delay_resp(&mut self, delay_resp: DelayRespMessage) -> TimeSyncResult<()> {
        if self.port_state != PortState::Slave {
            return Ok(());
        }

        // Verify this response matches our pending request
        let expected_seq = match self.pending_delay_seq {
            Some(seq) => seq,
            None => {
                debug!("Delay_Resp received without pending Delay_Req");
                return Ok(());
            }
        };

        if delay_resp.header.sequence_id != expected_seq {
            debug!(
                "Delay_Resp seq {} does not match pending Delay_Req seq {}",
                delay_resp.header.sequence_id, expected_seq
            );
            return Ok(());
        }

        debug!(
            "Received Delay_Resp seq={}, receive_timestamp={}s {}ns",
            delay_resp.header.sequence_id,
            delay_resp.receive_timestamp.seconds,
            delay_resp.receive_timestamp.nanoseconds,
        );

        // Now we have all four timestamps for the two-step calculation:
        //   t1 = origin timestamp (from Sync/Follow_Up)
        //   t2 = reception time of Sync (recorded locally)
        //   t3 = transmission time of Delay_Req (recorded locally)
        //   t4 = reception time of Delay_Req at master (from Delay_Resp)

        let t1 = self.follow_up_origin.or(self.sync_origin).ok_or_else(|| {
            TimeSyncError::InvalidPacket("Missing origin timestamp (t1)".to_string())
        })?;

        let t2 = self.sync_receive_time.ok_or_else(|| {
            TimeSyncError::InvalidPacket("Missing sync receive time (t2)".to_string())
        })?;

        let t3 = self.delay_req_send_time.ok_or_else(|| {
            TimeSyncError::InvalidPacket("Missing delay req send time (t3)".to_string())
        })?;

        let t4 = delay_resp.receive_timestamp;

        // IEEE 1588 two-step formulas:
        //   mean_path_delay = ((t2 - t1) + (t4 - t3)) / 2
        //   offset_from_master = (t2 - t1) - mean_path_delay
        //                      = ((t2 - t1) - (t4 - t3)) / 2

        let forward_delay = t2.diff(&t1); // t2 - t1
        let reverse_delay = t4.diff(&t3); // t4 - t3

        let mean_path_delay = (forward_delay + reverse_delay) / 2;
        let offset_from_master = (forward_delay - reverse_delay) / 2;

        self.current_ds.offset_from_master = offset_from_master;
        self.current_ds.mean_path_delay = mean_path_delay;

        info!(
            "PTP offset={}ns, delay={}ns",
            offset_from_master, mean_path_delay
        );

        // Clear pending state
        self.pending_sync_seq = None;
        self.pending_delay_seq = None;
        self.sync_receive_time = None;
        self.follow_up_origin = None;
        self.sync_origin = None;
        self.delay_req_send_time = None;

        Ok(())
    }

    async fn handle_announce(&mut self, announce: AnnounceMessage) -> TimeSyncResult<()> {
        let src_port = announce.header.source_port_identity;
        self.received_announces.insert(src_port, announce.clone());

        // Run BMCA
        let recommendation = recommend_state(&self.default_ds, Some(&announce), self.port_state);

        if recommendation.state != self.port_state {
            info!(
                "State transition: {:?} -> {:?}",
                self.port_state, recommendation.state
            );
            self.port_state = recommendation.state;
            self.current_master = recommendation.best_master;

            if self.port_state == PortState::Slave {
                info!("Became slave to {:?}", self.current_master);
                // Update parent dataset
                self.parent_ds.parent_port_identity = src_port;
                self.parent_ds.grandmaster_identity = announce.grandmaster_identity;
                self.parent_ds.grandmaster_clock_quality = announce.grandmaster_clock_quality;
                self.parent_ds.grandmaster_priority1 = announce.grandmaster_priority1;
                self.parent_ds.grandmaster_priority2 = announce.grandmaster_priority2;
                self.current_ds.steps_removed = announce.steps_removed + 1;
            }
        }

        Ok(())
    }
}

/// PTP Boundary Clock (BC) implementation.
///
/// A boundary clock has multiple PTP ports and can forward timing information.
pub struct BoundaryClock {
    /// Default dataset
    #[allow(dead_code)]
    default_ds: DefaultDataSet,
    /// Port states (indexed by port number)
    port_states: HashMap<u16, PortState>,
    /// Number of ports
    #[allow(dead_code)]
    num_ports: u16,
}

impl BoundaryClock {
    /// Create a new boundary clock.
    #[must_use]
    pub fn new(clock_identity: ClockIdentity, num_ports: u16) -> Self {
        let mut default_ds = DefaultDataSet::new(clock_identity);
        default_ds.number_ports = num_ports;

        let mut port_states = HashMap::new();
        for port_num in 1..=num_ports {
            port_states.insert(port_num, PortState::Initializing);
        }

        Self {
            default_ds,
            port_states,
            num_ports,
        }
    }

    /// Get port state.
    #[must_use]
    pub fn get_port_state(&self, port: u16) -> Option<PortState> {
        self.port_states.get(&port).copied()
    }

    /// Set port state.
    pub fn set_port_state(&mut self, port: u16, state: PortState) -> TimeSyncResult<()> {
        if port == 0 || port > self.num_ports {
            return Err(TimeSyncError::InvalidConfig(
                "Invalid port number".to_string(),
            ));
        }
        self.port_states.insert(port, state);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ordinary_clock_creation() {
        let clock_id = ClockIdentity::random();
        let clock = OrdinaryClock::new(clock_id, Domain::DEFAULT);

        assert_eq!(clock.port_state(), PortState::Initializing);
        assert_eq!(clock.offset_from_master(), 0);
    }

    #[test]
    fn test_boundary_clock_creation() {
        let clock_id = ClockIdentity::random();
        let clock = BoundaryClock::new(clock_id, 4);

        assert_eq!(clock.get_port_state(1), Some(PortState::Initializing));
        assert_eq!(clock.get_port_state(4), Some(PortState::Initializing));
        assert_eq!(clock.get_port_state(5), None);
    }
}
