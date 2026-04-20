#![allow(dead_code)]
//! Advanced IP multicast group manager with IGMP/MLD simulation.
//!
//! This module extends the basic [`crate::multicast`] support with:
//!
//! - **TTL-scoped routing** — groups are tagged with a [`MulticastScope`]
//!   derived from their TTL; packets are blocked at scope boundaries.
//! - **IGMP/MLD message simulation** — produces the correct wire message type
//!   for join/leave/report operations without touching the network.
//! - **Interface-aware membership** — the same group can be joined on multiple
//!   network interfaces independently.
//! - **Membership statistics** — per-group packet/byte counters with
//!   last-activity timestamps.
//! - **Subscription filtering** — query membership by scope, interface, or
//!   IGMP version.
//!
//! All operations are synchronous and allocation-light.

use std::collections::HashMap;
use std::fmt;

use crate::error::{NetError, NetResult};

// ─── Scope ────────────────────────────────────────────────────────────────────

/// Administrative scope derived from TTL/hop-limit (RFC 2365).
///
/// The scope determines how far a multicast packet is forwarded by routers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MulticastScope {
    /// TTL 0 — restricted to the originating host.
    NodeLocal,
    /// TTL 1 — restricted to the local link (subnet).
    LinkLocal,
    /// TTL ≤ 15 — site-local (single organisation site).
    SiteLocal,
    /// TTL ≤ 63 — regional scope.
    Regional,
    /// TTL > 63 — global internet scope.
    Global,
}

impl MulticastScope {
    /// Derives the scope from a TTL value.
    #[must_use]
    pub fn from_ttl(ttl: u8) -> Self {
        match ttl {
            0 => Self::NodeLocal,
            1 => Self::LinkLocal,
            2..=15 => Self::SiteLocal,
            16..=63 => Self::Regional,
            _ => Self::Global,
        }
    }

    /// Returns the maximum TTL that still maps to this scope.
    #[must_use]
    pub const fn max_ttl(self) -> u8 {
        match self {
            Self::NodeLocal => 0,
            Self::LinkLocal => 1,
            Self::SiteLocal => 15,
            Self::Regional => 63,
            Self::Global => u8::MAX,
        }
    }
}

impl fmt::Display for MulticastScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::NodeLocal => "NodeLocal",
            Self::LinkLocal => "LinkLocal",
            Self::SiteLocal => "SiteLocal",
            Self::Regional => "Regional",
            Self::Global => "Global",
        };
        f.write_str(s)
    }
}

// ─── Protocol version ─────────────────────────────────────────────────────────

/// Membership protocol version.
///
/// `Igmpv1`/`Igmpv2`/`Igmpv3` are used for IPv4; `Mldv1`/`Mldv2` for IPv6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MembershipProtocol {
    /// IGMPv1 — basic join only (RFC 1112).
    Igmpv1,
    /// IGMPv2 — adds leave message (RFC 2236).
    Igmpv2,
    /// IGMPv3 — source-specific multicast (RFC 3376).
    Igmpv3,
    /// MLDv1 — IPv6 equivalent of IGMPv2 (RFC 2710).
    Mldv1,
    /// MLDv2 — source-specific multicast for IPv6 (RFC 3810).
    Mldv2,
}

impl MembershipProtocol {
    /// Returns `true` for IPv4 protocols (IGMP variants).
    #[must_use]
    pub const fn is_ipv4(self) -> bool {
        matches!(self, Self::Igmpv1 | Self::Igmpv2 | Self::Igmpv3)
    }

    /// Returns `true` for IPv6 protocols (MLD variants).
    #[must_use]
    pub const fn is_ipv6(self) -> bool {
        matches!(self, Self::Mldv1 | Self::Mldv2)
    }

    /// Returns `true` if the protocol supports source-specific multicast.
    #[must_use]
    pub const fn supports_ssm(self) -> bool {
        matches!(self, Self::Igmpv3 | Self::Mldv2)
    }
}

impl fmt::Display for MembershipProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Igmpv1 => "IGMPv1",
            Self::Igmpv2 => "IGMPv2",
            Self::Igmpv3 => "IGMPv3",
            Self::Mldv1 => "MLDv1",
            Self::Mldv2 => "MLDv2",
        };
        f.write_str(s)
    }
}

// ─── Simulated wire messages ───────────────────────────────────────────────────

/// Simulated IGMP/MLD message type produced by the manager.
///
/// These are not real network packets; they represent the *logical* message
/// that would be sent on a real network interface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MembershipMessage {
    /// IGMPv1/v2 or MLDv1 — join a group.
    Join {
        /// Group address.
        group: String,
        /// Network interface.
        interface: String,
        /// Protocol used.
        protocol: MembershipProtocol,
    },
    /// IGMPv2 Leave Group or MLDv1 Done.
    Leave {
        /// Group address.
        group: String,
        /// Network interface.
        interface: String,
        /// Protocol used.
        protocol: MembershipProtocol,
    },
    /// IGMPv3 / MLDv2 INCLUDE record — accept only from specific sources.
    SourceInclude {
        /// Group address.
        group: String,
        /// Accepted source addresses.
        sources: Vec<String>,
        /// Network interface.
        interface: String,
    },
    /// IGMPv3 / MLDv2 EXCLUDE record — accept from all except listed sources.
    SourceExclude {
        /// Group address.
        group: String,
        /// Excluded source addresses.
        sources: Vec<String>,
        /// Network interface.
        interface: String,
    },
}

// ─── Group descriptor ─────────────────────────────────────────────────────────

/// Describes a multicast group membership entry.
#[derive(Debug, Clone)]
pub struct GroupEntry {
    /// Multicast group address (IPv4 or IPv6 string).
    pub group_addr: String,
    /// UDP port.
    pub port: u16,
    /// Network interface name.
    pub interface: String,
    /// IP time-to-live / hop limit.
    pub ttl: u8,
    /// Derived administrative scope.
    pub scope: MulticastScope,
    /// Membership protocol in use.
    pub protocol: MembershipProtocol,
    /// Optional SSM source filter (only relevant for IGMPv3/MLDv2).
    pub source_filter: Option<SourceFilter>,
}

/// Source-specific multicast filter mode and source list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFilter {
    /// INCLUDE: accept only listed sources.  EXCLUDE: block listed sources.
    pub mode: FilterMode,
    /// Source IP address list.
    pub sources: Vec<String>,
}

/// Filter mode for IGMPv3/MLDv2 source-specific multicast.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    /// Accept traffic only from the listed sources.
    Include,
    /// Accept traffic from all sources except the listed ones.
    Exclude,
}

impl GroupEntry {
    /// Creates a new [`GroupEntry`] without a source filter.
    #[must_use]
    pub fn new(
        group_addr: impl Into<String>,
        port: u16,
        interface: impl Into<String>,
        ttl: u8,
        protocol: MembershipProtocol,
    ) -> Self {
        let ttl_val = ttl;
        Self {
            group_addr: group_addr.into(),
            port,
            interface: interface.into(),
            ttl: ttl_val,
            scope: MulticastScope::from_ttl(ttl_val),
            protocol,
            source_filter: None,
        }
    }

    /// Attaches a source filter to this entry.
    #[must_use]
    pub fn with_source_filter(mut self, filter: SourceFilter) -> Self {
        self.source_filter = Some(filter);
        self
    }
}

// ─── Membership statistics ────────────────────────────────────────────────────

/// Per-membership packet/byte counters.
#[derive(Debug, Clone, Default)]
pub struct MembershipStats {
    /// Total packets received for this group.
    pub packets_rx: u64,
    /// Total bytes received for this group.
    pub bytes_rx: u64,
    /// Number of join operations performed.
    pub join_count: u32,
    /// Timestamp (ms) of the last received packet, or 0 if none.
    pub last_rx_ms: u64,
}

// ─── Manager key ──────────────────────────────────────────────────────────────

/// Stable key that uniquely identifies a membership entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MembershipKey {
    group_addr: String,
    port: u16,
    interface: String,
}

impl MembershipKey {
    fn new(group_addr: &str, port: u16, interface: &str) -> Self {
        Self {
            group_addr: group_addr.to_owned(),
            port,
            interface: interface.to_owned(),
        }
    }
}

// ─── Manager ─────────────────────────────────────────────────────────────────

/// Advanced multicast group manager.
///
/// Maintains a table of active memberships keyed by
/// `(group_addr, port, interface)` and produces simulated IGMP/MLD messages
/// for every state change.
#[derive(Debug, Default)]
pub struct MulticastGroupManager {
    memberships: HashMap<MembershipKey, (GroupEntry, MembershipStats)>,
    /// Log of all messages produced since creation or last [`Self::drain_messages`].
    message_log: Vec<MembershipMessage>,
}

impl MulticastGroupManager {
    /// Creates a new, empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // ── Validation ────────────────────────────────────────────────────────────

    /// Returns `Ok(())` if `addr` is a valid IPv4 multicast address
    /// (`224.0.0.0/4`).
    pub fn validate_ipv4_multicast(addr: &str) -> NetResult<()> {
        let parts: Vec<&str> = addr.split('.').collect();
        if parts.len() != 4 {
            return Err(NetError::protocol(format!("invalid IPv4 address: {addr}")));
        }
        let first: u8 = parts[0]
            .parse()
            .map_err(|_| NetError::protocol(format!("non-numeric octet in {addr}")))?;
        if first < 224 || first > 239 {
            return Err(NetError::protocol(format!(
                "{addr} is not in the IPv4 multicast range 224.0.0.0/4"
            )));
        }
        Ok(())
    }

    /// Returns `Ok(())` if `addr` is a valid IPv6 multicast address (`ff00::/8`).
    pub fn validate_ipv6_multicast(addr: &str) -> NetResult<()> {
        let lower = addr.to_ascii_lowercase();
        if lower.starts_with("ff") {
            Ok(())
        } else {
            Err(NetError::protocol(format!(
                "{addr} is not an IPv6 multicast address (must start with ff)"
            )))
        }
    }

    // ── Join / Leave ──────────────────────────────────────────────────────────

    /// Joins a multicast group.
    ///
    /// # Errors
    /// Returns `Err` if `entry.group_addr` is not a valid multicast address for
    /// the chosen protocol family, or if IGMPv3/MLDv2 is requested with a
    /// source filter but the protocol does not support SSM.
    pub fn join(&mut self, entry: GroupEntry) -> NetResult<MembershipMessage> {
        // Address validation
        if entry.protocol.is_ipv4() {
            Self::validate_ipv4_multicast(&entry.group_addr)?;
        } else {
            Self::validate_ipv6_multicast(&entry.group_addr)?;
        }

        // SSM sanity check
        if entry.source_filter.is_some() && !entry.protocol.supports_ssm() {
            return Err(NetError::protocol(format!(
                "protocol {} does not support source-specific multicast",
                entry.protocol
            )));
        }

        let key = MembershipKey::new(&entry.group_addr, entry.port, &entry.interface);

        // Build message before moving entry into the map
        let message = Self::build_join_message(&entry);

        self.memberships
            .entry(key)
            .and_modify(|(_e, stats)| stats.join_count += 1)
            .or_insert_with(|| {
                let mut stats = MembershipStats::default();
                stats.join_count = 1;
                (entry, stats)
            });

        self.message_log.push(message.clone());
        Ok(message)
    }

    /// Leaves a multicast group on a specific interface.
    ///
    /// Returns the generated [`MembershipMessage`], or `None` if the group was
    /// not joined on that interface.
    pub fn leave(
        &mut self,
        group_addr: &str,
        port: u16,
        interface: &str,
    ) -> Option<MembershipMessage> {
        let key = MembershipKey::new(group_addr, port, interface);
        let removed = self.memberships.remove(&key)?;
        let (entry, _stats) = removed;

        let msg = MembershipMessage::Leave {
            group: entry.group_addr.clone(),
            interface: entry.interface.clone(),
            protocol: entry.protocol,
        };
        self.message_log.push(msg.clone());
        Some(msg)
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Records arrival of a packet for the given group/port/interface.
    ///
    /// No-ops silently if the membership does not exist.
    pub fn record_packet(
        &mut self,
        group_addr: &str,
        port: u16,
        interface: &str,
        bytes: u64,
        now_ms: u64,
    ) {
        let key = MembershipKey::new(group_addr, port, interface);
        if let Some((_entry, stats)) = self.memberships.get_mut(&key) {
            stats.packets_rx += 1;
            stats.bytes_rx += bytes;
            stats.last_rx_ms = now_ms;
        }
    }

    /// Returns the statistics for a specific membership, if present.
    #[must_use]
    pub fn stats(&self, group_addr: &str, port: u16, interface: &str) -> Option<&MembershipStats> {
        let key = MembershipKey::new(group_addr, port, interface);
        self.memberships.get(&key).map(|(_, s)| s)
    }

    // ── Query ─────────────────────────────────────────────────────────────────

    /// Returns references to all active [`GroupEntry`] items.
    #[must_use]
    pub fn active_entries(&self) -> Vec<&GroupEntry> {
        self.memberships.values().map(|(e, _)| e).collect()
    }

    /// Returns all active entries on the given `interface`.
    #[must_use]
    pub fn entries_on_interface(&self, interface: &str) -> Vec<&GroupEntry> {
        self.memberships
            .values()
            .filter_map(|(e, _)| {
                if e.interface == interface {
                    Some(e)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns all active entries whose [`MulticastScope`] matches `scope`.
    #[must_use]
    pub fn entries_by_scope(&self, scope: MulticastScope) -> Vec<&GroupEntry> {
        self.memberships
            .values()
            .filter_map(|(e, _)| if e.scope == scope { Some(e) } else { None })
            .collect()
    }

    /// Total number of active memberships.
    #[must_use]
    pub fn membership_count(&self) -> usize {
        self.memberships.len()
    }

    /// Returns `true` if the group is currently joined on the given interface.
    #[must_use]
    pub fn is_joined(&self, group_addr: &str, port: u16, interface: &str) -> bool {
        let key = MembershipKey::new(group_addr, port, interface);
        self.memberships.contains_key(&key)
    }

    // ── Message log ───────────────────────────────────────────────────────────

    /// Drains all pending messages from the log and returns them.
    pub fn drain_messages(&mut self) -> Vec<MembershipMessage> {
        std::mem::take(&mut self.message_log)
    }

    /// Returns a reference to the message log without draining it.
    #[must_use]
    pub fn message_log(&self) -> &[MembershipMessage] {
        &self.message_log
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn build_join_message(entry: &GroupEntry) -> MembershipMessage {
        match &entry.source_filter {
            Some(filter) if entry.protocol.supports_ssm() => {
                let sources = filter.sources.clone();
                match filter.mode {
                    FilterMode::Include => MembershipMessage::SourceInclude {
                        group: entry.group_addr.clone(),
                        sources,
                        interface: entry.interface.clone(),
                    },
                    FilterMode::Exclude => MembershipMessage::SourceExclude {
                        group: entry.group_addr.clone(),
                        sources,
                        interface: entry.interface.clone(),
                    },
                }
            }
            _ => MembershipMessage::Join {
                group: entry.group_addr.clone(),
                interface: entry.interface.clone(),
                protocol: entry.protocol,
            },
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn basic_entry(addr: &str, protocol: MembershipProtocol) -> GroupEntry {
        GroupEntry::new(addr, 5004, "eth0", 15, protocol)
    }

    // 1. MulticastScope::from_ttl
    #[test]
    fn test_scope_from_ttl() {
        assert_eq!(MulticastScope::from_ttl(0), MulticastScope::NodeLocal);
        assert_eq!(MulticastScope::from_ttl(1), MulticastScope::LinkLocal);
        assert_eq!(MulticastScope::from_ttl(15), MulticastScope::SiteLocal);
        assert_eq!(MulticastScope::from_ttl(63), MulticastScope::Regional);
        assert_eq!(MulticastScope::from_ttl(128), MulticastScope::Global);
    }

    // 2. MulticastScope::max_ttl
    #[test]
    fn test_scope_max_ttl() {
        assert_eq!(MulticastScope::NodeLocal.max_ttl(), 0);
        assert_eq!(MulticastScope::LinkLocal.max_ttl(), 1);
        assert_eq!(MulticastScope::SiteLocal.max_ttl(), 15);
        assert_eq!(MulticastScope::Regional.max_ttl(), 63);
        assert_eq!(MulticastScope::Global.max_ttl(), 255);
    }

    // 3. validate_ipv4_multicast — valid addresses
    #[test]
    fn test_validate_ipv4_valid() {
        assert!(MulticastGroupManager::validate_ipv4_multicast("224.0.0.1").is_ok());
        assert!(MulticastGroupManager::validate_ipv4_multicast("239.255.255.255").is_ok());
    }

    // 4. validate_ipv4_multicast — invalid addresses
    #[test]
    fn test_validate_ipv4_invalid() {
        assert!(MulticastGroupManager::validate_ipv4_multicast("192.168.1.1").is_err());
        assert!(MulticastGroupManager::validate_ipv4_multicast("not.valid").is_err());
        assert!(MulticastGroupManager::validate_ipv4_multicast("10.0.0.1").is_err());
    }

    // 5. validate_ipv6_multicast
    #[test]
    fn test_validate_ipv6() {
        assert!(MulticastGroupManager::validate_ipv6_multicast("ff02::1").is_ok());
        assert!(MulticastGroupManager::validate_ipv6_multicast("fe80::1").is_err());
    }

    // 6. join produces a Join message and records membership
    #[test]
    fn test_join_basic() {
        let mut mgr = MulticastGroupManager::new();
        let entry = basic_entry("239.1.0.1", MembershipProtocol::Igmpv2);
        let msg = mgr.join(entry).expect("join should succeed");
        assert!(matches!(msg, MembershipMessage::Join { .. }));
        assert_eq!(mgr.membership_count(), 1);
        assert!(mgr.is_joined("239.1.0.1", 5004, "eth0"));
    }

    // 7. join invalid address returns Err
    #[test]
    fn test_join_invalid_addr() {
        let mut mgr = MulticastGroupManager::new();
        let entry = basic_entry("10.0.0.1", MembershipProtocol::Igmpv2);
        assert!(mgr.join(entry).is_err());
    }

    // 8. leave removes membership and produces Leave message
    #[test]
    fn test_leave() {
        let mut mgr = MulticastGroupManager::new();
        let entry = basic_entry("239.1.0.1", MembershipProtocol::Igmpv2);
        mgr.join(entry).expect("join should succeed");

        let msg = mgr.leave("239.1.0.1", 5004, "eth0");
        assert!(msg.is_some());
        assert!(matches!(
            msg.expect("msg is Some, checked above"),
            MembershipMessage::Leave { .. }
        ));
        assert_eq!(mgr.membership_count(), 0);
    }

    // 9. leave non-existent group returns None
    #[test]
    fn test_leave_not_joined() {
        let mut mgr = MulticastGroupManager::new();
        assert!(mgr.leave("239.1.0.1", 5004, "eth0").is_none());
    }

    // 10. SSM join produces SourceInclude message
    #[test]
    fn test_ssm_join_include() {
        let mut mgr = MulticastGroupManager::new();
        let filter = SourceFilter {
            mode: FilterMode::Include,
            sources: vec!["10.0.0.1".to_owned()],
        };
        let entry = GroupEntry::new("232.1.0.1", 5004, "eth0", 1, MembershipProtocol::Igmpv3)
            .with_source_filter(filter);
        let msg = mgr.join(entry).expect("SSM join should succeed");
        assert!(matches!(msg, MembershipMessage::SourceInclude { .. }));
    }

    // 11. SSM on non-SSM protocol returns Err
    #[test]
    fn test_ssm_on_igmpv2_fails() {
        let mut mgr = MulticastGroupManager::new();
        let filter = SourceFilter {
            mode: FilterMode::Include,
            sources: vec!["10.0.0.1".to_owned()],
        };
        let entry = GroupEntry::new("239.1.0.1", 5004, "eth0", 1, MembershipProtocol::Igmpv2)
            .with_source_filter(filter);
        assert!(mgr.join(entry).is_err());
    }

    // 12. record_packet updates stats
    #[test]
    fn test_record_packet() {
        let mut mgr = MulticastGroupManager::new();
        let entry = basic_entry("239.1.0.1", MembershipProtocol::Igmpv3);
        mgr.join(entry).expect("join should succeed");

        mgr.record_packet("239.1.0.1", 5004, "eth0", 1_316, 1_000);
        mgr.record_packet("239.1.0.1", 5004, "eth0", 1_316, 2_000);

        let stats = mgr
            .stats("239.1.0.1", 5004, "eth0")
            .expect("stats should be present");
        assert_eq!(stats.packets_rx, 2);
        assert_eq!(stats.bytes_rx, 2_632);
        assert_eq!(stats.last_rx_ms, 2_000);
    }

    // 13. entries_by_scope filters correctly
    #[test]
    fn test_entries_by_scope() {
        let mut mgr = MulticastGroupManager::new();
        // TTL 1 → LinkLocal
        let e1 = GroupEntry::new("239.1.0.1", 5004, "eth0", 1, MembershipProtocol::Igmpv2);
        // TTL 15 → SiteLocal
        let e2 = GroupEntry::new("239.2.0.1", 5005, "eth0", 15, MembershipProtocol::Igmpv2);
        mgr.join(e1).expect("join should succeed");
        mgr.join(e2).expect("join should succeed");

        let link_local = mgr.entries_by_scope(MulticastScope::LinkLocal);
        assert_eq!(link_local.len(), 1);
        assert_eq!(link_local[0].group_addr, "239.1.0.1");

        let site_local = mgr.entries_by_scope(MulticastScope::SiteLocal);
        assert_eq!(site_local.len(), 1);
    }

    // 14. drain_messages clears log
    #[test]
    fn test_drain_messages() {
        let mut mgr = MulticastGroupManager::new();
        let entry = basic_entry("239.1.0.1", MembershipProtocol::Igmpv2);
        mgr.join(entry).expect("join should succeed");

        let msgs = mgr.drain_messages();
        assert_eq!(msgs.len(), 1);
        assert!(mgr.message_log().is_empty());
    }

    // 15. MembershipProtocol helpers
    #[test]
    fn test_protocol_helpers() {
        assert!(MembershipProtocol::Igmpv3.is_ipv4());
        assert!(!MembershipProtocol::Igmpv3.is_ipv6());
        assert!(MembershipProtocol::Mldv2.is_ipv6());
        assert!(MembershipProtocol::Igmpv3.supports_ssm());
        assert!(MembershipProtocol::Mldv2.supports_ssm());
        assert!(!MembershipProtocol::Igmpv2.supports_ssm());
    }
}
