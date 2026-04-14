//! SRT Group connection manager.
//!
//! SRT supports *grouped connections* (also known as *bonded connections*)
//! that combine multiple underlying SRT links to achieve:
//!
//! - **Broadcast mode** — send every packet on all paths simultaneously for
//!   maximum redundancy (like SMPTE ST 2022-7 at the SRT layer).
//! - **Main/Backup mode** — one active path with one or more hot-standby paths;
//!   traffic automatically switches to a backup when the main path degrades.
//! - **Balancing mode** — distribute packets across multiple paths according
//!   to their current throughput capacity (experimental bonding).
//!
//! This module provides a **Group connection manager** that orchestrates
//! multiple SRT member links and presents a single logical send/receive
//! interface.  The manager:
//!
//! 1. Maintains a set of [`GroupMember`] entries, each with its own state.
//! 2. On send, applies the configured [`GroupMode`] to decide which members
//!    receive the packet.
//! 3. On receive, de-duplicates packets from all members using a rolling
//!    sequence-number window.
//! 4. Tracks per-member health and promotes/demotes members between active
//!    and standby roles.
//! 5. Provides aggregate statistics via [`GroupStats`].
//!
//! ## Relation to `srt::connection_mode`
//!
//! The low-level [`crate::srt::connection_mode`] module implements the per-link
//! handshake state machines (Caller / Listener / Rendezvous).  This module
//! sits above that layer and coordinates multiple such links as a logical group.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use crate::error::{NetError, NetResult};

// ─── Group Mode ───────────────────────────────────────────────────────────────

/// How the group distributes data across member links.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupMode {
    /// Send every packet on **all** active members (maximum redundancy).
    ///
    /// The receiver de-duplicates using sequence numbers.  This is equivalent
    /// to SMPTE ST 2022-7 seamless switching at the SRT layer.
    Broadcast,
    /// One member carries live traffic; others are hot-standby.
    ///
    /// Traffic is switched to a standby member when the main member's loss
    /// rate or RTT exceeds the configured thresholds.
    MainBackup,
    /// Distribute packets across members in proportion to their bandwidth.
    ///
    /// Each member sends a slice of the stream; receivers must reassemble.
    /// This increases aggregate throughput at the cost of receiver complexity.
    Balancing,
}

impl GroupMode {
    /// Returns a human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Broadcast => "broadcast",
            Self::MainBackup => "main-backup",
            Self::Balancing => "balancing",
        }
    }
}

impl std::fmt::Display for GroupMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ─── Member Role ──────────────────────────────────────────────────────────────

/// The role of a [`GroupMember`] within the group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemberRole {
    /// Actively transmitting / receiving.
    Active,
    /// Idle — receiving health probes but not carrying media traffic.
    Standby,
    /// Disconnected or failed — excluded from routing decisions.
    Inactive,
}

impl MemberRole {
    /// Returns `true` if this role can carry media traffic.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        matches!(self, Self::Active)
    }
}

// ─── Member Health ─────────────────────────────────────────────────────────────

/// Per-link health metrics.
#[derive(Debug, Clone, Default)]
pub struct MemberHealth {
    /// Estimated round-trip time in milliseconds.
    pub rtt_ms: f64,
    /// Packet loss fraction in the recent window (0.0–1.0).
    pub loss_fraction: f64,
    /// Estimated available bandwidth in bits per second.
    pub bandwidth_bps: u64,
    /// Packets sent on this link.
    pub packets_sent: u64,
    /// Packets received on this link.
    pub packets_received: u64,
    /// Packets retransmitted (ARQ).
    pub packets_retransmitted: u64,
    /// Timestamp of the most recent health sample.
    pub last_sample: Option<Instant>,
}

impl MemberHealth {
    /// Returns a composite health score (0.0 = worst, 1.0 = perfect).
    ///
    /// Score is based on loss and RTT; bandwidth is informational only.
    #[must_use]
    pub fn score(&self) -> f64 {
        let loss_score = (1.0 - self.loss_fraction).max(0.0);
        // Normalise RTT: 0 ms → 1.0, 200 ms → 0.0 (linear clamp).
        let rtt_score = (1.0 - self.rtt_ms / 200.0).clamp(0.0, 1.0);
        // Weight loss more heavily than RTT.
        loss_score * 0.7 + rtt_score * 0.3
    }

    /// Returns `true` if the member is healthy enough to carry traffic.
    #[must_use]
    pub fn is_healthy(&self, max_loss: f64, max_rtt_ms: f64) -> bool {
        self.loss_fraction <= max_loss && self.rtt_ms <= max_rtt_ms
    }

    /// Records a new health sample.
    pub fn update(&mut self, rtt_ms: f64, loss_fraction: f64, bandwidth_bps: u64) {
        // Exponential moving average for smoothing.
        const ALPHA: f64 = 0.2;
        if self.last_sample.is_none() {
            self.rtt_ms = rtt_ms;
            self.loss_fraction = loss_fraction;
            self.bandwidth_bps = bandwidth_bps;
        } else {
            self.rtt_ms = ALPHA * rtt_ms + (1.0 - ALPHA) * self.rtt_ms;
            self.loss_fraction = ALPHA * loss_fraction + (1.0 - ALPHA) * self.loss_fraction;
            self.bandwidth_bps = ((ALPHA * bandwidth_bps as f64)
                + ((1.0 - ALPHA) * self.bandwidth_bps as f64))
                as u64;
        }
        self.last_sample = Some(Instant::now());
    }
}

// ─── Group Member ─────────────────────────────────────────────────────────────

/// A single SRT link participating in the group.
#[derive(Debug)]
pub struct GroupMember {
    /// Unique member identifier within the group.
    pub id: u32,
    /// Remote peer address.
    pub peer_addr: SocketAddr,
    /// Current role of this member.
    pub role: MemberRole,
    /// Priority — lower number = higher priority for MainBackup mode.
    pub priority: u32,
    /// Per-link health metrics.
    pub health: MemberHealth,
    /// When this member was added to the group.
    pub added_at: Instant,
    /// When the role was last changed.
    pub role_changed_at: Instant,
    /// Sequence-number of the last packet sent on this link.
    pub last_send_seq: u32,
    /// Sequence-number of the last packet received on this link.
    pub last_recv_seq: u32,
}

impl GroupMember {
    /// Creates a new member with the given peer address and role.
    #[must_use]
    pub fn new(id: u32, peer_addr: SocketAddr, role: MemberRole, priority: u32) -> Self {
        let now = Instant::now();
        Self {
            id,
            peer_addr,
            role,
            priority,
            health: MemberHealth::default(),
            added_at: now,
            role_changed_at: now,
            last_send_seq: 0,
            last_recv_seq: 0,
        }
    }

    /// Updates the role and records the change timestamp.
    pub fn set_role(&mut self, role: MemberRole) {
        self.role = role;
        self.role_changed_at = Instant::now();
    }
}

// ─── Group Configuration ──────────────────────────────────────────────────────

/// Configuration for [`SrtGroupManager`].
#[derive(Debug, Clone)]
pub struct GroupConfig {
    /// How the group distributes data.
    pub mode: GroupMode,
    /// Maximum allowed packet loss fraction before a member is demoted.
    pub max_loss_fraction: f64,
    /// Maximum allowed RTT (ms) before a member is demoted.
    pub max_rtt_ms: f64,
    /// How long a standby member's health must be stable before promotion.
    pub promotion_hold: Duration,
    /// Sequence-number deduplication window size (in packets).
    pub dedup_window: usize,
    /// Maximum number of members in the group.
    pub max_members: usize,
}

impl Default for GroupConfig {
    fn default() -> Self {
        Self {
            mode: GroupMode::Broadcast,
            max_loss_fraction: 0.05,
            max_rtt_ms: 150.0,
            promotion_hold: Duration::from_secs(5),
            dedup_window: 1024,
            max_members: 8,
        }
    }
}

impl GroupConfig {
    /// Creates a MainBackup configuration with tight health thresholds.
    #[must_use]
    pub fn main_backup() -> Self {
        Self {
            mode: GroupMode::MainBackup,
            max_loss_fraction: 0.02,
            max_rtt_ms: 80.0,
            promotion_hold: Duration::from_secs(10),
            dedup_window: 512,
            max_members: 4,
        }
    }

    /// Creates a Balancing configuration.
    #[must_use]
    pub fn balancing() -> Self {
        Self {
            mode: GroupMode::Balancing,
            max_loss_fraction: 0.1,
            max_rtt_ms: 200.0,
            promotion_hold: Duration::from_secs(3),
            dedup_window: 2048,
            max_members: 4,
        }
    }
}

// ─── Group Statistics ─────────────────────────────────────────────────────────

/// Aggregate statistics for the group.
#[derive(Debug, Clone, Default)]
pub struct GroupStats {
    /// Total packets delivered to the application layer.
    pub packets_delivered: u64,
    /// Total duplicate packets discarded.
    pub duplicates_discarded: u64,
    /// Number of role changes (active ↔ standby) performed.
    pub role_switches: u64,
    /// Number of active members currently.
    pub active_member_count: usize,
    /// Number of standby members currently.
    pub standby_member_count: usize,
    /// Aggregate bytes sent across all active members.
    pub bytes_sent: u64,
}

// ─── Dedup Window ─────────────────────────────────────────────────────────────

/// A rolling bitset-based sequence-number deduplication window.
///
/// Sequence numbers are 32-bit and wrap around.  The window tracks the
/// `window_size` most recent sequence numbers.
struct DedupWindow {
    seen: VecDeque<u32>,
    window_size: usize,
}

impl DedupWindow {
    fn new(window_size: usize) -> Self {
        Self {
            seen: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Returns `true` if `seq` has already been seen (duplicate).
    ///
    /// If not seen, `seq` is recorded and `false` is returned.
    fn check_and_insert(&mut self, seq: u32) -> bool {
        if self.seen.contains(&seq) {
            return true; // duplicate
        }
        if self.seen.len() >= self.window_size {
            self.seen.pop_front();
        }
        self.seen.push_back(seq);
        false
    }
}

// ─── Group Manager ────────────────────────────────────────────────────────────

/// Manages a group of SRT member links presenting a unified send/receive API.
///
/// # Example
///
/// ```
/// use std::net::SocketAddr;
/// use oximedia_net::srt_group::{GroupConfig, GroupMode, MemberRole, SrtGroupManager};
///
/// let config = GroupConfig {
///     mode: GroupMode::Broadcast,
///     ..GroupConfig::default()
/// };
/// let mut mgr = SrtGroupManager::new(config);
///
/// let addr: SocketAddr = "192.168.1.10:9000".parse().unwrap();
/// mgr.add_member(addr, MemberRole::Active, 0).unwrap();
///
/// // Determine which members should receive a packet with seq 42.
/// let targets = mgr.send_targets(42);
/// assert_eq!(targets.len(), 1);
/// ```
pub struct SrtGroupManager {
    config: GroupConfig,
    members: HashMap<u32, GroupMember>,
    next_member_id: u32,
    dedup: DedupWindow,
    stats: GroupStats,
}

impl SrtGroupManager {
    /// Creates a new group manager with the given configuration.
    #[must_use]
    pub fn new(config: GroupConfig) -> Self {
        let window = config.dedup_window;
        Self {
            config,
            members: HashMap::new(),
            next_member_id: 1,
            dedup: DedupWindow::new(window),
            stats: GroupStats::default(),
        }
    }

    /// Returns the current group mode.
    #[must_use]
    pub fn mode(&self) -> GroupMode {
        self.config.mode
    }

    /// Adds a new member to the group.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::InvalidState`] if the group already has
    /// `max_members` members.
    pub fn add_member(
        &mut self,
        peer_addr: SocketAddr,
        role: MemberRole,
        priority: u32,
    ) -> NetResult<u32> {
        if self.members.len() >= self.config.max_members {
            return Err(NetError::invalid_state(format!(
                "group is full: max {} members",
                self.config.max_members
            )));
        }
        let id = self.next_member_id;
        self.next_member_id = self.next_member_id.wrapping_add(1);
        self.members
            .insert(id, GroupMember::new(id, peer_addr, role, priority));
        self.refresh_stats();
        Ok(id)
    }

    /// Removes a member from the group.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::NotFound`] if no member with `id` exists.
    pub fn remove_member(&mut self, id: u32) -> NetResult<()> {
        self.members
            .remove(&id)
            .ok_or_else(|| NetError::not_found(format!("group member {id} not found")))?;
        self.refresh_stats();
        Ok(())
    }

    /// Returns the set of member IDs that should receive packet `seq`.
    ///
    /// The selection depends on the configured [`GroupMode`]:
    ///
    /// - **Broadcast**: all active members.
    /// - **MainBackup**: the highest-priority active member; if none, the
    ///   highest-priority standby.
    /// - **Balancing**: all active members (caller is responsible for
    ///   splitting payload).
    #[must_use]
    pub fn send_targets(&self, _seq: u32) -> Vec<u32> {
        match self.config.mode {
            GroupMode::Broadcast | GroupMode::Balancing => self
                .members
                .values()
                .filter(|m| m.role.is_active())
                .map(|m| m.id)
                .collect(),
            GroupMode::MainBackup => {
                // Pick the lowest-priority-number active member.
                let best_active = self
                    .members
                    .values()
                    .filter(|m| m.role == MemberRole::Active)
                    .min_by_key(|m| m.priority);

                if let Some(m) = best_active {
                    vec![m.id]
                } else {
                    // Fall back to lowest-priority standby.
                    self.members
                        .values()
                        .filter(|m| m.role == MemberRole::Standby)
                        .min_by_key(|m| m.priority)
                        .map(|m| vec![m.id])
                        .unwrap_or_default()
                }
            }
        }
    }

    /// Processes an incoming packet with sequence number `seq` on `member_id`.
    ///
    /// Returns `true` if the packet is new (should be delivered to the
    /// application layer), or `false` if it is a duplicate.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::NotFound`] if `member_id` is unknown.
    pub fn receive(&mut self, member_id: u32, seq: u32, bytes: u64) -> NetResult<bool> {
        let member = self
            .members
            .get_mut(&member_id)
            .ok_or_else(|| NetError::not_found(format!("group member {member_id} not found")))?;

        member.last_recv_seq = seq;
        member.health.packets_received += 1;

        let is_dup = self.dedup.check_and_insert(seq);
        if is_dup {
            self.stats.duplicates_discarded += 1;
            return Ok(false);
        }

        self.stats.packets_delivered += 1;
        self.stats.bytes_sent += bytes;
        Ok(true)
    }

    /// Updates the health metrics for member `id` and re-evaluates its role.
    ///
    /// In MainBackup mode this may trigger a role switch if the current active
    /// member becomes unhealthy or a standby member becomes healthier.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::NotFound`] if `id` is unknown.
    pub fn update_health(
        &mut self,
        id: u32,
        rtt_ms: f64,
        loss_fraction: f64,
        bandwidth_bps: u64,
    ) -> NetResult<()> {
        {
            let member = self
                .members
                .get_mut(&id)
                .ok_or_else(|| NetError::not_found(format!("group member {id} not found")))?;
            member.health.update(rtt_ms, loss_fraction, bandwidth_bps);
        }

        if self.config.mode == GroupMode::MainBackup {
            self.evaluate_main_backup();
        }
        self.refresh_stats();
        Ok(())
    }

    /// Returns an immutable reference to a member.
    #[must_use]
    pub fn member(&self, id: u32) -> Option<&GroupMember> {
        self.members.get(&id)
    }

    /// Returns the current group statistics.
    #[must_use]
    pub fn stats(&self) -> &GroupStats {
        &self.stats
    }

    /// Returns all member IDs.
    #[must_use]
    pub fn member_ids(&self) -> Vec<u32> {
        self.members.keys().copied().collect()
    }

    /// Returns the number of members with the given role.
    #[must_use]
    pub fn member_count_with_role(&self, role: MemberRole) -> usize {
        self.members.values().filter(|m| m.role == role).count()
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Evaluates role assignments for MainBackup mode.
    ///
    /// - Demotes the current active member if it is unhealthy.
    /// - Promotes the healthiest standby member if the active is gone.
    fn evaluate_main_backup(&mut self) {
        let max_loss = self.config.max_loss_fraction;
        let max_rtt = self.config.max_rtt_ms;

        // Find the current active member (if any).
        let active_id = self
            .members
            .values()
            .find(|m| m.role == MemberRole::Active)
            .map(|m| m.id);

        if let Some(aid) = active_id {
            let unhealthy = {
                let m = &self.members[&aid];
                !m.health.is_healthy(max_loss, max_rtt)
            };
            if unhealthy {
                // Demote current active.
                if let Some(m) = self.members.get_mut(&aid) {
                    m.set_role(MemberRole::Standby);
                    self.stats.role_switches += 1;
                }

                // Promote the healthiest standby.
                self.promote_best_standby();
            }
        } else {
            self.promote_best_standby();
        }
    }

    /// Promotes the standby member with the best health score.
    fn promote_best_standby(&mut self) {
        let best_id = self
            .members
            .values()
            .filter(|m| m.role == MemberRole::Standby)
            .max_by(|a, b| {
                a.health
                    .score()
                    .partial_cmp(&b.health.score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|m| m.id);

        if let Some(id) = best_id {
            if let Some(m) = self.members.get_mut(&id) {
                m.set_role(MemberRole::Active);
                self.stats.role_switches += 1;
            }
        }
    }

    /// Refreshes aggregate stats from the current member set.
    fn refresh_stats(&mut self) {
        self.stats.active_member_count = self
            .members
            .values()
            .filter(|m| m.role == MemberRole::Active)
            .count();
        self.stats.standby_member_count = self
            .members
            .values()
            .filter(|m| m.role == MemberRole::Standby)
            .count();
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(port: u16) -> SocketAddr {
        format!("127.0.0.1:{port}").parse().unwrap()
    }

    fn broadcast_mgr() -> SrtGroupManager {
        SrtGroupManager::new(GroupConfig::default())
    }

    #[test]
    fn test_add_member_and_count() {
        let mut mgr = broadcast_mgr();
        let id = mgr.add_member(addr(9000), MemberRole::Active, 0).unwrap();
        assert_eq!(id, 1);
        assert_eq!(mgr.member_count_with_role(MemberRole::Active), 1);
    }

    #[test]
    fn test_remove_member() {
        let mut mgr = broadcast_mgr();
        let id = mgr.add_member(addr(9000), MemberRole::Active, 0).unwrap();
        mgr.remove_member(id).unwrap();
        assert_eq!(mgr.member_count_with_role(MemberRole::Active), 0);
    }

    #[test]
    fn test_remove_unknown_member_returns_error() {
        let mut mgr = broadcast_mgr();
        let err = mgr.remove_member(99).unwrap_err();
        assert!(matches!(err, NetError::NotFound(_)));
    }

    #[test]
    fn test_broadcast_targets_all_active() {
        let mut mgr = broadcast_mgr();
        mgr.add_member(addr(9001), MemberRole::Active, 0).unwrap();
        mgr.add_member(addr(9002), MemberRole::Active, 1).unwrap();
        mgr.add_member(addr(9003), MemberRole::Standby, 2).unwrap();

        let targets = mgr.send_targets(1);
        assert_eq!(targets.len(), 2, "only Active members are targeted");
    }

    #[test]
    fn test_main_backup_targets_lowest_priority_active() {
        let mut mgr = SrtGroupManager::new(GroupConfig::main_backup());
        mgr.add_member(addr(9001), MemberRole::Active, 10).unwrap();
        let hi_prio = mgr.add_member(addr(9002), MemberRole::Active, 1).unwrap();

        let targets = mgr.send_targets(1);
        assert_eq!(
            targets,
            vec![hi_prio],
            "should prefer member with priority 1"
        );
    }

    #[test]
    fn test_main_backup_fallback_to_standby() {
        let mut mgr = SrtGroupManager::new(GroupConfig::main_backup());
        let sb = mgr.add_member(addr(9001), MemberRole::Standby, 0).unwrap();

        let targets = mgr.send_targets(1);
        assert_eq!(targets, vec![sb], "should fall back to standby");
    }

    #[test]
    fn test_dedup_window_rejects_duplicate_seq() {
        let mut mgr = broadcast_mgr();
        let id = mgr.add_member(addr(9001), MemberRole::Active, 0).unwrap();

        let first = mgr.receive(id, 100, 1316).unwrap();
        let dup = mgr.receive(id, 100, 1316).unwrap();

        assert!(first, "first packet should be accepted");
        assert!(!dup, "duplicate should be rejected");
        assert_eq!(mgr.stats().duplicates_discarded, 1);
    }

    #[test]
    fn test_receive_unknown_member_returns_error() {
        let mut mgr = broadcast_mgr();
        let err = mgr.receive(999, 1, 100).unwrap_err();
        assert!(matches!(err, NetError::NotFound(_)));
    }

    #[test]
    fn test_health_update_and_score() {
        let mut mgr = broadcast_mgr();
        let id = mgr.add_member(addr(9001), MemberRole::Active, 0).unwrap();
        mgr.update_health(id, 20.0, 0.0, 10_000_000).unwrap();

        let score = mgr.member(id).unwrap().health.score();
        assert!(
            score > 0.9,
            "healthy member should score > 0.9, got {score}"
        );
    }

    #[test]
    fn test_main_backup_role_switch_on_unhealthy() {
        let config = GroupConfig {
            mode: GroupMode::MainBackup,
            max_loss_fraction: 0.05,
            max_rtt_ms: 100.0,
            ..GroupConfig::default()
        };
        let mut mgr = SrtGroupManager::new(config);

        let main = mgr.add_member(addr(9001), MemberRole::Active, 0).unwrap();
        let backup = mgr.add_member(addr(9002), MemberRole::Standby, 1).unwrap();

        // Inject bad health into main.
        mgr.update_health(main, 200.0, 0.5, 1_000_000).unwrap();

        // Main should now be standby, backup should be active.
        assert_eq!(
            mgr.member(main).unwrap().role,
            MemberRole::Standby,
            "degraded main should become standby"
        );
        assert_eq!(
            mgr.member(backup).unwrap().role,
            MemberRole::Active,
            "backup should be promoted"
        );
        assert!(mgr.stats().role_switches >= 1);
    }

    #[test]
    fn test_max_members_enforced() {
        let config = GroupConfig {
            max_members: 2,
            ..GroupConfig::default()
        };
        let mut mgr = SrtGroupManager::new(config);
        mgr.add_member(addr(9001), MemberRole::Active, 0).unwrap();
        mgr.add_member(addr(9002), MemberRole::Active, 1).unwrap();
        let err = mgr
            .add_member(addr(9003), MemberRole::Active, 2)
            .unwrap_err();
        assert!(matches!(err, NetError::InvalidState(_)));
    }

    #[test]
    fn test_group_mode_display() {
        assert_eq!(GroupMode::Broadcast.to_string(), "broadcast");
        assert_eq!(GroupMode::MainBackup.to_string(), "main-backup");
        assert_eq!(GroupMode::Balancing.to_string(), "balancing");
    }

    #[test]
    fn test_stats_after_successful_receives() {
        let mut mgr = broadcast_mgr();
        let id = mgr.add_member(addr(9001), MemberRole::Active, 0).unwrap();

        for seq in 1..=5u32 {
            mgr.receive(id, seq, 1316).unwrap();
        }

        assert_eq!(mgr.stats().packets_delivered, 5);
        assert_eq!(mgr.stats().duplicates_discarded, 0);
    }
}
