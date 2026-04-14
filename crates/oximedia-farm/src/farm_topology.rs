#![allow(dead_code)]
//! Farm topology management for the encoding farm.
//!
//! Models the physical and logical layout of the worker farm:
//!
//! - **Rack** — a physical rack (or cloud availability zone) that groups nodes
//!   sharing the same power domain and switch.
//! - **Zone** — a higher-level grouping (e.g. data-centre region, building)
//!   that contains one or more racks.
//! - **WorkerGroup** — a named set of workers that share capabilities or
//!   purpose (e.g. "gpu-transcoding", "archive-ingest").
//!
//! The module exposes:
//!
//! - [`FarmTopology`]: the authoritative store of topology information.
//! - [`TopologySchedulingHints`]: derived hints used by the scheduler to make
//!   topology-aware placement decisions (rack-local affinity, zone spreading,
//!   bandwidth constraints).
//! - [`BandwidthMatrix`]: a symmetric matrix of measured inter-rack bandwidths
//!   used to prefer intra-rack job distribution when output data is large.

use std::collections::{HashMap, HashSet};

use crate::{FarmError, WorkerId};

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Opaque rack identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RackId(pub String);

impl RackId {
    /// Create a new rack identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RackId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Opaque zone identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ZoneId(pub String);

impl ZoneId {
    /// Create a new zone identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ZoneId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Opaque worker-group identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct GroupId(pub String);

impl GroupId {
    /// Create a new group identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for GroupId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// Topology entities
// ---------------------------------------------------------------------------

/// A rack record within the topology.
#[derive(Debug, Clone)]
pub struct Rack {
    /// Stable rack identifier.
    pub id: RackId,
    /// Zone this rack belongs to.
    pub zone_id: ZoneId,
    /// Human-readable label.
    pub label: String,
    /// Maximum number of nodes (0 = unlimited).
    pub capacity: usize,
    /// Workers currently assigned to this rack.
    pub workers: HashSet<WorkerId>,
}

impl Rack {
    /// Create a new rack record.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::InvalidConfig`] if `label` is empty.
    pub fn new(
        id: RackId,
        zone_id: ZoneId,
        label: impl Into<String>,
        capacity: usize,
    ) -> crate::Result<Self> {
        let label = label.into();
        if label.is_empty() {
            return Err(FarmError::InvalidConfig("Rack label must not be empty".into()));
        }
        Ok(Self {
            id,
            zone_id,
            label,
            capacity,
            workers: HashSet::new(),
        })
    }

    /// Whether an additional worker can be assigned to this rack.
    #[must_use]
    pub fn has_capacity(&self) -> bool {
        self.capacity == 0 || self.workers.len() < self.capacity
    }
}

/// A zone record grouping racks together.
#[derive(Debug, Clone)]
pub struct Zone {
    /// Stable zone identifier.
    pub id: ZoneId,
    /// Human-readable label.
    pub label: String,
    /// Geographic or logical region tag (optional).
    pub region: Option<String>,
}

impl Zone {
    /// Create a new zone record.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::InvalidConfig`] if `label` is empty.
    pub fn new(id: ZoneId, label: impl Into<String>, region: Option<String>) -> crate::Result<Self> {
        let label = label.into();
        if label.is_empty() {
            return Err(FarmError::InvalidConfig("Zone label must not be empty".into()));
        }
        Ok(Self { id, label, region })
    }
}

/// A named group of workers sharing a common capability or purpose.
#[derive(Debug, Clone)]
pub struct WorkerGroup {
    /// Stable group identifier.
    pub id: GroupId,
    /// Human-readable label.
    pub label: String,
    /// Capability tags associated with this group (e.g. `"gpu"`, `"hdr"`).
    pub capabilities: HashSet<String>,
    /// Members of this group.
    pub members: HashSet<WorkerId>,
}

impl WorkerGroup {
    /// Create a new worker group.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::InvalidConfig`] if `label` is empty.
    pub fn new(id: GroupId, label: impl Into<String>) -> crate::Result<Self> {
        let label = label.into();
        if label.is_empty() {
            return Err(FarmError::InvalidConfig(
                "WorkerGroup label must not be empty".into(),
            ));
        }
        Ok(Self {
            id,
            label,
            capabilities: HashSet::new(),
            members: HashSet::new(),
        })
    }

    /// Add a capability tag.
    pub fn add_capability(&mut self, cap: impl Into<String>) {
        self.capabilities.insert(cap.into());
    }

    /// Whether this group advertises the given capability.
    #[must_use]
    pub fn has_capability(&self, cap: &str) -> bool {
        self.capabilities.contains(cap)
    }
}

// ---------------------------------------------------------------------------
// Bandwidth matrix
// ---------------------------------------------------------------------------

/// Symmetric matrix of measured inter-rack bandwidths (in Mbit/s).
///
/// Entries are stored as an upper-triangular map for space efficiency.
/// Intra-rack bandwidth (rack to itself) is always `f64::INFINITY` by
/// convention because local I/O is effectively unconstrained.
#[derive(Debug, Clone, Default)]
pub struct BandwidthMatrix {
    entries: HashMap<(String, String), f64>,
}

impl BandwidthMatrix {
    /// Create an empty bandwidth matrix.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the measured bandwidth between two racks (in Mbit/s).
    ///
    /// The pair is stored in lexicographic order so that lookups are symmetric.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::InvalidConfig`] if `bandwidth_mbps` is negative or
    /// NaN.
    pub fn set(&mut self, a: &RackId, b: &RackId, bandwidth_mbps: f64) -> crate::Result<()> {
        if !bandwidth_mbps.is_finite() || bandwidth_mbps < 0.0 {
            return Err(FarmError::InvalidConfig(
                "Bandwidth must be a non-negative finite number".into(),
            ));
        }
        let key = canonical_key(a.as_str(), b.as_str());
        self.entries.insert(key, bandwidth_mbps);
        Ok(())
    }

    /// Query the bandwidth between two racks.
    ///
    /// Returns `f64::INFINITY` when the two rack IDs are identical (intra-rack
    /// traffic), or `None` when no measurement is available.
    #[must_use]
    pub fn get(&self, a: &RackId, b: &RackId) -> Option<f64> {
        if a == b {
            return Some(f64::INFINITY);
        }
        let key = canonical_key(a.as_str(), b.as_str());
        self.entries.get(&key).copied()
    }
}

fn canonical_key(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

// ---------------------------------------------------------------------------
// Topology-aware scheduling hints
// ---------------------------------------------------------------------------

/// Hints derived from the topology for the scheduler.
#[derive(Debug, Clone)]
pub struct TopologySchedulingHints {
    /// Ordered list of workers sorted by network proximity to the "source"
    /// rack.  Workers in the same rack appear first.
    pub rack_preferred_workers: Vec<WorkerId>,

    /// Workers spread across distinct zones (for redundancy-aware placement).
    pub zone_spread_workers: Vec<WorkerId>,

    /// Workers that belong to groups advertising all required capabilities.
    pub capable_workers: Vec<WorkerId>,
}

// ---------------------------------------------------------------------------
// FarmTopology — the authoritative registry
// ---------------------------------------------------------------------------

/// Central registry of farm topology: zones, racks, workers, and groups.
#[derive(Debug)]
pub struct FarmTopology {
    zones: HashMap<ZoneId, Zone>,
    racks: HashMap<RackId, Rack>,
    groups: HashMap<GroupId, WorkerGroup>,
    /// Map from worker → rack for O(1) lookups.
    worker_rack: HashMap<WorkerId, RackId>,
    bandwidth: BandwidthMatrix,
}

impl FarmTopology {
    /// Create an empty topology.
    #[must_use]
    pub fn new() -> Self {
        Self {
            zones: HashMap::new(),
            racks: HashMap::new(),
            groups: HashMap::new(),
            worker_rack: HashMap::new(),
            bandwidth: BandwidthMatrix::new(),
        }
    }

    // --- Zone management ----------------------------------------------------

    /// Add a zone.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::AlreadyExists`] if a zone with the same ID is
    /// already registered.
    pub fn add_zone(&mut self, zone: Zone) -> crate::Result<()> {
        if self.zones.contains_key(&zone.id) {
            return Err(FarmError::AlreadyExists(format!(
                "Zone {} already exists",
                zone.id
            )));
        }
        self.zones.insert(zone.id.clone(), zone);
        Ok(())
    }

    /// Look up a zone by ID.
    pub fn zone(&self, id: &ZoneId) -> Option<&Zone> {
        self.zones.get(id)
    }

    // --- Rack management ----------------------------------------------------

    /// Add a rack.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::AlreadyExists`] if the rack already exists, or
    /// [`FarmError::NotFound`] if the referenced zone does not exist.
    pub fn add_rack(&mut self, rack: Rack) -> crate::Result<()> {
        if !self.zones.contains_key(&rack.zone_id) {
            return Err(FarmError::NotFound(format!(
                "Zone {} not found",
                rack.zone_id
            )));
        }
        if self.racks.contains_key(&rack.id) {
            return Err(FarmError::AlreadyExists(format!(
                "Rack {} already exists",
                rack.id
            )));
        }
        self.racks.insert(rack.id.clone(), rack);
        Ok(())
    }

    /// Look up a rack by ID.
    pub fn rack(&self, id: &RackId) -> Option<&Rack> {
        self.racks.get(id)
    }

    // --- Worker placement ---------------------------------------------------

    /// Assign a worker to a rack.
    ///
    /// # Errors
    ///
    /// - [`FarmError::NotFound`] when the rack does not exist.
    /// - [`FarmError::ResourceExhausted`] when the rack has no remaining
    ///   capacity.
    /// - [`FarmError::AlreadyExists`] when the worker is already placed in a
    ///   rack.
    pub fn place_worker(&mut self, worker_id: WorkerId, rack_id: &RackId) -> crate::Result<()> {
        if self.worker_rack.contains_key(&worker_id) {
            return Err(FarmError::AlreadyExists(format!(
                "Worker {} is already placed in a rack",
                worker_id
            )));
        }
        let rack = self
            .racks
            .get_mut(rack_id)
            .ok_or_else(|| FarmError::NotFound(format!("Rack {} not found", rack_id)))?;
        if !rack.has_capacity() {
            return Err(FarmError::ResourceExhausted(format!(
                "Rack {} is at full capacity ({})",
                rack_id, rack.capacity
            )));
        }
        rack.workers.insert(worker_id.clone());
        self.worker_rack.insert(worker_id, rack_id.clone());
        Ok(())
    }

    /// Remove a worker from its rack.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::NotFound`] when the worker is not placed anywhere.
    pub fn remove_worker(&mut self, worker_id: &WorkerId) -> crate::Result<()> {
        let rack_id = self
            .worker_rack
            .remove(worker_id)
            .ok_or_else(|| FarmError::NotFound(format!("Worker {} not placed", worker_id)))?;
        if let Some(rack) = self.racks.get_mut(&rack_id) {
            rack.workers.remove(worker_id);
        }
        Ok(())
    }

    /// Find which rack a worker is currently placed in.
    #[must_use]
    pub fn worker_rack(&self, worker_id: &WorkerId) -> Option<&RackId> {
        self.worker_rack.get(worker_id)
    }

    // --- Group management ---------------------------------------------------

    /// Add a worker group.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::AlreadyExists`] if the group already exists.
    pub fn add_group(&mut self, group: WorkerGroup) -> crate::Result<()> {
        if self.groups.contains_key(&group.id) {
            return Err(FarmError::AlreadyExists(format!(
                "Group {} already exists",
                group.id
            )));
        }
        self.groups.insert(group.id.clone(), group);
        Ok(())
    }

    /// Add a worker to an existing group.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::NotFound`] when the group does not exist.
    pub fn join_group(&mut self, worker_id: WorkerId, group_id: &GroupId) -> crate::Result<()> {
        let group = self
            .groups
            .get_mut(group_id)
            .ok_or_else(|| FarmError::NotFound(format!("Group {} not found", group_id)))?;
        group.members.insert(worker_id);
        Ok(())
    }

    /// Remove a worker from a group.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::NotFound`] when the group does not exist.
    pub fn leave_group(&mut self, worker_id: &WorkerId, group_id: &GroupId) -> crate::Result<()> {
        let group = self
            .groups
            .get_mut(group_id)
            .ok_or_else(|| FarmError::NotFound(format!("Group {} not found", group_id)))?;
        group.members.remove(worker_id);
        Ok(())
    }

    /// Look up a group by ID.
    pub fn group(&self, id: &GroupId) -> Option<&WorkerGroup> {
        self.groups.get(id)
    }

    // --- Bandwidth ----------------------------------------------------------

    /// Record inter-rack bandwidth measurement.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`BandwidthMatrix::set`].
    pub fn set_bandwidth(
        &mut self,
        a: &RackId,
        b: &RackId,
        bandwidth_mbps: f64,
    ) -> crate::Result<()> {
        self.bandwidth.set(a, b, bandwidth_mbps)
    }

    /// Query the bandwidth between two racks.
    #[must_use]
    pub fn bandwidth_mbps(&self, a: &RackId, b: &RackId) -> Option<f64> {
        self.bandwidth.get(a, b)
    }

    // --- Scheduling hints ---------------------------------------------------

    /// Generate [`TopologySchedulingHints`] for a job that originates from
    /// `source_rack`.
    ///
    /// * `source_rack` — the rack where the input data resides (for locality
    ///   preference).
    /// * `required_capabilities` — capability tags the assigned worker must
    ///   have (e.g. `["gpu"]`).
    /// * `candidate_workers` — the set of workers pre-filtered by the
    ///   scheduler as eligible.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::NotFound`] when `source_rack` is unknown.
    pub fn scheduling_hints(
        &self,
        source_rack: &RackId,
        required_capabilities: &[&str],
        candidate_workers: &[WorkerId],
    ) -> crate::Result<TopologySchedulingHints> {
        if !self.racks.contains_key(source_rack) {
            return Err(FarmError::NotFound(format!(
                "Source rack {} not found",
                source_rack
            )));
        }

        // 1. Rack-preferred: workers co-located in source_rack come first.
        let source_rack_workers: HashSet<&WorkerId> = self
            .racks
            .get(source_rack)
            .map(|r| r.workers.iter().collect())
            .unwrap_or_default();

        let mut rack_preferred: Vec<WorkerId> = candidate_workers
            .iter()
            .filter(|w| source_rack_workers.contains(w))
            .cloned()
            .collect();
        let others: Vec<WorkerId> = candidate_workers
            .iter()
            .filter(|w| !source_rack_workers.contains(w))
            .cloned()
            .collect();
        rack_preferred.extend(others);

        // 2. Zone-spread: pick at most one representative per zone.
        let mut seen_zones: HashSet<ZoneId> = HashSet::new();
        let mut zone_spread: Vec<WorkerId> = Vec::new();
        for worker in candidate_workers {
            if let Some(rack_id) = self.worker_rack.get(worker) {
                if let Some(rack) = self.racks.get(rack_id) {
                    if seen_zones.insert(rack.zone_id.clone()) {
                        zone_spread.push(worker.clone());
                    }
                }
            }
        }

        // 3. Capable: workers in any group that satisfies ALL required caps.
        let capable_group_workers: HashSet<WorkerId> = if required_capabilities.is_empty() {
            candidate_workers.iter().cloned().collect()
        } else {
            self.groups
                .values()
                .filter(|g| {
                    required_capabilities
                        .iter()
                        .all(|cap| g.has_capability(cap))
                })
                .flat_map(|g| g.members.iter().cloned())
                .collect()
        };

        let capable_workers: Vec<WorkerId> = candidate_workers
            .iter()
            .filter(|w| capable_group_workers.contains(w))
            .cloned()
            .collect();

        Ok(TopologySchedulingHints {
            rack_preferred_workers: rack_preferred,
            zone_spread_workers: zone_spread,
            capable_workers,
        })
    }

    // --- Summary ------------------------------------------------------------

    /// Return a compact [`TopologySummary`] for monitoring dashboards.
    #[must_use]
    pub fn summary(&self) -> TopologySummary {
        let total_workers: usize = self.worker_rack.len();
        let racks_with_workers: usize = self
            .racks
            .values()
            .filter(|r| !r.workers.is_empty())
            .count();

        TopologySummary {
            zone_count: self.zones.len(),
            rack_count: self.racks.len(),
            group_count: self.groups.len(),
            total_placed_workers: total_workers,
            active_racks: racks_with_workers,
        }
    }
}

impl Default for FarmTopology {
    fn default() -> Self {
        Self::new()
    }
}

/// High-level statistics about the current farm topology.
#[derive(Debug, Clone)]
pub struct TopologySummary {
    /// Number of zones in the topology.
    pub zone_count: usize,
    /// Number of racks across all zones.
    pub rack_count: usize,
    /// Number of worker groups.
    pub group_count: usize,
    /// Number of workers with a rack assignment.
    pub total_placed_workers: usize,
    /// Number of racks that have at least one worker.
    pub active_racks: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WorkerId;

    fn setup_topology() -> FarmTopology {
        let mut topo = FarmTopology::new();

        let zone_a = Zone::new(ZoneId::new("zone-a"), "Zone A", Some("us-east".into()))
            .expect("zone ok");
        let zone_b =
            Zone::new(ZoneId::new("zone-b"), "Zone B", None).expect("zone ok");
        topo.add_zone(zone_a).expect("add zone-a");
        topo.add_zone(zone_b).expect("add zone-b");

        let rack1 = Rack::new(
            RackId::new("rack-1"),
            ZoneId::new("zone-a"),
            "Rack 1",
            4,
        )
        .expect("rack ok");
        let rack2 = Rack::new(
            RackId::new("rack-2"),
            ZoneId::new("zone-b"),
            "Rack 2",
            4,
        )
        .expect("rack ok");
        topo.add_rack(rack1).expect("add rack-1");
        topo.add_rack(rack2).expect("add rack-2");

        topo
    }

    #[test]
    fn test_zone_rack_creation() {
        let topo = setup_topology();
        let summary = topo.summary();
        assert_eq!(summary.zone_count, 2);
        assert_eq!(summary.rack_count, 2);
    }

    #[test]
    fn test_place_and_remove_worker() {
        let mut topo = setup_topology();
        let w1 = WorkerId::new("worker-1");
        let rack1 = RackId::new("rack-1");

        topo.place_worker(w1.clone(), &rack1).expect("place ok");
        assert_eq!(topo.worker_rack(&w1), Some(&rack1));
        assert_eq!(topo.summary().total_placed_workers, 1);

        topo.remove_worker(&w1).expect("remove ok");
        assert!(topo.worker_rack(&w1).is_none());
        assert_eq!(topo.summary().total_placed_workers, 0);
    }

    #[test]
    fn test_rack_capacity_enforcement() {
        let mut topo = FarmTopology::new();
        let zone = Zone::new(ZoneId::new("z"), "Z", None).expect("ok");
        topo.add_zone(zone).expect("ok");
        let rack = Rack::new(RackId::new("r"), ZoneId::new("z"), "R", 1).expect("ok");
        topo.add_rack(rack).expect("ok");

        topo.place_worker(WorkerId::new("w1"), &RackId::new("r"))
            .expect("first placement ok");
        // Second placement should fail — capacity is 1
        assert!(topo
            .place_worker(WorkerId::new("w2"), &RackId::new("r"))
            .is_err());
    }

    #[test]
    fn test_worker_group_capabilities() {
        let mut group =
            WorkerGroup::new(GroupId::new("gpu-group"), "GPU Workers").expect("ok");
        group.add_capability("gpu");
        group.add_capability("hdr");
        assert!(group.has_capability("gpu"));
        assert!(!group.has_capability("dolby"));
    }

    #[test]
    fn test_join_leave_group() {
        let mut topo = setup_topology();
        let mut grp = WorkerGroup::new(GroupId::new("g1"), "Group 1").expect("ok");
        grp.add_capability("gpu");
        topo.add_group(grp).expect("ok");

        let w = WorkerId::new("w1");
        topo.join_group(w.clone(), &GroupId::new("g1")).expect("join ok");
        assert!(topo
            .group(&GroupId::new("g1"))
            .expect("group exists")
            .members
            .contains(&w));

        topo.leave_group(&w, &GroupId::new("g1")).expect("leave ok");
        assert!(!topo
            .group(&GroupId::new("g1"))
            .expect("group exists")
            .members
            .contains(&w));
    }

    #[test]
    fn test_bandwidth_matrix_symmetry() {
        let mut bw = BandwidthMatrix::new();
        let r1 = RackId::new("r1");
        let r2 = RackId::new("r2");
        bw.set(&r1, &r2, 10_000.0).expect("set ok");
        assert_eq!(bw.get(&r1, &r2), Some(10_000.0));
        assert_eq!(bw.get(&r2, &r1), Some(10_000.0));
    }

    #[test]
    fn test_bandwidth_intra_rack_is_infinity() {
        let bw = BandwidthMatrix::new();
        let r = RackId::new("r1");
        assert_eq!(bw.get(&r, &r), Some(f64::INFINITY));
    }

    #[test]
    fn test_scheduling_hints_rack_affinity() {
        let mut topo = setup_topology();
        let w1 = WorkerId::new("local-worker");
        let w2 = WorkerId::new("remote-worker");
        topo.place_worker(w1.clone(), &RackId::new("rack-1")).expect("ok");
        topo.place_worker(w2.clone(), &RackId::new("rack-2")).expect("ok");

        let hints = topo
            .scheduling_hints(&RackId::new("rack-1"), &[], &[w1.clone(), w2.clone()])
            .expect("hints ok");

        // Local worker should appear first
        assert_eq!(hints.rack_preferred_workers.first(), Some(&w1));
    }

    #[test]
    fn test_scheduling_hints_capable_workers() {
        let mut topo = setup_topology();
        let w_gpu = WorkerId::new("gpu-worker");
        let w_cpu = WorkerId::new("cpu-worker");
        topo.place_worker(w_gpu.clone(), &RackId::new("rack-1")).expect("ok");
        topo.place_worker(w_cpu.clone(), &RackId::new("rack-2")).expect("ok");

        let mut grp = WorkerGroup::new(GroupId::new("gpu"), "GPU").expect("ok");
        grp.add_capability("gpu");
        grp.members.insert(w_gpu.clone());
        topo.add_group(grp).expect("ok");

        let hints = topo
            .scheduling_hints(
                &RackId::new("rack-1"),
                &["gpu"],
                &[w_gpu.clone(), w_cpu.clone()],
            )
            .expect("hints ok");

        assert_eq!(hints.capable_workers, vec![w_gpu]);
        assert!(!hints.capable_workers.contains(&w_cpu));
    }

    #[test]
    fn test_duplicate_zone_rejected() {
        let mut topo = FarmTopology::new();
        let zone = Zone::new(ZoneId::new("z"), "Z", None).expect("ok");
        topo.add_zone(zone).expect("first add ok");
        let zone2 = Zone::new(ZoneId::new("z"), "Z duplicate", None).expect("ok");
        assert!(topo.add_zone(zone2).is_err());
    }
}
