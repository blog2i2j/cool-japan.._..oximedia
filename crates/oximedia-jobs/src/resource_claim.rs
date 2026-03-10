//! Resource claiming: CPU/GPU/memory allocation with conflict detection and release.

#![allow(dead_code)]

use std::collections::HashMap;

// ── Resource kinds ────────────────────────────────────────────────────────────

/// The kind of resource being claimed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    /// CPU threads.
    Cpu,
    /// GPU device slots.
    Gpu,
    /// Memory in bytes.
    Memory,
    /// Network bandwidth in bytes/sec.
    NetworkBandwidth,
    /// Disk I/O slots.
    DiskIo,
}

impl std::fmt::Display for ResourceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Cpu => "CPU",
            Self::Gpu => "GPU",
            Self::Memory => "Memory",
            Self::NetworkBandwidth => "NetworkBandwidth",
            Self::DiskIo => "DiskIO",
        };
        f.write_str(s)
    }
}

// ── Resource capacity ─────────────────────────────────────────────────────────

/// Total capacity of a single resource kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResourceCapacity {
    /// Kind of resource.
    pub kind: ResourceKind,
    /// Total available units.
    pub total: u64,
}

impl ResourceCapacity {
    /// Create a new capacity descriptor.
    #[must_use]
    pub const fn new(kind: ResourceKind, total: u64) -> Self {
        Self { kind, total }
    }
}

// ── Claim ─────────────────────────────────────────────────────────────────────

/// A request for a specific amount of a resource.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceClaim {
    /// Unique identifier of the claimant (e.g. job ID string).
    pub claimant: String,
    /// Kind of resource claimed.
    pub kind: ResourceKind,
    /// Amount claimed.
    pub amount: u64,
}

impl ResourceClaim {
    /// Create a new resource claim.
    #[must_use]
    pub fn new(claimant: impl Into<String>, kind: ResourceKind, amount: u64) -> Self {
        Self {
            claimant: claimant.into(),
            kind,
            amount,
        }
    }
}

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors from the resource manager.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResourceError {
    /// Not enough capacity available.
    InsufficientCapacity {
        /// Resource kind.
        kind: ResourceKind,
        /// Amount requested.
        requested: u64,
        /// Amount available.
        available: u64,
    },
    /// Unknown resource kind (not registered).
    UnknownResource(ResourceKind),
    /// Claimant has no active claims for this kind.
    NoClaim(String, ResourceKind),
}

impl std::fmt::Display for ResourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientCapacity {
                kind,
                requested,
                available,
            } => {
                write!(
                    f,
                    "insufficient {kind}: requested {requested}, available {available}"
                )
            }
            Self::UnknownResource(k) => write!(f, "unknown resource kind: {k}"),
            Self::NoClaim(id, k) => write!(f, "no claim by '{id}' on {k}"),
        }
    }
}

// ── ResourceManager ───────────────────────────────────────────────────────────

/// Tracks resource capacities and active claims.
pub struct ResourceManager {
    /// Total capacity per resource kind.
    capacities: HashMap<ResourceKind, u64>,
    /// Currently claimed amount per resource kind.
    claimed: HashMap<ResourceKind, u64>,
    /// Per-claimant, per-kind claim amounts.
    claimants: HashMap<String, HashMap<ResourceKind, u64>>,
}

impl ResourceManager {
    /// Create a new manager with the given capacities.
    #[must_use]
    pub fn new(capacities: &[ResourceCapacity]) -> Self {
        let cap_map = capacities.iter().map(|c| (c.kind, c.total)).collect();
        let claimed = capacities.iter().map(|c| (c.kind, 0u64)).collect();
        Self {
            capacities: cap_map,
            claimed,
            claimants: HashMap::new(),
        }
    }

    /// Returns the available (unclaimed) amount for `kind`.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceError::UnknownResource`] if the kind is not registered.
    pub fn available(&self, kind: ResourceKind) -> Result<u64, ResourceError> {
        let total = self
            .capacities
            .get(&kind)
            .ok_or(ResourceError::UnknownResource(kind))?;
        let used = self.claimed.get(&kind).copied().unwrap_or(0);
        Ok(total.saturating_sub(used))
    }

    /// Attempt to acquire a resource claim.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceError::UnknownResource`] if the kind is not registered.
    /// Returns [`ResourceError::InsufficientCapacity`] if not enough is available.
    pub fn acquire(&mut self, claim: ResourceClaim) -> Result<(), ResourceError> {
        let avail = self.available(claim.kind)?;
        if avail < claim.amount {
            return Err(ResourceError::InsufficientCapacity {
                kind: claim.kind,
                requested: claim.amount,
                available: avail,
            });
        }

        *self.claimed.entry(claim.kind).or_insert(0) += claim.amount;
        self.claimants
            .entry(claim.claimant.clone())
            .or_default()
            .entry(claim.kind)
            .and_modify(|v| *v += claim.amount)
            .or_insert(claim.amount);

        Ok(())
    }

    /// Release a previously acquired claim.
    ///
    /// # Errors
    ///
    /// Returns [`ResourceError::NoClaim`] if the claimant has no active claim.
    pub fn release(&mut self, claimant: &str, kind: ResourceKind) -> Result<(), ResourceError> {
        let amount = self
            .claimants
            .get_mut(claimant)
            .and_then(|m| m.remove(&kind))
            .ok_or_else(|| ResourceError::NoClaim(claimant.to_string(), kind))?;

        *self.claimed.entry(kind).or_insert(0) = self
            .claimed
            .get(&kind)
            .copied()
            .unwrap_or(0)
            .saturating_sub(amount);

        Ok(())
    }

    /// Returns the total claimed amount for `kind`.
    #[must_use]
    pub fn total_claimed(&self, kind: ResourceKind) -> u64 {
        self.claimed.get(&kind).copied().unwrap_or(0)
    }

    /// Returns a list of all claimant IDs that hold claims on `kind`.
    #[must_use]
    pub fn claimants_for(&self, kind: ResourceKind) -> Vec<&str> {
        self.claimants
            .iter()
            .filter(|(_, m)| m.contains_key(&kind))
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Returns `true` if `claimant` holds any active claims.
    #[must_use]
    pub fn has_claims(&self, claimant: &str) -> bool {
        self.claimants
            .get(claimant)
            .map(|m| !m.is_empty())
            .unwrap_or(false)
    }

    /// Forcibly release all claims held by `claimant`.
    pub fn release_all(&mut self, claimant: &str) {
        if let Some(claims) = self.claimants.remove(claimant) {
            for (kind, amount) in claims {
                *self.claimed.entry(kind).or_insert(0) = self
                    .claimed
                    .get(&kind)
                    .copied()
                    .unwrap_or(0)
                    .saturating_sub(amount);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> ResourceManager {
        ResourceManager::new(&[
            ResourceCapacity::new(ResourceKind::Cpu, 8),
            ResourceCapacity::new(ResourceKind::Gpu, 2),
            ResourceCapacity::new(ResourceKind::Memory, 16_000_000_000),
        ])
    }

    #[test]
    fn test_resource_kind_display() {
        assert_eq!(ResourceKind::Cpu.to_string(), "CPU");
        assert_eq!(ResourceKind::Memory.to_string(), "Memory");
    }

    #[test]
    fn test_available_full() {
        let mgr = make_manager();
        assert_eq!(
            mgr.available(ResourceKind::Cpu)
                .expect("available should succeed"),
            8
        );
    }

    #[test]
    fn test_unknown_resource() {
        let mgr = ResourceManager::new(&[]);
        assert!(matches!(
            mgr.available(ResourceKind::Cpu),
            Err(ResourceError::UnknownResource(ResourceKind::Cpu))
        ));
    }

    #[test]
    fn test_acquire_success() {
        let mut mgr = make_manager();
        let claim = ResourceClaim::new("job-1", ResourceKind::Cpu, 4);
        assert!(mgr.acquire(claim).is_ok());
        assert_eq!(
            mgr.available(ResourceKind::Cpu)
                .expect("available should succeed"),
            4
        );
    }

    #[test]
    fn test_acquire_insufficient() {
        let mut mgr = make_manager();
        let claim = ResourceClaim::new("job-1", ResourceKind::Cpu, 16);
        let err = mgr.acquire(claim).unwrap_err();
        assert!(matches!(err, ResourceError::InsufficientCapacity { .. }));
    }

    #[test]
    fn test_release_success() {
        let mut mgr = make_manager();
        mgr.acquire(ResourceClaim::new("job-1", ResourceKind::Cpu, 4))
            .expect("test expectation failed");
        mgr.release("job-1", ResourceKind::Cpu)
            .expect("release should succeed");
        assert_eq!(
            mgr.available(ResourceKind::Cpu)
                .expect("available should succeed"),
            8
        );
    }

    #[test]
    fn test_release_no_claim() {
        let mut mgr = make_manager();
        let err = mgr.release("job-x", ResourceKind::Gpu).unwrap_err();
        assert!(matches!(err, ResourceError::NoClaim(_, ResourceKind::Gpu)));
    }

    #[test]
    fn test_multiple_claimants() {
        let mut mgr = make_manager();
        mgr.acquire(ResourceClaim::new("job-1", ResourceKind::Cpu, 3))
            .expect("test expectation failed");
        mgr.acquire(ResourceClaim::new("job-2", ResourceKind::Cpu, 3))
            .expect("test expectation failed");
        assert_eq!(mgr.total_claimed(ResourceKind::Cpu), 6);
        assert_eq!(
            mgr.available(ResourceKind::Cpu)
                .expect("available should succeed"),
            2
        );
    }

    #[test]
    fn test_claimants_for() {
        let mut mgr = make_manager();
        mgr.acquire(ResourceClaim::new("job-1", ResourceKind::Gpu, 1))
            .expect("test expectation failed");
        let cl = mgr.claimants_for(ResourceKind::Gpu);
        assert!(cl.contains(&"job-1"));
    }

    #[test]
    fn test_has_claims() {
        let mut mgr = make_manager();
        assert!(!mgr.has_claims("job-1"));
        mgr.acquire(ResourceClaim::new("job-1", ResourceKind::Cpu, 1))
            .expect("test expectation failed");
        assert!(mgr.has_claims("job-1"));
    }

    #[test]
    fn test_release_all() {
        let mut mgr = make_manager();
        mgr.acquire(ResourceClaim::new("job-1", ResourceKind::Cpu, 4))
            .expect("test expectation failed");
        mgr.acquire(ResourceClaim::new("job-1", ResourceKind::Gpu, 1))
            .expect("test expectation failed");
        mgr.release_all("job-1");
        assert!(!mgr.has_claims("job-1"));
        assert_eq!(mgr.total_claimed(ResourceKind::Cpu), 0);
        assert_eq!(mgr.total_claimed(ResourceKind::Gpu), 0);
    }

    #[test]
    fn test_resource_capacity_new() {
        let cap = ResourceCapacity::new(ResourceKind::Memory, 8_000_000);
        assert_eq!(cap.kind, ResourceKind::Memory);
        assert_eq!(cap.total, 8_000_000);
    }

    #[test]
    fn test_resource_error_display() {
        let err = ResourceError::InsufficientCapacity {
            kind: ResourceKind::Gpu,
            requested: 4,
            available: 2,
        };
        let s = err.to_string();
        assert!(s.contains("GPU"));
        assert!(s.contains("4"));
    }

    #[test]
    fn test_resource_claim_new() {
        let c = ResourceClaim::new("job-42", ResourceKind::DiskIo, 10);
        assert_eq!(c.claimant, "job-42");
        assert_eq!(c.amount, 10);
    }
}
