//! Data sharding and consistent hashing for distributed workload placement.
//!
//! Provides a consistent hash ring for stable shard-to-node mapping, shard
//! metadata tracking, and rebalancing utilities.

#![allow(dead_code)]

use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// ShardKey
// ---------------------------------------------------------------------------

/// A key that identifies a data shard.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShardKey(pub String);

impl ShardKey {
    /// Create a new shard key.
    #[must_use]
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }

    /// Compute a 64-bit hash of the key bytes using SipHash-like mixing.
    ///
    /// Uses multiple rounds of mixing to ensure good distribution across the
    /// full u64 range.
    #[must_use]
    pub fn hash64(&self) -> u64 {
        // FNV-1a base
        const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
        const FNV_PRIME: u64 = 1_099_511_628_211;
        let mut h = FNV_OFFSET;
        for &b in self.0.as_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(FNV_PRIME);
        }
        // Finalisation mix (based on splitmix64) for better avalanche
        h ^= h >> 30;
        h = h.wrapping_mul(0xbf58476d1ce4e5b9);
        h ^= h >> 27;
        h = h.wrapping_mul(0x94d049bb133111eb);
        h ^= h >> 31;
        h
    }
}

// ---------------------------------------------------------------------------
// VirtualNode
// ---------------------------------------------------------------------------

/// A virtual node (token) in the consistent hash ring.
#[derive(Debug, Clone)]
pub struct VirtualNode {
    /// Ring position (hash of `node_id:replica_index`).
    pub position: u64,
    /// The physical node this virtual node belongs to.
    pub node_id: String,
    /// Replica index within the node (0-based).
    pub replica: u32,
}

// ---------------------------------------------------------------------------
// ConsistentHashRing
// ---------------------------------------------------------------------------

/// A consistent hash ring for stable shard-to-node assignment.
///
/// Each physical node is represented by `replicas_per_node` virtual nodes
/// spread across the ring.  Adding or removing a node only relocates
/// `1 / N` of the shards on average.
#[derive(Debug, Default)]
pub struct ConsistentHashRing {
    ring: BTreeMap<u64, VirtualNode>,
    replicas_per_node: u32,
}

impl ConsistentHashRing {
    /// Create a new ring with `replicas_per_node` virtual nodes per physical node.
    #[must_use]
    pub fn new(replicas_per_node: u32) -> Self {
        Self {
            ring: BTreeMap::new(),
            replicas_per_node: replicas_per_node.max(1),
        }
    }

    /// Add a node to the ring.
    pub fn add_node(&mut self, node_id: impl Into<String>) {
        let node_id = node_id.into();
        for replica in 0..self.replicas_per_node {
            let key = ShardKey::new(format!("{node_id}:{replica}"));
            let position = key.hash64();
            self.ring.insert(
                position,
                VirtualNode {
                    position,
                    node_id: node_id.clone(),
                    replica,
                },
            );
        }
    }

    /// Remove a node from the ring.
    pub fn remove_node(&mut self, node_id: &str) {
        for replica in 0..self.replicas_per_node {
            let key = ShardKey::new(format!("{node_id}:{replica}"));
            let position = key.hash64();
            self.ring.remove(&position);
        }
    }

    /// Find the responsible node for a shard key.
    ///
    /// Returns `None` if the ring is empty.
    #[must_use]
    pub fn get_node(&self, key: &ShardKey) -> Option<&str> {
        if self.ring.is_empty() {
            return None;
        }
        let hash = key.hash64();
        // Walk clockwise from hash; wrap around if needed.
        let node = self
            .ring
            .range(hash..)
            .next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, v)| v.node_id.as_str());
        node
    }

    /// Return the number of virtual nodes (tokens) currently in the ring.
    #[must_use]
    pub fn virtual_node_count(&self) -> usize {
        self.ring.len()
    }

    /// Return the number of distinct physical nodes in the ring.
    #[must_use]
    pub fn physical_node_count(&self) -> usize {
        let mut nodes: Vec<&str> = self.ring.values().map(|v| v.node_id.as_str()).collect();
        nodes.sort_unstable();
        nodes.dedup();
        nodes.len()
    }

    /// List all distinct physical node IDs.
    #[must_use]
    pub fn nodes(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.ring.values().map(|v| v.node_id.clone()).collect();
        ids.sort();
        ids.dedup();
        ids
    }
}

// ---------------------------------------------------------------------------
// ShardMetadata
// ---------------------------------------------------------------------------

/// Lifecycle state of a shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardState {
    /// Shard is available and fully up-to-date.
    Active,
    /// Shard is being migrated to another node.
    Migrating,
    /// Shard has been archived / is no longer hot.
    Archived,
}

/// Metadata about a single shard.
#[derive(Debug, Clone)]
pub struct ShardMetadata {
    /// The shard's key.
    pub key: ShardKey,
    /// Physical node currently responsible for this shard.
    pub owner_node: String,
    /// Approximate size in bytes.
    pub size_bytes: u64,
    /// Current lifecycle state.
    pub state: ShardState,
    /// Unix epoch ms when this metadata was last updated.
    pub updated_at_ms: u64,
}

impl ShardMetadata {
    /// Create a new active shard metadata record.
    #[must_use]
    pub fn new(key: ShardKey, owner_node: impl Into<String>, size_bytes: u64, now_ms: u64) -> Self {
        Self {
            key,
            owner_node: owner_node.into(),
            size_bytes,
            state: ShardState::Active,
            updated_at_ms: now_ms,
        }
    }

    /// Mark the shard as migrating to `target_node`.
    pub fn begin_migration(&mut self, target_node: impl Into<String>, now_ms: u64) {
        self.owner_node = target_node.into();
        self.state = ShardState::Migrating;
        self.updated_at_ms = now_ms;
    }

    /// Complete the migration (shard is now active on the new node).
    pub fn complete_migration(&mut self, now_ms: u64) {
        self.state = ShardState::Active;
        self.updated_at_ms = now_ms;
    }

    /// Returns `true` if the shard is currently active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.state == ShardState::Active
    }
}

// ---------------------------------------------------------------------------
// ShardCatalog
// ---------------------------------------------------------------------------

/// A catalog of all shards in the cluster.
#[derive(Debug, Default)]
pub struct ShardCatalog {
    shards: Vec<ShardMetadata>,
}

impl ShardCatalog {
    /// Create an empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or update a shard entry.
    pub fn upsert(&mut self, shard: ShardMetadata) {
        if let Some(existing) = self.shards.iter_mut().find(|s| s.key == shard.key) {
            *existing = shard;
        } else {
            self.shards.push(shard);
        }
    }

    /// Find a shard by key.
    #[must_use]
    pub fn get(&self, key: &ShardKey) -> Option<&ShardMetadata> {
        self.shards.iter().find(|s| &s.key == key)
    }

    /// Return all shards owned by a given node.
    #[must_use]
    pub fn shards_for_node(&self, node_id: &str) -> Vec<&ShardMetadata> {
        self.shards
            .iter()
            .filter(|s| s.owner_node == node_id)
            .collect()
    }

    /// Total data size across all shards in bytes.
    #[must_use]
    pub fn total_size_bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.size_bytes).sum()
    }

    /// Number of shards in the catalog.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shards.len()
    }

    /// Returns `true` if the catalog is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shards.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── ShardKey ─────────────────────────────────────────────────────────

    #[test]
    fn test_shard_key_hash64_deterministic() {
        let k = ShardKey::new("video-chunk-0042");
        assert_eq!(k.hash64(), k.hash64());
    }

    #[test]
    fn test_shard_key_different_keys_different_hashes() {
        let k1 = ShardKey::new("a");
        let k2 = ShardKey::new("b");
        assert_ne!(k1.hash64(), k2.hash64());
    }

    // ── ConsistentHashRing ───────────────────────────────────────────────

    #[test]
    fn test_ring_empty_returns_none() {
        let ring = ConsistentHashRing::new(3);
        assert!(ring.get_node(&ShardKey::new("key")).is_none());
    }

    #[test]
    fn test_ring_single_node_always_assigned() {
        let mut ring = ConsistentHashRing::new(10);
        ring.add_node("node-0");
        for i in 0..20_u32 {
            let key = ShardKey::new(format!("key-{i}"));
            assert_eq!(ring.get_node(&key), Some("node-0"));
        }
    }

    #[test]
    fn test_ring_virtual_node_count() {
        let mut ring = ConsistentHashRing::new(5);
        ring.add_node("n0");
        ring.add_node("n1");
        assert_eq!(ring.virtual_node_count(), 10);
    }

    #[test]
    fn test_ring_physical_node_count() {
        let mut ring = ConsistentHashRing::new(5);
        ring.add_node("n0");
        ring.add_node("n1");
        ring.add_node("n2");
        assert_eq!(ring.physical_node_count(), 3);
    }

    #[test]
    fn test_ring_remove_node() {
        let mut ring = ConsistentHashRing::new(5);
        ring.add_node("n0");
        ring.add_node("n1");
        ring.remove_node("n0");
        assert_eq!(ring.physical_node_count(), 1);
    }

    #[test]
    fn test_ring_nodes_list() {
        let mut ring = ConsistentHashRing::new(3);
        ring.add_node("alpha");
        ring.add_node("beta");
        let nodes = ring.nodes();
        assert!(nodes.contains(&"alpha".to_string()));
        assert!(nodes.contains(&"beta".to_string()));
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn test_ring_distribution_two_nodes() {
        let mut ring = ConsistentHashRing::new(50);
        ring.add_node("node-A");
        ring.add_node("node-B");
        let mut a_count = 0_u32;
        let mut b_count = 0_u32;
        for i in 0..100_u32 {
            let k = ShardKey::new(format!("shard-{i}"));
            match ring.get_node(&k) {
                Some("node-A") => a_count += 1,
                Some("node-B") => b_count += 1,
                _ => {}
            }
        }
        // With 50 replicas each, distribution should be roughly equal
        assert!(a_count > 20 && b_count > 20, "a={a_count}, b={b_count}");
    }

    // ── ShardMetadata ────────────────────────────────────────────────────

    #[test]
    fn test_shard_metadata_initial_state() {
        let meta = ShardMetadata::new(ShardKey::new("k"), "node-0", 1024, 1000);
        assert!(meta.is_active());
        assert_eq!(meta.state, ShardState::Active);
    }

    #[test]
    fn test_shard_metadata_begin_migration() {
        let mut meta = ShardMetadata::new(ShardKey::new("k"), "node-0", 1024, 1000);
        meta.begin_migration("node-1", 2000);
        assert_eq!(meta.state, ShardState::Migrating);
        assert_eq!(meta.owner_node, "node-1");
    }

    #[test]
    fn test_shard_metadata_complete_migration() {
        let mut meta = ShardMetadata::new(ShardKey::new("k"), "node-0", 1024, 1000);
        meta.begin_migration("node-1", 2000);
        meta.complete_migration(3000);
        assert!(meta.is_active());
    }

    // ── ShardCatalog ─────────────────────────────────────────────────────

    #[test]
    fn test_catalog_empty() {
        let catalog = ShardCatalog::new();
        assert!(catalog.is_empty());
        assert_eq!(catalog.len(), 0);
    }

    #[test]
    fn test_catalog_upsert_and_get() {
        let mut catalog = ShardCatalog::new();
        let key = ShardKey::new("shard-0");
        catalog.upsert(ShardMetadata::new(key.clone(), "n0", 512, 100));
        let meta = catalog.get(&key).expect("get should return a value");
        assert_eq!(meta.size_bytes, 512);
    }

    #[test]
    fn test_catalog_upsert_updates_existing() {
        let mut catalog = ShardCatalog::new();
        let key = ShardKey::new("shard-0");
        catalog.upsert(ShardMetadata::new(key.clone(), "n0", 512, 100));
        catalog.upsert(ShardMetadata::new(key.clone(), "n1", 1024, 200));
        assert_eq!(catalog.len(), 1);
        assert_eq!(
            catalog
                .get(&key)
                .expect("get should return a value")
                .size_bytes,
            1024
        );
    }

    #[test]
    fn test_catalog_shards_for_node() {
        let mut catalog = ShardCatalog::new();
        catalog.upsert(ShardMetadata::new(ShardKey::new("s0"), "n0", 100, 1));
        catalog.upsert(ShardMetadata::new(ShardKey::new("s1"), "n0", 200, 1));
        catalog.upsert(ShardMetadata::new(ShardKey::new("s2"), "n1", 300, 1));
        let shards = catalog.shards_for_node("n0");
        assert_eq!(shards.len(), 2);
    }

    #[test]
    fn test_catalog_total_size_bytes() {
        let mut catalog = ShardCatalog::new();
        catalog.upsert(ShardMetadata::new(ShardKey::new("s0"), "n0", 100, 1));
        catalog.upsert(ShardMetadata::new(ShardKey::new("s1"), "n0", 200, 1));
        assert_eq!(catalog.total_size_bytes(), 300);
    }
}
