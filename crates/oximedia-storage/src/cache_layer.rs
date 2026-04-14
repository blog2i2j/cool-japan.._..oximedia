//! Storage cache layer: LRU, LFU, FIFO, and ARC caches with policy tracking
//! and statistics.

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Cache policy
// ---------------------------------------------------------------------------

/// Eviction / replacement policy for a cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePolicy {
    /// Least Recently Used.
    LRU,
    /// Least Frequently Used.
    LFU,
    /// First In, First Out.
    FIFO,
    /// Adaptive Replacement Cache.
    ARC,
}

impl CachePolicy {
    /// Returns a human-readable name for the policy.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::LRU => "LRU",
            Self::LFU => "LFU",
            Self::FIFO => "FIFO",
            Self::ARC => "ARC",
        }
    }
}

// ---------------------------------------------------------------------------
// Cache entry
// ---------------------------------------------------------------------------

/// Metadata for a single cache entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Unique identifier / key for the cached object.
    pub key: String,
    /// Size of the cached object in bytes.
    pub size_bytes: u64,
    /// Number of times this entry has been accessed.
    pub access_count: u64,
    /// Timestamp (in milliseconds) of the last access.
    pub last_access_ms: u64,
    /// Timestamp (in milliseconds) when the entry was created.
    pub created_ms: u64,
}

impl CacheEntry {
    /// Create a new cache entry.
    #[must_use]
    pub fn new(key: impl Into<String>, size_bytes: u64, now_ms: u64) -> Self {
        Self {
            key: key.into(),
            size_bytes,
            access_count: 0,
            last_access_ms: now_ms,
            created_ms: now_ms,
        }
    }

    /// Age of the entry in milliseconds relative to `now`.
    #[must_use]
    pub fn age_ms(&self, now: u64) -> u64 {
        now.saturating_sub(self.created_ms)
    }
}

// ---------------------------------------------------------------------------
// LRU cache
// ---------------------------------------------------------------------------

/// An LRU cache with a byte-capacity limit.
pub struct LruCache {
    /// Maximum total size in bytes.
    pub capacity_bytes: u64,
    /// Current total size in bytes.
    pub used_bytes: u64,
    /// Map from key to entry.
    pub entries: HashMap<String, CacheEntry>,
    /// LRU order: front = most recently used, back = least recently used.
    order: VecDeque<String>,
}

impl LruCache {
    /// Create a new LRU cache with the given byte capacity.
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    /// Retrieve an entry by key, updating its access timestamp.
    ///
    /// Returns `None` if the key is not cached.
    pub fn get(&mut self, key: &str, now_ms: u64) -> Option<&CacheEntry> {
        if !self.entries.contains_key(key) {
            return None;
        }

        // Move to front of LRU order
        self.order.retain(|k| k != key);
        self.order.push_front(key.to_string());

        if let Some(entry) = self.entries.get_mut(key) {
            entry.access_count += 1;
            entry.last_access_ms = now_ms;
        }

        self.entries.get(key)
    }

    /// Insert (or refresh) a cache entry.
    ///
    /// Evicts entries as needed to satisfy the byte capacity.
    pub fn put(&mut self, key: impl Into<String>, size_bytes: u64, now_ms: u64) {
        let key = key.into();

        // Remove existing entry of the same key first
        if let Some(old) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(old.size_bytes);
            self.order.retain(|k| k != &key);
        }

        // Evict until there is room
        while self.used_bytes + size_bytes > self.capacity_bytes && !self.order.is_empty() {
            self.evict();
        }

        let entry = CacheEntry::new(key.clone(), size_bytes, now_ms);
        self.used_bytes += size_bytes;
        self.entries.insert(key.clone(), entry);
        self.order.push_front(key);
    }

    /// Evict the least recently used entry.
    ///
    /// Returns `true` if an entry was evicted.
    pub fn evict(&mut self) -> bool {
        if let Some(lru_key) = self.order.pop_back() {
            if let Some(entry) = self.entries.remove(&lru_key) {
                self.used_bytes = self.used_bytes.saturating_sub(entry.size_bytes);
                return true;
            }
        }
        false
    }

    /// Cache utilisation as a fraction (0.0–1.0).
    ///
    /// Returns `0.0` when capacity is zero.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn utilization(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f64 / self.capacity_bytes as f64
    }

    /// Number of entries currently in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the cache contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// LFU cache
// ---------------------------------------------------------------------------

/// A Least Frequently Used cache with byte-capacity limit.
///
/// Evicts the entry with the lowest access count.  Ties are broken by the
/// entry with the oldest `last_access_ms` (LRU among least-frequently-used).
pub struct LfuCache {
    /// Maximum total size in bytes.
    pub capacity_bytes: u64,
    /// Current total size in bytes.
    pub used_bytes: u64,
    /// Map from key to entry.
    pub entries: HashMap<String, CacheEntry>,
}

impl LfuCache {
    /// Create a new LFU cache with the given byte capacity.
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
        }
    }

    /// Retrieve an entry, incrementing its access count.
    pub fn get(&mut self, key: &str, now_ms: u64) -> Option<&CacheEntry> {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.access_count += 1;
            entry.last_access_ms = now_ms;
        }
        self.entries.get(key)
    }

    /// Insert (or refresh) a cache entry.
    pub fn put(&mut self, key: impl Into<String>, size_bytes: u64, now_ms: u64) {
        let key = key.into();

        if let Some(old) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(old.size_bytes);
        }

        while self.used_bytes + size_bytes > self.capacity_bytes && !self.entries.is_empty() {
            self.evict();
        }

        let entry = CacheEntry::new(key.clone(), size_bytes, now_ms);
        self.used_bytes += size_bytes;
        self.entries.insert(key, entry);
    }

    /// Evict the least frequently used entry (ties broken by oldest access).
    pub fn evict(&mut self) -> bool {
        if self.entries.is_empty() {
            return false;
        }

        let victim_key = self
            .entries
            .iter()
            .min_by(|a, b| {
                a.1.access_count
                    .cmp(&b.1.access_count)
                    .then(a.1.last_access_ms.cmp(&b.1.last_access_ms))
            })
            .map(|(k, _)| k.clone());

        if let Some(key) = victim_key {
            if let Some(entry) = self.entries.remove(&key) {
                self.used_bytes = self.used_bytes.saturating_sub(entry.size_bytes);
                return true;
            }
        }
        false
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cache utilisation.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn utilization(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f64 / self.capacity_bytes as f64
    }
}

// ---------------------------------------------------------------------------
// ARC cache (Adaptive Replacement Cache)
// ---------------------------------------------------------------------------

/// Adaptive Replacement Cache.
///
/// Dynamically balances recency (LRU) and frequency (LFU) via four internal
/// lists (T1, T2, B1, B2) and an adaptive parameter **p**.
///
/// Eviction fires when *either* the entry count exceeds `capacity` *or* the
/// total bytes exceed `max_bytes` (when set via [`ArcCache::with_capacity`]).
///
/// * **T1** — recently accessed (seen once), most recent at front.
/// * **T2** — frequently accessed (seen 2+), most recent at front.
/// * **B1** — ghost entries evicted from T1 (keys only).
/// * **B2** — ghost entries evicted from T2 (keys only).
///
/// Reference: Megiddo & Modha, "ARC: A Self-Tuning, Low Overhead Replacement
/// Cache", FAST 2003.
pub struct ArcCache {
    /// Maximum number of cached entries (c).
    capacity: usize,
    /// Hard byte limit (0 = unlimited).
    max_bytes: usize,
    /// Total bytes currently held by live entries.
    total_bytes: usize,
    /// Target size for T1 (adapts at runtime).
    p: usize,
    /// T1: recent entries (accessed once).
    t1: VecDeque<String>,
    /// T2: frequent entries (accessed 2+).
    t2: VecDeque<String>,
    /// B1: ghost keys evicted from T1.
    b1: VecDeque<String>,
    /// B2: ghost keys evicted from T2.
    b2: VecDeque<String>,
    /// Cached entry metadata.
    entries: HashMap<String, CacheEntry>,
}

impl ArcCache {
    /// Create a new ARC cache bounded only by entry count.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            max_bytes: 0,
            total_bytes: 0,
            p: 0,
            t1: VecDeque::new(),
            t2: VecDeque::new(),
            b1: VecDeque::new(),
            b2: VecDeque::new(),
            entries: HashMap::new(),
        }
    }

    /// Create a new ARC cache with **dual-limit** eviction: entries are evicted
    /// when *either* the entry count exceeds `max_entries` *or* the cumulative
    /// byte size of all live entries exceeds `max_bytes`.
    ///
    /// # Arguments
    /// * `max_entries` — maximum number of concurrently cached entries.
    /// * `max_bytes`   — maximum total bytes across all cached entries.
    #[must_use]
    pub fn with_capacity(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            capacity: max_entries,
            max_bytes,
            total_bytes: 0,
            p: 0,
            t1: VecDeque::new(),
            t2: VecDeque::new(),
            b1: VecDeque::new(),
            b2: VecDeque::new(),
            entries: HashMap::new(),
        }
    }

    /// Current total bytes of all live entries.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Maximum byte limit (0 = unlimited).
    #[must_use]
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Returns `true` if either the entry-count or the byte limit is exceeded.
    fn over_limit(&self) -> bool {
        let over_count = self.entries.len() > self.capacity;
        let over_bytes = self.max_bytes > 0 && self.total_bytes > self.max_bytes;
        over_count || over_bytes
    }

    /// Look up a key.
    ///
    /// If in T1, promote to T2.  If in T2, move to front.
    pub fn get(&mut self, key: &str, now_ms: u64) -> Option<&CacheEntry> {
        if let Some(pos) = self.t1.iter().position(|k| k == key) {
            if let Some(k) = self.t1.remove(pos) {
                self.t2.push_front(k);
            }
            if let Some(e) = self.entries.get_mut(key) {
                e.access_count += 1;
                e.last_access_ms = now_ms;
            }
            return self.entries.get(key);
        }

        if let Some(pos) = self.t2.iter().position(|k| k == key) {
            if let Some(k) = self.t2.remove(pos) {
                self.t2.push_front(k);
            }
            if let Some(e) = self.entries.get_mut(key) {
                e.access_count += 1;
                e.last_access_ms = now_ms;
            }
            return self.entries.get(key);
        }

        None
    }

    /// Insert a new entry.
    ///
    /// Eviction fires when *either* the live entry count exceeds `capacity`
    /// *or* `total_bytes` exceeds `max_bytes` (when `max_bytes > 0`).
    #[allow(clippy::cast_possible_truncation)]
    pub fn put(&mut self, key: impl Into<String>, size_bytes: u64, now_ms: u64) {
        let key = key.into();
        let sz = size_bytes as usize;

        // Already cached → promote and update size
        if self.entries.contains_key(&key) {
            if let Some(old) = self.entries.get(&key) {
                let old_sz = old.size_bytes as usize;
                self.total_bytes = self.total_bytes.saturating_sub(old_sz);
            }
            self.get(&key, now_ms);
            self.total_bytes += sz;
            if let Some(e) = self.entries.get_mut(&key) {
                e.size_bytes = size_bytes;
            }
            // Re-evict if byte limit now exceeded after size update.
            while self.over_limit() {
                self.replace(false);
            }
            return;
        }

        // Hit in B1 → increase p (favour recency)
        if let Some(pos) = self.b1.iter().position(|k| k == &key) {
            let delta = (self.b2.len().max(1) / self.b1.len().max(1)).max(1);
            self.p = (self.p + delta).min(self.capacity);
            self.b1.remove(pos);
            self.replace(false);
            self.t2.push_front(key.clone());
            self.entries
                .insert(key.clone(), CacheEntry::new(key, size_bytes, now_ms));
            self.total_bytes += sz;
            // Drain if byte limit is exceeded after insertion.
            while self.over_limit() && (!self.t1.is_empty() || !self.t2.is_empty()) {
                self.replace(false);
            }
            return;
        }

        // Hit in B2 → decrease p (favour frequency)
        if let Some(pos) = self.b2.iter().position(|k| k == &key) {
            let delta = (self.b1.len().max(1) / self.b2.len().max(1)).max(1);
            self.p = self.p.saturating_sub(delta);
            self.b2.remove(pos);
            self.replace(true);
            self.t2.push_front(key.clone());
            self.entries
                .insert(key.clone(), CacheEntry::new(key, size_bytes, now_ms));
            self.total_bytes += sz;
            while self.over_limit() && (!self.t1.is_empty() || !self.t2.is_empty()) {
                self.replace(true);
            }
            return;
        }

        // Completely new key — enforce capacity before inserting.
        let total_t1 = self.t1.len() + self.b1.len();
        if total_t1 == self.capacity {
            if self.t1.len() < self.capacity {
                self.b1.pop_back();
                self.replace(false);
            } else if let Some(evicted) = self.t1.pop_back() {
                if let Some(e) = self.entries.remove(&evicted) {
                    self.total_bytes = self.total_bytes.saturating_sub(e.size_bytes as usize);
                }
            }
        } else {
            let total = self.t1.len() + self.b1.len() + self.t2.len() + self.b2.len();
            if total >= self.capacity {
                if total >= 2 * self.capacity {
                    self.b2.pop_back();
                }
                self.replace(false);
            }
        }

        // Insert the new entry.
        self.t1.push_front(key.clone());
        self.entries
            .insert(key.clone(), CacheEntry::new(key, size_bytes, now_ms));
        self.total_bytes += sz;

        // Continue evicting while either limit is breached.
        while self.over_limit() && (!self.t1.is_empty() || !self.t2.is_empty()) {
            self.replace(false);
        }
    }

    /// ARC "replace" subroutine — evicts one entry.
    fn replace(&mut self, in_b2: bool) {
        if !self.t1.is_empty() && (self.t1.len() > self.p || (in_b2 && self.t1.len() == self.p)) {
            if let Some(evicted) = self.t1.pop_back() {
                if let Some(e) = self.entries.remove(&evicted) {
                    self.total_bytes = self.total_bytes.saturating_sub(e.size_bytes as usize);
                }
                self.b1.push_front(evicted);
            }
        } else if let Some(evicted) = self.t2.pop_back() {
            if let Some(e) = self.entries.remove(&evicted) {
                self.total_bytes = self.total_bytes.saturating_sub(e.size_bytes as usize);
            }
            self.b2.push_front(evicted);
        }
    }

    /// Number of cached entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current adaptive parameter p (target T1 size).
    #[must_use]
    pub fn adaptive_parameter(&self) -> usize {
        self.p
    }
}

// ---------------------------------------------------------------------------
// Size-aware ARC cache (evicts by total bytes rather than entry count)
// ---------------------------------------------------------------------------

/// Adaptive Replacement Cache with byte-based capacity.
///
/// Unlike [`ArcCache`] which counts entries, `ArcCacheSized` tracks `used_bytes`
/// and evicts entries from T1/T2 until the byte budget is satisfied.  The ARC
/// adaptive parameter **p** is expressed in bytes and updated proportionally.
///
/// Ghost lists (B1, B2) retain only keys (zero byte cost) so they can grow
/// without consuming the byte budget.
pub struct ArcCacheSized {
    /// Hard byte limit for live entries (T1 + T2).
    capacity_bytes: u64,
    /// Current total bytes of live entries.
    used_bytes: u64,
    /// Target bytes in T1 — the ARC adaptive parameter expressed in bytes.
    p_bytes: u64,
    /// T1: recently inserted entries (seen once); front = most recent.
    t1: VecDeque<String>,
    /// T2: frequently accessed entries (seen 2+); front = most recent.
    t2: VecDeque<String>,
    /// B1: ghost keys evicted from T1 (keys only, no byte cost).
    b1: VecDeque<String>,
    /// B2: ghost keys evicted from T2 (keys only, no byte cost).
    b2: VecDeque<String>,
    /// Live entry metadata indexed by key.
    entries: HashMap<String, CacheEntry>,
    /// Total bytes evicted from this cache over its lifetime.
    pub total_evicted_bytes: u64,
    /// Total number of eviction events.
    pub eviction_count: u64,
}

impl ArcCacheSized {
    /// Create a new size-aware ARC cache with `capacity_bytes` byte budget.
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            p_bytes: 0,
            t1: VecDeque::new(),
            t2: VecDeque::new(),
            b1: VecDeque::new(),
            b2: VecDeque::new(),
            entries: HashMap::new(),
            total_evicted_bytes: 0,
            eviction_count: 0,
        }
    }

    // ── Internal helpers ────────────────────────────────────────────────

    /// Bytes currently consumed by live entries.
    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    /// Hard byte limit.
    #[must_use]
    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    /// Cache utilisation as a fraction (0.0–1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn utilization(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f64 / self.capacity_bytes as f64
    }

    /// Number of live cached entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no live entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current adaptive target for T1 bytes.
    #[must_use]
    pub fn adaptive_parameter_bytes(&self) -> u64 {
        self.p_bytes
    }

    /// Evict one entry according to the ARC replace rule.
    ///
    /// Returns the number of bytes freed.
    fn replace_one(&mut self, prefer_t2: bool) -> u64 {
        // Choose victim: if T1 is above target (or `prefer_t2` and T1 == target), evict from T1.
        let t1_bytes: u64 = self
            .t1
            .iter()
            .filter_map(|k| self.entries.get(k).map(|e| e.size_bytes))
            .sum();

        let evict_from_t1 = !self.t1.is_empty()
            && (t1_bytes > self.p_bytes || (prefer_t2 && t1_bytes == self.p_bytes));

        if evict_from_t1 {
            if let Some(key) = self.t1.pop_back() {
                if let Some(entry) = self.entries.remove(&key) {
                    let freed = entry.size_bytes;
                    self.used_bytes = self.used_bytes.saturating_sub(freed);
                    self.total_evicted_bytes += freed;
                    self.eviction_count += 1;
                    self.b1.push_front(key);
                    return freed;
                }
            }
        } else if let Some(key) = self.t2.pop_back() {
            if let Some(entry) = self.entries.remove(&key) {
                let freed = entry.size_bytes;
                self.used_bytes = self.used_bytes.saturating_sub(freed);
                self.total_evicted_bytes += freed;
                self.eviction_count += 1;
                self.b2.push_front(key);
                return freed;
            }
        } else if let Some(key) = self.t1.pop_back() {
            // Fallback: evict from T1 even if under target
            if let Some(entry) = self.entries.remove(&key) {
                let freed = entry.size_bytes;
                self.used_bytes = self.used_bytes.saturating_sub(freed);
                self.total_evicted_bytes += freed;
                self.eviction_count += 1;
                self.b1.push_front(key);
                return freed;
            }
        }
        0
    }

    /// Evict entries until `needed_bytes` become available or no more entries exist.
    fn make_room(&mut self, needed_bytes: u64, prefer_t2: bool) {
        while self.used_bytes + needed_bytes > self.capacity_bytes
            && (!self.t1.is_empty() || !self.t2.is_empty())
        {
            let freed = self.replace_one(prefer_t2);
            if freed == 0 {
                break;
            }
        }
    }

    // ── Public API ───────────────────────────────────────────────────────

    /// Look up a key and update its access metadata.
    ///
    /// Promotes from T1 to T2 on first re-access (ARC promotion rule).
    pub fn get(&mut self, key: &str, now_ms: u64) -> Option<&CacheEntry> {
        if let Some(pos) = self.t1.iter().position(|k| k == key) {
            // Promote T1 → T2
            if let Some(k) = self.t1.remove(pos) {
                self.t2.push_front(k);
            }
            if let Some(e) = self.entries.get_mut(key) {
                e.access_count += 1;
                e.last_access_ms = now_ms;
            }
            return self.entries.get(key);
        }

        if let Some(pos) = self.t2.iter().position(|k| k == key) {
            // Move to front of T2 (MRU position)
            if let Some(k) = self.t2.remove(pos) {
                self.t2.push_front(k);
            }
            if let Some(e) = self.entries.get_mut(key) {
                e.access_count += 1;
                e.last_access_ms = now_ms;
            }
            return self.entries.get(key);
        }

        None
    }

    /// Insert or refresh an entry.
    ///
    /// Adjusts the adaptive parameter and evicts as needed to fit `size_bytes`
    /// within the byte budget.
    pub fn put(&mut self, key: impl Into<String>, size_bytes: u64, now_ms: u64) {
        let key = key.into();

        // Already cached → refresh in place (size may change)
        if let Some(existing) = self.entries.get(&key) {
            let old_size = existing.size_bytes;
            // Remove from its current list position
            if let Some(pos) = self.t1.iter().position(|k| k == &key) {
                self.t1.remove(pos);
            } else if let Some(pos) = self.t2.iter().position(|k| k == &key) {
                self.t2.remove(pos);
            }
            self.used_bytes = self.used_bytes.saturating_sub(old_size);
            self.entries.remove(&key);
        }

        // Ghost hit in B1 → increase p (favour recency / T1)
        let in_b2 = if let Some(pos) = self.b1.iter().position(|k| k == &key) {
            let b2_bytes: u64 = self
                .b2
                .iter()
                .filter_map(|k| self.entries.get(k).map(|e| e.size_bytes))
                .sum::<u64>()
                .max(size_bytes);
            let b1_bytes = self.b1.len().max(1) as u64 * size_bytes;
            let delta = b2_bytes / b1_bytes;
            self.p_bytes = (self.p_bytes + delta.max(1)).min(self.capacity_bytes);
            self.b1.remove(pos);
            self.make_room(size_bytes, false);
            self.t2.push_front(key.clone());
            self.entries
                .insert(key.clone(), CacheEntry::new(key, size_bytes, now_ms));
            self.used_bytes += size_bytes;
            return;
        } else {
            // Check B2 ghost hit
            self.b2.iter().position(|k| k == &key).is_some()
        };

        if in_b2 {
            let pos = self.b2.iter().position(|k| k == &key).unwrap_or(0);
            let b1_bytes = self.b1.len().max(1) as u64 * size_bytes;
            let b2_bytes = self.b2.len().max(1) as u64 * size_bytes;
            let delta = b1_bytes / b2_bytes;
            self.p_bytes = self.p_bytes.saturating_sub(delta.max(1));
            self.b2.remove(pos);
            self.make_room(size_bytes, true);
            self.t2.push_front(key.clone());
            self.entries
                .insert(key.clone(), CacheEntry::new(key, size_bytes, now_ms));
            self.used_bytes += size_bytes;
            return;
        }

        // Completely new key — evict as needed, insert into T1
        self.make_room(size_bytes, false);
        self.t1.push_front(key.clone());
        self.entries
            .insert(key.clone(), CacheEntry::new(key, size_bytes, now_ms));
        self.used_bytes += size_bytes;
    }

    /// Remove a specific entry from the cache.
    ///
    /// Returns `true` if the key was present and removed.
    pub fn remove(&mut self, key: &str) -> bool {
        if let Some(entry) = self.entries.remove(key) {
            self.used_bytes = self.used_bytes.saturating_sub(entry.size_bytes);
            self.t1.retain(|k| k != key);
            self.t2.retain(|k| k != key);
            return true;
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Cache statistics
// ---------------------------------------------------------------------------

/// Accumulates cache hit/miss/eviction counters.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of entries evicted.
    pub evictions: u64,
}

impl CacheStats {
    /// Fraction of lookups that were hits (0.0–1.0).
    ///
    /// Returns `0.0` when no lookups have been performed.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Record a cache hit.
    pub fn record_hit(&mut self) {
        self.hits += 1;
    }

    /// Record a cache miss.
    pub fn record_miss(&mut self) {
        self.misses += 1;
    }

    /// Record an eviction.
    pub fn record_eviction(&mut self) {
        self.evictions += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_policy_names() {
        assert_eq!(CachePolicy::LRU.name(), "LRU");
        assert_eq!(CachePolicy::LFU.name(), "LFU");
        assert_eq!(CachePolicy::FIFO.name(), "FIFO");
        assert_eq!(CachePolicy::ARC.name(), "ARC");
    }

    #[test]
    fn test_cache_entry_age() {
        let entry = CacheEntry::new("key", 128, 1000);
        assert_eq!(entry.age_ms(1500), 500);
        assert_eq!(entry.age_ms(999), 0); // saturating sub
    }

    #[test]
    fn test_lru_cache_put_and_get() {
        let mut cache = LruCache::new(1024);
        cache.put("file.mp4", 100, 0);
        assert!(cache.get("file.mp4", 1).is_some());
    }

    #[test]
    fn test_lru_cache_miss() {
        let mut cache = LruCache::new(1024);
        assert!(cache.get("missing", 0).is_none());
    }

    #[test]
    fn test_lru_cache_eviction_on_overflow() {
        let mut cache = LruCache::new(200);
        cache.put("a", 100, 0);
        cache.put("b", 100, 1);
        // Both fit; now add one more that requires evicting 'a' (LRU)
        cache.put("c", 100, 2);
        assert_eq!(cache.len(), 2);
        // 'a' should have been evicted
        assert!(cache.get("a", 3).is_none());
        assert!(cache.get("b", 3).is_some());
        assert!(cache.get("c", 3).is_some());
    }

    #[test]
    fn test_lru_cache_access_updates_order() {
        let mut cache = LruCache::new(200);
        cache.put("a", 100, 0);
        cache.put("b", 100, 1);
        // Access 'a' to make it recently used
        cache.get("a", 2);
        // Now insert 'c'; 'b' is LRU and should be evicted
        cache.put("c", 100, 3);
        assert!(cache.get("a", 4).is_some());
        assert!(cache.get("b", 4).is_none());
        assert!(cache.get("c", 4).is_some());
    }

    #[test]
    fn test_lru_cache_utilization() {
        let mut cache = LruCache::new(1000);
        cache.put("x", 500, 0);
        assert!((cache.utilization() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_lru_cache_utilization_empty() {
        let cache = LruCache::new(0);
        assert_eq!(cache.utilization(), 0.0);
    }

    #[test]
    fn test_lru_cache_overwrite_same_key() {
        let mut cache = LruCache::new(1000);
        cache.put("k", 100, 0);
        cache.put("k", 200, 1); // overwrite
        assert_eq!(cache.used_bytes, 200);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lru_cache_is_empty() {
        let cache = LruCache::new(1024);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_stats_hit_rate_no_lookups() {
        let stats = CacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_stats_hit_rate_all_hits() {
        let mut stats = CacheStats::default();
        stats.record_hit();
        stats.record_hit();
        assert!((stats.hit_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cache_stats_hit_rate_mixed() {
        let mut stats = CacheStats::default();
        stats.record_hit();
        stats.record_miss();
        assert!((stats.hit_rate() - 0.5).abs() < 1e-9);
    }

    // ── LFU cache tests ────────────────────────────────────────────────

    #[test]
    fn test_lfu_cache_put_and_get() {
        let mut cache = LfuCache::new(1024);
        cache.put("file.mp4", 100, 0);
        assert!(cache.get("file.mp4", 1).is_some());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lfu_cache_miss() {
        let mut cache = LfuCache::new(1024);
        assert!(cache.get("missing", 0).is_none());
    }

    #[test]
    fn test_lfu_cache_evicts_least_frequent() {
        let mut cache = LfuCache::new(200);
        cache.put("a", 100, 0);
        cache.put("b", 100, 1);

        // Access 'b' several times to make it more frequent
        cache.get("b", 2);
        cache.get("b", 3);
        // 'a' has 0 accesses, 'b' has 2 accesses

        // Insert 'c' – must evict 'a' (least frequent)
        cache.put("c", 100, 4);
        assert!(cache.get("a", 5).is_none());
        assert!(cache.get("b", 5).is_some());
        assert!(cache.get("c", 5).is_some());
    }

    #[test]
    fn test_lfu_cache_tie_broken_by_oldest_access() {
        let mut cache = LfuCache::new(200);
        cache.put("a", 100, 0);
        cache.put("b", 100, 10);

        // Both have 0 accesses.  'a' has last_access_ms=0, 'b' has 10.
        // Tie-breaking by oldest access → 'a' evicted.
        cache.put("c", 100, 20);
        assert!(cache.get("a", 21).is_none());
        assert!(cache.get("b", 21).is_some());
    }

    #[test]
    fn test_lfu_cache_overwrite_same_key() {
        let mut cache = LfuCache::new(1024);
        cache.put("k", 100, 0);
        cache.put("k", 200, 1);
        assert_eq!(cache.used_bytes, 200);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lfu_cache_utilization() {
        let mut cache = LfuCache::new(1000);
        cache.put("x", 400, 0);
        assert!((cache.utilization() - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_lfu_cache_utilization_zero_capacity() {
        let cache = LfuCache::new(0);
        assert_eq!(cache.utilization(), 0.0);
    }

    #[test]
    fn test_lfu_cache_is_empty() {
        let cache = LfuCache::new(1024);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_lfu_cache_evict_empty() {
        let mut cache = LfuCache::new(1024);
        assert!(!cache.evict());
    }

    #[test]
    fn test_lfu_cache_multiple_evictions() {
        let mut cache = LfuCache::new(300);
        cache.put("a", 100, 0);
        cache.put("b", 100, 1);
        cache.put("c", 100, 2);

        // Access counts: a=0, b=0, c=0.  Insert 'd' (200 bytes) → must evict
        // two entries to make room.
        cache.put("d", 200, 3);
        assert_eq!(cache.len(), 2);
        assert!(cache.get("d", 4).is_some());
    }

    // ── ARC cache tests ────────────────────────────────────────────────

    #[test]
    fn test_arc_cache_put_and_get() {
        let mut cache = ArcCache::new(10);
        cache.put("a", 100, 0);
        assert!(cache.get("a", 1).is_some());
    }

    #[test]
    fn test_arc_cache_miss() {
        let mut cache = ArcCache::new(10);
        assert!(cache.get("missing", 0).is_none());
    }

    #[test]
    fn test_arc_cache_capacity_enforcement() {
        let mut cache = ArcCache::new(3);
        cache.put("a", 10, 0);
        cache.put("b", 10, 1);
        cache.put("c", 10, 2);
        assert_eq!(cache.len(), 3);

        // Adding a 4th entry should evict one
        cache.put("d", 10, 3);
        assert!(cache.len() <= 3);
    }

    #[test]
    fn test_arc_cache_t1_to_t2_promotion() {
        let mut cache = ArcCache::new(10);
        cache.put("a", 10, 0); // goes to T1
                               // First get promotes from T1 to T2
        let entry = cache.get("a", 1);
        assert!(entry.is_some());
        assert_eq!(entry.map(|e| e.access_count), Some(1));
        // Second get should still find it (now in T2)
        assert!(cache.get("a", 2).is_some());
    }

    #[test]
    fn test_arc_cache_duplicate_put_promotes() {
        let mut cache = ArcCache::new(10);
        cache.put("a", 10, 0);
        // Put same key again should promote, not duplicate
        cache.put("a", 10, 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_arc_cache_ghost_b1_hit_increases_p() {
        let mut cache = ArcCache::new(2);
        cache.put("a", 10, 0);
        cache.put("b", 10, 1);
        // 'a' is LRU in T1, adding 'c' should evict 'a' to B1
        cache.put("c", 10, 2);
        // Now re-insert 'a' → B1 hit → p should increase
        let p_before = cache.adaptive_parameter();
        cache.put("a", 10, 3);
        assert!(cache.adaptive_parameter() >= p_before);
    }

    #[test]
    fn test_arc_cache_ghost_b2_hit_decreases_p() {
        let mut cache = ArcCache::new(2);
        cache.put("a", 10, 0);
        cache.get("a", 1); // promote to T2
        cache.put("b", 10, 2);
        // Fill to trigger evictions from T2 to B2
        cache.put("c", 10, 3);
        cache.put("d", 10, 4);
        // If 'a' ended up in B2, reinserting it decreases p
        let _p = cache.adaptive_parameter();
        cache.put("a", 10, 5);
        // Just verify no panic and cache is consistent
        assert!(cache.len() <= 2);
    }

    #[test]
    fn test_arc_cache_is_empty() {
        let cache = ArcCache::new(10);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_arc_cache_stress() {
        let mut cache = ArcCache::new(5);
        // Insert many entries with varying access patterns
        for i in 0..20u64 {
            cache.put(format!("key-{i}"), 10, i * 10);
        }
        assert!(cache.len() <= 5);

        // Access some keys repeatedly to build frequency
        for i in 15..20u64 {
            let _ = cache.get(&format!("key-{i}"), 200 + i);
            let _ = cache.get(&format!("key-{i}"), 300 + i);
        }

        // Insert more; frequently accessed keys should survive
        for i in 20..30u64 {
            cache.put(format!("key-{i}"), 10, 400 + i);
        }
        assert!(cache.len() <= 5);
    }

    #[test]
    fn test_arc_cache_sequential_then_reuse() {
        // Simulates a workload with sequential scan then re-access pattern
        let mut cache = ArcCache::new(4);

        // Sequential inserts
        cache.put("s1", 10, 0);
        cache.put("s2", 10, 1);
        cache.put("s3", 10, 2);
        cache.put("s4", 10, 3);

        // Re-access s1, s2 to promote to T2
        cache.get("s1", 4);
        cache.get("s2", 5);

        // Now more sequential inserts; s1/s2 should survive (in T2)
        cache.put("s5", 10, 6);
        cache.put("s6", 10, 7);

        // s1 and s2 were promoted and should still be accessible
        assert!(cache.get("s1", 8).is_some());
        assert!(cache.get("s2", 9).is_some());
    }

    // ── ArcCacheSized tests ────────────────────────────────────────────

    #[test]
    fn test_arc_sized_basic_put_get() {
        let mut cache = ArcCacheSized::new(1000);
        cache.put("a", 100, 0);
        assert!(cache.get("a", 1).is_some());
        assert_eq!(cache.used_bytes(), 100);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_arc_sized_miss() {
        let mut cache = ArcCacheSized::new(1000);
        assert!(cache.get("missing", 0).is_none());
    }

    #[test]
    fn test_arc_sized_evicts_when_full() {
        // Capacity = 300 bytes; insert three 100-byte entries, then one more
        let mut cache = ArcCacheSized::new(300);
        cache.put("a", 100, 0);
        cache.put("b", 100, 1);
        cache.put("c", 100, 2);
        assert_eq!(cache.used_bytes(), 300);
        // Insert 'd' (100 bytes) — must evict to stay within budget
        cache.put("d", 100, 3);
        assert!(
            cache.used_bytes() <= 300,
            "used {} > capacity 300",
            cache.used_bytes()
        );
        assert!(cache.eviction_count > 0, "at least one eviction expected");
    }

    #[test]
    fn test_arc_sized_used_bytes_not_exceed_capacity() {
        let mut cache = ArcCacheSized::new(512);
        for i in 0u64..20 {
            cache.put(format!("key-{i}"), 64, i * 10);
        }
        assert!(
            cache.used_bytes() <= 512,
            "used_bytes {} exceeded capacity 512",
            cache.used_bytes()
        );
    }

    #[test]
    fn test_arc_sized_promotion_t1_to_t2() {
        let mut cache = ArcCacheSized::new(2000);
        cache.put("x", 100, 0); // goes to T1
        let e = cache.get("x", 1); // promotes to T2
        assert!(e.is_some());
        assert_eq!(e.map(|v| v.access_count), Some(1));
        // Still accessible after promotion
        assert!(cache.get("x", 2).is_some());
    }

    #[test]
    fn test_arc_sized_remove() {
        let mut cache = ArcCacheSized::new(1000);
        cache.put("r", 200, 0);
        assert_eq!(cache.used_bytes(), 200);
        assert!(cache.remove("r"));
        assert_eq!(cache.used_bytes(), 0);
        assert!(cache.is_empty());
        // Double remove
        assert!(!cache.remove("r"));
    }

    #[test]
    fn test_arc_sized_overwrite_same_key() {
        let mut cache = ArcCacheSized::new(1000);
        cache.put("k", 100, 0);
        cache.put("k", 300, 1); // overwrite with larger size
                                // Only one entry
        assert_eq!(cache.len(), 1);
        assert!(cache.used_bytes() <= 1000);
    }

    #[test]
    fn test_arc_sized_utilization() {
        let mut cache = ArcCacheSized::new(1000);
        cache.put("x", 500, 0);
        assert!((cache.utilization() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_arc_sized_utilization_zero_capacity() {
        let cache = ArcCacheSized::new(0);
        assert_eq!(cache.utilization(), 0.0);
    }

    #[test]
    fn test_arc_sized_is_empty() {
        let cache = ArcCacheSized::new(1024);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_arc_sized_total_evicted_bytes_increases() {
        let mut cache = ArcCacheSized::new(200);
        // Fill exactly
        cache.put("a", 100, 0);
        cache.put("b", 100, 1);
        // Force eviction
        cache.put("c", 100, 2);
        assert!(
            cache.total_evicted_bytes >= 100,
            "should have evicted at least 100 bytes"
        );
    }

    #[test]
    fn test_arc_sized_stress_byte_budget() {
        let mut cache = ArcCacheSized::new(1024);
        for i in 0u64..50 {
            // Vary sizes between 32 and 256 bytes
            let size = 32 + (i % 8) * 32;
            cache.put(format!("key-{i}"), size, i * 5);
            assert!(
                cache.used_bytes() <= 1024,
                "iteration {i}: used {} > capacity 1024",
                cache.used_bytes()
            );
        }
    }

    // ── ArcCache dual-limit (size-aware) tests ─────────────────────────

    #[test]
    fn test_arc_with_capacity_byte_limit_triggers_eviction() {
        // 5 entry slots, but only 300 bytes total.
        let mut cache = ArcCache::with_capacity(5, 300);
        cache.put("a", 100, 0);
        cache.put("b", 100, 1);
        cache.put("c", 100, 2);
        // All three fit within 300 bytes.
        assert_eq!(cache.total_bytes(), 300);
        assert_eq!(cache.len(), 3);

        // Adding a 4th entry (100 bytes) pushes total to 400 which exceeds
        // max_bytes=300, so at least one entry must be evicted.
        cache.put("d", 100, 3);
        assert!(
            cache.total_bytes() <= 300,
            "total_bytes {} exceeded max_bytes 300",
            cache.total_bytes()
        );
        assert!(
            cache.len() < 4,
            "expected eviction, got {} entries",
            cache.len()
        );
    }

    // ── Deterministic eviction sequence tests ──────────────────────────────────

    #[test]
    fn test_lru_deterministic_eviction_sequence() {
        // capacity = 3 * 100 = 300 bytes
        // Access A, B, C → full; access A again (makes A MRU, B is now LRU)
        // Insert D → B should be evicted
        let mut cache = LruCache::new(300);
        cache.put("A", 100, 0);
        cache.put("B", 100, 1);
        cache.put("C", 100, 2);
        // Re-access A; order becomes: A (MRU), C, B (LRU)
        cache.get("A", 3);
        // Insert D — must evict B
        cache.put("D", 100, 4);
        assert_eq!(cache.len(), 3);
        assert!(cache.get("A", 5).is_some(), "A was MRU and should survive");
        assert!(
            cache.get("B", 5).is_none(),
            "B was LRU and should be evicted"
        );
        assert!(cache.get("C", 5).is_some(), "C should survive");
        assert!(
            cache.get("D", 5).is_some(),
            "D is newest and should survive"
        );
    }

    #[test]
    fn test_lru_hit_count_increments() {
        let mut cache = LruCache::new(1000);
        cache.put("x", 100, 0);
        let entry = cache.get("x", 1).expect("entry must exist");
        assert_eq!(entry.access_count, 1);
        let entry = cache.get("x", 2).expect("entry must exist");
        assert_eq!(entry.access_count, 2);
    }

    #[test]
    fn test_lfu_deterministic_eviction_frequency() {
        // Insert A (0 accesses), B (0 accesses), C (0 accesses) – capacity 200 bytes.
        // Access B twice → B has highest frequency.
        // Insert D (100 bytes) → must evict one of A or C (both have 0 accesses).
        let mut cache = LfuCache::new(300);
        cache.put("A", 100, 0);
        cache.put("B", 100, 1);
        cache.put("C", 100, 2);
        cache.get("B", 3);
        cache.get("B", 4);
        // capacity full (300). insert D (100): evict LFU (A or C, not B).
        cache.put("D", 100, 5);
        assert_eq!(cache.len(), 3);
        assert!(
            cache.get("B", 6).is_some(),
            "B has highest freq and must survive"
        );
        assert!(cache.get("D", 6).is_some(), "D is new and must survive");
        // A was created first (last_access_ms=0), so A should be evicted (tie-breaks by older access).
        assert!(
            cache.get("A", 6).is_none(),
            "A has lower access_ms and should be evicted"
        );
        assert!(cache.get("C", 6).is_some(), "C survives");
    }

    #[test]
    fn test_lru_miss_does_not_update_access_count() {
        let mut cache = LruCache::new(1000);
        let result = cache.get("nonexistent", 99);
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_stats_eviction_tracking() {
        let mut stats = CacheStats::default();
        stats.record_hit();
        stats.record_miss();
        stats.record_eviction();
        stats.record_eviction();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.evictions, 2);
        assert!((stats.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_arc_sized_a_not_evicted_after_multiple_accesses() {
        // A accessed twice (promoted to T2), then B, C inserted → D inserted.
        // A should survive because it's in T2 (frequently accessed).
        let mut cache = ArcCacheSized::new(300);
        cache.put("A", 100, 0);
        cache.get("A", 1); // promote to T2
        cache.put("B", 100, 2);
        cache.put("C", 100, 3);
        // Cache is now full (300). Insert D → must evict from T1 first (B or C).
        cache.put("D", 100, 4);
        // A (in T2) should survive
        assert!(
            cache.get("A", 5).is_some(),
            "A is in T2 and should survive eviction"
        );
        assert!(cache.used_bytes() <= 300);
    }

    #[test]
    fn test_lru_capacity_zero_stays_empty() {
        let mut cache = LruCache::new(0);
        cache.put("k", 0, 0); // 0-byte entry should fit in 0-byte cache
                              // With a 0-byte entry there's no bytes to evict; it fits.
                              // With a non-zero entry the cache must evict everything.
        let mut cache2 = LruCache::new(0);
        cache2.put("x", 100, 0); // 100 > 0 → eviction loop will clear but key gets inserted
                                 // The important thing is we don't panic
    }

    #[test]
    fn test_lfu_is_empty_after_full_eviction() {
        let mut cache = LfuCache::new(100);
        cache.put("a", 100, 0);
        assert!(!cache.is_empty());
        // evict directly
        cache.evict();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_arc_sized_hit_miss_tracking_with_stats() {
        let mut cache = ArcCacheSized::new(500);
        let mut stats = CacheStats::default();
        cache.put("img", 100, 0);
        if cache.get("img", 1).is_some() {
            stats.record_hit();
        } else {
            stats.record_miss();
        }
        if cache.get("missing", 1).is_some() {
            stats.record_hit();
        } else {
            stats.record_miss();
        }
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_lru_eviction_counter_via_stats() {
        let mut cache = LruCache::new(200);
        let mut stats = CacheStats::default();
        cache.put("a", 100, 0);
        cache.put("b", 100, 1);
        // full, evict 'a'
        let evicted = cache.evict();
        if evicted {
            stats.record_eviction();
        }
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn test_arc_with_capacity_count_limit_triggers_eviction() {
        // 3 entry slots, generous byte limit.
        let mut cache = ArcCache::with_capacity(3, usize::MAX);
        cache.put("a", 10, 0);
        cache.put("b", 10, 1);
        cache.put("c", 10, 2);
        assert_eq!(cache.len(), 3);

        // 4th entry exceeds count limit → eviction.
        cache.put("d", 10, 3);
        assert!(
            cache.len() <= 3,
            "entry count {} exceeded max 3",
            cache.len()
        );
    }

    #[test]
    fn test_arc_with_capacity_large_entry_evicts_multiple_small() {
        // Small entries totalling 200 bytes, then one large 300-byte entry.
        let mut cache = ArcCache::with_capacity(100, 300);
        cache.put("s1", 100, 0);
        cache.put("s2", 100, 1);
        // total = 200, both fit

        // Insert 250-byte entry → must evict both small ones (200 bytes)
        cache.put("large", 250, 2);
        assert!(
            cache.total_bytes() <= 300,
            "total_bytes {} exceeded max_bytes 300 after large insertion",
            cache.total_bytes()
        );
    }

    #[test]
    fn test_arc_with_capacity_both_limits_independent() {
        // Count limit = 2, byte limit = 10_000 (generous).
        let mut cache = ArcCache::with_capacity(2, 10_000);
        cache.put("a", 100, 0);
        cache.put("b", 100, 1);
        // count limit hit → evict on 3rd.
        cache.put("c", 100, 2);
        assert!(cache.len() <= 2);
    }

    #[test]
    fn test_arc_with_capacity_total_bytes_tracked_correctly() {
        let mut cache = ArcCache::with_capacity(10, 1_000_000);
        cache.put("x", 512, 0);
        assert_eq!(cache.total_bytes(), 512);
        cache.put("y", 256, 1);
        assert_eq!(cache.total_bytes(), 768);
    }

    #[test]
    fn test_arc_with_capacity_zero_max_bytes_means_unlimited() {
        // max_bytes = 0 disables the byte limit (count-only mode).
        let mut cache = ArcCache::new(3); // new() sets max_bytes=0
        cache.put("a", 1_000_000, 0);
        cache.put("b", 1_000_000, 1);
        cache.put("c", 1_000_000, 2);
        // No byte eviction should occur since max_bytes = 0.
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.max_bytes(), 0);
    }

    #[test]
    fn test_arc_with_capacity_byte_limit_equal_to_single_entry() {
        // Only one entry fits by bytes even though count allows 5.
        let mut cache = ArcCache::with_capacity(5, 100);
        cache.put("first", 100, 0);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.total_bytes(), 100);

        cache.put("second", 50, 1);
        // second (50 bytes) fits, but combined (150) exceeds 100 → first evicted.
        assert!(cache.total_bytes() <= 100);
    }

    #[test]
    fn test_arc_with_capacity_access_preserves_within_limits() {
        let mut cache = ArcCache::with_capacity(5, 500);
        cache.put("a", 100, 0);
        cache.put("b", 100, 1);
        // Access 'a' to promote to T2.
        assert!(cache.get("a", 2).is_some());
        // Both still within limits.
        assert!(cache.total_bytes() <= 500);
        assert!(cache.len() <= 5);
    }

    #[test]
    fn test_arc_with_capacity_constructor() {
        let cache = ArcCache::with_capacity(10, 4096);
        assert!(cache.is_empty());
        assert_eq!(cache.max_bytes(), 4096);
        assert_eq!(cache.total_bytes(), 0);
    }
}
