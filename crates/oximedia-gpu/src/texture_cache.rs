//! 2D tile-based texture cache simulation.
//!
//! Models the texture caching subsystem present on GPU hardware — a fixed-size
//! cache organised in 2D tiles.  Each cache line holds a rectangular tile of
//! texels.  The cache tracks:
//!
//! * **Hit/miss counts** per request.
//! * **LRU eviction** — the least-recently-used tile is evicted when the cache
//!   is full and a new tile must be fetched.
//! * **Prefetch hints** — callers can warm the cache for anticipated accesses
//!   without incrementing the hit counter.
//!
//! All structures are CPU-side simulations; no actual GPU memory is allocated.

use std::collections::{BTreeMap, VecDeque};
use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors returned by texture cache operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum CacheError {
    /// The requested texel coordinate is outside the texture bounds.
    #[error("Texel coordinate ({x}, {y}) out of bounds ({width}x{height})")]
    OutOfBounds {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    /// The cache capacity is zero, which is not permitted.
    #[error("Cache capacity must be at least 1")]
    ZeroCapacity,
    /// The tile size is zero.
    #[error("Tile size must be at least 1")]
    ZeroTileSize,
    /// Texture dimensions are zero.
    #[error("Texture dimensions must be at least 1×1")]
    ZeroDimensions,
}

// ─── TileKey ─────────────────────────────────────────────────────────────────

/// Identifies a 2D tile by its tile-grid coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileKey {
    /// Column index in the tile grid.
    pub tile_x: u32,
    /// Row index in the tile grid.
    pub tile_y: u32,
}

impl TileKey {
    /// Construct a `TileKey` from texel coordinates and tile size.
    #[must_use]
    pub fn from_texel(x: u32, y: u32, tile_size: u32) -> Self {
        Self {
            tile_x: x / tile_size,
            tile_y: y / tile_size,
        }
    }
}

// ─── CacheStats ───────────────────────────────────────────────────────────────

/// Aggregate cache performance statistics.
#[derive(Debug, Clone, Default)]
pub struct TexCacheStats {
    /// Number of cache hits (tile found in cache).
    pub hits: u64,
    /// Number of cache misses (tile not in cache; fetched from main memory).
    pub misses: u64,
    /// Number of cache lines written by prefetch operations.
    pub prefetches: u64,
    /// Number of tiles evicted to make room for new tiles.
    pub evictions: u64,
}

impl TexCacheStats {
    /// Hit rate: `hits / (hits + misses)`, or 0.0 if no accesses have occurred.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Total number of texel accesses (hits + misses).
    #[must_use]
    pub fn total_accesses(&self) -> u64 {
        self.hits + self.misses
    }
}

// ─── TextureCache ─────────────────────────────────────────────────────────────

/// 2D tile-based LRU texture cache.
///
/// The simulated texture has `width × height` texels divided into
/// `tile_size × tile_size` tiles.  At most `capacity` tiles reside in the
/// cache simultaneously.
///
/// # Cache line model
///
/// Each cache "line" corresponds to one complete tile.  On a miss the whole
/// tile is loaded from simulated main memory.  The tile's data is a flat
/// `Vec<u8>` with `tile_size * tile_size` bytes (one byte per texel).
pub struct TextureCache {
    /// Width of the texture in texels.
    pub width: u32,
    /// Height of the texture in texels.
    pub height: u32,
    /// Edge length (in texels) of each square tile / cache line.
    pub tile_size: u32,
    /// Maximum number of tiles the cache can hold.
    capacity: usize,
    /// Cached tiles: key → tile data.
    lines: BTreeMap<TileKey, Vec<u8>>,
    /// LRU order: front = most recent, back = oldest.
    lru: VecDeque<TileKey>,
    /// Aggregate statistics.
    stats: TexCacheStats,
    /// Simulated "main memory" — the backing texture data.
    texture_data: Vec<u8>,
}

impl TextureCache {
    /// Create a new texture cache backed by a flat texture buffer.
    ///
    /// `texture_data` must have exactly `width * height` bytes.  If it is
    /// shorter it is padded with zeros; if longer, the excess is ignored.
    ///
    /// # Errors
    ///
    /// Returns an error if `width`, `height`, `tile_size`, or `capacity` is 0.
    pub fn new(
        width: u32,
        height: u32,
        tile_size: u32,
        capacity: usize,
        texture_data: Vec<u8>,
    ) -> Result<Self, CacheError> {
        if width == 0 || height == 0 {
            return Err(CacheError::ZeroDimensions);
        }
        if tile_size == 0 {
            return Err(CacheError::ZeroTileSize);
        }
        if capacity == 0 {
            return Err(CacheError::ZeroCapacity);
        }
        let total = (width as usize) * (height as usize);
        let mut padded = texture_data;
        padded.resize(total, 0);

        Ok(Self {
            width,
            height,
            tile_size,
            capacity,
            lines: BTreeMap::new(),
            lru: VecDeque::new(),
            stats: TexCacheStats::default(),
            texture_data: padded,
        })
    }

    /// Fetch the texel value at `(x, y)`.
    ///
    /// If the tile containing `(x, y)` is not in the cache, it is loaded (a
    /// *miss*); otherwise the cached copy is used (a *hit*).
    ///
    /// # Errors
    ///
    /// Returns [`CacheError::OutOfBounds`] if `x >= width` or `y >= height`.
    pub fn fetch(&mut self, x: u32, y: u32) -> Result<u8, CacheError> {
        self.check_bounds(x, y)?;
        let key = TileKey::from_texel(x, y, self.tile_size);
        if self.lines.contains_key(&key) {
            self.stats.hits += 1;
            self.promote_lru(key);
        } else {
            self.stats.misses += 1;
            self.load_tile(key);
        }
        let tile = &self.lines[&key];
        let local_x = (x % self.tile_size) as usize;
        let local_y = (y % self.tile_size) as usize;
        Ok(tile[local_y * self.tile_size as usize + local_x])
    }

    /// Prefetch the tile containing `(x, y)` into the cache.
    ///
    /// If the tile is already cached, this is a no-op (the hit counter is *not*
    /// incremented).  If it is not cached, it is loaded and the prefetch counter
    /// is incremented.
    ///
    /// # Errors
    ///
    /// Returns [`CacheError::OutOfBounds`] if `x >= width` or `y >= height`.
    pub fn prefetch(&mut self, x: u32, y: u32) -> Result<(), CacheError> {
        self.check_bounds(x, y)?;
        let key = TileKey::from_texel(x, y, self.tile_size);
        if !self.lines.contains_key(&key) {
            self.load_tile(key);
            self.stats.prefetches += 1;
        }
        Ok(())
    }

    /// Invalidate (evict) all tiles from the cache.
    ///
    /// Statistics are preserved; only the tile storage is cleared.
    pub fn flush(&mut self) {
        self.lines.clear();
        self.lru.clear();
    }

    /// Invalidate a specific tile if present.
    ///
    /// Returns `true` if the tile was in the cache and was removed.
    pub fn invalidate_tile(&mut self, key: TileKey) -> bool {
        if self.lines.remove(&key).is_some() {
            self.lru.retain(|k| k != &key);
            true
        } else {
            false
        }
    }

    /// Number of tiles currently resident in the cache.
    #[must_use]
    pub fn resident_count(&self) -> usize {
        self.lines.len()
    }

    /// Whether a specific tile is currently resident in the cache.
    #[must_use]
    pub fn is_resident(&self, key: TileKey) -> bool {
        self.lines.contains_key(&key)
    }

    /// A reference to the current aggregate statistics.
    #[must_use]
    pub fn stats(&self) -> &TexCacheStats {
        &self.stats
    }

    /// Reset all statistics counters to zero.
    pub fn reset_stats(&mut self) {
        self.stats = TexCacheStats::default();
    }

    /// Maximum number of tiles the cache can hold.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of tiles in each dimension of the tile grid.
    #[must_use]
    pub fn grid_size(&self) -> (u32, u32) {
        let tiles_x = (self.width + self.tile_size - 1) / self.tile_size;
        let tiles_y = (self.height + self.tile_size - 1) / self.tile_size;
        (tiles_x, tiles_y)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn check_bounds(&self, x: u32, y: u32) -> Result<(), CacheError> {
        if x >= self.width || y >= self.height {
            Err(CacheError::OutOfBounds {
                x,
                y,
                width: self.width,
                height: self.height,
            })
        } else {
            Ok(())
        }
    }

    /// Load a tile from simulated main memory into the cache, evicting if needed.
    fn load_tile(&mut self, key: TileKey) {
        // Evict LRU tile if at capacity.
        if self.lines.len() >= self.capacity {
            if let Some(victim) = self.lru.pop_back() {
                self.lines.remove(&victim);
                self.stats.evictions += 1;
            }
        }
        // Build the tile data from the backing texture.
        let ts = self.tile_size as usize;
        let w = self.width as usize;
        let tile_x0 = key.tile_x as usize * ts;
        let tile_y0 = key.tile_y as usize * ts;
        let mut tile = vec![0u8; ts * ts];
        for local_y in 0..ts {
            let global_y = tile_y0 + local_y;
            if global_y >= self.height as usize {
                break;
            }
            for local_x in 0..ts {
                let global_x = tile_x0 + local_x;
                if global_x >= self.width as usize {
                    break;
                }
                tile[local_y * ts + local_x] = self.texture_data[global_y * w + global_x];
            }
        }
        self.lines.insert(key, tile);
        self.lru.push_front(key);
    }

    /// Move an existing cache line to the front of the LRU queue.
    fn promote_lru(&mut self, key: TileKey) {
        self.lru.retain(|k| k != &key);
        self.lru.push_front(key);
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple 16×16 texture where texel (x,y) = (y*16 + x) as u8.
    fn make_texture(width: u32, height: u32) -> Vec<u8> {
        (0..(width * height)).map(|i| (i % 256) as u8).collect()
    }

    fn make_cache(capacity: usize, tile_size: u32) -> TextureCache {
        let data = make_texture(16, 16);
        TextureCache::new(16, 16, tile_size, capacity, data).unwrap()
    }

    // ── TileKey ───────────────────────────────────────────────────────────────

    #[test]
    fn test_tile_key_from_texel() {
        let key = TileKey::from_texel(9, 5, 4);
        assert_eq!(key.tile_x, 2); // 9/4=2
        assert_eq!(key.tile_y, 1); // 5/4=1
    }

    #[test]
    fn test_tile_key_origin() {
        let key = TileKey::from_texel(0, 0, 8);
        assert_eq!(key.tile_x, 0);
        assert_eq!(key.tile_y, 0);
    }

    // ── TexCacheStats ─────────────────────────────────────────────────────────

    #[test]
    fn test_stats_hit_rate_zero_accesses() {
        let s = TexCacheStats::default();
        assert_eq!(s.hit_rate(), 0.0);
    }

    #[test]
    fn test_stats_hit_rate_all_hits() {
        let s = TexCacheStats {
            hits: 10,
            misses: 0,
            ..Default::default()
        };
        assert!((s.hit_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_total_accesses() {
        let s = TexCacheStats {
            hits: 3,
            misses: 7,
            ..Default::default()
        };
        assert_eq!(s.total_accesses(), 10);
    }

    // ── TextureCache construction ─────────────────────────────────────────────

    #[test]
    fn test_new_zero_capacity_error() {
        let err = TextureCache::new(8, 8, 4, 0, vec![]);
        assert!(matches!(err, Err(CacheError::ZeroCapacity)));
    }

    #[test]
    fn test_new_zero_tile_size_error() {
        let err = TextureCache::new(8, 8, 0, 4, vec![]);
        assert!(matches!(err, Err(CacheError::ZeroTileSize)));
    }

    #[test]
    fn test_new_zero_dimensions_error() {
        let err = TextureCache::new(0, 8, 4, 4, vec![]);
        assert!(matches!(err, Err(CacheError::ZeroDimensions)));
    }

    // ── fetch: hit / miss tracking ────────────────────────────────────────────

    #[test]
    fn test_fetch_first_access_is_miss() {
        let mut cache = make_cache(8, 4);
        let _ = cache.fetch(0, 0).unwrap();
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn test_fetch_second_access_same_tile_is_hit() {
        let mut cache = make_cache(8, 4);
        let _ = cache.fetch(0, 0).unwrap();
        let _ = cache.fetch(1, 1).unwrap(); // same tile (tile 0,0)
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_fetch_correct_texel_value() {
        let data = make_texture(16, 16);
        let expected_at_3_5 = data[5 * 16 + 3];
        let mut cache = TextureCache::new(16, 16, 4, 8, data).unwrap();
        let val = cache.fetch(3, 5).unwrap();
        assert_eq!(val, expected_at_3_5);
    }

    #[test]
    fn test_fetch_out_of_bounds() {
        let mut cache = make_cache(4, 4);
        let err = cache.fetch(16, 0);
        assert!(matches!(err, Err(CacheError::OutOfBounds { .. })));
    }

    // ── LRU eviction ─────────────────────────────────────────────────────────

    #[test]
    fn test_lru_eviction_when_full() {
        // 2-tile cache; access 3 different tiles → first tile should be evicted.
        let mut cache = make_cache(2, 4); // 16×16 / 4×4 = 16 tiles
        cache.fetch(0, 0).unwrap(); // tile (0,0) loaded
        cache.fetch(4, 0).unwrap(); // tile (1,0) loaded — cache full
                                    // Access a third tile → (0,0) should be evicted (LRU)
        cache.fetch(8, 0).unwrap(); // tile (2,0) loaded
        assert_eq!(cache.stats().evictions, 1);
        assert_eq!(cache.resident_count(), 2);
    }

    #[test]
    fn test_lru_promotion_prevents_eviction() {
        let mut cache = make_cache(2, 4);
        cache.fetch(0, 0).unwrap(); // (0,0) loaded → LRU = [(0,0)]
        cache.fetch(4, 0).unwrap(); // (1,0) loaded → LRU = [(1,0), (0,0)]
        cache.fetch(0, 0).unwrap(); // (0,0) hit → promoted → LRU = [(0,0),(1,0)]
                                    // Load new tile → (1,0) evicted, not (0,0)
        cache.fetch(8, 0).unwrap();
        let key_0_0 = TileKey {
            tile_x: 0,
            tile_y: 0,
        };
        assert!(
            cache.is_resident(key_0_0),
            "promoted tile should survive eviction"
        );
    }

    // ── prefetch ──────────────────────────────────────────────────────────────

    #[test]
    fn test_prefetch_loads_tile_without_hit_count() {
        let mut cache = make_cache(8, 4);
        cache.prefetch(0, 0).unwrap();
        assert_eq!(cache.stats().prefetches, 1);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
        // Subsequent fetch should be a hit.
        cache.fetch(0, 0).unwrap();
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_prefetch_already_resident_is_noop() {
        let mut cache = make_cache(8, 4);
        cache.fetch(0, 0).unwrap(); // miss → tile loaded
        cache.prefetch(0, 0).unwrap(); // already resident → no prefetch count
        assert_eq!(cache.stats().prefetches, 0);
    }

    // ── flush and invalidate ──────────────────────────────────────────────────

    #[test]
    fn test_flush_clears_cache() {
        let mut cache = make_cache(8, 4);
        cache.fetch(0, 0).unwrap();
        cache.fetch(4, 0).unwrap();
        assert_eq!(cache.resident_count(), 2);
        cache.flush();
        assert_eq!(cache.resident_count(), 0);
        // Stats should be preserved
        assert_eq!(cache.stats().misses, 2);
    }

    #[test]
    fn test_invalidate_tile() {
        let mut cache = make_cache(8, 4);
        cache.fetch(0, 0).unwrap();
        let key = TileKey {
            tile_x: 0,
            tile_y: 0,
        };
        assert!(cache.invalidate_tile(key));
        assert!(!cache.is_resident(key));
        // Second invalidation should return false.
        assert!(!cache.invalidate_tile(key));
    }

    // ── grid_size ─────────────────────────────────────────────────────────────

    #[test]
    fn test_grid_size_exact_division() {
        let data = vec![0u8; 64];
        let cache = TextureCache::new(8, 8, 4, 4, data).unwrap();
        assert_eq!(cache.grid_size(), (2, 2));
    }

    #[test]
    fn test_grid_size_non_exact() {
        let data = vec![0u8; 100];
        let cache = TextureCache::new(10, 10, 4, 4, data).unwrap();
        // ceil(10/4) = 3
        assert_eq!(cache.grid_size(), (3, 3));
    }
}
