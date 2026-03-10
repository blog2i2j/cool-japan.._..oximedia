//! Motion vector caching for video encoding.
//!
//! This module provides:
//! - MV cache for storing search results
//! - Reference frame MV storage
//! - Co-located MV lookup for temporal prediction
//!
//! Caching motion vectors improves encoding speed by:
//! - Providing better starting points for subsequent searches
//! - Enabling fast MV predictor derivation
//! - Supporting temporal prediction

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

use super::types::{BlockSize, MotionVector};

/// Cache entry for a motion vector.
#[derive(Clone, Copy, Debug, Default)]
pub struct MvCacheEntry {
    /// Motion vector.
    pub mv: MotionVector,
    /// Reference frame index.
    pub ref_idx: i8,
    /// SAD value.
    pub sad: u32,
    /// Is this entry valid?
    pub valid: bool,
    /// Is this an inter block?
    pub is_inter: bool,
}

impl MvCacheEntry {
    /// Creates an invalid entry.
    #[must_use]
    pub const fn invalid() -> Self {
        Self {
            mv: MotionVector::zero(),
            ref_idx: -1,
            sad: u32::MAX,
            valid: false,
            is_inter: false,
        }
    }

    /// Creates a valid inter entry.
    #[must_use]
    pub const fn inter(mv: MotionVector, ref_idx: i8, sad: u32) -> Self {
        Self {
            mv,
            ref_idx,
            sad,
            valid: true,
            is_inter: true,
        }
    }

    /// Creates an intra entry (no MV).
    #[must_use]
    pub const fn intra() -> Self {
        Self {
            mv: MotionVector::zero(),
            ref_idx: -1,
            sad: 0,
            valid: true,
            is_inter: false,
        }
    }
}

/// Motion vector cache for a frame.
///
/// Stores motion vectors in a grid aligned to 4x4 blocks (MI units).
#[derive(Clone, Debug)]
pub struct MvCache {
    /// Cache data.
    data: Vec<MvCacheEntry>,
    /// Width in MI units (4x4 blocks).
    mi_cols: usize,
    /// Height in MI units (4x4 blocks).
    mi_rows: usize,
    /// Number of reference frames supported.
    num_refs: usize,
}

impl Default for MvCache {
    fn default() -> Self {
        Self::new()
    }
}

impl MvCache {
    /// Creates a new empty cache.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            data: Vec::new(),
            mi_cols: 0,
            mi_rows: 0,
            num_refs: 3,
        }
    }

    /// Allocates cache for a frame.
    pub fn allocate(&mut self, width: usize, height: usize, num_refs: usize) {
        self.mi_cols = (width + 3) / 4;
        self.mi_rows = (height + 3) / 4;
        self.num_refs = num_refs;

        let size = self.mi_cols * self.mi_rows * num_refs;
        self.data = vec![MvCacheEntry::invalid(); size];
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.data.fill(MvCacheEntry::invalid());
    }

    /// Returns the width in MI units.
    #[must_use]
    pub const fn mi_cols(&self) -> usize {
        self.mi_cols
    }

    /// Returns the height in MI units.
    #[must_use]
    pub const fn mi_rows(&self) -> usize {
        self.mi_rows
    }

    /// Calculates the index for a position.
    fn index(&self, mi_row: usize, mi_col: usize, ref_idx: usize) -> Option<usize> {
        if mi_row >= self.mi_rows || mi_col >= self.mi_cols || ref_idx >= self.num_refs {
            return None;
        }
        Some((mi_row * self.mi_cols + mi_col) * self.num_refs + ref_idx)
    }

    /// Gets an entry.
    #[must_use]
    pub fn get(&self, mi_row: usize, mi_col: usize, ref_idx: usize) -> Option<&MvCacheEntry> {
        let idx = self.index(mi_row, mi_col, ref_idx)?;
        self.data.get(idx)
    }

    /// Gets a mutable entry.
    pub fn get_mut(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        ref_idx: usize,
    ) -> Option<&mut MvCacheEntry> {
        let idx = self.index(mi_row, mi_col, ref_idx)?;
        self.data.get_mut(idx)
    }

    /// Sets an entry.
    pub fn set(&mut self, mi_row: usize, mi_col: usize, ref_idx: usize, entry: MvCacheEntry) {
        if let Some(idx) = self.index(mi_row, mi_col, ref_idx) {
            if idx < self.data.len() {
                self.data[idx] = entry;
            }
        }
    }

    /// Fills a block region with an entry.
    pub fn fill_block(
        &mut self,
        mi_row: usize,
        mi_col: usize,
        block_size: BlockSize,
        ref_idx: usize,
        entry: MvCacheEntry,
    ) {
        let mi_width = block_size.width() / 4;
        let mi_height = block_size.height() / 4;

        for row in mi_row..mi_row + mi_height {
            for col in mi_col..mi_col + mi_width {
                self.set(row, col, ref_idx, entry);
            }
        }
    }

    /// Gets the left neighbor entry.
    #[must_use]
    pub fn get_left(&self, mi_row: usize, mi_col: usize, ref_idx: usize) -> Option<&MvCacheEntry> {
        if mi_col == 0 {
            return None;
        }
        self.get(mi_row, mi_col - 1, ref_idx)
    }

    /// Gets the top neighbor entry.
    #[must_use]
    pub fn get_top(&self, mi_row: usize, mi_col: usize, ref_idx: usize) -> Option<&MvCacheEntry> {
        if mi_row == 0 {
            return None;
        }
        self.get(mi_row - 1, mi_col, ref_idx)
    }

    /// Gets the top-right neighbor entry.
    #[must_use]
    pub fn get_top_right(
        &self,
        mi_row: usize,
        mi_col: usize,
        block_size: BlockSize,
        ref_idx: usize,
    ) -> Option<&MvCacheEntry> {
        if mi_row == 0 {
            return None;
        }
        let mi_width = block_size.width() / 4;
        let tr_col = mi_col + mi_width;
        if tr_col >= self.mi_cols {
            return None;
        }
        self.get(mi_row - 1, tr_col, ref_idx)
    }

    /// Gets the top-left neighbor entry.
    #[must_use]
    pub fn get_top_left(
        &self,
        mi_row: usize,
        mi_col: usize,
        ref_idx: usize,
    ) -> Option<&MvCacheEntry> {
        if mi_row == 0 || mi_col == 0 {
            return None;
        }
        self.get(mi_row - 1, mi_col - 1, ref_idx)
    }
}

/// Reference frame motion vector storage.
///
/// Stores MVs for decoded reference frames to enable temporal prediction.
#[derive(Clone, Debug)]
pub struct RefFrameMvs {
    /// MV data for each reference frame.
    frames: Vec<MvCache>,
    /// Maximum number of reference frames.
    max_refs: usize,
}

impl Default for RefFrameMvs {
    fn default() -> Self {
        Self::new()
    }
}

impl RefFrameMvs {
    /// Creates new reference frame MV storage.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            frames: Vec::new(),
            max_refs: 8,
        }
    }

    /// Sets the maximum number of references.
    #[must_use]
    pub fn with_max_refs(mut self, max: usize) -> Self {
        self.max_refs = max;
        self
    }

    /// Allocates storage for reference frames.
    pub fn allocate(&mut self, width: usize, height: usize, num_refs: usize) {
        self.frames.clear();
        for _ in 0..num_refs.min(self.max_refs) {
            let mut cache = MvCache::new();
            cache.allocate(width, height, 1);
            self.frames.push(cache);
        }
    }

    /// Gets the MV cache for a reference frame.
    #[must_use]
    pub fn get_frame(&self, frame_idx: usize) -> Option<&MvCache> {
        self.frames.get(frame_idx)
    }

    /// Gets mutable MV cache for a reference frame.
    pub fn get_frame_mut(&mut self, frame_idx: usize) -> Option<&mut MvCache> {
        self.frames.get_mut(frame_idx)
    }

    /// Stores MVs from current frame as reference.
    pub fn store_frame(&mut self, frame_idx: usize, source: &MvCache) {
        if frame_idx < self.frames.len() {
            self.frames[frame_idx] = source.clone();
        }
    }

    /// Gets co-located MV from reference frame.
    #[must_use]
    pub fn get_co_located(
        &self,
        frame_idx: usize,
        mi_row: usize,
        mi_col: usize,
    ) -> Option<MvCacheEntry> {
        let frame = self.frames.get(frame_idx)?;
        frame.get(mi_row, mi_col, 0).copied()
    }

    /// Shifts reference frames (for new frame insertion).
    pub fn shift_frames(&mut self) {
        if self.frames.len() > 1 {
            self.frames.rotate_right(1);
        }
    }
}

/// Co-located MV lookup helper.
#[derive(Clone, Debug, Default)]
pub struct CoLocatedMvLookup {
    /// Reference frame MVs.
    ref_mvs: RefFrameMvs,
    /// Temporal distance to each reference.
    temporal_distances: Vec<i32>,
}

impl CoLocatedMvLookup {
    /// Creates a new lookup helper.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ref_mvs: RefFrameMvs::new(),
            temporal_distances: Vec::new(),
        }
    }

    /// Allocates storage.
    pub fn allocate(&mut self, width: usize, height: usize, num_refs: usize) {
        self.ref_mvs.allocate(width, height, num_refs);
        self.temporal_distances = vec![1; num_refs];
    }

    /// Sets temporal distances.
    pub fn set_temporal_distances(&mut self, distances: &[i32]) {
        self.temporal_distances = distances.to_vec();
    }

    /// Gets co-located MV with temporal scaling.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn get_scaled_co_located(
        &self,
        frame_idx: usize,
        mi_row: usize,
        mi_col: usize,
        target_dist: i32,
    ) -> Option<MotionVector> {
        let entry = self.ref_mvs.get_co_located(frame_idx, mi_row, mi_col)?;

        if !entry.valid || !entry.is_inter {
            return None;
        }

        let src_dist = self.temporal_distances.get(frame_idx).copied().unwrap_or(1);

        if src_dist == target_dist || src_dist == 0 {
            return Some(entry.mv);
        }

        // Scale MV for different temporal distance
        let scale_x = (i64::from(entry.mv.dx) * i64::from(target_dist)) / i64::from(src_dist);
        let scale_y = (i64::from(entry.mv.dy) * i64::from(target_dist)) / i64::from(src_dist);

        Some(MotionVector::new(scale_x as i32, scale_y as i32))
    }

    /// Gets underlying reference MVs.
    #[must_use]
    pub fn ref_mvs(&self) -> &RefFrameMvs {
        &self.ref_mvs
    }

    /// Gets mutable reference MVs.
    pub fn ref_mvs_mut(&mut self) -> &mut RefFrameMvs {
        &mut self.ref_mvs
    }
}

/// Search result cache for avoiding redundant searches.
#[derive(Clone, Debug)]
pub struct SearchResultCache {
    /// Cached results.
    results: Vec<Option<(MotionVector, u32)>>,
    /// Width in blocks.
    width: usize,
    /// Height in blocks.
    height: usize,
    /// Block size for this cache.
    block_size: BlockSize,
}

impl Default for SearchResultCache {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchResultCache {
    /// Creates a new cache.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            results: Vec::new(),
            width: 0,
            height: 0,
            block_size: BlockSize::Block8x8,
        }
    }

    /// Allocates cache for a frame.
    pub fn allocate(&mut self, frame_width: usize, frame_height: usize, block_size: BlockSize) {
        self.block_size = block_size;
        self.width = (frame_width + block_size.width() - 1) / block_size.width();
        self.height = (frame_height + block_size.height() - 1) / block_size.height();
        self.results = vec![None; self.width * self.height];
    }

    /// Clears the cache.
    pub fn clear(&mut self) {
        self.results.fill(None);
    }

    /// Gets a cached result.
    #[must_use]
    pub fn get(&self, block_x: usize, block_y: usize) -> Option<(MotionVector, u32)> {
        let bx = block_x / self.block_size.width();
        let by = block_y / self.block_size.height();

        if bx >= self.width || by >= self.height {
            return None;
        }

        self.results[by * self.width + bx]
    }

    /// Stores a result.
    pub fn store(&mut self, block_x: usize, block_y: usize, mv: MotionVector, sad: u32) {
        let bx = block_x / self.block_size.width();
        let by = block_y / self.block_size.height();

        if bx < self.width && by < self.height {
            self.results[by * self.width + bx] = Some((mv, sad));
        }
    }

    /// Checks if a result is cached.
    #[must_use]
    pub fn has(&self, block_x: usize, block_y: usize) -> bool {
        self.get(block_x, block_y).is_some()
    }
}

/// Combined cache manager.
#[derive(Clone, Debug, Default)]
pub struct CacheManager {
    /// Current frame MV cache.
    pub current_frame: MvCache,
    /// Reference frame MVs.
    pub ref_frames: RefFrameMvs,
    /// Search result cache.
    pub search_cache: SearchResultCache,
    /// Co-located lookup.
    pub co_located: CoLocatedMvLookup,
}

impl CacheManager {
    /// Creates a new cache manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            current_frame: MvCache::new(),
            ref_frames: RefFrameMvs::new(),
            search_cache: SearchResultCache::new(),
            co_located: CoLocatedMvLookup::new(),
        }
    }

    /// Allocates all caches.
    pub fn allocate(&mut self, width: usize, height: usize, num_refs: usize) {
        self.current_frame.allocate(width, height, num_refs);
        self.ref_frames.allocate(width, height, num_refs);
        self.search_cache
            .allocate(width, height, BlockSize::Block8x8);
        self.co_located.allocate(width, height, num_refs);
    }

    /// Clears all caches for a new frame.
    pub fn new_frame(&mut self) {
        self.current_frame.clear();
        self.search_cache.clear();
    }

    /// Stores current frame as reference.
    pub fn finalize_frame(&mut self, frame_idx: usize) {
        self.ref_frames.store_frame(frame_idx, &self.current_frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mv_cache_entry_invalid() {
        let entry = MvCacheEntry::invalid();
        assert!(!entry.valid);
        assert!(!entry.is_inter);
    }

    #[test]
    fn test_mv_cache_entry_inter() {
        let mv = MotionVector::new(10, 20);
        let entry = MvCacheEntry::inter(mv, 0, 100);

        assert!(entry.valid);
        assert!(entry.is_inter);
        assert_eq!(entry.mv.dx, 10);
        assert_eq!(entry.sad, 100);
    }

    #[test]
    fn test_mv_cache_allocate() {
        let mut cache = MvCache::new();
        cache.allocate(64, 64, 3);

        assert_eq!(cache.mi_cols(), 16); // 64/4
        assert_eq!(cache.mi_rows(), 16);
    }

    #[test]
    fn test_mv_cache_get_set() {
        let mut cache = MvCache::new();
        cache.allocate(64, 64, 3);

        let entry = MvCacheEntry::inter(MotionVector::new(10, 20), 0, 100);
        cache.set(5, 5, 0, entry);

        let retrieved = cache.get(5, 5, 0).expect("get should return value");
        assert!(retrieved.valid);
        assert_eq!(retrieved.mv.dx, 10);
    }

    #[test]
    fn test_mv_cache_fill_block() {
        let mut cache = MvCache::new();
        cache.allocate(64, 64, 3);

        let entry = MvCacheEntry::inter(MotionVector::new(10, 20), 0, 100);
        cache.fill_block(0, 0, BlockSize::Block8x8, 0, entry);

        // 8x8 block = 2x2 MI units
        assert!(cache.get(0, 0, 0).expect("get should return value").valid);
        assert!(cache.get(0, 1, 0).expect("get should return value").valid);
        assert!(cache.get(1, 0, 0).expect("get should return value").valid);
        assert!(cache.get(1, 1, 0).expect("get should return value").valid);
    }

    #[test]
    fn test_mv_cache_neighbors() {
        let mut cache = MvCache::new();
        cache.allocate(64, 64, 3);

        // Set some entries
        let left = MvCacheEntry::inter(MotionVector::new(1, 1), 0, 10);
        let top = MvCacheEntry::inter(MotionVector::new(2, 2), 0, 20);

        cache.set(5, 4, 0, left);
        cache.set(4, 5, 0, top);

        let got_left = cache.get_left(5, 5, 0).expect("should succeed");
        assert_eq!(got_left.mv.dx, 1);

        let got_top = cache.get_top(5, 5, 0).expect("should succeed");
        assert_eq!(got_top.mv.dx, 2);
    }

    #[test]
    fn test_ref_frame_mvs() {
        let mut ref_mvs = RefFrameMvs::new();
        ref_mvs.allocate(64, 64, 3);

        // Store frame
        let mut cache = MvCache::new();
        cache.allocate(64, 64, 1);
        let entry = MvCacheEntry::inter(MotionVector::new(10, 20), 0, 100);
        cache.set(5, 5, 0, entry);

        ref_mvs.store_frame(0, &cache);

        // Retrieve co-located
        let co_loc = ref_mvs.get_co_located(0, 5, 5).expect("should succeed");
        assert!(co_loc.valid);
        assert_eq!(co_loc.mv.dx, 10);
    }

    #[test]
    fn test_co_located_lookup_scaling() {
        let mut lookup = CoLocatedMvLookup::new();
        lookup.allocate(64, 64, 3);
        lookup.set_temporal_distances(&[1, 2, 4]);

        // Store a co-located MV
        if let Some(frame) = lookup.ref_mvs_mut().get_frame_mut(0) {
            let entry = MvCacheEntry::inter(MotionVector::new(100, 200), 0, 50);
            frame.set(5, 5, 0, entry);
        }

        // Get scaled for different target distance
        let scaled = lookup.get_scaled_co_located(0, 5, 5, 2);
        assert!(scaled.is_some());
        let mv = scaled.expect("should succeed");
        assert_eq!(mv.dx, 200); // Scaled by 2
        assert_eq!(mv.dy, 400);
    }

    #[test]
    fn test_search_result_cache() {
        let mut cache = SearchResultCache::new();
        cache.allocate(64, 64, BlockSize::Block8x8);

        // Store result
        let mv = MotionVector::new(10, 20);
        cache.store(0, 0, mv, 100);

        // Retrieve
        let result = cache.get(0, 0);
        assert!(result.is_some());
        let (cached_mv, sad) = result.expect("should succeed");
        assert_eq!(cached_mv.dx, 10);
        assert_eq!(sad, 100);
    }

    #[test]
    fn test_search_result_cache_clear() {
        let mut cache = SearchResultCache::new();
        cache.allocate(64, 64, BlockSize::Block8x8);

        cache.store(0, 0, MotionVector::new(10, 20), 100);
        assert!(cache.has(0, 0));

        cache.clear();
        assert!(!cache.has(0, 0));
    }

    #[test]
    fn test_cache_manager() {
        let mut manager = CacheManager::new();
        manager.allocate(64, 64, 3);

        // Store in current frame
        let entry = MvCacheEntry::inter(MotionVector::new(10, 20), 0, 100);
        manager.current_frame.set(5, 5, 0, entry);

        // Store search result
        manager
            .search_cache
            .store(0, 0, MotionVector::new(5, 10), 50);

        assert!(
            manager
                .current_frame
                .get(5, 5, 0)
                .expect("get should return value")
                .valid
        );
        assert!(manager.search_cache.has(0, 0));
    }

    #[test]
    fn test_cache_manager_new_frame() {
        let mut manager = CacheManager::new();
        manager.allocate(64, 64, 3);

        // Store data
        let entry = MvCacheEntry::inter(MotionVector::new(10, 20), 0, 100);
        manager.current_frame.set(5, 5, 0, entry);
        manager
            .search_cache
            .store(0, 0, MotionVector::new(5, 10), 50);

        // New frame clears caches
        manager.new_frame();

        assert!(
            !manager
                .current_frame
                .get(5, 5, 0)
                .expect("get should return value")
                .valid
        );
        assert!(!manager.search_cache.has(0, 0));
    }
}
