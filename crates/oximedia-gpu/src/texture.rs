//! GPU texture management
//!
//! Describes texture formats, computes memory requirements, and provides a
//! simple pooled allocator for GPU textures.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Supported GPU texture formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    /// 8-bit RGBA (4 bytes / pixel)
    Rgba8,
    /// 16-bit half-float RGBA (8 bytes / pixel)
    Rgba16f,
    /// 10-bit RGB + 2-bit alpha packed (4 bytes / pixel)
    Rgb10A2,
    /// Single 8-bit red channel (1 byte / pixel)
    R8,
    /// Dual 8-bit RG channels (2 bytes / pixel)
    Rg8,
    /// Planar YUV 4:2:0 (1.5 bytes / pixel)
    Yuv420,
    /// Semi-planar NV12 YUV 4:2:0 (1.5 bytes / pixel)
    Nv12,
}

impl TextureFormat {
    /// Average bytes per pixel (may be fractional for planar formats)
    #[must_use]
    pub fn bytes_per_pixel(&self) -> f32 {
        match self {
            Self::Rgba8 | Self::Rgb10A2 => 4.0,
            Self::Rgba16f => 8.0,
            Self::R8 => 1.0,
            Self::Rg8 => 2.0,
            Self::Yuv420 | Self::Nv12 => 1.5,
        }
    }

    /// Whether this format is a YUV variant
    #[must_use]
    pub fn is_yuv(&self) -> bool {
        matches!(self, Self::Yuv420 | Self::Nv12)
    }

    /// Logical channel count (Y, Cb, Cr are each counted separately)
    #[must_use]
    pub fn channel_count(&self) -> u8 {
        match self {
            Self::R8 => 1,
            Self::Rg8 => 2,
            Self::Rgba8 | Self::Rgba16f | Self::Rgb10A2 => 4,
            Self::Yuv420 | Self::Nv12 => 3,
        }
    }
}

/// Descriptor for a single GPU texture
#[derive(Debug, Clone)]
pub struct TextureDescriptor {
    /// Width in texels
    pub width: u32,
    /// Height in texels
    pub height: u32,
    /// Texel format
    pub format: TextureFormat,
    /// Number of mip levels (1 = base level only)
    pub mip_levels: u8,
    /// Number of array layers (1 = single texture)
    pub array_layers: u16,
}

impl TextureDescriptor {
    /// Create a simple 2-D texture with no mip chain and a single layer
    #[must_use]
    pub fn new(width: u32, height: u32, format: TextureFormat) -> Self {
        Self {
            width,
            height,
            format,
            mip_levels: 1,
            array_layers: 1,
        }
    }

    /// Total size in bytes for the full mip chain and all array layers
    ///
    /// Each successive mip level has half the dimensions of the previous.
    /// Minimum mip size is 1×1.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        let bpp = self.format.bytes_per_pixel();
        let layers = self.array_layers as usize;
        let mut total_pixels: f64 = 0.0;
        let (mut w, mut h) = (f64::from(self.width), f64::from(self.height));
        for _ in 0..self.mip_levels {
            total_pixels += w * h;
            w = (w / 2.0).max(1.0);
            h = (h / 2.0).max(1.0);
        }
        (total_pixels * f64::from(bpp) * layers as f64) as usize
    }

    /// Total number of texels in the base mip level (ignoring arrays / mips)
    #[must_use]
    pub fn total_pixels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// A pool of GPU textures backed by a fixed memory budget and an optional
/// count-based capacity limit.
pub struct TexturePool {
    /// All allocated descriptors (index acts as texture handle)
    descriptors: Vec<Option<TextureDescriptor>>,
    /// Currently allocated bytes
    allocated_bytes: usize,
    /// Maximum bytes the pool may use
    pub(crate) max_bytes: usize,
    /// Maximum number of live textures (0 = unlimited).
    max_textures: usize,
    /// Monotonic clock for LRU tracking (incremented on each touch/allocate)
    access_clock: u64,
    /// Last-access timestamp per slot (parallel to `descriptors`)
    last_access: Vec<u64>,
}

impl TexturePool {
    /// Create a pool with a budget of `max_gb` gigabytes and no count limit.
    #[must_use]
    pub fn new(max_gb: f64) -> Self {
        Self {
            descriptors: Vec::new(),
            allocated_bytes: 0,
            max_bytes: (max_gb * 1024.0 * 1024.0 * 1024.0) as usize,
            max_textures: 0,
            access_clock: 0,
            last_access: Vec::new(),
        }
    }

    /// Create a pool with an explicit maximum texture **count**.
    ///
    /// When the pool holds `max` live textures, `allocate` returns `None`.
    /// Use `evict_lru` or `allocate_with_lru_eviction` to make room.
    /// The byte budget is set to `usize::MAX` (effectively unlimited).
    #[must_use]
    pub fn with_capacity(max: usize) -> Self {
        Self {
            descriptors: Vec::with_capacity(max),
            allocated_bytes: 0,
            max_bytes: usize::MAX,
            max_textures: max,
            access_clock: 0,
            last_access: Vec::with_capacity(max),
        }
    }

    /// Evict all LRU textures until the pool is below its `max_textures` limit
    /// (or until the pool is empty).
    ///
    /// Returns the number of textures evicted.
    pub fn evict_lru(&mut self) -> usize {
        let mut evicted = 0usize;
        while self.max_textures > 0 && self.live_count() > self.max_textures {
            match self.lru_handle() {
                Some(h) => {
                    self.free(h);
                    evicted += 1;
                }
                None => break,
            }
        }
        evicted
    }

    /// Allocate a texture in the pool
    ///
    /// Returns `Some(handle)` on success, or `None` if the byte budget or the
    /// count capacity (`max_textures`) is exceeded.
    pub fn allocate(&mut self, desc: TextureDescriptor) -> Option<usize> {
        let bytes = desc.size_bytes();
        if self.allocated_bytes + bytes > self.max_bytes {
            return None;
        }
        if self.max_textures > 0 && self.live_count() >= self.max_textures {
            return None;
        }
        // Reuse a freed slot if possible
        self.access_clock += 1;
        let ts = self.access_clock;
        if let Some(idx) = self
            .descriptors
            .iter()
            .position(std::option::Option::is_none)
        {
            self.descriptors[idx] = Some(desc);
            self.last_access[idx] = ts;
            self.allocated_bytes += bytes;
            return Some(idx);
        }
        let idx = self.descriptors.len();
        self.descriptors.push(Some(desc));
        self.last_access.push(ts);
        self.allocated_bytes += bytes;
        Some(idx)
    }

    /// Allocate a texture, evicting the LRU slot if the budget or count limit
    /// is exceeded.
    ///
    /// Returns `Some(handle)` on success, or `None` if eviction cannot free
    /// enough resources (e.g. a single remaining texture is larger than the
    /// requested one's byte budget).
    pub fn allocate_with_lru_eviction(&mut self, desc: TextureDescriptor) -> Option<usize> {
        let bytes = desc.size_bytes();
        // Try ordinary allocation first.
        let count_ok = self.max_textures == 0 || self.live_count() < self.max_textures;
        if self.allocated_bytes + bytes <= self.max_bytes && count_ok {
            return self.allocate(desc);
        }
        // Evict LRU entries until both byte budget and count limit are satisfied.
        loop {
            let bytes_ok = self.allocated_bytes + bytes <= self.max_bytes;
            let cnt_ok = self.max_textures == 0 || self.live_count() < self.max_textures;
            if bytes_ok && cnt_ok {
                return self.allocate(desc);
            }
            let lru = self.lru_handle()?;
            self.free(lru);
        }
    }

    /// Return the handle of the least-recently-used allocated texture, or
    /// `None` if the pool is empty.
    #[must_use]
    pub fn lru_handle(&self) -> Option<usize> {
        self.descriptors
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.as_ref().map(|_| i))
            .min_by_key(|&i| self.last_access[i])
    }

    /// Update the access timestamp of `handle` to mark it as recently used.
    pub fn touch(&mut self, handle: usize) {
        if handle < self.descriptors.len() && self.descriptors[handle].is_some() {
            self.access_clock += 1;
            self.last_access[handle] = self.access_clock;
        }
    }

    /// Free a texture by handle
    pub fn free(&mut self, id: usize) {
        if let Some(slot) = self.descriptors.get_mut(id) {
            if let Some(desc) = slot.take() {
                let bytes = desc.size_bytes();
                self.allocated_bytes = self.allocated_bytes.saturating_sub(bytes);
                self.last_access[id] = 0;
            }
        }
    }

    /// Utilisation in [0.0, 1.0]
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.max_bytes == 0 {
            return 0.0;
        }
        self.allocated_bytes as f64 / self.max_bytes as f64
    }

    /// Number of live (allocated) textures
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.descriptors.iter().filter(|s| s.is_some()).count()
    }

    /// Currently allocated bytes
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// Pool capacity in bytes
    #[must_use]
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }
}

// ============================================================
// Unit tests
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgba8_bytes_per_pixel() {
        assert!((TextureFormat::Rgba8.bytes_per_pixel() - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_yuv_formats_are_yuv() {
        assert!(TextureFormat::Yuv420.is_yuv());
        assert!(TextureFormat::Nv12.is_yuv());
        assert!(!TextureFormat::Rgba8.is_yuv());
    }

    #[test]
    fn test_channel_counts() {
        assert_eq!(TextureFormat::R8.channel_count(), 1);
        assert_eq!(TextureFormat::Rg8.channel_count(), 2);
        assert_eq!(TextureFormat::Rgba8.channel_count(), 4);
        assert_eq!(TextureFormat::Yuv420.channel_count(), 3);
    }

    #[test]
    fn test_descriptor_new_defaults() {
        let d = TextureDescriptor::new(1920, 1080, TextureFormat::Rgba8);
        assert_eq!(d.mip_levels, 1);
        assert_eq!(d.array_layers, 1);
    }

    #[test]
    fn test_descriptor_total_pixels() {
        let d = TextureDescriptor::new(100, 200, TextureFormat::R8);
        assert_eq!(d.total_pixels(), 20_000);
    }

    #[test]
    fn test_descriptor_size_bytes_rgba8() {
        let d = TextureDescriptor::new(4, 4, TextureFormat::Rgba8);
        // 4*4 = 16 pixels × 4 bytes = 64 bytes
        assert_eq!(d.size_bytes(), 64);
    }

    #[test]
    fn test_descriptor_size_bytes_with_mips() {
        // 4×4 + 2×2 + 1×1 = 16 + 4 + 1 = 21 pixels × 4 bytes = 84
        let mut d = TextureDescriptor::new(4, 4, TextureFormat::Rgba8);
        d.mip_levels = 3;
        assert_eq!(d.size_bytes(), 84);
    }

    #[test]
    fn test_pool_basic_allocation() {
        let mut pool = TexturePool::new(1.0);
        let desc = TextureDescriptor::new(64, 64, TextureFormat::Rgba8);
        let handle = pool.allocate(desc);
        assert!(handle.is_some());
        assert_eq!(pool.live_count(), 1);
    }

    #[test]
    fn test_pool_free_reduces_bytes() {
        let mut pool = TexturePool::new(1.0);
        let desc = TextureDescriptor::new(4, 4, TextureFormat::Rgba8);
        let handle = pool.allocate(desc).expect("allocation should succeed");
        let before = pool.allocated_bytes();
        pool.free(handle);
        assert!(pool.allocated_bytes() < before);
        assert_eq!(pool.live_count(), 0);
    }

    #[test]
    fn test_pool_reuses_freed_slot() {
        let mut pool = TexturePool::new(1.0);
        let d1 = TextureDescriptor::new(4, 4, TextureFormat::R8);
        let h1 = pool.allocate(d1).expect("allocation should succeed");
        pool.free(h1);
        let d2 = TextureDescriptor::new(4, 4, TextureFormat::R8);
        let h2 = pool.allocate(d2).expect("allocation should succeed");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_pool_budget_exceeded_returns_none() {
        // Pool with a tiny 1-byte budget
        let mut pool = TexturePool::new(0.0);
        pool.max_bytes = 1;
        let desc = TextureDescriptor::new(1920, 1080, TextureFormat::Rgba8);
        assert!(pool.allocate(desc).is_none());
    }

    #[test]
    fn test_pool_utilization_after_alloc() {
        let mut pool = TexturePool::new(0.0);
        // Set a precise budget
        let desc = TextureDescriptor::new(4, 4, TextureFormat::Rgba8); // 64 bytes
        pool.max_bytes = 128;
        pool.allocate(desc).expect("allocation should succeed");
        let util = pool.utilization();
        assert!((util - 0.5).abs() < 1e-6, "expected 0.5, got {util}");
    }

    // --- LRU eviction tests ---

    #[test]
    fn test_lru_handle_on_empty_pool() {
        let pool = TexturePool::new(1.0);
        assert!(pool.lru_handle().is_none());
    }

    #[test]
    fn test_lru_handle_returns_oldest() {
        let mut pool = TexturePool::new(1.0);
        let h0 = pool
            .allocate(TextureDescriptor::new(4, 4, TextureFormat::R8))
            .expect("alloc");
        let h1 = pool
            .allocate(TextureDescriptor::new(4, 4, TextureFormat::R8))
            .expect("alloc");
        // h0 was allocated first → lower timestamp → LRU
        assert_eq!(pool.lru_handle(), Some(h0));
        // Touch h0 — now h1 becomes LRU
        pool.touch(h0);
        assert_eq!(pool.lru_handle(), Some(h1));
        let _ = h1; // suppress unused warning
    }

    #[test]
    fn test_touch_updates_lru_order() {
        let mut pool = TexturePool::new(1.0);
        let h0 = pool
            .allocate(TextureDescriptor::new(4, 4, TextureFormat::R8))
            .expect("alloc");
        let h1 = pool
            .allocate(TextureDescriptor::new(4, 4, TextureFormat::R8))
            .expect("alloc");
        let h2 = pool
            .allocate(TextureDescriptor::new(4, 4, TextureFormat::R8))
            .expect("alloc");
        // Insertion order: h0 < h1 < h2 → LRU is h0
        assert_eq!(pool.lru_handle(), Some(h0));
        pool.touch(h0);
        pool.touch(h1);
        // Now order is: h2 < h0 < h1 → LRU is h2
        assert_eq!(pool.lru_handle(), Some(h2));
    }

    #[test]
    fn test_allocate_with_lru_eviction_makes_space() {
        let mut pool = TexturePool::new(0.0);
        // Budget = 64 bytes (one 4×4 Rgba8 texture)
        pool.max_bytes = 64;
        let h0 = pool
            .allocate(TextureDescriptor::new(4, 4, TextureFormat::Rgba8))
            .expect("first alloc should succeed");
        assert_eq!(pool.live_count(), 1);
        // Normal allocate fails — budget exhausted
        assert!(pool
            .allocate(TextureDescriptor::new(4, 4, TextureFormat::Rgba8))
            .is_none());
        // LRU eviction allocate should evict h0 and succeed
        let h1 = pool
            .allocate_with_lru_eviction(TextureDescriptor::new(4, 4, TextureFormat::Rgba8))
            .expect("lru eviction alloc should succeed");
        assert_eq!(pool.live_count(), 1);
        // The evicted slot is reused
        assert_eq!(h0, h1);
    }

    #[test]
    fn test_allocate_with_lru_eviction_preserves_mru() {
        let mut pool = TexturePool::new(0.0);
        // Budget = 128 bytes (two 4×4 Rgba8 textures = 64 each)
        pool.max_bytes = 128;
        let h0 = pool
            .allocate(TextureDescriptor::new(4, 4, TextureFormat::Rgba8))
            .expect("alloc h0");
        let _h1 = pool
            .allocate(TextureDescriptor::new(4, 4, TextureFormat::Rgba8))
            .expect("alloc h1");
        // Touch h0 so h1 becomes LRU
        pool.touch(h0);
        // Now request a third texture — must evict h1 (LRU)
        let h2 = pool
            .allocate_with_lru_eviction(TextureDescriptor::new(4, 4, TextureFormat::Rgba8))
            .expect("lru eviction");
        // h1 index should be reused by h2
        assert_eq!(h2, _h1);
        // h0 must still be alive
        assert_eq!(pool.live_count(), 2);
    }

    #[test]
    fn test_lru_eviction_returns_none_when_budget_impossible() {
        let mut pool = TexturePool::new(0.0);
        // Budget only fits 16 bytes but we want a 64-byte texture
        pool.max_bytes = 16;
        // Put a 16-byte 4×4 R8 texture in
        pool.allocate(TextureDescriptor::new(4, 4, TextureFormat::R8))
            .expect("alloc small");
        // Request 64-byte texture — cannot fit even after evicting the 16-byte one
        let result =
            pool.allocate_with_lru_eviction(TextureDescriptor::new(4, 4, TextureFormat::Rgba8));
        assert!(result.is_none());
    }

    // ─── Task F: with_capacity and evict_lru ─────────────────────────────────

    #[test]
    fn test_with_capacity_rejects_when_full() {
        let mut pool = TexturePool::with_capacity(2);
        let d = TextureDescriptor::new(4, 4, TextureFormat::Rgba8);
        assert!(
            pool.allocate(d.clone()).is_some(),
            "first alloc should succeed"
        );
        assert!(
            pool.allocate(d.clone()).is_some(),
            "second alloc should succeed"
        );
        // Third allocation must be rejected — count limit reached
        assert!(
            pool.allocate(d.clone()).is_none(),
            "third alloc must fail (capacity = 2)"
        );
    }

    #[test]
    fn test_evict_lru_reduces_count_to_capacity() {
        let mut pool = TexturePool::with_capacity(2);
        let d = TextureDescriptor::new(4, 4, TextureFormat::Rgba8);
        // Force-allocate 3 textures by bypassing the count limit temporarily
        pool.max_textures = 0; // unlimited
        pool.allocate(d.clone()).expect("alloc 1");
        pool.allocate(d.clone()).expect("alloc 2");
        pool.allocate(d.clone()).expect("alloc 3");
        assert_eq!(pool.live_count(), 3);

        pool.max_textures = 2; // re-enable limit
        let evicted = pool.evict_lru();
        assert_eq!(
            evicted, 1,
            "one texture should be evicted to reach capacity 2"
        );
        assert_eq!(pool.live_count(), 2);
    }

    #[test]
    fn test_evict_lru_correct_order() {
        let mut pool = TexturePool::with_capacity(3);
        let d = TextureDescriptor::new(4, 4, TextureFormat::R8);
        // Allocate in order h0, h1, h2 — all within capacity initially
        pool.max_textures = 0;
        let h0 = pool.allocate(d.clone()).expect("h0");
        let h1 = pool.allocate(d.clone()).expect("h1");
        let h2 = pool.allocate(d.clone()).expect("h2");
        // Touch h0 and h1 → h2 becomes LRU
        pool.touch(h0);
        pool.touch(h1);
        pool.max_textures = 2;
        let evicted = pool.evict_lru();
        assert_eq!(evicted, 1, "one eviction expected");
        // h2 (LRU) should be gone; h0, h1 should survive
        assert!(
            pool.descriptors[h2].is_none(),
            "h2 should have been evicted (LRU)"
        );
        assert!(pool.descriptors[h0].is_some(), "h0 should still be alive");
        assert!(pool.descriptors[h1].is_some(), "h1 should still be alive");
    }

    #[test]
    fn test_evict_lru_noop_when_under_capacity() {
        let mut pool = TexturePool::with_capacity(5);
        let d = TextureDescriptor::new(4, 4, TextureFormat::R8);
        pool.allocate(d.clone()).expect("alloc");
        pool.allocate(d.clone()).expect("alloc");
        // Two live textures; capacity is 5 — no eviction needed.
        let evicted = pool.evict_lru();
        assert_eq!(evicted, 0, "no eviction expected when under capacity");
    }

    #[test]
    fn test_evict_lru_on_empty_pool() {
        let mut pool = TexturePool::with_capacity(2);
        let evicted = pool.evict_lru();
        assert_eq!(evicted, 0, "no eviction on empty pool");
    }

    #[test]
    fn test_with_capacity_allocate_after_evict() {
        let mut pool = TexturePool::with_capacity(1);
        let d = TextureDescriptor::new(4, 4, TextureFormat::R8);
        let h0 = pool.allocate(d.clone()).expect("first alloc");
        // Pool is full — direct allocation must fail
        assert!(pool.allocate(d.clone()).is_none());
        // Evict via LRU, then allocate_with_lru_eviction
        let h1 = pool
            .allocate_with_lru_eviction(d.clone())
            .expect("evict+alloc");
        assert_eq!(pool.live_count(), 1, "still 1 live after evict+alloc");
        // The freed slot should be reused
        assert_eq!(h0, h1, "freed slot should be reused");
    }
}
