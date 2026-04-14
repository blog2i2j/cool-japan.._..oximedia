//! Glyph rasterization cache with LRU eviction.
//!
//! Rasterizing a glyph on every frame is expensive when the same characters
//! appear repeatedly (e.g. scoreboard digits, ticker text, broadcast clocks).
//! This module provides a [`GlyphCache`] that stores the RGBA bitmaps for
//! recently rendered glyphs and evicts the least-recently-used entries when
//! the cache reaches its capacity.
//!
//! The cache is keyed by [`GlyphKey`] — a combination of character, font-family
//! name, size-in-pixels, and sub-pixel mode — so glyphs rendered at different
//! sizes or with different fonts do not collide.
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_graphics::glyph_cache::{GlyphCache, GlyphKey, RasterizedGlyph};
//! use oximedia_graphics::font_metrics::SubPixelMode;
//!
//! let mut cache = GlyphCache::new(256);
//!
//! let key = GlyphKey::new('A', "Roboto", 24, SubPixelMode::Rgb);
//! if cache.get(&key).is_none() {
//!     // rasterize here …
//!     if let Some(glyph) = RasterizedGlyph::new(8, 12, vec![0u8; 8 * 12 * 4], 0, -2, 10.0) {
//!         cache.insert(key, glyph);
//!     }
//! }
//! ```

use std::collections::HashMap;

use crate::font_metrics::SubPixelMode;

// ---------------------------------------------------------------------------
// GlyphKey
// ---------------------------------------------------------------------------

/// Cache key uniquely identifying a rasterized glyph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GlyphKey {
    /// The Unicode code point being rendered.
    pub codepoint: char,
    /// Font family name.
    pub font_family: String,
    /// Glyph size in pixels (integer approximation to the requested pt size).
    pub size_px: u16,
    /// Sub-pixel rendering mode.
    pub sub_pixel: SubPixelMode,
}

impl GlyphKey {
    /// Create a new glyph key.
    #[must_use]
    pub fn new(
        codepoint: char,
        font_family: impl Into<String>,
        size_px: u16,
        sub_pixel: SubPixelMode,
    ) -> Self {
        Self {
            codepoint,
            font_family: font_family.into(),
            size_px,
            sub_pixel,
        }
    }
}

// ---------------------------------------------------------------------------
// RasterizedGlyph
// ---------------------------------------------------------------------------

/// The rasterized bitmap for a single glyph.
///
/// Pixel data is stored as tightly packed RGBA bytes:
/// `data[y * width * 4 + x * 4 .. y * width * 4 + x * 4 + 4]` = `[R, G, B, A]`.
#[derive(Debug, Clone)]
pub struct RasterizedGlyph {
    /// Bitmap width in pixels.
    pub width: u32,
    /// Bitmap height in pixels.
    pub height: u32,
    /// RGBA pixel data.  Length must equal `width * height * 4`.
    pub data: Vec<u8>,
    /// Horizontal bearing (offset from the pen position to the left edge of
    /// the glyph bitmap), in pixels.
    pub bearing_x: i32,
    /// Vertical bearing (offset from the baseline to the top of the bitmap),
    /// in pixels.  Negative values move the glyph down relative to the
    /// baseline.
    pub bearing_y: i32,
    /// Horizontal advance width in pixels (distance to advance the pen after
    /// this glyph).
    pub advance_x: f32,
}

impl RasterizedGlyph {
    /// Create a new rasterized glyph.
    ///
    /// Returns `None` if `data.len() != width * height * 4`.
    #[must_use]
    pub fn new(
        width: u32,
        height: u32,
        data: Vec<u8>,
        bearing_x: i32,
        bearing_y: i32,
        advance_x: f32,
    ) -> Option<Self> {
        let expected = (width as usize) * (height as usize) * 4;
        if data.len() != expected {
            return None;
        }
        Some(Self {
            width,
            height,
            data,
            bearing_x,
            bearing_y,
            advance_x,
        })
    }

    /// Create a new rasterized glyph from a greyscale coverage buffer.
    ///
    /// `coverage` has one byte per pixel (0 = transparent, 255 = fully opaque).
    /// Each byte is converted to an RGBA pixel using `color` with the coverage
    /// value as the alpha multiplier.
    ///
    /// Returns `None` if `coverage.len() != width * height`.
    #[must_use]
    pub fn from_coverage(
        width: u32,
        height: u32,
        coverage: &[u8],
        color: [u8; 4],
        bearing_x: i32,
        bearing_y: i32,
        advance_x: f32,
    ) -> Option<Self> {
        let px = (width as usize) * (height as usize);
        if coverage.len() != px {
            return None;
        }
        let mut data = Vec::with_capacity(px * 4);
        for &alpha in coverage {
            let a = (alpha as u32 * color[3] as u32 / 255) as u8;
            data.push(color[0]);
            data.push(color[1]);
            data.push(color[2]);
            data.push(a);
        }
        Some(Self {
            width,
            height,
            data,
            bearing_x,
            bearing_y,
            advance_x,
        })
    }

    /// Byte size of the pixel data.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the glyph has zero area (e.g., a space character).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

// ---------------------------------------------------------------------------
// LRU internal entry
// ---------------------------------------------------------------------------

/// Internal cache entry combining the glyph bitmap and LRU counter.
struct CacheEntry {
    glyph: RasterizedGlyph,
    /// Logical access clock value at the last access.  Higher means more
    /// recently used.
    last_used: u64,
}

// ---------------------------------------------------------------------------
// GlyphCache
// ---------------------------------------------------------------------------

/// LRU glyph cache keyed by [`GlyphKey`].
///
/// When the number of cached glyphs reaches `capacity`, the least-recently-used
/// entry is evicted before inserting a new one.
pub struct GlyphCache {
    entries: HashMap<GlyphKey, CacheEntry>,
    capacity: usize,
    clock: u64,
    hits: u64,
    misses: u64,
}

impl GlyphCache {
    /// Create a cache with the given maximum number of entries.
    ///
    /// The capacity is clamped to at least 1.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            entries: HashMap::with_capacity(capacity + 1),
            capacity,
            clock: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Cache capacity (maximum number of glyphs stored simultaneously).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of glyphs currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of cache hits since creation.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Number of cache misses since creation.
    #[must_use]
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Cache hit ratio in the range `[0.0, 1.0]`.
    ///
    /// Returns `0.0` if no lookups have been performed.
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    /// Look up a glyph by key.
    ///
    /// Returns a shared reference to the cached [`RasterizedGlyph`] if present,
    /// updating the LRU counter; returns `None` on a cache miss.
    pub fn get(&mut self, key: &GlyphKey) -> Option<&RasterizedGlyph> {
        // Check presence first (avoids borrow-checker issues with the
        // mutable borrow needed to update `last_used`).
        if self.entries.contains_key(key) {
            self.clock += 1;
            let clock = self.clock;
            if let Some(entry) = self.entries.get_mut(key) {
                entry.last_used = clock;
            }
            self.hits += 1;
            return self.entries.get(key).map(|e| &e.glyph);
        }
        self.misses += 1;
        None
    }

    /// Look up a glyph without updating access statistics or the LRU counter.
    ///
    /// Useful for read-only inspection (e.g., measuring sizes) without
    /// affecting eviction order.
    #[must_use]
    pub fn peek(&self, key: &GlyphKey) -> Option<&RasterizedGlyph> {
        self.entries.get(key).map(|e| &e.glyph)
    }

    /// Insert a glyph into the cache.
    ///
    /// If the cache is at capacity the LRU entry is evicted first.  If a glyph
    /// with the same key already exists it is replaced and its access time is
    /// updated.
    pub fn insert(&mut self, key: GlyphKey, glyph: RasterizedGlyph) {
        self.clock += 1;

        if self.entries.contains_key(&key) {
            // Update in place.
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.glyph = glyph;
                entry.last_used = self.clock;
            }
            return;
        }

        // Evict LRU entry if at capacity.
        if self.entries.len() >= self.capacity {
            self.evict_lru();
        }

        self.entries.insert(
            key,
            CacheEntry {
                glyph,
                last_used: self.clock,
            },
        );
    }

    /// Explicitly remove a single glyph from the cache.
    ///
    /// Returns `true` if the key was present.
    pub fn remove(&mut self, key: &GlyphKey) -> bool {
        self.entries.remove(key).is_some()
    }

    /// Remove all cached glyphs that match the given font family.
    ///
    /// Useful when a font is unloaded or replaced at runtime.
    pub fn invalidate_font(&mut self, font_family: &str) {
        self.entries
            .retain(|k, _| k.font_family != font_family);
    }

    /// Remove all cached glyphs of a given size.
    pub fn invalidate_size(&mut self, size_px: u16) {
        self.entries.retain(|k, _| k.size_px != size_px);
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Reset hit/miss statistics without clearing cached glyphs.
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }

    /// Total byte footprint of all cached bitmaps.
    #[must_use]
    pub fn total_byte_size(&self) -> usize {
        self.entries.values().map(|e| e.glyph.byte_size()).sum()
    }

    // -----------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------

    /// Evict the least-recently-used entry.
    fn evict_lru(&mut self) {
        let lru_key = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(k, _)| k.clone());

        if let Some(key) = lru_key {
            self.entries.remove(&key);
        }
    }
}

// ---------------------------------------------------------------------------
// GlyphAtlas — texture atlas for GPU upload
// ---------------------------------------------------------------------------

/// A simple packed glyph atlas that arranges rasterized glyphs in rows.
///
/// The atlas rasterises glyphs left-to-right, advancing to a new row when the
/// current row is full.  It provides UV coordinates for each glyph so that a
/// GPU renderer can sample them without per-glyph texture switches.
///
/// The atlas texture data is stored as RGBA bytes in row-major order.
pub struct GlyphAtlas {
    /// Atlas width in pixels.
    pub atlas_width: u32,
    /// Atlas height in pixels.
    pub atlas_height: u32,
    /// RGBA pixel data for the atlas texture.
    pub data: Vec<u8>,
    /// UV rectangles for each cached glyph: `(u_min, v_min, u_max, v_max)`.
    uvs: HashMap<GlyphKey, (f32, f32, f32, f32)>,
    /// Current pen x position.
    pen_x: u32,
    /// Current pen y (top of current row).
    pen_y: u32,
    /// Height of the tallest glyph in the current row.
    row_height: u32,
    /// Padding between glyphs (pixels).
    padding: u32,
}

impl GlyphAtlas {
    /// Create a new atlas of the given dimensions.
    ///
    /// `padding` is the gap between glyphs in pixels (helps prevent bleeding).
    #[must_use]
    pub fn new(atlas_width: u32, atlas_height: u32, padding: u32) -> Self {
        let size = (atlas_width as usize) * (atlas_height as usize) * 4;
        Self {
            atlas_width,
            atlas_height,
            data: vec![0u8; size],
            uvs: HashMap::new(),
            pen_x: 0,
            pen_y: 0,
            row_height: 0,
            padding,
        }
    }

    /// Attempt to pack a glyph into the atlas.
    ///
    /// Returns `true` on success.  Returns `false` if the glyph does not fit
    /// (the atlas is full).
    pub fn pack(&mut self, key: GlyphKey, glyph: &RasterizedGlyph) -> bool {
        if glyph.is_empty() {
            // Zero-size glyphs (spaces, etc.) get a zero-area UV entry.
            self.uvs.insert(key, (0.0, 0.0, 0.0, 0.0));
            return true;
        }

        let gw = glyph.width + self.padding;
        let gh = glyph.height + self.padding;

        // Advance to a new row if needed.
        if self.pen_x + gw > self.atlas_width {
            self.pen_y += self.row_height + self.padding;
            self.pen_x = 0;
            self.row_height = 0;
        }

        // Check vertical overflow.
        if self.pen_y + gh > self.atlas_height {
            return false;
        }

        // Blit glyph RGBA data into the atlas.
        for row in 0..glyph.height {
            let src_start = (row as usize) * (glyph.width as usize) * 4;
            let dst_row = self.pen_y + row;
            let dst_start = (dst_row as usize) * (self.atlas_width as usize) * 4
                + (self.pen_x as usize) * 4;
            let count = (glyph.width as usize) * 4;
            let src_end = src_start + count;
            let dst_end = dst_start + count;
            if src_end <= glyph.data.len() && dst_end <= self.data.len() {
                self.data[dst_start..dst_end]
                    .copy_from_slice(&glyph.data[src_start..src_end]);
            }
        }

        // Compute normalized UVs.
        let u_min = self.pen_x as f32 / self.atlas_width as f32;
        let v_min = self.pen_y as f32 / self.atlas_height as f32;
        let u_max = (self.pen_x + glyph.width) as f32 / self.atlas_width as f32;
        let v_max = (self.pen_y + glyph.height) as f32 / self.atlas_height as f32;
        self.uvs.insert(key, (u_min, v_min, u_max, v_max));

        // Advance pen.
        self.pen_x += gw;
        if glyph.height > self.row_height {
            self.row_height = glyph.height;
        }

        true
    }

    /// Return the UV rectangle `(u_min, v_min, u_max, v_max)` for the given
    /// glyph, or `None` if it has not been packed.
    #[must_use]
    pub fn uv(&self, key: &GlyphKey) -> Option<(f32, f32, f32, f32)> {
        self.uvs.get(key).copied()
    }

    /// Number of glyphs currently packed in the atlas.
    #[must_use]
    pub fn glyph_count(&self) -> usize {
        self.uvs.len()
    }

    /// Byte size of the atlas texture data.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    /// Clear the atlas, removing all packed glyphs and resetting the pen.
    pub fn clear(&mut self) {
        self.data.fill(0);
        self.uvs.clear();
        self.pen_x = 0;
        self.pen_y = 0;
        self.row_height = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_glyph(w: u32, h: u32) -> RasterizedGlyph {
        let data = vec![255u8; (w * h * 4) as usize];
        RasterizedGlyph::new(w, h, data, 0, -2, w as f32 + 1.0).expect("valid glyph")
    }

    fn make_key(ch: char) -> GlyphKey {
        GlyphKey::new(ch, "Roboto", 24, SubPixelMode::Rgb)
    }

    // --- GlyphKey ---

    #[test]
    fn test_glyph_key_equality() {
        let a = GlyphKey::new('A', "Roboto", 24, SubPixelMode::Rgb);
        let b = GlyphKey::new('A', "Roboto", 24, SubPixelMode::Rgb);
        assert_eq!(a, b);
    }

    #[test]
    fn test_glyph_key_different_char() {
        let a = GlyphKey::new('A', "Roboto", 24, SubPixelMode::Rgb);
        let b = GlyphKey::new('B', "Roboto", 24, SubPixelMode::Rgb);
        assert_ne!(a, b);
    }

    // --- RasterizedGlyph ---

    #[test]
    fn test_rasterized_glyph_valid() {
        let g = make_glyph(8, 12);
        assert_eq!(g.width, 8);
        assert_eq!(g.height, 12);
        assert_eq!(g.byte_size(), 8 * 12 * 4);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_rasterized_glyph_invalid_data_size() {
        let bad_data = vec![0u8; 10]; // wrong size
        let result = RasterizedGlyph::new(8, 12, bad_data, 0, 0, 9.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_rasterized_glyph_empty() {
        let g = RasterizedGlyph::new(0, 12, vec![], 0, 0, 5.0).expect("empty glyph ok");
        assert!(g.is_empty());
    }

    #[test]
    fn test_rasterized_glyph_from_coverage() {
        let coverage = vec![128u8; 8 * 12];
        let g = RasterizedGlyph::from_coverage(8, 12, &coverage, [255, 0, 0, 255], 0, 0, 9.0)
            .expect("valid");
        // Red channel should be 255, alpha should be proportional to coverage.
        assert_eq!(g.data[0], 255); // R
        assert_eq!(g.data[1], 0); // G
        assert_eq!(g.data[2], 0); // B
        // alpha = 128 * 255 / 255 = 128
        assert_eq!(g.data[3], 128);
    }

    #[test]
    fn test_rasterized_glyph_from_coverage_bad_len() {
        let bad_cov = vec![0u8; 5]; // wrong size for 8×12
        let result =
            RasterizedGlyph::from_coverage(8, 12, &bad_cov, [255, 255, 255, 255], 0, 0, 9.0);
        assert!(result.is_none());
    }

    // --- GlyphCache ---

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache = GlyphCache::new(10);
        let key = make_key('A');
        let glyph = make_glyph(8, 12);
        cache.insert(key.clone(), glyph);
        assert!(cache.get(&key).is_some());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_miss_increments_misses() {
        let mut cache = GlyphCache::new(10);
        let key = make_key('Z');
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn test_cache_hit_increments_hits() {
        let mut cache = GlyphCache::new(10);
        let key = make_key('A');
        cache.insert(key.clone(), make_glyph(8, 12));
        let _ = cache.get(&key);
        assert_eq!(cache.hits(), 1);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = GlyphCache::new(2);
        let k1 = make_key('1');
        let k2 = make_key('2');
        let k3 = make_key('3');
        cache.insert(k1.clone(), make_glyph(4, 4));
        cache.insert(k2.clone(), make_glyph(4, 4));
        // Access k1 to make it most recently used.
        let _ = cache.get(&k1);
        // Inserting k3 should evict k2 (LRU).
        cache.insert(k3.clone(), make_glyph(4, 4));
        assert!(cache.peek(&k1).is_some(), "k1 should still be cached");
        assert!(cache.peek(&k2).is_none(), "k2 should have been evicted");
        assert!(cache.peek(&k3).is_some(), "k3 should be in cache");
    }

    #[test]
    fn test_cache_invalidate_font() {
        let mut cache = GlyphCache::new(10);
        let k1 = GlyphKey::new('A', "Roboto", 24, SubPixelMode::None);
        let k2 = GlyphKey::new('B', "Arial", 24, SubPixelMode::None);
        cache.insert(k1.clone(), make_glyph(8, 12));
        cache.insert(k2.clone(), make_glyph(8, 12));
        cache.invalidate_font("Roboto");
        assert!(cache.peek(&k1).is_none());
        assert!(cache.peek(&k2).is_some());
    }

    #[test]
    fn test_cache_invalidate_size() {
        let mut cache = GlyphCache::new(10);
        let k24 = GlyphKey::new('A', "Roboto", 24, SubPixelMode::None);
        let k48 = GlyphKey::new('A', "Roboto", 48, SubPixelMode::None);
        cache.insert(k24.clone(), make_glyph(8, 12));
        cache.insert(k48.clone(), make_glyph(16, 24));
        cache.invalidate_size(24);
        assert!(cache.peek(&k24).is_none());
        assert!(cache.peek(&k48).is_some());
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = GlyphCache::new(10);
        for c in 'A'..='E' {
            cache.insert(make_key(c), make_glyph(8, 12));
        }
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_hit_ratio() {
        let mut cache = GlyphCache::new(10);
        let key = make_key('X');
        cache.insert(key.clone(), make_glyph(8, 12));
        let _ = cache.get(&key); // hit
        let _ = cache.get(&make_key('Y')); // miss
        let ratio = cache.hit_ratio();
        assert!((ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_cache_total_byte_size() {
        let mut cache = GlyphCache::new(10);
        cache.insert(make_key('A'), make_glyph(8, 12)); // 8*12*4 = 384 bytes
        cache.insert(make_key('B'), make_glyph(4, 4)); // 4*4*4 = 64 bytes
        assert_eq!(cache.total_byte_size(), 384 + 64);
    }

    #[test]
    fn test_cache_remove() {
        let mut cache = GlyphCache::new(10);
        let key = make_key('R');
        cache.insert(key.clone(), make_glyph(8, 8));
        assert!(cache.remove(&key));
        assert!(cache.peek(&key).is_none());
        assert!(!cache.remove(&key)); // second remove returns false
    }

    // --- GlyphAtlas ---

    #[test]
    fn test_atlas_pack_single_glyph() {
        let mut atlas = GlyphAtlas::new(256, 256, 1);
        let key = make_key('A');
        let glyph = make_glyph(8, 12);
        assert!(atlas.pack(key.clone(), &glyph));
        assert!(atlas.uv(&key).is_some());
        assert_eq!(atlas.glyph_count(), 1);
    }

    #[test]
    fn test_atlas_pack_fills_row_then_next() {
        let mut atlas = GlyphAtlas::new(32, 64, 0); // narrow atlas, 32px wide
        let k1 = make_key('1');
        let k2 = make_key('2');
        let k3 = make_key('3');
        let g = make_glyph(12, 12);
        // First two glyphs fit in row 1 (12+12=24 < 32).
        assert!(atlas.pack(k1, &g));
        assert!(atlas.pack(k2, &g));
        // Third glyph (12 more) overflows to row 2.
        assert!(atlas.pack(k3.clone(), &g));
        let (_, v_min, _, _) = atlas.uv(&k3).expect("k3 should be packed");
        // k3 should be in the second row, so v_min > 0.
        assert!(v_min > 0.0, "k3 should be in a new row");
    }

    #[test]
    fn test_atlas_overflow_returns_false() {
        let mut atlas = GlyphAtlas::new(16, 16, 0);
        let big = make_glyph(20, 20); // larger than atlas
        assert!(!atlas.pack(make_key('X'), &big));
    }

    #[test]
    fn test_atlas_clear() {
        let mut atlas = GlyphAtlas::new(256, 256, 1);
        assert!(atlas.pack(make_key('A'), &make_glyph(8, 12)));
        atlas.clear();
        assert_eq!(atlas.glyph_count(), 0);
        // All pixels should be zero after clear.
        assert!(atlas.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_atlas_blit_pixels_correct() {
        let mut atlas = GlyphAtlas::new(64, 64, 0);
        // Glyph with all pixels set to [255, 0, 0, 255].
        let w = 4u32;
        let h = 4u32;
        let data: Vec<u8> = (0..w * h).flat_map(|_| [255u8, 0, 0, 255]).collect();
        let g = RasterizedGlyph::new(w, h, data, 0, 0, 5.0).expect("valid");
        assert!(atlas.pack(make_key('Q'), &g));
        // The first pixel in the atlas (at row 0, col 0) should be red.
        assert_eq!(atlas.data[0], 255); // R
        assert_eq!(atlas.data[1], 0); // G
        assert_eq!(atlas.data[2], 0); // B
        assert_eq!(atlas.data[3], 255); // A
    }

    #[test]
    fn test_atlas_uv_normalized() {
        let atlas_w = 256u32;
        let atlas_h = 256u32;
        let glyph_w = 32u32;
        let glyph_h = 32u32;
        let mut atlas = GlyphAtlas::new(atlas_w, atlas_h, 0);
        let key = make_key('U');
        let g = make_glyph(glyph_w, glyph_h);
        assert!(atlas.pack(key.clone(), &g));
        let (u_min, v_min, u_max, v_max) = atlas.uv(&key).expect("present");
        assert!((u_min - 0.0).abs() < 1e-5);
        assert!((v_min - 0.0).abs() < 1e-5);
        assert!((u_max - glyph_w as f32 / atlas_w as f32).abs() < 1e-5);
        assert!((v_max - glyph_h as f32 / atlas_h as f32).abs() < 1e-5);
    }
}
