//! Glyph atlas caching for batch subtitle rendering.
//!
//! A glyph atlas packs rasterised glyph bitmaps into a single large texture,
//! enabling efficient GPU upload and reducing per-character draw calls to a
//! single textured quad per glyph.
//!
//! The atlas uses a shelf-packing algorithm: glyphs of similar height are
//! grouped onto horizontal "shelves", and new shelves are opened below the
//! current one when the current shelf is full.
//!
//! # Example
//!
//! ```
//! use oximedia_subtitle::glyph_atlas::{GlyphAtlas, AtlasConfig};
//!
//! let config = AtlasConfig { width: 512, height: 512, padding: 1 };
//! let mut atlas = GlyphAtlas::new(config);
//!
//! // Reserve space for a 16×20-pixel glyph.
//! let slot = atlas.allocate(16, 20).expect("space available");
//! assert!(slot.x + 16 <= 512);
//! assert!(slot.y + 20 <= 512);
//! ```

use crate::{SubtitleError, SubtitleResult};
use std::collections::HashMap;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a [`GlyphAtlas`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasConfig {
    /// Total atlas width in pixels.
    pub width: u32,
    /// Total atlas height in pixels.
    pub height: u32,
    /// Padding in pixels between glyphs (prevents bilinear bleeding).
    pub padding: u32,
}

impl Default for AtlasConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            padding: 1,
        }
    }
}

// ============================================================================
// Atlas slot (UV coordinates of a single glyph)
// ============================================================================

/// The position of a single glyph within the atlas texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasSlot {
    /// X offset in the atlas (pixels from left edge).
    pub x: u32,
    /// Y offset in the atlas (pixels from top edge).
    pub y: u32,
    /// Width of this glyph cell in pixels.
    pub width: u32,
    /// Height of this glyph cell in pixels.
    pub height: u32,
    /// Index of the shelf this slot lives on.
    pub shelf_index: usize,
}

impl AtlasSlot {
    /// Normalised U coordinate of the left edge (0.0–1.0).
    #[must_use]
    pub fn u0(&self, atlas_width: u32) -> f32 {
        self.x as f32 / atlas_width as f32
    }

    /// Normalised U coordinate of the right edge (0.0–1.0).
    #[must_use]
    pub fn u1(&self, atlas_width: u32) -> f32 {
        (self.x + self.width) as f32 / atlas_width as f32
    }

    /// Normalised V coordinate of the top edge (0.0–1.0).
    #[must_use]
    pub fn v0(&self, atlas_height: u32) -> f32 {
        self.y as f32 / atlas_height as f32
    }

    /// Normalised V coordinate of the bottom edge (0.0–1.0).
    #[must_use]
    pub fn v1(&self, atlas_height: u32) -> f32 {
        (self.y + self.height) as f32 / atlas_height as f32
    }
}

// ============================================================================
// Shelf
// ============================================================================

/// A horizontal strip within the atlas that packs glyphs of similar height.
#[derive(Debug, Clone)]
struct Shelf {
    /// Y coordinate of the top of this shelf.
    y: u32,
    /// Height of the shelf (set by the tallest glyph allocated so far).
    height: u32,
    /// Current X cursor (next free column, after padding).
    cursor_x: u32,
}

impl Shelf {
    /// Create a new shelf starting at the given Y offset with the given height.
    fn new(y: u32, height: u32) -> Self {
        Self {
            y,
            height,
            cursor_x: 0,
        }
    }

    /// Try to allocate `width` × `height` pixels on this shelf.
    ///
    /// Returns the `x` coordinate if successful, `None` if the shelf is full.
    fn try_allocate(&mut self, width: u32, padding: u32, atlas_width: u32) -> Option<u32> {
        let required = width + padding;
        if self.cursor_x + required > atlas_width {
            return None;
        }
        let x = self.cursor_x;
        self.cursor_x += required;
        Some(x)
    }
}

// ============================================================================
// Glyph key
// ============================================================================

/// Cache key for a glyph: Unicode codepoint + font size in integer units (px × 64).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GlyphKey {
    /// Unicode codepoint.
    pub codepoint: char,
    /// Font size in quarter-pixels (size_px * 4) to avoid floating-point hashing.
    pub size_q: u32,
    /// Font index (allows multiple fonts in one atlas).
    pub font_index: u8,
}

impl GlyphKey {
    /// Create a new glyph key from character, size in pixels, and font index.
    #[must_use]
    pub fn new(codepoint: char, size_px: f32, font_index: u8) -> Self {
        Self {
            codepoint,
            size_q: (size_px * 4.0) as u32,
            font_index,
        }
    }
}

// ============================================================================
// GlyphAtlas
// ============================================================================

/// Texture atlas that caches rasterised glyph bitmaps.
///
/// Glyphs are allocated using a shelf-first bin-packing strategy.
/// The backing pixel buffer (`pixels`) stores 8-bit alpha values in row-major
/// order; callers copy rasterised bitmaps into the slice returned by
/// [`GlyphAtlas::pixel_slice_mut`].
pub struct GlyphAtlas {
    /// Configuration (size, padding).
    pub config: AtlasConfig,
    /// RGBA pixel buffer (width × height × 4 bytes, RGBA).
    pixels: Vec<u8>,
    /// Horizontal shelves used for packing.
    shelves: Vec<Shelf>,
    /// Cursor tracking the Y position where the next shelf will open.
    next_shelf_y: u32,
    /// Cache: glyph key → atlas slot.
    cache: HashMap<GlyphKey, AtlasSlot>,
    /// Total number of successful allocations.
    allocation_count: u64,
    /// Number of cache hits.
    hit_count: u64,
}

impl GlyphAtlas {
    /// Create a new empty glyph atlas.
    #[must_use]
    pub fn new(config: AtlasConfig) -> Self {
        let size = (config.width * config.height * 4) as usize;
        Self {
            config,
            pixels: vec![0u8; size],
            shelves: Vec::new(),
            next_shelf_y: 0,
            cache: HashMap::new(),
            allocation_count: 0,
            hit_count: 0,
        }
    }

    /// Create an atlas with default 1024×1024 configuration.
    #[must_use]
    pub fn default_size() -> Self {
        Self::new(AtlasConfig::default())
    }

    // ------------------------------------------------------------------
    // Allocation
    // ------------------------------------------------------------------

    /// Allocate space for a glyph of the given pixel dimensions.
    ///
    /// Returns the [`AtlasSlot`] describing where to write the glyph bitmap.
    ///
    /// # Errors
    ///
    /// Returns [`SubtitleError::Internal`] if the atlas is full and cannot
    /// accommodate the requested glyph size.
    pub fn allocate(&mut self, width: u32, height: u32) -> SubtitleResult<AtlasSlot> {
        if width == 0 || height == 0 {
            return Err(SubtitleError::Internal(
                "glyph dimensions must be non-zero".to_string(),
            ));
        }
        if width > self.config.width || height > self.config.height {
            return Err(SubtitleError::Internal(format!(
                "glyph {}x{} exceeds atlas {}x{}",
                width, height, self.config.width, self.config.height
            )));
        }

        let padding = self.config.padding;

        // 1. Try existing shelves (best-fit height first).
        let shelf_index = self.find_best_shelf(height);

        if let Some(idx) = shelf_index {
            if let Some(x) = self.shelves[idx].try_allocate(width, padding, self.config.width) {
                let slot = AtlasSlot {
                    x,
                    y: self.shelves[idx].y,
                    width,
                    height,
                    shelf_index: idx,
                };
                self.allocation_count += 1;
                return Ok(slot);
            }
        }

        // 2. Open a new shelf.
        let new_shelf_y = self.next_shelf_y;
        let shelf_height = height + padding;
        if new_shelf_y + shelf_height > self.config.height {
            return Err(SubtitleError::Internal(
                "glyph atlas is full — increase atlas dimensions or flush cache".to_string(),
            ));
        }

        let new_idx = self.shelves.len();
        let mut shelf = Shelf::new(new_shelf_y, shelf_height);
        let x = shelf
            .try_allocate(width, padding, self.config.width)
            .ok_or_else(|| {
                SubtitleError::Internal("glyph wider than atlas".to_string())
            })?;

        self.next_shelf_y += shelf_height;
        self.shelves.push(shelf);

        let slot = AtlasSlot {
            x,
            y: new_shelf_y,
            width,
            height,
            shelf_index: new_idx,
        };
        self.allocation_count += 1;
        Ok(slot)
    }

    /// Find the best-fit shelf for a glyph of the given height.
    ///
    /// "Best fit" means the shelf whose height is ≥ `glyph_height` and closest
    /// to `glyph_height` (minimises wasted vertical space).
    fn find_best_shelf(&self, glyph_height: u32) -> Option<usize> {
        let mut best_idx: Option<usize> = None;
        let mut best_waste = u32::MAX;

        for (idx, shelf) in self.shelves.iter().enumerate() {
            if shelf.height < glyph_height {
                continue;
            }
            let waste = shelf.height - glyph_height;
            if waste < best_waste {
                best_waste = waste;
                best_idx = Some(idx);
            }
        }

        best_idx
    }

    // ------------------------------------------------------------------
    // Cache interface
    // ------------------------------------------------------------------

    /// Look up a glyph key in the cache.
    ///
    /// Returns the slot if the glyph has already been rasterised and uploaded.
    pub fn get(&mut self, key: &GlyphKey) -> Option<&AtlasSlot> {
        if let Some(slot) = self.cache.get(key) {
            self.hit_count += 1;
            Some(slot)
        } else {
            None
        }
    }

    /// Insert a freshly rasterised glyph into the atlas and cache it.
    ///
    /// `bitmap` must be an 8-bit grayscale (alpha-only) bitmap of size
    /// `slot.width × slot.height`.
    ///
    /// # Errors
    ///
    /// Returns an error if the bitmap length does not match the slot dimensions.
    pub fn insert(
        &mut self,
        key: GlyphKey,
        slot: AtlasSlot,
        bitmap: &[u8],
    ) -> SubtitleResult<()> {
        let expected = (slot.width * slot.height) as usize;
        if bitmap.len() != expected {
            return Err(SubtitleError::Internal(format!(
                "bitmap length {} != expected {}",
                bitmap.len(),
                expected
            )));
        }

        // Copy bitmap into the RGBA atlas buffer (write into alpha channel,
        // fill RGB with white so callers can tint by modulating colour).
        let atlas_w = self.config.width as usize;
        for row in 0..slot.height as usize {
            for col in 0..slot.width as usize {
                let atlas_x = slot.x as usize + col;
                let atlas_y = slot.y as usize + row;
                let pixel_idx = (atlas_y * atlas_w + atlas_x) * 4;
                let alpha = bitmap[row * slot.width as usize + col];
                self.pixels[pixel_idx] = 255; // R
                self.pixels[pixel_idx + 1] = 255; // G
                self.pixels[pixel_idx + 2] = 255; // B
                self.pixels[pixel_idx + 3] = alpha; // A
            }
        }

        self.cache.insert(key, slot);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Read-only view of the atlas RGBA pixel buffer.
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Mutable view of the atlas RGBA pixel buffer.
    ///
    /// Callers may write directly into this buffer for advanced use cases
    /// (e.g. uploading pre-scaled bitmaps without the `insert` helper).
    pub fn pixels_mut(&mut self) -> &mut [u8] {
        &mut self.pixels
    }

    /// Return a mutable slice corresponding to the pixel rectangle of `slot`.
    ///
    /// Returns `None` if the slot lies outside the atlas bounds.
    #[must_use]
    pub fn pixel_slice_mut(&mut self, slot: &AtlasSlot) -> Option<PixelSliceMut<'_>> {
        let atlas_w = self.config.width;
        if slot.x + slot.width > atlas_w || slot.y + slot.height > self.config.height {
            return None;
        }
        Some(PixelSliceMut {
            pixels: &mut self.pixels,
            atlas_width: atlas_w,
            slot: *slot,
        })
    }

    /// Number of glyphs currently stored in the cache.
    #[must_use]
    pub fn cached_count(&self) -> usize {
        self.cache.len()
    }

    /// Total successful allocations (including cache-miss fills).
    #[must_use]
    pub fn allocation_count(&self) -> u64 {
        self.allocation_count
    }

    /// Number of cache hits (glyph found without re-rasterising).
    #[must_use]
    pub fn hit_count(&self) -> u64 {
        self.hit_count
    }

    /// Cache hit ratio (0.0 if no lookups have occurred).
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        let total = self.allocation_count + self.hit_count;
        if total == 0 {
            return 0.0;
        }
        self.hit_count as f64 / total as f64
    }

    /// Evict all cached glyphs and clear the pixel buffer, resetting the atlas.
    pub fn clear(&mut self) {
        self.pixels.iter_mut().for_each(|p| *p = 0);
        self.shelves.clear();
        self.next_shelf_y = 0;
        self.cache.clear();
        self.allocation_count = 0;
        self.hit_count = 0;
    }

    /// Number of shelves currently in use.
    #[must_use]
    pub fn shelf_count(&self) -> usize {
        self.shelves.len()
    }

    /// Remaining vertical space in pixels.
    #[must_use]
    pub fn remaining_height(&self) -> u32 {
        self.config.height.saturating_sub(self.next_shelf_y)
    }

    /// Atlas utilisation: fraction of the vertical space that has been used.
    #[must_use]
    pub fn vertical_utilisation(&self) -> f32 {
        if self.config.height == 0 {
            return 0.0;
        }
        self.next_shelf_y as f32 / self.config.height as f32
    }
}

// ============================================================================
// PixelSliceMut — helper for writing directly into an atlas slot
// ============================================================================

/// Helper for writing pixels into a specific slot of the atlas.
pub struct PixelSliceMut<'a> {
    pixels: &'a mut Vec<u8>,
    atlas_width: u32,
    slot: AtlasSlot,
}

impl<'a> PixelSliceMut<'a> {
    /// Write an RGBA pixel at `(col, row)` within the slot.
    ///
    /// Silently ignores out-of-bounds writes.
    pub fn set_pixel(&mut self, col: u32, row: u32, r: u8, g: u8, b: u8, a: u8) {
        if col >= self.slot.width || row >= self.slot.height {
            return;
        }
        let ax = self.slot.x + col;
        let ay = self.slot.y + row;
        let idx = ((ay * self.atlas_width + ax) * 4) as usize;
        if idx + 3 < self.pixels.len() {
            self.pixels[idx] = r;
            self.pixels[idx + 1] = g;
            self.pixels[idx + 2] = b;
            self.pixels[idx + 3] = a;
        }
    }

    /// Write a grayscale alpha value at `(col, row)` within the slot.
    pub fn set_alpha(&mut self, col: u32, row: u32, alpha: u8) {
        self.set_pixel(col, row, 255, 255, 255, alpha);
    }

    /// Slot width.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.slot.width
    }

    /// Slot height.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.slot.height
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_atlas() -> GlyphAtlas {
        GlyphAtlas::new(AtlasConfig {
            width: 128,
            height: 128,
            padding: 1,
        })
    }

    #[test]
    fn test_allocate_single_glyph() {
        let mut atlas = small_atlas();
        let slot = atlas.allocate(16, 20).expect("allocation should succeed");
        assert_eq!(slot.x, 0);
        assert_eq!(slot.y, 0);
        assert_eq!(slot.width, 16);
        assert_eq!(slot.height, 20);
    }

    #[test]
    fn test_allocate_multiple_glyphs_same_shelf() {
        let mut atlas = small_atlas();
        let s1 = atlas.allocate(10, 16).expect("first allocation");
        let s2 = atlas.allocate(12, 16).expect("second allocation");
        // Both should land on shelf 0 (same height).
        assert_eq!(s1.shelf_index, s2.shelf_index);
        // s2 should be to the right of s1.
        assert!(s2.x > s1.x);
    }

    #[test]
    fn test_allocate_new_shelf_for_taller_glyph() {
        let mut atlas = GlyphAtlas::new(AtlasConfig {
            width: 128,
            height: 256,
            padding: 1,
        });
        let s1 = atlas.allocate(32, 16).expect("small glyph");
        let s2 = atlas.allocate(32, 32).expect("tall glyph");
        // Tall glyph should open a new shelf below the first.
        assert_ne!(s1.shelf_index, s2.shelf_index);
    }

    #[test]
    fn test_atlas_full_error() {
        // 4×4 atlas, 1 pixel padding — can hold very few glyphs.
        let mut atlas = GlyphAtlas::new(AtlasConfig {
            width: 4,
            height: 4,
            padding: 0,
        });
        // First allocation fills the atlas.
        atlas.allocate(4, 4).expect("should fit");
        // Next allocation must fail.
        let result = atlas.allocate(4, 4);
        assert!(result.is_err(), "atlas should be full");
    }

    #[test]
    fn test_insert_and_get_from_cache() {
        let mut atlas = small_atlas();
        let key = GlyphKey::new('A', 24.0, 0);
        let slot = atlas.allocate(8, 10).expect("allocation");

        let bitmap = vec![128u8; (slot.width * slot.height) as usize];
        atlas.insert(key, slot, &bitmap).expect("insert");

        let result = atlas.get(&key);
        assert!(result.is_some(), "should find glyph in cache");
    }

    #[test]
    fn test_cache_hit_count() {
        let mut atlas = small_atlas();
        let key = GlyphKey::new('B', 16.0, 0);
        let slot = atlas.allocate(6, 8).expect("allocation");
        let bitmap = vec![255u8; (slot.width * slot.height) as usize];
        atlas.insert(key, slot, &bitmap).expect("insert");

        // Two cache lookups.
        atlas.get(&key);
        atlas.get(&key);

        assert_eq!(atlas.hit_count(), 2);
    }

    #[test]
    fn test_insert_wrong_bitmap_size_errors() {
        let mut atlas = small_atlas();
        let key = GlyphKey::new('C', 12.0, 0);
        let slot = atlas.allocate(8, 8).expect("allocation");
        // Provide wrong-size bitmap.
        let bad_bitmap = vec![0u8; 10]; // should be 64
        let result = atlas.insert(key, slot, &bad_bitmap);
        assert!(result.is_err());
    }

    #[test]
    fn test_atlas_clear_resets_state() {
        let mut atlas = small_atlas();
        atlas.allocate(20, 20).expect("allocation");
        atlas.clear();
        assert_eq!(atlas.shelf_count(), 0);
        assert_eq!(atlas.cached_count(), 0);
        assert_eq!(atlas.allocation_count(), 0);
        assert_eq!(atlas.remaining_height(), 128);
    }

    #[test]
    fn test_uv_coordinates() {
        let slot = AtlasSlot {
            x: 0,
            y: 0,
            width: 512,
            height: 512,
            shelf_index: 0,
        };
        assert!((slot.u0(1024) - 0.0).abs() < 1e-6);
        assert!((slot.u1(1024) - 0.5).abs() < 1e-6);
        assert!((slot.v0(1024) - 0.0).abs() < 1e-6);
        assert!((slot.v1(1024) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_vertical_utilisation() {
        let mut atlas = GlyphAtlas::new(AtlasConfig {
            width: 64,
            height: 64,
            padding: 0,
        });
        let util_before = atlas.vertical_utilisation();
        assert!((util_before - 0.0).abs() < 1e-6);

        atlas.allocate(8, 32).expect("allocation");
        let util_after = atlas.vertical_utilisation();
        assert!(util_after > 0.0);
        assert!(util_after <= 1.0);
    }

    #[test]
    fn test_pixel_slice_mut_writes() {
        let mut atlas = small_atlas();
        let slot = atlas.allocate(4, 4).expect("allocation");
        {
            let mut slice = atlas.pixel_slice_mut(&slot).expect("pixel slice");
            slice.set_alpha(0, 0, 200);
            slice.set_pixel(1, 1, 100, 150, 200, 255);
        }
        // Verify via pixels buffer.
        let w = atlas.config.width as usize;
        let idx = (slot.y as usize * w + slot.x as usize) * 4;
        assert_eq!(atlas.pixels()[idx + 3], 200, "alpha at (0,0) should be 200");
    }

    #[test]
    fn test_glyph_key_equality() {
        let k1 = GlyphKey::new('A', 24.0, 0);
        let k2 = GlyphKey::new('A', 24.0, 0);
        let k3 = GlyphKey::new('B', 24.0, 0);
        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
    }

    #[test]
    fn test_hit_ratio_zero_on_empty() {
        let atlas = small_atlas();
        assert!((atlas.hit_ratio() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_dimension_error() {
        let mut atlas = small_atlas();
        assert!(atlas.allocate(0, 16).is_err());
        assert!(atlas.allocate(16, 0).is_err());
    }

    #[test]
    fn test_shelf_count_increases() {
        let mut atlas = GlyphAtlas::new(AtlasConfig {
            width: 256,
            height: 256,
            padding: 1,
        });
        assert_eq!(atlas.shelf_count(), 0);
        atlas.allocate(16, 16).expect("first");
        assert_eq!(atlas.shelf_count(), 1);
        // Glyph with different height should open a second shelf.
        atlas.allocate(16, 32).expect("second");
        assert_eq!(atlas.shelf_count(), 2);
    }
}
