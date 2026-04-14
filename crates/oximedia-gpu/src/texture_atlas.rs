//! Texture atlas packing for GPU uploads.
//!
//! Implements a shelf-based rectangle packer (Next-Fit Decreasing Height / NFDH)
//! that compacts many small textures into a single large atlas texture, reducing
//! the number of GPU bind-group changes and improving cache coherence on the GPU.
//!
//! # Algorithm
//!
//! Shelves are horizontal strips.  Each incoming rectangle is placed on the
//! current shelf if it fits; otherwise a new shelf is opened below the
//! previous one.  Rectangles should be submitted in decreasing-height order
//! for best packing efficiency (the caller is responsible for sorting if
//! desired).
//!
//! # Example
//!
//! ```rust
//! use oximedia_gpu::texture_atlas::{AtlasConfig, TextureAtlas, AtlasRect};
//!
//! let cfg = AtlasConfig { width: 2048, height: 2048, padding: 1 };
//! let mut atlas = TextureAtlas::new(cfg);
//!
//! let id = atlas.insert(64, 64).expect("fits in atlas");
//! let rect = atlas.rect(id).expect("valid id");
//! assert_eq!(rect.w, 64);
//! assert_eq!(rect.h, 64);
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

use crate::{GpuError, Result};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for a [`TextureAtlas`].
#[derive(Debug, Clone)]
pub struct AtlasConfig {
    /// Total width of the backing atlas texture in texels.
    pub width: u32,
    /// Total height of the backing atlas texture in texels.
    pub height: u32,
    /// Number of transparent texels added around each sub-image to prevent
    /// bleeding during bilinear sampling.
    pub padding: u32,
}

impl Default for AtlasConfig {
    fn default() -> Self {
        Self {
            width: 2048,
            height: 2048,
            padding: 1,
        }
    }
}

// ── Rect ──────────────────────────────────────────────────────────────────────

/// An allocated sub-region within the atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtlasRect {
    /// Left edge in atlas coordinates (texels).
    pub x: u32,
    /// Top edge in atlas coordinates (texels).
    pub y: u32,
    /// Width of the allocated region in texels (without padding).
    pub w: u32,
    /// Height of the allocated region in texels (without padding).
    pub h: u32,
}

impl AtlasRect {
    /// UV coordinates as `(u_min, v_min, u_max, v_max)` in `[0, 1]` space
    /// given the atlas dimensions.
    #[must_use]
    pub fn uv_normalized(&self, atlas_w: u32, atlas_h: u32) -> (f32, f32, f32, f32) {
        let aw = atlas_w as f32;
        let ah = atlas_h as f32;
        (
            self.x as f32 / aw,
            self.y as f32 / ah,
            (self.x + self.w) as f32 / aw,
            (self.y + self.h) as f32 / ah,
        )
    }
}

// ── Shelf ─────────────────────────────────────────────────────────────────────

/// A horizontal strip inside the atlas where rectangles are packed
/// left-to-right.
#[derive(Debug)]
struct Shelf {
    /// Y coordinate of the top of this shelf (texels).
    y: u32,
    /// Height of the tallest rectangle placed on this shelf.
    height: u32,
    /// X cursor: the next free horizontal position on this shelf.
    cursor_x: u32,
}

impl Shelf {
    fn new(y: u32) -> Self {
        Self {
            y,
            height: 0,
            cursor_x: 0,
        }
    }

    /// Try to place a rectangle of `w × h` on this shelf.
    ///
    /// Returns the allocated x position, or `None` if there is no room.
    fn try_place(&mut self, w: u32, h: u32, atlas_width: u32, padding: u32) -> Option<u32> {
        let padded_w = w + padding * 2;
        if self.cursor_x + padded_w > atlas_width {
            return None;
        }
        let x = self.cursor_x + padding;
        self.cursor_x += padded_w;
        if h + padding * 2 > self.height {
            self.height = h + padding * 2;
        }
        Some(x)
    }
}

// ── Atlas ─────────────────────────────────────────────────────────────────────

/// Texture atlas packer.
///
/// Allocates rectangular regions inside a fixed-size backing texture using a
/// shelf-packing strategy.  The CPU-side pixel data is stored in `pixels` so
/// the atlas can be uploaded to the GPU at any time.
pub struct TextureAtlas {
    /// Configuration.
    pub config: AtlasConfig,
    /// CPU-side RGBA pixel data for the backing texture.
    ///
    /// Size = `config.width * config.height * 4` bytes.
    pub pixels: Vec<u8>,
    /// Active shelves.
    shelves: Vec<Shelf>,
    /// Allocated rectangles, keyed by opaque `u32` ID.
    allocations: HashMap<u32, AtlasRect>,
    /// Monotonically increasing ID counter.
    next_id: u32,
    /// Y coordinate of the next shelf's top edge.
    next_shelf_y: u32,
}

impl TextureAtlas {
    /// Create a new, empty atlas.
    #[must_use]
    pub fn new(config: AtlasConfig) -> Self {
        let pixel_count = (config.width * config.height * 4) as usize;
        Self {
            pixels: vec![0u8; pixel_count],
            shelves: Vec::new(),
            allocations: HashMap::new(),
            next_id: 0,
            next_shelf_y: 0,
            config,
        }
    }

    /// Insert a sub-texture of `w × h` texels.
    ///
    /// Returns an opaque allocation ID on success.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Internal`] if the atlas has no space left or if the
    /// requested dimensions exceed the atlas size.
    pub fn insert(&mut self, w: u32, h: u32) -> Result<u32> {
        if w == 0 || h == 0 {
            return Err(GpuError::InvalidDimensions {
                width: w,
                height: h,
            });
        }
        if w + self.config.padding * 2 > self.config.width
            || h + self.config.padding * 2 > self.config.height
        {
            return Err(GpuError::Internal(format!(
                "Texture {w}×{h} does not fit in atlas {}×{}",
                self.config.width, self.config.height
            )));
        }

        // Try to place on an existing shelf.
        let atlas_w = self.config.width;
        let padding = self.config.padding;
        for shelf in self.shelves.iter_mut() {
            if h <= shelf.height || shelf.height == 0 {
                if let Some(x) = shelf.try_place(w, h, atlas_w, padding) {
                    let y = shelf.y + padding;
                    let rect = AtlasRect { x, y, w, h };
                    let id = self.next_id;
                    self.next_id = self.next_id.wrapping_add(1);
                    self.allocations.insert(id, rect);
                    return Ok(id);
                }
            }
        }

        // Open a new shelf.
        let needed_h = h + padding * 2;
        if self.next_shelf_y + needed_h > self.config.height {
            return Err(GpuError::Internal(
                "Atlas is full — no vertical space remaining".to_string(),
            ));
        }

        let mut shelf = Shelf::new(self.next_shelf_y);
        let x = shelf
            .try_place(w, h, atlas_w, padding)
            .ok_or_else(|| GpuError::Internal("Failed to place rect on new shelf".to_string()))?;
        let y = shelf.y + padding;
        self.next_shelf_y += shelf.height;
        self.shelves.push(shelf);

        let rect = AtlasRect { x, y, w, h };
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.allocations.insert(id, rect);
        Ok(id)
    }

    /// Upload RGBA pixel data for the sub-texture identified by `id`.
    ///
    /// `data` must be exactly `w * h * 4` bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if `id` is invalid or `data` has wrong length.
    pub fn upload_pixels(&mut self, id: u32, data: &[u8]) -> Result<()> {
        let rect = *self
            .allocations
            .get(&id)
            .ok_or_else(|| GpuError::Internal(format!("Unknown atlas ID {id}")))?;

        let expected = (rect.w * rect.h * 4) as usize;
        if data.len() != expected {
            return Err(GpuError::InvalidBufferSize {
                expected,
                actual: data.len(),
            });
        }

        let atlas_stride = (self.config.width * 4) as usize;
        for row in 0..rect.h {
            let src_start = (row * rect.w * 4) as usize;
            let src_end = src_start + (rect.w * 4) as usize;
            let dst_start = ((rect.y + row) as usize * atlas_stride) + rect.x as usize * 4;
            let dst_end = dst_start + (rect.w * 4) as usize;
            self.pixels[dst_start..dst_end].copy_from_slice(&data[src_start..src_end]);
        }
        Ok(())
    }

    /// Return the allocated [`AtlasRect`] for `id`, or `None` if unknown.
    #[must_use]
    pub fn rect(&self, id: u32) -> Option<AtlasRect> {
        self.allocations.get(&id).copied()
    }

    /// Number of allocated sub-textures.
    #[must_use]
    pub fn len(&self) -> usize {
        self.allocations.len()
    }

    /// `true` if no sub-textures have been allocated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.allocations.is_empty()
    }

    /// Utilisation as a fraction in `[0, 1]` (pixels used / total pixels).
    #[must_use]
    pub fn utilisation(&self) -> f32 {
        let total = (self.config.width * self.config.height) as f32;
        if total <= 0.0 {
            return 0.0;
        }
        let used: u32 = self.allocations.values().map(|r| r.w * r.h).sum();
        used as f32 / total
    }

    /// Clear all allocations, retaining the backing pixel buffer.
    pub fn clear(&mut self) {
        self.shelves.clear();
        self.allocations.clear();
        self.next_id = 0;
        self.next_shelf_y = 0;
        // Zero out pixel data.
        for b in self.pixels.iter_mut() {
            *b = 0;
        }
    }

    /// Atlas width in texels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.config.width
    }

    /// Atlas height in texels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.config.height
    }
}

// ── TextureAtlasPacker ────────────────────────────────────────────────────────

/// Simplified packer API that wraps [`TextureAtlas`].
///
/// Accepts a batch of `(width, height)` sprites and returns their allocated
/// `(x, y, w, h)` rectangles in the same order.
///
/// # Example
///
/// ```rust
/// use oximedia_gpu::texture_atlas::TextureAtlasPacker;
///
/// let mut packer = TextureAtlasPacker::new(512, 512);
/// let rects = packer.pack(&[(64, 64), (32, 32)]);
/// assert_eq!(rects.len(), 2);
/// ```
pub struct TextureAtlasPacker {
    atlas: TextureAtlas,
}

impl TextureAtlasPacker {
    /// Create a packer backed by an atlas of `max_w × max_h` texels.
    #[must_use]
    pub fn new(max_w: u32, max_h: u32) -> Self {
        Self {
            atlas: TextureAtlas::new(AtlasConfig {
                width: max_w,
                height: max_h,
                padding: 1,
            }),
        }
    }

    /// Pack a slice of `(width, height)` sprites.
    ///
    /// Returns a `Vec` of `(x, y, w, h)` tuples in the same order as the
    /// input.  Sprites that do not fit are represented by `(0, 0, 0, 0)`.
    #[must_use]
    pub fn pack(&mut self, sprites: &[(u32, u32)]) -> Vec<(u32, u32, u32, u32)> {
        sprites
            .iter()
            .map(|&(w, h)| match self.atlas.insert(w, h) {
                Ok(id) => {
                    let r = self.atlas.rect(id).unwrap_or(AtlasRect {
                        x: 0,
                        y: 0,
                        w: 0,
                        h: 0,
                    });
                    (r.x, r.y, r.w, r.h)
                }
                Err(_) => (0, 0, 0, 0),
            })
            .collect()
    }

    /// Total number of sprites currently packed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.atlas.len()
    }

    /// `true` if no sprites have been packed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.atlas.is_empty()
    }

    /// Current utilisation fraction `[0, 1]`.
    #[must_use]
    pub fn utilisation(&self) -> f32 {
        self.atlas.utilisation()
    }

    /// Reset the packer, discarding all packed sprites.
    pub fn clear(&mut self) {
        self.atlas.clear();
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_atlas() -> TextureAtlas {
        TextureAtlas::new(AtlasConfig {
            width: 256,
            height: 256,
            padding: 1,
        })
    }

    #[test]
    fn test_insert_single_rect() {
        let mut atlas = small_atlas();
        let id = atlas.insert(64, 64).expect("should succeed");
        let rect = atlas.rect(id).expect("should have rect");
        assert_eq!(rect.w, 64);
        assert_eq!(rect.h, 64);
        assert_eq!(atlas.len(), 1);
    }

    #[test]
    fn test_insert_multiple_rects() {
        let mut atlas = small_atlas();
        let ids: Vec<u32> = (0..4)
            .map(|_| atlas.insert(32, 32).expect("fits"))
            .collect();
        assert_eq!(ids.len(), 4);
        assert_eq!(atlas.len(), 4);
        // All IDs unique
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), 4);
    }

    #[test]
    fn test_rects_do_not_overlap() {
        let mut atlas = small_atlas();
        let mut rects = Vec::new();
        for _ in 0..9 {
            let id = atlas.insert(40, 40).expect("fits");
            rects.push(atlas.rect(id).expect("valid"));
        }
        // Check pairwise non-overlap
        for i in 0..rects.len() {
            for j in (i + 1)..rects.len() {
                let a = rects[i];
                let b = rects[j];
                let overlap_x = a.x < b.x + b.w && a.x + a.w > b.x;
                let overlap_y = a.y < b.y + b.h && a.y + a.h > b.y;
                assert!(
                    !(overlap_x && overlap_y),
                    "Rects {i} and {j} overlap: {:?} vs {:?}",
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn test_insert_too_large_returns_error() {
        let mut atlas = small_atlas();
        let result = atlas.insert(512, 512);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_dimension_returns_error() {
        let mut atlas = small_atlas();
        assert!(atlas.insert(0, 32).is_err());
        assert!(atlas.insert(32, 0).is_err());
    }

    #[test]
    fn test_upload_pixels_correct_dimensions() {
        let mut atlas = small_atlas();
        let id = atlas.insert(4, 4).expect("fits");
        let data = vec![255u8; 4 * 4 * 4]; // 4×4 RGBA
        atlas.upload_pixels(id, &data).expect("upload ok");
    }

    #[test]
    fn test_upload_pixels_wrong_size_returns_error() {
        let mut atlas = small_atlas();
        let id = atlas.insert(4, 4).expect("fits");
        let bad_data = vec![0u8; 10]; // wrong size
        assert!(atlas.upload_pixels(id, &bad_data).is_err());
    }

    #[test]
    fn test_upload_pixels_invalid_id_returns_error() {
        let mut atlas = small_atlas();
        let data = vec![0u8; 16];
        assert!(atlas.upload_pixels(9999, &data).is_err());
    }

    #[test]
    fn test_clear_resets_state() {
        let mut atlas = small_atlas();
        atlas.insert(64, 64).expect("ok");
        atlas.clear();
        assert_eq!(atlas.len(), 0);
        assert!(atlas.is_empty());
        // After clear we can use the full space again
        let id2 = atlas.insert(128, 128).expect("should fit after clear");
        assert!(atlas.rect(id2).is_some());
    }

    #[test]
    fn test_utilisation_increases_with_inserts() {
        let mut atlas = small_atlas();
        let u0 = atlas.utilisation();
        atlas.insert(64, 64).expect("ok");
        let u1 = atlas.utilisation();
        assert!(u1 > u0, "utilisation should increase");
    }

    #[test]
    fn test_uv_normalized_range() {
        let mut atlas = small_atlas();
        let id = atlas.insert(64, 64).expect("ok");
        let rect = atlas.rect(id).expect("valid");
        let (u0, v0, u1, v1) = rect.uv_normalized(atlas.width(), atlas.height());
        assert!(u0 >= 0.0 && u0 < 1.0);
        assert!(v0 >= 0.0 && v0 < 1.0);
        assert!(u1 > u0 && u1 <= 1.0);
        assert!(v1 > v0 && v1 <= 1.0);
    }

    // ── TextureAtlasPacker tests ──────────────────────────────────────────────

    #[test]
    fn test_packer_pack_single_sprite() {
        let mut packer = TextureAtlasPacker::new(512, 512);
        let rects = packer.pack(&[(64, 64)]);
        assert_eq!(rects.len(), 1);
        let (_, _, w, h) = rects[0];
        assert_eq!(w, 64);
        assert_eq!(h, 64);
    }

    #[test]
    fn test_packer_pack_multiple_sprites() {
        let mut packer = TextureAtlasPacker::new(512, 512);
        let sprites = vec![(32, 32), (64, 64), (16, 16)];
        let rects = packer.pack(&sprites);
        assert_eq!(rects.len(), 3);
        assert_eq!(packer.len(), 3);
    }

    #[test]
    fn test_packer_oversized_sprite_returns_zero_rect() {
        let mut packer = TextureAtlasPacker::new(64, 64);
        let rects = packer.pack(&[(256, 256)]);
        assert_eq!(rects[0], (0, 0, 0, 0));
    }

    #[test]
    fn test_packer_clear_resets() {
        let mut packer = TextureAtlasPacker::new(256, 256);
        let _ = packer.pack(&[(32, 32)]);
        packer.clear();
        assert!(packer.is_empty());
    }

    #[test]
    fn test_packer_utilisation_increases() {
        let mut packer = TextureAtlasPacker::new(256, 256);
        let u0 = packer.utilisation();
        let _ = packer.pack(&[(64, 64)]);
        assert!(packer.utilisation() > u0);
    }

    #[test]
    fn test_pixel_data_written_to_correct_position() {
        let mut atlas = TextureAtlas::new(AtlasConfig {
            width: 8,
            height: 8,
            padding: 0,
        });
        let id = atlas.insert(2, 2).expect("fits");
        let rect = atlas.rect(id).expect("valid");
        // Fill with 0xFF bytes
        let data = vec![0xFFu8; 2 * 2 * 4];
        atlas.upload_pixels(id, &data).expect("ok");
        // Check first pixel in atlas
        let idx = (rect.y as usize * 8 + rect.x as usize) * 4;
        assert_eq!(atlas.pixels[idx], 0xFF);
    }
}
