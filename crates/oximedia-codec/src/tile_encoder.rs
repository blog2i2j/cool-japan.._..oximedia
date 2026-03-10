//! Tile-based parallel frame encoding for OxiMedia codecs.
//!
//! This module provides pixel-level infrastructure for splitting raw video
//! frames into rectangular tiles, processing them concurrently, and
//! reassembling the result into a complete frame.
//!
//! Unlike [`crate::tile`] which works with [`crate::frame::VideoFrame`] and
//! codec-level bitstream output, this module operates on raw `&[u8]` / `Vec<u8>`
//! pixel buffers and is therefore codec-agnostic.
//!
//! # Architecture
//!
//! ```text
//! TileConfig  ─── tile grid parameters (cols, rows, frame size)
//!     │
//!     ▼
//! TileLayout  ─── pre-computed TileRegion grid (handles remainder pixels)
//!     │
//!     ▼
//! ParallelTileEncoder ─── split_frame → parallel encode_fn → merge_tiles
//! ```
//!
//! # Example
//!
//! ```
//! use oximedia_codec::tile_encoder::{TileConfig, ParallelTileEncoder};
//!
//! let config = TileConfig::new()
//!     .tile_cols(2)
//!     .tile_rows(2)
//!     .frame_width(64)
//!     .frame_height(64);
//!
//! let encoder = ParallelTileEncoder::new(config);
//!
//! // Create a simple 64×64 RGB frame (3 channels).
//! let frame: Vec<u8> = (0u8..=255).cycle().take(64 * 64 * 3).collect();
//!
//! let tiles = encoder.split_frame(&frame, 3);
//! assert_eq!(tiles.len(), 4);
//!
//! let merged = ParallelTileEncoder::merge_tiles(&tiles, 64, 64, 3);
//! assert_eq!(merged, frame);
//! ```

use rayon::prelude::*;
use std::ops::Range;

// =============================================================================
// TileConfig
// =============================================================================

/// Configuration for the tile grid and frame dimensions.
///
/// Use the builder-pattern methods to configure:
///
/// ```
/// use oximedia_codec::tile_encoder::TileConfig;
///
/// let cfg = TileConfig::new()
///     .tile_cols(4)
///     .tile_rows(4)
///     .num_threads(8)
///     .frame_width(1920)
///     .frame_height(1080);
///
/// assert_eq!(cfg.tile_cols, 4);
/// assert_eq!(cfg.num_threads, 8);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileConfig {
    /// Number of tile columns (1–64).
    pub tile_cols: u32,
    /// Number of tile rows (1–64).
    pub tile_rows: u32,
    /// Worker threads for parallel encoding (0 = use Rayon pool size).
    pub num_threads: usize,
    /// Frame width in pixels.
    pub frame_width: u32,
    /// Frame height in pixels.
    pub frame_height: u32,
}

impl TileConfig {
    /// Create a `TileConfig` with default values.
    ///
    /// Defaults: 1 column, 1 row, 0 threads (auto), 0×0 frame.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of tile columns (1–64).
    #[must_use]
    pub fn tile_cols(mut self, cols: u32) -> Self {
        self.tile_cols = cols.clamp(1, 64);
        self
    }

    /// Set the number of tile rows (1–64).
    #[must_use]
    pub fn tile_rows(mut self, rows: u32) -> Self {
        self.tile_rows = rows.clamp(1, 64);
        self
    }

    /// Set the worker thread count (0 = Rayon auto).
    #[must_use]
    pub fn num_threads(mut self, threads: usize) -> Self {
        self.num_threads = threads;
        self
    }

    /// Set the frame width in pixels.
    #[must_use]
    pub fn frame_width(mut self, width: u32) -> Self {
        self.frame_width = width;
        self
    }

    /// Set the frame height in pixels.
    #[must_use]
    pub fn frame_height(mut self, height: u32) -> Self {
        self.frame_height = height;
        self
    }

    /// Effective thread count (resolves 0 to the Rayon pool size).
    #[must_use]
    pub fn thread_count(&self) -> usize {
        if self.num_threads == 0 {
            rayon::current_num_threads()
        } else {
            self.num_threads
        }
    }
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            tile_cols: 1,
            tile_rows: 1,
            num_threads: 0,
            frame_width: 0,
            frame_height: 0,
        }
    }
}

// =============================================================================
// TileRegion
// =============================================================================

/// Pixel coordinates and dimensions of a single tile within a frame.
///
/// ```
/// use oximedia_codec::tile_encoder::TileRegion;
///
/// let region = TileRegion::new(1, 0, 512, 0, 512, 288);
/// assert_eq!(region.area(), 512 * 288);
/// assert!(region.contains(600, 100));
/// assert!(!region.contains(200, 100)); // left of tile
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileRegion {
    /// Tile column index (0-based).
    pub col: u32,
    /// Tile row index (0-based).
    pub row: u32,
    /// X pixel offset from the left of the frame.
    pub x: u32,
    /// Y pixel offset from the top of the frame.
    pub y: u32,
    /// Tile width in pixels.
    pub width: u32,
    /// Tile height in pixels.
    pub height: u32,
}

impl TileRegion {
    /// Create a new `TileRegion`.
    #[must_use]
    pub const fn new(col: u32, row: u32, x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            col,
            row,
            x,
            y,
            width,
            height,
        }
    }

    /// Area of this tile in pixels.
    #[must_use]
    pub const fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Returns `true` if the pixel `(px, py)` falls within this tile.
    #[must_use]
    pub const fn contains(&self, px: u32, py: u32) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }

    /// Pixel column range `x..(x + width)`.
    #[must_use]
    pub fn pixel_range_x(&self) -> Range<u32> {
        self.x..(self.x + self.width)
    }

    /// Pixel row range `y..(y + height)`.
    #[must_use]
    pub fn pixel_range_y(&self) -> Range<u32> {
        self.y..(self.y + self.height)
    }
}

// =============================================================================
// TileLayout
// =============================================================================

/// A grid of [`TileRegion`]s computed from a [`TileConfig`].
///
/// The last tile in each row/column absorbs any remainder pixels so the union
/// of all tiles exactly covers the full frame with no overlap.
///
/// ```
/// use oximedia_codec::tile_encoder::{TileConfig, TileLayout};
///
/// let cfg = TileConfig::new()
///     .tile_cols(2)
///     .tile_rows(2)
///     .frame_width(100)
///     .frame_height(100);
///
/// let layout = TileLayout::new(cfg);
/// assert_eq!(layout.tile_count(), 4);
///
/// // All tiles together cover 100×100 pixels.
/// let total: u64 = layout.tiles().iter().map(|t| t.area()).sum();
/// assert_eq!(total, 100 * 100);
/// ```
#[derive(Clone, Debug)]
pub struct TileLayout {
    /// The configuration used to build this layout.
    pub config: TileConfig,
    /// All tile regions in raster order (row-major).
    pub tiles: Vec<TileRegion>,
}

impl TileLayout {
    /// Compute a `TileLayout` from `config`.
    ///
    /// Tile boundaries are computed as `frame_width / tile_cols` (integer
    /// division); the last column and last row absorb the remainder pixels.
    #[must_use]
    pub fn new(config: TileConfig) -> Self {
        let cols = config.tile_cols.max(1);
        let rows = config.tile_rows.max(1);
        let fw = config.frame_width;
        let fh = config.frame_height;

        // Nominal tile sizes (last tile gets the remainder).
        let nominal_tw = fw / cols;
        let nominal_th = fh / rows;

        let mut tiles = Vec::with_capacity((cols * rows) as usize);

        for row in 0..rows {
            for col in 0..cols {
                let x = col * nominal_tw;
                let y = row * nominal_th;

                let width = if col == cols - 1 {
                    fw.saturating_sub(x)
                } else {
                    nominal_tw
                };
                let height = if row == rows - 1 {
                    fh.saturating_sub(y)
                } else {
                    nominal_th
                };

                tiles.push(TileRegion::new(col, row, x, y, width, height));
            }
        }

        Self { config, tiles }
    }

    /// Total number of tiles.
    #[must_use]
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Return the tile at grid position `(col, row)`, or `None` if out of bounds.
    #[must_use]
    pub fn get_tile(&self, col: u32, row: u32) -> Option<&TileRegion> {
        let cols = self.config.tile_cols;
        let rows = self.config.tile_rows;
        if col >= cols || row >= rows {
            return None;
        }
        self.tiles.get((row * cols + col) as usize)
    }

    /// All tile regions in raster order.
    #[must_use]
    pub fn tiles(&self) -> &[TileRegion] {
        &self.tiles
    }

    /// Find which tile contains the pixel `(px, py)`.
    ///
    /// Returns `None` if the pixel is outside the frame.
    #[must_use]
    pub fn tile_for_pixel(&self, px: u32, py: u32) -> Option<&TileRegion> {
        self.tiles.iter().find(|t| t.contains(px, py))
    }
}

// =============================================================================
// TileBuffer
// =============================================================================

/// Raw pixel data extracted from (or destined for) a single tile.
///
/// ```
/// use oximedia_codec::tile_encoder::{TileRegion, TileBuffer};
///
/// let region = TileRegion::new(0, 0, 0, 0, 4, 4);
/// let buf = TileBuffer::new(region, 3); // 3 channels (RGB)
/// assert_eq!(buf.data.len(), 4 * 4 * 3);
/// assert_eq!(buf.stride, 4 * 3);
/// ```
#[derive(Clone, Debug)]
pub struct TileBuffer {
    /// Spatial position of this tile in the frame.
    pub region: TileRegion,
    /// Raw pixel bytes for this tile (row-major, tightly packed).
    pub data: Vec<u8>,
    /// Row stride in bytes (`width * channels`).
    pub stride: usize,
    /// Bytes per pixel.
    pub channels: u8,
}

impl TileBuffer {
    /// Allocate an all-zero `TileBuffer` for `region` with `channels` bytes per pixel.
    #[must_use]
    pub fn new(region: TileRegion, channels: u8) -> Self {
        let ch = channels as usize;
        let stride = region.width as usize * ch;
        let data = vec![0u8; region.height as usize * stride];
        Self {
            region,
            data,
            stride,
            channels,
        }
    }

    /// Copy the tile's pixels from `frame` (a packed, row-major buffer).
    ///
    /// `frame_stride` is the number of bytes per row in the full frame
    /// (i.e. `frame_width * channels`).
    pub fn extract_from_frame(&mut self, frame: &[u8], frame_stride: usize) {
        let ch = self.channels as usize;
        let x_byte = self.region.x as usize * ch;
        let w_bytes = self.region.width as usize * ch;

        for row in 0..self.region.height as usize {
            let frame_row_start = (self.region.y as usize + row) * frame_stride + x_byte;
            let tile_row_start = row * self.stride;

            let src_end = (frame_row_start + w_bytes).min(frame.len());
            let copy_len = src_end.saturating_sub(frame_row_start);

            self.data[tile_row_start..tile_row_start + copy_len]
                .copy_from_slice(&frame[frame_row_start..src_end]);
        }
    }

    /// Write this tile's pixels back into `frame`.
    ///
    /// `frame_stride` must match the full frame's row stride.
    pub fn write_to_frame(&self, frame: &mut [u8], frame_stride: usize) {
        let ch = self.channels as usize;
        let x_byte = self.region.x as usize * ch;
        let w_bytes = self.region.width as usize * ch;

        for row in 0..self.region.height as usize {
            let frame_row_start = (self.region.y as usize + row) * frame_stride + x_byte;
            let tile_row_start = row * self.stride;

            let dst_end = (frame_row_start + w_bytes).min(frame.len());
            let copy_len = dst_end.saturating_sub(frame_row_start);

            frame[frame_row_start..frame_row_start + copy_len]
                .copy_from_slice(&self.data[tile_row_start..tile_row_start + copy_len]);
        }
    }
}

// =============================================================================
// ParallelTileEncoder
// =============================================================================

/// Splits a raw pixel frame into tiles, processes them in parallel, and
/// reassembles the result.
///
/// # Example
///
/// ```
/// use oximedia_codec::tile_encoder::{TileConfig, ParallelTileEncoder};
///
/// let config = TileConfig::new()
///     .tile_cols(2)
///     .tile_rows(2)
///     .frame_width(64)
///     .frame_height(64);
///
/// let encoder = ParallelTileEncoder::new(config);
///
/// let frame: Vec<u8> = (0u8..=255).cycle().take(64 * 64 * 3).collect();
/// let tiles = encoder.split_frame(&frame, 3);
/// assert_eq!(tiles.len(), 4);
///
/// // Identity encode: return each tile unchanged.
/// let processed = encoder
///     .encode_tiles_parallel(tiles, |tile| Ok(tile))
///     ?;
///
/// let merged = ParallelTileEncoder::merge_tiles(&processed, 64, 64, 3);
/// assert_eq!(merged, frame);
/// ```
pub struct ParallelTileEncoder {
    /// Pre-computed tile layout.
    pub layout: TileLayout,
}

impl ParallelTileEncoder {
    /// Create a `ParallelTileEncoder` from `config`.
    #[must_use]
    pub fn new(config: TileConfig) -> Self {
        Self {
            layout: TileLayout::new(config),
        }
    }

    /// Split `frame` into [`TileBuffer`]s, one per tile in the layout.
    ///
    /// `channels` is the number of bytes per pixel in `frame`.
    #[must_use]
    pub fn split_frame(&self, frame: &[u8], channels: u8) -> Vec<TileBuffer> {
        let fw = self.layout.config.frame_width;
        let frame_stride = fw as usize * channels as usize;

        self.layout
            .tiles
            .iter()
            .map(|region| {
                let mut buf = TileBuffer::new(region.clone(), channels);
                buf.extract_from_frame(frame, frame_stride);
                buf
            })
            .collect()
    }

    /// Merge a slice of [`TileBuffer`]s back into a complete frame.
    ///
    /// The returned `Vec<u8>` has `frame_width * frame_height * channels` bytes.
    #[must_use]
    pub fn merge_tiles(
        tiles: &[TileBuffer],
        frame_width: u32,
        frame_height: u32,
        channels: u8,
    ) -> Vec<u8> {
        let ch = channels as usize;
        let frame_stride = frame_width as usize * ch;
        let frame_size = frame_height as usize * frame_stride;
        let mut frame = vec![0u8; frame_size];

        for tile in tiles {
            tile.write_to_frame(&mut frame, frame_stride);
        }

        frame
    }

    /// Process `tiles` in parallel using `encode_fn`.
    ///
    /// Each tile is passed by value to `encode_fn`.  The closure must return
    /// either a (possibly modified) [`TileBuffer`] or an error string.
    ///
    /// Uses Rayon for parallel execution.  The output order matches the input
    /// order (raster order when produced by `split_frame`).
    ///
    /// # Errors
    ///
    /// Returns the first error string produced by any invocation of
    /// `encode_fn`.
    pub fn encode_tiles_parallel<F>(
        &self,
        tiles: Vec<TileBuffer>,
        encode_fn: F,
    ) -> Result<Vec<TileBuffer>, String>
    where
        F: Fn(TileBuffer) -> Result<TileBuffer, String> + Send + Sync,
    {
        let results: Vec<Result<TileBuffer, String>> =
            tiles.into_par_iter().map(|tile| encode_fn(tile)).collect();

        let mut out = Vec::with_capacity(results.len());
        for r in results {
            out.push(r?);
        }
        Ok(out)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // TileConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_tile_config_default() {
        let cfg = TileConfig::default();
        assert_eq!(cfg.tile_cols, 1);
        assert_eq!(cfg.tile_rows, 1);
        assert_eq!(cfg.num_threads, 0);
        assert_eq!(cfg.frame_width, 0);
        assert_eq!(cfg.frame_height, 0);
    }

    #[test]
    fn test_tile_config_builder() {
        let cfg = TileConfig::new()
            .tile_cols(4)
            .tile_rows(3)
            .num_threads(8)
            .frame_width(1920)
            .frame_height(1080);

        assert_eq!(cfg.tile_cols, 4);
        assert_eq!(cfg.tile_rows, 3);
        assert_eq!(cfg.num_threads, 8);
        assert_eq!(cfg.frame_width, 1920);
        assert_eq!(cfg.frame_height, 1080);
    }

    #[test]
    fn test_tile_config_clamp_cols() {
        // Values > 64 are clamped.
        let cfg = TileConfig::new().tile_cols(100);
        assert_eq!(cfg.tile_cols, 64);
    }

    #[test]
    fn test_tile_config_thread_count_auto() {
        let cfg = TileConfig::new().num_threads(0);
        assert!(cfg.thread_count() >= 1);
    }

    #[test]
    fn test_tile_config_thread_count_explicit() {
        let cfg = TileConfig::new().num_threads(4);
        assert_eq!(cfg.thread_count(), 4);
    }

    // -----------------------------------------------------------------------
    // TileRegion
    // -----------------------------------------------------------------------

    #[test]
    fn test_tile_region_area() {
        let r = TileRegion::new(0, 0, 0, 0, 100, 50);
        assert_eq!(r.area(), 5000);
    }

    #[test]
    fn test_tile_region_contains() {
        let r = TileRegion::new(1, 0, 50, 0, 50, 50);
        assert!(r.contains(50, 0));
        assert!(r.contains(99, 49));
        assert!(!r.contains(49, 0)); // left of region
        assert!(!r.contains(100, 0)); // right boundary (exclusive)
        assert!(!r.contains(50, 50)); // bottom boundary (exclusive)
    }

    #[test]
    fn test_tile_region_pixel_ranges() {
        let r = TileRegion::new(0, 1, 0, 100, 200, 80);
        assert_eq!(r.pixel_range_x(), 0..200);
        assert_eq!(r.pixel_range_y(), 100..180);
    }

    // -----------------------------------------------------------------------
    // TileLayout – divisible dimensions
    // -----------------------------------------------------------------------

    #[test]
    fn test_tile_layout_2x2_divisible() {
        let cfg = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(100)
            .frame_height(100);

        let layout = TileLayout::new(cfg);
        assert_eq!(layout.tile_count(), 4);

        // All tiles should be 50×50 for an evenly-divisible frame.
        for tile in layout.tiles() {
            assert_eq!(tile.width, 50);
            assert_eq!(tile.height, 50);
        }

        // Total area must equal frame area.
        let total: u64 = layout.tiles().iter().map(|t| t.area()).sum();
        assert_eq!(total, 100 * 100);
    }

    #[test]
    fn test_tile_layout_get_tile() {
        let cfg = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(100)
            .frame_height(100);

        let layout = TileLayout::new(cfg);

        let tl = layout.get_tile(0, 0).expect("should succeed");
        assert_eq!((tl.x, tl.y), (0, 0));

        let tr = layout.get_tile(1, 0).expect("should succeed");
        assert_eq!(tr.x, 50);

        let bl = layout.get_tile(0, 1).expect("should succeed");
        assert_eq!(bl.y, 50);

        assert!(layout.get_tile(2, 0).is_none());
    }

    // -----------------------------------------------------------------------
    // TileLayout – non-divisible dimensions
    // -----------------------------------------------------------------------

    #[test]
    fn test_tile_layout_2x2_non_divisible() {
        // 101×101 with 2×2 tiles: nominal 50×50, last col/row gets remainder.
        let cfg = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(101)
            .frame_height(101);

        let layout = TileLayout::new(cfg);
        assert_eq!(layout.tile_count(), 4);

        // Top-left: 50×50
        let tl = layout.get_tile(0, 0).expect("should succeed");
        assert_eq!(tl.width, 50);
        assert_eq!(tl.height, 50);

        // Top-right: 51×50 (gets the 1-pixel remainder in x)
        let tr = layout.get_tile(1, 0).expect("should succeed");
        assert_eq!(tr.width, 51);
        assert_eq!(tr.height, 50);

        // Bottom-left: 50×51
        let bl = layout.get_tile(0, 1).expect("should succeed");
        assert_eq!(bl.width, 50);
        assert_eq!(bl.height, 51);

        // Bottom-right: 51×51
        let br = layout.get_tile(1, 1).expect("should succeed");
        assert_eq!(br.width, 51);
        assert_eq!(br.height, 51);

        // Total area == 101×101
        let total: u64 = layout.tiles().iter().map(|t| t.area()).sum();
        assert_eq!(total, 101 * 101);
    }

    #[test]
    fn test_tile_layout_non_divisible_coverage() {
        // Verify every pixel is covered exactly once.
        let fw = 97u32;
        let fh = 83u32;
        let cfg = TileConfig::new()
            .tile_cols(3)
            .tile_rows(3)
            .frame_width(fw)
            .frame_height(fh);

        let layout = TileLayout::new(cfg);
        let mut counts = vec![0u32; (fw * fh) as usize];

        for tile in layout.tiles() {
            for py in tile.pixel_range_y() {
                for px in tile.pixel_range_x() {
                    counts[(py * fw + px) as usize] += 1;
                }
            }
        }

        assert!(
            counts.iter().all(|&c| c == 1),
            "some pixels are covered 0 or 2+ times"
        );
    }

    // -----------------------------------------------------------------------
    // TileLayout – tile_for_pixel
    // -----------------------------------------------------------------------

    #[test]
    fn test_tile_for_pixel() {
        let cfg = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(100)
            .frame_height(100);

        let layout = TileLayout::new(cfg);

        let t = layout.tile_for_pixel(25, 25).expect("should succeed");
        assert_eq!((t.col, t.row), (0, 0));

        let t = layout.tile_for_pixel(75, 25).expect("should succeed");
        assert_eq!((t.col, t.row), (1, 0));

        let t = layout.tile_for_pixel(25, 75).expect("should succeed");
        assert_eq!((t.col, t.row), (0, 1));

        let t = layout.tile_for_pixel(75, 75).expect("should succeed");
        assert_eq!((t.col, t.row), (1, 1));

        // Out-of-frame pixel.
        assert!(layout.tile_for_pixel(200, 200).is_none());
    }

    // -----------------------------------------------------------------------
    // TileBuffer
    // -----------------------------------------------------------------------

    #[test]
    fn test_tile_buffer_new() {
        let region = TileRegion::new(0, 0, 0, 0, 8, 6);
        let buf = TileBuffer::new(region, 3);
        assert_eq!(buf.stride, 8 * 3);
        assert_eq!(buf.data.len(), 8 * 6 * 3);
        assert!(buf.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_tile_buffer_extract() {
        // 4×4 single-channel frame: pixels 0..16
        let frame: Vec<u8> = (0u8..16).collect();
        let region = TileRegion::new(0, 0, 1, 1, 2, 2); // 2×2 tile at offset (1,1)
        let mut buf = TileBuffer::new(region, 1);
        buf.extract_from_frame(&frame, 4); // frame stride = 4

        // Row 1, col 1 → index 5; row 1, col 2 → index 6
        // Row 2, col 1 → index 9; row 2, col 2 → index 10
        assert_eq!(buf.data, vec![5, 6, 9, 10]);
    }

    #[test]
    fn test_tile_buffer_write_back() {
        let region = TileRegion::new(0, 0, 1, 1, 2, 2);
        let mut buf = TileBuffer::new(region, 1);
        buf.data = vec![5, 6, 9, 10];

        let mut frame = vec![0u8; 16];
        buf.write_to_frame(&mut frame, 4);

        assert_eq!(frame[5], 5);
        assert_eq!(frame[6], 6);
        assert_eq!(frame[9], 9);
        assert_eq!(frame[10], 10);
        // Other pixels untouched.
        assert_eq!(frame[0], 0);
    }

    // -----------------------------------------------------------------------
    // ParallelTileEncoder – split and merge roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_split_merge_roundtrip_divisible() {
        let fw = 64u32;
        let fh = 64u32;
        let channels = 3u8;

        let config = TileConfig::new()
            .tile_cols(4)
            .tile_rows(4)
            .frame_width(fw)
            .frame_height(fh);

        let encoder = ParallelTileEncoder::new(config);

        // Create a unique frame.
        let frame: Vec<u8> = (0u8..=255)
            .cycle()
            .take((fw * fh * channels as u32) as usize)
            .collect();
        let tiles = encoder.split_frame(&frame, channels);
        assert_eq!(tiles.len(), 16);

        let merged = ParallelTileEncoder::merge_tiles(&tiles, fw, fh, channels);
        assert_eq!(merged, frame, "roundtrip failed for divisible dimensions");
    }

    #[test]
    fn test_split_merge_roundtrip_non_divisible() {
        let fw = 101u32;
        let fh = 99u32;
        let channels = 1u8;

        let config = TileConfig::new()
            .tile_cols(3)
            .tile_rows(3)
            .frame_width(fw)
            .frame_height(fh);

        let encoder = ParallelTileEncoder::new(config);

        let frame: Vec<u8> = (0u8..=255).cycle().take((fw * fh) as usize).collect();
        let tiles = encoder.split_frame(&frame, channels);

        let merged = ParallelTileEncoder::merge_tiles(&tiles, fw, fh, channels);
        assert_eq!(
            merged, frame,
            "roundtrip failed for non-divisible dimensions"
        );
    }

    // -----------------------------------------------------------------------
    // ParallelTileEncoder – encode_tiles_parallel
    // -----------------------------------------------------------------------

    #[test]
    fn test_encode_tiles_parallel_identity() {
        let fw = 64u32;
        let fh = 64u32;
        let channels = 3u8;

        let config = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(fw)
            .frame_height(fh);

        let encoder = ParallelTileEncoder::new(config);

        let frame: Vec<u8> = (0u8..=255)
            .cycle()
            .take((fw * fh * channels as u32) as usize)
            .collect();
        let tiles = encoder.split_frame(&frame, channels);

        // Identity encode: return each tile unchanged.
        let processed = encoder
            .encode_tiles_parallel(tiles, |tile| Ok(tile))
            .expect("should succeed");

        let merged = ParallelTileEncoder::merge_tiles(&processed, fw, fh, channels);
        assert_eq!(merged, frame, "parallel identity encode broke the frame");
    }

    #[test]
    fn test_encode_tiles_parallel_error_propagates() {
        let config = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(64)
            .frame_height(64);

        let encoder = ParallelTileEncoder::new(config);
        let frame = vec![0u8; 64 * 64 * 3];
        let tiles = encoder.split_frame(&frame, 3);

        let result = encoder.encode_tiles_parallel(tiles, |_| Err("deliberate error".to_string()));
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_tiles_parallel_transform() {
        // Invert all pixel values and check the result.
        let fw = 32u32;
        let fh = 32u32;
        let channels = 1u8;

        let config = TileConfig::new()
            .tile_cols(2)
            .tile_rows(2)
            .frame_width(fw)
            .frame_height(fh);

        let encoder = ParallelTileEncoder::new(config);

        let frame: Vec<u8> = (0u8..=255).cycle().take((fw * fh) as usize).collect();
        let tiles = encoder.split_frame(&frame, channels);

        let inverted = encoder
            .encode_tiles_parallel(tiles, |mut tile| {
                for b in &mut tile.data {
                    *b = 255 - *b;
                }
                Ok(tile)
            })
            .expect("should succeed");

        let merged = ParallelTileEncoder::merge_tiles(&inverted, fw, fh, channels);
        let expected: Vec<u8> = frame.iter().map(|&b| 255 - b).collect();
        assert_eq!(merged, expected, "inversion result mismatch");
    }
}
