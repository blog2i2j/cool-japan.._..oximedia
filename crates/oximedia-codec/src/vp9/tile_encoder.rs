//! VP9 tile-based parallel frame encoding.
//!
//! VP9 supports tile columns and rows for parallel encoding. This module
//! implements frame-level tile splitting with sync points at tile boundaries,
//! matching the VP9 specification (Section 6.4).
//!
//! # Tile Layout
//!
//! VP9 tiles are aligned to superblock (64×64) boundaries.  The number of
//! tile columns is a power of two between 1 and 64; tile rows are 1 to 4 in
//! the spec but this implementation supports up to 64 for future extensions.
//!
//! ```text
//! ┌──────────┬──────────┬──────────┐
//! │  tile(0,0)│ tile(1,0)│ tile(2,0)│
//! ├──────────┼──────────┼──────────┤
//! │  tile(0,1)│ tile(1,1)│ tile(2,1)│
//! └──────────┴──────────┴──────────┘
//!   tile_cols = 3, tile_rows = 2
//! ```
//!
//! Tiles in the same row are independent and can be encoded in parallel.
//! The deblocking filter requires a one-superblock sync point between
//! adjacent column tiles.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use crate::error::{CodecError, CodecResult};
use crate::frame::VideoFrame;
use rayon::prelude::*;
use std::sync::Arc;

/// VP9 superblock size in pixels (always 64 for VP9; AV1 also supports 128).
pub const VP9_SB_SIZE: u32 = 64;

/// Maximum tile columns per VP9 spec.
pub const VP9_MAX_TILE_COLS: u32 = 64;

/// Maximum tile rows per VP9 spec.
pub const VP9_MAX_TILE_ROWS: u32 = 4;

// ============================================================================
// Tile Configuration
// ============================================================================

/// VP9 tile encoding configuration.
///
/// Both `tile_cols_log2` and `tile_rows_log2` must be non-negative integers.
/// The actual number of tile columns/rows is `2^tile_cols_log2` / `2^tile_rows_log2`.
#[derive(Clone, Debug)]
pub struct Vp9TileConfig {
    /// `log2` of the number of tile columns (0 = 1 column, 1 = 2, …, 6 = 64).
    pub tile_cols_log2: u32,
    /// `log2` of the number of tile rows (0 = 1 row, 1 = 2, 2 = 4).
    pub tile_rows_log2: u32,
    /// Number of worker threads (0 = use rayon's global thread pool).
    pub threads: usize,
    /// Enable deblocking filter boundary sync between column tiles.
    pub enable_deblock_sync: bool,
    /// Base quantiser index (0–255).
    pub base_qindex: u8,
}

impl Default for Vp9TileConfig {
    fn default() -> Self {
        Self {
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            threads: 0,
            enable_deblock_sync: true,
            base_qindex: 128,
        }
    }
}

impl Vp9TileConfig {
    /// Create a configuration with the given log2 tile counts.
    ///
    /// # Errors
    ///
    /// Returns an error if `tile_cols_log2 > 6` or `tile_rows_log2 > 2`.
    pub fn new(
        tile_cols_log2: u32,
        tile_rows_log2: u32,
        threads: usize,
    ) -> CodecResult<Self> {
        if tile_cols_log2 > 6 {
            return Err(CodecError::InvalidParameter(
                "tile_cols_log2 must be 0-6".to_string(),
            ));
        }
        if tile_rows_log2 > 2 {
            return Err(CodecError::InvalidParameter(
                "tile_rows_log2 must be 0-2 for VP9 compliance".to_string(),
            ));
        }
        Ok(Self {
            tile_cols_log2,
            tile_rows_log2,
            threads,
            enable_deblock_sync: true,
            base_qindex: 128,
        })
    }

    /// Automatically choose tile parameters for `width×height` and `threads`.
    ///
    /// Selects the largest power-of-two tile column count that fits both the
    /// frame width (minimum one superblock per tile column) and the available
    /// thread count.
    #[must_use]
    pub fn auto(width: u32, height: u32, threads: usize) -> Self {
        let max_sb_cols = width.div_ceil(VP9_SB_SIZE);
        let max_sb_rows = height.div_ceil(VP9_SB_SIZE);

        // Up to 6 bits for cols
        let mut cols_log2 = 0u32;
        while cols_log2 < 6 && (1u32 << (cols_log2 + 1)) <= max_sb_cols {
            cols_log2 += 1;
        }
        // Cap to thread count (rounded down to power-of-two)
        if threads > 1 {
            let thread_log2 = (threads as f32).log2().floor() as u32;
            cols_log2 = cols_log2.min(thread_log2);
        }

        // VP9 spec: max 2 bits for rows
        let mut rows_log2 = 0u32;
        while rows_log2 < 2 && (1u32 << (rows_log2 + 1)) <= max_sb_rows {
            rows_log2 += 1;
        }

        Self {
            tile_cols_log2: cols_log2,
            tile_rows_log2: rows_log2,
            threads,
            enable_deblock_sync: true,
            base_qindex: 128,
        }
    }

    /// Number of tile columns.
    #[must_use]
    pub fn tile_cols(&self) -> u32 {
        1u32 << self.tile_cols_log2
    }

    /// Number of tile rows.
    #[must_use]
    pub fn tile_rows(&self) -> u32 {
        1u32 << self.tile_rows_log2
    }

    /// Total tile count.
    #[must_use]
    pub fn tile_count(&self) -> u32 {
        self.tile_cols() * self.tile_rows()
    }

    /// Effective thread count (falls back to rayon's pool size when `threads == 0`).
    #[must_use]
    pub fn thread_count(&self) -> usize {
        if self.threads == 0 {
            rayon::current_num_threads()
        } else {
            self.threads
        }
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter is out of spec.
    pub fn validate(&self) -> CodecResult<()> {
        if self.tile_cols_log2 > 6 {
            return Err(CodecError::InvalidParameter(
                "tile_cols_log2 must be 0-6".to_string(),
            ));
        }
        if self.tile_rows_log2 > 2 {
            return Err(CodecError::InvalidParameter(
                "tile_rows_log2 must be 0-2".to_string(),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Tile Region
// ============================================================================

/// Describes a single VP9 tile region aligned to superblock boundaries.
#[derive(Clone, Debug)]
pub struct Vp9TileRegion {
    /// Tile column index.
    pub col: u32,
    /// Tile row index.
    pub row: u32,
    /// X offset in pixels (superblock-aligned).
    pub x: u32,
    /// Y offset in pixels (superblock-aligned).
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Linear tile index (row * tile_cols + col).
    pub index: u32,
    /// Superblock column start (inclusive).
    pub sb_col_start: u32,
    /// Superblock column end (exclusive).
    pub sb_col_end: u32,
    /// Superblock row start (inclusive).
    pub sb_row_start: u32,
    /// Superblock row end (exclusive).
    pub sb_row_end: u32,
}

impl Vp9TileRegion {
    /// Create a tile region.
    ///
    /// # Panics
    ///
    /// Never panics in normal use.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        col: u32,
        row: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        tile_cols: u32,
    ) -> Self {
        let sb_col_start = x / VP9_SB_SIZE;
        let sb_col_end = (x + width).div_ceil(VP9_SB_SIZE);
        let sb_row_start = y / VP9_SB_SIZE;
        let sb_row_end = (y + height).div_ceil(VP9_SB_SIZE);

        Self {
            col,
            row,
            x,
            y,
            width,
            height,
            index: row * tile_cols + col,
            sb_col_start,
            sb_col_end,
            sb_row_start,
            sb_row_end,
        }
    }

    /// Width in superblocks.
    #[must_use]
    pub fn sb_cols(&self) -> u32 {
        self.sb_col_end - self.sb_col_start
    }

    /// Height in superblocks.
    #[must_use]
    pub fn sb_rows(&self) -> u32 {
        self.sb_row_end - self.sb_row_start
    }

    /// True if the tile is at the left frame edge.
    #[must_use]
    pub fn is_left_edge(&self) -> bool {
        self.col == 0
    }

    /// True if the tile is at the top frame edge.
    #[must_use]
    pub fn is_top_edge(&self) -> bool {
        self.row == 0
    }
}

// ============================================================================
// Tile Frame Splitter
// ============================================================================

/// Splits a frame into VP9 tile regions.
#[derive(Debug)]
pub struct Vp9TileFrameSplitter {
    /// Configuration.
    config: Vp9TileConfig,
    /// Frame width.
    frame_width: u32,
    /// Frame height.
    frame_height: u32,
    /// Computed tile regions (raster order).
    regions: Vec<Vp9TileRegion>,
}

impl Vp9TileFrameSplitter {
    /// Create a new splitter and compute tile regions.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(
        config: Vp9TileConfig,
        frame_width: u32,
        frame_height: u32,
    ) -> CodecResult<Self> {
        config.validate()?;
        let mut splitter = Self {
            config,
            frame_width,
            frame_height,
            regions: Vec::new(),
        };
        splitter.compute_regions();
        Ok(splitter)
    }

    /// Compute tile regions with superblock alignment.
    fn compute_regions(&mut self) {
        self.regions.clear();

        let tile_cols = self.config.tile_cols();
        let tile_rows = self.config.tile_rows();

        // Total superblock columns and rows in the frame.
        let sb_cols = self.frame_width.div_ceil(VP9_SB_SIZE);
        let sb_rows = self.frame_height.div_ceil(VP9_SB_SIZE);

        for row in 0..tile_rows {
            for col in 0..tile_cols {
                // VP9: tile boundaries are SB-aligned, distributed uniformly.
                let sb_col_start = (col * sb_cols) / tile_cols;
                let sb_col_end = ((col + 1) * sb_cols) / tile_cols;
                let sb_row_start = (row * sb_rows) / tile_rows;
                let sb_row_end = ((row + 1) * sb_rows) / tile_rows;

                let x = sb_col_start * VP9_SB_SIZE;
                let y = sb_row_start * VP9_SB_SIZE;
                let width = (sb_col_end * VP9_SB_SIZE).min(self.frame_width) - x;
                let height = (sb_row_end * VP9_SB_SIZE).min(self.frame_height) - y;

                self.regions.push(Vp9TileRegion::new(
                    col, row, x, y, width, height, tile_cols,
                ));
            }
        }
    }

    /// Return a reference to the computed tile regions.
    #[must_use]
    pub fn regions(&self) -> &[Vp9TileRegion] {
        &self.regions
    }

    /// Return the number of tiles.
    #[must_use]
    pub fn tile_count(&self) -> usize {
        self.regions.len()
    }

    /// Return a single tile region by index.
    #[must_use]
    pub fn region(&self, index: usize) -> Option<&Vp9TileRegion> {
        self.regions.get(index)
    }
}

// ============================================================================
// Encoded Tile Data
// ============================================================================

/// The result of encoding a single VP9 tile.
#[derive(Clone, Debug)]
pub struct Vp9EncodedTile {
    /// Tile region that was encoded.
    pub region: Vp9TileRegion,
    /// Encoded tile bitstream bytes.
    pub data: Vec<u8>,
}

impl Vp9EncodedTile {
    /// Tile linear index.
    #[must_use]
    pub fn index(&self) -> u32 {
        self.region.index
    }

    /// Encoded data size in bytes.
    #[must_use]
    pub fn encoded_size(&self) -> usize {
        self.data.len()
    }
}

// ============================================================================
// Single-Tile Encoder
// ============================================================================

/// Encodes a single VP9 tile region.
///
/// This struct is designed to be created and used within a rayon parallel
/// iterator, so it is `Send + Sync`.
#[derive(Clone, Debug)]
pub struct Vp9TileEncoder {
    /// Region to encode.
    region: Vp9TileRegion,
    /// Base quantiser index (0-255).
    base_qindex: u8,
    /// True if this frame is a keyframe.
    is_keyframe: bool,
    /// Enable deblocking filter within the tile.
    enable_deblock: bool,
}

impl Vp9TileEncoder {
    /// Create a new tile encoder.
    #[must_use]
    pub fn new(
        region: Vp9TileRegion,
        base_qindex: u8,
        is_keyframe: bool,
        enable_deblock: bool,
    ) -> Self {
        Self {
            region,
            base_qindex,
            is_keyframe,
            enable_deblock,
        }
    }

    /// Encode the tile from a video frame.
    ///
    /// The returned [`Vp9EncodedTile`] contains the serialised bitstream for
    /// this tile including the per-tile header.
    ///
    /// # Errors
    ///
    /// Returns an error if the tile region falls outside the frame bounds.
    pub fn encode(&self, frame: &VideoFrame) -> CodecResult<Vp9EncodedTile> {
        if self.region.x + self.region.width > frame.width
            || self.region.y + self.region.height > frame.height
        {
            return Err(CodecError::InvalidParameter(format!(
                "VP9 tile region [{},{})+[{},{}] exceeds frame {}x{}",
                self.region.x,
                self.region.y,
                self.region.width,
                self.region.height,
                frame.width,
                frame.height,
            )));
        }

        let tile_data = self.extract_luma_data(frame)?;
        let encoded = self.encode_tile_bitstream(&tile_data);

        Ok(Vp9EncodedTile {
            region: self.region.clone(),
            data: encoded,
        })
    }

    /// Extract the luma tile pixels from the frame.
    fn extract_luma_data(&self, frame: &VideoFrame) -> CodecResult<Vec<u8>> {
        let mut pixels = Vec::with_capacity((self.region.width * self.region.height) as usize);

        if let Some(y_plane) = frame.planes.first() {
            for row in self.region.y..(self.region.y + self.region.height) {
                let row_start = (row as usize * y_plane.stride) + self.region.x as usize;
                let row_end = row_start + self.region.width as usize;
                if row_end <= y_plane.data.len() {
                    pixels.extend_from_slice(&y_plane.data[row_start..row_end]);
                } else {
                    // Pad with black if out of bounds (should not happen after
                    // the bounds check above).
                    let available = y_plane.data.len().saturating_sub(row_start);
                    pixels.extend_from_slice(&y_plane.data[row_start..row_start + available]);
                    pixels.resize(pixels.len() + (self.region.width as usize - available), 0);
                }
            }
        }

        Ok(pixels)
    }

    /// Serialise the tile into a minimal VP9 tile bitstream.
    ///
    /// A full implementation would perform:
    /// 1. Superblock partition decision (NONE / SPLIT)
    /// 2. Intra/inter prediction
    /// 3. Residual transform + quantisation
    /// 4. Entropy coding (boolean coder)
    /// 5. In-loop deblocking (with tile boundary sync)
    ///
    /// This implementation produces a valid header followed by a placeholder
    /// compressed block that can be replaced by real entropy-coded data.
    fn encode_tile_bitstream(&self, _luma: &[u8]) -> Vec<u8> {
        let mut bits = Vec::new();

        // --- Mini tile header ---
        // Byte 0: flags
        //   bit 7 = keyframe
        //   bit 6 = deblock enabled
        //   bits 1:0 = col / row parity (for debug)
        let flags = (u8::from(self.is_keyframe) << 7)
            | (u8::from(self.enable_deblock) << 6)
            | (self.region.col as u8 & 0x03);
        bits.push(flags);

        // Bytes 1: base_qindex
        bits.push(self.base_qindex);

        // Bytes 2-5: tile dimensions
        bits.extend_from_slice(&self.region.width.to_le_bytes());
        bits.extend_from_slice(&self.region.height.to_le_bytes());

        // Bytes 10-13: superblock coordinates
        bits.extend_from_slice(&self.region.sb_col_start.to_le_bytes());
        bits.extend_from_slice(&self.region.sb_row_start.to_le_bytes());

        // Placeholder compressed data: one zero byte per superblock in tile
        // (real coding would replace this with entropy-coded coefficients)
        let sb_count = (self.region.sb_cols() * self.region.sb_rows()) as usize;
        bits.resize(bits.len() + sb_count.max(1), 0x00);

        bits
    }

    /// Get the tile region.
    #[must_use]
    pub fn region(&self) -> &Vp9TileRegion {
        &self.region
    }
}

// ============================================================================
// Deblock Sync Point
// ============================================================================

/// Tracks deblocking filter synchronisation between adjacent tile columns.
///
/// VP9 requires that when encoding tile column *C* at superblock row *R*, the
/// deblocking for tile column *C−1* at row *R* has already completed before
/// the loop filter for tile *C* can start.  This structure records which
/// superblock rows have been deblocked per tile column.
///
/// In this implementation deblocking is done after all tiles are encoded
/// (single-pass), so the sync point is a marker for future pipelined
/// implementations.
#[derive(Debug)]
pub struct Vp9DeblockSync {
    /// Number of tile columns.
    tile_cols: u32,
    /// Number of superblock rows in the frame.
    sb_rows: u32,
    /// `completed_sb_rows[col]` = number of SB rows deblocked in column `col`.
    completed_sb_rows: Vec<std::sync::atomic::AtomicU32>,
}

impl Vp9DeblockSync {
    /// Create a new sync tracker.
    #[must_use]
    pub fn new(tile_cols: u32, sb_rows: u32) -> Self {
        let completed_sb_rows = (0..tile_cols)
            .map(|_| std::sync::atomic::AtomicU32::new(0))
            .collect();
        Self {
            tile_cols,
            sb_rows,
            completed_sb_rows,
        }
    }

    /// Mark SB rows `0..sb_row_count` as deblocked for `tile_col`.
    pub fn mark_complete(&self, tile_col: u32, sb_row_count: u32) {
        if tile_col < self.tile_cols {
            self.completed_sb_rows[tile_col as usize]
                .store(sb_row_count, std::sync::atomic::Ordering::Release);
        }
    }

    /// Return the number of completed SB rows for `tile_col`.
    #[must_use]
    pub fn completed_rows(&self, tile_col: u32) -> u32 {
        if tile_col < self.tile_cols {
            self.completed_sb_rows[tile_col as usize]
                .load(std::sync::atomic::Ordering::Acquire)
        } else {
            self.sb_rows
        }
    }

    /// Return `true` if `tile_col − 1` has deblocked at least `sb_row` rows,
    /// meaning `tile_col` may proceed with deblocking at `sb_row`.
    #[must_use]
    pub fn can_deblock(&self, tile_col: u32, sb_row: u32) -> bool {
        if tile_col == 0 {
            return true; // Leftmost tile has no dependency.
        }
        self.completed_rows(tile_col - 1) > sb_row
    }

    /// Total superblock rows in the frame.
    #[must_use]
    pub fn sb_rows(&self) -> u32 {
        self.sb_rows
    }
}

// ============================================================================
// Frame-Level Tile Encoder
// ============================================================================

/// Orchestrates parallel VP9 tile encoding for a complete frame.
///
/// Tiles are encoded with rayon's work-stealing pool.  Row-level dependency
/// sync (deblocking) is tracked via [`Vp9DeblockSync`].
#[derive(Debug)]
pub struct Vp9FrameTileEncoder {
    /// Configuration.
    config: Arc<Vp9TileConfig>,
    /// Frame splitter.
    splitter: Vp9TileFrameSplitter,
    /// Frame dimensions.
    frame_width: u32,
    /// Frame height.
    frame_height: u32,
}

impl Vp9FrameTileEncoder {
    /// Create a new frame-level tile encoder.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(
        config: Vp9TileConfig,
        frame_width: u32,
        frame_height: u32,
    ) -> CodecResult<Self> {
        let splitter =
            Vp9TileFrameSplitter::new(config.clone(), frame_width, frame_height)?;
        Ok(Self {
            config: Arc::new(config),
            splitter,
            frame_width,
            frame_height,
        })
    }

    /// Encode a frame in parallel, returning one [`Vp9EncodedTile`] per tile.
    ///
    /// Tiles are sorted in raster order (top-left to bottom-right) after
    /// parallel encoding.
    ///
    /// # Errors
    ///
    /// Returns an error if any tile encoding fails or the frame dimensions
    /// do not match the encoder configuration.
    pub fn encode_frame(
        &self,
        frame: &VideoFrame,
        is_keyframe: bool,
    ) -> CodecResult<Vec<Vp9EncodedTile>> {
        if frame.width != self.frame_width || frame.height != self.frame_height {
            return Err(CodecError::InvalidParameter(format!(
                "Frame {}x{} does not match encoder {}x{}",
                frame.width, frame.height, self.frame_width, self.frame_height,
            )));
        }

        let regions = self.splitter.regions();
        let base_qindex = self.config.base_qindex;
        let enable_deblock = self.config.enable_deblock_sync;

        // Encode all tiles in parallel using rayon.
        let results: Vec<CodecResult<Vp9EncodedTile>> = regions
            .par_iter()
            .map(|region| {
                let encoder = Vp9TileEncoder::new(
                    region.clone(),
                    base_qindex,
                    is_keyframe,
                    enable_deblock,
                );
                encoder.encode(frame)
            })
            .collect();

        // Collect results and propagate any errors.
        let mut tiles = Vec::with_capacity(results.len());
        for result in results {
            tiles.push(result?);
        }

        // Sort by tile index to guarantee raster order.
        tiles.sort_by_key(Vp9EncodedTile::index);

        // Simulate deblocking sync: mark all tiles as having their
        // SB rows deblocked after encoding completes.
        if enable_deblock {
            let sb_rows = self.frame_height.div_ceil(VP9_SB_SIZE);
            let sync = Vp9DeblockSync::new(self.config.tile_cols(), sb_rows);
            for col in 0..self.config.tile_cols() {
                sync.mark_complete(col, sb_rows);
            }
        }

        Ok(tiles)
    }

    /// Assemble encoded tiles into a VP9 tile group bitstream.
    ///
    /// The format is:
    /// ```text
    /// [1 byte: tile_group_header]
    /// For each tile except the last:
    ///   [4 bytes LE: tile_data_size][tile_data_size bytes: tile bitstream]
    /// [last tile data (no size prefix)]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if `tiles` is empty.
    pub fn assemble_frame(&self, tiles: &[Vp9EncodedTile]) -> CodecResult<Vec<u8>> {
        if tiles.is_empty() {
            return Err(CodecError::InvalidParameter(
                "Cannot assemble empty tile list".to_string(),
            ));
        }

        let mut out = Vec::new();

        // Tile group header byte.
        // bit 7 = has_tile_group (always 1 here)
        // bits 6:4 = tile_cols_log2
        // bits 3:1 = tile_rows_log2
        let header = 0x80u8
            | ((self.config.tile_cols_log2 as u8 & 0x07) << 4)
            | ((self.config.tile_rows_log2 as u8 & 0x07) << 1);
        out.push(header);

        for (i, tile) in tiles.iter().enumerate() {
            let is_last = i == tiles.len() - 1;
            if !is_last {
                let size = tile.data.len() as u32;
                out.extend_from_slice(&size.to_le_bytes());
            }
            out.extend_from_slice(&tile.data);
        }

        Ok(out)
    }

    /// Return the tile configuration.
    #[must_use]
    pub fn config(&self) -> &Vp9TileConfig {
        &self.config
    }

    /// Return the tile count.
    #[must_use]
    pub fn tile_count(&self) -> usize {
        self.splitter.tile_count()
    }

    /// Return the tile regions.
    #[must_use]
    pub fn regions(&self) -> &[Vp9TileRegion] {
        self.splitter.regions()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn make_frame(w: u32, h: u32) -> VideoFrame {
        let mut f = VideoFrame::new(PixelFormat::Yuv420p, w, h);
        f.allocate();
        f
    }

    // ---------- TileConfig --------------------------------------------------

    #[test]
    fn test_tile_config_default() {
        let cfg = Vp9TileConfig::default();
        assert_eq!(cfg.tile_cols(), 1);
        assert_eq!(cfg.tile_rows(), 1);
        assert_eq!(cfg.tile_count(), 1);
    }

    #[test]
    fn test_tile_config_new_valid() {
        let cfg = Vp9TileConfig::new(2, 1, 4).expect("should succeed");
        assert_eq!(cfg.tile_cols(), 4);
        assert_eq!(cfg.tile_rows(), 2);
        assert_eq!(cfg.tile_count(), 8);
    }

    #[test]
    fn test_tile_config_new_invalid_cols() {
        assert!(Vp9TileConfig::new(7, 0, 1).is_err());
    }

    #[test]
    fn test_tile_config_new_invalid_rows() {
        assert!(Vp9TileConfig::new(0, 3, 1).is_err());
    }

    #[test]
    fn test_tile_config_auto() {
        let cfg = Vp9TileConfig::auto(1920, 1080, 4);
        assert!(cfg.tile_count() >= 1);
        assert!(cfg.tile_count() <= VP9_MAX_TILE_COLS * (1u32 << 2));
    }

    // ---------- TileFrameSplitter -------------------------------------------

    #[test]
    fn test_splitter_single_tile() {
        let cfg = Vp9TileConfig::default();
        let s = Vp9TileFrameSplitter::new(cfg, 1920, 1080).expect("should succeed");
        assert_eq!(s.tile_count(), 1);
        let r = s.region(0).expect("should succeed");
        assert_eq!(r.x, 0);
        assert_eq!(r.y, 0);
        assert_eq!(r.width, 1920);
        assert_eq!(r.height, 1080);
    }

    #[test]
    fn test_splitter_2x1_tiles() {
        let cfg = Vp9TileConfig::new(1, 0, 2).expect("should succeed"); // 2 cols, 1 row
        let s = Vp9TileFrameSplitter::new(cfg, 1920, 1080).expect("should succeed");
        assert_eq!(s.tile_count(), 2);
        let r0 = s.region(0).expect("should succeed");
        let r1 = s.region(1).expect("should succeed");
        assert_eq!(r0.x, 0);
        assert!(r1.x > 0);
        assert_eq!(r0.y, 0);
        assert_eq!(r1.y, 0);
        // Total width should equal frame width.
        assert_eq!(r0.width + r1.width, 1920);
    }

    #[test]
    fn test_splitter_2x2_tiles() {
        let cfg = Vp9TileConfig::new(1, 1, 4).expect("should succeed"); // 2x2
        let s = Vp9TileFrameSplitter::new(cfg, 1920, 1088).expect("should succeed");
        assert_eq!(s.tile_count(), 4);
        // All rows should sum to full height.
        let top_height = s.region(0).expect("should succeed").height;
        let bot_height = s.region(2).expect("should succeed").height;
        assert_eq!(top_height + bot_height, 1088);
    }

    // ---------- DeblockSync -------------------------------------------------

    #[test]
    fn test_deblock_sync_left_edge() {
        let sync = Vp9DeblockSync::new(4, 16);
        assert!(sync.can_deblock(0, 0)); // Col 0 has no left dependency
    }

    #[test]
    fn test_deblock_sync_dependency() {
        let sync = Vp9DeblockSync::new(4, 16);
        // Col 1 depends on col 0 having completed SB row.
        assert!(!sync.can_deblock(1, 0)); // Col 0 hasn't completed row 0 yet.
        sync.mark_complete(0, 3);
        assert!(sync.can_deblock(1, 0)); // Col 0 completed rows 0,1,2 -> row 0 is done.
        assert!(sync.can_deblock(1, 2)); // Row 2 is < 3 completed.
        assert!(!sync.can_deblock(1, 3)); // Row 3 not yet completed.
    }

    // ---------- SingleTileEncoder -------------------------------------------

    #[test]
    fn test_single_tile_encode() {
        let region = Vp9TileRegion::new(0, 0, 0, 0, 320, 240, 1);
        let enc = Vp9TileEncoder::new(region, 128, true, true);
        let frame = make_frame(1920, 1080);
        let result = enc.encode(&frame);
        assert!(result.is_ok());
        let tile = result.expect("should succeed");
        assert!(tile.encoded_size() > 0);
    }

    #[test]
    fn test_single_tile_out_of_bounds() {
        let region = Vp9TileRegion::new(0, 0, 1900, 1070, 200, 200, 1);
        let enc = Vp9TileEncoder::new(region, 128, false, false);
        let frame = make_frame(1920, 1080);
        assert!(enc.encode(&frame).is_err());
    }

    // ---------- FrameTileEncoder --------------------------------------------

    #[test]
    fn test_frame_encoder_single_tile() {
        let cfg = Vp9TileConfig::default();
        let enc = Vp9FrameTileEncoder::new(cfg, 1920, 1080).expect("should succeed");
        let frame = make_frame(1920, 1080);
        let tiles = enc.encode_frame(&frame, true).expect("should succeed");
        assert_eq!(tiles.len(), 1);
    }

    #[test]
    fn test_frame_encoder_4x2_tiles() {
        let cfg = Vp9TileConfig::new(2, 1, 8).expect("should succeed"); // 4 cols, 2 rows
        let enc = Vp9FrameTileEncoder::new(cfg, 1920, 1088).expect("should succeed");
        let frame = make_frame(1920, 1088);
        let tiles = enc.encode_frame(&frame, false).expect("should succeed");
        assert_eq!(tiles.len(), 8);
        // Verify raster order
        for (i, tile) in tiles.iter().enumerate() {
            assert_eq!(tile.index() as usize, i);
        }
    }

    #[test]
    fn test_frame_encoder_wrong_dimensions() {
        let cfg = Vp9TileConfig::default();
        let enc = Vp9FrameTileEncoder::new(cfg, 1920, 1080).expect("should succeed");
        let frame = make_frame(1280, 720);
        assert!(enc.encode_frame(&frame, true).is_err());
    }

    #[test]
    fn test_assemble_frame() {
        let cfg = Vp9TileConfig::new(1, 0, 2).expect("should succeed"); // 2 cols, 1 row
        let enc = Vp9FrameTileEncoder::new(cfg, 1920, 1080).expect("should succeed");
        let frame = make_frame(1920, 1080);
        let tiles = enc.encode_frame(&frame, true).expect("should succeed");
        let assembled = enc.assemble_frame(&tiles).expect("should succeed");
        assert!(!assembled.is_empty());
        // Header byte must have bit 7 set.
        assert_eq!(assembled[0] & 0x80, 0x80);
    }

    #[test]
    fn test_assemble_empty_error() {
        let cfg = Vp9TileConfig::default();
        let enc = Vp9FrameTileEncoder::new(cfg, 1920, 1080).expect("should succeed");
        assert!(enc.assemble_frame(&[]).is_err());
    }

    #[test]
    fn test_tile_region_sb_alignment() {
        let region = Vp9TileRegion::new(1, 0, 64, 0, 128, 64, 2);
        assert_eq!(region.sb_col_start, 1);
        assert_eq!(region.sb_col_end, 3);
        assert_eq!(region.sb_cols(), 2);
        assert_eq!(region.sb_rows(), 1);
    }
}
