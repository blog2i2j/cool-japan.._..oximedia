//! Generic tile-based parallel frame encoding for OxiMedia codecs.
//!
//! This module provides codec-agnostic infrastructure for splitting video
//! frames into rectangular tiles and encoding them concurrently using
//! Rayon's work-stealing thread pool.  Individual codec implementations
//! (AV1, VP9, …) plug into the system by implementing [`TileEncodeOp`].
//!
//! # Architecture
//!
//! ```text
//! TileConfig  ─── describes the tile grid & thread count
//!     │
//!     ▼
//! TileEncoder ─── splits frame → parallel encode → collects TileResult
//!     │
//!     ▼
//! assemble_tiles() ─── merges sorted TileResults into a single bitstream
//! ```
//!
//! # Thread Safety
//!
//! All public types are `Send + Sync`.  Rayon's data-parallel iterators
//! ensure that no `unsafe` code is required.
//!
//! # Example
//!
//! ```
//! use oximedia_codec::tile::{TileConfig, TileEncoder, TileEncodeOp,
//!                             TileResult, assemble_tiles};
//! use oximedia_codec::error::CodecResult;
//! use oximedia_codec::frame::VideoFrame;
//! use oximedia_core::PixelFormat;
//!
//! /// Trivial encode op: store raw luma bytes.
//! struct RawLumaOp;
//!
//! impl TileEncodeOp for RawLumaOp {
//!     fn encode_tile(
//!         &self,
//!         frame: &VideoFrame,
//!         x: u32, y: u32, w: u32, h: u32,
//!     ) -> CodecResult<Vec<u8>> {
//!         let mut out = Vec::new();
//!         if let Some(plane) = frame.planes.first() {
//!             for row in y..(y + h) {
//!                 let start = row as usize * plane.stride + x as usize;
//!                 out.extend_from_slice(&plane.data[start..start + w as usize]);
//!             }
//!         }
//!         Ok(out)
//!     }
//! }
//!
//! let cfg = TileConfig::new(2, 2, 0)?;
//! let encoder = TileEncoder::new(cfg, 1920, 1080);
//!
//! let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
//! frame.allocate();
//!
//! let results = encoder.encode(&frame, &RawLumaOp)?;
//! assert_eq!(results.len(), 4);
//!
//! let bitstream = assemble_tiles(&results);
//! assert!(!bitstream.is_empty());
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

use crate::error::{CodecError, CodecResult};
use crate::frame::VideoFrame;
use rayon::prelude::*;
use std::sync::Arc;

// =============================================================================
// TileConfig
// =============================================================================

/// Configuration for the tile grid used during parallel encoding.
///
/// Tile counts must be positive integers ≤ 64.  A `threads` value of `0`
/// means "use Rayon's global thread pool size".
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileConfig {
    /// Number of tile columns (1–64).
    pub tile_cols: u32,
    /// Number of tile rows (1–64).
    pub tile_rows: u32,
    /// Worker thread count (0 = auto).
    pub threads: usize,
}

impl TileConfig {
    /// Create a validated `TileConfig`.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::InvalidParameter`] if:
    /// - `tile_cols` or `tile_rows` is 0 or greater than 64, or
    /// - the total tile count exceeds 4 096.
    pub fn new(tile_cols: u32, tile_rows: u32, threads: usize) -> CodecResult<Self> {
        if tile_cols == 0 || tile_cols > 64 {
            return Err(CodecError::InvalidParameter(format!(
                "tile_cols must be 1–64, got {tile_cols}"
            )));
        }
        if tile_rows == 0 || tile_rows > 64 {
            return Err(CodecError::InvalidParameter(format!(
                "tile_rows must be 1–64, got {tile_rows}"
            )));
        }
        if tile_cols * tile_rows > 4096 {
            return Err(CodecError::InvalidParameter(format!(
                "total tile count {} exceeds 4096",
                tile_cols * tile_rows
            )));
        }
        Ok(Self {
            tile_cols,
            tile_rows,
            threads,
        })
    }

    /// Total number of tiles.
    #[must_use]
    pub const fn tile_count(&self) -> u32 {
        self.tile_cols * self.tile_rows
    }

    /// Effective thread count (resolves `0` to the rayon pool size).
    #[must_use]
    pub fn thread_count(&self) -> usize {
        if self.threads == 0 {
            rayon::current_num_threads()
        } else {
            self.threads
        }
    }

    /// Choose a reasonable tile layout for `width × height` and `threads`.
    ///
    /// Selects the largest power-of-two tile counts that keep individual
    /// tile areas reasonable (≥ 64 × 64 pixels) while not exceeding the
    /// thread count.
    #[must_use]
    pub fn auto(width: u32, height: u32, threads: usize) -> Self {
        let t = if threads == 0 {
            rayon::current_num_threads()
        } else {
            threads
        };

        // Distribute threads across columns and rows proportional to aspect.
        let aspect = width as f32 / height.max(1) as f32;
        let target = t.next_power_of_two() as u32;

        let mut cols = ((target as f32 * aspect).sqrt().ceil() as u32)
            .next_power_of_two()
            .clamp(1, 64);
        let mut rows = ((target as f32 / aspect).sqrt().ceil() as u32)
            .next_power_of_two()
            .clamp(1, 64);

        // Clamp so that each tile is at least 64 pixels wide/tall.
        while cols > 1 && width / cols < 64 {
            cols /= 2;
        }
        while rows > 1 && height / rows < 64 {
            rows /= 2;
        }
        // Keep total ≤ 4096.
        while cols * rows > 4096 {
            if cols > rows {
                cols /= 2;
            } else {
                rows /= 2;
            }
        }

        Self {
            tile_cols: cols,
            tile_rows: rows,
            threads,
        }
    }
}

impl Default for TileConfig {
    /// Single-tile, auto-thread default.
    fn default() -> Self {
        Self {
            tile_cols: 1,
            tile_rows: 1,
            threads: 0,
        }
    }
}

// =============================================================================
// Tile coordinate helper
// =============================================================================

/// Pixel coordinates and dimensions of a single tile within a frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileCoord {
    /// Tile column index (0-based).
    pub col: u32,
    /// Tile row index (0-based).
    pub row: u32,
    /// X offset in pixels.
    pub x: u32,
    /// Y offset in pixels.
    pub y: u32,
    /// Tile width in pixels.
    pub width: u32,
    /// Tile height in pixels.
    pub height: u32,
    /// Linear raster index (`row * tile_cols + col`).
    pub index: u32,
}

impl TileCoord {
    /// Create a new `TileCoord`.
    #[must_use]
    pub const fn new(
        col: u32,
        row: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        tile_cols: u32,
    ) -> Self {
        Self {
            col,
            row,
            x,
            y,
            width,
            height,
            index: row * tile_cols + col,
        }
    }

    /// Tile area in pixels.
    #[must_use]
    pub const fn area(&self) -> u32 {
        self.width * self.height
    }

    /// True if this tile is at the left frame boundary.
    #[must_use]
    pub const fn is_left_edge(&self) -> bool {
        self.col == 0
    }

    /// True if this tile is at the top frame boundary.
    #[must_use]
    pub const fn is_top_edge(&self) -> bool {
        self.row == 0
    }
}

// =============================================================================
// TileResult
// =============================================================================

/// The output produced by encoding a single tile.
///
/// Results are collected after parallel encoding and then re-ordered into
/// raster order by [`TileEncoder::encode`] before being returned to the
/// caller.
#[derive(Clone, Debug)]
pub struct TileResult {
    /// Spatial coordinates of the tile within the frame.
    pub coord: TileCoord,
    /// Codec-specific encoded bytes for this tile.
    pub data: Vec<u8>,
}

impl TileResult {
    /// Create a new `TileResult`.
    #[must_use]
    pub fn new(coord: TileCoord, data: Vec<u8>) -> Self {
        Self { coord, data }
    }

    /// Raster index of this tile.
    #[must_use]
    pub const fn index(&self) -> u32 {
        self.coord.index
    }

    /// Encoded size in bytes.
    #[must_use]
    pub fn encoded_size(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the encoded data is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// =============================================================================
// TileEncodeOp trait
// =============================================================================

/// Codec-specific tile encoding operation.
///
/// Implementors receive a reference to the full frame plus the pixel
/// coordinates of the tile to encode.  They return the raw encoded bytes
/// for that tile or a [`CodecError`].
///
/// # Thread Safety
///
/// Implementations **must** be `Send + Sync` because [`TileEncoder`] drives
/// them from Rayon parallel iterators.
pub trait TileEncodeOp: Send + Sync {
    /// Encode the tile at `(x, y)` with size `(width, height)` pixels.
    ///
    /// # Errors
    ///
    /// Return a [`CodecError`] if encoding fails for any reason.
    fn encode_tile(
        &self,
        frame: &VideoFrame,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> CodecResult<Vec<u8>>;
}

// =============================================================================
// TileEncoder
// =============================================================================

/// Splits a frame into tiles and encodes them in parallel using Rayon.
///
/// # Example
///
/// ```
/// use oximedia_codec::tile::{TileConfig, TileEncoder, TileEncodeOp, TileResult};
/// use oximedia_codec::error::CodecResult;
/// use oximedia_codec::frame::VideoFrame;
/// use oximedia_core::PixelFormat;
///
/// struct NullOp;
/// impl TileEncodeOp for NullOp {
///     fn encode_tile(&self, _f: &VideoFrame, _x: u32, _y: u32, _w: u32, _h: u32)
///         -> CodecResult<Vec<u8>>
///     {
///         Ok(vec![0u8; 16])
///     }
/// }
///
/// let cfg = TileConfig::new(2, 2, 0)?;
/// let encoder = TileEncoder::new(cfg, 1920, 1080);
/// let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
/// frame.allocate();
///
/// let results = encoder.encode(&frame, &NullOp)?;
/// assert_eq!(results.len(), 4);
/// ```
pub struct TileEncoder {
    config: Arc<TileConfig>,
    frame_width: u32,
    frame_height: u32,
    /// Pre-computed tile coordinates in raster order.
    coords: Vec<TileCoord>,
}

impl TileEncoder {
    /// Create a `TileEncoder` for frames of size `frame_width × frame_height`.
    #[must_use]
    pub fn new(config: TileConfig, frame_width: u32, frame_height: u32) -> Self {
        let coords = Self::compute_coords(&config, frame_width, frame_height);
        Self {
            config: Arc::new(config),
            frame_width,
            frame_height,
            coords,
        }
    }

    /// Encode `frame` using `op` in parallel.
    ///
    /// The returned `Vec<TileResult>` is sorted in raster order (tile index
    /// 0 first).
    ///
    /// # Errors
    ///
    /// Returns the first [`CodecError`] produced by any tile's encoding, or
    /// [`CodecError::InvalidParameter`] if the frame dimensions do not match
    /// the encoder configuration.
    pub fn encode<O: TileEncodeOp>(
        &self,
        frame: &VideoFrame,
        op: &O,
    ) -> CodecResult<Vec<TileResult>> {
        if frame.width != self.frame_width || frame.height != self.frame_height {
            return Err(CodecError::InvalidParameter(format!(
                "frame {}×{} does not match encoder {}×{}",
                frame.width, frame.height, self.frame_width, self.frame_height
            )));
        }

        // Parallel encode.
        let results: Vec<CodecResult<TileResult>> = self
            .coords
            .par_iter()
            .map(|coord| {
                let data = op.encode_tile(frame, coord.x, coord.y, coord.width, coord.height)?;
                Ok(TileResult::new(coord.clone(), data))
            })
            .collect();

        // Propagate errors and sort.
        let mut tiles = Vec::with_capacity(results.len());
        for r in results {
            tiles.push(r?);
        }
        tiles.sort_by_key(TileResult::index);
        Ok(tiles)
    }

    /// The tile configuration.
    #[must_use]
    pub fn config(&self) -> &TileConfig {
        &self.config
    }

    /// Frame width.
    #[must_use]
    pub const fn frame_width(&self) -> u32 {
        self.frame_width
    }

    /// Frame height.
    #[must_use]
    pub const fn frame_height(&self) -> u32 {
        self.frame_height
    }

    /// Pre-computed tile coordinates.
    #[must_use]
    pub fn coords(&self) -> &[TileCoord] {
        &self.coords
    }

    /// Total number of tiles.
    #[must_use]
    pub fn tile_count(&self) -> usize {
        self.coords.len()
    }

    /// Compute uniform tile coordinates for the given config and frame size.
    fn compute_coords(config: &TileConfig, fw: u32, fh: u32) -> Vec<TileCoord> {
        let cols = config.tile_cols;
        let rows = config.tile_rows;
        let tw = fw.div_ceil(cols); // nominal tile width
        let th = fh.div_ceil(rows); // nominal tile height

        let mut coords = Vec::with_capacity((cols * rows) as usize);
        for row in 0..rows {
            for col in 0..cols {
                let x = col * tw;
                let y = row * th;
                let width = if col == cols - 1 { fw - x } else { tw };
                let height = if row == rows - 1 { fh - y } else { th };
                coords.push(TileCoord::new(col, row, x, y, width, height, cols));
            }
        }
        coords
    }
}

impl std::fmt::Debug for TileEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileEncoder")
            .field("config", &self.config)
            .field("frame_width", &self.frame_width)
            .field("frame_height", &self.frame_height)
            .field("tile_count", &self.tile_count())
            .finish()
    }
}

// =============================================================================
// assemble_tiles
// =============================================================================

/// Assemble an ordered slice of [`TileResult`]s into a single byte stream.
///
/// The format is:
///
/// ```text
/// [4 bytes LE: number of tiles]
/// For each tile except the last:
///     [4 bytes LE: tile_data_length]
///     [tile_data_length bytes]
/// [last tile bytes with no length prefix]
/// ```
///
/// Pass the output to a codec-specific container muxer that understands this
/// layout (or use [`decode_tile_stream`] to reverse the process).
///
/// # Panics
///
/// Panics if `tiles` is empty (use the guard in your calling code).
#[must_use]
pub fn assemble_tiles(tiles: &[TileResult]) -> Vec<u8> {
    if tiles.is_empty() {
        return Vec::new();
    }

    // Rough pre-allocation.
    let total_data: usize = tiles.iter().map(|t| t.encoded_size()).sum();
    let mut out = Vec::with_capacity(4 + total_data + (tiles.len() - 1) * 4);

    // Header: tile count.
    out.extend_from_slice(&(tiles.len() as u32).to_le_bytes());

    // Tile payloads.
    for (i, tile) in tiles.iter().enumerate() {
        let is_last = i == tiles.len() - 1;
        if !is_last {
            // Write size prefix for non-terminal tiles.
            out.extend_from_slice(&(tile.data.len() as u32).to_le_bytes());
        }
        out.extend_from_slice(&tile.data);
    }

    out
}

/// Decode a tile stream produced by [`assemble_tiles`].
///
/// Returns a `Vec` of raw per-tile byte payloads in the order they were
/// stored (which is raster order when the encoder was used correctly).
///
/// # Errors
///
/// Returns [`CodecError::InvalidBitstream`] if the stream is truncated or
/// the encoded tile count is inconsistent with the data length.
pub fn decode_tile_stream(stream: &[u8]) -> CodecResult<Vec<Vec<u8>>> {
    if stream.len() < 4 {
        return Err(CodecError::InvalidBitstream(
            "tile stream too short for header".to_string(),
        ));
    }

    let num_tiles = u32::from_le_bytes([stream[0], stream[1], stream[2], stream[3]]) as usize;
    if num_tiles == 0 {
        return Ok(Vec::new());
    }

    let mut tiles: Vec<Vec<u8>> = Vec::with_capacity(num_tiles);
    let mut pos = 4usize;

    for i in 0..num_tiles {
        let is_last = i == num_tiles - 1;

        if is_last {
            // Last tile: rest of stream.
            tiles.push(stream[pos..].to_vec());
            pos = stream.len();
        } else {
            if pos + 4 > stream.len() {
                return Err(CodecError::InvalidBitstream(format!(
                    "tile {i}: stream truncated before size field"
                )));
            }
            let tile_size = u32::from_le_bytes([
                stream[pos],
                stream[pos + 1],
                stream[pos + 2],
                stream[pos + 3],
            ]) as usize;
            pos += 4;

            if pos + tile_size > stream.len() {
                return Err(CodecError::InvalidBitstream(format!(
                    "tile {i}: declared size {tile_size} exceeds remaining stream bytes"
                )));
            }
            tiles.push(stream[pos..pos + tile_size].to_vec());
            pos += tile_size;
        }
    }

    Ok(tiles)
}

// =============================================================================
// Built-in encode ops
// =============================================================================

/// A simple encode op that extracts raw luma samples from a tile.
///
/// Useful as a reference implementation and for testing.
pub struct RawLumaEncodeOp;

impl TileEncodeOp for RawLumaEncodeOp {
    fn encode_tile(
        &self,
        frame: &VideoFrame,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> CodecResult<Vec<u8>> {
        let mut out = Vec::with_capacity((width * height) as usize);
        if let Some(plane) = frame.planes.first() {
            for row in y..(y + height) {
                let start = row as usize * plane.stride + x as usize;
                let end = start + width as usize;
                if end <= plane.data.len() {
                    out.extend_from_slice(&plane.data[start..end]);
                } else {
                    // Pad with zeros for out-of-plane rows.
                    let available = plane.data.len().saturating_sub(start);
                    out.extend_from_slice(&plane.data[start..start + available]);
                    out.resize(out.len() + (width as usize - available), 0);
                }
            }
        }
        Ok(out)
    }
}

/// A simple op that encodes a tile with a small header describing its
/// position and appends placeholder compressed data.
///
/// Header layout:
/// ```text
/// [4 bytes LE: x offset]
/// [4 bytes LE: y offset]
/// [4 bytes LE: width]
/// [4 bytes LE: height]
/// [1 byte: tile col index]
/// [1 byte: tile row index]
/// ```
/// Followed by raw luma bytes.
pub struct HeaderedTileEncodeOp;

impl TileEncodeOp for HeaderedTileEncodeOp {
    fn encode_tile(
        &self,
        frame: &VideoFrame,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> CodecResult<Vec<u8>> {
        let mut out = Vec::with_capacity(14 + (width * height) as usize);
        out.extend_from_slice(&x.to_le_bytes());
        out.extend_from_slice(&y.to_le_bytes());
        out.extend_from_slice(&width.to_le_bytes());
        out.extend_from_slice(&height.to_le_bytes());

        // Append raw luma.
        let raw = RawLumaEncodeOp.encode_tile(frame, x, y, width, height)?;
        out.extend_from_slice(&raw);
        Ok(out)
    }
}

// =============================================================================
// Parallel statistics helper
// =============================================================================

/// Summary statistics over a completed parallel encode run.
#[derive(Clone, Debug, Default)]
pub struct TileEncodeStats {
    /// Total encoded bytes across all tiles.
    pub total_bytes: usize,
    /// Smallest tile encoded size in bytes.
    pub min_tile_bytes: usize,
    /// Largest tile encoded size in bytes.
    pub max_tile_bytes: usize,
    /// Mean encoded bytes per tile.
    pub mean_tile_bytes: f64,
    /// Number of tiles.
    pub tile_count: usize,
}

impl TileEncodeStats {
    /// Compute stats from a slice of [`TileResult`]s.
    ///
    /// Returns `None` if `results` is empty.
    #[must_use]
    pub fn from_results(results: &[TileResult]) -> Option<Self> {
        if results.is_empty() {
            return None;
        }
        let sizes: Vec<usize> = results.iter().map(TileResult::encoded_size).collect();
        let total: usize = sizes.iter().sum();
        let min = *sizes.iter().min().unwrap_or(&0);
        let max = *sizes.iter().max().unwrap_or(&0);
        Some(Self {
            total_bytes: total,
            min_tile_bytes: min,
            max_tile_bytes: max,
            mean_tile_bytes: total as f64 / sizes.len() as f64,
            tile_count: sizes.len(),
        })
    }

    /// Compression ratio (encoded bytes / raw luma bytes).
    ///
    /// Returns `None` if `raw_luma_bytes` is 0.
    #[must_use]
    pub fn compression_ratio(&self, raw_luma_bytes: usize) -> Option<f64> {
        if raw_luma_bytes == 0 {
            return None;
        }
        Some(self.total_bytes as f64 / raw_luma_bytes as f64)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    // ---------- Helpers -----------------------------------------------------

    fn make_frame(w: u32, h: u32) -> VideoFrame {
        let mut f = VideoFrame::new(PixelFormat::Yuv420p, w, h);
        f.allocate();
        f
    }

    /// Encode op that returns a fixed-size payload of `n` bytes.
    struct FixedSizeOp(usize);

    impl TileEncodeOp for FixedSizeOp {
        fn encode_tile(
            &self,
            _frame: &VideoFrame,
            _x: u32,
            _y: u32,
            _w: u32,
            _h: u32,
        ) -> CodecResult<Vec<u8>> {
            Ok(vec![0xABu8; self.0])
        }
    }

    /// Encode op that always returns an error.
    struct ErrorOp;

    impl TileEncodeOp for ErrorOp {
        fn encode_tile(
            &self,
            _frame: &VideoFrame,
            _x: u32,
            _y: u32,
            _w: u32,
            _h: u32,
        ) -> CodecResult<Vec<u8>> {
            Err(CodecError::InvalidParameter("deliberate error".to_string()))
        }
    }

    // ---------- TileConfig --------------------------------------------------

    #[test]
    fn test_tile_config_default() {
        let cfg = TileConfig::default();
        assert_eq!(cfg.tile_cols, 1);
        assert_eq!(cfg.tile_rows, 1);
        assert_eq!(cfg.tile_count(), 1);
    }

    #[test]
    fn test_tile_config_new_valid() {
        let cfg = TileConfig::new(4, 2, 8).expect("should succeed");
        assert_eq!(cfg.tile_cols, 4);
        assert_eq!(cfg.tile_rows, 2);
        assert_eq!(cfg.tile_count(), 8);
    }

    #[test]
    fn test_tile_config_new_zero_cols() {
        assert!(TileConfig::new(0, 1, 0).is_err());
    }

    #[test]
    fn test_tile_config_new_zero_rows() {
        assert!(TileConfig::new(1, 0, 0).is_err());
    }

    #[test]
    fn test_tile_config_new_too_many_cols() {
        assert!(TileConfig::new(65, 1, 0).is_err());
    }

    #[test]
    fn test_tile_config_new_too_many_rows() {
        assert!(TileConfig::new(1, 65, 0).is_err());
    }

    #[test]
    fn test_tile_config_overflow() {
        // 64 × 64 = 4096, which is exactly the limit.
        assert!(TileConfig::new(64, 64, 0).is_ok());
        // 65 cols fails already, but a hypothetical 65×64 = 4160 would also fail.
    }

    #[test]
    fn test_tile_config_auto_wide() {
        let cfg = TileConfig::auto(3840, 1080, 8);
        assert!(
            cfg.tile_cols >= cfg.tile_rows,
            "wide frame should have more columns"
        );
        assert!(cfg.tile_count() >= 1);
    }

    #[test]
    fn test_tile_config_auto_tall() {
        let cfg = TileConfig::auto(1080, 3840, 8);
        assert!(
            cfg.tile_rows >= cfg.tile_cols,
            "tall frame should have more rows"
        );
    }

    #[test]
    fn test_tile_config_auto_single_thread() {
        let cfg = TileConfig::auto(1920, 1080, 1);
        // With 1 thread, tile count should still be valid.
        assert!(cfg.tile_count() >= 1);
    }

    #[test]
    fn test_tile_config_thread_count_auto() {
        let cfg = TileConfig::new(1, 1, 0).expect("should succeed");
        assert!(cfg.thread_count() >= 1);
    }

    #[test]
    fn test_tile_config_thread_count_explicit() {
        let cfg = TileConfig::new(1, 1, 4).expect("should succeed");
        assert_eq!(cfg.thread_count(), 4);
    }

    // ---------- TileCoord ---------------------------------------------------

    #[test]
    fn test_tile_coord_index() {
        // 2-column grid: (col=1, row=0) → index 1, (col=0, row=1) → index 2
        let c = TileCoord::new(1, 0, 960, 0, 960, 540, 2);
        assert_eq!(c.index, 1);
        assert_eq!(c.area(), 960 * 540);
        assert!(!c.is_left_edge());
        assert!(c.is_top_edge());
    }

    #[test]
    fn test_tile_coord_top_left() {
        let c = TileCoord::new(0, 0, 0, 0, 480, 270, 4);
        assert_eq!(c.index, 0);
        assert!(c.is_left_edge());
        assert!(c.is_top_edge());
    }

    // ---------- TileEncoder -------------------------------------------------

    #[test]
    fn test_encoder_single_tile() {
        let cfg = TileConfig::new(1, 1, 0).expect("should succeed");
        let encoder = TileEncoder::new(cfg, 1920, 1080);
        assert_eq!(encoder.tile_count(), 1);

        let c = &encoder.coords()[0];
        assert_eq!(c.x, 0);
        assert_eq!(c.y, 0);
        assert_eq!(c.width, 1920);
        assert_eq!(c.height, 1080);
    }

    #[test]
    fn test_encoder_2x2_coverage() {
        let cfg = TileConfig::new(2, 2, 0).expect("should succeed");
        let encoder = TileEncoder::new(cfg, 1920, 1080);
        assert_eq!(encoder.tile_count(), 4);

        // Every pixel should be covered exactly once.
        let mut covered = vec![0u32; 1920 * 1080];
        for coord in encoder.coords() {
            for row in coord.y..(coord.y + coord.height) {
                for col in coord.x..(coord.x + coord.width) {
                    covered[(row * 1920 + col) as usize] += 1;
                }
            }
        }
        assert!(
            covered.iter().all(|&c| c == 1),
            "some pixels covered ≠ 1 time"
        );
    }

    #[test]
    fn test_encoder_4x3_coverage() {
        let cfg = TileConfig::new(4, 3, 0).expect("should succeed");
        let encoder = TileEncoder::new(cfg, 1280, 720);
        assert_eq!(encoder.tile_count(), 12);

        let mut total_area: u64 = 0;
        for coord in encoder.coords() {
            assert!(coord.width > 0 && coord.height > 0, "empty tile");
            total_area += u64::from(coord.area());
        }
        assert_eq!(total_area, 1280 * 720, "total tile area != frame area");
    }

    #[test]
    fn test_encoder_raster_order() {
        let cfg = TileConfig::new(3, 2, 0).expect("should succeed");
        let encoder = TileEncoder::new(cfg, 1920, 1080);
        for (i, coord) in encoder.coords().iter().enumerate() {
            assert_eq!(coord.index as usize, i, "coords not in raster order");
        }
    }

    #[test]
    fn test_encoder_encode_parallel() {
        let cfg = TileConfig::new(2, 2, 0).expect("should succeed");
        let encoder = TileEncoder::new(cfg, 1920, 1080);
        let frame = make_frame(1920, 1080);

        let results = encoder
            .encode(&frame, &FixedSizeOp(64))
            .expect("encode should succeed");
        assert_eq!(results.len(), 4);
        // Results must be in raster order.
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.index() as usize, i);
            assert_eq!(r.encoded_size(), 64);
        }
    }

    #[test]
    fn test_encoder_encode_error_propagates() {
        let cfg = TileConfig::new(2, 2, 0).expect("should succeed");
        let encoder = TileEncoder::new(cfg, 1920, 1080);
        let frame = make_frame(1920, 1080);
        assert!(encoder.encode(&frame, &ErrorOp).is_err());
    }

    #[test]
    fn test_encoder_wrong_frame_dimensions() {
        let cfg = TileConfig::new(2, 2, 0).expect("should succeed");
        let encoder = TileEncoder::new(cfg, 1920, 1080);
        let frame = make_frame(1280, 720);
        assert!(encoder.encode(&frame, &FixedSizeOp(1)).is_err());
    }

    // ---------- assemble_tiles / decode_tile_stream -------------------------

    #[test]
    fn test_assemble_empty() {
        assert!(assemble_tiles(&[]).is_empty());
    }

    #[test]
    fn test_assemble_single_tile() {
        let coord = TileCoord::new(0, 0, 0, 0, 1920, 1080, 1);
        let result = TileResult::new(coord, vec![1u8, 2, 3, 4]);
        let stream = assemble_tiles(&[result]);

        // 4-byte header (tile count = 1) + 4 bytes data (last tile has no size prefix).
        assert_eq!(stream.len(), 4 + 4);
        assert_eq!(
            u32::from_le_bytes([stream[0], stream[1], stream[2], stream[3]]),
            1
        );
    }

    #[test]
    fn test_assemble_decode_roundtrip_two_tiles() {
        let payload_a = vec![0xAA; 128];
        let payload_b = vec![0xBB; 256];

        let ta = TileResult::new(TileCoord::new(0, 0, 0, 0, 960, 540, 2), payload_a.clone());
        let tb = TileResult::new(TileCoord::new(1, 0, 960, 0, 960, 540, 2), payload_b.clone());

        let stream = assemble_tiles(&[ta, tb]);
        let decoded = decode_tile_stream(&stream).expect("should succeed");

        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0], payload_a);
        assert_eq!(decoded[1], payload_b);
    }

    #[test]
    fn test_assemble_decode_roundtrip_four_tiles() {
        let cfg = TileConfig::new(2, 2, 0).expect("should succeed");
        let encoder = TileEncoder::new(cfg, 640, 480);
        let frame = make_frame(640, 480);

        let results = encoder
            .encode(&frame, &RawLumaEncodeOp)
            .expect("encode should succeed");
        let stream = assemble_tiles(&results);
        let decoded = decode_tile_stream(&stream).expect("should succeed");

        assert_eq!(decoded.len(), 4);
        // Each decoded tile must match the original result's data.
        for (orig, dec) in results.iter().zip(decoded.iter()) {
            assert_eq!(&orig.data, dec, "tile data mismatch after roundtrip");
        }
    }

    #[test]
    fn test_decode_tile_stream_truncated_header() {
        assert!(decode_tile_stream(&[0, 1]).is_err());
    }

    #[test]
    fn test_decode_tile_stream_truncated_size() {
        // Header says 2 tiles, but there is no size field for tile 0.
        let stream = [2u8, 0, 0, 0]; // 4 bytes header, nothing else
        assert!(decode_tile_stream(&stream).is_err());
    }

    #[test]
    fn test_decode_tile_stream_truncated_data() {
        // Header says 2 tiles; tile 0 claims 1000 bytes but stream is short.
        let mut stream = vec![2u8, 0, 0, 0]; // count = 2
        stream.extend_from_slice(&1000u32.to_le_bytes()); // tile 0 size
        stream.extend(vec![0u8; 10]); // only 10 bytes of data
        assert!(decode_tile_stream(&stream).is_err());
    }

    #[test]
    fn test_decode_empty_stream() {
        // A stream declaring 0 tiles should yield an empty vec.
        let stream = [0u8, 0, 0, 0];
        let decoded = decode_tile_stream(&stream).expect("should succeed");
        assert!(decoded.is_empty());
    }

    // ---------- Built-in encode ops -----------------------------------------

    #[test]
    fn test_raw_luma_op_size() {
        let frame = make_frame(320, 240);
        let op = RawLumaEncodeOp;
        let data = op
            .encode_tile(&frame, 0, 0, 320, 240)
            .expect("should succeed");
        // Should contain exactly 320*240 luma bytes.
        assert_eq!(data.len(), 320 * 240);
    }

    #[test]
    fn test_raw_luma_op_partial_tile() {
        let frame = make_frame(100, 50);
        let op = RawLumaEncodeOp;
        let data = op
            .encode_tile(&frame, 0, 0, 50, 25)
            .expect("should succeed");
        assert_eq!(data.len(), 50 * 25);
    }

    #[test]
    fn test_headered_tile_op_header_content() {
        let frame = make_frame(128, 64);
        let op = HeaderedTileEncodeOp;
        let data = op
            .encode_tile(&frame, 32, 16, 64, 32)
            .expect("should succeed");

        // First 16 bytes are the header.
        assert!(data.len() >= 16);
        let x = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let y = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let w = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let h = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);

        assert_eq!(x, 32);
        assert_eq!(y, 16);
        assert_eq!(w, 64);
        assert_eq!(h, 32);
        // Remaining bytes are luma samples.
        assert_eq!(data.len(), 16 + 64 * 32);
    }

    // ---------- TileEncodeStats ---------------------------------------------

    #[test]
    fn test_stats_from_empty() {
        assert!(TileEncodeStats::from_results(&[]).is_none());
    }

    #[test]
    fn test_stats_from_uniform() {
        let cfg = TileConfig::new(4, 2, 0).expect("should succeed");
        let encoder = TileEncoder::new(cfg, 1920, 1080);
        let frame = make_frame(1920, 1080);

        let results = encoder
            .encode(&frame, &FixedSizeOp(200))
            .expect("encode should succeed");
        let stats = TileEncodeStats::from_results(&results).expect("should succeed");

        assert_eq!(stats.tile_count, 8);
        assert_eq!(stats.total_bytes, 8 * 200);
        assert_eq!(stats.min_tile_bytes, 200);
        assert_eq!(stats.max_tile_bytes, 200);
        assert!((stats.mean_tile_bytes - 200.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_compression_ratio() {
        let cfg = TileConfig::new(1, 1, 0).expect("should succeed");
        let encoder = TileEncoder::new(cfg, 100, 100);
        let frame = make_frame(100, 100);

        let results = encoder
            .encode(&frame, &FixedSizeOp(500))
            .expect("encode should succeed");
        let stats = TileEncodeStats::from_results(&results).expect("should succeed");

        // raw luma = 100 * 100 = 10000 bytes; encoded = 500 → ratio ≈ 0.05
        let ratio = stats.compression_ratio(10000).expect("should succeed");
        assert!((ratio - 0.05).abs() < 1e-9);

        assert!(stats.compression_ratio(0).is_none());
    }

    // ---------- TileResult --------------------------------------------------

    #[test]
    fn test_tile_result_metadata() {
        let coord = TileCoord::new(2, 1, 640, 360, 320, 180, 4);
        let result = TileResult::new(coord.clone(), vec![1, 2, 3]);

        assert_eq!(result.index(), 1 * 4 + 2);
        assert_eq!(result.encoded_size(), 3);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_tile_result_empty() {
        let coord = TileCoord::new(0, 0, 0, 0, 10, 10, 1);
        let result = TileResult::new(coord, vec![]);
        assert!(result.is_empty());
        assert_eq!(result.encoded_size(), 0);
    }

    // ---------- Debug impls -------------------------------------------------

    #[test]
    fn test_tile_encoder_debug() {
        let cfg = TileConfig::new(2, 2, 0).expect("should succeed");
        let encoder = TileEncoder::new(cfg, 1920, 1080);
        let s = format!("{encoder:?}");
        assert!(s.contains("TileEncoder"));
        assert!(s.contains("1920"));
    }
}
