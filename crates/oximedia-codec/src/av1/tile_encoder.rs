//! Tile-based parallel encoding for AV1.
//!
//! This module provides infrastructure for parallel video encoding
//! using tile-based decomposition. Frames are split into independent
//! rectangular tiles that can be encoded concurrently.
//!
//! # Architecture
//!
//! - `TileEncoderConfig`: Configuration for tile splitting and threading
//! - `TileRegion`: Describes a rectangular tile region within a frame
//! - `TileEncoder`: Encodes a single tile region
//! - `ParallelTileEncoder`: Orchestrates parallel encoding of all tiles
//!
//! # Thread Safety
//!
//! All encoding is performed without unsafe code. Rayon's thread pool
//! provides safe parallelism through Rust's ownership system.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use super::tile::TileInfo;
use crate::error::{CodecError, CodecResult};
use crate::frame::VideoFrame;
use rayon::prelude::*;
use std::sync::Arc;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for tile-based encoding.
#[derive(Clone, Debug)]
pub struct TileEncoderConfig {
    /// Number of tile columns (power of 2, 1-64).
    pub tile_cols: u32,
    /// Number of tile rows (power of 2, 1-64).
    pub tile_rows: u32,
    /// Number of encoding threads (0 = auto-detect).
    pub threads: usize,
    /// Superblock size (64 or 128).
    pub sb_size: u32,
    /// Use uniform tile spacing.
    pub uniform_spacing: bool,
    /// Minimum tile width in superblocks.
    pub min_tile_width_sb: u32,
    /// Maximum tile width in superblocks.
    pub max_tile_width_sb: u32,
    /// Minimum tile height in superblocks.
    pub min_tile_height_sb: u32,
    /// Maximum tile height in superblocks.
    pub max_tile_height_sb: u32,
}

impl Default for TileEncoderConfig {
    fn default() -> Self {
        Self {
            tile_cols: 1,
            tile_rows: 1,
            threads: 0,
            sb_size: 64,
            uniform_spacing: true,
            min_tile_width_sb: 1,
            max_tile_width_sb: 64,
            min_tile_height_sb: 1,
            max_tile_height_sb: 64,
        }
    }
}

impl TileEncoderConfig {
    /// Create a new tile encoder config with automatic tile layout.
    ///
    /// Automatically determines optimal tile configuration based on
    /// frame dimensions and thread count.
    #[must_use]
    pub fn auto(width: u32, height: u32, threads: usize) -> Self {
        let mut config = Self::default();
        config.threads = threads;
        config.configure_for_dimensions(width, height);
        config
    }

    /// Create config with manual tile counts.
    ///
    /// # Errors
    ///
    /// Returns error if tile counts are invalid.
    pub fn with_tile_counts(tile_cols: u32, tile_rows: u32, threads: usize) -> CodecResult<Self> {
        if tile_cols == 0 || tile_rows == 0 {
            return Err(CodecError::InvalidParameter(
                "Tile counts must be positive".to_string(),
            ));
        }

        if tile_cols > 64 || tile_rows > 64 {
            return Err(CodecError::InvalidParameter(
                "Maximum 64 tile columns/rows".to_string(),
            ));
        }

        // Ensure power of 2
        if !tile_cols.is_power_of_two() || !tile_rows.is_power_of_two() {
            return Err(CodecError::InvalidParameter(
                "Tile counts must be power of 2".to_string(),
            ));
        }

        Ok(Self {
            tile_cols,
            tile_rows,
            threads,
            ..Default::default()
        })
    }

    /// Configure tile layout for given dimensions.
    pub fn configure_for_dimensions(&mut self, width: u32, height: u32) {
        let sb_cols = width.div_ceil(self.sb_size);
        let sb_rows = height.div_ceil(self.sb_size);

        // Determine thread count
        let thread_count = if self.threads == 0 {
            rayon::current_num_threads()
        } else {
            self.threads
        };

        // Calculate optimal tile counts based on thread count
        let target_tiles = thread_count.next_power_of_two() as u32;

        // Prefer horizontal splitting for wide frames
        let aspect_ratio = width as f32 / height.max(1) as f32;

        if aspect_ratio > 2.0 {
            // Wide frame: more columns than rows
            self.tile_cols = (target_tiles as f32).sqrt().ceil() as u32;
            self.tile_cols = self.tile_cols.next_power_of_two();
            self.tile_rows = (target_tiles / self.tile_cols).max(1);
            self.tile_rows = self.tile_rows.next_power_of_two();
        } else if aspect_ratio < 0.5 {
            // Tall frame: more rows than columns
            self.tile_rows = (target_tiles as f32).sqrt().ceil() as u32;
            self.tile_rows = self.tile_rows.next_power_of_two();
            self.tile_cols = (target_tiles / self.tile_rows).max(1);
            self.tile_cols = self.tile_cols.next_power_of_two();
        } else {
            // Balanced: split evenly
            let sqrt_tiles = (target_tiles as f32).sqrt() as u32;
            self.tile_cols = sqrt_tiles.next_power_of_two();
            self.tile_rows = sqrt_tiles.next_power_of_two();
        }

        // Clamp to valid ranges
        self.tile_cols = self.tile_cols.clamp(1, 64.min(sb_cols));
        self.tile_rows = self.tile_rows.clamp(1, 64.min(sb_rows));

        // Ensure we don't create too many tiles
        while self.tile_cols * self.tile_rows > 4096 {
            if self.tile_cols > self.tile_rows {
                self.tile_cols /= 2;
            } else {
                self.tile_rows /= 2;
            }
        }
    }

    /// Get total number of tiles.
    #[must_use]
    pub const fn tile_count(&self) -> u32 {
        self.tile_cols * self.tile_rows
    }

    /// Get effective thread count.
    #[must_use]
    pub fn thread_count(&self) -> usize {
        if self.threads == 0 {
            rayon::current_num_threads()
        } else {
            self.threads
        }
    }

    /// Validate configuration.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn validate(&self) -> CodecResult<()> {
        if self.tile_cols == 0 || self.tile_rows == 0 {
            return Err(CodecError::InvalidParameter(
                "Tile counts must be positive".to_string(),
            ));
        }

        if self.tile_cols > 64 || self.tile_rows > 64 {
            return Err(CodecError::InvalidParameter(
                "Maximum 64 tile columns/rows".to_string(),
            ));
        }

        if self.tile_count() > 4096 {
            return Err(CodecError::InvalidParameter(
                "Maximum 4096 total tiles".to_string(),
            ));
        }

        if self.sb_size != 64 && self.sb_size != 128 {
            return Err(CodecError::InvalidParameter(
                "Superblock size must be 64 or 128".to_string(),
            ));
        }

        Ok(())
    }
}

// =============================================================================
// Tile Region
// =============================================================================

/// Describes a rectangular tile region within a frame.
#[derive(Clone, Debug)]
pub struct TileRegion {
    /// Tile column index.
    pub col: u32,
    /// Tile row index.
    pub row: u32,
    /// X offset in pixels.
    pub x: u32,
    /// Y offset in pixels.
    pub y: u32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Tile index in raster order.
    pub index: u32,
}

impl TileRegion {
    /// Create a new tile region.
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

    /// Check if this region is valid.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        self.width > 0 && self.height > 0
    }

    /// Get area in pixels.
    #[must_use]
    pub const fn area(&self) -> u32 {
        self.width * self.height
    }

    /// Check if this tile is at the left edge.
    #[must_use]
    pub const fn is_left_edge(&self) -> bool {
        self.col == 0
    }

    /// Check if this tile is at the top edge.
    #[must_use]
    pub const fn is_top_edge(&self) -> bool {
        self.row == 0
    }
}

// =============================================================================
// Tile Frame Splitter
// =============================================================================

/// Splits frames into tile regions for parallel encoding.
#[derive(Clone, Debug)]
pub struct TileFrameSplitter {
    /// Encoder configuration.
    config: TileEncoderConfig,
    /// Frame width.
    frame_width: u32,
    /// Frame height.
    frame_height: u32,
    /// Tile regions.
    regions: Vec<TileRegion>,
}

impl TileFrameSplitter {
    /// Create a new tile frame splitter.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(
        config: TileEncoderConfig,
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

    /// Compute tile regions.
    fn compute_regions(&mut self) {
        self.regions.clear();

        if self.config.uniform_spacing {
            self.compute_uniform_regions();
        } else {
            self.compute_custom_regions();
        }
    }

    /// Compute uniformly-spaced tile regions.
    fn compute_uniform_regions(&mut self) {
        let tile_width = self.frame_width.div_ceil(self.config.tile_cols);
        let tile_height = self.frame_height.div_ceil(self.config.tile_rows);

        for row in 0..self.config.tile_rows {
            for col in 0..self.config.tile_cols {
                let x = col * tile_width;
                let y = row * tile_height;

                let width = if col == self.config.tile_cols - 1 {
                    self.frame_width - x
                } else {
                    tile_width
                };

                let height = if row == self.config.tile_rows - 1 {
                    self.frame_height - y
                } else {
                    tile_height
                };

                self.regions.push(TileRegion::new(
                    col,
                    row,
                    x,
                    y,
                    width,
                    height,
                    self.config.tile_cols,
                ));
            }
        }
    }

    /// Compute custom tile regions (non-uniform).
    fn compute_custom_regions(&mut self) {
        // For simplicity, fall back to uniform spacing
        // A full implementation would support custom tile sizes
        self.compute_uniform_regions();
    }

    /// Get all tile regions.
    #[must_use]
    pub fn regions(&self) -> &[TileRegion] {
        &self.regions
    }

    /// Get tile region by index.
    #[must_use]
    pub fn region(&self, index: usize) -> Option<&TileRegion> {
        self.regions.get(index)
    }

    /// Get number of tiles.
    #[must_use]
    pub fn tile_count(&self) -> usize {
        self.regions.len()
    }
}

// =============================================================================
// Tile Encoder
// =============================================================================

/// Encodes a single tile region.
#[derive(Clone, Debug)]
pub struct TileEncoder {
    /// Tile region being encoded.
    region: TileRegion,
    /// Quality parameter.
    quality: u8,
    /// Frame is keyframe.
    is_keyframe: bool,
}

impl TileEncoder {
    /// Create a new tile encoder.
    #[must_use]
    pub const fn new(region: TileRegion, quality: u8, is_keyframe: bool) -> Self {
        Self {
            region,
            quality,
            is_keyframe,
        }
    }

    /// Encode a tile region from a frame.
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails.
    pub fn encode(&self, frame: &VideoFrame) -> CodecResult<TileEncodedData> {
        // Validate region is within frame bounds
        if self.region.x + self.region.width > frame.width
            || self.region.y + self.region.height > frame.height
        {
            return Err(CodecError::InvalidParameter(
                "Tile region exceeds frame bounds".to_string(),
            ));
        }

        // Extract tile data from frame
        let tile_data = self.extract_tile_data(frame)?;

        // Encode tile data
        let encoded = self.encode_tile_data(&tile_data)?;

        Ok(TileEncodedData {
            region: self.region.clone(),
            data: encoded,
            size: 0, // Will be set during serialization
        })
    }

    /// Extract tile region data from frame.
    fn extract_tile_data(&self, frame: &VideoFrame) -> CodecResult<Vec<u8>> {
        let mut tile_pixels = Vec::new();

        // Extract Y plane
        if let Some(y_plane) = frame.planes.first() {
            for y in self.region.y..(self.region.y + self.region.height) {
                let row_start = (y as usize * y_plane.stride) + self.region.x as usize;
                let row_end = row_start + self.region.width as usize;
                if row_end <= y_plane.data.len() {
                    tile_pixels.extend_from_slice(&y_plane.data[row_start..row_end]);
                }
            }
        }

        // For YUV420, extract U and V planes with chroma subsampling
        let chroma_x = self.region.x / 2;
        let chroma_y = self.region.y / 2;
        let chroma_width = self.region.width / 2;
        let chroma_height = self.region.height / 2;

        for plane_idx in 1..frame.planes.len() {
            if let Some(plane) = frame.planes.get(plane_idx) {
                for y in chroma_y..(chroma_y + chroma_height) {
                    let row_start = (y as usize * plane.stride) + chroma_x as usize;
                    let row_end = row_start + chroma_width as usize;
                    if row_end <= plane.data.len() {
                        tile_pixels.extend_from_slice(&plane.data[row_start..row_end]);
                    }
                }
            }
        }

        Ok(tile_pixels)
    }

    /// Encode tile pixel data.
    fn encode_tile_data(&self, _tile_data: &[u8]) -> CodecResult<Vec<u8>> {
        // Simplified encoding: just wrap with minimal header
        // Real implementation would perform actual AV1 tile encoding:
        // - Transform coding (DCT/ADST)
        // - Quantization
        // - Entropy coding
        // - Loop filtering within tile

        let mut encoded = Vec::new();

        // Tile header (simplified)
        encoded.push(if self.is_keyframe { 0x80 } else { 0x00 });
        encoded.push(self.quality);

        // Tile size placeholders
        encoded.extend_from_slice(&(self.region.width).to_le_bytes());
        encoded.extend_from_slice(&(self.region.height).to_le_bytes());

        // Placeholder compressed data (zeros for now)
        let compressed_size = (self.region.width * self.region.height / 32) as usize;
        encoded.resize(encoded.len() + compressed_size, 0);

        Ok(encoded)
    }

    /// Get tile region.
    #[must_use]
    pub const fn region(&self) -> &TileRegion {
        &self.region
    }
}

// =============================================================================
// Encoded Tile Data
// =============================================================================

/// Encoded tile data result.
#[derive(Clone, Debug)]
pub struct TileEncodedData {
    /// Source tile region.
    pub region: TileRegion,
    /// Encoded bitstream data.
    pub data: Vec<u8>,
    /// Size in bytes (for OBU serialization).
    pub size: usize,
}

impl TileEncodedData {
    /// Get tile index.
    #[must_use]
    pub const fn index(&self) -> u32 {
        self.region.index
    }

    /// Get encoded size.
    #[must_use]
    pub fn encoded_size(&self) -> usize {
        self.data.len()
    }
}

// =============================================================================
// Parallel Tile Encoder
// =============================================================================

/// Orchestrates parallel encoding of all tiles in a frame.
#[derive(Debug)]
pub struct ParallelTileEncoder {
    /// Tile encoder configuration.
    config: Arc<TileEncoderConfig>,
    /// Frame splitter.
    splitter: TileFrameSplitter,
    /// Frame width.
    frame_width: u32,
    /// Frame height.
    frame_height: u32,
}

impl ParallelTileEncoder {
    /// Create a new parallel tile encoder.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(
        config: TileEncoderConfig,
        frame_width: u32,
        frame_height: u32,
    ) -> CodecResult<Self> {
        let splitter = TileFrameSplitter::new(config.clone(), frame_width, frame_height)?;

        Ok(Self {
            config: Arc::new(config),
            splitter,
            frame_width,
            frame_height,
        })
    }

    /// Encode a frame using parallel tile encoding.
    ///
    /// # Errors
    ///
    /// Returns error if encoding fails.
    pub fn encode_frame(
        &self,
        frame: &VideoFrame,
        quality: u8,
        is_keyframe: bool,
    ) -> CodecResult<Vec<TileEncodedData>> {
        // Validate frame dimensions
        if frame.width != self.frame_width || frame.height != self.frame_height {
            return Err(CodecError::InvalidParameter(format!(
                "Frame dimensions {}x{} don't match encoder {}x{}",
                frame.width, frame.height, self.frame_width, self.frame_height
            )));
        }

        // Configure rayon thread pool if specified
        if self.config.threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.config.threads)
                .build()
                .map_err(|e| {
                    CodecError::Internal(format!("Failed to create thread pool: {}", e))
                })?;
        }

        // Encode all tiles in parallel
        let encoded_tiles: Vec<CodecResult<TileEncodedData>> = self
            .splitter
            .regions()
            .par_iter()
            .map(|region| {
                let encoder = TileEncoder::new(region.clone(), quality, is_keyframe);
                encoder.encode(frame)
            })
            .collect();

        // Collect results and handle errors
        let mut tiles = Vec::with_capacity(encoded_tiles.len());
        for result in encoded_tiles {
            tiles.push(result?);
        }

        // Sort tiles by index to maintain raster order
        tiles.sort_by_key(TileEncodedData::index);

        Ok(tiles)
    }

    /// Merge encoded tiles into a single bitstream.
    ///
    /// # Errors
    ///
    /// Returns error if merging fails.
    pub fn merge_tiles(&self, tiles: &[TileEncodedData]) -> CodecResult<Vec<u8>> {
        if tiles.is_empty() {
            return Ok(Vec::new());
        }

        let mut merged = Vec::new();

        // Write tile group header
        self.write_tile_group_header(&mut merged, tiles.len() as u32);

        // Write each tile's data with size prefix (except last)
        for (i, tile) in tiles.iter().enumerate() {
            let is_last = i == tiles.len() - 1;

            if !is_last {
                // Write tile size (little-endian, variable-length)
                let size = tile.data.len() as u32;
                self.write_tile_size(&mut merged, size);
            }

            // Write tile data
            merged.extend_from_slice(&tile.data);
        }

        Ok(merged)
    }

    /// Write tile group header.
    fn write_tile_group_header(&self, output: &mut Vec<u8>, _num_tiles: u32) {
        // Simplified tile group header
        // Real implementation would write proper AV1 OBU tile group header
        if self.config.tile_count() > 1 {
            output.push(0x01); // Tile group marker
        }
    }

    /// Write tile size.
    fn write_tile_size(&self, output: &mut Vec<u8>, size: u32) {
        // Write as 4-byte little-endian
        output.extend_from_slice(&size.to_le_bytes());
    }

    /// Get tile configuration.
    #[must_use]
    pub fn config(&self) -> &TileEncoderConfig {
        &self.config
    }

    /// Get tile count.
    #[must_use]
    pub fn tile_count(&self) -> usize {
        self.splitter.tile_count()
    }

    /// Get tile regions.
    #[must_use]
    pub fn regions(&self) -> &[TileRegion] {
        self.splitter.regions()
    }
}

// =============================================================================
// Tile Info Builder
// =============================================================================

/// Builds TileInfo from encoder configuration.
pub struct TileInfoBuilder;

impl TileInfoBuilder {
    /// Build TileInfo from encoder configuration.
    #[must_use]
    pub fn from_config(
        config: &TileEncoderConfig,
        frame_width: u32,
        frame_height: u32,
    ) -> TileInfo {
        let sb_cols = frame_width.div_ceil(config.sb_size);
        let sb_rows = frame_height.div_ceil(config.sb_size);

        let tile_width_sb = sb_cols.div_ceil(config.tile_cols);
        let tile_height_sb = sb_rows.div_ceil(config.tile_rows);

        // Build column starts
        let mut tile_col_starts = Vec::new();
        for i in 0..=config.tile_cols {
            let start = (i * tile_width_sb).min(sb_cols);
            tile_col_starts.push(start);
        }

        // Build row starts
        let mut tile_row_starts = Vec::new();
        for i in 0..=config.tile_rows {
            let start = (i * tile_height_sb).min(sb_rows);
            tile_row_starts.push(start);
        }

        let tile_cols_log2 = (config.tile_cols as f32).log2() as u8;
        let tile_rows_log2 = (config.tile_rows as f32).log2() as u8;

        TileInfo {
            tile_cols: config.tile_cols,
            tile_rows: config.tile_rows,
            tile_col_starts,
            tile_row_starts,
            context_update_tile_id: 0,
            tile_size_bytes: 4,
            uniform_tile_spacing: config.uniform_spacing,
            tile_cols_log2,
            tile_rows_log2,
            min_tile_cols_log2: 0,
            max_tile_cols_log2: 6,
            min_tile_rows_log2: 0,
            max_tile_rows_log2: 6,
            sb_cols,
            sb_rows,
            sb_size: config.sb_size,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_tile_encoder_config_default() {
        let config = TileEncoderConfig::default();
        assert_eq!(config.tile_cols, 1);
        assert_eq!(config.tile_rows, 1);
        assert_eq!(config.tile_count(), 1);
    }

    #[test]
    fn test_tile_encoder_config_auto() {
        let config = TileEncoderConfig::auto(1920, 1080, 4);
        assert!(config.tile_count() > 0);
        assert!(config.tile_count() <= 4096);
    }

    #[test]
    fn test_tile_encoder_config_manual() {
        let config = TileEncoderConfig::with_tile_counts(2, 2, 4).expect("should succeed");
        assert_eq!(config.tile_cols, 2);
        assert_eq!(config.tile_rows, 2);
        assert_eq!(config.tile_count(), 4);
    }

    #[test]
    fn test_tile_encoder_config_validation() {
        let config = TileEncoderConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid = TileEncoderConfig::default();
        invalid.tile_cols = 0;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_tile_region() {
        let region = TileRegion::new(0, 0, 0, 0, 640, 480, 2);
        assert!(region.is_valid());
        assert_eq!(region.area(), 640 * 480);
        assert!(region.is_left_edge());
        assert!(region.is_top_edge());
    }

    #[test]
    fn test_tile_frame_splitter() {
        let config = TileEncoderConfig::with_tile_counts(2, 2, 4).expect("should succeed");
        let splitter = TileFrameSplitter::new(config, 1920, 1080).expect("should succeed");

        assert_eq!(splitter.tile_count(), 4);
        assert_eq!(splitter.regions().len(), 4);

        // Check first tile
        let region = splitter.region(0).expect("should succeed");
        assert_eq!(region.col, 0);
        assert_eq!(region.row, 0);
        assert_eq!(region.x, 0);
        assert_eq!(region.y, 0);
    }

    #[test]
    fn test_tile_encoder() {
        let region = TileRegion::new(0, 0, 0, 0, 320, 240, 1);
        let encoder = TileEncoder::new(region, 128, true);

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
        frame.allocate();

        let result = encoder.encode(&frame);
        assert!(result.is_ok());

        let encoded = result.expect("should succeed");
        assert!(encoded.encoded_size() > 0);
    }

    #[test]
    fn test_parallel_tile_encoder() {
        let config = TileEncoderConfig::with_tile_counts(2, 2, 4).expect("should succeed");
        let encoder = ParallelTileEncoder::new(config, 1920, 1080).expect("should succeed");

        assert_eq!(encoder.tile_count(), 4);
        assert_eq!(encoder.regions().len(), 4);
    }

    #[test]
    fn test_parallel_encode_frame() {
        let config = TileEncoderConfig::with_tile_counts(2, 2, 4).expect("should succeed");
        let encoder = ParallelTileEncoder::new(config, 1920, 1080).expect("should succeed");

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
        frame.allocate();

        let result = encoder.encode_frame(&frame, 128, true);
        assert!(result.is_ok());

        let tiles = result.expect("should succeed");
        assert_eq!(tiles.len(), 4);
    }

    #[test]
    fn test_merge_tiles() {
        let config = TileEncoderConfig::with_tile_counts(2, 2, 4).expect("should succeed");
        let encoder = ParallelTileEncoder::new(config, 1920, 1080).expect("should succeed");

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 1920, 1080);
        frame.allocate();

        let tiles = encoder
            .encode_frame(&frame, 128, true)
            .expect("should succeed");
        let merged = encoder.merge_tiles(&tiles);

        assert!(merged.is_ok());
        assert!(!merged.expect("should succeed").is_empty());
    }

    #[test]
    fn test_tile_info_builder() {
        let config = TileEncoderConfig::with_tile_counts(2, 2, 4).expect("should succeed");
        let tile_info = TileInfoBuilder::from_config(&config, 1920, 1080);

        assert_eq!(tile_info.tile_cols, 2);
        assert_eq!(tile_info.tile_rows, 2);
        assert_eq!(tile_info.tile_count(), 4);
    }

    #[test]
    fn test_aspect_ratio_configuration() {
        // Wide frame
        let mut config = TileEncoderConfig::default();
        config.configure_for_dimensions(3840, 1080);
        assert!(config.tile_cols >= config.tile_rows);

        // Tall frame
        let mut config = TileEncoderConfig::default();
        config.configure_for_dimensions(1080, 3840);
        assert!(config.tile_rows >= config.tile_cols);
    }
}
