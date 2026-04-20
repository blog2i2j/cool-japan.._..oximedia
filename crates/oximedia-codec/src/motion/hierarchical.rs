//! Hierarchical motion estimation using image pyramids.
//!
//! This module provides multi-resolution motion estimation that:
//! 1. Builds a pyramid of downsampled images
//! 2. Searches at coarse resolution first
//! 3. Refines motion vectors at finer resolutions
//!
//! This approach significantly reduces computational cost while
//! maintaining good accuracy for large motion vectors.

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::let_and_return)]
#![allow(clippy::manual_let_else)]

use super::diamond::AdaptiveDiamond;
use super::search::{MotionSearch, SearchConfig, SearchContext};
use super::types::{BlockMatch, BlockSize, MotionVector, SearchRange};

/// Maximum number of pyramid levels.
pub const MAX_PYRAMID_LEVELS: usize = 4;

/// Minimum dimension for pyramid level.
pub const MIN_PYRAMID_DIMENSION: usize = 16;

/// Configuration for hierarchical search.
#[derive(Clone, Debug)]
pub struct HierarchicalConfig {
    /// Number of pyramid levels.
    pub levels: usize,
    /// Search range at coarsest level.
    pub coarse_range: SearchRange,
    /// Refinement search range at each finer level.
    pub refine_range: SearchRange,
    /// Enable adaptive level selection.
    pub adaptive_levels: bool,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            levels: 3,
            coarse_range: SearchRange::symmetric(16),
            refine_range: SearchRange::symmetric(4),
            adaptive_levels: true,
        }
    }
}

impl HierarchicalConfig {
    /// Creates a new hierarchical config.
    #[must_use]
    pub const fn new(levels: usize) -> Self {
        Self {
            levels,
            coarse_range: SearchRange::symmetric(16),
            refine_range: SearchRange::symmetric(4),
            adaptive_levels: true,
        }
    }

    /// Sets the number of levels.
    #[must_use]
    pub const fn levels(mut self, levels: usize) -> Self {
        self.levels = levels;
        self
    }

    /// Sets the coarse level search range.
    #[must_use]
    pub const fn coarse_range(mut self, range: SearchRange) -> Self {
        self.coarse_range = range;
        self
    }

    /// Sets the refinement range.
    #[must_use]
    pub const fn refine_range(mut self, range: SearchRange) -> Self {
        self.refine_range = range;
        self
    }
}

/// A single level of the image pyramid.
#[derive(Clone, Debug)]
pub struct PyramidLevel {
    /// Pixel data.
    pub data: Vec<u8>,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Stride (bytes per row).
    pub stride: usize,
    /// Scale factor from original (1, 2, 4, ...).
    pub scale: usize,
}

impl PyramidLevel {
    /// Creates a new pyramid level.
    #[must_use]
    pub fn new(width: usize, height: usize, scale: usize) -> Self {
        let stride = width;
        Self {
            data: vec![0u8; stride * height],
            width,
            height,
            stride,
            scale,
        }
    }

    /// Creates a pyramid level from existing data.
    #[must_use]
    pub fn from_data(data: Vec<u8>, width: usize, height: usize, scale: usize) -> Self {
        let stride = width;
        Self {
            data,
            width,
            height,
            stride,
            scale,
        }
    }

    /// Gets the pixel value at (x, y).
    #[must_use]
    pub fn get_pixel(&self, x: usize, y: usize) -> u8 {
        if x < self.width && y < self.height {
            self.data[y * self.stride + x]
        } else {
            0
        }
    }

    /// Sets the pixel value at (x, y).
    pub fn set_pixel(&mut self, x: usize, y: usize, value: u8) {
        if x < self.width && y < self.height {
            self.data[y * self.stride + x] = value;
        }
    }

    /// Downsamples from another level (2:1).
    pub fn downsample_from(&mut self, src: &PyramidLevel) {
        for y in 0..self.height {
            for x in 0..self.width {
                let src_x = x * 2;
                let src_y = y * 2;

                // 2x2 box filter
                let p00 = u32::from(src.get_pixel(src_x, src_y));
                let p01 = u32::from(src.get_pixel(src_x + 1, src_y));
                let p10 = u32::from(src.get_pixel(src_x, src_y + 1));
                let p11 = u32::from(src.get_pixel(src_x + 1, src_y + 1));

                let avg = ((p00 + p01 + p10 + p11 + 2) / 4) as u8;
                self.set_pixel(x, y, avg);
            }
        }
    }

    /// Returns a slice of data for a block.
    #[must_use]
    pub fn block_data(&self, x: usize, y: usize) -> &[u8] {
        let offset = y * self.stride + x;
        &self.data[offset..]
    }
}

/// Image pyramid for multi-resolution search.
#[derive(Clone, Debug)]
pub struct ImagePyramid {
    /// Pyramid levels (index 0 = original resolution).
    levels: Vec<PyramidLevel>,
}

impl ImagePyramid {
    /// Creates a new empty pyramid.
    #[must_use]
    pub const fn new() -> Self {
        Self { levels: Vec::new() }
    }

    /// Builds the pyramid from source image data.
    pub fn build(&mut self, src: &[u8], width: usize, height: usize, num_levels: usize) {
        self.levels.clear();

        // Level 0: original resolution (copy)
        let level0 = PyramidLevel::from_data(src.to_vec(), width, height, 1);
        self.levels.push(level0);

        // Build downsampled levels
        let mut cur_width = width;
        let mut cur_height = height;
        let mut cur_scale = 1;

        for _ in 1..num_levels {
            cur_width /= 2;
            cur_height /= 2;
            cur_scale *= 2;

            if cur_width < MIN_PYRAMID_DIMENSION || cur_height < MIN_PYRAMID_DIMENSION {
                break;
            }

            if let Some(prev) = self.levels.last() {
                let mut level = PyramidLevel::new(cur_width, cur_height, cur_scale);
                level.downsample_from(prev);
                self.levels.push(level);
            }
        }
    }

    /// Returns the number of levels.
    #[must_use]
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Gets a pyramid level.
    #[must_use]
    pub fn get_level(&self, index: usize) -> Option<&PyramidLevel> {
        self.levels.get(index)
    }

    /// Gets the coarsest level.
    #[must_use]
    pub fn coarsest(&self) -> Option<&PyramidLevel> {
        self.levels.last()
    }

    /// Gets the finest level (original).
    #[must_use]
    pub fn finest(&self) -> Option<&PyramidLevel> {
        self.levels.first()
    }
}

impl Default for ImagePyramid {
    fn default() -> Self {
        Self::new()
    }
}

/// Hierarchical motion search using image pyramids.
#[derive(Clone, Debug)]
pub struct HierarchicalSearch {
    /// Source image pyramid.
    src_pyramid: ImagePyramid,
    /// Reference image pyramid.
    ref_pyramid: ImagePyramid,
    /// Search configuration.
    config: HierarchicalConfig,
    /// Underlying search algorithm.
    searcher: AdaptiveDiamond,
}

impl Default for HierarchicalSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl HierarchicalSearch {
    /// Creates a new hierarchical search.
    #[must_use]
    pub fn new() -> Self {
        Self {
            src_pyramid: ImagePyramid::new(),
            ref_pyramid: ImagePyramid::new(),
            config: HierarchicalConfig::default(),
            searcher: AdaptiveDiamond::new(),
        }
    }

    /// Sets the configuration.
    #[must_use]
    pub fn with_config(mut self, config: HierarchicalConfig) -> Self {
        self.config = config;
        self
    }

    /// Builds pyramids from source and reference frames.
    pub fn build_pyramids(
        &mut self,
        src: &[u8],
        src_width: usize,
        src_height: usize,
        reference: &[u8],
        ref_width: usize,
        ref_height: usize,
    ) {
        let levels = self.config.levels.min(MAX_PYRAMID_LEVELS);
        self.src_pyramid.build(src, src_width, src_height, levels);
        self.ref_pyramid
            .build(reference, ref_width, ref_height, levels);
    }

    /// Performs hierarchical search.
    ///
    /// Searches from coarsest to finest level, using the result from
    /// each level as the starting point for the next.
    pub fn search_hierarchical(
        &self,
        block_x: usize,
        block_y: usize,
        block_size: BlockSize,
        search_config: &SearchConfig,
    ) -> BlockMatch {
        let num_levels = self
            .src_pyramid
            .num_levels()
            .min(self.ref_pyramid.num_levels());

        if num_levels == 0 {
            return BlockMatch::worst();
        }

        // Start at coarsest level
        let mut current_mv = MotionVector::zero();

        // Search from coarsest to finest
        for level_idx in (0..num_levels).rev() {
            let src_level = match self.src_pyramid.get_level(level_idx) {
                Some(l) => l,
                None => continue,
            };
            let ref_level = match self.ref_pyramid.get_level(level_idx) {
                Some(l) => l,
                None => continue,
            };

            // Scale block position for this level
            let scale = src_level.scale;
            let scaled_x = block_x / scale;
            let scaled_y = block_y / scale;
            let scaled_width = block_size.width() / scale;
            let scaled_height = block_size.height() / scale;

            // Skip if block too small
            if scaled_width < 4 || scaled_height < 4 {
                continue;
            }

            // Determine search range for this level
            let level_range = if level_idx == num_levels - 1 {
                self.config.coarse_range
            } else {
                self.config.refine_range
            };

            // Create search config for this level
            let level_config = SearchConfig {
                range: level_range,
                ..search_config.clone()
            };

            // Scale MV from previous level
            let scaled_mv = MotionVector::from_full_pel(
                current_mv.full_pel_x() / scale as i32,
                current_mv.full_pel_y() / scale as i32,
            );

            // Create context for this level
            let src_offset = scaled_y * src_level.stride + scaled_x;
            if src_offset >= src_level.data.len() {
                continue;
            }

            let ctx = SearchContext::new(
                &src_level.data[src_offset..],
                src_level.stride,
                &ref_level.data,
                ref_level.stride,
                BlockSize::Block8x8, // Use fixed size for pyramid levels
                scaled_x,
                scaled_y,
                ref_level.width,
                ref_level.height,
            );

            // Search at this level
            let result = self
                .searcher
                .search_with_predictor(&ctx, &level_config, scaled_mv);

            // Scale MV back up for next level
            current_mv = MotionVector::from_full_pel(
                result.mv.full_pel_x() * scale as i32,
                result.mv.full_pel_y() * scale as i32,
            );
        }

        // Final search at full resolution
        if let (Some(src_level), Some(ref_level)) =
            (self.src_pyramid.finest(), self.ref_pyramid.finest())
        {
            let src_offset = block_y * src_level.stride + block_x;
            if src_offset < src_level.data.len() {
                let ctx = SearchContext::new(
                    &src_level.data[src_offset..],
                    src_level.stride,
                    &ref_level.data,
                    ref_level.stride,
                    block_size,
                    block_x,
                    block_y,
                    ref_level.width,
                    ref_level.height,
                );

                let final_config = SearchConfig {
                    range: self.config.refine_range,
                    ..search_config.clone()
                };

                return self
                    .searcher
                    .search_with_predictor(&ctx, &final_config, current_mv);
            }
        }

        let cost = search_config.mv_cost.rd_cost(&current_mv, u32::MAX);
        BlockMatch::new(current_mv, u32::MAX, cost)
    }

    /// Calculates the optimal number of pyramid levels.
    #[must_use]
    pub fn calculate_levels(width: usize, height: usize, max_levels: usize) -> usize {
        let min_dim = width.min(height);
        let mut levels = 1;

        let mut size = min_dim;
        while size >= MIN_PYRAMID_DIMENSION * 2 && levels < max_levels {
            size /= 2;
            levels += 1;
        }

        levels
    }
}

/// Coarse-to-fine refinement helper.
#[derive(Clone, Debug, Default)]
pub struct CoarseToFineRefiner {
    /// Refinement steps at each scale.
    steps: Vec<RefinementStep>,
}

/// A single refinement step.
#[derive(Clone, Debug)]
pub struct RefinementStep {
    /// Scale factor (1 = full resolution).
    pub scale: usize,
    /// Search range for this step.
    pub range: SearchRange,
    /// Number of iterations.
    pub iterations: u32,
}

impl Default for RefinementStep {
    fn default() -> Self {
        Self {
            scale: 1,
            range: SearchRange::symmetric(2),
            iterations: 4,
        }
    }
}

impl CoarseToFineRefiner {
    /// Creates a new refiner.
    #[must_use]
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Adds a refinement step.
    #[must_use]
    pub fn add_step(mut self, scale: usize, range: i32, iterations: u32) -> Self {
        self.steps.push(RefinementStep {
            scale,
            range: SearchRange::symmetric(range),
            iterations,
        });
        self
    }

    /// Creates default coarse-to-fine steps.
    #[must_use]
    pub fn default_steps() -> Self {
        Self::new()
            .add_step(4, 8, 8) // 1/4 resolution, wide search
            .add_step(2, 4, 6) // 1/2 resolution, medium search
            .add_step(1, 2, 4) // Full resolution, fine search
    }

    /// Returns the refinement steps.
    #[must_use]
    pub fn steps(&self) -> &[RefinementStep] {
        &self.steps
    }

    /// Scales a motion vector between levels.
    #[must_use]
    pub const fn scale_mv(mv: MotionVector, from_scale: usize, to_scale: usize) -> MotionVector {
        if from_scale == to_scale {
            return mv;
        }

        if from_scale > to_scale {
            // Upscaling (coarse to fine)
            let factor = (from_scale / to_scale) as i32;
            MotionVector::new(mv.dx * factor, mv.dy * factor)
        } else {
            // Downscaling (fine to coarse)
            let factor = (to_scale / from_scale) as i32;
            MotionVector::new(mv.dx / factor, mv.dy / factor)
        }
    }
}

/// Resolution scaling utilities.
pub struct ResolutionScaler;

impl ResolutionScaler {
    /// Downsamples image by factor of 2.
    pub fn downsample_2x(src: &[u8], width: usize, height: usize) -> Vec<u8> {
        let new_width = width / 2;
        let new_height = height / 2;
        let mut dst = vec![0u8; new_width * new_height];

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = x * 2;
                let src_y = y * 2;

                let p00 = u32::from(src[src_y * width + src_x]);
                let p01 = u32::from(src[src_y * width + src_x + 1]);
                let p10 = u32::from(src[(src_y + 1) * width + src_x]);
                let p11 = u32::from(src[(src_y + 1) * width + src_x + 1]);

                dst[y * new_width + x] = ((p00 + p01 + p10 + p11 + 2) / 4) as u8;
            }
        }

        dst
    }

    /// Downsamples image by factor of 4.
    pub fn downsample_4x(src: &[u8], width: usize, height: usize) -> Vec<u8> {
        let half = Self::downsample_2x(src, width, height);
        Self::downsample_2x(&half, width / 2, height / 2)
    }

    /// Upsamples motion vector coordinates.
    #[must_use]
    pub const fn upsample_mv(mv: MotionVector, factor: i32) -> MotionVector {
        MotionVector::new(mv.dx * factor, mv.dy * factor)
    }

    /// Downsamples motion vector coordinates.
    #[must_use]
    pub const fn downsample_mv(mv: MotionVector, factor: i32) -> MotionVector {
        MotionVector::new(mv.dx / factor, mv.dy / factor)
    }

    /// Scales block position.
    #[must_use]
    pub const fn scale_position(x: usize, y: usize, scale: usize) -> (usize, usize) {
        (x / scale, y / scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pyramid_level_creation() {
        let level = PyramidLevel::new(64, 64, 1);
        assert_eq!(level.width, 64);
        assert_eq!(level.height, 64);
        assert_eq!(level.scale, 1);
        assert_eq!(level.data.len(), 64 * 64);
    }

    #[test]
    fn test_pyramid_level_pixel_access() {
        let mut level = PyramidLevel::new(8, 8, 1);
        level.set_pixel(3, 4, 128);
        assert_eq!(level.get_pixel(3, 4), 128);
        assert_eq!(level.get_pixel(0, 0), 0);
    }

    #[test]
    fn test_pyramid_level_downsample() {
        let mut src_level = PyramidLevel::new(8, 8, 1);
        // Fill with gradient
        for y in 0..8 {
            for x in 0..8 {
                src_level.set_pixel(x, y, ((x + y) * 16) as u8);
            }
        }

        let mut dst_level = PyramidLevel::new(4, 4, 2);
        dst_level.downsample_from(&src_level);

        // Check that values are averaged
        assert!(dst_level.get_pixel(0, 0) > 0);
        assert!(dst_level.get_pixel(1, 1) > dst_level.get_pixel(0, 0));
    }

    #[test]
    fn test_image_pyramid_build() {
        let src = vec![128u8; 64 * 64];
        let mut pyramid = ImagePyramid::new();
        pyramid.build(&src, 64, 64, 3);

        assert_eq!(pyramid.num_levels(), 3);

        // Check level dimensions
        assert_eq!(pyramid.get_level(0).map(|l| l.width), Some(64));
        assert_eq!(pyramid.get_level(1).map(|l| l.width), Some(32));
        assert_eq!(pyramid.get_level(2).map(|l| l.width), Some(16));
    }

    #[test]
    fn test_image_pyramid_min_size() {
        let src = vec![128u8; 32 * 32];
        let mut pyramid = ImagePyramid::new();
        pyramid.build(&src, 32, 32, 5);

        // Should stop at MIN_PYRAMID_DIMENSION
        assert!(pyramid.num_levels() <= 2);
    }

    #[test]
    fn test_hierarchical_config() {
        let config = HierarchicalConfig::new(4)
            .coarse_range(SearchRange::symmetric(32))
            .refine_range(SearchRange::symmetric(8));

        assert_eq!(config.levels, 4);
        assert_eq!(config.coarse_range.horizontal, 32);
        assert_eq!(config.refine_range.horizontal, 8);
    }

    #[test]
    fn test_hierarchical_search_creation() {
        let search = HierarchicalSearch::new().with_config(HierarchicalConfig::new(3));

        assert_eq!(search.config.levels, 3);
    }

    #[test]
    fn test_calculate_pyramid_levels() {
        assert_eq!(HierarchicalSearch::calculate_levels(128, 128, 4), 4);
        assert_eq!(HierarchicalSearch::calculate_levels(64, 64, 4), 3);
        assert_eq!(HierarchicalSearch::calculate_levels(32, 32, 4), 2);
    }

    #[test]
    fn test_coarse_to_fine_refiner() {
        let refiner = CoarseToFineRefiner::default_steps();
        assert_eq!(refiner.steps().len(), 3);
    }

    #[test]
    fn test_scale_mv() {
        let mv = MotionVector::new(16, 32);

        // Coarse to fine (upscale)
        let scaled_up = CoarseToFineRefiner::scale_mv(mv, 2, 1);
        assert_eq!(scaled_up.dx, 32);
        assert_eq!(scaled_up.dy, 64);

        // Fine to coarse (downscale)
        let scaled_down = CoarseToFineRefiner::scale_mv(mv, 1, 2);
        assert_eq!(scaled_down.dx, 8);
        assert_eq!(scaled_down.dy, 16);
    }

    #[test]
    fn test_resolution_scaler_downsample() {
        // Create 4x4 image with known values
        let src = vec![
            100, 100, 200, 200, 100, 100, 200, 200, 50, 50, 150, 150, 50, 50, 150, 150,
        ];

        let dst = ResolutionScaler::downsample_2x(&src, 4, 4);
        assert_eq!(dst.len(), 4);

        // Check averaged values
        assert_eq!(dst[0], 100); // (100+100+100+100)/4
        assert_eq!(dst[1], 200); // (200+200+200+200)/4
        assert_eq!(dst[2], 50); // (50+50+50+50)/4
        assert_eq!(dst[3], 150); // (150+150+150+150)/4
    }

    #[test]
    fn test_resolution_scaler_mv() {
        let mv = MotionVector::new(8, 16);

        let up = ResolutionScaler::upsample_mv(mv, 2);
        assert_eq!(up.dx, 16);
        assert_eq!(up.dy, 32);

        let down = ResolutionScaler::downsample_mv(mv, 2);
        assert_eq!(down.dx, 4);
        assert_eq!(down.dy, 8);
    }

    #[test]
    fn test_hierarchical_search_integration() {
        let src = vec![100u8; 64 * 64];
        let reference = vec![100u8; 64 * 64];

        let mut search = HierarchicalSearch::new().with_config(HierarchicalConfig::new(3));

        search.build_pyramids(&src, 64, 64, &reference, 64, 64);

        let config = SearchConfig::default();
        let result = search.search_hierarchical(0, 0, BlockSize::Block8x8, &config);

        // Perfect match at origin
        assert_eq!(result.mv.full_pel_x(), 0);
        assert_eq!(result.mv.full_pel_y(), 0);
    }

    #[test]
    fn test_refinement_step() {
        let step = RefinementStep::default();
        assert_eq!(step.scale, 1);
        assert_eq!(step.iterations, 4);
    }
}
