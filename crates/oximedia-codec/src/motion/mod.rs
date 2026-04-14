//! Motion estimation module for video encoders.
//!
//! This module provides comprehensive motion estimation functionality for
//! inter-frame prediction in video encoding. Motion estimation finds the
//! best matching block in a reference frame for each block in the current
//! frame, enabling efficient temporal compression.
//!
//! # Architecture
//!
//! The module is organized into several submodules:
//!
//! - [`types`] - Core types: `MotionVector`, `BlockMatch`, `SearchRange`, etc.
//! - [`search`] - Motion search algorithms: full search, diamond, hexagon, UMH
//! - [`diamond`] - Diamond search patterns (SDSP, LDSP)
//! - [`hierarchical`] - Multi-resolution pyramid-based search
//! - [`subpel`] - Sub-pixel refinement and SATD computation
//! - [`predictor`] - MV prediction from spatial/temporal neighbors
//! - [`partition`] - Block partitioning decisions
//! - [`cache`] - MV caching for improved performance
//!
//! # Usage Example
//!
//! ```ignore
//! use oximedia_codec::motion::{
//!     MotionVector, SearchConfig, SearchRange, DiamondSearch, MotionSearch,
//! };
//!
//! // Configure search
//! let config = SearchConfig::default()
//!     .range(SearchRange::symmetric(32))
//!     .early_termination(true);
//!
//! // Create search algorithm
//! let searcher = DiamondSearch::new();
//!
//! // Perform search
//! let result = searcher.search(&context, &config);
//! println!("Best MV: ({}, {}), SAD: {}", result.mv.dx, result.mv.dy, result.sad);
//! ```
//!
//! # Search Algorithms
//!
//! The module provides several motion search algorithms with different
//! speed/quality tradeoffs:
//!
//! | Algorithm | Speed | Quality | Use Case |
//! |-----------|-------|---------|----------|
//! | `FullSearch` | Slow | Best | Reference, small ranges |
//! | `DiamondSearch` | Fast | Good | General purpose |
//! | `HexagonSearch` | Fast | Good | Alternative to diamond |
//! | `UmhSearch` | Medium | Very Good | High quality encoding |
//! | `HierarchicalSearch` | Medium | Good | Large motion, HD content |
//!
//! # Sub-pixel Precision
//!
//! Motion vectors support multiple precision levels:
//!
//! - **Full-pel** - Integer pixel precision
//! - **Half-pel** - 1/2 pixel precision
//! - **Quarter-pel** - 1/4 pixel precision (common in H.264/AV1)
//! - **Eighth-pel** - 1/8 pixel precision (VP9)
//!
//! Sub-pixel positions are interpolated using filters defined in [`subpel`].

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]

pub mod cache;
pub mod diamond;
pub mod hierarchical;
pub mod partition;
pub mod predictor;
pub mod search;
pub mod subpel;
pub mod types;

// Re-export primary types for convenient access
pub use cache::{CacheManager, CoLocatedMvLookup, MvCache, MvCacheEntry, RefFrameMvs};
pub use diamond::{
    AdaptiveDiamond, CrossDiamond, ExtendedDiamond, HexagonalSearch, LargeDiamond,
    PredictorDiamond, SmallDiamond, UMHexSearch,
};
pub use hierarchical::{
    CoarseToFineRefiner, HierarchicalConfig, HierarchicalSearch, ImagePyramid, PyramidLevel,
};
pub use partition::{
    InterMode, MergeCandidate, MergeCandidateList, PartitionContext, PartitionDecider,
    PartitionDecision, PartitionType, SkipDetector, SplitDecision,
};
pub use predictor::{
    MvCandidate, MvCostCalculator, MvPredContext, MvPredictor, MvPredictorList, MvpMode,
    NeighborInfo, NeighborPosition, SpatialPredictor, TemporalPredictor,
};
pub use search::{
    AdaptiveSearch, DiamondSearch, FullSearch, HexagonSearch, MotionSearch, SearchConfig,
    SearchContext, ThreeStepSearch, UmhSearch,
};
pub use subpel::{
    HadamardTransform, HalfPelFilter, HalfPelInterpolator, QuarterPelFilter,
    QuarterPelInterpolator, SatdCalculator, SubpelConfig, SubpelPatterns, SubpelRefiner,
};
pub use types::{BlockMatch, BlockSize, MotionVector, MvCost, MvPrecision, SearchRange};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_imports() {
        // Verify that primary types can be used
        let mv = MotionVector::new(10, 20);
        assert_eq!(mv.dx, 10);
        assert_eq!(mv.dy, 20);

        let range = SearchRange::symmetric(32);
        assert_eq!(range.horizontal, 32);

        let block_match = BlockMatch::zero_mv(100);
        assert_eq!(block_match.sad, 100);
    }

    #[test]
    fn test_search_algorithm_creation() {
        let _full = FullSearch::new();
        let _diamond = DiamondSearch::new();
        let _hexagon = HexagonSearch::new();
        let _umh = UmhSearch::new();
        let _adaptive = AdaptiveSearch::new();
        let _three_step = ThreeStepSearch::new();
    }

    #[test]
    fn test_diamond_patterns() {
        let small = SmallDiamond::new();
        let large = LargeDiamond::new();

        assert_eq!(small.size(), 4);
        assert_eq!(large.size(), 8);
    }

    #[test]
    fn test_predictor_creation() {
        let predictor = MvPredictor::new();
        let mvp = predictor.best_mvp();
        assert!(mvp.is_zero());
    }

    #[test]
    fn test_cache_creation() {
        let cache = MvCache::new();
        assert_eq!(cache.mi_cols(), 0);
        assert_eq!(cache.mi_rows(), 0);
    }

    #[test]
    fn test_partition_types() {
        assert_eq!(PartitionType::None.num_parts(), 1);
        assert_eq!(PartitionType::Split.num_parts(), 4);
    }

    #[test]
    fn test_subpel_components() {
        let _satd = SatdCalculator::new();
        let identical = vec![128u8; 16];
        let result = SatdCalculator::satd_4x4(&identical, 4, &identical, 4);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_hierarchical_components() {
        let pyramid = ImagePyramid::new();
        assert_eq!(pyramid.num_levels(), 0);

        let config = HierarchicalConfig::new(3);
        assert_eq!(config.levels, 3);
    }

    #[test]
    fn test_mv_precision() {
        assert_eq!(MvPrecision::FullPel.fractional_bits(), 0);
        assert_eq!(MvPrecision::HalfPel.fractional_bits(), 1);
        assert_eq!(MvPrecision::QuarterPel.fractional_bits(), 2);
        assert_eq!(MvPrecision::EighthPel.fractional_bits(), 3);
    }

    #[test]
    fn test_block_sizes() {
        assert_eq!(BlockSize::Block4x4.width(), 4);
        assert_eq!(BlockSize::Block8x8.width(), 8);
        assert_eq!(BlockSize::Block16x16.width(), 16);
        assert_eq!(BlockSize::Block64x64.width(), 64);
        assert_eq!(BlockSize::Block128x128.width(), 128);
    }

    #[test]
    fn test_integration_search_workflow() {
        // Create test data
        let src = vec![100u8; 64]; // 8x8 block
        let mut reference = vec![50u8; 256]; // 16x16 frame

        // Place matching block at (4, 4)
        for row in 0..8 {
            for col in 0..8 {
                reference[(row + 4) * 16 + col + 4] = 100;
            }
        }

        // Setup context
        let ctx = SearchContext::new(&src, 8, &reference, 16, BlockSize::Block8x8, 0, 0, 16, 16);

        // Configure search
        let config = SearchConfig::default().range(SearchRange::symmetric(8));

        // Search with different algorithms
        let full_result = FullSearch::new().search(&ctx, &config);
        let diamond_result = DiamondSearch::new().search(&ctx, &config);
        let hex_result = HexagonSearch::new().search(&ctx, &config);

        // All should find reasonably good matches
        assert!(full_result.sad < 1000);
        assert!(diamond_result.sad < 1000);
        assert!(hex_result.sad < 1000);

        // Full search should find optimal
        assert_eq!(full_result.mv.full_pel_x(), 4);
        assert_eq!(full_result.mv.full_pel_y(), 4);
    }
}
