//! AV1 codec implementation.
//!
//! AV1 (`AOMedia` Video 1) is a royalty-free video codec developed by the
//! Alliance for Open Media. It provides excellent compression efficiency
//! and is the primary video codec for `OxiMedia`.
//!
//! # Features
//!
//! - OBU (Open Bitstream Unit) parsing
//! - 10-bit and 12-bit HDR support
//! - Film grain synthesis
//! - Scalable video coding (SVC)
//! - Reference frame management
//!
//! # Architecture
//!
//! The AV1 implementation is split into several modules:
//!
//! - `obu` - OBU parsing and serialization
//! - `sequence` - Sequence header handling
//! - `frame_header` - Frame header parsing
//! - `decoder` - Frame decoding
//! - `encoder` - Frame encoding
//! - `frame` - Frame-level processing
//! - `tile` - Tile processing
//! - `transform` - DCT/ADST transforms
//! - `prediction` - Intra/inter prediction
//! - `entropy` - Entropy coding (symbol coding)
//! - `entropy_tables` - Default CDF tables for entropy coding
//! - `coefficients` - Transform coefficient parsing
//! - `block` - Block-level data structures
//! - `loop_filter` - Loop filter parameters
//! - `cdef` - CDEF (Constrained Directional Enhancement Filter)
//! - `quantization` - Quantization parameters and dequantizer tables
//!
//! # Example
//!
//! ```ignore
//! use oximedia_codec::{Av1Decoder, DecoderConfig, VideoDecoder};
//!
//! let config = DecoderConfig::default();
//! let mut decoder = Av1Decoder::new(config)?;
//!
//! decoder.send_packet(bitstream_data, pts)?;
//! while let Some(frame) = decoder.receive_frame()? {
//!     // Process decoded frame
//! }
//! ```

mod block;
mod cdef;
mod coeff_decode;
mod coeff_encode;
mod coefficients;
pub mod conformance;
mod decoder;
mod encoder;
mod entropy;
mod entropy_encoder;
mod entropy_tables;
pub mod film_grain;
pub mod film_grain_perblock;
pub mod film_grain_table;
mod frame;
mod frame_header;
mod loop_filter;
mod loop_optimization;
mod mode_decision;
mod obu;
mod parallel_tile_decoder;
pub mod parallel_tile_encoder;
mod prediction;
mod quantization;
mod sequence;
mod symbols;
mod tile;
mod tile_encoder;
mod transform;

// Public re-exports for main types
pub use block::{
    BlockContextManager, BlockModeInfo, BlockSize, InterMode, IntraMode, PartitionType,
    PlaneBlockContext,
};
pub use cdef::{CdefParams, CdefStrength};
pub use coeff_decode::{BatchedCoeffDecoder, CoeffAnalysis, CoeffDecoder};
pub use coeff_encode::{
    dequantize_coeffs, quantize_coeffs, CoeffEncoder as TransformCoeffEncoder,
    CoeffStats as TransformCoeffStats,
};
pub use coefficients::{
    CoeffBuffer, CoeffContext, CoeffStats, EobContext, EobPt, LevelContext, ScanOrderCache,
};
pub use conformance::{ObuValidator, SequenceHeaderValidator, ValidationResult};
pub use decoder::Av1Decoder;
pub use encoder::Av1Encoder;
pub use entropy::{ArithmeticDecoder, SymbolReader, SymbolWriter};
pub use entropy_encoder::{
    ArithmeticEncoder as Av1ArithmeticEncoder, BitstreamWriter, ObuWriter,
    SymbolEncoder as Av1SymbolEncoder,
};
pub use entropy_tables::{CdfContext, CDF_PROB_BITS, CDF_PROB_TOP};
pub use film_grain::{
    FilmGrainParams as Av1FilmGrainParams, FilmGrainSynthesizer, ScalingPoint, GRAIN_BLOCK_SIZE,
    MAX_AR_COEFFS_CHROMA, MAX_AR_COEFFS_LUMA, MAX_AR_LAG, MAX_CHROMA_SCALING_POINTS,
    MAX_LUMA_SCALING_POINTS,
};
pub use film_grain_perblock::{apply_grain_per_block_bilinear, GrainBlock, BLEND_ZONE};
pub use film_grain_table::{
    BlockGrainOverride, FilmGrainTable, GrainIntensity, GrainPatternBuilder, GrainPreset,
    PerBlockGrainTable,
};
pub use frame_header::{
    FrameHeader, FrameSize, FrameType, GlobalMotion, InterpolationFilter, ReferenceMode,
    RenderSize, SegmentationParams,
};
pub use loop_filter::LoopFilterParams;
pub use loop_optimization::{
    CdefOptimizer, FilmGrainParams, LoopFilterOptimizer, LoopOptimizer, RestorationOptimizer,
    RestorationType,
};
pub use mode_decision::{
    compute_lambda_from_qp, ModeCandidate, ModeDecision, ModeDecisionConfig, PredictionMode,
};
pub use obu::{ObuHeader, ObuType};
pub use parallel_tile_decoder::{ParallelTileDecoder, TileJob};
pub use parallel_tile_encoder::{
    encode_tiles_parallel, EncodedTile, ParallelTileEncoder as RawParallelTileEncoder,
    TileEncoderConfig as RawTileEncoderConfig, TileRegionInfo,
};
pub use prediction::PredictionEngine;
pub use quantization::{
    apply_adaptive_qm, get_ac_dequant, get_dc_dequant, qindex_to_qp, qp_to_qindex,
    select_adaptive_qm, AdaptiveQmSelection, DeltaQState, QmContentType, QuantizationParams,
    AC_QLOOKUP, DC_QLOOKUP, MAX_Q_INDEX, MIN_Q_INDEX, NUM_QM_LEVELS, QINDEX_RANGE,
};
pub use sequence::{
    OperatingPoint, SequenceHeader, SpatialLayerConfig, SvcConfig, SvcReferenceMode,
    TemporalLayerConfig, MAX_OPERATING_POINTS, MAX_SPATIAL_LAYERS, MAX_TEMPORAL_LAYERS,
};
pub use symbols::{MvPredictor, SymbolDecoder, SymbolEncoder};
pub use tile::{TileData, TileGroup, TileGroupObu, TileInfo};
pub use tile_encoder::{
    ParallelTileEncoder, TileEncodedData, TileEncoder, TileEncoderConfig, TileFrameSplitter,
    TileInfoBuilder, TileRegion,
};
pub use transform::{Transform2D, TransformContext, TxClass, TxSize, TxSizeSqr, TxType, TxType1D};
