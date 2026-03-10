//! Audio and video codec implementations for `OxiMedia`.
//!
//! This crate provides encoding and decoding for royalty-free codecs:
//!
//! ## Video Codecs
//!
//! - **AV1** - Alliance for Open Media codec (primary)
//! - **VP9** - Google's royalty-free codec
//! - **VP8** - Google's earlier royalty-free codec
//! - **Theora** - Xiph.Org Foundation codec (VP3-based)
//!
//! ## Audio Codecs
//!
//! - **Opus** - Modern low-latency audio codec
//!
//! # Architecture
//!
//! All codecs implement unified traits:
//!
//! - [`VideoDecoder`] - Decode compressed packets to frames
//! - [`VideoEncoder`] - Encode frames to compressed packets
//!
//! # Example
//!
//! ```ignore
//! use oximedia_codec::{VideoDecoder, Av1Decoder};
//!
//! let mut decoder = Av1Decoder::new(&codec_params)?;
//! decoder.send_packet(&packet)?;
//! while let Some(frame) = decoder.receive_frame()? {
//!     // Process decoded frame
//! }
//! ```
//!
//! # Audio Example
//!
//! ```ignore
//! use oximedia_codec::opus::OpusDecoder;
//!
//! let mut decoder = OpusDecoder::new(48000, 2)?;
//! let audio_frame = decoder.decode_packet(&packet_data)?;
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::unused_self)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::self_assignment)]
#![allow(clippy::redundant_else)]
#![allow(clippy::no_effect_underscore_binding)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::if_not_else)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::bool_to_int_with_if)]
#![allow(unused_variables)]
#![allow(unused_imports)]
// Additional clippy allows for codec implementation
#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::explicit_iter_loop)]
#![allow(clippy::implicit_clone)]
#![allow(clippy::assertions_on_constants)]
#![allow(clippy::no_effect)]
#![allow(clippy::unnecessary_operation)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_strip)]
#![allow(clippy::fn_params_excessive_bools)]
#![allow(clippy::struct_excessive_bools)]
#![allow(dead_code)]
#![allow(clippy::missing_fields_in_debug)]
#![allow(clippy::useless_vec)]
#![allow(clippy::used_underscore_binding)]
#![allow(clippy::unnecessary_unwrap)]
#![allow(clippy::needless_late_init)]
#![allow(clippy::never_loop)]
#![allow(clippy::while_let_on_iterator)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::bool_assert_comparison)]
#![allow(clippy::overly_complex_bool_expr)]
#![allow(clippy::double_comparisons)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::float_cmp)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::option_as_ref_deref)]
#![allow(clippy::single_match_else)]
#![allow(clippy::cast_abs_to_unsigned)]
#![allow(clippy::semicolon_if_nothing_returned)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::iter_without_into_iter)]
#![allow(clippy::collapsible_else_if)]
#![allow(clippy::naive_bytecount)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::unnecessary_literal_unwrap)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::match_wildcard_for_single_variants)]
#![allow(clippy::unnecessary_filter_map)]
#![allow(clippy::ref_binding_to_reference)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::borrowed_box)]
#![allow(clippy::needless_question_mark)]
#![allow(clippy::type_complexity)]
#![allow(clippy::bind_instead_of_map)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::ref_option)]
#![allow(clippy::new_without_default)]
#![allow(clippy::erasing_op)]
#![allow(clippy::identity_op)]
#![allow(clippy::op_ref)]
#![allow(clippy::manual_flatten)]
#![allow(clippy::while_let_loop)]
#![allow(clippy::from_over_into)]
#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::collapsible_match)]
#![allow(clippy::inefficient_to_string)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::enum_glob_use)]
#![allow(clippy::cloned_ref_to_slice_refs)]
#![allow(clippy::verbose_bit_mask)]
#![allow(clippy::let_and_return)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::self_only_used_in_recursion)]
#![allow(clippy::unnecessary_map_or)]

pub mod av1_obu;
pub mod bitrate_model;
pub mod bitstream_writer;
pub mod codec_caps;
pub mod color_range;
pub mod entropy_coding;
pub mod error;
pub mod frame;
pub mod frame_types;
pub mod gop_structure;
pub mod hdr;
pub mod intra;
pub mod motion;
pub mod multipass;
pub mod nal_unit;
pub mod packet_queue;
pub mod picture_timing;
pub mod profile_level;
pub mod rate_control;
pub mod reconstruct;
pub mod reference_frames;
pub mod simd;
pub mod slice_header;
pub mod stream_info;
pub mod tile;
pub mod tile_encoder;
pub mod traits;

// Audio support
pub mod audio;

// Standalone SILK frame decoding types
pub mod silk;

// Standalone CELT frame decoding types
pub mod celt;

// PNG codec
pub mod png;

// GIF codec
pub mod gif;

// JPEG-XL codec
#[cfg(feature = "jpegxl")]
pub mod jpegxl;

// Image I/O support
#[cfg(feature = "image-io")]
pub mod image;

#[cfg(feature = "av1")]
pub mod av1;

#[cfg(feature = "vp9")]
pub mod vp9;

#[cfg(feature = "vp8")]
pub mod vp8;

#[cfg(feature = "theora")]
pub mod theora;

#[cfg(feature = "h263")]
pub mod h263;

#[cfg(feature = "opus")]
pub mod opus;

#[cfg(feature = "ffv1")]
pub mod ffv1;

// Re-exports
pub use audio::{AudioFrame, ChannelLayout, SampleFormat};
pub use error::{CodecError, CodecResult};
pub use frame::{
    ColorInfo, ColorPrimaries, FrameType, MatrixCoefficients, Plane, TransferCharacteristics,
    VideoFrame,
};
pub use multipass::{
    allocation::AllocationStrategy, Allocator, Analyzer, Buffer, BufferConfig, ComplexityStats,
    EncoderConfig as MultiPassConfig, EncoderError, EncodingResult, FrameAllocation,
    FrameComplexity, LookaheadAnalysis, LookaheadFrame, MultiPassEncoder, PassStats, PassType,
    SceneChangeDetector, Stats, VbvStatistics,
};
pub use rate_control::{
    AdaptiveQuantization, AllocationResult, AnalysisResult, AqMode, BitrateAllocator, BlockQpMap,
    BufferModel, CbrController, ComplexityEstimator, ContentAnalyzer, ContentType, CqpController,
    CrfController, FrameStats, GopAllocationStatus, GopStats, Lookahead, QpResult, QpSelector,
    QpStrategy, RateControlMode, RateController, RcConfig, RcOutput, SceneChangeThreshold,
    TextureMetrics, VbrController,
};
pub use reconstruct::{
    BufferPool, CdefApplicator, CdefBlockConfig, CdefFilterResult, ChromaSubsampling,
    DeblockFilter, DeblockParams, DecoderPipeline, EdgeFilter, FilmGrainParams,
    FilmGrainSynthesizer, FilterDirection, FilterStrength, FrameBuffer, GrainBlock,
    LoopFilterPipeline, OutputConfig, OutputFormat, OutputFormatter, PipelineConfig, PipelineStage,
    PlaneBuffer, PlaneType, ReconstructResult, ReconstructionError, ResidualBuffer, ResidualPlane,
    StageResult, SuperResConfig, SuperResUpscaler, UpscaleMethod,
};
pub use tile::{
    assemble_tiles, decode_tile_stream, HeaderedTileEncodeOp, RawLumaEncodeOp, TileConfig,
    TileCoord, TileEncodeOp, TileEncodeStats, TileEncoder, TileResult,
};
pub use traits::{
    BitrateMode, DecoderConfig, EncodedPacket, EncoderConfig, EncoderPreset, VideoDecoder,
    VideoEncoder,
};

#[cfg(feature = "av1")]
pub use av1::{Av1Decoder, Av1Encoder};

#[cfg(feature = "vp9")]
pub use vp9::Vp9Decoder;

#[cfg(feature = "vp8")]
pub use vp8::Vp8Decoder;

#[cfg(feature = "theora")]
pub use theora::{TheoraConfig, TheoraDecoder, TheoraEncoder};

#[cfg(feature = "h263")]
pub use h263::{H263Decoder, H263Encoder, PictureFormat};

#[cfg(feature = "opus")]
pub use opus::{OpusDecoder, OpusEncoder, OpusEncoderConfig};

#[cfg(feature = "ffv1")]
pub use ffv1::{Ffv1Decoder, Ffv1Encoder};

#[cfg(feature = "image-io")]
pub use image::{
    convert_rgb_to_yuv420p, convert_yuv420p_to_rgb, rgb_to_yuv, yuv_to_rgb,
    EncoderConfig as ImageEncoderConfig, ImageDecoder, ImageEncoder, ImageFormat,
};

pub use png::{
    batch_encode as batch_encode_png, best_encoder, decode as decode_png,
    encode_grayscale as encode_png_grayscale, encode_rgb as encode_png_rgb,
    encode_rgba as encode_png_rgba, encoder_from_profile, fast_encoder, get_info as png_info,
    is_png, optimize as optimize_png, transcode as transcode_png, validate as validate_png,
    Chromaticity, ColorType as PngColorType, CompressionLevel as PngCompressionLevel,
    DecodedImage as PngImage, EncoderBuilder as PngEncoderBuilder,
    EncoderConfig as PngEncoderConfig, EncodingProfile, EncodingStats, FilterStrategy, FilterType,
    ImageHeader as PngHeader, PaletteEntry, PaletteOptimizer, ParallelPngEncoder,
    PhysicalDimensions, PngDecoder, PngDecoderExtended, PngEncoder, PngEncoderExtended, PngInfo,
    PngMetadata, SignificantBits, TextChunk,
};

pub use gif::{
    is_gif, DisposalMethod, DitheringMethod, GifDecoder, GifEncoder, GifEncoderConfig, GifFrame,
    GifFrameConfig, GraphicsControlExtension, ImageDescriptor, LogicalScreenDescriptor,
    QuantizationMethod,
};

#[cfg(feature = "jpegxl")]
pub use jpegxl::{
    AnsDecoder, AnsDistribution, AnsEncoder, BitReader as JxlBitReader, BitWriter as JxlBitWriter,
    DecodedImage as JxlImage, JxlColorSpace, JxlConfig, JxlDecoder, JxlEncoder, JxlFrameEncoding,
    JxlHeader, ModularDecoder, ModularEncoder,
};
