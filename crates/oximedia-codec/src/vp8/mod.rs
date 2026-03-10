//! VP8 codec implementation.
//!
//! This module provides a pure Rust VP8 decoder based on RFC 6386.
//! VP8 is a royalty-free video codec developed by Google as part of
//! the `WebM` project.
//!
//! # Features
//!
//! - Boolean arithmetic decoder for entropy coding
//! - Frame header parsing for keyframes and inter frames
//! - Reference frame management (last, golden, altref)
//! - 4x4 DCT/IDCT transforms
//! - Intra prediction (DC, V, H, TM modes)
//! - Inter prediction with sub-pixel motion compensation
//! - Loop filter (deblocking)
//!
//! # Codec Details
//!
//! VP8 always outputs YUV 4:2:0 planar format (`Yuv420p`).
//! It supports two frame types:
//! - Keyframes: Can be decoded independently
//! - Inter frames: Use motion compensation from reference frames
//!
//! # Architecture
//!
//! VP8 operates on 16x16 macroblocks which can use:
//! - Intra prediction: I16 (16x16) or I4 (4x4) modes
//! - Inter prediction: Motion compensation with quarter-pixel precision
//! - Transform: 4x4 DCT or WHT for DC coefficients
//! - Loop filter: Deblocking filter at macroblock boundaries
//!
//! # Examples
//!
//! ```
//! use oximedia_codec::vp8::{Vp8Decoder, FrameHeader, FrameType};
//! use oximedia_codec::traits::{VideoDecoder, DecoderConfig};
//!
//! // Create a decoder
//! let config = DecoderConfig::default();
//! let mut decoder = Vp8Decoder::new(config)?;
//!
//! // Parse a frame header directly
//! let header_data = [
//!     0x10, 0x00, 0x00,       // frame tag
//!     0x9D, 0x01, 0x2A,       // sync code
//!     0x40, 0x01, 0xF0, 0x00, // 320x240
//! ];
//! let header = FrameHeader::parse(&header_data)?;
//! assert!(header.is_keyframe());
//! assert_eq!(header.width, 320);
//! assert_eq!(header.height, 240);
//!
//! // Decode a frame
//! decoder.send_packet(&header_data, 0)?;
//! if let Some(frame) = decoder.receive_frame()? {
//!     assert_eq!(frame.width, 320);
//!     assert_eq!(frame.height, 240);
//! }
//! ```
//!
//! # References
//!
//! - [RFC 6386: VP8 Data Format and Decoding Guide](https://tools.ietf.org/html/rfc6386)
//! - [WebM Project](https://www.webmproject.org/)

mod bool_decoder;
mod dct;
mod decoder;
mod frame_header;
mod loopfilter;
mod mb_mode;
mod motion;
mod prediction;

pub use bool_decoder::BoolDecoder;
pub use dct::{dequantize_block, dequantize_coeff, idct4x4, iwht4x4, Block4x4, PixelBlock4x4};
pub use decoder::Vp8Decoder;
pub use frame_header::{ClampingType, ColorSpace, FrameHeader, FrameType};
pub use loopfilter::{
    calculate_filter_params, filter_horizontal_edge, filter_vertical_edge, LoopFilterConfig,
    MAX_LOOP_FILTER, MAX_SHARPNESS,
};
pub use mb_mode::{
    ChromaMode, InterMode, IntraMode16, IntraMode4, MacroblockType, PartitionType, RefFrame,
    NUM_CHROMA_MODES, NUM_I16_MODES, NUM_I4_MODES, NUM_MV_MODES,
};
pub use motion::{clamp_mv, motion_compensate, MotionVector};
pub use prediction::{predict_chroma, predict_intra_16x16, predict_intra_4x4};
