//! WebP codec support.
//!
//! Provides VP8 (lossy) and VP8L (lossless) WebP encoding, alpha channel
//! handling, and RIFF container parsing/writing.

pub mod alpha;
pub mod encoder;
pub mod riff;
pub mod vp8l_decoder;
pub mod vp8l_encoder;

pub use alpha::{
    decode_alpha, encode_alpha, AlphaCompression, AlphaFilter, AlphaHeader,
};
pub use encoder::WebPLossyEncoder;
pub use riff::{ChunkType, RiffChunk, Vp8xFeatures, WebPContainer, WebPEncoding, WebPWriter};
pub use vp8l_decoder::{
    ColorTransformElement, DecodedImage, HuffmanCode, HuffmanTree, Transform, Vp8lBitReader,
    Vp8lDecoder, Vp8lHeader,
};
pub use vp8l_encoder::{Vp8lBitWriter, Vp8lEncoder};
