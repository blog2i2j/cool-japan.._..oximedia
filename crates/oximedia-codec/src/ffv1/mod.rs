//! FFV1 (FF Video Codec 1) lossless video codec.
//!
//! This module implements the FFV1 lossless video codec as specified in
//! RFC 9043 / ISO/IEC 24114. FFV1 is a patent-free, intra-frame-only
//! lossless video codec widely used for archival and professional workflows.
//!
//! # Features
//!
//! - Lossless compression (bit-perfect round-trip)
//! - YCbCr 4:2:0, 4:2:2, and 4:4:4 colorspaces
//! - 8-bit sample depth
//! - Range coder entropy coding (version 3)
//! - Per-slice CRC-32 error detection (version 3)
//! - Median prediction for spatial decorrelation
//!
//! # Architecture
//!
//! - [`Ffv1Encoder`] implements [`VideoEncoder`](crate::traits::VideoEncoder)
//! - [`Ffv1Decoder`] implements [`VideoDecoder`](crate::traits::VideoDecoder)
//! - Entropy coding: [`range_coder`] (v3) and [`golomb`] (v0/v1)
//! - Prediction: [`prediction`] (median predictor)
//! - Error detection: [`crc32`] (MPEG-2 CRC-32)
//!
//! # Example
//!
//! ```ignore
//! use oximedia_codec::ffv1::{Ffv1Encoder, Ffv1Decoder};
//! use oximedia_codec::traits::{VideoEncoder, VideoDecoder, EncoderConfig};
//!
//! // Encode
//! let config = EncoderConfig { /* ... */ };
//! let mut encoder = Ffv1Encoder::new(config)?;
//! encoder.send_frame(&frame)?;
//! let packet = encoder.receive_packet()?;
//!
//! // Decode
//! let extradata = encoder.extradata();
//! let mut decoder = Ffv1Decoder::with_extradata(&extradata)?;
//! decoder.send_packet(&packet?.data, 0)?;
//! let decoded_frame = decoder.receive_frame()?;
//! ```

pub mod crc32;
pub mod decoder;
pub mod encoder;
pub mod golomb;
pub mod prediction;
pub mod range_coder;
pub mod types;

// Public re-exports
pub use decoder::Ffv1Decoder;
pub use encoder::Ffv1Encoder;
pub use types::{
    Ffv1ChromaType, Ffv1Colorspace, Ffv1Config, Ffv1Version, SliceHeader, CONTEXT_COUNT,
    INITIAL_STATE,
};
