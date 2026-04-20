//! APV (Advanced Professional Video) codec implementation.
//!
//! APV is a royalty-free, intra-frame professional video codec standardized as
//! ISO/IEC 23009-13. It is designed for professional video production workflows
//! requiring high quality, low latency, and random-access editing.
//!
//! # Architecture
//!
//! APV-S (Simple Profile) encoding pipeline:
//!
//! ```text
//! Input Frame → YCbCr conversion → Tile partitioning
//!   → 8x8 DCT → Quantization → Zigzag scan → Entropy coding
//!   → APV Access Unit (header + tile data)
//! ```
//!
//! Key characteristics:
//! - **Intra-frame only**: each frame is independently encoded (all keyframes)
//! - **8x8 DCT**: orthonormal Type-II DCT with f64 precision
//! - **QP-based quantization**: exponential QP-to-step mapping (6 steps ≈ 2×)
//! - **Exp-Golomb entropy**: variable-length coding with zero run-length
//! - **Tile structure**: frame divided into tiles for parallel processing
//! - **Bit depths**: 8-bit, 10-bit, 12-bit
//! - **Chroma formats**: 4:2:0, 4:2:2, 4:4:4
//!
//! # Example
//!
//! ```ignore
//! use oximedia_codec::apv::{ApvEncoder, ApvDecoder, ApvConfig};
//! use oximedia_codec::traits::{VideoEncoder, VideoDecoder};
//!
//! // Encode
//! let config = ApvConfig::new(640, 480)?.with_qp(22);
//! let mut encoder = ApvEncoder::new(config)?;
//! encoder.send_frame(&video_frame)?;
//! while let Some(packet) = encoder.receive_packet()? {
//!     // Write APV packet to container
//! }
//!
//! // Decode
//! let mut decoder = ApvDecoder::new();
//! decoder.send_packet(&packet_data, pts)?;
//! while let Some(frame) = decoder.receive_frame()? {
//!     // Process decoded frame
//! }
//! ```

pub mod dct;
pub mod decoder;
pub mod encoder;
pub mod entropy;
pub mod types;

pub use decoder::ApvDecoder;
pub use encoder::ApvEncoder;
pub use types::{
    ApvBitDepth, ApvChromaFormat, ApvConfig, ApvError, ApvFrameHeader, ApvProfile, ApvTileInfo,
};
