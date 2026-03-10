//! JPEG-XL (ISO/IEC 18181) codec implementation.
//!
//! This module provides encoding and decoding for JPEG-XL images with:
//!
//! ## Features
//!
//! - **Lossless mode**: Modular encoding with reversible color transform and
//!   adaptive prediction (gradient + weighted average of neighbors)
//! - **8-bit and 16-bit** sample support
//! - **RGB, RGBA, and Grayscale** color types
//! - **Codestream format**: Direct JXL codestream (0xFF 0x0A signature)
//!
//! ## Architecture
//!
//! JPEG-XL lossless encoding uses the Modular sub-codec:
//!
//! 1. Apply Reversible Color Transform (RCT) to decorrelate channels
//! 2. Predict each sample using adaptive weighted predictor
//! 3. Entropy-code the residuals using ANS (Asymmetric Numeral Systems)
//!
//! ## Examples
//!
//! ### Lossless encoding
//!
//! ```ignore
//! use oximedia_codec::jpegxl::{JxlEncoder, JxlConfig};
//!
//! let encoder = JxlEncoder::lossless();
//! let jxl_data = encoder.encode(&rgb_pixels, 1920, 1080, 3, 8)?;
//! ```
//!
//! ### Decoding
//!
//! ```ignore
//! use oximedia_codec::jpegxl::JxlDecoder;
//!
//! let decoder = JxlDecoder::new();
//! let image = decoder.decode(&jxl_data)?;
//! println!("Decoded {}x{} image", image.width, image.height);
//! ```
//!
//! ## Safety
//!
//! This implementation uses no unsafe code and is fully memory-safe.
//! 100% pure Rust with no C/Fortran dependencies.

mod bitreader;
mod decoder;
mod encoder;
mod entropy;
mod modular;
mod types;

// Re-export public API
pub use bitreader::{BitReader, BitWriter};
pub use decoder::{DecodedImage, JxlDecoder};
pub use encoder::JxlEncoder;
pub use entropy::{AnsDecoder, AnsDistribution, AnsEncoder};
pub use modular::{forward_rct, inverse_rct, ModularDecoder, ModularEncoder};
pub use types::{JxlColorSpace, JxlConfig, JxlFrameEncoding, JxlHeader};
