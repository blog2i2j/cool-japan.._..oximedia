//! DNG (Digital Negative) RAW image format reader/writer.
//!
//! DNG is an open RAW image format based on TIFF/EP (ISO 12234-2).
//! It preserves the raw sensor data from digital cameras along with
//! rich metadata about the capture conditions.
//!
//! # Features
//!
//! - Full TIFF-based parsing (little-endian and big-endian)
//! - CFA (Color Filter Array) pattern detection (RGGB, BGGR, GRBG, GBRG)
//! - Bit unpacking for 10, 12, 14, and 16-bit sensor data
//! - Bilinear demosaicing (Bayer to RGB)
//! - White balance and color matrix application
//! - DNG writing with uncompressed raw data
//! - Round-trip DNG read/write
//!
//! # Example
//!
//! ```no_run
//! use oximedia_image::dng::{DngReader, DngImage};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let data = std::fs::read("photo.dng")?;
//! if DngReader::is_dng(&data) {
//!     let image = DngReader::read(&data)?;
//!     println!("DNG: {}x{} @ {} bits", image.width, image.height, image.bit_depth);
//! }
//! # Ok(())
//! # }
//! ```

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::similar_names)]

pub mod constants;
pub mod conversion;
pub mod demosaic;
pub(crate) mod parser;
pub mod processing;
pub mod reader;
#[cfg(test)]
mod tests;
pub mod types;
pub mod writer;

// Re-export public API
pub use constants::*;
pub use conversion::{dng_to_image_frame, image_frame_to_dng};
pub use demosaic::demosaic_bilinear;
pub use processing::{apply_color_matrix, apply_white_balance};
pub use reader::DngReader;
pub use types::*;
pub use writer::DngWriter;
