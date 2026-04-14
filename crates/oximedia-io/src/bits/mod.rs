//! Bit-level reading utilities for parsing binary formats.
//!
//! This module provides the [`BitReader`] type for reading individual bits
//! and multi-bit values from byte arrays, along with support for Exp-Golomb
//! coded integers used in video codecs like H.264/AVC.
//!
//! # Example
//!
//! ```
//! use oximedia_io::bits::BitReader;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let data = [0b10110100, 0b11001010];
//! let mut reader = BitReader::new(&data);
//!
//! // Read 4 bits at a time
//! assert_eq!(reader.read_bits(4)?, 0b1011);
//! assert_eq!(reader.read_bits(4)?, 0b0100);
//! # Ok(())
//! # }
//! ```

mod exp_golomb;
mod reader;

pub use reader::BitReader;
