//! Core types for multimedia processing.
//!
//! This module provides fundamental types used throughout `OxiMedia`:
//!
//! - [`Rational`] - Precise rational number arithmetic for timing
//! - [`Timestamp`] - Timestamps with timebase context
//! - [`PixelFormat`] - Video pixel format definitions
//! - [`SampleFormat`] - Audio sample format definitions
//! - [`CodecId`] - Codec identifiers (Green List only)
//! - [`MediaType`] - Media stream type classification
//! - [`fourcc::FourCc`] - Typed FourCC value type and constants

mod codec_id;
pub mod color_meta;
pub mod fourcc;
mod pixel_format;
mod rational;
mod sample_format;
mod timestamp;

pub use codec_id::{CodecId, MediaType};
pub use color_meta::{ColorPrimaries, ColorRange, MatrixCoefficients, TransferCharacteristics};
pub use fourcc::FourCc;
pub use pixel_format::PixelFormat;
pub use rational::Rational;
pub use sample_format::SampleFormat;
pub use timestamp::Timestamp;
