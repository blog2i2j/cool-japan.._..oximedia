//! Wide-gamut colour space definitions and conversions.
//!
//! This module contains primary chromaticity tables, conversion matrices, and
//! utility functions for individual wide colour gamuts.

pub mod display_p3;

pub use display_p3::{srgb_to_display_p3, DisplayP3};
