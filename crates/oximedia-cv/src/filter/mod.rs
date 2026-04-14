//! Image filtering and convolution modules.
//!
//! This module groups filter-based image processing algorithms.
//!
//! # Modules
//!
//! - [`canny`]: Canny edge detection with hysteresis thresholding

pub mod canny;

pub use canny::CannyEdge;
