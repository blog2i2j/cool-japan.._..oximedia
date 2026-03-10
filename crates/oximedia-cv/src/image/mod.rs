//! Image processing operations.
//!
//! This module provides various image processing algorithms including:
//!
//! - [`resize`]: Image resizing with multiple interpolation methods
//! - [`convert`]: Color space conversions (RGB, YUV, HSV, etc.)
//! - [`histogram`]: Histogram computation and equalization
//! - [`filter`]: Image filtering (blur, sharpen, etc.)
//! - [`edge`]: Edge detection algorithms
//! - [`morph`]: Morphological operations

pub mod convert;
pub mod edge;
pub mod filter;
pub mod histogram;
pub mod histogram_ext;
pub mod morph;
pub mod resize;

// Re-export commonly used items
pub use convert::{bgr_to_yuv420p, rgb_to_grayscale_bt601, ColorSpace};
pub use edge::{CannyEdge, EdgeDetector, LaplacianEdge, SobelEdge};
pub use filter::{BilateralFilter, BoxBlur, GaussianBlur, ImageFilter, MedianFilter};
pub use histogram::Histogram;
pub use morph::{MorphOperation, StructuringElement};
pub use resize::ResizeMethod;
