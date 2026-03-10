//! Geometric transformation module.
//!
//! This module provides geometric transformations including:
//!
//! - [`affine`]: Affine transformations (rotation, scaling, translation, shear)
//! - [`perspective`]: Perspective transformations and homography
//!
//! # Example
//!
//! ```
//! use oximedia_cv::transform::AffineTransform;
//!
//! let transform = AffineTransform::identity();
//! let rotated = transform.rotate(45.0_f64.to_radians());
//! ```

pub mod affine;
pub mod perspective;

// Re-export commonly used items
pub use affine::{warp_affine_image, AffineTransform};
pub use perspective::PerspectiveTransform;
