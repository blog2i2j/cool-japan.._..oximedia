//! High-level operations built on compute kernels.

pub mod affine_gpu;
pub mod color;
pub mod convolution;
pub mod dct_gpu;
pub mod deinterlace;
pub mod histogram_gpu;
pub mod motion;
pub mod scale;

// Re-export the alpha blending API at the ops level for convenience.
pub use color::{alpha_blend, alpha_blend_rgba};

// Re-export GPU operations
pub use affine_gpu::{apply_affine, AffineTransform};
pub use dct_gpu::{
    forward_dct_8x8, forward_dct_batch, inverse_dct_8x8, inverse_dct_batch, DctBlock,
};
pub use histogram_gpu::{compute_histogram, GpuHistogram, HISTOGRAM_BINS};
