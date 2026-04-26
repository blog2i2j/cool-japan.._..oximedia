//! Transform operations (DCT, FFT, geometric transforms)

use crate::{GpuDevice, Result};

/// Transform operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformType {
    /// Discrete Cosine Transform (DCT)
    DCT,
    /// Inverse DCT
    IDCT,
    /// Fast Fourier Transform (FFT)
    FFT,
    /// Inverse FFT
    IFFT,
    /// Rotate 90 degrees
    Rotate90,
    /// Rotate 180 degrees
    Rotate180,
    /// Rotate 270 degrees
    Rotate270,
    /// Flip horizontal
    FlipHorizontal,
    /// Flip vertical
    FlipVertical,
    /// Transpose
    Transpose,
    /// Affine transform
    Affine,
    /// Perspective transform
    Perspective,
}

/// Transform kernel for frequency domain and geometric operations
pub struct TransformKernel {
    transform_type: TransformType,
}

impl TransformKernel {
    /// Create a new transform kernel
    #[must_use]
    pub fn new(transform_type: TransformType) -> Self {
        Self { transform_type }
    }

    /// Create a DCT transform kernel
    #[must_use]
    pub fn dct() -> Self {
        Self::new(TransformType::DCT)
    }

    /// Create an IDCT transform kernel
    #[must_use]
    pub fn idct() -> Self {
        Self::new(TransformType::IDCT)
    }

    /// Create a rotate kernel
    #[must_use]
    pub fn rotate(degrees: i32) -> Self {
        let transform_type = match degrees % 360 {
            90 | -270 => TransformType::Rotate90,
            180 | -180 => TransformType::Rotate180,
            270 | -90 => TransformType::Rotate270,
            _ => TransformType::Rotate90, // Default
        };
        Self::new(transform_type)
    }

    /// Create a flip kernel
    #[must_use]
    pub fn flip(horizontal: bool) -> Self {
        let transform_type = if horizontal {
            TransformType::FlipHorizontal
        } else {
            TransformType::FlipVertical
        };
        Self::new(transform_type)
    }

    /// Execute the transform operation (frequency-domain, f32 data).
    ///
    /// Handles DCT and IDCT which operate on `f32` frequency-domain data.
    /// For pixel-level geometric transforms (rotate, flip, transpose) use
    /// [`TransformKernel::execute_u8`] instead.
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `input` - Input data buffer
    /// * `output` - Output data buffer
    /// * `width` - Data width
    /// * `height` - Data height
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails or is not supported for f32 data.
    pub fn execute(
        &self,
        device: &GpuDevice,
        input: &[f32],
        output: &mut [f32],
        width: u32,
        height: u32,
    ) -> Result<()> {
        match self.transform_type {
            TransformType::DCT => {
                crate::ops::TransformOperation::dct_2d(device, input, output, width, height)
            }
            TransformType::IDCT => {
                crate::ops::TransformOperation::idct_2d(device, input, output, width, height)
            }
            TransformType::FFT
            | TransformType::IFFT
            | TransformType::Affine
            | TransformType::Perspective => Err(crate::GpuError::NotSupported(format!(
                "Transform type {:?} not yet implemented",
                self.transform_type
            ))),
            _ => Err(crate::GpuError::NotSupported(format!(
                "Transform type {:?} requires u8 pixel data — use execute_u8()",
                self.transform_type
            ))),
        }
    }

    /// Execute a geometric pixel transform on an interleaved `u8` image buffer.
    ///
    /// Handles `Rotate90`, `Rotate180`, `Rotate270`, `FlipHorizontal`,
    /// `FlipVertical`, and `Transpose`.  `FFT`, `IFFT`, `Affine`, and
    /// `Perspective` are deliberately left as `NotSupported`.
    ///
    /// The `_device` parameter is accepted for API symmetry but is not used
    /// by the CPU-side implementations (the geometric ops are fully pure-Rust).
    ///
    /// # Arguments
    ///
    /// * `_device` - GPU device (unused; present for API consistency)
    /// * `input` - Input pixel buffer (`width * height * channels` bytes)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `channels` - Bytes per pixel (e.g. 3 for RGB, 4 for RGBA)
    ///
    /// # Errors
    ///
    /// Returns [`crate::GpuError::NotSupported`] for frequency-domain,
    /// `Affine`, and `Perspective` transform types.
    pub fn execute_u8(
        &self,
        _device: &GpuDevice,
        input: &[u8],
        width: u32,
        height: u32,
        channels: u32,
    ) -> Result<Vec<u8>> {
        match self.transform_type {
            TransformType::Rotate90 => Ok(crate::ops::TransformOperation::rotate90(
                input, width, height, channels,
            )),
            TransformType::Rotate180 => Ok(crate::ops::TransformOperation::rotate180(
                input, width, height, channels,
            )),
            TransformType::Rotate270 => Ok(crate::ops::TransformOperation::rotate270(
                input, width, height, channels,
            )),
            TransformType::FlipHorizontal => Ok(crate::ops::TransformOperation::flip_horizontal(
                input, width, height, channels,
            )),
            TransformType::FlipVertical => Ok(crate::ops::TransformOperation::flip_vertical(
                input, width, height, channels,
            )),
            TransformType::Transpose => Ok(crate::ops::TransformOperation::transpose(
                input, width, height, channels,
            )),
            TransformType::FFT
            | TransformType::IFFT
            | TransformType::Affine
            | TransformType::Perspective => Err(crate::GpuError::NotSupported(format!(
                "Transform type {:?} not yet implemented",
                self.transform_type
            ))),
            TransformType::DCT | TransformType::IDCT => {
                Err(crate::GpuError::NotSupported(format!(
                    "Transform type {:?} operates on f32 data — use execute()",
                    self.transform_type
                )))
            }
        }
    }

    /// Get the transform type
    #[must_use]
    pub fn transform_type(&self) -> TransformType {
        self.transform_type
    }

    /// Check if this is a frequency domain transform
    #[must_use]
    pub fn is_frequency_domain(&self) -> bool {
        matches!(
            self.transform_type,
            TransformType::DCT | TransformType::IDCT | TransformType::FFT | TransformType::IFFT
        )
    }

    /// Check if this is a geometric transform
    #[must_use]
    pub fn is_geometric(&self) -> bool {
        matches!(
            self.transform_type,
            TransformType::Rotate90
                | TransformType::Rotate180
                | TransformType::Rotate270
                | TransformType::FlipHorizontal
                | TransformType::FlipVertical
                | TransformType::Transpose
                | TransformType::Affine
                | TransformType::Perspective
        )
    }

    /// Estimate FLOPS for the transform operation
    #[must_use]
    pub fn estimate_flops(width: u32, height: u32, transform_type: TransformType) -> u64 {
        let n = u64::from(width) * u64::from(height);

        match transform_type {
            TransformType::DCT | TransformType::IDCT => {
                // DCT complexity: O(N^2 log N) for 2D
                let log_n = (n as f64).log2().ceil() as u64;
                n * n * log_n
            }
            TransformType::FFT | TransformType::IFFT => {
                // FFT complexity: O(N log N)
                let log_n = (n as f64).log2().ceil() as u64;
                n * log_n * 5 // 5 ops per butterfly
            }
            _ => {
                // Geometric transforms: O(N)
                n
            }
        }
    }
}

/// Affine transformation matrix
#[derive(Debug, Clone, Copy)]
pub struct AffineMatrix {
    /// Matrix elements [a, b, c, d, tx, ty]
    /// [ a  b  tx ]
    /// [ c  d  ty ]
    /// [ 0  0  1  ]
    pub elements: [f32; 6],
}

impl AffineMatrix {
    /// Create an identity matrix
    #[must_use]
    pub fn identity() -> Self {
        Self {
            elements: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        }
    }

    /// Create a translation matrix
    #[must_use]
    pub fn translation(tx: f32, ty: f32) -> Self {
        Self {
            elements: [1.0, 0.0, tx, 0.0, 1.0, ty],
        }
    }

    /// Create a rotation matrix
    #[must_use]
    pub fn rotation(angle_radians: f32) -> Self {
        let cos = angle_radians.cos();
        let sin = angle_radians.sin();
        Self {
            elements: [cos, -sin, 0.0, sin, cos, 0.0],
        }
    }

    /// Create a scaling matrix
    #[must_use]
    pub fn scaling(sx: f32, sy: f32) -> Self {
        Self {
            elements: [sx, 0.0, 0.0, 0.0, sy, 0.0],
        }
    }

    /// Combine two affine transformations
    #[must_use]
    pub fn combine(&self, other: &Self) -> Self {
        let a1 = self.elements;
        let a2 = other.elements;

        Self {
            elements: [
                a1[0] * a2[0] + a1[1] * a2[3],
                a1[0] * a2[1] + a1[1] * a2[4],
                a1[0] * a2[2] + a1[1] * a2[5] + a1[2],
                a1[3] * a2[0] + a1[4] * a2[3],
                a1[3] * a2[1] + a1[4] * a2[4],
                a1[3] * a2[2] + a1[4] * a2[5] + a1[5],
            ],
        }
    }

    /// Get matrix elements
    #[must_use]
    pub fn as_array(&self) -> [f32; 6] {
        self.elements
    }
}

impl Default for AffineMatrix {
    fn default() -> Self {
        Self::identity()
    }
}

/// Warp kernel for geometric transformations
pub struct WarpKernel {
    matrix: AffineMatrix,
}

impl WarpKernel {
    /// Create a new warp kernel
    #[must_use]
    pub fn new(matrix: AffineMatrix) -> Self {
        Self { matrix }
    }

    /// Create a rotation warp
    #[must_use]
    pub fn rotation(angle_degrees: f32, center_x: f32, center_y: f32) -> Self {
        let angle_radians = angle_degrees.to_radians();

        // Translate to origin, rotate, translate back
        let t1 = AffineMatrix::translation(-center_x, -center_y);
        let r = AffineMatrix::rotation(angle_radians);
        let t2 = AffineMatrix::translation(center_x, center_y);

        let matrix = t1.combine(&r).combine(&t2);

        Self::new(matrix)
    }

    /// Create a scaling warp
    #[must_use]
    pub fn scaling(sx: f32, sy: f32, center_x: f32, center_y: f32) -> Self {
        let t1 = AffineMatrix::translation(-center_x, -center_y);
        let s = AffineMatrix::scaling(sx, sy);
        let t2 = AffineMatrix::translation(center_x, center_y);

        let matrix = t1.combine(&s).combine(&t2);

        Self::new(matrix)
    }

    /// Get the transformation matrix
    #[must_use]
    pub fn matrix(&self) -> &AffineMatrix {
        &self.matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_kernel_creation() {
        let kernel = TransformKernel::dct();
        assert_eq!(kernel.transform_type(), TransformType::DCT);
        assert!(kernel.is_frequency_domain());
        assert!(!kernel.is_geometric());

        let kernel = TransformKernel::rotate(90);
        assert_eq!(kernel.transform_type(), TransformType::Rotate90);
        assert!(!kernel.is_frequency_domain());
        assert!(kernel.is_geometric());
    }

    #[test]
    fn test_affine_matrix_identity() {
        let identity = AffineMatrix::identity();
        let elements = identity.as_array();
        assert_eq!(elements, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_affine_matrix_translation() {
        let trans = AffineMatrix::translation(10.0, 20.0);
        let elements = trans.as_array();
        assert_eq!(elements[2], 10.0);
        assert_eq!(elements[5], 20.0);
    }

    #[test]
    fn test_affine_matrix_scaling() {
        let scale = AffineMatrix::scaling(2.0, 3.0);
        let elements = scale.as_array();
        assert_eq!(elements[0], 2.0);
        assert_eq!(elements[4], 3.0);
    }

    #[test]
    fn test_affine_matrix_combination() {
        let t1 = AffineMatrix::translation(10.0, 20.0);
        let s = AffineMatrix::scaling(2.0, 2.0);
        let combined = t1.combine(&s);

        // The result should be a combined transformation
        assert!(combined.elements[0] > 0.0);
    }

    #[test]
    fn test_flops_estimation() {
        let flops_dct = TransformKernel::estimate_flops(64, 64, TransformType::DCT);
        let flops_rotate = TransformKernel::estimate_flops(64, 64, TransformType::Rotate90);

        assert!(flops_dct > 0);
        assert!(flops_rotate > 0);
        assert!(flops_dct > flops_rotate); // DCT should be more expensive
    }
}
