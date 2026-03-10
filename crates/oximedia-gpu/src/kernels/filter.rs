//! Image filtering and convolution kernels

use crate::{GpuDevice, Result};

/// Filter operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    /// Gaussian blur
    GaussianBlur,
    /// Box blur (simple average)
    BoxBlur,
    /// Median filter
    Median,
    /// Bilateral filter (edge-preserving)
    Bilateral,
    /// Unsharp mask (sharpening)
    UnsharpMask,
    /// Sobel edge detection
    Sobel,
    /// Scharr edge detection
    Scharr,
    /// Laplacian edge detection
    Laplacian,
    /// Custom convolution
    Custom,
}

/// Convolution kernel
pub struct ConvolutionKernel {
    kernel: Vec<f32>,
    width: u32,
    height: u32,
    normalize: bool,
}

impl ConvolutionKernel {
    /// Create a new convolution kernel
    ///
    /// # Arguments
    ///
    /// * `kernel` - Kernel weights (must be width * height in size)
    /// * `width` - Kernel width (must be odd)
    /// * `height` - Kernel height (must be odd)
    /// * `normalize` - Whether to normalize the kernel
    pub fn new(kernel: Vec<f32>, width: u32, height: u32, normalize: bool) -> Result<Self> {
        if kernel.len() != (width * height) as usize {
            return Err(crate::GpuError::Internal(
                "Kernel size mismatch".to_string(),
            ));
        }

        if width % 2 == 0 || height % 2 == 0 {
            return Err(crate::GpuError::Internal(
                "Kernel dimensions must be odd".to_string(),
            ));
        }

        Ok(Self {
            kernel,
            width,
            height,
            normalize,
        })
    }

    /// Create a Sobel X kernel (3x3)
    #[must_use]
    pub fn sobel_x() -> Self {
        Self::new(
            vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0],
            3,
            3,
            false,
        )
        .expect("hardcoded 3x3 Sobel X kernel is always valid")
    }

    /// Create a Sobel Y kernel (3x3)
    #[must_use]
    pub fn sobel_y() -> Self {
        Self::new(
            vec![-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0],
            3,
            3,
            false,
        )
        .expect("hardcoded 3x3 Sobel Y kernel is always valid")
    }

    /// Create a Laplacian kernel (3x3)
    #[must_use]
    pub fn laplacian() -> Self {
        Self::new(
            vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0],
            3,
            3,
            false,
        )
        .expect("hardcoded 3x3 Laplacian kernel is always valid")
    }

    /// Create a box blur kernel
    pub fn box_blur(size: u32) -> Result<Self> {
        let total = (size * size) as usize;
        let value = 1.0 / total as f32;
        let kernel = vec![value; total];
        Self::new(kernel, size, size, false)
    }

    /// Create a sharpening kernel
    #[must_use]
    pub fn sharpen() -> Self {
        Self::new(
            vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
            3,
            3,
            false,
        )
        .expect("hardcoded 3x3 sharpen kernel is always valid")
    }

    /// Get the kernel data
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.kernel
    }

    /// Get kernel dimensions
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Check if normalization is enabled
    #[must_use]
    pub fn is_normalized(&self) -> bool {
        self.normalize
    }

    /// Apply the convolution kernel
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `input` - Input image buffer
    /// * `output` - Output image buffer
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn apply(
        &self,
        device: &GpuDevice,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        crate::ops::FilterOperation::convolve(
            device,
            input,
            output,
            width,
            height,
            &self.kernel,
            self.normalize,
        )
    }
}

/// Image filter kernel
pub struct FilterKernel {
    filter_type: FilterType,
    sigma: f32,
    kernel_size: u32,
}

impl FilterKernel {
    /// Create a new filter kernel
    #[must_use]
    pub fn new(filter_type: FilterType, sigma: f32, kernel_size: u32) -> Self {
        Self {
            filter_type,
            sigma,
            kernel_size,
        }
    }

    /// Create a Gaussian blur filter
    #[must_use]
    pub fn gaussian_blur(sigma: f32) -> Self {
        let kernel_size = Self::gaussian_kernel_size(sigma);
        Self::new(FilterType::GaussianBlur, sigma, kernel_size)
    }

    /// Create a box blur filter
    #[must_use]
    pub fn box_blur(radius: u32) -> Self {
        let kernel_size = radius * 2 + 1;
        Self::new(FilterType::BoxBlur, 0.0, kernel_size)
    }

    /// Create an unsharp mask filter (sharpening)
    #[must_use]
    pub fn sharpen(amount: f32) -> Self {
        Self::new(FilterType::UnsharpMask, amount, 5)
    }

    /// Create a Sobel edge detection filter
    #[must_use]
    pub fn sobel() -> Self {
        Self::new(FilterType::Sobel, 0.0, 3)
    }

    /// Create a bilateral filter
    #[must_use]
    pub fn bilateral(sigma_spatial: f32, _sigma_range: f32) -> Self {
        let kernel_size = Self::gaussian_kernel_size(sigma_spatial);
        Self::new(FilterType::Bilateral, sigma_spatial, kernel_size)
    }

    /// Execute the filter operation
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `input` - Input image buffer
    /// * `output` - Output image buffer
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn execute(
        &self,
        device: &GpuDevice,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        match self.filter_type {
            FilterType::GaussianBlur => crate::ops::FilterOperation::gaussian_blur(
                device, input, output, width, height, self.sigma,
            ),
            FilterType::UnsharpMask => crate::ops::FilterOperation::sharpen(
                device, input, output, width, height, self.sigma,
            ),
            FilterType::Sobel | FilterType::Scharr | FilterType::Laplacian => {
                crate::ops::FilterOperation::edge_detect(device, input, output, width, height)
            }
            _ => Err(crate::GpuError::NotSupported(format!(
                "Filter type {:?} not yet implemented",
                self.filter_type
            ))),
        }
    }

    /// Get the filter type
    #[must_use]
    pub fn filter_type(&self) -> FilterType {
        self.filter_type
    }

    /// Get the sigma parameter
    #[must_use]
    pub fn sigma(&self) -> f32 {
        self.sigma
    }

    /// Get the kernel size
    #[must_use]
    pub fn kernel_size(&self) -> u32 {
        self.kernel_size
    }

    /// Calculate Gaussian kernel size from sigma
    fn gaussian_kernel_size(sigma: f32) -> u32 {
        let radius = (3.0 * sigma).ceil() as u32;
        2 * radius + 1
    }

    /// Estimate FLOPS for filter operation
    #[must_use]
    pub fn estimate_flops(width: u32, height: u32, kernel_size: u32) -> u64 {
        let pixels = u64::from(width) * u64::from(height);
        let ops_per_pixel = u64::from(kernel_size) * u64::from(kernel_size) * 4; // 4 channels
        pixels * ops_per_pixel * 2 // multiply + add
    }
}

/// Separable filter for optimized 2D filtering
pub struct SeparableFilter {
    horizontal_kernel: Vec<f32>,
    vertical_kernel: Vec<f32>,
}

impl SeparableFilter {
    /// Create a new separable filter
    #[must_use]
    pub fn new(horizontal: Vec<f32>, vertical: Vec<f32>) -> Self {
        Self {
            horizontal_kernel: horizontal,
            vertical_kernel: vertical,
        }
    }

    /// Create a Gaussian separable filter
    #[must_use]
    pub fn gaussian(sigma: f32, size: u32) -> Self {
        let kernel = Self::gaussian_kernel_1d(sigma, size);
        Self::new(kernel.clone(), kernel)
    }

    /// Generate 1D Gaussian kernel
    fn gaussian_kernel_1d(sigma: f32, size: u32) -> Vec<f32> {
        let radius = (size / 2) as i32;
        let mut kernel = Vec::with_capacity(size as usize);
        let two_sigma_sq = 2.0 * sigma * sigma;

        let mut sum = 0.0;
        for i in -radius..=radius {
            let value = (-((i * i) as f32) / two_sigma_sq).exp();
            kernel.push(value);
            sum += value;
        }

        // Normalize
        for value in &mut kernel {
            *value /= sum;
        }

        kernel
    }

    /// Get the horizontal kernel
    #[must_use]
    pub fn horizontal(&self) -> &[f32] {
        &self.horizontal_kernel
    }

    /// Get the vertical kernel
    #[must_use]
    pub fn vertical(&self) -> &[f32] {
        &self.vertical_kernel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convolution_kernel_creation() {
        let kernel = ConvolutionKernel::sobel_x();
        assert_eq!(kernel.dimensions(), (3, 3));
        assert_eq!(kernel.data().len(), 9);

        let kernel = ConvolutionKernel::laplacian();
        assert_eq!(kernel.dimensions(), (3, 3));
    }

    #[test]
    fn test_filter_kernel_creation() {
        let filter = FilterKernel::gaussian_blur(2.0);
        assert_eq!(filter.filter_type(), FilterType::GaussianBlur);
        assert_eq!(filter.sigma(), 2.0);

        let filter = FilterKernel::sobel();
        assert_eq!(filter.filter_type(), FilterType::Sobel);
        assert_eq!(filter.kernel_size(), 3);
    }

    #[test]
    fn test_separable_filter() {
        let filter = SeparableFilter::gaussian(1.0, 5);
        assert_eq!(filter.horizontal().len(), 5);
        assert_eq!(filter.vertical().len(), 5);

        // Check normalization
        let sum: f32 = filter.horizontal().iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_box_blur_kernel() {
        let kernel =
            ConvolutionKernel::box_blur(3).expect("box blur kernel creation should succeed");
        assert_eq!(kernel.dimensions(), (3, 3));
        let expected_value = 1.0 / 9.0;
        for &value in kernel.data() {
            assert!((value - expected_value).abs() < 0.001);
        }
    }
}
