//! Motion smoothing filters for video stabilization.
//!
//! This module provides various smoothing algorithms to reduce jitter in motion trajectories:
//!
//! - Gaussian smoothing filter
//! - Low-pass filter
//! - Adaptive smoothing based on motion magnitude

use crate::error::{CvError, CvResult};
use crate::stabilize::MotionParameters;
use std::f64::consts::PI;

/// Motion smoother trait.
///
/// Defines the interface for smoothing motion parameters.
pub trait MotionSmoother {
    /// Smooth motion parameters.
    ///
    /// # Arguments
    ///
    /// * `params` - Input motion parameters
    ///
    /// # Errors
    ///
    /// Returns an error if smoothing fails.
    fn smooth(&mut self, params: &MotionParameters) -> CvResult<MotionParameters>;
}

/// Gaussian smoothing filter.
///
/// Applies Gaussian convolution to motion trajectories.
///
/// # Examples
///
/// ```
/// use oximedia_cv::stabilize::smooth::GaussianSmoother;
///
/// let smoother = GaussianSmoother::new(15, 3.0);
/// ```
#[derive(Debug, Clone)]
pub struct GaussianSmoother {
    /// Window size (must be odd).
    window_size: usize,
    /// Gaussian sigma parameter.
    sigma: f64,
    /// Pre-computed Gaussian kernel.
    kernel: Vec<f64>,
}

impl GaussianSmoother {
    /// Create a new Gaussian smoother.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Size of the smoothing window (must be odd)
    /// * `sigma` - Standard deviation of the Gaussian
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::smooth::GaussianSmoother;
    ///
    /// let smoother = GaussianSmoother::new(15, 3.0);
    /// ```
    #[must_use]
    pub fn new(window_size: usize, sigma: f64) -> Self {
        let kernel = Self::compute_gaussian_kernel(window_size, sigma);
        Self {
            window_size,
            sigma,
            kernel,
        }
    }

    /// Set window size.
    #[must_use]
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self.kernel = Self::compute_gaussian_kernel(size, self.sigma);
        self
    }

    /// Set sigma parameter.
    #[must_use]
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self.kernel = Self::compute_gaussian_kernel(self.window_size, sigma);
        self
    }

    /// Compute Gaussian kernel.
    fn compute_gaussian_kernel(size: usize, sigma: f64) -> Vec<f64> {
        let half = (size / 2) as i32;
        let mut kernel = Vec::with_capacity(size);
        let mut sum = 0.0;

        for i in -half..=half {
            let x = i as f64;
            let value = (-x * x / (2.0 * sigma * sigma)).exp();
            kernel.push(value);
            sum += value;
        }

        // Normalize kernel
        for value in &mut kernel {
            *value /= sum;
        }

        kernel
    }

    /// Apply Gaussian smoothing to a signal.
    fn smooth_signal(&self, signal: &[f64]) -> Vec<f64> {
        let len = signal.len();
        let mut smoothed = vec![0.0; len];
        let half = (self.window_size / 2) as i32;

        for i in 0..len {
            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            for (k, &kernel_value) in self.kernel.iter().enumerate() {
                let offset = k as i32 - half;
                let idx = i as i32 + offset;

                if idx >= 0 && idx < len as i32 {
                    sum += signal[idx as usize] * kernel_value;
                    weight_sum += kernel_value;
                }
            }

            smoothed[i] = if weight_sum > 0.0 {
                sum / weight_sum
            } else {
                signal[i]
            };
        }

        smoothed
    }
}

impl MotionSmoother for GaussianSmoother {
    fn smooth(&mut self, params: &MotionParameters) -> CvResult<MotionParameters> {
        Ok(MotionParameters {
            dx: self.smooth_signal(&params.dx),
            dy: self.smooth_signal(&params.dy),
            da: self.smooth_signal(&params.da),
            ds: self.smooth_signal(&params.ds),
        })
    }
}

/// Low-pass filter for motion smoothing.
///
/// Implements a simple IIR low-pass filter.
///
/// # Examples
///
/// ```
/// use oximedia_cv::stabilize::smooth::LowPassFilter;
///
/// let filter = LowPassFilter::new(0.3);
/// ```
#[derive(Debug, Clone)]
pub struct LowPassFilter {
    /// Filter cutoff frequency (0.0-1.0).
    cutoff: f64,
    /// Filter coefficient.
    alpha: f64,
}

impl LowPassFilter {
    /// Create a new low-pass filter.
    ///
    /// # Arguments
    ///
    /// * `cutoff` - Cutoff frequency (0.0-1.0)
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::smooth::LowPassFilter;
    ///
    /// let filter = LowPassFilter::new(0.3);
    /// ```
    #[must_use]
    pub fn new(cutoff: f64) -> Self {
        let alpha = Self::compute_alpha(cutoff);
        Self { cutoff, alpha }
    }

    /// Set cutoff frequency.
    #[must_use]
    pub fn with_cutoff(mut self, cutoff: f64) -> Self {
        self.cutoff = cutoff.clamp(0.0, 1.0);
        self.alpha = Self::compute_alpha(self.cutoff);
        self
    }

    /// Compute filter coefficient from cutoff frequency.
    fn compute_alpha(cutoff: f64) -> f64 {
        let rc = 1.0 / (2.0 * PI * cutoff);
        let dt = 1.0;
        dt / (rc + dt)
    }

    /// Apply low-pass filter to a signal.
    fn filter_signal(&self, signal: &[f64]) -> Vec<f64> {
        if signal.is_empty() {
            return Vec::new();
        }

        let mut filtered = Vec::with_capacity(signal.len());
        filtered.push(signal[0]);

        for i in 1..signal.len() {
            let value = self.alpha * signal[i] + (1.0 - self.alpha) * filtered[i - 1];
            filtered.push(value);
        }

        filtered
    }
}

impl MotionSmoother for LowPassFilter {
    fn smooth(&mut self, params: &MotionParameters) -> CvResult<MotionParameters> {
        Ok(MotionParameters {
            dx: self.filter_signal(&params.dx),
            dy: self.filter_signal(&params.dy),
            da: self.filter_signal(&params.da),
            ds: self.filter_signal(&params.ds),
        })
    }
}

/// Adaptive smoother based on motion magnitude.
///
/// Applies stronger smoothing to small motions and weaker smoothing to large motions.
///
/// # Examples
///
/// ```
/// use oximedia_cv::stabilize::smooth::AdaptiveSmoother;
///
/// let smoother = AdaptiveSmoother::new(15, 50.0);
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveSmoother {
    /// Base window size.
    base_window: usize,
    /// Motion magnitude threshold.
    magnitude_threshold: f64,
    /// Gaussian smoother for small motions.
    gaussian_smoother: GaussianSmoother,
}

impl AdaptiveSmoother {
    /// Create a new adaptive smoother.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Base window size
    /// * `magnitude_threshold` - Motion magnitude threshold
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::smooth::AdaptiveSmoother;
    ///
    /// let smoother = AdaptiveSmoother::new(15, 50.0);
    /// ```
    #[must_use]
    pub fn new(window_size: usize, magnitude_threshold: f64) -> Self {
        Self {
            base_window: window_size,
            magnitude_threshold,
            gaussian_smoother: GaussianSmoother::new(window_size, 3.0),
        }
    }

    /// Set base window size.
    #[must_use]
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.base_window = size;
        self.gaussian_smoother = self.gaussian_smoother.with_window_size(size);
        self
    }

    /// Set magnitude threshold.
    #[must_use]
    pub fn with_magnitude_threshold(mut self, threshold: f64) -> Self {
        self.magnitude_threshold = threshold;
        self
    }

    /// Compute motion magnitude at each frame.
    fn compute_magnitudes(&self, params: &MotionParameters) -> Vec<f64> {
        let len = params.dx.len();
        let mut magnitudes = Vec::with_capacity(len);

        for i in 0..len {
            let mag = (params.dx[i] * params.dx[i] + params.dy[i] * params.dy[i]).sqrt();
            magnitudes.push(mag);
        }

        magnitudes
    }

    /// Compute adaptive weights based on motion magnitude.
    fn compute_adaptive_weights(&self, magnitudes: &[f64]) -> Vec<f64> {
        magnitudes
            .iter()
            .map(|&mag| {
                // Use sigmoid function to compute weight
                let normalized = mag / self.magnitude_threshold;
                1.0 / (1.0 + (-5.0 * (normalized - 1.0)).exp())
            })
            .collect()
    }

    /// Apply adaptive smoothing to a signal.
    fn smooth_signal_adaptive(
        &self,
        signal: &[f64],
        weights: &[f64],
        smoothed: &[f64],
    ) -> Vec<f64> {
        signal
            .iter()
            .zip(smoothed.iter())
            .zip(weights.iter())
            .map(|((&original, &smooth), &weight)| {
                // Blend between original and smoothed based on weight
                original * weight + smooth * (1.0 - weight)
            })
            .collect()
    }

    /// Smooth motion parameters adaptively.
    ///
    /// # Arguments
    ///
    /// * `params` - Input motion parameters
    /// * `max_magnitude` - Maximum motion magnitude for normalization
    ///
    /// # Errors
    ///
    /// Returns an error if smoothing fails.
    pub fn smooth(
        &mut self,
        params: &MotionParameters,
        max_magnitude: f64,
    ) -> CvResult<MotionParameters> {
        // Update magnitude threshold
        self.magnitude_threshold = max_magnitude;

        // Compute motion magnitudes
        let magnitudes = self.compute_magnitudes(params);

        // Compute adaptive weights
        let weights = self.compute_adaptive_weights(&magnitudes);

        // Apply Gaussian smoothing
        let gaussian_smoothed = self.gaussian_smoother.smooth(params)?;

        // Apply adaptive blending
        Ok(MotionParameters {
            dx: self.smooth_signal_adaptive(&params.dx, &weights, &gaussian_smoothed.dx),
            dy: self.smooth_signal_adaptive(&params.dy, &weights, &gaussian_smoothed.dy),
            da: self.smooth_signal_adaptive(&params.da, &weights, &gaussian_smoothed.da),
            ds: self.smooth_signal_adaptive(&params.ds, &weights, &gaussian_smoothed.ds),
        })
    }
}

/// Moving average smoother.
///
/// Applies a simple moving average filter to motion trajectories.
///
/// # Examples
///
/// ```
/// use oximedia_cv::stabilize::smooth::MovingAverageSmoother;
///
/// let smoother = MovingAverageSmoother::new(15);
/// ```
#[derive(Debug, Clone)]
pub struct MovingAverageSmoother {
    /// Window size.
    window_size: usize,
}

impl MovingAverageSmoother {
    /// Create a new moving average smoother.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Size of the moving average window
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::smooth::MovingAverageSmoother;
    ///
    /// let smoother = MovingAverageSmoother::new(15);
    /// ```
    #[must_use]
    pub const fn new(window_size: usize) -> Self {
        Self { window_size }
    }

    /// Set window size.
    #[must_use]
    pub const fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Apply moving average to a signal.
    fn smooth_signal(&self, signal: &[f64]) -> Vec<f64> {
        let len = signal.len();
        let mut smoothed = Vec::with_capacity(len);
        let half = (self.window_size / 2) as i32;

        for i in 0..len {
            let mut sum = 0.0;
            let mut count = 0;

            let start = (i as i32 - half).max(0) as usize;
            let end = (i as i32 + half + 1).min(len as i32) as usize;

            for j in start..end {
                sum += signal[j];
                count += 1;
            }

            smoothed.push(sum / count as f64);
        }

        smoothed
    }
}

impl MotionSmoother for MovingAverageSmoother {
    fn smooth(&mut self, params: &MotionParameters) -> CvResult<MotionParameters> {
        Ok(MotionParameters {
            dx: self.smooth_signal(&params.dx),
            dy: self.smooth_signal(&params.dy),
            da: self.smooth_signal(&params.da),
            ds: self.smooth_signal(&params.ds),
        })
    }
}

/// Median filter smoother.
///
/// Applies median filtering to remove outliers from motion trajectories.
///
/// # Examples
///
/// ```
/// use oximedia_cv::stabilize::smooth::MedianFilterSmoother;
///
/// let smoother = MedianFilterSmoother::new(5);
/// ```
#[derive(Debug, Clone)]
pub struct MedianFilterSmoother {
    /// Window size (must be odd).
    window_size: usize,
}

impl MedianFilterSmoother {
    /// Create a new median filter smoother.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Size of the median filter window (must be odd)
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::smooth::MedianFilterSmoother;
    ///
    /// let smoother = MedianFilterSmoother::new(5);
    /// ```
    #[must_use]
    pub const fn new(window_size: usize) -> Self {
        Self { window_size }
    }

    /// Set window size.
    #[must_use]
    pub const fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Apply median filter to a signal.
    fn smooth_signal(&self, signal: &[f64]) -> Vec<f64> {
        let len = signal.len();
        let mut smoothed = Vec::with_capacity(len);
        let half = (self.window_size / 2) as i32;

        for i in 0..len {
            let mut window = Vec::new();

            let start = (i as i32 - half).max(0) as usize;
            let end = (i as i32 + half + 1).min(len as i32) as usize;

            for j in start..end {
                window.push(signal[j]);
            }

            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = window[window.len() / 2];
            smoothed.push(median);
        }

        smoothed
    }
}

impl MotionSmoother for MedianFilterSmoother {
    fn smooth(&mut self, params: &MotionParameters) -> CvResult<MotionParameters> {
        Ok(MotionParameters {
            dx: self.smooth_signal(&params.dx),
            dy: self.smooth_signal(&params.dy),
            da: self.smooth_signal(&params.da),
            ds: self.smooth_signal(&params.ds),
        })
    }
}

/// Exponential moving average smoother.
///
/// Applies exponential weighting to recent motion values.
///
/// # Examples
///
/// ```
/// use oximedia_cv::stabilize::smooth::ExponentialSmoother;
///
/// let smoother = ExponentialSmoother::new(0.8);
/// ```
#[derive(Debug, Clone)]
pub struct ExponentialSmoother {
    /// Smoothing factor (0.0-1.0).
    alpha: f64,
}

impl ExponentialSmoother {
    /// Create a new exponential smoother.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Smoothing factor (0.0 = no smoothing, 1.0 = no memory)
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::smooth::ExponentialSmoother;
    ///
    /// let smoother = ExponentialSmoother::new(0.8);
    /// ```
    #[must_use]
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Set smoothing factor.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Apply exponential smoothing to a signal.
    fn smooth_signal(&self, signal: &[f64]) -> Vec<f64> {
        if signal.is_empty() {
            return Vec::new();
        }

        let mut smoothed = Vec::with_capacity(signal.len());
        smoothed.push(signal[0]);

        for i in 1..signal.len() {
            let value = self.alpha * signal[i] + (1.0 - self.alpha) * smoothed[i - 1];
            smoothed.push(value);
        }

        smoothed
    }
}

impl MotionSmoother for ExponentialSmoother {
    fn smooth(&mut self, params: &MotionParameters) -> CvResult<MotionParameters> {
        Ok(MotionParameters {
            dx: self.smooth_signal(&params.dx),
            dy: self.smooth_signal(&params.dy),
            da: self.smooth_signal(&params.da),
            ds: self.smooth_signal(&params.ds),
        })
    }
}

/// Savitzky-Golay filter smoother.
///
/// Applies polynomial smoothing to preserve features while reducing noise.
///
/// # Examples
///
/// ```
/// use oximedia_cv::stabilize::smooth::SavitzkyGolaySmoother;
///
/// let smoother = SavitzkyGolaySmoother::new(5, 2);
/// ```
#[derive(Debug, Clone)]
pub struct SavitzkyGolaySmoother {
    /// Window size (must be odd).
    window_size: usize,
    /// Polynomial order.
    polynomial_order: usize,
}

impl SavitzkyGolaySmoother {
    /// Create a new Savitzky-Golay smoother.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Size of the smoothing window (must be odd)
    /// * `polynomial_order` - Order of the polynomial (must be < window_size)
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::stabilize::smooth::SavitzkyGolaySmoother;
    ///
    /// let smoother = SavitzkyGolaySmoother::new(5, 2);
    /// ```
    #[must_use]
    pub const fn new(window_size: usize, polynomial_order: usize) -> Self {
        Self {
            window_size,
            polynomial_order,
        }
    }

    /// Apply Savitzky-Golay smoothing to a signal (simplified implementation).
    fn smooth_signal(&self, signal: &[f64]) -> Vec<f64> {
        // For simplicity, use a moving average as an approximation
        // A full implementation would compute the convolution coefficients
        let len = signal.len();
        let mut smoothed = Vec::with_capacity(len);
        let half = (self.window_size / 2) as i32;

        for i in 0..len {
            let mut sum = 0.0;
            let mut count = 0;

            let start = (i as i32 - half).max(0) as usize;
            let end = (i as i32 + half + 1).min(len as i32) as usize;

            for j in start..end {
                sum += signal[j];
                count += 1;
            }

            smoothed.push(sum / count as f64);
        }

        smoothed
    }
}

impl MotionSmoother for SavitzkyGolaySmoother {
    fn smooth(&mut self, params: &MotionParameters) -> CvResult<MotionParameters> {
        Ok(MotionParameters {
            dx: self.smooth_signal(&params.dx),
            dy: self.smooth_signal(&params.dy),
            da: self.smooth_signal(&params.da),
            ds: self.smooth_signal(&params.ds),
        })
    }
}
