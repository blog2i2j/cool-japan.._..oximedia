//! Smoothing filters for motion trajectories.
//!
//! Provides various filtering algorithms including Gaussian, moving average,
//! Kalman filter, and low-pass filtering.

use crate::error::{StabilizeError, StabilizeResult};
use crate::motion::trajectory::Trajectory;
use scirs2_core::ndarray::Array1;

/// Main trajectory smoother that combines multiple filtering techniques.
#[derive(Debug)]
pub struct TrajectorySmoother {
    window_size: usize,
    strength: f64,
    gaussian_filter: GaussianFilter,
    low_pass_filter: LowPassFilter,
    kalman_filter: KalmanFilter,
    use_kalman: bool,
}

impl TrajectorySmoother {
    /// Create a new trajectory smoother.
    #[must_use]
    pub fn new(window_size: usize, strength: f64) -> Self {
        Self {
            window_size,
            strength: strength.clamp(0.0, 1.0),
            gaussian_filter: GaussianFilter::new(window_size),
            low_pass_filter: LowPassFilter::new(0.3),
            kalman_filter: KalmanFilter::new(),
            use_kalman: true,
        }
    }

    /// Set smoothing strength.
    pub fn set_strength(&mut self, strength: f64) {
        self.strength = strength.clamp(0.0, 1.0);
    }

    /// Enable or disable Kalman filtering.
    pub fn set_use_kalman(&mut self, use_kalman: bool) {
        self.use_kalman = use_kalman;
    }

    /// Smooth a trajectory.
    ///
    /// # Errors
    ///
    /// Returns an error if the trajectory is empty or smoothing fails.
    pub fn smooth(&mut self, trajectory: &Trajectory) -> StabilizeResult<Trajectory> {
        if trajectory.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        // Choose smoothing method based on strength
        let smoothed = if self.strength < 0.3 {
            // Low strength: use simple moving average
            self.moving_average_smooth(trajectory)?
        } else if self.strength < 0.7 {
            // Medium strength: use Gaussian smoothing
            self.gaussian_smooth(trajectory)?
        } else if self.use_kalman {
            // High strength: use Kalman filter
            self.kalman_smooth(trajectory)?
        } else {
            // High strength without Kalman: use low-pass filter
            self.low_pass_smooth(trajectory)?
        };

        Ok(smoothed)
    }

    /// Apply moving average smoothing.
    fn moving_average_smooth(&self, trajectory: &Trajectory) -> StabilizeResult<Trajectory> {
        let window = (self.window_size as f64 * self.strength) as usize;
        let window = window.max(1);

        Ok(Trajectory {
            x: self.moving_average(&trajectory.x, window),
            y: self.moving_average(&trajectory.y, window),
            angle: self.moving_average(&trajectory.angle, window),
            scale: self.moving_average_scale(&trajectory.scale, window),
            frame_count: trajectory.frame_count,
        })
    }

    /// Apply Gaussian smoothing.
    fn gaussian_smooth(&self, trajectory: &Trajectory) -> StabilizeResult<Trajectory> {
        let sigma = self.window_size as f64 * self.strength / 3.0;
        self.gaussian_filter.smooth(trajectory, sigma)
    }

    /// Apply Kalman filter smoothing.
    fn kalman_smooth(&mut self, trajectory: &Trajectory) -> StabilizeResult<Trajectory> {
        self.kalman_filter
            .set_process_noise(0.1 * (1.0 - self.strength));
        self.kalman_filter
            .set_measurement_noise(0.1 * self.strength);
        self.kalman_filter.smooth(trajectory)
    }

    /// Apply low-pass filter smoothing.
    fn low_pass_smooth(&self, trajectory: &Trajectory) -> StabilizeResult<Trajectory> {
        let cutoff = 0.5 * (1.0 - self.strength);
        self.low_pass_filter.smooth(trajectory, cutoff)
    }

    /// Simple moving average.
    fn moving_average(&self, data: &Array1<f64>, window: usize) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);
        let half = window / 2;

        for i in 0..n {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(n);
            let count = end - start;

            let sum: f64 = data.slice(scirs2_core::ndarray::s![start..end]).sum();
            result[i] = sum / count as f64;
        }

        result
    }

    /// Moving average for scale (multiplicative).
    fn moving_average_scale(&self, data: &Array1<f64>, window: usize) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);
        let half = window / 2;

        for i in 0..n {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(n);
            let count = end - start;

            // Geometric mean for scale factors
            let product: f64 = data
                .slice(scirs2_core::ndarray::s![start..end])
                .iter()
                .product();
            result[i] = product.powf(1.0 / count as f64);
        }

        result
    }
}

/// Gaussian smoothing filter.
#[derive(Debug)]
pub struct GaussianFilter {
    window_size: usize,
}

impl GaussianFilter {
    /// Create a new Gaussian filter.
    #[must_use]
    pub const fn new(window_size: usize) -> Self {
        Self { window_size }
    }

    /// Smooth a trajectory using Gaussian kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if the trajectory is empty.
    pub fn smooth(&self, trajectory: &Trajectory, sigma: f64) -> StabilizeResult<Trajectory> {
        if trajectory.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        let kernel = self.create_gaussian_kernel(sigma);

        Ok(Trajectory {
            x: self.convolve(&trajectory.x, &kernel),
            y: self.convolve(&trajectory.y, &kernel),
            angle: self.convolve(&trajectory.angle, &kernel),
            scale: self.convolve_scale(&trajectory.scale, &kernel),
            frame_count: trajectory.frame_count,
        })
    }

    /// Create Gaussian kernel.
    fn create_gaussian_kernel(&self, sigma: f64) -> Array1<f64> {
        let half = self.window_size / 2;
        let mut kernel = Array1::zeros(self.window_size);

        let mut sum = 0.0;
        for i in 0..self.window_size {
            let x = (i as f64 - half as f64) / sigma;
            let value = (-0.5 * x * x).exp();
            kernel[i] = value;
            sum += value;
        }

        // Normalize
        kernel.mapv_inplace(|v| v / sum);
        kernel
    }

    /// Convolve signal with kernel.
    fn convolve(&self, data: &Array1<f64>, kernel: &Array1<f64>) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);
        let half = kernel.len() / 2;

        for i in 0..n {
            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            for (j, &k_val) in kernel.iter().enumerate() {
                let idx = i as i32 + j as i32 - half as i32;
                if idx >= 0 && idx < n as i32 {
                    sum += data[idx as usize] * k_val;
                    weight_sum += k_val;
                }
            }

            result[i] = if weight_sum > 0.0 {
                sum / weight_sum
            } else {
                data[i]
            };
        }

        result
    }

    /// Convolve scale values (geometric mean).
    fn convolve_scale(&self, data: &Array1<f64>, kernel: &Array1<f64>) -> Array1<f64> {
        // For scale, use log-space convolution
        let log_data = data.mapv(|v| v.ln());
        let smoothed = self.convolve(&log_data, kernel);
        smoothed.mapv(|v| v.exp())
    }
}

/// Low-pass filter for trajectory smoothing.
#[derive(Debug)]
pub struct LowPassFilter {
    alpha: f64,
}

impl LowPassFilter {
    /// Create a new low-pass filter.
    #[must_use]
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Set filter cutoff frequency.
    pub fn set_cutoff(&mut self, alpha: f64) {
        self.alpha = alpha.clamp(0.0, 1.0);
    }

    /// Smooth a trajectory using low-pass filter.
    ///
    /// # Errors
    ///
    /// Returns an error if the trajectory is empty.
    pub fn smooth(&self, trajectory: &Trajectory, cutoff: f64) -> StabilizeResult<Trajectory> {
        if trajectory.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        let alpha = cutoff.clamp(0.0, 1.0);

        Ok(Trajectory {
            x: self.filter_signal(&trajectory.x, alpha),
            y: self.filter_signal(&trajectory.y, alpha),
            angle: self.filter_signal(&trajectory.angle, alpha),
            scale: self.filter_signal(&trajectory.scale, alpha),
            frame_count: trajectory.frame_count,
        })
    }

    /// Apply exponential moving average filter.
    fn filter_signal(&self, data: &Array1<f64>, alpha: f64) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);

        if n == 0 {
            return result;
        }

        result[0] = data[0];

        for i in 1..n {
            result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }
}

/// Kalman filter for trajectory smoothing.
#[derive(Debug)]
pub struct KalmanFilter {
    process_noise: f64,
    measurement_noise: f64,
}

impl KalmanFilter {
    /// Create a new Kalman filter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            process_noise: 0.01,
            measurement_noise: 0.1,
        }
    }

    /// Set process noise.
    pub fn set_process_noise(&mut self, noise: f64) {
        self.process_noise = noise.max(0.0);
    }

    /// Set measurement noise.
    pub fn set_measurement_noise(&mut self, noise: f64) {
        self.measurement_noise = noise.max(0.0);
    }

    /// Smooth a trajectory using Kalman filter.
    ///
    /// # Errors
    ///
    /// Returns an error if the trajectory is empty.
    pub fn smooth(&self, trajectory: &Trajectory) -> StabilizeResult<Trajectory> {
        if trajectory.is_empty() {
            return Err(StabilizeError::EmptyFrameSequence);
        }

        Ok(Trajectory {
            x: self.filter_signal(&trajectory.x),
            y: self.filter_signal(&trajectory.y),
            angle: self.filter_signal(&trajectory.angle),
            scale: self.filter_signal(&trajectory.scale),
            frame_count: trajectory.frame_count,
        })
    }

    /// Apply 1D Kalman filter.
    fn filter_signal(&self, data: &Array1<f64>) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);

        if n == 0 {
            return result;
        }

        // Initialize state
        let mut x = data[0]; // State estimate
        let mut p = 1.0; // Error covariance

        let q = self.process_noise; // Process noise
        let r = self.measurement_noise; // Measurement noise

        for i in 0..n {
            // Predict
            let x_pred = x;
            let p_pred = p + q;

            // Update
            let k = p_pred / (p_pred + r); // Kalman gain
            x = x_pred + k * (data[i] - x_pred);
            p = (1.0 - k) * p_pred;

            result[i] = x;
        }

        result
    }
}

impl Default for KalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoother_creation() {
        let smoother = TrajectorySmoother::new(30, 0.5);
        assert_eq!(smoother.window_size, 30);
        assert!((smoother.strength - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gaussian_filter() {
        let filter = GaussianFilter::new(11);
        let kernel = filter.create_gaussian_kernel(2.0);
        assert_eq!(kernel.len(), 11);

        // Check normalization
        let sum: f64 = kernel.sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_low_pass_filter() {
        let filter = LowPassFilter::new(0.3);
        let data = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let filtered = filter.filter_signal(&data, 0.5);
        assert_eq!(filtered.len(), 5);
    }

    #[test]
    fn test_kalman_filter() {
        let filter = KalmanFilter::new();
        let data = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let filtered = filter.filter_signal(&data);
        assert_eq!(filtered.len(), 5);
    }

    #[test]
    fn test_moving_average() {
        let smoother = TrajectorySmoother::new(5, 0.5);
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let averaged = smoother.moving_average(&data, 3);
        assert_eq!(averaged.len(), 5);
        assert!((averaged[2] - 3.0).abs() < f64::EPSILON);
    }
}

/// Butterworth low-pass filter implementation.
pub struct ButterworthFilter {
    order: usize,
    cutoff: f64,
}

impl ButterworthFilter {
    /// Create a new Butterworth filter.
    #[must_use]
    pub fn new(order: usize, cutoff: f64) -> Self {
        Self {
            order,
            cutoff: cutoff.clamp(0.0, 0.5),
        }
    }

    /// Apply Butterworth filter to signal.
    #[must_use]
    pub fn filter(&self, data: &Array1<f64>) -> Array1<f64> {
        // Simplified implementation - forward-backward filter
        let forward = self.filter_forward(data);
        let backward = self.filter_backward(&forward);
        backward
    }

    /// Forward pass of filter.
    fn filter_forward(&self, data: &Array1<f64>) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);

        if n == 0 {
            return result;
        }

        let alpha = 2.0 * std::f64::consts::PI * self.cutoff;
        let a = alpha / (alpha + 1.0);

        result[0] = data[0];

        for i in 1..n {
            result[i] = a * data[i] + (1.0 - a) * result[i - 1];
        }

        result
    }

    /// Backward pass of filter.
    fn filter_backward(&self, data: &Array1<f64>) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);

        if n == 0 {
            return result;
        }

        let alpha = 2.0 * std::f64::consts::PI * self.cutoff;
        let a = alpha / (alpha + 1.0);

        result[n - 1] = data[n - 1];

        for i in (0..n - 1).rev() {
            result[i] = a * data[i] + (1.0 - a) * result[i + 1];
        }

        result
    }
}

/// Savitzky-Golay smoothing filter.
pub struct SavitzkyGolayFilter {
    window_size: usize,
    poly_order: usize,
}

impl SavitzkyGolayFilter {
    /// Create a new Savitzky-Golay filter.
    #[must_use]
    pub fn new(window_size: usize, poly_order: usize) -> Self {
        Self {
            window_size,
            poly_order,
        }
    }

    /// Apply Savitzky-Golay smoothing.
    #[must_use]
    pub fn smooth(&self, data: &Array1<f64>) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);
        let half = self.window_size / 2;

        for i in 0..n {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(n);

            // Simple moving average as approximation
            let sum: f64 = data.slice(scirs2_core::ndarray::s![start..end]).sum();
            let count = end - start;
            result[i] = sum / count as f64;
        }

        result
    }
}

/// Median filter for outlier removal.
pub struct MedianFilter {
    window_size: usize,
}

impl MedianFilter {
    /// Create a new median filter.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }

    /// Apply median filtering.
    #[must_use]
    pub fn filter(&self, data: &Array1<f64>) -> Array1<f64> {
        let n = data.len();
        let mut result = Array1::zeros(n);
        let half = self.window_size / 2;

        for i in 0..n {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(n);

            let mut window: Vec<f64> = data.slice(scirs2_core::ndarray::s![start..end]).to_vec();
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            result[i] = if window.len() % 2 == 0 {
                (window[window.len() / 2 - 1] + window[window.len() / 2]) / 2.0
            } else {
                window[window.len() / 2]
            };
        }

        result
    }
}

#[cfg(test)]
mod filter_tests {
    use super::*;

    #[test]
    fn test_butterworth_filter() {
        let filter = ButterworthFilter::new(2, 0.1);
        let data = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0]);
        let filtered = filter.filter(&data);
        assert_eq!(filtered.len(), 5);
    }

    #[test]
    fn test_savitzky_golay() {
        let filter = SavitzkyGolayFilter::new(5, 2);
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let smoothed = filter.smooth(&data);
        assert_eq!(smoothed.len(), 5);
    }

    #[test]
    fn test_median_filter() {
        let filter = MedianFilter::new(3);
        let data = Array1::from_vec(vec![1.0, 10.0, 2.0, 3.0, 4.0]);
        let filtered = filter.filter(&data);
        // Median should remove the outlier (10.0)
        assert!((filtered[1] - 2.0).abs() < 1.0);
    }
}
