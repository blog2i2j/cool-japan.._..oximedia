//! Kalman filter for motion prediction and state estimation.
//!
//! This module provides a Kalman filter implementation for tracking
//! with motion prediction, supporting various motion models.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::tracking::KalmanFilter;
//!
//! let mut kf = KalmanFilter::new(4, 2); // 4D state, 2D measurement
//! ```

use crate::error::{CvError, CvResult};

/// Kalman filter for state estimation and prediction.
///
/// Implements the discrete Kalman filter for linear dynamic systems.
///
/// # Examples
///
/// ```
/// use oximedia_cv::tracking::KalmanFilter;
///
/// // Track 2D position with velocity (4D state: x, y, vx, vy)
/// let mut kf = KalmanFilter::new(4, 2);
/// ```
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    /// State dimension.
    state_dim: usize,
    /// Measurement dimension.
    measure_dim: usize,
    /// State vector.
    state: Vec<f64>,
    /// State covariance matrix.
    pub covariance: Vec<f64>,
    /// State transition matrix (F).
    pub transition: Vec<f64>,
    /// Measurement matrix (H).
    pub measurement: Vec<f64>,
    /// Process noise covariance (Q).
    pub process_noise: Vec<f64>,
    /// Measurement noise covariance (R).
    pub measurement_noise: Vec<f64>,
}

impl KalmanFilter {
    /// Create a new Kalman filter.
    ///
    /// # Arguments
    ///
    /// * `state_dim` - State vector dimension
    /// * `measure_dim` - Measurement vector dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::KalmanFilter;
    ///
    /// let kf = KalmanFilter::new(4, 2);
    /// ```
    #[must_use]
    pub fn new(state_dim: usize, measure_dim: usize) -> Self {
        let mut kf = Self {
            state_dim,
            measure_dim,
            state: vec![0.0; state_dim],
            covariance: vec![0.0; state_dim * state_dim],
            transition: vec![0.0; state_dim * state_dim],
            measurement: vec![0.0; measure_dim * state_dim],
            process_noise: vec![0.0; state_dim * state_dim],
            measurement_noise: vec![0.0; measure_dim * measure_dim],
        };

        // Initialize with identity matrices
        for i in 0..state_dim {
            kf.transition[i * state_dim + i] = 1.0;
            kf.covariance[i * state_dim + i] = 1.0;
            kf.process_noise[i * state_dim + i] = 0.01;
        }

        for i in 0..measure_dim {
            kf.measurement[i * state_dim + i] = 1.0;
            kf.measurement_noise[i * measure_dim + i] = 0.1;
        }

        kf
    }

    /// Create a constant velocity motion model.
    ///
    /// State: [x, y, vx, vy]
    /// Measurement: [x, y]
    ///
    /// # Arguments
    ///
    /// * `dt` - Time step
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::KalmanFilter;
    ///
    /// let kf = KalmanFilter::constant_velocity(1.0);
    /// ```
    #[must_use]
    pub fn constant_velocity(dt: f64) -> Self {
        let mut kf = Self::new(4, 2);

        // State transition: x_new = x + vx*dt, vx_new = vx
        kf.transition = vec![
            1.0, 0.0, dt, 0.0, 0.0, 1.0, 0.0, dt, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];

        // Measurement matrix: measure x and y only
        kf.measurement = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

        // Process noise
        let q = 0.1;
        kf.process_noise = vec![
            q, 0.0, 0.0, 0.0, 0.0, q, 0.0, 0.0, 0.0, 0.0, q, 0.0, 0.0, 0.0, 0.0, q,
        ];

        // Measurement noise
        let r = 1.0;
        kf.measurement_noise = vec![r, 0.0, 0.0, r];

        // Initial covariance
        kf.covariance = vec![
            10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0,
        ];

        kf
    }

    /// Create a constant acceleration motion model.
    ///
    /// State: [x, y, vx, vy, ax, ay]
    /// Measurement: [x, y]
    ///
    /// # Arguments
    ///
    /// * `dt` - Time step
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::KalmanFilter;
    ///
    /// let kf = KalmanFilter::constant_acceleration(1.0);
    /// ```
    #[must_use]
    pub fn constant_acceleration(dt: f64) -> Self {
        let mut kf = Self::new(6, 2);

        // State transition with acceleration
        let dt2 = dt * dt / 2.0;
        kf.transition = vec![
            1.0, 0.0, dt, 0.0, dt2, 0.0, 0.0, 1.0, 0.0, dt, 0.0, dt2, 0.0, 0.0, 1.0, 0.0, dt, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, dt, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0,
        ];

        // Measurement matrix
        kf.measurement = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        // Process noise
        let q = 0.1;
        for i in 0..6 {
            kf.process_noise[i * 6 + i] = q;
        }

        // Measurement noise
        kf.measurement_noise = vec![1.0, 0.0, 0.0, 1.0];

        // Initial covariance
        for i in 0..6 {
            kf.covariance[i * 6 + i] = 10.0;
        }

        kf
    }

    /// Set initial state.
    ///
    /// # Errors
    ///
    /// Returns an error if state dimension doesn't match.
    pub fn set_state(&mut self, state: Vec<f64>) -> CvResult<()> {
        if state.len() != self.state_dim {
            return Err(CvError::invalid_parameter(
                "state dimension",
                format!("expected {}, got {}", self.state_dim, state.len()),
            ));
        }
        self.state = state;
        Ok(())
    }

    /// Set process noise covariance.
    pub fn set_process_noise(&mut self, q: f64) {
        for i in 0..self.state_dim {
            self.process_noise[i * self.state_dim + i] = q;
        }
    }

    /// Set measurement noise covariance.
    pub fn set_measurement_noise(&mut self, r: f64) {
        for i in 0..self.measure_dim {
            self.measurement_noise[i * self.measure_dim + i] = r;
        }
    }

    /// Predict next state.
    ///
    /// Performs the prediction step: x = F*x, P = F*P*F' + Q
    ///
    /// # Returns
    ///
    /// Predicted state vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::KalmanFilter;
    ///
    /// let mut kf = KalmanFilter::constant_velocity(1.0);
    /// kf.set_state(vec![0.0, 0.0, 1.0, 1.0])?;
    ///
    /// let predicted = kf.predict();
    /// // After 1 second at velocity (1, 1), position should be (1, 1)
    /// assert!((predicted[0] - 1.0).abs() < 0.001);
    /// ```
    pub fn predict(&mut self) -> Vec<f64> {
        // x = F * x
        let new_state = matrix_vector_multiply(
            &self.transition,
            &self.state,
            self.state_dim,
            self.state_dim,
        );
        self.state = new_state;

        // P = F * P * F' + Q
        let fp = matrix_multiply(
            &self.transition,
            &self.covariance,
            self.state_dim,
            self.state_dim,
            self.state_dim,
        );

        let fpft = matrix_multiply_transpose(
            &fp,
            &self.transition,
            self.state_dim,
            self.state_dim,
            self.state_dim,
        );

        self.covariance = matrix_add(&fpft, &self.process_noise, self.state_dim, self.state_dim);

        self.state.clone()
    }

    /// Update state with measurement.
    ///
    /// Performs the update step with Kalman gain computation.
    ///
    /// # Arguments
    ///
    /// * `measurement` - Measurement vector
    ///
    /// # Returns
    ///
    /// Updated state vector.
    ///
    /// # Errors
    ///
    /// Returns an error if measurement dimension doesn't match.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::tracking::KalmanFilter;
    ///
    /// let mut kf = KalmanFilter::constant_velocity(1.0);
    /// kf.set_state(vec![0.0, 0.0, 1.0, 1.0])?;
    ///
    /// let measurement = vec![2.0, 3.0];
    /// let updated = kf.update(&measurement)?;
    /// ```
    pub fn update(&mut self, measurement: &[f64]) -> CvResult<Vec<f64>> {
        if measurement.len() != self.measure_dim {
            return Err(CvError::invalid_parameter(
                "measurement dimension",
                format!("expected {}, got {}", self.measure_dim, measurement.len()),
            ));
        }

        // Innovation: y = z - H*x
        let hx = matrix_vector_multiply(
            &self.measurement,
            &self.state,
            self.measure_dim,
            self.state_dim,
        );

        let mut innovation = vec![0.0; self.measure_dim];
        for i in 0..self.measure_dim {
            innovation[i] = measurement[i] - hx[i];
        }

        // Innovation covariance: S = H*P*H' + R
        let hp = matrix_multiply(
            &self.measurement,
            &self.covariance,
            self.measure_dim,
            self.state_dim,
            self.state_dim,
        );

        let hpht = matrix_multiply_transpose(
            &hp,
            &self.measurement,
            self.measure_dim,
            self.state_dim,
            self.measure_dim,
        );

        let s = matrix_add(
            &hpht,
            &self.measurement_noise,
            self.measure_dim,
            self.measure_dim,
        );

        // Kalman gain: K = P*H' * inv(S)
        let s_inv = matrix_inverse(&s, self.measure_dim)?;

        let pht = matrix_multiply_transpose(
            &self.covariance,
            &self.measurement,
            self.state_dim,
            self.state_dim,
            self.measure_dim,
        );

        let kalman_gain = matrix_multiply(
            &pht,
            &s_inv,
            self.state_dim,
            self.measure_dim,
            self.measure_dim,
        );

        // Update state: x = x + K*y
        let ky =
            matrix_vector_multiply(&kalman_gain, &innovation, self.state_dim, self.measure_dim);

        for i in 0..self.state_dim {
            self.state[i] += ky[i];
        }

        // Update covariance: P = (I - K*H) * P
        let kh = matrix_multiply(
            &kalman_gain,
            &self.measurement,
            self.state_dim,
            self.measure_dim,
            self.state_dim,
        );

        let mut i_kh = identity_matrix(self.state_dim);
        for i in 0..self.state_dim * self.state_dim {
            i_kh[i] -= kh[i];
        }

        self.covariance = matrix_multiply(
            &i_kh,
            &self.covariance,
            self.state_dim,
            self.state_dim,
            self.state_dim,
        );

        Ok(self.state.clone())
    }

    /// Get current state.
    #[must_use]
    pub fn state(&self) -> &[f64] {
        &self.state
    }

    /// Get state covariance.
    #[must_use]
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }
}

/// Matrix-vector multiplication: result = A * x
fn matrix_vector_multiply(a: &[f64], x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut result = vec![0.0; rows];

    for i in 0..rows {
        let mut sum = 0.0;
        for j in 0..cols {
            sum += a[i * cols + j] * x[j];
        }
        result[i] = sum;
    }

    result
}

/// Matrix-matrix multiplication: C = A * B
fn matrix_multiply(a: &[f64], b: &[f64], m: usize, n: usize, p: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * p];

    for i in 0..m {
        for j in 0..p {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * p + j];
            }
            c[i * p + j] = sum;
        }
    }

    c
}

/// Matrix multiplication with transpose: C = A * B'
fn matrix_multiply_transpose(a: &[f64], b: &[f64], m: usize, n: usize, p: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * p];

    for i in 0..m {
        for j in 0..p {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[j * n + k]; // Note: b[j*n+k] instead of b[k*p+j]
            }
            c[i * p + j] = sum;
        }
    }

    c
}

/// Matrix addition: C = A + B
fn matrix_add(a: &[f64], b: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let size = rows * cols;
    let mut c = vec![0.0; size];

    for i in 0..size {
        c[i] = a[i] + b[i];
    }

    c
}

/// Create identity matrix
fn identity_matrix(n: usize) -> Vec<f64> {
    let mut mat = vec![0.0; n * n];
    for i in 0..n {
        mat[i * n + i] = 1.0;
    }
    mat
}

/// Matrix inversion (using Gauss-Jordan elimination).
///
/// # Errors
///
/// Returns an error if matrix is singular.
fn matrix_inverse(a: &[f64], n: usize) -> CvResult<Vec<f64>> {
    // Create augmented matrix [A | I]
    let mut aug = vec![0.0; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        let mut max_val = aug[i * 2 * n + i].abs();

        for k in (i + 1)..n {
            let val = aug[k * 2 * n + i].abs();
            if val > max_val {
                max_val = val;
                max_row = k;
            }
        }

        if max_val < 1e-10 {
            return Err(CvError::matrix_error("Matrix is singular"));
        }

        // Swap rows
        if max_row != i {
            for j in 0..2 * n {
                let tmp = aug[i * 2 * n + j];
                aug[i * 2 * n + j] = aug[max_row * 2 * n + j];
                aug[max_row * 2 * n + j] = tmp;
            }
        }

        // Scale pivot row
        let pivot = aug[i * 2 * n + i];
        for j in 0..2 * n {
            aug[i * 2 * n + j] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[k * 2 * n + i];
                for j in 0..2 * n {
                    aug[k * 2 * n + j] -= factor * aug[i * 2 * n + j];
                }
            }
        }
    }

    // Extract inverse from right half
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }

    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_filter_new() {
        let kf = KalmanFilter::new(4, 2);
        assert_eq!(kf.state_dim, 4);
        assert_eq!(kf.measure_dim, 2);
        assert_eq!(kf.state.len(), 4);
    }

    #[test]
    fn test_constant_velocity_model() {
        let kf = KalmanFilter::constant_velocity(1.0);
        assert_eq!(kf.state_dim, 4);
        assert_eq!(kf.measure_dim, 2);
    }

    #[test]
    fn test_constant_acceleration_model() {
        let kf = KalmanFilter::constant_acceleration(1.0);
        assert_eq!(kf.state_dim, 6);
        assert_eq!(kf.measure_dim, 2);
    }

    #[test]
    fn test_set_state() {
        let mut kf = KalmanFilter::new(4, 2);
        let state = vec![1.0, 2.0, 3.0, 4.0];
        kf.set_state(state.clone())
            .expect("operation should succeed");

        assert_eq!(kf.state(), &state[..]);
    }

    #[test]
    fn test_predict() {
        let mut kf = KalmanFilter::constant_velocity(1.0);
        kf.set_state(vec![0.0, 0.0, 1.0, 1.0])
            .expect("set_state should succeed");

        let predicted = kf.predict();
        assert!((predicted[0] - 1.0).abs() < 0.001); // x += vx*dt
        assert!((predicted[1] - 1.0).abs() < 0.001); // y += vy*dt
    }

    #[test]
    fn test_update() {
        let mut kf = KalmanFilter::constant_velocity(1.0);
        kf.set_state(vec![0.0, 0.0, 1.0, 1.0])
            .expect("set_state should succeed");

        let measurement = vec![1.0, 1.0];
        let updated = kf.update(&measurement).expect("update should succeed");

        // State should be updated towards measurement
        assert!(updated[0] > 0.0);
        assert!(updated[1] > 0.0);
    }

    #[test]
    fn test_predict_update_cycle() {
        let mut kf = KalmanFilter::constant_velocity(1.0);
        kf.set_state(vec![0.0, 0.0, 1.0, 0.0])
            .expect("set_state should succeed");

        // Predict
        kf.predict();

        // Update with measurement
        let measurement = vec![1.5, 0.0];
        kf.update(&measurement).expect("update should succeed");

        let state = kf.state();
        assert!(state[0] > 0.0);
    }

    #[test]
    fn test_matrix_vector_multiply() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![2.0, 1.0];

        let result = matrix_vector_multiply(&a, &x, 2, 2);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 4.0).abs() < 1e-10); // 1*2 + 2*1
        assert!((result[1] - 10.0).abs() < 1e-10); // 3*2 + 4*1
    }

    #[test]
    fn test_matrix_multiply() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 0.0, 1.0, 2.0];

        let c = matrix_multiply(&a, &b, 2, 2, 2);
        assert_eq!(c.len(), 4);
        assert!((c[0] - 4.0).abs() < 1e-10); // 1*2 + 2*1
        assert!((c[1] - 4.0).abs() < 1e-10); // 1*0 + 2*2
    }

    #[test]
    fn test_matrix_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];

        let c = matrix_add(&a, &b, 2, 2);
        assert_eq!(c, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_identity_matrix() {
        let i = identity_matrix(3);
        assert_eq!(i.len(), 9);
        assert!((i[0] - 1.0).abs() < 1e-10);
        assert!((i[4] - 1.0).abs() < 1e-10);
        assert!((i[8] - 1.0).abs() < 1e-10);
        assert!((i[1]).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_inverse() {
        let a = vec![4.0, 3.0, 3.0, 2.0];

        let inv = matrix_inverse(&a, 2).expect("matrix_inverse should succeed");
        assert_eq!(inv.len(), 4);

        // A * A^-1 should be identity
        let product = matrix_multiply(&a, &inv, 2, 2, 2);
        assert!((product[0] - 1.0).abs() < 1e-8);
        assert!((product[3] - 1.0).abs() < 1e-8);
        assert!((product[1]).abs() < 1e-8);
        assert!((product[2]).abs() < 1e-8);
    }
}
