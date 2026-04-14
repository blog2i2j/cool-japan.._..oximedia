//! IMU sensor fusion for head orientation tracking.
//!
//! This module provides a full 9-DOF inertial measurement unit (IMU) fusion
//! pipeline that combines readings from three sensors:
//!
//! - **Gyroscope** — angular velocity (deg/s), integrated to obtain orientation.
//!   Drifts over time due to integration of noise.
//! - **Accelerometer** — measures gravity + linear acceleration (g).
//!   Provides pitch and roll reference when the device is quasi-static.
//! - **Magnetometer** — measures Earth's magnetic field (µT).
//!   Provides absolute yaw (heading) reference, subject to soft/hard-iron distortion.
//!
//! # Algorithm: Mahony complementary filter (extended to 9-DOF)
//!
//! The Mahony filter fuses gyroscope data with accelerometer and magnetometer
//! corrections using a pair of PI controllers that drive the estimated gravity
//! and magnetic field directions toward their measured values.
//!
//! ```text
//! error_accel  = a_est × a_meas       (cross product — rotation axis to correct)
//! error_mag    = m_est × m_meas
//! ω_corrected  = ω_gyro + Kp*(error_accel + error_mag) + Ki*integral
//! q            = integrate(ω_corrected, dt)
//! ```
//!
//! # Coordinate convention
//! - X axis: forward (north when level and pointing north)
//! - Y axis: right
//! - Z axis: down (ENU: up when Z is flipped)
//! - Angles: right-hand rule, ZYX Euler (yaw → pitch → roll)
//!
//! # References
//! Mahony, R. et al. (2008). "Nonlinear complementary filters on the special
//! orthogonal group." *IEEE Transactions on Automatic Control*, 53(5), 1203–1218.

use std::f32::consts::PI;

use crate::SpatialError;

// ─── Constants ────────────────────────────────────────────────────────────────

/// Degrees-to-radians conversion factor.
const DEG_TO_RAD: f32 = PI / 180.0;

/// Radians-to-degrees conversion factor.
const RAD_TO_DEG: f32 = 180.0 / PI;

// ─── Raw sensor readings ──────────────────────────────────────────────────────

/// Raw 3-axis gyroscope reading (angular velocity in deg/s).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GyroReading {
    /// Angular velocity around X axis (deg/s).
    pub x: f32,
    /// Angular velocity around Y axis (deg/s).
    pub y: f32,
    /// Angular velocity around Z axis (deg/s).
    pub z: f32,
}

/// Raw 3-axis accelerometer reading (acceleration in g).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AccelReading {
    /// Acceleration along X axis (g).
    pub x: f32,
    /// Acceleration along Y axis (g).
    pub y: f32,
    /// Acceleration along Z axis (g).
    pub z: f32,
}

/// Raw 3-axis magnetometer reading (magnetic field in µT or arbitrary units).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MagReading {
    /// Magnetic field along X axis.
    pub x: f32,
    /// Magnetic field along Y axis.
    pub y: f32,
    /// Magnetic field along Z axis.
    pub z: f32,
}

/// Combined IMU sensor sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuSample {
    /// Gyroscope angular velocity (deg/s).
    pub gyro: GyroReading,
    /// Accelerometer measurement (g).
    pub accel: AccelReading,
    /// Magnetometer measurement (µT or arbitrary units). Optional — when
    /// `None`, only 6-DOF (accel + gyro) fusion is performed and yaw drifts.
    pub mag: Option<MagReading>,
}

// ─── Quaternion ────────────────────────────────────────────────────────────────

/// Unit quaternion representing a 3-D orientation.
///
/// Stored as `(w, x, y, z)` where `w` is the scalar part.
#[derive(Debug, Clone, Copy)]
pub struct FusionQuaternion {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl FusionQuaternion {
    /// Identity quaternion (no rotation).
    pub fn identity() -> Self {
        Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 }
    }

    /// Normalise to unit length.  Returns identity if near-zero.
    pub fn normalise(self) -> Self {
        let norm = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if norm < 1e-9 {
            Self::identity()
        } else {
            Self {
                w: self.w / norm,
                x: self.x / norm,
                y: self.y / norm,
                z: self.z / norm,
            }
        }
    }

    /// Quaternion multiplication `self ⊗ rhs`.
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }

    /// Convert quaternion to ZYX Euler angles in degrees: `(yaw, pitch, roll)`.
    ///
    /// - **yaw**   — rotation around Z axis (heading, -180..+180 deg)
    /// - **pitch** — rotation around Y axis (-90..+90 deg)
    /// - **roll**  — rotation around X axis (-180..+180 deg)
    pub fn to_euler_deg(self) -> EulerAngles {
        let q = self.normalise();

        // Roll (rotation about X axis)
        let sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
        let cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
        let roll = sinr_cosp.atan2(cosr_cosp) * RAD_TO_DEG;

        // Pitch (rotation about Y axis) — clamped to avoid gimbal lock singularity
        let sinp = 2.0 * (q.w * q.y - q.z * q.x);
        let pitch = if sinp.abs() >= 1.0 {
            sinp.signum() * 90.0
        } else {
            sinp.asin() * RAD_TO_DEG
        };

        // Yaw (rotation about Z axis)
        let siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        let cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        let yaw = siny_cosp.atan2(cosy_cosp) * RAD_TO_DEG;

        EulerAngles { yaw_deg: yaw, pitch_deg: pitch, roll_deg: roll }
    }
}

impl Default for FusionQuaternion {
    fn default() -> Self {
        Self::identity()
    }
}

// ─── Output ────────────────────────────────────────────────────────────────────

/// ZYX Euler angles output from the fusion filter (all in degrees).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EulerAngles {
    /// Heading — rotation around the vertical (Z) axis.  Range: -180..+180 deg.
    pub yaw_deg: f32,
    /// Pitch — rotation around the lateral (Y) axis.  Range: -90..+90 deg.
    pub pitch_deg: f32,
    /// Roll — rotation around the forward (X) axis.  Range: -180..+180 deg.
    pub roll_deg: f32,
}

// ─── Mahony filter configuration ──────────────────────────────────────────────

/// Configuration for the Mahony complementary filter.
#[derive(Debug, Clone)]
pub struct MahonyConfig {
    /// Proportional gain for accelerometer feedback.  Higher = faster correction,
    /// but more susceptible to linear acceleration disturbances.  Default: 2.0.
    pub kp_accel: f32,
    /// Integral gain for accelerometer bias estimation.  Default: 0.005.
    pub ki_accel: f32,
    /// Proportional gain for magnetometer feedback.  Default: 2.0.
    pub kp_mag: f32,
    /// Integral gain for magnetometer bias estimation.  Default: 0.005.
    pub ki_mag: f32,
    /// If `true`, magnetometer readings are ignored even when supplied,
    /// forcing 6-DOF (gyro + accel only) operation.
    pub disable_magnetometer: bool,
}

impl MahonyConfig {
    /// Create default Mahony configuration (suitable for head-tracking use cases).
    pub fn new() -> Self {
        Self {
            kp_accel: 2.0,
            ki_accel: 0.005,
            kp_mag: 2.0,
            ki_mag: 0.005,
            disable_magnetometer: false,
        }
    }

    /// Create a 6-DOF configuration (gyro + accelerometer only, no magnetometer).
    pub fn six_dof() -> Self {
        Self { disable_magnetometer: true, ..Self::new() }
    }
}

impl Default for MahonyConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Magnetometer calibration ─────────────────────────────────────────────────

/// Hard-iron and soft-iron magnetometer calibration parameters.
///
/// Hard-iron offsets arise from permanent magnetic fields near the sensor.
/// Soft-iron distortion results from the sensor being mounted in a magnetically
/// permeable material that distorts the field shape.
#[derive(Debug, Clone)]
pub struct MagCalibration {
    /// Hard-iron offset vector `[x, y, z]`.  Subtracted from raw readings.
    pub hard_iron: [f32; 3],
    /// Soft-iron 3×3 correction matrix (row-major).  Applied after hard-iron removal.
    pub soft_iron: [[f32; 3]; 3],
}

impl MagCalibration {
    /// Create a default (identity) calibration (no correction applied).
    pub fn identity() -> Self {
        Self {
            hard_iron: [0.0; 3],
            soft_iron: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }
    }

    /// Apply calibration to a raw magnetometer reading.
    pub fn apply(&self, raw: MagReading) -> MagReading {
        let v = [raw.x - self.hard_iron[0], raw.y - self.hard_iron[1], raw.z - self.hard_iron[2]];
        let m = &self.soft_iron;
        MagReading {
            x: m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            y: m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            z: m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        }
    }
}

impl Default for MagCalibration {
    fn default() -> Self {
        Self::identity()
    }
}

// ─── Gyroscope bias estimation ────────────────────────────────────────────────

/// Online gyroscope bias estimator using a running mean over a static window.
///
/// When the IMU is detected to be static (accel magnitude close to 1 g,
/// gyro magnitude below a threshold), the gyroscope readings are averaged
/// to estimate the bias offset.
#[derive(Debug, Clone)]
pub struct GyroBiasEstimator {
    /// Accumulated gyro sum for bias computation.
    sum: [f32; 3],
    /// Number of samples in the accumulation window.
    count: u32,
    /// Maximum number of samples to average before freezing the estimate.
    window: u32,
    /// Estimated bias `[x, y, z]` in deg/s.
    pub bias: [f32; 3],
    /// Gyro magnitude threshold (deg/s) below which the device is considered static.
    pub static_gyro_threshold: f32,
    /// Accel magnitude deviation tolerance from 1 g.
    pub static_accel_tolerance: f32,
}

impl GyroBiasEstimator {
    /// Create a new estimator.
    ///
    /// * `window` — number of static samples to average (e.g. 200 at 100 Hz = 2 s).
    pub fn new(window: u32) -> Self {
        Self {
            sum: [0.0; 3],
            count: 0,
            window: window.max(1),
            bias: [0.0; 3],
            static_gyro_threshold: 1.0,
            static_accel_tolerance: 0.05,
        }
    }

    /// Update the estimator with a new sensor sample.
    ///
    /// If the device appears static, the gyro reading is accumulated.
    /// Once `window` static samples have been collected, the bias is updated.
    pub fn update(&mut self, gyro: GyroReading, accel: AccelReading) {
        let gyro_mag = (gyro.x * gyro.x + gyro.y * gyro.y + gyro.z * gyro.z).sqrt();
        let accel_mag = (accel.x * accel.x + accel.y * accel.y + accel.z * accel.z).sqrt();
        let is_static = gyro_mag < self.static_gyro_threshold
            && (accel_mag - 1.0).abs() < self.static_accel_tolerance;

        if is_static {
            self.sum[0] += gyro.x;
            self.sum[1] += gyro.y;
            self.sum[2] += gyro.z;
            self.count += 1;

            if self.count >= self.window {
                let n = self.count as f32;
                self.bias[0] = self.sum[0] / n;
                self.bias[1] = self.sum[1] / n;
                self.bias[2] = self.sum[2] / n;
                // Reset accumulator for the next window.
                self.sum = [0.0; 3];
                self.count = 0;
            }
        } else {
            // Reset if we're not static — avoid mixing motion and static samples.
            self.sum = [0.0; 3];
            self.count = 0;
        }
    }

    /// Remove estimated bias from a gyro reading.
    pub fn debias(&self, gyro: GyroReading) -> GyroReading {
        GyroReading {
            x: gyro.x - self.bias[0],
            y: gyro.y - self.bias[1],
            z: gyro.z - self.bias[2],
        }
    }
}

// ─── Mahony IMU Fusion Filter ──────────────────────────────────────────────────

/// 9-DOF Mahony complementary filter for IMU sensor fusion.
///
/// Fuses gyroscope, accelerometer, and (optionally) magnetometer readings into
/// a quaternion orientation estimate with automatic gyroscope bias correction.
///
/// # Example
///
/// ```rust
/// use oximedia_spatial::imu_fusion::{MahonyFilter, MahonyConfig, ImuSample,
///                                    GyroReading, AccelReading};
///
/// let mut filter = MahonyFilter::new(MahonyConfig::six_dof());
/// let sample = ImuSample {
///     gyro: GyroReading { x: 0.0, y: 0.0, z: 0.0 },
///     accel: AccelReading { x: 0.0, y: 0.0, z: 1.0 },
///     mag: None,
/// };
/// let angles = filter.update(sample, 1.0 / 100.0);
/// assert!((angles.pitch_deg).abs() < 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct MahonyFilter {
    /// Current orientation estimate.
    pub quaternion: FusionQuaternion,
    /// PI filter integral error accumulator `[x, y, z]`.
    integral_error: [f32; 3],
    /// Filter configuration.
    pub config: MahonyConfig,
}

impl MahonyFilter {
    /// Create a new Mahony filter with the supplied configuration.
    pub fn new(config: MahonyConfig) -> Self {
        Self {
            quaternion: FusionQuaternion::identity(),
            integral_error: [0.0; 3],
            config,
        }
    }

    /// Update the filter with a new IMU sample and time-step `dt` (seconds).
    ///
    /// Returns the current orientation as ZYX Euler angles in degrees.
    pub fn update(&mut self, sample: ImuSample, dt: f32) -> EulerAngles {
        let q = self.quaternion;

        // ── 1. Normalise accelerometer ──────────────────────────────────────
        let ax = sample.accel.x;
        let ay = sample.accel.y;
        let az = sample.accel.z;
        let accel_norm = (ax * ax + ay * ay + az * az).sqrt();

        let mut error = [0.0_f32; 3];

        if accel_norm > 1e-9 {
            let (ax, ay, az) = (ax / accel_norm, ay / accel_norm, az / accel_norm);

            // Estimated gravity direction from quaternion (third column of rotation matrix).
            let vx = 2.0 * (q.x * q.z - q.w * q.y);
            let vy = 2.0 * (q.w * q.x + q.y * q.z);
            let vz = q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z;

            // Cross product error between estimated and measured gravity.
            error[0] += ay * vz - az * vy;
            error[1] += az * vx - ax * vz;
            error[2] += ax * vy - ay * vx;
        }

        // ── 2. Magnetometer correction (9-DOF) ──────────────────────────────
        if !self.config.disable_magnetometer {
            if let Some(mag) = sample.mag {
                let mx = mag.x;
                let my = mag.y;
                let mz = mag.z;
                let mag_norm = (mx * mx + my * my + mz * mz).sqrt();

                if mag_norm > 1e-9 {
                    let (mx, my, mz) = (mx / mag_norm, my / mag_norm, mz / mag_norm);

                    // Reference direction of Earth's magnetic field in body frame.
                    // Computed by rotating the measured field back to the world frame
                    // and zeroing the Y component (so only inclination angle is preserved).
                    let hx = 2.0 * mx * (0.5 - q.y * q.y - q.z * q.z)
                        + 2.0 * my * (q.x * q.y - q.w * q.z)
                        + 2.0 * mz * (q.x * q.z + q.w * q.y);
                    let hz = 2.0 * mx * (q.x * q.z - q.w * q.y)
                        + 2.0 * my * (q.y * q.z + q.w * q.x)
                        + 2.0 * mz * (0.5 - q.x * q.x - q.y * q.y);
                    let bx = (hx * hx + hz * hz).sqrt(); // project onto XZ plane
                    let bz = hz;

                    // Estimated magnetic field direction from current quaternion.
                    let wx = 2.0 * bx * (0.5 - q.y * q.y - q.z * q.z)
                        + 2.0 * bz * (q.x * q.z - q.w * q.y);
                    let wy = 2.0 * bx * (q.x * q.y - q.w * q.z)
                        + 2.0 * bz * (q.w * q.x + q.y * q.z);
                    let wz = 2.0 * bx * (q.w * q.y + q.x * q.z)
                        + 2.0 * bz * (0.5 - q.x * q.x - q.y * q.y);

                    // Cross product error.
                    error[0] += my * wz - mz * wy;
                    error[1] += mz * wx - mx * wz;
                    error[2] += mx * wy - my * wx;
                }
            }
        }

        // ── 3. PI correction on gyroscope ────────────────────────────────────
        self.integral_error[0] += error[0] * dt;
        self.integral_error[1] += error[1] * dt;
        self.integral_error[2] += error[2] * dt;

        let gx = sample.gyro.x * DEG_TO_RAD
            + self.config.kp_accel * error[0]
            + self.config.ki_accel * self.integral_error[0];
        let gy = sample.gyro.y * DEG_TO_RAD
            + self.config.kp_accel * error[1]
            + self.config.ki_accel * self.integral_error[1];
        let gz = sample.gyro.z * DEG_TO_RAD
            + self.config.kp_accel * error[2]
            + self.config.ki_accel * self.integral_error[2];

        // ── 4. Integrate quaternion rate ─────────────────────────────────────
        let half_dt = 0.5 * dt;
        let dq = FusionQuaternion {
            w: -q.x * gx - q.y * gy - q.z * gz,
            x:  q.w * gx + q.y * gz - q.z * gy,
            y:  q.w * gy - q.x * gz + q.z * gx,
            z:  q.w * gz + q.x * gy - q.y * gx,
        };
        self.quaternion = FusionQuaternion {
            w: q.w + dq.w * half_dt,
            x: q.x + dq.x * half_dt,
            y: q.y + dq.y * half_dt,
            z: q.z + dq.z * half_dt,
        }
        .normalise();

        self.quaternion.to_euler_deg()
    }

    /// Reset the filter to the identity orientation.
    pub fn reset(&mut self) {
        self.quaternion = FusionQuaternion::identity();
        self.integral_error = [0.0; 3];
    }

    /// Return the current orientation estimate as Euler angles in degrees.
    pub fn euler_angles(&self) -> EulerAngles {
        self.quaternion.to_euler_deg()
    }
}

// ─── Full 9-DOF fusion pipeline ───────────────────────────────────────────────

/// High-level 9-DOF IMU fusion pipeline combining:
/// - Mahony complementary filter for orientation estimation
/// - Gyroscope bias estimation during static periods
/// - Magnetometer hard/soft-iron calibration
///
/// # Example
///
/// ```rust
/// use oximedia_spatial::imu_fusion::{ImuFusionPipeline, ImuSample,
///                                    GyroReading, AccelReading};
///
/// let mut pipeline = ImuFusionPipeline::new(100); // 100 Hz sample rate
/// let sample = ImuSample {
///     gyro: GyroReading { x: 0.0, y: 0.0, z: 0.0 },
///     accel: AccelReading { x: 0.0, y: 0.0, z: 1.0 },
///     mag: None,
/// };
/// let angles = pipeline.process(sample);
/// assert!(angles.is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct ImuFusionPipeline {
    /// Mahony filter performing the quaternion integration.
    filter: MahonyFilter,
    /// Gyroscope bias estimator.
    bias_estimator: GyroBiasEstimator,
    /// Magnetometer calibration parameters.
    mag_calibration: MagCalibration,
    /// Nominal sample rate in Hz (used to compute `dt`).
    sample_rate_hz: u32,
}

impl ImuFusionPipeline {
    /// Create a new pipeline.
    ///
    /// * `sample_rate_hz` — expected IMU sample rate (e.g. 100 for 100 Hz).
    pub fn new(sample_rate_hz: u32) -> Self {
        let rate = sample_rate_hz.max(1);
        Self {
            filter: MahonyFilter::new(MahonyConfig::new()),
            bias_estimator: GyroBiasEstimator::new(rate * 2), // 2-second window
            mag_calibration: MagCalibration::identity(),
            sample_rate_hz: rate,
        }
    }

    /// Override the Mahony filter configuration.
    pub fn with_config(mut self, config: MahonyConfig) -> Self {
        self.filter = MahonyFilter::new(config);
        self
    }

    /// Set magnetometer calibration parameters.
    pub fn with_mag_calibration(mut self, cal: MagCalibration) -> Self {
        self.mag_calibration = cal;
        self
    }

    /// Process a single IMU sample, returning the estimated orientation.
    ///
    /// Internally:
    /// 1. Applies magnetometer calibration (if mag is present).
    /// 2. Updates the gyro bias estimator.
    /// 3. Removes estimated bias from the gyro reading.
    /// 4. Runs the Mahony filter update step.
    pub fn process(&mut self, sample: ImuSample) -> Result<EulerAngles, SpatialError> {
        // Calibrate magnetometer.
        let calibrated_mag = sample.mag.map(|m| self.mag_calibration.apply(m));

        // Update bias estimator.
        self.bias_estimator.update(sample.gyro, sample.accel);

        // Remove gyro bias.
        let debiased_gyro = self.bias_estimator.debias(sample.gyro);

        let corrected = ImuSample {
            gyro: debiased_gyro,
            accel: sample.accel,
            mag: calibrated_mag,
        };

        let dt = 1.0 / self.sample_rate_hz as f32;
        Ok(self.filter.update(corrected, dt))
    }

    /// Return the current orientation quaternion.
    pub fn quaternion(&self) -> FusionQuaternion {
        self.filter.quaternion
    }

    /// Return the current Euler angle estimate.
    pub fn euler_angles(&self) -> EulerAngles {
        self.filter.euler_angles()
    }

    /// Return the estimated gyroscope bias (deg/s per axis).
    pub fn gyro_bias(&self) -> [f32; 3] {
        self.bias_estimator.bias
    }

    /// Reset the filter and bias estimator to initial state.
    pub fn reset(&mut self) {
        self.filter.reset();
        self.bias_estimator = GyroBiasEstimator::new(self.sample_rate_hz * 2);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_sample(gyro_x: f32, gyro_y: f32, gyro_z: f32) -> ImuSample {
        ImuSample {
            gyro: GyroReading { x: gyro_x, y: gyro_y, z: gyro_z },
            accel: AccelReading { x: 0.0, y: 0.0, z: 1.0 },
            mag: None,
        }
    }

    #[test]
    fn test_identity_quaternion() {
        let q = FusionQuaternion::identity();
        assert!((q.w - 1.0).abs() < 1e-6);
        assert!(q.x.abs() < 1e-6);
        assert!(q.y.abs() < 1e-6);
        assert!(q.z.abs() < 1e-6);
    }

    #[test]
    fn test_quaternion_normalise() {
        let q = FusionQuaternion { w: 2.0, x: 0.0, y: 0.0, z: 0.0 }.normalise();
        assert!((q.w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quaternion_near_zero_normalise() {
        let q = FusionQuaternion { w: 0.0, x: 0.0, y: 0.0, z: 0.0 }.normalise();
        // Should return identity
        assert!((q.w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_euler_identity_is_zero() {
        let angles = FusionQuaternion::identity().to_euler_deg();
        assert!(angles.yaw_deg.abs() < 1e-3);
        assert!(angles.pitch_deg.abs() < 1e-3);
        assert!(angles.roll_deg.abs() < 1e-3);
    }

    #[test]
    fn test_static_sensor_convergence() {
        // Feed 200 samples with zero gyro and gravity pointing down.
        let mut filter = MahonyFilter::new(MahonyConfig::six_dof());
        let sample = flat_sample(0.0, 0.0, 0.0);
        let mut last_angles = EulerAngles { yaw_deg: 0.0, pitch_deg: 0.0, roll_deg: 0.0 };
        for _ in 0..200 {
            last_angles = filter.update(sample, 0.01);
        }
        // With gravity pointing in +Z, pitch and roll should stay near zero.
        assert!(last_angles.pitch_deg.abs() < 2.0, "pitch={}", last_angles.pitch_deg);
        assert!(last_angles.roll_deg.abs() < 2.0, "roll={}", last_angles.roll_deg);
    }

    #[test]
    fn test_gyro_bias_estimator_static() {
        let mut est = GyroBiasEstimator::new(50);
        for _ in 0..50 {
            est.update(
                GyroReading { x: 0.3, y: -0.1, z: 0.05 },
                AccelReading { x: 0.0, y: 0.0, z: 1.0 },
            );
        }
        // After 50 static samples, bias should be estimated.
        assert!((est.bias[0] - 0.3).abs() < 0.01, "bias_x={}", est.bias[0]);
        assert!((est.bias[1] - (-0.1)).abs() < 0.01, "bias_y={}", est.bias[1]);
    }

    #[test]
    fn test_gyro_bias_debias() {
        let est = GyroBiasEstimator {
            sum: [0.0; 3],
            count: 0,
            window: 100,
            bias: [1.0, 2.0, 3.0],
            static_gyro_threshold: 1.0,
            static_accel_tolerance: 0.05,
        };
        let debiased = est.debias(GyroReading { x: 1.5, y: 2.5, z: 3.5 });
        assert!((debiased.x - 0.5).abs() < 1e-6);
        assert!((debiased.y - 0.5).abs() < 1e-6);
        assert!((debiased.z - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_mag_calibration_identity() {
        let cal = MagCalibration::identity();
        let raw = MagReading { x: 30.0, y: -10.0, z: 5.0 };
        let out = cal.apply(raw);
        assert!((out.x - 30.0).abs() < 1e-5);
        assert!((out.y - (-10.0)).abs() < 1e-5);
        assert!((out.z - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_mag_calibration_hard_iron_offset() {
        let cal = MagCalibration {
            hard_iron: [10.0, 5.0, 2.0],
            soft_iron: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        };
        let raw = MagReading { x: 40.0, y: 15.0, z: 7.0 };
        let out = cal.apply(raw);
        assert!((out.x - 30.0).abs() < 1e-5);
        assert!((out.y - 10.0).abs() < 1e-5);
        assert!((out.z - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_pipeline_process_returns_ok() {
        let mut pipeline = ImuFusionPipeline::new(100);
        let sample = flat_sample(0.0, 0.0, 0.0);
        let result = pipeline.process(sample);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pipeline_reset_returns_to_identity() {
        let mut pipeline = ImuFusionPipeline::new(100);
        // Feed some non-zero motion.
        for _ in 0..50 {
            let _ = pipeline.process(ImuSample {
                gyro: GyroReading { x: 10.0, y: 5.0, z: 2.0 },
                accel: AccelReading { x: 0.1, y: 0.0, z: 0.99 },
                mag: None,
            });
        }
        pipeline.reset();
        let q = pipeline.quaternion();
        assert!((q.w - 1.0).abs() < 1e-6, "w={}", q.w);
    }

    #[test]
    fn test_pipeline_with_magnetometer() {
        let mut pipeline = ImuFusionPipeline::new(100);
        let sample = ImuSample {
            gyro: GyroReading { x: 0.0, y: 0.0, z: 0.0 },
            accel: AccelReading { x: 0.0, y: 0.0, z: 1.0 },
            mag: Some(MagReading { x: 30.0, y: 0.0, z: -50.0 }),
        };
        for _ in 0..100 {
            let result = pipeline.process(sample);
            assert!(result.is_ok());
        }
        // After convergence with a north-pointing mag, pitch should be bounded.
        let angles = pipeline.euler_angles();
        assert!(angles.pitch_deg.abs() < 30.0, "pitch={}", angles.pitch_deg);
    }

    #[test]
    fn test_mahony_config_six_dof() {
        let cfg = MahonyConfig::six_dof();
        assert!(cfg.disable_magnetometer);
    }

    #[test]
    fn test_gyro_bias_no_update_when_moving() {
        let mut est = GyroBiasEstimator::new(50);
        // Motion: gyro magnitude > threshold
        est.update(
            GyroReading { x: 90.0, y: 0.0, z: 0.0 },
            AccelReading { x: 0.0, y: 0.0, z: 1.0 },
        );
        // Bias should remain zero (not updated from a single motion sample).
        assert!(est.bias[0].abs() < 1e-6);
    }
}
