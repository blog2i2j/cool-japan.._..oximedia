//! IMU (Inertial Measurement Unit) sensor integration
//!
//! Provides high-frequency orientation tracking using gyroscope and
//! accelerometer data with sensor fusion.

use super::CameraPose;
use crate::math::{Point3, Unit, UnitQuaternion, Vector3};
use crate::{Result, VirtualProductionError};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// IMU sensor data
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ImuData {
    /// Acceleration in m/s² (x, y, z)
    pub acceleration: Vector3<f64>,
    /// Angular velocity in rad/s (x, y, z)
    pub angular_velocity: Vector3<f64>,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
}

impl ImuData {
    /// Create new IMU data
    #[must_use]
    pub fn new(
        acceleration: Vector3<f64>,
        angular_velocity: Vector3<f64>,
        timestamp_ns: u64,
    ) -> Self {
        Self {
            acceleration,
            angular_velocity,
            timestamp_ns,
        }
    }
}

impl Default for ImuData {
    fn default() -> Self {
        Self {
            acceleration: Vector3::new(0.0, 0.0, -9.81), // Gravity
            angular_velocity: Vector3::zeros(),
            timestamp_ns: 0,
        }
    }
}

/// IMU sensor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImuSensorConfig {
    /// Sample rate in Hz
    pub sample_rate: f64,
    /// Accelerometer noise (m/s²)
    pub accel_noise: f64,
    /// Gyroscope noise (rad/s)
    pub gyro_noise: f64,
    /// Complementary filter coefficient (0.0 - 1.0)
    pub filter_coefficient: f64,
    /// History buffer size
    pub buffer_size: usize,
}

impl Default for ImuSensorConfig {
    fn default() -> Self {
        Self {
            sample_rate: 200.0,
            accel_noise: 0.01,
            gyro_noise: 0.001,
            filter_coefficient: 0.98,
            buffer_size: 100,
        }
    }
}

/// IMU sensor for orientation tracking
pub struct ImuSensor {
    config: ImuSensorConfig,
    data_buffer: VecDeque<ImuData>,
    current_orientation: UnitQuaternion<f64>,
    current_position: Point3<f64>,
    velocity: Vector3<f64>,
    last_timestamp_ns: u64,
}

impl ImuSensor {
    /// Create new IMU sensor
    pub fn new(config: ImuSensorConfig) -> Result<Self> {
        let buffer_size = config.buffer_size;
        Ok(Self {
            config,
            data_buffer: VecDeque::with_capacity(buffer_size),
            current_orientation: UnitQuaternion::identity(),
            current_position: Point3::origin(),
            velocity: Vector3::zeros(),
            last_timestamp_ns: 0,
        })
    }

    /// Update sensor with new IMU data
    pub fn update(&mut self, data: ImuData) -> Result<()> {
        // Add to buffer
        self.data_buffer.push_back(data);
        if self.data_buffer.len() > self.config.buffer_size {
            self.data_buffer.pop_front();
        }

        // Update orientation from gyroscope
        if self.last_timestamp_ns > 0 {
            let dt = (data.timestamp_ns - self.last_timestamp_ns) as f64 * 1e-9;
            self.integrate_gyroscope(data.angular_velocity, dt);
            self.integrate_accelerometer(data.acceleration, dt);
        }

        self.last_timestamp_ns = data.timestamp_ns;

        Ok(())
    }

    /// Integrate gyroscope data to update orientation
    fn integrate_gyroscope(&mut self, angular_velocity: Vector3<f64>, dt: f64) {
        // Compute rotation increment
        let angle = angular_velocity.norm() * dt;
        if angle > 1e-10 {
            let axis = angular_velocity / angular_velocity.norm();
            let rotation = UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), angle);
            self.current_orientation *= rotation;
        }
    }

    /// Integrate accelerometer data to update position
    fn integrate_accelerometer(&mut self, acceleration: Vector3<f64>, dt: f64) {
        // Remove gravity (assumes orientation is correct)
        let gravity = Vector3::new(0.0, 0.0, -9.81);
        let world_accel = self.current_orientation * acceleration - gravity;

        // Apply complementary filter to reduce drift
        let filtered_accel = world_accel * (1.0 - self.config.filter_coefficient);

        // Update velocity and position
        self.velocity += filtered_accel * dt;
        self.current_position += self.velocity * dt;

        // Apply velocity damping to reduce drift
        self.velocity *= 0.99;
    }

    /// Get current pose estimate
    pub fn get_pose(&self, timestamp_ns: u64) -> Result<CameraPose> {
        // Confidence decreases with time since last update
        let time_since_update = if self.last_timestamp_ns > 0 {
            (timestamp_ns.saturating_sub(self.last_timestamp_ns)) as f64 * 1e-9
        } else {
            0.0
        };

        let confidence = (1.0 / (1.0 + time_since_update)).min(1.0) as f32;

        Ok(CameraPose {
            position: self.current_position,
            orientation: self.current_orientation,
            timestamp_ns,
            confidence,
        })
    }

    /// Reset IMU state
    pub fn reset(&mut self) {
        self.data_buffer.clear();
        self.current_orientation = UnitQuaternion::identity();
        self.current_position = Point3::origin();
        self.velocity = Vector3::zeros();
        self.last_timestamp_ns = 0;
    }

    /// Calibrate IMU (zero out bias)
    pub fn calibrate(&mut self, num_samples: usize) -> Result<()> {
        if self.data_buffer.len() < num_samples {
            return Err(VirtualProductionError::CameraTracking(format!(
                "Not enough samples for calibration: {} < {}",
                self.data_buffer.len(),
                num_samples
            )));
        }

        // Compute average bias from recent samples
        let mut accel_bias = Vector3::zeros();
        let mut gyro_bias = Vector3::zeros();

        let samples_to_use = self.data_buffer.len().min(num_samples);
        for data in self.data_buffer.iter().rev().take(samples_to_use) {
            accel_bias += data.acceleration;
            gyro_bias += data.angular_velocity;
        }

        accel_bias /= samples_to_use as f64;
        gyro_bias /= samples_to_use as f64;

        // Subtract bias from all buffered data
        for data in &mut self.data_buffer {
            data.acceleration -= accel_bias;
            data.angular_velocity -= gyro_bias;
        }

        Ok(())
    }

    /// Get number of buffered samples
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.data_buffer.len()
    }

    /// Get current orientation
    #[must_use]
    pub fn orientation(&self) -> &UnitQuaternion<f64> {
        &self.current_orientation
    }

    /// Get current position
    #[must_use]
    pub fn position(&self) -> &Point3<f64> {
        &self.current_position
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &ImuSensorConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imu_data() {
        let data = ImuData::default();
        assert_eq!(data.angular_velocity, Vector3::zeros());
        assert!((data.acceleration.z + 9.81).abs() < 1e-6);
    }

    #[test]
    fn test_imu_sensor_creation() {
        let config = ImuSensorConfig::default();
        let sensor = ImuSensor::new(config);
        assert!(sensor.is_ok());
    }

    #[test]
    fn test_imu_sensor_update() {
        let config = ImuSensorConfig::default();
        let mut sensor = ImuSensor::new(config).expect("should succeed in test");

        let data = ImuData::new(Vector3::new(0.0, 0.0, -9.81), Vector3::zeros(), 1000000);

        let result = sensor.update(data);
        assert!(result.is_ok());
        assert_eq!(sensor.buffer_size(), 1);
    }

    #[test]
    fn test_imu_sensor_reset() {
        let config = ImuSensorConfig::default();
        let mut sensor = ImuSensor::new(config).expect("should succeed in test");

        sensor
            .update(ImuData::default())
            .expect("should succeed in test");
        sensor.reset();

        assert_eq!(sensor.buffer_size(), 0);
        assert_eq!(sensor.position(), &Point3::origin());
    }

    #[test]
    fn test_imu_integration() {
        let config = ImuSensorConfig::default();
        let mut sensor = ImuSensor::new(config).expect("should succeed in test");

        // Simulate constant rotation
        for i in 0..10 {
            let data = ImuData::new(
                Vector3::new(0.0, 0.0, -9.81),
                Vector3::new(0.1, 0.0, 0.0), // Rotate around X-axis
                i * 1_000_000,
            );
            sensor.update(data).expect("should succeed in test");
        }

        // Orientation should have changed
        assert_ne!(sensor.orientation(), &UnitQuaternion::identity());
    }

    #[test]
    fn test_get_pose() {
        let config = ImuSensorConfig::default();
        let sensor = ImuSensor::new(config).expect("should succeed in test");
        let pose = sensor.get_pose(1000000);
        assert!(pose.is_ok());
    }
}
