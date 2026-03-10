//! Telemetry data (drones, cameras).
//!
//! Handles IMU and sensor telemetry data.

#![forbid(unsafe_code)]

use bytes::{BufMut, Bytes, BytesMut};
use oximedia_core::{OxiError, OxiResult};

/// IMU (Inertial Measurement Unit) data.
#[derive(Debug, Clone, Copy)]
pub struct ImuData {
    /// Accelerometer X (m/s²).
    pub accel_x: f32,
    /// Accelerometer Y (m/s²).
    pub accel_y: f32,
    /// Accelerometer Z (m/s²).
    pub accel_z: f32,
    /// Gyroscope X (rad/s).
    pub gyro_x: f32,
    /// Gyroscope Y (rad/s).
    pub gyro_y: f32,
    /// Gyroscope Z (rad/s).
    pub gyro_z: f32,
    /// Magnetometer X (µT).
    pub mag_x: f32,
    /// Magnetometer Y (µT).
    pub mag_y: f32,
    /// Magnetometer Z (µT).
    pub mag_z: f32,
}

impl ImuData {
    /// Creates a new IMU data point.
    #[must_use]
    pub const fn new(
        accel_x: f32,
        accel_y: f32,
        accel_z: f32,
        gyro_x: f32,
        gyro_y: f32,
        gyro_z: f32,
    ) -> Self {
        Self {
            accel_x,
            accel_y,
            accel_z,
            gyro_x,
            gyro_y,
            gyro_z,
            mag_x: 0.0,
            mag_y: 0.0,
            mag_z: 0.0,
        }
    }

    /// Sets magnetometer values.
    #[must_use]
    pub const fn with_magnetometer(mut self, mag_x: f32, mag_y: f32, mag_z: f32) -> Self {
        self.mag_x = mag_x;
        self.mag_y = mag_y;
        self.mag_z = mag_z;
        self
    }

    /// Serializes to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(36);

        buf.put_f32(self.accel_x);
        buf.put_f32(self.accel_y);
        buf.put_f32(self.accel_z);
        buf.put_f32(self.gyro_x);
        buf.put_f32(self.gyro_y);
        buf.put_f32(self.gyro_z);
        buf.put_f32(self.mag_x);
        buf.put_f32(self.mag_y);
        buf.put_f32(self.mag_z);

        buf.freeze()
    }

    /// Deserializes from bytes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the data slice is shorter than 36 bytes.
    ///
    /// # Panics
    ///
    /// Panics if byte slice conversion fails (should not happen given the length check).
    pub fn from_bytes(data: &[u8]) -> OxiResult<Self> {
        if data.len() < 36 {
            return Err(OxiError::InvalidData("IMU data too short".into()));
        }

        Ok(Self {
            accel_x: f32::from_be_bytes(data[0..4].try_into().expect("slice length checked")),
            accel_y: f32::from_be_bytes(data[4..8].try_into().expect("slice length checked")),
            accel_z: f32::from_be_bytes(data[8..12].try_into().expect("slice length checked")),
            gyro_x: f32::from_be_bytes(data[12..16].try_into().expect("slice length checked")),
            gyro_y: f32::from_be_bytes(data[16..20].try_into().expect("slice length checked")),
            gyro_z: f32::from_be_bytes(data[20..24].try_into().expect("slice length checked")),
            mag_x: f32::from_be_bytes(data[24..28].try_into().expect("slice length checked")),
            mag_y: f32::from_be_bytes(data[28..32].try_into().expect("slice length checked")),
            mag_z: f32::from_be_bytes(data[32..36].try_into().expect("slice length checked")),
        })
    }
}

/// Camera exposure data.
#[derive(Debug, Clone, Copy)]
pub struct ExposureData {
    /// ISO value.
    pub iso: u16,
    /// Shutter speed (1/n seconds).
    pub shutter_speed: u16,
    /// Aperture (f-stop * 10).
    pub aperture: u16,
    /// White balance in Kelvin.
    pub white_balance: u16,
}

impl ExposureData {
    /// Creates a new exposure data point.
    #[must_use]
    pub const fn new(iso: u16, shutter_speed: u16, aperture: u16, white_balance: u16) -> Self {
        Self {
            iso,
            shutter_speed,
            aperture,
            white_balance,
        }
    }

    /// Serializes to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(8);

        buf.put_u16(self.iso);
        buf.put_u16(self.shutter_speed);
        buf.put_u16(self.aperture);
        buf.put_u16(self.white_balance);

        buf.freeze()
    }

    /// Deserializes from bytes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the data slice is shorter than 8 bytes.
    ///
    /// # Panics
    ///
    /// Panics if byte slice conversion fails (should not happen given the length check).
    pub fn from_bytes(data: &[u8]) -> OxiResult<Self> {
        if data.len() < 8 {
            return Err(OxiError::InvalidData("Exposure data too short".into()));
        }

        Ok(Self {
            iso: u16::from_be_bytes(data[0..2].try_into().expect("slice length checked")),
            shutter_speed: u16::from_be_bytes(data[2..4].try_into().expect("slice length checked")),
            aperture: u16::from_be_bytes(data[4..6].try_into().expect("slice length checked")),
            white_balance: u16::from_be_bytes(data[6..8].try_into().expect("slice length checked")),
        })
    }
}

/// Combined telemetry data point.
#[derive(Debug, Clone, Copy)]
pub struct TelemetryData {
    /// IMU data (optional).
    pub imu: Option<ImuData>,
    /// Exposure data (optional).
    pub exposure: Option<ExposureData>,
    /// Temperature in Celsius.
    pub temperature: f32,
    /// Battery level (0-100%).
    pub battery_level: u8,
}

impl TelemetryData {
    /// Creates a new telemetry data point.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            imu: None,
            exposure: None,
            temperature: 0.0,
            battery_level: 100,
        }
    }

    /// Sets IMU data.
    #[must_use]
    pub const fn with_imu(mut self, imu: ImuData) -> Self {
        self.imu = Some(imu);
        self
    }

    /// Sets exposure data.
    #[must_use]
    pub const fn with_exposure(mut self, exposure: ExposureData) -> Self {
        self.exposure = Some(exposure);
        self
    }

    /// Sets temperature.
    #[must_use]
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Sets battery level.
    #[must_use]
    pub const fn with_battery(mut self, level: u8) -> Self {
        self.battery_level = level;
        self
    }
}

impl Default for TelemetryData {
    fn default() -> Self {
        Self::new()
    }
}

/// Telemetry track containing multiple data points.
#[derive(Debug, Clone)]
pub struct TelemetryTrack {
    points: Vec<(i64, TelemetryData)>, // (timestamp, data)
}

impl TelemetryTrack {
    /// Creates a new telemetry track.
    #[must_use]
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Adds a telemetry data point.
    pub fn add_point(&mut self, timestamp: i64, data: TelemetryData) {
        self.points.push((timestamp, data));
    }

    /// Returns all telemetry points.
    #[must_use]
    pub fn points(&self) -> &[(i64, TelemetryData)] {
        &self.points
    }

    /// Gets the telemetry data at a specific timestamp.
    #[must_use]
    pub fn get_point_at(&self, timestamp: i64) -> Option<&TelemetryData> {
        self.points
            .iter()
            .rev()
            .find(|(ts, _)| *ts <= timestamp)
            .map(|(_, data)| data)
    }

    /// Returns the number of points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns true if there are no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

impl Default for TelemetryTrack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imu_data() {
        let imu = ImuData::new(0.0, 9.8, 0.0, 0.0, 0.0, 0.1).with_magnetometer(20.0, 0.0, -10.0);

        assert_eq!(imu.accel_y, 9.8);
        assert_eq!(imu.gyro_z, 0.1);
        assert_eq!(imu.mag_x, 20.0);
    }

    #[test]
    fn test_imu_serialization() {
        let imu = ImuData::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);

        let bytes = imu.to_bytes();
        let decoded = ImuData::from_bytes(&bytes).expect("operation should succeed");

        assert_eq!(decoded.accel_x, 1.0);
        assert_eq!(decoded.gyro_z, 6.0);
    }

    #[test]
    fn test_exposure_data() {
        let exposure = ExposureData::new(800, 1000, 28, 5600);

        assert_eq!(exposure.iso, 800);
        assert_eq!(exposure.aperture, 28); // f/2.8
        assert_eq!(exposure.white_balance, 5600);
    }

    #[test]
    fn test_exposure_serialization() {
        let exposure = ExposureData::new(800, 1000, 28, 5600);

        let bytes = exposure.to_bytes();
        let decoded = ExposureData::from_bytes(&bytes).expect("operation should succeed");

        assert_eq!(decoded.iso, 800);
        assert_eq!(decoded.white_balance, 5600);
    }

    #[test]
    fn test_telemetry_data() {
        let imu = ImuData::new(0.0, 9.8, 0.0, 0.0, 0.0, 0.1);
        let exposure = ExposureData::new(800, 1000, 28, 5600);

        let telemetry = TelemetryData::new()
            .with_imu(imu)
            .with_exposure(exposure)
            .with_temperature(25.5)
            .with_battery(85);

        assert!(telemetry.imu.is_some());
        assert!(telemetry.exposure.is_some());
        assert_eq!(telemetry.temperature, 25.5);
        assert_eq!(telemetry.battery_level, 85);
    }

    #[test]
    fn test_telemetry_track() {
        let mut track = TelemetryTrack::new();

        let data1 = TelemetryData::new().with_temperature(20.0);
        let data2 = TelemetryData::new().with_temperature(25.0);

        track.add_point(0, data1);
        track.add_point(1000, data2);

        assert_eq!(track.len(), 2);
        assert!(!track.is_empty());

        let found = track.get_point_at(500);
        assert!(found.is_some());
        assert_eq!(found.expect("operation should succeed").temperature, 20.0);
    }
}
