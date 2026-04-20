//! GPS data track (GoPro-style).
//!
//! Handles GPS/location data in containers.

#![forbid(unsafe_code)]

use bytes::{BufMut, Bytes, BytesMut};
use oximedia_core::{OxiError, OxiResult};

/// GPS coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpsCoordinate {
    /// Latitude in degrees (-90 to 90).
    pub latitude: f64,
    /// Longitude in degrees (-180 to 180).
    pub longitude: f64,
    /// Altitude in meters.
    pub altitude: f64,
}

impl GpsCoordinate {
    /// Creates a new GPS coordinate.
    ///
    /// # Errors
    ///
    /// Returns `Err` if latitude is not in `[-90, 90]` or longitude is not in `[-180, 180]`.
    pub fn new(latitude: f64, longitude: f64, altitude: f64) -> OxiResult<Self> {
        if !(-90.0..=90.0).contains(&latitude) {
            return Err(OxiError::InvalidData(
                "Latitude must be between -90 and 90".into(),
            ));
        }
        if !(-180.0..=180.0).contains(&longitude) {
            return Err(OxiError::InvalidData(
                "Longitude must be between -180 and 180".into(),
            ));
        }

        Ok(Self {
            latitude,
            longitude,
            altitude,
        })
    }
}

/// GPS data point with velocity and accuracy.
#[derive(Debug, Clone, Copy)]
pub struct GpsDataPoint {
    /// GPS coordinate.
    pub coordinate: GpsCoordinate,
    /// Speed in meters per second.
    pub speed: f64,
    /// Heading in degrees (0-360).
    pub heading: f64,
    /// Horizontal accuracy in meters.
    pub horizontal_accuracy: f64,
    /// Vertical accuracy in meters.
    pub vertical_accuracy: f64,
    /// Number of satellites.
    pub satellites: u8,
}

impl GpsDataPoint {
    /// Creates a new GPS data point.
    #[must_use]
    pub const fn new(coordinate: GpsCoordinate) -> Self {
        Self {
            coordinate,
            speed: 0.0,
            heading: 0.0,
            horizontal_accuracy: 0.0,
            vertical_accuracy: 0.0,
            satellites: 0,
        }
    }

    /// Sets the speed.
    #[must_use]
    pub const fn with_speed(mut self, speed: f64) -> Self {
        self.speed = speed;
        self
    }

    /// Sets the heading.
    #[must_use]
    pub const fn with_heading(mut self, heading: f64) -> Self {
        self.heading = heading;
        self
    }

    /// Sets the accuracy.
    #[must_use]
    pub const fn with_accuracy(mut self, horizontal: f64, vertical: f64) -> Self {
        self.horizontal_accuracy = horizontal;
        self.vertical_accuracy = vertical;
        self
    }

    /// Sets the number of satellites.
    #[must_use]
    pub const fn with_satellites(mut self, satellites: u8) -> Self {
        self.satellites = satellites;
        self
    }

    /// Serializes to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(64);

        buf.put_f64(self.coordinate.latitude);
        buf.put_f64(self.coordinate.longitude);
        buf.put_f64(self.coordinate.altitude);
        buf.put_f64(self.speed);
        buf.put_f64(self.heading);
        buf.put_f64(self.horizontal_accuracy);
        buf.put_f64(self.vertical_accuracy);
        buf.put_u8(self.satellites);

        buf.freeze()
    }

    /// Deserializes from bytes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the data is too short, if slice conversion fails, or contains invalid coordinate values.
    pub fn from_bytes(data: &[u8]) -> OxiResult<Self> {
        if data.len() < 57 {
            return Err(OxiError::InvalidData("GPS data too short".into()));
        }

        let conv = |s: &[u8]| -> OxiResult<[u8; 8]> {
            s.try_into()
                .map_err(|_| OxiError::InvalidData("GPS slice conversion failed".into()))
        };

        let latitude = f64::from_be_bytes(conv(&data[0..8])?);
        let longitude = f64::from_be_bytes(conv(&data[8..16])?);
        let altitude = f64::from_be_bytes(conv(&data[16..24])?);
        let speed = f64::from_be_bytes(conv(&data[24..32])?);
        let heading = f64::from_be_bytes(conv(&data[32..40])?);
        let horizontal_accuracy = f64::from_be_bytes(conv(&data[40..48])?);
        let vertical_accuracy = f64::from_be_bytes(conv(&data[48..56])?);
        let satellites = data[56];

        let coordinate = GpsCoordinate::new(latitude, longitude, altitude)?;

        Ok(Self {
            coordinate,
            speed,
            heading,
            horizontal_accuracy,
            vertical_accuracy,
            satellites,
        })
    }
}

/// GPS track containing multiple GPS data points.
#[derive(Debug, Clone)]
pub struct GpsTrack {
    points: Vec<(i64, GpsDataPoint)>, // (timestamp, data)
}

impl GpsTrack {
    /// Creates a new GPS track.
    #[must_use]
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Adds a GPS data point.
    pub fn add_point(&mut self, timestamp: i64, point: GpsDataPoint) {
        self.points.push((timestamp, point));
    }

    /// Returns all GPS points.
    #[must_use]
    pub fn points(&self) -> &[(i64, GpsDataPoint)] {
        &self.points
    }

    /// Gets the GPS point at a specific timestamp.
    #[must_use]
    pub fn get_point_at(&self, timestamp: i64) -> Option<&GpsDataPoint> {
        self.points
            .iter()
            .rev()
            .find(|(ts, _)| *ts <= timestamp)
            .map(|(_, point)| point)
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

    /// Calculates the total distance traveled.
    #[must_use]
    pub fn total_distance(&self) -> f64 {
        let mut distance = 0.0;

        for window in self.points.windows(2) {
            let (_, p1) = &window[0];
            let (_, p2) = &window[1];
            distance += haversine_distance(
                p1.coordinate.latitude,
                p1.coordinate.longitude,
                p2.coordinate.latitude,
                p2.coordinate.longitude,
            );
        }

        distance
    }
}

impl Default for GpsTrack {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculates the haversine distance between two GPS coordinates.
#[must_use]
#[allow(clippy::suboptimal_flops)]
fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS_M: f64 = 6_371_000.0;

    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let delta_lat = (lat2 - lat1).to_radians();
    let delta_lon = (lon2 - lon1).to_radians();

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_RADIUS_M * c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gps_coordinate() {
        let coord = GpsCoordinate::new(37.7749, -122.4194, 10.0).expect("operation should succeed");
        assert_eq!(coord.latitude, 37.7749);
        assert_eq!(coord.longitude, -122.4194);
        assert_eq!(coord.altitude, 10.0);

        // Invalid latitude
        assert!(GpsCoordinate::new(100.0, 0.0, 0.0).is_err());
        assert!(GpsCoordinate::new(0.0, 200.0, 0.0).is_err());
    }

    #[test]
    fn test_gps_data_point() {
        let coord = GpsCoordinate::new(37.7749, -122.4194, 10.0).expect("operation should succeed");
        let point = GpsDataPoint::new(coord)
            .with_speed(5.0)
            .with_heading(90.0)
            .with_accuracy(10.0, 5.0)
            .with_satellites(8);

        assert_eq!(point.speed, 5.0);
        assert_eq!(point.heading, 90.0);
        assert_eq!(point.satellites, 8);
    }

    #[test]
    fn test_gps_serialization() {
        let coord = GpsCoordinate::new(37.7749, -122.4194, 10.0).expect("operation should succeed");
        let point = GpsDataPoint::new(coord).with_speed(5.0);

        let bytes = point.to_bytes();
        let decoded = GpsDataPoint::from_bytes(&bytes).expect("operation should succeed");

        assert!((decoded.coordinate.latitude - 37.7749).abs() < 0.0001);
        assert!((decoded.speed - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_gps_track() {
        let mut track = GpsTrack::new();

        let coord1 =
            GpsCoordinate::new(37.7749, -122.4194, 10.0).expect("operation should succeed");
        let point1 = GpsDataPoint::new(coord1);
        track.add_point(0, point1);

        let coord2 =
            GpsCoordinate::new(37.7750, -122.4195, 11.0).expect("operation should succeed");
        let point2 = GpsDataPoint::new(coord2);
        track.add_point(1000, point2);

        assert_eq!(track.len(), 2);
        assert!(!track.is_empty());

        let found = track.get_point_at(500);
        assert!(found.is_some());

        let distance = track.total_distance();
        assert!(distance > 0.0);
    }

    #[test]
    fn test_haversine_distance() {
        // Distance between San Francisco and Los Angeles (approximately 559 km)
        let distance = haversine_distance(37.7749, -122.4194, 34.0522, -118.2437);
        assert!(distance > 500_000.0 && distance < 600_000.0);
    }
}
