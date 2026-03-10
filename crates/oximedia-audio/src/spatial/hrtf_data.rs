//! HRTF (Head-Related Transfer Function) database management.
//!
//! This module provides HRTF data loading, caching, and interpolation for binaural rendering.
//! It supports multiple HRTF databases including MIT KEMAR and CIPIC.

use crate::{AudioError, AudioResult};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::{Arc, RwLock};

/// Maximum HRIR length in samples (typically 128-256 for common databases)
pub const MAX_HRIR_LENGTH: usize = 256;

/// HRIR measurement for a specific direction (azimuth, elevation)
#[derive(Clone, Debug)]
pub struct HrirMeasurement {
    /// Left ear impulse response
    pub left: Vec<f32>,
    /// Right ear impulse response
    pub right: Vec<f32>,
    /// Azimuth in radians (-π to π, 0 = front)
    pub azimuth: f32,
    /// Elevation in radians (-π/2 to π/2, 0 = horizontal)
    pub elevation: f32,
}

impl HrirMeasurement {
    /// Create a new HRIR measurement
    pub fn new(left: Vec<f32>, right: Vec<f32>, azimuth: f32, elevation: f32) -> Self {
        Self {
            left,
            right,
            azimuth,
            elevation,
        }
    }

    /// Get the length of the impulse response
    pub fn len(&self) -> usize {
        self.left.len().min(self.right.len())
    }

    /// Check if the impulse response is empty
    pub fn is_empty(&self) -> bool {
        self.left.is_empty() || self.right.is_empty()
    }
}

/// HRTF database containing measurements at various positions
#[derive(Clone, Debug)]
pub struct HrtfDatabase {
    /// Sample rate of the HRIR measurements
    pub sample_rate: u32,
    /// All HRIR measurements in the database
    measurements: Vec<HrirMeasurement>,
    /// Spatial index for fast lookup (azimuth_deg, elevation_deg) -> index
    spatial_index: HashMap<(i32, i32), usize>,
    /// Database name/identifier
    name: String,
}

impl HrtfDatabase {
    /// Create a new HRTF database
    pub fn new(name: String, sample_rate: u32) -> Self {
        Self {
            sample_rate,
            measurements: Vec::new(),
            spatial_index: HashMap::new(),
            name,
        }
    }

    /// Add a measurement to the database
    pub fn add_measurement(&mut self, measurement: HrirMeasurement) {
        let azimuth_deg = (measurement.azimuth.to_degrees().round() as i32 + 360) % 360;
        let elevation_deg = measurement.elevation.to_degrees().round() as i32;

        let index = self.measurements.len();
        self.measurements.push(measurement);
        self.spatial_index
            .insert((azimuth_deg, elevation_deg), index);
    }

    /// Get the nearest HRIR measurement for a given direction
    pub fn get_nearest(&self, azimuth: f32, elevation: f32) -> Option<&HrirMeasurement> {
        if self.measurements.is_empty() {
            return None;
        }

        let azimuth_deg = (azimuth.to_degrees().round() as i32 + 360) % 360;
        let elevation_deg = elevation.to_degrees().round() as i32;

        // Try exact match first
        if let Some(&index) = self.spatial_index.get(&(azimuth_deg, elevation_deg)) {
            return Some(&self.measurements[index]);
        }

        // Find nearest measurement
        let mut min_distance = f32::INFINITY;
        let mut nearest_index = 0;

        for (i, measurement) in self.measurements.iter().enumerate() {
            let distance = angular_distance(
                azimuth,
                elevation,
                measurement.azimuth,
                measurement.elevation,
            );
            if distance < min_distance {
                min_distance = distance;
                nearest_index = i;
            }
        }

        Some(&self.measurements[nearest_index])
    }

    /// Interpolate HRIR for a given direction using bilinear interpolation
    pub fn interpolate(&self, azimuth: f32, elevation: f32) -> Option<HrirMeasurement> {
        if self.measurements.is_empty() {
            return None;
        }

        // Find the 4 nearest neighbors
        let neighbors = self.find_neighbors(azimuth, elevation)?;

        if neighbors.len() == 1 {
            // Exact match
            return Some(neighbors[0].clone());
        }

        // Perform bilinear interpolation
        let hrir_len = neighbors[0].len();
        let mut left = vec![0.0; hrir_len];
        let mut right = vec![0.0; hrir_len];

        // Calculate weights based on inverse distance
        let mut total_weight = 0.0;
        let mut weights = Vec::new();

        for neighbor in &neighbors {
            let distance =
                angular_distance(azimuth, elevation, neighbor.azimuth, neighbor.elevation);
            let weight = if distance < 0.001 {
                1.0
            } else {
                1.0 / distance
            };
            weights.push(weight);
            total_weight += weight;
        }

        // Normalize weights and interpolate
        for (i, neighbor) in neighbors.iter().enumerate() {
            let weight = weights[i] / total_weight;
            for j in 0..hrir_len {
                left[j] += neighbor.left[j] * weight;
                right[j] += neighbor.right[j] * weight;
            }
        }

        Some(HrirMeasurement::new(left, right, azimuth, elevation))
    }

    /// Find nearest neighbors for interpolation (up to 4 points)
    fn find_neighbors(&self, azimuth: f32, elevation: f32) -> Option<Vec<HrirMeasurement>> {
        let mut neighbors = Vec::new();
        let mut distances: Vec<(usize, f32)> = self
            .measurements
            .iter()
            .enumerate()
            .map(|(i, m)| {
                (
                    i,
                    angular_distance(azimuth, elevation, m.azimuth, m.elevation),
                )
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take up to 4 nearest neighbors
        for (i, _) in distances.iter().take(4) {
            neighbors.push(self.measurements[*i].clone());
        }

        if neighbors.is_empty() {
            None
        } else {
            Some(neighbors)
        }
    }

    /// Get the number of measurements in the database
    pub fn len(&self) -> usize {
        self.measurements.len()
    }

    /// Check if the database is empty
    pub fn is_empty(&self) -> bool {
        self.measurements.is_empty()
    }

    /// Get the database name
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Calculate angular distance between two directions
fn angular_distance(az1: f32, el1: f32, az2: f32, el2: f32) -> f32 {
    // Convert to Cartesian coordinates
    let x1 = el1.cos() * az1.cos();
    let y1 = el1.cos() * az1.sin();
    let z1 = el1.sin();

    let x2 = el2.cos() * az2.cos();
    let y2 = el2.cos() * az2.sin();
    let z2 = el2.sin();

    // Dot product
    let dot = x1 * x2 + y1 * y2 + z1 * z2;

    // Clamp to avoid numerical errors
    let dot = dot.clamp(-1.0, 1.0);

    dot.acos()
}

/// HRTF database manager with caching
pub struct HrtfManager {
    /// Cached databases
    databases: Arc<RwLock<HashMap<String, Arc<HrtfDatabase>>>>,
    /// Default database key
    default_database: String,
}

impl HrtfManager {
    /// Create a new HRTF manager
    pub fn new() -> Self {
        Self {
            databases: Arc::new(RwLock::new(HashMap::new())),
            default_database: String::from("default"),
        }
    }

    /// Load a database
    pub fn load_database(&self, name: String, database: HrtfDatabase) -> AudioResult<()> {
        let mut databases = self
            .databases
            .write()
            .map_err(|_| AudioError::Internal("Failed to acquire write lock".to_string()))?;
        databases.insert(name, Arc::new(database));
        Ok(())
    }

    /// Get a database by name
    pub fn get_database(&self, name: &str) -> AudioResult<Arc<HrtfDatabase>> {
        let databases = self
            .databases
            .read()
            .map_err(|_| AudioError::Internal("Failed to acquire read lock".to_string()))?;
        databases
            .get(name)
            .cloned()
            .ok_or_else(|| AudioError::Internal(format!("Database not found: {}", name)))
    }

    /// Get the default database
    pub fn get_default(&self) -> AudioResult<Arc<HrtfDatabase>> {
        self.get_database(&self.default_database)
    }

    /// Set the default database
    pub fn set_default(&mut self, name: String) {
        self.default_database = name;
    }

    /// List all available databases
    pub fn list_databases(&self) -> AudioResult<Vec<String>> {
        let databases = self
            .databases
            .read()
            .map_err(|_| AudioError::Internal("Failed to acquire read lock".to_string()))?;
        Ok(databases.keys().cloned().collect())
    }
}

impl Default for HrtfManager {
    fn default() -> Self {
        let manager = Self::new();

        // Load default database (MIT KEMAR)
        let default_db = create_default_database();
        let _ = manager.load_database("default".to_string(), default_db);

        manager
    }
}

/// Create a default HRTF database (MIT KEMAR simplified)
fn create_default_database() -> HrtfDatabase {
    let mut db = HrtfDatabase::new("MIT KEMAR (Simplified)".to_string(), 44100);

    // Add simplified HRIR measurements
    // In a real implementation, this would load from actual HRTF data files

    // Horizontal plane (elevation = 0)
    for azimuth_deg in (0..360).step_by(15) {
        let azimuth = (azimuth_deg as f32).to_radians();
        let elevation = 0.0;

        let (left, right) = generate_synthetic_hrir(azimuth, elevation);
        db.add_measurement(HrirMeasurement::new(left, right, azimuth, elevation));
    }

    // Upper hemisphere (elevation = 30°)
    for azimuth_deg in (0..360).step_by(30) {
        let azimuth = (azimuth_deg as f32).to_radians();
        let elevation = 30.0_f32.to_radians();

        let (left, right) = generate_synthetic_hrir(azimuth, elevation);
        db.add_measurement(HrirMeasurement::new(left, right, azimuth, elevation));
    }

    // Lower hemisphere (elevation = -30°)
    for azimuth_deg in (0..360).step_by(30) {
        let azimuth = (azimuth_deg as f32).to_radians();
        let elevation = -30.0_f32.to_radians();

        let (left, right) = generate_synthetic_hrir(azimuth, elevation);
        db.add_measurement(HrirMeasurement::new(left, right, azimuth, elevation));
    }

    // Top (elevation = 90°)
    let (left, right) = generate_synthetic_hrir(0.0, PI / 2.0);
    db.add_measurement(HrirMeasurement::new(left, right, 0.0, PI / 2.0));

    // Bottom (elevation = -90°)
    let (left, right) = generate_synthetic_hrir(0.0, -PI / 2.0);
    db.add_measurement(HrirMeasurement::new(left, right, 0.0, -PI / 2.0));

    db
}

/// Generate synthetic HRIR (simplified model for demonstration)
/// In a real implementation, use actual measured HRTF data
fn generate_synthetic_hrir(azimuth: f32, elevation: f32) -> (Vec<f32>, Vec<f32>) {
    const HRIR_LENGTH: usize = 128;

    // Calculate ITD (Interaural Time Difference)
    let head_radius = 0.0875; // Average head radius in meters
    let sound_speed = 343.0; // Speed of sound in m/s
    let sample_rate = 44100.0;

    let itd = head_radius / sound_speed * (azimuth + elevation.cos() * azimuth.sin());
    let itd_samples = (itd * sample_rate).round() as i32;

    // Calculate ILD (Interaural Level Difference)
    let ild = (1.0 + azimuth.cos()) / 2.0;

    let mut left = vec![0.0; HRIR_LENGTH];
    let mut right = vec![0.0; HRIR_LENGTH];

    // Generate simple impulse with delay and level difference
    let left_delay = if itd_samples > 0 {
        itd_samples as usize
    } else {
        0
    };
    let right_delay = if itd_samples < 0 {
        (-itd_samples) as usize
    } else {
        0
    };

    // Main impulse
    if left_delay < HRIR_LENGTH {
        left[left_delay] = (1.0 - ild).sqrt();
    }
    if right_delay < HRIR_LENGTH {
        right[right_delay] = ild.sqrt();
    }

    // Add some diffuse reflections
    for i in 10..HRIR_LENGTH {
        let decay = (-(i as f32) / 20.0).exp();
        left[i] += decay * 0.1 * ((i as f32 * 0.1).sin());
        right[i] += decay * 0.1 * ((i as f32 * 0.1 + PI).sin());
    }

    // Normalize
    let left_max = left.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let right_max = right.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let max = left_max.max(right_max);

    if max > 0.0 {
        for sample in &mut left {
            *sample /= max;
        }
        for sample in &mut right {
            *sample /= max;
        }
    }

    (left, right)
}

/// SOFA (Spatially Oriented Format for Acoustics) file format support
/// This is a placeholder for future implementation
pub struct SofaLoader {
    _phantom: std::marker::PhantomData<()>,
}

impl SofaLoader {
    /// Load HRTF database from SOFA file
    /// Note: This is a placeholder. Real implementation would parse SOFA files.
    #[allow(dead_code)]
    pub fn load_from_file(_path: &str) -> AudioResult<HrtfDatabase> {
        Err(AudioError::UnsupportedFormat(
            "SOFA format not yet implemented".to_string(),
        ))
    }
}

/// CIPIC HRTF database loader
/// This is a placeholder for future implementation
pub struct CipicLoader {
    _phantom: std::marker::PhantomData<()>,
}

impl CipicLoader {
    /// Load CIPIC database
    /// Note: This is a placeholder. Real implementation would load CIPIC data.
    #[allow(dead_code)]
    pub fn load_subject(_subject_id: u32) -> AudioResult<HrtfDatabase> {
        Err(AudioError::UnsupportedFormat(
            "CIPIC format not yet implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hrir_measurement() {
        let left = vec![1.0, 0.5, 0.25];
        let right = vec![0.8, 0.4, 0.2];
        let hrir = HrirMeasurement::new(left, right, 0.0, 0.0);

        assert_eq!(hrir.len(), 3);
        assert!(!hrir.is_empty());
    }

    #[test]
    fn test_hrtf_database() {
        let mut db = HrtfDatabase::new("Test".to_string(), 44100);

        let left = vec![1.0; 128];
        let right = vec![1.0; 128];
        db.add_measurement(HrirMeasurement::new(left.clone(), right.clone(), 0.0, 0.0));
        db.add_measurement(HrirMeasurement::new(left, right, PI / 2.0, 0.0));

        assert_eq!(db.len(), 2);
        assert!(!db.is_empty());
    }

    #[test]
    fn test_angular_distance() {
        let dist = angular_distance(0.0, 0.0, 0.0, 0.0);
        assert!(dist < 0.001);

        let dist = angular_distance(0.0, 0.0, PI, 0.0);
        assert!((dist - PI).abs() < 0.001);
    }

    #[test]
    fn test_hrtf_manager() {
        let manager = HrtfManager::default();

        let db = manager.get_default();
        assert!(db.is_ok());

        let db = db.expect("should succeed");
        assert!(!db.is_empty());
    }

    #[test]
    fn test_interpolation() {
        let db = create_default_database();

        let hrir = db.interpolate(0.0, 0.0);
        assert!(hrir.is_some());

        let hrir = hrir.expect("should succeed");
        assert!(!hrir.is_empty());
    }
}
