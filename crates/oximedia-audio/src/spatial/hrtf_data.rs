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

/// SOFA (Spatially Oriented Format for Acoustics) file format support.
///
/// Supports the OxiMedia Simplified SOFA text format (`.sofa.txt` or `.sofa.csv`).
/// Real SOFA/HDF5 binary files are not supported without the `hdf5` feature.
pub struct SofaLoader {
    _phantom: std::marker::PhantomData<()>,
}

impl SofaLoader {
    /// Load an HRTF database from a file.
    ///
    /// Accepts the **OxiMedia Simplified SOFA text format** (`.sofa.txt` or `.sofa.csv`).
    ///
    /// # Simplified SOFA Text Format
    ///
    /// ```text
    /// # OxiMedia Simplified SOFA Format v1
    /// # Comments start with #
    /// DATABASE_NAME  MIT KEMAR
    /// SAMPLE_RATE    44100
    /// HRIR_LENGTH    128
    /// # azimuth_deg elevation_deg left_ir_samples... right_ir_samples...
    /// 0.0 0.0 0.001 0.002 ... 0.001 0.002 ...
    /// 45.0 0.0 ...
    /// ```
    ///
    /// Each measurement line contains: `azimuth elevation [HRIR_LENGTH left samples] [HRIR_LENGTH right samples]`
    ///
    /// # Errors
    ///
    /// - Returns [`AudioError::UnsupportedFormat`] if the path has a `.sofa` extension (real HDF5 binary).
    /// - Returns [`AudioError::Io`] if the file cannot be opened.
    /// - Returns [`AudioError::InvalidData`] if required headers are missing or a measurement line
    ///   has the wrong number of values.
    pub fn load_from_file(path: &str) -> AudioResult<HrtfDatabase> {
        use std::io::{BufRead, BufReader};

        // Dispatch on extension: reject real HDF5 .sofa files early
        let p = std::path::Path::new(path);
        let ext = p
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ext == "sofa" {
            return Err(AudioError::UnsupportedFormat(
                "Real SOFA/HDF5 format requires the 'hdf5' feature which is not currently \
                 enabled. Use the .sofa.txt simplified format instead."
                    .to_string(),
            ));
        }

        let file =
            std::fs::File::open(path).map_err(|e| AudioError::Io(format!("{}: {}", path, e)))?;
        let reader = BufReader::new(file);

        let mut db_name = String::from("SOFA Database");
        let mut sample_rate: u32 = 44100;
        let mut hrir_length: usize = 0;
        let mut db: Option<HrtfDatabase> = None;

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result.map_err(|e| AudioError::Io(e.to_string()))?;
            let trimmed = line.trim();

            // Skip blank lines and comments
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Try to parse as a key-value header first (no leading digit or sign)
            let first_char = trimmed.chars().next().unwrap_or(' ');
            if first_char.is_alphabetic() {
                // Parse header key-value pair
                let mut parts = trimmed.splitn(2, |c: char| c.is_whitespace());
                let key = parts.next().unwrap_or("").trim();
                let value = parts.next().unwrap_or("").trim();
                match key {
                    "DATABASE_NAME" => db_name = value.to_string(),
                    "SAMPLE_RATE" => {
                        sample_rate = value.parse::<u32>().map_err(|_| {
                            AudioError::InvalidData(format!(
                                "line {}: invalid SAMPLE_RATE value: {}",
                                line_num + 1,
                                value
                            ))
                        })?;
                    }
                    "HRIR_LENGTH" => {
                        hrir_length = value.parse::<usize>().map_err(|_| {
                            AudioError::InvalidData(format!(
                                "line {}: invalid HRIR_LENGTH value: {}",
                                line_num + 1,
                                value
                            ))
                        })?;
                    }
                    "MEASUREMENT_COUNT" | "SOFA_VERSION" => {
                        // Informational — no action needed
                    }
                    _ => {
                        // Unknown header key — skip silently
                    }
                }
                continue;
            }

            // Lazily initialise the database on the first data line
            if db.is_none() {
                if hrir_length == 0 {
                    return Err(AudioError::InvalidData(
                        "HRIR_LENGTH header must appear before measurement data".to_string(),
                    ));
                }
                db = Some(HrtfDatabase::new(db_name.clone(), sample_rate));
            }

            // Parse measurement line: azimuth elevation [hrir_length left] [hrir_length right]
            let tokens: Vec<&str> = trimmed.split_ascii_whitespace().collect();
            let expected = 2 + 2 * hrir_length;
            if tokens.len() != expected {
                return Err(AudioError::InvalidData(format!(
                    "line {}: expected {} tokens (azimuth + elevation + {} left + {} right), got {}",
                    line_num + 1,
                    expected,
                    hrir_length,
                    hrir_length,
                    tokens.len()
                )));
            }

            let azimuth_deg = tokens[0].parse::<f32>().map_err(|_| {
                AudioError::InvalidData(format!(
                    "line {}: invalid azimuth value: {}",
                    line_num + 1,
                    tokens[0]
                ))
            })?;
            let elevation_deg = tokens[1].parse::<f32>().map_err(|_| {
                AudioError::InvalidData(format!(
                    "line {}: invalid elevation value: {}",
                    line_num + 1,
                    tokens[1]
                ))
            })?;

            let mut left = Vec::with_capacity(hrir_length);
            let mut right = Vec::with_capacity(hrir_length);

            for i in 0..hrir_length {
                let v = tokens[2 + i].parse::<f32>().map_err(|_| {
                    AudioError::InvalidData(format!(
                        "line {}: invalid left sample at index {}: {}",
                        line_num + 1,
                        i,
                        tokens[2 + i]
                    ))
                })?;
                left.push(v);
            }
            for i in 0..hrir_length {
                let v = tokens[2 + hrir_length + i].parse::<f32>().map_err(|_| {
                    AudioError::InvalidData(format!(
                        "line {}: invalid right sample at index {}: {}",
                        line_num + 1,
                        i,
                        tokens[2 + hrir_length + i]
                    ))
                })?;
                right.push(v);
            }

            let azimuth_rad = azimuth_deg.to_radians();
            let elevation_rad = elevation_deg.to_radians();
            let measurement = HrirMeasurement::new(left, right, azimuth_rad, elevation_rad);

            if let Some(ref mut database) = db {
                database.add_measurement(measurement);
            }
        }

        db.ok_or_else(|| {
            AudioError::InvalidData("No measurement data found in SOFA file".to_string())
        })
    }
}

/// CIPIC HRTF database loader.
///
/// Loads HRTF measurements in the OxiMedia simplified CIPIC text format.
/// The file is resolved via the `OXIMEDIA_CIPIC_DATA_DIR` environment variable
/// (or the current working directory as fallback) at the path:
///
/// ```text
/// {OXIMEDIA_CIPIC_DATA_DIR}/cipic/subject_{id:03}/hrtf.cipic.txt
/// ```
///
/// If the file does not exist the loader falls back to a **synthetic CIPIC-compatible**
/// database that covers the full CIPIC azimuth/elevation grid using the built-in
/// synthetic HRIR generator.  All other I/O and parse errors are propagated as `Err`.
pub struct CipicLoader {
    _phantom: std::marker::PhantomData<()>,
}

/// CIPIC azimuth grid (degrees) — 25 values matching the standard CIPIC positions.
const CIPIC_AZIMUTHS: [f32; 25] = [
    -80.0, -65.0, -55.0, -45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0,
    10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 55.0, 65.0, 80.0,
];

/// CIPIC elevation grid (degrees) — 50 values matching the standard CIPIC positions.
const CIPIC_ELEVATIONS: [f32; 50] = [
    -45.0, -39.375, -33.75, -28.125, -22.5, -16.875, -11.25, -5.625, 0.0, 5.625, 11.25, 16.875,
    22.5, 28.125, 33.75, 39.375, 45.0, 50.625, 56.25, 61.875, 67.5, 73.125, 78.75, 84.375, 90.0,
    95.625, 101.25, 106.875, 112.5, 118.125, 123.75, 129.375, 135.0, 140.625, 146.25, 151.875,
    157.5, 163.125, 168.75, 174.375, 180.0, 185.625, 191.25, 196.875, 202.5, 208.125, 213.75,
    219.375, 225.0, 230.625,
];

impl CipicLoader {
    /// Load a CIPIC HRTF database for the given subject identifier.
    ///
    /// The function first attempts to read the file at:
    /// ```text
    /// $OXIMEDIA_CIPIC_DATA_DIR/cipic/subject_{subject_id:03}/hrtf.cipic.txt
    /// ```
    /// (falling back to the current working directory when the env var is unset).
    ///
    /// If the file is **not found**, a synthetic CIPIC-compatible database covering
    /// all 25×50 standard CIPIC grid positions is generated and returned.
    ///
    /// Parse errors and other I/O problems are propagated as [`Err`].
    ///
    /// # CIPIC Text Format
    ///
    /// ```text
    /// # CIPIC HRTF Subject 1
    /// DATABASE_NAME CIPIC Subject 1
    /// SAMPLE_RATE 44100
    /// HRIR_LENGTH 200
    /// # azimuth_deg elevation_deg left_samples... right_samples...
    /// -80.0 -45.0 0.001 -0.002 ... (200 left) 0.001 -0.002 ... (200 right)
    /// ```
    pub fn load_subject(subject_id: u32) -> AudioResult<HrtfDatabase> {
        use std::io::{BufRead, BufReader};

        // Resolve the base data directory from the environment or CWD
        let base_dir = std::env::var("OXIMEDIA_CIPIC_DATA_DIR").unwrap_or_else(|_| ".".to_string());
        let file_path = format!(
            "{}/cipic/subject_{:03}/hrtf.cipic.txt",
            base_dir, subject_id
        );

        let file = match std::fs::File::open(&file_path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Gracefully fall back to synthetic data
                return Ok(Self::generate_synthetic(subject_id));
            }
            Err(e) => {
                return Err(AudioError::Io(format!("{}: {}", file_path, e)));
            }
        };

        let reader = BufReader::new(file);

        let mut db_name = format!("CIPIC Subject {}", subject_id);
        let mut sample_rate: u32 = 44100;
        let mut hrir_length: usize = 0;
        let mut db: Option<HrtfDatabase> = None;

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result.map_err(|e| AudioError::Io(e.to_string()))?;
            let trimmed = line.trim();

            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let first_char = trimmed.chars().next().unwrap_or(' ');
            if first_char.is_alphabetic() {
                let mut parts = trimmed.splitn(2, |c: char| c.is_whitespace());
                let key = parts.next().unwrap_or("").trim();
                let value = parts.next().unwrap_or("").trim();
                match key {
                    "DATABASE_NAME" => db_name = value.to_string(),
                    "SAMPLE_RATE" => {
                        sample_rate = value.parse::<u32>().map_err(|_| {
                            AudioError::InvalidData(format!(
                                "line {}: invalid SAMPLE_RATE value: {}",
                                line_num + 1,
                                value
                            ))
                        })?;
                    }
                    "HRIR_LENGTH" => {
                        hrir_length = value.parse::<usize>().map_err(|_| {
                            AudioError::InvalidData(format!(
                                "line {}: invalid HRIR_LENGTH value: {}",
                                line_num + 1,
                                value
                            ))
                        })?;
                    }
                    _ => {}
                }
                continue;
            }

            if db.is_none() {
                if hrir_length == 0 {
                    return Err(AudioError::InvalidData(
                        "HRIR_LENGTH header must appear before measurement data".to_string(),
                    ));
                }
                db = Some(HrtfDatabase::new(db_name.clone(), sample_rate));
            }

            let tokens: Vec<&str> = trimmed.split_ascii_whitespace().collect();
            let expected = 2 + 2 * hrir_length;
            if tokens.len() != expected {
                return Err(AudioError::InvalidData(format!(
                    "line {}: expected {} tokens, got {}",
                    line_num + 1,
                    expected,
                    tokens.len()
                )));
            }

            let azimuth_deg = tokens[0].parse::<f32>().map_err(|_| {
                AudioError::InvalidData(format!(
                    "line {}: invalid azimuth: {}",
                    line_num + 1,
                    tokens[0]
                ))
            })?;
            let elevation_deg = tokens[1].parse::<f32>().map_err(|_| {
                AudioError::InvalidData(format!(
                    "line {}: invalid elevation: {}",
                    line_num + 1,
                    tokens[1]
                ))
            })?;

            let mut left = Vec::with_capacity(hrir_length);
            let mut right = Vec::with_capacity(hrir_length);

            for i in 0..hrir_length {
                let v = tokens[2 + i].parse::<f32>().map_err(|_| {
                    AudioError::InvalidData(format!(
                        "line {}: invalid left sample {}: {}",
                        line_num + 1,
                        i,
                        tokens[2 + i]
                    ))
                })?;
                left.push(v);
            }
            for i in 0..hrir_length {
                let v = tokens[2 + hrir_length + i].parse::<f32>().map_err(|_| {
                    AudioError::InvalidData(format!(
                        "line {}: invalid right sample {}: {}",
                        line_num + 1,
                        i,
                        tokens[2 + hrir_length + i]
                    ))
                })?;
                right.push(v);
            }

            let azimuth_rad = azimuth_deg.to_radians();
            let elevation_rad = elevation_deg.to_radians();
            let measurement = HrirMeasurement::new(left, right, azimuth_rad, elevation_rad);

            if let Some(ref mut database) = db {
                database.add_measurement(measurement);
            }
        }

        db.ok_or_else(|| {
            AudioError::InvalidData("No measurement data found in CIPIC file".to_string())
        })
    }

    /// Generate a synthetic CIPIC-compatible HRTF database for the given subject.
    ///
    /// Covers all positions on the standard CIPIC 25×50 azimuth/elevation grid.
    fn generate_synthetic(subject_id: u32) -> HrtfDatabase {
        let name = format!("CIPIC Subject {} (Synthetic)", subject_id);
        let mut db = HrtfDatabase::new(name, 44100);

        for &az_deg in CIPIC_AZIMUTHS.iter() {
            for &el_deg in CIPIC_ELEVATIONS.iter() {
                let azimuth_rad = az_deg.to_radians();
                let elevation_rad = el_deg.to_radians();
                let (left, right) = generate_synthetic_hrir(azimuth_rad, elevation_rad);
                db.add_measurement(HrirMeasurement::new(
                    left,
                    right,
                    azimuth_rad,
                    elevation_rad,
                ));
            }
        }

        db
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

    // ---------------------------------------------------------------
    // SofaLoader tests
    // ---------------------------------------------------------------

    #[test]
    fn test_sofa_loader_file_not_found() {
        let result = SofaLoader::load_from_file("/nonexistent/path/to/file.sofa.txt");
        assert!(
            result.is_err(),
            "Expected Err for a nonexistent file, got Ok"
        );
    }

    #[test]
    fn test_sofa_loader_real_hdf5_returns_unsupported() {
        // A .sofa extension (without .txt) should be rejected as HDF5
        let result = SofaLoader::load_from_file("/any/path/kemar.sofa");
        match result {
            Err(AudioError::UnsupportedFormat(_)) => {}
            other => panic!("Expected UnsupportedFormat, got {:?}", other),
        }
    }

    #[test]
    fn test_sofa_loader_valid_file() {
        use std::io::Write;

        // HRIR length = 4 for brevity; 3 measurements
        let hrir_length = 4usize;
        let mut dir = std::env::temp_dir();
        dir.push(format!("oximedia_sofa_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let file_path = dir.join("test.sofa.txt");

        {
            let mut f = std::fs::File::create(&file_path).expect("create temp file");
            writeln!(f, "# OxiMedia Simplified SOFA Format v1").expect("write");
            writeln!(f, "DATABASE_NAME  Test DB").expect("write");
            writeln!(f, "SAMPLE_RATE    44100").expect("write");
            writeln!(f, "HRIR_LENGTH    {}", hrir_length).expect("write");

            // Write 3 measurement lines: 2 + 2*hrir_length tokens each
            for az in [0.0f32, 45.0, 90.0] {
                let samples: Vec<String> = (0..hrir_length * 2)
                    .map(|i| format!("{:.4}", 0.001 * (i + 1) as f32))
                    .collect();
                writeln!(f, "{} 0.0 {}", az, samples.join(" ")).expect("write");
            }
        }

        let result = SofaLoader::load_from_file(file_path.to_str().expect("valid path"));
        // Cleanup regardless of result
        let _ = std::fs::remove_dir_all(&dir);

        let db = result.expect("load_from_file should succeed for valid file");
        assert_eq!(db.len(), 3, "Expected 3 measurements");
        assert_eq!(db.sample_rate, 44100);
        assert!(db.name().contains("Test DB"));
    }

    // ---------------------------------------------------------------
    // CipicLoader tests
    // ---------------------------------------------------------------

    #[test]
    fn test_cipic_loader_returns_database() {
        // Subject 1 with no real data on disk → synthetic fallback
        let result = CipicLoader::load_subject(1);
        let db = result.expect("load_subject should always succeed (synthetic fallback)");
        assert!(
            db.name().contains("CIPIC"),
            "Database name should contain 'CIPIC', got: {}",
            db.name()
        );
        assert!(
            db.len() > 0,
            "Expected at least one measurement in synthetic database"
        );
        // CIPIC grid: 25 azimuths × 50 elevations = 1250 measurements
        assert_eq!(
            db.len(),
            CIPIC_AZIMUTHS.len() * CIPIC_ELEVATIONS.len(),
            "Synthetic CIPIC database should cover the full 25×50 grid"
        );
    }

    #[test]
    fn test_cipic_loader_valid_file() {
        use std::io::Write;

        let hrir_length = 4usize;
        let subject_id = 42u32;

        // Build the expected directory structure inside a temp dir
        let mut base = std::env::temp_dir();
        base.push(format!("oximedia_cipic_test_{}", std::process::id()));
        let subject_dir = base.join(format!("cipic/subject_{:03}", subject_id));
        std::fs::create_dir_all(&subject_dir).expect("create temp dirs");
        let file_path = subject_dir.join("hrtf.cipic.txt");

        {
            let mut f = std::fs::File::create(&file_path).expect("create temp file");
            writeln!(f, "# CIPIC HRTF Subject {}", subject_id).expect("write");
            writeln!(f, "DATABASE_NAME CIPIC Subject {}", subject_id).expect("write");
            writeln!(f, "SAMPLE_RATE 44100").expect("write");
            writeln!(f, "HRIR_LENGTH {}", hrir_length).expect("write");

            // 2 measurements
            for az in [-80.0f32, 0.0] {
                let samples: Vec<String> = (0..hrir_length * 2)
                    .map(|i| format!("{:.5}", 0.0005 * (i + 1) as f32))
                    .collect();
                writeln!(f, "{} -45.0 {}", az, samples.join(" ")).expect("write");
            }
        }

        // Point the loader at our temp directory via the env var
        std::env::set_var(
            "OXIMEDIA_CIPIC_DATA_DIR",
            base.to_str().expect("valid path"),
        );
        let result = CipicLoader::load_subject(subject_id);
        // Always remove env var and temp dir, regardless of result
        std::env::remove_var("OXIMEDIA_CIPIC_DATA_DIR");
        let _ = std::fs::remove_dir_all(&base);

        let db = result.expect("load_subject should succeed for valid CIPIC file");
        assert_eq!(db.len(), 2, "Expected 2 measurements from file");
        assert_eq!(db.sample_rate, 44100);
        assert!(db.name().contains("CIPIC"));
    }
}
