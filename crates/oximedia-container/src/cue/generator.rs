//! Cue point generation for seeking.
//!
//! Generates optimal cue points for seeking in containers.

#![forbid(unsafe_code)]

use oximedia_core::{OxiError, OxiResult};

/// A cue point for seeking.
#[derive(Debug, Clone, Copy)]
pub struct CuePoint {
    /// Timestamp in the track's timebase.
    pub timestamp: i64,
    /// File position in bytes.
    pub file_position: u64,
    /// Cluster/segment position (for Matroska).
    pub cluster_position: Option<u64>,
    /// Track number.
    pub track_number: u16,
}

impl CuePoint {
    /// Creates a new cue point.
    #[must_use]
    pub const fn new(timestamp: i64, file_position: u64, track_number: u16) -> Self {
        Self {
            timestamp,
            file_position,
            cluster_position: None,
            track_number,
        }
    }

    /// Sets the cluster position.
    #[must_use]
    pub const fn with_cluster_position(mut self, position: u64) -> Self {
        self.cluster_position = Some(position);
        self
    }
}

/// Configuration for cue point generation.
#[derive(Debug, Clone)]
pub struct CueGeneratorConfig {
    /// Minimum interval between cue points in seconds.
    pub min_interval_secs: f64,
    /// Maximum interval between cue points in seconds.
    pub max_interval_secs: f64,
    /// Generate cue points only on keyframes.
    pub keyframes_only: bool,
    /// Tracks to generate cue points for.
    pub tracks: Option<Vec<u16>>,
}

impl Default for CueGeneratorConfig {
    fn default() -> Self {
        Self {
            min_interval_secs: 0.5,
            max_interval_secs: 5.0,
            keyframes_only: true,
            tracks: None,
        }
    }
}

impl CueGeneratorConfig {
    /// Creates a new configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            min_interval_secs: 0.5,
            max_interval_secs: 5.0,
            keyframes_only: true,
            tracks: None,
        }
    }

    /// Sets the minimum interval.
    #[must_use]
    pub const fn with_min_interval(mut self, interval_secs: f64) -> Self {
        self.min_interval_secs = interval_secs;
        self
    }

    /// Sets the maximum interval.
    #[must_use]
    pub const fn with_max_interval(mut self, interval_secs: f64) -> Self {
        self.max_interval_secs = interval_secs;
        self
    }

    /// Sets whether to generate cue points only on keyframes.
    #[must_use]
    pub const fn with_keyframes_only(mut self, enabled: bool) -> Self {
        self.keyframes_only = enabled;
        self
    }

    /// Sets the tracks to generate cue points for.
    #[must_use]
    pub fn with_tracks(mut self, tracks: Vec<u16>) -> Self {
        self.tracks = Some(tracks);
        self
    }
}

/// Generator for creating cue points.
pub struct CueGenerator {
    config: CueGeneratorConfig,
    cue_points: Vec<CuePoint>,
    last_cue_time: Option<f64>,
}

impl CueGenerator {
    /// Creates a new cue point generator.
    #[must_use]
    pub fn new(config: CueGeneratorConfig) -> Self {
        Self {
            config,
            cue_points: Vec::new(),
            last_cue_time: None,
        }
    }

    /// Considers adding a cue point for a frame.
    #[allow(clippy::cast_precision_loss)]
    pub fn consider_frame(
        &mut self,
        timestamp: i64,
        file_position: u64,
        track_number: u16,
        is_keyframe: bool,
        timebase_den: u32,
    ) {
        // Check if this track should have cue points
        if let Some(ref tracks) = self.config.tracks {
            if !tracks.contains(&track_number) {
                return;
            }
        }

        // Check if keyframe-only is enabled
        if self.config.keyframes_only && !is_keyframe {
            return;
        }

        // Calculate time in seconds
        let time_secs = timestamp as f64 / f64::from(timebase_den);

        // Check interval
        if let Some(last_time) = self.last_cue_time {
            let interval = time_secs - last_time;
            if interval < self.config.min_interval_secs {
                return;
            }
        }

        // Add cue point
        let cue = CuePoint::new(timestamp, file_position, track_number);
        self.cue_points.push(cue);
        self.last_cue_time = Some(time_secs);
    }

    /// Returns all generated cue points.
    #[must_use]
    pub fn cue_points(&self) -> &[CuePoint] {
        &self.cue_points
    }

    /// Clears all cue points.
    pub fn clear(&mut self) {
        self.cue_points.clear();
        self.last_cue_time = None;
    }

    /// Validates the cue points.
    ///
    /// # Errors
    ///
    /// Returns `Err` if timestamps are not monotonically increasing per track.
    pub fn validate(&self) -> OxiResult<()> {
        // Check for monotonically increasing timestamps per track
        let mut track_times: std::collections::HashMap<u16, i64> = std::collections::HashMap::new();

        for cue in &self.cue_points {
            if let Some(&last_time) = track_times.get(&cue.track_number) {
                if cue.timestamp <= last_time {
                    return Err(OxiError::InvalidData(
                        "Cue point timestamps are not monotonically increasing".into(),
                    ));
                }
            }
            track_times.insert(cue.track_number, cue.timestamp);
        }

        Ok(())
    }
}

/// Helper for finding the nearest cue point.
pub struct CueSeeker;

impl CueSeeker {
    /// Finds the cue point closest to a target timestamp.
    #[must_use]
    pub fn find_nearest(cue_points: &[CuePoint], timestamp: i64, track: u16) -> Option<&CuePoint> {
        cue_points
            .iter()
            .filter(|c| c.track_number == track && c.timestamp <= timestamp)
            .max_by_key(|c| c.timestamp)
    }

    /// Finds all cue points before a timestamp.
    #[must_use]
    pub fn find_before(cue_points: &[CuePoint], timestamp: i64, track: u16) -> Vec<&CuePoint> {
        cue_points
            .iter()
            .filter(|c| c.track_number == track && c.timestamp < timestamp)
            .collect()
    }

    /// Finds all cue points in a time range.
    #[must_use]
    pub fn find_in_range(
        cue_points: &[CuePoint],
        start_ts: i64,
        end_ts: i64,
        track: u16,
    ) -> Vec<&CuePoint> {
        cue_points
            .iter()
            .filter(|c| c.track_number == track && c.timestamp >= start_ts && c.timestamp <= end_ts)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cue_point() {
        let cue = CuePoint::new(1000, 12345, 1).with_cluster_position(10000);

        assert_eq!(cue.timestamp, 1000);
        assert_eq!(cue.file_position, 12345);
        assert_eq!(cue.track_number, 1);
        assert_eq!(cue.cluster_position, Some(10000));
    }

    #[test]
    fn test_cue_generator_config() {
        let config = CueGeneratorConfig::new()
            .with_min_interval(1.0)
            .with_max_interval(10.0)
            .with_keyframes_only(false)
            .with_tracks(vec![1, 2]);

        assert_eq!(config.min_interval_secs, 1.0);
        assert_eq!(config.max_interval_secs, 10.0);
        assert!(!config.keyframes_only);
        assert_eq!(config.tracks, Some(vec![1, 2]));
    }

    #[test]
    fn test_cue_generator() {
        let config = CueGeneratorConfig::new().with_min_interval(1.0);
        let mut generator = CueGenerator::new(config);

        // Add frames
        generator.consider_frame(0, 0, 1, true, 1000);
        generator.consider_frame(500, 5000, 1, false, 1000); // Too soon
        generator.consider_frame(1500, 15000, 1, true, 1000); // Should be added

        assert_eq!(generator.cue_points().len(), 2);
        assert!(generator.validate().is_ok());
    }

    #[test]
    fn test_cue_seeker() {
        let cues = vec![
            CuePoint::new(0, 0, 1),
            CuePoint::new(1000, 10000, 1),
            CuePoint::new(2000, 20000, 1),
        ];

        let nearest = CueSeeker::find_nearest(&cues, 1500, 1);
        assert!(nearest.is_some());
        assert_eq!(nearest.expect("operation should succeed").timestamp, 1000);

        let before = CueSeeker::find_before(&cues, 1500, 1);
        assert_eq!(before.len(), 2);

        let in_range = CueSeeker::find_in_range(&cues, 500, 1500, 1);
        assert_eq!(in_range.len(), 1);
    }
}
