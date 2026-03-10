//! Generic data track support.
//!
//! Handles arbitrary data tracks in containers.

#![forbid(unsafe_code)]

use bytes::Bytes;
use oximedia_core::{OxiError, OxiResult};

/// Type of data track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataTrackType {
    /// GPS/location data.
    Gps,
    /// Telemetry data (IMU, sensors).
    Telemetry,
    /// Custom binary data.
    Binary,
    /// JSON metadata.
    Json,
    /// XML metadata.
    Xml,
}

impl DataTrackType {
    /// Returns the MIME type for this data track type.
    #[must_use]
    pub const fn mime_type(&self) -> &'static str {
        match self {
            Self::Gps => "application/x-gps",
            Self::Telemetry => "application/x-telemetry",
            Self::Binary => "application/octet-stream",
            Self::Json => "application/json",
            Self::Xml => "application/xml",
        }
    }
}

/// A data sample in a data track.
#[derive(Debug, Clone)]
pub struct DataSample {
    /// Timestamp in track timebase.
    pub timestamp: i64,
    /// Sample data.
    pub data: Bytes,
    /// Duration (optional).
    pub duration: Option<i64>,
}

impl DataSample {
    /// Creates a new data sample.
    #[must_use]
    pub const fn new(timestamp: i64, data: Bytes) -> Self {
        Self {
            timestamp,
            data,
            duration: None,
        }
    }

    /// Sets the duration.
    #[must_use]
    pub const fn with_duration(mut self, duration: i64) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Returns the size of the data in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

/// A data track in a container.
#[derive(Debug, Clone)]
pub struct DataTrack {
    /// Track type.
    pub track_type: DataTrackType,
    /// Track ID.
    pub track_id: u32,
    /// Samples.
    samples: Vec<DataSample>,
    /// Codec ID (optional).
    pub codec_id: Option<String>,
    /// Codec private data (optional).
    pub codec_private: Option<Bytes>,
}

impl DataTrack {
    /// Creates a new data track.
    #[must_use]
    pub const fn new(track_type: DataTrackType, track_id: u32) -> Self {
        Self {
            track_type,
            track_id,
            samples: Vec::new(),
            codec_id: None,
            codec_private: None,
        }
    }

    /// Sets the codec ID.
    #[must_use]
    pub fn with_codec_id(mut self, codec_id: impl Into<String>) -> Self {
        self.codec_id = Some(codec_id.into());
        self
    }

    /// Sets the codec private data.
    #[must_use]
    pub fn with_codec_private(mut self, data: Bytes) -> Self {
        self.codec_private = Some(data);
        self
    }

    /// Adds a sample to the track.
    pub fn add_sample(&mut self, sample: DataSample) {
        self.samples.push(sample);
    }

    /// Returns all samples.
    #[must_use]
    pub fn samples(&self) -> &[DataSample] {
        &self.samples
    }

    /// Gets a sample at a specific timestamp.
    #[must_use]
    pub fn get_sample_at(&self, timestamp: i64) -> Option<&DataSample> {
        self.samples.iter().rev().find(|s| s.timestamp <= timestamp)
    }

    /// Returns the number of samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Returns the total size of all samples.
    #[must_use]
    pub fn total_size(&self) -> usize {
        self.samples.iter().map(DataSample::size).sum()
    }

    /// Validates the data track.
    ///
    /// # Errors
    ///
    /// Returns `Err` if timestamps are not monotonically increasing.
    pub fn validate(&self) -> OxiResult<()> {
        // Check for monotonically increasing timestamps
        let mut last_timestamp = None;

        for sample in &self.samples {
            if let Some(last) = last_timestamp {
                if sample.timestamp < last {
                    return Err(OxiError::InvalidData(
                        "Timestamps are not monotonically increasing".into(),
                    ));
                }
            }
            last_timestamp = Some(sample.timestamp);
        }

        Ok(())
    }
}

/// Builder for data tracks.
pub struct DataTrackBuilder {
    track: DataTrack,
}

impl DataTrackBuilder {
    /// Creates a new builder.
    #[must_use]
    pub const fn new(track_type: DataTrackType, track_id: u32) -> Self {
        Self {
            track: DataTrack::new(track_type, track_id),
        }
    }

    /// Sets the codec ID.
    pub fn codec_id(&mut self, codec_id: impl Into<String>) -> &mut Self {
        self.track.codec_id = Some(codec_id.into());
        self
    }

    /// Sets the codec private data.
    pub fn codec_private(&mut self, data: Bytes) -> &mut Self {
        self.track.codec_private = Some(data);
        self
    }

    /// Adds a sample.
    pub fn add_sample(&mut self, sample: DataSample) -> &mut Self {
        self.track.add_sample(sample);
        self
    }

    /// Builds the data track.
    #[must_use]
    pub fn build(self) -> DataTrack {
        self.track
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_track_type() {
        assert_eq!(DataTrackType::Gps.mime_type(), "application/x-gps");
        assert_eq!(DataTrackType::Json.mime_type(), "application/json");
    }

    #[test]
    fn test_data_sample() {
        let data = Bytes::from_static(b"test");
        let sample = DataSample::new(1000, data).with_duration(100);

        assert_eq!(sample.timestamp, 1000);
        assert_eq!(sample.duration, Some(100));
        assert_eq!(sample.size(), 4);
    }

    #[test]
    fn test_data_track() {
        let mut track = DataTrack::new(DataTrackType::Gps, 1);

        let sample1 = DataSample::new(0, Bytes::from_static(b"gps1"));
        let sample2 = DataSample::new(1000, Bytes::from_static(b"gps2"));

        track.add_sample(sample1);
        track.add_sample(sample2);

        assert_eq!(track.sample_count(), 2);
        assert_eq!(track.total_size(), 8);

        let found = track.get_sample_at(500);
        assert!(found.is_some());
        assert_eq!(found.expect("operation should succeed").timestamp, 0);

        assert!(track.validate().is_ok());
    }

    #[test]
    fn test_data_track_builder() {
        let mut builder = DataTrackBuilder::new(DataTrackType::Telemetry, 1);
        builder
            .codec_id("telemetry/v1")
            .add_sample(DataSample::new(0, Bytes::from_static(b"data1")))
            .add_sample(DataSample::new(1000, Bytes::from_static(b"data2")));

        let track = builder.build();
        assert_eq!(track.track_id, 1);
        assert_eq!(track.codec_id, Some("telemetry/v1".into()));
        assert_eq!(track.sample_count(), 2);
    }
}
