//! ISO recording management for multi-camera production.
//!
//! ISO (Isolated) recording captures each camera angle to a separate file,
//! allowing maximum editorial flexibility in post-production.

#![allow(dead_code)]

/// Quality preset for an ISO recording stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsoQuality {
    /// Proxy quality - low bitrate, suitable for offline editing and review.
    Proxy,
    /// Half-resolution - reduced bitrate and dimensions.
    HalfRes,
    /// Full resolution - broadcast quality.
    FullRes,
    /// Raw sensor output - maximum fidelity, very high bitrate.
    Raw,
}

impl IsoQuality {
    /// Returns a multiplier relative to the base bitrate for this quality level.
    ///
    /// `Proxy` = 0.1×, `HalfRes` = 0.3×, `FullRes` = 1.0×, `Raw` = 5.0×.
    #[must_use]
    pub fn bitrate_multiplier(&self) -> f32 {
        match self {
            Self::Proxy => 0.1,
            Self::HalfRes => 0.3,
            Self::FullRes => 1.0,
            Self::Raw => 5.0,
        }
    }

    /// Returns a short uppercase label for use in file names.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Proxy => "PRXY",
            Self::HalfRes => "HALF",
            Self::FullRes => "FULL",
            Self::Raw => "RAW",
        }
    }
}

/// Configuration for a single ISO stream (one camera angle at one quality level).
#[derive(Debug, Clone)]
pub struct IsoStream {
    /// Camera identifier (maps to a physical or virtual camera).
    pub camera_id: u32,
    /// Quality preset for this stream.
    pub quality: IsoQuality,
    /// Codec name (e.g. `"ProRes422"`, `"H.264"`, `"XDCAM"`).
    pub codec: String,
    /// Number of audio channels to record for this stream.
    pub audio_channels: u8,
}

impl IsoStream {
    /// Creates a new `IsoStream`.
    pub fn new(
        camera_id: u32,
        quality: IsoQuality,
        codec: impl Into<String>,
        audio_channels: u8,
    ) -> Self {
        Self {
            camera_id,
            quality,
            codec: codec.into(),
            audio_channels,
        }
    }
}

/// Manages the configuration and lifecycle of an ISO recording session.
#[derive(Debug, Default)]
pub struct IsoRecorder {
    /// Configured streams (cameras × quality levels).
    streams: Vec<IsoStream>,
    /// Active session ID; `None` when not recording.
    session_id: Option<String>,
    /// Recording start time in milliseconds since epoch.
    start_ms: Option<u64>,
}

impl IsoRecorder {
    /// Creates a new empty `IsoRecorder`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a stream configuration to the recorder.
    ///
    /// If a stream for the same `camera_id` and `quality` already exists, it is replaced.
    pub fn add_stream(&mut self, stream: IsoStream) {
        self.streams
            .retain(|s| !(s.camera_id == stream.camera_id && s.quality == stream.quality));
        self.streams.push(stream);
    }

    /// Removes a stream for the given camera ID and quality.
    pub fn remove_stream(&mut self, camera_id: u32, quality: IsoQuality) {
        self.streams
            .retain(|s| !(s.camera_id == camera_id && s.quality == quality));
    }

    /// Starts recording with the given session ID.
    ///
    /// Returns `false` if already recording.
    pub fn start_recording(&mut self, session_id: &str) -> bool {
        if self.session_id.is_some() {
            return false;
        }
        self.session_id = Some(session_id.to_string());
        self.start_ms = Some(current_time_ms());
        true
    }

    /// Stops the current recording and returns an `IsoSession` summary.
    ///
    /// Returns `None` if not currently recording.
    pub fn stop_recording(&mut self) -> Option<IsoSession> {
        let session_id = self.session_id.take()?;
        let start_ms = self.start_ms.take().unwrap_or(0);
        let end_ms = current_time_ms();

        // Estimate total size: assume base bitrate of 50 Mbps for FullRes
        let base_bitrate_bps: u64 = 50_000_000 / 8; // bytes per second
        let duration_s = (end_ms - start_ms) / 1000;

        let total_size_bytes: u64 = self
            .streams
            .iter()
            .map(|s| (base_bitrate_bps as f32 * s.quality.bitrate_multiplier()) as u64 * duration_s)
            .sum();

        Some(IsoSession {
            session_id,
            streams: self.streams.clone(),
            start_ms,
            end_ms,
            total_size_bytes,
        })
    }

    /// Returns whether the recorder is currently recording.
    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.session_id.is_some()
    }

    /// Returns a reference to the configured streams.
    #[must_use]
    pub fn streams(&self) -> &[IsoStream] {
        &self.streams
    }
}

/// Summary of a completed ISO recording session.
#[derive(Debug, Clone)]
pub struct IsoSession {
    /// Session identifier used to name output files.
    pub session_id: String,
    /// Streams that were recorded in this session.
    pub streams: Vec<IsoStream>,
    /// Recording start time in milliseconds since epoch.
    pub start_ms: u64,
    /// Recording end time in milliseconds since epoch.
    pub end_ms: u64,
    /// Estimated total size of all recorded files in bytes.
    pub total_size_bytes: u64,
}

impl IsoSession {
    /// Returns the duration of the session in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Returns streams sorted by quality (ascending bitrate multiplier).
    #[must_use]
    pub fn streams_by_quality(&self) -> Vec<&IsoStream> {
        let mut sorted: Vec<&IsoStream> = self.streams.iter().collect();
        sorted.sort_by(|a, b| {
            a.quality
                .bitrate_multiplier()
                .partial_cmp(&b.quality.bitrate_multiplier())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }
}

/// Generates standardised ISO file names for recorded streams.
pub struct IsoFileNaming;

impl IsoFileNaming {
    /// Generates a file name for an ISO recording.
    ///
    /// Format: `ISO_CAM<camera_id>_<date>_<session_id>_<quality>.mxf`
    ///
    /// Where:
    /// - `<camera_id>` is zero-padded to two digits.
    /// - `<date>` is in `YYYYMMDD` format derived from the current date (hardcoded for
    ///   deterministic testing; production code would use `chrono`).
    /// - `<session_id>` is upper-cased.
    /// - `<quality>` is the quality label (e.g. `FULL`, `PRXY`).
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_multicam::iso_recording::{IsoFileNaming, IsoQuality};
    /// let name = IsoFileNaming::generate(1, "sess001", &IsoQuality::FullRes);
    /// assert!(name.starts_with("ISO_CAM01_"));
    /// assert!(name.ends_with("_FULL.mxf"));
    /// ```
    #[must_use]
    pub fn generate(camera_id: u32, session_id: &str, quality: &IsoQuality) -> String {
        // Format the current UTC date as YYYYMMDD for the filename.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        // Compute YYYY, MM, DD from Unix timestamp using the proleptic Gregorian
        // calendar (no external crate required).
        let days_since_epoch = now / 86400;
        // Shift epoch from 1970-01-01 to the fictitious day 0 of the algorithm.
        // Using the civil_from_days algorithm (Howard Hinnant, public domain).
        let z = days_since_epoch as i64 + 719468;
        let era = if z >= 0 { z } else { z - 146096 } / 146097;
        let doe = z - era * 146097; // day of era [0, 146096]
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // year of era [0, 399]
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year [0, 365]
        let mp = (5 * doy + 2) / 153; // month prime [0, 11]
        let d = doy - (153 * mp + 2) / 5 + 1; // day [1, 31]
        let m = if mp < 10 { mp + 3 } else { mp - 9 }; // month [1, 12]
        let y = if m <= 2 { y + 1 } else { y };
        let date = format!("{:04}{:02}{:02}", y, m, d);
        format!(
            "ISO_CAM{:02}_{}_{}_{}.mxf",
            camera_id,
            date,
            session_id.to_uppercase(),
            quality.label()
        )
    }
}

/// Returns the current time as milliseconds since Unix epoch.
///
/// Uses `std::time::SystemTime` for a monotonically-advancing clock.
fn current_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iso_quality_bitrate_multiplier() {
        assert!(IsoQuality::Proxy.bitrate_multiplier() < IsoQuality::HalfRes.bitrate_multiplier());
        assert!(
            IsoQuality::HalfRes.bitrate_multiplier() < IsoQuality::FullRes.bitrate_multiplier()
        );
        assert!(IsoQuality::FullRes.bitrate_multiplier() < IsoQuality::Raw.bitrate_multiplier());
    }

    #[test]
    fn test_iso_quality_labels() {
        assert_eq!(IsoQuality::Proxy.label(), "PRXY");
        assert_eq!(IsoQuality::HalfRes.label(), "HALF");
        assert_eq!(IsoQuality::FullRes.label(), "FULL");
        assert_eq!(IsoQuality::Raw.label(), "RAW");
    }

    #[test]
    fn test_iso_stream_creation() {
        let stream = IsoStream::new(0, IsoQuality::FullRes, "ProRes422", 2);
        assert_eq!(stream.camera_id, 0);
        assert_eq!(stream.quality, IsoQuality::FullRes);
        assert_eq!(stream.codec, "ProRes422");
        assert_eq!(stream.audio_channels, 2);
    }

    #[test]
    fn test_iso_recorder_add_stream() {
        let mut recorder = IsoRecorder::new();
        recorder.add_stream(IsoStream::new(0, IsoQuality::FullRes, "ProRes422", 2));
        recorder.add_stream(IsoStream::new(1, IsoQuality::Proxy, "H.264", 2));
        assert_eq!(recorder.streams().len(), 2);
    }

    #[test]
    fn test_iso_recorder_replace_stream() {
        let mut recorder = IsoRecorder::new();
        recorder.add_stream(IsoStream::new(0, IsoQuality::FullRes, "ProRes422", 2));
        recorder.add_stream(IsoStream::new(0, IsoQuality::FullRes, "DNxHD", 4));
        assert_eq!(recorder.streams().len(), 1);
        assert_eq!(recorder.streams()[0].codec, "DNxHD");
    }

    #[test]
    fn test_iso_recorder_start_stop() {
        let mut recorder = IsoRecorder::new();
        recorder.add_stream(IsoStream::new(0, IsoQuality::FullRes, "ProRes422", 2));

        assert!(recorder.start_recording("SESS001"));
        assert!(recorder.is_recording());

        let session = recorder
            .stop_recording()
            .expect("multicam test operation should succeed");
        assert_eq!(session.session_id, "SESS001");
        assert!(!recorder.is_recording());
    }

    #[test]
    fn test_iso_recorder_double_start() {
        let mut recorder = IsoRecorder::new();
        assert!(recorder.start_recording("S1"));
        assert!(!recorder.start_recording("S2")); // should fail
    }

    #[test]
    fn test_iso_session_duration() {
        let session = IsoSession {
            session_id: "S1".to_string(),
            streams: vec![],
            start_ms: 1000,
            end_ms: 5000,
            total_size_bytes: 0,
        };
        assert_eq!(session.duration_ms(), 4000);
    }

    #[test]
    fn test_iso_session_streams_by_quality() {
        let session = IsoSession {
            session_id: "S1".to_string(),
            streams: vec![
                IsoStream::new(0, IsoQuality::Raw, "RAW", 2),
                IsoStream::new(1, IsoQuality::Proxy, "H.264", 2),
                IsoStream::new(2, IsoQuality::FullRes, "ProRes422", 2),
            ],
            start_ms: 0,
            end_ms: 1000,
            total_size_bytes: 0,
        };

        let sorted = session.streams_by_quality();
        assert_eq!(sorted[0].quality, IsoQuality::Proxy);
        assert_eq!(sorted[2].quality, IsoQuality::Raw);
    }

    #[test]
    fn test_file_naming_format() {
        let name = IsoFileNaming::generate(1, "sess001", &IsoQuality::FullRes);
        assert!(name.starts_with("ISO_CAM01_"));
        assert!(name.contains("SESS001"));
        assert!(name.ends_with("_FULL.mxf"));
    }

    #[test]
    fn test_file_naming_proxy() {
        let name = IsoFileNaming::generate(5, "take3", &IsoQuality::Proxy);
        assert!(name.starts_with("ISO_CAM05_"));
        assert!(name.ends_with("_PRXY.mxf"));
    }

    #[test]
    fn test_file_naming_session_uppercase() {
        let name = IsoFileNaming::generate(0, "mySession", &IsoQuality::Raw);
        assert!(name.contains("MYSESSION"));
    }
}
