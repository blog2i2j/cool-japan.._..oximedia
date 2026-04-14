// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Media properties detection.

use std::time::Duration;

/// Properties of a media file.
#[derive(Debug, Clone)]
pub struct MediaProperties {
    /// Container format
    pub format: String,
    /// File size in bytes
    pub file_size: u64,
    /// Duration of the media
    pub duration: Option<Duration>,
    /// Video width in pixels
    pub width: Option<u32>,
    /// Video height in pixels
    pub height: Option<u32>,
    /// Video codec
    pub video_codec: Option<String>,
    /// Audio codec
    pub audio_codec: Option<String>,
    /// Video bitrate in bits per second
    pub video_bitrate: Option<u64>,
    /// Audio bitrate in bits per second
    pub audio_bitrate: Option<u64>,
    /// Frame rate in frames per second
    pub frame_rate: Option<f64>,
    /// Audio sample rate in Hz
    pub audio_sample_rate: Option<u32>,
    /// Number of audio channels
    pub audio_channels: Option<u32>,
}

impl MediaProperties {
    /// Check if the media has video.
    #[must_use]
    pub fn has_video(&self) -> bool {
        self.video_codec.is_some()
    }

    /// Check if the media has audio.
    #[must_use]
    pub fn has_audio(&self) -> bool {
        self.audio_codec.is_some()
    }

    /// Get the aspect ratio.
    #[must_use]
    pub fn aspect_ratio(&self) -> Option<f64> {
        match (self.width, self.height) {
            (Some(w), Some(h)) if h > 0 => Some(f64::from(w) / f64::from(h)),
            _ => None,
        }
    }

    /// Get the total bitrate (video + audio).
    #[must_use]
    pub fn total_bitrate(&self) -> Option<u64> {
        match (self.video_bitrate, self.audio_bitrate) {
            (Some(v), Some(a)) => Some(v + a),
            (Some(v), None) => Some(v),
            (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }

    /// Get the resolution as a string (e.g., "1920x1080").
    #[must_use]
    pub fn resolution_string(&self) -> Option<String> {
        match (self.width, self.height) {
            (Some(w), Some(h)) => Some(format!("{w}x{h}")),
            _ => None,
        }
    }

    /// Get the standard resolution name (e.g., "1080p", "4K").
    #[must_use]
    pub fn resolution_name(&self) -> Option<&'static str> {
        match (self.width, self.height) {
            (Some(3840), Some(2160)) => Some("4K"),
            (Some(2560), Some(1440)) => Some("1440p"),
            (Some(1920), Some(1080)) => Some("1080p"),
            (Some(1280), Some(720)) => Some("720p"),
            (Some(854), Some(480)) => Some("480p"),
            (Some(640), Some(360)) => Some("360p"),
            _ => None,
        }
    }

    /// Check if the video is high definition (720p or higher).
    #[must_use]
    pub fn is_hd(&self) -> bool {
        match self.height {
            Some(h) => h >= 720,
            None => false,
        }
    }

    /// Check if the video is 4K or higher.
    #[must_use]
    pub fn is_4k(&self) -> bool {
        match (self.width, self.height) {
            (Some(w), Some(h)) => w >= 3840 && h >= 2160,
            _ => false,
        }
    }

    /// Get the duration in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> Option<f64> {
        self.duration.map(|d| d.as_secs_f64())
    }

    /// Format the duration as HH:MM:SS.
    #[must_use]
    pub fn duration_formatted(&self) -> Option<String> {
        self.duration.map(|d| {
            let total_secs = d.as_secs();
            let hours = total_secs / 3600;
            let minutes = (total_secs % 3600) / 60;
            let seconds = total_secs % 60;

            if hours > 0 {
                format!("{hours:02}:{minutes:02}:{seconds:02}")
            } else {
                format!("{minutes:02}:{seconds:02}")
            }
        })
    }

    /// Get the file size in a human-readable format.
    #[must_use]
    pub fn file_size_formatted(&self) -> String {
        format_bytes(self.file_size)
    }

    /// Calculate the average bitrate from file size and duration.
    #[must_use]
    pub fn calculated_bitrate(&self) -> Option<u64> {
        self.duration.map(|d| {
            let duration_secs = d.as_secs_f64();
            if duration_secs > 0.0 {
                ((self.file_size as f64 * 8.0) / duration_secs) as u64
            } else {
                0
            }
        })
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_index])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_properties() -> MediaProperties {
        MediaProperties {
            format: "mp4".to_string(),
            file_size: 1024 * 1024 * 10, // 10 MB
            duration: Some(Duration::from_secs(60)),
            width: Some(1920),
            height: Some(1080),
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            video_bitrate: Some(5_000_000),
            audio_bitrate: Some(128_000),
            frame_rate: Some(30.0),
            audio_sample_rate: Some(48000),
            audio_channels: Some(2),
        }
    }

    #[test]
    fn test_has_streams() {
        let props = create_test_properties();
        assert!(props.has_video());
        assert!(props.has_audio());
    }

    #[test]
    fn test_aspect_ratio() {
        let props = create_test_properties();
        let ratio = props.aspect_ratio().unwrap();
        assert!((ratio - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_total_bitrate() {
        let props = create_test_properties();
        assert_eq!(props.total_bitrate(), Some(5_128_000));
    }

    #[test]
    fn test_resolution_string() {
        let props = create_test_properties();
        assert_eq!(props.resolution_string(), Some("1920x1080".to_string()));
    }

    #[test]
    fn test_resolution_name() {
        let props = create_test_properties();
        assert_eq!(props.resolution_name(), Some("1080p"));

        let mut props_4k = props.clone();
        props_4k.width = Some(3840);
        props_4k.height = Some(2160);
        assert_eq!(props_4k.resolution_name(), Some("4K"));
    }

    #[test]
    fn test_is_hd() {
        let props = create_test_properties();
        assert!(props.is_hd());

        let mut props_sd = props.clone();
        props_sd.height = Some(480);
        assert!(!props_sd.is_hd());
    }

    #[test]
    fn test_is_4k() {
        let mut props = create_test_properties();
        assert!(!props.is_4k());

        props.width = Some(3840);
        props.height = Some(2160);
        assert!(props.is_4k());
    }

    #[test]
    fn test_duration_seconds() {
        let props = create_test_properties();
        assert_eq!(props.duration_seconds(), Some(60.0));
    }

    #[test]
    fn test_duration_formatted() {
        let props = create_test_properties();
        assert_eq!(props.duration_formatted(), Some("01:00".to_string()));

        let mut props_long = props.clone();
        props_long.duration = Some(Duration::from_secs(3665));
        assert_eq!(
            props_long.duration_formatted(),
            Some("01:01:05".to_string())
        );
    }

    #[test]
    fn test_file_size_formatted() {
        let props = create_test_properties();
        let formatted = props.file_size_formatted();
        assert!(formatted.contains("MB"));
    }

    #[test]
    fn test_calculated_bitrate() {
        let props = create_test_properties();
        let bitrate = props.calculated_bitrate().unwrap();
        // 10 MB over 60 seconds = ~1.4 Mbps
        assert!(bitrate > 1_000_000);
        assert!(bitrate < 2_000_000);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }
}
