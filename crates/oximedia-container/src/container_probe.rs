#![allow(dead_code)]
//! Higher-level container probing beyond magic-byte detection.
//!
//! Provides `ContainerProbeResult`, `ContainerInfo`, and `ContainerProber`
//! for interrogating container structure without a full demux pass.

/// Summary flags produced by probing a container's header region.
#[derive(Debug, Clone, PartialEq)]
pub struct ContainerProbeResult {
    /// Whether at least one video track was detected.
    pub video_present: bool,
    /// Whether at least one audio track was detected.
    pub audio_present: bool,
    /// Whether at least one subtitle track was detected.
    pub subtitle_present: bool,
    /// Confidence of the format detection in the range `[0.0, 1.0]`.
    pub confidence: f32,
    /// Raw format name string as reported by the container layer.
    pub format_label: String,
}

impl ContainerProbeResult {
    /// Creates a new probe result with default confidence of 1.0.
    #[must_use]
    pub fn new(format_label: impl Into<String>) -> Self {
        Self {
            video_present: false,
            audio_present: false,
            subtitle_present: false,
            confidence: 1.0,
            format_label: format_label.into(),
        }
    }

    /// Returns `true` when at least one video track was detected.
    #[must_use]
    pub fn has_video(&self) -> bool {
        self.video_present
    }

    /// Returns `true` when at least one audio track was detected.
    #[must_use]
    pub fn has_audio(&self) -> bool {
        self.audio_present
    }

    /// Returns `true` for multimedia containers that have both video and audio.
    #[must_use]
    pub fn is_av(&self) -> bool {
        self.video_present && self.audio_present
    }

    /// Returns `true` when confidence is at or above `threshold`.
    #[must_use]
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

/// Detailed structural information about a container, produced after a
/// more thorough header scan than a simple magic-byte probe.
#[derive(Debug, Clone)]
pub struct ContainerInfo {
    /// Short format name (e.g. `"matroska"`, `"mp4"`, `"ogg"`).
    format_name: String,
    /// Total number of tracks (all types).
    total_tracks: usize,
    /// Number of video tracks.
    video_count: usize,
    /// Number of audio tracks.
    audio_count: usize,
    /// Total container duration in milliseconds, if signalled.
    duration_ms: Option<u64>,
    /// Container file size in bytes, if known.
    file_size: Option<u64>,
}

impl ContainerInfo {
    /// Creates a new `ContainerInfo`.
    #[must_use]
    pub fn new(format_name: impl Into<String>) -> Self {
        Self {
            format_name: format_name.into(),
            total_tracks: 0,
            video_count: 0,
            audio_count: 0,
            duration_ms: None,
            file_size: None,
        }
    }

    /// Sets video and audio track counts, automatically deriving `total_tracks`.
    #[must_use]
    pub fn with_tracks(mut self, video: usize, audio: usize) -> Self {
        self.video_count = video;
        self.audio_count = audio;
        self.total_tracks = video + audio;
        self
    }

    /// Sets the duration.
    #[must_use]
    pub fn with_duration_ms(mut self, ms: u64) -> Self {
        self.duration_ms = Some(ms);
        self
    }

    /// Sets the file size.
    #[must_use]
    pub fn with_file_size(mut self, bytes: u64) -> Self {
        self.file_size = Some(bytes);
        self
    }

    /// Returns the short format name.
    #[must_use]
    pub fn format_name(&self) -> &str {
        &self.format_name
    }

    /// Returns the total track count (all types).
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.total_tracks
    }

    /// Returns the number of video tracks.
    #[must_use]
    pub fn video_count(&self) -> usize {
        self.video_count
    }

    /// Returns the number of audio tracks.
    #[must_use]
    pub fn audio_count(&self) -> usize {
        self.audio_count
    }

    /// Returns the duration in milliseconds, if known.
    #[must_use]
    pub fn duration_ms(&self) -> Option<u64> {
        self.duration_ms
    }

    /// Estimates the average bit rate in kbps from file size and duration.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn estimated_bitrate_kbps(&self) -> Option<f64> {
        match (self.file_size, self.duration_ms) {
            (Some(bytes), Some(ms)) if ms > 0 => Some((bytes as f64 * 8.0) / (ms as f64)),
            _ => None,
        }
    }
}

/// A thin prober that inspects raw bytes and fills a `ContainerInfo`.
#[derive(Debug, Default)]
pub struct ContainerProber {
    probed_count: usize,
}

impl ContainerProber {
    /// Creates a new `ContainerProber`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of containers probed so far.
    #[must_use]
    pub fn probed_count(&self) -> usize {
        self.probed_count
    }

    /// Inspects the first bytes of a container and returns a
    /// `ContainerProbeResult`.
    ///
    /// Detection is based on well-known magic sequences:
    /// - `[0x1A, 0x45, 0xDF, 0xA3]` → Matroska / `WebM`
    /// - `[0x66, 0x4C, 0x61, 0x43]` (`fLaC`) → FLAC
    /// - `[0x4F, 0x67, 0x67, 0x53]` (`OggS`) → Ogg
    /// - `[0x52, 0x49, 0x46, 0x46]` (`RIFF`) → WAV
    /// - `[0x00, 0x00, 0x00, _, 0x66, 0x74, 0x79, 0x70]` → MP4/ftyp
    pub fn probe_header(&mut self, header: &[u8]) -> ContainerProbeResult {
        self.probed_count += 1;

        if header.len() >= 4 && header[..4] == [0x1A, 0x45, 0xDF, 0xA3] {
            let mut r = ContainerProbeResult::new("matroska");
            r.video_present = true;
            r.audio_present = true;
            return r;
        }
        if header.len() >= 4 && &header[..4] == b"fLaC" {
            let mut r = ContainerProbeResult::new("flac");
            r.audio_present = true;
            return r;
        }
        if header.len() >= 4 && &header[..4] == b"OggS" {
            let mut r = ContainerProbeResult::new("ogg");
            r.audio_present = true;
            return r;
        }
        if header.len() >= 4 && &header[..4] == b"RIFF" {
            let mut r = ContainerProbeResult::new("wav");
            r.audio_present = true;
            return r;
        }
        // MP4: check bytes 4-7 for "ftyp"
        if header.len() >= 8 && &header[4..8] == b"ftyp" {
            let mut r = ContainerProbeResult::new("mp4");
            r.video_present = true;
            r.audio_present = true;
            return r;
        }

        let mut r = ContainerProbeResult::new("unknown");
        r.confidence = 0.0;
        r
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // 1. has_video – true
    #[test]
    fn test_has_video_true() {
        let mut r = ContainerProbeResult::new("mkv");
        r.video_present = true;
        assert!(r.has_video());
    }

    // 2. has_video – false
    #[test]
    fn test_has_video_false() {
        let r = ContainerProbeResult::new("flac");
        assert!(!r.has_video());
    }

    // 3. has_audio – true
    #[test]
    fn test_has_audio_true() {
        let mut r = ContainerProbeResult::new("ogg");
        r.audio_present = true;
        assert!(r.has_audio());
    }

    // 4. is_av – both present
    #[test]
    fn test_is_av_both() {
        let mut r = ContainerProbeResult::new("mp4");
        r.video_present = true;
        r.audio_present = true;
        assert!(r.is_av());
    }

    // 5. is_av – audio only
    #[test]
    fn test_is_av_audio_only() {
        let mut r = ContainerProbeResult::new("wav");
        r.audio_present = true;
        assert!(!r.is_av());
    }

    // 6. is_confident threshold
    #[test]
    fn test_is_confident() {
        let r = ContainerProbeResult::new("matroska");
        assert!(r.is_confident(0.9));
        assert!(!r.is_confident(1.1));
    }

    // 7. ContainerInfo format_name
    #[test]
    fn test_container_info_format_name() {
        let info = ContainerInfo::new("matroska");
        assert_eq!(info.format_name(), "matroska");
    }

    // 8. ContainerInfo track_count
    #[test]
    fn test_container_info_track_count() {
        let info = ContainerInfo::new("mp4").with_tracks(1, 2);
        assert_eq!(info.track_count(), 3);
    }

    // 9. ContainerInfo video_count
    #[test]
    fn test_container_info_video_count() {
        let info = ContainerInfo::new("mkv").with_tracks(2, 4);
        assert_eq!(info.video_count(), 2);
        assert_eq!(info.audio_count(), 4);
    }

    // 10. estimated_bitrate_kbps – computes correctly
    #[test]
    fn test_estimated_bitrate_kbps() {
        let info = ContainerInfo::new("mp4")
            .with_file_size(1_000_000)
            .with_duration_ms(1000);
        // 1 MB in 1 s = 8 Mbps = 8000 kbps
        let kbps = info
            .estimated_bitrate_kbps()
            .expect("operation should succeed");
        assert!((kbps - 8000.0).abs() < 1.0);
    }

    // 11. estimated_bitrate_kbps – None when no duration
    #[test]
    fn test_estimated_bitrate_kbps_no_duration() {
        let info = ContainerInfo::new("mkv").with_file_size(1_000_000);
        assert!(info.estimated_bitrate_kbps().is_none());
    }

    // 12. ContainerProber detects Matroska
    #[test]
    fn test_probe_matroska() {
        let mut p = ContainerProber::new();
        let magic = [0x1A, 0x45, 0xDF, 0xA3, 0x00, 0x00, 0x00, 0x00];
        let r = p.probe_header(&magic);
        assert_eq!(r.format_label, "matroska");
        assert!(r.has_video());
        assert!(r.has_audio());
    }

    // 13. ContainerProber detects FLAC
    #[test]
    fn test_probe_flac() {
        let mut p = ContainerProber::new();
        let r = p.probe_header(b"fLaC\x00\x00\x00\x22");
        assert_eq!(r.format_label, "flac");
        assert!(!r.has_video());
        assert!(r.has_audio());
    }

    // 14. ContainerProber detects MP4 via ftyp box
    #[test]
    fn test_probe_mp4() {
        let mut p = ContainerProber::new();
        // 4-byte box size + "ftyp"
        let header = b"\x00\x00\x00\x18ftyp\x69\x73\x6f\x6d";
        let r = p.probe_header(header);
        assert_eq!(r.format_label, "mp4");
        assert!(r.has_video());
        assert_eq!(p.probed_count(), 1);
    }

    // 15. ContainerProber unknown returns confidence 0
    #[test]
    fn test_probe_unknown() {
        let mut p = ContainerProber::new();
        let r = p.probe_header(b"\xFF\xFF\xFF\xFF");
        assert_eq!(r.format_label, "unknown");
        assert_eq!(r.confidence, 0.0);
    }
}
