//! Metadata-based deduplication and fuzzy matching.
//!
//! This module provides:
//! - Filename similarity (Levenshtein distance, fuzzy matching)
//! - Duration matching with tolerance
//! - Resolution matching
//! - Codec and format matching
//! - Fuzzy metadata comparison

use crate::{DedupError, DedupResult};
use std::path::{Path, PathBuf};

/// Media metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaMetadata {
    /// File path
    pub path: PathBuf,

    /// File size in bytes
    pub size: u64,

    /// Duration in seconds (for audio/video)
    pub duration: Option<f64>,

    /// Video width (if video)
    pub width: Option<u32>,

    /// Video height (if video)
    pub height: Option<u32>,

    /// Bitrate in bits per second
    pub bitrate: Option<u64>,

    /// Frame rate (if video)
    pub framerate: Option<f64>,

    /// Sample rate (if audio)
    pub sample_rate: Option<u32>,

    /// Number of audio channels
    pub channels: Option<u16>,

    /// Video codec
    pub video_codec: Option<String>,

    /// Audio codec
    pub audio_codec: Option<String>,

    /// Container format
    pub container: Option<String>,

    /// Creation timestamp
    pub created: Option<i64>,

    /// Modified timestamp
    pub modified: Option<i64>,
}

impl MediaMetadata {
    /// Create new metadata.
    #[must_use]
    pub fn new(path: PathBuf, size: u64) -> Self {
        Self {
            path,
            size,
            duration: None,
            width: None,
            height: None,
            bitrate: None,
            framerate: None,
            sample_rate: None,
            channels: None,
            video_codec: None,
            audio_codec: None,
            container: None,
            created: None,
            modified: None,
        }
    }

    /// Get filename without extension.
    #[must_use]
    pub fn filename(&self) -> String {
        self.path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string()
    }

    /// Get file extension.
    #[must_use]
    pub fn extension(&self) -> String {
        self.path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase()
    }

    /// Get resolution as string (e.g., "1920x1080").
    #[must_use]
    pub fn resolution(&self) -> Option<String> {
        match (self.width, self.height) {
            (Some(w), Some(h)) => Some(format!("{w}x{h}")),
            _ => None,
        }
    }

    /// Check if this is a video file.
    #[must_use]
    pub fn is_video(&self) -> bool {
        self.width.is_some() && self.height.is_some()
    }

    /// Check if this is an audio file.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        self.sample_rate.is_some() && !self.is_video()
    }

    /// Calculate aspect ratio.
    #[must_use]
    pub fn aspect_ratio(&self) -> Option<f64> {
        match (self.width, self.height) {
            (Some(w), Some(h)) if h > 0 => Some(f64::from(w) / f64::from(h)),
            _ => None,
        }
    }
}

/// Metadata similarity result.
#[derive(Debug, Clone)]
pub struct MetadataSimilarity {
    /// Filename similarity (0.0-1.0)
    pub filename_similarity: f64,

    /// Duration match (0.0-1.0)
    pub duration_match: f64,

    /// Resolution match (0.0-1.0)
    pub resolution_match: f64,

    /// Codec match (0.0-1.0)
    pub codec_match: f64,

    /// Size similarity (0.0-1.0)
    pub size_similarity: f64,

    /// Container match (0.0 or 1.0)
    pub container_match: f64,
}

impl MetadataSimilarity {
    /// Calculate overall similarity score.
    #[must_use]
    pub fn overall_score(&self) -> f64 {
        // Weighted average
        self.filename_similarity * 0.3
            + self.duration_match * 0.2
            + self.resolution_match * 0.2
            + self.codec_match * 0.15
            + self.size_similarity * 0.1
            + self.container_match * 0.05
    }

    /// Check if metadata is similar above threshold.
    #[must_use]
    pub fn is_similar(&self, threshold: f64) -> bool {
        self.overall_score() >= threshold
    }
}

/// Compare two metadata objects.
#[must_use]
pub fn compare_metadata(meta1: &MediaMetadata, meta2: &MediaMetadata) -> MetadataSimilarity {
    let filename_similarity = compare_filenames(&meta1.filename(), &meta2.filename());
    let duration_match = compare_durations(meta1.duration, meta2.duration);
    let resolution_match = compare_resolutions(meta1, meta2);
    let codec_match = compare_codecs(meta1, meta2);
    let size_similarity = compare_sizes(meta1.size, meta2.size);
    let container_match = compare_containers(&meta1.container, &meta2.container);

    MetadataSimilarity {
        filename_similarity,
        duration_match,
        resolution_match,
        codec_match,
        size_similarity,
        container_match,
    }
}

/// Compare filenames using normalized Levenshtein distance.
#[must_use]
pub fn compare_filenames(name1: &str, name2: &str) -> f64 {
    // Normalize: lowercase, remove special characters
    let norm1 = normalize_filename(name1);
    let norm2 = normalize_filename(name2);

    if norm1 == norm2 {
        return 1.0;
    }

    if norm1.is_empty() || norm2.is_empty() {
        return 0.0;
    }

    let distance = levenshtein_distance(&norm1, &norm2);
    let max_len = norm1.len().max(norm2.len());

    1.0 - (distance as f64 / max_len as f64)
}

/// Normalize filename for comparison.
fn normalize_filename(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Calculate Levenshtein distance between two strings.
fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();

    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    let mut matrix = vec![vec![0usize; len2 + 1]; len1 + 1];

    // Initialize first row and column
    for i in 0..=len1 {
        matrix[i][0] = i;
    }
    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    // Fill matrix
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    for i in 1..=len1 {
        for j in 1..=len2 {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                0
            } else {
                1
            };

            matrix[i][j] = (matrix[i - 1][j] + 1) // deletion
                .min(matrix[i][j - 1] + 1) // insertion
                .min(matrix[i - 1][j - 1] + cost); // substitution
        }
    }

    matrix[len1][len2]
}

/// Compare durations with tolerance.
#[must_use]
pub fn compare_durations(dur1: Option<f64>, dur2: Option<f64>) -> f64 {
    match (dur1, dur2) {
        (Some(d1), Some(d2)) => {
            if d1 == 0.0 && d2 == 0.0 {
                return 1.0;
            }

            let max_dur = d1.max(d2);
            if max_dur == 0.0 {
                return 1.0;
            }

            let diff = (d1 - d2).abs();
            let tolerance = max_dur * 0.02; // 2% tolerance

            if diff <= tolerance {
                1.0
            } else {
                (1.0 - diff / max_dur).max(0.0)
            }
        }
        (None, None) => 0.5, // Unknown
        _ => 0.0,            // One has duration, other doesn't
    }
}

/// Compare resolutions.
#[must_use]
pub fn compare_resolutions(meta1: &MediaMetadata, meta2: &MediaMetadata) -> f64 {
    match ((meta1.width, meta1.height), (meta2.width, meta2.height)) {
        ((Some(w1), Some(h1)), (Some(w2), Some(h2))) => {
            if w1 == w2 && h1 == h2 {
                1.0
            } else {
                // Compare aspect ratios
                let ar1 = f64::from(w1) / f64::from(h1);
                let ar2 = f64::from(w2) / f64::from(h2);

                let ar_diff = (ar1 - ar2).abs();
                if ar_diff < 0.01 {
                    // Same aspect ratio, different resolution
                    0.5
                } else {
                    0.0
                }
            }
        }
        ((None, None), (None, None)) => 0.5, // Both unknown
        _ => 0.0,                            // One has resolution, other doesn't
    }
}

/// Compare codecs.
#[must_use]
pub fn compare_codecs(meta1: &MediaMetadata, meta2: &MediaMetadata) -> f64 {
    let video_match = compare_strings(&meta1.video_codec, &meta2.video_codec);
    let audio_match = compare_strings(&meta1.audio_codec, &meta2.audio_codec);

    // Average of video and audio codec matches
    (video_match + audio_match) / 2.0
}

/// Compare optional strings.
fn compare_strings(s1: &Option<String>, s2: &Option<String>) -> f64 {
    match (s1, s2) {
        (Some(a), Some(b)) => {
            if a.eq_ignore_ascii_case(b) {
                1.0
            } else {
                0.0
            }
        }
        (None, None) => 0.5, // Both unknown
        _ => 0.0,            // One known, other unknown
    }
}

/// Compare file sizes.
#[must_use]
pub fn compare_sizes(size1: u64, size2: u64) -> f64 {
    if size1 == 0 && size2 == 0 {
        return 1.0;
    }

    let max_size = size1.max(size2);
    if max_size == 0 {
        return 1.0;
    }

    let diff = (size1 as i64 - size2 as i64).unsigned_abs();
    let tolerance = (max_size as f64 * 0.05) as u64; // 5% tolerance

    if diff <= tolerance {
        1.0
    } else {
        (1.0 - diff as f64 / max_size as f64).max(0.0)
    }
}

/// Compare containers.
#[must_use]
pub fn compare_containers(cont1: &Option<String>, cont2: &Option<String>) -> f64 {
    compare_strings(cont1, cont2)
}

/// Extract metadata from file.
///
/// # Errors
///
/// Returns an error if the file cannot be read.
pub fn extract_metadata(path: impl AsRef<Path>) -> DedupResult<MediaMetadata> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(DedupError::FileNotFound(path.to_path_buf()));
    }

    let file_metadata = std::fs::metadata(path)?;
    let size = file_metadata.len();

    let mut metadata = MediaMetadata::new(path.to_path_buf(), size);

    // Get timestamps
    if let Ok(created) = file_metadata.created() {
        if let Ok(duration) = created.duration_since(std::time::UNIX_EPOCH) {
            metadata.created = Some(duration.as_secs() as i64);
        }
    }

    if let Ok(modified) = file_metadata.modified() {
        if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
            metadata.modified = Some(duration.as_secs() as i64);
        }
    }

    // Set container based on file extension
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    metadata.container = Some(ext);

    // Extract detailed metadata using magic-byte detection
    detect_format_from_magic(path, &mut metadata);

    Ok(metadata)
}

/// Detect media format and codec information from file magic bytes.
fn detect_format_from_magic(path: &Path, metadata: &mut MediaMetadata) {
    use std::io::Read;

    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return,
    };

    let mut buf = [0u8; 64];
    let n = match file.read(&mut buf) {
        Ok(n) => n,
        Err(_) => return,
    };

    if n < 4 {
        return;
    }

    let bytes = &buf[..n];

    // EBML / Matroska / WebM: magic [0x1A, 0x45, 0xDF, 0xA3]
    if bytes.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        // Look for "webm" string in the header area (bytes 4..32)
        let search_range = &bytes[4..n.min(32)];
        let is_webm = search_range.windows(4).any(|w| w == b"webm");
        if is_webm {
            metadata.container = Some("webm".to_string());
        } else {
            metadata.container = Some("mkv".to_string());
        }
        metadata.video_codec = Some("vp9".to_string());
        metadata.audio_codec = Some("opus".to_string());
        return;
    }

    // ftyp box: MP4 family — bytes[4..8] == b"ftyp"
    if n >= 12 && &bytes[4..8] == b"ftyp" {
        let brand = &bytes[8..12];
        if brand == b"qt  " {
            metadata.container = Some("mov".to_string());
        } else if brand == b"M4A " {
            metadata.container = Some("m4a".to_string());
        } else if brand == b"M4V " {
            metadata.container = Some("m4v".to_string());
        } else {
            metadata.container = Some("mp4".to_string());
        }
        metadata.video_codec = Some("h264".to_string());
        metadata.audio_codec = Some("aac".to_string());
        return;
    }

    // RIFF / WAV: b"RIFF" at start and bytes[8..12] == b"WAVE"
    if n >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WAVE" {
        metadata.container = Some("wav".to_string());
        metadata.audio_codec = Some("pcm".to_string());
        // Parse WAV fmt chunk: channels at bytes 22..24 (u16 LE),
        // sample_rate at bytes 24..28 (u32 LE).
        if n >= 28 {
            let channels = u16::from_le_bytes([bytes[22], bytes[23]]);
            let sample_rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
            if channels > 0 {
                metadata.channels = Some(channels);
            }
            if sample_rate > 0 {
                metadata.sample_rate = Some(sample_rate);
            }
        }
        return;
    }

    // FLAC: b"fLaC" at start
    if bytes.starts_with(b"fLaC") {
        metadata.container = Some("flac".to_string());
        metadata.audio_codec = Some("flac".to_string());
        // STREAMINFO block begins at byte 4.
        // Bytes 4: block type/last-metadata-block flag (0x00 = STREAMINFO, not last).
        // Sample rate is stored as a 20-bit big-endian field starting at offset 18
        // within the STREAMINFO data (i.e., file byte 4+4+14 = 22).
        // Layout: bytes[8..10] = min block size, bytes[10..12] = max block size,
        //         bytes[12..15] = min frame size (24-bit),
        //         bytes[15..18] = max frame size (24-bit),
        //         bytes[18..21] = sample_rate (20 bits) | channels (3 bits) | bits/sample-1 (5 bits)
        // We need offset 18 within STREAMINFO which is file offset 4 (fLaC) + 4 (block header) + 18 = 26.
        if n >= 29 {
            // sample_rate is top 20 bits of 3 bytes at file offsets 26,27,28
            let b0 = bytes[26] as u32;
            let b1 = bytes[27] as u32;
            let b2 = bytes[28] as u32;
            let sample_rate = (b0 << 12) | (b1 << 4) | (b2 >> 4);
            if sample_rate > 0 {
                metadata.sample_rate = Some(sample_rate);
            }
            // channels: bits [3:1] of byte 28 (3 bits, value+1)
            let channels = ((b2 >> 1) & 0x07) + 1;
            metadata.channels = Some(channels as u16);
        }
        return;
    }

    // OGG: b"OggS" at start
    if bytes.starts_with(b"OggS") {
        metadata.container = Some("ogg".to_string());
        // Check for Opus or Vorbis identification in the first page payload.
        // The first page data typically starts around byte 28.
        let page_data = if n > 28 { &bytes[28..] } else { &bytes[4..] };
        if page_data.windows(8).any(|w| w == b"OpusHead") {
            metadata.audio_codec = Some("opus".to_string());
        } else if page_data
            .windows(7)
            .any(|w| w == b"\x01vorbis" || w == b"\x03vorbis")
        {
            metadata.audio_codec = Some("vorbis".to_string());
        } else if page_data.windows(6).any(|w| w == b"vorbis") {
            metadata.audio_codec = Some("vorbis".to_string());
        } else {
            metadata.audio_codec = Some("vorbis".to_string());
        }
        return;
    }

    // MPEG-TS: sync byte 0x47 at intervals of 188 bytes
    if n >= 1 && bytes[0] == 0x47 {
        // Check if additional sync bytes appear at expected 188-byte intervals.
        let is_ts = (n >= 189 && bytes[188] == 0x47) || (n >= 1 && bytes[0] == 0x47 && n < 189);
        if is_ts {
            metadata.container = Some("ts".to_string());
            metadata.video_codec = Some("h264".to_string());
            metadata.audio_codec = Some("aac".to_string());
            return;
        }
    }

    // ID3 tag (MP3): b"ID3"
    if bytes.starts_with(b"ID3") {
        metadata.container = Some("mp3".to_string());
        metadata.audio_codec = Some("mp3".to_string());
        return;
    }

    // MP3 sync word: 0xFF followed by 0xE0..=0xFF
    if n >= 2 && bytes[0] == 0xFF && bytes[1] >= 0xE0 {
        metadata.container = Some("mp3".to_string());
        metadata.audio_codec = Some("mp3".to_string());
    }
}

/// Find potential duplicates based on metadata.
#[must_use]
pub fn find_metadata_duplicates(
    metadata_list: &[MediaMetadata],
    threshold: f64,
) -> Vec<Vec<usize>> {
    let mut groups = Vec::new();
    let mut processed = vec![false; metadata_list.len()];

    for i in 0..metadata_list.len() {
        if processed[i] {
            continue;
        }

        let mut group = vec![i];

        for j in (i + 1)..metadata_list.len() {
            if processed[j] {
                continue;
            }

            let similarity = compare_metadata(&metadata_list[i], &metadata_list[j]);

            if similarity.is_similar(threshold) {
                group.push(j);
                processed[j] = true;
            }
        }

        if group.len() > 1 {
            groups.push(group);
        }

        processed[i] = true;
    }

    groups
}

/// Fuzzy search for similar filenames.
#[must_use]
pub fn fuzzy_search(query: &str, candidates: &[String], threshold: f64) -> Vec<(usize, f64)> {
    let mut results = Vec::new();

    for (i, candidate) in candidates.iter().enumerate() {
        let similarity = compare_filenames(query, candidate);

        if similarity >= threshold {
            results.push((i, similarity));
        }
    }

    // Sort by similarity (highest first)
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    results
}

/// Calculate metadata quality score (completeness).
#[must_use]
pub fn metadata_quality(metadata: &MediaMetadata) -> f64 {
    let mut score = 0.0;
    let mut total = 0.0;

    // Check each field
    total += 1.0;
    if metadata.duration.is_some() {
        score += 1.0;
    }

    total += 1.0;
    if metadata.width.is_some() && metadata.height.is_some() {
        score += 1.0;
    }

    total += 1.0;
    if metadata.bitrate.is_some() {
        score += 1.0;
    }

    total += 1.0;
    if metadata.framerate.is_some() || metadata.sample_rate.is_some() {
        score += 1.0;
    }

    total += 1.0;
    if metadata.video_codec.is_some() || metadata.audio_codec.is_some() {
        score += 1.0;
    }

    total += 1.0;
    if metadata.container.is_some() {
        score += 1.0;
    }

    score / total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_metadata(name: &str, duration: f64, width: u32, height: u32) -> MediaMetadata {
        let mut meta = MediaMetadata::new(PathBuf::from(name), 1000000);
        meta.duration = Some(duration);
        meta.width = Some(width);
        meta.height = Some(height);
        meta
    }

    #[test]
    fn test_metadata_creation() {
        let meta = MediaMetadata::new(PathBuf::from("test.mp4"), 1000);
        assert_eq!(meta.size, 1000);
        assert_eq!(meta.extension(), "mp4");
    }

    #[test]
    fn test_filename_extraction() {
        let meta = MediaMetadata::new(PathBuf::from("/path/to/video.mp4"), 1000);
        assert_eq!(meta.filename(), "video");
        assert_eq!(meta.extension(), "mp4");
    }

    #[test]
    fn test_resolution() {
        let mut meta = MediaMetadata::new(PathBuf::from("test.mp4"), 1000);
        meta.width = Some(1920);
        meta.height = Some(1080);

        assert_eq!(meta.resolution(), Some("1920x1080".to_string()));
        assert!(meta.is_video());
    }

    #[test]
    fn test_aspect_ratio() {
        let mut meta = MediaMetadata::new(PathBuf::from("test.mp4"), 1000);
        meta.width = Some(1920);
        meta.height = Some(1080);

        let ar = meta.aspect_ratio().expect("operation should succeed");
        assert!((ar - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_filename_comparison() {
        assert_eq!(compare_filenames("video", "video"), 1.0);
        assert!(compare_filenames("video1", "video2") > 0.5);
        assert!(compare_filenames("test", "completely_different") < 0.5);

        // Case insensitive
        assert_eq!(compare_filenames("VIDEO", "video"), 1.0);

        // Special characters ignored
        assert_eq!(compare_filenames("my-video", "my_video"), 1.0);
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("abc", "abc"), 0);
        assert_eq!(levenshtein_distance("abc", "ab"), 1);
        assert_eq!(levenshtein_distance("abc", "def"), 3);
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn test_duration_comparison() {
        assert_eq!(compare_durations(Some(100.0), Some(100.0)), 1.0);
        assert!(compare_durations(Some(100.0), Some(101.0)) > 0.9); // Within tolerance
        assert!(compare_durations(Some(100.0), Some(200.0)) < 0.9);
        assert_eq!(compare_durations(None, None), 0.5);
        assert_eq!(compare_durations(Some(100.0), None), 0.0);
    }

    #[test]
    fn test_resolution_comparison() {
        let meta1 = create_test_metadata("video1.mp4", 100.0, 1920, 1080);
        let meta2 = create_test_metadata("video2.mp4", 100.0, 1920, 1080);
        let meta3 = create_test_metadata("video3.mp4", 100.0, 1280, 720);
        let meta4 = create_test_metadata("video4.mp4", 100.0, 3840, 2160);

        assert_eq!(compare_resolutions(&meta1, &meta2), 1.0); // Same resolution
        assert_eq!(compare_resolutions(&meta1, &meta4), 0.5); // Same aspect ratio
        assert_eq!(compare_resolutions(&meta1, &meta3), 0.5); // Same aspect ratio
    }

    #[test]
    fn test_size_comparison() {
        assert_eq!(compare_sizes(1000, 1000), 1.0);
        assert!(compare_sizes(1000, 1040) > 0.9); // Within 5% tolerance
        assert!(compare_sizes(1000, 2000) < 0.9);
    }

    #[test]
    fn test_codec_comparison() {
        let mut meta1 = create_test_metadata("video1.mp4", 100.0, 1920, 1080);
        meta1.video_codec = Some("av1".to_string());
        meta1.audio_codec = Some("opus".to_string());

        let mut meta2 = create_test_metadata("video2.mp4", 100.0, 1920, 1080);
        meta2.video_codec = Some("av1".to_string());
        meta2.audio_codec = Some("opus".to_string());

        let mut meta3 = create_test_metadata("video3.mp4", 100.0, 1920, 1080);
        meta3.video_codec = Some("vp9".to_string());
        meta3.audio_codec = Some("opus".to_string());

        let match12 = compare_codecs(&meta1, &meta2);
        let match13 = compare_codecs(&meta1, &meta3);

        assert_eq!(match12, 1.0); // Same codecs
        assert_eq!(match13, 0.5); // One codec different
    }

    #[test]
    fn test_metadata_similarity() {
        let meta1 = create_test_metadata("video_clip.mp4", 100.0, 1920, 1080);
        let meta2 = create_test_metadata("video_clip_copy.mp4", 100.0, 1920, 1080);

        let similarity = compare_metadata(&meta1, &meta2);

        assert!(similarity.filename_similarity > 0.6);
        assert_eq!(similarity.duration_match, 1.0);
        assert_eq!(similarity.resolution_match, 1.0);
        assert!(similarity.is_similar(0.8));
    }

    #[test]
    fn test_fuzzy_search() {
        let candidates = vec![
            "video_clip.mp4".to_string(),
            "audio_track.mp3".to_string(),
            "video_clip_2.mp4".to_string(),
            "completely_different.mov".to_string(),
        ];

        let results = fuzzy_search("video clip", &candidates, 0.5);

        assert!(!results.is_empty());
        assert!(results[0].1 > 0.5); // First result should be most similar
    }

    #[test]
    fn test_metadata_quality() {
        let mut meta = MediaMetadata::new(PathBuf::from("test.mp4"), 1000);
        assert!(metadata_quality(&meta) < 0.2); // Minimal metadata

        meta.duration = Some(100.0);
        meta.width = Some(1920);
        meta.height = Some(1080);
        meta.bitrate = Some(5000000);
        meta.framerate = Some(30.0);
        meta.video_codec = Some("av1".to_string());
        meta.container = Some("mp4".to_string());

        assert!(metadata_quality(&meta) > 0.9); // Complete metadata
    }

    #[test]
    fn test_find_metadata_duplicates() {
        let metadata_list = vec![
            create_test_metadata("video1.mp4", 100.0, 1920, 1080),
            create_test_metadata("video1_copy.mp4", 100.0, 1920, 1080),
            create_test_metadata("video2.mp4", 200.0, 1280, 720),
            create_test_metadata("video1_copy2.mp4", 100.0, 1920, 1080),
        ];

        let groups = find_metadata_duplicates(&metadata_list, 0.8);

        assert_eq!(groups.len(), 1); // One group of duplicates
        assert!(groups[0].len() >= 2); // At least two files in the group
    }
}
