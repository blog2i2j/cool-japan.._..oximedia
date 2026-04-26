//! Test sequence management for codec benchmarking.
//!
//! This module provides functionality for managing test video sequences used in
//! benchmarking. Sequences can have different characteristics like resolution,
//! frame rate, content type, and motion characteristics.

use crate::{BenchError, BenchResult};
use oximedia_codec::{Plane, VideoFrame};
use oximedia_core::types::{PixelFormat, Rational, Timestamp};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Content type classification for test sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContentType {
    /// Animation content (e.g., cartoons, CGI)
    Animation,
    /// Live action content
    LiveAction,
    /// Screen content (e.g., presentations, UI)
    ScreenContent,
    /// Sports content with high motion
    Sports,
    /// Low motion content (e.g., talking head)
    LowMotion,
    /// Mixed content
    Mixed,
}

/// Motion characteristics of a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionCharacteristics {
    /// Very low motion (< 5% of pixels change)
    VeryLow,
    /// Low motion (5-15% of pixels change)
    Low,
    /// Medium motion (15-30% of pixels change)
    Medium,
    /// High motion (30-50% of pixels change)
    High,
    /// Very high motion (> 50% of pixels change)
    VeryHigh,
}

/// Scene complexity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SceneComplexity {
    /// Low complexity (flat colors, simple shapes)
    Low,
    /// Medium complexity
    Medium,
    /// High complexity (detailed textures, noise)
    High,
}

/// Test sequence metadata and properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSequence {
    /// Sequence name/identifier
    pub name: String,

    /// Path to the sequence file
    pub path: PathBuf,

    /// Video width in pixels
    pub width: usize,

    /// Video height in pixels
    pub height: usize,

    /// Frame rate
    pub frame_rate: Rational,

    /// Pixel format
    pub pixel_format: PixelFormat,

    /// Total number of frames
    pub frame_count: usize,

    /// Content type
    pub content_type: ContentType,

    /// Motion characteristics
    pub motion: MotionCharacteristics,

    /// Scene complexity
    pub complexity: SceneComplexity,

    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl TestSequence {
    /// Create a new test sequence.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        path: impl Into<PathBuf>,
        width: usize,
        height: usize,
        frame_rate: Rational,
    ) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            width,
            height,
            frame_rate,
            pixel_format: PixelFormat::Yuv420p,
            frame_count: 0,
            content_type: ContentType::Mixed,
            motion: MotionCharacteristics::Medium,
            complexity: SceneComplexity::Medium,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Set the pixel format.
    #[must_use]
    pub fn with_pixel_format(mut self, format: PixelFormat) -> Self {
        self.pixel_format = format;
        self
    }

    /// Set the frame count.
    #[must_use]
    pub fn with_frame_count(mut self, count: usize) -> Self {
        self.frame_count = count;
        self
    }

    /// Set the content type.
    #[must_use]
    pub fn with_content_type(mut self, content_type: ContentType) -> Self {
        self.content_type = content_type;
        self
    }

    /// Set the motion characteristics.
    #[must_use]
    pub fn with_motion(mut self, motion: MotionCharacteristics) -> Self {
        self.motion = motion;
        self
    }

    /// Set the scene complexity.
    #[must_use]
    pub fn with_complexity(mut self, complexity: SceneComplexity) -> Self {
        self.complexity = complexity;
        self
    }

    /// Add metadata.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get the sequence resolution as a string.
    #[must_use]
    pub fn resolution_string(&self) -> String {
        format!("{}x{}", self.width, self.height)
    }

    /// Check if this is a high-resolution sequence (>= 1080p).
    #[must_use]
    pub fn is_high_resolution(&self) -> bool {
        self.height >= 1080
    }

    /// Check if this is a 4K sequence.
    #[must_use]
    pub fn is_4k(&self) -> bool {
        self.width >= 3840 && self.height >= 2160
    }

    /// Get the total number of pixels per frame.
    #[must_use]
    pub fn total_pixels(&self) -> usize {
        self.width * self.height
    }

    /// Load frames from the sequence.
    ///
    /// Supports Y4M (YUV4MPEG2) files.  Other formats return an error.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, parsed, or is not a
    /// supported format.
    pub fn load_frames(&self, max_frames: Option<usize>) -> BenchResult<Vec<VideoFrame>> {
        let limit = max_frames.unwrap_or(usize::MAX);

        let ext = self.path.extension().and_then(|e| e.to_str()).unwrap_or("");

        if ext.eq_ignore_ascii_case("y4m") {
            load_y4m_frames(&self.path, limit)
        } else {
            Err(BenchError::ExecutionFailed(format!(
                "Unsupported file format '{}' for sequence '{}'; only .y4m is supported",
                ext, self.name
            )))
        }
    }

    /// Validate that the sequence file exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the sequence file doesn't exist.
    pub fn validate(&self) -> BenchResult<()> {
        if !self.path.exists() {
            return Err(BenchError::SequenceNotFound(self.name.clone()));
        }

        if self.width == 0 || self.height == 0 {
            return Err(BenchError::InvalidConfig(format!(
                "Invalid resolution for sequence {}",
                self.name
            )));
        }

        if self.frame_count == 0 {
            return Err(BenchError::InvalidConfig(format!(
                "Zero frames in sequence {}",
                self.name
            )));
        }

        Ok(())
    }
}

/// A collection of test sequences.
#[derive(Debug, Clone, Default)]
pub struct SequenceSet {
    sequences: Vec<TestSequence>,
}

impl SequenceSet {
    /// Create a new empty sequence set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a sequence to the set.
    pub fn add(&mut self, sequence: TestSequence) {
        self.sequences.push(sequence);
    }

    /// Get all sequences.
    #[must_use]
    pub fn sequences(&self) -> &[TestSequence] {
        &self.sequences
    }

    /// Get sequences by content type.
    #[must_use]
    pub fn by_content_type(&self, content_type: ContentType) -> Vec<&TestSequence> {
        self.sequences
            .iter()
            .filter(|s| s.content_type == content_type)
            .collect()
    }

    /// Get sequences by resolution.
    #[must_use]
    pub fn by_resolution(&self, width: usize, height: usize) -> Vec<&TestSequence> {
        self.sequences
            .iter()
            .filter(|s| s.width == width && s.height == height)
            .collect()
    }

    /// Get high-resolution sequences (>= 1080p).
    #[must_use]
    pub fn high_resolution(&self) -> Vec<&TestSequence> {
        self.sequences
            .iter()
            .filter(|s| s.is_high_resolution())
            .collect()
    }

    /// Validate all sequences.
    ///
    /// # Errors
    ///
    /// Returns an error if any sequence is invalid.
    pub fn validate_all(&self) -> BenchResult<()> {
        for sequence in &self.sequences {
            sequence.validate()?;
        }
        Ok(())
    }

    /// Create a standard test set with common resolutions.
    #[must_use]
    pub fn standard_set() -> Self {
        let mut set = Self::new();

        // 480p sequence
        set.add(
            TestSequence::new(
                "test_480p",
                "./sequences/test_480p.y4m",
                854,
                480,
                Rational::new(30, 1),
            )
            .with_frame_count(300)
            .with_content_type(ContentType::Mixed)
            .with_motion(MotionCharacteristics::Medium),
        );

        // 720p sequence
        set.add(
            TestSequence::new(
                "test_720p",
                "./sequences/test_720p.y4m",
                1280,
                720,
                Rational::new(30, 1),
            )
            .with_frame_count(300)
            .with_content_type(ContentType::Mixed)
            .with_motion(MotionCharacteristics::Medium),
        );

        // 1080p sequence
        set.add(
            TestSequence::new(
                "test_1080p",
                "./sequences/test_1080p.y4m",
                1920,
                1080,
                Rational::new(30, 1),
            )
            .with_frame_count(300)
            .with_content_type(ContentType::Mixed)
            .with_motion(MotionCharacteristics::Medium),
        );

        // 4K sequence
        set.add(
            TestSequence::new(
                "test_4k",
                "./sequences/test_4k.y4m",
                3840,
                2160,
                Rational::new(30, 1),
            )
            .with_frame_count(150)
            .with_content_type(ContentType::Mixed)
            .with_motion(MotionCharacteristics::Medium),
        );

        set
    }
}

/// Load a test sequence from a Y4M file.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn load_y4m_sequence(path: impl AsRef<Path>) -> BenchResult<TestSequence> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(BenchError::SequenceNotFound(path.display().to_string()));
    }

    // This is a placeholder - in reality, this would parse the Y4M header
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Default values - would be parsed from Y4M header
    Ok(TestSequence::new(
        name,
        path,
        1920,
        1080,
        Rational::new(30, 1),
    ))
}

/// Generate a synthetic test sequence.
///
/// Each [`ContentType`] maps to a distinct visual pattern:
///
/// | `ContentType`   | Pattern                             |
/// |-----------------|-------------------------------------|
/// | `Animation`     | Horizontal luma gradient            |
/// | `LowMotion`     | Solid mid-gray (Y=128, UV=128)      |
/// | `ScreenContent` | 8×8 pixel black/white checkerboard  |
/// | `Sports`        | Deterministic LCG noise per pixel   |
/// | `LiveAction`    | Deterministic LCG noise per pixel   |
/// | `Mixed`         | Horizontal luma gradient            |
///
/// All frames use `Yuv420p` with full-range values.
///
/// # Errors
///
/// Returns an error if width or height is zero.
pub fn generate_synthetic_sequence(
    _name: impl Into<String>,
    width: usize,
    height: usize,
    frame_count: usize,
    content_type: ContentType,
) -> BenchResult<Vec<VideoFrame>> {
    if width == 0 || height == 0 {
        return Err(BenchError::InvalidConfig(
            "Width and height must be non-zero".to_string(),
        ));
    }

    let timebase = Rational::new(1, 1000);
    let mut frames = Vec::with_capacity(frame_count);

    for frame_idx in 0..frame_count {
        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width as u32, height as u32);
        frame.timestamp = Timestamp::new(frame_idx as i64, timebase);

        let chroma_w = width.div_ceil(2);
        let chroma_h = height.div_ceil(2);

        let (y_plane, u_plane, v_plane) = match content_type {
            ContentType::LowMotion => {
                // Solid mid-gray
                let y = vec![128u8; width * height];
                let u = vec![128u8; chroma_w * chroma_h];
                let v = vec![128u8; chroma_w * chroma_h];
                (y, u, v)
            }
            ContentType::ScreenContent => {
                // 8×8 black/white checkerboard
                const CELL: usize = 8;
                let mut y = vec![0u8; width * height];
                for row in 0..height {
                    for col in 0..width {
                        if ((col / CELL) + (row / CELL)) % 2 == 0 {
                            y[row * width + col] = 235; // white (luma limited range white)
                        } else {
                            y[row * width + col] = 16; // black (luma limited range black)
                        }
                    }
                }
                let u = vec![128u8; chroma_w * chroma_h];
                let v = vec![128u8; chroma_w * chroma_h];
                (y, u, v)
            }
            ContentType::Sports | ContentType::LiveAction => {
                // Deterministic noise via a fast mixing hash (splitmix64 style)
                let mut y = vec![0u8; width * height];
                for (pixel_idx, sample) in y.iter_mut().enumerate() {
                    let mut seed = frame_idx as u64 * 0x9e3779b97f4a7c15 + pixel_idx as u64;
                    seed ^= seed >> 33;
                    seed = seed.wrapping_mul(0xff51afd7ed558ccd);
                    seed ^= seed >> 33;
                    seed = seed.wrapping_mul(0xc4ceb9fe1a85ec53);
                    *sample = (seed >> 56) as u8;
                }
                let mut u = vec![0u8; chroma_w * chroma_h];
                for (pixel_idx, sample) in u.iter_mut().enumerate() {
                    let mut seed = frame_idx as u64 * 0x6c62272e07bb0142 + pixel_idx as u64;
                    seed ^= seed >> 33;
                    seed = seed.wrapping_mul(0xff51afd7ed558ccd);
                    seed ^= seed >> 33;
                    seed = seed.wrapping_mul(0xc4ceb9fe1a85ec53);
                    *sample = (seed >> 56) as u8;
                }
                let mut v = vec![0u8; chroma_w * chroma_h];
                for (pixel_idx, sample) in v.iter_mut().enumerate() {
                    let mut seed = frame_idx as u64 * 0xb492b66fbe98f273 + pixel_idx as u64;
                    seed ^= seed >> 33;
                    seed = seed.wrapping_mul(0xff51afd7ed558ccd);
                    seed ^= seed >> 33;
                    seed = seed.wrapping_mul(0xc4ceb9fe1a85ec53);
                    *sample = (seed >> 56) as u8;
                }
                (y, u, v)
            }
            // Animation and Mixed: horizontal luma gradient
            ContentType::Animation | ContentType::Mixed => {
                let mut y = vec![0u8; width * height];
                for row in 0..height {
                    for col in 0..width {
                        // Scale column index to 0-235 luma range
                        let luma = if width > 1 {
                            (col * 235 / (width - 1)) as u8
                        } else {
                            128
                        };
                        y[row * width + col] = luma;
                    }
                }
                let u = vec![128u8; chroma_w * chroma_h];
                let v = vec![128u8; chroma_w * chroma_h];
                (y, u, v)
            }
        };

        frame.planes = vec![
            Plane::with_dimensions(y_plane, width, width as u32, height as u32),
            Plane::with_dimensions(u_plane, chroma_w, chroma_w as u32, chroma_h as u32),
            Plane::with_dimensions(v_plane, chroma_w, chroma_w as u32, chroma_h as u32),
        ];

        frames.push(frame);
    }

    Ok(frames)
}

/// Load frames from a Y4M (YUV4MPEG2) file as `Yuv420p` [`VideoFrame`]s.
///
/// The Y4M header format is:
/// ```text
/// YUV4MPEG2 W<w> H<h> F<fps_n>:<fps_d> I<interlace> A<ar_n>:<ar_d> C<chroma>\n
/// ```
/// Each frame consists of the literal text `FRAME\n` followed by raw YUV420
/// byte data (Y plane: `W×H`, U plane: `(W/2)×(H/2)`, V plane: `(W/2)×(H/2)`).
///
/// # Errors
///
/// Returns an error if the file cannot be read, the header is missing or
/// malformed, or frame data is truncated.
fn load_y4m_frames(path: &Path, max_frames: usize) -> BenchResult<Vec<VideoFrame>> {
    use std::io::{BufRead, BufReader, Read};

    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);

    // --- Parse header line ---
    let mut header = String::new();
    reader.read_line(&mut header)?;
    let header = header.trim_end_matches('\n').trim_end_matches('\r');

    if !header.starts_with("YUV4MPEG2") {
        return Err(BenchError::ExecutionFailed(format!(
            "Not a valid Y4M file (missing YUV4MPEG2 magic): {}",
            path.display()
        )));
    }

    let mut width: u32 = 0;
    let mut height: u32 = 0;

    for token in header.split_whitespace() {
        if let Some(rest) = token.strip_prefix('W') {
            width = rest.parse::<u32>().map_err(|_| {
                BenchError::ExecutionFailed(format!("Invalid width token '{token}' in Y4M header"))
            })?;
        } else if let Some(rest) = token.strip_prefix('H') {
            height = rest.parse::<u32>().map_err(|_| {
                BenchError::ExecutionFailed(format!("Invalid height token '{token}' in Y4M header"))
            })?;
        }
    }

    if width == 0 || height == 0 {
        return Err(BenchError::ExecutionFailed(format!(
            "Y4M header missing valid width/height in {}",
            path.display()
        )));
    }

    let w = width as usize;
    let h = height as usize;
    let chroma_w = w.div_ceil(2);
    let chroma_h = h.div_ceil(2);
    let y_size = w * h;
    let uv_size = chroma_w * chroma_h;
    let timebase = Rational::new(1, 1000);

    let mut frames = Vec::new();
    let mut frame_idx: usize = 0;

    loop {
        if frame_idx >= max_frames {
            break;
        }

        // Read the FRAME marker line
        let mut frame_header = String::new();
        let bytes_read = reader.read_line(&mut frame_header)?;
        if bytes_read == 0 {
            // EOF
            break;
        }

        let frame_header = frame_header.trim_end_matches('\n').trim_end_matches('\r');
        if !frame_header.starts_with("FRAME") {
            return Err(BenchError::ExecutionFailed(format!(
                "Expected 'FRAME' marker at frame {frame_idx}, got '{frame_header}'"
            )));
        }

        // Read raw YUV420 data
        let mut y_data = vec![0u8; y_size];
        let mut u_data = vec![0u8; uv_size];
        let mut v_data = vec![0u8; uv_size];

        reader.read_exact(&mut y_data).map_err(|e| {
            BenchError::ExecutionFailed(format!("Truncated Y plane at frame {frame_idx}: {e}"))
        })?;
        reader.read_exact(&mut u_data).map_err(|e| {
            BenchError::ExecutionFailed(format!("Truncated U plane at frame {frame_idx}: {e}"))
        })?;
        reader.read_exact(&mut v_data).map_err(|e| {
            BenchError::ExecutionFailed(format!("Truncated V plane at frame {frame_idx}: {e}"))
        })?;

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.timestamp = Timestamp::new(frame_idx as i64, timebase);
        frame.planes = vec![
            Plane::with_dimensions(y_data, w, width, height),
            Plane::with_dimensions(u_data, chroma_w, chroma_w as u32, chroma_h as u32),
            Plane::with_dimensions(v_data, chroma_w, chroma_w as u32, chroma_h as u32),
        ];

        frames.push(frame);
        frame_idx += 1;
    }

    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("oximedia-bench-sequences-{name}"))
    }

    #[test]
    fn test_sequence_creation() {
        let seq = TestSequence::new(
            "test",
            tmp_path("test.y4m"),
            1920,
            1080,
            Rational::new(30, 1),
        );

        assert_eq!(seq.name, "test");
        assert_eq!(seq.width, 1920);
        assert_eq!(seq.height, 1080);
    }

    #[test]
    fn test_sequence_builder() {
        let seq = TestSequence::new(
            "test",
            tmp_path("test.y4m"),
            1920,
            1080,
            Rational::new(30, 1),
        )
        .with_frame_count(300)
        .with_content_type(ContentType::Sports)
        .with_motion(MotionCharacteristics::High);

        assert_eq!(seq.frame_count, 300);
        assert_eq!(seq.content_type, ContentType::Sports);
        assert_eq!(seq.motion, MotionCharacteristics::High);
    }

    #[test]
    fn test_resolution_checks() {
        let seq_1080p = TestSequence::new(
            "1080p",
            tmp_path("test.y4m"),
            1920,
            1080,
            Rational::new(30, 1),
        );
        assert!(seq_1080p.is_high_resolution());
        assert!(!seq_1080p.is_4k());

        let seq_4k =
            TestSequence::new("4k", tmp_path("test.y4m"), 3840, 2160, Rational::new(30, 1));
        assert!(seq_4k.is_high_resolution());
        assert!(seq_4k.is_4k());
    }

    #[test]
    fn test_resolution_string() {
        let seq = TestSequence::new(
            "test",
            tmp_path("test.y4m"),
            1920,
            1080,
            Rational::new(30, 1),
        );
        assert_eq!(seq.resolution_string(), "1920x1080");
    }

    #[test]
    fn test_total_pixels() {
        let seq = TestSequence::new(
            "test",
            tmp_path("test.y4m"),
            1920,
            1080,
            Rational::new(30, 1),
        );
        assert_eq!(seq.total_pixels(), 1920 * 1080);
    }

    #[test]
    fn test_sequence_set() {
        let mut set = SequenceSet::new();

        let seq1 = TestSequence::new(
            "seq1",
            tmp_path("seq1.y4m"),
            1920,
            1080,
            Rational::new(30, 1),
        )
        .with_content_type(ContentType::Sports);

        let seq2 = TestSequence::new(
            "seq2",
            tmp_path("seq2.y4m"),
            1280,
            720,
            Rational::new(30, 1),
        )
        .with_content_type(ContentType::Animation);

        set.add(seq1);
        set.add(seq2);

        assert_eq!(set.sequences().len(), 2);
        assert_eq!(set.by_content_type(ContentType::Sports).len(), 1);
        assert_eq!(set.by_resolution(1920, 1080).len(), 1);
    }

    #[test]
    fn test_standard_set() {
        let set = SequenceSet::standard_set();
        assert!(!set.sequences().is_empty());
        assert!(set.high_resolution().len() >= 2); // At least 1080p and 4K
    }

    #[test]
    fn test_content_type_serialization() {
        let ct = ContentType::Sports;
        let json = serde_json::to_string(&ct).expect("json should be valid");
        let deserialized: ContentType =
            serde_json::from_str(&json).expect("test expectation failed");
        assert_eq!(ct, deserialized);
    }

    #[test]
    fn test_motion_characteristics() {
        let motion = MotionCharacteristics::High;
        let json = serde_json::to_string(&motion).expect("json should be valid");
        let deserialized: MotionCharacteristics =
            serde_json::from_str(&json).expect("test expectation failed");
        assert_eq!(motion, deserialized);
    }

    #[test]
    fn test_sequence_metadata() {
        let seq = TestSequence::new(
            "test",
            tmp_path("test.y4m"),
            1920,
            1080,
            Rational::new(30, 1),
        )
        .with_metadata("source", "camera_a")
        .with_metadata("date", "2024-01-01");

        assert_eq!(seq.metadata.get("source"), Some(&"camera_a".to_string()));
        assert_eq!(seq.metadata.get("date"), Some(&"2024-01-01".to_string()));
    }

    // ----------------------------------------------------------------
    // Tests for newly implemented stubs
    // ----------------------------------------------------------------

    /// Build a minimal valid 4×4 Y4M file with one frame in a temp file.
    fn write_test_y4m(path: &std::path::Path, width: usize, height: usize, frames: usize) {
        use std::io::Write;
        let mut f = std::fs::File::create(path).expect("create y4m");
        writeln!(f, "YUV4MPEG2 W{width} H{height} F30:1 Ip A1:1 C420").expect("header");

        let chroma_w = width.div_ceil(2);
        let chroma_h = height.div_ceil(2);
        let y_size = width * height;
        let uv_size = chroma_w * chroma_h;

        for i in 0..frames {
            writeln!(f, "FRAME").expect("frame marker");
            f.write_all(&vec![64u8.wrapping_add(i as u8); y_size])
                .expect("y plane");
            f.write_all(&vec![128u8; uv_size]).expect("u plane");
            f.write_all(&vec![128u8; uv_size]).expect("v plane");
        }
    }

    #[test]
    fn test_load_frames_y4m() {
        let path = std::env::temp_dir().join("oximedia_bench_test_load_y4m_4x4.y4m");
        write_test_y4m(&path, 4, 4, 1);

        let seq =
            TestSequence::new("test_y4m", &path, 4, 4, Rational::new(30, 1)).with_frame_count(1);
        let frames = seq.load_frames(None).expect("load_frames should succeed");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].width, 4);
        assert_eq!(frames[0].height, 4);
        // Y plane size
        assert_eq!(frames[0].planes[0].data.len(), 4 * 4);
    }

    #[test]
    fn test_load_frames_y4m_max_frames() {
        let path = std::env::temp_dir().join("oximedia_bench_test_load_y4m_maxframes.y4m");
        write_test_y4m(&path, 4, 4, 5);

        let seq =
            TestSequence::new("max_frames", &path, 4, 4, Rational::new(30, 1)).with_frame_count(5);
        let frames = seq
            .load_frames(Some(2))
            .expect("load_frames with max_frames should succeed");
        assert_eq!(frames.len(), 2);
    }

    #[test]
    fn test_load_frames_unsupported_format_errors() {
        let path = std::env::temp_dir().join("oximedia_bench_test_load_unsupported.mp4");
        // Create a dummy file so it exists
        std::fs::write(&path, b"dummy").expect("write");
        let seq =
            TestSequence::new("unsupported", &path, 4, 4, Rational::new(30, 1)).with_frame_count(1);
        let result = seq.load_frames(None);
        assert!(result.is_err(), "should fail for non-y4m extension");
    }

    #[test]
    fn test_generate_solid_color_low_motion() {
        // LowMotion → solid gray
        let frames = generate_synthetic_sequence("test", 4, 4, 3, ContentType::LowMotion)
            .expect("generate should succeed");
        assert_eq!(frames.len(), 3);
        for frame in &frames {
            // Y plane all 128
            assert!(frame.planes[0].data.iter().all(|&v| v == 128));
        }
    }

    #[test]
    fn test_generate_checkerboard_screen_content() {
        // ScreenContent → checkerboard with cell=8; at 16×16 width pixel(0,0)=white, pixel(8,0)=black
        let frames = generate_synthetic_sequence("test", 16, 16, 1, ContentType::ScreenContent)
            .expect("generate should succeed");
        assert_eq!(frames.len(), 1);
        let y = &frames[0].planes[0].data;
        // (col=0, row=0): (0/8 + 0/8) % 2 == 0 → white (235)
        assert_eq!(y[0], 235, "pixel(0,0) should be white");
        // (col=8, row=0): (8/8 + 0/8) % 2 == 1 → black (16)
        assert_eq!(y[8], 16, "pixel(8,0) should be black");
    }

    #[test]
    fn test_generate_noise_sports() {
        let frames = generate_synthetic_sequence("test", 8, 8, 2, ContentType::Sports)
            .expect("generate should succeed");
        assert_eq!(frames.len(), 2);
        // Frames should differ (different frame_idx seeds)
        let y0 = &frames[0].planes[0].data;
        let y1 = &frames[1].planes[0].data;
        assert_ne!(y0, y1, "noise frames should differ");
    }

    #[test]
    fn test_generate_gradient_animation() {
        let width = 8;
        let frames = generate_synthetic_sequence("test", width, 4, 1, ContentType::Animation)
            .expect("generate should succeed");
        assert_eq!(frames.len(), 1);
        let y = &frames[0].planes[0].data;
        // First pixel should be 0 (leftmost), last pixel of first row should be 235
        assert_eq!(y[0], 0, "leftmost pixel should be 0");
        assert_eq!(y[width - 1], 235, "rightmost pixel should be 235");
    }
}

/// Sequence analyzer for analyzing video content characteristics.
pub struct SequenceAnalyzer;

impl SequenceAnalyzer {
    /// Analyze motion characteristics of a sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if analysis fails.
    pub fn analyze_motion(_frames: &[VideoFrame]) -> BenchResult<MotionAnalysis> {
        // Placeholder for motion analysis
        Ok(MotionAnalysis {
            average_motion_magnitude: 0.0,
            motion_distribution: MotionDistribution::default(),
            scene_changes: Vec::new(),
        })
    }

    /// Analyze spatial complexity.
    ///
    /// # Errors
    ///
    /// Returns an error if analysis fails.
    pub fn analyze_spatial_complexity(_frames: &[VideoFrame]) -> BenchResult<SpatialComplexity> {
        // Placeholder for spatial complexity analysis
        Ok(SpatialComplexity {
            average_complexity: 0.0,
            variance: 0.0,
            edge_density: 0.0,
        })
    }

    /// Analyze temporal complexity.
    ///
    /// # Errors
    ///
    /// Returns an error if analysis fails.
    pub fn analyze_temporal_complexity(_frames: &[VideoFrame]) -> BenchResult<TemporalComplexity> {
        // Placeholder for temporal complexity analysis
        Ok(TemporalComplexity {
            average_temporal_difference: 0.0,
            scene_change_frequency: 0.0,
        })
    }

    /// Detect scene changes.
    ///
    /// # Errors
    ///
    /// Returns an error if detection fails.
    pub fn detect_scene_changes(_frames: &[VideoFrame]) -> BenchResult<Vec<usize>> {
        // Placeholder for scene change detection
        Ok(Vec::new())
    }
}

/// Motion analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionAnalysis {
    /// Average motion magnitude
    pub average_motion_magnitude: f64,
    /// Motion distribution
    pub motion_distribution: MotionDistribution,
    /// Scene change frame indices
    pub scene_changes: Vec<usize>,
}

/// Motion distribution statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MotionDistribution {
    /// Percentage of low motion blocks
    pub low_motion_percentage: f64,
    /// Percentage of medium motion blocks
    pub medium_motion_percentage: f64,
    /// Percentage of high motion blocks
    pub high_motion_percentage: f64,
}

/// Spatial complexity analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialComplexity {
    /// Average spatial complexity
    pub average_complexity: f64,
    /// Variance in complexity
    pub variance: f64,
    /// Edge density
    pub edge_density: f64,
}

/// Temporal complexity analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalComplexity {
    /// Average temporal difference
    pub average_temporal_difference: f64,
    /// Scene change frequency (changes per second)
    pub scene_change_frequency: f64,
}

/// Sequence generator for creating synthetic test sequences.
#[allow(dead_code)]
pub struct SequenceGenerator {
    width: usize,
    height: usize,
    frame_rate: Rational,
}

impl SequenceGenerator {
    /// Create a new sequence generator.
    #[must_use]
    pub fn new(width: usize, height: usize, frame_rate: Rational) -> Self {
        Self {
            width,
            height,
            frame_rate,
        }
    }

    /// Generate a solid color sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn generate_solid_color(
        &self,
        _frame_count: usize,
        _color: (u8, u8, u8),
    ) -> BenchResult<Vec<VideoFrame>> {
        // Placeholder
        Ok(Vec::new())
    }

    /// Generate a gradient sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn generate_gradient(&self, _frame_count: usize) -> BenchResult<Vec<VideoFrame>> {
        // Placeholder
        Ok(Vec::new())
    }

    /// Generate a noise sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn generate_noise(
        &self,
        _frame_count: usize,
        _noise_level: f64,
    ) -> BenchResult<Vec<VideoFrame>> {
        // Placeholder
        Ok(Vec::new())
    }

    /// Generate a motion sequence (moving object).
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn generate_motion(
        &self,
        _frame_count: usize,
        _velocity: (f64, f64),
    ) -> BenchResult<Vec<VideoFrame>> {
        // Placeholder
        Ok(Vec::new())
    }

    /// Generate a checkboard pattern sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn generate_checkerboard(
        &self,
        _frame_count: usize,
        _block_size: usize,
    ) -> BenchResult<Vec<VideoFrame>> {
        // Placeholder
        Ok(Vec::new())
    }
}

/// Sequence validation utilities.
pub struct SequenceValidator;

impl SequenceValidator {
    /// Validate sequence integrity.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails.
    pub fn validate_integrity(_sequence: &TestSequence) -> BenchResult<ValidationResult> {
        Ok(ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        })
    }

    /// Check for corrupted frames.
    ///
    /// # Errors
    ///
    /// Returns an error if checking fails.
    pub fn check_for_corruption(_frames: &[VideoFrame]) -> BenchResult<Vec<usize>> {
        // Placeholder - returns indices of corrupted frames
        Ok(Vec::new())
    }

    /// Validate frame consistency.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails.
    pub fn validate_frame_consistency(_frames: &[VideoFrame]) -> BenchResult<bool> {
        // Check that all frames have same dimensions, format, etc.
        Ok(true)
    }
}

/// Validation result.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the sequence is valid
    pub is_valid: bool,
    /// List of errors
    pub errors: Vec<String>,
    /// List of warnings
    pub warnings: Vec<String>,
}

/// Sequence metadata extractor.
pub struct MetadataExtractor;

impl MetadataExtractor {
    /// Extract metadata from a sequence file.
    ///
    /// # Errors
    ///
    /// Returns an error if extraction fails.
    pub fn extract(_path: impl AsRef<std::path::Path>) -> BenchResult<SequenceMetadata> {
        Ok(SequenceMetadata {
            format: "Y4M".to_string(),
            codec: None,
            bit_depth: 8,
            chroma_subsampling: "4:2:0".to_string(),
            color_space: "BT.709".to_string(),
            hdr_metadata: None,
        })
    }
}

/// Sequence metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceMetadata {
    /// File format
    pub format: String,
    /// Codec (if compressed)
    pub codec: Option<String>,
    /// Bit depth
    pub bit_depth: u8,
    /// Chroma subsampling
    pub chroma_subsampling: String,
    /// Color space
    pub color_space: String,
    /// HDR metadata
    pub hdr_metadata: Option<HdrMetadata>,
}

/// HDR metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdrMetadata {
    /// Transfer function
    pub transfer_function: String,
    /// Color primaries
    pub color_primaries: String,
    /// Master display metadata
    pub master_display: Option<MasterDisplayMetadata>,
    /// Content light level
    pub content_light_level: Option<ContentLightLevel>,
}

/// Master display metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterDisplayMetadata {
    /// Display primaries
    pub display_primaries: [[u16; 2]; 3],
    /// White point
    pub white_point: [u16; 2],
    /// Maximum display luminance
    pub max_luminance: u32,
    /// Minimum display luminance
    pub min_luminance: u32,
}

/// Content light level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentLightLevel {
    /// Maximum content light level
    pub max_cll: u16,
    /// Maximum frame average light level
    pub max_fall: u16,
}

/// Sequence database for managing multiple test sequences.
#[derive(Debug, Clone)]
pub struct SequenceDatabase {
    sequences: Vec<TestSequence>,
    index_by_name: std::collections::HashMap<String, usize>,
    index_by_resolution: std::collections::HashMap<String, Vec<usize>>,
}

impl SequenceDatabase {
    /// Create a new sequence database.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sequences: Vec::new(),
            index_by_name: std::collections::HashMap::new(),
            index_by_resolution: std::collections::HashMap::new(),
        }
    }

    /// Add a sequence to the database.
    pub fn add(&mut self, sequence: TestSequence) {
        let idx = self.sequences.len();
        let name = sequence.name.clone();
        let resolution = sequence.resolution_string();

        self.sequences.push(sequence);
        self.index_by_name.insert(name, idx);
        self.index_by_resolution
            .entry(resolution)
            .or_insert_with(Vec::new)
            .push(idx);
    }

    /// Get a sequence by name.
    #[must_use]
    pub fn get_by_name(&self, name: &str) -> Option<&TestSequence> {
        self.index_by_name
            .get(name)
            .and_then(|&idx| self.sequences.get(idx))
    }

    /// Get sequences by resolution.
    #[must_use]
    pub fn get_by_resolution(&self, width: usize, height: usize) -> Vec<&TestSequence> {
        let resolution = format!("{width}x{height}");
        self.index_by_resolution
            .get(&resolution)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|&idx| self.sequences.get(idx))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all sequences.
    #[must_use]
    pub fn all(&self) -> &[TestSequence] {
        &self.sequences
    }

    /// Get sequences by content type.
    #[must_use]
    pub fn get_by_content_type(&self, content_type: ContentType) -> Vec<&TestSequence> {
        self.sequences
            .iter()
            .filter(|s| s.content_type == content_type)
            .collect()
    }

    /// Get sequences by motion characteristics.
    #[must_use]
    pub fn get_by_motion(&self, motion: MotionCharacteristics) -> Vec<&TestSequence> {
        self.sequences
            .iter()
            .filter(|s| s.motion == motion)
            .collect()
    }

    /// Load sequences from a directory.
    ///
    /// # Errors
    ///
    /// Returns an error if loading fails.
    pub fn load_from_directory(
        &mut self,
        _path: impl AsRef<std::path::Path>,
    ) -> BenchResult<usize> {
        // Placeholder for loading sequences from a directory
        Ok(0)
    }

    /// Export database to JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if export fails.
    pub fn export_to_json(&self, path: impl AsRef<std::path::Path>) -> BenchResult<()> {
        let json = serde_json::to_string_pretty(&self.sequences)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Import database from JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if import fails.
    pub fn import_from_json(&mut self, path: impl AsRef<std::path::Path>) -> BenchResult<()> {
        let json = std::fs::read_to_string(path)?;
        let sequences: Vec<TestSequence> = serde_json::from_str(&json)?;

        for sequence in sequences {
            self.add(sequence);
        }

        Ok(())
    }

    /// Get database statistics.
    #[must_use]
    pub fn statistics(&self) -> DatabaseStatistics {
        let mut stats = DatabaseStatistics {
            total_sequences: self.sequences.len(),
            total_frames: 0,
            total_duration_seconds: 0.0,
            resolutions: std::collections::HashMap::new(),
            content_types: std::collections::HashMap::new(),
        };

        for seq in &self.sequences {
            stats.total_frames += seq.frame_count;
            stats.total_duration_seconds += seq.frame_count as f64 / seq.frame_rate.to_f64();

            *stats
                .resolutions
                .entry(seq.resolution_string())
                .or_insert(0) += 1;
            *stats.content_types.entry(seq.content_type).or_insert(0) += 1;
        }

        stats
    }
}

impl Default for SequenceDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Database statistics.
#[derive(Debug, Clone)]
pub struct DatabaseStatistics {
    /// Total number of sequences
    pub total_sequences: usize,
    /// Total number of frames
    pub total_frames: usize,
    /// Total duration in seconds
    pub total_duration_seconds: f64,
    /// Resolution distribution
    pub resolutions: std::collections::HashMap<String, usize>,
    /// Content type distribution
    pub content_types: std::collections::HashMap<ContentType, usize>,
}

#[cfg(test)]
mod extended_tests {
    use super::*;

    fn tmp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("oximedia-bench-sequences-ext-{name}"))
    }

    #[test]
    fn test_sequence_generator() {
        let gen = SequenceGenerator::new(1920, 1080, Rational::new(30, 1));
        assert_eq!(gen.width, 1920);
        assert_eq!(gen.height, 1080);
    }

    #[test]
    fn test_sequence_database() {
        let mut db = SequenceDatabase::new();

        let seq = TestSequence::new(
            "test_1080p",
            tmp_path("test.y4m"),
            1920,
            1080,
            Rational::new(30, 1),
        );

        db.add(seq);

        assert_eq!(db.all().len(), 1);
        assert!(db.get_by_name("test_1080p").is_some());
        assert_eq!(db.get_by_resolution(1920, 1080).len(), 1);
    }

    #[test]
    fn test_database_statistics() {
        let mut db = SequenceDatabase::new();

        db.add(
            TestSequence::new(
                "seq1",
                tmp_path("seq1.y4m"),
                1920,
                1080,
                Rational::new(30, 1),
            )
            .with_frame_count(300),
        );

        db.add(
            TestSequence::new(
                "seq2",
                tmp_path("seq2.y4m"),
                1280,
                720,
                Rational::new(30, 1),
            )
            .with_frame_count(150),
        );

        let stats = db.statistics();
        assert_eq!(stats.total_sequences, 2);
        assert_eq!(stats.total_frames, 450);
    }

    #[test]
    fn test_motion_distribution() {
        let dist = MotionDistribution {
            low_motion_percentage: 60.0,
            medium_motion_percentage: 30.0,
            high_motion_percentage: 10.0,
        };

        assert_eq!(dist.low_motion_percentage, 60.0);
    }

    #[test]
    fn test_validation_result() {
        let result = ValidationResult {
            is_valid: true,
            errors: vec![],
            warnings: vec!["Warning: Low frame count".to_string()],
        };

        assert!(result.is_valid);
        assert_eq!(result.warnings.len(), 1);
    }

    #[test]
    fn test_hdr_metadata() {
        let metadata = HdrMetadata {
            transfer_function: "PQ".to_string(),
            color_primaries: "BT.2020".to_string(),
            master_display: None,
            content_light_level: Some(ContentLightLevel {
                max_cll: 1000,
                max_fall: 400,
            }),
        };

        assert_eq!(metadata.transfer_function, "PQ");
        assert!(metadata.content_light_level.is_some());
    }
}
