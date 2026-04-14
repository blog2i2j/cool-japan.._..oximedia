#![allow(dead_code)]
//! Streaming iterator protocol for frame-by-frame decoding.
//!
//! Implements Python's `__iter__` / `__next__` protocol so callers can write:
//!
//! ```python
//! import oximedia
//!
//! decoder = oximedia.FrameIterator(width=1920, height=1080, frame_count=300)
//! for frame in decoder:
//!     process(frame)
//! ```
//!
//! This module provides:
//! - [`DecodedFrame`] — a single decoded video frame with metadata
//! - [`FrameIterator`] — a Python iterator that yields `DecodedFrame` objects
//! - [`AudioFrameIterator`] — a Python iterator that yields `DecodedAudioChunk` objects
//! - [`DecodedAudioChunk`] — a single decoded audio chunk with metadata

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// DecodedFrame
// ---------------------------------------------------------------------------

/// A single decoded video frame returned by [`FrameIterator`].
#[pyclass]
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    /// Frame index (0-based).
    #[pyo3(get)]
    pub index: u64,
    /// Frame width in pixels.
    #[pyo3(get)]
    pub width: u32,
    /// Frame height in pixels.
    #[pyo3(get)]
    pub height: u32,
    /// Presentation timestamp in microseconds.
    #[pyo3(get)]
    pub pts_us: i64,
    /// Whether this is a keyframe.
    #[pyo3(get)]
    pub is_keyframe: bool,
    /// Pixel format string (e.g. "yuv420p", "rgb24").
    #[pyo3(get)]
    pub pixel_format: String,
    /// Raw pixel data (Y plane or packed RGB).
    #[pyo3(get)]
    pub data: Vec<u8>,
}

#[pymethods]
impl DecodedFrame {
    #[new]
    #[pyo3(signature = (index=0, width=0, height=0, pts_us=0, is_keyframe=false, pixel_format="yuv420p"))]
    pub fn new(
        index: u64,
        width: u32,
        height: u32,
        pts_us: i64,
        is_keyframe: bool,
        pixel_format: &str,
    ) -> Self {
        let data_size = (width as usize) * (height as usize) * 3 / 2; // YUV420
        Self {
            index,
            width,
            height,
            pts_us,
            is_keyframe,
            pixel_format: pixel_format.to_string(),
            data: vec![0u8; data_size],
        }
    }

    /// Number of bytes in the frame data.
    pub fn data_size(&self) -> usize {
        self.data.len()
    }

    /// Pixel count (width * height).
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Timestamp in seconds (from microsecond PTS).
    pub fn pts_seconds(&self) -> f64 {
        self.pts_us as f64 / 1_000_000.0
    }

    fn __repr__(&self) -> String {
        format!(
            "DecodedFrame(index={}, {}x{}, pts={}, keyframe={}, fmt={:?})",
            self.index, self.width, self.height, self.pts_us, self.is_keyframe, self.pixel_format
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// FrameIterator
// ---------------------------------------------------------------------------

/// A Python-compatible iterator that yields [`DecodedFrame`] objects.
///
/// Simulates frame-by-frame decoding. The iterator yields `frame_count` frames
/// and then raises `StopIteration`.
///
/// This demonstrates the `__iter__` / `__next__` protocol for integration
/// with Python `for` loops.
#[pyclass]
#[derive(Debug, Clone)]
pub struct FrameIterator {
    /// Output frame width.
    width: u32,
    /// Output frame height.
    height: u32,
    /// Total number of frames to yield.
    frame_count: u64,
    /// Current frame index.
    current: u64,
    /// Microseconds per frame (derived from fps).
    us_per_frame: i64,
    /// Keyframe interval (every N frames).
    keyframe_interval: u64,
    /// Pixel format string.
    pixel_format: String,
}

#[pymethods]
impl FrameIterator {
    /// Create a new frame iterator.
    ///
    /// Parameters
    /// ----------
    /// width : int
    ///     Frame width in pixels.
    /// height : int
    ///     Frame height in pixels.
    /// frame_count : int
    ///     Total frames to yield.
    /// fps : float
    ///     Frames per second (used to compute PTS). Defaults to 30.0.
    /// keyframe_interval : int
    ///     Emit a keyframe every N frames. Defaults to 30.
    /// pixel_format : str
    ///     Pixel format label. Defaults to "yuv420p".
    #[new]
    #[pyo3(signature = (width=1920, height=1080, frame_count=300, fps=30.0, keyframe_interval=30, pixel_format="yuv420p"))]
    pub fn new(
        width: u32,
        height: u32,
        frame_count: u64,
        fps: f64,
        keyframe_interval: u64,
        pixel_format: &str,
    ) -> Self {
        let us_per_frame = if fps > 0.0 {
            (1_000_000.0 / fps) as i64
        } else {
            33_333 // ~30fps fallback
        };
        Self {
            width,
            height,
            frame_count,
            current: 0,
            us_per_frame,
            keyframe_interval: keyframe_interval.max(1),
            pixel_format: pixel_format.to_string(),
        }
    }

    /// Python iterator protocol: return self.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Python iterator protocol: return next frame or raise StopIteration.
    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<DecodedFrame> {
        if slf.current >= slf.frame_count {
            return None;
        }
        let index = slf.current;
        let pts_us = index as i64 * slf.us_per_frame;
        let is_keyframe = index % slf.keyframe_interval == 0;
        let pixel_format = slf.pixel_format.clone();
        let width = slf.width;
        let height = slf.height;
        slf.current += 1;

        // Generate synthetic frame data with a simple pattern
        let data_size = (width as usize) * (height as usize) * 3 / 2;
        let pattern_byte = (index & 0xFF) as u8;
        let data = vec![pattern_byte; data_size];

        Some(DecodedFrame {
            index,
            width,
            height,
            pts_us,
            is_keyframe,
            pixel_format,
            data,
        })
    }

    /// Number of frames remaining.
    fn __length_hint__(&self) -> usize {
        (self.frame_count.saturating_sub(self.current)) as usize
    }

    /// Reset the iterator to the beginning.
    pub fn reset(&mut self) {
        self.current = 0;
    }

    /// Skip ahead by `n` frames.
    pub fn skip(&mut self, n: u64) {
        self.current = self.current.saturating_add(n).min(self.frame_count);
    }

    /// Seek to a specific frame index.
    pub fn seek(&mut self, index: u64) {
        self.current = index.min(self.frame_count);
    }

    /// Total frame count.
    #[getter]
    pub fn total_frames(&self) -> u64 {
        self.frame_count
    }

    /// Current frame position.
    #[getter]
    pub fn position(&self) -> u64 {
        self.current
    }

    /// Whether the iterator is exhausted.
    #[getter]
    pub fn is_exhausted(&self) -> bool {
        self.current >= self.frame_count
    }

    fn __repr__(&self) -> String {
        format!(
            "FrameIterator({}x{}, {}/{} frames, fmt={:?})",
            self.width, self.height, self.current, self.frame_count, self.pixel_format
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// DecodedAudioChunk
// ---------------------------------------------------------------------------

/// A single decoded audio chunk returned by [`AudioFrameIterator`].
#[pyclass]
#[derive(Debug, Clone)]
pub struct DecodedAudioChunk {
    /// Chunk index (0-based).
    #[pyo3(get)]
    pub index: u64,
    /// Number of audio samples in this chunk.
    #[pyo3(get)]
    pub sample_count: u32,
    /// Sample rate in Hz.
    #[pyo3(get)]
    pub sample_rate: u32,
    /// Number of channels.
    #[pyo3(get)]
    pub channels: u32,
    /// Presentation timestamp in microseconds.
    #[pyo3(get)]
    pub pts_us: i64,
    /// Sample format string (e.g. "f32", "s16").
    #[pyo3(get)]
    pub sample_format: String,
    /// Raw audio sample data.
    #[pyo3(get)]
    pub data: Vec<u8>,
}

#[pymethods]
impl DecodedAudioChunk {
    #[new]
    #[pyo3(signature = (index=0, sample_count=1024, sample_rate=48000, channels=2, pts_us=0, sample_format="f32"))]
    pub fn new(
        index: u64,
        sample_count: u32,
        sample_rate: u32,
        channels: u32,
        pts_us: i64,
        sample_format: &str,
    ) -> Self {
        let bytes_per_sample: usize = match sample_format {
            "s16" => 2,
            "s32" | "f32" => 4,
            "f64" => 8,
            _ => 4, // default f32
        };
        let data_size = (sample_count as usize) * (channels as usize) * bytes_per_sample;
        Self {
            index,
            sample_count,
            sample_rate,
            channels,
            pts_us,
            sample_format: sample_format.to_string(),
            data: vec![0u8; data_size],
        }
    }

    /// Duration of this chunk in seconds.
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        f64::from(self.sample_count) / f64::from(self.sample_rate)
    }

    /// Timestamp in seconds.
    pub fn pts_seconds(&self) -> f64 {
        self.pts_us as f64 / 1_000_000.0
    }

    /// Data size in bytes.
    pub fn data_size(&self) -> usize {
        self.data.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "DecodedAudioChunk(index={}, {} samples, {}Hz, {}ch, pts={}, fmt={:?})",
            self.index, self.sample_count, self.sample_rate, self.channels,
            self.pts_us, self.sample_format
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// AudioFrameIterator
// ---------------------------------------------------------------------------

/// Python-compatible iterator for audio chunks.
#[pyclass]
#[derive(Debug, Clone)]
pub struct AudioFrameIterator {
    /// Total chunks to yield.
    chunk_count: u64,
    /// Current chunk index.
    current: u64,
    /// Samples per chunk.
    samples_per_chunk: u32,
    /// Sample rate.
    sample_rate: u32,
    /// Channel count.
    channels: u32,
    /// Sample format.
    sample_format: String,
    /// Microseconds per chunk.
    us_per_chunk: i64,
}

#[pymethods]
impl AudioFrameIterator {
    #[new]
    #[pyo3(signature = (chunk_count=100, samples_per_chunk=1024, sample_rate=48000, channels=2, sample_format="f32"))]
    pub fn new(
        chunk_count: u64,
        samples_per_chunk: u32,
        sample_rate: u32,
        channels: u32,
        sample_format: &str,
    ) -> Self {
        let us_per_chunk = if sample_rate > 0 {
            (f64::from(samples_per_chunk) / f64::from(sample_rate) * 1_000_000.0) as i64
        } else {
            0
        };
        Self {
            chunk_count,
            current: 0,
            samples_per_chunk,
            sample_rate,
            channels,
            sample_format: sample_format.to_string(),
            us_per_chunk,
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<DecodedAudioChunk> {
        if slf.current >= slf.chunk_count {
            return None;
        }
        let index = slf.current;
        let pts_us = index as i64 * slf.us_per_chunk;
        let sample_count = slf.samples_per_chunk;
        let sample_rate = slf.sample_rate;
        let channels = slf.channels;
        let sample_format = slf.sample_format.clone();
        slf.current += 1;

        let bytes_per_sample: usize = match sample_format.as_str() {
            "s16" => 2,
            "s32" | "f32" => 4,
            "f64" => 8,
            _ => 4,
        };
        let data_size = (sample_count as usize) * (channels as usize) * bytes_per_sample;

        Some(DecodedAudioChunk {
            index,
            sample_count,
            sample_rate,
            channels,
            pts_us,
            sample_format,
            data: vec![0u8; data_size],
        })
    }

    fn __length_hint__(&self) -> usize {
        (self.chunk_count.saturating_sub(self.current)) as usize
    }

    /// Reset to the beginning.
    pub fn reset(&mut self) {
        self.current = 0;
    }

    /// Skip ahead by `n` chunks.
    pub fn skip(&mut self, n: u64) {
        self.current = self.current.saturating_add(n).min(self.chunk_count);
    }

    #[getter]
    pub fn total_chunks(&self) -> u64 {
        self.chunk_count
    }

    #[getter]
    pub fn position(&self) -> u64 {
        self.current
    }

    #[getter]
    pub fn is_exhausted(&self) -> bool {
        self.current >= self.chunk_count
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioFrameIterator({}/{} chunks, {} samples/chunk, {}Hz, {}ch)",
            self.current, self.chunk_count, self.samples_per_chunk,
            self.sample_rate, self.channels
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register frame iterator types into the parent module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DecodedFrame>()?;
    m.add_class::<FrameIterator>()?;
    m.add_class::<DecodedAudioChunk>()?;
    m.add_class::<AudioFrameIterator>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoded_frame_new() {
        let frame = DecodedFrame::new(0, 320, 240, 0, true, "yuv420p");
        assert_eq!(frame.index, 0);
        assert_eq!(frame.width, 320);
        assert_eq!(frame.height, 240);
        assert!(frame.is_keyframe);
        assert_eq!(frame.pixel_format, "yuv420p");
        // YUV420: w*h * 3/2
        assert_eq!(frame.data_size(), 320 * 240 * 3 / 2);
    }

    #[test]
    fn test_decoded_frame_pixel_count() {
        let frame = DecodedFrame::new(0, 1920, 1080, 0, false, "rgb24");
        assert_eq!(frame.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_decoded_frame_pts_seconds() {
        let frame = DecodedFrame::new(0, 320, 240, 2_000_000, false, "yuv420p");
        assert!((frame.pts_seconds() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_decoded_frame_repr() {
        let frame = DecodedFrame::new(5, 320, 240, 166666, true, "yuv420p");
        let repr = frame.__repr__();
        assert!(repr.contains("index=5"));
        assert!(repr.contains("320x240"));
        assert!(repr.contains("keyframe=true"));
    }

    #[test]
    fn test_frame_iterator_creation() {
        let it = FrameIterator::new(640, 480, 10, 30.0, 5, "yuv420p");
        assert_eq!(it.total_frames(), 10);
        assert_eq!(it.position(), 0);
        assert!(!it.is_exhausted());
    }

    #[test]
    fn test_frame_iterator_exhaustion() {
        let mut it = FrameIterator::new(64, 48, 3, 30.0, 1, "yuv420p");
        // Simulate iteration without Python runtime
        assert_eq!(it.current, 0);
        it.current = 3;
        assert!(it.is_exhausted());
    }

    #[test]
    fn test_frame_iterator_reset() {
        let mut it = FrameIterator::new(64, 48, 10, 30.0, 5, "yuv420p");
        it.current = 7;
        it.reset();
        assert_eq!(it.position(), 0);
    }

    #[test]
    fn test_frame_iterator_skip() {
        let mut it = FrameIterator::new(64, 48, 100, 30.0, 30, "yuv420p");
        it.skip(50);
        assert_eq!(it.position(), 50);
        it.skip(200); // should clamp to frame_count
        assert_eq!(it.position(), 100);
    }

    #[test]
    fn test_frame_iterator_seek() {
        let mut it = FrameIterator::new(64, 48, 100, 30.0, 30, "yuv420p");
        it.seek(42);
        assert_eq!(it.position(), 42);
        it.seek(9999);
        assert_eq!(it.position(), 100);
    }

    #[test]
    fn test_frame_iterator_length_hint() {
        let mut it = FrameIterator::new(64, 48, 100, 30.0, 30, "yuv420p");
        assert_eq!(it.__length_hint__(), 100);
        it.current = 30;
        assert_eq!(it.__length_hint__(), 70);
    }

    #[test]
    fn test_frame_iterator_repr() {
        let it = FrameIterator::new(1920, 1080, 300, 30.0, 30, "yuv420p");
        let repr = it.__repr__();
        assert!(repr.contains("1920x1080"));
        assert!(repr.contains("0/300"));
    }

    #[test]
    fn test_frame_iterator_zero_fps_fallback() {
        let it = FrameIterator::new(320, 240, 10, 0.0, 1, "yuv420p");
        assert_eq!(it.us_per_frame, 33_333); // fallback
    }

    #[test]
    fn test_audio_chunk_new() {
        let chunk = DecodedAudioChunk::new(0, 1024, 48000, 2, 0, "f32");
        assert_eq!(chunk.sample_count, 1024);
        assert_eq!(chunk.sample_rate, 48000);
        assert_eq!(chunk.channels, 2);
        assert_eq!(chunk.data_size(), 1024 * 2 * 4); // f32 = 4 bytes
    }

    #[test]
    fn test_audio_chunk_duration() {
        let chunk = DecodedAudioChunk::new(0, 48000, 48000, 1, 0, "f32");
        assert!((chunk.duration_secs() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_audio_chunk_zero_sample_rate() {
        let chunk = DecodedAudioChunk::new(0, 1024, 0, 2, 0, "f32");
        assert!((chunk.duration_secs()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_audio_chunk_pts_seconds() {
        let chunk = DecodedAudioChunk::new(0, 1024, 48000, 2, 5_000_000, "s16");
        assert!((chunk.pts_seconds() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_audio_chunk_s16_data_size() {
        let chunk = DecodedAudioChunk::new(0, 512, 44100, 1, 0, "s16");
        assert_eq!(chunk.data_size(), 512 * 1 * 2); // s16 = 2 bytes
    }

    #[test]
    fn test_audio_iterator_creation() {
        let it = AudioFrameIterator::new(50, 1024, 48000, 2, "f32");
        assert_eq!(it.total_chunks(), 50);
        assert_eq!(it.position(), 0);
        assert!(!it.is_exhausted());
    }

    #[test]
    fn test_audio_iterator_skip_and_reset() {
        let mut it = AudioFrameIterator::new(100, 1024, 48000, 2, "f32");
        it.skip(50);
        assert_eq!(it.position(), 50);
        it.reset();
        assert_eq!(it.position(), 0);
    }

    #[test]
    fn test_audio_iterator_exhaustion() {
        let mut it = AudioFrameIterator::new(5, 1024, 48000, 2, "f32");
        it.current = 5;
        assert!(it.is_exhausted());
        assert_eq!(it.__length_hint__(), 0);
    }

    #[test]
    fn test_audio_iterator_repr() {
        let it = AudioFrameIterator::new(100, 1024, 48000, 2, "f32");
        let repr = it.__repr__();
        assert!(repr.contains("0/100"));
        assert!(repr.contains("1024 samples"));
        assert!(repr.contains("48000Hz"));
    }
}
