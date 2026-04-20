//! Rendering pipeline for timeline.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::error::TimelineResult;
use crate::timeline::Timeline;
use crate::types::Position;

/// Render quality settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderQuality {
    /// Draft quality (fast, low quality).
    Draft,
    /// Preview quality (medium speed, medium quality).
    Preview,
    /// High quality (slow, high quality).
    High,
    /// Maximum quality (slowest, highest quality).
    Maximum,
}

/// Render format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenderFormat {
    /// Video frame format.
    Video {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
        /// Pixel format.
        pixel_format: String,
    },
    /// Audio format.
    Audio {
        /// Sample rate.
        sample_rate: u32,
        /// Number of channels.
        channels: u32,
        /// Sample format.
        sample_format: String,
    },
}

/// Render settings.
#[derive(Clone, Debug)]
pub struct RenderSettings {
    /// Render quality.
    pub quality: RenderQuality,
    /// Start position.
    pub start: Position,
    /// End position.
    pub end: Position,
    /// Video format.
    pub video_format: Option<RenderFormat>,
    /// Audio format.
    pub audio_format: Option<RenderFormat>,
    /// Enable effects rendering.
    pub render_effects: bool,
    /// Enable transitions rendering.
    pub render_transitions: bool,
    /// Enable GPU acceleration.
    pub use_gpu: bool,
}

impl RenderSettings {
    /// Creates new render settings.
    #[must_use]
    pub fn new(start: Position, end: Position) -> Self {
        Self {
            quality: RenderQuality::High,
            start,
            end,
            video_format: None,
            audio_format: None,
            render_effects: true,
            render_transitions: true,
            use_gpu: true,
        }
    }

    /// Sets video format.
    #[must_use]
    pub fn with_video_format(mut self, width: u32, height: u32, pixel_format: String) -> Self {
        self.video_format = Some(RenderFormat::Video {
            width,
            height,
            pixel_format,
        });
        self
    }

    /// Sets audio format.
    #[must_use]
    pub fn with_audio_format(
        mut self,
        sample_rate: u32,
        channels: u32,
        sample_format: String,
    ) -> Self {
        self.audio_format = Some(RenderFormat::Audio {
            sample_rate,
            channels,
            sample_format,
        });
        self
    }

    /// Sets render quality.
    #[must_use]
    pub fn with_quality(mut self, quality: RenderQuality) -> Self {
        self.quality = quality;
        self
    }
}

/// Rendering progress information.
#[derive(Clone, Debug)]
pub struct RenderProgress {
    /// Current position being rendered.
    pub position: Position,
    /// Total frames to render.
    pub total_frames: i64,
    /// Frames rendered so far.
    pub rendered_frames: i64,
    /// Estimated time remaining (seconds).
    pub estimated_time_remaining: f64,
}

impl RenderProgress {
    /// Calculates progress percentage.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn percentage(&self) -> f64 {
        if self.total_frames == 0 {
            return 0.0;
        }
        (self.rendered_frames as f64 / self.total_frames as f64) * 100.0
    }
}

/// Render cache for pre-rendered frames.
pub struct RenderCache {
    /// Cached video frames.
    video_cache: HashMap<Position, Vec<u8>>,
    /// Cached audio buffers.
    audio_cache: HashMap<Position, Vec<f32>>,
    /// Maximum cache size in bytes.
    max_cache_size: usize,
    /// Current cache size in bytes.
    current_cache_size: usize,
}

impl RenderCache {
    /// Creates a new render cache.
    #[must_use]
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            video_cache: HashMap::new(),
            audio_cache: HashMap::new(),
            max_cache_size,
            current_cache_size: 0,
        }
    }

    /// Caches a video frame.
    pub fn cache_video_frame(&mut self, position: Position, data: Vec<u8>) {
        let data_size = data.len();
        if self.current_cache_size + data_size > self.max_cache_size {
            self.evict_oldest();
        }
        self.video_cache.insert(position, data);
        self.current_cache_size += data_size;
    }

    /// Gets a cached video frame.
    #[must_use]
    pub fn get_video_frame(&self, position: Position) -> Option<&Vec<u8>> {
        self.video_cache.get(&position)
    }

    /// Caches an audio buffer.
    pub fn cache_audio_buffer(&mut self, position: Position, data: Vec<f32>) {
        let data_size = data.len() * std::mem::size_of::<f32>();
        if self.current_cache_size + data_size > self.max_cache_size {
            self.evict_oldest();
        }
        self.audio_cache.insert(position, data);
        self.current_cache_size += data_size;
    }

    /// Gets a cached audio buffer.
    #[must_use]
    pub fn get_audio_buffer(&self, position: Position) -> Option<&Vec<f32>> {
        self.audio_cache.get(&position)
    }

    /// Clears the cache.
    pub fn clear(&mut self) {
        self.video_cache.clear();
        self.audio_cache.clear();
        self.current_cache_size = 0;
    }

    /// Evicts oldest cache entries.
    fn evict_oldest(&mut self) {
        // Simple eviction: remove half the cache
        let target_size = self.max_cache_size / 2;
        while self.current_cache_size > target_size {
            if let Some((pos, _)) = self.video_cache.iter().next() {
                let pos = *pos;
                if let Some(data) = self.video_cache.remove(&pos) {
                    self.current_cache_size -= data.len();
                }
            } else if let Some((pos, _)) = self.audio_cache.iter().next() {
                let pos = *pos;
                if let Some(data) = self.audio_cache.remove(&pos) {
                    self.current_cache_size -= data.len() * std::mem::size_of::<f32>();
                }
            } else {
                break;
            }
        }
    }

    /// Returns cache statistics.
    #[must_use]
    pub fn stats(&self) -> RenderCacheStats {
        RenderCacheStats {
            video_frames: self.video_cache.len(),
            audio_buffers: self.audio_cache.len(),
            total_size: self.current_cache_size,
            max_size: self.max_cache_size,
        }
    }
}

/// Render cache statistics.
#[derive(Clone, Debug)]
pub struct RenderCacheStats {
    /// Number of cached video frames.
    pub video_frames: usize,
    /// Number of cached audio buffers.
    pub audio_buffers: usize,
    /// Total cache size in bytes.
    pub total_size: usize,
    /// Maximum cache size in bytes.
    pub max_size: usize,
}

impl RenderCacheStats {
    /// Calculates cache usage percentage.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn usage_percentage(&self) -> f64 {
        if self.max_size == 0 {
            return 0.0;
        }
        (self.total_size as f64 / self.max_size as f64) * 100.0
    }
}

/// Renderer for timeline.
#[allow(dead_code)]
pub struct TimelineRenderer {
    /// Timeline to render.
    timeline: Arc<Timeline>,
    /// Render cache.
    cache: RenderCache,
    /// Render settings.
    settings: RenderSettings,
    /// Progress callback: called with (`current_frame`, `total_frames`).
    progress_callback: Option<Box<dyn Fn(u64, u64) + Send>>,
}

impl TimelineRenderer {
    /// Creates a new timeline renderer.
    #[must_use]
    pub fn new(timeline: Arc<Timeline>, settings: RenderSettings) -> Self {
        Self {
            timeline,
            cache: RenderCache::new(1024 * 1024 * 1024), // 1GB default cache
            settings,
            progress_callback: None,
        }
    }

    /// Sets a progress callback called with (`current_frame`, `total_frames`).
    pub fn set_progress_callback<F>(&mut self, callback: F)
    where
        F: Fn(u64, u64) + Send + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
    }

    /// Renders a single frame at a position.
    ///
    /// Composites all video clips that overlap `position` from bottom to top
    /// using simple alpha blending. Each clip is filled with a deterministic
    /// gradient colour derived from the clip name hash.
    ///
    /// # Errors
    ///
    /// Returns error if rendering fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn render_frame(&mut self, position: Position) -> TimelineResult<Vec<u8>> {
        // Check cache first
        if let Some(cached) = self.cache.get_video_frame(position) {
            return Ok(cached.clone());
        }

        // Determine output dimensions from settings.
        let (width, height) = match &self.settings.video_format {
            Some(RenderFormat::Video { width, height, .. }) => (*width, *height),
            _ => (1920u32, 1080u32),
        };

        let pixel_count = (width as usize) * (height as usize);
        // Start with a black transparent background.
        let mut composite = vec![0u8; pixel_count * 4];

        // Gather video clips that cover this position (bottom track first, then
        // higher z-index tracks on top).
        let timeline = Arc::clone(&self.timeline);
        let mut active_clips: Vec<(u8, u8, u8)> = Vec::new(); // (r, g, b) per clip layer

        let mut tracks: Vec<_> = timeline.video_tracks.iter().collect();
        tracks.sort_by_key(|t| t.z_index);

        for track in &tracks {
            if track.muted {
                continue;
            }
            for clip in &track.clips {
                if !clip.enabled {
                    continue;
                }
                let clip_in = clip.timeline_in;
                let clip_out = clip.timeline_out();
                if position >= clip_in && position < clip_out {
                    // Derive a deterministic colour from the clip name.
                    let mut hasher = DefaultHasher::new();
                    clip.name.hash(&mut hasher);
                    clip.id.as_uuid().hash(&mut hasher);
                    let hash = hasher.finish();
                    let r = ((hash >> 16) & 0xFF) as u8;
                    let g = ((hash >> 8) & 0xFF) as u8;
                    let b = (hash & 0xFF) as u8;
                    active_clips.push((r, g, b));
                }
            }
        }

        if active_clips.is_empty() {
            // Render black frame.
            for i in (0..composite.len()).step_by(4) {
                composite[i] = 0;
                composite[i + 1] = 0;
                composite[i + 2] = 0;
                composite[i + 3] = 255;
            }
        } else {
            // Fill base with the bottom-most clip colour, then alpha-blend upper clips.
            let (base_r, base_g, base_b) = active_clips[0];
            for px in 0..pixel_count {
                let i = px * 4;
                // Gradient: vary brightness left-to-right.
                let x = (px % width as usize) as f64 / f64::from(width);
                let factor = 0.6 + 0.4 * x;
                composite[i] = (f64::from(base_r) * factor) as u8;
                composite[i + 1] = (f64::from(base_g) * factor) as u8;
                composite[i + 2] = (f64::from(base_b) * factor) as u8;
                composite[i + 3] = 255;
            }

            // Composite subsequent layers with 50% alpha each.
            for &(r, g, b) in active_clips.iter().skip(1) {
                for px in 0..pixel_count {
                    let i = px * 4;
                    let x = (px % width as usize) as f64 / f64::from(width);
                    let factor = 0.6 + 0.4 * x;
                    let src_r = (f64::from(r) * factor) as u8;
                    let src_g = (f64::from(g) * factor) as u8;
                    let src_b = (f64::from(b) * factor) as u8;
                    // Simple 50% alpha blend (over operator, src_alpha = 0.5).
                    composite[i] = ((u16::from(src_r) + u16::from(composite[i])) / 2) as u8;
                    composite[i + 1] = ((u16::from(src_g) + u16::from(composite[i + 1])) / 2) as u8;
                    composite[i + 2] = ((u16::from(src_b) + u16::from(composite[i + 2])) / 2) as u8;
                    // Alpha stays fully opaque.
                }
            }
        }

        self.cache.cache_video_frame(position, composite.clone());
        Ok(composite)
    }

    /// Renders audio samples at a position.
    ///
    /// Mixes all audio clips that overlap `position` by simple addition,
    /// clamping the result to [-1.0, 1.0].
    ///
    /// # Errors
    ///
    /// Returns error if rendering fails.
    pub fn render_audio(
        &mut self,
        position: Position,
        sample_count: usize,
    ) -> TimelineResult<Vec<f32>> {
        // Check cache first
        if let Some(cached) = self.cache.get_audio_buffer(position) {
            return Ok(cached.clone());
        }

        let mut mix = vec![0.0f32; sample_count];

        let timeline = Arc::clone(&self.timeline);

        for track in &timeline.audio_tracks {
            if track.muted {
                continue;
            }
            for clip in &track.clips {
                if !clip.enabled {
                    continue;
                }
                let clip_in = clip.timeline_in;
                let clip_out = clip.timeline_out();
                if position >= clip_in && position < clip_out {
                    // Generate a simple test tone (sine-like) based on clip name hash.
                    let mut hasher = DefaultHasher::new();
                    clip.name.hash(&mut hasher);
                    let hash = hasher.finish();
                    // Frequency index 0–7 → 220–880 Hz range (not real audio, just composition).
                    let freq_idx = (hash & 0x7) as usize;
                    let amplitude = 0.1f32; // Keep quiet when mixing.
                    for (i, sample) in mix.iter_mut().enumerate() {
                        // Pseudo-sinusoidal: alternate +/- amplitude with period based on freq_idx.
                        let period = 16 + freq_idx * 4;
                        let sign = if (i % period) < period / 2 {
                            1.0f32
                        } else {
                            -1.0f32
                        };
                        *sample += sign * amplitude;
                    }
                }
            }
        }

        // Clamp to [-1.0, 1.0].
        for s in &mut mix {
            *s = s.clamp(-1.0, 1.0);
        }

        self.cache.cache_audio_buffer(position, mix.clone());
        Ok(mix)
    }

    /// Renders the entire timeline, calling the progress callback after each frame.
    ///
    /// # Errors
    ///
    /// Returns error if rendering fails.
    #[allow(clippy::cast_sign_loss)]
    pub fn render_all(&mut self) -> TimelineResult<RenderProgress> {
        let start = self.settings.start;
        let end = self.settings.end;
        let total_frames = end.value() - start.value();
        let total_frames_u64 = total_frames.max(0) as u64;

        let mut rendered_frames = 0i64;
        for frame in start.value()..end.value() {
            let position = Position::new(frame);
            self.render_frame(position)?;
            rendered_frames += 1;

            // Call progress callback if set.
            if let Some(ref cb) = self.progress_callback {
                cb(rendered_frames as u64, total_frames_u64);
            }
        }

        Ok(RenderProgress {
            position: end,
            total_frames,
            rendered_frames,
            estimated_time_remaining: 0.0,
        })
    }

    /// Clears the render cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Gets cache statistics.
    #[must_use]
    pub fn cache_stats(&self) -> RenderCacheStats {
        self.cache.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::Rational;

    fn create_test_timeline() -> Arc<Timeline> {
        Arc::new(
            Timeline::new("Test", Rational::new(24, 1), 48000).expect("should succeed in test"),
        )
    }

    #[test]
    fn test_render_quality() {
        assert_eq!(RenderQuality::Draft, RenderQuality::Draft);
        assert_ne!(RenderQuality::Draft, RenderQuality::High);
    }

    #[test]
    fn test_render_settings() {
        let settings = RenderSettings::new(Position::new(0), Position::new(100))
            .with_quality(RenderQuality::High)
            .with_video_format(1920, 1080, "rgba".to_string());

        assert_eq!(settings.quality, RenderQuality::High);
        assert!(settings.video_format.is_some());
    }

    #[test]
    fn test_render_progress() {
        let progress = RenderProgress {
            position: Position::new(50),
            total_frames: 100,
            rendered_frames: 50,
            estimated_time_remaining: 10.0,
        };

        assert!((progress.percentage() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_render_cache() {
        let mut cache = RenderCache::new(1024);
        let data = vec![0u8; 256];
        cache.cache_video_frame(Position::new(0), data.clone());
        assert!(cache.get_video_frame(Position::new(0)).is_some());
        assert_eq!(
            cache
                .get_video_frame(Position::new(0))
                .expect("should succeed in test"),
            &data
        );
    }

    #[test]
    fn test_render_cache_eviction() {
        let mut cache = RenderCache::new(512);
        cache.cache_video_frame(Position::new(0), vec![0u8; 256]);
        cache.cache_video_frame(Position::new(1), vec![0u8; 256]);
        cache.cache_video_frame(Position::new(2), vec![0u8; 256]); // Should trigger eviction

        let stats = cache.stats();
        assert!(stats.total_size <= stats.max_size);
    }

    #[test]
    fn test_cache_stats() {
        let cache = RenderCache::new(1024);
        let stats = cache.stats();
        assert_eq!(stats.max_size, 1024);
        assert_eq!(stats.total_size, 0);
        assert!((stats.usage_percentage() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_timeline_renderer() {
        let timeline = create_test_timeline();
        let settings = RenderSettings::new(Position::new(0), Position::new(10));
        let mut renderer = TimelineRenderer::new(timeline, settings);

        assert!(renderer.render_frame(Position::new(0)).is_ok());
        assert!(renderer.render_audio(Position::new(0), 1024).is_ok());
    }

    #[test]
    fn test_render_frame_returns_rgba() {
        let timeline = create_test_timeline();
        let settings = RenderSettings::new(Position::new(0), Position::new(5)).with_video_format(
            320,
            240,
            "rgba".to_string(),
        );
        let mut renderer = TimelineRenderer::new(timeline, settings);

        let frame = renderer
            .render_frame(Position::new(0))
            .expect("should succeed in test");
        // 320 * 240 * 4 bytes
        assert_eq!(frame.len(), 320 * 240 * 4);
        // Fully opaque black frame (no clips).
        assert_eq!(frame[3], 255); // alpha
        assert_eq!(frame[0], 0); // red (black)
    }

    #[test]
    fn test_render_audio_clamped() {
        let timeline = create_test_timeline();
        let settings = RenderSettings::new(Position::new(0), Position::new(5));
        let mut renderer = TimelineRenderer::new(timeline, settings);

        let audio = renderer
            .render_audio(Position::new(0), 1024)
            .expect("should succeed in test");
        assert_eq!(audio.len(), 1024);
        // Silence for timeline with no clips.
        for s in &audio {
            assert!(*s >= -1.0 && *s <= 1.0, "Sample out of range: {s}");
        }
    }

    #[test]
    fn test_render_all_calls_progress_callback() {
        use std::sync::{Arc as StdArc, Mutex};

        let timeline = create_test_timeline();
        let settings = RenderSettings::new(Position::new(0), Position::new(5));
        let mut renderer = TimelineRenderer::new(Arc::clone(&timeline), settings);

        let calls = StdArc::new(Mutex::new(0u64));
        let calls_clone = StdArc::clone(&calls);
        renderer.set_progress_callback(move |current, total| {
            let mut c = calls_clone.lock().expect("should succeed in test");
            *c += 1;
            assert!(current <= total, "current > total");
        });

        let result = renderer.render_all();
        assert!(result.is_ok());
        let count = *calls.lock().expect("should succeed in test");
        assert_eq!(
            count, 5,
            "Progress callback should be called once per frame"
        );
    }

    #[test]
    fn test_render_frame_with_video_clip() {
        use crate::clip::{Clip, MediaSource};

        let mut tl =
            crate::timeline::Timeline::new("T", oximedia_core::Rational::new(24, 1), 48000)
                .expect("should succeed in test");
        let tid = tl.add_video_track("V1").expect("should succeed in test");
        let clip = Clip::new(
            "test_clip".to_string(),
            MediaSource::file(std::env::temp_dir().join("oximedia-timeline-render-test.mov")),
            Position::new(0),
            Position::new(50),
            Position::new(0),
        )
        .expect("should succeed in test");
        tl.add_clip(tid, clip).expect("should succeed in test");

        let timeline = Arc::new(tl);
        let settings = RenderSettings::new(Position::new(0), Position::new(50)).with_video_format(
            64,
            36,
            "rgba".to_string(),
        );
        let mut renderer = TimelineRenderer::new(timeline, settings);

        let frame = renderer
            .render_frame(Position::new(25))
            .expect("should succeed in test");
        assert_eq!(frame.len(), 64 * 36 * 4);
        // Should not be all black - clip contributes colour.
        let has_colour = frame
            .chunks(4)
            .any(|px| px[0] != 0 || px[1] != 0 || px[2] != 0);
        assert!(has_colour, "Frame should contain clip colour");
    }
}
