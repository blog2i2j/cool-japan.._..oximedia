//! Game capture extensions: delta detection, replay simple API, highlight callbacks,
//! chapter marking, platform stream configs, scene transitions, game overlay,
//! audio mixer, input recorder, and stream metrics.

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

use crate::{GamingError, GamingResult};
use std::collections::HashMap;
use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// DirtyBlock / RegionCapture::capture_delta
// ─────────────────────────────────────────────────────────────────────────────

/// A block of the screen that changed between two frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirtyBlock {
    /// Horizontal block index (block_x * block_size = pixel x).
    pub block_x: u32,
    /// Vertical block index.
    pub block_y: u32,
    /// Width of the block in pixels (may be smaller at the right/bottom edge).
    pub block_w: u32,
    /// Height of the block in pixels.
    pub block_h: u32,
}

/// Scan two raw RGBA frames and return the set of blocks that changed.
///
/// `prev` and `curr` must both contain `w * h * 4` bytes of RGBA data.
/// `block_size` is the side length of each comparison block in pixels.
/// A block is "dirty" when any pixel within it differs between the two frames.
///
/// Returns an empty `Vec` when the frames are identical, the inputs are empty,
/// or `block_size` is zero.
#[must_use]
pub fn capture_delta(prev: &[u8], curr: &[u8], w: u32, h: u32, block_size: u32) -> Vec<DirtyBlock> {
    if prev.len() != curr.len() || prev.is_empty() || block_size == 0 || w == 0 || h == 0 {
        return Vec::new();
    }
    let expected = (w as usize) * (h as usize) * 4;
    if prev.len() != expected || curr.len() != expected {
        return Vec::new();
    }

    let blocks_x = (w + block_size - 1) / block_size;
    let blocks_y = (h + block_size - 1) / block_size;
    let mut dirty = Vec::new();

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let px_start = bx * block_size;
            let py_start = by * block_size;
            let px_end = (px_start + block_size).min(w);
            let py_end = (py_start + block_size).min(h);

            'block: for py in py_start..py_end {
                for px in px_start..px_end {
                    let idx = ((py * w + px) * 4) as usize;
                    // Compare all 4 channels
                    if prev[idx] != curr[idx]
                        || prev[idx + 1] != curr[idx + 1]
                        || prev[idx + 2] != curr[idx + 2]
                        || prev[idx + 3] != curr[idx + 3]
                    {
                        dirty.push(DirtyBlock {
                            block_x: bx,
                            block_y: by,
                            block_w: px_end - px_start,
                            block_h: py_end - py_start,
                        });
                        break 'block;
                    }
                }
            }
        }
    }
    dirty
}

// ─────────────────────────────────────────────────────────────────────────────
// Simple ReplayBuffer (capacity-based, frame-bytes ring buffer)
// ─────────────────────────────────────────────────────────────────────────────

/// Simple ring-buffer replay buffer keyed by duration + fps rather than byte budget.
///
/// Stores raw frame bytes; wraps around at capacity.
pub struct SimpleReplayBuffer {
    /// Maximum number of frames (capacity_seconds * fps).
    capacity: usize,
    /// Ring buffer of frames.
    frames: VecDeque<Vec<u8>>,
    /// Target frames per second (used for time-based export).
    fps: f32,
}

impl SimpleReplayBuffer {
    /// Create a new replay buffer.
    ///
    /// # Errors
    ///
    /// Returns an error when `capacity_seconds <= 0`, `fps <= 0`, or
    /// `capacity_seconds * fps` overflows a reasonable usize limit.
    pub fn new(capacity_seconds: f32, fps: f32) -> GamingResult<Self> {
        if capacity_seconds <= 0.0 {
            return Err(GamingError::ReplayBufferError(
                "capacity_seconds must be positive".to_string(),
            ));
        }
        if fps <= 0.0 {
            return Err(GamingError::ReplayBufferError(
                "fps must be positive".to_string(),
            ));
        }
        let capacity = (capacity_seconds * fps).ceil() as usize;
        if capacity == 0 {
            return Err(GamingError::ReplayBufferError(
                "computed capacity is zero".to_string(),
            ));
        }
        Ok(Self {
            capacity,
            frames: VecDeque::with_capacity(capacity.min(32_768)),
            fps,
        })
    }

    /// Push a new frame, evicting the oldest if the buffer is full.
    pub fn push_frame(&mut self, frame: Vec<u8>) {
        if self.frames.len() >= self.capacity {
            self.frames.pop_front();
        }
        self.frames.push_back(frame);
    }

    /// Export the last `secs` seconds of frames (most recent at the end).
    ///
    /// Returns at most `ceil(secs * fps)` frames.
    #[must_use]
    pub fn export_last_n_seconds(&self, secs: f32) -> Vec<Vec<u8>> {
        if secs <= 0.0 || self.frames.is_empty() {
            return Vec::new();
        }
        let n = (secs * self.fps).ceil() as usize;
        let skip = self.frames.len().saturating_sub(n);
        self.frames.iter().skip(skip).cloned().collect()
    }

    /// Number of frames currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` when no frames are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Maximum number of frames the buffer can hold.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HighlightDetector (event-driven, game-events focused)
// ─────────────────────────────────────────────────────────────────────────────

use crate::game_event::GameEvent;

/// Event-driven highlight detector that reacts to audio spikes and game events.
#[derive(Debug, Clone)]
pub struct EventHighlightDetector {
    /// Timestamp of the most recently detected highlight, in ms.
    pub last_highlight_ms: Option<u64>,
    /// Number of highlights detected in this session.
    pub highlight_count: u64,
}

impl EventHighlightDetector {
    /// Create a new event-driven highlight detector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            last_highlight_ms: None,
            highlight_count: 0,
        }
    }

    /// Returns `true` and records a highlight when the RMS audio level exceeds `threshold`.
    pub fn on_audio_spike(&mut self, rms: f32, threshold: f32) -> bool {
        if rms > threshold {
            self.highlight_count += 1;
            true
        } else {
            false
        }
    }

    /// Returns `true` and records a highlight when the game event is a kill or multi-kill.
    pub fn on_kill_event(&mut self, event: &GameEvent) -> bool {
        use crate::game_event::GameEventType;
        let is_kill = matches!(
            event.event_type,
            GameEventType::Kill | GameEventType::MultiKill(_)
        );
        if is_kill {
            self.highlight_count += 1;
        }
        is_kill
    }
}

impl Default for EventHighlightDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ChapterMarker
// ─────────────────────────────────────────────────────────────────────────────

/// A single chapter entry.
#[derive(Debug, Clone)]
pub struct Chapter {
    /// Timestamp in milliseconds from the start of the recording.
    pub timestamp_ms: u64,
    /// Human-readable chapter title.
    pub title: String,
}

/// Maintains an ordered list of chapters and can export them in FFmpeg metadata format.
#[derive(Debug, Clone, Default)]
pub struct ChapterMarker {
    chapters: Vec<Chapter>,
}

impl ChapterMarker {
    /// Create a new, empty chapter marker list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a chapter at `timestamp_ms` with the given `title`.
    ///
    /// Chapters are kept sorted by timestamp.
    pub fn add_chapter(&mut self, timestamp_ms: u64, title: &str) {
        self.chapters.push(Chapter {
            timestamp_ms,
            title: title.to_string(),
        });
        self.chapters.sort_by_key(|c| c.timestamp_ms);
    }

    /// Return the number of chapters.
    #[must_use]
    pub fn chapter_count(&self) -> usize {
        self.chapters.len()
    }

    /// Produce a string in FFmpeg's `ffmetadata` chapter format.
    ///
    /// The format is:
    /// ```text
    /// ;FFMETADATA1
    /// [CHAPTER]
    /// TIMEBASE=1/1000
    /// START=<ms>
    /// END=<next_ms - 1>
    /// title=<title>
    /// ```
    #[must_use]
    pub fn to_ffmetadata(&self) -> String {
        let mut out = String::from(";FFMETADATA1\n");

        for (i, ch) in self.chapters.iter().enumerate() {
            let end = if i + 1 < self.chapters.len() {
                self.chapters[i + 1].timestamp_ms.saturating_sub(1)
            } else {
                // Last chapter: use start + 30 000 ms as a generous end sentinel
                ch.timestamp_ms + 30_000
            };

            out.push_str("[CHAPTER]\n");
            out.push_str("TIMEBASE=1/1000\n");
            out.push_str(&format!("START={}\n", ch.timestamp_ms));
            out.push_str(&format!("END={end}\n"));
            out.push_str(&format!("title={}\n", ch.title));
        }

        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Platform stream configs
// ─────────────────────────────────────────────────────────────────────────────

/// Twitch stream configuration (simple, key-based variant used for output).
#[derive(Debug, Clone)]
pub struct TwitchStreamConfig {
    /// RTMP stream key.
    pub stream_key: String,
    /// Twitch ingest server URL.
    pub server_url: String,
    /// Target video bitrate in kbps.
    pub bitrate_kbps: u32,
}

/// YouTube Live stream configuration (simple, key-based variant used for output).
#[derive(Debug, Clone)]
pub struct YouTubeStreamConfig {
    /// RTMP stream key.
    pub stream_key: String,
    /// Video resolution string (e.g. `"1920x1080"`).
    pub resolution: String,
}

/// Marker trait for stream platform configs.
pub trait StreamConfig: std::fmt::Debug {
    /// Validate the configuration, returning `Ok(())` when valid.
    fn validate(&self) -> GamingResult<()>;
    /// Display name of this platform.
    fn platform_name(&self) -> &'static str;
}

impl StreamConfig for TwitchStreamConfig {
    fn validate(&self) -> GamingResult<()> {
        if self.stream_key.is_empty() {
            return Err(GamingError::InvalidConfig(
                "Twitch stream_key must not be empty".to_string(),
            ));
        }
        if self.server_url.is_empty() {
            return Err(GamingError::InvalidConfig(
                "Twitch server_url must not be empty".to_string(),
            ));
        }
        if self.bitrate_kbps == 0 {
            return Err(GamingError::InvalidConfig(
                "Twitch bitrate_kbps must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    fn platform_name(&self) -> &'static str {
        "Twitch"
    }
}

impl StreamConfig for YouTubeStreamConfig {
    fn validate(&self) -> GamingResult<()> {
        if self.stream_key.is_empty() {
            return Err(GamingError::InvalidConfig(
                "YouTube stream_key must not be empty".to_string(),
            ));
        }
        if self.resolution.is_empty() {
            return Err(GamingError::InvalidConfig(
                "YouTube resolution must not be empty".to_string(),
            ));
        }
        Ok(())
    }

    fn platform_name(&self) -> &'static str {
        "YouTube"
    }
}

/// Validates a generic stream platform config.
///
/// # Errors
///
/// Returns the inner validation error when the config is invalid.
pub fn validate_stream_config(config: &dyn StreamConfig) -> GamingResult<()> {
    config.validate()
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene transitions
// ─────────────────────────────────────────────────────────────────────────────

/// A scene transition event produced by `SceneTransitionBuilder`.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionEvent {
    /// Source scene name.
    pub from_scene: String,
    /// Destination scene name.
    pub to_scene: String,
    /// Type of transition.
    pub transition_type: GameTransitionType,
    /// Number of frames over which the transition occurs (0 for cut).
    pub frames: u32,
}

/// Type of scene transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameTransitionType {
    /// Instantaneous cut.
    Cut,
    /// Dissolve / cross-fade.
    Dissolve,
}

/// Factory for creating `TransitionEvent` values.
pub struct SceneTransitionBuilder;

impl SceneTransitionBuilder {
    /// Create an instant cut transition.
    #[must_use]
    pub fn cut(from_scene: &str, to_scene: &str) -> TransitionEvent {
        TransitionEvent {
            from_scene: from_scene.to_string(),
            to_scene: to_scene.to_string(),
            transition_type: GameTransitionType::Cut,
            frames: 0,
        }
    }

    /// Create a dissolve (cross-fade) transition over `frames` frames.
    #[must_use]
    pub fn dissolve(from: &str, to: &str, frames: u32) -> TransitionEvent {
        TransitionEvent {
            from_scene: from.to_string(),
            to_scene: to.to_string(),
            transition_type: GameTransitionType::Dissolve,
            frames,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GameOverlay
// ─────────────────────────────────────────────────────────────────────────────

/// A text element placed on the overlay.
#[derive(Debug, Clone)]
struct TextElement {
    x: u32,
    y: u32,
    text: String,
    color: [u8; 4],
}

/// An image element placed on the overlay.
#[derive(Debug, Clone)]
struct ImageElement {
    x: u32,
    y: u32,
    rgba: Vec<u8>,
    w: u32,
    h: u32,
}

/// Simple RGBA overlay compositor.
pub struct GameOverlay {
    width: u32,
    height: u32,
    text_elements: Vec<TextElement>,
    image_elements: Vec<ImageElement>,
}

impl GameOverlay {
    /// Create a new overlay for a canvas of `width × height` pixels.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            text_elements: Vec::new(),
            image_elements: Vec::new(),
        }
    }

    /// Add a text element at `(x, y)` with the given RGBA `color`.
    ///
    /// Text is rendered as a single block glyph per character (8×8 pixels)
    /// for simplicity.
    pub fn add_text(&mut self, x: u32, y: u32, text: &str, color: [u8; 4]) {
        self.text_elements.push(TextElement {
            x,
            y,
            text: text.to_string(),
            color,
        });
    }

    /// Add an RGBA image at `(x, y)`.  `rgba` must be `w * h * 4` bytes.
    pub fn add_image(&mut self, x: u32, y: u32, rgba: &[u8], w: u32, h: u32) {
        self.image_elements.push(ImageElement {
            x,
            y,
            rgba: rgba.to_vec(),
            w,
            h,
        });
    }

    /// Composite all overlay elements onto `bg` (a `width × height × 4` RGBA buffer)
    /// using straight-alpha compositing and return the result.
    ///
    /// Elements are composited in insertion order (text first, then images).
    /// The input `bg` buffer must be exactly `self.width * self.height * 4` bytes.
    /// If `bg` is the wrong size, a black frame is returned instead.
    #[must_use]
    pub fn render_to_buffer(&self, bg: &[u8]) -> Vec<u8> {
        let expected = (self.width as usize) * (self.height as usize) * 4;
        let mut out = if bg.len() == expected {
            bg.to_vec()
        } else {
            vec![0u8; expected]
        };

        // Render text blocks (8×8 block per character)
        const GLYPH_W: u32 = 8;
        const GLYPH_H: u32 = 8;
        for elem in &self.text_elements {
            for (ci, _ch) in elem.text.chars().enumerate() {
                let gx = elem.x + (ci as u32) * (GLYPH_W + 1);
                let gy = elem.y;
                for dy in 0..GLYPH_H {
                    for dx in 0..GLYPH_W {
                        let px = gx + dx;
                        let py = gy + dy;
                        if px < self.width && py < self.height {
                            let idx = ((py * self.width + px) * 4) as usize;
                            if idx + 3 < out.len() {
                                Self::blend_straight(&mut out[idx..idx + 4], elem.color);
                            }
                        }
                    }
                }
            }
        }

        // Render images
        for img in &self.image_elements {
            for dy in 0..img.h {
                for dx in 0..img.w {
                    let src_idx = ((dy * img.w + dx) * 4) as usize;
                    if src_idx + 3 >= img.rgba.len() {
                        continue;
                    }
                    let src = [
                        img.rgba[src_idx],
                        img.rgba[src_idx + 1],
                        img.rgba[src_idx + 2],
                        img.rgba[src_idx + 3],
                    ];
                    let px = img.x + dx;
                    let py = img.y + dy;
                    if px < self.width && py < self.height {
                        let dst_idx = ((py * self.width + px) * 4) as usize;
                        if dst_idx + 3 < out.len() {
                            Self::blend_straight(&mut out[dst_idx..dst_idx + 4], src);
                        }
                    }
                }
            }
        }

        out
    }

    /// Straight-alpha blend: dst = src_alpha * src + (1 - src_alpha) * dst
    #[inline]
    fn blend_straight(dst: &mut [u8], src: [u8; 4]) {
        let sa = src[3] as f32 / 255.0;
        let da = 1.0 - sa;
        dst[0] = ((src[0] as f32 * sa) + (dst[0] as f32 * da)).min(255.0) as u8;
        dst[1] = ((src[1] as f32 * sa) + (dst[1] as f32 * da)).min(255.0) as u8;
        dst[2] = ((src[2] as f32 * sa) + (dst[2] as f32 * da)).min(255.0) as u8;
        dst[3] = 255;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GameAudioMixer
// ─────────────────────────────────────────────────────────────────────────────

/// A source entry in the game audio mixer.
#[derive(Debug, Clone)]
pub struct MixerSource {
    /// Numeric identifier for the source.
    pub id: u32,
    /// Volume in `[0.0, ∞)` — typically 0.0–1.0.
    pub volume: f32,
}

/// Simple weighted-sum audio mixer for game sources.
pub struct GameAudioMixer {
    sources: Vec<MixerSource>,
}

impl GameAudioMixer {
    /// Create a new mixer with no sources.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    /// Add a source with the given `id` and `volume`.
    ///
    /// If a source with this `id` already exists, its volume is updated.
    pub fn add_source(&mut self, id: u32, volume: f32) {
        if let Some(s) = self.sources.iter_mut().find(|s| s.id == id) {
            s.volume = volume;
        } else {
            self.sources.push(MixerSource { id, volume });
        }
    }

    /// Remove the source with the given `id`.
    pub fn remove_source(&mut self, id: u32) {
        self.sources.retain(|s| s.id != id);
    }

    /// Mix all sources by computing a weighted sum of the provided sample slices.
    ///
    /// `sources` maps source ID to a slice of `f32` samples.  Only sources that
    /// are registered in the mixer are mixed; unknown IDs are silently ignored.
    /// The output length equals the length of the longest slice.
    ///
    /// Clipping is **not** applied — the caller is responsible for normalisation
    /// if needed.
    #[must_use]
    pub fn mix_sources(&self, sources: &HashMap<u32, &[f32]>) -> Vec<f32> {
        let max_len = sources.values().map(|s| s.len()).max().unwrap_or(0);

        if max_len == 0 {
            return Vec::new();
        }

        let mut out = vec![0.0_f32; max_len];

        for src in &self.sources {
            if let Some(samples) = sources.get(&src.id) {
                let vol = src.volume;
                for (i, &s) in samples.iter().enumerate() {
                    out[i] += s * vol;
                }
            }
        }

        out
    }
}

impl Default for GameAudioMixer {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InputRecorder
// ─────────────────────────────────────────────────────────────────────────────

/// A single recorded input event.
#[derive(Debug, Clone)]
pub struct InputEntry {
    /// Key or button code.
    pub key_code: u32,
    /// `true` = pressed, `false` = released.
    pub pressed: bool,
    /// Timestamp in milliseconds.
    pub timestamp_ms: u64,
}

/// Records keyboard / button input events and can export them as a CSV replay script.
#[derive(Debug, Clone, Default)]
pub struct InputRecorder {
    entries: Vec<InputEntry>,
}

impl InputRecorder {
    /// Create a new, empty recorder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a key press or release event.
    pub fn record_key(&mut self, key_code: u32, pressed: bool, timestamp_ms: u64) {
        self.entries.push(InputEntry {
            key_code,
            pressed,
            timestamp_ms,
        });
    }

    /// Export all recorded events as a CSV replay script.
    ///
    /// Format:
    /// ```text
    /// timestamp_ms,key_code,action
    /// 100,65,pressed
    /// 150,65,released
    /// ```
    #[must_use]
    pub fn to_replay_script(&self) -> String {
        let mut out = String::from("timestamp_ms,key_code,action\n");
        for e in &self.entries {
            let action = if e.pressed { "pressed" } else { "released" };
            out.push_str(&format!("{},{},{}\n", e.timestamp_ms, e.key_code, action));
        }
        out
    }

    /// Number of recorded events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.entries.len()
    }

    /// Clear all recorded events.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamMetrics
// ─────────────────────────────────────────────────────────────────────────────

/// Real-time stream health metrics with computed health score.
#[derive(Debug, Clone, Default)]
pub struct StreamMetrics {
    /// Current frames-per-second.
    pub fps: f32,
    /// Current bitrate in kbps.
    pub bitrate_kbps: u32,
    /// Dropped frame count in the current measurement window.
    pub dropped_frames: u32,
    /// Total frames sent in the current measurement window.
    pub total_frames: u32,
    /// Target FPS as configured.
    pub target_fps: f32,
}

impl StreamMetrics {
    /// Create a new `StreamMetrics` with `target_fps` configured.
    #[must_use]
    pub fn new(target_fps: f32) -> Self {
        Self {
            target_fps,
            ..Default::default()
        }
    }

    /// Record a new measurement.
    pub fn update(&mut self, fps: f32, bitrate_kbps: u32, dropped_frames: u32) {
        self.fps = fps;
        self.bitrate_kbps = bitrate_kbps;
        self.dropped_frames += dropped_frames;
        // Approximate total frames from fps and a 1-second window
        self.total_frames += fps.round() as u32;
    }

    /// Compute a composite health score in `[0.0, 1.0]`.
    ///
    /// Formula:
    /// `(fps / target_fps * 0.5 + (1.0 - dropped_ratio) * 0.5).min(1.0)`
    ///
    /// Where `dropped_ratio = dropped_frames / total_frames`.
    #[must_use]
    pub fn health_score(&self) -> f32 {
        let fps_score = if self.target_fps > 0.0 {
            (self.fps / self.target_fps).min(1.0) * 0.5
        } else {
            0.5
        };

        let drop_ratio = if self.total_frames > 0 {
            self.dropped_frames as f32 / self.total_frames as f32
        } else {
            0.0
        };

        let drop_score = (1.0 - drop_ratio).clamp(0.0, 1.0) * 0.5;

        (fps_score + drop_score).min(1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game_event::{GameEvent, GameEventType};

    // ── capture_delta ────────────────────────────────────────────────────────

    #[test]
    fn test_capture_delta_identical_frames_no_dirty() {
        let frame = vec![128u8; 4 * 4 * 4]; // 4×4 RGBA
        let dirty = capture_delta(&frame, &frame, 4, 4, 2);
        assert!(dirty.is_empty());
    }

    #[test]
    fn test_capture_delta_changed_single_pixel() {
        let w = 4u32;
        let h = 4u32;
        let mut prev = vec![0u8; (w * h * 4) as usize];
        let mut curr = prev.clone();
        // Change pixel (2, 1)
        let idx = ((1 * w + 2) * 4) as usize;
        curr[idx] = 255;
        let dirty = capture_delta(&prev, &curr, w, h, 2);
        // Block (1, 0) should be dirty
        assert!(!dirty.is_empty());
        let b = dirty[0];
        assert_eq!(b.block_x, 1);
        assert_eq!(b.block_y, 0);

        // Swap back: identical → no dirty
        prev = curr.clone();
        let dirty2 = capture_delta(&prev, &curr, w, h, 2);
        assert!(dirty2.is_empty());
    }

    #[test]
    fn test_capture_delta_empty_inputs() {
        let dirty = capture_delta(&[], &[], 0, 0, 8);
        assert!(dirty.is_empty());
    }

    #[test]
    fn test_capture_delta_all_blocks_dirty() {
        let prev = vec![0u8; 4 * 4 * 4];
        let curr = vec![255u8; 4 * 4 * 4];
        let dirty = capture_delta(&prev, &curr, 4, 4, 2);
        // 2×2 = 4 blocks, all dirty
        assert_eq!(dirty.len(), 4);
    }

    // ── SimpleReplayBuffer ───────────────────────────────────────────────────

    #[test]
    fn test_replay_buffer_wraps_at_capacity() {
        let mut rb = SimpleReplayBuffer::new(1.0, 3.0).expect("valid"); // capacity = 3
        for i in 0u8..5 {
            rb.push_frame(vec![i]);
        }
        // After 5 pushes into capacity 3, frames should be [2,3,4]
        assert_eq!(rb.len(), 3);
        let exported = rb.export_last_n_seconds(1.0);
        assert_eq!(exported.len(), 3);
        assert_eq!(exported[0], vec![2]);
        assert_eq!(exported[2], vec![4]);
    }

    #[test]
    fn test_replay_buffer_export_partial() {
        let mut rb = SimpleReplayBuffer::new(10.0, 2.0).expect("valid"); // capacity = 20
        for i in 0u8..10 {
            rb.push_frame(vec![i]);
        }
        // Export last 2 seconds at 2fps → ceil(2*2)=4 frames
        let exp = rb.export_last_n_seconds(2.0);
        assert_eq!(exp.len(), 4);
        assert_eq!(exp[0], vec![6]);
    }

    #[test]
    fn test_replay_buffer_invalid_params() {
        assert!(SimpleReplayBuffer::new(0.0, 30.0).is_err());
        assert!(SimpleReplayBuffer::new(10.0, 0.0).is_err());
        assert!(SimpleReplayBuffer::new(-1.0, 30.0).is_err());
    }

    // ── EventHighlightDetector ───────────────────────────────────────────────

    #[test]
    fn test_highlight_audio_spike_above_threshold() {
        let mut det = EventHighlightDetector::new();
        assert!(det.on_audio_spike(0.9, 0.8));
        assert_eq!(det.highlight_count, 1);
    }

    #[test]
    fn test_highlight_audio_spike_below_threshold() {
        let mut det = EventHighlightDetector::new();
        assert!(!det.on_audio_spike(0.5, 0.8));
        assert_eq!(det.highlight_count, 0);
    }

    #[test]
    fn test_highlight_on_kill_event() {
        let mut det = EventHighlightDetector::new();
        let event = GameEvent::new(GameEventType::Kill);
        assert!(det.on_kill_event(&event));
        assert_eq!(det.highlight_count, 1);
    }

    #[test]
    fn test_highlight_on_death_event_not_kill() {
        let mut det = EventHighlightDetector::new();
        let event = GameEvent::new(GameEventType::Death);
        assert!(!det.on_kill_event(&event));
        assert_eq!(det.highlight_count, 0);
    }

    #[test]
    fn test_highlight_on_multi_kill() {
        let mut det = EventHighlightDetector::new();
        let event = GameEvent::new(GameEventType::MultiKill(4));
        assert!(det.on_kill_event(&event));
        assert_eq!(det.highlight_count, 1);
    }

    // ── ChapterMarker ────────────────────────────────────────────────────────

    #[test]
    fn test_chapter_marker_add_and_count() {
        let mut cm = ChapterMarker::new();
        cm.add_chapter(0, "Intro");
        cm.add_chapter(60_000, "Battle");
        assert_eq!(cm.chapter_count(), 2);
    }

    #[test]
    fn test_chapter_marker_to_ffmetadata_contains_required_fields() {
        let mut cm = ChapterMarker::new();
        cm.add_chapter(0, "Start");
        cm.add_chapter(30_000, "Middle");
        let meta = cm.to_ffmetadata();
        assert!(meta.contains(";FFMETADATA1"));
        assert!(meta.contains("[CHAPTER]"));
        assert!(meta.contains("TIMEBASE=1/1000"));
        assert!(meta.contains("START=0"));
        assert!(meta.contains("title=Start"));
        assert!(meta.contains("START=30000"));
        assert!(meta.contains("title=Middle"));
    }

    #[test]
    fn test_chapter_marker_sorted_by_timestamp() {
        let mut cm = ChapterMarker::new();
        cm.add_chapter(5000, "Late");
        cm.add_chapter(1000, "Early");
        let meta = cm.to_ffmetadata();
        let early_pos = meta.find("title=Early").expect("Early chapter present");
        let late_pos = meta.find("title=Late").expect("Late chapter present");
        assert!(early_pos < late_pos);
    }

    // ── Platform configs ─────────────────────────────────────────────────────

    #[test]
    fn test_twitch_config_valid() {
        let cfg = TwitchStreamConfig {
            stream_key: "live_key".to_string(),
            server_url: "rtmp://live.twitch.tv/app".to_string(),
            bitrate_kbps: 6000,
        };
        assert!(validate_stream_config(&cfg).is_ok());
    }

    #[test]
    fn test_twitch_config_empty_key_invalid() {
        let cfg = TwitchStreamConfig {
            stream_key: String::new(),
            server_url: "rtmp://live.twitch.tv/app".to_string(),
            bitrate_kbps: 6000,
        };
        assert!(validate_stream_config(&cfg).is_err());
    }

    #[test]
    fn test_youtube_config_valid() {
        let cfg = YouTubeStreamConfig {
            stream_key: "yt-key".to_string(),
            resolution: "1920x1080".to_string(),
        };
        assert!(validate_stream_config(&cfg).is_ok());
    }

    #[test]
    fn test_youtube_config_empty_resolution_invalid() {
        let cfg = YouTubeStreamConfig {
            stream_key: "yt-key".to_string(),
            resolution: String::new(),
        };
        assert!(validate_stream_config(&cfg).is_err());
    }

    // ── SceneTransitionBuilder ───────────────────────────────────────────────

    #[test]
    fn test_scene_cut_transition() {
        let ev = SceneTransitionBuilder::cut("main", "game");
        assert_eq!(ev.from_scene, "main");
        assert_eq!(ev.to_scene, "game");
        assert_eq!(ev.transition_type, GameTransitionType::Cut);
        assert_eq!(ev.frames, 0);
    }

    #[test]
    fn test_scene_dissolve_transition() {
        let ev = SceneTransitionBuilder::dissolve("game", "outro", 30);
        assert_eq!(ev.transition_type, GameTransitionType::Dissolve);
        assert_eq!(ev.frames, 30);
    }

    // ── GameOverlay ──────────────────────────────────────────────────────────

    #[test]
    fn test_game_overlay_alpha_composite() {
        let mut overlay = GameOverlay::new(8, 8);
        // Add a fully opaque red image pixel at (0,0)
        let red_pixel = vec![255u8, 0, 0, 255];
        overlay.add_image(0, 0, &red_pixel, 1, 1);

        // Background is all blue
        let bg = vec![0u8, 0, 255, 255].repeat(64);
        let out = overlay.render_to_buffer(&bg);

        // First pixel should be red (fully opaque overlay)
        assert_eq!(out[0], 255); // R
        assert_eq!(out[1], 0); // G
        assert_eq!(out[2], 0); // B
    }

    #[test]
    fn test_game_overlay_semi_transparent_image() {
        let mut overlay = GameOverlay::new(4, 4);
        // 50%-alpha white pixel
        let white_half = vec![255u8, 255, 255, 128];
        overlay.add_image(0, 0, &white_half, 1, 1);

        // Background: black
        let bg = vec![0u8; 4 * 4 * 4];
        let out = overlay.render_to_buffer(&bg);

        // The blended red channel should be ~128/2 ≈ 50 (not 0, not 255)
        assert!(out[0] > 0 && out[0] < 255);
    }

    #[test]
    fn test_game_overlay_text_renders_something() {
        let mut overlay = GameOverlay::new(64, 16);
        overlay.add_text(0, 0, "Hi", [255, 255, 255, 255]);
        let bg = vec![0u8; 64 * 16 * 4];
        let out = overlay.render_to_buffer(&bg);
        // At least one pixel should be non-zero
        assert!(out.iter().any(|&b| b > 0));
    }

    // ── GameAudioMixer ───────────────────────────────────────────────────────

    #[test]
    fn test_audio_mixer_weighted_sum() {
        let mut mixer = GameAudioMixer::new();
        mixer.add_source(1, 1.0);
        mixer.add_source(2, 0.5);

        let s1 = [1.0_f32, 1.0];
        let s2 = [1.0_f32, 1.0];
        let mut map: HashMap<u32, &[f32]> = HashMap::new();
        map.insert(1, &s1);
        map.insert(2, &s2);

        let out = mixer.mix_sources(&map);
        // Expected: 1.0*1.0 + 0.5*1.0 = 1.5 for each sample
        assert_eq!(out.len(), 2);
        assert!((out[0] - 1.5).abs() < 1e-6);
        assert!((out[1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_audio_mixer_unknown_source_ignored() {
        let mixer = GameAudioMixer::new();
        let s = [1.0_f32; 4];
        let mut map: HashMap<u32, &[f32]> = HashMap::new();
        map.insert(99, &s); // not registered
        let out = mixer.mix_sources(&map);
        // All zeros since no registered sources
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_audio_mixer_empty_sources() {
        let mixer = GameAudioMixer::new();
        let map: HashMap<u32, &[f32]> = HashMap::new();
        let out = mixer.mix_sources(&map);
        assert!(out.is_empty());
    }

    // ── InputRecorder ────────────────────────────────────────────────────────

    #[test]
    fn test_input_recorder_csv_format() {
        let mut recorder = InputRecorder::new();
        recorder.record_key(65, true, 100);
        recorder.record_key(65, false, 150);

        let script = recorder.to_replay_script();
        assert!(script.starts_with("timestamp_ms,key_code,action\n"));
        assert!(script.contains("100,65,pressed"));
        assert!(script.contains("150,65,released"));
    }

    #[test]
    fn test_input_recorder_count_and_clear() {
        let mut recorder = InputRecorder::new();
        recorder.record_key(1, true, 0);
        recorder.record_key(2, true, 100);
        assert_eq!(recorder.event_count(), 2);
        recorder.clear();
        assert_eq!(recorder.event_count(), 0);
    }

    // ── StreamMetrics ────────────────────────────────────────────────────────

    #[test]
    fn test_stream_metrics_perfect_health() {
        let mut m = StreamMetrics::new(60.0);
        m.update(60.0, 6000, 0);
        let score = m.health_score();
        // fps_score = (60/60)*0.5 = 0.5, drop_score = 1.0*0.5 = 0.5 → 1.0
        assert!((score - 1.0).abs() < 1e-3, "score={score}");
    }

    #[test]
    fn test_stream_metrics_half_fps() {
        let mut m = StreamMetrics::new(60.0);
        m.update(30.0, 6000, 0);
        let score = m.health_score();
        // fps_score = 0.5*0.5 = 0.25, drop_score = 0.5 → 0.75
        assert!((score - 0.75).abs() < 1e-3, "score={score}");
    }

    #[test]
    fn test_stream_metrics_all_dropped() {
        let mut m = StreamMetrics::new(30.0);
        // 30 fps, 30 total frames, all 30 dropped
        m.update(30.0, 3000, 30);
        let score = m.health_score();
        // fps_score = 0.5, drop_score = 0 → 0.5
        assert!(score <= 0.5 + 1e-3, "score={score}");
    }
}
