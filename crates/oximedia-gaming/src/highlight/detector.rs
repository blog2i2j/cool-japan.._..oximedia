//! Automated highlight detection combining audio and chat signals.
//!
//! Provides RMS-based audio excitement detection, chat-rate hype detection,
//! and a sliding-window highlight combiner that produces scored [`HighlightEvent`]s.

/// Type of gaming highlight.
#[derive(Debug, Clone, PartialEq)]
pub enum HighlightType {
    /// Player eliminated an opponent
    Kill,
    /// Player was eliminated
    Death,
    /// In-game achievement unlocked
    Achievement,
    /// Clip-worthy moment (multi-kill, clutch, etc.)
    Clipworthy,
    /// User-defined custom event
    Custom(String),
}

/// A single detected highlight moment.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct HighlightEvent {
    /// Start of the highlight in milliseconds
    pub timestamp_ms: u64,
    /// Duration of the highlight in milliseconds
    pub duration_ms: u64,
    /// Classification of the event
    pub event_type: HighlightType,
    /// Excitement score (0.0–1.0)
    pub score: f32,
}

/// Detects excitement in audio samples using RMS energy in sliding windows.
pub struct AudioEventDetector {
    /// Window size in milliseconds
    window_ms: u32,
    /// Minimum RMS threshold to register as excited
    excitement_threshold: f32,
}

/// Detects chat hype based on message rate relative to a baseline.
pub struct ChatActivityDetector {
    /// Multiplier above baseline that counts as hype
    hype_multiplier: f32,
}

/// Combines audio and chat signals to produce highlight events in a sliding window.
pub struct HighlightDetector {
    audio_detector: AudioEventDetector,
    chat_detector: ChatActivityDetector,
    /// Weight of audio signal vs chat signal (0.0–1.0; rest goes to chat)
    audio_weight: f32,
    /// Minimum combined score to register a highlight
    min_score: f32,
    /// Window duration in milliseconds
    window_ms: u64,
}

/// An ordered collection of detected highlights with merge/rank helpers.
#[derive(Debug, Default)]
pub struct HighlightTimeline {
    events: Vec<HighlightEvent>,
}

// ── AudioEventDetector ───────────────────────────────────────────────────────

impl AudioEventDetector {
    /// Create a new detector.
    ///
    /// * `window_ms` – length of each RMS analysis window in milliseconds
    /// * `excitement_threshold` – normalised RMS above which the window is "excited"
    #[must_use]
    pub fn new(window_ms: u32, excitement_threshold: f32) -> Self {
        Self {
            window_ms,
            excitement_threshold,
        }
    }

    /// Detect excitement in `samples` recorded at `sample_rate` Hz.
    ///
    /// The function splits the sample buffer into 100ms windows, computes the RMS
    /// for each, normalises it, and returns the proportion of windows that exceed
    /// the configured threshold (0.0–1.0).
    #[must_use]
    pub fn detect_excitement(&self, samples: &[f32], sample_rate: u32) -> f32 {
        if samples.is_empty() || sample_rate == 0 {
            return 0.0;
        }

        let window_size = (sample_rate as usize * self.window_ms as usize) / 1000;
        let window_size = window_size.max(1);

        let windows: Vec<&[f32]> = samples.chunks(window_size).collect();
        if windows.is_empty() {
            return 0.0;
        }

        // Compute RMS for each window
        let rms_values: Vec<f32> = windows
            .iter()
            .map(|w| {
                let mean_sq = w.iter().map(|&s| s * s).sum::<f32>() / w.len() as f32;
                mean_sq.sqrt()
            })
            .collect();

        // Normalise: find the peak RMS
        let max_rms = rms_values.iter().copied().fold(0.0_f32, f32::max);
        if max_rms == 0.0 {
            return 0.0;
        }

        let normalised: Vec<f32> = rms_values.iter().map(|&r| r / max_rms).collect();

        // Fraction of windows above the excitement threshold
        let excited = normalised
            .iter()
            .filter(|&&n| n >= self.excitement_threshold)
            .count();
        excited as f32 / normalised.len() as f32
    }
}

impl Default for AudioEventDetector {
    fn default() -> Self {
        Self::new(100, 0.7)
    }
}

// ── ChatActivityDetector ─────────────────────────────────────────────────────

impl ChatActivityDetector {
    /// Create a new detector.
    ///
    /// * `hype_multiplier` – how many times the baseline rate qualifies as hype
    #[must_use]
    pub fn new(hype_multiplier: f32) -> Self {
        Self { hype_multiplier }
    }

    /// Compute a hype score (0.0–1.0) given the current message rate and baseline.
    ///
    /// Returns 0.0 when `baseline_rate` is zero or `message_rate` is below the
    /// hype threshold.  Saturates at 1.0.
    #[must_use]
    pub fn detect_hype(&self, message_rate: f32, baseline_rate: f32) -> f32 {
        if baseline_rate <= 0.0 || message_rate <= 0.0 {
            return 0.0;
        }
        let ratio = message_rate / baseline_rate;
        if ratio < self.hype_multiplier {
            return 0.0;
        }
        // Saturate: ratio / hype_multiplier mapped to [1.0, …], clamped to 1.0
        (ratio / self.hype_multiplier - 1.0).clamp(0.0, 1.0)
    }
}

impl Default for ChatActivityDetector {
    fn default() -> Self {
        Self::new(2.5)
    }
}

// ── HighlightDetector ────────────────────────────────────────────────────────

impl HighlightDetector {
    /// Create a new combined detector.
    #[must_use]
    pub fn new(
        audio_detector: AudioEventDetector,
        chat_detector: ChatActivityDetector,
        audio_weight: f32,
        min_score: f32,
        window_ms: u64,
    ) -> Self {
        Self {
            audio_detector,
            chat_detector,
            audio_weight: audio_weight.clamp(0.0, 1.0),
            min_score,
            window_ms,
        }
    }

    /// Analyse a window of audio + chat activity and produce an optional highlight.
    ///
    /// * `timestamp_ms` – start of the window
    /// * `samples` – raw audio samples for the window
    /// * `sample_rate` – sample rate in Hz
    /// * `message_rate` – chat messages per second in this window
    /// * `baseline_rate` – typical chat messages per second
    /// * `event_type` – the type to assign if a highlight is detected
    #[must_use]
    pub fn analyse_window(
        &self,
        timestamp_ms: u64,
        samples: &[f32],
        sample_rate: u32,
        message_rate: f32,
        baseline_rate: f32,
        event_type: HighlightType,
    ) -> Option<HighlightEvent> {
        let audio_score = self.audio_detector.detect_excitement(samples, sample_rate);
        let chat_score = self.chat_detector.detect_hype(message_rate, baseline_rate);

        let chat_weight = 1.0 - self.audio_weight;
        let combined = self.audio_weight * audio_score + chat_weight * chat_score;

        if combined >= self.min_score {
            Some(HighlightEvent {
                timestamp_ms,
                duration_ms: self.window_ms,
                event_type,
                score: combined,
            })
        } else {
            None
        }
    }
}

impl Default for HighlightDetector {
    fn default() -> Self {
        Self::new(
            AudioEventDetector::default(),
            ChatActivityDetector::default(),
            0.6,  // 60% audio weight
            0.4,  // minimum score to register
            5000, // 5-second window
        )
    }
}

// ── HighlightTimeline ────────────────────────────────────────────────────────

impl HighlightTimeline {
    /// Create an empty timeline.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an event to the timeline.
    pub fn add_event(&mut self, event: HighlightEvent) {
        self.events.push(event);
    }

    /// Return the number of events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Return `true` if there are no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Merge overlapping events (by timestamp + duration), keeping the higher score.
    pub fn merge_overlapping(&mut self) {
        if self.events.len() < 2 {
            return;
        }

        // Sort by timestamp
        self.events.sort_by_key(|e| e.timestamp_ms);

        let mut merged: Vec<HighlightEvent> = Vec::new();
        for event in self.events.drain(..) {
            if let Some(last) = merged.last_mut() {
                let last_end = last.timestamp_ms + last.duration_ms;
                if event.timestamp_ms < last_end {
                    // Overlap: extend the window and keep the better score
                    let new_end = last_end.max(event.timestamp_ms + event.duration_ms);
                    last.duration_ms = new_end - last.timestamp_ms;
                    if event.score > last.score {
                        last.score = event.score;
                        last.event_type = event.event_type;
                    }
                    continue;
                }
            }
            merged.push(event);
        }
        self.events = merged;
    }

    /// Return the top `n` highlights sorted by score (descending).
    #[must_use]
    pub fn top_n(&self, n: usize) -> Vec<HighlightEvent> {
        let mut sorted = self.events.clone();
        sorted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(n);
        sorted
    }

    /// Return all events sorted by timestamp.
    #[must_use]
    pub fn events_sorted(&self) -> Vec<&HighlightEvent> {
        let mut refs: Vec<&HighlightEvent> = self.events.iter().collect();
        refs.sort_by_key(|e| e.timestamp_ms);
        refs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- AudioEventDetector ---

    #[test]
    fn test_audio_detector_empty() {
        let det = AudioEventDetector::default();
        assert_eq!(det.detect_excitement(&[], 44100), 0.0);
    }

    #[test]
    fn test_audio_detector_silence() {
        let det = AudioEventDetector::default();
        let samples = vec![0.0_f32; 44100];
        assert_eq!(det.detect_excitement(&samples, 44100), 0.0);
    }

    #[test]
    fn test_audio_detector_loud() {
        let det = AudioEventDetector::new(100, 0.5);
        // All samples at max amplitude → every window will be at 100% RMS → excited
        let samples = vec![1.0_f32; 44100];
        let score = det.detect_excitement(&samples, 44100);
        assert!(score > 0.5, "Expected high excitement, got {score}");
    }

    #[test]
    fn test_audio_detector_partial() {
        // First half loud, second half silent
        let det = AudioEventDetector::new(100, 0.5);
        let mut samples = vec![1.0_f32; 22050];
        samples.extend(vec![0.0_f32; 22050]);
        let score = det.detect_excitement(&samples, 44100);
        // Should be between 0 and 1 exclusive
        assert!(score > 0.0 && score < 1.0, "score={score}");
    }

    // --- ChatActivityDetector ---

    #[test]
    fn test_chat_no_hype_below_threshold() {
        let det = ChatActivityDetector::new(3.0);
        // 2x baseline but threshold is 3x → no hype
        assert_eq!(det.detect_hype(2.0, 1.0), 0.0);
    }

    #[test]
    fn test_chat_hype_above_threshold() {
        let det = ChatActivityDetector::new(2.0);
        let score = det.detect_hype(6.0, 1.0);
        assert!(score > 0.0, "Expected hype score > 0");
    }

    #[test]
    fn test_chat_zero_baseline() {
        let det = ChatActivityDetector::default();
        assert_eq!(det.detect_hype(10.0, 0.0), 0.0);
    }

    #[test]
    fn test_chat_hype_saturates() {
        let det = ChatActivityDetector::new(2.0);
        // 1000x baseline → must saturate at 1.0
        assert_eq!(det.detect_hype(1000.0, 1.0), 1.0);
    }

    // --- HighlightDetector ---

    #[test]
    fn test_highlight_detector_no_signal() {
        let det = HighlightDetector::default();
        let silence = vec![0.0_f32; 44100];
        let result = det.analyse_window(0, &silence, 44100, 1.0, 1.0, HighlightType::Kill);
        // Silence + no hype → below min_score
        assert!(result.is_none());
    }

    #[test]
    fn test_highlight_detector_strong_signal() {
        let det = HighlightDetector::new(
            AudioEventDetector::new(100, 0.5),
            ChatActivityDetector::new(2.0),
            0.5,
            0.3,
            5000,
        );
        let samples = vec![1.0_f32; 44100];
        let result = det.analyse_window(0, &samples, 44100, 10.0, 1.0, HighlightType::Clipworthy);
        assert!(result.is_some());
        let ev = result.expect("result should be ok");
        assert!(ev.score >= 0.3);
        assert_eq!(ev.event_type, HighlightType::Clipworthy);
    }

    // --- HighlightTimeline ---

    #[test]
    fn test_timeline_empty() {
        let timeline = HighlightTimeline::new();
        assert!(timeline.is_empty());
        assert!(timeline.top_n(5).is_empty());
    }

    fn make_event(ts: u64, dur: u64, score: f32) -> HighlightEvent {
        HighlightEvent {
            timestamp_ms: ts,
            duration_ms: dur,
            event_type: HighlightType::Kill,
            score,
        }
    }

    #[test]
    fn test_timeline_add_and_top() {
        let mut tl = HighlightTimeline::new();
        tl.add_event(make_event(0, 5000, 0.9));
        tl.add_event(make_event(10_000, 5000, 0.5));
        tl.add_event(make_event(20_000, 5000, 0.7));

        let top = tl.top_n(2);
        assert_eq!(top.len(), 2);
        assert!(top[0].score >= top[1].score);
    }

    #[test]
    fn test_timeline_merge_overlapping() {
        let mut tl = HighlightTimeline::new();
        // Two events that overlap
        tl.add_event(make_event(0, 6000, 0.6));
        tl.add_event(make_event(4000, 6000, 0.8));
        // One isolated event
        tl.add_event(make_event(20_000, 5000, 0.5));

        tl.merge_overlapping();
        assert_eq!(tl.len(), 2);
    }

    #[test]
    fn test_timeline_no_overlap() {
        let mut tl = HighlightTimeline::new();
        tl.add_event(make_event(0, 3000, 0.5));
        tl.add_event(make_event(10_000, 3000, 0.6));
        tl.merge_overlapping();
        assert_eq!(tl.len(), 2);
    }

    #[test]
    fn test_highlight_type_custom() {
        let ht = HighlightType::Custom("triple_kill".to_string());
        assert_eq!(ht, HighlightType::Custom("triple_kill".to_string()));
        assert_ne!(ht, HighlightType::Kill);
    }
}
