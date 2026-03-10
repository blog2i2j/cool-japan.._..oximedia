//! Speaker diarization and identification for captioning.
//!
//! Provides speaker registration, identification via cosine similarity, and
//! change-point detection over feature vectors.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A registered speaker with a voice feature vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Speaker {
    /// Unique numeric identifier.
    pub id: u32,
    /// Human-readable label (e.g. "Anchor A").
    pub label: String,
    /// 16-dimensional voice feature vector.
    pub voice_signature: Vec<f32>,
}

impl Speaker {
    /// Create a new speaker. `features` is truncated or padded to 16 dimensions.
    #[must_use]
    pub fn new(id: u32, label: impl Into<String>, features: Vec<f32>) -> Self {
        let mut voice_signature = features;
        voice_signature.resize(16, 0.0);
        Self {
            id,
            label: label.into(),
            voice_signature,
        }
    }
}

/// Registers speakers and identifies them from feature vectors.
#[derive(Debug, Default)]
pub struct SpeakerDiarizer {
    speakers: HashMap<u32, Speaker>,
}

impl SpeakerDiarizer {
    /// Create a new empty diarizer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a speaker with the given ID, label, and feature vector.
    pub fn register_speaker(&mut self, id: u32, label: &str, features: Vec<f32>) {
        self.speakers.insert(id, Speaker::new(id, label, features));
    }

    /// Identify the closest registered speaker for the given feature vector.
    ///
    /// Returns `Some((speaker_id, cosine_similarity))` for the best match,
    /// or `None` if no speakers are registered.
    #[must_use]
    pub fn identify(&self, features: &[f32]) -> Option<(u32, f32)> {
        if self.speakers.is_empty() {
            return None;
        }

        self.speakers
            .values()
            .map(|spk| (spk.id, cosine_similarity(&spk.voice_signature, features)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Number of registered speakers.
    #[must_use]
    pub fn speaker_count(&self) -> usize {
        self.speakers.len()
    }

    /// Look up a registered speaker by ID.
    #[must_use]
    pub fn get_speaker(&self, id: u32) -> Option<&Speaker> {
        self.speakers.get(&id)
    }
}

/// Cosine similarity between two equal-length vectors.
///
/// Returns a value in [-1, 1]. Returns 0.0 if either vector has zero norm.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let dot: f32 = a[..len]
        .iter()
        .zip(b[..len].iter())
        .map(|(x, y)| x * y)
        .sum();
    let norm_a: f32 = a[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// A segment attributed to a specific speaker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationSegment {
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// Identified speaker ID.
    pub speaker_id: u32,
    /// Confidence of the identification (0.0–1.0, derived from cosine similarity).
    pub confidence: f32,
}

impl DiarizationSegment {
    /// Create a new diarization segment.
    #[must_use]
    pub fn new(start_ms: u64, end_ms: u64, speaker_id: u32, confidence: f32) -> Self {
        Self {
            start_ms,
            end_ms,
            speaker_id,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Duration of this segment in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// The full diarization result for a media clip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationResult {
    /// Ordered list of speaker segments.
    pub segments: Vec<DiarizationSegment>,
    /// Total number of unique speakers detected.
    pub speaker_count: u32,
}

impl DiarizationResult {
    /// Create a new result.
    #[must_use]
    pub fn new(segments: Vec<DiarizationSegment>, speaker_count: u32) -> Self {
        Self {
            segments,
            speaker_count,
        }
    }

    /// Total speaking time for a given speaker ID, in milliseconds.
    #[must_use]
    pub fn speaker_time(&self, id: u32) -> u64 {
        self.segments
            .iter()
            .filter(|s| s.speaker_id == id)
            .map(DiarizationSegment::duration_ms)
            .sum()
    }

    /// Return the ID of the speaker with the most total speaking time.
    ///
    /// Returns `None` if there are no segments.
    #[must_use]
    pub fn dominant_speaker(&self) -> Option<u32> {
        if self.segments.is_empty() {
            return None;
        }

        // Collect unique speaker IDs
        let mut ids: Vec<u32> = self.segments.iter().map(|s| s.speaker_id).collect();
        ids.sort_unstable();
        ids.dedup();

        ids.into_iter().max_by_key(|&id| self.speaker_time(id))
    }
}

/// Detects speaker change-points in a sequence of feature vectors.
pub struct SpeakerChangeDetector;

impl SpeakerChangeDetector {
    /// Detect timestamps of speaker changes in a feature sequence.
    ///
    /// Each element of `features` represents audio extracted from one
    /// `window_ms`-long window. A speaker change is flagged when the cosine
    /// similarity between adjacent windows drops below 0.7.
    ///
    /// Returns a list of start-of-change timestamps (in ms, 0-based).
    #[must_use]
    pub fn detect(features: &[Vec<f32>], window_ms: u32) -> Vec<u64> {
        const CHANGE_THRESHOLD: f32 = 0.7;
        let mut changes = Vec::new();

        for i in 1..features.len() {
            let sim = cosine_similarity(&features[i - 1], &features[i]);
            if sim < CHANGE_THRESHOLD {
                changes.push((i as u64) * u64::from(window_ms));
            }
        }

        changes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_features(val: f32, len: usize) -> Vec<f32> {
        vec![val; len]
    }

    #[test]
    fn test_speaker_creation() {
        let spk = Speaker::new(1, "Anchor", vec![1.0; 16]);
        assert_eq!(spk.id, 1);
        assert_eq!(spk.label, "Anchor");
        assert_eq!(spk.voice_signature.len(), 16);
    }

    #[test]
    fn test_speaker_features_padded_to_16() {
        let spk = Speaker::new(1, "Short", vec![1.0; 5]);
        assert_eq!(spk.voice_signature.len(), 16);
    }

    #[test]
    fn test_diarizer_register_and_count() {
        let mut diarizer = SpeakerDiarizer::new();
        diarizer.register_speaker(1, "Alice", vec![1.0; 16]);
        diarizer.register_speaker(2, "Bob", vec![0.5; 16]);
        assert_eq!(diarizer.speaker_count(), 2);
    }

    #[test]
    fn test_diarizer_identify_correct_speaker() {
        let mut diarizer = SpeakerDiarizer::new();
        diarizer.register_speaker(1, "Alice", uniform_features(1.0, 16));
        diarizer.register_speaker(2, "Bob", uniform_features(0.0, 16));

        // Query matches Alice's profile
        let result = diarizer.identify(&uniform_features(1.0, 16));
        assert!(result.is_some());
        assert_eq!(result.expect("operation should succeed in test").0, 1);
    }

    #[test]
    fn test_diarizer_empty_returns_none() {
        let diarizer = SpeakerDiarizer::new();
        let result = diarizer.identify(&uniform_features(1.0, 16));
        assert!(result.is_none());
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0f32, 0.0, 0.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0f32; 4];
        let b = vec![1.0f32; 4];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_diarization_segment_duration() {
        let seg = DiarizationSegment::new(1000, 4000, 1, 0.9);
        assert_eq!(seg.duration_ms(), 3000);
    }

    #[test]
    fn test_diarization_result_speaker_time() {
        let segments = vec![
            DiarizationSegment::new(0, 5000, 1, 0.9),
            DiarizationSegment::new(5000, 8000, 2, 0.85),
            DiarizationSegment::new(8000, 15000, 1, 0.92),
        ];
        let result = DiarizationResult::new(segments, 2);
        assert_eq!(result.speaker_time(1), 12000); // 5000 + 7000
        assert_eq!(result.speaker_time(2), 3000);
    }

    #[test]
    fn test_diarization_result_dominant_speaker() {
        let segments = vec![
            DiarizationSegment::new(0, 2000, 1, 0.9),
            DiarizationSegment::new(2000, 10000, 2, 0.85), // 8000 ms
        ];
        let result = DiarizationResult::new(segments, 2);
        assert_eq!(result.dominant_speaker(), Some(2));
    }

    #[test]
    fn test_diarization_result_no_segments() {
        let result = DiarizationResult::new(vec![], 0);
        assert_eq!(result.dominant_speaker(), None);
    }

    #[test]
    fn test_change_detector_no_change() {
        // All windows are identical → similarity = 1.0 → no change
        let features: Vec<Vec<f32>> = (0..5).map(|_| vec![1.0f32; 16]).collect();
        let changes = SpeakerChangeDetector::detect(&features, 1000);
        assert!(changes.is_empty());
    }

    #[test]
    fn test_change_detector_detects_change() {
        // Window 0 and 1 are identical, window 2 is orthogonal
        let features: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0], // change here
        ];
        let changes = SpeakerChangeDetector::detect(&features, 500);
        assert!(!changes.is_empty());
        assert!(changes.contains(&1000)); // at window index 2, t = 2 * 500 ms
    }
}
