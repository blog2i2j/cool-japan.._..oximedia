//! Genre classification using audio features.

use crate::genre::features::GenreFeatures;
use crate::types::GenreResult;
use crate::MirResult;
use std::collections::HashMap;

/// Genre classifier.
pub struct GenreClassifier {
    sample_rate: f32,
}

impl GenreClassifier {
    /// Create a new genre classifier.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self { sample_rate }
    }

    /// Classify genre from audio signal.
    ///
    /// # Errors
    ///
    /// Returns error if classification fails.
    pub fn classify(&self, signal: &[f32]) -> MirResult<GenreResult> {
        // Extract features
        let feature_extractor = GenreFeatures::new(self.sample_rate);
        let features = feature_extractor.extract(signal)?;

        // Simple rule-based classification (in practice, would use ML model)
        let mut genre_scores = HashMap::new();

        // Electronic: high spectral centroid, low zero crossing
        let electronic_score =
            features.spectral_centroid * 0.6 + (1.0 - features.zero_crossing_rate) * 0.4;
        genre_scores.insert("electronic".to_string(), electronic_score);

        // Rock: high energy, moderate tempo
        let rock_score =
            features.energy * 0.5 + self.tempo_score(features.tempo, 120.0, 160.0) * 0.5;
        genre_scores.insert("rock".to_string(), rock_score);

        // Classical: low energy variation, wide spectral range
        let classical_score =
            (1.0 - features.energy_variance) * 0.6 + features.spectral_bandwidth * 0.4;
        genre_scores.insert("classical".to_string(), classical_score);

        // Jazz: moderate tempo, high harmonic complexity
        let jazz_score = self.tempo_score(features.tempo, 100.0, 140.0) * 0.4
            + features.harmonic_complexity * 0.6;
        genre_scores.insert("jazz".to_string(), jazz_score);

        // Hip-hop: strong beats, low tempo
        let hiphop_score =
            features.beat_strength * 0.6 + self.tempo_score(features.tempo, 80.0, 110.0) * 0.4;
        genre_scores.insert("hip-hop".to_string(), hiphop_score);

        // Pop: moderate everything, strong beats
        let pop_score =
            features.beat_strength * 0.5 + self.tempo_score(features.tempo, 100.0, 130.0) * 0.5;
        genre_scores.insert("pop".to_string(), pop_score);

        // Find top genre
        let (top_genre, top_confidence) = genre_scores
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(("unknown".to_string(), 0.0), |(g, &c)| (g.clone(), c));

        Ok(GenreResult {
            genres: genre_scores,
            top_genre_name: top_genre,
            top_genre_confidence: top_confidence,
        })
    }

    /// Score based on tempo range.
    fn tempo_score(&self, tempo: f32, min_bpm: f32, max_bpm: f32) -> f32 {
        if tempo >= min_bpm && tempo <= max_bpm {
            1.0 - ((tempo - (min_bpm + max_bpm) / 2.0).abs() / ((max_bpm - min_bpm) / 2.0))
        } else {
            0.0
        }
    }

    /// Multi-label genre classification.
    ///
    /// Returns all genres whose confidence exceeds the given threshold,
    /// allowing a track to be labelled with multiple genres simultaneously.
    ///
    /// # Errors
    ///
    /// Returns error if feature extraction or classification fails.
    pub fn classify_multi_label(
        &self,
        signal: &[f32],
        threshold: f32,
    ) -> MirResult<Vec<(String, f32)>> {
        let result = self.classify(signal)?;

        let mut labels: Vec<(String, f32)> = result
            .genres
            .into_iter()
            .filter(|(_, conf)| *conf >= threshold)
            .collect();

        // Sort descending by confidence
        labels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(labels)
    }
}

/// Streaming genre classifier that processes audio in chunks.
///
/// Maintains a running feature accumulator and produces updated genre
/// predictions as new audio arrives, suitable for real-time applications.
pub struct StreamingGenreClassifier {
    sample_rate: f32,
    /// Accumulated spectral centroid values per chunk.
    centroids: Vec<f32>,
    /// Accumulated spectral bandwidth values per chunk.
    bandwidths: Vec<f32>,
    /// Accumulated energy values per chunk.
    energies: Vec<f32>,
    /// Accumulated zero-crossing rates per chunk.
    zcr_values: Vec<f32>,
    /// Total samples processed.
    total_samples: usize,
}

impl StreamingGenreClassifier {
    /// Create a new streaming genre classifier.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            centroids: Vec::new(),
            bandwidths: Vec::new(),
            energies: Vec::new(),
            zcr_values: Vec::new(),
            total_samples: 0,
        }
    }

    /// Process a chunk of audio and update internal feature accumulators.
    ///
    /// # Errors
    ///
    /// Returns error if feature extraction fails.
    pub fn process_chunk(&mut self, chunk: &[f32]) -> MirResult<()> {
        if chunk.is_empty() {
            return Ok(());
        }

        let feature_extractor = GenreFeatures::new(self.sample_rate);

        // Only attempt STFT-based extraction if chunk is large enough
        if chunk.len() >= 2048 {
            let features = feature_extractor.extract(chunk)?;
            self.centroids.push(features.spectral_centroid);
            self.bandwidths.push(features.spectral_bandwidth);
            self.energies.push(features.energy);
            self.zcr_values.push(features.zero_crossing_rate);
        } else {
            // For small chunks, compute ZCR only
            let zcr = Self::compute_zcr(chunk);
            self.zcr_values.push(zcr);
        }

        self.total_samples += chunk.len();
        Ok(())
    }

    /// Get current genre prediction based on accumulated features.
    ///
    /// # Errors
    ///
    /// Returns error if insufficient data has been accumulated.
    pub fn current_prediction(&self) -> MirResult<GenreResult> {
        if self.centroids.is_empty() {
            return Err(crate::MirError::InsufficientData(
                "Not enough data for genre prediction".to_string(),
            ));
        }

        let classifier = GenreClassifier::new(self.sample_rate);

        // Build aggregated features as a synthetic signal metric
        let avg_centroid = crate::utils::mean(&self.centroids);
        let avg_bandwidth = crate::utils::mean(&self.bandwidths);
        let avg_energy = crate::utils::mean(&self.energies);
        let energy_variance = crate::utils::std_dev(&self.energies);
        let avg_zcr = crate::utils::mean(&self.zcr_values);

        let harmonic_complexity = avg_bandwidth / (avg_centroid + 1.0);

        // Build genre scores using the same rule-based approach
        let mut genre_scores = HashMap::new();

        let electronic_score = avg_centroid * 0.6 + (1.0 - avg_zcr) * 0.4;
        genre_scores.insert("electronic".to_string(), electronic_score);

        let rock_score = avg_energy * 0.5 + classifier.tempo_score(120.0, 120.0, 160.0) * 0.5;
        genre_scores.insert("rock".to_string(), rock_score);

        let classical_score = (1.0 - energy_variance) * 0.6 + avg_bandwidth * 0.4;
        genre_scores.insert("classical".to_string(), classical_score);

        let jazz_score =
            classifier.tempo_score(110.0, 100.0, 140.0) * 0.4 + harmonic_complexity * 0.6;
        genre_scores.insert("jazz".to_string(), jazz_score);

        let hiphop_score = avg_energy * 0.6 + classifier.tempo_score(90.0, 80.0, 110.0) * 0.4;
        genre_scores.insert("hip-hop".to_string(), hiphop_score);

        let pop_score = avg_energy * 0.5 + classifier.tempo_score(115.0, 100.0, 130.0) * 0.5;
        genre_scores.insert("pop".to_string(), pop_score);

        let (top_genre, top_confidence) = genre_scores
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(("unknown".to_string(), 0.0), |(g, &c)| (g.clone(), c));

        Ok(GenreResult {
            genres: genre_scores,
            top_genre_name: top_genre,
            top_genre_confidence: top_confidence,
        })
    }

    /// Reset the streaming classifier state.
    pub fn reset(&mut self) {
        self.centroids.clear();
        self.bandwidths.clear();
        self.energies.clear();
        self.zcr_values.clear();
        self.total_samples = 0;
    }

    /// Number of samples processed so far.
    #[must_use]
    pub fn samples_processed(&self) -> usize {
        self.total_samples
    }

    /// Compute zero-crossing rate for a small chunk.
    #[allow(clippy::cast_precision_loss)]
    fn compute_zcr(signal: &[f32]) -> f32 {
        if signal.len() < 2 {
            return 0.0;
        }
        let mut crossings = 0_u32;
        for i in 1..signal.len() {
            if (signal[i] >= 0.0 && signal[i - 1] < 0.0)
                || (signal[i] < 0.0 && signal[i - 1] >= 0.0)
            {
                crossings += 1;
            }
        }
        crossings as f32 / signal.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genre_classifier_creation() {
        let classifier = GenreClassifier::new(44100.0);
        assert_eq!(classifier.sample_rate, 44100.0);
    }

    #[test]
    fn test_tempo_score() {
        let classifier = GenreClassifier::new(44100.0);
        assert_eq!(classifier.tempo_score(120.0, 100.0, 140.0), 1.0);
        assert!(classifier.tempo_score(200.0, 100.0, 140.0) < 0.1);
    }

    #[test]
    fn test_streaming_classifier_creation() {
        let streaming = StreamingGenreClassifier::new(44100.0);
        assert_eq!(streaming.samples_processed(), 0);
    }

    #[test]
    fn test_streaming_classifier_insufficient_data() {
        let streaming = StreamingGenreClassifier::new(44100.0);
        let result = streaming.current_prediction();
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_classifier_reset() {
        let mut streaming = StreamingGenreClassifier::new(44100.0);
        // Process a sine wave chunk
        let chunk: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin()).collect();
        let _ = streaming.process_chunk(&chunk);
        assert!(streaming.samples_processed() > 0);
        streaming.reset();
        assert_eq!(streaming.samples_processed(), 0);
    }
}
