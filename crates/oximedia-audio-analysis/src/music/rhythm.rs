//! Rhythmic analysis extending MIR capabilities.

use crate::transient::TransientDetector;
use crate::{AnalysisConfig, Result};

/// Rhythm analyzer for advanced rhythmic features.
pub struct RhythmAnalyzer {
    #[allow(dead_code)]
    config: AnalysisConfig,
    transient_detector: TransientDetector,
}

impl RhythmAnalyzer {
    /// Create a new rhythm analyzer.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        let transient_detector = TransientDetector::new(config.clone());

        Self {
            config,
            transient_detector,
        }
    }

    /// Analyze rhythmic features.
    pub fn analyze(&self, samples: &[f32], sample_rate: f32) -> Result<RhythmFeatures> {
        // Detect onsets/transients
        let transients = self.transient_detector.detect(samples, sample_rate)?;

        // Compute inter-onset intervals
        let iois = self.compute_inter_onset_intervals(&transients.transient_times);

        // Estimate tempo from IOIs
        let tempo = self.estimate_tempo(&iois, sample_rate);

        // Compute rhythm regularity
        let regularity = self.compute_regularity(&iois);

        // Compute syncopation (deviation from expected beat)
        let syncopation = self.compute_syncopation(&transients.transient_times, tempo);

        Ok(RhythmFeatures {
            tempo,
            num_onsets: transients.num_transients,
            inter_onset_intervals: iois,
            rhythm_regularity: regularity,
            syncopation,
        })
    }

    /// Compute inter-onset intervals.
    #[allow(clippy::unused_self)]
    fn compute_inter_onset_intervals(&self, onset_times: &[f32]) -> Vec<f32> {
        if onset_times.len() < 2 {
            return vec![];
        }

        onset_times.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Estimate tempo from inter-onset intervals.
    #[allow(clippy::unused_self)]
    fn estimate_tempo(&self, iois: &[f32], _sample_rate: f32) -> f32 {
        if iois.is_empty() {
            return 0.0;
        }

        // Median IOI as beat period estimate
        let mut sorted_iois = iois.to_vec();
        sorted_iois.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median_ioi = sorted_iois[sorted_iois.len() / 2];

        // Convert to BPM
        if median_ioi > 0.0 {
            60.0 / median_ioi
        } else {
            0.0
        }
    }

    /// Compute rhythm regularity.
    #[allow(clippy::unused_self)]
    fn compute_regularity(&self, iois: &[f32]) -> f32 {
        if iois.len() < 2 {
            return 0.0;
        }

        let mean = iois.iter().sum::<f32>() / iois.len() as f32;
        let variance = iois.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / iois.len() as f32;

        let std_dev = variance.sqrt();

        // Regularity is inverse of coefficient of variation
        if mean > 0.0 {
            1.0 / (1.0 + std_dev / mean)
        } else {
            0.0
        }
    }

    /// Compute syncopation measure.
    #[allow(clippy::unused_self)]
    fn compute_syncopation(&self, onset_times: &[f32], tempo: f32) -> f32 {
        if onset_times.is_empty() || tempo <= 0.0 {
            return 0.0;
        }

        let beat_period = 60.0 / tempo;
        let mut syncopation_score = 0.0;

        for &onset in onset_times {
            // Distance to nearest beat
            let beat_phase = (onset % beat_period) / beat_period;
            let dist_to_beat = (beat_phase - 0.5).abs() * 2.0; // 0 at beat, 1 at off-beat

            syncopation_score += dist_to_beat;
        }

        syncopation_score / onset_times.len() as f32
    }
}

/// Rhythmic features.
#[derive(Debug, Clone)]
pub struct RhythmFeatures {
    /// Estimated tempo in BPM
    pub tempo: f32,
    /// Number of detected onsets
    pub num_onsets: usize,
    /// Inter-onset intervals in seconds
    pub inter_onset_intervals: Vec<f32>,
    /// Rhythm regularity (0-1, higher = more regular)
    pub rhythm_regularity: f32,
    /// Syncopation measure (0-1, higher = more syncopated)
    pub syncopation: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rhythm_analyzer() {
        let config = AnalysisConfig::default();
        let analyzer = RhythmAnalyzer::new(config);

        // Generate regular beats at 120 BPM
        let sample_rate = 44100.0;
        let beat_interval = 60.0 / 120.0; // 0.5 seconds
        let mut samples = vec![0.0; (sample_rate * 4.0) as usize];

        for i in 0..8 {
            let pos = (i as f32 * beat_interval * sample_rate) as usize;
            if pos < samples.len() {
                samples[pos] = 1.0;
            }
        }

        let result = analyzer.analyze(&samples, sample_rate);
        assert!(result.is_ok());
    }
}
