//! Speaker identification and verification.

use crate::voice::VoiceCharacteristics;
use crate::{AnalysisConfig, Result};

/// Speaker identifier for speaker recognition tasks.
pub struct SpeakerIdentifier {
    #[allow(dead_code)]
    config: AnalysisConfig,
    enrolled_speakers: Vec<SpeakerProfile>,
}

impl SpeakerIdentifier {
    /// Create a new speaker identifier.
    #[must_use]
    pub fn new(config: AnalysisConfig) -> Self {
        Self {
            config,
            enrolled_speakers: Vec::new(),
        }
    }

    /// Enroll a new speaker with their voice characteristics.
    pub fn enroll_speaker(&mut self, id: String, characteristics: VoiceCharacteristics) {
        let profile = SpeakerProfile {
            id,
            characteristics,
        };
        self.enrolled_speakers.push(profile);
    }

    /// Identify speaker from voice characteristics.
    ///
    /// Returns speaker ID and confidence score.
    #[must_use]
    pub fn identify(&self, characteristics: &VoiceCharacteristics) -> Option<(String, f32)> {
        if self.enrolled_speakers.is_empty() {
            return None;
        }

        let mut best_match = None;
        let mut best_score = 0.0;

        for speaker in &self.enrolled_speakers {
            let score = self.compute_similarity(&speaker.characteristics, characteristics);
            if score > best_score {
                best_score = score;
                best_match = Some(speaker.id.clone());
            }
        }

        if best_score > 0.5 {
            best_match.map(|id| (id, best_score))
        } else {
            None
        }
    }

    /// Verify if voice matches a specific speaker.
    pub fn verify(&self, speaker_id: &str, characteristics: &VoiceCharacteristics) -> Result<bool> {
        let speaker = self.enrolled_speakers.iter().find(|s| s.id == speaker_id);

        if let Some(speaker) = speaker {
            let score = self.compute_similarity(&speaker.characteristics, characteristics);
            Ok(score > 0.7) // Higher threshold for verification
        } else {
            Ok(false)
        }
    }

    /// Compute similarity between two voice characteristic sets.
    #[allow(clippy::unused_self)]
    fn compute_similarity(&self, a: &VoiceCharacteristics, b: &VoiceCharacteristics) -> f32 {
        // F0 similarity (normalized by average)
        let f0_diff = (a.f0 - b.f0).abs() / ((a.f0 + b.f0) / 2.0);
        let f0_sim = (1.0 - f0_diff).max(0.0);

        // Formant similarity
        let mut formant_sim = 0.0;
        let min_len = a.formants.len().min(b.formants.len());
        if min_len > 0 {
            for i in 0..min_len {
                let diff =
                    (a.formants[i] - b.formants[i]).abs() / ((a.formants[i] + b.formants[i]) / 2.0);
                formant_sim += (1.0 - diff).max(0.0);
            }
            formant_sim /= min_len as f32;
        }

        // Jitter similarity
        let jitter_diff = (a.jitter - b.jitter).abs();
        let jitter_sim = (1.0 - jitter_diff * 50.0).max(0.0);

        // Shimmer similarity
        let shimmer_diff = (a.shimmer - b.shimmer).abs();
        let shimmer_sim = (1.0 - shimmer_diff * 10.0).max(0.0);

        // Weighted average

        f0_sim * 0.3 + formant_sim * 0.4 + jitter_sim * 0.15 + shimmer_sim * 0.15
    }

    /// Get number of enrolled speakers.
    #[must_use]
    pub fn num_speakers(&self) -> usize {
        self.enrolled_speakers.len()
    }

    /// Remove a speaker from the database.
    pub fn remove_speaker(&mut self, speaker_id: &str) -> bool {
        if let Some(pos) = self
            .enrolled_speakers
            .iter()
            .position(|s| s.id == speaker_id)
        {
            self.enrolled_speakers.remove(pos);
            true
        } else {
            false
        }
    }
}

/// Speaker profile for identification.
#[derive(Debug, Clone)]
struct SpeakerProfile {
    id: String,
    characteristics: VoiceCharacteristics,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::age::AgeGroup;
    use crate::voice::{Emotion, Gender};

    #[test]
    fn test_speaker_identification() {
        let config = AnalysisConfig::default();
        let mut identifier = SpeakerIdentifier::new(config);

        // Enroll speaker 1
        let speaker1 = VoiceCharacteristics {
            f0: 120.0,
            formants: vec![500.0, 1500.0, 2500.0],
            jitter: 0.01,
            shimmer: 0.05,
            hnr: 15.0,
            gender: Gender::Male,
            age_group: AgeGroup::YoungAdult,
            emotion: Emotion::Neutral,
        };
        identifier.enroll_speaker("speaker1".to_string(), speaker1);

        // Enroll speaker 2
        let speaker2 = VoiceCharacteristics {
            f0: 220.0,
            formants: vec![600.0, 1800.0, 2800.0],
            jitter: 0.008,
            shimmer: 0.04,
            hnr: 18.0,
            gender: Gender::Female,
            age_group: AgeGroup::YoungAdult,
            emotion: Emotion::Neutral,
        };
        identifier.enroll_speaker("speaker2".to_string(), speaker2);

        // Test identification with similar characteristics to speaker1
        let test_voice = VoiceCharacteristics {
            f0: 125.0,
            formants: vec![510.0, 1520.0, 2480.0],
            jitter: 0.011,
            shimmer: 0.052,
            hnr: 14.5,
            gender: Gender::Male,
            age_group: AgeGroup::YoungAdult,
            emotion: Emotion::Neutral,
        };

        let result = identifier.identify(&test_voice);
        assert!(result.is_some());
        if let Some((id, _score)) = result {
            assert_eq!(id, "speaker1");
        }
    }

    #[test]
    fn test_speaker_verification() {
        let config = AnalysisConfig::default();
        let mut identifier = SpeakerIdentifier::new(config);

        let speaker_chars = VoiceCharacteristics {
            f0: 150.0,
            formants: vec![550.0, 1600.0, 2600.0],
            jitter: 0.009,
            shimmer: 0.045,
            hnr: 16.0,
            gender: Gender::Male,
            age_group: AgeGroup::MiddleAged,
            emotion: Emotion::Neutral,
        };
        identifier.enroll_speaker("test_speaker".to_string(), speaker_chars.clone());

        // Should verify with same characteristics
        assert!(identifier
            .verify("test_speaker", &speaker_chars)
            .expect("verification should succeed"));
    }
}
