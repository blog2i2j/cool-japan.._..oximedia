//! Instrument identification from audio features.

use crate::spectral::SpectralFeatures;
use crate::transient::TransientResult;

/// Musical instrument classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Instrument {
    /// Piano
    Piano,
    /// Guitar (acoustic or electric)
    Guitar,
    /// Violin or string instrument
    Violin,
    /// Flute or wind instrument
    Flute,
    /// Trumpet or brass instrument
    Trumpet,
    /// Drums or percussion
    Drums,
    /// Bass guitar or bass
    Bass,
    /// Synthesizer
    Synthesizer,
    /// Vocals
    Vocals,
    /// Unknown/unclassified
    Unknown,
}

/// Detect instrument from audio features.
///
/// Uses spectral and temporal features to classify instruments:
/// - Piano: Sharp attacks, wide spectral range, harmonic
/// - Guitar: Moderate attacks, characteristic formants
/// - Violin: Smooth attacks, strong harmonics
/// - Flute: Pure tone, low harmonics
/// - Drums: Very sharp attacks, noise-like spectrum
/// - Bass: Low spectral centroid
/// - Vocals: Formant structure, vibrato
///
/// # Arguments
/// * `spectral` - Spectral features
/// * `transients` - Transient detection result
/// * `f0` - Fundamental frequency (if detected)
///
/// # Returns
/// Detected instrument
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn detect_instrument(
    spectral: &SpectralFeatures,
    transients: &TransientResult,
    f0: Option<f32>,
) -> Instrument {
    // Feature extraction
    let is_harmonic = spectral.flatness < 0.3;
    let is_noisy = spectral.flatness > 0.7;
    let has_strong_transients = transients.avg_strength > 0.5;
    let low_centroid = spectral.centroid < 500.0;
    let high_centroid = spectral.centroid > 2000.0;

    // Decision tree classification
    if is_noisy && has_strong_transients {
        return Instrument::Drums;
    }

    if low_centroid && is_harmonic {
        return Instrument::Bass;
    }

    if let Some(fundamental) = f0 {
        // Voice characteristics
        if (80.0..=1000.0).contains(&fundamental) {
            // Check for formant-like structure in spectrum
            let has_formants = check_formant_structure(&spectral.magnitude_spectrum);
            if has_formants {
                return Instrument::Vocals;
            }
        }

        // Flute characteristics
        if fundamental >= 250.0 && is_harmonic && spectral.flatness < 0.15 {
            return Instrument::Flute;
        }

        // Piano characteristics
        if has_strong_transients && is_harmonic && spectral.bandwidth > 1000.0 {
            return Instrument::Piano;
        }

        // Guitar characteristics
        if is_harmonic && !has_strong_transients && fundamental >= 80.0 {
            return Instrument::Guitar;
        }

        // Violin characteristics
        if is_harmonic && high_centroid && fundamental >= 200.0 {
            return Instrument::Violin;
        }

        // Trumpet characteristics
        if is_harmonic && spectral.centroid > 800.0 && spectral.centroid < 2000.0 {
            return Instrument::Trumpet;
        }
    }

    // Synthesizer (often has non-natural spectral characteristics)
    if !is_noisy && !check_formant_structure(&spectral.magnitude_spectrum) {
        return Instrument::Synthesizer;
    }

    Instrument::Unknown
}

/// Check for formant-like structure in spectrum.
fn check_formant_structure(spectrum: &[f32]) -> bool {
    if spectrum.len() < 20 {
        return false;
    }

    // Look for multiple peaks in spectrum
    let mut peaks = 0;
    for i in 2..(spectrum.len() - 2) {
        if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] && spectrum[i] > 0.1 {
            peaks += 1;
        }
    }

    // Formants typically have 2-4 prominent peaks
    (2..=4).contains(&peaks)
}

/// Detect instrument with confidence scores.
#[must_use]
pub fn detect_instrument_scores(
    spectral: &SpectralFeatures,
    transients: &TransientResult,
    f0: Option<f32>,
) -> Vec<(Instrument, f32)> {
    let mut scores = vec![
        (Instrument::Piano, 0.0),
        (Instrument::Guitar, 0.0),
        (Instrument::Violin, 0.0),
        (Instrument::Flute, 0.0),
        (Instrument::Trumpet, 0.0),
        (Instrument::Drums, 0.0),
        (Instrument::Bass, 0.0),
        (Instrument::Vocals, 0.0),
        (Instrument::Synthesizer, 0.0),
    ];

    // Drums score
    if spectral.flatness > 0.5 && transients.avg_strength > 0.4 {
        scores[5].1 = 0.8;
    }

    // Bass score
    if spectral.centroid < 500.0 {
        scores[6].1 = 0.7;
    }

    if let Some(fundamental) = f0 {
        // Vocals score
        if (80.0..=1000.0).contains(&fundamental)
            && check_formant_structure(&spectral.magnitude_spectrum)
        {
            scores[7].1 = 0.8;
        }

        // Flute score
        if fundamental >= 250.0 && spectral.flatness < 0.15 {
            scores[3].1 = 0.7;
        }

        // Piano score
        if transients.avg_strength > 0.5 && spectral.bandwidth > 1000.0 {
            scores[0].1 = 0.7;
        }

        // Guitar score
        if spectral.flatness < 0.3 && fundamental >= 80.0 {
            scores[1].1 = 0.6;
        }

        // Violin score
        if spectral.centroid > 2000.0 && spectral.flatness < 0.3 {
            scores[2].1 = 0.6;
        }

        // Trumpet score
        if spectral.centroid > 800.0 && spectral.centroid < 2000.0 {
            scores[4].1 = 0.6;
        }
    }

    scores.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instrument_detection() {
        // Create features for drums (noisy, transient)
        let spectral = SpectralFeatures {
            centroid: 1000.0,
            flatness: 0.8,
            crest: 5.0,
            bandwidth: 3000.0,
            rolloff: 5000.0,
            flux: 0.0,
            magnitude_spectrum: vec![0.5; 100],
        };

        let transients = TransientResult {
            transient_times: vec![0.1, 0.2, 0.3],
            onset_strength: vec![0.8, 0.7, 0.9],
            num_transients: 3,
            avg_strength: 0.8,
        };

        let instrument = detect_instrument(&spectral, &transients, None);
        assert_eq!(instrument, Instrument::Drums);
    }

    #[test]
    fn test_instrument_scores() {
        let spectral = SpectralFeatures {
            centroid: 300.0,
            flatness: 0.2,
            crest: 3.0,
            bandwidth: 500.0,
            rolloff: 800.0,
            flux: 0.0,
            magnitude_spectrum: vec![0.5; 100],
        };

        let transients = TransientResult::default();

        let scores = detect_instrument_scores(&spectral, &transients, Some(100.0));

        // Bass should have high score due to low centroid
        let bass_score = scores
            .iter()
            .find(|(i, _)| *i == Instrument::Bass)
            .expect("unexpected None/Err")
            .1;
        assert!(bass_score > 0.5);
    }
}
