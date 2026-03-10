//! Pitch contour analysis.

use super::PitchResult;

/// Pitch contour representation.
#[derive(Debug, Clone)]
pub struct PitchContour {
    /// Time points in seconds
    pub times: Vec<f32>,
    /// Pitch values in Hz
    pub frequencies: Vec<f32>,
    /// Confidence values
    pub confidences: Vec<f32>,
}

impl PitchContour {
    /// Create pitch contour from tracking result.
    #[must_use]
    pub fn from_pitch_result(result: &PitchResult, hop_size: usize, sample_rate: f32) -> Self {
        let hop_duration = hop_size as f32 / sample_rate;
        let times: Vec<f32> = (0..result.estimates.len())
            .map(|i| i as f32 * hop_duration)
            .collect();

        Self {
            times,
            frequencies: result.estimates.clone(),
            confidences: result.confidences.clone(),
        }
    }

    /// Smooth the pitch contour using median filtering.
    pub fn smooth(&mut self, window_size: usize) {
        if window_size < 3 || self.frequencies.is_empty() {
            return;
        }

        let half_window = window_size / 2;
        let mut smoothed = Vec::with_capacity(self.frequencies.len());

        for i in 0..self.frequencies.len() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(self.frequencies.len());

            let mut window: Vec<f32> = self.frequencies[start..end]
                .iter()
                .zip(&self.confidences[start..end])
                .filter(|(_, &conf)| conf > 0.5)
                .map(|(&freq, _)| freq)
                .collect();

            if window.is_empty() {
                smoothed.push(0.0);
            } else {
                window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                smoothed.push(window[window.len() / 2]);
            }
        }

        self.frequencies = smoothed;
    }

    /// Compute pitch range (min to max).
    pub fn range(&self) -> (f32, f32) {
        let voiced: Vec<f32> = self
            .frequencies
            .iter()
            .zip(&self.confidences)
            .filter(|(_, &conf)| conf > 0.5)
            .map(|(&f, _)| f)
            .collect();

        if voiced.is_empty() {
            return (0.0, 0.0);
        }

        let min = voiced.iter().copied().fold(f32::INFINITY, f32::min);
        let max = voiced.iter().copied().fold(0.0_f32, f32::max);

        (min, max)
    }

    /// Compute pitch variation (standard deviation).
    #[must_use]
    pub fn variation(&self) -> f32 {
        let voiced: Vec<f32> = self
            .frequencies
            .iter()
            .zip(&self.confidences)
            .filter(|(_, &conf)| conf > 0.5)
            .map(|(&f, _)| f)
            .collect();

        if voiced.len() < 2 {
            return 0.0;
        }

        let mean = voiced.iter().sum::<f32>() / voiced.len() as f32;
        let variance =
            voiced.iter().map(|&f| (f - mean).powi(2)).sum::<f32>() / (voiced.len() - 1) as f32;

        variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pitch_contour() {
        let result = PitchResult {
            estimates: vec![440.0, 442.0, 438.0, 440.0],
            confidences: vec![0.9, 0.85, 0.88, 0.92],
            mean_f0: 440.0,
            voicing_rate: 1.0,
        };

        let contour = PitchContour::from_pitch_result(&result, 512, 44100.0);
        assert_eq!(contour.times.len(), 4);
        assert_eq!(contour.frequencies.len(), 4);

        let (min, max) = contour.range();
        assert!(min >= 438.0 && min <= 439.0);
        assert!(max >= 441.0 && max <= 443.0);

        let variation = contour.variation();
        assert!(variation > 0.0 && variation < 5.0);
    }

    #[test]
    fn test_contour_smoothing() {
        let result = PitchResult {
            estimates: vec![440.0, 500.0, 442.0, 438.0, 440.0],
            confidences: vec![0.9, 0.3, 0.85, 0.88, 0.92],
            mean_f0: 440.0,
            voicing_rate: 0.8,
        };

        let mut contour = PitchContour::from_pitch_result(&result, 512, 44100.0);
        contour.smooth(3);

        // The 500 Hz outlier with low confidence should be filtered out
        assert!(contour.frequencies[1] < 460.0);
    }
}
