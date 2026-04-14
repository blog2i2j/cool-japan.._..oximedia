//! Missing frame substitution for damaged video streams.
//!
//! When one or more consecutive frames are missing from a video stream,
//! [`FrameSubstitutor`] generates substitute frames using a previous valid
//! frame as the reference.  The default strategy is frame-freeze (copy the
//! previous frame), with optional lightweight fade-to-black for multi-frame
//! gaps.

/// Substitution strategy for missing frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubstitutionStrategy {
    /// Repeat the previous frame unchanged for every missing frame.
    Freeze,
    /// Fade the previous frame toward black linearly across the gap.
    FadeToBlack,
    /// Fill with solid black (all zeros).
    Black,
}

impl Default for SubstitutionStrategy {
    fn default() -> Self {
        Self::Freeze
    }
}

/// Generator for substitute frames when media data is missing.
#[derive(Debug, Default)]
pub struct FrameSubstitutor {
    strategy: SubstitutionStrategy,
}

impl FrameSubstitutor {
    /// Create a new substitutor with the default [`SubstitutionStrategy::Freeze`] strategy.
    #[must_use]
    pub fn new() -> Self {
        Self {
            strategy: SubstitutionStrategy::Freeze,
        }
    }

    /// Create a substitutor with an explicit strategy.
    #[must_use]
    pub fn with_strategy(strategy: SubstitutionStrategy) -> Self {
        Self { strategy }
    }

    /// Generate `missing_count` substitute frames using `prev_frame` as the
    /// reference.
    ///
    /// Returns a `Vec<Vec<u8>>` where each inner `Vec<u8>` represents one
    /// substitute frame's raw byte data.  Returns an empty `Vec` when
    /// `missing_count == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_repair::frame_sub::FrameSubstitutor;
    ///
    /// let prev = vec![128u8; 1920 * 1080 * 3]; // 1080p RGB
    /// let subs = FrameSubstitutor::new().substitute(&prev, 3);
    /// assert_eq!(subs.len(), 3);
    /// assert_eq!(subs[0], prev);
    /// ```
    #[must_use]
    pub fn substitute(&self, prev_frame: &[u8], missing_count: usize) -> Vec<Vec<u8>> {
        if missing_count == 0 {
            return Vec::new();
        }

        match self.strategy {
            SubstitutionStrategy::Freeze => {
                (0..missing_count)
                    .map(|_| prev_frame.to_vec())
                    .collect()
            }
            SubstitutionStrategy::FadeToBlack => {
                (0..missing_count)
                    .map(|i| {
                        // Linear fade: frame 0 == prev_frame, last frame == black
                        let alpha = if missing_count == 1 {
                            0.0f32
                        } else {
                            i as f32 / (missing_count - 1) as f32
                        };
                        prev_frame
                            .iter()
                            .map(|&b| (b as f32 * (1.0 - alpha)).round() as u8)
                            .collect()
                    })
                    .collect()
            }
            SubstitutionStrategy::Black => {
                (0..missing_count)
                    .map(|_| vec![0u8; prev_frame.len()])
                    .collect()
            }
        }
    }

    /// Convenience static method: freeze-substitute using `prev_frame`.
    ///
    /// Equivalent to `FrameSubstitutor::new().substitute(prev_frame, missing_count)`.
    #[must_use]
    pub fn freeze(prev_frame: &[u8], missing_count: usize) -> Vec<Vec<u8>> {
        Self::new().substitute(prev_frame, missing_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME_SIZE: usize = 16; // tiny test frame

    fn make_frame(val: u8) -> Vec<u8> {
        vec![val; FRAME_SIZE]
    }

    #[test]
    fn zero_missing_returns_empty() {
        let prev = make_frame(100);
        let subs = FrameSubstitutor::new().substitute(&prev, 0);
        assert!(subs.is_empty());
    }

    #[test]
    fn freeze_returns_exact_copies() {
        let prev = make_frame(200);
        let subs = FrameSubstitutor::new().substitute(&prev, 4);
        assert_eq!(subs.len(), 4);
        for f in &subs {
            assert_eq!(f, &prev);
        }
    }

    #[test]
    fn freeze_static_method() {
        let prev = make_frame(50);
        let subs = FrameSubstitutor::freeze(&prev, 2);
        assert_eq!(subs.len(), 2);
        assert_eq!(subs[0], prev);
    }

    #[test]
    fn black_fills_with_zeros() {
        let prev = make_frame(0xFF);
        let subs = FrameSubstitutor::with_strategy(SubstitutionStrategy::Black)
            .substitute(&prev, 3);
        assert_eq!(subs.len(), 3);
        for f in &subs {
            assert!(f.iter().all(|&b| b == 0));
        }
    }

    #[test]
    fn fade_to_black_single_frame_is_black() {
        let prev = make_frame(200);
        let subs = FrameSubstitutor::with_strategy(SubstitutionStrategy::FadeToBlack)
            .substitute(&prev, 1);
        // Single frame: alpha = 0.0 → should equal prev
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0], prev);
    }

    #[test]
    fn fade_to_black_decreases_brightness() {
        let prev = make_frame(200);
        let subs = FrameSubstitutor::with_strategy(SubstitutionStrategy::FadeToBlack)
            .substitute(&prev, 5);
        assert_eq!(subs.len(), 5);
        // First frame should equal prev (alpha=0)
        assert_eq!(subs[0], prev);
        // Last frame should be darker than first
        let last_avg: f32 = subs[4].iter().map(|&b| b as f32).sum::<f32>() / FRAME_SIZE as f32;
        let first_avg = 200.0f32;
        assert!(
            last_avg < first_avg,
            "last={last_avg} should be darker than first={first_avg}"
        );
    }

    #[test]
    fn substitute_preserves_frame_size() {
        let prev = vec![42u8; 1920 * 1080 * 3];
        let subs = FrameSubstitutor::new().substitute(&prev, 2);
        for f in &subs {
            assert_eq!(f.len(), prev.len());
        }
    }

    #[test]
    fn default_strategy_is_freeze() {
        assert_eq!(SubstitutionStrategy::default(), SubstitutionStrategy::Freeze);
    }
}
