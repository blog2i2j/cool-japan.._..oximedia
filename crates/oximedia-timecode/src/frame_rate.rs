#![allow(dead_code)]
//! Rational frame-rate representation independent of the SMPTE enum in lib.rs.

/// A rational frame rate expressed as `numerator / denominator`.
///
/// This complements the [`crate::FrameRate`] enum with arbitrary-precision
/// rational arithmetic so that custom frame rates (e.g. 48000/1001) can be
/// represented without adding new enum variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRateRatio {
    /// Numerator of the frame rate fraction.
    pub numerator: u32,
    /// Denominator of the frame rate fraction.
    pub denominator: u32,
}

impl FrameRateRatio {
    /// Create a new [`FrameRateRatio`].
    ///
    /// Returns `None` if `denominator` is zero.
    pub fn new(numerator: u32, denominator: u32) -> Option<Self> {
        if denominator == 0 {
            None
        } else {
            Some(Self {
                numerator,
                denominator,
            })
        }
    }

    /// Exact floating-point value of the frame rate.
    #[allow(clippy::cast_precision_loss)]
    pub fn fps_f64(&self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }

    /// Whether this frame rate is compatible with SMPTE drop-frame timecode.
    ///
    /// Drop frame is only defined for the 30000/1001 (≈ 29.97) and
    /// 60000/1001 (≈ 59.94) rates.
    pub fn is_drop_frame_compatible(&self) -> bool {
        matches!(
            (self.numerator, self.denominator),
            (30000, 1001) | (60000, 1001)
        )
    }

    /// Whether this ratio is numerically equal to `other` (cross-multiply check).
    pub fn matches(&self, other: &FrameRateRatio) -> bool {
        // Cross-multiply to avoid floating-point comparison.
        (self.numerator as u64) * (other.denominator as u64)
            == (other.numerator as u64) * (self.denominator as u64)
    }

    /// A list of common broadcast frame rates.
    pub fn common_frame_rates() -> Vec<FrameRateRatio> {
        vec![
            FrameRateRatio {
                numerator: 24000,
                denominator: 1001,
            }, // 23.976
            FrameRateRatio {
                numerator: 24,
                denominator: 1,
            }, // 24
            FrameRateRatio {
                numerator: 25,
                denominator: 1,
            }, // 25
            FrameRateRatio {
                numerator: 30000,
                denominator: 1001,
            }, // 29.97
            FrameRateRatio {
                numerator: 30,
                denominator: 1,
            }, // 30
            FrameRateRatio {
                numerator: 50,
                denominator: 1,
            }, // 50
            FrameRateRatio {
                numerator: 60000,
                denominator: 1001,
            }, // 59.94
            FrameRateRatio {
                numerator: 60,
                denominator: 1,
            }, // 60
        ]
    }

    /// Nominal integer frames per second (rounded).
    #[allow(clippy::cast_possible_truncation)]
    pub fn nominal_fps(&self) -> u32 {
        ((self.numerator as f64 / self.denominator as f64).round()) as u32
    }
}

impl std::fmt::Display for FrameRateRatio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.denominator == 1 {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid() {
        let fr = FrameRateRatio::new(25, 1).expect("valid frame rate ratio");
        assert_eq!(fr.numerator, 25);
        assert_eq!(fr.denominator, 1);
    }

    #[test]
    fn test_new_zero_denominator() {
        assert!(FrameRateRatio::new(30, 0).is_none());
    }

    #[test]
    fn test_fps_f64_exact() {
        let fr = FrameRateRatio::new(25, 1).expect("valid frame rate ratio");
        assert!((fr.fps_f64() - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_fps_f64_fractional() {
        let fr = FrameRateRatio::new(30000, 1001).expect("valid frame rate ratio");
        assert!((fr.fps_f64() - 29.97002997).abs() < 1e-6);
    }

    #[test]
    fn test_is_drop_frame_compatible_true() {
        let fr2997 = FrameRateRatio::new(30000, 1001).expect("valid frame rate ratio");
        let fr5994 = FrameRateRatio::new(60000, 1001).expect("valid frame rate ratio");
        assert!(fr2997.is_drop_frame_compatible());
        assert!(fr5994.is_drop_frame_compatible());
    }

    #[test]
    fn test_is_drop_frame_compatible_false() {
        let fr = FrameRateRatio::new(25, 1).expect("valid frame rate ratio");
        assert!(!fr.is_drop_frame_compatible());
    }

    #[test]
    fn test_matches_equal() {
        let a = FrameRateRatio::new(25, 1).expect("valid frame rate ratio");
        let b = FrameRateRatio::new(50, 2).expect("valid frame rate ratio");
        assert!(a.matches(&b));
    }

    #[test]
    fn test_matches_not_equal() {
        let a = FrameRateRatio::new(24, 1).expect("valid frame rate ratio");
        let b = FrameRateRatio::new(25, 1).expect("valid frame rate ratio");
        assert!(!a.matches(&b));
    }

    #[test]
    fn test_common_frame_rates_count() {
        let rates = FrameRateRatio::common_frame_rates();
        assert_eq!(rates.len(), 8);
    }

    #[test]
    fn test_common_frame_rates_contains_25() {
        let rates = FrameRateRatio::common_frame_rates();
        assert!(rates
            .iter()
            .any(|r| r.numerator == 25 && r.denominator == 1));
    }

    #[test]
    fn test_nominal_fps_exact() {
        let fr = FrameRateRatio::new(30, 1).expect("valid frame rate ratio");
        assert_eq!(fr.nominal_fps(), 30);
    }

    #[test]
    fn test_nominal_fps_rounded() {
        let fr = FrameRateRatio::new(30000, 1001).expect("valid frame rate ratio");
        // 29.97 rounds to 30
        assert_eq!(fr.nominal_fps(), 30);
    }

    #[test]
    fn test_display_integer() {
        let fr = FrameRateRatio::new(25, 1).expect("valid frame rate ratio");
        assert_eq!(fr.to_string(), "25");
    }

    #[test]
    fn test_display_fractional() {
        let fr = FrameRateRatio::new(30000, 1001).expect("valid frame rate ratio");
        assert_eq!(fr.to_string(), "30000/1001");
    }
}
