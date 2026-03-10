//! Frame rate conversion helpers for the IVTC filter.

use oximedia_core::Rational;

use super::TelecinePattern;

/// Common frame rates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StandardFrameRate {
    /// Film rate: 24 fps.
    Film24,
    /// Film rate: 23.976 fps (24000/1001).
    Film23976,
    /// PAL rate: 25 fps.
    Pal25,
    /// NTSC rate: 29.97 fps (30000/1001).
    Ntsc2997,
    /// NTSC rate: 30 fps.
    Ntsc30,
    /// High frame rate: 50 fps.
    High50,
    /// High frame rate: 59.94 fps (60000/1001).
    High5994,
    /// High frame rate: 60 fps.
    High60,
}

impl StandardFrameRate {
    /// Get the rational representation.
    #[must_use]
    pub fn as_rational(&self) -> Rational {
        match self {
            Self::Film24 => Rational { num: 24, den: 1 },
            Self::Film23976 => Rational {
                num: 24000,
                den: 1001,
            },
            Self::Pal25 => Rational { num: 25, den: 1 },
            Self::Ntsc2997 => Rational {
                num: 30000,
                den: 1001,
            },
            Self::Ntsc30 => Rational { num: 30, den: 1 },
            Self::High50 => Rational { num: 50, den: 1 },
            Self::High5994 => Rational {
                num: 60000,
                den: 1001,
            },
            Self::High60 => Rational { num: 60, den: 1 },
        }
    }

    /// Get the decimal representation.
    #[must_use]
    pub fn as_f64(&self) -> f64 {
        let r = self.as_rational();
        r.num as f64 / r.den as f64
    }

    /// Detect frame rate from rational.
    #[must_use]
    pub fn from_rational(rate: Rational) -> Option<Self> {
        let fps = rate.num as f64 / rate.den as f64;

        if (fps - 23.976).abs() < 0.01 {
            Some(Self::Film23976)
        } else if (fps - 24.0).abs() < 0.01 {
            Some(Self::Film24)
        } else if (fps - 25.0).abs() < 0.01 {
            Some(Self::Pal25)
        } else if (fps - 29.97).abs() < 0.01 {
            Some(Self::Ntsc2997)
        } else if (fps - 30.0).abs() < 0.01 {
            Some(Self::Ntsc30)
        } else if (fps - 50.0).abs() < 0.01 {
            Some(Self::High50)
        } else if (fps - 59.94).abs() < 0.01 {
            Some(Self::High5994)
        } else if (fps - 60.0).abs() < 0.01 {
            Some(Self::High60)
        } else {
            None
        }
    }
}

/// Calculate target frame rate after IVTC.
#[must_use]
pub fn calculate_ivtc_framerate(source_rate: Rational, pattern: TelecinePattern) -> Rational {
    let cycle_len = pattern.cycle_length();
    let output_frames = pattern.output_frames();

    if cycle_len == 0 || output_frames == 0 {
        return source_rate;
    }

    // Calculate decimation ratio
    let ratio = output_frames as i64 / cycle_len as i64;

    Rational {
        num: source_rate.num * ratio,
        den: source_rate.den,
    }
}

/// Detect telecine pattern from frame rate ratio.
#[must_use]
pub fn detect_pattern_from_rates(
    source: StandardFrameRate,
    expected_output: StandardFrameRate,
) -> Option<TelecinePattern> {
    let src_fps = source.as_f64();
    let out_fps = expected_output.as_f64();
    let ratio = src_fps / out_fps;

    // 3:2 pulldown: 29.97 -> 23.976 (ratio ~1.25)
    if (ratio - 1.25).abs() < 0.01 {
        Some(TelecinePattern::Pattern32)
    }
    // 2:2 pulldown: 25 -> 25 (ratio 1.0)
    else if (ratio - 1.0).abs() < 0.01 {
        Some(TelecinePattern::Pattern22)
    }
    // Euro pulldown: 25 -> 24 (ratio ~1.04)
    else if (ratio - 1.04167).abs() < 0.01 {
        Some(TelecinePattern::EuroPulldown)
    } else {
        None
    }
}
