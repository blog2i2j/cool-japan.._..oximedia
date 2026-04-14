//! Audio sample format definitions.
//!
//! This module provides the [`SampleFormat`] enum representing the various
//! ways audio sample data can be stored in memory.

/// Audio sample format.
///
/// Defines how audio samples are stored in memory, including bit depth,
/// signedness, and whether samples are interleaved or planar.
///
/// Formats ending with 'p' are planar (one plane per channel).
///
/// # Examples
///
/// ```
/// use oximedia_core::types::SampleFormat;
///
/// let format = SampleFormat::F32;
/// assert!(!format.is_planar());
/// assert_eq!(format.bytes_per_sample(), 4);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[derive(Default)]
pub enum SampleFormat {
    /// Unsigned 8-bit integer, interleaved.
    U8,

    /// Signed 16-bit integer, interleaved.
    S16,

    /// Signed 32-bit integer, interleaved.
    S32,

    /// 32-bit floating point, interleaved.
    #[default]
    F32,

    /// 64-bit floating point, interleaved.
    F64,

    /// Signed 16-bit integer, planar.
    S16p,

    /// Signed 32-bit integer, planar.
    S32p,

    /// 32-bit floating point, planar.
    F32p,

    /// 64-bit floating point, planar.
    F64p,

    /// Signed 24-bit integer, interleaved (stored in 3 bytes per sample).
    /// Common in professional audio interfaces and WAV files.
    S24,

    /// Signed 24-bit integer, planar (stored in 3 bytes per sample).
    /// Planar variant for professional audio pipelines.
    S24p,
}

impl SampleFormat {
    /// Returns the number of bytes per sample.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::SampleFormat;
    ///
    /// assert_eq!(SampleFormat::U8.bytes_per_sample(), 1);
    /// assert_eq!(SampleFormat::S16.bytes_per_sample(), 2);
    /// assert_eq!(SampleFormat::F32.bytes_per_sample(), 4);
    /// assert_eq!(SampleFormat::F64.bytes_per_sample(), 8);
    /// ```
    #[must_use]
    pub const fn bytes_per_sample(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::S16 | Self::S16p => 2,
            Self::S24 | Self::S24p => 3,
            Self::S32 | Self::S32p | Self::F32 | Self::F32p => 4,
            Self::F64 | Self::F64p => 8,
        }
    }

    /// Returns whether this format uses planar storage.
    ///
    /// Planar formats store each channel in a separate contiguous
    /// memory region, as opposed to interleaved formats where
    /// samples alternate between channels.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::SampleFormat;
    ///
    /// assert!(!SampleFormat::F32.is_planar());
    /// assert!(SampleFormat::F32p.is_planar());
    /// ```
    #[must_use]
    pub const fn is_planar(&self) -> bool {
        matches!(
            self,
            Self::S16p | Self::S24p | Self::S32p | Self::F32p | Self::F64p
        )
    }

    /// Returns the number of bits per sample.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::SampleFormat;
    ///
    /// assert_eq!(SampleFormat::U8.bits_per_sample(), 8);
    /// assert_eq!(SampleFormat::S16.bits_per_sample(), 16);
    /// assert_eq!(SampleFormat::F32.bits_per_sample(), 32);
    /// ```
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn bits_per_sample(&self) -> u32 {
        (self.bytes_per_sample() * 8) as u32
    }

    /// Returns whether this format uses floating point samples.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::SampleFormat;
    ///
    /// assert!(SampleFormat::F32.is_float());
    /// assert!(SampleFormat::F64p.is_float());
    /// assert!(!SampleFormat::S16.is_float());
    /// ```
    #[must_use]
    pub const fn is_float(&self) -> bool {
        matches!(self, Self::F32 | Self::F64 | Self::F32p | Self::F64p)
    }

    /// Returns whether this format uses signed integer samples.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::SampleFormat;
    ///
    /// assert!(SampleFormat::S16.is_signed());
    /// assert!(SampleFormat::S32p.is_signed());
    /// assert!(!SampleFormat::U8.is_signed());
    /// ```
    #[must_use]
    pub const fn is_signed(&self) -> bool {
        !matches!(self, Self::U8)
    }

    /// Returns the packed (interleaved) equivalent of this format.
    ///
    /// If the format is already packed, returns self.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::SampleFormat;
    ///
    /// assert_eq!(SampleFormat::F32p.to_packed(), SampleFormat::F32);
    /// assert_eq!(SampleFormat::S16.to_packed(), SampleFormat::S16);
    /// ```
    #[must_use]
    pub const fn to_packed(&self) -> Self {
        match self {
            Self::S16p => Self::S16,
            Self::S24p => Self::S24,
            Self::S32p => Self::S32,
            Self::F32p => Self::F32,
            Self::F64p => Self::F64,
            other => *other,
        }
    }

    /// Returns the planar equivalent of this format.
    ///
    /// If the format is already planar, returns self.
    /// Note: U8 has no planar equivalent and returns U8.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::SampleFormat;
    ///
    /// assert_eq!(SampleFormat::F32.to_planar(), SampleFormat::F32p);
    /// assert_eq!(SampleFormat::S16p.to_planar(), SampleFormat::S16p);
    /// ```
    #[must_use]
    pub const fn to_planar(&self) -> Self {
        match self {
            Self::S16 => Self::S16p,
            Self::S24 => Self::S24p,
            Self::S32 => Self::S32p,
            Self::F32 => Self::F32p,
            Self::F64 => Self::F64p,
            other => *other,
        }
    }
}

impl SampleFormat {
    /// Returns the bit depth of this sample format.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::SampleFormat;
    ///
    /// assert_eq!(SampleFormat::U8.bit_depth(), 8);
    /// assert_eq!(SampleFormat::S16.bit_depth(), 16);
    /// assert_eq!(SampleFormat::S24.bit_depth(), 24);
    /// assert_eq!(SampleFormat::F32.bit_depth(), 32);
    /// assert_eq!(SampleFormat::F64.bit_depth(), 64);
    /// ```
    #[must_use]
    pub const fn bit_depth(&self) -> u8 {
        match self {
            Self::U8 => 8,
            Self::S16 | Self::S16p => 16,
            Self::S24 | Self::S24p => 24,
            Self::S32 | Self::S32p | Self::F32 | Self::F32p => 32,
            Self::F64 | Self::F64p => 64,
        }
    }

    /// Returns the theoretical dynamic range in decibels for integer formats.
    ///
    /// For floating-point formats, returns an approximate practical range.
    /// The formula for integer formats is `6.02 * bits + 1.76` dB.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::SampleFormat;
    ///
    /// let dr = SampleFormat::S16.dynamic_range_db();
    /// assert!((dr - 98.09).abs() < 0.1); // ~96.3 + 1.76
    /// ```
    #[must_use]
    pub fn dynamic_range_db(&self) -> f64 {
        match self {
            Self::U8 => 6.02 * 8.0 + 1.76,                // ~49.9 dB
            Self::S16 | Self::S16p => 6.02 * 16.0 + 1.76, // ~98.1 dB
            Self::S24 | Self::S24p => 6.02 * 24.0 + 1.76, // ~146.2 dB
            Self::S32 | Self::S32p => 6.02 * 32.0 + 1.76, // ~194.4 dB
            Self::F32 | Self::F32p => 144.0,              // ~24-bit equivalent practical range
            Self::F64 | Self::F64p => 1022.0, // exponent range gives enormous dynamic range
        }
    }

    /// Returns the buffer size in bytes needed for `sample_count` samples
    /// across `channels` channels.
    ///
    /// For interleaved formats, all channels are stored in a single buffer.
    /// For planar formats, this returns the total size of all per-channel planes.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::SampleFormat;
    ///
    /// // 1024 stereo F32 samples: 1024 * 2 * 4 = 8192 bytes
    /// assert_eq!(SampleFormat::F32.buffer_size(1024, 2), 8192);
    ///
    /// // Planar is the same total: 1024 * 4 * 2 planes = 8192
    /// assert_eq!(SampleFormat::F32p.buffer_size(1024, 2), 8192);
    /// ```
    #[must_use]
    pub const fn buffer_size(&self, sample_count: usize, channels: usize) -> usize {
        self.bytes_per_sample() * sample_count * channels
    }
}

impl std::str::FromStr for SampleFormat {
    type Err = crate::OxiError;

    /// Parses a sample format name string.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::SampleFormat;
    ///
    /// let fmt: SampleFormat = "f32".parse().expect("should parse");
    /// assert_eq!(fmt, SampleFormat::F32);
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "u8" => Ok(Self::U8),
            "s16" => Ok(Self::S16),
            "s24" => Ok(Self::S24),
            "s32" => Ok(Self::S32),
            "f32" | "float" | "float32" => Ok(Self::F32),
            "f64" | "double" | "float64" => Ok(Self::F64),
            "s16p" => Ok(Self::S16p),
            "s24p" => Ok(Self::S24p),
            "s32p" => Ok(Self::S32p),
            "f32p" | "floatp" => Ok(Self::F32p),
            "f64p" | "doublep" => Ok(Self::F64p),
            _ => Err(crate::OxiError::Unsupported(format!(
                "Unknown sample format: {s}"
            ))),
        }
    }
}

impl std::fmt::Display for SampleFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::U8 => "u8",
            Self::S16 => "s16",
            Self::S24 => "s24",
            Self::S32 => "s32",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::S16p => "s16p",
            Self::S24p => "s24p",
            Self::S32p => "s32p",
            Self::F32p => "f32p",
            Self::F64p => "f64p",
        };
        write!(f, "{name}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_per_sample() {
        assert_eq!(SampleFormat::U8.bytes_per_sample(), 1);
        assert_eq!(SampleFormat::S16.bytes_per_sample(), 2);
        assert_eq!(SampleFormat::S16p.bytes_per_sample(), 2);
        assert_eq!(SampleFormat::S32.bytes_per_sample(), 4);
        assert_eq!(SampleFormat::F32.bytes_per_sample(), 4);
        assert_eq!(SampleFormat::F64.bytes_per_sample(), 8);
    }

    #[test]
    fn test_is_planar() {
        assert!(!SampleFormat::U8.is_planar());
        assert!(!SampleFormat::S16.is_planar());
        assert!(!SampleFormat::F32.is_planar());
        assert!(SampleFormat::S16p.is_planar());
        assert!(SampleFormat::F32p.is_planar());
        assert!(SampleFormat::F64p.is_planar());
    }

    #[test]
    fn test_is_float() {
        assert!(!SampleFormat::U8.is_float());
        assert!(!SampleFormat::S16.is_float());
        assert!(SampleFormat::F32.is_float());
        assert!(SampleFormat::F64.is_float());
        assert!(SampleFormat::F32p.is_float());
    }

    #[test]
    fn test_is_signed() {
        assert!(!SampleFormat::U8.is_signed());
        assert!(SampleFormat::S16.is_signed());
        assert!(SampleFormat::S32.is_signed());
        assert!(SampleFormat::F32.is_signed());
    }

    #[test]
    fn test_to_packed() {
        assert_eq!(SampleFormat::F32p.to_packed(), SampleFormat::F32);
        assert_eq!(SampleFormat::S16p.to_packed(), SampleFormat::S16);
        assert_eq!(SampleFormat::F32.to_packed(), SampleFormat::F32);
    }

    #[test]
    fn test_to_planar() {
        assert_eq!(SampleFormat::F32.to_planar(), SampleFormat::F32p);
        assert_eq!(SampleFormat::S16.to_planar(), SampleFormat::S16p);
        assert_eq!(SampleFormat::F32p.to_planar(), SampleFormat::F32p);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", SampleFormat::F32), "f32");
        assert_eq!(format!("{}", SampleFormat::S16p), "s16p");
    }

    // ── S24 / S24p tests ─────────────────────────────────────────────

    #[test]
    fn test_s24_bytes_per_sample() {
        assert_eq!(SampleFormat::S24.bytes_per_sample(), 3);
        assert_eq!(SampleFormat::S24p.bytes_per_sample(), 3);
    }

    #[test]
    fn test_s24_bits_per_sample() {
        assert_eq!(SampleFormat::S24.bits_per_sample(), 24);
        assert_eq!(SampleFormat::S24p.bits_per_sample(), 24);
    }

    #[test]
    fn test_s24_is_planar() {
        assert!(!SampleFormat::S24.is_planar());
        assert!(SampleFormat::S24p.is_planar());
    }

    #[test]
    fn test_s24_is_float() {
        assert!(!SampleFormat::S24.is_float());
        assert!(!SampleFormat::S24p.is_float());
    }

    #[test]
    fn test_s24_is_signed() {
        assert!(SampleFormat::S24.is_signed());
        assert!(SampleFormat::S24p.is_signed());
    }

    #[test]
    fn test_s24_to_packed() {
        assert_eq!(SampleFormat::S24p.to_packed(), SampleFormat::S24);
        assert_eq!(SampleFormat::S24.to_packed(), SampleFormat::S24);
    }

    #[test]
    fn test_s24_to_planar() {
        assert_eq!(SampleFormat::S24.to_planar(), SampleFormat::S24p);
        assert_eq!(SampleFormat::S24p.to_planar(), SampleFormat::S24p);
    }

    #[test]
    fn test_s24_display() {
        assert_eq!(format!("{}", SampleFormat::S24), "s24");
        assert_eq!(format!("{}", SampleFormat::S24p), "s24p");
    }

    #[test]
    fn test_f64_properties() {
        // F64 already existed but verify full coverage
        assert_eq!(SampleFormat::F64.bytes_per_sample(), 8);
        assert!(SampleFormat::F64.is_float());
        assert!(SampleFormat::F64.is_signed());
        assert!(!SampleFormat::F64.is_planar());
        assert_eq!(SampleFormat::F64.to_planar(), SampleFormat::F64p);
        assert_eq!(SampleFormat::F64p.to_packed(), SampleFormat::F64);
        assert_eq!(format!("{}", SampleFormat::F64), "f64");
        assert_eq!(format!("{}", SampleFormat::F64p), "f64p");
    }

    // ── dynamic_range_db tests ──────────────────────────────────────

    #[test]
    fn test_dynamic_range_u8() {
        let dr = SampleFormat::U8.dynamic_range_db();
        assert!((dr - 49.92).abs() < 0.1);
    }

    #[test]
    fn test_dynamic_range_s16() {
        let dr = SampleFormat::S16.dynamic_range_db();
        assert!((dr - 98.08).abs() < 0.1);
        // Planar variant has same range
        assert!((SampleFormat::S16p.dynamic_range_db() - dr).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dynamic_range_s24() {
        let dr = SampleFormat::S24.dynamic_range_db();
        assert!((dr - 146.24).abs() < 0.1);
        assert!((SampleFormat::S24p.dynamic_range_db() - dr).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dynamic_range_s32() {
        let dr = SampleFormat::S32.dynamic_range_db();
        assert!((dr - 194.4).abs() < 0.1);
    }

    #[test]
    fn test_dynamic_range_f32() {
        assert!(SampleFormat::F32.dynamic_range_db() > 140.0);
        assert!(
            (SampleFormat::F32p.dynamic_range_db() - SampleFormat::F32.dynamic_range_db()).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_dynamic_range_f64() {
        assert!(SampleFormat::F64.dynamic_range_db() > SampleFormat::F32.dynamic_range_db());
        assert!(
            (SampleFormat::F64p.dynamic_range_db() - SampleFormat::F64.dynamic_range_db()).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_dynamic_range_ordering() {
        // U8 < S16 < F32 < S24 < S32 < F64
        // F32 practical range (~144 dB) is between S16 (~98) and S24 (~146)
        assert!(SampleFormat::U8.dynamic_range_db() < SampleFormat::S16.dynamic_range_db());
        assert!(SampleFormat::S16.dynamic_range_db() < SampleFormat::F32.dynamic_range_db());
        assert!(SampleFormat::F32.dynamic_range_db() < SampleFormat::S24.dynamic_range_db());
        assert!(SampleFormat::S24.dynamic_range_db() < SampleFormat::S32.dynamic_range_db());
        assert!(SampleFormat::S32.dynamic_range_db() < SampleFormat::F64.dynamic_range_db());
    }

    // ── buffer_size tests ───────────────────────────────────────────

    #[test]
    fn test_buffer_size_f32_stereo() {
        assert_eq!(SampleFormat::F32.buffer_size(1024, 2), 1024 * 2 * 4);
    }

    #[test]
    fn test_buffer_size_s16_mono() {
        assert_eq!(SampleFormat::S16.buffer_size(48000, 1), 48000 * 2);
    }

    #[test]
    fn test_buffer_size_s24_stereo() {
        assert_eq!(SampleFormat::S24.buffer_size(1024, 2), 1024 * 2 * 3);
    }

    #[test]
    fn test_buffer_size_f64_5_1() {
        assert_eq!(SampleFormat::F64.buffer_size(1024, 6), 1024 * 6 * 8);
    }

    #[test]
    fn test_buffer_size_planar_equals_interleaved() {
        // Total byte count is the same for planar vs interleaved
        assert_eq!(
            SampleFormat::F32.buffer_size(1024, 2),
            SampleFormat::F32p.buffer_size(1024, 2),
        );
    }

    // ── FromStr tests ───────────────────────────────────────────────

    #[test]
    fn test_from_str_all_formats() {
        assert_eq!(
            "u8".parse::<SampleFormat>().expect("parse"),
            SampleFormat::U8
        );
        assert_eq!(
            "s16".parse::<SampleFormat>().expect("parse"),
            SampleFormat::S16
        );
        assert_eq!(
            "s24".parse::<SampleFormat>().expect("parse"),
            SampleFormat::S24
        );
        assert_eq!(
            "s32".parse::<SampleFormat>().expect("parse"),
            SampleFormat::S32
        );
        assert_eq!(
            "f32".parse::<SampleFormat>().expect("parse"),
            SampleFormat::F32
        );
        assert_eq!(
            "f64".parse::<SampleFormat>().expect("parse"),
            SampleFormat::F64
        );
        assert_eq!(
            "s16p".parse::<SampleFormat>().expect("parse"),
            SampleFormat::S16p
        );
        assert_eq!(
            "s24p".parse::<SampleFormat>().expect("parse"),
            SampleFormat::S24p
        );
        assert_eq!(
            "s32p".parse::<SampleFormat>().expect("parse"),
            SampleFormat::S32p
        );
        assert_eq!(
            "f32p".parse::<SampleFormat>().expect("parse"),
            SampleFormat::F32p
        );
        assert_eq!(
            "f64p".parse::<SampleFormat>().expect("parse"),
            SampleFormat::F64p
        );
    }

    #[test]
    fn test_from_str_aliases() {
        assert_eq!(
            "float".parse::<SampleFormat>().expect("parse"),
            SampleFormat::F32
        );
        assert_eq!(
            "float32".parse::<SampleFormat>().expect("parse"),
            SampleFormat::F32
        );
        assert_eq!(
            "double".parse::<SampleFormat>().expect("parse"),
            SampleFormat::F64
        );
        assert_eq!(
            "float64".parse::<SampleFormat>().expect("parse"),
            SampleFormat::F64
        );
        assert_eq!(
            "floatp".parse::<SampleFormat>().expect("parse"),
            SampleFormat::F32p
        );
        assert_eq!(
            "doublep".parse::<SampleFormat>().expect("parse"),
            SampleFormat::F64p
        );
    }

    #[test]
    fn test_from_str_case_insensitive() {
        assert_eq!(
            "F32".parse::<SampleFormat>().expect("parse"),
            SampleFormat::F32
        );
        assert_eq!(
            "S16P".parse::<SampleFormat>().expect("parse"),
            SampleFormat::S16p
        );
        assert_eq!(
            "FLOAT".parse::<SampleFormat>().expect("parse"),
            SampleFormat::F32
        );
    }

    #[test]
    fn test_from_str_unknown() {
        assert!("aac".parse::<SampleFormat>().is_err());
        assert!("unknown".parse::<SampleFormat>().is_err());
    }

    #[test]
    fn test_from_str_roundtrip() {
        let formats = [
            SampleFormat::U8,
            SampleFormat::S16,
            SampleFormat::S24,
            SampleFormat::S32,
            SampleFormat::F32,
            SampleFormat::F64,
            SampleFormat::S16p,
            SampleFormat::S24p,
            SampleFormat::S32p,
            SampleFormat::F32p,
            SampleFormat::F64p,
        ];
        for fmt in &formats {
            let s = format!("{fmt}");
            let parsed: SampleFormat = s.parse().expect("roundtrip should work");
            assert_eq!(*fmt, parsed);
        }
    }
}
