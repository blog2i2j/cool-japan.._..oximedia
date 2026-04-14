#![allow(dead_code)]

//! Time scale conversions for precision timing applications.
//!
//! This module handles conversions between different time scales used in
//! professional media and telecommunications:
//!
//! - **TAI**: International Atomic Time (no leap seconds)
//! - **UTC**: Coordinated Universal Time (includes leap seconds)
//! - **GPS**: GPS Time (offset from TAI by 19 seconds)
//! - **PTP**: IEEE 1588 PTP epoch (same as TAI but epoch = 1970-01-01)
//! - **NTP**: NTP epoch (1900-01-01 UTC)

/// Supported time scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeScale {
    /// International Atomic Time.
    Tai,
    /// Coordinated Universal Time (with leap seconds).
    Utc,
    /// GPS Time (TAI - 19 seconds, epoch 1980-01-06).
    Gps,
    /// IEEE 1588 PTP Time (TAI-based, epoch 1970-01-01).
    Ptp,
    /// NTP Time (UTC-based, epoch 1900-01-01).
    Ntp,
}

impl TimeScale {
    /// Returns a human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Tai => "TAI (International Atomic Time)",
            Self::Utc => "UTC (Coordinated Universal Time)",
            Self::Gps => "GPS Time",
            Self::Ptp => "PTP (IEEE 1588)",
            Self::Ntp => "NTP Time",
        }
    }

    /// Returns the epoch description.
    #[must_use]
    pub const fn epoch_description(&self) -> &'static str {
        match self {
            Self::Tai => "1958-01-01T00:00:00 TAI",
            Self::Utc => "1970-01-01T00:00:00 UTC",
            Self::Gps => "1980-01-06T00:00:00 UTC",
            Self::Ptp => "1970-01-01T00:00:00 TAI",
            Self::Ntp => "1900-01-01T00:00:00 UTC",
        }
    }

    /// Returns whether this scale accounts for leap seconds.
    #[must_use]
    pub const fn has_leap_seconds(&self) -> bool {
        matches!(self, Self::Utc | Self::Ntp)
    }
}

/// Fixed offset between TAI and GPS time in seconds.
pub const TAI_GPS_OFFSET_S: i64 = 19;

/// Fixed offset between TAI epoch (1958) and Unix epoch (1970) in seconds.
/// TAI epoch is 12 years before Unix epoch: 4383 days * 86400 s/day.
pub const TAI_UNIX_EPOCH_OFFSET_S: i64 = 378_691_200;

/// Offset between NTP epoch (1900) and Unix epoch (1970) in seconds.
/// 70 years = 25567 days * 86400 s/day.
pub const NTP_UNIX_EPOCH_OFFSET_S: i64 = 2_208_988_800;

/// Offset between GPS epoch (1980-01-06) and Unix epoch (1970) in seconds.
pub const GPS_UNIX_EPOCH_OFFSET_S: i64 = 315_964_800;

/// A leap second table entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeapSecondEntry {
    /// Unix timestamp (UTC) at which the leap second was introduced.
    pub utc_epoch: i64,
    /// Cumulative TAI-UTC offset after this leap second (in seconds).
    pub tai_utc_offset: i32,
}

/// Provides the current leap second table.
///
/// This is a simplified table with key entries. A production system
/// would load this from IERS bulletins.
#[must_use]
pub fn leap_second_table() -> Vec<LeapSecondEntry> {
    vec![
        LeapSecondEntry {
            utc_epoch: 63_072_000,
            tai_utc_offset: 10,
        }, // 1972-01-01
        LeapSecondEntry {
            utc_epoch: 78_796_800,
            tai_utc_offset: 11,
        }, // 1972-07-01
        LeapSecondEntry {
            utc_epoch: 94_694_400,
            tai_utc_offset: 12,
        }, // 1973-01-01
        LeapSecondEntry {
            utc_epoch: 126_230_400,
            tai_utc_offset: 13,
        }, // 1974-01-01
        LeapSecondEntry {
            utc_epoch: 157_766_400,
            tai_utc_offset: 14,
        }, // 1975-01-01
        LeapSecondEntry {
            utc_epoch: 189_302_400,
            tai_utc_offset: 15,
        }, // 1976-01-01
        LeapSecondEntry {
            utc_epoch: 220_924_800,
            tai_utc_offset: 16,
        }, // 1977-01-01
        LeapSecondEntry {
            utc_epoch: 252_460_800,
            tai_utc_offset: 17,
        }, // 1978-01-01
        LeapSecondEntry {
            utc_epoch: 283_996_800,
            tai_utc_offset: 18,
        }, // 1979-01-01
        LeapSecondEntry {
            utc_epoch: 315_532_800,
            tai_utc_offset: 19,
        }, // 1980-01-01
        LeapSecondEntry {
            utc_epoch: 362_793_600,
            tai_utc_offset: 20,
        }, // 1981-07-01
        LeapSecondEntry {
            utc_epoch: 394_329_600,
            tai_utc_offset: 21,
        }, // 1982-07-01
        LeapSecondEntry {
            utc_epoch: 425_865_600,
            tai_utc_offset: 22,
        }, // 1983-07-01
        LeapSecondEntry {
            utc_epoch: 489_024_000,
            tai_utc_offset: 23,
        }, // 1985-07-01
        LeapSecondEntry {
            utc_epoch: 567_993_600,
            tai_utc_offset: 24,
        }, // 1988-01-01
        LeapSecondEntry {
            utc_epoch: 631_152_000,
            tai_utc_offset: 25,
        }, // 1990-01-01
        LeapSecondEntry {
            utc_epoch: 662_688_000,
            tai_utc_offset: 26,
        }, // 1991-01-01
        LeapSecondEntry {
            utc_epoch: 709_948_800,
            tai_utc_offset: 27,
        }, // 1992-07-01
        LeapSecondEntry {
            utc_epoch: 741_484_800,
            tai_utc_offset: 28,
        }, // 1993-07-01
        LeapSecondEntry {
            utc_epoch: 773_020_800,
            tai_utc_offset: 29,
        }, // 1994-07-01
        LeapSecondEntry {
            utc_epoch: 820_454_400,
            tai_utc_offset: 30,
        }, // 1996-01-01
        LeapSecondEntry {
            utc_epoch: 867_715_200,
            tai_utc_offset: 31,
        }, // 1997-07-01
        LeapSecondEntry {
            utc_epoch: 915_148_800,
            tai_utc_offset: 32,
        }, // 1999-01-01
        LeapSecondEntry {
            utc_epoch: 1_136_073_600,
            tai_utc_offset: 33,
        }, // 2006-01-01
        LeapSecondEntry {
            utc_epoch: 1_230_768_000,
            tai_utc_offset: 34,
        }, // 2009-01-01
        LeapSecondEntry {
            utc_epoch: 1_341_100_800,
            tai_utc_offset: 35,
        }, // 2012-07-01
        LeapSecondEntry {
            utc_epoch: 1_435_708_800,
            tai_utc_offset: 36,
        }, // 2015-07-01
        LeapSecondEntry {
            utc_epoch: 1_483_228_800,
            tai_utc_offset: 37,
        }, // 2017-01-01
    ]
}

/// Returns the TAI-UTC offset (leap seconds) for a given UTC Unix timestamp.
#[must_use]
pub fn tai_utc_offset_at(utc_unix: i64) -> i32 {
    let table = leap_second_table();
    let mut offset = 0_i32;
    for entry in &table {
        if utc_unix >= entry.utc_epoch {
            offset = entry.tai_utc_offset;
        } else {
            break;
        }
    }
    offset
}

/// Converts a UTC Unix timestamp to TAI Unix timestamp.
#[must_use]
pub fn utc_to_tai(utc_unix: i64) -> i64 {
    utc_unix + i64::from(tai_utc_offset_at(utc_unix))
}

/// Converts a TAI Unix timestamp to UTC Unix timestamp (approximate inverse).
#[must_use]
pub fn tai_to_utc(tai_unix: i64) -> i64 {
    // Approximate: use the offset at the estimated UTC time
    let approx_utc = tai_unix - 37; // Start with latest offset
    let offset = tai_utc_offset_at(approx_utc);
    tai_unix - i64::from(offset)
}

/// Converts a UTC Unix timestamp to GPS time (seconds since GPS epoch).
#[must_use]
pub fn utc_to_gps(utc_unix: i64) -> i64 {
    let tai = utc_to_tai(utc_unix);
    tai - TAI_GPS_OFFSET_S - GPS_UNIX_EPOCH_OFFSET_S
}

/// Converts GPS time to UTC Unix timestamp.
#[must_use]
pub fn gps_to_utc(gps_seconds: i64) -> i64 {
    let tai_unix = gps_seconds + GPS_UNIX_EPOCH_OFFSET_S + TAI_GPS_OFFSET_S;
    tai_to_utc(tai_unix)
}

/// Converts a UTC Unix timestamp to NTP timestamp (seconds since NTP epoch).
#[must_use]
pub fn utc_to_ntp(utc_unix: i64) -> i64 {
    utc_unix + NTP_UNIX_EPOCH_OFFSET_S
}

/// Converts an NTP timestamp to UTC Unix timestamp.
#[must_use]
pub fn ntp_to_utc(ntp_seconds: i64) -> i64 {
    ntp_seconds - NTP_UNIX_EPOCH_OFFSET_S
}

/// High-level converter between any two time scales.
#[derive(Debug, Clone)]
pub struct TimeScaleConverter {
    /// Cached leap second table length for diagnostics.
    leap_table_len: usize,
}

impl TimeScaleConverter {
    /// Creates a new converter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            leap_table_len: leap_second_table().len(),
        }
    }

    /// Converts a timestamp from one time scale to another.
    ///
    /// The `value` is in seconds relative to the source scale's epoch.
    #[must_use]
    pub fn convert(&self, value: i64, from: TimeScale, to: TimeScale) -> i64 {
        if from == to {
            return value;
        }

        // First convert to UTC Unix
        let utc_unix = match from {
            TimeScale::Utc => value,
            TimeScale::Tai => tai_to_utc(value),
            TimeScale::Gps => gps_to_utc(value),
            TimeScale::Ptp => tai_to_utc(value), // PTP uses TAI with Unix epoch
            TimeScale::Ntp => ntp_to_utc(value),
        };

        // Then convert from UTC Unix to target
        match to {
            TimeScale::Utc => utc_unix,
            TimeScale::Tai => utc_to_tai(utc_unix),
            TimeScale::Gps => utc_to_gps(utc_unix),
            TimeScale::Ptp => utc_to_tai(utc_unix), // PTP uses TAI with Unix epoch
            TimeScale::Ntp => utc_to_ntp(utc_unix),
        }
    }

    /// Returns the number of entries in the leap second table.
    #[must_use]
    pub fn leap_table_entries(&self) -> usize {
        self.leap_table_len
    }
}

impl Default for TimeScaleConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_scale_names() {
        assert!(TimeScale::Tai.name().contains("TAI"));
        assert!(TimeScale::Utc.name().contains("UTC"));
        assert!(TimeScale::Gps.name().contains("GPS"));
        assert!(TimeScale::Ptp.name().contains("PTP"));
        assert!(TimeScale::Ntp.name().contains("NTP"));
    }

    #[test]
    fn test_has_leap_seconds() {
        assert!(!TimeScale::Tai.has_leap_seconds());
        assert!(TimeScale::Utc.has_leap_seconds());
        assert!(!TimeScale::Gps.has_leap_seconds());
        assert!(!TimeScale::Ptp.has_leap_seconds());
        assert!(TimeScale::Ntp.has_leap_seconds());
    }

    #[test]
    fn test_tai_utc_offset_before_1972() {
        let offset = tai_utc_offset_at(0);
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_tai_utc_offset_2020() {
        // After 2017-01-01 the offset should be 37
        let offset = tai_utc_offset_at(1_600_000_000);
        assert_eq!(offset, 37);
    }

    #[test]
    fn test_utc_to_tai_roundtrip() {
        let utc = 1_600_000_000_i64;
        let tai = utc_to_tai(utc);
        let back = tai_to_utc(tai);
        assert_eq!(back, utc);
    }

    #[test]
    fn test_utc_to_ntp_roundtrip() {
        let utc = 1_600_000_000_i64;
        let ntp = utc_to_ntp(utc);
        let back = ntp_to_utc(ntp);
        assert_eq!(back, utc);
    }

    #[test]
    fn test_ntp_epoch_offset() {
        // NTP epoch is 70 years before Unix epoch
        let ntp = utc_to_ntp(0);
        assert_eq!(ntp, NTP_UNIX_EPOCH_OFFSET_S);
    }

    #[test]
    fn test_gps_conversion() {
        let utc = 1_600_000_000_i64;
        let gps = utc_to_gps(utc);
        let back = gps_to_utc(gps);
        assert_eq!(back, utc);
    }

    #[test]
    fn test_converter_identity() {
        let conv = TimeScaleConverter::new();
        assert_eq!(conv.convert(12345, TimeScale::Utc, TimeScale::Utc), 12345);
    }

    #[test]
    fn test_converter_utc_tai() {
        let conv = TimeScaleConverter::new();
        let utc = 1_600_000_000_i64;
        let tai = conv.convert(utc, TimeScale::Utc, TimeScale::Tai);
        assert_eq!(tai, utc + 37);
    }

    #[test]
    fn test_converter_utc_ntp() {
        let conv = TimeScaleConverter::new();
        let utc = 1_600_000_000_i64;
        let ntp = conv.convert(utc, TimeScale::Utc, TimeScale::Ntp);
        assert_eq!(ntp, utc + NTP_UNIX_EPOCH_OFFSET_S);
    }

    #[test]
    fn test_converter_leap_table_entries() {
        let conv = TimeScaleConverter::new();
        assert!(conv.leap_table_entries() >= 28);
    }

    #[test]
    fn test_epoch_descriptions() {
        assert!(TimeScale::Tai.epoch_description().contains("1958"));
        assert!(TimeScale::Gps.epoch_description().contains("1980"));
        assert!(TimeScale::Ntp.epoch_description().contains("1900"));
    }
}
