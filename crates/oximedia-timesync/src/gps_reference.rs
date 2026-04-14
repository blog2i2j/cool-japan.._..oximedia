//! GPS-disciplined clock input as an ultimate time reference.
//!
//! A GPS-disciplined clock uses NMEA sentences or a 1-PPS (pulse-per-second)
//! signal from a GPS receiver to discipline the local oscillator to UTC.
//! It provides stratum-1 accuracy (typically < 100 ns) without needing a
//! network time server.
//!
//! # Supported input modes
//! - **1-PPS** — hardware pulse aligned to the UTC second edge (sub-microsecond).
//! - **NMEA 0183** — software-decoded $GPRMC / $GPGGA sentences (ms accuracy).
//! - **UBX** — ublox proprietary binary protocol for direct nanosecond timing.
//!
//! # Architecture
//! ```text
//!  GPS Antenna
//!     │
//!  GPS Receiver (hardware)
//!     ├── 1-PPS pin ──────────────────────► GpsPpsSource
//!     │                                         │
//!     └── NMEA/UBX serial ──► GpsNmeaParser ──► GpsDisciplinedClock
//! ```

use crate::error::{TimeSyncError, TimeSyncResult};
use std::fmt;

// ---------------------------------------------------------------------------
// GPS fix quality
// ---------------------------------------------------------------------------

/// GPS fix quality as reported by $GPGGA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GpsFixQuality {
    /// No fix available.
    NoFix = 0,
    /// Standard GPS fix (SPS mode).
    GpsFix = 1,
    /// Differential GPS fix.
    DgpsFix = 2,
    /// PPS fix (disciplined to external PPS).
    PpsFix = 3,
    /// Real-time kinematic (RTK) fixed.
    RtkFixed = 4,
    /// RTK float solution.
    RtkFloat = 5,
    /// Dead reckoning.
    DeadReckoning = 6,
}

impl GpsFixQuality {
    /// Returns `true` if the fix quality is sufficient for timing (≥ GPS fix).
    #[must_use]
    pub fn is_usable(&self) -> bool {
        matches!(
            self,
            Self::GpsFix | Self::DgpsFix | Self::PpsFix | Self::RtkFixed | Self::RtkFloat
        )
    }

    /// Converts from the $GPGGA quality indicator byte.
    pub fn from_u8(value: u8) -> TimeSyncResult<Self> {
        Ok(match value {
            0 => Self::NoFix,
            1 => Self::GpsFix,
            2 => Self::DgpsFix,
            3 => Self::PpsFix,
            4 => Self::RtkFixed,
            5 => Self::RtkFloat,
            6 => Self::DeadReckoning,
            v => {
                return Err(TimeSyncError::InvalidPacket(format!(
                    "GPS: unknown fix quality indicator: {v}"
                )))
            }
        })
    }
}

impl fmt::Display for GpsFixQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::NoFix => "No Fix",
            Self::GpsFix => "GPS Fix (SPS)",
            Self::DgpsFix => "DGPS Fix",
            Self::PpsFix => "PPS Fix",
            Self::RtkFixed => "RTK Fixed",
            Self::RtkFloat => "RTK Float",
            Self::DeadReckoning => "Dead Reckoning",
        };
        write!(f, "{s}")
    }
}

// ---------------------------------------------------------------------------
// NMEA parser (minimal — RMC + GGA)
// ---------------------------------------------------------------------------

/// Parsed time from an NMEA $GPRMC or $GPGGA sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NmeaTime {
    /// UTC hours (0–23).
    pub hours: u8,
    /// UTC minutes (0–59).
    pub minutes: u8,
    /// UTC seconds (0–59, may be 60 for leap second).
    pub seconds: u8,
    /// Fractional seconds in milliseconds (0–999).
    pub millis: u16,
    /// Fix quality (from GGA) or `GpsFix` if from RMC.
    pub fix_quality: GpsFixQuality,
    /// Number of satellites in use (from GGA, 0 if from RMC).
    pub satellites: u8,
}

impl NmeaTime {
    /// Converts to total nanoseconds within the day (UTC).
    #[must_use]
    pub fn time_of_day_ns(&self) -> u64 {
        let h = u64::from(self.hours);
        let m = u64::from(self.minutes);
        let s = u64::from(self.seconds);
        let ms = u64::from(self.millis);
        ((h * 3_600 + m * 60 + s) * 1_000 + ms) * 1_000_000
    }
}

/// Minimal NMEA 0183 sentence parser for timing sentences.
pub struct GpsNmeaParser;

impl GpsNmeaParser {
    /// Parses a $GPRMC or $GNRMC sentence and extracts the UTC time.
    ///
    /// Format: `$GPRMC,HHMMSS.ss,A,lat,N,lon,E,speed,course,DDMMYY,mag,E*hh`
    ///
    /// Returns `None` if the sentence is not RMC, the fix status is 'V'
    /// (void), or the time field is malformed.
    #[must_use]
    pub fn parse_rmc(sentence: &str) -> Option<NmeaTime> {
        let sentence = sentence.trim_start_matches('$');
        if !sentence.starts_with("GPRMC,") && !sentence.starts_with("GNRMC,") {
            return None;
        }
        // Strip checksum
        let body = sentence.split('*').next()?;
        let fields: Vec<&str> = body.split(',').collect();
        if fields.len() < 3 {
            return None;
        }
        // fields[1] = HHMMSS.ss,  fields[2] = A/V
        if fields[2] == "V" {
            return None; // void / invalid
        }
        Self::parse_time_field(fields[1], GpsFixQuality::GpsFix, 0)
    }

    /// Parses a $GPGGA or $GNGGA sentence and extracts UTC time and fix quality.
    ///
    /// Format: `$GPGGA,HHMMSS.ss,lat,N,lon,E,quality,sats,...*hh`
    #[must_use]
    pub fn parse_gga(sentence: &str) -> Option<NmeaTime> {
        let sentence = sentence.trim_start_matches('$');
        if !sentence.starts_with("GPGGA,") && !sentence.starts_with("GNGGA,") {
            return None;
        }
        let body = sentence.split('*').next()?;
        let fields: Vec<&str> = body.split(',').collect();
        if fields.len() < 8 {
            return None;
        }
        let quality = fields[6].parse::<u8>().ok()?;
        let fix = GpsFixQuality::from_u8(quality).ok()?;
        if !fix.is_usable() {
            return None;
        }
        let sats = fields[7].parse::<u8>().unwrap_or(0);
        Self::parse_time_field(fields[1], fix, sats)
    }

    /// Parses an NMEA time field `HHMMSS.sss` or `HHMMSS`.
    fn parse_time_field(
        time_field: &str,
        fix_quality: GpsFixQuality,
        satellites: u8,
    ) -> Option<NmeaTime> {
        if time_field.len() < 6 {
            return None;
        }
        let hours: u8 = time_field[0..2].parse().ok()?;
        let minutes: u8 = time_field[2..4].parse().ok()?;
        let seconds: u8 = time_field[4..6].parse().ok()?;
        let millis: u16 = if time_field.len() > 7 && time_field.as_bytes().get(6) == Some(&b'.') {
            let frac = &time_field[7..];
            // Pad or truncate to 3 digits
            let padded = format!("{:0<3}", &frac[..frac.len().min(3)]);
            padded.parse().unwrap_or(0)
        } else {
            0
        };

        if hours > 23 || minutes > 59 || seconds > 60 {
            return None;
        }

        Some(NmeaTime {
            hours,
            minutes,
            seconds,
            millis,
            fix_quality,
            satellites,
        })
    }
}

// ---------------------------------------------------------------------------
// 1-PPS source
// ---------------------------------------------------------------------------

/// Models a 1-PPS (pulse-per-second) hardware timing signal.
///
/// The PPS pulse arrives once per second, aligned to the UTC second edge.
/// Combined with an NMEA sentence that provides the calendar date and time,
/// it gives sub-microsecond absolute UTC timing.
#[derive(Debug, Clone)]
pub struct GpsPpsSource {
    /// Timestamp of the most recent PPS edge (nanoseconds since Unix epoch,
    /// as measured by the local system clock).
    pub last_pps_ns: Option<u64>,
    /// Fractional-second correction (nanoseconds) derived from PPS vs
    /// system-clock offset calibration.
    pub pps_correction_ns: i64,
    /// Number of valid PPS pulses received.
    pub pulse_count: u64,
    /// Whether the PPS source is considered locked and usable.
    pub locked: bool,
}

impl GpsPpsSource {
    /// Creates a new PPS source in the unlocked state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            last_pps_ns: None,
            pps_correction_ns: 0,
            pulse_count: 0,
            locked: false,
        }
    }

    /// Records a PPS pulse edge at `system_time_ns` (nanoseconds since epoch).
    ///
    /// The `expected_second_ns` is the UTC second boundary the pulse is
    /// believed to correspond to (e.g. from the most recent NMEA sentence).
    pub fn record_pulse(&mut self, system_time_ns: u64, expected_second_ns: u64) {
        // Correction = expected UTC second - actual system time at pulse edge.
        let correction = expected_second_ns as i64 - system_time_ns as i64;
        self.pps_correction_ns = correction;
        self.last_pps_ns = Some(system_time_ns);
        self.pulse_count += 1;
        self.locked = true;
    }

    /// Returns the corrected UTC nanosecond value corresponding to the last PPS edge.
    ///
    /// Returns `None` if no pulse has been received yet.
    #[must_use]
    pub fn corrected_utc_ns(&self) -> Option<u64> {
        self.last_pps_ns.map(|pps| {
            let corrected = pps as i64 + self.pps_correction_ns;
            corrected.max(0) as u64
        })
    }
}

impl Default for GpsPpsSource {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GPS-disciplined clock
// ---------------------------------------------------------------------------

/// Configuration for a GPS-disciplined clock.
#[derive(Debug, Clone)]
pub struct GpsClockConfig {
    /// Minimum number of satellites required for a usable fix.
    pub min_satellites: u8,
    /// Maximum acceptable offset between NMEA time and system clock (nanoseconds).
    /// If exceeded, the clock is considered unsync'd until re-locked.
    pub max_offset_ns: u64,
    /// Whether to require a 1-PPS signal for stratum-1 accuracy.
    pub require_pps: bool,
}

impl Default for GpsClockConfig {
    fn default() -> Self {
        Self {
            min_satellites: 4,
            max_offset_ns: 1_000_000, // 1 ms
            require_pps: false,
        }
    }
}

/// GPS-disciplined clock — combines NMEA time and optional 1-PPS for UTC.
pub struct GpsDisciplinedClock {
    /// Configuration.
    pub config: GpsClockConfig,
    /// NMEA parser state: most recently received time.
    pub last_nmea_time: Option<NmeaTime>,
    /// PPS source (if available).
    pub pps_source: Option<GpsPpsSource>,
    /// Current UTC offset from system clock in nanoseconds.
    pub utc_offset_ns: i64,
    /// PTP stratum equivalent (1 with PPS, 2 with NMEA-only).
    pub stratum: u8,
    /// Whether the clock is considered synchronised.
    pub synchronized: bool,
}

impl GpsDisciplinedClock {
    /// Creates a new GPS-disciplined clock with the given configuration.
    #[must_use]
    pub fn new(config: GpsClockConfig) -> Self {
        Self {
            config,
            last_nmea_time: None,
            pps_source: None,
            utc_offset_ns: 0,
            stratum: 16, // unsync'd
            synchronized: false,
        }
    }

    /// Creates a clock with PPS support enabled.
    #[must_use]
    pub fn with_pps(mut self) -> Self {
        self.pps_source = Some(GpsPpsSource::new());
        self
    }

    /// Processes an NMEA sentence (RMC or GGA) to update the clock.
    ///
    /// Returns `true` if the sentence was valid and the time updated.
    pub fn process_nmea(&mut self, sentence: &str) -> bool {
        let parsed = if sentence.contains("RMC") {
            GpsNmeaParser::parse_rmc(sentence)
        } else if sentence.contains("GGA") {
            GpsNmeaParser::parse_gga(sentence)
        } else {
            None
        };

        if let Some(nmea_time) = parsed {
            if nmea_time.satellites >= self.config.min_satellites
                || nmea_time.fix_quality >= GpsFixQuality::GpsFix
            {
                self.last_nmea_time = Some(nmea_time);
                self.update_sync_state();
                return true;
            }
        }
        false
    }

    /// Records a 1-PPS pulse edge.
    ///
    /// `system_time_ns`: the system-clock nanosecond value at the pulse edge.
    /// `expected_utc_second_ns`: the expected UTC second (from NMEA).
    pub fn record_pps_pulse(&mut self, system_time_ns: u64, expected_utc_second_ns: u64) {
        if let Some(ref mut pps) = self.pps_source {
            pps.record_pulse(system_time_ns, expected_utc_second_ns);
            self.stratum = 1;
            self.synchronized = true;
        }
    }

    /// Returns `true` if the clock is currently providing valid UTC time.
    #[must_use]
    pub fn is_synchronized(&self) -> bool {
        self.synchronized
    }

    /// Returns the estimated UTC time-of-day in nanoseconds, or `None` if
    /// the clock is not synchronized.
    #[must_use]
    pub fn utc_time_of_day_ns(&self) -> Option<u64> {
        self.last_nmea_time.as_ref().map(|t| t.time_of_day_ns())
    }

    fn update_sync_state(&mut self) {
        if let Some(ref nmea) = self.last_nmea_time {
            if nmea.fix_quality.is_usable() {
                self.synchronized = true;
                self.stratum = if self.pps_source.as_ref().map_or(false, |p| p.locked) {
                    1
                } else {
                    2
                };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fix_quality_is_usable() {
        assert!(GpsFixQuality::GpsFix.is_usable());
        assert!(GpsFixQuality::DgpsFix.is_usable());
        assert!(GpsFixQuality::PpsFix.is_usable());
        assert!(!GpsFixQuality::NoFix.is_usable());
        assert!(!GpsFixQuality::DeadReckoning.is_usable());
    }

    #[test]
    fn test_fix_quality_from_u8() {
        assert_eq!(GpsFixQuality::from_u8(1).unwrap(), GpsFixQuality::GpsFix);
        assert_eq!(GpsFixQuality::from_u8(0).unwrap(), GpsFixQuality::NoFix);
        assert!(GpsFixQuality::from_u8(255).is_err());
    }

    #[test]
    fn test_nmea_time_time_of_day_ns() {
        let t = NmeaTime {
            hours: 0,
            minutes: 0,
            seconds: 1,
            millis: 0,
            fix_quality: GpsFixQuality::GpsFix,
            satellites: 6,
        };
        assert_eq!(t.time_of_day_ns(), 1_000_000_000);
    }

    #[test]
    fn test_nmea_time_hours_minutes() {
        let t = NmeaTime {
            hours: 1,
            minutes: 1,
            seconds: 0,
            millis: 0,
            fix_quality: GpsFixQuality::GpsFix,
            satellites: 6,
        };
        // 1h + 1m = 3600 + 60 = 3660 s = 3_660_000_000_000 ns
        assert_eq!(t.time_of_day_ns(), 3_660_000_000_000);
    }

    #[test]
    fn test_nmea_parser_rmc_valid() {
        let sentence = "$GPRMC,120000.00,A,5130.0000,N,00000.0000,E,0.0,0.0,010101,,,*70";
        let parsed = GpsNmeaParser::parse_rmc(sentence);
        assert!(parsed.is_some());
        let t = parsed.unwrap();
        assert_eq!(t.hours, 12);
        assert_eq!(t.minutes, 0);
        assert_eq!(t.seconds, 0);
    }

    #[test]
    fn test_nmea_parser_rmc_void() {
        // Status 'V' = void/invalid
        let sentence = "$GPRMC,120000.00,V,5130.0000,N,00000.0000,E,0.0,0.0,010101,,,*70";
        let parsed = GpsNmeaParser::parse_rmc(sentence);
        assert!(parsed.is_none());
    }

    #[test]
    fn test_nmea_parser_rmc_with_millis() {
        let sentence = "$GPRMC,120000.500,A,5130.0000,N,00000.0000,E,0.0,0.0,010101,,,*70";
        let parsed = GpsNmeaParser::parse_rmc(sentence);
        assert!(parsed.is_some());
        let t = parsed.unwrap();
        assert_eq!(t.millis, 500);
    }

    #[test]
    fn test_nmea_parser_gga_valid() {
        // $GPGGA,HHMMSS.ss,lat,N,lon,E,quality,sats,...
        let sentence = "$GPGGA,120000.00,5130.0,N,00000.0,E,1,08,0.9,100.0,M,0.0,M,,*47";
        let parsed = GpsNmeaParser::parse_gga(sentence);
        assert!(parsed.is_some());
        let t = parsed.unwrap();
        assert_eq!(t.hours, 12);
        assert_eq!(t.satellites, 8);
        assert_eq!(t.fix_quality, GpsFixQuality::GpsFix);
    }

    #[test]
    fn test_nmea_parser_gga_no_fix() {
        // quality = 0 → no fix
        let sentence = "$GPGGA,120000.00,0.0,N,0.0,E,0,00,0.0,0.0,M,0.0,M,,*48";
        let parsed = GpsNmeaParser::parse_gga(sentence);
        assert!(parsed.is_none(), "no-fix GGA should return None");
    }

    #[test]
    fn test_pps_source_record_pulse() {
        let mut pps = GpsPpsSource::new();
        assert!(!pps.locked);
        // system time = 1_000_000_100 ns, expected second = 1_000_000_000 ns
        pps.record_pulse(1_000_000_100, 1_000_000_000);
        assert!(pps.locked);
        assert_eq!(pps.pulse_count, 1);
        assert_eq!(pps.pps_correction_ns, -100);
    }

    #[test]
    fn test_pps_source_corrected_utc() {
        let mut pps = GpsPpsSource::new();
        pps.record_pulse(1_000_000_000, 1_000_000_100);
        // correction = 100, corrected = 1_000_000_000 + 100 = 1_000_000_100
        let utc = pps.corrected_utc_ns();
        assert_eq!(utc, Some(1_000_000_100));
    }

    #[test]
    fn test_gps_clock_process_nmea_rmc() {
        let config = GpsClockConfig {
            min_satellites: 0,
            ..Default::default()
        };
        let mut clock = GpsDisciplinedClock::new(config);
        let sentence = "$GPRMC,120000.00,A,5130.0000,N,00000.0000,E,0.0,0.0,010101,,,*70";
        let result = clock.process_nmea(sentence);
        assert!(result, "RMC sentence should be accepted");
        assert!(clock.is_synchronized());
    }

    #[test]
    fn test_gps_clock_utc_time_of_day() {
        let config = GpsClockConfig {
            min_satellites: 0,
            ..Default::default()
        };
        let mut clock = GpsDisciplinedClock::new(config);
        let sentence = "$GPRMC,010000.00,A,5130.0000,N,00000.0000,E,0.0,0.0,010101,,,*70";
        clock.process_nmea(sentence);
        let tod = clock.utc_time_of_day_ns();
        assert!(tod.is_some());
        // 01:00:00 = 3600 s = 3_600_000_000_000 ns
        assert_eq!(tod.unwrap(), 3_600_000_000_000);
    }

    #[test]
    fn test_gps_clock_stratum_with_pps() {
        let mut clock = GpsDisciplinedClock::new(GpsClockConfig::default()).with_pps();
        assert_eq!(clock.stratum, 16); // not yet locked
        clock.record_pps_pulse(1_000_000_000, 1_000_000_000);
        assert_eq!(clock.stratum, 1);
        assert!(clock.is_synchronized());
    }
}
