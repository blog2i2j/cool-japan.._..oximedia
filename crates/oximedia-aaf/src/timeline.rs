//! Timeline and edit rate support
//!
//! This module provides timeline-related types and utilities for AAF:
//! - `EditRate`: Rational time representation
//! - Position: Timeline position
//! - Duration calculations
//! - Time conversions

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Sub};

/// Edit rate (rational number representing frames per second)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditRate {
    /// Numerator
    pub numerator: i32,
    /// Denominator
    pub denominator: i32,
}

impl EditRate {
    /// Create a new edit rate
    #[must_use]
    pub fn new(numerator: i32, denominator: i32) -> Self {
        assert!(denominator != 0, "Edit rate denominator cannot be zero");
        Self {
            numerator,
            denominator,
        }
    }

    /// Create from floating point value
    #[must_use]
    pub fn from_float(value: f64) -> Self {
        // Convert to rational approximation
        let (num, den) = approximate_rational(value, 1000000);
        Self::new(num, den)
    }

    /// Convert to floating point
    #[must_use]
    pub fn to_float(&self) -> f64 {
        f64::from(self.numerator) / f64::from(self.denominator)
    }

    /// Get the reciprocal (duration per frame)
    #[must_use]
    pub fn reciprocal(&self) -> Self {
        Self::new(self.denominator, self.numerator)
    }

    /// Simplify the fraction
    #[must_use]
    pub fn simplify(&self) -> Self {
        let gcd = gcd(self.numerator.abs(), self.denominator.abs());
        Self::new(self.numerator / gcd, self.denominator / gcd)
    }

    /// Check if this is NTSC rate (contains 1001 in denominator)
    #[must_use]
    pub fn is_ntsc(&self) -> bool {
        self.denominator == 1001 || self.denominator % 1001 == 0
    }

    /// Common edit rates
    pub const FILM_24: EditRate = EditRate {
        numerator: 24,
        denominator: 1,
    };

    pub const FILM_23_976: EditRate = EditRate {
        numerator: 24000,
        denominator: 1001,
    };

    pub const PAL_25: EditRate = EditRate {
        numerator: 25,
        denominator: 1,
    };

    pub const NTSC_29_97: EditRate = EditRate {
        numerator: 30000,
        denominator: 1001,
    };

    pub const NTSC_30: EditRate = EditRate {
        numerator: 30,
        denominator: 1,
    };

    pub const PAL_50: EditRate = EditRate {
        numerator: 50,
        denominator: 1,
    };

    pub const NTSC_59_94: EditRate = EditRate {
        numerator: 60000,
        denominator: 1001,
    };

    pub const NTSC_60: EditRate = EditRate {
        numerator: 60,
        denominator: 1,
    };

    pub const AUDIO_48K: EditRate = EditRate {
        numerator: 48000,
        denominator: 1,
    };

    pub const AUDIO_96K: EditRate = EditRate {
        numerator: 96000,
        denominator: 1,
    };
}

impl fmt::Display for EditRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == 1 {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

impl Default for EditRate {
    fn default() -> Self {
        Self::PAL_25
    }
}

/// Timeline position (in edit units)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Position(pub i64);

impl Position {
    /// Create a new position
    #[must_use]
    pub fn new(value: i64) -> Self {
        Self(value)
    }

    /// Zero position
    #[must_use]
    pub fn zero() -> Self {
        Self(0)
    }

    /// Convert to seconds given an edit rate
    #[must_use]
    pub fn to_seconds(&self, edit_rate: EditRate) -> f64 {
        (self.0 as f64 * f64::from(edit_rate.denominator)) / f64::from(edit_rate.numerator)
    }

    /// Create from seconds given an edit rate
    #[must_use]
    pub fn from_seconds(seconds: f64, edit_rate: EditRate) -> Self {
        let value =
            (seconds * f64::from(edit_rate.numerator) / f64::from(edit_rate.denominator)).round();
        Self(value as i64)
    }

    /// Convert to frames given an edit rate (assuming edit rate is frames per second)
    #[must_use]
    pub fn to_frames(&self, edit_rate: EditRate) -> i64 {
        // If edit rate is already in the correct units, just return the value
        // Otherwise, convert based on the edit rate
        if edit_rate.denominator == 1 {
            self.0
        } else {
            (self.0 * i64::from(edit_rate.numerator)) / i64::from(edit_rate.denominator)
        }
    }

    /// Create from frames given an edit rate
    #[must_use]
    pub fn from_frames(frames: i64, edit_rate: EditRate) -> Self {
        if edit_rate.denominator == 1 {
            Self(frames)
        } else {
            Self((frames * i64::from(edit_rate.denominator)) / i64::from(edit_rate.numerator))
        }
    }

    /// Convert between different edit rates
    #[must_use]
    pub fn convert(&self, from_rate: EditRate, to_rate: EditRate) -> Self {
        // Convert position from from_rate to to_rate
        // time = position / from_rate
        // new_position = time * to_rate
        let value = (self.0 * i64::from(to_rate.numerator) * i64::from(from_rate.denominator))
            / (i64::from(from_rate.numerator) * i64::from(to_rate.denominator));
        Self(value)
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::zero()
    }
}

impl Add for Position {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl Sub for Position {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl Add<i64> for Position {
    type Output = Self;

    fn add(self, other: i64) -> Self {
        Self(self.0 + other)
    }
}

impl Sub<i64> for Position {
    type Output = Self;

    fn sub(self, other: i64) -> Self {
        Self(self.0 - other)
    }
}

/// Duration (length in edit units)
pub type Duration = i64;

/// Timeline range
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineRange {
    /// Start position
    pub start: Position,
    /// Duration
    pub duration: Duration,
}

impl TimelineRange {
    /// Create a new timeline range
    #[must_use]
    pub fn new(start: Position, duration: Duration) -> Self {
        Self { start, duration }
    }

    /// Get the end position (exclusive)
    #[must_use]
    pub fn end(&self) -> Position {
        Position(self.start.0 + self.duration)
    }

    /// Check if this range contains a position
    #[must_use]
    pub fn contains(&self, position: Position) -> bool {
        position >= self.start && position < self.end()
    }

    /// Check if this range overlaps with another
    #[must_use]
    pub fn overlaps(&self, other: &TimelineRange) -> bool {
        self.start < other.end() && other.start < self.end()
    }

    /// Get the intersection with another range
    #[must_use]
    pub fn intersection(&self, other: &TimelineRange) -> Option<TimelineRange> {
        if !self.overlaps(other) {
            return None;
        }

        let start = self.start.max(other.start);
        let end = self.end().min(other.end());
        let duration = end.0 - start.0;

        Some(TimelineRange { start, duration })
    }

    /// Offset this range by a position
    #[must_use]
    pub fn offset(&self, offset: Position) -> TimelineRange {
        TimelineRange {
            start: self.start + offset,
            duration: self.duration,
        }
    }
}

/// Greatest common divisor
fn gcd(a: i32, b: i32) -> i32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Approximate a floating point value as a rational number
fn approximate_rational(value: f64, max_denominator: i32) -> (i32, i32) {
    if value.is_nan() || value.is_infinite() {
        return (0, 1);
    }

    let sign = if value < 0.0 { -1 } else { 1 };
    let value = value.abs();

    // Continued fraction approximation
    let mut h1 = 1;
    let mut h2 = 0;
    let mut k1 = 0;
    let mut k2 = 1;

    let mut b = value;
    loop {
        let a = b.floor() as i32;
        let mut aux = h1;
        h1 = a * h1 + h2;
        h2 = aux;
        aux = k1;
        k1 = a * k1 + k2;
        k2 = aux;

        b = 1.0 / (b - f64::from(a));

        if k1 > max_denominator || (f64::from(h1) / f64::from(k1) - value).abs() < 1e-8 {
            break;
        }
    }

    (sign * h1, k1)
}

/// Timeline utilities
pub struct TimelineUtils;

impl TimelineUtils {
    /// Calculate duration in edit units from seconds
    #[must_use]
    pub fn duration_from_seconds(seconds: f64, edit_rate: EditRate) -> Duration {
        ((seconds * f64::from(edit_rate.numerator)) / f64::from(edit_rate.denominator)).round()
            as i64
    }

    /// Calculate duration in seconds from edit units
    #[must_use]
    pub fn duration_to_seconds(duration: Duration, edit_rate: EditRate) -> f64 {
        (duration as f64 * f64::from(edit_rate.denominator)) / f64::from(edit_rate.numerator)
    }

    /// Calculate frames from edit units
    #[must_use]
    pub fn edit_units_to_frames(units: i64, edit_rate: EditRate) -> i64 {
        (units * i64::from(edit_rate.numerator)) / i64::from(edit_rate.denominator)
    }

    /// Calculate edit units from frames
    #[must_use]
    pub fn frames_to_edit_units(frames: i64, edit_rate: EditRate) -> i64 {
        (frames * i64::from(edit_rate.denominator)) / i64::from(edit_rate.numerator)
    }

    /// Convert timecode to position
    #[must_use]
    pub fn timecode_to_position(
        hours: u8,
        minutes: u8,
        seconds: u8,
        frames: u8,
        edit_rate: EditRate,
    ) -> Position {
        let fps = edit_rate.to_float().round() as i64;
        let total_frames = i64::from(hours) * 3600 * fps
            + i64::from(minutes) * 60 * fps
            + i64::from(seconds) * fps
            + i64::from(frames);

        Position::from_frames(total_frames, edit_rate)
    }

    /// Convert position to timecode components
    #[must_use]
    pub fn position_to_timecode(position: Position, edit_rate: EditRate) -> (u8, u8, u8, u8) {
        let fps = edit_rate.to_float().round() as i64;
        let total_frames = position.to_frames(edit_rate);

        let hours = (total_frames / (fps * 3600)) as u8;
        let remaining = total_frames % (fps * 3600);
        let minutes = (remaining / (fps * 60)) as u8;
        let remaining = remaining % (fps * 60);
        let seconds = (remaining / fps) as u8;
        let frames = (remaining % fps) as u8;

        (hours, minutes, seconds, frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_rate_creation() {
        let rate = EditRate::new(25, 1);
        assert_eq!(rate.numerator, 25);
        assert_eq!(rate.denominator, 1);
        assert_eq!(rate.to_float(), 25.0);
    }

    #[test]
    fn test_edit_rate_simplify() {
        let rate = EditRate::new(50, 2);
        let simplified = rate.simplify();
        assert_eq!(simplified.numerator, 25);
        assert_eq!(simplified.denominator, 1);
    }

    #[test]
    fn test_edit_rate_from_float() {
        let rate = EditRate::from_float(29.97);
        // Should approximate to 30000/1001 or similar
        let value = rate.to_float();
        assert!((value - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_position_creation() {
        let pos = Position::new(100);
        assert_eq!(pos.0, 100);
    }

    #[test]
    fn test_position_arithmetic() {
        let pos1 = Position::new(100);
        let pos2 = Position::new(50);
        assert_eq!((pos1 + pos2).0, 150);
        assert_eq!((pos1 - pos2).0, 50);
    }

    #[test]
    fn test_position_to_seconds() {
        let pos = Position::new(25);
        let rate = EditRate::new(25, 1);
        assert_eq!(pos.to_seconds(rate), 1.0);
    }

    #[test]
    fn test_position_from_seconds() {
        let rate = EditRate::new(25, 1);
        let pos = Position::from_seconds(2.0, rate);
        assert_eq!(pos.0, 50);
    }

    #[test]
    fn test_position_convert() {
        let pos = Position::new(30);
        let from_rate = EditRate::new(30, 1);
        let to_rate = EditRate::new(25, 1);
        let converted = pos.convert(from_rate, to_rate);
        assert_eq!(converted.0, 25);
    }

    #[test]
    fn test_timeline_range() {
        let range = TimelineRange::new(Position::new(10), 50);
        assert_eq!(range.start.0, 10);
        assert_eq!(range.duration, 50);
        assert_eq!(range.end().0, 60);
    }

    #[test]
    fn test_timeline_range_contains() {
        let range = TimelineRange::new(Position::new(10), 50);
        assert!(range.contains(Position::new(30)));
        assert!(!range.contains(Position::new(5)));
        assert!(!range.contains(Position::new(60)));
    }

    #[test]
    fn test_timeline_range_overlaps() {
        let range1 = TimelineRange::new(Position::new(10), 50);
        let range2 = TimelineRange::new(Position::new(40), 30);
        let range3 = TimelineRange::new(Position::new(100), 20);

        assert!(range1.overlaps(&range2));
        assert!(!range1.overlaps(&range3));
    }

    #[test]
    fn test_timeline_range_intersection() {
        let range1 = TimelineRange::new(Position::new(10), 50);
        let range2 = TimelineRange::new(Position::new(40), 30);

        let intersection = range1
            .intersection(&range2)
            .expect("intersection should be valid");
        assert_eq!(intersection.start.0, 40);
        assert_eq!(intersection.duration, 20);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(48, 18), 6);
        assert_eq!(gcd(100, 50), 50);
        assert_eq!(gcd(17, 13), 1);
    }

    #[test]
    fn test_approximate_rational() {
        let (num, den) = approximate_rational(29.97, 100000);
        let value = num as f64 / den as f64;
        assert!((value - 29.97).abs() < 0.01);
    }

    #[test]
    fn test_timeline_utils_timecode() {
        let rate = EditRate::new(25, 1);
        let pos = TimelineUtils::timecode_to_position(1, 0, 0, 0, rate);
        let (h, m, s, f) = TimelineUtils::position_to_timecode(pos, rate);
        assert_eq!(h, 1);
        assert_eq!(m, 0);
        assert_eq!(s, 0);
        assert_eq!(f, 0);
    }
}
