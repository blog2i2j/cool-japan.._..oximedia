//! Timestamp types for multimedia timing.
//!
//! This module provides a [`Timestamp`] type that represents a point in time
//! within a media stream, including presentation time, decode time, and duration.

use super::Rational;

/// Timestamp with timebase context.
///
/// Represents a point in time within a media stream. Times are stored
/// as integer values that must be interpreted relative to the timebase.
///
/// # Examples
///
/// ```
/// use oximedia_core::types::{Timestamp, Rational};
///
/// // Create a timestamp at 1.5 seconds with a 1/1000 timebase
/// let ts = Timestamp::new(1500, Rational::new(1, 1000));
/// assert!((ts.to_seconds() - 1.5).abs() < 0.001);
/// ```
#[derive(Clone, Copy, Debug, Eq, serde::Serialize, serde::Deserialize)]
pub struct Timestamp {
    /// Presentation timestamp - when this frame should be displayed.
    pub pts: i64,
    /// Decode timestamp - when this frame should be decoded.
    /// May differ from PTS for codecs with B-frames.
    pub dts: Option<i64>,
    /// The timebase used to interpret pts/dts values.
    pub timebase: Rational,
    /// Duration of this frame in timebase units.
    pub duration: Option<i64>,
}

impl Timestamp {
    /// Creates a new timestamp with the given PTS and timebase.
    ///
    /// # Arguments
    ///
    /// * `pts` - Presentation timestamp value
    /// * `timebase` - The timebase for interpreting the timestamp
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::{Timestamp, Rational};
    ///
    /// let ts = Timestamp::new(90000, Rational::new(1, 90000));
    /// assert!((ts.to_seconds() - 1.0).abs() < f64::EPSILON);
    /// ```
    #[must_use]
    pub fn new(pts: i64, timebase: Rational) -> Self {
        Self {
            pts,
            dts: None,
            timebase,
            duration: None,
        }
    }

    /// Creates a new timestamp with all fields specified.
    ///
    /// # Arguments
    ///
    /// * `pts` - Presentation timestamp value
    /// * `dts` - Optional decode timestamp value
    /// * `timebase` - The timebase for interpreting timestamps
    /// * `duration` - Optional duration in timebase units
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::{Timestamp, Rational};
    ///
    /// let ts = Timestamp::with_dts(
    ///     3000,
    ///     Some(2000),
    ///     Rational::new(1, 1000),
    ///     Some(33),
    /// );
    /// assert_eq!(ts.dts, Some(2000));
    /// assert_eq!(ts.duration, Some(33));
    /// ```
    #[must_use]
    pub fn with_dts(pts: i64, dts: Option<i64>, timebase: Rational, duration: Option<i64>) -> Self {
        Self {
            pts,
            dts,
            timebase,
            duration,
        }
    }

    /// Converts the presentation timestamp to seconds.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::{Timestamp, Rational};
    ///
    /// let ts = Timestamp::new(48000, Rational::new(1, 48000));
    /// assert!((ts.to_seconds() - 1.0).abs() < f64::EPSILON);
    /// ```
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_seconds(&self) -> f64 {
        self.pts as f64 * self.timebase.to_f64()
    }

    /// Converts the decode timestamp to seconds, if present.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::{Timestamp, Rational};
    ///
    /// let ts = Timestamp::with_dts(3000, Some(2000), Rational::new(1, 1000), None);
    /// assert!((ts.dts_to_seconds()? - 2.0).abs() < 0.001);
    /// ```
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn dts_to_seconds(&self) -> Option<f64> {
        self.dts.map(|dts| dts as f64 * self.timebase.to_f64())
    }

    /// Returns the duration in seconds, if present.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::{Timestamp, Rational};
    ///
    /// let ts = Timestamp::with_dts(0, None, Rational::new(1, 1000), Some(33));
    /// assert!((ts.duration_seconds()? - 0.033).abs() < 0.001);
    /// ```
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> Option<f64> {
        self.duration.map(|dur| dur as f64 * self.timebase.to_f64())
    }

    /// Rescales the timestamp to a new timebase.
    ///
    /// Converts all timestamp values (pts, dts, duration) from the current
    /// timebase to the target timebase.
    ///
    /// # Arguments
    ///
    /// * `target_timebase` - The new timebase to convert to
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::{Timestamp, Rational};
    ///
    /// // Convert from milliseconds to microseconds
    /// let ts = Timestamp::new(1000, Rational::new(1, 1000));
    /// let rescaled = ts.rescale(Rational::new(1, 1_000_000));
    /// assert_eq!(rescaled.pts, 1_000_000);
    /// ```
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn rescale(&self, target_timebase: Rational) -> Self {
        // new_pts = old_pts * (old_timebase / new_timebase)
        // = old_pts * old_timebase.num * new_timebase.den / (old_timebase.den * new_timebase.num)
        let scale_num = self.timebase.num * target_timebase.den;
        let scale_den = self.timebase.den * target_timebase.num;

        let rescale = |value: i64| -> i64 {
            // Use 128-bit arithmetic to avoid overflow
            let result = (i128::from(value) * i128::from(scale_num)) / i128::from(scale_den);
            result as i64
        };

        Self {
            pts: rescale(self.pts),
            dts: self.dts.map(rescale),
            timebase: target_timebase,
            duration: self.duration.map(rescale),
        }
    }

    /// Returns the effective decode timestamp.
    ///
    /// Returns the DTS if present, otherwise returns the PTS.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::{Timestamp, Rational};
    ///
    /// let ts = Timestamp::new(1000, Rational::new(1, 1000));
    /// assert_eq!(ts.effective_dts(), 1000);
    ///
    /// let ts_with_dts = Timestamp::with_dts(3000, Some(2000), Rational::new(1, 1000), None);
    /// assert_eq!(ts_with_dts.effective_dts(), 2000);
    /// ```
    #[must_use]
    pub fn effective_dts(&self) -> i64 {
        self.dts.unwrap_or(self.pts)
    }
}

impl Default for Timestamp {
    fn default() -> Self {
        Self {
            pts: 0,
            dts: None,
            timebase: Rational::new(1, 1000),
            duration: None,
        }
    }
}

impl PartialEq for Timestamp {
    fn eq(&self, other: &Self) -> bool {
        // Compare timestamps by converting to a common timebase
        let self_seconds = self.to_seconds();
        let other_seconds = other.to_seconds();
        (self_seconds - other_seconds).abs() < 1e-9
    }
}
impl PartialOrd for Timestamp {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Timestamp {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let self_seconds = self.to_seconds();
        let other_seconds = other.to_seconds();
        self_seconds
            .partial_cmp(&other_seconds)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let ts = Timestamp::new(1000, Rational::new(1, 1000));
        assert_eq!(ts.pts, 1000);
        assert!(ts.dts.is_none());
        assert!(ts.duration.is_none());
    }

    #[test]
    fn test_with_dts() {
        let ts = Timestamp::with_dts(3000, Some(2000), Rational::new(1, 1000), Some(33));
        assert_eq!(ts.pts, 3000);
        assert_eq!(ts.dts, Some(2000));
        assert_eq!(ts.duration, Some(33));
    }

    #[test]
    fn test_to_seconds() {
        let ts = Timestamp::new(1500, Rational::new(1, 1000));
        assert!((ts.to_seconds() - 1.5).abs() < 0.001);

        let ts = Timestamp::new(90000, Rational::new(1, 90000));
        assert!((ts.to_seconds() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dts_to_seconds() {
        let ts = Timestamp::new(1000, Rational::new(1, 1000));
        assert!(ts.dts_to_seconds().is_none());

        let ts = Timestamp::with_dts(3000, Some(2000), Rational::new(1, 1000), None);
        assert!(
            (ts.dts_to_seconds()
                .expect("dts_to_seconds should return value")
                - 2.0)
                .abs()
                < 0.001
        );
    }

    #[test]
    fn test_duration_seconds() {
        let ts = Timestamp::with_dts(0, None, Rational::new(1, 1000), Some(33));
        assert!(
            (ts.duration_seconds()
                .expect("duration_seconds should return value")
                - 0.033)
                .abs()
                < 0.001
        );
    }

    #[test]
    fn test_rescale() {
        // From milliseconds to microseconds
        let ts = Timestamp::new(1000, Rational::new(1, 1000));
        let rescaled = ts.rescale(Rational::new(1, 1_000_000));
        assert_eq!(rescaled.pts, 1_000_000);
        assert!((rescaled.to_seconds() - 1.0).abs() < f64::EPSILON);

        // From 90kHz to milliseconds
        let ts = Timestamp::new(90000, Rational::new(1, 90000));
        let rescaled = ts.rescale(Rational::new(1, 1000));
        assert_eq!(rescaled.pts, 1000);
    }

    #[test]
    fn test_rescale_with_dts() {
        let ts = Timestamp::with_dts(3000, Some(2000), Rational::new(1, 1000), Some(33));
        let rescaled = ts.rescale(Rational::new(1, 1_000_000));
        assert_eq!(rescaled.pts, 3_000_000);
        assert_eq!(rescaled.dts, Some(2_000_000));
        assert_eq!(rescaled.duration, Some(33_000));
    }

    #[test]
    fn test_effective_dts() {
        let ts = Timestamp::new(1000, Rational::new(1, 1000));
        assert_eq!(ts.effective_dts(), 1000);

        let ts = Timestamp::with_dts(3000, Some(2000), Rational::new(1, 1000), None);
        assert_eq!(ts.effective_dts(), 2000);
    }

    #[test]
    fn test_default() {
        let ts = Timestamp::default();
        assert_eq!(ts.pts, 0);
        assert!(ts.dts.is_none());
        assert_eq!(ts.timebase, Rational::new(1, 1000));
    }

    #[test]
    fn test_partial_eq() {
        let ts1 = Timestamp::new(1000, Rational::new(1, 1000));
        let ts2 = Timestamp::new(1_000_000, Rational::new(1, 1_000_000));
        assert_eq!(ts1, ts2);
    }
}
