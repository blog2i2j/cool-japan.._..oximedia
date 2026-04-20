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
    /// assert!((ts.dts_to_seconds().expect("dts present") - 2.0).abs() < 0.001);
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
    /// assert!((ts.duration_seconds().expect("duration present") - 0.033).abs() < 0.001);
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

    /// Creates a timestamp from a floating-point seconds value.
    ///
    /// The timestamp is quantised to the given timebase.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::{Timestamp, Rational};
    ///
    /// let ts = Timestamp::from_seconds(1.5, Rational::new(1, 1000));
    /// assert_eq!(ts.pts, 1500);
    /// ```
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_seconds(seconds: f64, timebase: Rational) -> Self {
        let tb_f64 = timebase.to_f64();
        let pts = if tb_f64.abs() < f64::EPSILON {
            0
        } else {
            (seconds / tb_f64).round() as i64
        };
        Self {
            pts,
            dts: None,
            timebase,
            duration: None,
        }
    }

    /// Checked addition returning `Result` instead of using the `Add` operator.
    ///
    /// Adds another timestamp, rescaling to this timestamp's timebase.
    ///
    /// # Errors
    ///
    /// Returns `OxiError::InvalidData` on arithmetic overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::{Timestamp, Rational};
    ///
    /// let a = Timestamp::new(1000, Rational::new(1, 1000));
    /// let b = Timestamp::new(500, Rational::new(1, 1000));
    /// let sum = a.checked_add(b).expect("should succeed");
    /// assert_eq!(sum.pts, 1500);
    /// ```
    pub fn checked_add(self, rhs: Self) -> Result<Self, crate::OxiError> {
        self + rhs
    }

    /// Checked subtraction returning `Result` instead of using the `Sub` operator.
    ///
    /// Subtracts another timestamp, rescaling to this timestamp's timebase.
    ///
    /// # Errors
    ///
    /// Returns `OxiError::InvalidData` on arithmetic overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::{Timestamp, Rational};
    ///
    /// let a = Timestamp::new(1500, Rational::new(1, 1000));
    /// let b = Timestamp::new(500, Rational::new(1, 1000));
    /// let diff = a.checked_sub(b).expect("should succeed");
    /// assert_eq!(diff.pts, 1000);
    /// ```
    pub fn checked_sub(self, rhs: Self) -> Result<Self, crate::OxiError> {
        self - rhs
    }

    /// Scales all timestamp values by an integer factor with overflow protection.
    ///
    /// # Errors
    ///
    /// Returns `OxiError::InvalidData` on arithmetic overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::{Timestamp, Rational};
    ///
    /// let ts = Timestamp::new(1000, Rational::new(1, 1000));
    /// let doubled = ts.checked_mul_i64(2).expect("should succeed");
    /// assert_eq!(doubled.pts, 2000);
    /// ```
    pub fn checked_mul_i64(self, factor: i64) -> Result<Self, crate::OxiError> {
        self * Rational::new(factor, 1)
    }

    /// Divides all timestamp values by a `Rational` factor with overflow protection.
    ///
    /// This is equivalent to multiplying by the reciprocal of the given rational.
    ///
    /// # Errors
    ///
    /// Returns `OxiError::InvalidData` if the rational numerator is zero or on overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::{Timestamp, Rational};
    ///
    /// let ts = Timestamp::new(1000, Rational::new(1, 1000));
    /// let halved = ts.checked_div(Rational::new(2, 1)).expect("should succeed");
    /// assert_eq!(halved.pts, 500);
    /// ```
    pub fn checked_div(self, rhs: Rational) -> Result<Self, crate::OxiError> {
        if rhs.num == 0 {
            return Err(crate::OxiError::InvalidData(
                "Timestamp division by zero rational".to_string(),
            ));
        }
        self * Rational::new(rhs.den, rhs.num)
    }

    /// Returns the absolute difference between two timestamps in seconds.
    ///
    /// Always returns a non-negative value.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::types::{Timestamp, Rational};
    ///
    /// let a = Timestamp::new(3000, Rational::new(1, 1000));
    /// let b = Timestamp::new(1000, Rational::new(1, 1000));
    /// assert!((a.distance_seconds(&b) - 2.0).abs() < 1e-9);
    /// assert!((b.distance_seconds(&a) - 2.0).abs() < 1e-9);
    /// ```
    #[must_use]
    pub fn distance_seconds(&self, other: &Self) -> f64 {
        (self.to_seconds() - other.to_seconds()).abs()
    }

    /// Adds a wall-clock duration to this timestamp using saturating arithmetic.
    ///
    /// Overflow is clamped to `i64::MAX`.
    #[must_use]
    pub fn duration_add(self, duration: std::time::Duration) -> Self {
        let delta = duration_to_timebase_units(duration, self.timebase, i64::MAX);
        self.with_saturated_pts(self.pts.saturating_add(delta))
    }

    /// Subtracts a wall-clock duration from this timestamp using saturating arithmetic.
    ///
    /// Underflow is clamped to `0`.
    #[must_use]
    pub fn duration_sub(self, duration: std::time::Duration) -> Self {
        let delta = duration_to_timebase_units(duration, self.timebase, i64::MAX);
        self.with_saturated_pts(self.pts.saturating_sub(delta).max(0))
    }

    /// Scales the timestamp by `num / den` using saturating arithmetic.
    ///
    /// If `den` is zero, returns `self` unchanged.
    #[must_use]
    pub fn scale_by(self, num: i64, den: i64) -> Self {
        if den == 0 {
            return self;
        }

        self.with_saturated_pts(saturating_mul_div_i64(self.pts, num, den))
    }

    #[must_use]
    fn with_saturated_pts(self, pts: i64) -> Self {
        let clamped_pts = pts.max(0);
        let dts = self.dts.map(|value| value.clamp(0, clamped_pts));

        Self {
            pts: clamped_pts,
            dts,
            timebase: self.timebase,
            duration: self.duration,
        }
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

impl std::ops::Add for Timestamp {
    type Output = Result<Self, crate::OxiError>;

    /// Adds two timestamps, rescaling the right-hand side to the left's timebase.
    ///
    /// Returns `Err` on arithmetic overflow.
    fn add(self, rhs: Self) -> Self::Output {
        let rhs_rescaled = rhs.rescale(self.timebase);
        let pts = checked_add_i64(self.pts, rhs_rescaled.pts)?;
        let dts = match (self.dts, rhs_rescaled.dts) {
            (Some(a), Some(b)) => Some(checked_add_i64(a, b)?),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };
        let duration = match (self.duration, rhs_rescaled.duration) {
            (Some(a), Some(b)) => Some(checked_add_i64(a, b)?),
            (d @ Some(_), None) | (None, d @ Some(_)) => d,
            (None, None) => None,
        };
        Ok(Self {
            pts,
            dts,
            timebase: self.timebase,
            duration,
        })
    }
}

impl std::ops::Sub for Timestamp {
    type Output = Result<Self, crate::OxiError>;

    /// Subtracts the right-hand timestamp from the left, rescaling to a common timebase.
    ///
    /// Returns `Err` on arithmetic overflow.
    fn sub(self, rhs: Self) -> Self::Output {
        let rhs_rescaled = rhs.rescale(self.timebase);
        let pts = checked_sub_i64(self.pts, rhs_rescaled.pts)?;
        let dts = match (self.dts, rhs_rescaled.dts) {
            (Some(a), Some(b)) => Some(checked_sub_i64(a, b)?),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(checked_sub_i64(0, b)?),
            (None, None) => None,
        };
        Ok(Self {
            pts,
            dts,
            timebase: self.timebase,
            duration: self.duration,
        })
    }
}

impl std::ops::Mul<Rational> for Timestamp {
    type Output = Result<Self, crate::OxiError>;

    /// Scales all timestamp values by a `Rational` factor with overflow protection.
    fn mul(self, rhs: Rational) -> Self::Output {
        let scale = |v: i64| -> Result<i64, crate::OxiError> { checked_mul_rational(v, rhs) };
        let pts = scale(self.pts)?;
        let dts = self.dts.map(scale).transpose()?;
        let duration = self.duration.map(scale).transpose()?;
        Ok(Self {
            pts,
            dts,
            timebase: self.timebase,
            duration,
        })
    }
}

impl std::ops::Mul<i64> for Timestamp {
    type Output = Result<Self, crate::OxiError>;

    /// Scales all timestamp values by an integer factor with overflow protection.
    fn mul(self, rhs: i64) -> Self::Output {
        self * Rational::new(rhs, 1)
    }
}

impl std::ops::Div<Rational> for Timestamp {
    type Output = Result<Self, crate::OxiError>;

    /// Divides all timestamp values by a `Rational` factor with overflow protection.
    ///
    /// Returns `Err` if the rational numerator is zero or on overflow.
    fn div(self, rhs: Rational) -> Self::Output {
        if rhs.num == 0 {
            return Err(crate::OxiError::InvalidData(
                "Timestamp division by zero rational".to_string(),
            ));
        }
        self * Rational::new(rhs.den, rhs.num)
    }
}

/// Checked addition returning an `OxiError` on overflow.
fn checked_add_i64(a: i64, b: i64) -> Result<i64, crate::OxiError> {
    a.checked_add(b).ok_or_else(|| {
        crate::OxiError::InvalidData(format!("Timestamp addition overflow: {a} + {b}"))
    })
}

/// Checked subtraction returning an `OxiError` on overflow.
fn checked_sub_i64(a: i64, b: i64) -> Result<i64, crate::OxiError> {
    a.checked_sub(b).ok_or_else(|| {
        crate::OxiError::InvalidData(format!("Timestamp subtraction overflow: {a} - {b}"))
    })
}

/// Multiplies an `i64` by a `Rational` using 128-bit intermediates with overflow checking.
#[allow(clippy::cast_possible_truncation)]
fn checked_mul_rational(value: i64, r: Rational) -> Result<i64, crate::OxiError> {
    let numerator = i128::from(value)
        .checked_mul(i128::from(r.num))
        .ok_or_else(|| {
            crate::OxiError::InvalidData(format!(
                "Timestamp multiply overflow: {value} * {}/{}",
                r.num, r.den
            ))
        })?;
    let result = numerator / i128::from(r.den);
    if result > i128::from(i64::MAX) || result < i128::from(i64::MIN) {
        return Err(crate::OxiError::InvalidData(format!(
            "Timestamp multiply overflow: result {result} exceeds i64 range"
        )));
    }
    Ok(result as i64)
}

#[allow(clippy::cast_possible_truncation)]
fn duration_to_timebase_units(
    duration: std::time::Duration,
    timebase: Rational,
    fallback: i64,
) -> i64 {
    if timebase.num <= 0 || timebase.den <= 0 {
        return fallback;
    }

    let nanos_per_second = 1_000_000_000_i128;
    let duration_nanos = i128::from(duration.as_secs())
        .saturating_mul(nanos_per_second)
        .saturating_add(i128::from(duration.subsec_nanos()));
    let numerator = duration_nanos
        .saturating_mul(i128::from(timebase.den))
        .checked_div(i128::from(timebase.num).saturating_mul(nanos_per_second));

    match numerator {
        Some(value) => value.clamp(0, i128::from(i64::MAX)) as i64,
        None => fallback,
    }
}

#[allow(clippy::cast_possible_truncation)]
fn saturating_mul_div_i64(value: i64, num: i64, den: i64) -> i64 {
    let denominator = i128::from(den);
    if denominator == 0 {
        return value;
    }

    let scaled = i128::from(value)
        .saturating_mul(i128::from(num))
        .checked_div(denominator);

    match scaled {
        Some(result) if result < 0 => 0,
        Some(result) if result > i128::from(i64::MAX) => i64::MAX,
        Some(result) => result as i64,
        None => {
            if (value > 0 && num > 0 && den > 0) || (value < 0 && num < 0 && den > 0) {
                i64::MAX
            } else {
                0
            }
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

    // ── Timestamp arithmetic tests ────────────────────────────────────

    #[test]
    fn test_add_same_timebase() {
        let a = Timestamp::new(1000, Rational::new(1, 1000));
        let b = Timestamp::new(500, Rational::new(1, 1000));
        let sum = (a + b).expect("addition should succeed");
        assert_eq!(sum.pts, 1500);
        assert_eq!(sum.timebase, Rational::new(1, 1000));
    }

    #[test]
    fn test_add_different_timebases() {
        let a = Timestamp::new(1000, Rational::new(1, 1000)); // 1 second
        let b = Timestamp::new(90000, Rational::new(1, 90000)); // 1 second
        let sum = (a + b).expect("addition should succeed");
        assert_eq!(sum.timebase, Rational::new(1, 1000));
        assert_eq!(sum.pts, 2000); // 2 seconds in ms
    }

    #[test]
    fn test_add_with_dts_and_duration() {
        let a = Timestamp::with_dts(100, Some(90), Rational::new(1, 1000), Some(33));
        let b = Timestamp::with_dts(200, Some(180), Rational::new(1, 1000), Some(33));
        let sum = (a + b).expect("addition should succeed");
        assert_eq!(sum.pts, 300);
        assert_eq!(sum.dts, Some(270));
        assert_eq!(sum.duration, Some(66));
    }

    #[test]
    fn test_sub_same_timebase() {
        let a = Timestamp::new(1500, Rational::new(1, 1000));
        let b = Timestamp::new(500, Rational::new(1, 1000));
        let diff = (a - b).expect("subtraction should succeed");
        assert_eq!(diff.pts, 1000);
    }

    #[test]
    fn test_sub_different_timebases() {
        let a = Timestamp::new(2000, Rational::new(1, 1000)); // 2 seconds
        let b = Timestamp::new(90000, Rational::new(1, 90000)); // 1 second
        let diff = (a - b).expect("subtraction should succeed");
        assert_eq!(diff.pts, 1000); // 1 second in ms
    }

    #[test]
    fn test_mul_by_rational() {
        let ts = Timestamp::new(1000, Rational::new(1, 1000));
        let doubled = (ts * Rational::new(2, 1)).expect("multiply should succeed");
        assert_eq!(doubled.pts, 2000);
        assert_eq!(doubled.timebase, Rational::new(1, 1000));
    }

    #[test]
    fn test_mul_with_dts_and_duration() {
        let ts = Timestamp::with_dts(100, Some(90), Rational::new(1, 1000), Some(33));
        let scaled = (ts * Rational::new(3, 1)).expect("multiply should succeed");
        assert_eq!(scaled.pts, 300);
        assert_eq!(scaled.dts, Some(270));
        assert_eq!(scaled.duration, Some(99));
    }

    #[test]
    fn test_mul_by_fraction() {
        let ts = Timestamp::new(1000, Rational::new(1, 1000));
        let halved = (ts * Rational::new(1, 2)).expect("multiply should succeed");
        assert_eq!(halved.pts, 500);
    }

    #[test]
    fn test_add_overflow_protection() {
        let a = Timestamp::new(i64::MAX, Rational::new(1, 1000));
        let b = Timestamp::new(1, Rational::new(1, 1000));
        let result = a + b;
        assert!(result.is_err());
    }

    #[test]
    fn test_sub_overflow_protection() {
        let a = Timestamp::new(i64::MIN, Rational::new(1, 1000));
        let b = Timestamp::new(1, Rational::new(1, 1000));
        let result = a - b;
        assert!(result.is_err());
    }

    #[test]
    fn test_mul_overflow_protection() {
        let ts = Timestamp::new(i64::MAX, Rational::new(1, 1000));
        let result = ts * Rational::new(i64::MAX, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_mul_zero_rational() {
        let ts = Timestamp::new(1000, Rational::new(1, 1000));
        let zeroed = (ts * Rational::new(0, 1)).expect("multiply by zero should succeed");
        assert_eq!(zeroed.pts, 0);
    }

    #[test]
    fn test_add_preserves_timebase() {
        let tb = Rational::new(1, 48000);
        let a = Timestamp::new(48000, tb);
        let b = Timestamp::new(24000, tb);
        let sum = (a + b).expect("addition should succeed");
        assert_eq!(sum.timebase, tb);
        assert_eq!(sum.pts, 72000);
    }

    #[test]
    fn test_sub_negative_result() {
        let a = Timestamp::new(100, Rational::new(1, 1000));
        let b = Timestamp::new(500, Rational::new(1, 1000));
        let diff = (a - b).expect("subtraction should succeed");
        assert_eq!(diff.pts, -400);
    }

    // ── Div<Rational> tests ──────────────────────────────────────────

    #[test]
    fn test_div_by_rational() {
        let ts = Timestamp::new(1000, Rational::new(1, 1000));
        let halved = (ts / Rational::new(2, 1)).expect("division should succeed");
        assert_eq!(halved.pts, 500);
        assert_eq!(halved.timebase, Rational::new(1, 1000));
    }

    #[test]
    fn test_div_by_fraction() {
        let ts = Timestamp::new(1000, Rational::new(1, 1000));
        // Dividing by 1/2 is multiplying by 2
        let doubled = (ts / Rational::new(1, 2)).expect("division should succeed");
        assert_eq!(doubled.pts, 2000);
    }

    #[test]
    fn test_div_with_dts_and_duration() {
        let ts = Timestamp::with_dts(300, Some(270), Rational::new(1, 1000), Some(99));
        let scaled = (ts / Rational::new(3, 1)).expect("division should succeed");
        assert_eq!(scaled.pts, 100);
        assert_eq!(scaled.dts, Some(90));
        assert_eq!(scaled.duration, Some(33));
    }

    #[test]
    fn test_div_by_zero_rational() {
        let ts = Timestamp::new(1000, Rational::new(1, 1000));
        let result = ts / Rational::new(0, 1);
        assert!(result.is_err());
    }

    // ── Mul<i64> tests ──────────────────────────────────────────────

    #[test]
    fn test_mul_by_i64() {
        let ts = Timestamp::new(500, Rational::new(1, 1000));
        let tripled = (ts * 3_i64).expect("multiply should succeed");
        assert_eq!(tripled.pts, 1500);
    }

    #[test]
    fn test_mul_by_i64_with_dts() {
        let ts = Timestamp::with_dts(100, Some(80), Rational::new(1, 1000), Some(33));
        let scaled = (ts * 5_i64).expect("multiply should succeed");
        assert_eq!(scaled.pts, 500);
        assert_eq!(scaled.dts, Some(400));
        assert_eq!(scaled.duration, Some(165));
    }

    #[test]
    fn test_mul_by_i64_zero() {
        let ts = Timestamp::new(1000, Rational::new(1, 1000));
        let zeroed = (ts * 0_i64).expect("multiply by zero should succeed");
        assert_eq!(zeroed.pts, 0);
    }

    #[test]
    fn test_mul_by_i64_overflow() {
        let ts = Timestamp::new(i64::MAX, Rational::new(1, 1000));
        let result = ts * 2_i64;
        assert!(result.is_err());
    }

    // ── from_seconds tests ──────────────────────────────────────────

    #[test]
    fn test_from_seconds_millisecond_timebase() {
        let ts = Timestamp::from_seconds(1.5, Rational::new(1, 1000));
        assert_eq!(ts.pts, 1500);
        assert_eq!(ts.timebase, Rational::new(1, 1000));
        assert!(ts.dts.is_none());
        assert!(ts.duration.is_none());
    }

    #[test]
    fn test_from_seconds_90khz_timebase() {
        let ts = Timestamp::from_seconds(1.0, Rational::new(1, 90000));
        assert_eq!(ts.pts, 90000);
    }

    #[test]
    fn test_from_seconds_48khz_audio() {
        let ts = Timestamp::from_seconds(2.0, Rational::new(1, 48000));
        assert_eq!(ts.pts, 96000);
    }

    #[test]
    fn test_from_seconds_zero() {
        let ts = Timestamp::from_seconds(0.0, Rational::new(1, 1000));
        assert_eq!(ts.pts, 0);
    }

    #[test]
    fn test_from_seconds_negative() {
        let ts = Timestamp::from_seconds(-1.0, Rational::new(1, 1000));
        assert_eq!(ts.pts, -1000);
    }

    #[test]
    fn test_from_seconds_roundtrip() {
        let original_secs = 3.141;
        let ts = Timestamp::from_seconds(original_secs, Rational::new(1, 1_000_000));
        assert!((ts.to_seconds() - original_secs).abs() < 1e-6);
    }

    // ── checked_add / checked_sub / checked_mul_i64 / checked_div tests ─

    #[test]
    fn test_checked_add_success() {
        let a = Timestamp::new(1000, Rational::new(1, 1000));
        let b = Timestamp::new(500, Rational::new(1, 1000));
        let sum = a.checked_add(b).expect("should succeed");
        assert_eq!(sum.pts, 1500);
    }

    #[test]
    fn test_checked_add_overflow() {
        let a = Timestamp::new(i64::MAX, Rational::new(1, 1000));
        let b = Timestamp::new(1, Rational::new(1, 1000));
        assert!(a.checked_add(b).is_err());
    }

    #[test]
    fn test_checked_sub_success() {
        let a = Timestamp::new(1500, Rational::new(1, 1000));
        let b = Timestamp::new(500, Rational::new(1, 1000));
        let diff = a.checked_sub(b).expect("should succeed");
        assert_eq!(diff.pts, 1000);
    }

    #[test]
    fn test_checked_sub_overflow() {
        let a = Timestamp::new(i64::MIN, Rational::new(1, 1000));
        let b = Timestamp::new(1, Rational::new(1, 1000));
        assert!(a.checked_sub(b).is_err());
    }

    #[test]
    fn test_checked_mul_i64_success() {
        let ts = Timestamp::new(1000, Rational::new(1, 1000));
        let doubled = ts.checked_mul_i64(2).expect("should succeed");
        assert_eq!(doubled.pts, 2000);
    }

    #[test]
    fn test_checked_div_success() {
        let ts = Timestamp::new(1000, Rational::new(1, 1000));
        let halved = ts.checked_div(Rational::new(2, 1)).expect("should succeed");
        assert_eq!(halved.pts, 500);
    }

    #[test]
    fn test_checked_div_by_zero() {
        let ts = Timestamp::new(1000, Rational::new(1, 1000));
        assert!(ts.checked_div(Rational::new(0, 1)).is_err());
    }

    // ── distance_seconds tests ──────────────────────────────────────

    #[test]
    fn test_distance_seconds_same_timebase() {
        let a = Timestamp::new(3000, Rational::new(1, 1000));
        let b = Timestamp::new(1000, Rational::new(1, 1000));
        assert!((a.distance_seconds(&b) - 2.0).abs() < 1e-9);
        assert!((b.distance_seconds(&a) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_distance_seconds_different_timebases() {
        let a = Timestamp::new(2000, Rational::new(1, 1000)); // 2s
        let b = Timestamp::new(90000, Rational::new(1, 90000)); // 1s
        assert!((a.distance_seconds(&b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_distance_seconds_zero() {
        let a = Timestamp::new(1000, Rational::new(1, 1000));
        let b = Timestamp::new(1_000_000, Rational::new(1, 1_000_000));
        assert!(a.distance_seconds(&b) < 1e-9);
    }

    #[test]
    fn test_duration_add_saturating() {
        let ts = Timestamp::new(i64::MAX, Rational::new(1, 1));
        let result = ts.duration_add(std::time::Duration::from_secs(1));
        assert_eq!(result.pts, i64::MAX);
    }

    #[test]
    fn test_duration_sub_saturating() {
        let ts = Timestamp::new(0, Rational::new(1, 1));
        let result = ts.duration_sub(std::time::Duration::from_secs(1));
        assert!(result.pts >= 0);
        assert_eq!(result.pts, 0);
    }

    #[test]
    fn test_scale_by() {
        let ts = Timestamp::new(100, Rational::new(1, 1000));
        assert_eq!(ts.scale_by(3, 2).pts, 150);
        assert_eq!(ts.scale_by(1, 0), ts);
    }

    // ── Timestamp conversion accuracy tests ─────────────────────────

    #[test]
    fn test_conversion_90khz_to_48khz() {
        let ts = Timestamp::new(90000, Rational::new(1, 90000)); // 1 second
        let rescaled = ts.rescale(Rational::new(1, 48000));
        assert_eq!(rescaled.pts, 48000);
        assert!((rescaled.to_seconds() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_conversion_ms_to_fps30() {
        // 1000ms = 1 second, at 30fps timebase that's 30 frames
        let ts = Timestamp::new(1000, Rational::new(1, 1000));
        let rescaled = ts.rescale(Rational::new(1, 30));
        assert_eq!(rescaled.pts, 30);
    }

    #[test]
    fn test_conversion_ntsc_timebase() {
        // NTSC: 30000/1001 fps -> timebase 1001/30000
        // 30 frames at this timebase: pts=30, tb=1001/30000
        // seconds = 30 * 1001/30000 = 30030/30000 = 1.001
        let ts = Timestamp::new(30, Rational::new(1001, 30000));
        assert!((ts.to_seconds() - 1.001).abs() < 1e-6);
    }
}
