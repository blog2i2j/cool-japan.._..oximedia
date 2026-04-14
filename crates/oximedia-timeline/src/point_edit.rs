//! Three-point and four-point editing operations.
//!
//! Professional NLEs support two fundamental editing modes that differ in how
//! many mark-points the editor sets before performing an insert or overwrite:
//!
//! - **Three-point editing** — the editor sets any three of the four possible
//!   points (source in, source out, record in, record out). The fourth is
//!   calculated automatically.
//!
//! - **Four-point editing** — all four points are explicitly set. When the
//!   source duration differs from the record duration, a *fit-to-fill* speed
//!   change is computed so the source fits exactly into the record window.
//!
//! # Example
//!
//! ```
//! use oximedia_timeline::point_edit::{ThreePointEdit, FourPointEdit, PointEditMode};
//! use oximedia_timeline::types::Position;
//!
//! // Three-point edit: source in/out + record in → record out is computed.
//! let edit = ThreePointEdit::with_source_range_and_record_in(
//!     Position::new(100),  // source in
//!     Position::new(200),  // source out
//!     Position::new(500),  // record in
//! ).expect("valid edit");
//! assert_eq!(edit.record_out().value(), 600);
//!
//! // Four-point edit: compute fit-to-fill speed.
//! let edit = FourPointEdit::new(
//!     Position::new(0),    // source in
//!     Position::new(100),  // source out
//!     Position::new(0),    // record in
//!     Position::new(200),  // record out
//! ).expect("valid edit");
//! let speed = edit.fit_to_fill_speed();
//! assert!((speed - 0.5).abs() < f64::EPSILON);
//! ```

use serde::{Deserialize, Serialize};

use crate::error::{TimelineError, TimelineResult};
use crate::types::{Duration, Position};

// ---------------------------------------------------------------------------
// PointEditMode
// ---------------------------------------------------------------------------

/// The operation to perform when executing a point edit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PointEditMode {
    /// Insert — existing material is pushed downstream.
    Insert,
    /// Overwrite — existing material is replaced in place.
    Overwrite,
}

// ---------------------------------------------------------------------------
// ThreePointEdit
// ---------------------------------------------------------------------------

/// A three-point edit where three of the four points (source in/out, record
/// in/out) are specified and the missing fourth is derived.
///
/// Exactly one of the four points is `None`; the constructor methods enforce
/// this invariant.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThreePointEdit {
    source_in: Position,
    source_out: Position,
    record_in: Position,
    record_out: Position,
    /// Which point was computed (not set by the user).
    derived: DerivedPoint,
    /// Operation mode.
    pub mode: PointEditMode,
}

/// Indicates which of the four points was automatically derived.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DerivedPoint {
    /// Source in was derived.
    SourceIn,
    /// Source out was derived.
    SourceOut,
    /// Record in was derived.
    RecordIn,
    /// Record out was derived.
    RecordOut,
}

impl ThreePointEdit {
    // ---- factory helpers (one per missing point) ----

    /// Source in/out + record in → record out is computed.
    ///
    /// # Errors
    ///
    /// Returns an error if `source_in >= source_out`.
    pub fn with_source_range_and_record_in(
        source_in: Position,
        source_out: Position,
        record_in: Position,
    ) -> TimelineResult<Self> {
        let dur = Self::checked_duration(source_in, source_out)?;
        Ok(Self {
            source_in,
            source_out,
            record_in,
            record_out: Position::new(record_in.value() + dur),
            derived: DerivedPoint::RecordOut,
            mode: PointEditMode::Insert,
        })
    }

    /// Source in/out + record out → record in is computed.
    ///
    /// # Errors
    ///
    /// Returns an error if `source_in >= source_out`.
    pub fn with_source_range_and_record_out(
        source_in: Position,
        source_out: Position,
        record_out: Position,
    ) -> TimelineResult<Self> {
        let dur = Self::checked_duration(source_in, source_out)?;
        Ok(Self {
            source_in,
            source_out,
            record_in: Position::new(record_out.value() - dur),
            record_out,
            derived: DerivedPoint::RecordIn,
            mode: PointEditMode::Insert,
        })
    }

    /// Source in + record in/out → source out is computed.
    ///
    /// # Errors
    ///
    /// Returns an error if `record_in >= record_out`.
    pub fn with_source_in_and_record_range(
        source_in: Position,
        record_in: Position,
        record_out: Position,
    ) -> TimelineResult<Self> {
        let dur = Self::checked_duration(record_in, record_out)?;
        Ok(Self {
            source_in,
            source_out: Position::new(source_in.value() + dur),
            record_in,
            record_out,
            derived: DerivedPoint::SourceOut,
            mode: PointEditMode::Insert,
        })
    }

    /// Source out + record in/out → source in is computed.
    ///
    /// # Errors
    ///
    /// Returns an error if `record_in >= record_out`.
    pub fn with_source_out_and_record_range(
        source_out: Position,
        record_in: Position,
        record_out: Position,
    ) -> TimelineResult<Self> {
        let dur = Self::checked_duration(record_in, record_out)?;
        Ok(Self {
            source_in: Position::new(source_out.value() - dur),
            source_out,
            record_in,
            record_out,
            derived: DerivedPoint::SourceIn,
            mode: PointEditMode::Insert,
        })
    }

    // ---- accessors ----

    /// Returns the source in-point.
    #[must_use]
    pub fn source_in(&self) -> Position {
        self.source_in
    }

    /// Returns the source out-point.
    #[must_use]
    pub fn source_out(&self) -> Position {
        self.source_out
    }

    /// Returns the record in-point.
    #[must_use]
    pub fn record_in(&self) -> Position {
        self.record_in
    }

    /// Returns the record out-point.
    #[must_use]
    pub fn record_out(&self) -> Position {
        self.record_out
    }

    /// Returns which point was derived.
    #[must_use]
    pub fn derived_point(&self) -> DerivedPoint {
        self.derived
    }

    /// Returns the edit duration (in frames).
    #[must_use]
    pub fn duration(&self) -> Duration {
        Duration::new(self.record_out.value() - self.record_in.value())
    }

    /// Returns the source duration (in frames).
    #[must_use]
    pub fn source_duration(&self) -> Duration {
        Duration::new(self.source_out.value() - self.source_in.value())
    }

    /// Sets the edit mode.
    pub fn set_mode(&mut self, mode: PointEditMode) {
        self.mode = mode;
    }

    /// Validates that all four points form a consistent edit.
    ///
    /// # Errors
    ///
    /// Returns an error if source or record ranges are negative, or if the
    /// source duration does not match the record duration.
    pub fn validate(&self) -> TimelineResult<()> {
        if self.source_in >= self.source_out {
            return Err(TimelineError::InvalidPosition(
                "source_in must be before source_out".into(),
            ));
        }
        if self.record_in >= self.record_out {
            return Err(TimelineError::InvalidPosition(
                "record_in must be before record_out".into(),
            ));
        }
        let src_dur = self.source_out.value() - self.source_in.value();
        let rec_dur = self.record_out.value() - self.record_in.value();
        if src_dur != rec_dur {
            return Err(TimelineError::InvalidDuration(format!(
                "source duration ({src_dur}) != record duration ({rec_dur}) in three-point edit"
            )));
        }
        Ok(())
    }

    // ---- internal helpers ----

    fn checked_duration(from: Position, to: Position) -> TimelineResult<i64> {
        let dur = to.value() - from.value();
        if dur <= 0 {
            return Err(TimelineError::InvalidDuration(format!(
                "Range must be positive, got {from} .. {to}"
            )));
        }
        Ok(dur)
    }
}

// ---------------------------------------------------------------------------
// FourPointEdit
// ---------------------------------------------------------------------------

/// A four-point edit where all four mark-points are explicitly set.
///
/// When the source duration differs from the record duration, the clip speed
/// must be adjusted (*fit-to-fill*) so that the source material fills the
/// record window exactly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FourPointEdit {
    source_in: Position,
    source_out: Position,
    record_in: Position,
    record_out: Position,
    /// Operation mode.
    pub mode: PointEditMode,
}

impl FourPointEdit {
    /// Creates a new four-point edit.
    ///
    /// # Errors
    ///
    /// Returns an error if any range is non-positive.
    pub fn new(
        source_in: Position,
        source_out: Position,
        record_in: Position,
        record_out: Position,
    ) -> TimelineResult<Self> {
        if source_in >= source_out {
            return Err(TimelineError::InvalidDuration(
                "source range must be positive".into(),
            ));
        }
        if record_in >= record_out {
            return Err(TimelineError::InvalidDuration(
                "record range must be positive".into(),
            ));
        }
        Ok(Self {
            source_in,
            source_out,
            record_in,
            record_out,
            mode: PointEditMode::Insert,
        })
    }

    /// Returns the source in-point.
    #[must_use]
    pub fn source_in(&self) -> Position {
        self.source_in
    }

    /// Returns the source out-point.
    #[must_use]
    pub fn source_out(&self) -> Position {
        self.source_out
    }

    /// Returns the record in-point.
    #[must_use]
    pub fn record_in(&self) -> Position {
        self.record_in
    }

    /// Returns the record out-point.
    #[must_use]
    pub fn record_out(&self) -> Position {
        self.record_out
    }

    /// Returns the source duration (in frames).
    #[must_use]
    pub fn source_duration(&self) -> Duration {
        Duration::new(self.source_out.value() - self.source_in.value())
    }

    /// Returns the record duration (in frames).
    #[must_use]
    pub fn record_duration(&self) -> Duration {
        Duration::new(self.record_out.value() - self.record_in.value())
    }

    /// Returns whether a speed change is needed (source duration != record duration).
    #[must_use]
    pub fn needs_speed_change(&self) -> bool {
        self.source_duration() != self.record_duration()
    }

    /// Computes the fit-to-fill speed multiplier.
    ///
    /// The returned value is `source_duration / record_duration`.
    /// - `1.0` means durations match (no speed change).
    /// - `> 1.0` means the source is longer than the record window, so playback
    ///   must be sped up.
    /// - `< 1.0` means the source is shorter, so playback must be slowed down.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn fit_to_fill_speed(&self) -> f64 {
        let src = self.source_duration().value() as f64;
        let rec = self.record_duration().value() as f64;
        if rec == 0.0 {
            return 1.0;
        }
        src / rec
    }

    /// Sets the edit mode.
    pub fn set_mode(&mut self, mode: PointEditMode) {
        self.mode = mode;
    }

    /// Converts this four-point edit to a three-point edit by dropping the
    /// record out-point and using the source duration instead.
    ///
    /// This discards the fit-to-fill speed change and produces an edit whose
    /// record duration equals the source duration.
    #[must_use]
    pub fn to_three_point_drop_record_out(&self) -> ThreePointEdit {
        let dur = self.source_duration().value();
        ThreePointEdit {
            source_in: self.source_in,
            source_out: self.source_out,
            record_in: self.record_in,
            record_out: Position::new(self.record_in.value() + dur),
            derived: DerivedPoint::RecordOut,
            mode: self.mode,
        }
    }

    /// Converts this four-point edit to a three-point edit by dropping the
    /// source out-point and fitting the source to the record window.
    ///
    /// The source out-point is recalculated so the source duration matches
    /// the record duration.
    #[must_use]
    pub fn to_three_point_drop_source_out(&self) -> ThreePointEdit {
        let dur = self.record_duration().value();
        ThreePointEdit {
            source_in: self.source_in,
            source_out: Position::new(self.source_in.value() + dur),
            record_in: self.record_in,
            record_out: self.record_out,
            derived: DerivedPoint::SourceOut,
            mode: self.mode,
        }
    }

    /// Checks whether the fit-to-fill speed is within a practical range.
    ///
    /// Returns `true` if the speed is between `min` and `max` (inclusive).
    #[must_use]
    pub fn speed_in_range(&self, min: f64, max: f64) -> bool {
        let speed = self.fit_to_fill_speed();
        speed >= min && speed <= max
    }

    /// Validates the edit.
    ///
    /// # Errors
    ///
    /// Returns an error if ranges are non-positive.
    pub fn validate(&self) -> TimelineResult<()> {
        if self.source_in >= self.source_out {
            return Err(TimelineError::InvalidPosition(
                "source_in must be before source_out".into(),
            ));
        }
        if self.record_in >= self.record_out {
            return Err(TimelineError::InvalidPosition(
                "record_in must be before record_out".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ThreePointEdit ---------------------------------------------------

    #[test]
    fn test_three_point_source_range_record_in() {
        let edit = ThreePointEdit::with_source_range_and_record_in(
            Position::new(100),
            Position::new(200),
            Position::new(500),
        )
        .expect("valid");
        assert_eq!(edit.record_out().value(), 600);
        assert_eq!(edit.derived_point(), DerivedPoint::RecordOut);
        assert_eq!(edit.duration().value(), 100);
    }

    #[test]
    fn test_three_point_source_range_record_out() {
        let edit = ThreePointEdit::with_source_range_and_record_out(
            Position::new(0),
            Position::new(50),
            Position::new(200),
        )
        .expect("valid");
        assert_eq!(edit.record_in().value(), 150);
        assert_eq!(edit.derived_point(), DerivedPoint::RecordIn);
    }

    #[test]
    fn test_three_point_source_in_record_range() {
        let edit = ThreePointEdit::with_source_in_and_record_range(
            Position::new(10),
            Position::new(0),
            Position::new(75),
        )
        .expect("valid");
        assert_eq!(edit.source_out().value(), 85);
        assert_eq!(edit.derived_point(), DerivedPoint::SourceOut);
    }

    #[test]
    fn test_three_point_source_out_record_range() {
        let edit = ThreePointEdit::with_source_out_and_record_range(
            Position::new(100),
            Position::new(0),
            Position::new(40),
        )
        .expect("valid");
        assert_eq!(edit.source_in().value(), 60);
        assert_eq!(edit.derived_point(), DerivedPoint::SourceIn);
    }

    #[test]
    fn test_three_point_invalid_source_range() {
        let result = ThreePointEdit::with_source_range_and_record_in(
            Position::new(200),
            Position::new(100),
            Position::new(0),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_three_point_invalid_record_range() {
        let result = ThreePointEdit::with_source_in_and_record_range(
            Position::new(0),
            Position::new(100),
            Position::new(50),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_three_point_validate() {
        let edit = ThreePointEdit::with_source_range_and_record_in(
            Position::new(0),
            Position::new(100),
            Position::new(0),
        )
        .expect("valid");
        assert!(edit.validate().is_ok());
    }

    #[test]
    fn test_three_point_set_mode() {
        let mut edit = ThreePointEdit::with_source_range_and_record_in(
            Position::new(0),
            Position::new(50),
            Position::new(0),
        )
        .expect("valid");
        edit.set_mode(PointEditMode::Overwrite);
        assert_eq!(edit.mode, PointEditMode::Overwrite);
    }

    #[test]
    fn test_three_point_source_duration() {
        let edit = ThreePointEdit::with_source_range_and_record_in(
            Position::new(10),
            Position::new(60),
            Position::new(0),
        )
        .expect("valid");
        assert_eq!(edit.source_duration().value(), 50);
    }

    // -- FourPointEdit ----------------------------------------------------

    #[test]
    fn test_four_point_creation() {
        let edit = FourPointEdit::new(
            Position::new(0),
            Position::new(100),
            Position::new(0),
            Position::new(100),
        )
        .expect("valid");
        assert!(!edit.needs_speed_change());
        assert!((edit.fit_to_fill_speed() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_four_point_speed_up() {
        let edit = FourPointEdit::new(
            Position::new(0),
            Position::new(200),
            Position::new(0),
            Position::new(100),
        )
        .expect("valid");
        assert!(edit.needs_speed_change());
        assert!((edit.fit_to_fill_speed() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_four_point_slow_down() {
        let edit = FourPointEdit::new(
            Position::new(0),
            Position::new(100),
            Position::new(0),
            Position::new(200),
        )
        .expect("valid");
        assert!((edit.fit_to_fill_speed() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_four_point_invalid_source() {
        let result = FourPointEdit::new(
            Position::new(100),
            Position::new(50),
            Position::new(0),
            Position::new(100),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_four_point_invalid_record() {
        let result = FourPointEdit::new(
            Position::new(0),
            Position::new(100),
            Position::new(200),
            Position::new(100),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_four_point_to_three_point_drop_record_out() {
        let edit = FourPointEdit::new(
            Position::new(0),
            Position::new(100),
            Position::new(500),
            Position::new(700),
        )
        .expect("valid");
        let three = edit.to_three_point_drop_record_out();
        // Source duration (100) should be used, not record duration (200)
        assert_eq!(three.record_out().value(), 600);
        assert_eq!(three.derived_point(), DerivedPoint::RecordOut);
    }

    #[test]
    fn test_four_point_to_three_point_drop_source_out() {
        let edit = FourPointEdit::new(
            Position::new(0),
            Position::new(100),
            Position::new(0),
            Position::new(200),
        )
        .expect("valid");
        let three = edit.to_three_point_drop_source_out();
        // Record duration (200) should be used
        assert_eq!(three.source_out().value(), 200);
        assert_eq!(three.derived_point(), DerivedPoint::SourceOut);
    }

    #[test]
    fn test_four_point_speed_in_range() {
        let edit = FourPointEdit::new(
            Position::new(0),
            Position::new(100),
            Position::new(0),
            Position::new(200),
        )
        .expect("valid");
        assert!(edit.speed_in_range(0.25, 4.0));
        assert!(!edit.speed_in_range(1.0, 4.0));
    }

    #[test]
    fn test_four_point_validate() {
        let edit = FourPointEdit::new(
            Position::new(0),
            Position::new(100),
            Position::new(0),
            Position::new(100),
        )
        .expect("valid");
        assert!(edit.validate().is_ok());
    }

    #[test]
    fn test_four_point_set_mode() {
        let mut edit = FourPointEdit::new(
            Position::new(0),
            Position::new(100),
            Position::new(0),
            Position::new(100),
        )
        .expect("valid");
        edit.set_mode(PointEditMode::Overwrite);
        assert_eq!(edit.mode, PointEditMode::Overwrite);
    }

    #[test]
    fn test_four_point_durations() {
        let edit = FourPointEdit::new(
            Position::new(10),
            Position::new(60),
            Position::new(100),
            Position::new(250),
        )
        .expect("valid");
        assert_eq!(edit.source_duration().value(), 50);
        assert_eq!(edit.record_duration().value(), 150);
    }
}
