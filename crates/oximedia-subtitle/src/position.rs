//! Subtitle position normalisation.
//!
//! Converts raw [`SubtitlePosition`] values (which may use arbitrary
//! coordinate systems, pixel positions, or percentage-based offsets) into a
//! canonical [`NormalizedPosition`] with percentage-based coordinates in the
//! range `[0.0, 100.0]` and one of three vertical placement modes:
//! **top**, **center**, or **bottom**.

use crate::style::Position;

// ── Public types ──────────────────────────────────────────────────────────────

/// Vertical placement mode for a normalised subtitle position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerticalMode {
    /// Subtitle is placed near the top of the video frame (≤ 33 %).
    Top,
    /// Subtitle is placed near the vertical center (34 %–66 %).
    Center,
    /// Subtitle is placed near the bottom of the video frame (≥ 67 %).
    Bottom,
}

/// A normalised subtitle position with percentage coordinates and a named
/// vertical placement mode.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedPosition {
    /// Horizontal position as a percentage of frame width, in `[0.0, 100.0]`.
    pub x_pct: f32,
    /// Vertical position as a percentage of frame height, in `[0.0, 100.0]`.
    pub y_pct: f32,
    /// Named vertical placement derived from `y_pct`.
    pub vertical_mode: VerticalMode,
}

// ── SubtitlePositionNormalizer ────────────────────────────────────────────────

/// Normalises raw subtitle positions to canonical percentage-based coordinates.
///
/// # Example
///
/// ```rust
/// use oximedia_subtitle::position::{SubtitlePositionNormalizer, VerticalMode};
/// use oximedia_subtitle::style::Position;
///
/// let pos = Position::new(0.5, 0.9);
/// let norm = SubtitlePositionNormalizer::normalize(&pos);
/// assert_eq!(norm.vertical_mode, VerticalMode::Bottom);
/// assert!((norm.y_pct - 90.0).abs() < 0.1);
/// ```
pub struct SubtitlePositionNormalizer;

impl SubtitlePositionNormalizer {
    /// Normalise a raw [`Position`] to a [`NormalizedPosition`].
    ///
    /// The `Position::x` and `Position::y` fields are expected to be in the
    /// range `[0.0, 100.0]` (percentage of frame).  Values outside this range
    /// are clamped.  Values greater than `1.0` are treated as percentages;
    /// values ≤ `1.0` (fractional) are multiplied by 100 to convert them.
    #[must_use]
    pub fn normalize(pos: &Position) -> NormalizedPosition {
        let x_pct = normalise_coord(pos.x);
        let y_pct = normalise_coord(pos.y);

        let vertical_mode = if y_pct <= 33.0 {
            VerticalMode::Top
        } else if y_pct <= 66.0 {
            VerticalMode::Center
        } else {
            VerticalMode::Bottom
        };

        NormalizedPosition {
            x_pct,
            y_pct,
            vertical_mode,
        }
    }

    /// Create a default bottom-center position (standard subtitle placement).
    #[must_use]
    pub fn default_bottom() -> NormalizedPosition {
        NormalizedPosition {
            x_pct: 50.0,
            y_pct: 90.0,
            vertical_mode: VerticalMode::Bottom,
        }
    }

    /// Create a top-center position (e.g. for upper third titles).
    #[must_use]
    pub fn default_top() -> NormalizedPosition {
        NormalizedPosition {
            x_pct: 50.0,
            y_pct: 10.0,
            vertical_mode: VerticalMode::Top,
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Normalise a single coordinate value.
///
/// - Values in `(1.0, 100.0]` are treated as percentages and clamped.
/// - Values in `[0.0, 1.0]` are treated as fractions and scaled to `[0, 100]`.
/// - Negative values are clamped to 0.
/// - Values > 100 are clamped to 100.
fn normalise_coord(v: f32) -> f32 {
    if v <= 1.0 && v >= 0.0 {
        // Fractional (0–1): scale to percentage.
        v * 100.0
    } else {
        v.clamp(0.0, 100.0)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pos(x: f32, y: f32) -> Position {
        Position::new(x, y)
    }

    #[test]
    fn test_bottom_mode_fractional() {
        // y=0.9 → 90% → Bottom
        let n = SubtitlePositionNormalizer::normalize(&pos(0.5, 0.9));
        assert_eq!(n.vertical_mode, VerticalMode::Bottom);
        assert!((n.x_pct - 50.0).abs() < 0.5);
        assert!((n.y_pct - 90.0).abs() < 0.5);
    }

    #[test]
    fn test_top_mode() {
        // y=0.1 → 10% → Top
        let n = SubtitlePositionNormalizer::normalize(&pos(0.25, 0.1));
        assert_eq!(n.vertical_mode, VerticalMode::Top);
    }

    #[test]
    fn test_center_mode() {
        // y=0.5 → 50% → Center
        let n = SubtitlePositionNormalizer::normalize(&pos(0.5, 0.5));
        assert_eq!(n.vertical_mode, VerticalMode::Center);
    }

    #[test]
    fn test_fractional_input_scaled() {
        let n = SubtitlePositionNormalizer::normalize(&pos(0.5, 0.9));
        assert!((n.x_pct - 50.0).abs() < 0.5);
        assert!((n.y_pct - 90.0).abs() < 0.5);
        assert_eq!(n.vertical_mode, VerticalMode::Bottom);
    }

    #[test]
    fn test_over_1_clamped_to_100() {
        // Values > 1 but ≤ 100 are treated as percentages.
        let n = SubtitlePositionNormalizer::normalize(&pos(50.0, 90.0));
        assert!((n.x_pct - 50.0).abs() < 0.01);
        assert!((n.y_pct - 90.0).abs() < 0.01);
    }

    #[test]
    fn test_boundary_top_center() {
        // y=0.33 → 33% → Top boundary
        let n = SubtitlePositionNormalizer::normalize(&pos(0.0, 0.33));
        assert_eq!(n.vertical_mode, VerticalMode::Top);
    }

    #[test]
    fn test_boundary_center() {
        // y=0.5 → 50% → Center
        let n = SubtitlePositionNormalizer::normalize(&pos(0.0, 0.5));
        assert_eq!(n.vertical_mode, VerticalMode::Center);
    }

    #[test]
    fn test_boundary_bottom() {
        // y=0.67 → 67% → Bottom
        let n = SubtitlePositionNormalizer::normalize(&pos(0.0, 0.67));
        assert_eq!(n.vertical_mode, VerticalMode::Bottom);
    }
}
