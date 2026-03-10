//! Dolby Vision level-based luminance mapping
//!
//! Implements luminance mapping curves used by DV metadata levels to convert
//! between source and target display luminance ranges. Includes polynomial,
//! multi-piece, and spline-based mapping strategies.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A polynomial mapping curve of degree N (max 3).
///
/// Maps normalised PQ values `[0,1]` to `[0,1]` using:
/// `y = c0 + c1*x + c2*x^2 + c3*x^3`
#[derive(Debug, Clone, PartialEq)]
pub struct PolynomialCurve {
    /// Constant coefficient (c0).
    pub c0: f64,
    /// Linear coefficient (c1).
    pub c1: f64,
    /// Quadratic coefficient (c2).
    pub c2: f64,
    /// Cubic coefficient (c3).
    pub c3: f64,
}

impl PolynomialCurve {
    /// Create a new polynomial curve.
    #[must_use]
    pub const fn new(c0: f64, c1: f64, c2: f64, c3: f64) -> Self {
        Self { c0, c1, c2, c3 }
    }

    /// Identity curve (y = x).
    #[must_use]
    pub const fn identity() -> Self {
        Self::new(0.0, 1.0, 0.0, 0.0)
    }

    /// Evaluate the polynomial at a given input.
    #[must_use]
    pub fn evaluate(&self, x: f64) -> f64 {
        self.c0 + self.c1 * x + self.c2 * x * x + self.c3 * x * x * x
    }

    /// Evaluate and clamp to [0, 1].
    #[must_use]
    pub fn evaluate_clamped(&self, x: f64) -> f64 {
        self.evaluate(x).clamp(0.0, 1.0)
    }

    /// Derivative at a given input.
    #[must_use]
    pub fn derivative(&self, x: f64) -> f64 {
        self.c1 + 2.0 * self.c2 * x + 3.0 * self.c3 * x * x
    }

    /// Whether this is effectively an identity mapping.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        (self.c0).abs() < 1e-9
            && (self.c1 - 1.0).abs() < 1e-9
            && (self.c2).abs() < 1e-9
            && (self.c3).abs() < 1e-9
    }
}

impl Default for PolynomialCurve {
    fn default() -> Self {
        Self::identity()
    }
}

/// A single piece in a multi-piece mapping curve.
#[derive(Debug, Clone, PartialEq)]
pub struct MappingPiece {
    /// Start of the input range for this piece (inclusive).
    pub input_start: f64,
    /// End of the input range for this piece (exclusive).
    pub input_end: f64,
    /// Polynomial curve for this piece.
    pub curve: PolynomialCurve,
}

impl MappingPiece {
    /// Create a new mapping piece.
    #[must_use]
    pub fn new(input_start: f64, input_end: f64, curve: PolynomialCurve) -> Self {
        Self {
            input_start,
            input_end,
            curve,
        }
    }

    /// Whether the given input falls within this piece's range.
    #[must_use]
    pub fn contains(&self, x: f64) -> bool {
        x >= self.input_start && x < self.input_end
    }

    /// Width of the input range.
    #[must_use]
    pub fn width(&self) -> f64 {
        self.input_end - self.input_start
    }

    /// Evaluate the piece at a given input (does not check bounds).
    #[must_use]
    pub fn evaluate(&self, x: f64) -> f64 {
        // Normalise x within this piece's range
        let t = if self.width() > 0.0 {
            (x - self.input_start) / self.width()
        } else {
            0.0
        };
        self.curve.evaluate(t)
    }
}

/// A multi-piece luminance mapping curve.
///
/// Composed of consecutive non-overlapping polynomial pieces covering [0, 1].
#[derive(Debug, Clone, Default)]
pub struct MultiPieceCurve {
    /// Pieces sorted by input_start.
    pub pieces: Vec<MappingPiece>,
}

impl MultiPieceCurve {
    /// Create an empty multi-piece curve.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a piece. Pieces should be added in order of `input_start`.
    pub fn add_piece(&mut self, piece: MappingPiece) {
        self.pieces.push(piece);
    }

    /// Number of pieces.
    #[must_use]
    pub fn piece_count(&self) -> usize {
        self.pieces.len()
    }

    /// Evaluate the curve at a given input by finding the correct piece.
    #[must_use]
    pub fn evaluate(&self, x: f64) -> f64 {
        for piece in &self.pieces {
            if piece.contains(x) {
                return piece.evaluate(x);
            }
        }
        // Fallback: clamp to last piece or identity
        if let Some(last) = self.pieces.last() {
            last.evaluate(x)
        } else {
            x
        }
    }

    /// Validate the curve for continuity and coverage.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.pieces.is_empty() {
            errors.push("No pieces defined".to_string());
            return errors;
        }

        // Check coverage starts at 0
        if (self.pieces[0].input_start).abs() > 1e-6 {
            errors.push(format!(
                "First piece starts at {} instead of 0.0",
                self.pieces[0].input_start
            ));
        }

        // Check for gaps and overlaps between consecutive pieces
        for i in 0..self.pieces.len() - 1 {
            let gap = self.pieces[i + 1].input_start - self.pieces[i].input_end;
            if gap.abs() > 1e-6 {
                if gap > 0.0 {
                    errors.push(format!("Gap between piece {} and {}: {:.6}", i, i + 1, gap));
                } else {
                    errors.push(format!(
                        "Overlap between piece {} and {}: {:.6}",
                        i,
                        i + 1,
                        -gap
                    ));
                }
            }
        }

        errors
    }
}

/// Mapping strategy selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MappingStrategy {
    /// Single polynomial curve.
    Polynomial,
    /// Multi-piece polynomial curve.
    MultiPiece,
    /// Pass-through (no mapping).
    Passthrough,
}

impl MappingStrategy {
    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Polynomial => "Polynomial",
            Self::MultiPiece => "Multi-Piece",
            Self::Passthrough => "Passthrough",
        }
    }
}

/// Complete level mapping configuration for a target display.
#[derive(Debug, Clone)]
pub struct LevelMapping {
    /// Target display peak luminance in nits.
    pub target_nits: f32,
    /// Mapping strategy used.
    pub strategy: MappingStrategy,
    /// Single polynomial curve (if strategy is Polynomial).
    pub polynomial: Option<PolynomialCurve>,
    /// Multi-piece curve (if strategy is MultiPiece).
    pub multi_piece: Option<MultiPieceCurve>,
}

impl LevelMapping {
    /// Create a passthrough (identity) mapping for the given target.
    #[must_use]
    pub fn passthrough(target_nits: f32) -> Self {
        Self {
            target_nits,
            strategy: MappingStrategy::Passthrough,
            polynomial: None,
            multi_piece: None,
        }
    }

    /// Create a polynomial mapping.
    #[must_use]
    pub fn with_polynomial(target_nits: f32, curve: PolynomialCurve) -> Self {
        Self {
            target_nits,
            strategy: MappingStrategy::Polynomial,
            polynomial: Some(curve),
            multi_piece: None,
        }
    }

    /// Create a multi-piece mapping.
    #[must_use]
    pub fn with_multi_piece(target_nits: f32, curve: MultiPieceCurve) -> Self {
        Self {
            target_nits,
            strategy: MappingStrategy::MultiPiece,
            polynomial: None,
            multi_piece: Some(curve),
        }
    }

    /// Evaluate the mapping for a normalised input value.
    #[must_use]
    pub fn evaluate(&self, x: f64) -> f64 {
        match self.strategy {
            MappingStrategy::Passthrough => x,
            MappingStrategy::Polynomial => self
                .polynomial
                .as_ref()
                .map_or(x, |c| c.evaluate_clamped(x)),
            MappingStrategy::MultiPiece => self.multi_piece.as_ref().map_or(x, |c| c.evaluate(x)),
        }
    }
}

/// A set of level mappings for multiple target displays.
#[derive(Debug, Default)]
pub struct LevelMappingSet {
    /// Mappings indexed by target nits.
    mappings: Vec<LevelMapping>,
}

impl LevelMappingSet {
    /// Create an empty mapping set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a mapping.
    pub fn add(&mut self, mapping: LevelMapping) {
        self.mappings.push(mapping);
    }

    /// Number of mappings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mappings.len()
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mappings.is_empty()
    }

    /// Find the mapping closest to the requested target nits.
    #[must_use]
    pub fn find_closest(&self, target_nits: f32) -> Option<&LevelMapping> {
        self.mappings.iter().min_by(|a, b| {
            let da = (a.target_nits - target_nits).abs();
            let db = (b.target_nits - target_nits).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Evaluate the mapping closest to the target nits.
    #[must_use]
    pub fn evaluate_for_target(&self, target_nits: f32, x: f64) -> f64 {
        self.find_closest(target_nits).map_or(x, |m| m.evaluate(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_identity() {
        let c = PolynomialCurve::identity();
        assert!(c.is_identity());
        assert!((c.evaluate(0.5) - 0.5).abs() < 1e-9);
        assert!((c.evaluate(0.0) - 0.0).abs() < 1e-9);
        assert!((c.evaluate(1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_polynomial_quadratic() {
        let c = PolynomialCurve::new(0.0, 0.0, 1.0, 0.0);
        assert!((c.evaluate(0.5) - 0.25).abs() < 1e-9);
        assert!((c.evaluate(1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_polynomial_derivative() {
        let c = PolynomialCurve::identity();
        assert!((c.derivative(0.5) - 1.0).abs() < 1e-9);

        let c2 = PolynomialCurve::new(0.0, 0.0, 1.0, 0.0);
        assert!((c2.derivative(0.5) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_polynomial_clamped() {
        let c = PolynomialCurve::new(0.0, 2.0, 0.0, 0.0);
        assert!((c.evaluate_clamped(0.8) - 1.0).abs() < 1e-9);
        assert!((c.evaluate_clamped(-0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_mapping_piece_contains() {
        let p = MappingPiece::new(0.0, 0.5, PolynomialCurve::identity());
        assert!(p.contains(0.0));
        assert!(p.contains(0.25));
        assert!(!p.contains(0.5));
        assert!(!p.contains(1.0));
    }

    #[test]
    fn test_mapping_piece_width() {
        let p = MappingPiece::new(0.2, 0.8, PolynomialCurve::identity());
        assert!((p.width() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_multi_piece_curve_evaluate() {
        let mut curve = MultiPieceCurve::new();
        // Lower half: identity
        curve.add_piece(MappingPiece::new(0.0, 0.5, PolynomialCurve::identity()));
        // Upper half: quadratic
        curve.add_piece(MappingPiece::new(
            0.5,
            1.0,
            PolynomialCurve::new(0.0, 0.0, 1.0, 0.0),
        ));
        assert_eq!(curve.piece_count(), 2);
        // Lower half: identity maps 0.25 in [0, 0.5] to t=0.5, identity(0.5)=0.5
        let val = curve.evaluate(0.25);
        assert!((val - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_multi_piece_curve_validate_ok() {
        let mut curve = MultiPieceCurve::new();
        curve.add_piece(MappingPiece::new(0.0, 0.5, PolynomialCurve::identity()));
        curve.add_piece(MappingPiece::new(0.5, 1.0, PolynomialCurve::identity()));
        assert!(curve.validate().is_empty());
    }

    #[test]
    fn test_multi_piece_curve_validate_gap() {
        let mut curve = MultiPieceCurve::new();
        curve.add_piece(MappingPiece::new(0.0, 0.4, PolynomialCurve::identity()));
        curve.add_piece(MappingPiece::new(0.6, 1.0, PolynomialCurve::identity()));
        let errs = curve.validate();
        assert!(!errs.is_empty());
        assert!(errs[0].contains("Gap"));
    }

    #[test]
    fn test_level_mapping_passthrough() {
        let m = LevelMapping::passthrough(1000.0);
        assert_eq!(m.strategy, MappingStrategy::Passthrough);
        assert!((m.evaluate(0.7) - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_level_mapping_polynomial() {
        let m = LevelMapping::with_polynomial(400.0, PolynomialCurve::new(0.0, 0.5, 0.5, 0.0));
        assert_eq!(m.strategy, MappingStrategy::Polynomial);
        let val = m.evaluate(1.0);
        assert!((val - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_level_mapping_set_find_closest() {
        let mut set = LevelMappingSet::new();
        set.add(LevelMapping::passthrough(100.0));
        set.add(LevelMapping::passthrough(600.0));
        set.add(LevelMapping::passthrough(4000.0));

        let closest = set.find_closest(500.0).expect("closest should be valid");
        assert!((closest.target_nits - 600.0).abs() < 0.1);
    }

    #[test]
    fn test_level_mapping_set_evaluate_for_target() {
        let mut set = LevelMappingSet::new();
        set.add(LevelMapping::with_polynomial(
            100.0,
            PolynomialCurve::new(0.0, 0.5, 0.0, 0.0),
        ));
        set.add(LevelMapping::passthrough(1000.0));

        // Close to 1000 => passthrough
        let v = set.evaluate_for_target(900.0, 0.8);
        assert!((v - 0.8).abs() < 1e-9);

        // Close to 100 => polynomial x*0.5
        let v2 = set.evaluate_for_target(120.0, 0.8);
        assert!((v2 - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_mapping_strategy_label() {
        assert_eq!(MappingStrategy::Polynomial.label(), "Polynomial");
        assert_eq!(MappingStrategy::MultiPiece.label(), "Multi-Piece");
        assert_eq!(MappingStrategy::Passthrough.label(), "Passthrough");
    }

    #[test]
    fn test_level_mapping_set_empty() {
        let set = LevelMappingSet::new();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
        assert!(set.find_closest(100.0).is_none());
        // evaluate_for_target returns x when empty
        assert!((set.evaluate_for_target(100.0, 0.5) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_polynomial_default() {
        let c = PolynomialCurve::default();
        assert!(c.is_identity());
    }
}
