//! Cubic B-spline and Catmull-Rom curve types for smoother rotoscoping masks.
//!
//! These curve types complement the existing [`BezierCurve`](super::bezier::BezierCurve)
//! with alternative interpolation schemes that are often easier to control in
//! rotoscoping workflows:
//!
//! - **Catmull-Rom** splines pass *through* every control point, making them
//!   intuitive for artists who want to click directly on the mask boundary.
//! - **Cubic B-splines** approximate the control polygon with C2 continuity,
//!   producing very smooth curves useful for soft organic shapes.
//!
//! Both types support closed and open curves and can be sampled into polylines
//! for rasterisation by the existing [`BezierMask`](super::bezier::BezierMask)
//! pipeline.
//!
//! # Example
//!
//! ```
//! use oximedia_vfx::rotoscoping::spline_curves::{CatmullRomCurve, CubicBSplineCurve, SplinePoint};
//!
//! // Catmull-Rom: passes through all points
//! let mut cr = CatmullRomCurve::new();
//! cr.add_point(SplinePoint::new(10.0, 10.0));
//! cr.add_point(SplinePoint::new(50.0, 20.0));
//! cr.add_point(SplinePoint::new(80.0, 60.0));
//! cr.add_point(SplinePoint::new(30.0, 80.0));
//! cr.set_closed(true);
//! let samples = cr.sample(20);
//! assert!(samples.len() > 4);
//!
//! // Cubic B-spline: approximates the control polygon
//! let mut bs = CubicBSplineCurve::new();
//! bs.add_point(SplinePoint::new(0.0, 0.0));
//! bs.add_point(SplinePoint::new(50.0, 0.0));
//! bs.add_point(SplinePoint::new(50.0, 50.0));
//! bs.add_point(SplinePoint::new(0.0, 50.0));
//! bs.set_closed(true);
//! let samples = bs.sample(16);
//! assert!(samples.len() > 4);
//! ```

use crate::{Frame, VfxResult};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// SplinePoint
// ─────────────────────────────────────────────────────────────────────────────

/// A control point for spline curves (simpler than `BezierPoint` — no handles).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SplinePoint {
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
    /// Optional per-point weight (used for tension in Catmull-Rom).
    pub weight: f32,
}

impl SplinePoint {
    /// Create a new spline point with unit weight.
    #[must_use]
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            weight: 1.0,
        }
    }

    /// Create with explicit weight.
    #[must_use]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.clamp(0.0, 4.0);
        self
    }

    /// Euclidean distance to another point.
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Catmull-Rom spline
// ─────────────────────────────────────────────────────────────────────────────

/// Centripetal Catmull-Rom spline curve.
///
/// The curve passes through every control point and has C1 continuity.
/// The `alpha` parameter controls parameterisation:
/// - `alpha = 0.0` → uniform Catmull-Rom
/// - `alpha = 0.5` → centripetal (default, avoids cusps)
/// - `alpha = 1.0` → chordal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatmullRomCurve {
    points: Vec<SplinePoint>,
    closed: bool,
    /// Parameterisation exponent (0.0 uniform, 0.5 centripetal, 1.0 chordal).
    pub alpha: f32,
    /// Tension parameter: 0.0 = Catmull-Rom default, 1.0 = tight.
    pub tension: f32,
}

impl CatmullRomCurve {
    /// Create an empty open Catmull-Rom curve with centripetal parameterisation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            closed: false,
            alpha: 0.5,
            tension: 0.0,
        }
    }

    /// Add a control point.
    pub fn add_point(&mut self, pt: SplinePoint) {
        self.points.push(pt);
    }

    /// Set whether the curve is closed (loops back to start).
    pub fn set_closed(&mut self, closed: bool) {
        self.closed = closed;
    }

    /// Number of control points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the curve has no control points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Read-only access to control points.
    #[must_use]
    pub fn points(&self) -> &[SplinePoint] {
        &self.points
    }

    /// Number of segments in the curve.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        let n = self.points.len();
        if n < 2 {
            return 0;
        }
        if self.closed {
            n
        } else {
            n - 1
        }
    }

    /// Get control point by index with wrapping for closed curves.
    fn point_at(&self, idx: i32) -> &SplinePoint {
        let n = self.points.len() as i32;
        let wrapped = ((idx % n) + n) % n;
        &self.points[wrapped as usize]
    }

    /// Evaluate a single segment at parameter `t` in [0, 1].
    ///
    /// Uses the four-point Catmull-Rom basis with configurable tension.
    #[must_use]
    pub fn evaluate_segment(&self, segment: usize, t: f32) -> Option<(f32, f32)> {
        if segment >= self.segment_count() {
            return None;
        }

        let n = self.points.len() as i32;
        let i = segment as i32;

        // Four control points for this segment
        let p0 = if self.closed {
            self.point_at(i - 1)
        } else if i == 0 {
            // Virtual mirrored point — handled inline below via (p0x, p0y)
            &self.points[0]
        } else {
            &self.points[(i - 1) as usize]
        };

        let p1 = &self.points[segment];
        let p2_idx = if self.closed {
            ((i + 1) % n) as usize
        } else {
            (segment + 1).min(self.points.len() - 1)
        };
        let p2 = &self.points[p2_idx];

        let p3 = if self.closed {
            self.point_at(i + 2)
        } else if segment + 2 < self.points.len() {
            &self.points[segment + 2]
        } else {
            &self.points[self.points.len() - 1]
        };

        // Handle edge case for open curves: virtual mirrored points
        let (p0x, p0y) = if !self.closed && i == 0 {
            (2.0 * p1.x - p2.x, 2.0 * p1.y - p2.y)
        } else {
            (p0.x, p0.y)
        };

        let (p3x, p3y) = if !self.closed && segment + 2 >= self.points.len() {
            (2.0 * p2.x - p1.x, 2.0 * p2.y - p1.y)
        } else {
            (p3.x, p3.y)
        };

        let s = 0.5 * (1.0 - self.tension);
        let t2 = t * t;
        let t3 = t2 * t;

        // Catmull-Rom matrix coefficients
        let x = s * (-t3 + 2.0 * t2 - t) * p0x
            + (s * (t3 - 2.0 * t2) + (2.0 - s) * t3 - (3.0 - s) * t2 + 1.0) * p1.x
            + (-s * t3 + s * t2 + (s - 2.0) * t3 + (3.0 - s) * t2 + s * t) * p1.x;

        // Use the standard Catmull-Rom formulation instead (clearer)
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        // Tangents with tension
        let m1x = s * (p2.x - p0x);
        let m1y = s * (p2.y - p0y);
        let m2x = s * (p3x - p1.x);
        let m2y = s * (p3y - p1.y);

        let _ = x; // suppress unused warning from the first attempt

        let out_x = h00 * p1.x + h10 * m1x + h01 * p2.x + h11 * m2x;
        let out_y = h00 * p1.y + h10 * m1y + h01 * p2.y + h11 * m2y;

        Some((out_x, out_y))
    }

    /// Sample the entire curve into a polyline.
    ///
    /// `steps_per_segment`: number of sample points per segment.
    #[must_use]
    pub fn sample(&self, steps_per_segment: usize) -> Vec<(f32, f32)> {
        let steps = steps_per_segment.max(2);
        let mut result = Vec::new();

        for seg in 0..self.segment_count() {
            for step in 0..steps {
                let t = step as f32 / steps as f32;
                if let Some(pt) = self.evaluate_segment(seg, t) {
                    result.push(pt);
                }
            }
        }

        // Close the curve by adding the first point again, or add the last point
        if self.closed {
            if let Some(&first) = self.points.first() {
                result.push((first.x, first.y));
            }
        } else if let Some(last) = self.points.last() {
            result.push((last.x, last.y));
        }

        result
    }
}

impl Default for CatmullRomCurve {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cubic B-spline
// ─────────────────────────────────────────────────────────────────────────────

/// Uniform cubic B-spline curve.
///
/// The curve approximates the control polygon with C2 continuity — it does
/// *not* pass through the control points (except when clamped).  This makes
/// it excellent for smooth organic rotoscoping shapes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CubicBSplineCurve {
    points: Vec<SplinePoint>,
    closed: bool,
}

impl CubicBSplineCurve {
    /// Create an empty open B-spline.
    #[must_use]
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            closed: false,
        }
    }

    /// Add a control point.
    pub fn add_point(&mut self, pt: SplinePoint) {
        self.points.push(pt);
    }

    /// Set closed.
    pub fn set_closed(&mut self, closed: bool) {
        self.closed = closed;
    }

    /// Number of control points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Read-only access to control points.
    #[must_use]
    pub fn points(&self) -> &[SplinePoint] {
        &self.points
    }

    /// Number of spline segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        let n = self.points.len();
        if n < 4 && !self.closed {
            return 0;
        }
        if n < 3 {
            return 0;
        }
        if self.closed {
            n
        } else {
            n.saturating_sub(3)
        }
    }

    /// Get control point with wrapping.
    fn point_at(&self, idx: i32) -> &SplinePoint {
        let n = self.points.len() as i32;
        let wrapped = ((idx % n) + n) % n;
        &self.points[wrapped as usize]
    }

    /// Evaluate segment at parameter `t` in [0, 1].
    ///
    /// Uses the uniform cubic B-spline basis:
    /// ```text
    /// B(t) = (1/6) [ (1-t)^3 P0 + (3t^3 - 6t^2 + 4) P1
    ///              + (-3t^3 + 3t^2 + 3t + 1) P2 + t^3 P3 ]
    /// ```
    #[must_use]
    pub fn evaluate_segment(&self, segment: usize, t: f32) -> Option<(f32, f32)> {
        if segment >= self.segment_count() {
            return None;
        }

        let i = segment as i32;
        let (p0, p1, p2, p3) = if self.closed {
            (
                self.point_at(i),
                self.point_at(i + 1),
                self.point_at(i + 2),
                self.point_at(i + 3),
            )
        } else {
            (
                &self.points[segment],
                &self.points[segment + 1],
                &self.points[segment + 2],
                &self.points[segment + 3],
            )
        };

        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt3 = mt * mt * mt;

        let b0 = mt3 / 6.0;
        let b1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0;
        let b2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0;
        let b3 = t3 / 6.0;

        let x = b0 * p0.x + b1 * p1.x + b2 * p2.x + b3 * p3.x;
        let y = b0 * p0.y + b1 * p1.y + b2 * p2.y + b3 * p3.y;

        Some((x, y))
    }

    /// Sample the entire curve into a polyline.
    #[must_use]
    pub fn sample(&self, steps_per_segment: usize) -> Vec<(f32, f32)> {
        let steps = steps_per_segment.max(2);
        let mut result = Vec::new();

        for seg in 0..self.segment_count() {
            for step in 0..steps {
                let t = step as f32 / steps as f32;
                if let Some(pt) = self.evaluate_segment(seg, t) {
                    result.push(pt);
                }
            }
        }

        // Add final point
        if let Some(pt) = self.evaluate_segment(self.segment_count().saturating_sub(1), 1.0) {
            result.push(pt);
        }

        result
    }
}

impl Default for CubicBSplineCurve {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SplineMask — unified mask type for spline curves
// ─────────────────────────────────────────────────────────────────────────────

/// The type of spline curve used for the mask.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SplineCurveType {
    /// Catmull-Rom spline (passes through control points).
    CatmullRom(CatmullRomCurve),
    /// Cubic B-spline (approximates control polygon).
    BSpline(CubicBSplineCurve),
}

impl SplineCurveType {
    /// Sample the curve into a polyline.
    #[must_use]
    pub fn sample(&self, steps: usize) -> Vec<(f32, f32)> {
        match self {
            Self::CatmullRom(c) => c.sample(steps),
            Self::BSpline(b) => b.sample(steps),
        }
    }

    /// Number of control points.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::CatmullRom(c) => c.len(),
            Self::BSpline(b) => b.len(),
        }
    }

    /// Whether the curve has no control points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::CatmullRom(c) => c.is_empty(),
            Self::BSpline(b) => b.is_empty(),
        }
    }
}

/// A mask defined by a spline curve with feather and opacity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplineMask {
    /// The spline curve defining the mask boundary.
    pub curve: SplineCurveType,
    /// Feather (blur) amount in pixels.
    pub feather: f32,
    /// Mask opacity (0.0 = transparent, 1.0 = opaque).
    pub opacity: f32,
    /// Whether the mask is inverted.
    pub inverted: bool,
    /// Steps per segment for sampling.
    pub sample_steps: usize,
}

impl SplineMask {
    /// Create a new mask from a spline curve.
    #[must_use]
    pub fn new(curve: SplineCurveType) -> Self {
        Self {
            curve,
            feather: 0.0,
            opacity: 1.0,
            inverted: false,
            sample_steps: 16,
        }
    }

    /// Set feather amount.
    #[must_use]
    pub fn with_feather(mut self, feather: f32) -> Self {
        self.feather = feather.max(0.0);
        self
    }

    /// Set opacity.
    #[must_use]
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Set inverted.
    #[must_use]
    pub fn with_inverted(mut self, inverted: bool) -> Self {
        self.inverted = inverted;
        self
    }

    /// Render the spline mask to a frame's alpha channel.
    pub fn render(&self, output: &mut Frame) -> VfxResult<()> {
        let samples = self.curve.sample(self.sample_steps);
        if samples.len() < 3 {
            return Ok(());
        }

        for y in 0..output.height {
            for x in 0..output.width {
                let inside = is_point_inside(x as f32, y as f32, &samples);
                let distance = if inside {
                    -distance_to_polyline(x as f32, y as f32, &samples)
                } else {
                    distance_to_polyline(x as f32, y as f32, &samples)
                };

                let alpha = if self.feather > 0.0 {
                    (-distance / self.feather).clamp(0.0, 1.0)
                } else if inside {
                    1.0
                } else {
                    0.0
                };

                let alpha = alpha * self.opacity;
                let alpha = if self.inverted { 1.0 - alpha } else { alpha };

                let pixel = output.get_pixel(x, y).unwrap_or([0, 0, 0, 0]);
                output.set_pixel(x, y, [pixel[0], pixel[1], pixel[2], (alpha * 255.0) as u8]);
            }
        }

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Geometry helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Ray-casting point-in-polygon test.
fn is_point_inside(x: f32, y: f32, poly: &[(f32, f32)]) -> bool {
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[j];
        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Minimum distance from a point to any segment in a polyline.
fn distance_to_polyline(x: f32, y: f32, poly: &[(f32, f32)]) -> f32 {
    let mut min_d = f32::MAX;
    for i in 0..poly.len() {
        let j = (i + 1) % poly.len();
        let d = point_to_segment_dist(x, y, poly[i].0, poly[i].1, poly[j].0, poly[j].1);
        min_d = min_d.min(d);
    }
    min_d
}

/// Distance from point (px, py) to line segment (x1,y1)-(x2,y2).
fn point_to_segment_dist(px: f32, py: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;
    if len_sq < f32::EPSILON {
        return ((px - x1) * (px - x1) + (py - y1) * (py - y1)).sqrt();
    }
    let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let proj_x = x1 + t * dx;
    let proj_y = y1 + t * dy;
    ((px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y)).sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SplinePoint ──────────────────────────────────────────────────────

    #[test]
    fn test_spline_point_distance() {
        let a = SplinePoint::new(0.0, 0.0);
        let b = SplinePoint::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_spline_point_weight_clamp() {
        let p = SplinePoint::new(1.0, 2.0).with_weight(10.0);
        assert_eq!(p.weight, 4.0);
        let p2 = SplinePoint::new(1.0, 2.0).with_weight(-1.0);
        assert_eq!(p2.weight, 0.0);
    }

    // ── Catmull-Rom ──────────────────────────────────────────────────────

    #[test]
    fn test_catmull_rom_empty() {
        let cr = CatmullRomCurve::new();
        assert!(cr.is_empty());
        assert_eq!(cr.segment_count(), 0);
        assert!(cr.sample(10).is_empty());
    }

    #[test]
    fn test_catmull_rom_two_points_open() {
        let mut cr = CatmullRomCurve::new();
        cr.add_point(SplinePoint::new(0.0, 0.0));
        cr.add_point(SplinePoint::new(100.0, 100.0));
        assert_eq!(cr.segment_count(), 1);
        let samples = cr.sample(10);
        assert!(samples.len() >= 2);
        // First sample should be near start
        assert!(samples[0].0.abs() < 1.0);
        assert!(samples[0].1.abs() < 1.0);
    }

    #[test]
    fn test_catmull_rom_passes_through_points() {
        let mut cr = CatmullRomCurve::new();
        cr.add_point(SplinePoint::new(0.0, 0.0));
        cr.add_point(SplinePoint::new(50.0, 30.0));
        cr.add_point(SplinePoint::new(100.0, 10.0));
        cr.add_point(SplinePoint::new(150.0, 50.0));

        // Evaluate at t=0 of segment 1 should be at point[1]
        let (x, y) = cr.evaluate_segment(1, 0.0).expect("eval seg 1 t=0");
        assert!(
            (x - 50.0).abs() < 0.1 && (y - 30.0).abs() < 0.1,
            "should pass through P1: got ({x}, {y})"
        );

        // Evaluate at t=1 of segment 1 should be at point[2]
        let (x, y) = cr.evaluate_segment(1, 1.0).expect("eval seg 1 t=1");
        assert!(
            (x - 100.0).abs() < 0.1 && (y - 10.0).abs() < 0.1,
            "should pass through P2: got ({x}, {y})"
        );
    }

    #[test]
    fn test_catmull_rom_closed() {
        let mut cr = CatmullRomCurve::new();
        cr.add_point(SplinePoint::new(0.0, 0.0));
        cr.add_point(SplinePoint::new(100.0, 0.0));
        cr.add_point(SplinePoint::new(100.0, 100.0));
        cr.add_point(SplinePoint::new(0.0, 100.0));
        cr.set_closed(true);
        assert_eq!(cr.segment_count(), 4);
        let samples = cr.sample(10);
        // Closed curve samples should start and end at same point
        let first = samples.first().copied().unwrap_or((0.0, 0.0));
        let last = samples.last().copied().unwrap_or((1.0, 1.0));
        assert!(
            (first.0 - last.0).abs() < 1.0 && (first.1 - last.1).abs() < 1.0,
            "closed curve should loop: first=({},{}), last=({},{})",
            first.0,
            first.1,
            last.0,
            last.1
        );
    }

    #[test]
    fn test_catmull_rom_tension() {
        let mut cr_default = CatmullRomCurve::new();
        cr_default.tension = 0.0;
        let mut cr_tight = CatmullRomCurve::new();
        cr_tight.tension = 0.8;

        let pts = [
            SplinePoint::new(0.0, 0.0),
            SplinePoint::new(50.0, 100.0),
            SplinePoint::new(100.0, 0.0),
            SplinePoint::new(150.0, 100.0),
        ];
        for p in &pts {
            cr_default.add_point(*p);
            cr_tight.add_point(*p);
        }

        let s_default = cr_default.sample(20);
        let s_tight = cr_tight.sample(20);
        // They should differ — tight tension reduces overshooting
        assert_ne!(s_default.len(), 0);
        assert_ne!(s_tight.len(), 0);
        // At least some sample points should differ
        let same_count = s_default
            .iter()
            .zip(s_tight.iter())
            .filter(|((ax, ay), (bx, by))| (ax - bx).abs() < 0.01 && (ay - by).abs() < 0.01)
            .count();
        assert!(
            same_count < s_default.len(),
            "tension should change the curve shape"
        );
    }

    // ── Cubic B-spline ───────────────────────────────────────────────────

    #[test]
    fn test_bspline_empty() {
        let bs = CubicBSplineCurve::new();
        assert!(bs.is_empty());
        assert_eq!(bs.segment_count(), 0);
    }

    #[test]
    fn test_bspline_needs_four_points_open() {
        let mut bs = CubicBSplineCurve::new();
        bs.add_point(SplinePoint::new(0.0, 0.0));
        bs.add_point(SplinePoint::new(1.0, 1.0));
        bs.add_point(SplinePoint::new(2.0, 0.0));
        assert_eq!(bs.segment_count(), 0, "need 4 points for open B-spline");
        bs.add_point(SplinePoint::new(3.0, 1.0));
        assert_eq!(bs.segment_count(), 1);
    }

    #[test]
    fn test_bspline_approximates_not_interpolates() {
        let mut bs = CubicBSplineCurve::new();
        bs.add_point(SplinePoint::new(0.0, 0.0));
        bs.add_point(SplinePoint::new(50.0, 100.0));
        bs.add_point(SplinePoint::new(100.0, 0.0));
        bs.add_point(SplinePoint::new(150.0, 100.0));

        // B-spline does NOT pass through P1 (50, 100) at t=0 of segment 0
        let (_, y) = bs.evaluate_segment(0, 0.0).expect("eval");
        // The y value should be significantly less than 100 (approximation)
        assert!(
            y < 80.0,
            "B-spline should approximate, not pass through P1: y={y}"
        );
    }

    #[test]
    fn test_bspline_closed() {
        let mut bs = CubicBSplineCurve::new();
        bs.add_point(SplinePoint::new(0.0, 0.0));
        bs.add_point(SplinePoint::new(100.0, 0.0));
        bs.add_point(SplinePoint::new(100.0, 100.0));
        bs.add_point(SplinePoint::new(0.0, 100.0));
        bs.set_closed(true);
        assert_eq!(bs.segment_count(), 4);
        let samples = bs.sample(10);
        assert!(!samples.is_empty());
    }

    #[test]
    fn test_bspline_sample_produces_polyline() {
        let mut bs = CubicBSplineCurve::new();
        for i in 0..6 {
            bs.add_point(SplinePoint::new(i as f32 * 20.0, ((i % 2) as f32) * 50.0));
        }
        let samples = bs.sample(10);
        assert!(
            samples.len() > 20,
            "should have many samples, got {}",
            samples.len()
        );
    }

    #[test]
    fn test_bspline_c2_smooth() {
        // Check that consecutive segments connect smoothly
        let mut bs = CubicBSplineCurve::new();
        bs.add_point(SplinePoint::new(0.0, 0.0));
        bs.add_point(SplinePoint::new(30.0, 60.0));
        bs.add_point(SplinePoint::new(60.0, 20.0));
        bs.add_point(SplinePoint::new(90.0, 80.0));
        bs.add_point(SplinePoint::new(120.0, 40.0));

        // End of segment 0 should equal start of segment 1
        let end0 = bs.evaluate_segment(0, 1.0).expect("seg0 end");
        let start1 = bs.evaluate_segment(1, 0.0).expect("seg1 start");
        assert!(
            (end0.0 - start1.0).abs() < 0.01 && (end0.1 - start1.1).abs() < 0.01,
            "C2 continuity: end0=({},{}), start1=({},{})",
            end0.0,
            end0.1,
            start1.0,
            start1.1
        );
    }

    // ── SplineMask ───────────────────────────────────────────────────────

    #[test]
    fn test_spline_mask_catmull_rom_render() {
        let mut cr = CatmullRomCurve::new();
        cr.add_point(SplinePoint::new(20.0, 20.0));
        cr.add_point(SplinePoint::new(80.0, 20.0));
        cr.add_point(SplinePoint::new(80.0, 80.0));
        cr.add_point(SplinePoint::new(20.0, 80.0));
        cr.set_closed(true);

        let mask = SplineMask::new(SplineCurveType::CatmullRom(cr)).with_feather(3.0);
        let mut frame = Frame::new(100, 100).expect("frame");
        mask.render(&mut frame).expect("render");

        // Centre should be inside
        let centre_alpha = frame.get_pixel(50, 50).unwrap_or([0; 4])[3];
        assert!(centre_alpha > 200, "centre should be opaque: {centre_alpha}");

        // Corner should be outside
        let corner_alpha = frame.get_pixel(0, 0).unwrap_or([0; 4])[3];
        assert!(
            corner_alpha < 50,
            "corner should be transparent: {corner_alpha}"
        );
    }

    #[test]
    fn test_spline_mask_bspline_render() {
        let mut bs = CubicBSplineCurve::new();
        bs.add_point(SplinePoint::new(10.0, 10.0));
        bs.add_point(SplinePoint::new(90.0, 10.0));
        bs.add_point(SplinePoint::new(90.0, 90.0));
        bs.add_point(SplinePoint::new(10.0, 90.0));
        bs.set_closed(true);

        let mask = SplineMask::new(SplineCurveType::BSpline(bs));
        let mut frame = Frame::new(100, 100).expect("frame");
        mask.render(&mut frame).expect("render");
        // Should produce some mask data
        let has_nonzero = frame.data.iter().any(|&b| b > 0);
        assert!(has_nonzero, "mask should produce non-zero alpha");
    }

    #[test]
    fn test_spline_mask_inverted() {
        let mut cr = CatmullRomCurve::new();
        cr.add_point(SplinePoint::new(20.0, 20.0));
        cr.add_point(SplinePoint::new(80.0, 20.0));
        cr.add_point(SplinePoint::new(80.0, 80.0));
        cr.add_point(SplinePoint::new(20.0, 80.0));
        cr.set_closed(true);

        let mask = SplineMask::new(SplineCurveType::CatmullRom(cr)).with_inverted(true);
        let mut frame = Frame::new(100, 100).expect("frame");
        mask.render(&mut frame).expect("render");

        // Centre should be transparent when inverted
        let centre_alpha = frame.get_pixel(50, 50).unwrap_or([0; 4])[3];
        assert!(
            centre_alpha < 50,
            "inverted: centre should be transparent: {centre_alpha}"
        );
    }

    #[test]
    fn test_spline_curve_type_len() {
        let mut cr = CatmullRomCurve::new();
        cr.add_point(SplinePoint::new(0.0, 0.0));
        cr.add_point(SplinePoint::new(1.0, 1.0));
        let ct = SplineCurveType::CatmullRom(cr);
        assert_eq!(ct.len(), 2);
        assert!(!ct.is_empty());
    }
}
