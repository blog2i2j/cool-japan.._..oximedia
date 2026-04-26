#![allow(dead_code)]
//! Bezier curve path construction and evaluation for broadcast graphics.
//!
//! Provides cubic and quadratic Bezier curve primitives, path building utilities,
//! and evaluation functions for constructing smooth motion paths, text-on-path
//! layouts, and decorative border elements in broadcast overlays.

use std::fmt;

/// A 2D point used in Bezier path operations.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Point2D {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
}

impl Point2D {
    /// Create a new point.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Compute the Euclidean distance to another point.
    #[allow(clippy::cast_precision_loss)]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Linear interpolation between two points.
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }
}

impl fmt::Display for Point2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2})", self.x, self.y)
    }
}

/// A quadratic Bezier segment (start, control, end).
#[derive(Clone, Debug, PartialEq)]
pub struct QuadraticBezier {
    /// Start point.
    pub p0: Point2D,
    /// Control point.
    pub p1: Point2D,
    /// End point.
    pub p2: Point2D,
}

impl QuadraticBezier {
    /// Create a new quadratic Bezier segment.
    pub fn new(p0: Point2D, p1: Point2D, p2: Point2D) -> Self {
        Self { p0, p1, p2 }
    }

    /// Evaluate the curve at parameter `t` in [0, 1].
    pub fn evaluate(&self, t: f64) -> Point2D {
        let t_clamped = t.clamp(0.0, 1.0);
        let inv = 1.0 - t_clamped;
        let inv2 = inv * inv;
        let t2 = t_clamped * t_clamped;
        Point2D {
            x: inv2 * self.p0.x + 2.0 * inv * t_clamped * self.p1.x + t2 * self.p2.x,
            y: inv2 * self.p0.y + 2.0 * inv * t_clamped * self.p1.y + t2 * self.p2.y,
        }
    }

    /// Compute the tangent direction at parameter `t`.
    pub fn tangent(&self, t: f64) -> Point2D {
        let t_clamped = t.clamp(0.0, 1.0);
        let inv = 1.0 - t_clamped;
        Point2D {
            x: 2.0 * inv * (self.p1.x - self.p0.x) + 2.0 * t_clamped * (self.p2.x - self.p1.x),
            y: 2.0 * inv * (self.p1.y - self.p0.y) + 2.0 * t_clamped * (self.p2.y - self.p1.y),
        }
    }

    /// Approximate the arc length by sampling `n` segments.
    #[allow(clippy::cast_precision_loss)]
    pub fn approximate_length(&self, n: usize) -> f64 {
        let steps = n.max(1);
        let mut length = 0.0;
        let mut prev = self.evaluate(0.0);
        for i in 1..=steps {
            let t = i as f64 / steps as f64;
            let curr = self.evaluate(t);
            length += prev.distance_to(&curr);
            prev = curr;
        }
        length
    }
}

/// A cubic Bezier segment (start, control1, control2, end).
#[derive(Clone, Debug, PartialEq)]
pub struct CubicBezier {
    /// Start point.
    pub p0: Point2D,
    /// First control point.
    pub p1: Point2D,
    /// Second control point.
    pub p2: Point2D,
    /// End point.
    pub p3: Point2D,
}

impl CubicBezier {
    /// Create a new cubic Bezier segment.
    pub fn new(p0: Point2D, p1: Point2D, p2: Point2D, p3: Point2D) -> Self {
        Self { p0, p1, p2, p3 }
    }

    /// Evaluate the curve at parameter `t` in [0, 1].
    pub fn evaluate(&self, t: f64) -> Point2D {
        let t_clamped = t.clamp(0.0, 1.0);
        let inv = 1.0 - t_clamped;
        let inv2 = inv * inv;
        let inv3 = inv2 * inv;
        let t2 = t_clamped * t_clamped;
        let t3 = t2 * t_clamped;
        Point2D {
            x: inv3 * self.p0.x
                + 3.0 * inv2 * t_clamped * self.p1.x
                + 3.0 * inv * t2 * self.p2.x
                + t3 * self.p3.x,
            y: inv3 * self.p0.y
                + 3.0 * inv2 * t_clamped * self.p1.y
                + 3.0 * inv * t2 * self.p2.y
                + t3 * self.p3.y,
        }
    }

    /// Compute the tangent direction at parameter `t`.
    pub fn tangent(&self, t: f64) -> Point2D {
        let t_clamped = t.clamp(0.0, 1.0);
        let inv = 1.0 - t_clamped;
        let inv2 = inv * inv;
        let t2 = t_clamped * t_clamped;
        Point2D {
            x: 3.0 * inv2 * (self.p1.x - self.p0.x)
                + 6.0 * inv * t_clamped * (self.p2.x - self.p1.x)
                + 3.0 * t2 * (self.p3.x - self.p2.x),
            y: 3.0 * inv2 * (self.p1.y - self.p0.y)
                + 6.0 * inv * t_clamped * (self.p2.y - self.p1.y)
                + 3.0 * t2 * (self.p3.y - self.p2.y),
        }
    }

    /// Approximate the arc length by sampling `n` segments.
    #[allow(clippy::cast_precision_loss)]
    pub fn approximate_length(&self, n: usize) -> f64 {
        let steps = n.max(1);
        let mut length = 0.0;
        let mut prev = self.evaluate(0.0);
        for i in 1..=steps {
            let t = i as f64 / steps as f64;
            let curr = self.evaluate(t);
            length += prev.distance_to(&curr);
            prev = curr;
        }
        length
    }

    /// Split the curve at parameter `t` into two sub-curves using de Casteljau.
    pub fn split_at(&self, t: f64) -> (CubicBezier, CubicBezier) {
        let t_clamped = t.clamp(0.0, 1.0);
        let a = self.p0.lerp(&self.p1, t_clamped);
        let b = self.p1.lerp(&self.p2, t_clamped);
        let c = self.p2.lerp(&self.p3, t_clamped);
        let d = a.lerp(&b, t_clamped);
        let e = b.lerp(&c, t_clamped);
        let f = d.lerp(&e, t_clamped);
        (
            CubicBezier::new(self.p0, a, d, f),
            CubicBezier::new(f, e, c, self.p3),
        )
    }

    /// Compute the axis-aligned bounding box of the curve.
    pub fn bounding_box(&self) -> BoundingBox {
        let mut min_x = self.p0.x.min(self.p3.x);
        let mut max_x = self.p0.x.max(self.p3.x);
        let mut min_y = self.p0.y.min(self.p3.y);
        let mut max_y = self.p0.y.max(self.p3.y);
        // Sample internal points
        for i in 1..=20 {
            let t = i as f64 / 20.0;
            let p = self.evaluate(t);
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }
        BoundingBox {
            min: Point2D::new(min_x, min_y),
            max: Point2D::new(max_x, max_y),
        }
    }
}

/// Axis-aligned bounding box.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundingBox {
    /// Minimum corner.
    pub min: Point2D,
    /// Maximum corner.
    pub max: Point2D,
}

impl BoundingBox {
    /// Width of the bounding box.
    pub fn width(&self) -> f64 {
        (self.max.x - self.min.x).abs()
    }

    /// Height of the bounding box.
    pub fn height(&self) -> f64 {
        (self.max.y - self.min.y).abs()
    }

    /// Center point.
    pub fn center(&self) -> Point2D {
        Point2D::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
        )
    }

    /// Check if a point is inside the bounding box.
    pub fn contains(&self, p: &Point2D) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }
}

/// A segment of a Bezier path.
#[derive(Clone, Debug, PartialEq)]
pub enum PathSegment {
    /// Move to a point (pen up).
    MoveTo(Point2D),
    /// Line to a point.
    LineTo(Point2D),
    /// Quadratic Bezier to a point.
    QuadTo {
        /// Control point.
        control: Point2D,
        /// End point.
        end: Point2D,
    },
    /// Cubic Bezier to a point.
    CubicTo {
        /// First control point.
        control1: Point2D,
        /// Second control point.
        control2: Point2D,
        /// End point.
        end: Point2D,
    },
    /// Close the current sub-path.
    Close,
}

/// A composite Bezier path composed of multiple segments.
#[derive(Clone, Debug, Default)]
pub struct BezierPath {
    /// All segments in this path.
    segments: Vec<PathSegment>,
    /// Current pen position.
    current: Point2D,
}

impl BezierPath {
    /// Create a new empty path.
    pub fn new() -> Self {
        Self::default()
    }

    /// Move the pen to a position without drawing.
    pub fn move_to(&mut self, x: f64, y: f64) -> &mut Self {
        let p = Point2D::new(x, y);
        self.segments.push(PathSegment::MoveTo(p));
        self.current = p;
        self
    }

    /// Draw a straight line to a position.
    pub fn line_to(&mut self, x: f64, y: f64) -> &mut Self {
        let p = Point2D::new(x, y);
        self.segments.push(PathSegment::LineTo(p));
        self.current = p;
        self
    }

    /// Draw a quadratic Bezier curve.
    pub fn quad_to(&mut self, cx: f64, cy: f64, x: f64, y: f64) -> &mut Self {
        let end = Point2D::new(x, y);
        self.segments.push(PathSegment::QuadTo {
            control: Point2D::new(cx, cy),
            end,
        });
        self.current = end;
        self
    }

    /// Draw a cubic Bezier curve.
    pub fn cubic_to(
        &mut self,
        c1x: f64,
        c1y: f64,
        c2x: f64,
        c2y: f64,
        x: f64,
        y: f64,
    ) -> &mut Self {
        let end = Point2D::new(x, y);
        self.segments.push(PathSegment::CubicTo {
            control1: Point2D::new(c1x, c1y),
            control2: Point2D::new(c2x, c2y),
            end,
        });
        self.current = end;
        self
    }

    /// Close the current sub-path.
    pub fn close(&mut self) -> &mut Self {
        self.segments.push(PathSegment::Close);
        self
    }

    /// Get the number of segments in this path.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Get all segments.
    pub fn segments(&self) -> &[PathSegment] {
        &self.segments
    }

    /// Check if the path is empty.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Approximate the total arc length of the path.
    #[allow(clippy::cast_precision_loss)]
    pub fn approximate_length(&self, samples_per_segment: usize) -> f64 {
        let mut total = 0.0;
        let mut pen = Point2D::default();
        for seg in &self.segments {
            match seg {
                PathSegment::MoveTo(p) => pen = *p,
                PathSegment::LineTo(p) => {
                    total += pen.distance_to(p);
                    pen = *p;
                }
                PathSegment::QuadTo { control, end } => {
                    let q = QuadraticBezier::new(pen, *control, *end);
                    total += q.approximate_length(samples_per_segment);
                    pen = *end;
                }
                PathSegment::CubicTo {
                    control1,
                    control2,
                    end,
                } => {
                    let c = CubicBezier::new(pen, *control1, *control2, *end);
                    total += c.approximate_length(samples_per_segment);
                    pen = *end;
                }
                PathSegment::Close => {}
            }
        }
        total
    }

    /// Sample points along the path at equal arc-length intervals.
    #[allow(clippy::cast_precision_loss)]
    pub fn sample_uniform(&self, count: usize) -> Vec<Point2D> {
        if count == 0 || self.is_empty() {
            return Vec::new();
        }
        // Flatten to dense polyline first
        let polyline = self.flatten(64);
        if polyline.len() < 2 {
            return polyline;
        }
        // Compute cumulative arc lengths
        let mut cum_lengths = Vec::with_capacity(polyline.len());
        cum_lengths.push(0.0);
        for i in 1..polyline.len() {
            let prev_len = cum_lengths[i - 1];
            cum_lengths.push(prev_len + polyline[i - 1].distance_to(&polyline[i]));
        }
        let total = *cum_lengths.last().unwrap_or(&0.0);
        if total <= 0.0 {
            return vec![polyline[0]];
        }
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let target = if count == 1 {
                0.0
            } else {
                total * i as f64 / (count - 1) as f64
            };
            // Find the segment that contains this arc length
            let idx = match cum_lengths
                .binary_search_by(|v| v.partial_cmp(&target).unwrap_or(std::cmp::Ordering::Equal))
            {
                Ok(i) => i,
                Err(i) => i.saturating_sub(1),
            };
            if idx >= polyline.len() - 1 {
                if let Some(last) = polyline.last() {
                    result.push(*last);
                }
            } else {
                let seg_len = cum_lengths[idx + 1] - cum_lengths[idx];
                let t = if seg_len > 0.0 {
                    (target - cum_lengths[idx]) / seg_len
                } else {
                    0.0
                };
                result.push(polyline[idx].lerp(&polyline[idx + 1], t));
            }
        }
        result
    }

    /// Flatten the path into a polyline with `steps_per_curve` for each curve segment.
    #[allow(clippy::cast_precision_loss)]
    fn flatten(&self, steps_per_curve: usize) -> Vec<Point2D> {
        let mut points = Vec::new();
        let mut pen = Point2D::default();
        for seg in &self.segments {
            match seg {
                PathSegment::MoveTo(p) => {
                    pen = *p;
                    points.push(pen);
                }
                PathSegment::LineTo(p) => {
                    points.push(*p);
                    pen = *p;
                }
                PathSegment::QuadTo { control, end } => {
                    let q = QuadraticBezier::new(pen, *control, *end);
                    for i in 1..=steps_per_curve {
                        let t = i as f64 / steps_per_curve as f64;
                        points.push(q.evaluate(t));
                    }
                    pen = *end;
                }
                PathSegment::CubicTo {
                    control1,
                    control2,
                    end,
                } => {
                    let c = CubicBezier::new(pen, *control1, *control2, *end);
                    for i in 1..=steps_per_curve {
                        let t = i as f64 / steps_per_curve as f64;
                        points.push(c.evaluate(t));
                    }
                    pen = *end;
                }
                PathSegment::Close => {}
            }
        }
        points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_new() {
        let p = Point2D::new(3.0, 4.0);
        assert!((p.x - 3.0).abs() < 1e-12);
        assert!((p.y - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_point_distance() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_point_lerp() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(10.0, 20.0);
        let mid = a.lerp(&b, 0.5);
        assert!((mid.x - 5.0).abs() < 1e-12);
        assert!((mid.y - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_point_display() {
        let p = Point2D::new(1.5, 2.75);
        let s = format!("{p}");
        assert!(s.contains("1.50"));
        assert!(s.contains("2.75"));
    }

    #[test]
    fn test_quadratic_endpoints() {
        let q = QuadraticBezier::new(
            Point2D::new(0.0, 0.0),
            Point2D::new(50.0, 100.0),
            Point2D::new(100.0, 0.0),
        );
        let start = q.evaluate(0.0);
        let end = q.evaluate(1.0);
        assert!((start.x - 0.0).abs() < 1e-12);
        assert!((end.x - 100.0).abs() < 1e-12);
    }

    #[test]
    fn test_quadratic_midpoint() {
        let q = QuadraticBezier::new(
            Point2D::new(0.0, 0.0),
            Point2D::new(50.0, 100.0),
            Point2D::new(100.0, 0.0),
        );
        let mid = q.evaluate(0.5);
        // At t=0.5: x = 0.25*0 + 0.5*50 + 0.25*100 = 50
        assert!((mid.x - 50.0).abs() < 1e-12);
        // At t=0.5: y = 0.25*0 + 0.5*100 + 0.25*0 = 50
        assert!((mid.y - 50.0).abs() < 1e-12);
    }

    #[test]
    fn test_quadratic_tangent() {
        let q = QuadraticBezier::new(
            Point2D::new(0.0, 0.0),
            Point2D::new(50.0, 100.0),
            Point2D::new(100.0, 0.0),
        );
        let tan_start = q.tangent(0.0);
        // tangent at 0: 2*(p1-p0) = (100, 200)
        assert!((tan_start.x - 100.0).abs() < 1e-12);
        assert!((tan_start.y - 200.0).abs() < 1e-12);
    }

    #[test]
    fn test_quadratic_length() {
        // Straight line as degenerate quadratic
        let q = QuadraticBezier::new(
            Point2D::new(0.0, 0.0),
            Point2D::new(50.0, 0.0),
            Point2D::new(100.0, 0.0),
        );
        let len = q.approximate_length(100);
        assert!((len - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_cubic_endpoints() {
        let c = CubicBezier::new(
            Point2D::new(0.0, 0.0),
            Point2D::new(33.0, 100.0),
            Point2D::new(66.0, 100.0),
            Point2D::new(100.0, 0.0),
        );
        let start = c.evaluate(0.0);
        let end = c.evaluate(1.0);
        assert!((start.x - 0.0).abs() < 1e-12);
        assert!((end.x - 100.0).abs() < 1e-12);
    }

    #[test]
    fn test_cubic_split() {
        let c = CubicBezier::new(
            Point2D::new(0.0, 0.0),
            Point2D::new(33.0, 100.0),
            Point2D::new(66.0, 100.0),
            Point2D::new(100.0, 0.0),
        );
        let (left, right) = c.split_at(0.5);
        // Left curve should end at midpoint, right should start there
        let mid = c.evaluate(0.5);
        assert!((left.p3.x - mid.x).abs() < 1e-12);
        assert!((right.p0.x - mid.x).abs() < 1e-12);
    }

    #[test]
    fn test_cubic_bounding_box() {
        let c = CubicBezier::new(
            Point2D::new(0.0, 0.0),
            Point2D::new(0.0, 100.0),
            Point2D::new(100.0, 100.0),
            Point2D::new(100.0, 0.0),
        );
        let bb = c.bounding_box();
        assert!(bb.min.x >= -1.0);
        assert!(bb.max.x <= 101.0);
        assert!(bb.min.y >= -1.0);
        assert!(bb.max.y <= 101.0);
        assert!(bb.width() > 90.0);
    }

    #[test]
    fn test_bounding_box_contains() {
        let bb = BoundingBox {
            min: Point2D::new(0.0, 0.0),
            max: Point2D::new(100.0, 100.0),
        };
        assert!(bb.contains(&Point2D::new(50.0, 50.0)));
        assert!(!bb.contains(&Point2D::new(150.0, 50.0)));
    }

    #[test]
    fn test_path_build_and_length() {
        let mut path = BezierPath::new();
        path.move_to(0.0, 0.0)
            .line_to(100.0, 0.0)
            .line_to(100.0, 100.0)
            .close();
        assert_eq!(path.segment_count(), 4);
        let len = path.approximate_length(32);
        // Two sides of 100 each
        assert!((len - 200.0).abs() < 1.0);
    }

    #[test]
    fn test_path_empty() {
        let path = BezierPath::new();
        assert!(path.is_empty());
        assert_eq!(path.segment_count(), 0);
    }

    #[test]
    fn test_path_sample_uniform() {
        let mut path = BezierPath::new();
        path.move_to(0.0, 0.0).line_to(100.0, 0.0);
        let samples = path.sample_uniform(5);
        assert_eq!(samples.len(), 5);
        // First and last should be near endpoints
        assert!((samples[0].x - 0.0).abs() < 1.0);
        assert!((samples[4].x - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_path_cubic_segment() {
        let mut path = BezierPath::new();
        path.move_to(0.0, 0.0)
            .cubic_to(33.0, 100.0, 66.0, 100.0, 100.0, 0.0);
        assert_eq!(path.segment_count(), 2);
        let len = path.approximate_length(64);
        // Curve should be longer than straight distance of 100
        assert!(len > 100.0);
    }
}
