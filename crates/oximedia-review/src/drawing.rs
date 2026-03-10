//! Drawing tools and annotations for visual feedback.

use crate::{DrawingId, SessionId};
use serde::{Deserialize, Serialize};

pub mod annotation;
pub mod color;
pub mod export;
pub mod tools;

pub use annotation::{Annotation, AnnotationLayer};
pub use color::{Color, StrokeStyle};
pub use tools::{DrawingTool, Shape};

/// Point in 2D space (normalized coordinates 0.0-1.0).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    /// X coordinate (0.0 = left, 1.0 = right).
    pub x: f32,
    /// Y coordinate (0.0 = top, 1.0 = bottom).
    pub y: f32,
}

impl Point {
    /// Create a new point.
    #[must_use]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Calculate distance to another point.
    #[must_use]
    pub fn distance_to(&self, other: &Point) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Rectangle defined by top-left and bottom-right corners.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rectangle {
    /// Top-left corner.
    pub top_left: Point,
    /// Bottom-right corner.
    pub bottom_right: Point,
}

impl Rectangle {
    /// Create a new rectangle.
    #[must_use]
    pub fn new(top_left: Point, bottom_right: Point) -> Self {
        Self {
            top_left,
            bottom_right,
        }
    }

    /// Get the width of the rectangle.
    #[must_use]
    pub fn width(&self) -> f32 {
        (self.bottom_right.x - self.top_left.x).abs()
    }

    /// Get the height of the rectangle.
    #[must_use]
    pub fn height(&self) -> f32 {
        (self.bottom_right.y - self.top_left.y).abs()
    }

    /// Get the center point.
    #[must_use]
    pub fn center(&self) -> Point {
        Point::new(
            (self.top_left.x + self.bottom_right.x) / 2.0,
            (self.top_left.y + self.bottom_right.y) / 2.0,
        )
    }

    /// Check if a point is inside the rectangle.
    #[must_use]
    pub fn contains(&self, point: &Point) -> bool {
        point.x >= self.top_left.x
            && point.x <= self.bottom_right.x
            && point.y >= self.top_left.y
            && point.y <= self.bottom_right.y
    }
}

/// Circle defined by center and radius.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Circle {
    /// Center point.
    pub center: Point,
    /// Radius (in normalized coordinates).
    pub radius: f32,
}

impl Circle {
    /// Create a new circle.
    #[must_use]
    pub fn new(center: Point, radius: f32) -> Self {
        Self { center, radius }
    }

    /// Check if a point is inside the circle.
    #[must_use]
    pub fn contains(&self, point: &Point) -> bool {
        self.center.distance_to(point) <= self.radius
    }
}

/// Arrow from one point to another.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Arrow {
    /// Start point.
    pub start: Point,
    /// End point.
    pub end: Point,
    /// Arrow head size.
    pub head_size: f32,
}

impl Arrow {
    /// Create a new arrow.
    #[must_use]
    pub fn new(start: Point, end: Point, head_size: f32) -> Self {
        Self {
            start,
            end,
            head_size,
        }
    }

    /// Get the length of the arrow.
    #[must_use]
    pub fn length(&self) -> f32 {
        self.start.distance_to(&self.end)
    }
}

/// Text annotation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextAnnotation {
    /// Position of the text.
    pub position: Point,
    /// Text content.
    pub text: String,
    /// Font size.
    pub font_size: f32,
}

impl TextAnnotation {
    /// Create a new text annotation.
    #[must_use]
    pub fn new(position: Point, text: String, font_size: f32) -> Self {
        Self {
            position,
            text,
            font_size,
        }
    }
}

/// Freehand path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FreehandPath {
    /// Points in the path.
    pub points: Vec<Point>,
    /// Whether the path is smooth.
    pub smooth: bool,
}

impl FreehandPath {
    /// Create a new freehand path.
    #[must_use]
    pub fn new(points: Vec<Point>, smooth: bool) -> Self {
        Self { points, smooth }
    }

    /// Add a point to the path.
    pub fn add_point(&mut self, point: Point) {
        self.points.push(point);
    }

    /// Get the total length of the path.
    #[must_use]
    pub fn length(&self) -> f32 {
        self.points
            .windows(2)
            .map(|w| w[0].distance_to(&w[1]))
            .sum()
    }
}

/// Drawing on a frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Drawing {
    /// Drawing ID.
    pub id: DrawingId,
    /// Session ID.
    pub session_id: SessionId,
    /// Frame number.
    pub frame: i64,
    /// Drawing tool used.
    pub tool: DrawingTool,
    /// Shape data.
    pub shape: Shape,
    /// Color and style.
    pub style: StrokeStyle,
    /// Author.
    pub author: String,
}

// ---------------------------------------------------------------------------
// Drawing primitives: rasterization helpers for rendering annotations
// onto pixel buffers (RGBA, row-major).
// ---------------------------------------------------------------------------

/// Rasterize a line segment using Bresenham's algorithm.
///
/// Writes `color` (RGBA) into `buffer` for every pixel along the line
/// from `(x0, y0)` to `(x1, y1)`.  Coordinates outside the buffer are
/// silently clipped.
#[allow(clippy::cast_sign_loss)]
pub fn rasterize_line(
    buffer: &mut [u8],
    width: u32,
    height: u32,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: [u8; 4],
) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx: i32 = if x0 < x1 { 1 } else { -1 };
    let sy: i32 = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut cx = x0;
    let mut cy = y0;

    loop {
        if cx >= 0 && cy >= 0 && (cx as u32) < width && (cy as u32) < height {
            let idx = ((cy as u32 * width + cx as u32) as usize) * 4;
            if idx + 3 < buffer.len() {
                buffer[idx] = color[0];
                buffer[idx + 1] = color[1];
                buffer[idx + 2] = color[2];
                buffer[idx + 3] = color[3];
            }
        }
        if cx == x1 && cy == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            cx += sx;
        }
        if e2 <= dx {
            err += dx;
            cy += sy;
        }
    }
}

/// Rasterize a rectangle outline.
#[allow(clippy::cast_possible_truncation)]
pub fn rasterize_rectangle(
    buffer: &mut [u8],
    width: u32,
    height: u32,
    rect: &Rectangle,
    color: [u8; 4],
) {
    let x0 = (rect.top_left.x * width as f32) as i32;
    let y0 = (rect.top_left.y * height as f32) as i32;
    let x1 = (rect.bottom_right.x * width as f32) as i32;
    let y1 = (rect.bottom_right.y * height as f32) as i32;

    rasterize_line(buffer, width, height, x0, y0, x1, y0, color);
    rasterize_line(buffer, width, height, x1, y0, x1, y1, color);
    rasterize_line(buffer, width, height, x1, y1, x0, y1, color);
    rasterize_line(buffer, width, height, x0, y1, x0, y0, color);
}

/// Rasterize a circle outline using the midpoint algorithm.
#[allow(clippy::cast_possible_truncation)]
pub fn rasterize_circle(
    buffer: &mut [u8],
    width: u32,
    height: u32,
    circle: &Circle,
    color: [u8; 4],
) {
    let cx = (circle.center.x * width as f32) as i32;
    let cy = (circle.center.y * height as f32) as i32;
    let r = (circle.radius * width.min(height) as f32) as i32;

    let mut x = r;
    let mut y = 0_i32;
    let mut err = 1 - r;

    while x >= y {
        // Plot the eight octants
        for &(px, py) in &[
            (cx + x, cy + y),
            (cx - x, cy + y),
            (cx + x, cy - y),
            (cx - x, cy - y),
            (cx + y, cy + x),
            (cx - y, cy + x),
            (cx + y, cy - x),
            (cx - y, cy - x),
        ] {
            if px >= 0 && py >= 0 && (px as u32) < width && (py as u32) < height {
                let idx = ((py as u32 * width + px as u32) as usize) * 4;
                if idx + 3 < buffer.len() {
                    buffer[idx] = color[0];
                    buffer[idx + 1] = color[1];
                    buffer[idx + 2] = color[2];
                    buffer[idx + 3] = color[3];
                }
            }
        }

        y += 1;
        if err < 0 {
            err += 2 * y + 1;
        } else {
            x -= 1;
            err += 2 * (y - x) + 1;
        }
    }
}

/// Rasterize an arrow (line with arrowhead).
#[allow(clippy::cast_possible_truncation)]
pub fn rasterize_arrow(buffer: &mut [u8], width: u32, height: u32, arrow: &Arrow, color: [u8; 4]) {
    let x0 = (arrow.start.x * width as f32) as i32;
    let y0 = (arrow.start.y * height as f32) as i32;
    let x1 = (arrow.end.x * width as f32) as i32;
    let y1 = (arrow.end.y * height as f32) as i32;

    // Draw the shaft
    rasterize_line(buffer, width, height, x0, y0, x1, y1, color);

    // Draw the arrowhead
    let dx = (x1 - x0) as f32;
    let dy = (y1 - y0) as f32;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1.0 {
        return;
    }
    let head_px = (arrow.head_size * width.min(height) as f32).max(4.0);
    let ux = dx / len;
    let uy = dy / len;

    // Two barb endpoints
    let bx0 = x1 as f32 - head_px * (ux + uy * 0.5);
    let by0 = y1 as f32 - head_px * (uy - ux * 0.5);
    let bx1 = x1 as f32 - head_px * (ux - uy * 0.5);
    let by1 = y1 as f32 - head_px * (uy + ux * 0.5);

    rasterize_line(buffer, width, height, x1, y1, bx0 as i32, by0 as i32, color);
    rasterize_line(buffer, width, height, x1, y1, bx1 as i32, by1 as i32, color);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_distance() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(3.0, 4.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_rectangle_dimensions() {
        let rect = Rectangle::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0));
        assert!((rect.width() - 1.0).abs() < 0.001);
        assert!((rect.height() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_rectangle_center() {
        let rect = Rectangle::new(Point::new(0.0, 0.0), Point::new(2.0, 2.0));
        let center = rect.center();
        assert!((center.x - 1.0).abs() < 0.001);
        assert!((center.y - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_rectangle_contains() {
        let rect = Rectangle::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0));
        assert!(rect.contains(&Point::new(0.5, 0.5)));
        assert!(!rect.contains(&Point::new(1.5, 0.5)));
    }

    #[test]
    fn test_circle_contains() {
        let circle = Circle::new(Point::new(0.0, 0.0), 1.0);
        assert!(circle.contains(&Point::new(0.5, 0.0)));
        assert!(!circle.contains(&Point::new(2.0, 0.0)));
    }

    #[test]
    fn test_arrow_length() {
        let arrow = Arrow::new(Point::new(0.0, 0.0), Point::new(3.0, 4.0), 0.1);
        assert!((arrow.length() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_freehand_path() {
        let mut path = FreehandPath::new(Vec::new(), true);
        path.add_point(Point::new(0.0, 0.0));
        path.add_point(Point::new(1.0, 0.0));
        assert_eq!(path.points.len(), 2);
        assert!((path.length() - 1.0).abs() < 0.001);
    }
}
