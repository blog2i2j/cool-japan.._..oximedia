//! Drawing tools for creating annotations.

use crate::drawing::{Arrow, Circle, FreehandPath, Point, Rectangle, TextAnnotation};
use serde::{Deserialize, Serialize};

/// Available drawing tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DrawingTool {
    /// Arrow tool - point to specific elements.
    Arrow,
    /// Circle tool - highlight areas.
    Circle,
    /// Rectangle tool - mark regions.
    Rectangle,
    /// Freehand tool - draw custom shapes.
    Freehand,
    /// Text tool - add text notes.
    Text,
    /// Highlighter tool - semi-transparent highlighting.
    Highlighter,
    /// Pen tool - precise drawing.
    Pen,
    /// Eraser tool - remove drawings.
    Eraser,
}

impl DrawingTool {
    /// Get all available tools.
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::Arrow,
            Self::Circle,
            Self::Rectangle,
            Self::Freehand,
            Self::Text,
            Self::Highlighter,
            Self::Pen,
            Self::Eraser,
        ]
    }

    /// Check if tool creates a closed shape.
    #[must_use]
    pub fn is_closed_shape(self) -> bool {
        matches!(self, Self::Circle | Self::Rectangle)
    }

    /// Check if tool supports fill.
    #[must_use]
    pub fn supports_fill(self) -> bool {
        matches!(self, Self::Circle | Self::Rectangle | Self::Highlighter)
    }
}

/// Shape created by a drawing tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Shape {
    /// Arrow shape.
    Arrow(Arrow),
    /// Circle shape.
    Circle(Circle),
    /// Rectangle shape.
    Rectangle(Rectangle),
    /// Freehand path.
    Freehand(FreehandPath),
    /// Text annotation.
    Text(TextAnnotation),
}

impl Shape {
    /// Get the bounding box of the shape.
    #[must_use]
    pub fn bounding_box(&self) -> Rectangle {
        match self {
            Shape::Arrow(arrow) => {
                let min_x = arrow.start.x.min(arrow.end.x);
                let max_x = arrow.start.x.max(arrow.end.x);
                let min_y = arrow.start.y.min(arrow.end.y);
                let max_y = arrow.start.y.max(arrow.end.y);
                Rectangle::new(Point::new(min_x, min_y), Point::new(max_x, max_y))
            }
            Shape::Circle(circle) => Rectangle::new(
                Point::new(
                    circle.center.x - circle.radius,
                    circle.center.y - circle.radius,
                ),
                Point::new(
                    circle.center.x + circle.radius,
                    circle.center.y + circle.radius,
                ),
            ),
            Shape::Rectangle(rect) => *rect,
            Shape::Freehand(path) => {
                let min_x = path
                    .points
                    .iter()
                    .map(|p| p.x)
                    .fold(f32::INFINITY, f32::min);
                let max_x = path
                    .points
                    .iter()
                    .map(|p| p.x)
                    .fold(f32::NEG_INFINITY, f32::max);
                let min_y = path
                    .points
                    .iter()
                    .map(|p| p.y)
                    .fold(f32::INFINITY, f32::min);
                let max_y = path
                    .points
                    .iter()
                    .map(|p| p.y)
                    .fold(f32::NEG_INFINITY, f32::max);
                Rectangle::new(Point::new(min_x, min_y), Point::new(max_x, max_y))
            }
            Shape::Text(text) => {
                // Simple bounding box for text
                Rectangle::new(
                    text.position,
                    Point::new(text.position.x + 0.1, text.position.y + 0.05),
                )
            }
        }
    }

    /// Check if shape contains a point.
    ///
    /// For closed shapes (circle, rectangle) performs an interior test.
    /// For freehand paths uses the even-odd ray-casting rule.
    /// Arrows and text have no fill and always return `false`.
    #[must_use]
    pub fn contains(&self, point: &Point) -> bool {
        match self {
            Shape::Arrow(_) => false,
            Shape::Circle(circle) => circle.contains(point),
            Shape::Rectangle(rect) => rect.contains(point),
            Shape::Freehand(path) => Self::freehand_contains(path, point),
            Shape::Text(_) => false,
        }
    }

    /// Even-odd ray casting test for a closed freehand polygon.
    fn freehand_contains(path: &FreehandPath, point: &Point) -> bool {
        let pts = &path.points;
        if pts.len() < 3 {
            return false;
        }
        let mut inside = false;
        let n = pts.len();
        let mut j = n - 1;
        for i in 0..n {
            let yi = pts[i].y;
            let yj = pts[j].y;
            let xi = pts[i].x;
            let xj = pts[j].x;
            if ((yi > point.y) != (yj > point.y))
                && (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi)
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }
}

/// Tool configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    /// Selected tool.
    pub tool: DrawingTool,
    /// Stroke width.
    pub stroke_width: f32,
    /// Whether shape should be filled.
    pub filled: bool,
    /// Opacity (0.0-1.0).
    pub opacity: f32,
}

impl Default for ToolConfig {
    fn default() -> Self {
        Self {
            tool: DrawingTool::Pen,
            stroke_width: 2.0,
            filled: false,
            opacity: 1.0,
        }
    }
}

impl ToolConfig {
    /// Create a new tool configuration.
    #[must_use]
    pub fn new(tool: DrawingTool) -> Self {
        Self {
            tool,
            ..Default::default()
        }
    }

    /// Set stroke width.
    #[must_use]
    pub fn with_stroke_width(mut self, width: f32) -> Self {
        self.stroke_width = width;
        self
    }

    /// Set fill.
    #[must_use]
    pub fn with_fill(mut self, filled: bool) -> Self {
        self.filled = filled;
        self
    }

    /// Set opacity.
    #[must_use]
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drawing_tool_all() {
        let tools = DrawingTool::all();
        assert_eq!(tools.len(), 8);
        assert!(tools.contains(&DrawingTool::Arrow));
    }

    #[test]
    fn test_drawing_tool_is_closed_shape() {
        assert!(DrawingTool::Circle.is_closed_shape());
        assert!(DrawingTool::Rectangle.is_closed_shape());
        assert!(!DrawingTool::Arrow.is_closed_shape());
    }

    #[test]
    fn test_drawing_tool_supports_fill() {
        assert!(DrawingTool::Circle.supports_fill());
        assert!(DrawingTool::Rectangle.supports_fill());
        assert!(!DrawingTool::Arrow.supports_fill());
    }

    #[test]
    fn test_shape_bounding_box() {
        let circle = Shape::Circle(Circle::new(Point::new(0.5, 0.5), 0.2));
        let bbox = circle.bounding_box();
        assert!((bbox.top_left.x - 0.3).abs() < 0.001);
        assert!((bbox.bottom_right.x - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_shape_contains() {
        let circle = Shape::Circle(Circle::new(Point::new(0.0, 0.0), 1.0));
        assert!(circle.contains(&Point::new(0.5, 0.0)));
        assert!(!circle.contains(&Point::new(2.0, 0.0)));
    }

    #[test]
    fn test_tool_config_default() {
        let config = ToolConfig::default();
        assert_eq!(config.tool, DrawingTool::Pen);
        assert!((config.stroke_width - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_tool_config_builder() {
        let config = ToolConfig::new(DrawingTool::Arrow)
            .with_stroke_width(3.0)
            .with_fill(true)
            .with_opacity(0.5);

        assert_eq!(config.tool, DrawingTool::Arrow);
        assert!((config.stroke_width - 3.0).abs() < 0.001);
        assert!(config.filled);
        assert!((config.opacity - 0.5).abs() < 0.001);
    }
}
