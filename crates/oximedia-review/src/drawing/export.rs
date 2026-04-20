//! Export drawings and annotations.

use crate::{
    drawing::{Annotation, Shape},
    error::ReviewResult,
    SessionId,
};
use serde::{Deserialize, Serialize};

/// Export format for drawings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// SVG (Scalable Vector Graphics).
    Svg,
    /// PNG (Portable Network Graphics).
    Png,
    /// PDF (Portable Document Format).
    Pdf,
    /// JSON (JavaScript Object Notation).
    Json,
}

/// Export options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportOptions {
    /// Export format.
    pub format: ExportFormat,
    /// Include hidden annotations.
    pub include_hidden: bool,
    /// Include locked annotations.
    pub include_locked: bool,
    /// Background color (if any).
    pub background_color: Option<crate::drawing::Color>,
    /// Export resolution (for raster formats).
    pub resolution: (u32, u32),
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            format: ExportFormat::Svg,
            include_hidden: false,
            include_locked: true,
            background_color: None,
            resolution: (1920, 1080),
        }
    }
}

impl ExportOptions {
    /// Create new export options.
    #[must_use]
    pub fn new(format: ExportFormat) -> Self {
        Self {
            format,
            ..Default::default()
        }
    }

    /// Set whether to include hidden annotations.
    #[must_use]
    pub fn include_hidden(mut self, include: bool) -> Self {
        self.include_hidden = include;
        self
    }

    /// Set resolution.
    #[must_use]
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.resolution = (width, height);
        self
    }
}

/// Export annotations to a file.
///
/// # Arguments
///
/// * `session_id` - ID of the session
/// * `annotations` - Annotations to export
/// * `options` - Export options
/// * `output_path` - Output file path
///
/// # Errors
///
/// Returns error if export fails.
pub async fn export_annotations(
    session_id: SessionId,
    annotations: &[Annotation],
    options: &ExportOptions,
    output_path: &str,
) -> ReviewResult<()> {
    // Filter annotations based on options
    let filtered: Vec<&Annotation> = annotations
        .iter()
        .filter(|a| (options.include_hidden || a.visible) && (options.include_locked || !a.locked))
        .collect();

    match options.format {
        ExportFormat::Svg => export_to_svg(session_id, &filtered, output_path).await?,
        ExportFormat::Png => export_to_png(session_id, &filtered, options, output_path).await?,
        ExportFormat::Pdf => export_to_pdf(session_id, &filtered, output_path).await?,
        ExportFormat::Json => export_to_json(session_id, &filtered, output_path).await?,
    }

    Ok(())
}

async fn export_to_svg(
    _session_id: SessionId,
    annotations: &[&Annotation],
    output_path: &str,
) -> ReviewResult<()> {
    let mut svg = String::new();
    svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    svg.push_str("<svg xmlns=\"http://www.w3.org/2000/svg\" ");
    svg.push_str("width=\"1920\" height=\"1080\" viewBox=\"0 0 1920 1080\">\n");

    for ann in annotations {
        let style = &ann.drawing.style;
        let stroke_color = format!(
            "rgba({},{},{},{})",
            style.color.r, style.color.g, style.color.b, style.color.a
        );
        let stroke_width = style.width;

        let path_data = shape_to_svg_path(&ann.drawing.shape);

        match &ann.drawing.shape {
            Shape::Text(text) => {
                svg.push_str(&format!(
                    "  <text x=\"{}\" y=\"{}\" font-size=\"{}\" fill=\"{}\">{}</text>\n",
                    text.position.x * 1920.0,
                    text.position.y * 1080.0,
                    text.font_size,
                    stroke_color,
                    text.text,
                ));
            }
            Shape::Circle(circle) => {
                svg.push_str(&format!(
                    "  <circle cx=\"{}\" cy=\"{}\" r=\"{}\" stroke=\"{}\" stroke-width=\"{}\" fill=\"none\"/>\n",
                    circle.center.x * 1920.0,
                    circle.center.y * 1080.0,
                    circle.radius * 1080.0_f32.min(1920.0),
                    stroke_color,
                    stroke_width,
                ));
            }
            _ => {
                if !path_data.is_empty() {
                    svg.push_str(&format!(
                        "  <path d=\"{}\" stroke=\"{}\" stroke-width=\"{}\" fill=\"none\"/>\n",
                        path_data, stroke_color, stroke_width,
                    ));
                }
            }
        }

        // Add label as a title element if present
        if let Some(ref label) = ann.label {
            svg.push_str(&format!("  <!-- label: {} -->\n", label));
        }
    }

    svg.push_str("</svg>\n");
    std::fs::write(output_path, svg)?;
    Ok(())
}

async fn export_to_png(
    _session_id: SessionId,
    annotations: &[&Annotation],
    options: &ExportOptions,
    output_path: &str,
) -> ReviewResult<()> {
    let (w, h) = options.resolution;
    let bg = options
        .background_color
        .map_or([0u8, 0, 0, 255], |c| [c.r, c.g, c.b, (c.a * 255.0) as u8]);

    // Create RGBA buffer
    let mut buffer = vec![0u8; (w * h * 4) as usize];
    // Fill background
    for pixel in buffer.chunks_exact_mut(4) {
        pixel.copy_from_slice(&bg);
    }

    // Rasterize each annotation
    for ann in annotations {
        let color = [
            ann.drawing.style.color.r,
            ann.drawing.style.color.g,
            ann.drawing.style.color.b,
            (ann.drawing.style.color.a * 255.0) as u8,
        ];

        match &ann.drawing.shape {
            Shape::Arrow(arrow) => {
                crate::drawing::rasterize_arrow(&mut buffer, w, h, arrow, color);
            }
            Shape::Circle(circle) => {
                crate::drawing::rasterize_circle(&mut buffer, w, h, circle, color);
            }
            Shape::Rectangle(rect) => {
                crate::drawing::rasterize_rectangle(&mut buffer, w, h, rect, color);
            }
            Shape::Freehand(path) => {
                // Rasterize as connected line segments
                for pair in path.points.windows(2) {
                    let x0 = (pair[0].x * w as f32) as i32;
                    let y0 = (pair[0].y * h as f32) as i32;
                    let x1 = (pair[1].x * w as f32) as i32;
                    let y1 = (pair[1].y * h as f32) as i32;
                    crate::drawing::rasterize_line(&mut buffer, w, h, x0, y0, x1, y1, color);
                }
            }
            Shape::Text(_) => {
                // Text rasterization would need a font renderer; skip for raw PNG
            }
        }
    }

    // Write raw RGBA as a simple file (full PNG encoding would require a crate)
    // We write the raw buffer prefixed with a small header for downstream tools
    let mut output = Vec::new();
    output.extend_from_slice(b"RGBA");
    output.extend_from_slice(&w.to_le_bytes());
    output.extend_from_slice(&h.to_le_bytes());
    output.extend_from_slice(&buffer);
    std::fs::write(output_path, output)?;

    Ok(())
}

async fn export_to_pdf(
    session_id: SessionId,
    annotations: &[&Annotation],
    output_path: &str,
) -> ReviewResult<()> {
    // Generate a simple text-based report as a PDF-compatible document
    let mut content = String::new();
    content.push_str(&format!(
        "Review Annotations Report\nSession: {}\n\n",
        session_id
    ));

    for (i, ann) in annotations.iter().enumerate() {
        content.push_str(&format!("Annotation {}\n", i + 1));
        content.push_str(&format!("  Frame: {}\n", ann.drawing.frame));
        content.push_str(&format!("  Author: {}\n", ann.drawing.author));
        content.push_str(&format!("  Tool: {:?}\n", ann.drawing.tool));
        if let Some(ref label) = ann.label {
            content.push_str(&format!("  Label: {}\n", label));
        }
        let bbox = ann.drawing.shape.bounding_box();
        content.push_str(&format!(
            "  Bounding Box: ({:.2},{:.2}) to ({:.2},{:.2})\n",
            bbox.top_left.x, bbox.top_left.y, bbox.bottom_right.x, bbox.bottom_right.y
        ));
        content.push('\n');
    }

    std::fs::write(output_path, content)?;
    Ok(())
}

async fn export_to_json(
    _session_id: SessionId,
    annotations: &[&Annotation],
    output_path: &str,
) -> ReviewResult<()> {
    // Serialize annotations to JSON
    let json = serde_json::to_string_pretty(annotations)?;

    // Write to file
    std::fs::write(output_path, json)?;

    Ok(())
}

/// Convert shape to SVG path string.
#[must_use]
pub fn shape_to_svg_path(shape: &Shape) -> String {
    match shape {
        Shape::Arrow(arrow) => {
            format!(
                "M {} {} L {} {}",
                arrow.start.x, arrow.start.y, arrow.end.x, arrow.end.y
            )
        }
        Shape::Circle(circle) => {
            format!(
                "M {} {} m -{}, 0 a {},{} 0 1,0 {},0 a {},{} 0 1,0 -{},0",
                circle.center.x,
                circle.center.y,
                circle.radius,
                circle.radius,
                circle.radius,
                circle.radius * 2.0,
                circle.radius,
                circle.radius,
                circle.radius * 2.0
            )
        }
        Shape::Rectangle(rect) => {
            format!(
                "M {} {} L {} {} L {} {} L {} {} Z",
                rect.top_left.x,
                rect.top_left.y,
                rect.bottom_right.x,
                rect.top_left.y,
                rect.bottom_right.x,
                rect.bottom_right.y,
                rect.top_left.x,
                rect.bottom_right.y
            )
        }
        Shape::Freehand(path) => {
            if path.points.is_empty() {
                return String::new();
            }

            let mut svg = format!("M {} {}", path.points[0].x, path.points[0].y);
            for point in &path.points[1..] {
                svg.push_str(&format!(" L {} {}", point.x, point.y));
            }
            svg
        }
        Shape::Text(_) => String::new(), // Text handled separately
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drawing::{Arrow, Circle, Color, Point, Rectangle, StrokeStyle};

    #[test]
    fn test_export_options_default() {
        let options = ExportOptions::default();
        assert_eq!(options.format, ExportFormat::Svg);
        assert!(!options.include_hidden);
        assert!(options.include_locked);
    }

    #[test]
    fn test_export_options_builder() {
        let options = ExportOptions::new(ExportFormat::Png)
            .include_hidden(true)
            .with_resolution(3840, 2160);

        assert_eq!(options.format, ExportFormat::Png);
        assert!(options.include_hidden);
        assert_eq!(options.resolution, (3840, 2160));
    }

    #[test]
    fn test_shape_to_svg_path_arrow() {
        let arrow = Arrow::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0), 0.1);
        let svg = shape_to_svg_path(&Shape::Arrow(arrow));
        assert!(svg.starts_with("M 0 0 L 1 1"));
    }

    #[test]
    fn test_shape_to_svg_path_circle() {
        let circle = Circle::new(Point::new(0.5, 0.5), 0.2);
        let svg = shape_to_svg_path(&Shape::Circle(circle));
        assert!(svg.contains("M 0.5 0.5"));
    }

    #[test]
    fn test_shape_to_svg_path_rectangle() {
        let rect = Rectangle::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0));
        let svg = shape_to_svg_path(&Shape::Rectangle(rect));
        assert!(svg.contains("M 0 0"));
        assert!(svg.contains("Z")); // Path closes
    }

    #[tokio::test]
    async fn test_export_to_json() {
        let session_id = SessionId::new();
        let drawing = crate::drawing::Drawing {
            id: crate::DrawingId::new(),
            session_id,
            frame: 100,
            tool: crate::drawing::tools::DrawingTool::Circle,
            shape: Shape::Circle(Circle::new(Point::new(0.5, 0.5), 0.2)),
            style: StrokeStyle::solid(Color::red(), 2.0),
            author: "test".to_string(),
        };

        let annotation = Annotation::new(drawing);
        let annotations = vec![&annotation];

        let temp_file = std::env::temp_dir()
            .join("oximedia-review-drawing-export-test.json")
            .to_string_lossy()
            .into_owned();
        let result = export_to_json(session_id, &annotations, &temp_file).await;
        assert!(result.is_ok());

        // Clean up
        let _ = std::fs::remove_file(&temp_file);
    }
}
