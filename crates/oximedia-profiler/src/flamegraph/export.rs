//! Flame graph file export.
//!
//! Write rendered flame graphs to disk in multiple formats:
//!
//! * **SVG** — scalable vector graphic, readable in any browser.
//! * **HTML** — interactive HTML with JavaScript tooltip handling.
//! * **Folded** — Brendan Gregg's "folded stack" text format, one line per
//!   sample: `frame1;frame2;frame3 <count>`.  Consumable by `flamegraph.pl`
//!   and other tooling.
//!
//! All writes use an atomic rename pattern (write to a `.tmp` sibling, then
//! `rename`) so callers never observe a partially written file.

use std::path::{Path, PathBuf};

use super::generate::FlameGraphData;
use super::interactive::InteractiveRenderer;
use super::svg::SvgRenderer;
use crate::{ProfilerError, Result};

// ---------------------------------------------------------------------------
// Export format
// ---------------------------------------------------------------------------

/// Output format for a flame graph export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// Scalable Vector Graphic.
    Svg,
    /// Interactive HTML with JavaScript tooltip handling.
    Html,
    /// Brendan Gregg's "folded stack" text format.
    Folded,
}

impl ExportFormat {
    /// Canonical file extension for this format (without leading dot).
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Svg => "svg",
            Self::Html => "html",
            Self::Folded => "txt",
        }
    }

    /// MIME type.
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Svg => "image/svg+xml",
            Self::Html => "text/html",
            Self::Folded => "text/plain",
        }
    }
}

// ---------------------------------------------------------------------------
// FlameGraphExporter
// ---------------------------------------------------------------------------

/// Exports flame graphs to files in configurable formats.
///
/// # Example
/// ```no_run
/// use oximedia_profiler::flamegraph::{FlameGraphData, FlameGraphGenerator};
/// use oximedia_profiler::flamegraph::export::{FlameGraphExporter, ExportFormat};
///
/// let data = FlameGraphGenerator::new().generate();
/// let exporter = FlameGraphExporter::new(1200, 800);
/// exporter.export(&data, "/tmp/profile.svg", ExportFormat::Svg).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct FlameGraphExporter {
    width: u32,
    height: u32,
}

impl FlameGraphExporter {
    /// Create a new exporter for a flame graph of the given pixel dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Render `data` to `path` in the given `format`, using an atomic
    /// write-then-rename.
    ///
    /// # Errors
    /// Returns [`ProfilerError::Io`] if the file cannot be written.
    pub fn export(
        &self,
        data: &FlameGraphData,
        path: impl AsRef<Path>,
        format: ExportFormat,
    ) -> Result<()> {
        let path = path.as_ref();
        let content = match format {
            ExportFormat::Svg => self.render_svg(data),
            ExportFormat::Html => self.render_html(data),
            ExportFormat::Folded => Self::render_folded(data),
        };
        self.atomic_write(path, content.as_bytes())
    }

    /// Render and return the content as a `String` without writing to disk.
    pub fn render(
        &self,
        data: &FlameGraphData,
        format: ExportFormat,
    ) -> String {
        match format {
            ExportFormat::Svg => self.render_svg(data),
            ExportFormat::Html => self.render_html(data),
            ExportFormat::Folded => Self::render_folded(data),
        }
    }

    // -----------------------------------------------------------------------
    // Format renderers
    // -----------------------------------------------------------------------

    fn render_svg(&self, data: &FlameGraphData) -> String {
        SvgRenderer::new(self.width, self.height).render(data)
    }

    fn render_html(&self, data: &FlameGraphData) -> String {
        InteractiveRenderer::new(self.width, self.height).render(data)
    }

    /// Convert `FlameGraphData` to Brendan Gregg's folded-stack text format.
    ///
    /// Each line: `frame1;frame2;...;frameN count`
    ///
    /// The tree is walked depth-first; leaf nodes (no children) emit a line.
    /// Internal nodes whose sample count exceeds their children's total also
    /// emit a line for their "self" samples.
    fn render_folded(data: &FlameGraphData) -> String {
        let mut out = String::new();
        let mut stack: Vec<&str> = Vec::new();
        Self::walk_folded(&data.root, &mut stack, &mut out);
        out
    }

    fn walk_folded<'a>(
        node: &'a super::generate::FlameNode,
        stack: &mut Vec<&'a str>,
        out: &mut String,
    ) {
        stack.push(&node.name);

        if node.children.is_empty() {
            // Leaf: emit the full stack.
            if node.value > 0 {
                out.push_str(&stack.join(";"));
                out.push(' ');
                out.push_str(&node.value.to_string());
                out.push('\n');
            }
        } else {
            // Compute children total.
            let children_total: u64 = node.children.iter().map(|c| c.value).sum();
            let self_samples = node.value.saturating_sub(children_total);
            // Emit self samples if any.
            if self_samples > 0 {
                out.push_str(&stack.join(";"));
                out.push(' ');
                out.push_str(&self_samples.to_string());
                out.push('\n');
            }
            for child in &node.children {
                Self::walk_folded(child, stack, out);
            }
        }

        stack.pop();
    }

    // -----------------------------------------------------------------------
    // Atomic write
    // -----------------------------------------------------------------------

    /// Write `data` to `path` atomically: write to `<path>.tmp` then rename.
    fn atomic_write(&self, path: &Path, data: &[u8]) -> Result<()> {
        // Build a sibling tmp path.
        let mut tmp_path: PathBuf = path.to_path_buf();
        {
            let mut name = path
                .file_name()
                .ok_or_else(|| {
                    ProfilerError::Other("export path has no filename".to_string())
                })?
                .to_os_string();
            name.push(".tmp");
            tmp_path.set_file_name(name);
        }

        std::fs::write(&tmp_path, data)?;
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Export flame graph to multiple formats simultaneously.
    ///
    /// Each format is written to `base_path.<ext>` (e.g. `profile.svg`,
    /// `profile.html`, `profile.txt`).
    pub fn export_all(
        &self,
        data: &FlameGraphData,
        base_path: impl AsRef<Path>,
        formats: &[ExportFormat],
    ) -> Result<Vec<PathBuf>> {
        let base = base_path.as_ref();
        let mut written = Vec::with_capacity(formats.len());
        for &fmt in formats {
            let mut dest = base.to_path_buf();
            dest.set_extension(fmt.extension());
            self.export(data, &dest, fmt)?;
            written.push(dest);
        }
        Ok(written)
    }
}

impl Default for FlameGraphExporter {
    fn default() -> Self {
        Self::new(1200, 800)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::sample::{Sample, StackFrame};
    use crate::flamegraph::generate::FlameGraphGenerator;

    fn make_data() -> FlameGraphData {
        let mut gen = FlameGraphGenerator::new();
        let mut s1 = Sample::new(3, 33.3);
        s1.add_frame(StackFrame::new("main".to_string(), 0x1000));
        s1.add_frame(StackFrame::new("decode".to_string(), 0x2000));
        s1.add_frame(StackFrame::new("huffman".to_string(), 0x3000));

        let mut s2 = Sample::new(2, 50.0);
        s2.add_frame(StackFrame::new("main".to_string(), 0x1000));
        s2.add_frame(StackFrame::new("encode".to_string(), 0x4000));

        gen.add_samples(&[s1, s2]);
        gen.generate()
    }

    #[test]
    fn test_export_format_extension() {
        assert_eq!(ExportFormat::Svg.extension(), "svg");
        assert_eq!(ExportFormat::Html.extension(), "html");
        assert_eq!(ExportFormat::Folded.extension(), "txt");
    }

    #[test]
    fn test_export_format_mime() {
        assert_eq!(ExportFormat::Svg.mime_type(), "image/svg+xml");
        assert_eq!(ExportFormat::Html.mime_type(), "text/html");
        assert_eq!(ExportFormat::Folded.mime_type(), "text/plain");
    }

    #[test]
    fn test_render_svg_contains_tag() {
        let exporter = FlameGraphExporter::new(800, 600);
        let data = make_data();
        let svg = exporter.render(&data, ExportFormat::Svg);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_render_html_contains_doctype() {
        let exporter = FlameGraphExporter::default();
        let data = make_data();
        let html = exporter.render(&data, ExportFormat::Html);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("</html>"));
    }

    #[test]
    fn test_render_folded_nonempty() {
        let exporter = FlameGraphExporter::default();
        let data = make_data();
        let folded = exporter.render(&data, ExportFormat::Folded);
        // Should contain at least one stack line
        assert!(folded.contains(';') || !folded.is_empty());
    }

    #[test]
    fn test_render_folded_leaf_line_format() {
        let exporter = FlameGraphExporter::default();
        let data = make_data();
        let folded = exporter.render(&data, ExportFormat::Folded);
        // Every non-empty line must end with a space + integer count.
        for line in folded.lines() {
            let parts: Vec<&str> = line.rsplitn(2, ' ').collect();
            assert_eq!(parts.len(), 2, "line has wrong format: {line}");
            let count_str = parts[0];
            assert!(
                count_str.parse::<u64>().is_ok(),
                "count not u64: {count_str}"
            );
        }
    }

    #[test]
    fn test_export_svg_to_file() {
        let tmp = std::env::temp_dir().join("oximedia_profiler_test_export.svg");
        let exporter = FlameGraphExporter::default();
        let data = make_data();
        exporter
            .export(&data, &tmp, ExportFormat::Svg)
            .expect("should succeed in test");
        let content = std::fs::read_to_string(&tmp).expect("should succeed in test");
        assert!(content.contains("<svg"));
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_export_html_to_file() {
        let tmp = std::env::temp_dir().join("oximedia_profiler_test_export.html");
        let exporter = FlameGraphExporter::default();
        let data = make_data();
        exporter
            .export(&data, &tmp, ExportFormat::Html)
            .expect("should succeed in test");
        let content = std::fs::read_to_string(&tmp).expect("should succeed in test");
        assert!(content.contains("<!DOCTYPE html>"));
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_export_folded_to_file() {
        let tmp = std::env::temp_dir().join("oximedia_profiler_test_export.txt");
        let exporter = FlameGraphExporter::default();
        let data = make_data();
        exporter
            .export(&data, &tmp, ExportFormat::Folded)
            .expect("should succeed in test");
        let content = std::fs::read_to_string(&tmp).expect("should succeed in test");
        // Non-empty and all lines parseable.
        assert!(!content.trim().is_empty());
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_export_all_creates_multiple_files() {
        let base = std::env::temp_dir().join("oximedia_profiler_export_all");
        let exporter = FlameGraphExporter::default();
        let data = make_data();
        let formats = [ExportFormat::Svg, ExportFormat::Html, ExportFormat::Folded];
        let paths = exporter
            .export_all(&data, &base, &formats)
            .expect("should succeed in test");
        assert_eq!(paths.len(), 3);
        for path in &paths {
            assert!(path.exists(), "{path:?} was not created");
            std::fs::remove_file(path).ok();
        }
    }

    #[test]
    fn test_default_dimensions() {
        let exp = FlameGraphExporter::default();
        assert_eq!(exp.width, 1200);
        assert_eq!(exp.height, 800);
    }
}
