//! Clip list export to various formats.

use crate::clip::Clip;
use crate::error::{ClipError, ClipResult};

/// Exporter for clip lists.
#[derive(Debug, Clone, Default)]
pub struct ClipListExporter;

impl ClipListExporter {
    /// Creates a new clip list exporter.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Exports clips to CSV format.
    ///
    /// # Errors
    ///
    /// Returns an error if the export fails.
    #[allow(clippy::format_in_format_args)]
    pub fn to_csv(&self, clips: &[Clip]) -> ClipResult<String> {
        let mut output =
            String::from("ID,Name,File Path,Rating,Favorite,Rejected,Keywords,Duration\n");

        for clip in clips {
            let keywords = clip.keywords.join(";");
            let duration = clip
                .effective_duration()
                .map_or(String::new(), |d| d.to_string());

            output.push_str(&format!(
                "{},{},{},{},{},{},{},{}\n",
                clip.id,
                Self::escape_csv(&clip.name),
                Self::escape_csv(&clip.file_path.to_string_lossy()),
                clip.rating.to_value(),
                clip.is_favorite,
                clip.is_rejected,
                Self::escape_csv(&keywords),
                duration
            ));
        }

        Ok(output)
    }

    /// Exports clips to JSON format.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self, clips: &[Clip]) -> ClipResult<String> {
        serde_json::to_string_pretty(clips).map_err(|e| ClipError::Serialization(e.to_string()))
    }

    /// Exports clips to plain text list.
    #[must_use]
    pub fn to_text(&self, clips: &[Clip]) -> String {
        let mut output = String::new();

        for clip in clips {
            output.push_str(&format!("- {} ({})\n", clip.name, clip.file_path.display()));
        }

        output
    }

    fn escape_csv(s: &str) -> String {
        if s.contains(',') || s.contains('"') || s.contains('\n') {
            format!("\"{}\"", s.replace('"', "\"\""))
        } else {
            s.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_export_csv() {
        let mut clip = Clip::new(PathBuf::from("/test.mov"));
        clip.set_name("Test Clip");
        clip.add_keyword("test");

        let clips = vec![clip];
        let exporter = ClipListExporter::new();
        let csv = exporter.to_csv(&clips).expect("to_csv should succeed");

        assert!(csv.contains("ID,Name,File Path"));
        assert!(csv.contains("Test Clip"));
    }

    #[test]
    fn test_export_json() {
        let clip = Clip::new(PathBuf::from("/test.mov"));
        let clips = vec![clip];

        let exporter = ClipListExporter::new();
        let json = exporter.to_json(&clips).expect("to_json should succeed");

        assert!(json.contains("file_path"));
    }

    #[test]
    fn test_csv_escape() {
        assert_eq!(ClipListExporter::escape_csv("simple"), "simple");
        assert_eq!(ClipListExporter::escape_csv("has,comma"), "\"has,comma\"");
        assert_eq!(
            ClipListExporter::escape_csv("has\"quote"),
            "\"has\"\"quote\""
        );
    }
}
