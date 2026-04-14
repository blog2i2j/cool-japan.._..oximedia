//! Dolby Vision XML manifest builder.
//!
//! Generates a minimal but spec-aligned Dolby Vision XML manifest that
//! describes the shot structure of a content item.  Each shot carries a
//! frame range (`start` / `end` frame numbers) and the peak luminance of the
//! L1 content-mastering metadata.
//!
//! # Output format
//!
//! The XML produced follows the pattern used by Dolby's CMU (Content Mapping
//! Unit) tools:
//!
//! ```xml
//! <?xml version="1.0" encoding="UTF-8"?>
//! <DolbyVisionManifest version="1.0">
//!   <Shots>
//!     <Shot start="0" end="95" l1_max_pq="3079.00"/>
//!     <!-- … -->
//!   </Shots>
//! </DolbyVisionManifest>
//! ```

/// A single shot entry in the Dolby Vision manifest.
#[derive(Debug, Clone)]
pub struct DvManifestShot {
    /// First frame number of the shot (inclusive).
    pub start: u64,
    /// Last frame number of the shot (inclusive).
    pub end: u64,
    /// Peak luminance for the shot as an L1 max_pq value (cd/m² or PQ-coded).
    pub l1_max: f32,
}

/// Builder for a Dolby Vision XML manifest.
///
/// Accumulate shots with [`DvManifestBuilder::add_shot`], then call
/// [`DvManifestBuilder::build_xml`] to generate the manifest string.
#[derive(Debug, Clone, Default)]
pub struct DvManifestBuilder {
    shots: Vec<DvManifestShot>,
    /// Content title (optional, written as an XML attribute when set).
    title: Option<String>,
    /// Frame rate as a rational number (e.g. `(24000, 1001)` for 23.976 fps).
    frame_rate: Option<(u32, u32)>,
}

impl DvManifestBuilder {
    /// Create a new, empty manifest builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set an optional content title that will appear in the manifest header.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the source frame rate as a rational `(numerator, denominator)` pair.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_dolbyvision::xml_manifest::DvManifestBuilder;
    /// let builder = DvManifestBuilder::new().with_frame_rate(24000, 1001);
    /// ```
    #[must_use]
    pub fn with_frame_rate(mut self, numerator: u32, denominator: u32) -> Self {
        self.frame_rate = Some((numerator, denominator));
        self
    }

    /// Add a shot to the manifest.
    ///
    /// # Arguments
    ///
    /// * `start` — first frame of the shot (0-based, inclusive).
    /// * `end`   — last frame of the shot (0-based, inclusive).
    /// * `l1_max` — peak luminance / L1 `max_pq` value for the shot.
    pub fn add_shot(&mut self, start: u64, end: u64, l1_max: f32) {
        self.shots.push(DvManifestShot { start, end, l1_max });
    }

    /// Build the XML manifest string.
    ///
    /// Returns a UTF-8 XML document.  Special XML characters (`<`, `>`, `&`,
    /// `"`, `'`) in `title` are escaped.
    #[must_use]
    pub fn build_xml(&self) -> String {
        let mut xml = String::with_capacity(256 + self.shots.len() * 80);
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");

        // Root element open tag
        xml.push_str("<DolbyVisionManifest version=\"1.0\"");
        if let Some(ref title) = self.title {
            xml.push_str(" title=\"");
            xml.push_str(&escape_xml_attr(title));
            xml.push('"');
        }
        if let Some((num, den)) = self.frame_rate {
            xml.push_str(&format!(" frameRate=\"{num}/{den}\""));
        }
        xml.push_str(">\n");

        // Shots block
        xml.push_str("  <Shots>\n");
        for shot in &self.shots {
            xml.push_str(&format!(
                "    <Shot start=\"{}\" end=\"{}\" l1_max_pq=\"{:.2}\"/>\n",
                shot.start, shot.end, shot.l1_max
            ));
        }
        xml.push_str("  </Shots>\n");

        xml.push_str("</DolbyVisionManifest>\n");
        xml
    }

    /// Returns the number of shots accumulated so far.
    #[must_use]
    pub fn shot_count(&self) -> usize {
        self.shots.len()
    }
}

/// Escape special XML attribute characters.
fn escape_xml_attr(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_manifest() {
        let builder = DvManifestBuilder::new();
        let xml = builder.build_xml();
        assert!(xml.contains("<?xml version=\"1.0\""));
        assert!(xml.contains("<DolbyVisionManifest"));
        assert!(xml.contains("<Shots>"));
        assert!(xml.contains("</Shots>"));
        assert!(xml.contains("</DolbyVisionManifest>"));
    }

    #[test]
    fn test_manifest_with_shots() {
        let mut builder = DvManifestBuilder::new();
        builder.add_shot(0, 95, 3079.0);
        builder.add_shot(96, 240, 2500.5);

        let xml = builder.build_xml();

        assert!(xml.contains("start=\"0\""), "shot start");
        assert!(xml.contains("end=\"95\""), "shot end");
        assert!(xml.contains("l1_max_pq=\"3079.00\""), "l1_max");
        assert!(xml.contains("start=\"96\""));
        assert!(xml.contains("end=\"240\""));
        assert!(xml.contains("l1_max_pq=\"2500.50\""));
        assert_eq!(builder.shot_count(), 2);
    }

    #[test]
    fn test_manifest_with_title() {
        let builder = DvManifestBuilder::new().with_title("My Movie");
        let xml = builder.build_xml();
        assert!(xml.contains("title=\"My Movie\""), "title attribute: {xml}");
    }

    #[test]
    fn test_title_xml_escaping() {
        let builder = DvManifestBuilder::new().with_title("Film & \"Crew\"");
        let xml = builder.build_xml();
        assert!(xml.contains("Film &amp; &quot;Crew&quot;"), "escaped: {xml}");
    }

    #[test]
    fn test_manifest_with_frame_rate() {
        let builder = DvManifestBuilder::new().with_frame_rate(24000, 1001);
        let xml = builder.build_xml();
        assert!(xml.contains("frameRate=\"24000/1001\""), "frame rate: {xml}");
    }

    #[test]
    fn test_shot_count() {
        let mut builder = DvManifestBuilder::new();
        assert_eq!(builder.shot_count(), 0);
        builder.add_shot(0, 10, 1000.0);
        builder.add_shot(11, 20, 2000.0);
        assert_eq!(builder.shot_count(), 2);
    }

    #[test]
    fn test_xml_structure_ordering() {
        let mut builder = DvManifestBuilder::new();
        builder.add_shot(100, 200, 1500.0);
        let xml = builder.build_xml();

        // XML declaration must come first
        let decl_pos = xml.find("<?xml").expect("xml decl");
        let root_pos = xml.find("<DolbyVisionManifest").expect("root");
        let shots_pos = xml.find("<Shots>").expect("shots");
        let shot_pos = xml.find("<Shot ").expect("shot");

        assert!(decl_pos < root_pos);
        assert!(root_pos < shots_pos);
        assert!(shots_pos < shot_pos);
    }
}
