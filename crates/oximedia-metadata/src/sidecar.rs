//! Sidecar metadata file management.
//!
//! Provides support for external sidecar files that accompany media files,
//! including XMP, JSON, YAML, and CSV formats.

use std::collections::HashMap;

/// Supported sidecar file formats.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SidecarFormat {
    /// Adobe XMP sidecar (.xmp)
    XmpSidecar,
    /// JSON sidecar (.json)
    JsonSidecar,
    /// YAML sidecar (.yaml / .yml)
    YamlSidecar,
    /// CSV sidecar (.csv)
    CsvSidecar,
}

impl SidecarFormat {
    /// Return the canonical file extension for this format (without the dot).
    #[allow(dead_code)]
    pub fn extension(self) -> &'static str {
        match self {
            Self::XmpSidecar => "xmp",
            Self::JsonSidecar => "json",
            Self::YamlSidecar => "yaml",
            Self::CsvSidecar => "csv",
        }
    }
}

/// An external sidecar metadata file.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SidecarFile {
    /// Path to the sidecar file.
    pub path: String,
    /// Format of the sidecar.
    pub format: SidecarFormat,
    /// Key-value metadata content.
    pub content: HashMap<String, String>,
}

impl SidecarFile {
    /// Create a new, empty sidecar file descriptor.
    #[allow(dead_code)]
    pub fn new(path: &str, format: SidecarFormat) -> Self {
        Self {
            path: path.to_string(),
            format,
            content: HashMap::new(),
        }
    }

    /// Set a metadata field.
    #[allow(dead_code)]
    pub fn set(&mut self, key: &str, value: &str) {
        self.content.insert(key.to_string(), value.to_string());
    }

    /// Get a metadata field value.
    #[allow(dead_code)]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.content.get(key).map(String::as_str)
    }

    /// Serialize the sidecar content to a string in the appropriate format.
    #[allow(dead_code)]
    pub fn serialize(&self) -> String {
        match self.format {
            SidecarFormat::XmpSidecar => self.serialize_xmp(),
            SidecarFormat::JsonSidecar => self.serialize_json(),
            SidecarFormat::YamlSidecar => self.serialize_yaml(),
            SidecarFormat::CsvSidecar => self.serialize_csv(),
        }
    }

    fn serialize_xmp(&self) -> String {
        let mut out = String::from(
            "<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>\n\
             <x:xmpmeta xmlns:x='adobe:ns:meta/'>\n\
             <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>\n\
             <rdf:Description rdf:about='' xmlns:dc='http://purl.org/dc/elements/1.1/'>\n",
        );
        // Sort keys for deterministic output
        let mut keys: Vec<&String> = self.content.keys().collect();
        keys.sort();
        for key in keys {
            let value = &self.content[key];
            // Escape XML special characters
            let escaped = value
                .replace('&', "&amp;")
                .replace('<', "&lt;")
                .replace('>', "&gt;")
                .replace('"', "&quot;");
            out.push_str(&format!("  <dc:{key}>{escaped}</dc:{key}>\n"));
        }
        out.push_str(
            "</rdf:Description>\n\
             </rdf:RDF>\n\
             </x:xmpmeta>\n\
             <?xpacket end='w'?>",
        );
        out
    }

    fn serialize_json(&self) -> String {
        let mut pairs: Vec<String> = self
            .content
            .iter()
            .map(|(k, v)| {
                let escaped_v = v.replace('\\', "\\\\").replace('"', "\\\"");
                format!("  \"{k}\": \"{escaped_v}\"")
            })
            .collect();
        pairs.sort(); // deterministic
        format!("{{\n{}\n}}", pairs.join(",\n"))
    }

    fn serialize_yaml(&self) -> String {
        let mut keys: Vec<&String> = self.content.keys().collect();
        keys.sort();
        keys.iter()
            .map(|k| {
                let v = &self.content[*k];
                // Quote values that contain YAML special chars
                if v.contains(':') || v.contains('#') || v.starts_with(' ') {
                    format!("{k}: \"{v}\"")
                } else {
                    format!("{k}: {v}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn serialize_csv(&self) -> String {
        let mut keys: Vec<&String> = self.content.keys().collect();
        keys.sort();
        let header = keys
            .iter()
            .map(|k| k.as_str())
            .collect::<Vec<_>>()
            .join(",");
        let values = keys
            .iter()
            .map(|k| {
                let v = &self.content[*k];
                if v.contains(',') || v.contains('"') || v.contains('\n') {
                    format!("\"{}\"", v.replace('"', "\"\""))
                } else {
                    v.clone()
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        format!("{header}\n{values}")
    }

    /// Parse a sidecar from its string representation.
    #[allow(dead_code)]
    pub fn from_str(s: &str, format: SidecarFormat) -> Self {
        let mut sidecar = Self::new("", format);
        match format {
            SidecarFormat::JsonSidecar => parse_json_into(&mut sidecar.content, s),
            SidecarFormat::YamlSidecar => parse_yaml_into(&mut sidecar.content, s),
            SidecarFormat::CsvSidecar => parse_csv_into(&mut sidecar.content, s),
            SidecarFormat::XmpSidecar => parse_xmp_into(&mut sidecar.content, s),
        }
        sidecar
    }
}

/// Parse a minimal JSON object (string values only) into a map.
fn parse_json_into(map: &mut HashMap<String, String>, s: &str) {
    // Lightweight parser: look for "key": "value" pairs
    let s = s.trim().trim_start_matches('{').trim_end_matches('}');
    for pair in s.split(',') {
        let pair = pair.trim();
        if let Some((k, v)) = pair.split_once(':') {
            let k = k.trim().trim_matches('"').to_string();
            let v = v.trim().trim_matches('"').to_string();
            if !k.is_empty() {
                map.insert(k, v);
            }
        }
    }
}

/// Parse minimal YAML (key: value lines) into a map.
fn parse_yaml_into(map: &mut HashMap<String, String>, s: &str) {
    for line in s.lines() {
        if let Some((k, v)) = line.split_once(':') {
            let k = k.trim().to_string();
            let v = v.trim().trim_matches('"').to_string();
            if !k.is_empty() && !k.starts_with('#') {
                map.insert(k, v);
            }
        }
    }
}

/// Parse minimal CSV (header + one data row) into a map.
fn parse_csv_into(map: &mut HashMap<String, String>, s: &str) {
    let mut lines = s.lines();
    if let (Some(header), Some(values)) = (lines.next(), lines.next()) {
        let keys: Vec<&str> = header.split(',').collect();
        let vals: Vec<&str> = values.split(',').collect();
        for (k, v) in keys.iter().zip(vals.iter()) {
            let k = k.trim().trim_matches('"').to_string();
            let v = v.trim().trim_matches('"').to_string();
            if !k.is_empty() {
                map.insert(k, v);
            }
        }
    }
}

/// Parse minimal XMP (dc:tag content) into a map.
fn parse_xmp_into(map: &mut HashMap<String, String>, s: &str) {
    // Look for patterns like <dc:key>value</dc:key>
    let mut remaining = s;
    while let Some(open_start) = remaining.find("<dc:") {
        let tag_rest = &remaining[open_start + 4..];
        if let Some(close_angle) = tag_rest.find('>') {
            let tag_name = &tag_rest[..close_angle];
            let after_open = &tag_rest[close_angle + 1..];
            let close_tag = format!("</dc:{tag_name}>");
            if let Some(close_start) = after_open.find(&close_tag) {
                let value = &after_open[..close_start];
                // Unescape basic XML entities
                let value = value
                    .replace("&amp;", "&")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                    .replace("&quot;", "\"");
                map.insert(tag_name.to_string(), value);
                remaining = &after_open[close_start + close_tag.len()..];
                continue;
            }
        }
        break;
    }
}

/// Check for a sidecar file alongside `media_path` by testing common extensions.
/// Returns the first existing sidecar path found, or `None`.
#[allow(dead_code)]
pub fn find_sidecar(media_path: &str) -> Option<String> {
    let extensions = ["xmp", "json", "yaml", "yml", "csv"];
    for ext in &extensions {
        let candidate = format!("{media_path}.{ext}");
        if std::path::Path::new(&candidate).exists() {
            return Some(candidate);
        }
    }
    // Also try replacing the original extension
    if let Some(dot_pos) = media_path.rfind('.') {
        let base = &media_path[..dot_pos];
        for ext in &extensions {
            let candidate = format!("{base}.{ext}");
            if std::path::Path::new(&candidate).exists() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Return the conventional sidecar path for `media_path` and the given format.
#[allow(dead_code)]
pub fn sidecar_path(media_path: &str, format: SidecarFormat) -> String {
    format!("{}.{}", media_path, format.extension())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-metadata-sidecar-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn test_sidecar_format_extension() {
        assert_eq!(SidecarFormat::XmpSidecar.extension(), "xmp");
        assert_eq!(SidecarFormat::JsonSidecar.extension(), "json");
        assert_eq!(SidecarFormat::YamlSidecar.extension(), "yaml");
        assert_eq!(SidecarFormat::CsvSidecar.extension(), "csv");
    }

    #[test]
    fn test_sidecar_path() {
        let p = sidecar_path("/media/clip.mp4", SidecarFormat::XmpSidecar);
        assert_eq!(p, "/media/clip.mp4.xmp");
        let p2 = sidecar_path("/media/clip.mp4", SidecarFormat::JsonSidecar);
        assert_eq!(p2, "/media/clip.mp4.json");
    }

    #[test]
    fn test_sidecar_new_empty() {
        let p = tmp_path("test.xmp");
        let s = SidecarFile::new(&p, SidecarFormat::XmpSidecar);
        assert_eq!(s.path, p);
        assert!(s.content.is_empty());
    }

    #[test]
    fn test_sidecar_set_get() {
        let mut s = SidecarFile::new(&tmp_path("test.json"), SidecarFormat::JsonSidecar);
        s.set("title", "My Video");
        assert_eq!(s.get("title"), Some("My Video"));
        assert_eq!(s.get("missing"), None);
    }

    #[test]
    fn test_serialize_json() {
        let mut s = SidecarFile::new(&tmp_path("test.json"), SidecarFormat::JsonSidecar);
        s.set("title", "Test");
        let out = s.serialize();
        assert!(out.contains("\"title\""));
        assert!(out.contains("\"Test\""));
    }

    #[test]
    fn test_serialize_yaml() {
        let mut s = SidecarFile::new(&tmp_path("test.yaml"), SidecarFormat::YamlSidecar);
        s.set("author", "Alice");
        let out = s.serialize();
        assert!(out.contains("author: Alice"));
    }

    #[test]
    fn test_serialize_csv() {
        let mut s = SidecarFile::new(&tmp_path("test.csv"), SidecarFormat::CsvSidecar);
        s.set("title", "Movie");
        s.set("year", "2024");
        let out = s.serialize();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_serialize_xmp() {
        let mut s = SidecarFile::new(&tmp_path("test.xmp"), SidecarFormat::XmpSidecar);
        s.set("title", "Test Movie");
        let out = s.serialize();
        assert!(out.contains("<dc:title>Test Movie</dc:title>"));
        assert!(out.contains("<?xpacket"));
    }

    #[test]
    fn test_xmp_escaping() {
        let mut s = SidecarFile::new(&tmp_path("test.xmp"), SidecarFormat::XmpSidecar);
        s.set("description", "A & B <example>");
        let out = s.serialize();
        assert!(out.contains("&amp;"));
        assert!(out.contains("&lt;"));
    }

    #[test]
    fn test_from_str_yaml() {
        let yaml = "title: Hello World\nauthor: Bob";
        let s = SidecarFile::from_str(yaml, SidecarFormat::YamlSidecar);
        assert_eq!(s.get("title"), Some("Hello World"));
        assert_eq!(s.get("author"), Some("Bob"));
    }

    #[test]
    fn test_from_str_csv() {
        let csv = "title,year\nMyFilm,2023";
        let s = SidecarFile::from_str(csv, SidecarFormat::CsvSidecar);
        assert_eq!(s.get("title"), Some("MyFilm"));
        assert_eq!(s.get("year"), Some("2023"));
    }

    #[test]
    fn test_from_str_xmp_roundtrip() {
        let mut s = SidecarFile::new(&tmp_path("r.xmp"), SidecarFormat::XmpSidecar);
        s.set("title", "Roundtrip");
        s.set("creator", "Alice");
        let serialized = s.serialize();
        let parsed = SidecarFile::from_str(&serialized, SidecarFormat::XmpSidecar);
        assert_eq!(parsed.get("title"), Some("Roundtrip"));
        assert_eq!(parsed.get("creator"), Some("Alice"));
    }

    #[test]
    fn test_find_sidecar_nonexistent() {
        // A path that certainly doesn't exist should return None
        let result = find_sidecar("/nonexistent/path/clip.mp4");
        assert!(result.is_none());
    }
}
