//! IIIF (International Image Interoperability Framework) manifest generation.
//!
//! Generates IIIF Presentation API 3.0 manifests for archived media objects,
//! enabling interoperability with IIIF-compatible viewers (Universal Viewer,
//! Mirador, etc.).
//!
//! Reference: <https://iiif.io/api/presentation/3.0/>

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// IIIF API context URI for Presentation API 3.0.
const IIIF_CONTEXT: &str = "http://iiif.io/api/presentation/3/context.json";

/// A language-tagged string value as used throughout IIIF.
///
/// Maps a BCP 47 language tag (e.g. `"en"`) to one or more string values.
/// The special key `"none"` is used for language-neutral values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageMap(pub HashMap<String, Vec<String>>);

impl LanguageMap {
    /// Creates a language map with a single value tagged `"none"`.
    #[must_use]
    pub fn plain(value: impl Into<String>) -> Self {
        let mut map = HashMap::new();
        map.insert("none".to_string(), vec![value.into()]);
        Self(map)
    }

    /// Creates a language map with a value for a specific language tag.
    #[must_use]
    pub fn with_lang(lang: impl Into<String>, value: impl Into<String>) -> Self {
        let mut map = HashMap::new();
        map.insert(lang.into(), vec![value.into()]);
        Self(map)
    }

    /// Adds an additional value for the given language.
    pub fn add(&mut self, lang: impl Into<String>, value: impl Into<String>) {
        self.0.entry(lang.into()).or_default().push(value.into());
    }
}

/// A IIIF resource body (image or audio/video service endpoint).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IiifBody {
    /// `id` URI for this body resource.
    pub id: String,
    /// Resource type (e.g. `"Image"`, `"Video"`, `"Sound"`).
    #[serde(rename = "type")]
    pub type_: String,
    /// MIME type (e.g. `"image/jpeg"`, `"video/mp4"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Width in pixels (images/video).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    /// Height in pixels (images/video).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    /// Duration in seconds (audio/video).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
}

/// A IIIF Annotation (associates a resource body with a canvas target).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IiifAnnotation {
    /// `id` URI.
    pub id: String,
    /// Always `"Annotation"`.
    #[serde(rename = "type")]
    pub type_: String,
    /// Motivation (e.g. `"painting"`, `"supplementing"`).
    pub motivation: String,
    /// The resource body being annotated onto the canvas.
    pub body: IiifBody,
    /// Target canvas URI (may include spatial/temporal fragment).
    pub target: String,
}

/// A IIIF Annotation Page groups one or more annotations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IiifAnnotationPage {
    /// `id` URI.
    pub id: String,
    /// Always `"AnnotationPage"`.
    #[serde(rename = "type")]
    pub type_: String,
    /// The annotations in this page.
    pub items: Vec<IiifAnnotation>,
}

/// A IIIF Canvas represents a virtual container for a single view/frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IiifCanvas {
    /// `id` URI (typically `<manifest_id>/canvas/<n>`).
    pub id: String,
    /// Always `"Canvas"`.
    #[serde(rename = "type")]
    pub type_: String,
    /// Human-readable label.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<LanguageMap>,
    /// Canvas width in pixels (required for image canvases).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    /// Canvas height in pixels (required for image canvases).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    /// Canvas duration in seconds (required for AV canvases).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    /// Annotation pages containing painting annotations.
    pub items: Vec<IiifAnnotationPage>,
}

/// A IIIF Presentation API 3.0 Manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IiifManifest {
    /// JSON-LD context.
    #[serde(rename = "@context")]
    pub context: String,
    /// `id` URI — must be an HTTP(S) URL that dereferences to this manifest.
    pub id: String,
    /// Always `"Manifest"`.
    #[serde(rename = "type")]
    pub type_: String,
    /// Human-readable label.
    pub label: LanguageMap,
    /// Optional summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<LanguageMap>,
    /// Rights statement URI (e.g. a Creative Commons URL).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rights: Option<String>,
    /// Required statement (attribution).
    #[serde(rename = "requiredStatement", skip_serializing_if = "Option::is_none")]
    pub required_statement: Option<RequiredStatement>,
    /// Metadata key-value pairs displayed in IIIF viewers.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub metadata: Vec<MetadataEntry>,
    /// Canvases forming the sequence of views.
    pub items: Vec<IiifCanvas>,
}

/// A metadata key-value pair for display in viewers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataEntry {
    /// Field label.
    pub label: LanguageMap,
    /// Field value.
    pub value: LanguageMap,
}

/// A required statement (used for attribution/license notices).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequiredStatement {
    /// Label (e.g. `"Attribution"`).
    pub label: LanguageMap,
    /// Value (the attribution text or HTML).
    pub value: LanguageMap,
}

impl IiifManifest {
    /// Serialises this manifest to a pretty-printed JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if JSON serialisation fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Saves this manifest to a file as pretty-printed JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = self.to_json().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;
        std::fs::write(path, json)
    }
}

/// Builds a IIIF Presentation API 3.0 manifest for an archived object.
pub struct IiifManifestBuilder {
    manifest_id: String,
    label: LanguageMap,
    summary: Option<LanguageMap>,
    rights: Option<String>,
    required_statement: Option<RequiredStatement>,
    metadata: Vec<MetadataEntry>,
    canvases: Vec<IiifCanvas>,
    canvas_counter: u32,
    annotation_counter: u32,
}

impl IiifManifestBuilder {
    /// Creates a new builder.
    ///
    /// * `manifest_id` – the HTTP(S) URI at which this manifest will be served.
    /// * `label`       – human-readable title of the object.
    #[must_use]
    pub fn new(manifest_id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            manifest_id: manifest_id.into(),
            label: LanguageMap::plain(label),
            summary: None,
            rights: None,
            required_statement: None,
            metadata: Vec::new(),
            canvases: Vec::new(),
            canvas_counter: 0,
            annotation_counter: 0,
        }
    }

    /// Sets the summary (displayed in viewers as a short description).
    #[must_use]
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(LanguageMap::plain(summary));
        self
    }

    /// Sets the rights statement URI.
    #[must_use]
    pub fn with_rights(mut self, rights_uri: impl Into<String>) -> Self {
        self.rights = Some(rights_uri.into());
        self
    }

    /// Adds a metadata key-value pair.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.push(MetadataEntry {
            label: LanguageMap::plain(key),
            value: LanguageMap::plain(value),
        });
        self
    }

    /// Sets the required attribution statement.
    #[must_use]
    pub fn with_attribution(mut self, label: impl Into<String>, text: impl Into<String>) -> Self {
        self.required_statement = Some(RequiredStatement {
            label: LanguageMap::plain(label),
            value: LanguageMap::plain(text),
        });
        self
    }

    /// Adds an image canvas.
    ///
    /// * `image_uri`  – the URI of the image resource.
    /// * `mime_type`  – MIME type (e.g. `"image/tiff"`).
    /// * `width`      – image width in pixels.
    /// * `height`     – image height in pixels.
    /// * `canvas_label` – optional human-readable label for this canvas.
    #[must_use]
    pub fn add_image_canvas(
        mut self,
        image_uri: impl Into<String>,
        mime_type: impl Into<String>,
        width: u32,
        height: u32,
        canvas_label: Option<&str>,
    ) -> Self {
        self.canvas_counter += 1;
        self.annotation_counter += 1;
        let n = self.canvas_counter;
        let a = self.annotation_counter;
        let canvas_id = format!("{}/canvas/{n}", self.manifest_id);
        let annotation_page_id = format!("{canvas_id}/page/1");
        let annotation_id = format!("{canvas_id}/annotation/{a}");
        let image_uri: String = image_uri.into();

        let body = IiifBody {
            id: image_uri,
            type_: "Image".to_string(),
            format: Some(mime_type.into()),
            width: Some(width),
            height: Some(height),
            duration: None,
        };

        let annotation = IiifAnnotation {
            id: annotation_id,
            type_: "Annotation".to_string(),
            motivation: "painting".to_string(),
            body,
            target: canvas_id.clone(),
        };

        let page = IiifAnnotationPage {
            id: annotation_page_id,
            type_: "AnnotationPage".to_string(),
            items: vec![annotation],
        };

        let canvas = IiifCanvas {
            id: canvas_id,
            type_: "Canvas".to_string(),
            label: canvas_label.map(LanguageMap::plain),
            width: Some(width),
            height: Some(height),
            duration: None,
            items: vec![page],
        };

        self.canvases.push(canvas);
        self
    }

    /// Adds an audio or video canvas.
    ///
    /// * `media_uri`  – the URI of the audio/video resource.
    /// * `type_`      – `"Sound"` or `"Video"`.
    /// * `mime_type`  – MIME type (e.g. `"video/mp4"`, `"audio/flac"`).
    /// * `duration`   – duration in seconds.
    #[must_use]
    pub fn add_av_canvas(
        mut self,
        media_uri: impl Into<String>,
        type_: impl Into<String>,
        mime_type: impl Into<String>,
        duration: f64,
        canvas_label: Option<&str>,
    ) -> Self {
        self.canvas_counter += 1;
        self.annotation_counter += 1;
        let n = self.canvas_counter;
        let a = self.annotation_counter;
        let canvas_id = format!("{}/canvas/{n}", self.manifest_id);
        let annotation_page_id = format!("{canvas_id}/page/1");
        let annotation_id = format!("{canvas_id}/annotation/{a}");
        let type_str = type_.into();
        let media_uri: String = media_uri.into();

        let body = IiifBody {
            id: media_uri,
            type_: type_str.clone(),
            format: Some(mime_type.into()),
            width: None,
            height: None,
            duration: Some(duration),
        };

        let annotation = IiifAnnotation {
            id: annotation_id,
            type_: "Annotation".to_string(),
            motivation: "painting".to_string(),
            body,
            target: canvas_id.clone(),
        };

        let page = IiifAnnotationPage {
            id: annotation_page_id,
            type_: "AnnotationPage".to_string(),
            items: vec![annotation],
        };

        let canvas = IiifCanvas {
            id: canvas_id,
            type_: "Canvas".to_string(),
            label: canvas_label.map(LanguageMap::plain),
            width: None,
            height: None,
            duration: Some(duration),
            items: vec![page],
        };

        self.canvases.push(canvas);
        self
    }

    /// Builds and returns the `IiifManifest`.
    #[must_use]
    pub fn build(self) -> IiifManifest {
        IiifManifest {
            context: IIIF_CONTEXT.to_string(),
            id: self.manifest_id,
            type_: "Manifest".to_string(),
            label: self.label,
            summary: self.summary,
            rights: self.rights,
            required_statement: self.required_statement,
            metadata: self.metadata,
            items: self.canvases,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_json_contains_context() {
        let manifest = IiifManifestBuilder::new(
            "https://example.org/iiif/manifest/1",
            "Test Film",
        )
        .build();
        let json = manifest.to_json().expect("json");
        assert!(json.contains(IIIF_CONTEXT));
        assert!(json.contains("Manifest"));
    }

    #[test]
    fn test_manifest_with_image_canvas() {
        let manifest = IiifManifestBuilder::new(
            "https://example.org/iiif/m/1",
            "Film Still",
        )
        .add_image_canvas(
            "https://example.org/images/still001.tif",
            "image/tiff",
            3840,
            2160,
            Some("Frame 1"),
        )
        .build();

        assert_eq!(manifest.items.len(), 1);
        let canvas = &manifest.items[0];
        assert_eq!(canvas.width, Some(3840));
        assert_eq!(canvas.height, Some(2160));
        assert_eq!(canvas.items[0].items[0].body.type_, "Image");
    }

    #[test]
    fn test_manifest_with_av_canvas() {
        let manifest = IiifManifestBuilder::new(
            "https://example.org/iiif/m/2",
            "News Broadcast",
        )
        .add_av_canvas(
            "https://example.org/media/news.mkv",
            "Video",
            "video/x-matroska",
            1800.0,
            Some("Main Programme"),
        )
        .build();

        assert_eq!(manifest.items.len(), 1);
        let canvas = &manifest.items[0];
        assert_eq!(canvas.duration, Some(1800.0));
        assert_eq!(canvas.items[0].items[0].body.type_, "Video");
    }

    #[test]
    fn test_manifest_metadata_fields() {
        let manifest = IiifManifestBuilder::new("https://example.org/iiif/m/3", "Doc Film")
            .with_metadata("Date", "2024-01-01")
            .with_metadata("Creator", "WGBH")
            .build();
        assert_eq!(manifest.metadata.len(), 2);
    }

    #[test]
    fn test_manifest_rights_and_attribution() {
        let manifest = IiifManifestBuilder::new("https://example.org/iiif/m/4", "Archive Reel")
            .with_rights("https://creativecommons.org/licenses/by/4.0/")
            .with_attribution("Attribution", "© 2024 Archive Corp")
            .build();
        assert!(manifest.rights.is_some());
        assert!(manifest.required_statement.is_some());
    }

    #[test]
    fn test_manifest_save_to_file() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let path = tmp.path().join("manifest.json");

        let manifest = IiifManifestBuilder::new("https://example.org/iiif/m/5", "Test")
            .build();
        manifest.save(&path).expect("save");
        assert!(path.exists());
        let content = std::fs::read_to_string(&path).expect("read");
        assert!(content.contains("Manifest"));
    }

    #[test]
    fn test_language_map_plain() {
        let lm = LanguageMap::plain("Hello");
        assert_eq!(lm.0.get("none").and_then(|v| v.first()).map(String::as_str), Some("Hello"));
    }

    #[test]
    fn test_language_map_with_lang() {
        let lm = LanguageMap::with_lang("en", "Hello");
        assert_eq!(lm.0.get("en").and_then(|v| v.first()).map(String::as_str), Some("Hello"));
    }

    #[test]
    fn test_multi_canvas_manifest() {
        let manifest = IiifManifestBuilder::new("https://example.org/iiif/multi", "Film Stills")
            .add_image_canvas("https://example.org/f1.jpg", "image/jpeg", 1920, 1080, None)
            .add_image_canvas("https://example.org/f2.jpg", "image/jpeg", 1920, 1080, None)
            .add_image_canvas("https://example.org/f3.jpg", "image/jpeg", 1920, 1080, None)
            .build();
        assert_eq!(manifest.items.len(), 3);
        // Canvas IDs must be unique
        let ids: std::collections::HashSet<_> = manifest.items.iter().map(|c| &c.id).collect();
        assert_eq!(ids.len(), 3);
    }
}
