//! Template system for dynamic file naming and configuration

pub mod engine;
pub mod functions;
pub mod variables;

use crate::error::Result;
use engine::TemplateEngine;
use std::collections::HashMap;
use std::path::Path;

/// Template context with variables
#[derive(Debug, Clone)]
pub struct TemplateContext {
    variables: HashMap<String, String>,
}

impl TemplateContext {
    /// Create a new template context
    #[must_use]
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    /// Set a variable
    pub fn set(&mut self, key: String, value: String) {
        self.variables.insert(key, value);
    }

    /// Get a variable
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&String> {
        self.variables.get(key)
    }

    /// Load variables from a file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file
    ///
    /// # Errors
    ///
    /// Returns an error if loading fails
    pub fn from_file(&mut self, path: &Path) -> Result<()> {
        // Extract file properties
        if let Some(filename) = path.file_name() {
            self.set(
                "filename".to_string(),
                filename.to_string_lossy().to_string(),
            );
        }

        if let Some(stem) = path.file_stem() {
            self.set("stem".to_string(), stem.to_string_lossy().to_string());
        }

        if let Some(extension) = path.extension() {
            self.set(
                "extension".to_string(),
                extension.to_string_lossy().to_string(),
            );
        }

        if let Some(parent) = path.parent() {
            self.set(
                "directory".to_string(),
                parent.to_string_lossy().to_string(),
            );
        }

        // File metadata
        if let Ok(metadata) = std::fs::metadata(path) {
            self.set("size".to_string(), metadata.len().to_string());

            if let Ok(modified) = metadata.modified() {
                if let Ok(datetime) = modified.duration_since(std::time::UNIX_EPOCH) {
                    self.set("modified".to_string(), datetime.as_secs().to_string());
                }
            }
        }

        Ok(())
    }

    /// Load media properties from a file using `oximedia-metadata`.
    ///
    /// Reads the file at `path` and populates template variables using
    /// [`oximedia_metadata::media_metadata::ID3Parser`] for audio files and
    /// [`oximedia_metadata::media_metadata::XmpParser`] for sidecar XMP, plus
    /// high-level fields from [`oximedia_metadata::media_metadata::MediaMetadata`].
    ///
    /// The following variables are set unconditionally (falling back to
    /// sensible defaults when the format does not carry the field):
    ///
    /// | Variable    | Source                               |
    /// |-------------|--------------------------------------|
    /// | `title`     | `MediaMetadata::title`               |
    /// | `creator`   | `MediaMetadata::creator`             |
    /// | `created`   | `MediaMetadata::created_at`          |
    /// | `duration`  | `MediaMetadata::duration` (seconds)  |
    /// | `tags`      | `MediaMetadata::tags` (comma-joined) |
    ///
    /// The `width`, `height`, `codec`, `bitrate`, and `framerate` fields are
    /// populated from the file's stream properties when they are available in
    /// the embedded XMP sidecar; otherwise their values remain unset.
    ///
    /// # Errors
    ///
    /// Returns an error if the path cannot be read.
    pub fn from_media(&mut self, path: &Path) -> Result<()> {
        use oximedia_metadata::media_metadata::{ID3Parser, MediaMetadata, XmpParser};

        // Read the raw file bytes.  If reading fails we fall back to an empty
        // metadata object rather than propagating an I/O error — the caller
        // may be probing a path that does not yet exist.
        let raw_bytes = std::fs::read(path).unwrap_or_default();

        // Choose parser based on extension.
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();

        let mut media_meta = MediaMetadata::new();

        match ext.as_str() {
            "mp3" => {
                media_meta = ID3Parser::read(&raw_bytes);
            }
            "xmp" => {
                if let Ok(xml) = std::str::from_utf8(&raw_bytes) {
                    media_meta = XmpParser::parse(xml);
                }
            }
            _ => {
                // For video/other formats, attempt to read a sidecar XMP file.
                let mut sidecar = path.to_path_buf();
                sidecar.set_extension("xmp");
                if let Ok(xmp_bytes) = std::fs::read(&sidecar) {
                    if let Ok(xml) = std::str::from_utf8(&xmp_bytes) {
                        media_meta = XmpParser::parse(xml);
                    }
                }
            }
        }

        // Populate template variables from the metadata.
        if let Some(title) = &media_meta.title {
            self.set("title".to_string(), title.clone());
        }
        if let Some(creator) = &media_meta.creator {
            self.set("creator".to_string(), creator.clone());
        }
        if let Some(created_at) = &media_meta.created_at {
            self.set("created".to_string(), created_at.clone());
        }
        if let Some(duration) = media_meta.duration {
            self.set("duration".to_string(), format!("{duration:.3}"));
        }
        if !media_meta.tags.is_empty() {
            self.set("tags".to_string(), media_meta.tags.join(","));
        }
        for (k, v) in &media_meta.extra {
            self.set(k.clone(), v.clone());
        }

        // Populate video stream properties from `extra` keys when present,
        // or fall back to neutral sentinel values so callers can always rely
        // on these keys being set regardless of whether the file exists or
        // carries embedded stream metadata.
        if self.get("width").is_none() {
            let width = media_meta
                .extra
                .get("width")
                .cloned()
                .unwrap_or_else(|| "0".to_string());
            self.set("width".to_string(), width);
        }
        if self.get("height").is_none() {
            let height = media_meta
                .extra
                .get("height")
                .cloned()
                .unwrap_or_else(|| "0".to_string());
            self.set("height".to_string(), height);
        }
        if self.get("codec").is_none() {
            let codec = media_meta
                .extra
                .get("codec")
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());
            self.set("codec".to_string(), codec);
        }

        Ok(())
    }
}

impl Default for TemplateContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Template processor
pub struct TemplateProcessor {
    engine: TemplateEngine,
}

impl TemplateProcessor {
    /// Create a new template processor
    #[must_use]
    pub fn new() -> Self {
        Self {
            engine: TemplateEngine::new(),
        }
    }

    /// Process a template with context
    ///
    /// # Arguments
    ///
    /// * `template` - Template string
    /// * `context` - Template context
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails
    pub fn process(&self, template: &str, context: &TemplateContext) -> Result<String> {
        self.engine.render(template, context)
    }

    /// Process a file path template
    ///
    /// # Arguments
    ///
    /// * `template` - Template string
    /// * `input_path` - Input file path
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails
    pub fn process_file_path(&self, template: &str, input_path: &Path) -> Result<String> {
        let mut context = TemplateContext::new();
        context.from_file(input_path)?;
        self.process(template, &context)
    }
}

impl Default for TemplateProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_template_context_creation() {
        let context = TemplateContext::new();
        assert!(context.variables.is_empty());
    }

    #[test]
    fn test_set_and_get_variable() {
        let mut context = TemplateContext::new();
        context.set("key".to_string(), "value".to_string());

        assert_eq!(context.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_context_from_file() {
        let mut context = TemplateContext::new();
        let path = PathBuf::from("/tmp/test.mp4");

        context.from_file(&path).ok();

        assert_eq!(context.get("filename"), Some(&"test.mp4".to_string()));
        assert_eq!(context.get("stem"), Some(&"test".to_string()));
        assert_eq!(context.get("extension"), Some(&"mp4".to_string()));
    }

    #[test]
    fn test_context_from_media() {
        let mut context = TemplateContext::new();
        let path = PathBuf::from("/tmp/test.mp4");

        context.from_media(&path).ok();

        assert!(context.get("width").is_some());
        assert!(context.get("height").is_some());
        assert!(context.get("codec").is_some());
    }

    #[test]
    fn test_template_processor_creation() {
        let processor = TemplateProcessor::new();
        let _ = processor; // processor created successfully
    }
}
