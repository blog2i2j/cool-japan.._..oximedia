//! Matroska attachments (fonts, cover art, etc.).
//!
//! Provides attachment handling for Matroska/WebM containers.

#![forbid(unsafe_code)]

use bytes::Bytes;
use oximedia_core::{OxiError, OxiResult};

/// An attached file in Matroska.
#[derive(Debug, Clone)]
pub struct MatroskaAttachment {
    /// Unique file ID.
    pub uid: u64,
    /// Filename.
    pub filename: String,
    /// MIME type.
    pub mime_type: String,
    /// File data.
    pub data: Bytes,
    /// File description.
    pub description: Option<String>,
}

impl MatroskaAttachment {
    /// Creates a new attachment.
    #[must_use]
    pub fn new(
        uid: u64,
        filename: impl Into<String>,
        mime_type: impl Into<String>,
        data: Bytes,
    ) -> Self {
        Self {
            uid,
            filename: filename.into(),
            mime_type: mime_type.into(),
            data,
            description: None,
        }
    }

    /// Sets the description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Returns the file size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns true if this is an image.
    #[must_use]
    pub fn is_image(&self) -> bool {
        self.mime_type.starts_with("image/")
    }

    /// Returns true if this is a font.
    #[must_use]
    pub fn is_font(&self) -> bool {
        self.mime_type.starts_with("font/")
            || self.mime_type.starts_with("application/x-font-")
            || self.mime_type.contains("truetype")
            || self.mime_type.contains("opentype")
    }

    /// Returns the file extension from the filename.
    #[must_use]
    pub fn extension(&self) -> Option<&str> {
        self.filename.rsplit('.').next()
    }
}

/// Collection of Matroska attachments.
#[derive(Debug, Clone)]
pub struct MatroskaAttachments {
    attachments: Vec<MatroskaAttachment>,
}

impl MatroskaAttachments {
    /// Creates a new attachments collection.
    #[must_use]
    pub fn new() -> Self {
        Self {
            attachments: Vec::new(),
        }
    }

    /// Adds an attachment.
    pub fn add(&mut self, attachment: MatroskaAttachment) {
        self.attachments.push(attachment);
    }

    /// Returns all attachments.
    #[must_use]
    pub fn attachments(&self) -> &[MatroskaAttachment] {
        &self.attachments
    }

    /// Finds an attachment by UID.
    #[must_use]
    pub fn find_by_uid(&self, uid: u64) -> Option<&MatroskaAttachment> {
        self.attachments.iter().find(|a| a.uid == uid)
    }

    /// Finds an attachment by filename.
    #[must_use]
    pub fn find_by_filename(&self, filename: &str) -> Option<&MatroskaAttachment> {
        self.attachments.iter().find(|a| a.filename == filename)
    }

    /// Returns all image attachments.
    #[must_use]
    pub fn images(&self) -> Vec<&MatroskaAttachment> {
        self.attachments.iter().filter(|a| a.is_image()).collect()
    }

    /// Returns all font attachments.
    #[must_use]
    pub fn fonts(&self) -> Vec<&MatroskaAttachment> {
        self.attachments.iter().filter(|a| a.is_font()).collect()
    }

    /// Returns the total size of all attachments.
    #[must_use]
    pub fn total_size(&self) -> usize {
        self.attachments.iter().map(MatroskaAttachment::size).sum()
    }

    /// Validates the attachments.
    ///
    /// # Errors
    ///
    /// Returns `Err` if duplicate UIDs are found.
    pub fn validate(&self) -> OxiResult<()> {
        // Check for duplicate UIDs
        let mut uids = std::collections::HashSet::new();
        for attachment in &self.attachments {
            if !uids.insert(attachment.uid) {
                return Err(OxiError::InvalidData(format!(
                    "Duplicate attachment UID: {}",
                    attachment.uid
                )));
            }
        }
        Ok(())
    }

    /// Returns the number of attachments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.attachments.len()
    }

    /// Returns true if there are no attachments.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.attachments.is_empty()
    }
}

impl Default for MatroskaAttachments {
    fn default() -> Self {
        Self::new()
    }
}

/// Common MIME types for attachments.
pub struct MimeTypes;

impl MimeTypes {
    /// JPEG image.
    pub const JPEG: &'static str = "image/jpeg";
    /// PNG image.
    pub const PNG: &'static str = "image/png";
    /// WebP image.
    pub const WEBP: &'static str = "image/webp";
    /// TrueType font.
    pub const TTF: &'static str = "font/ttf";
    /// OpenType font.
    pub const OTF: &'static str = "font/otf";
    /// WOFF font.
    pub const WOFF: &'static str = "font/woff";
    /// WOFF2 font.
    pub const WOFF2: &'static str = "font/woff2";
    /// Generic binary data.
    pub const BINARY: &'static str = "application/octet-stream";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matroska_attachment() {
        let data = Bytes::from_static(b"test data");
        let attachment = MatroskaAttachment::new(1, "cover.jpg", MimeTypes::JPEG, data)
            .with_description("Cover art");

        assert_eq!(attachment.uid, 1);
        assert_eq!(attachment.filename, "cover.jpg");
        assert_eq!(attachment.mime_type, MimeTypes::JPEG);
        assert_eq!(attachment.size(), 9);
        assert!(attachment.is_image());
        assert!(!attachment.is_font());
        assert_eq!(attachment.extension(), Some("jpg"));
        assert_eq!(attachment.description, Some("Cover art".into()));
    }

    #[test]
    fn test_font_detection() {
        let data = Bytes::new();
        let font = MatroskaAttachment::new(1, "font.ttf", MimeTypes::TTF, data);
        assert!(font.is_font());
        assert!(!font.is_image());
    }

    #[test]
    fn test_matroska_attachments() {
        let mut attachments = MatroskaAttachments::new();

        let cover = MatroskaAttachment::new(
            1,
            "cover.jpg",
            MimeTypes::JPEG,
            Bytes::from_static(b"image"),
        );
        let font =
            MatroskaAttachment::new(2, "font.ttf", MimeTypes::TTF, Bytes::from_static(b"font"));

        attachments.add(cover);
        attachments.add(font);

        assert_eq!(attachments.len(), 2);
        assert_eq!(attachments.images().len(), 1);
        assert_eq!(attachments.fonts().len(), 1);
        assert_eq!(attachments.total_size(), 9);

        let found = attachments.find_by_filename("cover.jpg");
        assert!(found.is_some());
        assert_eq!(found.expect("operation should succeed").uid, 1);
    }

    #[test]
    fn test_validate_attachments() {
        let mut attachments = MatroskaAttachments::new();
        assert!(attachments.validate().is_ok());

        attachments.add(MatroskaAttachment::new(
            1,
            "a.jpg",
            MimeTypes::JPEG,
            Bytes::new(),
        ));
        assert!(attachments.validate().is_ok());

        // Duplicate UID
        attachments.add(MatroskaAttachment::new(
            1,
            "b.jpg",
            MimeTypes::JPEG,
            Bytes::new(),
        ));
        assert!(attachments.validate().is_err());
    }
}
