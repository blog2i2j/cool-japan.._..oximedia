//! Vorbis comment format utilities.
//!
//! Vorbis comments are used by:
//! - FLAC (in metadata blocks)
//! - Ogg Vorbis and Ogg Opus (in codec headers)
//!
//! Format specification: <https://www.xiph.org/vorbis/doc/v-comment.html>

use oximedia_core::{OxiError, OxiResult};

use super::tags::{TagMap, TagValue};

/// Vorbis comment data structure.
///
/// Vorbis comments consist of a vendor string followed by
/// a list of `KEY=VALUE` pairs.
#[derive(Clone, Debug, Default)]
pub struct VorbisComments {
    /// Vendor string (encoder identification).
    pub vendor: String,
    /// Tag map containing all comments.
    pub tags: TagMap,
}

impl VorbisComments {
    /// Creates a new empty Vorbis comments structure.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates Vorbis comments with a vendor string.
    #[must_use]
    pub fn with_vendor(vendor: impl Into<String>) -> Self {
        Self {
            vendor: vendor.into(),
            tags: TagMap::new(),
        }
    }

    /// Parses Vorbis comments from raw data.
    ///
    /// # Format
    ///
    /// - 4 bytes: vendor string length (little-endian)
    /// - N bytes: vendor string (UTF-8)
    /// - 4 bytes: comment count (little-endian)
    /// - For each comment:
    ///   - 4 bytes: comment length (little-endian)
    ///   - N bytes: comment string "KEY=VALUE" (UTF-8)
    ///
    /// # Errors
    ///
    /// Returns an error if the data is truncated or malformed.
    pub fn parse(data: &[u8]) -> OxiResult<Self> {
        if data.len() < 8 {
            return Err(OxiError::UnexpectedEof);
        }

        let mut offset = 0;

        // Vendor string length (little-endian)
        let vendor_len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        if offset + vendor_len > data.len() {
            return Err(OxiError::UnexpectedEof);
        }

        let vendor = String::from_utf8_lossy(&data[offset..offset + vendor_len]).into_owned();
        offset += vendor_len;

        if offset + 4 > data.len() {
            return Err(OxiError::UnexpectedEof);
        }

        // Comment count (little-endian)
        let comment_count = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        let mut tags = TagMap::new();

        for _ in 0..comment_count {
            if offset + 4 > data.len() {
                break;
            }

            let comment_len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + comment_len > data.len() {
                break;
            }

            let comment = String::from_utf8_lossy(&data[offset..offset + comment_len]);
            offset += comment_len;

            // Parse KEY=VALUE
            if let Some((key, value)) = comment.split_once('=') {
                tags.add(key, value.to_string());
            }
        }

        Ok(Self { vendor, tags })
    }

    /// Encodes Vorbis comments to raw bytes.
    ///
    /// Returns the encoded data in the Vorbis comment format.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn encode(&self) -> Vec<u8> {
        let mut data = Vec::new();

        // Vendor string
        let vendor_bytes = self.vendor.as_bytes();
        data.extend_from_slice(&(vendor_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(vendor_bytes);

        // Count comments
        let comment_count = self.tags.iter().filter(|(_, v)| v.is_text()).count();
        data.extend_from_slice(&(comment_count as u32).to_le_bytes());

        // Write comments
        for (key, value) in self.tags.iter() {
            if let Some(text) = value.as_text() {
                let comment = format!("{key}={text}");
                let comment_bytes = comment.as_bytes();
                data.extend_from_slice(&(comment_bytes.len() as u32).to_le_bytes());
                data.extend_from_slice(comment_bytes);
            }
        }

        data
    }

    /// Returns true if there are no comments (vendor string may still exist).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// Returns the number of comment entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tags.len()
    }
}

/// Builds Vorbis comments incrementally.
pub struct VorbisCommentsBuilder {
    vendor: String,
    tags: TagMap,
}

impl VorbisCommentsBuilder {
    /// Creates a new builder with default vendor string.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vendor: "OxiMedia".to_string(),
            tags: TagMap::new(),
        }
    }

    /// Sets the vendor string.
    #[must_use]
    pub fn vendor(mut self, vendor: impl Into<String>) -> Self {
        self.vendor = vendor.into();
        self
    }

    /// Adds a tag.
    #[must_use]
    pub fn tag(mut self, key: impl AsRef<str>, value: impl Into<TagValue>) -> Self {
        self.tags.add(key, value);
        self
    }

    /// Builds the Vorbis comments.
    #[must_use]
    pub fn build(self) -> VorbisComments {
        VorbisComments {
            vendor: self.vendor,
            tags: self.tags,
        }
    }
}

impl Default for VorbisCommentsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vorbis_comments_new() {
        let vc = VorbisComments::new();
        assert!(vc.vendor.is_empty());
        assert!(vc.is_empty());
    }

    #[test]
    fn test_vorbis_comments_with_vendor() {
        let vc = VorbisComments::with_vendor("TestVendor");
        assert_eq!(vc.vendor, "TestVendor");
    }

    #[test]
    fn test_vorbis_comments_parse_empty() {
        let mut data = Vec::new();
        data.extend_from_slice(&0u32.to_le_bytes()); // vendor length
        data.extend_from_slice(&0u32.to_le_bytes()); // comment count

        let vc = VorbisComments::parse(&data).expect("operation should succeed");
        assert!(vc.vendor.is_empty());
        assert!(vc.is_empty());
    }

    #[test]
    fn test_vorbis_comments_parse_with_tags() {
        let mut data = Vec::new();

        // Vendor: "Test"
        let vendor = b"Test";
        data.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        data.extend_from_slice(vendor);

        // 2 comments
        data.extend_from_slice(&2u32.to_le_bytes());

        // TITLE=Test Title
        let comment1 = b"TITLE=Test Title";
        data.extend_from_slice(&(comment1.len() as u32).to_le_bytes());
        data.extend_from_slice(comment1);

        // ARTIST=Test Artist
        let comment2 = b"ARTIST=Test Artist";
        data.extend_from_slice(&(comment2.len() as u32).to_le_bytes());
        data.extend_from_slice(comment2);

        let vc = VorbisComments::parse(&data).expect("operation should succeed");
        assert_eq!(vc.vendor, "Test");
        assert_eq!(vc.len(), 2);
        assert_eq!(vc.tags.get_text("TITLE"), Some("Test Title"));
        assert_eq!(vc.tags.get_text("ARTIST"), Some("Test Artist"));
    }

    #[test]
    fn test_vorbis_comments_encode() {
        let mut vc = VorbisComments::with_vendor("TestVendor");
        vc.tags.set("TITLE", "Test Title");
        vc.tags.set("ARTIST", "Test Artist");

        let encoded = vc.encode();

        // Parse it back
        let decoded = VorbisComments::parse(&encoded).expect("operation should succeed");
        assert_eq!(decoded.vendor, "TestVendor");
        assert_eq!(decoded.tags.get_text("TITLE"), Some("Test Title"));
        assert_eq!(decoded.tags.get_text("ARTIST"), Some("Test Artist"));
    }

    #[test]
    fn test_vorbis_comments_encode_empty() {
        let vc = VorbisComments::new();
        let encoded = vc.encode();

        let decoded = VorbisComments::parse(&encoded).expect("operation should succeed");
        assert!(decoded.vendor.is_empty());
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_vorbis_comments_builder() {
        let vc = VorbisCommentsBuilder::new()
            .vendor("TestVendor")
            .tag("TITLE", "Test")
            .tag("ARTIST", "Artist")
            .build();

        assert_eq!(vc.vendor, "TestVendor");
        assert_eq!(vc.tags.get_text("TITLE"), Some("Test"));
        assert_eq!(vc.tags.get_text("ARTIST"), Some("Artist"));
    }

    #[test]
    fn test_vorbis_comments_parse_truncated() {
        let data = vec![0, 0, 0]; // Too short
        assert!(VorbisComments::parse(&data).is_err());
    }

    #[test]
    fn test_vorbis_comments_parse_malformed_vendor() {
        let mut data = Vec::new();
        data.extend_from_slice(&100u32.to_le_bytes()); // vendor length > data
        assert!(VorbisComments::parse(&data).is_err());
    }
}
