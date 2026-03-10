//! Metadata reading from container formats.

use async_trait::async_trait;
use oximedia_core::OxiResult;
use oximedia_io::MediaSource;

use super::tags::TagMap;
use super::util::MediaSourceExt;
use super::vorbis::VorbisComments;
use crate::demux::flac::metadata::BlockType;
use crate::demux::matroska::types::{SimpleTag, Tag};
use crate::ContainerFormat;

/// Trait for reading metadata from a media source.
#[async_trait]
pub trait MetadataReader: Sized {
    /// Reads metadata from the source.
    ///
    /// # Errors
    ///
    /// Returns an error if reading or parsing fails.
    async fn read<R: MediaSource>(source: R) -> OxiResult<TagMap>;
}

/// FLAC metadata reader.
pub struct FlacMetadataReader;

#[async_trait]
impl MetadataReader for FlacMetadataReader {
    async fn read<R: MediaSource>(mut source: R) -> OxiResult<TagMap> {
        // Read FLAC magic
        let mut magic = [0u8; 4];
        source.read_exact(&mut magic).await?;

        if &magic != b"fLaC" {
            return Err(oximedia_core::OxiError::UnknownFormat);
        }

        // Read metadata blocks until we find Vorbis comments
        loop {
            let mut header = [0u8; 4];
            source.read_exact(&mut header).await?;

            let is_last = header[0] & 0x80 != 0;
            let block_type = BlockType::from(header[0]);
            let length = u32::from_be_bytes([0, header[1], header[2], header[3]]);

            let mut block_data = vec![0u8; length as usize];
            source.read_exact(&mut block_data).await?;

            if block_type == BlockType::VorbisComment {
                let comments = VorbisComments::parse(&block_data)?;
                return Ok(comments.tags);
            }

            if is_last {
                break;
            }
        }

        // No Vorbis comment block found
        Ok(TagMap::new())
    }
}

/// Matroska metadata reader.
pub struct MatroskaMetadataReader;

impl MatroskaMetadataReader {
    /// Converts Matroska tags to a unified tag map.
    #[must_use]
    pub fn convert_tags(tags: &[Tag]) -> TagMap {
        let mut tag_map = TagMap::new();

        for tag in tags {
            Self::convert_simple_tags(&tag.simple_tags, &mut tag_map);
        }

        tag_map
    }

    /// Recursively converts simple tags.
    fn convert_simple_tags(simple_tags: &[SimpleTag], tag_map: &mut TagMap) {
        for simple_tag in simple_tags {
            if let Some(ref value) = simple_tag.string {
                tag_map.add(&simple_tag.name, value.clone());
            }

            // Handle nested tags
            if !simple_tag.children.is_empty() {
                Self::convert_simple_tags(&simple_tag.children, tag_map);
            }
        }
    }
}

/// Determines the container format from magic bytes.
///
/// # Errors
///
/// Returns an error if the format cannot be determined.
pub fn detect_format(magic: &[u8]) -> OxiResult<ContainerFormat> {
    if magic.len() < 4 {
        return Err(oximedia_core::OxiError::UnknownFormat);
    }

    if &magic[..4] == b"fLaC" {
        Ok(ContainerFormat::Flac)
    } else if magic.len() >= 4 && &magic[..4] == b"OggS" {
        Ok(ContainerFormat::Ogg)
    } else if magic.len() >= 4
        && magic[0] == 0x1A
        && magic[1] == 0x45
        && magic[2] == 0xDF
        && magic[3] == 0xA3
    {
        // EBML header - could be Matroska or WebM
        Ok(ContainerFormat::Matroska)
    } else {
        Err(oximedia_core::OxiError::UnknownFormat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_io::MemorySource;

    #[tokio::test]
    async fn test_detect_format_flac() {
        let format = detect_format(b"fLaC").expect("operation should succeed");
        assert_eq!(format, ContainerFormat::Flac);
    }

    #[tokio::test]
    async fn test_detect_format_ogg() {
        let format = detect_format(b"OggS").expect("operation should succeed");
        assert_eq!(format, ContainerFormat::Ogg);
    }

    #[tokio::test]
    async fn test_detect_format_matroska() {
        let magic = [0x1A, 0x45, 0xDF, 0xA3];
        let format = detect_format(&magic).expect("operation should succeed");
        assert_eq!(format, ContainerFormat::Matroska);
    }

    #[tokio::test]
    async fn test_detect_format_invalid() {
        assert!(detect_format(b"UNKN").is_err());
    }

    #[tokio::test]
    async fn test_flac_metadata_reader_no_magic() {
        let data = b"NOTF";
        let source = MemorySource::new(bytes::Bytes::from_static(data));
        let result = FlacMetadataReader::read(source).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_matroska_convert_empty_tags() {
        let tags = vec![];
        let tag_map = MatroskaMetadataReader::convert_tags(&tags);
        assert!(tag_map.is_empty());
    }

    #[test]
    fn test_matroska_convert_simple_tags() {
        use crate::demux::matroska::types::{SimpleTag, Tag, TagTargets};

        let tags = vec![Tag {
            targets: TagTargets::default(),
            simple_tags: vec![
                SimpleTag {
                    name: "TITLE".to_string(),
                    string: Some("Test Title".to_string()),
                    ..Default::default()
                },
                SimpleTag {
                    name: "ARTIST".to_string(),
                    string: Some("Test Artist".to_string()),
                    ..Default::default()
                },
            ],
        }];

        let tag_map = MatroskaMetadataReader::convert_tags(&tags);
        assert_eq!(tag_map.get_text("TITLE"), Some("Test Title"));
        assert_eq!(tag_map.get_text("ARTIST"), Some("Test Artist"));
    }

    #[test]
    fn test_matroska_convert_nested_tags() {
        use crate::demux::matroska::types::{SimpleTag, Tag, TagTargets};

        let tags = vec![Tag {
            targets: TagTargets::default(),
            simple_tags: vec![SimpleTag {
                name: "PARENT".to_string(),
                string: Some("Parent Value".to_string()),
                children: vec![SimpleTag {
                    name: "CHILD".to_string(),
                    string: Some("Child Value".to_string()),
                    ..Default::default()
                }],
                ..Default::default()
            }],
        }];

        let tag_map = MatroskaMetadataReader::convert_tags(&tags);
        assert_eq!(tag_map.get_text("PARENT"), Some("Parent Value"));
        assert_eq!(tag_map.get_text("CHILD"), Some("Child Value"));
    }
}
