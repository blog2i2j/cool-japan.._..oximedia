//! WASM metadata extraction.
//!
//! This module provides WebAssembly bindings for extracting metadata from
//! media container headers. It supports all major metadata formats including
//! ID3v2, Vorbis Comments, APEv2, iTunes, Matroska, and more.
//!
//! # Format Detection
//!
//! The module auto-detects the metadata format from the input bytes by
//! examining magic bytes and header signatures:
//!
//! - **ID3v2**: Starts with `"ID3"`
//! - **FLAC/Vorbis Comments**: Starts with `"fLaC"` or contains `"\x03vorbis"`
//! - **APEv2**: Starts with `"APETAGEX"`
//! - **iTunes/MP4**: Contains `"ftyp"` atom
//! - **Matroska/WebM**: Starts with EBML header (`0x1A 0x45 0xDF 0xA3`)
//! - **EXIF**: Starts with `"Exif"` or TIFF byte order markers
//!
//! # JavaScript Example
//!
//! ```javascript
//! const result = JSON.parse(oximedia.wasm_parse_metadata(headerBytes));
//! console.log('Format:', result.format);
//! console.log('Title:', result.fields.title);
//! console.log('Artist:', result.fields.artist);
//! ```

use wasm_bindgen::prelude::*;

use oximedia_metadata::{Metadata, MetadataFormat, MetadataValue};

/// Detect metadata format from raw bytes.
///
/// Examines magic bytes and header signatures to determine the metadata format.
/// Returns `None` if the format cannot be detected.
fn detect_metadata_format(data: &[u8]) -> Option<MetadataFormat> {
    if data.len() < 3 {
        return None;
    }

    // ID3v2: starts with "ID3"
    if data.starts_with(b"ID3") {
        return Some(MetadataFormat::Id3v2);
    }

    // FLAC: starts with "fLaC" - Vorbis Comments are embedded
    if data.starts_with(b"fLaC") {
        return Some(MetadataFormat::VorbisComments);
    }

    // APEv2: starts with "APETAGEX"
    if data.len() >= 8 && data.starts_with(b"APETAGEX") {
        return Some(MetadataFormat::Apev2);
    }

    // Matroska/WebM EBML header
    if data.len() >= 4 && data[0] == 0x1A && data[1] == 0x45 && data[2] == 0xDF && data[3] == 0xA3 {
        return Some(MetadataFormat::Matroska);
    }

    // EXIF: starts with "Exif\0\0" or TIFF byte order markers
    if data.starts_with(b"Exif") {
        return Some(MetadataFormat::Exif);
    }
    // TIFF big-endian or little-endian
    if data.len() >= 4
        && ((data[0] == 0x4D && data[1] == 0x4D) || (data[0] == 0x49 && data[1] == 0x49))
    {
        return Some(MetadataFormat::Exif);
    }

    // Vorbis comment header: "\x03vorbis"
    if data.len() >= 7 {
        for i in 0..data.len().saturating_sub(7) {
            if data[i] == 0x03 && data.get(i + 1..i + 7) == Some(b"vorbis") {
                return Some(MetadataFormat::VorbisComments);
            }
        }
    }

    // iTunes/MP4: look for "ftyp" atom near the start
    if data.len() >= 8 {
        // ftyp atom is usually at offset 4 (after 4-byte size)
        if data.get(4..8) == Some(b"ftyp") {
            return Some(MetadataFormat::iTunes);
        }
    }

    // XMP: look for XML processing instruction or xmpmeta tag
    if data.starts_with(b"<?xpacket") || data.starts_with(b"<x:xmpmeta") {
        return Some(MetadataFormat::Xmp);
    }

    // QuickTime: look for "moov" or "mdat" atoms
    if data.len() >= 8 {
        if data.get(4..8) == Some(b"moov") || data.get(4..8) == Some(b"mdat") {
            return Some(MetadataFormat::QuickTime);
        }
    }

    None
}

/// Convert a `MetadataValue` to a JSON-serializable string representation.
fn metadata_value_to_json(value: &MetadataValue) -> serde_json::Value {
    match value {
        MetadataValue::Text(s) => serde_json::Value::String(s.clone()),
        MetadataValue::TextList(list) => serde_json::Value::Array(
            list.iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect(),
        ),
        MetadataValue::Integer(i) => serde_json::json!(*i),
        MetadataValue::Float(f) => serde_json::json!(*f),
        MetadataValue::Boolean(b) => serde_json::Value::Bool(*b),
        MetadataValue::DateTime(dt) => serde_json::Value::String(dt.clone()),
        MetadataValue::Binary(data) => {
            serde_json::json!({
                "type": "binary",
                "size": data.len(),
            })
        }
        MetadataValue::Picture(pic) => {
            serde_json::json!({
                "type": "picture",
                "mime_type": pic.mime_type,
                "picture_type": format!("{}", pic.picture_type),
                "description": pic.description,
                "size": pic.data.len(),
                "width": pic.width,
                "height": pic.height,
            })
        }
        MetadataValue::Pictures(pics) => serde_json::Value::Array(
            pics.iter()
                .map(|pic| {
                    serde_json::json!({
                        "type": "picture",
                        "mime_type": pic.mime_type,
                        "picture_type": format!("{}", pic.picture_type),
                        "description": pic.description,
                        "size": pic.data.len(),
                        "width": pic.width,
                        "height": pic.height,
                    })
                })
                .collect(),
        ),
    }
}

/// Parse metadata from container header bytes.
///
/// Auto-detects the metadata format and extracts all available fields.
/// Returns a JSON string with the detected format and a map of fields.
///
/// # Arguments
///
/// * `data` - Raw bytes containing metadata (e.g., file header)
///
/// # Returns
///
/// JSON string with structure:
/// ```json
/// {
///   "format": "ID3v2",
///   "fields": {
///     "TIT2": "Song Title",
///     "TPE1": "Artist Name",
///     "TALB": "Album Name"
///   },
///   "common": {
///     "title": "Song Title",
///     "artist": "Artist Name",
///     "album": "Album Name"
///   }
/// }
/// ```
///
/// # Errors
///
/// Returns an error if:
/// - The data is empty or too short
/// - The metadata format cannot be detected
/// - Parsing fails for the detected format
///
/// # Example (JavaScript)
///
/// ```javascript
/// const response = await fetch('song.mp3');
/// const headerBytes = new Uint8Array(await response.arrayBuffer());
/// const metadata = JSON.parse(oximedia.wasm_parse_metadata(headerBytes));
/// document.title = metadata.common.title || 'Unknown';
/// ```
#[wasm_bindgen]
pub fn wasm_parse_metadata(data: &[u8]) -> Result<String, JsValue> {
    if data.is_empty() {
        return Err(crate::utils::js_err(
            "No data provided for metadata parsing",
        ));
    }

    let format = detect_metadata_format(data)
        .ok_or_else(|| crate::utils::js_err("Could not detect metadata format from input data"))?;

    let metadata = Metadata::parse(data, format)
        .map_err(|e| crate::utils::js_err(&format!("Metadata parse error: {e}")))?;

    // Convert fields to JSON map
    let mut fields_map = serde_json::Map::new();
    for (key, value) in metadata.fields() {
        fields_map.insert(key.clone(), metadata_value_to_json(value));
    }

    // Extract common fields
    let common = metadata.common();
    let common_json = serde_json::json!({
        "title": common.title,
        "artist": common.artist,
        "album": common.album,
        "album_artist": common.album_artist,
        "genre": common.genre,
        "year": common.year,
        "track_number": common.track_number,
        "disc_number": common.disc_number,
        "comment": common.comment,
    });

    let result = serde_json::json!({
        "format": format.to_string(),
        "fields": fields_map,
        "common": common_json,
    });

    serde_json::to_string(&result)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization error: {e}")))
}
