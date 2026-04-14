//! Standards-conformance validation for metadata fields and values.
//!
//! Validates metadata against format-specific rules:
//!
//! - **ID3v2** — Frame IDs must be exactly 4 ASCII uppercase alphanumeric characters.
//!   Selected frames have additional length or encoding constraints.
//! - **Vorbis Comments** — Field names must be 7-bit ASCII, no `=` or NUL, and
//!   non-empty. Values must be valid UTF-8 (satisfied by Rust's `String`).
//! - **iTunes/MP4** — Atom keys must start with `©` or be known freeform atom names.
//!   Values should not exceed the maximum atom payload size (≤ 16 MiB).
//! - **APEv2** — Keys must be 2–255 bytes, ASCII 0x20-0x7E, not a reserved word.
//!   Values may not exceed 8 MiB.
//!
//! # Example
//!
//! ```
//! use oximedia_metadata::metadata_validate::{validate_field, ValidationContext};
//!
//! let result = validate_field("TIT2", "My Song", ValidationContext::Id3v2);
//! assert!(result.is_ok());
//!
//! let bad = validate_field("bad key!", "value", ValidationContext::Id3v2);
//! assert!(bad.is_err());
//! ```

use crate::Error;

// ─── Context ─────────────────────────────────────────────────────────────────

/// The metadata format context used for validation rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationContext {
    /// ID3v2 frame validation rules.
    Id3v2,
    /// Vorbis Comment field validation rules.
    VorbisComment,
    /// iTunes/MP4 atom name validation rules.
    Itunes,
    /// APEv2 tag key validation rules.
    Apev2,
}

// ─── ID3v2 ───────────────────────────────────────────────────────────────────

/// Known valid ID3v2 v2.3/v2.4 frame IDs (a representative set).
const KNOWN_ID3V2_FRAMES: &[&str] = &[
    "AENC", "APIC", "ASPI", "CHAP", "COMM", "COMR", "CTOC", "ENCR", "EQU2", "EQUA",
    "ETCO", "GEOB", "GRID", "IPLS", "LINK", "MCDI", "MLLT", "OWNE", "PCNT", "POPM",
    "POSS", "PRIV", "RBUF", "RVA2", "RVAD", "RVRB", "SEEK", "SIGN", "SYLT", "SYTC",
    "TALB", "TBPM", "TCOM", "TCON", "TCOP", "TDAT", "TDEN", "TDLY", "TDOR", "TDRC",
    "TDRL", "TDTG", "TENC", "TEXT", "TFLT", "TIPL", "TIT1", "TIT2", "TIT3", "TKEY",
    "TLAN", "TLEN", "TMCL", "TMED", "TMOO", "TOAL", "TOFN", "TOLY", "TOPE", "TORY",
    "TOWN", "TPE1", "TPE2", "TPE3", "TPE4", "TPOS", "TPRO", "TPUB", "TRCK", "TRDA",
    "TRSN", "TRSO", "TSIZ", "TSOA", "TSOP", "TSOT", "TSRC", "TSSE", "TSST", "TYER",
    "TXXX", "UFID", "USER", "USLT", "WCOM", "WCOP", "WOAF", "WOAR", "WOAS", "WORS",
    "WPAY", "WPUB", "WXXX",
];

/// Maximum value length in bytes for text frames (ID3v2 spec recommends < 1 MB).
const ID3V2_MAX_TEXT_BYTES: usize = 1_048_576;

/// Validate an ID3v2 frame ID.
///
/// Rules:
/// 1. Must be exactly 4 characters long.
/// 2. Must consist only of ASCII uppercase letters or digits.
/// 3. Should be a known frame ID or a user-defined `T`, `W`, `X` frame
///    (lenient: unknown IDs produce a warning, not an error).
fn validate_id3v2_key(key: &str) -> Result<(), Error> {
    if key.len() != 4 {
        return Err(Error::InvalidFormat(format!(
            "ID3v2 frame ID must be 4 characters, got {}: '{key}'",
            key.len()
        )));
    }
    if !key
        .chars()
        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
    {
        return Err(Error::InvalidFormat(format!(
            "ID3v2 frame ID must be uppercase ASCII alphanumeric, got '{key}'"
        )));
    }
    Ok(())
}

/// Validate an ID3v2 text frame value.
fn validate_id3v2_value(key: &str, value: &str) -> Result<(), Error> {
    if value.len() > ID3V2_MAX_TEXT_BYTES {
        return Err(Error::InvalidFormat(format!(
            "ID3v2 frame '{key}' value exceeds maximum size ({} > {ID3V2_MAX_TEXT_BYTES})",
            value.len()
        )));
    }
    // TRCK and TPOS must be numeric (optionally "N/M")
    if key == "TRCK" || key == "TPOS" {
        let clean = value.trim();
        if !clean.is_empty() && !is_track_pos_format(clean) {
            return Err(Error::InvalidFormat(format!(
                "ID3v2 '{key}' must be numeric (or 'N/M'), got '{clean}'"
            )));
        }
    }
    // TBPM must be a non-negative integer
    if key == "TBPM" {
        let clean = value.trim();
        if !clean.is_empty() && clean.parse::<u32>().is_err() {
            return Err(Error::InvalidFormat(format!(
                "ID3v2 'TBPM' must be a non-negative integer, got '{clean}'"
            )));
        }
    }
    Ok(())
}

/// Check whether a string matches `N` or `N/M` where N, M are positive integers.
fn is_track_pos_format(s: &str) -> bool {
    let parts: Vec<&str> = s.split('/').collect();
    match parts.as_slice() {
        [n] => n.parse::<u32>().is_ok(),
        [n, m] => n.parse::<u32>().is_ok() && m.parse::<u32>().is_ok(),
        _ => false,
    }
}

// ─── Vorbis Comment ───────────────────────────────────────────────────────────

/// Maximum Vorbis Comment field name length (spec: no explicit limit, but 64 is generous).
const VORBIS_MAX_KEY_LEN: usize = 64;

/// Validate a Vorbis Comment field name.
///
/// Rules (from RFC 5334 / Vorbis spec):
/// 1. Must not be empty.
/// 2. Must be <= 64 bytes (we enforce a reasonable limit).
/// 3. Must consist only of 7-bit ASCII in the range 0x20-0x7D (excluding `=` 0x3D).
///    The spec disallows `=` (used as separator) and NUL.
fn validate_vorbis_key(key: &str) -> Result<(), Error> {
    if key.is_empty() {
        return Err(Error::InvalidFormat(
            "Vorbis Comment field name must not be empty".to_string(),
        ));
    }
    if key.len() > VORBIS_MAX_KEY_LEN {
        return Err(Error::InvalidFormat(format!(
            "Vorbis Comment field name exceeds maximum length ({} > {VORBIS_MAX_KEY_LEN}): '{key}'",
            key.len()
        )));
    }
    for (i, b) in key.bytes().enumerate() {
        if b == b'=' || b < 0x20 || b > 0x7D {
            return Err(Error::InvalidFormat(format!(
                "Vorbis Comment field name contains invalid byte 0x{b:02x} at position {i}: '{key}'"
            )));
        }
    }
    Ok(())
}

// ─── iTunes / MP4 ────────────────────────────────────────────────────────────

/// Well-known iTunes metadata atom names (non-© atoms).
const ITUNES_STANDARD_ATOMS: &[&str] = &[
    "aART", "cpil", "disk", "gnre", "pgap", "rtng", "tmpo", "trkn", "soal", "soar",
    "soaa", "sonm", "soco", "sosn", "tvsh", "tvsn", "tvep", "tvnn", "desc", "ldes",
    "purd", "apID", "cmID", "plID", "cnID", "sfID", "atID", "geID", "akID", "catg",
    "hdvd", "stik", "pcst", "keyw", "itnu",
];

/// Maximum iTunes atom value payload size (16 MiB).
const ITUNES_MAX_VALUE_BYTES: usize = 16 * 1024 * 1024;

/// Validate an iTunes/MP4 atom key.
///
/// Rules:
/// 1. Must not be empty.
/// 2. Must start with `©` (copyright sign, U+00A9) for standard text atoms, OR
///    be a known non-© atom name, OR be a freeform atom (starts with "----:").
fn validate_itunes_key(key: &str) -> Result<(), Error> {
    if key.is_empty() {
        return Err(Error::InvalidFormat(
            "iTunes atom key must not be empty".to_string(),
        ));
    }
    if key.starts_with('\u{00A9}') {
        // Standard text atom — must be exactly © + 3 ASCII chars
        let rest: &str = &key[2..]; // © is 2 bytes in UTF-8
        if rest.len() != 3 || !rest.is_ascii() {
            return Err(Error::InvalidFormat(format!(
                "iTunes © atom key must be '©' + 3 ASCII chars, got '{key}'"
            )));
        }
        return Ok(());
    }
    if key.starts_with("----:") {
        // Freeform atom; just require non-empty after prefix
        if key.len() <= 5 {
            return Err(Error::InvalidFormat(format!(
                "iTunes freeform atom '----:' must have a non-empty name: '{key}'"
            )));
        }
        return Ok(());
    }
    if ITUNES_STANDARD_ATOMS.contains(&key) {
        return Ok(());
    }
    // Unknown but 4-char ASCII atoms are permissible (forward compat)
    if key.len() == 4 && key.is_ascii() {
        return Ok(());
    }
    Err(Error::InvalidFormat(format!(
        "iTunes atom key is not a recognised atom name: '{key}'"
    )))
}

/// Validate an iTunes atom value.
fn validate_itunes_value(key: &str, value: &str) -> Result<(), Error> {
    if value.len() > ITUNES_MAX_VALUE_BYTES {
        return Err(Error::InvalidFormat(format!(
            "iTunes atom '{key}' value exceeds maximum size ({} > {ITUNES_MAX_VALUE_BYTES})",
            value.len()
        )));
    }
    Ok(())
}

// ─── APEv2 ───────────────────────────────────────────────────────────────────

/// Reserved APEv2 item keys that must not be used as user tags.
const APE_RESERVED_KEYS: &[&str] = &[
    "Id3v1Comment", "Id", "Tag", "OggS",
];

/// Maximum APEv2 item key length (255 bytes per spec).
const APE_MAX_KEY_LEN: usize = 255;
/// Minimum APEv2 item key length (2 bytes per spec).
const APE_MIN_KEY_LEN: usize = 2;
/// Maximum APEv2 item value size (8 MiB).
const APE_MAX_VALUE_BYTES: usize = 8 * 1024 * 1024;

/// Validate an APEv2 tag key.
///
/// Rules (from the APEv2 specification):
/// 1. Must be 2–255 bytes.
/// 2. Must consist only of printable ASCII characters (0x20–0x7E).
/// 3. Must not be one of the reserved key names.
fn validate_apev2_key(key: &str) -> Result<(), Error> {
    let len = key.len();
    if len < APE_MIN_KEY_LEN || len > APE_MAX_KEY_LEN {
        return Err(Error::InvalidFormat(format!(
            "APEv2 key length must be {APE_MIN_KEY_LEN}–{APE_MAX_KEY_LEN} bytes, got {len}: '{key}'"
        )));
    }
    for (i, b) in key.bytes().enumerate() {
        if b < 0x20 || b > 0x7E {
            return Err(Error::InvalidFormat(format!(
                "APEv2 key contains non-printable ASCII byte 0x{b:02x} at position {i}: '{key}'"
            )));
        }
    }
    for &reserved in APE_RESERVED_KEYS {
        if key.eq_ignore_ascii_case(reserved) {
            return Err(Error::InvalidFormat(format!(
                "APEv2 key '{key}' is reserved and must not be used"
            )));
        }
    }
    Ok(())
}

/// Validate an APEv2 item value.
fn validate_apev2_value(key: &str, value: &str) -> Result<(), Error> {
    if value.len() > APE_MAX_VALUE_BYTES {
        return Err(Error::InvalidFormat(format!(
            "APEv2 item '{key}' value exceeds maximum size ({} > {APE_MAX_VALUE_BYTES})",
            value.len()
        )));
    }
    Ok(())
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Validate a single metadata field `(key, value)` pair for the given format context.
///
/// Returns `Ok(())` if the field is valid, or an `Error::InvalidFormat` if it
/// violates the format's constraints.
///
/// # Errors
///
/// Returns an error describing the specific constraint that was violated.
pub fn validate_field(key: &str, value: &str, ctx: ValidationContext) -> Result<(), Error> {
    match ctx {
        ValidationContext::Id3v2 => {
            validate_id3v2_key(key)?;
            validate_id3v2_value(key, value)?;
        }
        ValidationContext::VorbisComment => {
            validate_vorbis_key(key)?;
        }
        ValidationContext::Itunes => {
            validate_itunes_key(key)?;
            validate_itunes_value(key, value)?;
        }
        ValidationContext::Apev2 => {
            validate_apev2_key(key)?;
            validate_apev2_value(key, value)?;
        }
    }
    Ok(())
}

/// Validate all fields in a `HashMap<String, String>` for the given context.
///
/// Returns a list of `(key, error_message)` pairs for every invalid field.
/// Returns an empty vector if all fields are valid.
pub fn validate_fields(
    fields: &std::collections::HashMap<String, String>,
    ctx: ValidationContext,
) -> Vec<(String, String)> {
    let mut errors = Vec::new();
    for (key, value) in fields {
        if let Err(e) = validate_field(key, value, ctx) {
            errors.push((key.clone(), e.to_string()));
        }
    }
    errors.sort_by(|a, b| a.0.cmp(&b.0));
    errors
}

/// Check whether a given ID3v2 frame ID is a well-known standard frame.
pub fn is_known_id3v2_frame(key: &str) -> bool {
    KNOWN_ID3V2_FRAMES.contains(&key)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ID3v2 ────────────────────────────────────────────────────────────────

    #[test]
    fn test_id3v2_valid_key() {
        assert!(validate_field("TIT2", "My Song", ValidationContext::Id3v2).is_ok());
        assert!(validate_field("TRCK", "5", ValidationContext::Id3v2).is_ok());
        assert!(validate_field("TCON", "Rock", ValidationContext::Id3v2).is_ok());
    }

    #[test]
    fn test_id3v2_key_wrong_length() {
        assert!(validate_field("TIT", "x", ValidationContext::Id3v2).is_err());
        assert!(validate_field("TIT22", "x", ValidationContext::Id3v2).is_err());
        assert!(validate_field("", "x", ValidationContext::Id3v2).is_err());
    }

    #[test]
    fn test_id3v2_key_invalid_chars() {
        assert!(validate_field("tit2", "x", ValidationContext::Id3v2).is_err()); // lowercase
        assert!(validate_field("TIT!", "x", ValidationContext::Id3v2).is_err()); // symbol
        assert!(validate_field("TIT\n", "x", ValidationContext::Id3v2).is_err()); // control
    }

    #[test]
    fn test_id3v2_trck_valid_formats() {
        assert!(validate_field("TRCK", "3", ValidationContext::Id3v2).is_ok());
        assert!(validate_field("TRCK", "3/12", ValidationContext::Id3v2).is_ok());
        assert!(validate_field("TRCK", "", ValidationContext::Id3v2).is_ok()); // empty ok
    }

    #[test]
    fn test_id3v2_trck_invalid_formats() {
        assert!(validate_field("TRCK", "three", ValidationContext::Id3v2).is_err());
        assert!(validate_field("TRCK", "1/2/3", ValidationContext::Id3v2).is_err());
    }

    #[test]
    fn test_id3v2_tbpm_valid() {
        assert!(validate_field("TBPM", "128", ValidationContext::Id3v2).is_ok());
        assert!(validate_field("TBPM", "", ValidationContext::Id3v2).is_ok());
    }

    #[test]
    fn test_id3v2_tbpm_invalid() {
        assert!(validate_field("TBPM", "one hundred", ValidationContext::Id3v2).is_err());
    }

    #[test]
    fn test_id3v2_known_frame_check() {
        assert!(is_known_id3v2_frame("TIT2"));
        assert!(is_known_id3v2_frame("APIC"));
        assert!(!is_known_id3v2_frame("ZZZZ"));
    }

    // ── Vorbis Comment ───────────────────────────────────────────────────────

    #[test]
    fn test_vorbis_valid_key() {
        assert!(validate_field("TITLE", "My Song", ValidationContext::VorbisComment).is_ok());
        assert!(validate_field("ARTIST", "Band", ValidationContext::VorbisComment).is_ok());
        assert!(validate_field("TRACKNUMBER", "1", ValidationContext::VorbisComment).is_ok());
    }

    #[test]
    fn test_vorbis_empty_key_rejected() {
        assert!(validate_field("", "val", ValidationContext::VorbisComment).is_err());
    }

    #[test]
    fn test_vorbis_key_with_equals_rejected() {
        assert!(validate_field("TIT=LE", "val", ValidationContext::VorbisComment).is_err());
    }

    #[test]
    fn test_vorbis_key_with_control_char_rejected() {
        assert!(validate_field("TITL\x01E", "val", ValidationContext::VorbisComment).is_err());
    }

    #[test]
    fn test_vorbis_key_too_long_rejected() {
        let long_key = "A".repeat(65);
        assert!(validate_field(&long_key, "val", ValidationContext::VorbisComment).is_err());
    }

    // ── iTunes ───────────────────────────────────────────────────────────────

    #[test]
    fn test_itunes_valid_copyright_atom() {
        assert!(validate_field("\u{00A9}nam", "Song", ValidationContext::Itunes).is_ok()); // ©nam
        assert!(validate_field("\u{00A9}ART", "Band", ValidationContext::Itunes).is_ok()); // ©ART
    }

    #[test]
    fn test_itunes_valid_standard_atom() {
        assert!(validate_field("aART", "Album Artist", ValidationContext::Itunes).is_ok());
        assert!(validate_field("cpil", "1", ValidationContext::Itunes).is_ok());
    }

    #[test]
    fn test_itunes_valid_freeform_atom() {
        assert!(validate_field("----:com.example:MyTag", "val", ValidationContext::Itunes).is_ok());
    }

    #[test]
    fn test_itunes_freeform_too_short() {
        assert!(validate_field("----:", "val", ValidationContext::Itunes).is_err());
    }

    #[test]
    fn test_itunes_invalid_atom() {
        assert!(validate_field("not_a_valid_atom_name_long", "val", ValidationContext::Itunes).is_err());
    }

    // ── APEv2 ────────────────────────────────────────────────────────────────

    #[test]
    fn test_apev2_valid_key() {
        assert!(validate_field("Title", "My Song", ValidationContext::Apev2).is_ok());
        assert!(validate_field("Artist", "Band", ValidationContext::Apev2).is_ok());
        assert!(validate_field("MP", "val", ValidationContext::Apev2).is_ok()); // exactly 2 chars
    }

    #[test]
    fn test_apev2_key_too_short() {
        assert!(validate_field("X", "val", ValidationContext::Apev2).is_err());
    }

    #[test]
    fn test_apev2_key_with_non_ascii() {
        assert!(validate_field("Tïtle", "val", ValidationContext::Apev2).is_err());
    }

    #[test]
    fn test_apev2_reserved_key_rejected() {
        assert!(validate_field("Id3v1Comment", "val", ValidationContext::Apev2).is_err());
        assert!(validate_field("id3v1comment", "val", ValidationContext::Apev2).is_err()); // case-insensitive
    }

    #[test]
    fn test_validate_fields_returns_errors() {
        let mut fields = std::collections::HashMap::new();
        fields.insert("TIT2".to_string(), "Good".to_string());
        fields.insert("bad!".to_string(), "Bad key".to_string());
        fields.insert("tit2".to_string(), "Lowercase".to_string());

        let errors = validate_fields(&fields, ValidationContext::Id3v2);
        assert!(!errors.is_empty());
        // "bad!" and "tit2" should both fail; "TIT2" should pass
        let failed_keys: Vec<&str> = errors.iter().map(|(k, _)| k.as_str()).collect();
        assert!(failed_keys.contains(&"bad!") || failed_keys.contains(&"tit2"));
    }

    #[test]
    fn test_validate_fields_all_valid() {
        let mut fields = std::collections::HashMap::new();
        fields.insert("TIT2".to_string(), "My Song".to_string());
        fields.insert("TPE1".to_string(), "My Artist".to_string());

        let errors = validate_fields(&fields, ValidationContext::Id3v2);
        assert!(errors.is_empty());
    }
}
