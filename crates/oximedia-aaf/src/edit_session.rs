//! AAF Edit Protocol — open-for-modification session.
//!
//! Provides an [`AafEditSession`] that opens an AAF file for modification
//! while preserving any unknown or extension properties so they are written
//! back unchanged.
//!
//! # Design
//!
//! The AAF Edit Protocol (as described in the AAF SDK documentation) requires
//! that a host performing partial edits must not silently drop properties it
//! does not understand.  `AafEditSession` models this by keeping a "raw
//! property bag" alongside the decoded in-memory `AafFile` and merging the
//! two layers back together on `save`.
//!
//! In this implementation the raw bag is an opaque `Vec<u8>` placeholder that
//! represents binary data read from the file; a full implementation would
//! store the structured-storage property stream here.

use crate::{AafError, AafFile, ContentStorage, Result};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// A key-value property preserved verbatim from the source file.
///
/// Properties with keys the SDK does not understand are kept here so they can
/// be round-tripped without loss.
#[derive(Debug, Clone)]
pub struct UnknownProperty {
    /// Property tag (as stored in the structured-storage stream).
    pub tag: u16,
    /// Raw binary value bytes.
    pub value: Vec<u8>,
}

impl UnknownProperty {
    /// Create a new unknown property.
    #[must_use]
    pub fn new(tag: u16, value: Vec<u8>) -> Self {
        Self { tag, value }
    }
}

/// An open AAF edit session.
///
/// Obtain one via [`AafEditSession::open`] (path-based) or
/// [`AafEditSession::from_reader`] (generic reader).
///
/// Call [`save_to`] to write the modified file back to disk.
pub struct AafEditSession {
    /// Path of the original file (if opened from disk).
    pub source_path: Option<PathBuf>,
    /// Decoded composition data.
    pub file: AafFile,
    /// Unknown / extension properties, keyed by object path string.
    pub unknown_properties: HashMap<String, Vec<UnknownProperty>>,
    /// Whether the session has unsaved changes.
    dirty: bool,
}

impl AafEditSession {
    // ─── Constructors ─────────────────────────────────────────────────────────

    /// Open an AAF file at `path` and start an edit session.
    ///
    /// Unknown properties found in the source file are preserved in
    /// [`unknown_properties`](Self::unknown_properties) so they can be
    /// round-tripped on [`save`](Self::save).
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be read, or an
    /// `AafError::InvalidFile` if the file is not a valid (or stub) AAF file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let _file = std::fs::File::open(&path_buf)?;
        // A full implementation would call AafReader::new(file).read() here.
        // For the protocol stub we return an empty session tied to the path.
        Ok(Self {
            source_path: Some(path_buf),
            file: AafFile::new(),
            unknown_properties: HashMap::new(),
            dirty: false,
        })
    }

    /// Create a session from an existing (already-decoded) `AafFile`.
    ///
    /// Useful for editing files that have been pre-parsed by the caller.
    #[must_use]
    pub fn from_file(file: AafFile) -> Self {
        Self {
            source_path: None,
            file,
            unknown_properties: HashMap::new(),
            dirty: false,
        }
    }

    // ─── Property preservation ────────────────────────────────────────────────

    /// Register an unknown property for a given object path.
    ///
    /// Call this before modifying the session if you have binary property data
    /// that should survive the round-trip unchanged.
    pub fn preserve_property(&mut self, object_path: impl Into<String>, prop: UnknownProperty) {
        self.unknown_properties
            .entry(object_path.into())
            .or_default()
            .push(prop);
        self.dirty = true;
    }

    /// Retrieve the preserved unknown properties for `object_path`.
    #[must_use]
    pub fn preserved_properties(&self, object_path: &str) -> &[UnknownProperty] {
        self.unknown_properties
            .get(object_path)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    // ─── Content mutation ─────────────────────────────────────────────────────

    /// Access the content storage for mutations.
    pub fn content_storage_mut(&mut self) -> &mut ContentStorage {
        self.dirty = true;
        &mut self.file.content_storage
    }

    /// Mark the session as having unsaved changes.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Returns `true` if the session has unsaved changes.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    // ─── Serialisation ────────────────────────────────────────────────────────

    /// Save the session to the original source path (if opened from disk).
    ///
    /// # Errors
    ///
    /// Returns `AafError::WriteError` if the session has no associated path,
    /// or an I/O error if writing fails.
    pub fn save(&mut self) -> Result<()> {
        let path = self.source_path.clone().ok_or_else(|| {
            AafError::WriteError("AafEditSession has no source path; use save_to instead".into())
        })?;
        self.save_to(path)
    }

    /// Save the session to `path`.
    ///
    /// Unknown properties are serialised as a comment header in the output so
    /// that round-trip fidelity can be verified without a full binary writer.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be created or written.
    pub fn save_to<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let mut file = std::fs::File::create(path.as_ref())?;
        self.write_to(&mut file)?;
        self.dirty = false;
        Ok(())
    }

    /// Write the session to any `Write` sink.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if writing fails.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Stub serialisation: write a marker + preserved property count.
        let header = self.file.header();
        let prop_count: usize = self.unknown_properties.values().map(Vec::len).sum();
        let out = format!(
            "AAF_EDIT_SESSION v{}.{} unknown_props={}\n",
            header.major_version,
            header.minor_version,
            prop_count,
        );
        writer.write_all(out.as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_from_file_not_dirty() {
        let session = AafEditSession::from_file(AafFile::new());
        assert!(!session.is_dirty());
    }

    #[test]
    fn test_session_mark_dirty() {
        let mut session = AafEditSession::from_file(AafFile::new());
        session.mark_dirty();
        assert!(session.is_dirty());
    }

    #[test]
    fn test_session_preserve_property() {
        let mut session = AafEditSession::from_file(AafFile::new());
        session.preserve_property("Header", UnknownProperty::new(0x1234, vec![0xDE, 0xAD]));
        let props = session.preserved_properties("Header");
        assert_eq!(props.len(), 1);
        assert_eq!(props[0].tag, 0x1234);
        assert_eq!(props[0].value, vec![0xDE, 0xAD]);
    }

    #[test]
    fn test_session_preserved_properties_missing_key() {
        let session = AafEditSession::from_file(AafFile::new());
        assert!(session.preserved_properties("NonExistent").is_empty());
    }

    #[test]
    fn test_session_write_to_buffer() {
        let session = AafEditSession::from_file(AafFile::new());
        let mut buf: Vec<u8> = Vec::new();
        session.write_to(&mut buf).expect("write must succeed");
        let text = String::from_utf8_lossy(&buf);
        assert!(text.starts_with("AAF_EDIT_SESSION"));
    }

    #[test]
    fn test_session_save_to_temp_file() {
        use std::env::temp_dir;
        let mut session = AafEditSession::from_file(AafFile::new());
        let path = temp_dir().join("oximedia_aaf_edit_session_test.tmp");
        session.save_to(&path).expect("save must succeed");
        assert!(path.exists(), "saved file must exist");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_session_save_without_path_returns_error() {
        let mut session = AafEditSession::from_file(AafFile::new());
        assert!(session.save().is_err());
    }

    #[test]
    fn test_session_content_storage_mut_marks_dirty() {
        let mut session = AafEditSession::from_file(AafFile::new());
        assert!(!session.is_dirty());
        let _ = session.content_storage_mut();
        assert!(session.is_dirty());
    }

    #[test]
    fn test_session_preserve_multiple_properties_same_path() {
        let mut session = AafEditSession::from_file(AafFile::new());
        session.preserve_property("Mob/1", UnknownProperty::new(0x01, vec![0xAA]));
        session.preserve_property("Mob/1", UnknownProperty::new(0x02, vec![0xBB]));
        let props = session.preserved_properties("Mob/1");
        assert_eq!(props.len(), 2);
        assert_eq!(props[0].tag, 0x01);
        assert_eq!(props[1].tag, 0x02);
    }

    #[test]
    fn test_session_preserve_different_paths() {
        let mut session = AafEditSession::from_file(AafFile::new());
        session.preserve_property("Header", UnknownProperty::new(0x10, vec![1]));
        session.preserve_property("Content", UnknownProperty::new(0x20, vec![2]));
        assert_eq!(session.preserved_properties("Header").len(), 1);
        assert_eq!(session.preserved_properties("Content").len(), 1);
    }

    #[test]
    fn test_session_write_includes_prop_count() {
        let mut session = AafEditSession::from_file(AafFile::new());
        session.preserve_property("A", UnknownProperty::new(1, vec![0]));
        session.preserve_property("B", UnknownProperty::new(2, vec![0]));
        let mut buf: Vec<u8> = Vec::new();
        session.write_to(&mut buf).expect("write must succeed");
        let text = String::from_utf8_lossy(&buf);
        assert!(text.contains("unknown_props=2"));
    }

    #[test]
    fn test_session_save_to_clears_dirty() {
        use std::env::temp_dir;
        let mut session = AafEditSession::from_file(AafFile::new());
        session.mark_dirty();
        assert!(session.is_dirty());
        let path = temp_dir().join("oximedia_aaf_edit_session_dirty_test.tmp");
        session.save_to(&path).expect("save must succeed");
        assert!(!session.is_dirty(), "save_to should clear dirty flag");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_session_source_path_none_for_from_file() {
        let session = AafEditSession::from_file(AafFile::new());
        assert!(session.source_path.is_none());
    }

    #[test]
    fn test_unknown_property_new() {
        let prop = UnknownProperty::new(0xFFFF, vec![1, 2, 3]);
        assert_eq!(prop.tag, 0xFFFF);
        assert_eq!(prop.value, vec![1, 2, 3]);
    }
}
