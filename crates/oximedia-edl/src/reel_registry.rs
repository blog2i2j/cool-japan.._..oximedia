//! EDL reel registry.
//!
//! Provides `ReelFormat`, `ReelInfo`, and `ReelRegistry` for tracking
//! physical and virtual reels referenced within an EDL.

#![allow(dead_code)]

use std::collections::HashMap;

/// The origination format of a reel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReelFormat {
    /// 16 mm film reel.
    Film16mm,
    /// 35 mm film reel.
    Film35mm,
    /// 65/70 mm film reel.
    Film65mm,
    /// Digital tape (e.g. HDCAM, XDCAM).
    DigitalTape,
    /// File-based / virtual reel.
    FileBased,
    /// Unknown format.
    Unknown,
}

impl ReelFormat {
    /// Returns `true` if the reel originates from film stock.
    #[must_use]
    pub fn is_film(self) -> bool {
        matches!(self, Self::Film16mm | Self::Film35mm | Self::Film65mm)
    }

    /// Returns `true` if the reel is file-based (not physical media).
    #[must_use]
    pub fn is_file_based(self) -> bool {
        matches!(self, Self::FileBased)
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Film16mm => "16mm Film",
            Self::Film35mm => "35mm Film",
            Self::Film65mm => "65mm Film",
            Self::DigitalTape => "Digital Tape",
            Self::FileBased => "File-Based",
            Self::Unknown => "Unknown",
        }
    }
}

/// Metadata about a single reel.
#[derive(Debug, Clone)]
pub struct ReelInfo {
    /// Reel name/identifier (up to 8 characters in CMX 3600).
    pub name: String,
    /// Physical format of this reel.
    pub format: ReelFormat,
    /// Optional description.
    pub description: Option<String>,
    /// Optional total duration in frames.
    pub total_frames: Option<u64>,
}

impl ReelInfo {
    /// Create a new `ReelInfo`.
    #[must_use]
    pub fn new(name: impl Into<String>, format: ReelFormat) -> Self {
        Self {
            name: name.into(),
            format,
            description: None,
            total_frames: None,
        }
    }

    /// Returns `true` if the reel name is non-empty and not longer than 8 chars.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.name.is_empty() && self.name.len() <= 8
    }

    /// Set a descriptive label.
    pub fn set_description(&mut self, desc: impl Into<String>) {
        self.description = Some(desc.into());
    }

    /// Set the total duration in frames.
    pub fn set_total_frames(&mut self, frames: u64) {
        self.total_frames = Some(frames);
    }
}

/// A registry mapping reel names to `ReelInfo` records.
#[derive(Debug, Clone, Default)]
pub struct ReelRegistry {
    reels: HashMap<String, ReelInfo>,
}

impl ReelRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a reel, overwriting any existing entry with the same name.
    pub fn register(&mut self, info: ReelInfo) {
        self.reels.insert(info.name.clone(), info);
    }

    /// Register a reel by name and format (convenience).
    pub fn register_by_name(&mut self, name: impl Into<String>, format: ReelFormat) {
        let info = ReelInfo::new(name, format);
        self.register(info);
    }

    /// Look up a reel by name.
    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<&ReelInfo> {
        self.reels.get(name)
    }

    /// Look up a mutable reel by name.
    pub fn lookup_mut(&mut self, name: &str) -> Option<&mut ReelInfo> {
        self.reels.get_mut(name)
    }

    /// Total number of registered reels.
    #[must_use]
    pub fn reel_count(&self) -> usize {
        self.reels.len()
    }

    /// Returns `true` if no reels are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.reels.is_empty()
    }

    /// Return all reels of a given format.
    #[must_use]
    pub fn by_format(&self, format: ReelFormat) -> Vec<&ReelInfo> {
        self.reels.values().filter(|r| r.format == format).collect()
    }

    /// Return sorted list of all reel names.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.reels.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ReelFormat tests ---

    #[test]
    fn test_film16mm_is_film() {
        assert!(ReelFormat::Film16mm.is_film());
    }

    #[test]
    fn test_film35mm_is_film() {
        assert!(ReelFormat::Film35mm.is_film());
    }

    #[test]
    fn test_digital_tape_not_film() {
        assert!(!ReelFormat::DigitalTape.is_film());
    }

    #[test]
    fn test_file_based_is_file_based() {
        assert!(ReelFormat::FileBased.is_file_based());
    }

    #[test]
    fn test_film_not_file_based() {
        assert!(!ReelFormat::Film35mm.is_file_based());
    }

    #[test]
    fn test_format_label() {
        assert_eq!(ReelFormat::Film35mm.label(), "35mm Film");
        assert_eq!(ReelFormat::FileBased.label(), "File-Based");
        assert_eq!(ReelFormat::Unknown.label(), "Unknown");
    }

    // --- ReelInfo tests ---

    #[test]
    fn test_reel_info_valid() {
        let r = ReelInfo::new("A001", ReelFormat::DigitalTape);
        assert!(r.is_valid());
    }

    #[test]
    fn test_reel_info_empty_name_invalid() {
        let r = ReelInfo::new("", ReelFormat::FileBased);
        assert!(!r.is_valid());
    }

    #[test]
    fn test_reel_info_name_too_long_invalid() {
        let r = ReelInfo::new("VERYLONGNAME", ReelFormat::FileBased);
        assert!(!r.is_valid());
    }

    #[test]
    fn test_reel_info_set_description() {
        let mut r = ReelInfo::new("B002", ReelFormat::Film35mm);
        assert!(r.description.is_none());
        r.set_description("Scene 3 daylight");
        assert_eq!(r.description.as_deref(), Some("Scene 3 daylight"));
    }

    #[test]
    fn test_reel_info_set_frames() {
        let mut r = ReelInfo::new("C003", ReelFormat::DigitalTape);
        r.set_total_frames(86400);
        assert_eq!(r.total_frames, Some(86400));
    }

    // --- ReelRegistry tests ---

    #[test]
    fn test_registry_register_and_count() {
        let mut reg = ReelRegistry::new();
        assert!(reg.is_empty());
        reg.register_by_name("A001", ReelFormat::DigitalTape);
        reg.register_by_name("B002", ReelFormat::Film35mm);
        assert_eq!(reg.reel_count(), 2);
    }

    #[test]
    fn test_registry_lookup_found() {
        let mut reg = ReelRegistry::new();
        reg.register_by_name("X001", ReelFormat::FileBased);
        let r = reg.lookup("X001");
        assert!(r.is_some());
        assert_eq!(r.expect("r should be valid").name, "X001");
    }

    #[test]
    fn test_registry_lookup_missing() {
        let reg = ReelRegistry::new();
        assert!(reg.lookup("GHOST").is_none());
    }

    #[test]
    fn test_registry_overwrite() {
        let mut reg = ReelRegistry::new();
        reg.register_by_name("A001", ReelFormat::DigitalTape);
        reg.register_by_name("A001", ReelFormat::Film35mm);
        assert_eq!(reg.reel_count(), 1);
        assert_eq!(
            reg.lookup("A001").expect("lookup should succeed").format,
            ReelFormat::Film35mm
        );
    }

    #[test]
    fn test_registry_by_format() {
        let mut reg = ReelRegistry::new();
        reg.register_by_name("A001", ReelFormat::Film35mm);
        reg.register_by_name("A002", ReelFormat::Film35mm);
        reg.register_by_name("B001", ReelFormat::DigitalTape);
        assert_eq!(reg.by_format(ReelFormat::Film35mm).len(), 2);
        assert_eq!(reg.by_format(ReelFormat::DigitalTape).len(), 1);
    }

    #[test]
    fn test_registry_names_sorted() {
        let mut reg = ReelRegistry::new();
        reg.register_by_name("C001", ReelFormat::FileBased);
        reg.register_by_name("A001", ReelFormat::FileBased);
        reg.register_by_name("B001", ReelFormat::FileBased);
        assert_eq!(reg.names(), vec!["A001", "B001", "C001"]);
    }
}
