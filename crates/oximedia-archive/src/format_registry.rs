#![allow(dead_code)]
//! Archive format registry: capabilities, selection, and metadata.

use std::collections::HashMap;

/// Supported archive container formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchiveFormat {
    /// ZIP archive (DEFLATE/STORE compressed).
    Zip,
    /// Tape Archive (GNU tar / POSIX ustar).
    Tar,
    /// 7-Zip archive (LZMA2 compression).
    SevenZip,
    /// Linear Tape-Open (LTO) tape cartridge format.
    Lto,
}

impl ArchiveFormat {
    /// Returns `true` if this format uses tape media.
    #[must_use]
    pub fn is_tape_format(&self) -> bool {
        matches!(self, Self::Lto)
    }

    /// Returns a human-readable name for the format.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Zip => "ZIP",
            Self::Tar => "TAR",
            Self::SevenZip => "7-Zip",
            Self::Lto => "LTO Tape",
        }
    }

    /// Returns the canonical file extension (without dot).
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Zip => "zip",
            Self::Tar => "tar",
            Self::SevenZip => "7z",
            Self::Lto => "lto",
        }
    }
}

/// Describes the capabilities of an archive format.
#[derive(Debug, Clone)]
pub struct FormatCapability {
    /// The format this capability record describes.
    pub format: ArchiveFormat,
    /// Maximum single-archive size in bytes (`u64::MAX` = unlimited).
    pub max_size_bytes: u64,
    /// Whether this format supports random-access reads.
    pub random_access: bool,
    /// Whether this format supports native compression.
    pub native_compression: bool,
    /// Typical write throughput in MiB/s (0 = unknown).
    pub write_throughput_mib_s: u32,
}

impl FormatCapability {
    /// Creates a new `FormatCapability`.
    #[must_use]
    pub fn new(
        format: ArchiveFormat,
        max_size_bytes: u64,
        random_access: bool,
        native_compression: bool,
        write_throughput_mib_s: u32,
    ) -> Self {
        Self {
            format,
            max_size_bytes,
            random_access,
            native_compression,
            write_throughput_mib_s,
        }
    }

    /// Returns the maximum archive size in bytes.
    #[must_use]
    pub fn max_size_bytes(&self) -> u64 {
        self.max_size_bytes
    }

    /// Returns `true` if the format can hold at least `required_bytes`.
    #[must_use]
    pub fn can_hold(&self, required_bytes: u64) -> bool {
        self.max_size_bytes >= required_bytes
    }
}

/// Registry mapping `ArchiveFormat` variants to their `FormatCapability`.
#[derive(Debug, Default)]
pub struct FormatRegistry {
    capabilities: HashMap<ArchiveFormat, FormatCapability>,
}

impl FormatRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            capabilities: HashMap::new(),
        }
    }

    /// Creates a registry pre-populated with sensible defaults.
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut r = Self::new();
        // ZIP: 4 GiB limit without ZIP64; with ZIP64 effectively unlimited → use u64::MAX
        r.register(FormatCapability::new(
            ArchiveFormat::Zip,
            u64::MAX,
            true,
            true,
            200,
        ));
        r.register(FormatCapability::new(
            ArchiveFormat::Tar,
            u64::MAX,
            false,
            false,
            500,
        ));
        r.register(FormatCapability::new(
            ArchiveFormat::SevenZip,
            u64::MAX,
            true,
            true,
            150,
        ));
        // LTO-9: ~18 TB native capacity
        r.register(FormatCapability::new(
            ArchiveFormat::Lto,
            18 * 1024 * 1024 * 1024 * 1024_u64, // 18 TiB
            false,
            true,
            300,
        ));
        r
    }

    /// Registers or overwrites the capability for a format.
    pub fn register(&mut self, cap: FormatCapability) {
        self.capabilities.insert(cap.format, cap);
    }

    /// Returns the capability for the given format, if registered.
    #[must_use]
    pub fn capability(&self, format: ArchiveFormat) -> Option<&FormatCapability> {
        self.capabilities.get(&format)
    }

    /// Returns the best format for the given total size (highest throughput that can hold the data).
    ///
    /// Tape formats are preferred for sizes exceeding 1 TiB.
    #[must_use]
    pub fn best_for_size(&self, size_bytes: u64) -> Option<ArchiveFormat> {
        const ONE_TIB: u64 = 1024 * 1024 * 1024 * 1024;

        let candidates: Vec<&FormatCapability> = self
            .capabilities
            .values()
            .filter(|c| c.can_hold(size_bytes))
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Prefer tape for large archives
        if size_bytes >= ONE_TIB {
            if let Some(tape) = candidates.iter().find(|c| c.format.is_tape_format()) {
                return Some(tape.format);
            }
        }

        // Otherwise pick highest throughput
        candidates
            .into_iter()
            .max_by_key(|c| c.write_throughput_mib_s)
            .map(|c| c.format)
    }

    /// Returns the number of registered formats.
    #[must_use]
    pub fn len(&self) -> usize {
        self.capabilities.len()
    }

    /// Returns `true` if no formats are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.capabilities.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ArchiveFormat ---

    #[test]
    fn lto_is_tape() {
        assert!(ArchiveFormat::Lto.is_tape_format());
    }

    #[test]
    fn zip_not_tape() {
        assert!(!ArchiveFormat::Zip.is_tape_format());
    }

    #[test]
    fn tar_extension() {
        assert_eq!(ArchiveFormat::Tar.extension(), "tar");
    }

    #[test]
    fn sevenz_extension() {
        assert_eq!(ArchiveFormat::SevenZip.extension(), "7z");
    }

    #[test]
    fn display_names_nonempty() {
        for fmt in [
            ArchiveFormat::Zip,
            ArchiveFormat::Tar,
            ArchiveFormat::SevenZip,
            ArchiveFormat::Lto,
        ] {
            assert!(!fmt.display_name().is_empty());
        }
    }

    // --- FormatCapability ---

    #[test]
    fn can_hold_exact_size() {
        let cap = FormatCapability::new(ArchiveFormat::Zip, 1000, true, true, 100);
        assert!(cap.can_hold(1000));
    }

    #[test]
    fn cannot_hold_over_limit() {
        let cap = FormatCapability::new(ArchiveFormat::Zip, 1000, true, true, 100);
        assert!(!cap.can_hold(1001));
    }

    #[test]
    fn max_size_bytes_accessor() {
        let cap = FormatCapability::new(ArchiveFormat::Tar, 5000, false, false, 200);
        assert_eq!(cap.max_size_bytes(), 5000);
    }

    // --- FormatRegistry ---

    #[test]
    fn default_registry_has_four_formats() {
        let r = FormatRegistry::with_defaults();
        assert_eq!(r.len(), 4);
    }

    #[test]
    fn registry_not_empty() {
        let r = FormatRegistry::with_defaults();
        assert!(!r.is_empty());
    }

    #[test]
    fn capability_lookup_zip() {
        let r = FormatRegistry::with_defaults();
        let cap = r
            .capability(ArchiveFormat::Zip)
            .expect("cap should be valid");
        assert!(cap.native_compression);
    }

    #[test]
    fn best_for_size_small_not_tape() {
        let r = FormatRegistry::with_defaults();
        let fmt = r
            .best_for_size(100 * 1024 * 1024)
            .expect("fmt should be valid"); // 100 MiB
        assert!(!fmt.is_tape_format());
    }

    #[test]
    fn best_for_size_large_prefers_tape() {
        let r = FormatRegistry::with_defaults();
        let one_tib: u64 = 1024 * 1024 * 1024 * 1024;
        let fmt = r.best_for_size(one_tib).expect("fmt should be valid");
        assert!(fmt.is_tape_format());
    }

    #[test]
    fn best_for_size_returns_none_for_empty_registry() {
        let r = FormatRegistry::new();
        assert!(r.best_for_size(1).is_none());
    }

    #[test]
    fn register_overrides_existing() {
        let mut r = FormatRegistry::with_defaults();
        let custom = FormatCapability::new(ArchiveFormat::Zip, 42, false, false, 1);
        r.register(custom);
        assert_eq!(
            r.capability(ArchiveFormat::Zip)
                .expect("capability should succeed")
                .max_size_bytes(),
            42
        );
    }
}
