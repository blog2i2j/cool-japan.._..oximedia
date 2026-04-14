//! PKL (Packing List) XML generator.
//!
//! [`PklGenerator`] provides a simple, dependency-free way to produce a
//! SMPTE ST 429-8 conformant PKL XML string without requiring file I/O.
//! Intended for programmatic package construction workflows.
//!
//! # Example
//!
//! ```
//! use oximedia_imf::pkl_gen::PklGenerator;
//!
//! let mut gen = PklGenerator::new();
//! gen.add_asset("video.mxf", 1_048_576, "deadbeef01234567");
//! gen.add_asset("audio.mxf", 262_144, "0123456789abcdef");
//! let xml = gen.build_xml();
//! assert!(xml.contains("<PackingList>"));
//! assert!(xml.contains("video.mxf"));
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Asset entry
// ---------------------------------------------------------------------------

/// A single asset entry in the PKL.
#[derive(Debug, Clone)]
pub struct PklAsset {
    /// Original path / file name of the asset.
    pub path: String,
    /// File size in bytes.
    pub size: u64,
    /// Hex-encoded hash (SHA-1, MD-5, or SHA-256 depending on caller).
    pub hash: String,
}

// ---------------------------------------------------------------------------
// PklGenerator
// ---------------------------------------------------------------------------

/// Fluent builder that accumulates asset entries and serialises them to a
/// minimal PKL XML document.
#[derive(Debug, Clone, Default)]
pub struct PklGenerator {
    assets: Vec<PklAsset>,
}

impl PklGenerator {
    /// Create an empty `PklGenerator`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an asset with its path, byte size, and pre-computed hash.
    pub fn add_asset(&mut self, path: &str, size: u64, hash: &str) {
        self.assets.push(PklAsset {
            path: path.to_string(),
            size,
            hash: hash.to_string(),
        });
    }

    /// Serialise the accumulated assets to a minimal PKL XML string.
    ///
    /// Produces well-formed XML that mirrors the SMPTE ST 429-8 outline:
    ///
    /// ```xml
    /// <?xml version="1.0" encoding="UTF-8"?>
    /// <PackingList>
    ///   <AssetList>
    ///     <Asset>
    ///       <OriginalFileName>video.mxf</OriginalFileName>
    ///       <Size>1048576</Size>
    ///       <Hash>deadbeef...</Hash>
    ///     </Asset>
    ///   </AssetList>
    /// </PackingList>
    /// ```
    #[must_use]
    pub fn build_xml(&self) -> String {
        let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<PackingList>\n");
        xml.push_str("  <AssetList>\n");
        for asset in &self.assets {
            xml.push_str("    <Asset>\n");
            xml.push_str(&format!(
                "      <OriginalFileName>{}</OriginalFileName>\n",
                xml_escape(&asset.path)
            ));
            xml.push_str(&format!("      <Size>{}</Size>\n", asset.size));
            xml.push_str(&format!(
                "      <Hash>{}</Hash>\n",
                xml_escape(&asset.hash)
            ));
            xml.push_str("    </Asset>\n");
        }
        xml.push_str("  </AssetList>\n</PackingList>\n");
        xml
    }

    /// Number of assets registered in this generator.
    #[must_use]
    pub fn len(&self) -> usize {
        self.assets.len()
    }

    /// Returns `true` when no assets have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.assets.is_empty()
    }

    /// Return a reference to the registered assets.
    #[must_use]
    pub fn assets(&self) -> &[PklAsset] {
        &self.assets
    }
}

/// Escape XML special characters.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── PklGenerator::new ────────────────────────────────────────────────────

    #[test]
    fn test_new_is_empty() {
        let g = PklGenerator::new();
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
    }

    // ── add_asset ────────────────────────────────────────────────────────────

    #[test]
    fn test_add_asset_increments_len() {
        let mut g = PklGenerator::new();
        g.add_asset("video.mxf", 1024, "hash1");
        g.add_asset("audio.mxf", 512, "hash2");
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn test_add_asset_stores_fields() {
        let mut g = PklGenerator::new();
        g.add_asset("test.mxf", 999, "abc123");
        let a = &g.assets()[0];
        assert_eq!(a.path, "test.mxf");
        assert_eq!(a.size, 999);
        assert_eq!(a.hash, "abc123");
    }

    // ── build_xml ────────────────────────────────────────────────────────────

    #[test]
    fn test_build_xml_contains_root_element() {
        let g = PklGenerator::new();
        let xml = g.build_xml();
        assert!(xml.contains("<PackingList>"));
        assert!(xml.contains("</PackingList>"));
    }

    #[test]
    fn test_build_xml_contains_asset_list() {
        let g = PklGenerator::new();
        let xml = g.build_xml();
        assert!(xml.contains("<AssetList>"));
        assert!(xml.contains("</AssetList>"));
    }

    #[test]
    fn test_build_xml_empty_list_is_valid() {
        let g = PklGenerator::new();
        let xml = g.build_xml();
        // Should not contain any <Asset> elements
        assert!(!xml.contains("<Asset>"));
    }

    #[test]
    fn test_build_xml_contains_asset_path() {
        let mut g = PklGenerator::new();
        g.add_asset("my_clip.mxf", 2048, "aabbccdd");
        let xml = g.build_xml();
        assert!(xml.contains("my_clip.mxf"));
    }

    #[test]
    fn test_build_xml_contains_size() {
        let mut g = PklGenerator::new();
        g.add_asset("x.mxf", 1_048_576, "hash");
        let xml = g.build_xml();
        assert!(xml.contains("1048576"));
    }

    #[test]
    fn test_build_xml_contains_hash() {
        let mut g = PklGenerator::new();
        g.add_asset("x.mxf", 1, "deadbeef01234567");
        let xml = g.build_xml();
        assert!(xml.contains("deadbeef01234567"));
    }

    #[test]
    fn test_build_xml_multiple_assets() {
        let mut g = PklGenerator::new();
        g.add_asset("a.mxf", 100, "hash_a");
        g.add_asset("b.mxf", 200, "hash_b");
        let xml = g.build_xml();
        assert!(xml.contains("a.mxf"));
        assert!(xml.contains("b.mxf"));
        assert!(xml.contains("hash_a"));
        assert!(xml.contains("hash_b"));
    }

    #[test]
    fn test_build_xml_escapes_special_chars_in_path() {
        let mut g = PklGenerator::new();
        g.add_asset("clip & reel.mxf", 1, "hash");
        let xml = g.build_xml();
        assert!(xml.contains("clip &amp; reel.mxf"));
    }

    #[test]
    fn test_build_xml_has_prolog() {
        let g = PklGenerator::new();
        let xml = g.build_xml();
        assert!(xml.starts_with("<?xml"));
    }
}
