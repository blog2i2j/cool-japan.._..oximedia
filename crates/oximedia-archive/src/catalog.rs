//! Archive catalog management.
//!
//! Provides catalog entries, access control, full-text search, date-range
//! search, CSV import/export, and OAI-PMH XML export.

#![allow(dead_code)]

/// Access control level for catalog entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessLevel {
    /// Freely available to all.
    Public,
    /// Available to authenticated users.
    Restricted,
    /// Sensitive; accessible only to named individuals.
    Confidential,
    /// Internal staff use only.
    Internal,
}

impl AccessLevel {
    /// Returns `true` if a user with the given `user_role` may access this level.
    ///
    /// Role hierarchy (most privileged first): `"admin"`, `"staff"`, `"user"`, anything else.
    #[must_use]
    pub fn can_access(&self, user_role: &str) -> bool {
        match self {
            Self::Public => true,
            Self::Restricted => matches!(user_role, "admin" | "staff" | "user"),
            Self::Internal => matches!(user_role, "admin" | "staff"),
            Self::Confidential => user_role == "admin",
        }
    }

    /// Short string label.
    #[must_use]
    pub const fn label(&self) -> &str {
        match self {
            Self::Public => "public",
            Self::Restricted => "restricted",
            Self::Confidential => "confidential",
            Self::Internal => "internal",
        }
    }
}

/// A single catalog record describing a media asset.
#[derive(Clone, Debug)]
pub struct CatalogEntry {
    /// Unique identifier.
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// Extended description.
    pub description: String,
    /// Creation timestamp (Unix milliseconds).
    pub date_created_ms: u64,
    /// File format identifier (e.g., `"dpx"`, `"mp4"`).
    pub format: String,
    /// Duration in seconds, if applicable.
    pub duration_secs: Option<f64>,
    /// Physical shelf location, if applicable.
    pub physical_location: Option<String>,
    /// Path or URI to the digital file.
    pub digital_path: Option<String>,
    /// Rights statement or licence.
    pub rights: String,
    /// Access control level.
    pub access_level: AccessLevel,
}

impl CatalogEntry {
    /// Create a new catalog entry with mandatory fields.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        format: impl Into<String>,
        date_created_ms: u64,
        rights: impl Into<String>,
        access_level: AccessLevel,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            description: String::new(),
            date_created_ms,
            format: format.into(),
            duration_secs: None,
            physical_location: None,
            digital_path: None,
            rights: rights.into(),
            access_level,
        }
    }
}

/// In-memory catalog index supporting search operations.
#[derive(Default)]
pub struct CatalogIndex {
    entries: Vec<CatalogEntry>,
}

impl CatalogIndex {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entry to the index.
    pub fn add(&mut self, entry: CatalogEntry) {
        self.entries.push(entry);
    }

    /// Search entries whose `title` contains `query` (case-insensitive).
    #[must_use]
    pub fn search_by_title(&self, query: &str) -> Vec<&CatalogEntry> {
        let q = query.to_lowercase();
        self.entries
            .iter()
            .filter(|e| e.title.to_lowercase().contains(&q))
            .collect()
    }

    /// Return entries created within the given millisecond timestamp range (inclusive).
    #[must_use]
    pub fn search_by_date_range(&self, start_ms: u64, end_ms: u64) -> Vec<&CatalogEntry> {
        self.entries
            .iter()
            .filter(|e| e.date_created_ms >= start_ms && e.date_created_ms <= end_ms)
            .collect()
    }

    /// Total number of entries in the index.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.entries.len()
    }

    /// Look up an entry by its exact ID.
    #[must_use]
    pub fn get_by_id(&self, id: &str) -> Option<&CatalogEntry> {
        self.entries.iter().find(|e| e.id == id)
    }
}

/// Catalog export utilities.
pub struct CatalogExport;

impl CatalogExport {
    /// Export entries to CSV format.
    ///
    /// Columns: id, title, format, date_created_ms, duration_secs, rights, access_level
    #[must_use]
    pub fn to_csv(entries: &[CatalogEntry]) -> String {
        let mut out =
            String::from("id,title,format,date_created_ms,duration_secs,rights,access_level\n");
        for e in entries {
            let duration = e.duration_secs.map(|d| d.to_string()).unwrap_or_default();
            out.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                csv_escape(&e.id),
                csv_escape(&e.title),
                csv_escape(&e.format),
                e.date_created_ms,
                duration,
                csv_escape(&e.rights),
                e.access_level.label(),
            ));
        }
        out
    }

    /// Export entries as a minimal OAI-PMH `ListRecords` XML response.
    #[must_use]
    pub fn to_oai_pmh(entries: &[CatalogEntry]) -> String {
        let mut out = String::from(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <OAI-PMH xmlns=\"http://www.openarchives.org/OAI/2.0/\">\n\
             <responseDate>2026-01-01T00:00:00Z</responseDate>\n\
             <request verb=\"ListRecords\"/>\n\
             <ListRecords>\n",
        );

        for e in entries {
            out.push_str("  <record>\n    <header>\n");
            out.push_str(&format!(
                "      <identifier>{}</identifier>\n",
                xml_escape(&e.id)
            ));
            out.push_str(&format!(
                "      <datestamp>{}</datestamp>\n",
                ms_to_iso8601(e.date_created_ms)
            ));
            out.push_str("    </header>\n    <metadata>\n      <oai_dc:dc\n");
            out.push_str("        xmlns:oai_dc=\"http://www.openarchives.org/OAI/2.0/oai_dc/\"\n");
            out.push_str("        xmlns:dc=\"http://purl.org/dc/elements/1.1/\">\n");
            out.push_str(&format!(
                "        <dc:title>{}</dc:title>\n",
                xml_escape(&e.title)
            ));
            if !e.description.is_empty() {
                out.push_str(&format!(
                    "        <dc:description>{}</dc:description>\n",
                    xml_escape(&e.description)
                ));
            }
            out.push_str(&format!(
                "        <dc:format>{}</dc:format>\n",
                xml_escape(&e.format)
            ));
            out.push_str(&format!(
                "        <dc:rights>{}</dc:rights>\n",
                xml_escape(&e.rights)
            ));
            out.push_str("      </oai_dc:dc>\n    </metadata>\n  </record>\n");
        }

        out.push_str("</ListRecords>\n</OAI-PMH>");
        out
    }
}

/// Catalog import utilities.
pub struct CatalogImport;

impl CatalogImport {
    /// Parse catalog entries from a CSV string.
    ///
    /// Expects the header row `id,title,format,date_created_ms,duration_secs,rights,access_level`.
    /// Lines that cannot be parsed are silently skipped.
    #[must_use]
    pub fn from_csv(csv: &str) -> Vec<CatalogEntry> {
        let mut entries = Vec::new();
        let mut lines = csv.lines();

        // Skip header
        if lines.next().is_none() {
            return entries;
        }

        for line in lines {
            let cols: Vec<&str> = line.splitn(7, ',').collect();
            if cols.len() < 7 {
                continue;
            }

            let id = csv_unescape(cols[0]);
            let title = csv_unescape(cols[1]);
            let format = csv_unescape(cols[2]);
            let date_created_ms: u64 = cols[3].trim().parse().unwrap_or(0);
            let duration_secs: Option<f64> = cols[4]
                .trim()
                .parse()
                .ok()
                .filter(|_| !cols[4].trim().is_empty());
            let rights = csv_unescape(cols[5]);
            let access_level = match cols[6].trim() {
                "public" => AccessLevel::Public,
                "restricted" => AccessLevel::Restricted,
                "confidential" => AccessLevel::Confidential,
                "internal" => AccessLevel::Internal,
                _ => AccessLevel::Public,
            };

            let mut entry =
                CatalogEntry::new(id, title, format, date_created_ms, rights, access_level);
            entry.duration_secs = duration_secs;
            entries.push(entry);
        }

        entries
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Wrap a CSV field in quotes if it contains a comma, newline, or quote.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('\n') || s.contains('"') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Strip surrounding quotes from a CSV field.
fn csv_unescape(s: &str) -> String {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        s[1..s.len() - 1].replace("\"\"", "\"")
    } else {
        s.to_string()
    }
}

/// Convert Unix milliseconds to a simplified ISO 8601 date string.
fn ms_to_iso8601(ms: u64) -> String {
    let secs = ms / 1_000;
    let days = secs / 86_400;
    let year = 1970 + days / 365;
    // Approximate month/day
    let day_of_year = days % 365;
    let month = day_of_year / 30 + 1;
    let day = day_of_year % 30 + 1;
    format!("{year:04}-{month:02}-{day:02}T00:00:00Z")
}

/// Minimal XML character escaping.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry(id: &str, title: &str, ms: u64) -> CatalogEntry {
        CatalogEntry::new(id, title, "dpx", ms, "CC0", AccessLevel::Public)
    }

    #[test]
    fn test_access_level_public() {
        assert!(AccessLevel::Public.can_access("anyone"));
        assert!(AccessLevel::Public.can_access("guest"));
    }

    #[test]
    fn test_access_level_restricted() {
        assert!(AccessLevel::Restricted.can_access("user"));
        assert!(!AccessLevel::Restricted.can_access("guest"));
    }

    #[test]
    fn test_access_level_internal() {
        assert!(AccessLevel::Internal.can_access("staff"));
        assert!(!AccessLevel::Internal.can_access("user"));
    }

    #[test]
    fn test_access_level_confidential() {
        assert!(AccessLevel::Confidential.can_access("admin"));
        assert!(!AccessLevel::Confidential.can_access("staff"));
    }

    #[test]
    fn test_catalog_index_add_and_count() {
        let mut idx = CatalogIndex::new();
        idx.add(sample_entry("a1", "Sunset Reel", 1_000_000));
        idx.add(sample_entry("a2", "Night Scene", 2_000_000));
        assert_eq!(idx.total_count(), 2);
    }

    #[test]
    fn test_search_by_title() {
        let mut idx = CatalogIndex::new();
        idx.add(sample_entry("a1", "Sunset Reel", 1_000_000));
        idx.add(sample_entry("a2", "Night Scene", 2_000_000));
        idx.add(sample_entry("a3", "Sunset Beach", 3_000_000));

        let results = idx.search_by_title("sunset");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_by_date_range() {
        let mut idx = CatalogIndex::new();
        idx.add(sample_entry("a1", "A", 1_000));
        idx.add(sample_entry("a2", "B", 5_000));
        idx.add(sample_entry("a3", "C", 9_000));

        let results = idx.search_by_date_range(2_000, 8_000);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a2");
    }

    #[test]
    fn test_catalog_export_csv_header() {
        let entries = vec![sample_entry("id1", "My Film", 0)];
        let csv = CatalogExport::to_csv(&entries);
        assert!(csv.starts_with("id,title,format,"));
        assert!(csv.contains("My Film"));
    }

    #[test]
    fn test_catalog_export_oai_pmh() {
        let entries = vec![sample_entry("oai:1", "Test", 86_400_000)];
        let xml = CatalogExport::to_oai_pmh(&entries);
        assert!(xml.contains("<OAI-PMH"));
        assert!(xml.contains("<dc:title>Test</dc:title>"));
        assert!(xml.contains("oai:1"));
    }

    #[test]
    fn test_catalog_import_from_csv() {
        let csv = "id,title,format,date_created_ms,duration_secs,rights,access_level\n\
                   film001,My Documentary,mp4,1700000000000,3600.5,CC-BY,public\n";
        let entries = CatalogImport::from_csv(csv);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "film001");
        assert_eq!(entries[0].title, "My Documentary");
        assert_eq!(entries[0].format, "mp4");
        assert!((entries[0].duration_secs.expect("test expectation failed") - 3600.5).abs() < 1e-6);
        assert_eq!(entries[0].access_level, AccessLevel::Public);
    }

    #[test]
    fn test_catalog_csv_roundtrip() {
        let original = vec![
            sample_entry("r1", "Film A", 1_000_000),
            sample_entry("r2", "Film, B", 2_000_000),
        ];
        let csv = CatalogExport::to_csv(&original);
        let imported = CatalogImport::from_csv(&csv);
        assert_eq!(imported.len(), 2);
        assert_eq!(imported[0].id, original[0].id);
    }
}

// ── New asset-catalog types ───────────────────────────────────────────────────

/// Classification of a media asset.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssetKind {
    /// Motion picture or video clip.
    Video,
    /// Audio-only file.
    Audio,
    /// Still image.
    Image,
    /// Text document or PDF.
    Document,
    /// Subtitle or caption file.
    Subtitle,
    /// Sidecar / companion file.
    Sidecar,
}

impl AssetKind {
    /// Returns `true` for primary media kinds (Video, Audio, Image).
    #[must_use]
    pub const fn is_media(self) -> bool {
        matches!(self, Self::Video | Self::Audio | Self::Image)
    }
}

/// A single entry in the asset inventory.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct AssetCatalogEntry {
    /// Unique numeric identifier.
    pub id: u64,
    /// Relative or absolute path.
    pub path: String,
    /// Asset classification.
    pub kind: AssetKind,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Creation time as Unix epoch seconds.
    pub created_epoch: u64,
    /// Arbitrary string tags for this asset.
    pub tags: Vec<String>,
}

impl AssetCatalogEntry {
    /// Returns `true` if `t` is in the tag list (exact match).
    #[must_use]
    pub fn has_tag(&self, t: &str) -> bool {
        self.tags.iter().any(|tag| tag == t)
    }
}

/// In-memory searchable asset inventory.
#[allow(dead_code)]
#[derive(Default, Debug)]
pub struct ArchiveCatalog {
    /// All catalog entries.
    pub entries: Vec<AssetCatalogEntry>,
    next_id: u64,
}

impl ArchiveCatalog {
    /// Create an empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entry; assigns and returns a unique id.
    pub fn add(
        &mut self,
        path: impl Into<String>,
        kind: AssetKind,
        size_bytes: u64,
        created_epoch: u64,
        tags: Vec<String>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(AssetCatalogEntry {
            id,
            path: path.into(),
            kind,
            size_bytes,
            created_epoch,
            tags,
        });
        id
    }

    /// Find an entry by its numeric id.
    #[must_use]
    pub fn find_by_id(&self, id: u64) -> Option<&AssetCatalogEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Find all entries that carry the given tag.
    #[must_use]
    pub fn find_by_tag(&self, tag: &str) -> Vec<&AssetCatalogEntry> {
        self.entries.iter().filter(|e| e.has_tag(tag)).collect()
    }

    /// Find entries whose path contains `query` (case-insensitive).
    #[must_use]
    pub fn search_path(&self, query: &str) -> Vec<&AssetCatalogEntry> {
        let q = query.to_lowercase();
        self.entries
            .iter()
            .filter(|e| e.path.to_lowercase().contains(&q))
            .collect()
    }

    /// Sum of all entry sizes.
    #[must_use]
    pub fn total_size_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.size_bytes).sum()
    }

    /// Count of entries matching `kind`.
    #[must_use]
    pub fn kind_count(&self, kind: AssetKind) -> usize {
        self.entries.iter().filter(|e| e.kind == kind).count()
    }
}

#[cfg(test)]
mod asset_catalog_tests {
    use super::*;

    fn make_catalog() -> ArchiveCatalog {
        let mut c = ArchiveCatalog::new();
        c.add(
            "videos/intro.mp4",
            AssetKind::Video,
            1_000_000,
            1_000,
            vec!["featured".into()],
        );
        c.add(
            "audio/bg.wav",
            AssetKind::Audio,
            500_000,
            2_000,
            vec!["music".into(), "featured".into()],
        );
        c.add(
            "images/thumb.jpg",
            AssetKind::Image,
            200_000,
            3_000,
            vec!["thumbnail".into()],
        );
        c.add(
            "docs/readme.pdf",
            AssetKind::Document,
            50_000,
            4_000,
            vec![],
        );
        c
    }

    #[test]
    fn test_asset_kind_is_media_video() {
        assert!(AssetKind::Video.is_media());
    }

    #[test]
    fn test_asset_kind_is_media_audio() {
        assert!(AssetKind::Audio.is_media());
    }

    #[test]
    fn test_asset_kind_not_media_document() {
        assert!(!AssetKind::Document.is_media());
    }

    #[test]
    fn test_asset_kind_not_media_sidecar() {
        assert!(!AssetKind::Sidecar.is_media());
    }

    #[test]
    fn test_has_tag_true() {
        let c = make_catalog();
        assert!(c
            .find_by_id(0)
            .expect("find_by_id should succeed")
            .has_tag("featured"));
    }

    #[test]
    fn test_has_tag_false() {
        let c = make_catalog();
        assert!(!c
            .find_by_id(0)
            .expect("find_by_id should succeed")
            .has_tag("music"));
    }

    #[test]
    fn test_find_by_id_present() {
        let c = make_catalog();
        assert!(c.find_by_id(2).is_some());
    }

    #[test]
    fn test_find_by_id_missing() {
        let c = make_catalog();
        assert!(c.find_by_id(999).is_none());
    }

    #[test]
    fn test_find_by_tag() {
        let c = make_catalog();
        let results = c.find_by_tag("featured");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_path_case_insensitive() {
        let c = make_catalog();
        let results = c.search_path("VIDEOS");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "videos/intro.mp4");
    }

    #[test]
    fn test_total_size_bytes() {
        let c = make_catalog();
        assert_eq!(c.total_size_bytes(), 1_750_000);
    }

    #[test]
    fn test_kind_count_video() {
        let c = make_catalog();
        assert_eq!(c.kind_count(AssetKind::Video), 1);
    }

    #[test]
    fn test_kind_count_zero() {
        let c = make_catalog();
        assert_eq!(c.kind_count(AssetKind::Subtitle), 0);
    }
}
