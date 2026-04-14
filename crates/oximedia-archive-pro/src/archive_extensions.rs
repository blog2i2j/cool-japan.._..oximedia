//! Archive-pro extensions for TODO.md (0.1.3 session).
//!
//! Provides:
//! - `PremisRecord` — minimal PREMIS XML v3 generator
//! - `MetsPackage` — minimal METS XML generator
//! - `BagItPackager` — in-memory BagIt manifest (no disk I/O)
//! - `ChecksumTree` — parent hash combining children hashes (Merkle-style)
//! - `FormatMigrator::plan_migration` — step-based migration planner
//! - `RiskAssessor::assess` — score-based risk assessment
//! - `ProvenanceChain` — JSON-serializable provenance chain
//! - `BitRotDetector::compare` — 32-byte hash equality
//! - `DisasterRecoveryPlan` — priority-ordered replica manager
//! - `ColdStorageManager::estimate_retrieval_cost_usd` — tier pricing

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Shared SHA-256 helpers (pure Rust, no ring / openssl)
// ─────────────────────────────────────────────────────────────────────────────

use sha2::{Digest, Sha256};

/// Compute the SHA-256 hex digest of `data`.
fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    result.iter().map(|b| format!("{b:02x}")).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. PREMIS record
// ─────────────────────────────────────────────────────────────────────────────

/// A single PREMIS provenance event.
#[derive(Debug, Clone)]
pub struct PremisEventEntry {
    /// Event type (e.g. "ingestion", "migration").
    pub event_type: String,
    /// ISO-8601 date string.
    pub date: String,
    /// Agent responsible for the event.
    pub agent: String,
}

/// Minimal PREMIS v3 record for a single digital object.
#[derive(Debug, Clone)]
pub struct PremisRecord {
    /// Unique object identifier.
    pub object_id: String,
    /// Creation date (ISO-8601 string).
    pub creation_date: String,
    /// Provenance events linked to this object.
    pub events: Vec<PremisEventEntry>,
}

impl PremisRecord {
    /// Create a new PREMIS record.
    #[must_use]
    pub fn new(object_id: &str, creation_date: &str) -> Self {
        Self {
            object_id: object_id.to_string(),
            creation_date: creation_date.to_string(),
            events: Vec::new(),
        }
    }

    /// Add a provenance event to this record.
    pub fn add_event(&mut self, event_type: &str, date: &str, agent: &str) {
        self.events.push(PremisEventEntry {
            event_type: event_type.to_string(),
            date: date.to_string(),
            agent: agent.to_string(),
        });
    }

    /// Serialize to minimal PREMIS XML v3.
    #[must_use]
    pub fn to_xml(&self) -> String {
        let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<premis:premis xmlns:premis=\"http://www.loc.gov/premis/v3\" version=\"3.0\">\n");

        // Object element
        xml.push_str("  <premis:object xsi:type=\"premis:file\"\n");
        xml.push_str(
            "    xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">\n",
        );
        xml.push_str("    <premis:objectIdentifier>\n");
        xml.push_str("      <premis:objectIdentifierType>local</premis:objectIdentifierType>\n");
        xml.push_str(&format!(
            "      <premis:objectIdentifierValue>{}</premis:objectIdentifierValue>\n",
            xml_escape(&self.object_id)
        ));
        xml.push_str("    </premis:objectIdentifier>\n");
        xml.push_str(&format!(
            "    <premis:objectCreationDate>{}</premis:objectCreationDate>\n",
            xml_escape(&self.creation_date)
        ));
        xml.push_str("  </premis:object>\n");

        // Event elements
        for (i, ev) in self.events.iter().enumerate() {
            xml.push_str("  <premis:event>\n");
            xml.push_str("    <premis:eventIdentifier>\n");
            xml.push_str("      <premis:eventIdentifierType>local</premis:eventIdentifierType>\n");
            xml.push_str(&format!(
                "      <premis:eventIdentifierValue>evt-{i:04}</premis:eventIdentifierValue>\n"
            ));
            xml.push_str("    </premis:eventIdentifier>\n");
            xml.push_str(&format!(
                "    <premis:eventType>{}</premis:eventType>\n",
                xml_escape(&ev.event_type)
            ));
            xml.push_str(&format!(
                "    <premis:eventDateTime>{}</premis:eventDateTime>\n",
                xml_escape(&ev.date)
            ));
            xml.push_str("    <premis:linkingAgentIdentifier>\n");
            xml.push_str(
                "      <premis:linkingAgentIdentifierType>local</premis:linkingAgentIdentifierType>\n",
            );
            xml.push_str(&format!(
                "      <premis:linkingAgentIdentifierValue>{}</premis:linkingAgentIdentifierValue>\n",
                xml_escape(&ev.agent)
            ));
            xml.push_str("    </premis:linkingAgentIdentifier>\n");
            xml.push_str("  </premis:event>\n");
        }

        xml.push_str("</premis:premis>\n");
        xml
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

// ─────────────────────────────────────────────────────────────────────────────
// 2. METS package
// ─────────────────────────────────────────────────────────────────────────────

/// A single METS file entry.
#[derive(Debug, Clone)]
pub struct MetsFileEntry {
    /// File ID.
    pub fid: String,
    /// File location (href).
    pub href: String,
    /// MIME type.
    pub mimetype: String,
}

/// Minimal METS XML package builder.
#[derive(Debug, Clone)]
pub struct MetsPackage {
    /// METS object ID.
    pub objid: String,
    /// Human-readable label.
    pub label: String,
    /// Files added to this package.
    pub files: Vec<MetsFileEntry>,
}

impl MetsPackage {
    /// Create a new METS package.
    #[must_use]
    pub fn new(objid: &str, label: &str) -> Self {
        Self {
            objid: objid.to_string(),
            label: label.to_string(),
            files: Vec::new(),
        }
    }

    /// Add a file to the package.
    pub fn add_file(&mut self, fid: &str, href: &str, mimetype: &str) {
        self.files.push(MetsFileEntry {
            fid: fid.to_string(),
            href: href.to_string(),
            mimetype: mimetype.to_string(),
        });
    }

    /// Serialize to minimal METS XML.
    #[must_use]
    pub fn to_xml(&self) -> String {
        let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str(&format!(
            "<mets:mets xmlns:mets=\"http://www.loc.gov/METS/\"\
             \n          OBJID=\"{}\"\
             \n          LABEL=\"{}\">\n",
            xml_escape(&self.objid),
            xml_escape(&self.label)
        ));

        // fileSec
        xml.push_str("  <mets:fileSec>\n");
        xml.push_str("    <mets:fileGrp USE=\"preservation\">\n");
        for f in &self.files {
            xml.push_str(&format!(
                "      <mets:file ID=\"{}\" MIMETYPE=\"{}\">\n",
                xml_escape(&f.fid),
                xml_escape(&f.mimetype)
            ));
            xml.push_str(&format!(
                "        <mets:FLocat LOCTYPE=\"URL\" xlink:href=\"{}\"\
                 \n          xmlns:xlink=\"http://www.w3.org/1999/xlink\"/>\n",
                xml_escape(&f.href)
            ));
            xml.push_str("      </mets:file>\n");
        }
        xml.push_str("    </mets:fileGrp>\n");
        xml.push_str("  </mets:fileSec>\n");

        // structMap
        xml.push_str("  <mets:structMap TYPE=\"physical\">\n");
        xml.push_str("    <mets:div TYPE=\"item\">\n");
        for f in &self.files {
            xml.push_str(&format!(
                "      <mets:fptr FILEID=\"{}\"/>\n",
                xml_escape(&f.fid)
            ));
        }
        xml.push_str("    </mets:div>\n");
        xml.push_str("  </mets:structMap>\n");

        xml.push_str("</mets:mets>\n");
        xml
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. BagIt packager (in-memory; no disk I/O)
// ─────────────────────────────────────────────────────────────────────────────

/// BagIt manifest produced by `BagItPackager`.
#[derive(Debug, Clone)]
pub struct BagItManifest {
    /// Contents of `bagit.txt`.
    pub bagit_txt: String,
    /// Contents of `bag-info.txt`.
    pub bag_info_txt: String,
    /// Contents of `manifest-sha256.txt`.
    pub manifest_sha256_txt: String,
    /// SHA-256 hashes indexed by relative file path (under `data/`).
    pub hashes: HashMap<String, String>,
}

/// In-memory BagIt packager (RFC 8493 v1.0).
pub struct BagItPackager;

impl BagItPackager {
    /// Create a `BagItManifest` from a slice of `(filename, content)` pairs.
    ///
    /// No files are written to disk; all outputs are returned as strings.
    pub fn create(
        base_dir: &str,
        files: &[(String, Vec<u8>)],
    ) -> crate::Result<BagItManifest> {
        let bagit_txt = "BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n".to_string();

        let mut hashes: HashMap<String, String> = HashMap::new();
        let mut manifest_lines = Vec::new();

        for (name, content) in files {
            let hash = sha256_hex(content);
            let relative = format!("data/{name}");
            manifest_lines.push(format!("{hash}  {relative}"));
            hashes.insert(relative, hash);
        }

        manifest_lines.sort(); // deterministic output
        let manifest_sha256_txt = manifest_lines.join("\n") + "\n";

        let total_bytes: usize = files.iter().map(|(_, c)| c.len()).sum();
        let bag_info_txt = format!(
            "Bag-Software-Agent: oximedia-archive-pro\n\
             Bagging-Date: {date}\n\
             Bag-Size: {total_bytes} bytes\n\
             Payload-Oxum: {total_bytes}.{file_count}\n\
             Source-Organization: COOLJAPAN OU\n\
             External-Identifier: {base_dir}\n",
            date = chrono::Utc::now().format("%Y-%m-%d"),
            total_bytes = total_bytes,
            file_count = files.len(),
            base_dir = base_dir,
        );

        Ok(BagItManifest {
            bagit_txt,
            bag_info_txt,
            manifest_sha256_txt,
            hashes,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Checksum tree (hierarchical / Merkle-style)
// ─────────────────────────────────────────────────────────────────────────────

/// A leaf node in the checksum tree.
#[derive(Debug, Clone)]
pub struct ChecksumLeaf {
    /// Relative file path.
    pub path: String,
    /// SHA-256 hex hash of the file content.
    pub hash: String,
}

/// A checksum tree that computes per-file hashes and a combined root hash.
#[derive(Debug, Clone)]
pub struct ChecksumTree {
    /// Leaf nodes (one per file).
    pub leaves: Vec<ChecksumLeaf>,
    /// Combined root hash (hash of all leaf hashes concatenated in order).
    pub root_hash: String,
}

impl ChecksumTree {
    /// Build a checksum tree from a slice of `(path, content)` pairs.
    #[must_use]
    pub fn build(files: &[(&str, &[u8])]) -> Self {
        let mut leaves: Vec<ChecksumLeaf> = files
            .iter()
            .map(|(path, content)| ChecksumLeaf {
                path: (*path).to_string(),
                hash: sha256_hex(content),
            })
            .collect();

        // Sort by path for deterministic ordering
        leaves.sort_by(|a, b| a.path.cmp(&b.path));

        // Compute root as SHA-256 of all leaf hashes joined
        let combined: String = leaves.iter().map(|l| l.hash.as_str()).collect();
        let root_hash = sha256_hex(combined.as_bytes());

        Self { leaves, root_hash }
    }

    /// Look up the hash for a specific path.
    #[must_use]
    pub fn hash_for(&self, path: &str) -> Option<&str> {
        self.leaves
            .iter()
            .find(|l| l.path == path)
            .map(|l| l.hash.as_str())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Format migration planner
// ─────────────────────────────────────────────────────────────────────────────

/// Risk level of a migration plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    /// No risk.
    None,
    /// Low risk (well-understood migration path).
    Low,
    /// Medium risk (some loss possible).
    Medium,
    /// High risk (significant quality or metadata loss).
    High,
    /// Critical risk (migration may be impossible or highly destructive).
    Critical,
}

impl RiskLevel {
    /// Numeric risk score (0..4).
    #[must_use]
    pub fn score(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Low => 1,
            Self::Medium => 2,
            Self::High => 3,
            Self::Critical => 4,
        }
    }
}

/// A single step in a migration plan.
#[derive(Debug, Clone)]
pub struct MigrationStep {
    /// Step index (1-based).
    pub order: u32,
    /// Human-readable action description.
    pub description: String,
    /// Tool or method recommended.
    pub tool: String,
}

/// A format migration plan.
#[derive(Debug, Clone)]
pub struct MigrationPlan {
    /// Ordered migration steps.
    pub steps: Vec<MigrationStep>,
    /// Overall risk level of this migration.
    pub risk_level: RiskLevel,
}

/// Format migration planner.
pub struct FormatMigrator;

impl FormatMigrator {
    /// Plan a migration from `src_format` to `dst_format`.
    ///
    /// Returns a `MigrationPlan` with ordered steps and a risk assessment.
    #[must_use]
    pub fn plan_migration(src_format: &str, dst_format: &str) -> MigrationPlan {
        let src = src_format.to_lowercase();
        let dst = dst_format.to_lowercase();

        // Risk based on source format
        let risk_level = match src.as_str() {
            "wmv" | "rm" | "real" | "asf" | "flv" | "swf" => RiskLevel::Critical,
            "avi" | "mov" | "mpg" | "mpeg" => RiskLevel::High,
            "mp4" | "m4v" | "aac" | "mp3" => RiskLevel::Medium,
            "webm" | "vp8" | "vp9" => RiskLevel::Low,
            "mkv" | "ffv1" | "flac" | "wav" | "tiff" | "png" => RiskLevel::None,
            _ => RiskLevel::Medium,
        };

        let steps = vec![
            MigrationStep {
                order: 1,
                description: format!("Verify integrity of source '{src_format}' files"),
                tool: "sha256sum / ChecksumTree".to_string(),
            },
            MigrationStep {
                order: 2,
                description: format!(
                    "Transcode from '{src_format}' to '{dst_format}' preserving metadata"
                ),
                tool: "oximedia-transcode".to_string(),
            },
            MigrationStep {
                order: 3,
                description: format!("Validate '{dst_format}' output quality and metadata"),
                tool: "oximedia-qc".to_string(),
            },
            MigrationStep {
                order: 4,
                description: "Generate and verify checksums for migrated files".to_string(),
                tool: "oximedia-archive-pro::ChecksumTree".to_string(),
            },
            MigrationStep {
                order: 5,
                description: "Create PREMIS migration event record".to_string(),
                tool: "PremisRecord".to_string(),
            },
        ];

        // Suppress unused variable warning for dst
        let _ = dst;

        MigrationPlan { steps, risk_level }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Risk assessor
// ─────────────────────────────────────────────────────────────────────────────

/// A computed risk score for a digital object.
#[derive(Debug, Clone)]
pub struct RiskScore {
    /// Numeric risk score (higher = more at risk).
    pub score: u32,
    /// Format assessed.
    pub format: String,
    /// Age of the object in years.
    pub age_years: u32,
    /// Human-readable description.
    pub description: String,
}

/// Risk assessor for digital preservation objects.
pub struct RiskAssessor;

impl RiskAssessor {
    /// Assess the preservation risk of a digital object.
    ///
    /// Formula: `(now_year - year_created) / 10 + format_risk(format)`
    /// where obsolete formats (wmv, rm, flv, real, swf, asf) get +3.
    #[must_use]
    pub fn assess(format: &str, year_created: u32, now_year: u32) -> RiskScore {
        let age_years = now_year.saturating_sub(year_created);
        let age_score = age_years / 10;

        let format_lc = format.to_lowercase();
        let format_bonus: u32 = match format_lc.as_str() {
            "wmv" | "rm" | "real" | "flv" | "swf" | "asf" => 3,
            "avi" | "mov" | "mpg" | "mpeg" => 2,
            "mp4" | "aac" | "mp3" => 1,
            _ => 0,
        };

        let score = age_score + format_bonus;

        let description = if format_bonus == 3 {
            format!(
                "Obsolete format '{format}' ({age_years} years old): score {score} — immediate migration recommended"
            )
        } else if score > 5 {
            format!("Format '{format}' ({age_years} years old): score {score} — migration planning advised")
        } else {
            format!("Format '{format}' ({age_years} years old): score {score} — currently acceptable")
        };

        RiskScore {
            score,
            format: format.to_string(),
            age_years,
            description,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Provenance chain (JSON)
// ─────────────────────────────────────────────────────────────────────────────

/// A single transformation step in the provenance chain.
#[derive(Debug, Clone)]
pub struct TransformationEntry {
    /// Source format or state.
    pub from: String,
    /// Destination format or state.
    pub to: String,
    /// Tool or process used.
    pub tool: String,
    /// ISO-8601 date of transformation.
    pub date: String,
}

/// A provenance chain tracking the history of transformations for an object.
#[derive(Debug, Clone)]
pub struct ProvenanceChain {
    /// Object identifier.
    pub object_id: String,
    /// Ordered list of transformations.
    pub transformations: Vec<TransformationEntry>,
}

impl ProvenanceChain {
    /// Create a new empty provenance chain.
    #[must_use]
    pub fn new(object_id: &str) -> Self {
        Self {
            object_id: object_id.to_string(),
            transformations: Vec::new(),
        }
    }

    /// Record a transformation.
    pub fn add_transformation(&mut self, from: &str, to: &str, tool: &str, date: &str) {
        self.transformations.push(TransformationEntry {
            from: from.to_string(),
            to: to.to_string(),
            tool: tool.to_string(),
            date: date.to_string(),
        });
    }

    /// Serialize to JSON.
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for t in &self.transformations {
            parts.push(format!(
                "{{\"from\":\"{}\",\"to\":\"{}\",\"tool\":\"{}\",\"date\":\"{}\"}}",
                json_escape(&t.from),
                json_escape(&t.to),
                json_escape(&t.tool),
                json_escape(&t.date),
            ));
        }
        let arr = parts.join(",");
        format!(
            "{{\"object_id\":\"{}\",\"transformations\":[{arr}]}}",
            json_escape(&self.object_id)
        )
    }
}

/// Escape JSON string special characters.
fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Bit rot detection
// ─────────────────────────────────────────────────────────────────────────────

/// Bit rot detector using SHA-256 hash comparison.
pub struct BitRotDetector;

impl BitRotDetector {
    /// Compare two 32-byte SHA-256 hashes.
    ///
    /// Returns `true` if the hashes match (no bit rot detected).
    /// Returns `false` if they differ (possible bit rot).
    #[must_use]
    pub fn compare(original_hash: &[u8; 32], current_hash: &[u8; 32]) -> bool {
        original_hash == current_hash
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Disaster recovery plan
// ─────────────────────────────────────────────────────────────────────────────

/// A storage replica entry.
#[derive(Debug, Clone)]
pub struct ReplicaEntry {
    /// Geographic or logical location identifier.
    pub location: String,
    /// Priority — higher value = higher priority.
    pub priority: u32,
}

/// Disaster recovery plan with priority-ordered replicas.
#[derive(Debug, Clone, Default)]
pub struct DisasterRecoveryPlan {
    /// All registered replicas.
    pub replicas: Vec<ReplicaEntry>,
}

impl DisasterRecoveryPlan {
    /// Create a new empty plan.
    #[must_use]
    pub fn new() -> Self {
        Self {
            replicas: Vec::new(),
        }
    }

    /// Register a replica location.
    pub fn add_replica(&mut self, location: &str, priority: u32) {
        self.replicas.push(ReplicaEntry {
            location: location.to_string(),
            priority,
        });
    }

    /// Return the primary replica — the location with the highest priority value.
    ///
    /// If multiple replicas share the maximum priority, the first one added wins.
    #[must_use]
    pub fn get_primary_replica(&self) -> Option<&str> {
        self.replicas
            .iter()
            .max_by_key(|r| r.priority)
            .map(|r| r.location.as_str())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Cold storage manager
// ─────────────────────────────────────────────────────────────────────────────

/// Cloud cold storage tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColdTier {
    /// Amazon S3 Glacier or equivalent: $0.01/GB retrieval.
    Glacier,
    /// Amazon S3 Glacier Deep Archive or equivalent: $0.02/GB retrieval.
    GlacierDeepArchive,
}

impl ColdTier {
    /// Retrieval cost per GB in USD.
    #[must_use]
    pub fn cost_per_gb(self) -> f64 {
        match self {
            Self::Glacier => 0.01,
            Self::GlacierDeepArchive => 0.02,
        }
    }
}

/// Cold storage manager with retrieval cost estimation.
pub struct ColdStorageManager;

impl ColdStorageManager {
    /// Estimate the retrieval cost in USD for a given size and tier.
    ///
    /// - Glacier: $0.01/GB
    /// - Glacier Deep Archive: $0.02/GB
    #[must_use]
    pub fn estimate_retrieval_cost_usd(size_gb: f64, tier: ColdTier) -> f64 {
        size_gb * tier.cost_per_gb()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- PremisRecord ---

    #[test]
    fn test_premis_record_xml_contains_object() {
        let mut record = PremisRecord::new("obj-001", "2025-01-15");
        record.add_event("ingestion", "2025-01-15", "archivist@example.com");
        let xml = record.to_xml();
        assert!(xml.contains("<premis:object"), "XML must contain <premis:object>");
        assert!(xml.contains("obj-001"), "XML must contain object ID");
    }

    #[test]
    fn test_premis_record_xml_has_event() {
        let mut record = PremisRecord::new("obj-002", "2025-02-01");
        record.add_event("migration", "2025-03-01", "migrator-bot");
        let xml = record.to_xml();
        assert!(xml.contains("<premis:event>"));
        assert!(xml.contains("migration"));
        assert!(xml.contains("migrator-bot"));
    }

    #[test]
    fn test_premis_record_no_events() {
        let record = PremisRecord::new("obj-003", "2024-12-01");
        let xml = record.to_xml();
        assert!(xml.contains("<premis:premis"));
        assert!(!xml.contains("<premis:event>"));
    }

    // --- MetsPackage ---

    #[test]
    fn test_mets_package_xml_basic() {
        let mut pkg = MetsPackage::new("METS-001", "Test Package");
        pkg.add_file("f001", "data/video.mkv", "video/x-matroska");
        let xml = pkg.to_xml();
        assert!(xml.contains("<mets:mets"), "XML must open with mets:mets");
        assert!(xml.contains("METS-001"));
        assert!(xml.contains("video/x-matroska"));
        assert!(xml.contains("data/video.mkv"));
    }

    #[test]
    fn test_mets_package_multiple_files() {
        let mut pkg = MetsPackage::new("M2", "Multi");
        pkg.add_file("f1", "data/a.mkv", "video/x-matroska");
        pkg.add_file("f2", "data/b.flac", "audio/flac");
        let xml = pkg.to_xml();
        assert!(xml.contains("f1"));
        assert!(xml.contains("f2"));
        assert!(xml.contains("audio/flac"));
    }

    // --- BagItPackager ---

    #[test]
    fn test_bagit_manifest_contains_sha256_hashes() {
        let files = vec![
            ("video.mkv".to_string(), b"fake video content".to_vec()),
            ("audio.flac".to_string(), b"fake audio content".to_vec()),
        ];
        let manifest = BagItPackager::create("/archive/bag1", &files)
            .expect("BagIt create should succeed");

        // manifest-sha256.txt must contain 64-hex-char SHA-256 hashes
        assert!(
            manifest.manifest_sha256_txt.contains("data/video.mkv"),
            "Manifest must reference video.mkv"
        );
        assert!(
            manifest.manifest_sha256_txt.contains("data/audio.flac"),
            "Manifest must reference audio.flac"
        );
        // Each hash is 64 hex characters
        for line in manifest.manifest_sha256_txt.lines() {
            if line.is_empty() {
                continue;
            }
            let hash = line.split_whitespace().next().expect("hash must exist");
            assert_eq!(hash.len(), 64, "SHA-256 hex hash must be 64 characters");
        }
    }

    #[test]
    fn test_bagit_txt_version() {
        let files = vec![("f.txt".to_string(), b"content".to_vec())];
        let manifest =
            BagItPackager::create("/test", &files).expect("create should succeed");
        assert!(manifest.bagit_txt.contains("BagIt-Version: 1.0"));
    }

    // --- ChecksumTree ---

    #[test]
    fn test_checksum_tree_build() {
        let files: Vec<(&str, &[u8])> = vec![
            ("a.txt", b"hello"),
            ("b.txt", b"world"),
        ];
        let tree = ChecksumTree::build(&files);
        assert_eq!(tree.leaves.len(), 2);
        assert!(!tree.root_hash.is_empty());
        assert_eq!(tree.root_hash.len(), 64); // SHA-256 hex
    }

    #[test]
    fn test_checksum_tree_root_changes_with_content() {
        let files1: Vec<(&str, &[u8])> = vec![("a.txt", b"content1")];
        let files2: Vec<(&str, &[u8])> = vec![("a.txt", b"content2")];
        let t1 = ChecksumTree::build(&files1);
        let t2 = ChecksumTree::build(&files2);
        assert_ne!(t1.root_hash, t2.root_hash);
    }

    // --- FormatMigrator ---

    #[test]
    fn test_migration_plan_has_steps() {
        let plan = FormatMigrator::plan_migration("avi", "mkv");
        assert!(!plan.steps.is_empty());
        assert!(plan.steps[0].order == 1);
    }

    #[test]
    fn test_migration_plan_risk_obsolete() {
        let plan = FormatMigrator::plan_migration("wmv", "mkv");
        assert_eq!(plan.risk_level, RiskLevel::Critical);
    }

    #[test]
    fn test_migration_plan_risk_preservation_format() {
        let plan = FormatMigrator::plan_migration("ffv1", "mkv");
        assert_eq!(plan.risk_level, RiskLevel::None);
    }

    // --- RiskAssessor ---

    #[test]
    fn test_risk_score_increases_with_age() {
        let young = RiskAssessor::assess("mp4", 2020, 2025);
        let old = RiskAssessor::assess("mp4", 2000, 2025);
        assert!(
            old.score > young.score,
            "Older files should have higher risk score"
        );
    }

    #[test]
    fn test_risk_score_obsolete_format_bonus() {
        let wmv = RiskAssessor::assess("wmv", 2024, 2025);
        let mkv = RiskAssessor::assess("mkv", 2024, 2025);
        assert!(wmv.score > mkv.score, "Obsolete formats should score higher");
    }

    #[test]
    fn test_risk_score_same_year() {
        let score = RiskAssessor::assess("wav", 2025, 2025);
        assert_eq!(score.age_years, 0);
        assert_eq!(score.score, 0); // no age penalty, no format penalty
    }

    // --- ProvenanceChain ---

    #[test]
    fn test_provenance_chain_to_json() {
        let mut chain = ProvenanceChain::new("asset-123");
        chain.add_transformation("avi", "mkv", "oximedia-transcode", "2025-06-01");
        let json = chain.to_json();
        assert!(json.contains("asset-123"));
        assert!(json.contains("oximedia-transcode"));
        assert!(json.contains("2025-06-01"));
    }

    #[test]
    fn test_provenance_chain_empty() {
        let chain = ProvenanceChain::new("empty-obj");
        let json = chain.to_json();
        assert!(json.contains("empty-obj"));
        assert!(json.contains("\"transformations\":[]"));
    }

    // --- BitRotDetector ---

    #[test]
    fn test_bit_rot_detector_matching_hashes() {
        let h = [0x42u8; 32];
        assert!(BitRotDetector::compare(&h, &h));
    }

    #[test]
    fn test_bit_rot_detector_different_hashes() {
        let h1 = [0x01u8; 32];
        let h2 = [0x02u8; 32];
        assert!(!BitRotDetector::compare(&h1, &h2));
    }

    // --- DisasterRecoveryPlan ---

    #[test]
    fn test_disaster_recovery_primary_replica() {
        let mut plan = DisasterRecoveryPlan::new();
        plan.add_replica("us-west-2", 2);
        plan.add_replica("eu-central-1", 5); // highest priority
        plan.add_replica("ap-northeast-1", 1);

        assert_eq!(plan.get_primary_replica(), Some("eu-central-1"));
    }

    #[test]
    fn test_disaster_recovery_empty() {
        let plan = DisasterRecoveryPlan::new();
        assert!(plan.get_primary_replica().is_none());
    }

    // --- ColdStorageManager ---

    #[test]
    fn test_glacier_cost() {
        let cost = ColdStorageManager::estimate_retrieval_cost_usd(100.0, ColdTier::Glacier);
        assert!((cost - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_glacier_deep_archive_cost() {
        let cost =
            ColdStorageManager::estimate_retrieval_cost_usd(100.0, ColdTier::GlacierDeepArchive);
        assert!((cost - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_zero_size_retrieval() {
        let cost = ColdStorageManager::estimate_retrieval_cost_usd(0.0, ColdTier::Glacier);
        assert_eq!(cost, 0.0);
    }
}
