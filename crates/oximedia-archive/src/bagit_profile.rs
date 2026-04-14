//! BagIt profile validation for digital preservation archives.
//!
//! Implements the BagIt Profile specification (version 1.3.0) which allows
//! archive producers to declare constraints on bag structure, tag files,
//! and payload manifests.
//!
//! A *profile* is a JSON document that lists required tag files, required tag
//! values, allowed manifest algorithms, and other constraints. This module
//! can parse such documents and validate that a given bag conforms to them.
//!
//! Reference: <https://bagit-profiles.github.io/bagit-profiles-specification/>

#![allow(dead_code)]

use crate::{ArchiveError, ArchiveResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Profile structures
// ---------------------------------------------------------------------------

/// A BagIt profile document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BagItProfile {
    /// URL or identifier for this profile.
    #[serde(rename = "BagIt-Profile-Info")]
    pub profile_info: ProfileInfo,

    /// Required tag files that must be present in the bag.
    /// Maps tag-file name → tag field constraints.
    #[serde(rename = "Bag-Info", default)]
    pub bag_info: HashMap<String, TagConstraint>,

    /// Manifests that must be present.
    /// e.g. `["sha256", "blake3"]`
    #[serde(rename = "Manifests-Required", default)]
    pub manifests_required: Vec<String>,

    /// Manifests that are allowed (empty means all are allowed).
    #[serde(rename = "Manifests-Allowed", default)]
    pub manifests_allowed: Vec<String>,

    /// Tag manifests that must be present.
    #[serde(rename = "Tag-Manifests-Required", default)]
    pub tag_manifests_required: Vec<String>,

    /// Tag manifests that are allowed (empty means all are allowed).
    #[serde(rename = "Tag-Manifests-Allowed", default)]
    pub tag_manifests_allowed: Vec<String>,

    /// Tag files that must be present beyond bagit.txt and bag-info.txt.
    #[serde(rename = "Tag-Files-Required", default)]
    pub tag_files_required: Vec<String>,

    /// Tag files that are allowed (empty means all are allowed).
    #[serde(rename = "Tag-Files-Allowed", default)]
    pub tag_files_allowed: Vec<String>,

    /// Whether the fetch.txt file is allowed.
    #[serde(rename = "Allow-Fetch.txt", default = "default_true")]
    pub allow_fetch_txt: bool,

    /// Whether `serialized` bags (zip, tar) are allowed.
    #[serde(rename = "Serialization", default)]
    pub serialization: SerializationPolicy,

    /// Accepted MIME types for serialized bags.
    #[serde(rename = "Accept-Serialization", default)]
    pub accept_serialization: Vec<String>,

    /// Required BagIt version(s).
    #[serde(rename = "BagIt-Version-Required", default)]
    pub bagit_version_required: Vec<String>,
}

fn default_true() -> bool {
    true
}

/// Metadata about the profile itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileInfo {
    /// Profile identifier URI.
    #[serde(rename = "BagIt-Profile-Identifier", default)]
    pub identifier: String,
    /// Human-readable source organization name.
    #[serde(rename = "Source-Organization", default)]
    pub source_organization: String,
    /// External description of the profile.
    #[serde(rename = "External-Description", default)]
    pub external_description: String,
    /// Version of this profile document.
    #[serde(rename = "Version", default)]
    pub version: String,
    /// Contact name.
    #[serde(rename = "Contact-Name", default)]
    pub contact_name: String,
    /// Contact email.
    #[serde(rename = "Contact-Email", default)]
    pub contact_email: String,
}

/// Constraint on a tag field value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagConstraint {
    /// Whether this tag is required.
    #[serde(rename = "required", default)]
    pub required: bool,

    /// If present, the field value must be one of these strings.
    #[serde(rename = "values", default)]
    pub values: Vec<String>,

    /// Human-readable description.
    #[serde(rename = "description", default)]
    pub description: String,

    /// Whether the field may be repeated.
    #[serde(rename = "repeatable", default = "default_true")]
    pub repeatable: bool,

    /// Recommended value (informational only).
    #[serde(rename = "recommended", default)]
    pub recommended: bool,
}

/// Bag serialization policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SerializationPolicy {
    /// Bags must not be serialized.
    Forbidden,
    /// Bags may optionally be serialized.
    #[default]
    Optional,
    /// Bags must be serialized.
    Required,
}

// ---------------------------------------------------------------------------
// Bag descriptor — lightweight representation of a bag on disk
// ---------------------------------------------------------------------------

/// A minimal in-memory descriptor of a BagIt bag.
#[derive(Debug, Clone)]
pub struct BagDescriptor {
    /// Root directory of the bag.
    pub root: PathBuf,
    /// BagIt version declared in `bagit.txt`.
    pub bagit_version: String,
    /// Tag encoding declared in `bagit.txt`.
    pub tag_encoding: String,
    /// Tag file names present in the bag root (relative, e.g. `"bag-info.txt"`).
    pub tag_files: HashSet<String>,
    /// Manifest algorithms present (e.g. `"sha256"`, `"blake3"`).
    pub manifest_algorithms: HashSet<String>,
    /// Tag manifest algorithms present.
    pub tag_manifest_algorithms: HashSet<String>,
    /// Whether `fetch.txt` is present.
    pub has_fetch_txt: bool,
    /// Parsed bag-info fields (lowercase key → values).
    pub bag_info_fields: HashMap<String, Vec<String>>,
}

impl BagDescriptor {
    /// Scan a directory to build a `BagDescriptor`.
    pub fn from_dir(root: &Path) -> ArchiveResult<Self> {
        let bagit_txt = root.join("bagit.txt");
        if !bagit_txt.exists() {
            return Err(ArchiveError::Validation(format!(
                "bagit.txt not found in {}",
                root.display()
            )));
        }

        let (version, encoding) = Self::parse_bagit_txt(&bagit_txt)?;

        let mut tag_files = HashSet::new();
        let mut manifest_algorithms = HashSet::new();
        let mut tag_manifest_algorithms = HashSet::new();
        let mut has_fetch_txt = false;

        for entry in std::fs::read_dir(root)? {
            let entry = entry?;
            let file_name = entry
                .file_name()
                .into_string()
                .map_err(|_| ArchiveError::Validation("non-UTF8 filename".to_string()))?;

            if entry.file_type()?.is_file() {
                let lower = file_name.to_lowercase();
                if lower == "fetch.txt" {
                    has_fetch_txt = true;
                } else if let Some(alg) = lower
                    .strip_prefix("manifest-")
                    .and_then(|s| s.strip_suffix(".txt"))
                {
                    manifest_algorithms.insert(alg.to_string());
                } else if let Some(alg) = lower
                    .strip_prefix("tagmanifest-")
                    .and_then(|s| s.strip_suffix(".txt"))
                {
                    tag_manifest_algorithms.insert(alg.to_string());
                } else {
                    tag_files.insert(file_name);
                }
            }
        }

        // Parse bag-info.txt
        let bag_info_fields = if root.join("bag-info.txt").exists() {
            Self::parse_bag_info(&root.join("bag-info.txt"))?
        } else {
            HashMap::new()
        };

        Ok(Self {
            root: root.to_path_buf(),
            bagit_version: version,
            tag_encoding: encoding,
            tag_files,
            manifest_algorithms,
            tag_manifest_algorithms,
            has_fetch_txt,
            bag_info_fields,
        })
    }

    fn parse_bagit_txt(path: &Path) -> ArchiveResult<(String, String)> {
        let content = std::fs::read_to_string(path)?;
        let mut version = String::from("1.0");
        let mut encoding = String::from("UTF-8");

        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("BagIt-Version:") {
                version = rest.trim().to_string();
            } else if let Some(rest) = line.strip_prefix("Tag-File-Character-Encoding:") {
                encoding = rest.trim().to_string();
            }
        }
        Ok((version, encoding))
    }

    fn parse_bag_info(path: &Path) -> ArchiveResult<HashMap<String, Vec<String>>> {
        let content = std::fs::read_to_string(path)?;
        let mut fields: HashMap<String, Vec<String>> = HashMap::new();
        let mut current_key: Option<String> = None;
        let mut current_val = String::new();

        for line in content.lines() {
            if line.starts_with(' ') || line.starts_with('\t') {
                // Folded line continuation
                current_val.push(' ');
                current_val.push_str(line.trim());
            } else if let Some((key, val)) = line.split_once(':') {
                // Save previous
                if let Some(k) = current_key.take() {
                    fields
                        .entry(k.to_lowercase())
                        .or_default()
                        .push(current_val.trim().to_string());
                }
                current_key = Some(key.trim().to_string());
                current_val = val.trim().to_string();
            }
        }
        if let Some(k) = current_key {
            fields
                .entry(k.to_lowercase())
                .or_default()
                .push(current_val.trim().to_string());
        }

        Ok(fields)
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// A single validation finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileViolation {
    /// Short error code.
    pub code: String,
    /// Human-readable description.
    pub message: String,
}

impl ProfileViolation {
    fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ProfileViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

/// The outcome of a profile validation run.
#[derive(Debug, Clone)]
pub struct ProfileValidationResult {
    /// Whether the bag is fully conformant.
    pub conforms: bool,
    /// All violations found.
    pub violations: Vec<ProfileViolation>,
    /// Warnings (non-conformance for recommended fields).
    pub warnings: Vec<String>,
}

impl ProfileValidationResult {
    /// Return `true` if there are no violations.
    pub fn is_conformant(&self) -> bool {
        self.conforms
    }
}

/// Validate a bag against a BagIt profile.
pub fn validate_bag_against_profile(
    bag: &BagDescriptor,
    profile: &BagItProfile,
) -> ProfileValidationResult {
    let mut violations = Vec::new();
    let mut warnings = Vec::new();

    // 1. Required manifests
    for alg in &profile.manifests_required {
        if !bag.manifest_algorithms.contains(alg.as_str()) {
            violations.push(ProfileViolation::new(
                "MANIFEST_MISSING",
                format!("Required manifest algorithm '{alg}' not present"),
            ));
        }
    }

    // 2. Allowed manifests (if non-empty, acts as whitelist)
    if !profile.manifests_allowed.is_empty() {
        let allowed: HashSet<&str> = profile
            .manifests_allowed
            .iter()
            .map(String::as_str)
            .collect();
        for alg in &bag.manifest_algorithms {
            if !allowed.contains(alg.as_str()) {
                violations.push(ProfileViolation::new(
                    "MANIFEST_DISALLOWED",
                    format!("Manifest algorithm '{alg}' is not in the allowed list"),
                ));
            }
        }
    }

    // 3. Required tag manifests
    for alg in &profile.tag_manifests_required {
        if !bag.tag_manifest_algorithms.contains(alg.as_str()) {
            violations.push(ProfileViolation::new(
                "TAG_MANIFEST_MISSING",
                format!("Required tag-manifest algorithm '{alg}' not present"),
            ));
        }
    }

    // 4. Required tag files
    let bag_tag_files_lower: HashSet<String> =
        bag.tag_files.iter().map(|s| s.to_lowercase()).collect();
    for tf in &profile.tag_files_required {
        if !bag_tag_files_lower.contains(&tf.to_lowercase()) {
            violations.push(ProfileViolation::new(
                "TAG_FILE_MISSING",
                format!("Required tag file '{tf}' not present"),
            ));
        }
    }

    // 5. fetch.txt policy
    if !profile.allow_fetch_txt && bag.has_fetch_txt {
        violations.push(ProfileViolation::new(
            "FETCH_TXT_FORBIDDEN",
            "Profile forbids fetch.txt but it is present in the bag",
        ));
    }

    // 6. Bag-Info tag constraints
    for (field_name, constraint) in &profile.bag_info {
        let key = field_name.to_lowercase();
        let present_values = bag.bag_info_fields.get(&key);

        if constraint.required && present_values.is_none() {
            violations.push(ProfileViolation::new(
                "BAG_INFO_FIELD_MISSING",
                format!("Required Bag-Info field '{field_name}' is absent"),
            ));
            continue;
        }

        if constraint.recommended && present_values.is_none() {
            warnings.push(format!(
                "Recommended Bag-Info field '{field_name}' is absent"
            ));
        }

        if !constraint.values.is_empty() {
            if let Some(actual_values) = present_values {
                let allowed: HashSet<&str> = constraint.values.iter().map(String::as_str).collect();
                for v in actual_values {
                    if !allowed.contains(v.as_str()) {
                        violations.push(ProfileViolation::new(
                            "BAG_INFO_VALUE_INVALID",
                            format!(
                                "Bag-Info field '{field_name}' has value '{v}' not in allowed list {:?}",
                                constraint.values
                            ),
                        ));
                    }
                }
            }
        }
    }

    // 7. BagIt-Version-Required
    if !profile.bagit_version_required.is_empty() {
        let allowed_versions: HashSet<&str> = profile
            .bagit_version_required
            .iter()
            .map(String::as_str)
            .collect();
        if !allowed_versions.contains(bag.bagit_version.as_str()) {
            violations.push(ProfileViolation::new(
                "BAGIT_VERSION_INVALID",
                format!(
                    "BagIt version '{}' not in required list {:?}",
                    bag.bagit_version, profile.bagit_version_required
                ),
            ));
        }
    }

    let conforms = violations.is_empty();
    ProfileValidationResult {
        conforms,
        violations,
        warnings,
    }
}

/// Parse a BagIt profile from a JSON string.
pub fn parse_profile(json: &str) -> ArchiveResult<BagItProfile> {
    serde_json::from_str(json)
        .map_err(|e| ArchiveError::Validation(format!("Invalid BagIt profile JSON: {e}")))
}

// ---------------------------------------------------------------------------
// Helpers for building test bags on disk
// ---------------------------------------------------------------------------

/// Write a minimal valid bag to `dir` for testing.
pub fn write_minimal_bag(
    dir: &Path,
    manifest_algs: &[&str],
    tag_manifest_algs: &[&str],
    extra_tag_files: &[&str],
    has_fetch_txt: bool,
    bag_info_fields: &[(&str, &str)],
) -> ArchiveResult<()> {
    // bagit.txt
    std::fs::write(
        dir.join("bagit.txt"),
        "BagIt-Version: 1.0\nTag-File-Character-Encoding: UTF-8\n",
    )?;

    // data directory + placeholder
    let data_dir = dir.join("data");
    std::fs::create_dir_all(&data_dir)?;
    std::fs::write(data_dir.join("placeholder.txt"), b"payload")?;

    // manifests
    for alg in manifest_algs {
        std::fs::write(
            dir.join(format!("manifest-{alg}.txt")),
            format!("deadbeef  data/placeholder.txt\n"),
        )?;
    }

    // tag manifests
    for alg in tag_manifest_algs {
        std::fs::write(
            dir.join(format!("tagmanifest-{alg}.txt")),
            format!("deadbeef  bagit.txt\n"),
        )?;
    }

    // extra tag files
    for tf in extra_tag_files {
        std::fs::write(dir.join(tf), b"")?;
    }

    // fetch.txt
    if has_fetch_txt {
        std::fs::write(dir.join("fetch.txt"), b"")?;
    }

    // bag-info.txt
    if !bag_info_fields.is_empty() {
        let content = bag_info_fields
            .iter()
            .map(|(k, v)| format!("{k}: {v}\n"))
            .collect::<String>();
        std::fs::write(dir.join("bag-info.txt"), content.as_bytes())?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("oximedia_bagit_{name}"));
        std::fs::create_dir_all(&d).expect("mkdir");
        d
    }

    fn empty_profile() -> BagItProfile {
        BagItProfile {
            profile_info: ProfileInfo {
                identifier: "https://example.com/profile".into(),
                source_organization: "Test Org".into(),
                external_description: "".into(),
                version: "1.0".into(),
                contact_name: "".into(),
                contact_email: "".into(),
            },
            bag_info: HashMap::new(),
            manifests_required: vec![],
            manifests_allowed: vec![],
            tag_manifests_required: vec![],
            tag_manifests_allowed: vec![],
            tag_files_required: vec![],
            tag_files_allowed: vec![],
            allow_fetch_txt: true,
            serialization: SerializationPolicy::Optional,
            accept_serialization: vec![],
            bagit_version_required: vec![],
        }
    }

    // ── BagDescriptor::from_dir ───────────────────────────────────────────────

    #[test]
    fn test_bag_descriptor_parses_minimal_bag() {
        let dir = tmp_dir("minimal");
        write_minimal_bag(&dir, &["sha256"], &["sha256"], &[], false, &[]).expect("write bag");

        let desc = BagDescriptor::from_dir(&dir).expect("parse bag");
        assert_eq!(desc.bagit_version, "1.0");
        assert_eq!(desc.tag_encoding, "UTF-8");
        assert!(desc.manifest_algorithms.contains("sha256"));
        assert!(desc.tag_manifest_algorithms.contains("sha256"));
        assert!(!desc.has_fetch_txt);
    }

    #[test]
    fn test_bag_descriptor_detects_fetch_txt() {
        let dir = tmp_dir("fetch");
        write_minimal_bag(&dir, &["sha256"], &[], &[], true, &[]).expect("write bag");
        let desc = BagDescriptor::from_dir(&dir).expect("parse");
        assert!(desc.has_fetch_txt);
    }

    #[test]
    fn test_bag_descriptor_parses_bag_info() {
        let dir = tmp_dir("baginfo");
        write_minimal_bag(
            &dir,
            &["sha256"],
            &[],
            &[],
            false,
            &[
                ("Source-Organization", "ACME Corp"),
                ("Contact-Email", "admin@acme.com"),
            ],
        )
        .expect("write");
        let desc = BagDescriptor::from_dir(&dir).expect("parse");
        assert_eq!(
            desc.bag_info_fields
                .get("source-organization")
                .and_then(|v| v.first())
                .map(String::as_str),
            Some("ACME Corp")
        );
    }

    #[test]
    fn test_bag_descriptor_missing_bagit_txt_error() {
        let dir = tmp_dir("nobagit");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let err = BagDescriptor::from_dir(&dir);
        assert!(err.is_err());
    }

    // ── validate_bag_against_profile ─────────────────────────────────────────

    #[test]
    fn test_valid_bag_conforms_to_empty_profile() {
        let dir = tmp_dir("conform_empty");
        write_minimal_bag(&dir, &["sha256"], &[], &[], false, &[]).expect("write");
        let bag = BagDescriptor::from_dir(&dir).expect("parse");
        let profile = empty_profile();
        let result = validate_bag_against_profile(&bag, &profile);
        assert!(result.conforms);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_required_manifest_missing_violation() {
        let dir = tmp_dir("missing_manifest");
        write_minimal_bag(&dir, &["sha256"], &[], &[], false, &[]).expect("write");
        let bag = BagDescriptor::from_dir(&dir).expect("parse");
        let mut profile = empty_profile();
        profile.manifests_required = vec!["blake3".to_string()];
        let result = validate_bag_against_profile(&bag, &profile);
        assert!(!result.conforms);
        assert!(result
            .violations
            .iter()
            .any(|v| v.code == "MANIFEST_MISSING"));
    }

    #[test]
    fn test_disallowed_manifest_violation() {
        let dir = tmp_dir("disallowed");
        write_minimal_bag(&dir, &["md5", "sha256"], &[], &[], false, &[]).expect("write");
        let bag = BagDescriptor::from_dir(&dir).expect("parse");
        let mut profile = empty_profile();
        // Only sha256 allowed; md5 present → violation
        profile.manifests_allowed = vec!["sha256".to_string()];
        let result = validate_bag_against_profile(&bag, &profile);
        assert!(!result.conforms);
        assert!(result
            .violations
            .iter()
            .any(|v| v.code == "MANIFEST_DISALLOWED"));
    }

    #[test]
    fn test_fetch_txt_forbidden_violation() {
        let dir = tmp_dir("fetch_forbidden");
        write_minimal_bag(&dir, &["sha256"], &[], &[], true, &[]).expect("write");
        let bag = BagDescriptor::from_dir(&dir).expect("parse");
        let mut profile = empty_profile();
        profile.allow_fetch_txt = false;
        let result = validate_bag_against_profile(&bag, &profile);
        assert!(!result.conforms);
        assert!(result
            .violations
            .iter()
            .any(|v| v.code == "FETCH_TXT_FORBIDDEN"));
    }

    #[test]
    fn test_required_tag_file_missing() {
        let dir = tmp_dir("no_dcat");
        write_minimal_bag(&dir, &["sha256"], &[], &[], false, &[]).expect("write");
        let bag = BagDescriptor::from_dir(&dir).expect("parse");
        let mut profile = empty_profile();
        profile.tag_files_required = vec!["metadata/dcat.xml".to_string()];
        let result = validate_bag_against_profile(&bag, &profile);
        assert!(!result.conforms);
        assert!(result
            .violations
            .iter()
            .any(|v| v.code == "TAG_FILE_MISSING"));
    }

    #[test]
    fn test_required_bag_info_field_missing() {
        let dir = tmp_dir("no_contact");
        write_minimal_bag(&dir, &["sha256"], &[], &[], false, &[]).expect("write");
        let bag = BagDescriptor::from_dir(&dir).expect("parse");
        let mut profile = empty_profile();
        profile.bag_info.insert(
            "Contact-Email".to_string(),
            TagConstraint {
                required: true,
                values: vec![],
                description: "".into(),
                repeatable: true,
                recommended: false,
            },
        );
        let result = validate_bag_against_profile(&bag, &profile);
        assert!(!result.conforms);
        assert!(result
            .violations
            .iter()
            .any(|v| v.code == "BAG_INFO_FIELD_MISSING"));
    }

    #[test]
    fn test_bagit_version_required_violation() {
        let dir = tmp_dir("bad_version");
        write_minimal_bag(&dir, &["sha256"], &[], &[], false, &[]).expect("write");
        let bag = BagDescriptor::from_dir(&dir).expect("parse");
        let mut profile = empty_profile();
        profile.bagit_version_required = vec!["0.97".to_string()];
        let result = validate_bag_against_profile(&bag, &profile);
        assert!(!result.conforms);
        assert!(result
            .violations
            .iter()
            .any(|v| v.code == "BAGIT_VERSION_INVALID"));
    }

    // ── parse_profile ─────────────────────────────────────────────────────────

    #[test]
    fn test_parse_profile_from_json() {
        let json = r#"{
            "BagIt-Profile-Info": {
                "BagIt-Profile-Identifier": "https://example.com/profile/1.0",
                "Source-Organization": "Example Org",
                "Version": "1.0"
            },
            "Manifests-Required": ["sha256"],
            "Allow-Fetch.txt": false
        }"#;
        let profile = parse_profile(json).expect("parse");
        assert_eq!(profile.manifests_required, vec!["sha256"]);
        assert!(!profile.allow_fetch_txt);
        assert_eq!(profile.profile_info.source_organization, "Example Org");
    }

    #[test]
    fn test_parse_profile_invalid_json_error() {
        let err = parse_profile("{not valid json}");
        assert!(err.is_err());
    }
}
