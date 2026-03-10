//! Plugin manifest file parsing and validation.
//!
//! Each plugin can ship with a `plugin.json` manifest that describes
//! its metadata and capabilities. This allows the host to discover
//! plugins without loading their shared libraries first.

use crate::error::{PluginError, PluginResult};
use crate::traits::PLUGIN_API_VERSION;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ── Semantic Versioning ─────────────────────────────────────────────────

/// A parsed semantic version (major.minor.patch with optional pre-release).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemVer {
    /// Major version number.
    pub major: u64,
    /// Minor version number.
    pub minor: u64,
    /// Patch version number.
    pub patch: u64,
    /// Optional pre-release identifier (e.g. "alpha.1").
    pub pre: Option<String>,
}

impl SemVer {
    /// Parse a semver string such as `"1.2.3"` or `"1.0.0-beta.2"`.
    ///
    /// # Errors
    ///
    /// Returns a descriptive string on parse failure.
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        let (version_part, pre) = if let Some((v, p)) = s.split_once('-') {
            (v, Some(p.to_string()))
        } else {
            (s, None)
        };

        let parts: Vec<&str> = version_part.split('.').collect();
        if parts.len() < 2 || parts.len() > 3 {
            return Err(format!("Expected 2-3 dot-separated numbers, got '{s}'"));
        }

        let major = parts[0]
            .parse::<u64>()
            .map_err(|e| format!("Invalid major: {e}"))?;
        let minor = parts[1]
            .parse::<u64>()
            .map_err(|e| format!("Invalid minor: {e}"))?;
        let patch = if parts.len() == 3 {
            parts[2]
                .parse::<u64>()
                .map_err(|e| format!("Invalid patch: {e}"))?
        } else {
            0
        };

        Ok(Self {
            major,
            minor,
            patch,
            pre,
        })
    }

    /// Compare two versions, ignoring pre-release (numeric only).
    fn cmp_numeric(&self, other: &Self) -> std::cmp::Ordering {
        self.major
            .cmp(&other.major)
            .then(self.minor.cmp(&other.minor))
            .then(self.patch.cmp(&other.patch))
    }

    /// Check if this version is compatible with `other` under caret semantics.
    ///
    /// Caret compatibility (`^`): versions are compatible if the left-most
    /// non-zero component is the same.
    ///
    /// - `^1.2.3` matches `>=1.2.3, <2.0.0`
    /// - `^0.2.3` matches `>=0.2.3, <0.3.0`
    /// - `^0.0.3` matches `>=0.0.3, <0.0.4`
    pub fn is_caret_compatible(&self, req: &Self) -> bool {
        if self.cmp_numeric(req) == std::cmp::Ordering::Less {
            return false;
        }
        if req.major > 0 {
            self.major == req.major
        } else if req.minor > 0 {
            self.major == 0 && self.minor == req.minor
        } else {
            self.major == 0 && self.minor == 0 && self.patch == req.patch
        }
    }

    /// Check if this version is compatible under tilde semantics.
    ///
    /// - `~1.2.3` matches `>=1.2.3, <1.3.0`
    /// - `~1.2`   matches `>=1.2.0, <1.3.0`
    pub fn is_tilde_compatible(&self, req: &Self) -> bool {
        if self.cmp_numeric(req) == std::cmp::Ordering::Less {
            return false;
        }
        self.major == req.major && self.minor == req.minor
    }
}

impl std::fmt::Display for SemVer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(ref pre) = self.pre {
            write!(f, "-{pre}")?;
        }
        Ok(())
    }
}

/// Operator used in a semver requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemVerOp {
    /// Exact match `=` / bare version.
    Exact,
    /// Greater than `>`.
    Gt,
    /// Greater or equal `>=`.
    Gte,
    /// Less than `<`.
    Lt,
    /// Less or equal `<=`.
    Lte,
    /// Caret `^` (compatible updates).
    Caret,
    /// Tilde `~` (patch-level updates).
    Tilde,
    /// Wildcard `*` (any version).
    Wildcard,
}

/// A semver requirement such as `">=1.0.0"` or `"^2.3"`.
#[derive(Debug, Clone)]
pub struct SemVerReq {
    /// The comparison operator.
    pub op: SemVerOp,
    /// The version to compare against (ignored for `Wildcard`).
    pub version: SemVer,
}

impl SemVerReq {
    /// Parse a requirement string.
    ///
    /// Accepted forms:
    /// - `"*"` — any version
    /// - `"1.2.3"` — exact match
    /// - `">=1.0.0"`, `">1.0.0"`, `"<=1.0.0"`, `"<1.0.0"` — comparison
    /// - `"^1.0.0"` — caret
    /// - `"~1.0.0"` — tilde
    /// - `"=1.0.0"` — explicit exact
    ///
    /// # Errors
    ///
    /// Returns a descriptive string on parse failure.
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        if s == "*" {
            return Ok(Self {
                op: SemVerOp::Wildcard,
                version: SemVer {
                    major: 0,
                    minor: 0,
                    patch: 0,
                    pre: None,
                },
            });
        }

        let (op, rest) = if let Some(r) = s.strip_prefix(">=") {
            (SemVerOp::Gte, r)
        } else if let Some(r) = s.strip_prefix("<=") {
            (SemVerOp::Lte, r)
        } else if let Some(r) = s.strip_prefix('>') {
            (SemVerOp::Gt, r)
        } else if let Some(r) = s.strip_prefix('<') {
            (SemVerOp::Lt, r)
        } else if let Some(r) = s.strip_prefix('^') {
            (SemVerOp::Caret, r)
        } else if let Some(r) = s.strip_prefix('~') {
            (SemVerOp::Tilde, r)
        } else if let Some(r) = s.strip_prefix('=') {
            (SemVerOp::Exact, r)
        } else {
            (SemVerOp::Exact, s)
        };

        let version = SemVer::parse(rest.trim())?;
        Ok(Self { op, version })
    }

    /// Check whether a given version satisfies this requirement.
    pub fn matches(&self, version: &SemVer) -> bool {
        match self.op {
            SemVerOp::Wildcard => true,
            SemVerOp::Exact => version.cmp_numeric(&self.version) == std::cmp::Ordering::Equal,
            SemVerOp::Gt => version.cmp_numeric(&self.version) == std::cmp::Ordering::Greater,
            SemVerOp::Gte => version.cmp_numeric(&self.version) != std::cmp::Ordering::Less,
            SemVerOp::Lt => version.cmp_numeric(&self.version) == std::cmp::Ordering::Less,
            SemVerOp::Lte => version.cmp_numeric(&self.version) != std::cmp::Ordering::Greater,
            SemVerOp::Caret => version.is_caret_compatible(&self.version),
            SemVerOp::Tilde => version.is_tilde_compatible(&self.version),
        }
    }
}

impl std::fmt::Display for SemVerReq {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.op {
            SemVerOp::Wildcard => write!(f, "*"),
            SemVerOp::Exact => write!(f, "={}", self.version),
            SemVerOp::Gt => write!(f, ">{}", self.version),
            SemVerOp::Gte => write!(f, ">={}", self.version),
            SemVerOp::Lt => write!(f, "<{}", self.version),
            SemVerOp::Lte => write!(f, "<={}", self.version),
            SemVerOp::Caret => write!(f, "^{}", self.version),
            SemVerOp::Tilde => write!(f, "~{}", self.version),
        }
    }
}

// ── Dependency Resolution ──────────────────────────────────────────────

/// Result of resolving plugin dependencies.
#[derive(Debug, Clone)]
pub struct DependencyResolution {
    /// Plugins in topological load order (dependencies first).
    pub load_order: Vec<String>,
    /// Any unresolvable dependencies: (plugin, missing dep, requirement).
    pub missing: Vec<(String, String, String)>,
    /// Any version conflicts: (plugin, dep, requirement, available version).
    pub conflicts: Vec<(String, String, String, String)>,
}

impl DependencyResolution {
    /// Returns `true` if all dependencies are satisfied.
    pub fn is_satisfied(&self) -> bool {
        self.missing.is_empty() && self.conflicts.is_empty()
    }
}

/// Resolve dependency order for a set of plugin manifests.
///
/// Performs a topological sort on the dependency graph and checks that
/// each dependency's version satisfies the declared requirement.
///
/// # Errors
///
/// Returns [`PluginError::InvalidManifest`] if a dependency cycle is detected.
pub fn resolve_dependencies(manifests: &[PluginManifest]) -> PluginResult<DependencyResolution> {
    // Build name → manifest index lookup.
    let mut index_by_name: HashMap<&str, usize> = HashMap::new();
    for (i, m) in manifests.iter().enumerate() {
        index_by_name.insert(&m.name, i);
    }

    let mut missing: Vec<(String, String, String)> = Vec::new();
    let mut conflicts: Vec<(String, String, String, String)> = Vec::new();

    // Validate every dependency's version requirement.
    for manifest in manifests {
        for (dep_name, req_str) in &manifest.dependencies {
            if let Some(&dep_idx) = index_by_name.get(dep_name.as_str()) {
                let dep_manifest = &manifests[dep_idx];
                let req = SemVerReq::parse(req_str).map_err(|e| {
                    PluginError::InvalidManifest(format!(
                        "Bad requirement for dep '{dep_name}' of '{}': {e}",
                        manifest.name
                    ))
                })?;
                let dep_ver = SemVer::parse(&dep_manifest.version).map_err(|e| {
                    PluginError::InvalidManifest(format!(
                        "Bad version in dep '{}': {e}",
                        dep_manifest.name
                    ))
                })?;
                if !req.matches(&dep_ver) {
                    conflicts.push((
                        manifest.name.clone(),
                        dep_name.clone(),
                        req_str.clone(),
                        dep_manifest.version.clone(),
                    ));
                }
            } else {
                missing.push((manifest.name.clone(), dep_name.clone(), req_str.clone()));
            }
        }
    }

    // Topological sort (Kahn's algorithm).
    let n = manifests.len();
    let mut in_degree = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (i, manifest) in manifests.iter().enumerate() {
        for dep_name in manifest.dependencies.keys() {
            if let Some(&dep_idx) = index_by_name.get(dep_name.as_str()) {
                adj[dep_idx].push(i);
                in_degree[i] += 1;
            }
        }
    }

    let mut queue: std::collections::VecDeque<usize> = in_degree
        .iter()
        .enumerate()
        .filter(|(_, &d)| d == 0)
        .map(|(i, _)| i)
        .collect();

    let mut load_order: Vec<String> = Vec::with_capacity(n);

    while let Some(idx) = queue.pop_front() {
        load_order.push(manifests[idx].name.clone());
        for &succ in &adj[idx] {
            in_degree[succ] = in_degree[succ].saturating_sub(1);
            if in_degree[succ] == 0 {
                queue.push_back(succ);
            }
        }
    }

    if load_order.len() < n {
        return Err(PluginError::InvalidManifest(
            "Dependency cycle detected among plugins".to_string(),
        ));
    }

    Ok(DependencyResolution {
        load_order,
        missing,
        conflicts,
    })
}

// ────────────────────────────────────────────────────────────────────────

/// Plugin manifest file (`plugin.json`).
///
/// This file sits alongside the shared library and provides metadata
/// for plugin discovery without loading the binary.
///
/// # Example JSON
///
/// ```json
/// {
///   "name": "oximedia-plugin-h264",
///   "version": "1.0.0",
///   "api_version": 1,
///   "description": "H.264 decoder/encoder plugin",
///   "author": "Example Corp",
///   "license": "proprietary",
///   "patent_encumbered": true,
///   "library": "libh264_plugin.so",
///   "codecs": [
///     {
///       "name": "h264",
///       "decode": true,
///       "encode": true,
///       "description": "H.264/AVC video codec"
///     }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    /// Plugin name (must be unique).
    pub name: String,
    /// Plugin version (semver).
    pub version: String,
    /// API version this plugin targets.
    pub api_version: u32,
    /// Human-readable description.
    pub description: String,
    /// Plugin author or organization.
    pub author: String,
    /// License identifier.
    pub license: String,
    /// Whether the plugin contains patent-encumbered codecs.
    pub patent_encumbered: bool,
    /// Shared library filename (e.g., "libh264_plugin.so").
    pub library: String,
    /// List of codecs provided by this plugin.
    pub codecs: Vec<ManifestCodec>,
    /// Dependencies on other plugins (plugin name → semver requirement).
    #[serde(default)]
    pub dependencies: HashMap<String, String>,
    /// Minimum host version required (semver requirement string, e.g. ">=0.1.0").
    #[serde(default)]
    pub min_host_version: Option<String>,
}

/// A codec entry in the plugin manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestCodec {
    /// Codec name (e.g., "h264", "aac").
    pub name: String,
    /// Whether decoding is supported.
    pub decode: bool,
    /// Whether encoding is supported.
    pub encode: bool,
    /// Human-readable description.
    pub description: String,
}

impl PluginManifest {
    /// Parse a manifest from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::InvalidManifest`] if the JSON is malformed.
    pub fn from_json(json: &str) -> PluginResult<Self> {
        serde_json::from_str(json).map_err(|e| PluginError::InvalidManifest(e.to_string()))
    }

    /// Serialize the manifest to a JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Json`] if serialization fails.
    pub fn to_json(&self) -> PluginResult<String> {
        serde_json::to_string_pretty(self).map_err(PluginError::Json)
    }

    /// Load a manifest from a file on disk.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::Io`] if the file cannot be read, or
    /// [`PluginError::InvalidManifest`] if the content is invalid.
    pub fn from_file(path: &Path) -> PluginResult<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_json(&content)
    }

    /// Validate the manifest for correctness and compatibility.
    ///
    /// Checks:
    /// - Name is non-empty
    /// - Version is non-empty
    /// - API version matches current host
    /// - Library filename is non-empty
    /// - At least one codec is declared
    /// - Each codec has a non-empty name
    /// - Each codec supports at least decode or encode
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::InvalidManifest`] with a description
    /// of the first validation failure.
    pub fn validate(&self) -> PluginResult<()> {
        if self.name.is_empty() {
            return Err(PluginError::InvalidManifest(
                "Plugin name must not be empty".to_string(),
            ));
        }

        if self.version.is_empty() {
            return Err(PluginError::InvalidManifest(
                "Plugin version must not be empty".to_string(),
            ));
        }

        if self.api_version != PLUGIN_API_VERSION {
            return Err(PluginError::ApiIncompatible(format!(
                "Manifest declares API v{}, host expects v{PLUGIN_API_VERSION}",
                self.api_version
            )));
        }

        if self.library.is_empty() {
            return Err(PluginError::InvalidManifest(
                "Library filename must not be empty".to_string(),
            ));
        }

        if self.codecs.is_empty() {
            return Err(PluginError::InvalidManifest(
                "Plugin must declare at least one codec".to_string(),
            ));
        }

        for (i, codec) in self.codecs.iter().enumerate() {
            if codec.name.is_empty() {
                return Err(PluginError::InvalidManifest(format!(
                    "Codec at index {i} has empty name"
                )));
            }

            if !codec.decode && !codec.encode {
                return Err(PluginError::InvalidManifest(format!(
                    "Codec '{}' must support at least decode or encode",
                    codec.name
                )));
            }
        }

        Ok(())
    }

    /// Validate that the plugin version is valid semver.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::InvalidManifest`] if the version string is not
    /// valid semantic versioning.
    pub fn validate_version(&self) -> PluginResult<SemVer> {
        SemVer::parse(&self.version).map_err(|e| {
            PluginError::InvalidManifest(format!(
                "Invalid plugin version '{}': {}",
                self.version, e
            ))
        })
    }

    /// Check if this manifest declares a specific codec.
    pub fn has_codec(&self, name: &str) -> bool {
        self.codecs.iter().any(|c| c.name == name)
    }

    /// Check whether this plugin's version satisfies a semver requirement string.
    ///
    /// The requirement string can be:
    /// - `"1.2.3"` — exact match
    /// - `">=1.0.0"` — greater or equal
    /// - `"^1.0.0"` — caret (compatible with 1.x.y)
    /// - `"~1.2.0"` — tilde (compatible with 1.2.x)
    ///
    /// # Errors
    ///
    /// Returns error if either the version or requirement cannot be parsed.
    pub fn satisfies_requirement(&self, requirement: &str) -> PluginResult<bool> {
        let version = self.validate_version()?;
        let req = SemVerReq::parse(requirement).map_err(|e| {
            PluginError::InvalidManifest(format!("Invalid requirement '{requirement}': {e}"))
        })?;
        Ok(req.matches(&version))
    }

    /// Validate all dependency requirement strings are parseable.
    ///
    /// # Errors
    ///
    /// Returns error on the first invalid dependency requirement string.
    pub fn validate_dependencies(&self) -> PluginResult<()> {
        for (dep_name, req_str) in &self.dependencies {
            SemVerReq::parse(req_str).map_err(|e| {
                PluginError::InvalidManifest(format!(
                    "Invalid dependency requirement for '{dep_name}': '{req_str}' — {e}"
                ))
            })?;
        }
        Ok(())
    }

    /// Get the library path relative to the manifest directory.
    ///
    /// Given the path to the manifest file, returns the expected
    /// path to the shared library.
    pub fn library_path(&self, manifest_path: &Path) -> Option<std::path::PathBuf> {
        manifest_path.parent().map(|dir| dir.join(&self.library))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> PluginManifest {
        PluginManifest {
            name: "test-plugin".to_string(),
            version: "1.0.0".to_string(),
            api_version: PLUGIN_API_VERSION,
            description: "A test plugin".to_string(),
            author: "Test Author".to_string(),
            license: "MIT".to_string(),
            patent_encumbered: false,
            library: "libtest_plugin.so".to_string(),
            codecs: vec![ManifestCodec {
                name: "test-codec".to_string(),
                decode: true,
                encode: false,
                description: "A test codec".to_string(),
            }],
            dependencies: HashMap::new(),
            min_host_version: None,
        }
    }

    #[test]
    fn test_manifest_roundtrip() {
        let manifest = sample_manifest();
        let json = manifest.to_json().expect("serialization should succeed");
        let parsed = PluginManifest::from_json(&json).expect("deserialization should succeed");
        assert_eq!(parsed.name, "test-plugin");
        assert_eq!(parsed.version, "1.0.0");
        assert_eq!(parsed.codecs.len(), 1);
        assert_eq!(parsed.codecs[0].name, "test-codec");
    }

    #[test]
    fn test_manifest_validate_success() {
        let manifest = sample_manifest();
        manifest.validate().expect("validation should succeed");
    }

    #[test]
    fn test_manifest_validate_empty_name() {
        let mut manifest = sample_manifest();
        manifest.name = String::new();
        let err = manifest.validate().expect_err("should fail");
        assert!(err.to_string().contains("name must not be empty"));
    }

    #[test]
    fn test_manifest_validate_empty_version() {
        let mut manifest = sample_manifest();
        manifest.version = String::new();
        let err = manifest.validate().expect_err("should fail");
        assert!(err.to_string().contains("version must not be empty"));
    }

    #[test]
    fn test_manifest_validate_wrong_api_version() {
        let mut manifest = sample_manifest();
        manifest.api_version = 999;
        let err = manifest.validate().expect_err("should fail");
        assert!(err.to_string().contains("API"));
    }

    #[test]
    fn test_manifest_validate_empty_library() {
        let mut manifest = sample_manifest();
        manifest.library = String::new();
        let err = manifest.validate().expect_err("should fail");
        assert!(err.to_string().contains("Library filename"));
    }

    #[test]
    fn test_manifest_validate_no_codecs() {
        let mut manifest = sample_manifest();
        manifest.codecs.clear();
        let err = manifest.validate().expect_err("should fail");
        assert!(err.to_string().contains("at least one codec"));
    }

    #[test]
    fn test_manifest_validate_codec_empty_name() {
        let mut manifest = sample_manifest();
        manifest.codecs[0].name = String::new();
        let err = manifest.validate().expect_err("should fail");
        assert!(err.to_string().contains("empty name"));
    }

    #[test]
    fn test_manifest_validate_codec_no_capability() {
        let mut manifest = sample_manifest();
        manifest.codecs[0].decode = false;
        manifest.codecs[0].encode = false;
        let err = manifest.validate().expect_err("should fail");
        assert!(err.to_string().contains("at least decode or encode"));
    }

    #[test]
    fn test_manifest_has_codec() {
        let manifest = sample_manifest();
        assert!(manifest.has_codec("test-codec"));
        assert!(!manifest.has_codec("nonexistent"));
    }

    #[test]
    fn test_manifest_library_path() {
        let manifest = sample_manifest();
        let manifest_path = Path::new("/usr/lib/oximedia/plugins/test/plugin.json");
        let lib_path = manifest.library_path(manifest_path);
        assert_eq!(
            lib_path,
            Some(std::path::PathBuf::from(
                "/usr/lib/oximedia/plugins/test/libtest_plugin.so"
            ))
        );
    }

    #[test]
    fn test_manifest_from_invalid_json() {
        let result = PluginManifest::from_json("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_manifest_from_file_not_found() {
        let result = PluginManifest::from_file(Path::new("/nonexistent/plugin.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_manifest_from_file_roundtrip() {
        let manifest = sample_manifest();
        let json = manifest.to_json().expect("serialization should succeed");

        let dir = std::env::temp_dir().join("oximedia-plugin-test-manifest");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        let path = dir.join("plugin.json");
        std::fs::write(&path, &json).expect("write should succeed");

        let loaded = PluginManifest::from_file(&path).expect("load should succeed");
        assert_eq!(loaded.name, "test-plugin");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── SemVer tests ──

    #[test]
    fn test_semver_parse_full() {
        let v = SemVer::parse("1.2.3").expect("parse should succeed");
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
        assert!(v.pre.is_none());
    }

    #[test]
    fn test_semver_parse_with_pre() {
        let v = SemVer::parse("0.1.0-alpha.1").expect("parse should succeed");
        assert_eq!(v.major, 0);
        assert_eq!(v.minor, 1);
        assert_eq!(v.patch, 0);
        assert_eq!(v.pre, Some("alpha.1".to_string()));
    }

    #[test]
    fn test_semver_parse_two_parts() {
        let v = SemVer::parse("2.5").expect("parse should succeed");
        assert_eq!(v.major, 2);
        assert_eq!(v.minor, 5);
        assert_eq!(v.patch, 0);
    }

    #[test]
    fn test_semver_parse_invalid() {
        assert!(SemVer::parse("abc").is_err());
        assert!(SemVer::parse("1.2.3.4").is_err());
    }

    #[test]
    fn test_semver_display() {
        let v = SemVer::parse("1.2.3").expect("parse should succeed");
        assert_eq!(v.to_string(), "1.2.3");

        let v2 = SemVer::parse("0.1.0-beta").expect("parse should succeed");
        assert_eq!(v2.to_string(), "0.1.0-beta");
    }

    #[test]
    fn test_semver_caret_compatibility() {
        let v = SemVer::parse("1.5.2").expect("parse should succeed");
        let req = SemVer::parse("1.0.0").expect("parse should succeed");
        assert!(v.is_caret_compatible(&req));

        let req2 = SemVer::parse("2.0.0").expect("parse should succeed");
        assert!(!v.is_caret_compatible(&req2));

        // ^0.2.0 → [0.2.0, 0.3.0)
        let v03 = SemVer::parse("0.2.5").expect("parse should succeed");
        let req03 = SemVer::parse("0.2.0").expect("parse should succeed");
        assert!(v03.is_caret_compatible(&req03));

        let v04 = SemVer::parse("0.3.0").expect("parse should succeed");
        assert!(!v04.is_caret_compatible(&req03));
    }

    #[test]
    fn test_semver_tilde_compatibility() {
        let v = SemVer::parse("1.2.5").expect("parse should succeed");
        let req = SemVer::parse("1.2.0").expect("parse should succeed");
        assert!(v.is_tilde_compatible(&req));

        let v_bad = SemVer::parse("1.3.0").expect("parse should succeed");
        assert!(!v_bad.is_tilde_compatible(&req));
    }

    // ── SemVerReq tests ──

    #[test]
    fn test_semver_req_exact() {
        let req = SemVerReq::parse("1.0.0").expect("parse should succeed");
        assert!(req.matches(&SemVer::parse("1.0.0").expect("parse")));
        assert!(!req.matches(&SemVer::parse("1.0.1").expect("parse")));
    }

    #[test]
    fn test_semver_req_gte() {
        let req = SemVerReq::parse(">=1.0.0").expect("parse should succeed");
        assert!(req.matches(&SemVer::parse("1.0.0").expect("parse")));
        assert!(req.matches(&SemVer::parse("2.0.0").expect("parse")));
        assert!(!req.matches(&SemVer::parse("0.9.9").expect("parse")));
    }

    #[test]
    fn test_semver_req_caret() {
        let req = SemVerReq::parse("^1.2.0").expect("parse should succeed");
        assert!(req.matches(&SemVer::parse("1.5.0").expect("parse")));
        assert!(!req.matches(&SemVer::parse("2.0.0").expect("parse")));
        assert!(!req.matches(&SemVer::parse("1.1.0").expect("parse")));
    }

    #[test]
    fn test_semver_req_tilde() {
        let req = SemVerReq::parse("~1.2.0").expect("parse should succeed");
        assert!(req.matches(&SemVer::parse("1.2.5").expect("parse")));
        assert!(!req.matches(&SemVer::parse("1.3.0").expect("parse")));
    }

    #[test]
    fn test_semver_req_wildcard() {
        let req = SemVerReq::parse("*").expect("parse should succeed");
        assert!(req.matches(&SemVer::parse("0.0.1").expect("parse")));
        assert!(req.matches(&SemVer::parse("99.99.99").expect("parse")));
    }

    #[test]
    fn test_semver_req_lt() {
        let req = SemVerReq::parse("<2.0.0").expect("parse should succeed");
        assert!(req.matches(&SemVer::parse("1.9.9").expect("parse")));
        assert!(!req.matches(&SemVer::parse("2.0.0").expect("parse")));
    }

    // ── Manifest version validation tests ──

    #[test]
    fn test_manifest_validate_version() {
        let m = sample_manifest();
        let v = m.validate_version().expect("should parse");
        assert_eq!(v.major, 1);
    }

    #[test]
    fn test_manifest_satisfies_requirement() {
        let m = sample_manifest(); // version = "1.0.0"
        assert!(m.satisfies_requirement(">=0.5.0").expect("should parse"));
        assert!(m.satisfies_requirement("^1.0.0").expect("should parse"));
        assert!(!m.satisfies_requirement(">=2.0.0").expect("should parse"));
    }

    #[test]
    fn test_manifest_validate_dependencies_ok() {
        let mut m = sample_manifest();
        m.dependencies
            .insert("other-plugin".to_string(), "^1.0.0".to_string());
        assert!(m.validate_dependencies().is_ok());
    }

    #[test]
    fn test_manifest_validate_dependencies_bad_req() {
        let mut m = sample_manifest();
        m.dependencies
            .insert("other".to_string(), "not-a-version!!!".to_string());
        assert!(m.validate_dependencies().is_err());
    }

    // ── Dependency resolution tests ──

    #[test]
    fn test_resolve_dependencies_no_deps() {
        let m = sample_manifest();
        let res = resolve_dependencies(&[m]).expect("should resolve");
        assert!(res.is_satisfied());
        assert_eq!(res.load_order.len(), 1);
    }

    #[test]
    fn test_resolve_dependencies_linear_chain() {
        let mut a = sample_manifest();
        a.name = "plugin-a".to_string();
        a.version = "1.0.0".to_string();
        a.dependencies.clear();

        let mut b = sample_manifest();
        b.name = "plugin-b".to_string();
        b.version = "2.0.0".to_string();
        b.dependencies
            .insert("plugin-a".to_string(), "^1.0.0".to_string());

        let res = resolve_dependencies(&[a, b]).expect("should resolve");
        assert!(res.is_satisfied());
        // plugin-a should come before plugin-b
        let pos_a = res
            .load_order
            .iter()
            .position(|n| n == "plugin-a")
            .expect("should exist");
        let pos_b = res
            .load_order
            .iter()
            .position(|n| n == "plugin-b")
            .expect("should exist");
        assert!(pos_a < pos_b);
    }

    #[test]
    fn test_resolve_dependencies_missing() {
        let mut m = sample_manifest();
        m.dependencies
            .insert("nonexistent".to_string(), ">=1.0.0".to_string());

        let res = resolve_dependencies(&[m]).expect("should resolve");
        assert!(!res.is_satisfied());
        assert_eq!(res.missing.len(), 1);
        assert_eq!(res.missing[0].1, "nonexistent");
    }

    #[test]
    fn test_resolve_dependencies_version_conflict() {
        let mut a = sample_manifest();
        a.name = "provider".to_string();
        a.version = "1.0.0".to_string();

        let mut b = sample_manifest();
        b.name = "consumer".to_string();
        b.dependencies
            .insert("provider".to_string(), ">=2.0.0".to_string());

        let res = resolve_dependencies(&[a, b]).expect("should resolve");
        assert!(!res.is_satisfied());
        assert_eq!(res.conflicts.len(), 1);
    }

    #[test]
    fn test_resolve_dependencies_cycle() {
        let mut a = sample_manifest();
        a.name = "cycle-a".to_string();
        a.dependencies
            .insert("cycle-b".to_string(), "*".to_string());

        let mut b = sample_manifest();
        b.name = "cycle-b".to_string();
        b.dependencies
            .insert("cycle-a".to_string(), "*".to_string());

        let res = resolve_dependencies(&[a, b]);
        assert!(res.is_err());
        assert!(res
            .expect_err("should be an error")
            .to_string()
            .contains("cycle"));
    }
}
