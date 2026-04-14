#![allow(dead_code)]
//! Extended farm configuration and profile management.
//!
//! Provides a layered configuration model on top of
//! [`crate::CoordinatorConfig`] / [`crate::WorkerConfig`] with support for
//! named encoding profiles, resource quotas, and environment-specific
//! overrides. Supports YAML and TOML serialisation/deserialisation.

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when loading or saving farm configuration.
#[derive(Debug, thiserror::Error)]
pub enum FarmConfigError {
    /// I/O error reading or writing a file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// YAML parse or serialization error.
    #[error("YAML error: {0}")]
    Yaml(String),
    /// TOML parse error.
    #[error("TOML parse error: {0}")]
    TomlDe(String),
    /// TOML serialization error.
    #[error("TOML serialize error: {0}")]
    TomlSer(String),
}

// ---------------------------------------------------------------------------
// Custom serde helper for Duration (stored as seconds)
// ---------------------------------------------------------------------------

mod duration_secs {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(dur: &Duration, ser: S) -> Result<S::Ok, S::Error> {
        dur.as_secs().serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<Duration, D::Error> {
        let secs = u64::deserialize(de)?;
        Ok(Duration::from_secs(secs))
    }
}

// ---------------------------------------------------------------------------
// Encoding profile
// ---------------------------------------------------------------------------

/// A named encoding profile describing target codec parameters.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EncodingProfile {
    /// Human-readable profile name (e.g. "broadcast-hd").
    pub name: String,
    /// Target codec (e.g. "h264", "hevc", "av1").
    pub codec: String,
    /// Container format (e.g. "mp4", "mkv").
    pub container: String,
    /// Target bitrate in kbps (0 = CRF/CQ mode).
    pub bitrate_kbps: u32,
    /// CRF / CQ value (0 = bitrate mode).
    pub crf: u8,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame rate numerator.
    pub fps_num: u32,
    /// Frame rate denominator.
    pub fps_den: u32,
    /// Number of encoding passes.
    pub passes: u8,
    /// Arbitrary extra key-value parameters.
    pub extra: HashMap<String, String>,
}

impl Default for EncodingProfile {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            codec: "h264".to_string(),
            container: "mp4".to_string(),
            bitrate_kbps: 5000,
            crf: 0,
            width: 1920,
            height: 1080,
            fps_num: 30,
            fps_den: 1,
            passes: 1,
            extra: HashMap::new(),
        }
    }
}

impl EncodingProfile {
    /// Create a minimal profile with the given name and codec.
    #[must_use]
    pub fn new(name: impl Into<String>, codec: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            codec: codec.into(),
            ..Default::default()
        }
    }

    /// Compute the frame rate as a floating-point value.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn fps(&self) -> f64 {
        if self.fps_den == 0 {
            return 0.0;
        }
        f64::from(self.fps_num) / f64::from(self.fps_den)
    }

    /// Check whether this profile uses CRF / constant-quality mode.
    #[must_use]
    pub fn is_crf_mode(&self) -> bool {
        self.crf > 0 && self.bitrate_kbps == 0
    }
}

// ---------------------------------------------------------------------------
// Resource quota
// ---------------------------------------------------------------------------

/// Resource quota that limits what a single job may consume.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ResourceQuota {
    /// Maximum CPU cores a single job may use.
    pub max_cpu_cores: u32,
    /// Maximum memory in MiB.
    pub max_memory_mib: u64,
    /// Maximum GPU devices.
    pub max_gpus: u32,
    /// Maximum wall-clock duration in seconds.
    #[serde(with = "duration_secs")]
    pub max_wall_time: Duration,
    /// Maximum disk scratch space in MiB.
    pub max_scratch_mib: u64,
}

impl Default for ResourceQuota {
    fn default() -> Self {
        Self {
            max_cpu_cores: 4,
            max_memory_mib: 8192,
            max_gpus: 1,
            max_wall_time: Duration::from_secs(3600),
            max_scratch_mib: 10240,
        }
    }
}

impl ResourceQuota {
    /// Create a quota with custom CPU and memory limits.
    #[must_use]
    pub fn new(cpu: u32, mem_mib: u64) -> Self {
        Self {
            max_cpu_cores: cpu,
            max_memory_mib: mem_mib,
            ..Default::default()
        }
    }

    /// Check whether this quota allows the requested resources.
    #[must_use]
    pub fn allows(&self, cpu: u32, mem_mib: u64, gpus: u32) -> bool {
        cpu <= self.max_cpu_cores && mem_mib <= self.max_memory_mib && gpus <= self.max_gpus
    }
}

// ---------------------------------------------------------------------------
// Farm-wide configuration
// ---------------------------------------------------------------------------

/// Farm-wide configuration aggregating profiles, quotas, and policies.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FarmConfig {
    /// Named encoding profiles.
    pub profiles: HashMap<String, EncodingProfile>,
    /// Named resource quotas.
    pub quotas: HashMap<String, ResourceQuota>,
    /// Default profile name to use when none is specified.
    pub default_profile: String,
    /// Default quota name.
    pub default_quota: String,
    /// Whether to allow jobs to exceed quotas in emergency mode.
    pub emergency_override: bool,
    /// Maximum number of retries for any task in the farm.
    pub global_max_retries: u32,
    /// Grace period after a worker misses a heartbeat before it is
    /// considered dead (stored as seconds).
    #[serde(with = "duration_secs")]
    pub heartbeat_grace: Duration,
}

impl Default for FarmConfig {
    fn default() -> Self {
        let mut profiles = HashMap::new();
        profiles.insert("default".to_string(), EncodingProfile::default());
        let mut quotas = HashMap::new();
        quotas.insert("default".to_string(), ResourceQuota::default());
        Self {
            profiles,
            quotas,
            default_profile: "default".to_string(),
            default_quota: "default".to_string(),
            emergency_override: false,
            global_max_retries: 3,
            heartbeat_grace: Duration::from_secs(90),
        }
    }
}

impl FarmConfig {
    /// Create a new configuration with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // ── I/O helpers ──────────────────────────────────────────────────────────

    /// Load a `FarmConfig` from a YAML file.
    ///
    /// # Errors
    ///
    /// Returns `FarmConfigError::Io` if the file cannot be read, or
    /// `FarmConfigError::Yaml` if the content is not valid YAML.
    pub fn from_yaml_file(path: &Path) -> Result<Self, FarmConfigError> {
        let s = std::fs::read_to_string(path)?;
        Self::from_str_yaml(&s)
    }

    /// Load a `FarmConfig` from a TOML file.
    ///
    /// # Errors
    ///
    /// Returns `FarmConfigError::Io` if the file cannot be read, or
    /// `FarmConfigError::TomlDe` if the content is not valid TOML.
    pub fn from_toml_file(path: &Path) -> Result<Self, FarmConfigError> {
        let s = std::fs::read_to_string(path)?;
        Self::from_str_toml(&s)
    }

    /// Parse a `FarmConfig` from a YAML string.
    ///
    /// # Errors
    ///
    /// Returns `FarmConfigError::Yaml` if the string is not valid YAML or
    /// cannot be deserialised into `FarmConfig`.
    pub fn from_str_yaml(s: &str) -> Result<Self, FarmConfigError> {
        serde_yaml_ng::from_str(s).map_err(|e| FarmConfigError::Yaml(e.to_string()))
    }

    /// Parse a `FarmConfig` from a TOML string.
    ///
    /// # Errors
    ///
    /// Returns `FarmConfigError::TomlDe` if the string is not valid TOML or
    /// cannot be deserialised into `FarmConfig`.
    pub fn from_str_toml(s: &str) -> Result<Self, FarmConfigError> {
        toml::from_str(s).map_err(|e| FarmConfigError::TomlDe(e.to_string()))
    }

    /// Serialise this `FarmConfig` to a YAML string.
    ///
    /// # Errors
    ///
    /// Returns `FarmConfigError::Yaml` if serialisation fails.
    pub fn to_yaml(&self) -> Result<String, FarmConfigError> {
        serde_yaml_ng::to_string(self).map_err(|e| FarmConfigError::Yaml(e.to_string()))
    }

    /// Serialise this `FarmConfig` to a TOML string.
    ///
    /// # Errors
    ///
    /// Returns `FarmConfigError::TomlSer` if serialisation fails.
    pub fn to_toml(&self) -> Result<String, FarmConfigError> {
        toml::to_string(self).map_err(|e| FarmConfigError::TomlSer(e.to_string()))
    }

    // ── Profile / quota management ────────────────────────────────────────────

    /// Add or replace an encoding profile.
    pub fn add_profile(&mut self, profile: EncodingProfile) {
        self.profiles.insert(profile.name.clone(), profile);
    }

    /// Add or replace a resource quota.
    pub fn add_quota(&mut self, name: impl Into<String>, quota: ResourceQuota) {
        self.quotas.insert(name.into(), quota);
    }

    /// Look up an encoding profile by name.
    #[must_use]
    pub fn get_profile(&self, name: &str) -> Option<&EncodingProfile> {
        self.profiles.get(name)
    }

    /// Look up a resource quota by name.
    #[must_use]
    pub fn get_quota(&self, name: &str) -> Option<&ResourceQuota> {
        self.quotas.get(name)
    }

    /// Return the default encoding profile.
    #[must_use]
    pub fn default_profile(&self) -> Option<&EncodingProfile> {
        self.profiles.get(&self.default_profile)
    }

    /// Return the default resource quota.
    #[must_use]
    pub fn default_quota(&self) -> Option<&ResourceQuota> {
        self.quotas.get(&self.default_quota)
    }

    /// Total number of registered profiles.
    #[must_use]
    pub fn profile_count(&self) -> usize {
        self.profiles.len()
    }

    /// Total number of registered quotas.
    #[must_use]
    pub fn quota_count(&self) -> usize {
        self.quotas.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    // ── Original unit tests ───────────────────────────────────────────────────

    #[test]
    fn test_encoding_profile_default() {
        let p = EncodingProfile::default();
        assert_eq!(p.codec, "h264");
        assert_eq!(p.width, 1920);
        assert_eq!(p.height, 1080);
    }

    #[test]
    fn test_encoding_profile_fps() {
        let p = EncodingProfile {
            fps_num: 24000,
            fps_den: 1001,
            ..Default::default()
        };
        let fps = p.fps();
        assert!((fps - 23.976).abs() < 0.01);
    }

    #[test]
    fn test_encoding_profile_fps_zero_den() {
        let p = EncodingProfile {
            fps_den: 0,
            ..Default::default()
        };
        assert!((p.fps() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_is_crf_mode() {
        let mut p = EncodingProfile::default();
        p.crf = 23;
        p.bitrate_kbps = 0;
        assert!(p.is_crf_mode());
    }

    #[test]
    fn test_is_not_crf_mode() {
        let p = EncodingProfile::default();
        assert!(!p.is_crf_mode());
    }

    #[test]
    fn test_resource_quota_allows() {
        let q = ResourceQuota::new(8, 16384);
        assert!(q.allows(4, 8192, 1));
        assert!(!q.allows(16, 8192, 1));
        assert!(!q.allows(4, 32768, 1));
    }

    #[test]
    fn test_farm_config_default() {
        let cfg = FarmConfig::new();
        assert_eq!(cfg.profile_count(), 1);
        assert_eq!(cfg.quota_count(), 1);
        assert!(cfg.default_profile().is_some());
        assert!(cfg.default_quota().is_some());
    }

    #[test]
    fn test_add_profile() {
        let mut cfg = FarmConfig::new();
        cfg.add_profile(EncodingProfile::new("4k-hevc", "hevc"));
        assert_eq!(cfg.profile_count(), 2);
        let p = cfg.get_profile("4k-hevc").expect("profile should exist");
        assert_eq!(p.codec, "hevc");
    }

    #[test]
    fn test_add_quota() {
        let mut cfg = FarmConfig::new();
        cfg.add_quota("heavy", ResourceQuota::new(32, 65536));
        assert_eq!(cfg.quota_count(), 2);
        let q = cfg.get_quota("heavy").expect("quota should exist");
        assert_eq!(q.max_cpu_cores, 32);
    }

    #[test]
    fn test_missing_profile() {
        let cfg = FarmConfig::new();
        assert!(cfg.get_profile("nonexistent").is_none());
    }

    #[test]
    fn test_missing_quota() {
        let cfg = FarmConfig::new();
        assert!(cfg.get_quota("nonexistent").is_none());
    }

    #[test]
    fn test_encoding_profile_new() {
        let p = EncodingProfile::new("web", "vp9");
        assert_eq!(p.name, "web");
        assert_eq!(p.codec, "vp9");
        assert_eq!(p.container, "mp4");
    }

    // ── TOML round-trip ───────────────────────────────────────────────────────

    #[test]
    fn test_toml_round_trip_default() {
        let cfg = FarmConfig::new();
        let toml_str = cfg.to_toml().expect("serialize to TOML");
        assert!(!toml_str.is_empty());
        let restored: FarmConfig = FarmConfig::from_str_toml(&toml_str).expect("deserialize TOML");
        assert_eq!(restored.default_profile, cfg.default_profile);
        assert_eq!(restored.default_quota, cfg.default_quota);
        assert_eq!(restored.global_max_retries, cfg.global_max_retries);
        assert_eq!(restored.heartbeat_grace, cfg.heartbeat_grace);
        assert_eq!(restored.emergency_override, cfg.emergency_override);
    }

    #[test]
    fn test_toml_round_trip_with_extra_profile() {
        let mut cfg = FarmConfig::new();
        let mut extra = HashMap::new();
        extra.insert("x265-preset".to_string(), "slow".to_string());
        cfg.add_profile(EncodingProfile {
            name: "cinema-4k".to_string(),
            codec: "hevc".to_string(),
            container: "mkv".to_string(),
            bitrate_kbps: 0,
            crf: 18,
            width: 3840,
            height: 2160,
            fps_num: 24,
            fps_den: 1,
            passes: 2,
            extra,
        });
        cfg.add_quota("heavy", ResourceQuota::new(16, 32768));

        let toml_str = cfg.to_toml().expect("serialize to TOML");
        let restored = FarmConfig::from_str_toml(&toml_str).expect("deserialize TOML");
        assert_eq!(cfg, restored);
    }

    #[test]
    fn test_toml_file_round_trip() {
        let cfg = FarmConfig::new();
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        let toml_str = cfg.to_toml().expect("serialize");
        f.write_all(toml_str.as_bytes()).expect("write");
        let path = f.path().to_path_buf();
        let restored = FarmConfig::from_toml_file(&path).expect("from_toml_file");
        assert_eq!(restored.default_profile, cfg.default_profile);
        assert_eq!(restored.heartbeat_grace, cfg.heartbeat_grace);
    }

    // ── YAML round-trip ───────────────────────────────────────────────────────

    #[test]
    fn test_yaml_round_trip_default() {
        let cfg = FarmConfig::new();
        let yaml_str = cfg.to_yaml().expect("serialize to YAML");
        assert!(!yaml_str.is_empty());
        let restored = FarmConfig::from_str_yaml(&yaml_str).expect("deserialize YAML");
        assert_eq!(restored.default_profile, cfg.default_profile);
        assert_eq!(restored.default_quota, cfg.default_quota);
        assert_eq!(restored.global_max_retries, cfg.global_max_retries);
        assert_eq!(restored.heartbeat_grace, cfg.heartbeat_grace);
    }

    #[test]
    fn test_yaml_round_trip_with_extra_profile() {
        let mut cfg = FarmConfig::new();
        cfg.add_profile(EncodingProfile::new("web-vp9", "vp9"));
        cfg.add_quota("light", ResourceQuota::new(2, 4096));
        cfg.emergency_override = true;

        let yaml_str = cfg.to_yaml().expect("serialize to YAML");
        let restored = FarmConfig::from_str_yaml(&yaml_str).expect("deserialize YAML");
        assert_eq!(cfg, restored);
    }

    #[test]
    fn test_yaml_file_round_trip() {
        let mut cfg = FarmConfig::new();
        cfg.global_max_retries = 7;
        cfg.heartbeat_grace = Duration::from_secs(120);

        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        let yaml_str = cfg.to_yaml().expect("serialize");
        f.write_all(yaml_str.as_bytes()).expect("write");
        let path = f.path().to_path_buf();
        let restored = FarmConfig::from_yaml_file(&path).expect("from_yaml_file");
        assert_eq!(restored.global_max_retries, 7);
        assert_eq!(restored.heartbeat_grace, Duration::from_secs(120));
    }

    #[test]
    fn test_yaml_invalid_returns_error() {
        let result = FarmConfig::from_str_yaml("not: valid: yaml: {{{");
        assert!(result.is_err());
    }

    #[test]
    fn test_toml_invalid_returns_error() {
        let result = FarmConfig::from_str_toml("not = [valid toml");
        assert!(result.is_err());
    }

    #[test]
    fn test_duration_serde_via_toml() {
        // Verify that Duration fields survive TOML round-trip with correct seconds
        let mut cfg = FarmConfig::new();
        cfg.heartbeat_grace = Duration::from_secs(45);
        let toml_str = cfg.to_toml().expect("serialize");
        let restored = FarmConfig::from_str_toml(&toml_str).expect("deserialize");
        assert_eq!(restored.heartbeat_grace.as_secs(), 45);
    }

    #[test]
    fn test_duration_serde_via_yaml() {
        let mut cfg = FarmConfig::new();
        cfg.heartbeat_grace = Duration::from_secs(300);
        let yaml_str = cfg.to_yaml().expect("serialize");
        let restored = FarmConfig::from_str_yaml(&yaml_str).expect("deserialize");
        assert_eq!(restored.heartbeat_grace.as_secs(), 300);
    }

    #[test]
    fn test_toml_contains_expected_keys() {
        let cfg = FarmConfig::new();
        let toml_str = cfg.to_toml().expect("serialize");
        assert!(toml_str.contains("default_profile"));
        assert!(toml_str.contains("global_max_retries"));
        assert!(toml_str.contains("heartbeat_grace"));
    }

    #[test]
    fn test_yaml_contains_expected_keys() {
        let cfg = FarmConfig::new();
        let yaml_str = cfg.to_yaml().expect("serialize");
        assert!(yaml_str.contains("default_profile"));
        assert!(yaml_str.contains("global_max_retries"));
        assert!(yaml_str.contains("heartbeat_grace"));
    }
}
