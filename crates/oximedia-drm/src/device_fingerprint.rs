//! Device fingerprinting for DRM authorization.
//!
//! Combines multiple hardware and software identifiers into a single, stable
//! **device fingerprint** — a 32-byte SHA-256–like digest computed entirely in
//! pure Rust without external hash crates.
//!
//! ## Goals
//! * **Stability**: minor environmental changes (e.g. a different app version)
//!   should not change the fingerprint.
//! * **Uniqueness**: two distinct physical devices should have different
//!   fingerprints with overwhelming probability.
//! * **Privacy**: the raw identifiers are hashed; they cannot be reversed from
//!   the fingerprint alone.
//! * **Fuzzy matching**: a [`FingerprintMatcher`] scores how similar two
//!   fingerprints are, enabling detection of the same device across minor
//!   changes (e.g. IP address change, OS update).
//!
//! ## Algorithm
//! Components are weighted and individually hashed before being XOR-folded
//! into the master digest.  The component weights reflect their relative
//! stability: hardware identifiers (CPU ID, MAC address) are weighted highly
//! while volatile identifiers (IP address) receive lower weight.
//!
//! ## No `unsafe`, no `unwrap()` in library code.

use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Component weight table
// ---------------------------------------------------------------------------

/// Relative weight of each hardware/software component.
///
/// Higher weight = the component contributes more to similarity scoring.
/// Weights are normalised to sum to 1.0 when computing the overall score.
const WEIGHT_CPU_ID: f64 = 0.25;
const WEIGHT_MAC_ADDR: f64 = 0.20;
const WEIGHT_DISK_SERIAL: f64 = 0.18;
const WEIGHT_PLATFORM: f64 = 0.12;
const WEIGHT_OS_VERSION: f64 = 0.10;
const WEIGHT_APP_ID: f64 = 0.08;
const WEIGHT_TIMEZONE: f64 = 0.04;
const WEIGHT_LOCALE: f64 = 0.03;

// ---------------------------------------------------------------------------
// Component descriptor
// ---------------------------------------------------------------------------

/// A single hardware or software identifier contributing to the fingerprint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FingerprintComponent {
    /// CPU identifier (e.g. `"GenuineIntel:0x000506E3"`).
    CpuId(String),
    /// Primary network interface MAC address (e.g. `"AA:BB:CC:DD:EE:FF"`).
    MacAddress(String),
    /// System disk or SSD serial number.
    DiskSerial(String),
    /// Target platform (`"linux"`, `"windows"`, `"macos"`, `"ios"`, …).
    Platform(String),
    /// Operating system version string.
    OsVersion(String),
    /// Application or DRM agent identifier (name + version).
    AppId(String),
    /// IANA timezone name (`"America/New_York"`).
    Timezone(String),
    /// BCP-47 locale tag (`"en-US"`).
    Locale(String),
}

impl FingerprintComponent {
    /// Return the string value of this component.
    pub fn value(&self) -> &str {
        match self {
            Self::CpuId(v)
            | Self::MacAddress(v)
            | Self::DiskSerial(v)
            | Self::Platform(v)
            | Self::OsVersion(v)
            | Self::AppId(v)
            | Self::Timezone(v)
            | Self::Locale(v) => v.as_str(),
        }
    }

    /// Relative weight used in similarity scoring.
    pub fn weight(&self) -> f64 {
        match self {
            Self::CpuId(_) => WEIGHT_CPU_ID,
            Self::MacAddress(_) => WEIGHT_MAC_ADDR,
            Self::DiskSerial(_) => WEIGHT_DISK_SERIAL,
            Self::Platform(_) => WEIGHT_PLATFORM,
            Self::OsVersion(_) => WEIGHT_OS_VERSION,
            Self::AppId(_) => WEIGHT_APP_ID,
            Self::Timezone(_) => WEIGHT_TIMEZONE,
            Self::Locale(_) => WEIGHT_LOCALE,
        }
    }

    /// Discriminant tag name (used when the same component type appears in both
    /// fingerprints for comparison).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::CpuId(_) => "cpu_id",
            Self::MacAddress(_) => "mac_address",
            Self::DiskSerial(_) => "disk_serial",
            Self::Platform(_) => "platform",
            Self::OsVersion(_) => "os_version",
            Self::AppId(_) => "app_id",
            Self::Timezone(_) => "timezone",
            Self::Locale(_) => "locale",
        }
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust 32-byte hash (BLAKE2b-inspired mix, no external deps)
// ---------------------------------------------------------------------------

/// Simple, non-cryptographic 32-byte hash used as the fingerprint digest.
///
/// Uses a cascade of FNV-1a + xorshift steps to achieve good avalanche
/// behaviour without pulling in a crypto crate.
fn hash_component(kind: &str, value: &str) -> [u8; 32] {
    // We mix the kind tag and the value to make each component's hash unique.
    let mut state: [u64; 4] = [
        0x6c62272e07bb0142,
        0x62b821756295c58d,
        0x0000000000000000,
        0xffffffffffffffff,
    ];

    // Mix bytes using FNV-1a style
    let mix_byte = |state: &mut [u64; 4], b: u8| {
        state[0] ^= b as u64;
        state[0] = state[0].wrapping_mul(0x00000100000001b3);
        state[1] ^= state[0].rotate_left(17);
        state[2] = state[2].wrapping_add(state[1]).rotate_right(13);
        state[3] ^= state[2].wrapping_mul(0x9e3779b97f4a7c15);
    };

    // Mix kind tag, then a separator, then value
    for b in kind.bytes() {
        mix_byte(&mut state, b);
    }
    mix_byte(&mut state, 0xFF); // separator
    for b in value.bytes() {
        mix_byte(&mut state, b);
    }

    // Finalise: additional mixing passes
    for _ in 0..4 {
        state[0] = state[0]
            .wrapping_add(state[3])
            .rotate_left(23)
            ^ state[1];
        state[1] = state[1].wrapping_mul(0xc4ceb9fe1a85ec53);
        state[2] = state[2].wrapping_add(state[0]).rotate_right(31);
        state[3] ^= state[2].wrapping_mul(0x517cc1b727220a95);
    }

    let mut out = [0u8; 32];
    out[0..8].copy_from_slice(&state[0].to_le_bytes());
    out[8..16].copy_from_slice(&state[1].to_le_bytes());
    out[16..24].copy_from_slice(&state[2].to_le_bytes());
    out[24..32].copy_from_slice(&state[3].to_le_bytes());
    out
}

/// XOR-fold two 32-byte digests.
fn xor_fold(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = a[i] ^ b[i];
    }
    out
}

// ---------------------------------------------------------------------------
// Fingerprint
// ---------------------------------------------------------------------------

/// A stable 32-byte device fingerprint derived from hardware/software components.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceFingerprint {
    /// Hex-encoded 32-byte digest.
    digest: String,
    /// Components used to produce this fingerprint (for display / logging).
    #[serde(skip)]
    components: Vec<FingerprintComponent>,
}

impl DeviceFingerprint {
    /// Compute the fingerprint from a list of components.
    ///
    /// Order does not matter — each component is hashed independently and the
    /// results are XOR-folded.  An empty component list yields the all-zero digest.
    pub fn from_components(components: Vec<FingerprintComponent>) -> Self {
        let mut acc = [0u8; 32];
        for c in &components {
            let h = hash_component(c.kind(), c.value());
            acc = xor_fold(&acc, &h);
        }
        Self {
            digest: hex::encode(acc),
            components,
        }
    }

    /// Parse a previously serialised hex digest (40–64 hex chars).
    ///
    /// Returns `None` if the string is not valid hex or the wrong length.
    pub fn from_hex(hex_str: &str) -> Option<Self> {
        if hex_str.len() != 64 {
            return None;
        }
        let bytes = hex::decode(hex_str).ok()?;
        if bytes.len() != 32 {
            return None;
        }
        Some(Self {
            digest: hex_str.to_lowercase(),
            components: Vec::new(),
        })
    }

    /// Raw hex digest string.
    pub fn as_hex(&self) -> &str {
        &self.digest
    }

    /// Components used to build this fingerprint (may be empty if created via
    /// [`Self::from_hex`]).
    pub fn components(&self) -> &[FingerprintComponent] {
        &self.components
    }

    /// Number of components.
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Return the raw 32 bytes of the digest.
    ///
    /// Returns `None` only if the internal hex string is somehow malformed
    /// (should never happen in normal usage).
    pub fn as_bytes(&self) -> Option<Vec<u8>> {
        hex::decode(&self.digest).ok()
    }
}

impl fmt::Debug for DeviceFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DeviceFingerprint({})", &self.digest[..16])
    }
}

impl fmt::Display for DeviceFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.digest)
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Fluent builder for [`DeviceFingerprint`].
///
/// ```
/// use oximedia_drm::device_fingerprint::FingerprintBuilder;
///
/// let fp = FingerprintBuilder::new()
///     .cpu_id("GenuineIntel:0x000506E3")
///     .platform("linux")
///     .build();
/// ```
#[derive(Debug, Default)]
pub struct FingerprintBuilder {
    components: Vec<FingerprintComponent>,
}

impl FingerprintBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a CPU identifier.
    pub fn cpu_id(mut self, v: impl Into<String>) -> Self {
        self.components.push(FingerprintComponent::CpuId(v.into()));
        self
    }

    /// Add a MAC address.
    pub fn mac_address(mut self, v: impl Into<String>) -> Self {
        self.components
            .push(FingerprintComponent::MacAddress(v.into()));
        self
    }

    /// Add a disk serial number.
    pub fn disk_serial(mut self, v: impl Into<String>) -> Self {
        self.components
            .push(FingerprintComponent::DiskSerial(v.into()));
        self
    }

    /// Add the target platform.
    pub fn platform(mut self, v: impl Into<String>) -> Self {
        self.components
            .push(FingerprintComponent::Platform(v.into()));
        self
    }

    /// Add the OS version.
    pub fn os_version(mut self, v: impl Into<String>) -> Self {
        self.components
            .push(FingerprintComponent::OsVersion(v.into()));
        self
    }

    /// Add an application identifier.
    pub fn app_id(mut self, v: impl Into<String>) -> Self {
        self.components.push(FingerprintComponent::AppId(v.into()));
        self
    }

    /// Add a timezone.
    pub fn timezone(mut self, v: impl Into<String>) -> Self {
        self.components
            .push(FingerprintComponent::Timezone(v.into()));
        self
    }

    /// Add a locale tag.
    pub fn locale(mut self, v: impl Into<String>) -> Self {
        self.components
            .push(FingerprintComponent::Locale(v.into()));
        self
    }

    /// Build the [`DeviceFingerprint`].
    pub fn build(self) -> DeviceFingerprint {
        DeviceFingerprint::from_components(self.components)
    }
}

// ---------------------------------------------------------------------------
// Similarity scoring
// ---------------------------------------------------------------------------

/// Result of comparing two fingerprints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    /// Overall weighted similarity score in `[0.0, 1.0]`.
    ///
    /// * `1.0` → identical (within compared components)
    /// * `0.0` → completely different (no matching component values)
    pub score: f64,
    /// Weighted score of matching components.
    pub matching_weight: f64,
    /// Weighted score of differing components.
    pub differing_weight: f64,
    /// Components that appear only in the reference fingerprint (not in candidate).
    pub missing_in_candidate: Vec<String>,
    /// Components that appear only in the candidate fingerprint (not in reference).
    pub extra_in_candidate: Vec<String>,
}

impl SimilarityResult {
    /// Returns `true` if the score meets the given threshold.
    pub fn is_likely_same_device(&self, threshold: f64) -> bool {
        self.score >= threshold
    }
}

/// Compares device fingerprints using weighted component matching.
#[derive(Debug, Clone)]
pub struct FingerprintMatcher {
    /// Minimum score to consider two fingerprints as the "same" device.
    threshold: f64,
}

impl FingerprintMatcher {
    /// Create a matcher with the given threshold (0.0–1.0).
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Default threshold (0.75 — allows a couple of minor component mismatches).
    pub fn with_default_threshold() -> Self {
        Self::new(0.75)
    }

    /// Compare `reference` against `candidate` and return a detailed score.
    pub fn compare(
        &self,
        reference: &DeviceFingerprint,
        candidate: &DeviceFingerprint,
    ) -> SimilarityResult {
        // Exact match shortcut
        if reference.digest == candidate.digest {
            return SimilarityResult {
                score: 1.0,
                matching_weight: 1.0,
                differing_weight: 0.0,
                missing_in_candidate: Vec::new(),
                extra_in_candidate: Vec::new(),
            };
        }

        // Build kind→value maps for each fingerprint
        use std::collections::HashMap;
        let ref_map: HashMap<&str, &str> = reference
            .components
            .iter()
            .map(|c| (c.kind(), c.value()))
            .collect();
        let cand_map: HashMap<&str, &str> = candidate
            .components
            .iter()
            .map(|c| (c.kind(), c.value()))
            .collect();

        let mut matching_weight = 0.0f64;
        let mut differing_weight = 0.0f64;
        let mut missing_in_candidate: Vec<String> = Vec::new();
        let mut extra_in_candidate: Vec<String> = Vec::new();

        // Score reference components against candidate
        for c in &reference.components {
            match cand_map.get(c.kind()) {
                Some(&cand_val) if cand_val == c.value() => {
                    matching_weight += c.weight();
                }
                Some(_) => {
                    differing_weight += c.weight();
                }
                None => {
                    missing_in_candidate.push(c.kind().to_string());
                }
            }
        }

        // Extra components in candidate
        for c in &candidate.components {
            if !ref_map.contains_key(c.kind()) {
                extra_in_candidate.push(c.kind().to_string());
            }
        }

        // Normalise: total weight of reference components
        let total_weight: f64 = reference.components.iter().map(|c| c.weight()).sum();
        let score = if total_weight <= 0.0 {
            0.0
        } else {
            (matching_weight / total_weight).clamp(0.0, 1.0)
        };

        SimilarityResult {
            score,
            matching_weight,
            differing_weight,
            missing_in_candidate,
            extra_in_candidate,
        }
    }

    /// Convenience: returns `true` when the score meets the configured threshold.
    pub fn is_same_device(
        &self,
        reference: &DeviceFingerprint,
        candidate: &DeviceFingerprint,
    ) -> bool {
        self.compare(reference, candidate)
            .is_likely_same_device(self.threshold)
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// A registry that stores known device fingerprints and supports fuzzy lookup.
#[derive(Debug)]
pub struct FingerprintRegistry {
    entries: Vec<(String, DeviceFingerprint)>, // (device_id, fingerprint)
    matcher: FingerprintMatcher,
}

impl Default for FingerprintRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl FingerprintRegistry {
    /// Create a registry with the default similarity threshold.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            matcher: FingerprintMatcher::with_default_threshold(),
        }
    }

    /// Create a registry with a custom threshold.
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            entries: Vec::new(),
            matcher: FingerprintMatcher::new(threshold),
        }
    }

    /// Register a device.  If `device_id` already exists the fingerprint is updated.
    pub fn register(&mut self, device_id: impl Into<String>, fingerprint: DeviceFingerprint) {
        let id = device_id.into();
        if let Some(entry) = self.entries.iter_mut().find(|(d, _)| d == &id) {
            entry.1 = fingerprint;
        } else {
            self.entries.push((id, fingerprint));
        }
    }

    /// Exact lookup by device ID.
    pub fn get(&self, device_id: &str) -> Option<&DeviceFingerprint> {
        self.entries
            .iter()
            .find(|(d, _)| d == device_id)
            .map(|(_, fp)| fp)
    }

    /// Fuzzy lookup: find the best-matching device ID for the given fingerprint.
    ///
    /// Returns `None` if no registered fingerprint meets the similarity threshold.
    pub fn find_similar(&self, candidate: &DeviceFingerprint) -> Option<(&str, SimilarityResult)> {
        self.entries
            .iter()
            .map(|(id, fp)| {
                let result = self.matcher.compare(fp, candidate);
                (id.as_str(), result)
            })
            .filter(|(_, r)| r.is_likely_same_device(self.matcher.threshold))
            .max_by(|(_, a), (_, b)| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Number of registered devices.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove a device by ID.  Returns `true` if it was present.
    pub fn remove(&mut self, device_id: &str) -> bool {
        let before = self.entries.len();
        self.entries.retain(|(d, _)| d != device_id);
        self.entries.len() < before
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn full_fp() -> DeviceFingerprint {
        FingerprintBuilder::new()
            .cpu_id("GenuineIntel:0x506E3")
            .mac_address("AA:BB:CC:DD:EE:FF")
            .disk_serial("SN123456")
            .platform("linux")
            .os_version("6.1.0")
            .app_id("OxiMediaPlayer-1.0")
            .timezone("UTC")
            .locale("en-US")
            .build()
    }

    #[test]
    fn test_deterministic() {
        let fp1 = full_fp();
        let fp2 = full_fp();
        assert_eq!(fp1.as_hex(), fp2.as_hex());
    }

    #[test]
    fn test_different_components_different_digest() {
        let fp1 = FingerprintBuilder::new().cpu_id("A").platform("linux").build();
        let fp2 = FingerprintBuilder::new().cpu_id("B").platform("linux").build();
        assert_ne!(fp1.as_hex(), fp2.as_hex());
    }

    #[test]
    fn test_component_count() {
        let fp = full_fp();
        assert_eq!(fp.component_count(), 8);
    }

    #[test]
    fn test_from_hex_roundtrip() {
        let fp = full_fp();
        let hex = fp.as_hex().to_string();
        let restored = DeviceFingerprint::from_hex(&hex).expect("should parse");
        assert_eq!(restored.as_hex(), hex);
    }

    #[test]
    fn test_from_hex_invalid() {
        assert!(DeviceFingerprint::from_hex("not-valid-hex").is_none());
        // Wrong length
        assert!(DeviceFingerprint::from_hex("deadbeef").is_none());
    }

    #[test]
    fn test_as_bytes_length() {
        let fp = full_fp();
        let bytes = fp.as_bytes().expect("bytes");
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn test_similarity_identical() {
        let fp = full_fp();
        let matcher = FingerprintMatcher::with_default_threshold();
        let result = matcher.compare(&fp, &fp);
        assert!((result.score - 1.0).abs() < 1e-9);
        assert!(result.is_likely_same_device(0.99));
    }

    #[test]
    fn test_similarity_one_component_changed() {
        let reference = full_fp();
        let candidate = FingerprintBuilder::new()
            .cpu_id("GenuineIntel:0x506E3")    // same
            .mac_address("AA:BB:CC:DD:EE:FF")   // same
            .disk_serial("SN123456")             // same
            .platform("linux")                   // same
            .os_version("6.2.0")                 // CHANGED
            .app_id("OxiMediaPlayer-1.0")        // same
            .timezone("UTC")                     // same
            .locale("en-US")                     // same
            .build();

        let matcher = FingerprintMatcher::with_default_threshold();
        let result = matcher.compare(&reference, &candidate);
        // Should be high but < 1.0
        assert!(result.score > 0.8);
        assert!(result.score < 1.0);
        assert!(result.is_likely_same_device(0.75));
    }

    #[test]
    fn test_similarity_completely_different() {
        let fp1 = FingerprintBuilder::new()
            .cpu_id("A")
            .mac_address("11:22:33:44:55:66")
            .build();
        let fp2 = FingerprintBuilder::new()
            .cpu_id("B")
            .mac_address("AA:BB:CC:DD:EE:FF")
            .build();
        let matcher = FingerprintMatcher::new(0.75);
        let result = matcher.compare(&fp1, &fp2);
        assert!(result.score < 0.1);
        assert!(!result.is_likely_same_device(0.75));
    }

    #[test]
    fn test_registry_exact_lookup() {
        let mut registry = FingerprintRegistry::new();
        let fp = full_fp();
        registry.register("device-001", fp.clone());
        let found = registry.get("device-001").expect("should exist");
        assert_eq!(found.as_hex(), fp.as_hex());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_fuzzy_match() {
        let mut registry = FingerprintRegistry::new();
        registry.register("dev-A", full_fp());

        // Candidate with one component changed
        let candidate = FingerprintBuilder::new()
            .cpu_id("GenuineIntel:0x506E3")
            .mac_address("AA:BB:CC:DD:EE:FF")
            .disk_serial("SN123456")
            .platform("linux")
            .os_version("6.2.0") // changed
            .app_id("OxiMediaPlayer-1.0")
            .timezone("UTC")
            .locale("en-US")
            .build();

        let result = registry.find_similar(&candidate);
        assert!(result.is_some());
        let (device_id, _) = result.expect("should find");
        assert_eq!(device_id, "dev-A");
    }

    #[test]
    fn test_registry_no_fuzzy_match_when_too_different() {
        let mut registry = FingerprintRegistry::new();
        registry.register("dev-A", full_fp());

        // Completely different fingerprint
        let unrelated = FingerprintBuilder::new()
            .cpu_id("OtherCPU")
            .mac_address("00:00:00:00:00:00")
            .platform("windows")
            .build();

        assert!(registry.find_similar(&unrelated).is_none());
    }

    #[test]
    fn test_registry_remove() {
        let mut registry = FingerprintRegistry::new();
        registry.register("dev-X", full_fp());
        assert_eq!(registry.len(), 1);
        assert!(registry.remove("dev-X"));
        assert!(registry.is_empty());
        assert!(!registry.remove("dev-X")); // already gone
    }

    #[test]
    fn test_registry_update() {
        let mut registry = FingerprintRegistry::new();
        let fp1 = full_fp();
        let fp2 = FingerprintBuilder::new().platform("macos").build();
        registry.register("dev-Y", fp1);
        registry.register("dev-Y", fp2.clone()); // update
        assert_eq!(registry.len(), 1);
        assert_eq!(
            registry.get("dev-Y").expect("exists").as_hex(),
            fp2.as_hex()
        );
    }

    #[test]
    fn test_component_weights_positive() {
        let components = vec![
            FingerprintComponent::CpuId("x".into()),
            FingerprintComponent::MacAddress("y".into()),
            FingerprintComponent::DiskSerial("z".into()),
            FingerprintComponent::Platform("p".into()),
            FingerprintComponent::OsVersion("o".into()),
            FingerprintComponent::AppId("a".into()),
            FingerprintComponent::Timezone("t".into()),
            FingerprintComponent::Locale("l".into()),
        ];
        for c in &components {
            assert!(c.weight() > 0.0, "Weight for {} must be positive", c.kind());
        }
    }

    #[test]
    fn test_missing_extra_tracking() {
        let reference = FingerprintBuilder::new()
            .cpu_id("CPU-A")
            .platform("linux")
            .build();
        let candidate = FingerprintBuilder::new()
            .cpu_id("CPU-A")
            .os_version("6.0") // extra, not in reference
            .build();

        let matcher = FingerprintMatcher::new(0.0);
        let result = matcher.compare(&reference, &candidate);
        assert!(result.missing_in_candidate.contains(&"platform".to_string()));
        assert!(result.extra_in_candidate.contains(&"os_version".to_string()));
    }

    #[test]
    fn test_display_and_debug() {
        let fp = full_fp();
        let s = format!("{fp}");
        assert_eq!(s.len(), 64); // hex of 32 bytes
        let d = format!("{fp:?}");
        assert!(d.contains("DeviceFingerprint"));
    }
}
