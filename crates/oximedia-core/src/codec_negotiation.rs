//! Codec negotiation utilities for `OxiMedia`.
//!
//! This module provides types and functions for negotiating codec parameters
//! between local and remote endpoints, preferring hardware-accelerated codecs
//! when available.
//!
//! # Example
//!
//! ```
//! use oximedia_core::codec_negotiation::{CodecCapability, CodecNegotiator, negotiate};
//!
//! let local = vec![
//!     CodecCapability {
//!         name: "av1".to_string(),
//!         profiles: vec!["main".to_string()],
//!         max_level: 40,
//!         hardware_accelerated: false,
//!     },
//! ];
//! let remote = vec![
//!     CodecCapability {
//!         name: "av1".to_string(),
//!         profiles: vec!["main".to_string()],
//!         max_level: 30,
//!         hardware_accelerated: false,
//!     },
//! ];
//! let result = negotiate(&local, &remote);
//! assert!(result.is_some());
//! assert_eq!(result?.selected_codec, "av1");
//! ```

#![allow(dead_code)]

/// Describes the codec capabilities of one endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecCapability {
    /// Codec name (e.g. `"av1"`, `"vp9"`).
    pub name: String,
    /// List of supported codec profiles.
    pub profiles: Vec<String>,
    /// Maximum codec level supported (e.g. 40 for level 4.0).
    pub max_level: u32,
    /// Whether the codec benefits from hardware acceleration on this device.
    pub hardware_accelerated: bool,
}

impl CodecCapability {
    /// Creates a new `CodecCapability`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        profiles: Vec<String>,
        max_level: u32,
        hardware_accelerated: bool,
    ) -> Self {
        Self {
            name: name.into(),
            profiles,
            max_level,
            hardware_accelerated,
        }
    }

    /// Returns `true` if `profile` is listed in this capability.
    #[must_use]
    pub fn supports_profile(&self, profile: &str) -> bool {
        self.profiles.iter().any(|p| p == profile)
    }

    /// Returns `true` if this codec uses hardware acceleration.
    #[must_use]
    pub fn is_hw_accelerated(&self) -> bool {
        self.hardware_accelerated
    }
}

/// Handles codec negotiation between a local and a remote set of capabilities.
#[derive(Debug, Default)]
pub struct CodecNegotiator {
    /// Codecs supported locally.
    pub local_caps: Vec<CodecCapability>,
    /// Codecs supported by the remote endpoint.
    pub remote_caps: Vec<CodecCapability>,
}

impl CodecNegotiator {
    /// Creates an empty `CodecNegotiator`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a local codec capability.
    pub fn add_local(&mut self, cap: CodecCapability) {
        self.local_caps.push(cap);
    }

    /// Adds a remote codec capability.
    pub fn add_remote(&mut self, cap: CodecCapability) {
        self.remote_caps.push(cap);
    }

    /// Returns the names of codecs supported by both endpoints.
    #[must_use]
    pub fn common_codecs(&self) -> Vec<&str> {
        self.local_caps
            .iter()
            .filter(|l| self.remote_caps.iter().any(|r| r.name == l.name))
            .map(|l| l.name.as_str())
            .collect()
    }

    /// Returns the preferred common codec, favouring hardware-accelerated ones.
    ///
    /// Returns `None` when there are no codecs in common.
    #[must_use]
    pub fn preferred_codec(&self) -> Option<&str> {
        // Collect common names first.
        let common = self.common_codecs();
        if common.is_empty() {
            return None;
        }
        // Prefer hardware-accelerated; fall back to first common.
        for name in &common {
            if let Some(cap) = self.local_caps.iter().find(|c| &c.name.as_str() == name) {
                if cap.hardware_accelerated {
                    return Some(name);
                }
            }
        }
        common.into_iter().next()
    }
}

/// The result of a successful codec negotiation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegotiationResult {
    /// The codec selected by both endpoints.
    pub selected_codec: String,
    /// The agreed-upon profile.
    pub profile: String,
    /// The agreed-upon level (minimum of local and remote max levels).
    pub level: u32,
    /// Whether the selected codec uses hardware acceleration on the local side.
    pub hardware_accelerated: bool,
}

impl NegotiationResult {
    /// Returns `true` if the selected codec uses hardware acceleration.
    #[must_use]
    pub fn is_hardware(&self) -> bool {
        self.hardware_accelerated
    }
}

/// Attempts to negotiate a codec between `local` and `remote` capability sets.
///
/// Hardware-accelerated codecs are preferred. The first common codec whose
/// profile list has at least one entry in common is selected.
///
/// Returns `None` when no mutually supported codec/profile pair exists.
#[must_use]
pub fn negotiate(
    local: &[CodecCapability],
    remote: &[CodecCapability],
) -> Option<NegotiationResult> {
    // Build a prioritised list: hw-accelerated local caps first.
    let mut ordered: Vec<&CodecCapability> = local.iter().collect();
    ordered.sort_by_key(|c| u8::from(!c.hardware_accelerated));

    for local_cap in ordered {
        if let Some(remote_cap) = remote.iter().find(|r| r.name == local_cap.name) {
            // Find a common profile.
            let common_profile = local_cap
                .profiles
                .iter()
                .find(|p| remote_cap.profiles.contains(p));
            if let Some(profile) = common_profile {
                let level = local_cap.max_level.min(remote_cap.max_level);
                return Some(NegotiationResult {
                    selected_codec: local_cap.name.clone(),
                    profile: profile.clone(),
                    level,
                    hardware_accelerated: local_cap.hardware_accelerated,
                });
            }
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn av1_cap(hw: bool) -> CodecCapability {
        CodecCapability::new("av1", vec!["main".to_string(), "high".to_string()], 40, hw)
    }

    fn vp9_cap() -> CodecCapability {
        CodecCapability::new("vp9", vec!["profile0".to_string()], 50, false)
    }

    // 1. supports_profile – positive
    #[test]
    fn test_supports_profile_positive() {
        let cap = av1_cap(false);
        assert!(cap.supports_profile("main"));
        assert!(cap.supports_profile("high"));
    }

    // 2. supports_profile – negative
    #[test]
    fn test_supports_profile_negative() {
        let cap = av1_cap(false);
        assert!(!cap.supports_profile("baseline"));
    }

    // 3. is_hw_accelerated
    #[test]
    fn test_is_hw_accelerated() {
        assert!(av1_cap(true).is_hw_accelerated());
        assert!(!av1_cap(false).is_hw_accelerated());
    }

    // 4. CodecNegotiator::common_codecs – overlap
    #[test]
    fn test_common_codecs_overlap() {
        let mut neg = CodecNegotiator::new();
        neg.add_local(av1_cap(false));
        neg.add_local(vp9_cap());
        neg.add_remote(av1_cap(false));
        let common = neg.common_codecs();
        assert_eq!(common, vec!["av1"]);
    }

    // 5. CodecNegotiator::common_codecs – no overlap
    #[test]
    fn test_common_codecs_no_overlap() {
        let mut neg = CodecNegotiator::new();
        neg.add_local(vp9_cap());
        neg.add_remote(av1_cap(false));
        assert!(neg.common_codecs().is_empty());
    }

    // 6. preferred_codec – hw preferred
    #[test]
    fn test_preferred_codec_hw_first() {
        let mut neg = CodecNegotiator::new();
        neg.add_local(vp9_cap()); // software
        neg.add_local(av1_cap(true)); // hardware
        neg.add_remote(vp9_cap());
        neg.add_remote(av1_cap(true));
        // av1 is hw, should be preferred
        assert_eq!(neg.preferred_codec(), Some("av1"));
    }

    // 7. preferred_codec – no common
    #[test]
    fn test_preferred_codec_none() {
        let mut neg = CodecNegotiator::new();
        neg.add_local(vp9_cap());
        neg.add_remote(av1_cap(false));
        assert!(neg.preferred_codec().is_none());
    }

    // 8. preferred_codec – falls back to first when no hw
    #[test]
    fn test_preferred_codec_fallback() {
        let mut neg = CodecNegotiator::new();
        neg.add_local(av1_cap(false));
        neg.add_local(vp9_cap());
        neg.add_remote(av1_cap(false));
        neg.add_remote(vp9_cap());
        // No hw acceleration; first common codec returned
        let pref = neg.preferred_codec();
        assert!(pref.is_some());
    }

    // 9. negotiate – success
    #[test]
    fn test_negotiate_success() {
        let local = vec![av1_cap(false)];
        let remote = vec![av1_cap(false)];
        let result = negotiate(&local, &remote).expect("negotiation should succeed");
        assert_eq!(result.selected_codec, "av1");
        assert!(result.profile == "main" || result.profile == "high");
        assert_eq!(result.level, 40);
        assert!(!result.is_hardware());
    }

    // 10. negotiate – hw preferred
    #[test]
    fn test_negotiate_prefers_hw() {
        let local = vec![vp9_cap(), av1_cap(true)];
        let remote = vec![vp9_cap(), av1_cap(true)];
        let result = negotiate(&local, &remote).expect("negotiation should succeed");
        assert_eq!(result.selected_codec, "av1");
        assert!(result.is_hardware());
    }

    // 11. negotiate – level is min of both
    #[test]
    fn test_negotiate_level_min() {
        let local = vec![CodecCapability::new(
            "av1",
            vec!["main".to_string()],
            50,
            false,
        )];
        let remote = vec![CodecCapability::new(
            "av1",
            vec!["main".to_string()],
            30,
            false,
        )];
        let result = negotiate(&local, &remote).expect("negotiation should succeed");
        assert_eq!(result.level, 30);
    }

    // 12. negotiate – no common codec returns None
    #[test]
    fn test_negotiate_no_common() {
        let local = vec![av1_cap(false)];
        let remote = vec![vp9_cap()];
        assert!(negotiate(&local, &remote).is_none());
    }

    // 13. negotiate – profile mismatch returns None
    #[test]
    fn test_negotiate_profile_mismatch() {
        let local = vec![CodecCapability::new(
            "av1",
            vec!["high".to_string()],
            40,
            false,
        )];
        let remote = vec![CodecCapability::new(
            "av1",
            vec!["baseline".to_string()],
            40,
            false,
        )];
        assert!(negotiate(&local, &remote).is_none());
    }

    // 14. NegotiationResult::is_hardware
    #[test]
    fn test_negotiation_result_is_hardware() {
        let r = NegotiationResult {
            selected_codec: "av1".to_string(),
            profile: "main".to_string(),
            level: 40,
            hardware_accelerated: true,
        };
        assert!(r.is_hardware());
        let r2 = NegotiationResult {
            hardware_accelerated: false,
            ..r
        };
        assert!(!r2.is_hardware());
    }
}
