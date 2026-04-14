//! Pre-roll and bumper injection at packaging time.
//!
//! A *pre-roll* is a fixed segment inserted before the main content (e.g., a
//! studio logo or countdown slate).  A *bumper* is a short segment inserted at
//! the end of the content (e.g., a "stay tuned" card or end card).
//!
//! This module provides:
//!
//! - [`PreRollConfig`] — describes the pre-roll/bumper segment to inject.
//! - [`BumperConfig`] — describes an end bumper.
//! - [`InjectionPoint`] — either a pre-roll or bumper position.
//! - [`PreRollInjector`] — builds an ordered segment list with injections applied.
//!
//! # Segment ordering
//!
//! Given a main content segment list `[A, B, C]` and a pre-roll segment `P`,
//! the injected list is `[P, A, B, C]`.  With a bumper `Q` appended, it
//! becomes `[P, A, B, C, Q]`.
//!
//! # Example
//!
//! ```rust
//! use oximedia_packager::pre_roll::{PreRollConfig, BumperConfig, PreRollInjector};
//!
//! let pre = PreRollConfig {
//!     segment_uri: "preroll.ts".to_string(),
//!     duration_secs: 5.0,
//!     is_skippable: false,
//! };
//! let injector = PreRollInjector::new().with_pre_roll(pre);
//!
//! let content = vec!["seg0.ts", "seg1.ts", "seg2.ts"];
//! let result = injector.inject_uris(&content);
//! assert_eq!(result[0], "preroll.ts");
//! assert_eq!(result[1], "seg0.ts");
//! ```

// ---------------------------------------------------------------------------
// PreRollConfig
// ---------------------------------------------------------------------------

/// Configuration for a pre-roll segment.
#[derive(Debug, Clone)]
pub struct PreRollConfig {
    /// URI of the pre-roll segment file.
    pub segment_uri: String,
    /// Duration of the pre-roll segment in seconds.
    pub duration_secs: f64,
    /// Whether the viewer can skip the pre-roll.
    pub is_skippable: bool,
}

// ---------------------------------------------------------------------------
// BumperConfig
// ---------------------------------------------------------------------------

/// Configuration for a bumper (end card) segment.
#[derive(Debug, Clone)]
pub struct BumperConfig {
    /// URI of the bumper segment file.
    pub segment_uri: String,
    /// Duration of the bumper in seconds.
    pub duration_secs: f64,
}

// ---------------------------------------------------------------------------
// SegmentEntry
// ---------------------------------------------------------------------------

/// A single entry in the injected segment list.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentEntry {
    /// URI of the segment.
    pub uri: String,
    /// Duration in seconds.
    pub duration_secs: f64,
    /// Whether this is an injected segment (pre-roll or bumper) vs content.
    pub is_injected: bool,
    /// Tag for injected segment type.
    pub injection_tag: Option<InjectionTag>,
}

/// Tag identifying the type of an injected segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InjectionTag {
    /// Pre-roll segment.
    PreRoll,
    /// Bumper (end card) segment.
    Bumper,
}

// ---------------------------------------------------------------------------
// PreRollInjector
// ---------------------------------------------------------------------------

/// Injects pre-roll and bumper segments into a packaging segment list.
#[derive(Debug, Default)]
pub struct PreRollInjector {
    pre_roll: Option<PreRollConfig>,
    bumper: Option<BumperConfig>,
}

impl PreRollInjector {
    /// Create an injector with no configured injections.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the pre-roll configuration.
    #[must_use]
    pub fn with_pre_roll(mut self, config: PreRollConfig) -> Self {
        self.pre_roll = Some(config);
        self
    }

    /// Set the bumper configuration.
    #[must_use]
    pub fn with_bumper(mut self, config: BumperConfig) -> Self {
        self.bumper = Some(config);
        self
    }

    /// Inject segments into `content`, returning a new ordered `Vec<SegmentEntry>`.
    ///
    /// Pre-roll (if configured) is prepended; bumper (if configured) is appended.
    #[must_use]
    pub fn inject(
        &self,
        content: &[SegmentEntry],
    ) -> Vec<SegmentEntry> {
        let mut result: Vec<SegmentEntry> = Vec::with_capacity(
            content.len()
                + usize::from(self.pre_roll.is_some())
                + usize::from(self.bumper.is_some()),
        );

        if let Some(ref pre) = self.pre_roll {
            result.push(SegmentEntry {
                uri: pre.segment_uri.clone(),
                duration_secs: pre.duration_secs,
                is_injected: true,
                injection_tag: Some(InjectionTag::PreRoll),
            });
        }

        result.extend_from_slice(content);

        if let Some(ref bump) = self.bumper {
            result.push(SegmentEntry {
                uri: bump.segment_uri.clone(),
                duration_secs: bump.duration_secs,
                is_injected: true,
                injection_tag: Some(InjectionTag::Bumper),
            });
        }

        result
    }

    /// Convenience: inject pre-roll / bumper into a simple `&[&str]` URI list
    /// with a default duration of 0.0 seconds for content segments.
    #[must_use]
    pub fn inject_uris<S: AsRef<str>>(&self, uris: &[S]) -> Vec<String> {
        let entries: Vec<SegmentEntry> = uris
            .iter()
            .map(|u| SegmentEntry {
                uri: u.as_ref().to_string(),
                duration_secs: 0.0,
                is_injected: false,
                injection_tag: None,
            })
            .collect();
        self.inject(&entries)
            .into_iter()
            .map(|e| e.uri)
            .collect()
    }

    /// Generate an HLS `#EXTINF` + URI block for all injected entries.
    ///
    /// This is a minimal representation suitable for insertion into an `.m3u8`
    /// playlist before writing full HLS manifest entries.
    #[must_use]
    pub fn to_hls_extinf_block(&self, content: &[SegmentEntry]) -> String {
        let mut out = String::new();
        for entry in self.inject(content) {
            out.push_str(&format!("#EXTINF:{:.3},\n", entry.duration_secs));
            out.push_str(&entry.uri);
            out.push('\n');
        }
        out
    }

    /// Total injected pre-roll duration in seconds (0.0 if no pre-roll).
    #[must_use]
    pub fn pre_roll_duration_secs(&self) -> f64 {
        self.pre_roll
            .as_ref()
            .map_or(0.0, |p| p.duration_secs)
    }

    /// Total bumper duration in seconds (0.0 if no bumper).
    #[must_use]
    pub fn bumper_duration_secs(&self) -> f64 {
        self.bumper.as_ref().map_or(0.0, |b| b.duration_secs)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_content(uris: &[&str]) -> Vec<SegmentEntry> {
        uris.iter()
            .map(|u| SegmentEntry {
                uri: u.to_string(),
                duration_secs: 6.0,
                is_injected: false,
                injection_tag: None,
            })
            .collect()
    }

    #[test]
    fn test_inject_pre_roll_prepended() {
        let pre = PreRollConfig {
            segment_uri: "preroll.ts".to_string(),
            duration_secs: 5.0,
            is_skippable: false,
        };
        let injector = PreRollInjector::new().with_pre_roll(pre);
        let content = make_content(&["seg0.ts", "seg1.ts"]);
        let result = injector.inject(&content);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].uri, "preroll.ts");
        assert_eq!(result[0].injection_tag, Some(InjectionTag::PreRoll));
        assert_eq!(result[1].uri, "seg0.ts");
        assert!(!result[1].is_injected);
    }

    #[test]
    fn test_inject_bumper_appended() {
        let bumper = BumperConfig {
            segment_uri: "bumper.ts".to_string(),
            duration_secs: 3.0,
        };
        let injector = PreRollInjector::new().with_bumper(bumper);
        let content = make_content(&["seg0.ts", "seg1.ts"]);
        let result = injector.inject(&content);
        assert_eq!(result.len(), 3);
        assert_eq!(result[2].uri, "bumper.ts");
        assert_eq!(result[2].injection_tag, Some(InjectionTag::Bumper));
    }

    #[test]
    fn test_inject_both_pre_roll_and_bumper() {
        let pre = PreRollConfig {
            segment_uri: "pre.ts".to_string(),
            duration_secs: 4.0,
            is_skippable: true,
        };
        let bumper = BumperConfig {
            segment_uri: "bump.ts".to_string(),
            duration_secs: 2.0,
        };
        let injector = PreRollInjector::new()
            .with_pre_roll(pre)
            .with_bumper(bumper);
        let content = make_content(&["a.ts", "b.ts", "c.ts"]);
        let result = injector.inject(&content);
        assert_eq!(result.len(), 5);
        assert_eq!(result[0].uri, "pre.ts");
        assert_eq!(result[4].uri, "bump.ts");
    }

    #[test]
    fn test_inject_no_injections() {
        let injector = PreRollInjector::new();
        let content = make_content(&["seg0.ts"]);
        let result = injector.inject(&content);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].uri, "seg0.ts");
    }

    #[test]
    fn test_inject_uris() {
        let pre = PreRollConfig {
            segment_uri: "preroll.ts".to_string(),
            duration_secs: 5.0,
            is_skippable: false,
        };
        let injector = PreRollInjector::new().with_pre_roll(pre);
        let uris = ["seg0.ts", "seg1.ts"];
        let result = injector.inject_uris(&uris);
        assert_eq!(result[0], "preroll.ts");
        assert_eq!(result[1], "seg0.ts");
    }

    #[test]
    fn test_to_hls_extinf_block() {
        let pre = PreRollConfig {
            segment_uri: "pre.ts".to_string(),
            duration_secs: 5.0,
            is_skippable: false,
        };
        let injector = PreRollInjector::new().with_pre_roll(pre);
        let content = make_content(&["seg.ts"]);
        let block = injector.to_hls_extinf_block(&content);
        assert!(block.contains("#EXTINF:5.000,"), "block should have pre-roll EXTINF");
        assert!(block.contains("pre.ts"), "block should have pre-roll URI");
        assert!(block.contains("seg.ts"), "block should have content URI");
    }

    #[test]
    fn test_pre_roll_duration_secs() {
        let injector = PreRollInjector::new().with_pre_roll(PreRollConfig {
            segment_uri: "p.ts".to_string(),
            duration_secs: 7.5,
            is_skippable: false,
        });
        assert!((injector.pre_roll_duration_secs() - 7.5).abs() < 1e-9);
        assert!((injector.bumper_duration_secs()).abs() < 1e-9);
    }
}
