//! Low-Latency DASH (LL-DASH) with CMAF chunked transfer encoding.
//!
//! This module provides the high-level LL-DASH API conforming to
//! ISO/IEC 23009-1:2022 Annex K, including:
//!
//! - [`LlDashConfig`] — millisecond-granularity latency configuration
//! - [`DashRepresentation`] — video/audio representation descriptor
//! - [`LlDashMpd`] — live DASH Media Presentation Description (MPD)
//!   with `availabilityTimeComplete="false"` and `availabilityTimeOffset`
//!
//! # LL-DASH over CMAF
//!
//! In LL-DASH, a DASH Adaptation Set uses a `SegmentTemplate` with
//! `availabilityTimeOffset` set to `(segment_duration - fragment_duration)`.
//! The MPD's `type="dynamic"` and `minimumUpdatePeriod` drive clients to
//! poll for new segments.
//!
//! # Example
//!
//! ```
//! use oximedia_net::ll_dash::{DashRepresentation, LlDashConfig, LlDashMpd};
//!
//! let config = LlDashConfig {
//!     segment_duration_ms: 2000,
//!     fragment_duration_ms: 500,
//!     availability_time_offset_sec: 1.5,
//! };
//!
//! let rep = DashRepresentation {
//!     id: "video_1080p".to_owned(),
//!     bandwidth: 4_000_000,
//!     codecs: "avc1.640028".to_owned(),
//!     width: 1920,
//!     height: 1080,
//! };
//!
//! let mpd = LlDashMpd {
//!     config,
//!     representations: vec![rep],
//! };
//!
//! let xml = mpd.build_mpd_xml();
//! assert!(xml.contains("availabilityTimeComplete=\"false\""));
//! assert!(xml.contains("availabilityTimeOffset"));
//! ```

use std::fmt::Write as FmtWrite;

// ─── LlDashConfig ─────────────────────────────────────────────────────────────

/// LL-DASH timing and latency configuration.
///
/// All three time fields control how early clients can request segment data
/// before the full segment duration elapses.
#[derive(Debug, Clone)]
pub struct LlDashConfig {
    /// Duration of a full DASH segment in milliseconds (e.g., 2000 ms).
    pub segment_duration_ms: u32,
    /// Duration of each CMAF chunk / fragment in milliseconds (e.g., 500 ms).
    ///
    /// Must divide `segment_duration_ms` evenly.
    pub fragment_duration_ms: u32,
    /// `availabilityTimeOffset` value in seconds.
    ///
    /// Indicates how many seconds before a segment's nominal availability
    /// time its first fragment is accessible.  Typically set to
    /// `(segment_duration_ms - fragment_duration_ms) / 1000`.
    pub availability_time_offset_sec: f32,
}

impl Default for LlDashConfig {
    /// Standard LL-DASH configuration: 2-second segments, 500 ms fragments.
    fn default() -> Self {
        Self {
            segment_duration_ms: 2000,
            fragment_duration_ms: 500,
            availability_time_offset_sec: 1.5,
        }
    }
}

impl LlDashConfig {
    /// Returns the `availability_time_offset_sec` derived from the segment
    /// and fragment durations: `(segment_duration_ms - fragment_duration_ms) / 1000`.
    #[must_use]
    pub fn derived_ato(&self) -> f32 {
        let diff = self
            .segment_duration_ms
            .saturating_sub(self.fragment_duration_ms);
        diff as f32 / 1000.0
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - `fragment_duration_ms` is zero or exceeds `segment_duration_ms`
    /// - `fragment_duration_ms` does not evenly divide `segment_duration_ms`
    pub fn validate(&self) -> Result<(), String> {
        if self.fragment_duration_ms == 0 {
            return Err("fragment_duration_ms must be > 0".to_owned());
        }
        if self.fragment_duration_ms > self.segment_duration_ms {
            return Err(format!(
                "fragment_duration_ms ({}) must not exceed segment_duration_ms ({})",
                self.fragment_duration_ms, self.segment_duration_ms
            ));
        }
        if self.segment_duration_ms % self.fragment_duration_ms != 0 {
            return Err(format!(
                "fragment_duration_ms ({}) must evenly divide segment_duration_ms ({})",
                self.fragment_duration_ms, self.segment_duration_ms
            ));
        }
        Ok(())
    }

    /// Number of fragments per segment.
    #[must_use]
    pub fn fragments_per_segment(&self) -> u32 {
        if self.fragment_duration_ms == 0 {
            return 1;
        }
        self.segment_duration_ms / self.fragment_duration_ms
    }

    /// Returns the segment duration in seconds as a float.
    #[must_use]
    pub fn segment_duration_secs(&self) -> f64 {
        self.segment_duration_ms as f64 / 1000.0
    }

    /// Returns the fragment duration in seconds as a float.
    #[must_use]
    pub fn fragment_duration_secs(&self) -> f64 {
        self.fragment_duration_ms as f64 / 1000.0
    }
}

// ─── DashRepresentation ───────────────────────────────────────────────────────

/// A DASH Representation descriptor for inclusion in an Adaptation Set.
///
/// Each representation corresponds to a distinct quality level
/// (e.g., 1080p @ 4 Mbps, 720p @ 2 Mbps).
#[derive(Debug, Clone)]
pub struct DashRepresentation {
    /// Unique representation ID (e.g., `"video_1080p"` or `"1"`).
    pub id: String,
    /// Encoded bandwidth in bits per second.
    pub bandwidth: u32,
    /// RFC 6381 codec string (e.g., `"avc1.640028"` or `"av01.0.08M.08"`).
    pub codecs: String,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

impl DashRepresentation {
    /// Creates a representation for AVC High Profile at the given resolution and bitrate.
    #[must_use]
    pub fn avc_high(id: impl Into<String>, width: u32, height: u32, bandwidth: u32) -> Self {
        Self {
            id: id.into(),
            bandwidth,
            codecs: "avc1.640028".to_owned(),
            width,
            height,
        }
    }

    /// Creates a representation for AV1 Main Profile at the given resolution and bitrate.
    #[must_use]
    pub fn av1(id: impl Into<String>, width: u32, height: u32, bandwidth: u32) -> Self {
        Self {
            id: id.into(),
            bandwidth,
            codecs: "av01.0.08M.08".to_owned(),
            width,
            height,
        }
    }

    /// Renders the `<Representation>` XML element.
    #[must_use]
    pub fn to_xml(&self) -> String {
        format!(
            "      <Representation id=\"{}\" bandwidth=\"{}\" codecs=\"{}\" width=\"{}\" height=\"{}\"/>",
            self.id, self.bandwidth, self.codecs, self.width, self.height
        )
    }
}

// ─── LlDashMpd ────────────────────────────────────────────────────────────────

/// A live LL-DASH Media Presentation Description (MPD).
///
/// Generates an XML MPD with:
/// - `type="dynamic"` (live stream)
/// - `availabilityTimeComplete="false"` on the `SegmentTemplate`
/// - `availabilityTimeOffset` from [`LlDashConfig::availability_time_offset_sec`]
/// - A `<ServiceDescription>` with `<Latency>` element
/// - One `<AdaptationSet>` for video with all [`DashRepresentation`]s
pub struct LlDashMpd {
    /// LL-DASH timing configuration.
    pub config: LlDashConfig,
    /// Video representations to include in the MPD.
    pub representations: Vec<DashRepresentation>,
}

impl LlDashMpd {
    /// Creates a new MPD with the given configuration.
    #[must_use]
    pub fn new(config: LlDashConfig) -> Self {
        Self {
            config,
            representations: Vec::new(),
        }
    }

    /// Adds a representation to this MPD.
    pub fn add_representation(&mut self, rep: DashRepresentation) {
        self.representations.push(rep);
    }

    /// Build the LL-DASH MPD XML string.
    ///
    /// Includes all mandatory LL-DASH attributes per ISO/IEC 23009-1:2022:
    ///
    /// - `type="dynamic"` on the root `<MPD>` element
    /// - `<ServiceDescription>` / `<Latency>` for client latency guidance
    /// - `availabilityTimeComplete="false"` on `<SegmentTemplate>` signals
    ///   that segment data is available before `segment_duration` elapses
    /// - `availabilityTimeOffset` tells clients how many seconds early to
    ///   request segment data
    #[must_use]
    pub fn build_mpd_xml(&self) -> String {
        let mut xml = String::with_capacity(2048);

        let seg_dur = self.config.segment_duration_secs();
        let frag_dur = self.config.fragment_duration_secs();
        let ato = self.config.availability_time_offset_sec;

        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<MPD xmlns=\"urn:mpeg:dash:schema:mpd:2011\"\n");
        xml.push_str("     xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n");
        xml.push_str("     type=\"dynamic\"\n");
        let _ = writeln!(xml, "     minimumUpdatePeriod=\"PT{frag_dur:.3}S\"");
        let _ = writeln!(xml, "     minBufferTime=\"PT{seg_dur:.3}S\"");
        xml.push_str(
            "     profiles=\"urn:mpeg:dash:profile:isoff-live:2011,urn:mpeg:dash:profile:cmaf:2019\">\n",
        );

        // ServiceDescription with latency guidance
        let target_latency_ms = (seg_dur * 1.5 * 1000.0) as u32;
        xml.push_str("  <ServiceDescription id=\"0\">\n");
        let _ = writeln!(
            xml,
            "    <Latency target=\"{target_latency_ms}\" min=\"{}\" max=\"{}\"/>",
            (target_latency_ms as f64 * 0.5) as u32,
            target_latency_ms * 3
        );
        xml.push_str("    <PlaybackRate min=\"0.96\" max=\"1.04\"/>\n");
        xml.push_str("  </ServiceDescription>\n");

        // Period and AdaptationSet
        xml.push_str("  <Period id=\"0\" start=\"PT0S\">\n");
        xml.push_str("    <AdaptationSet mimeType=\"video/mp4\" contentType=\"video\">\n");

        // SegmentTemplate with availabilityTimeComplete=false and ato
        let _ = writeln!(
            xml,
            "      <SegmentTemplate timescale=\"90000\"\n         media=\"segment_$Number$.m4s\"\n         initialization=\"init.mp4\"\n         duration=\"{}\"\n         availabilityTimeComplete=\"false\"\n         availabilityTimeOffset=\"{ato:.3}\">",
            (seg_dur * 90000.0) as u64
        );
        xml.push_str("        <SegmentTimeline/>\n");
        xml.push_str("      </SegmentTemplate>\n");

        // Representations
        for rep in &self.representations {
            let _ = writeln!(xml, "{}", rep.to_xml());
        }

        xml.push_str("    </AdaptationSet>\n");
        xml.push_str("  </Period>\n");
        xml.push_str("</MPD>\n");

        xml
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> LlDashConfig {
        LlDashConfig::default()
    }

    fn make_rep() -> DashRepresentation {
        DashRepresentation::avc_high("1080p", 1920, 1080, 4_000_000)
    }

    // 1. Default config has expected values
    #[test]
    fn test_default_config_values() {
        let cfg = default_config();
        assert_eq!(cfg.segment_duration_ms, 2000);
        assert_eq!(cfg.fragment_duration_ms, 500);
        assert!((cfg.availability_time_offset_sec - 1.5).abs() < 1e-6);
    }

    // 2. validate passes for default config
    #[test]
    fn test_validate_default_ok() {
        assert!(default_config().validate().is_ok());
    }

    // 3. validate rejects zero fragment_duration
    #[test]
    fn test_validate_zero_fragment() {
        let mut cfg = default_config();
        cfg.fragment_duration_ms = 0;
        assert!(cfg.validate().is_err());
    }

    // 4. validate rejects non-divisible fragment
    #[test]
    fn test_validate_non_divisible() {
        let mut cfg = default_config();
        cfg.fragment_duration_ms = 300;
        assert!(cfg.validate().is_err());
    }

    // 5. validate rejects fragment > segment
    #[test]
    fn test_validate_fragment_exceeds_segment() {
        let mut cfg = default_config();
        cfg.fragment_duration_ms = 3000;
        assert!(cfg.validate().is_err());
    }

    // 6. fragments_per_segment correct
    #[test]
    fn test_fragments_per_segment() {
        let cfg = default_config(); // 2000 / 500 = 4
        assert_eq!(cfg.fragments_per_segment(), 4);
    }

    // 7. derived_ato matches (seg - frag) / 1000
    #[test]
    fn test_derived_ato() {
        let cfg = default_config(); // (2000 - 500) / 1000 = 1.5
        assert!((cfg.derived_ato() - 1.5_f32).abs() < 1e-6);
    }

    // 8. DashRepresentation::avc_high sets codecs
    #[test]
    fn test_avc_high_codecs() {
        let rep = DashRepresentation::avc_high("r1", 1920, 1080, 4_000_000);
        assert_eq!(rep.codecs, "avc1.640028");
        assert_eq!(rep.width, 1920);
        assert_eq!(rep.height, 1080);
        assert_eq!(rep.bandwidth, 4_000_000);
    }

    // 9. DashRepresentation::av1 sets codecs
    #[test]
    fn test_av1_codecs() {
        let rep = DashRepresentation::av1("r2", 1280, 720, 2_000_000);
        assert!(rep.codecs.starts_with("av01"));
    }

    // 10. DashRepresentation::to_xml contains id and bandwidth
    #[test]
    fn test_representation_to_xml() {
        let rep = make_rep();
        let xml = rep.to_xml();
        assert!(xml.contains("id=\"1080p\""));
        assert!(xml.contains("bandwidth=\"4000000\""));
        assert!(xml.contains("codecs=\"avc1.640028\""));
    }

    // 11. LlDashMpd::build_mpd_xml contains MPD opening
    #[test]
    fn test_mpd_xml_opening() {
        let mpd = LlDashMpd {
            config: default_config(),
            representations: vec![make_rep()],
        };
        let xml = mpd.build_mpd_xml();
        assert!(xml.contains("<MPD"));
        assert!(xml.contains("type=\"dynamic\""));
    }

    // 12. build_mpd_xml contains availabilityTimeComplete="false"
    #[test]
    fn test_mpd_xml_atc_false() {
        let mpd = LlDashMpd {
            config: default_config(),
            representations: vec![],
        };
        let xml = mpd.build_mpd_xml();
        assert!(xml.contains("availabilityTimeComplete=\"false\""));
    }

    // 13. build_mpd_xml contains availabilityTimeOffset
    #[test]
    fn test_mpd_xml_ato() {
        let cfg = LlDashConfig {
            segment_duration_ms: 2000,
            fragment_duration_ms: 500,
            availability_time_offset_sec: 1.5,
        };
        let mpd = LlDashMpd {
            config: cfg,
            representations: vec![],
        };
        let xml = mpd.build_mpd_xml();
        assert!(xml.contains("availabilityTimeOffset=\"1.500\""));
    }

    // 14. build_mpd_xml contains ServiceDescription
    #[test]
    fn test_mpd_xml_service_description() {
        let mpd = LlDashMpd {
            config: default_config(),
            representations: vec![],
        };
        let xml = mpd.build_mpd_xml();
        assert!(xml.contains("<ServiceDescription"));
        assert!(xml.contains("<Latency"));
    }

    // 15. build_mpd_xml contains all representations
    #[test]
    fn test_mpd_xml_representations() {
        let mut mpd = LlDashMpd::new(default_config());
        mpd.add_representation(DashRepresentation::avc_high("1080p", 1920, 1080, 4_000_000));
        mpd.add_representation(DashRepresentation::avc_high("720p", 1280, 720, 2_000_000));
        let xml = mpd.build_mpd_xml();
        assert!(xml.contains("id=\"1080p\""));
        assert!(xml.contains("id=\"720p\""));
    }

    // 16. add_representation appends correctly
    #[test]
    fn test_add_representation() {
        let mut mpd = LlDashMpd::new(default_config());
        assert_eq!(mpd.representations.len(), 0);
        mpd.add_representation(make_rep());
        assert_eq!(mpd.representations.len(), 1);
    }

    // 17. segment_duration_secs is correct
    #[test]
    fn test_segment_duration_secs() {
        let cfg = default_config();
        assert!((cfg.segment_duration_secs() - 2.0).abs() < 1e-9);
    }

    // 18. fragment_duration_secs is correct
    #[test]
    fn test_fragment_duration_secs() {
        let cfg = default_config();
        assert!((cfg.fragment_duration_secs() - 0.5).abs() < 1e-9);
    }

    // 19. MPD XML is valid UTF-8 and closes MPD element
    #[test]
    fn test_mpd_xml_closes_mpd_element() {
        let mpd = LlDashMpd {
            config: default_config(),
            representations: vec![make_rep()],
        };
        let xml = mpd.build_mpd_xml();
        assert!(xml.ends_with("</MPD>\n"));
    }

    // 20. Ultra-low-latency config: 1s segment, 250ms fragment
    #[test]
    fn test_ull_config() {
        let cfg = LlDashConfig {
            segment_duration_ms: 1000,
            fragment_duration_ms: 250,
            availability_time_offset_sec: 0.75,
        };
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.fragments_per_segment(), 4);
        assert!((cfg.derived_ato() - 0.75_f32).abs() < 1e-5);
    }
}
