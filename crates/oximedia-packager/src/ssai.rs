//! Server-Side Ad Insertion (SSAI) markers and manifest manipulation.
//!
//! SSAI enables the packaging layer to embed ad break markers directly into
//! adaptive streaming manifests (HLS M3U8 and DASH MPD), allowing the CDN
//! or origin server to stitch ad content seamlessly without requiring
//! client-side ad decision logic.
//!
//! # HLS Ad Markers
//!
//! HLS uses `#EXT-X-CUE-OUT` / `#EXT-X-CUE-IN` and `#EXT-X-DATERANGE` tags
//! (per SCTE-35 mapping) to signal ad opportunity boundaries.
//!
//! # DASH Ad Markers
//!
//! DASH uses `<EventStream>` inside a `<Period>` with scheme URI
//! `urn:scte:scte35:2014:xml+bin` or `urn:scte:scte35:2013:bin`.
//!
//! # Example
//!
//! ```rust
//! use oximedia_packager::ssai::{AdBreak, AdBreakType, SsaiMarkerWriter};
//!
//! let breaks = vec![
//!     AdBreak { offset_secs: 60.0, duration_secs: 30.0, break_id: "ad1".to_string(), break_type: AdBreakType::PreRoll },
//!     AdBreak { offset_secs: 300.0, duration_secs: 60.0, break_id: "ad2".to_string(), break_type: AdBreakType::MidRoll },
//! ];
//! let writer = SsaiMarkerWriter::new(breaks);
//! let hls = writer.hls_cue_markers();
//! assert!(hls.contains("EXT-X-CUE-OUT"), "HLS markers should include CUE-OUT");
//! ```

// ---------------------------------------------------------------------------
// AdBreakType
// ---------------------------------------------------------------------------

/// Classification of an ad break position relative to content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdBreakType {
    /// Ad plays before the main content begins.
    PreRoll,
    /// Ad plays at a mid-point within the main content.
    MidRoll,
    /// Ad plays after the main content finishes.
    PostRoll,
    /// A short overlay bumper (lower-third / interstitial).
    Overlay,
}

impl AdBreakType {
    /// SCTE-35 `avail_descriptor` type label.
    #[must_use]
    pub fn scte35_label(self) -> &'static str {
        match self {
            Self::PreRoll => "pre-roll",
            Self::MidRoll => "mid-roll",
            Self::PostRoll => "post-roll",
            Self::Overlay => "overlay",
        }
    }
}

// ---------------------------------------------------------------------------
// AdBreak
// ---------------------------------------------------------------------------

/// A single ad break in the content timeline.
#[derive(Debug, Clone)]
pub struct AdBreak {
    /// Break start position from the beginning of the content (seconds).
    pub offset_secs: f64,
    /// Expected duration of the ad break (seconds).
    pub duration_secs: f64,
    /// Unique identifier for this break (used in EXT-X-DATERANGE ID=).
    pub break_id: String,
    /// Classification of the break.
    pub break_type: AdBreakType,
}

impl AdBreak {
    /// Create a new mid-roll ad break.
    #[must_use]
    pub fn mid_roll(id: &str, offset_secs: f64, duration_secs: f64) -> Self {
        Self {
            offset_secs,
            duration_secs,
            break_id: id.to_string(),
            break_type: AdBreakType::MidRoll,
        }
    }

    /// Create a pre-roll ad break (offset = 0).
    #[must_use]
    pub fn pre_roll(id: &str, duration_secs: f64) -> Self {
        Self {
            offset_secs: 0.0,
            duration_secs,
            break_id: id.to_string(),
            break_type: AdBreakType::PreRoll,
        }
    }

    /// Create a post-roll ad break.
    #[must_use]
    pub fn post_roll(id: &str, content_duration_secs: f64, duration_secs: f64) -> Self {
        Self {
            offset_secs: content_duration_secs,
            duration_secs,
            break_id: id.to_string(),
            break_type: AdBreakType::PostRoll,
        }
    }
}

// ---------------------------------------------------------------------------
// SsaiMarkerWriter
// ---------------------------------------------------------------------------

/// Writes SSAI ad markers for HLS and DASH manifests.
pub struct SsaiMarkerWriter {
    breaks: Vec<AdBreak>,
}

impl SsaiMarkerWriter {
    /// Create a new writer for the given ad breaks.
    ///
    /// Breaks are sorted by `offset_secs` on construction.
    #[must_use]
    pub fn new(mut breaks: Vec<AdBreak>) -> Self {
        breaks.sort_by(|a, b| a.offset_secs.partial_cmp(&b.offset_secs).unwrap_or(std::cmp::Ordering::Equal));
        Self { breaks }
    }

    /// Generate HLS `#EXT-X-CUE-OUT` / `#EXT-X-CUE-IN` marker lines.
    ///
    /// Returns one string per ad break, with `CUE-OUT:duration` followed by
    /// `CUE-IN` on separate lines, ready for insertion at the appropriate
    /// segment boundary in an `.m3u8` playlist.
    #[must_use]
    pub fn hls_cue_markers(&self) -> String {
        let mut out = String::new();
        for brk in &self.breaks {
            out.push_str(&format!(
                "# Break: {} ({}) at {:.3}s\n",
                brk.break_id,
                brk.break_type.scte35_label(),
                brk.offset_secs,
            ));
            out.push_str(&format!("#EXT-X-CUE-OUT:{:.3}\n", brk.duration_secs));
            out.push_str("#EXT-X-CUE-IN\n");
        }
        out
    }

    /// Generate HLS `#EXT-X-DATERANGE` tags for ad breaks.
    ///
    /// Uses `CLASS="com.apple.hls.interstitial"` for Apple interstitial
    /// compatibility and includes `PLANNED-DURATION` and `X-AD-TYPE` attributes.
    #[must_use]
    pub fn hls_daterange_tags(&self, programme_start_utc: &str) -> String {
        let mut out = String::new();
        for brk in &self.breaks {
            let start = format!(
                "{}.{:03}Z",
                programme_start_utc.trim_end_matches('Z'),
                (brk.offset_secs * 1_000.0) as u64,
            );
            out.push_str(&format!(
                "#EXT-X-DATERANGE:ID=\"{}\",CLASS=\"com.apple.hls.interstitial\",\
                 START-DATE=\"{}\",PLANNED-DURATION={:.3},\
                 X-AD-TYPE=\"{}\"\n",
                brk.break_id,
                start,
                brk.duration_secs,
                brk.break_type.scte35_label(),
            ));
        }
        out
    }

    /// Generate a DASH `<EventStream>` XML fragment for SCTE-35 signalling.
    ///
    /// Uses `urn:scte:scte35:2014:xml+bin` as the scheme URI.
    #[must_use]
    pub fn dash_event_stream(&self, timescale: u32) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "<EventStream schemeIdUri=\"urn:scte:scte35:2014:xml+bin\" timescale=\"{timescale}\">\n"
        ));
        for brk in &self.breaks {
            let presentation_time = (brk.offset_secs * timescale as f64) as u64;
            let duration = (brk.duration_secs * timescale as f64) as u64;
            out.push_str(&format!(
                "  <Event id=\"{}\" presentationTime=\"{}\" duration=\"{}\">\n",
                brk.break_id, presentation_time, duration,
            ));
            out.push_str(&format!(
                "    <!-- {} ad break, type={} -->\n",
                brk.break_type.scte35_label(),
                brk.break_type.scte35_label(),
            ));
            out.push_str("  </Event>\n");
        }
        out.push_str("</EventStream>");
        out
    }

    /// Return the total number of ad breaks.
    #[must_use]
    pub fn break_count(&self) -> usize {
        self.breaks.len()
    }

    /// Return the total ad time across all breaks (seconds).
    #[must_use]
    pub fn total_ad_duration_secs(&self) -> f64 {
        self.breaks.iter().map(|b| b.duration_secs).sum()
    }

    /// Return ad breaks that fall within a given time window `[start, end)`.
    #[must_use]
    pub fn breaks_in_range(&self, start_secs: f64, end_secs: f64) -> Vec<&AdBreak> {
        self.breaks
            .iter()
            .filter(|b| b.offset_secs >= start_secs && b.offset_secs < end_secs)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_breaks() -> Vec<AdBreak> {
        vec![
            AdBreak::pre_roll("ad0", 15.0),
            AdBreak::mid_roll("ad1", 60.0, 30.0),
            AdBreak::mid_roll("ad2", 180.0, 60.0),
            AdBreak::post_roll("ad3", 300.0, 20.0),
        ]
    }

    #[test]
    fn test_break_count() {
        let writer = SsaiMarkerWriter::new(sample_breaks());
        assert_eq!(writer.break_count(), 4);
    }

    #[test]
    fn test_total_ad_duration() {
        let writer = SsaiMarkerWriter::new(sample_breaks());
        let total = writer.total_ad_duration_secs();
        assert!((total - 125.0).abs() < 1e-9, "expected 125s total, got {total}");
    }

    #[test]
    fn test_hls_cue_markers_contains_cue_out() {
        let writer = SsaiMarkerWriter::new(sample_breaks());
        let hls = writer.hls_cue_markers();
        assert!(hls.contains("EXT-X-CUE-OUT"), "HLS output must contain CUE-OUT tag");
        assert!(hls.contains("EXT-X-CUE-IN"), "HLS output must contain CUE-IN tag");
    }

    #[test]
    fn test_hls_cue_markers_duration() {
        let writer = SsaiMarkerWriter::new(vec![AdBreak::mid_roll("m1", 60.0, 30.0)]);
        let hls = writer.hls_cue_markers();
        assert!(hls.contains("30.000"), "CUE-OUT should include 30-second duration");
    }

    #[test]
    fn test_hls_daterange_tags_id() {
        let writer = SsaiMarkerWriter::new(vec![AdBreak::mid_roll("mid1", 60.0, 30.0)]);
        let tags = writer.hls_daterange_tags("2024-01-01T00:00:00");
        assert!(tags.contains("mid1"), "DATERANGE tag should include break ID");
        assert!(tags.contains("EXT-X-DATERANGE"), "should include DATERANGE tag");
    }

    #[test]
    fn test_dash_event_stream_structure() {
        let writer = SsaiMarkerWriter::new(vec![AdBreak::mid_roll("ev1", 60.0, 30.0)]);
        let xml = writer.dash_event_stream(90_000);
        assert!(xml.contains("EventStream"), "DASH output should have EventStream element");
        assert!(xml.contains("ev1"), "DASH output should contain break ID");
        assert!(xml.contains("presentationTime"), "DASH output should have presentationTime");
    }

    #[test]
    fn test_breaks_in_range() {
        let writer = SsaiMarkerWriter::new(sample_breaks());
        let in_range = writer.breaks_in_range(50.0, 200.0);
        assert_eq!(in_range.len(), 2, "should find ad1 and ad2 in range [50, 200)");
        assert_eq!(in_range[0].break_id, "ad1");
        assert_eq!(in_range[1].break_id, "ad2");
    }

    #[test]
    fn test_breaks_sorted_by_offset() {
        let unsorted = vec![
            AdBreak::mid_roll("c", 300.0, 30.0),
            AdBreak::mid_roll("a", 60.0, 30.0),
            AdBreak::mid_roll("b", 180.0, 30.0),
        ];
        let writer = SsaiMarkerWriter::new(unsorted);
        let offsets: Vec<f64> = writer.breaks.iter().map(|b| b.offset_secs).collect();
        assert_eq!(offsets, vec![60.0, 180.0, 300.0], "breaks should be sorted by offset");
    }

    #[test]
    fn test_empty_breaks() {
        let writer = SsaiMarkerWriter::new(vec![]);
        assert_eq!(writer.break_count(), 0);
        assert!((writer.total_ad_duration_secs()).abs() < 1e-9);
        assert!(writer.hls_cue_markers().is_empty());
    }

    #[test]
    fn test_ad_break_type_labels() {
        assert_eq!(AdBreakType::PreRoll.scte35_label(), "pre-roll");
        assert_eq!(AdBreakType::MidRoll.scte35_label(), "mid-roll");
        assert_eq!(AdBreakType::PostRoll.scte35_label(), "post-roll");
        assert_eq!(AdBreakType::Overlay.scte35_label(), "overlay");
    }
}
