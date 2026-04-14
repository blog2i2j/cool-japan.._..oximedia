//! MPEG-DASH (Dynamic Adaptive Streaming over HTTP) segment template generation.
//!
//! Produces `AdaptationSet` and `SegmentTemplate` XML fragments for DASH MPD
//! manifests.  The implementation covers the most common single-period,
//! multi-adaptation-set use case: one video adaptation set and one audio
//! adaptation set, each with a single representation.
//!
//! # Example
//!
//! ```
//! use oximedia_container::streaming::dash::{DashManifest, AdaptationSetKind};
//!
//! let mut manifest = DashManifest::new("PT0H1M0S".to_string());
//! manifest.add_video_adaptation("1280x720", 2_500_000, "init-video.mp4", "chunk-video-$Number$.m4s");
//! manifest.add_audio_adaptation("stereo", 128_000, "init-audio.mp4", "chunk-audio-$Number$.m4s");
//!
//! let mpd = manifest.build_mpd();
//! assert!(mpd.contains("<MPD"));
//! assert!(mpd.contains("SegmentTemplate"));
//! ```

/// The kind of an adaptation set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptationSetKind {
    /// Video content.
    Video,
    /// Audio content.
    Audio,
    /// Subtitle or closed-caption content.
    Subtitle,
}

impl AdaptationSetKind {
    fn content_type(&self) -> &'static str {
        match self {
            Self::Video => "video",
            Self::Audio => "audio",
            Self::Subtitle => "text",
        }
    }

    fn mime_type(&self) -> &'static str {
        match self {
            Self::Video => "video/mp4",
            Self::Audio => "audio/mp4",
            Self::Subtitle => "application/mp4",
        }
    }
}

/// A single DASH representation within an adaptation set.
#[derive(Debug, Clone)]
pub struct DashRepresentation {
    /// Unique representation ID string.
    pub id: String,
    /// Bit rate in bits per second.
    pub bandwidth: u64,
    /// Optional codec string (e.g. `"av01.0.08M.08"`, `"opus"`).
    pub codecs: Option<String>,
    /// Initialisation segment URL.
    pub init_url: String,
    /// Media segment URL template (use `$Number$` for segment index).
    pub media_url: String,
    /// Segment duration in milliseconds.
    pub segment_duration_ms: u64,
    /// Optional video width (pixels).
    pub width: Option<u32>,
    /// Optional video height (pixels).
    pub height: Option<u32>,
    /// Optional audio sample rate (Hz).
    pub sample_rate: Option<u32>,
}

impl DashRepresentation {
    /// Create a minimal video representation.
    #[must_use]
    pub fn video(
        id: impl Into<String>,
        bandwidth: u64,
        init_url: impl Into<String>,
        media_url: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            bandwidth,
            codecs: None,
            init_url: init_url.into(),
            media_url: media_url.into(),
            segment_duration_ms: 4000,
            width: None,
            height: None,
            sample_rate: None,
        }
    }

    /// Create a minimal audio representation.
    #[must_use]
    pub fn audio(
        id: impl Into<String>,
        bandwidth: u64,
        init_url: impl Into<String>,
        media_url: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            bandwidth,
            codecs: None,
            init_url: init_url.into(),
            media_url: media_url.into(),
            segment_duration_ms: 4000,
            width: None,
            height: None,
            sample_rate: Some(48000),
        }
    }
}

/// A DASH adaptation set containing one or more representations.
#[derive(Debug, Clone)]
pub struct DashAdaptationSet {
    /// Content type of this adaptation set.
    pub kind: AdaptationSetKind,
    /// ISO 639-2 language tag (optional; e.g. `"eng"`).
    pub lang: Option<String>,
    /// Representations included in this adaptation set.
    pub representations: Vec<DashRepresentation>,
}

impl DashAdaptationSet {
    /// Create an adaptation set of the given kind.
    #[must_use]
    pub fn new(kind: AdaptationSetKind) -> Self {
        Self {
            kind,
            lang: None,
            representations: Vec::new(),
        }
    }

    /// Attach an optional language tag.
    #[must_use]
    pub fn with_lang(mut self, lang: impl Into<String>) -> Self {
        self.lang = Some(lang.into());
        self
    }

    /// Add a representation.
    pub fn add_representation(&mut self, rep: DashRepresentation) {
        self.representations.push(rep);
    }
}

/// Builder for a minimal MPEG-DASH MPD document.
#[derive(Debug, Clone, Default)]
pub struct DashManifest {
    /// ISO 8601 duration string for the total media duration (e.g.
    /// `"PT0H1M0S"`).
    pub media_duration: String,
    /// Adaptation sets in presentation order.
    pub adaptation_sets: Vec<DashAdaptationSet>,
    /// Optional minimum buffer time hint (ISO 8601, e.g. `"PT2S"`).
    pub min_buffer_time: Option<String>,
}

impl DashManifest {
    /// Create a new manifest with the given total media duration string.
    #[must_use]
    pub fn new(media_duration: impl Into<String>) -> Self {
        Self {
            media_duration: media_duration.into(),
            ..Default::default()
        }
    }

    /// Add a video adaptation set with a single representation.
    ///
    /// `resolution` is a human-readable string such as `"1280x720"` (stored
    /// as a representation ID).
    pub fn add_video_adaptation(
        &mut self,
        resolution: impl Into<String>,
        bandwidth: u64,
        init_url: impl Into<String>,
        media_url: impl Into<String>,
    ) {
        let mut aset = DashAdaptationSet::new(AdaptationSetKind::Video);
        aset.add_representation(DashRepresentation::video(
            resolution, bandwidth, init_url, media_url,
        ));
        self.adaptation_sets.push(aset);
    }

    /// Add an audio adaptation set with a single representation.
    pub fn add_audio_adaptation(
        &mut self,
        id: impl Into<String>,
        bandwidth: u64,
        init_url: impl Into<String>,
        media_url: impl Into<String>,
    ) {
        let mut aset = DashAdaptationSet::new(AdaptationSetKind::Audio);
        aset.add_representation(DashRepresentation::audio(
            id, bandwidth, init_url, media_url,
        ));
        self.adaptation_sets.push(aset);
    }

    /// Serialise to an MPEG-DASH MPD XML string.
    ///
    /// The produced document is a minimal but valid static (VOD) MPD.
    #[must_use]
    pub fn build_mpd(&self) -> String {
        let min_buf = self
            .min_buffer_time
            .as_deref()
            .unwrap_or("PT2S");

        let mut out = String::with_capacity(1024);
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str("<MPD\n");
        out.push_str("  xmlns=\"urn:mpeg:dash:schema:mpd:2011\"\n");
        out.push_str("  profiles=\"urn:mpeg:dash:profile:isoff-on-demand:2011\"\n");
        out.push_str("  type=\"static\"\n");
        out.push_str(&format!(
            "  mediaPresentationDuration=\"{}\"\n",
            self.media_duration
        ));
        out.push_str(&format!("  minBufferTime=\"{min_buf}\">\n"));
        out.push_str("  <Period start=\"PT0S\">\n");

        for (idx, aset) in self.adaptation_sets.iter().enumerate() {
            let content_type = aset.kind.content_type();
            let mime_type = aset.kind.mime_type();
            out.push_str(&format!(
                "    <AdaptationSet id=\"{idx}\" contentType=\"{content_type}\" mimeType=\"{mime_type}\""
            ));
            if let Some(ref lang) = aset.lang {
                out.push_str(&format!(" lang=\"{lang}\""));
            }
            out.push_str(">\n");

            for rep in &aset.representations {
                out.push_str(&format!(
                    "      <Representation id=\"{}\" bandwidth=\"{}\"",
                    rep.id, rep.bandwidth
                ));
                if let Some(ref codecs) = rep.codecs {
                    out.push_str(&format!(" codecs=\"{codecs}\""));
                }
                if let (Some(w), Some(h)) = (rep.width, rep.height) {
                    out.push_str(&format!(" width=\"{w}\" height=\"{h}\""));
                }
                if let Some(sr) = rep.sample_rate {
                    out.push_str(&format!(" audioSamplingRate=\"{sr}\""));
                }
                out.push_str(">\n");

                let dur_ms = rep.segment_duration_ms;
                out.push_str(&format!(
                    "        <SegmentTemplate\n          initialization=\"{}\"\n          media=\"{}\"\n          duration=\"{dur_ms}\"\n          timescale=\"1000\"\n          startNumber=\"1\"/>\n",
                    rep.init_url, rep.media_url
                ));

                out.push_str("      </Representation>\n");
            }

            out.push_str("    </AdaptationSet>\n");
        }

        out.push_str("  </Period>\n");
        out.push_str("</MPD>\n");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_manifest() {
        let manifest = DashManifest::new("PT0H0M30S");
        let mpd = manifest.build_mpd();
        assert!(mpd.contains("<MPD"));
        assert!(mpd.contains("mediaPresentationDuration=\"PT0H0M30S\""));
        assert!(mpd.contains("</MPD>"));
    }

    #[test]
    fn test_manifest_with_video_and_audio() {
        let mut manifest = DashManifest::new("PT0H1M0S".to_string());
        manifest.add_video_adaptation(
            "1280x720",
            2_500_000,
            "init-video.mp4",
            "chunk-video-$Number$.m4s",
        );
        manifest.add_audio_adaptation(
            "stereo",
            128_000,
            "init-audio.mp4",
            "chunk-audio-$Number$.m4s",
        );

        let mpd = manifest.build_mpd();
        assert!(mpd.contains("contentType=\"video\""));
        assert!(mpd.contains("contentType=\"audio\""));
        assert!(mpd.contains("SegmentTemplate"));
        assert!(mpd.contains("init-video.mp4"));
        assert!(mpd.contains("chunk-video-$Number$.m4s"));
        assert!(mpd.contains("bandwidth=\"2500000\""));
        assert!(mpd.contains("bandwidth=\"128000\""));
    }

    #[test]
    fn test_manifest_audio_sample_rate() {
        let mut manifest = DashManifest::new("PT30S");
        manifest.add_audio_adaptation("audio", 128_000, "init.mp4", "seg-$Number$.m4s");
        let mpd = manifest.build_mpd();
        assert!(mpd.contains("audioSamplingRate=\"48000\""));
    }

    #[test]
    fn test_adaptation_set_with_lang() {
        let mut aset = DashAdaptationSet::new(AdaptationSetKind::Audio)
            .with_lang("eng");
        aset.add_representation(DashRepresentation::audio(
            "audio-eng", 128_000, "init.mp4", "seg-$N$.m4s",
        ));
        let mut manifest = DashManifest::new("PT30S");
        manifest.adaptation_sets.push(aset);
        let mpd = manifest.build_mpd();
        assert!(mpd.contains("lang=\"eng\""));
    }

    #[test]
    fn test_representation_video_dimensions() {
        let mut rep = DashRepresentation::video("v1", 5_000_000, "init.mp4", "seg-$N$.m4s");
        rep.width = Some(1920);
        rep.height = Some(1080);
        let mut aset = DashAdaptationSet::new(AdaptationSetKind::Video);
        aset.add_representation(rep);
        let mut manifest = DashManifest::new("PT60S");
        manifest.adaptation_sets.push(aset);
        let mpd = manifest.build_mpd();
        assert!(mpd.contains("width=\"1920\""));
        assert!(mpd.contains("height=\"1080\""));
    }

    #[test]
    fn test_mpd_is_valid_xml_structure() {
        let mut manifest = DashManifest::new("PT30S");
        manifest.add_video_adaptation("720p", 2_000_000, "i.mp4", "s-$N$.m4s");
        let mpd = manifest.build_mpd();
        // Check nesting: Period is inside MPD, AdaptationSet inside Period
        let mpd_open = mpd.find("<MPD").expect("<MPD");
        let period_open = mpd.find("<Period").expect("<Period");
        let aset_open = mpd.find("<AdaptationSet").expect("<AdaptationSet");
        let period_close = mpd.find("</Period>").expect("</Period>");
        let mpd_close = mpd.find("</MPD>").expect("</MPD>");
        assert!(mpd_open < period_open);
        assert!(period_open < aset_open);
        assert!(aset_open < period_close);
        assert!(period_close < mpd_close);
    }
}
