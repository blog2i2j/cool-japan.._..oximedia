//! FCPXML 1.10 export — Final Cut Pro XML (Apple, 2023+)
//!
//! Exports an AAF `CompositionMob` to FCPXML format version 1.10, the modern
//! Final Cut Pro X interchange format introduced with FCP 10.6.5.
//!
//! FCPXML 1.10 differs substantially from the legacy FCP 7 `xmeml` format:
//! - Root element: `<fcpxml version="1.10">` (not `<xmeml>`)
//! - Projects are wrapped in a `<library>` → `<event>` → `<project>` hierarchy
//! - Sequences use a `<spine>` element instead of flat track lists
//! - Clips reference `<asset>` elements declared in `<resources>`
//! - Timecode is expressed as rational `s/N` values (frame-accurate fractions)
//! - Roles are attached per-clip for audio/video classification
//!
//! Reference: Apple FCPXML Reference 1.10 (developer.apple.com/documentation/professional_video_applications)

use crate::composition::{CompositionMob, Sequence, SequenceComponent, SourceClip, Track};
use crate::timeline::EditRate;
use crate::{AafError, Result};
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use uuid::Uuid;

// ─── Public types ─────────────────────────────────────────────────────────────

/// FCPXML 1.10 exporter for AAF `CompositionMob` objects.
///
/// Use [`FcpXml10Exporter::new`] to obtain an instance, then call
/// [`export`](Self::export) to produce the XML string.
pub struct FcpXml10Exporter {
    /// Library name (wraps the event/project hierarchy)
    library_name: String,
    /// Whether to emit video clips
    include_video: bool,
    /// Whether to emit audio clips
    include_audio: bool,
    /// Whether to embed `<asset>` declarations in `<resources>`
    include_assets: bool,
}

impl FcpXml10Exporter {
    /// Create a new exporter with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            library_name: "OxiMedia Library".to_string(),
            include_video: true,
            include_audio: true,
            include_assets: true,
        }
    }

    /// Set the library name that appears in the FCPXML hierarchy.
    #[must_use]
    pub fn with_library_name(mut self, name: impl Into<String>) -> Self {
        self.library_name = name.into();
        self
    }

    /// Control whether video clips are included.
    #[must_use]
    pub fn with_video(mut self, include: bool) -> Self {
        self.include_video = include;
        self
    }

    /// Control whether audio clips are included.
    #[must_use]
    pub fn with_audio(mut self, include: bool) -> Self {
        self.include_audio = include;
        self
    }

    /// Control whether `<asset>` declarations are emitted in `<resources>`.
    #[must_use]
    pub fn with_assets(mut self, include: bool) -> Self {
        self.include_assets = include;
        self
    }

    /// Export `composition` to an FCPXML 1.10 string.
    ///
    /// # Errors
    ///
    /// Returns [`AafError::ConversionError`] if string formatting fails.
    pub fn export(&self, composition: &CompositionMob) -> Result<String> {
        let mut xml = String::new();

        // XML declaration
        writeln!(xml, r#"<?xml version="1.0" encoding="UTF-8"?>"#)
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(
            xml,
            r#"<!DOCTYPE fcpxml>"#
        )
        .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, r#"<fcpxml version="1.10">"#)
            .map_err(|e| AafError::ConversionError(e.to_string()))?;

        // Collect assets from all clips
        let assets = self.collect_assets(composition);

        // <resources>
        if self.include_assets {
            self.write_resources(&assets, &mut xml)?;
        }

        // <library> → <event> → <project>
        self.write_library(composition, &assets, &mut xml)?;

        writeln!(xml, "</fcpxml>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;

        Ok(xml)
    }

    // ─── Resources ───────────────────────────────────────────────────────────

    fn collect_assets(&self, composition: &CompositionMob) -> HashMap<Uuid, AssetInfo> {
        let mut assets: HashMap<Uuid, AssetInfo> = HashMap::new();
        for track in composition.tracks() {
            let include = (track.is_picture() && self.include_video)
                || (track.is_sound() && self.include_audio);
            if !include {
                continue;
            }
            if let Some(ref seq) = track.sequence {
                for comp in &seq.components {
                    if let SequenceComponent::SourceClip(clip) = comp {
                        let id = clip.source_mob_id;
                        assets.entry(id).or_insert_with(|| AssetInfo {
                            mob_id: id,
                            has_video: track.is_picture(),
                            has_audio: track.is_sound(),
                        });
                    }
                }
            }
        }
        assets
    }

    fn write_resources(
        &self,
        assets: &HashMap<Uuid, AssetInfo>,
        xml: &mut String,
    ) -> Result<()> {
        writeln!(xml, "  <resources>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;

        // Format element
        writeln!(
            xml,
            r#"    <format id="r1" name="FFVideoFormat1080p25" frameDuration="100/2500s" width="1920" height="1080"/>"#
        )
        .map_err(|e| AafError::ConversionError(e.to_string()))?;

        // Asset elements
        for info in assets.values() {
            let short_id = &info.mob_id.to_string()[..8];
            let has_v = if info.has_video { "1" } else { "0" };
            let has_a = if info.has_audio { "1" } else { "0" };
            writeln!(
                xml,
                r#"    <asset id="asset-{short_id}" name="mob_{short_id}" uid="{}" hasVideo="{has_v}" hasAudio="{has_a}" format="r1"/>"#,
                info.mob_id
            )
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        }

        writeln!(xml, "  </resources>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        Ok(())
    }

    // ─── Library / Event / Project ────────────────────────────────────────────

    fn write_library(
        &self,
        composition: &CompositionMob,
        assets: &HashMap<Uuid, AssetInfo>,
        xml: &mut String,
    ) -> Result<()> {
        let lib_name = xml_escape(&self.library_name);
        writeln!(xml, r#"  <library name="{lib_name}">"#)
            .map_err(|e| AafError::ConversionError(e.to_string()))?;

        let event_name = xml_escape(composition.name());
        writeln!(xml, r#"    <event name="{event_name}">"#)
            .map_err(|e| AafError::ConversionError(e.to_string()))?;

        self.write_project(composition, assets, xml)?;

        writeln!(xml, "    </event>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "  </library>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        Ok(())
    }

    fn write_project(
        &self,
        composition: &CompositionMob,
        assets: &HashMap<Uuid, AssetInfo>,
        xml: &mut String,
    ) -> Result<()> {
        let proj_name = xml_escape(composition.name());
        let uid = composition.mob_id();
        writeln!(xml, r#"      <project name="{proj_name}" uid="{uid}">"#)
            .map_err(|e| AafError::ConversionError(e.to_string()))?;

        self.write_sequence(composition, assets, xml)?;

        writeln!(xml, "      </project>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        Ok(())
    }

    fn write_sequence(
        &self,
        composition: &CompositionMob,
        assets: &HashMap<Uuid, AssetInfo>,
        xml: &mut String,
    ) -> Result<()> {
        let edit_rate = composition.edit_rate().unwrap_or(EditRate::PAL_25);
        let duration_frames = composition.duration().unwrap_or(0);
        let dur_str = frames_to_rational_time(duration_frames, edit_rate);
        let tc_format = if edit_rate.is_ntsc() {
            "DF"
        } else {
            "NDF"
        };

        writeln!(
            xml,
            r#"        <sequence duration="{dur_str}" format="r1" tcStart="0s" tcFormat="{tc_format}">"#
        )
        .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "          <spine>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;

        // Emit all clips from all eligible tracks on the spine
        for track in composition.tracks() {
            let include = (track.is_picture() && self.include_video)
                || (track.is_sound() && self.include_audio);
            if !include {
                continue;
            }
            if let Some(ref seq) = track.sequence {
                self.write_spine_clips(seq, &track, edit_rate, assets, xml)?;
            }
        }

        writeln!(xml, "          </spine>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        writeln!(xml, "        </sequence>")
            .map_err(|e| AafError::ConversionError(e.to_string()))?;
        Ok(())
    }

    fn write_spine_clips(
        &self,
        sequence: &Sequence,
        track: &Track,
        edit_rate: EditRate,
        _assets: &HashMap<Uuid, AssetInfo>,
        xml: &mut String,
    ) -> Result<()> {
        let role = if track.is_picture() {
            "video"
        } else {
            "audio"
        };
        let mut clip_index = 1u32;

        for component in &sequence.components {
            match component {
                SequenceComponent::SourceClip(clip) => {
                    self.write_asset_clip(clip, clip_index, role, edit_rate, xml)?;
                    clip_index += 1;
                }
                SequenceComponent::Filler(filler) => {
                    // Gaps are expressed as <gap> elements in FCPXML 1.10
                    let dur = frames_to_rational_time(filler.length, edit_rate);
                    writeln!(xml, r#"            <gap duration="{dur}"/>"#)
                        .map_err(|e| AafError::ConversionError(e.to_string()))?;
                }
                SequenceComponent::Transition(trans) => {
                    // Transitions become <transition> elements
                    let dur = frames_to_rational_time(trans.length, edit_rate);
                    writeln!(xml, r#"            <transition duration="{dur}"/>"#)
                        .map_err(|e| AafError::ConversionError(e.to_string()))?;
                }
                SequenceComponent::Effect(_) => {
                    // Effects are attached to clips in FCPXML; skip at top-level
                }
            }
        }
        Ok(())
    }

    fn write_asset_clip(
        &self,
        clip: &SourceClip,
        index: u32,
        role: &str,
        edit_rate: EditRate,
        xml: &mut String,
    ) -> Result<()> {
        let short_id = &clip.source_mob_id.to_string()[..8];
        let asset_ref = format!("asset-{short_id}");
        let name = format!("Clip_{index}_{short_id}");
        let start = frames_to_rational_time(clip.start_time.0, edit_rate);
        let duration = frames_to_rational_time(clip.length, edit_rate);

        writeln!(
            xml,
            r#"            <asset-clip ref="{asset_ref}" name="{name}" start="{start}" duration="{duration}" role="{role}"/>"#
        )
        .map_err(|e| AafError::ConversionError(e.to_string()))?;

        Ok(())
    }
}

impl Default for FcpXml10Exporter {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Internal asset info ──────────────────────────────────────────────────────

struct AssetInfo {
    mob_id: Uuid,
    has_video: bool,
    has_audio: bool,
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Escape XML special characters.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Convert a frame count to FCPXML rational time notation `N/Ds`.
///
/// In FCPXML 1.10, durations and timestamps are expressed as rational fractions
/// of a second: e.g. 25 frames at 25 fps = `"1s"`, 10 frames at 25 fps = `"2/5s"`.
fn frames_to_rational_time(frames: i64, rate: EditRate) -> String {
    if frames == 0 {
        return "0s".to_string();
    }
    let num = rate.numerator as i64;
    let den = rate.denominator as i64;
    if den == 0 || num == 0 {
        return "0s".to_string();
    }
    // frames / (num/den) seconds = frames * den / num seconds
    let numer = frames * den;
    let denom = num;
    let g = gcd(numer.unsigned_abs(), denom.unsigned_abs());
    let numer_r = numer / g as i64;
    let denom_r = denom / g as i64;
    if denom_r == 1 {
        format!("{numer_r}s")
    } else {
        format!("{numer_r}/{denom_r}s")
    }
}

/// Compute GCD via binary algorithm (no stdlib dependency).
fn gcd(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::{
        CompositionMob, Filler, Sequence, SequenceComponent, SourceClip, Track, TrackType,
    };
    use crate::dictionary::Auid;
    use crate::timeline::{EditRate, Position};
    use uuid::Uuid;

    fn make_composition() -> CompositionMob {
        let src1 = Uuid::new_v4();
        let src2 = Uuid::new_v4();

        let mut comp = CompositionMob::new(Uuid::new_v4(), "Post Edit");

        // Video track
        let mut vid_seq = Sequence::new(Auid::PICTURE);
        vid_seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            100,
            Position::zero(),
            src1,
            1,
        )));
        vid_seq.add_component(SequenceComponent::Filler(Filler::new(10)));
        vid_seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            50,
            Position::new(100),
            src2,
            1,
        )));
        let mut vid_track = Track::new(1, "V1", EditRate::PAL_25, TrackType::Picture);
        vid_track.set_sequence(vid_seq);
        comp.add_track(vid_track);

        // Audio track
        let mut aud_seq = Sequence::new(Auid::SOUND);
        aud_seq.add_component(SequenceComponent::SourceClip(SourceClip::new(
            160,
            Position::zero(),
            src1,
            2,
        )));
        let mut aud_track = Track::new(2, "A1", EditRate::PAL_25, TrackType::Sound);
        aud_track.set_sequence(aud_seq);
        comp.add_track(aud_track);

        comp
    }

    #[test]
    fn test_fcpxml10_root_element() {
        let comp = make_composition();
        let xml = FcpXml10Exporter::new()
            .export(&comp)
            .expect("export must succeed");
        assert!(xml.contains(r#"<fcpxml version="1.10">"#));
        assert!(xml.contains("</fcpxml>"));
    }

    #[test]
    fn test_fcpxml10_library_element() {
        let comp = make_composition();
        let xml = FcpXml10Exporter::new()
            .with_library_name("My Library")
            .export(&comp)
            .expect("export must succeed");
        assert!(xml.contains(r#"<library name="My Library">"#));
        assert!(xml.contains("</library>"));
    }

    #[test]
    fn test_fcpxml10_event_element() {
        let comp = make_composition();
        let xml = FcpXml10Exporter::new()
            .export(&comp)
            .expect("export must succeed");
        assert!(xml.contains(r#"<event name="Post Edit">"#));
        assert!(xml.contains("</event>"));
    }

    #[test]
    fn test_fcpxml10_project_element() {
        let comp = make_composition();
        let uid = comp.mob_id();
        let xml = FcpXml10Exporter::new()
            .export(&comp)
            .expect("export must succeed");
        assert!(xml.contains(r#"<project name="Post Edit""#));
        assert!(xml.contains(&format!("uid=\"{uid}\"")));
    }

    #[test]
    fn test_fcpxml10_spine_element() {
        let comp = make_composition();
        let xml = FcpXml10Exporter::new()
            .export(&comp)
            .expect("export must succeed");
        assert!(xml.contains("<spine>"));
        assert!(xml.contains("</spine>"));
    }

    #[test]
    fn test_fcpxml10_asset_clip_present() {
        let comp = make_composition();
        let xml = FcpXml10Exporter::new()
            .export(&comp)
            .expect("export must succeed");
        assert!(xml.contains("<asset-clip"));
    }

    #[test]
    fn test_fcpxml10_gap_for_filler() {
        let comp = make_composition();
        let xml = FcpXml10Exporter::new()
            .export(&comp)
            .expect("export must succeed");
        assert!(xml.contains("<gap"), "filler should produce a <gap> element");
    }

    #[test]
    fn test_fcpxml10_resources_section() {
        let comp = make_composition();
        let xml = FcpXml10Exporter::new()
            .export(&comp)
            .expect("export must succeed");
        assert!(xml.contains("<resources>"));
        assert!(xml.contains("<asset "));
        assert!(xml.contains("</resources>"));
    }

    #[test]
    fn test_fcpxml10_video_only_excludes_audio() {
        let comp = make_composition();
        let xml = FcpXml10Exporter::new()
            .with_audio(false)
            .export(&comp)
            .expect("export must succeed");
        // No audio-role asset-clips
        assert!(!xml.contains(r#"role="audio""#));
        assert!(xml.contains(r#"role="video""#));
    }

    #[test]
    fn test_fcpxml10_audio_only_excludes_video() {
        let comp = make_composition();
        let xml = FcpXml10Exporter::new()
            .with_video(false)
            .export(&comp)
            .expect("export must succeed");
        assert!(xml.contains(r#"role="audio""#));
        assert!(!xml.contains(r#"role="video""#));
    }

    #[test]
    fn test_fcpxml10_rational_time_whole_second() {
        // 25 frames at 25 fps = 1s
        assert_eq!(frames_to_rational_time(25, EditRate::PAL_25), "1s");
    }

    #[test]
    fn test_fcpxml10_rational_time_fraction() {
        // 10 frames at 25 fps = 10/25s = 2/5s
        assert_eq!(frames_to_rational_time(10, EditRate::PAL_25), "2/5s");
    }

    #[test]
    fn test_fcpxml10_rational_time_zero() {
        assert_eq!(frames_to_rational_time(0, EditRate::PAL_25), "0s");
    }

    #[test]
    fn test_fcpxml10_sequence_ndf_for_pal() {
        let comp = make_composition();
        let xml = FcpXml10Exporter::new()
            .export(&comp)
            .expect("export must succeed");
        assert!(xml.contains(r#"tcFormat="NDF""#));
    }

    #[test]
    fn test_fcpxml10_xml_escape_in_names() {
        let mut comp = CompositionMob::new(Uuid::new_v4(), "My & Project");
        let mut seq = Sequence::new(Auid::PICTURE);
        seq.add_component(SequenceComponent::Filler(Filler::new(25)));
        let mut track = Track::new(1, "V1", EditRate::PAL_25, TrackType::Picture);
        track.set_sequence(seq);
        comp.add_track(track);
        let xml = FcpXml10Exporter::new()
            .export(&comp)
            .expect("export must succeed");
        assert!(xml.contains("My &amp; Project"));
    }

    #[test]
    fn test_gcd_basic() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(7, 0), 7);
        assert_eq!(gcd(15, 25), 5);
    }
}
