//! FCPXML (Final Cut Pro XML) format parser and writer.
//!
//! FCPXML is Apple's Final Cut Pro project exchange format. This module supports
//! FCPXML version 1.9 and reads the `<spine>` element of a sequence to build
//! an [`Edl`], and writes an `Edl` back to a minimal FCPXML 1.9 document.
//!
//! # Format overview
//!
//! ```xml
//! <?xml version="1.0" encoding="UTF-8"?>
//! <fcpxml version="1.9">
//!   <resources>
//!     <format id="r1" frameDuration="1001/30000s"/>
//!     <asset id="r2" name="clip" src="file:///clip.mov"/>
//!   </resources>
//!   <library>
//!     <event name="My Event">
//!       <project name="My Project">
//!         <sequence duration="200/30000s" tcStart="0s" tcFormat="NDF">
//!           <spine>
//!             <clip name="clip" ref="r2" offset="0s" duration="100/30000s" start="0s"/>
//!             <transition name="Cross Dissolve" offset="100/30000s" duration="30/30000s"/>
//!           </spine>
//!         </sequence>
//!       </project>
//!     </event>
//!   </library>
//! </fcpxml>
//! ```

use super::{EditType, Edl, EdlError, EdlEvent, EdlResult, Timecode};
use oximedia_core::Rational;
use quick_xml::events::Event;
use quick_xml::Reader;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Parse an FCP time string (e.g. `"0s"`, `"300s"`, `"1001/30000s"`) to a
/// rational value in *seconds* as `(numerator, denominator)`.
fn parse_fcp_time_rational(s: &str) -> EdlResult<(i64, i64)> {
    let s = s.trim().trim_end_matches('s');
    if s.is_empty() {
        return Ok((0, 1));
    }
    if let Some(pos) = s.find('/') {
        let num: i64 = s[..pos]
            .parse()
            .map_err(|_| EdlError::InvalidTimecode(format!("bad numerator in '{}'", s)))?;
        let den: i64 = s[pos + 1..]
            .parse()
            .map_err(|_| EdlError::InvalidTimecode(format!("bad denominator in '{}'", s)))?;
        if den == 0 {
            return Err(EdlError::InvalidTimecode(
                "zero denominator in FCP time".to_string(),
            ));
        }
        Ok((num, den))
    } else {
        let num: i64 = s
            .parse()
            .map_err(|_| EdlError::InvalidTimecode(format!("cannot parse FCP time '{}'", s)))?;
        Ok((num, 1))
    }
}

/// Convert an FCP time string to a frame count given the sequence frame rate.
///
/// `fps` is passed as `Rational` so we can compute `round(time_secs * fps)`.
fn fcp_time_to_frames(s: &str, fps: Rational) -> EdlResult<i64> {
    let (num, den) = parse_fcp_time_rational(s)?;
    // frames = round( (num / den) * (fps_num / fps_den) )
    // Use i128 intermediates to avoid overflow on large values.
    let fps_num = fps.num;
    let fps_den = fps.den;
    // frames = num * fps_num / (den * fps_den) — round to nearest
    let numer = num as i128 * fps_num as i128;
    let denom = den as i128 * fps_den as i128;
    if denom == 0 {
        return Err(EdlError::InvalidTimecode(
            "zero denominator when converting FCP time to frames".to_string(),
        ));
    }
    // Round-half-away-from-zero
    let frames = if numer >= 0 {
        (numer + denom / 2) / denom
    } else {
        (numer - denom / 2) / denom
    };
    Ok(frames as i64)
}

/// Emit an FCP time string `"N/Ds"` for the given frame count and frame rate.
fn frames_to_fcp_time(frames: i64, fps: Rational) -> String {
    if frames == 0 {
        return "0s".to_string();
    }
    // time_seconds = frames / fps = frames * fps_den / fps_num
    let num = frames as i128 * fps.den as i128;
    let den = fps.num as i128;
    if den == 0 {
        return "0s".to_string();
    }
    // Simplify by GCD
    let g = gcd(num.unsigned_abs(), den.unsigned_abs()) as i128;
    let num = num / g;
    let den = den / g;
    format!("{}/{}s", num, den)
}

fn gcd(a: u128, b: u128) -> u128 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// A thin wrapper around an attribute map extracted from an XML start tag.
type AttrMap = HashMap<String, String>;

/// Extract all attributes from a `quick_xml::events::BytesStart` into a `HashMap`.
fn attrs_to_map(e: &quick_xml::events::BytesStart<'_>) -> EdlResult<AttrMap> {
    let mut map = HashMap::new();
    for attr_result in e.attributes() {
        let attr =
            attr_result.map_err(|err| EdlError::XmlError(format!("attribute error: {}", err)))?;
        let key = String::from_utf8_lossy(attr.key.as_ref()).into_owned();
        let value = attr
            .unescape_value()
            .map_err(|err| EdlError::XmlError(format!("unescape error: {}", err)))?
            .into_owned();
        map.insert(key, value);
    }
    Ok(map)
}

// ---------------------------------------------------------------------------
// Parser state machine
// ---------------------------------------------------------------------------

/// Intermediate resource tables built during the first pass over `<resources>`.
#[derive(Default)]
struct Resources {
    /// `format_id -> frameDuration rational (num, den)`.
    formats: HashMap<String, (i64, i64)>,
    /// `asset_id -> asset_name`.
    asset_names: HashMap<String, String>,
}

/// Full FCPXML parser.
struct FcpxmlParser {
    resources: Resources,
    project_name: String,
    sequence_fps: Rational,
    drop_frame: bool,
    events: Vec<EdlEvent>,
    event_counter: u32,
}

impl FcpxmlParser {
    fn new() -> Self {
        Self {
            resources: Resources::default(),
            project_name: String::new(),
            sequence_fps: Rational::new(30, 1),
            drop_frame: false,
            events: Vec::new(),
            event_counter: 0,
        }
    }

    /// Top-level parse entry point.
    fn parse(mut self, content: &str) -> EdlResult<Edl> {
        let mut reader = Reader::from_str(content);
        reader.config_mut().trim_text(true);

        // Track nesting depth for spine elements
        let mut in_resources = false;
        let mut in_spine = false;
        let mut spine_depth: usize = 0;

        let mut buf = Vec::new();
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Eof) => break,
                Ok(Event::Start(ref e)) => {
                    let tag = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                    match tag.as_str() {
                        "resources" => {
                            in_resources = true;
                        }
                        "format" if in_resources => {
                            let attrs = attrs_to_map(e)?;
                            self.handle_format(&attrs)?;
                        }
                        "asset" if in_resources => {
                            let attrs = attrs_to_map(e)?;
                            self.handle_asset(&attrs);
                        }
                        "project" => {
                            let attrs = attrs_to_map(e)?;
                            if let Some(name) = attrs.get("name") {
                                self.project_name = name.clone();
                            }
                        }
                        "sequence" => {
                            let attrs = attrs_to_map(e)?;
                            self.handle_sequence(&attrs)?;
                        }
                        "spine" => {
                            in_spine = true;
                            spine_depth = 0;
                        }
                        "clip" if in_spine && spine_depth == 0 => {
                            let attrs = attrs_to_map(e)?;
                            self.handle_clip(&attrs)?;
                        }
                        "transition" if in_spine && spine_depth == 0 => {
                            let attrs = attrs_to_map(e)?;
                            self.handle_transition(&attrs)?;
                        }
                        _ if in_spine => {
                            spine_depth += 1;
                        }
                        _ => {}
                    }
                }
                Ok(Event::Empty(ref e)) => {
                    let tag = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                    match tag.as_str() {
                        "format" if in_resources => {
                            let attrs = attrs_to_map(e)?;
                            self.handle_format(&attrs)?;
                        }
                        "asset" if in_resources => {
                            let attrs = attrs_to_map(e)?;
                            self.handle_asset(&attrs);
                        }
                        "clip" if in_spine && spine_depth == 0 => {
                            let attrs = attrs_to_map(e)?;
                            self.handle_clip(&attrs)?;
                        }
                        "transition" if in_spine && spine_depth == 0 => {
                            let attrs = attrs_to_map(e)?;
                            self.handle_transition(&attrs)?;
                        }
                        _ => {}
                    }
                }
                Ok(Event::End(ref e)) => {
                    let tag = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                    match tag.as_str() {
                        "resources" => {
                            in_resources = false;
                        }
                        "spine" => {
                            in_spine = false;
                            spine_depth = 0;
                        }
                        _ if in_spine && spine_depth > 0 => {
                            spine_depth -= 1;
                        }
                        _ => {}
                    }
                }
                Err(e) => {
                    return Err(EdlError::XmlError(format!("XML parse error: {}", e)));
                }
                _ => {}
            }
            buf.clear();
        }

        Ok(Edl {
            title: self.project_name,
            frame_rate: self.sequence_fps,
            drop_frame: self.drop_frame,
            events: self.events,
            comments: Vec::new(),
            metadata: HashMap::new(),
        })
    }

    fn handle_format(&mut self, attrs: &AttrMap) -> EdlResult<()> {
        if let (Some(id), Some(dur)) = (attrs.get("id"), attrs.get("frameDuration")) {
            let rational = parse_fcp_time_rational(dur)?;
            self.resources.formats.insert(id.clone(), rational);
        }
        Ok(())
    }

    fn handle_asset(&mut self, attrs: &AttrMap) {
        if let (Some(id), Some(name)) = (attrs.get("id"), attrs.get("name")) {
            self.resources.asset_names.insert(id.clone(), name.clone());
        }
    }

    fn handle_sequence(&mut self, attrs: &AttrMap) -> EdlResult<()> {
        // Determine frame rate from first available format resource
        // FCPXML sequences reference a format via `format="r1"` attribute or
        // we fall back to the first format in resources.
        let format_id = attrs.get("format").cloned();
        let frame_dur = if let Some(ref id) = format_id {
            self.resources.formats.get(id).copied()
        } else {
            self.resources.formats.values().next().copied()
        };

        if let Some((dur_num, dur_den)) = frame_dur {
            // frameDuration = per-frame duration in seconds (num/den)
            // fps = 1 / frameDuration = den / num
            if dur_num == 0 {
                return Err(EdlError::InvalidFormat(
                    "frameDuration numerator is zero".to_string(),
                ));
            }
            self.sequence_fps = Rational::new(dur_den, dur_num);
        }

        // Determine drop-frame from tcFormat attribute
        if let Some(tc_format) = attrs.get("tcFormat") {
            self.drop_frame = tc_format == "DF";
        }
        Ok(())
    }

    fn handle_clip(&mut self, attrs: &AttrMap) -> EdlResult<()> {
        let fps = self.sequence_fps;

        // `offset` = record-in position on the timeline
        let offset_str = attrs.get("offset").map(|s| s.as_str()).unwrap_or("0s");
        let record_in_frames = fcp_time_to_frames(offset_str, fps)?;

        // `duration` = duration of the clip in the timeline
        let dur_str = attrs.get("duration").map(|s| s.as_str()).unwrap_or("0s");
        let duration_frames = fcp_time_to_frames(dur_str, fps)?;

        // `start` = source-media in-point
        let start_str = attrs.get("start").map(|s| s.as_str()).unwrap_or("0s");
        let source_in_frames = fcp_time_to_frames(start_str, fps)?;

        let record_out_frames = record_in_frames + duration_frames;
        let source_out_frames = source_in_frames + duration_frames;

        // Resolve asset name for reel
        let reel = if let Some(ref_id) = attrs.get("ref") {
            self.resources
                .asset_names
                .get(ref_id)
                .cloned()
                .unwrap_or_else(|| "AX".to_string())
        } else {
            attrs
                .get("name")
                .cloned()
                .unwrap_or_else(|| "AX".to_string())
        };

        self.event_counter += 1;
        self.events.push(EdlEvent {
            number: self.event_counter,
            reel,
            track: "V".to_string(),
            edit_type: EditType::Cut,
            source_in: Timecode::from_frames(source_in_frames, fps, self.drop_frame),
            source_out: Timecode::from_frames(source_out_frames, fps, self.drop_frame),
            record_in: Timecode::from_frames(record_in_frames, fps, self.drop_frame),
            record_out: Timecode::from_frames(record_out_frames, fps, self.drop_frame),
            transition_duration: None,
            motion_effect: None,
            comments: Vec::new(),
            metadata: HashMap::new(),
        });
        Ok(())
    }

    fn handle_transition(&mut self, attrs: &AttrMap) -> EdlResult<()> {
        let fps = self.sequence_fps;

        let offset_str = attrs.get("offset").map(|s| s.as_str()).unwrap_or("0s");
        let record_in_frames = fcp_time_to_frames(offset_str, fps)?;

        let dur_str = attrs.get("duration").map(|s| s.as_str()).unwrap_or("0s");
        let duration_frames = fcp_time_to_frames(dur_str, fps)?;

        let record_out_frames = record_in_frames + duration_frames;

        self.event_counter += 1;
        self.events.push(EdlEvent {
            number: self.event_counter,
            reel: "AX".to_string(),
            track: "V".to_string(),
            edit_type: EditType::Dissolve,
            source_in: Timecode::from_frames(0, fps, self.drop_frame),
            source_out: Timecode::from_frames(duration_frames, fps, self.drop_frame),
            record_in: Timecode::from_frames(record_in_frames, fps, self.drop_frame),
            record_out: Timecode::from_frames(record_out_frames, fps, self.drop_frame),
            transition_duration: Some(duration_frames as u32),
            motion_effect: None,
            comments: Vec::new(),
            metadata: HashMap::new(),
        });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Write an [`Edl`] as FCPXML 1.9.
struct FcpxmlWriter;

impl FcpxmlWriter {
    fn write(&self, edl: &Edl) -> EdlResult<String> {
        let fps = edl.frame_rate;

        // Collect unique reels → assets
        let mut asset_map: HashMap<String, String> = HashMap::new(); // reel -> asset_id
        let mut asset_counter = 2usize;
        for event in &edl.events {
            if event.edit_type == EditType::Cut && !asset_map.contains_key(&event.reel) {
                asset_map.insert(event.reel.clone(), format!("r{}", asset_counter));
                asset_counter += 1;
            }
        }

        // Compute total timeline duration
        let total_frames = edl
            .events
            .iter()
            .map(|e| e.record_out.to_frames())
            .max()
            .unwrap_or(0);

        let mut out = String::new();
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str("<!DOCTYPE fcpxml>\n");
        out.push_str("<fcpxml version=\"1.9\">\n");

        // <resources>
        out.push_str("  <resources>\n");
        // frame duration = 1 / fps = fps_den / fps_num
        let frame_dur = format!("{}/{}s", fps.den, fps.num);
        out.push_str(&format!(
            "    <format id=\"r1\" name=\"FFVideoFormat\" frameDuration=\"{}\"/>\n",
            frame_dur
        ));
        // Assets: one per unique reel (for Cut events)
        let mut sorted_assets: Vec<(&String, &String)> = asset_map.iter().collect();
        sorted_assets.sort_by_key(|(_, id)| id.as_str());
        for (reel, asset_id) in &sorted_assets {
            let safe_name = reel.replace('"', "&quot;");
            out.push_str(&format!(
                "    <asset id=\"{}\" name=\"{}\" src=\"file:///media/{}.mov\"/>\n",
                asset_id, safe_name, safe_name
            ));
        }
        out.push_str("  </resources>\n");

        // <library><event><project><sequence>
        let seq_duration = frames_to_fcp_time(total_frames, fps);
        let tc_format = if edl.drop_frame { "DF" } else { "NDF" };
        let project_name = edl.title.replace('"', "&quot;");
        out.push_str("  <library>\n");
        out.push_str("    <event name=\"Untitled Event\">\n");
        out.push_str(&format!("      <project name=\"{}\">\n", project_name));
        out.push_str(&format!(
            "        <sequence format=\"r1\" duration=\"{}\" tcStart=\"0s\" tcFormat=\"{}\" audioLayout=\"stereo\" audioRate=\"48k\">\n",
            seq_duration, tc_format
        ));
        out.push_str("          <spine>\n");

        for event in &edl.events {
            match event.edit_type {
                EditType::Cut => {
                    let asset_id = asset_map
                        .get(&event.reel)
                        .map(|s| s.as_str())
                        .unwrap_or("r2");
                    let offset = frames_to_fcp_time(event.record_in.to_frames(), fps);
                    let duration = frames_to_fcp_time(
                        event.record_out.to_frames() - event.record_in.to_frames(),
                        fps,
                    );
                    let start = frames_to_fcp_time(event.source_in.to_frames(), fps);
                    let clip_name = event.reel.replace('"', "&quot;");
                    out.push_str(&format!(
                        "            <clip name=\"{}\" ref=\"{}\" offset=\"{}\" duration=\"{}\" start=\"{}\"/>\n",
                        clip_name, asset_id, offset, duration, start
                    ));
                }
                EditType::Dissolve => {
                    let offset = frames_to_fcp_time(event.record_in.to_frames(), fps);
                    let duration = frames_to_fcp_time(
                        event.record_out.to_frames() - event.record_in.to_frames(),
                        fps,
                    );
                    out.push_str(&format!(
                        "            <transition name=\"Cross Dissolve\" offset=\"{}\" duration=\"{}\"/>\n",
                        offset, duration
                    ));
                }
                EditType::Wipe => {
                    let offset = frames_to_fcp_time(event.record_in.to_frames(), fps);
                    let duration = frames_to_fcp_time(
                        event.record_out.to_frames() - event.record_in.to_frames(),
                        fps,
                    );
                    out.push_str(&format!(
                        "            <transition name=\"Wipe\" offset=\"{}\" duration=\"{}\"/>\n",
                        offset, duration
                    ));
                }
                EditType::Key => {
                    // Key edits are represented as clips with a `compositeMode` attribute
                    let asset_id = asset_map
                        .get(&event.reel)
                        .map(|s| s.as_str())
                        .unwrap_or("r2");
                    let offset = frames_to_fcp_time(event.record_in.to_frames(), fps);
                    let duration = frames_to_fcp_time(
                        event.record_out.to_frames() - event.record_in.to_frames(),
                        fps,
                    );
                    let start = frames_to_fcp_time(event.source_in.to_frames(), fps);
                    let clip_name = event.reel.replace('"', "&quot;");
                    out.push_str(&format!(
                        "            <clip name=\"{}\" ref=\"{}\" offset=\"{}\" duration=\"{}\" start=\"{}\" compositeMode=\"normal\"/>\n",
                        clip_name, asset_id, offset, duration, start
                    ));
                }
            }
        }

        out.push_str("          </spine>\n");
        out.push_str("        </sequence>\n");
        out.push_str("      </project>\n");
        out.push_str("    </event>\n");
        out.push_str("  </library>\n");
        out.push_str("</fcpxml>\n");

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse an FCPXML string into an [`Edl`].
pub fn parse(content: &str) -> EdlResult<Edl> {
    FcpxmlParser::new().parse(content)
}

/// Write an [`Edl`] as an FCPXML 1.9 string.
pub fn write(edl: &Edl) -> EdlResult<String> {
    FcpxmlWriter.write(edl)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_fps_30() -> Rational {
        Rational::new(30, 1)
    }

    fn make_tc(frames: i64, fps: Rational) -> Timecode {
        Timecode::from_frames(frames, fps, false)
    }

    fn make_event(
        number: u32,
        reel: &str,
        edit_type: EditType,
        src_in: i64,
        src_out: i64,
        rec_in: i64,
        rec_out: i64,
        fps: Rational,
        trans_dur: Option<u32>,
    ) -> EdlEvent {
        EdlEvent {
            number,
            reel: reel.to_string(),
            track: "V".to_string(),
            edit_type,
            source_in: make_tc(src_in, fps),
            source_out: make_tc(src_out, fps),
            record_in: make_tc(rec_in, fps),
            record_out: make_tc(rec_out, fps),
            transition_duration: trans_dur,
            motion_effect: None,
            comments: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Round-trip test
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_round_trip() {
        let fps = make_fps_30();
        let mut edl = Edl::new("RoundTrip Test".to_string(), fps, false);
        edl.add_event(make_event(
            1,
            "CLIP_A",
            EditType::Cut,
            0,
            90,
            0,
            90,
            fps,
            None,
        ));
        edl.add_event(make_event(
            2,
            "AX",
            EditType::Dissolve,
            0,
            30,
            90,
            120,
            fps,
            Some(30),
        ));
        edl.add_event(make_event(
            3,
            "CLIP_B",
            EditType::Cut,
            0,
            120,
            120,
            240,
            fps,
            None,
        ));

        let xml = write(&edl).expect("write should succeed");
        let edl2 = parse(&xml).expect("parse should succeed");

        assert_eq!(edl2.title, "RoundTrip Test");
        assert_eq!(edl2.events.len(), 3, "event count mismatch");

        // Verify cut clip round-trips within ±1 frame
        let e0_orig = &edl.events[0];
        let e0 = &edl2.events[0];
        assert!(
            (e0.record_in.to_frames() - e0_orig.record_in.to_frames()).abs() <= 1,
            "record_in mismatch: {} vs {}",
            e0.record_in.to_frames(),
            e0_orig.record_in.to_frames()
        );
        assert!(
            (e0.record_out.to_frames() - e0_orig.record_out.to_frames()).abs() <= 1,
            "record_out mismatch"
        );

        // Transition
        let e1 = &edl2.events[1];
        assert_eq!(e1.edit_type, EditType::Dissolve);
        assert!(
            (e1.record_in.to_frames() - 90).abs() <= 1,
            "transition record_in mismatch"
        );
        assert_eq!(e1.transition_duration, Some(30));
    }

    // -----------------------------------------------------------------------
    // Empty timeline
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_empty_spine() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<fcpxml version="1.9">
  <resources>
    <format id="r1" frameDuration="1/30s"/>
  </resources>
  <library>
    <event name="My Event">
      <project name="Empty Project">
        <sequence format="r1" duration="0s" tcStart="0s" tcFormat="NDF">
          <spine/>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>"#;
        let edl = parse(xml).expect("parse should succeed");
        assert_eq!(edl.events.len(), 0);
        assert_eq!(edl.title, "Empty Project");
    }

    // -----------------------------------------------------------------------
    // Malformed XML
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_malformed_xml() {
        let bad = "<fcpxml version=\"1.9\"><resources><format id=BROKEN";
        let result = parse(bad);
        assert!(result.is_err(), "malformed XML should fail");
    }

    // -----------------------------------------------------------------------
    // Multiple clips
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_clips_and_transition() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<fcpxml version="1.9">
  <resources>
    <format id="r1" frameDuration="1/24s"/>
    <asset id="r2" name="alpha"/>
    <asset id="r3" name="beta"/>
  </resources>
  <library>
    <event name="E">
      <project name="Proj">
        <sequence format="r1" duration="260/24s" tcStart="0s" tcFormat="NDF">
          <spine>
            <clip name="alpha" ref="r2" offset="0s" duration="120/24s" start="0s"/>
            <transition name="Cross Dissolve" offset="120/24s" duration="20/24s"/>
            <clip name="beta" ref="r3" offset="140/24s" duration="120/24s" start="0s"/>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>"#;
        let edl = parse(xml).expect("parse should succeed");
        assert_eq!(edl.events.len(), 3);
        assert_eq!(edl.events[0].edit_type, EditType::Cut);
        assert_eq!(edl.events[0].reel, "alpha");
        assert_eq!(edl.events[1].edit_type, EditType::Dissolve);
        assert_eq!(edl.events[1].transition_duration, Some(20));
        assert_eq!(edl.events[2].edit_type, EditType::Cut);
        assert_eq!(edl.events[2].reel, "beta");

        // Verify timecodes (24fps integer)
        assert_eq!(edl.events[0].record_in.to_frames(), 0);
        assert_eq!(edl.events[0].record_out.to_frames(), 120);
        assert_eq!(edl.events[2].record_in.to_frames(), 140);
        assert_eq!(edl.events[2].record_out.to_frames(), 260);
    }

    // -----------------------------------------------------------------------
    // Frame duration parsing — 29.97 (1001/30000s per frame)
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_fractional_framerate() {
        // 1001/30000s per frame → fps = 30000/1001 ≈ 29.97
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<fcpxml version="1.9">
  <resources>
    <format id="r1" frameDuration="1001/30000s"/>
    <asset id="r2" name="clip"/>
  </resources>
  <library>
    <event name="E">
      <project name="Proj">
        <sequence format="r1" duration="30030/30000s" tcStart="0s" tcFormat="NDF">
          <spine>
            <clip name="clip" ref="r2" offset="0s" duration="30030/30000s" start="0s"/>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>"#;
        let edl = parse(xml).expect("parse should succeed");
        assert_eq!(edl.events.len(), 1);
        // 30030/30000 seconds * (30000/1001 fps) = 30030/1001 ≈ 30 frames
        let clip = &edl.events[0];
        assert!(
            (clip.record_out.to_frames() - clip.record_in.to_frames() - 30).abs() <= 1,
            "expected ~30 frames duration"
        );
    }

    // -----------------------------------------------------------------------
    // FCP time helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_fcp_time_roundtrip() {
        let fps = Rational::new(24, 1);
        for frames in [0i64, 1, 24, 100, 1200, 86400] {
            let s = frames_to_fcp_time(frames, fps);
            let back = fcp_time_to_frames(&s, fps).expect("should parse");
            assert_eq!(back, frames, "frames={} -> '{}' -> {}", frames, s, back);
        }
    }

    #[test]
    fn test_fcp_time_zero() {
        let fps = Rational::new(30, 1);
        assert_eq!(
            fcp_time_to_frames("0s", fps).expect("zero time should parse"),
            0
        );
        assert_eq!(frames_to_fcp_time(0, fps), "0s");
    }

    // -----------------------------------------------------------------------
    // Write produces valid XML that round-trips
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_produces_xml() {
        let fps = make_fps_30();
        let mut edl = Edl::new("Write Test".to_string(), fps, false);
        edl.add_event(make_event(
            1,
            "REEL_A",
            EditType::Cut,
            0,
            60,
            0,
            60,
            fps,
            None,
        ));
        let xml = write(&edl).expect("write should succeed");

        assert!(xml.contains("fcpxml"), "should contain fcpxml element");
        assert!(xml.contains("REEL_A"), "should contain reel name");
        assert!(xml.contains("Write Test"), "should contain project name");
        assert!(
            xml.contains("frameDuration"),
            "should contain frameDuration"
        );
    }

    // -----------------------------------------------------------------------
    // Drop-frame flag preserved
    // -----------------------------------------------------------------------

    #[test]
    fn test_drop_frame_preserved() {
        let fps = make_fps_30();
        let edl = Edl::new("DF Test".to_string(), fps, true);
        let xml = write(&edl).expect("write should succeed");
        assert!(xml.contains("DF"), "should contain DF marker");

        let edl2 = parse(&xml).expect("parse should succeed");
        assert!(edl2.drop_frame);
    }
}
