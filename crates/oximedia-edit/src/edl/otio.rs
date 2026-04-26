//! OpenTimelineIO (OTIO) JSON format parser and writer.
//!
//! OpenTimelineIO is a JSON-based interchange format for editorial data. This
//! module reads the first Video `Track` inside a `Timeline`'s `Stack`, mapping
//! `Clip` children to [`EdlEvent`]s with [`EditType::Cut`] and `Transition`
//! children to [`EdlEvent`]s with [`EditType::Dissolve`].  `Gap` items advance
//! the record-position counter without producing an event.
//!
//! # Example OTIO JSON
//!
//! ```json
//! {
//!   "OTIO_SCHEMA": "Timeline.1",
//!   "name": "My Timeline",
//!   "global_start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": 24.0, "value": 0.0},
//!   "tracks": {
//!     "OTIO_SCHEMA": "Stack.1",
//!     "children": [
//!       {
//!         "OTIO_SCHEMA": "Track.1",
//!         "kind": "Video",
//!         "children": [
//!           {
//!             "OTIO_SCHEMA": "Clip.1",
//!             "name": "clip_a",
//!             "source_range": {
//!               "OTIO_SCHEMA": "TimeRange.1",
//!               "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": 24.0, "value": 0.0},
//!               "duration":   {"OTIO_SCHEMA": "RationalTime.1", "rate": 24.0, "value": 100.0}
//!             }
//!           }
//!         ]
//!       }
//!     ]
//!   }
//! }
//! ```

use super::{EditType, Edl, EdlError, EdlEvent, EdlResult, Timecode};
use oximedia_core::Rational;
use serde_json::Value;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Read `OTIO_SCHEMA` field from a JSON object. Returns `""` if absent.
fn schema(obj: &Value) -> &str {
    obj.get("OTIO_SCHEMA")
        .and_then(|v| v.as_str())
        .unwrap_or("")
}

/// Extract `rate` and `value` from a `RationalTime` JSON object.
/// Returns `(frames, fps)`.
fn rational_time(obj: &Value) -> EdlResult<(f64, f64)> {
    let rate = obj
        .get("rate")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| EdlError::MissingField("RationalTime.rate".to_string()))?;
    let value = obj
        .get("value")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| EdlError::MissingField("RationalTime.value".to_string()))?;
    Ok((value, rate))
}

/// Convert a `TimeRange` object to `(start_frames, duration_frames, fps)`.
fn time_range(obj: &Value) -> EdlResult<(i64, i64, f64)> {
    let start_obj = obj
        .get("start_time")
        .ok_or_else(|| EdlError::MissingField("TimeRange.start_time".to_string()))?;
    let dur_obj = obj
        .get("duration")
        .ok_or_else(|| EdlError::MissingField("TimeRange.duration".to_string()))?;
    let (start_val, rate) = rational_time(start_obj)?;
    let (dur_val, _) = rational_time(dur_obj)?;
    Ok((start_val.round() as i64, dur_val.round() as i64, rate))
}

/// Build a `RationalTime` JSON value.
fn make_rational_time(value: i64, rate: f64) -> Value {
    serde_json::json!({
        "OTIO_SCHEMA": "RationalTime.1",
        "rate": rate,
        "value": value as f64
    })
}

/// Build a `TimeRange` JSON value.
fn make_time_range(start: i64, duration: i64, rate: f64) -> Value {
    serde_json::json!({
        "OTIO_SCHEMA": "TimeRange.1",
        "start_time": make_rational_time(start, rate),
        "duration": make_rational_time(duration, rate)
    })
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parse an OpenTimelineIO JSON string into an [`Edl`].
pub fn parse(content: &str) -> EdlResult<Edl> {
    let root: Value = serde_json::from_str(content)
        .map_err(|e| EdlError::JsonError(format!("JSON parse error: {}", e)))?;

    // Validate top-level schema
    let top_schema = schema(&root);
    if !top_schema.starts_with("Timeline") {
        return Err(EdlError::InvalidFormat(format!(
            "expected Timeline schema, got '{}'",
            top_schema
        )));
    }

    let title = root
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("Untitled")
        .to_string();

    // Determine frame rate from global_start_time or first clip
    let global_rate = root
        .get("global_start_time")
        .and_then(|v| v.get("rate"))
        .and_then(|v| v.as_f64());

    // Walk tracks to find Video track
    let tracks_obj = root
        .get("tracks")
        .ok_or_else(|| EdlError::MissingField("tracks".to_string()))?;

    let track_children = tracks_obj
        .get("children")
        .and_then(|v| v.as_array())
        .ok_or_else(|| EdlError::MissingField("tracks.children".to_string()))?;

    // Find first Video track
    let video_track = track_children.iter().find(|t| {
        let kind = t.get("kind").and_then(|v| v.as_str()).unwrap_or("");
        kind == "Video" || schema(t).starts_with("Track")
    });

    let children = if let Some(track) = video_track {
        track
            .get("children")
            .and_then(|v| v.as_array())
            .ok_or_else(|| EdlError::MissingField("track.children".to_string()))?
    } else {
        return Ok(Edl::new(title, Rational::new(24, 1), false));
    };

    // Determine fps from first clip's source_range rate, falling back to global
    let clip_rate = children
        .iter()
        .find_map(|child| {
            child
                .get("source_range")
                .and_then(|r| r.get("start_time"))
                .and_then(|s| s.get("rate"))
                .and_then(|r| r.as_f64())
        })
        .or(global_rate)
        .unwrap_or(24.0);

    // Convert float fps to Rational: detect common NTSC rates.
    let fps_rational = float_to_rational(clip_rate);

    let mut events: Vec<EdlEvent> = Vec::new();
    let mut event_counter: u32 = 0;
    let mut record_pos: i64 = 0; // running record-in position in frames

    for child in children {
        let child_schema = schema(child);

        if child_schema.starts_with("Clip") {
            let name = child
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("AX")
                .to_string();

            let (src_start, dur, _rate) = if let Some(sr) = child.get("source_range") {
                time_range(sr)?
            } else {
                (0i64, 0i64, clip_rate)
            };

            let record_in = record_pos;
            let record_out = record_in + dur;
            record_pos = record_out;

            event_counter += 1;
            events.push(EdlEvent {
                number: event_counter,
                reel: name,
                track: "V".to_string(),
                edit_type: EditType::Cut,
                source_in: Timecode::from_frames(src_start, fps_rational, false),
                source_out: Timecode::from_frames(src_start + dur, fps_rational, false),
                record_in: Timecode::from_frames(record_in, fps_rational, false),
                record_out: Timecode::from_frames(record_out, fps_rational, false),
                transition_duration: None,
                motion_effect: None,
                comments: Vec::new(),
                metadata: HashMap::new(),
            });
        } else if child_schema.starts_with("Transition") {
            // OTIO Transition has in_offset + out_offset (both RationalTime)
            let in_offset = child
                .get("in_offset")
                .map(|v| {
                    rational_time(v)
                        .map(|(val, _)| val.round() as i64)
                        .unwrap_or(0)
                })
                .unwrap_or(0);
            let out_offset = child
                .get("out_offset")
                .map(|v| {
                    rational_time(v)
                        .map(|(val, _)| val.round() as i64)
                        .unwrap_or(0)
                })
                .unwrap_or(0);
            let total_dur = in_offset + out_offset;

            // Transition sits between clips; its record-in is the current playhead
            // minus in_offset (it overlaps the preceding clip). For EDL mapping we
            // record it at the current playhead and do NOT advance the playhead.
            let record_in = record_pos;
            let record_out = record_in + total_dur;

            event_counter += 1;
            events.push(EdlEvent {
                number: event_counter,
                reel: "AX".to_string(),
                track: "V".to_string(),
                edit_type: EditType::Dissolve,
                source_in: Timecode::from_frames(0, fps_rational, false),
                source_out: Timecode::from_frames(total_dur, fps_rational, false),
                record_in: Timecode::from_frames(record_in, fps_rational, false),
                record_out: Timecode::from_frames(record_out, fps_rational, false),
                transition_duration: Some(total_dur as u32),
                motion_effect: None,
                comments: Vec::new(),
                metadata: HashMap::new(),
            });
            // Transitions do not advance the timeline playhead in OTIO model
        } else if child_schema.starts_with("Gap") {
            // Gap: advance playhead without emitting an event
            let dur = child
                .get("source_range")
                .map(|sr| time_range(sr).map(|(_, d, _)| d).unwrap_or(0))
                .unwrap_or(0);
            record_pos += dur;
        }
        // Other schema types (e.g. Stack, unknown) are skipped
    }

    Ok(Edl {
        title,
        frame_rate: fps_rational,
        drop_frame: false,
        events,
        comments: Vec::new(),
        metadata: HashMap::new(),
    })
}

/// Convert a float fps (e.g. 29.97) to the nearest useful `Rational`.
fn float_to_rational(fps: f64) -> Rational {
    // Common broadcast rates
    let known: &[(f64, i64, i64)] = &[
        (23.976, 24000, 1001),
        (24.0, 24, 1),
        (25.0, 25, 1),
        (29.97, 30000, 1001),
        (30.0, 30, 1),
        (47.952, 48000, 1001),
        (48.0, 48, 1),
        (59.94, 60000, 1001),
        (60.0, 60, 1),
    ];
    for (ref_fps, num, den) in known {
        if (fps - ref_fps).abs() < 0.01 {
            return Rational::new(*num, *den);
        }
    }
    // Fallback: round to nearest integer fps
    Rational::new(fps.round() as i64, 1)
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Write an [`Edl`] as an OpenTimelineIO JSON string.
pub fn write(edl: &Edl) -> EdlResult<String> {
    let fps = edl.frame_rate;
    // Compute float rate for OTIO (OTIO uses float rates)
    let rate_float = fps.to_f64();

    let mut clip_children: Vec<Value> = Vec::new();

    for event in &edl.events {
        match event.edit_type {
            EditType::Cut | EditType::Key => {
                let src_start = event.source_in.to_frames();
                let duration = event.record_out.to_frames() - event.record_in.to_frames();
                let clip = serde_json::json!({
                    "OTIO_SCHEMA": "Clip.1",
                    "name": event.reel,
                    "source_range": make_time_range(src_start, duration, rate_float),
                    "effects": [],
                    "markers": [],
                    "metadata": {}
                });
                clip_children.push(clip);
            }
            EditType::Dissolve | EditType::Wipe => {
                let total_dur = event.record_out.to_frames() - event.record_in.to_frames();
                let half = total_dur / 2;
                let transition = serde_json::json!({
                    "OTIO_SCHEMA": "Transition.1",
                    "name": if event.edit_type == EditType::Dissolve { "Cross Dissolve" } else { "Wipe" },
                    "transition_type": "SMPTE_Dissolve",
                    "in_offset": make_rational_time(half, rate_float),
                    "out_offset": make_rational_time(total_dur - half, rate_float),
                    "metadata": {}
                });
                clip_children.push(transition);
            }
        }
    }

    let video_track = serde_json::json!({
        "OTIO_SCHEMA": "Track.1",
        "name": "Video 1",
        "kind": "Video",
        "children": clip_children,
        "effects": [],
        "markers": [],
        "metadata": {}
    });

    let stack = serde_json::json!({
        "OTIO_SCHEMA": "Stack.1",
        "name": "tracks",
        "children": [video_track],
        "effects": [],
        "markers": [],
        "metadata": {}
    });

    let timeline = serde_json::json!({
        "OTIO_SCHEMA": "Timeline.1",
        "name": edl.title,
        "global_start_time": make_rational_time(0, rate_float),
        "tracks": stack,
        "metadata": {}
    });

    serde_json::to_string_pretty(&timeline)
        .map_err(|e| EdlError::JsonError(format!("JSON serialization error: {}", e)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn fps_24() -> Rational {
        Rational::new(24, 1)
    }

    fn make_tc(frames: i64, fps: Rational) -> Timecode {
        Timecode::from_frames(frames, fps, false)
    }

    fn make_clip_event(
        number: u32,
        reel: &str,
        src_in: i64,
        src_out: i64,
        rec_in: i64,
        rec_out: i64,
        fps: Rational,
    ) -> EdlEvent {
        EdlEvent {
            number,
            reel: reel.to_string(),
            track: "V".to_string(),
            edit_type: EditType::Cut,
            source_in: make_tc(src_in, fps),
            source_out: make_tc(src_out, fps),
            record_in: make_tc(rec_in, fps),
            record_out: make_tc(rec_out, fps),
            transition_duration: None,
            motion_effect: None,
            comments: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_round_trip() {
        let fps = fps_24();
        let mut edl = Edl::new("OTIO RoundTrip".to_string(), fps, false);
        edl.add_event(make_clip_event(1, "clip_a", 0, 100, 0, 100, fps));
        edl.add_event(make_clip_event(2, "clip_b", 200, 350, 100, 250, fps));

        let json = write(&edl).expect("write should succeed");
        let edl2 = parse(&json).expect("parse should succeed");

        assert_eq!(edl2.title, "OTIO RoundTrip");
        assert_eq!(edl2.events.len(), 2);

        let e0 = &edl2.events[0];
        let e0_orig = &edl.events[0];
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
        assert_eq!(e0.reel, "clip_a");

        let e1 = &edl2.events[1];
        assert_eq!(e1.reel, "clip_b");
        assert!(
            (e1.record_in.to_frames() - 100).abs() <= 1,
            "clip_b record_in should be 100, got {}",
            e1.record_in.to_frames()
        );
    }

    // -----------------------------------------------------------------------
    // Empty timeline
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_empty_timeline() {
        let json = r#"{
  "OTIO_SCHEMA": "Timeline.1",
  "name": "Empty",
  "global_start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": 24.0, "value": 0.0},
  "tracks": {
    "OTIO_SCHEMA": "Stack.1",
    "children": [
      {
        "OTIO_SCHEMA": "Track.1",
        "kind": "Video",
        "children": []
      }
    ]
  },
  "metadata": {}
}"#;
        let edl = parse(json).expect("parse should succeed");
        assert_eq!(edl.events.len(), 0);
        assert_eq!(edl.title, "Empty");
    }

    // -----------------------------------------------------------------------
    // Malformed JSON
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_malformed_json() {
        let bad = r#"{ "OTIO_SCHEMA": "Timeline.1", "name": "bad", "#;
        let result = parse(bad);
        assert!(result.is_err(), "malformed JSON should fail");
    }

    // -----------------------------------------------------------------------
    // Wrong schema
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_wrong_schema() {
        let json = r#"{"OTIO_SCHEMA": "Clip.1", "name": "not a timeline"}"#;
        let result = parse(json);
        assert!(result.is_err(), "non-Timeline schema should fail");
    }

    // -----------------------------------------------------------------------
    // Gap advances playhead
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_gap_advances_record_position() {
        let json = r#"{
  "OTIO_SCHEMA": "Timeline.1",
  "name": "Gap Test",
  "global_start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": 24.0, "value": 0.0},
  "tracks": {
    "OTIO_SCHEMA": "Stack.1",
    "children": [
      {
        "OTIO_SCHEMA": "Track.1",
        "kind": "Video",
        "children": [
          {
            "OTIO_SCHEMA": "Clip.1",
            "name": "clip_a",
            "source_range": {
              "OTIO_SCHEMA": "TimeRange.1",
              "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": 24.0, "value": 0.0},
              "duration":   {"OTIO_SCHEMA": "RationalTime.1", "rate": 24.0, "value": 50.0}
            }
          },
          {
            "OTIO_SCHEMA": "Gap.1",
            "name": "gap",
            "source_range": {
              "OTIO_SCHEMA": "TimeRange.1",
              "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": 24.0, "value": 0.0},
              "duration":   {"OTIO_SCHEMA": "RationalTime.1", "rate": 24.0, "value": 25.0}
            }
          },
          {
            "OTIO_SCHEMA": "Clip.1",
            "name": "clip_b",
            "source_range": {
              "OTIO_SCHEMA": "TimeRange.1",
              "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": 24.0, "value": 0.0},
              "duration":   {"OTIO_SCHEMA": "RationalTime.1", "rate": 24.0, "value": 75.0}
            }
          }
        ]
      }
    ]
  },
  "metadata": {}
}"#;
        let edl = parse(json).expect("parse should succeed");
        // Gap does not produce an event
        assert_eq!(edl.events.len(), 2, "gap should not produce event");
        // clip_a: record 0..50
        assert_eq!(edl.events[0].record_in.to_frames(), 0);
        assert_eq!(edl.events[0].record_out.to_frames(), 50);
        // clip_b: record 75..150 (50 clip + 25 gap)
        assert_eq!(edl.events[1].record_in.to_frames(), 75);
        assert_eq!(edl.events[1].record_out.to_frames(), 150);
    }

    // -----------------------------------------------------------------------
    // Transition produces dissolve event
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_transition() {
        let json = r#"{
  "OTIO_SCHEMA": "Timeline.1",
  "name": "Trans Test",
  "global_start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": 24.0, "value": 0.0},
  "tracks": {
    "OTIO_SCHEMA": "Stack.1",
    "children": [
      {
        "OTIO_SCHEMA": "Track.1",
        "kind": "Video",
        "children": [
          {
            "OTIO_SCHEMA": "Clip.1",
            "name": "clip_a",
            "source_range": {
              "OTIO_SCHEMA": "TimeRange.1",
              "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": 24.0, "value": 0.0},
              "duration":   {"OTIO_SCHEMA": "RationalTime.1", "rate": 24.0, "value": 100.0}
            }
          },
          {
            "OTIO_SCHEMA": "Transition.1",
            "name": "Cross Dissolve",
            "transition_type": "SMPTE_Dissolve",
            "in_offset":  {"OTIO_SCHEMA": "RationalTime.1", "rate": 24.0, "value": 12.0},
            "out_offset": {"OTIO_SCHEMA": "RationalTime.1", "rate": 24.0, "value": 12.0}
          },
          {
            "OTIO_SCHEMA": "Clip.1",
            "name": "clip_b",
            "source_range": {
              "OTIO_SCHEMA": "TimeRange.1",
              "start_time": {"OTIO_SCHEMA": "RationalTime.1", "rate": 24.0, "value": 0.0},
              "duration":   {"OTIO_SCHEMA": "RationalTime.1", "rate": 24.0, "value": 100.0}
            }
          }
        ]
      }
    ]
  },
  "metadata": {}
}"#;
        let edl = parse(json).expect("parse should succeed");
        assert_eq!(edl.events.len(), 3);
        assert_eq!(edl.events[0].edit_type, EditType::Cut);
        assert_eq!(edl.events[1].edit_type, EditType::Dissolve);
        assert_eq!(edl.events[1].transition_duration, Some(24));
        assert_eq!(edl.events[2].edit_type, EditType::Cut);
    }

    // -----------------------------------------------------------------------
    // Float FPS detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_float_to_rational_known_rates() {
        let r = float_to_rational(24.0);
        assert_eq!(r.num, 24);
        assert_eq!(r.den, 1);

        let r = float_to_rational(29.97);
        assert_eq!(r.num, 30000);
        assert_eq!(r.den, 1001);

        let r = float_to_rational(25.0);
        assert_eq!(r.num, 25);
        assert_eq!(r.den, 1);
    }

    // -----------------------------------------------------------------------
    // Write output structure
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_produces_valid_json() {
        let fps = fps_24();
        let mut edl = Edl::new("JSON Write Test".to_string(), fps, false);
        edl.add_event(make_clip_event(1, "reel_1", 0, 48, 0, 48, fps));

        let json_str = write(&edl).expect("write should succeed");
        let parsed: Value = serde_json::from_str(&json_str).expect("output should be valid JSON");

        assert_eq!(schema(&parsed), "Timeline.1");
        assert_eq!(
            parsed.get("name").and_then(|v| v.as_str()),
            Some("JSON Write Test")
        );

        let tracks = parsed.get("tracks").expect("tracks should exist");
        let children = tracks
            .get("children")
            .and_then(|v| v.as_array())
            .expect("stack children should exist");
        assert!(!children.is_empty(), "at least one track");

        let track_clips = children[0]
            .get("children")
            .and_then(|v| v.as_array())
            .expect("track children should exist");
        assert_eq!(track_clips.len(), 1);
        assert_eq!(schema(&track_clips[0]), "Clip.1");
        assert_eq!(
            track_clips[0].get("name").and_then(|v| v.as_str()),
            Some("reel_1")
        );
    }
}
