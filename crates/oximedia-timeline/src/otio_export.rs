//! OpenTimelineIO JSON export.
//!
//! Exports timelines to the OpenTimelineIO (OTIO) JSON format, supporting
//! tracks, clips, transitions, markers, and effects.

use serde_json::{json, Value};

use crate::clip::MediaSource;
use crate::error::{TimelineError, TimelineResult};
use crate::marker::MarkerType;
use crate::timeline::Timeline;
use crate::track::TrackType;
use crate::transition::TransitionType;

/// OTIO export options.
#[derive(Clone, Debug)]
pub struct OtioExportOptions {
    /// Whether to include markers in the export.
    pub include_markers: bool,
    /// Whether to include transitions.
    pub include_transitions: bool,
    /// Whether to include effects metadata.
    pub include_effects: bool,
    /// Pretty-print the JSON output.
    pub pretty_print: bool,
}

impl Default for OtioExportOptions {
    fn default() -> Self {
        Self {
            include_markers: true,
            include_transitions: true,
            include_effects: true,
            pretty_print: true,
        }
    }
}

/// OTIO exporter.
pub struct OtioExporter;

impl OtioExporter {
    /// Exports the timeline to OTIO JSON format.
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn export(timeline: &Timeline, options: &OtioExportOptions) -> TimelineResult<String> {
        let fps = timeline.frame_rate.num as f64 / timeline.frame_rate.den as f64;

        let mut tracks = Vec::new();

        // Video tracks
        for track in &timeline.video_tracks {
            tracks.push(Self::export_track(
                track,
                TrackType::Video,
                timeline,
                options,
                fps,
            )?);
        }

        // Audio tracks
        for track in &timeline.audio_tracks {
            tracks.push(Self::export_track(
                track,
                TrackType::Audio,
                timeline,
                options,
                fps,
            )?);
        }

        // Markers
        let markers = if options.include_markers {
            Self::export_markers(timeline, fps)
        } else {
            vec![]
        };

        let otio = json!({
            "OTIO_SCHEMA": "Timeline.1",
            "name": timeline.name,
            "global_start_time": {
                "OTIO_SCHEMA": "RationalTime.1",
                "value": 0.0,
                "rate": fps
            },
            "tracks": {
                "OTIO_SCHEMA": "Stack.1",
                "name": "tracks",
                "children": tracks,
                "markers": markers,
                "metadata": {}
            },
            "metadata": {
                "sample_rate": timeline.sample_rate,
                "frame_rate_num": timeline.frame_rate.num,
                "frame_rate_den": timeline.frame_rate.den
            }
        });

        if options.pretty_print {
            serde_json::to_string_pretty(&otio)
                .map_err(|e| TimelineError::ExportError(format!("JSON serialization error: {e}")))
        } else {
            serde_json::to_string(&otio)
                .map_err(|e| TimelineError::ExportError(format!("JSON serialization error: {e}")))
        }
    }

    /// Exports a single track to OTIO JSON.
    #[allow(clippy::cast_precision_loss)]
    fn export_track(
        track: &crate::track::Track,
        track_type: TrackType,
        timeline: &Timeline,
        options: &OtioExportOptions,
        fps: f64,
    ) -> TimelineResult<Value> {
        let kind = match track_type {
            TrackType::Video => "Video",
            TrackType::Audio => "Audio",
            TrackType::Subtitle => "Video", // OTIO doesn't have subtitle kind natively
        };

        let mut children: Vec<Value> = Vec::new();

        // Insert gaps and clips
        let mut current_pos = 0i64;

        for clip in &track.clips {
            // Insert gap if there's space before this clip
            if clip.timeline_in.value() > current_pos {
                let gap_duration = clip.timeline_in.value() - current_pos;
                children.push(json!({
                    "OTIO_SCHEMA": "Gap.1",
                    "name": "",
                    "source_range": {
                        "OTIO_SCHEMA": "TimeRange.1",
                        "start_time": {
                            "OTIO_SCHEMA": "RationalTime.1",
                            "value": 0.0,
                            "rate": fps
                        },
                        "duration": {
                            "OTIO_SCHEMA": "RationalTime.1",
                            "value": gap_duration as f64,
                            "rate": fps
                        }
                    }
                }));
            }

            // Add transition before clip if present
            if options.include_transitions {
                if let Some(transition) = timeline.transitions.get(&clip.id) {
                    let trans_name = match transition.transition_type {
                        TransitionType::Dissolve => "CrossDissolve",
                        TransitionType::DipToBlack => "FadeToBlack",
                        TransitionType::DipToWhite => "FadeToWhite",
                        TransitionType::Wipe => "Wipe",
                        TransitionType::Push => "Push",
                        TransitionType::Slide => "Slide",
                        _ => "CrossDissolve",
                    };
                    children.push(json!({
                        "OTIO_SCHEMA": "Transition.1",
                        "name": trans_name,
                        "transition_type": "SMPTE_Dissolve",
                        "in_offset": {
                            "OTIO_SCHEMA": "RationalTime.1",
                            "value": transition.duration.value() as f64 / 2.0,
                            "rate": fps
                        },
                        "out_offset": {
                            "OTIO_SCHEMA": "RationalTime.1",
                            "value": transition.duration.value() as f64 / 2.0,
                            "rate": fps
                        }
                    }));
                }
            }

            // Build clip effects
            let effects = if options.include_effects {
                Self::export_effects(clip)
            } else {
                vec![]
            };

            // Build media reference
            let media_ref = Self::export_media_reference(&clip.source, fps);

            children.push(json!({
                "OTIO_SCHEMA": "Clip.2",
                "name": clip.name,
                "source_range": {
                    "OTIO_SCHEMA": "TimeRange.1",
                    "start_time": {
                        "OTIO_SCHEMA": "RationalTime.1",
                        "value": clip.source_in.value() as f64,
                        "rate": fps
                    },
                    "duration": {
                        "OTIO_SCHEMA": "RationalTime.1",
                        "value": clip.source_duration().value() as f64,
                        "rate": fps
                    }
                },
                "media_reference": media_ref,
                "effects": effects,
                "metadata": clip.metadata
            }));

            current_pos = clip.timeline_out().value();
        }

        Ok(json!({
            "OTIO_SCHEMA": "Track.1",
            "name": track.name,
            "kind": kind,
            "children": children,
            "metadata": {
                "muted": track.muted,
                "locked": track.locked,
                "volume": track.volume,
                "pan": track.pan
            }
        }))
    }

    /// Exports a media reference to OTIO JSON.
    #[allow(clippy::cast_precision_loss)]
    fn export_media_reference(source: &MediaSource, _fps: f64) -> Value {
        match source {
            MediaSource::File { path, .. } => {
                json!({
                    "OTIO_SCHEMA": "ExternalReference.1",
                    "target_url": path.to_string_lossy(),
                    "available_range": null
                })
            }
            MediaSource::Sequence {
                pattern,
                start,
                end,
            } => {
                json!({
                    "OTIO_SCHEMA": "ImageSequenceReference.1",
                    "target_url_base": pattern,
                    "start_frame": start,
                    "end_frame": end
                })
            }
            MediaSource::Color { rgba } => {
                json!({
                    "OTIO_SCHEMA": "GeneratorReference.1",
                    "generator_kind": "SolidColor",
                    "parameters": {
                        "color": rgba
                    }
                })
            }
            _ => {
                json!({
                    "OTIO_SCHEMA": "MissingReference.1"
                })
            }
        }
    }

    /// Exports clip effects to OTIO JSON.
    fn export_effects(clip: &crate::clip::Clip) -> Vec<Value> {
        let mut effects = Vec::new();

        // Export speed if not normal
        if !clip.speed.is_normal() {
            effects.push(json!({
                "OTIO_SCHEMA": "LinearTimeWarp.1",
                "name": "Speed",
                "time_scalar": clip.speed.value()
            }));
        }

        // Export other effects
        for effect in clip.effects.effects() {
            let mut params = serde_json::Map::new();
            for (key, param) in &effect.parameters {
                let val = match param {
                    crate::effects::EffectParameter::Float(v) => json!(v),
                    crate::effects::EffectParameter::Int(v) => json!(v),
                    crate::effects::EffectParameter::Bool(v) => json!(v),
                    crate::effects::EffectParameter::String(v) => json!(v),
                    crate::effects::EffectParameter::Color(c) => json!(c),
                    crate::effects::EffectParameter::Point2D(p) => json!(p),
                };
                params.insert(key.clone(), val);
            }

            effects.push(json!({
                "OTIO_SCHEMA": "Effect.1",
                "name": effect.name,
                "effect_name": format!("{:?}", effect.effect_type),
                "metadata": params
            }));
        }

        effects
    }

    /// Exports markers to OTIO JSON.
    #[allow(clippy::cast_precision_loss)]
    fn export_markers(timeline: &Timeline, fps: f64) -> Vec<Value> {
        timeline
            .markers
            .markers()
            .iter()
            .map(|marker| {
                let color = match marker.marker_type {
                    MarkerType::Standard => "BLUE",
                    MarkerType::Chapter => "GREEN",
                    MarkerType::Comment => "YELLOW",
                    MarkerType::Todo => "RED",
                    MarkerType::WebLink => "PURPLE",
                };

                let dur_val = marker.duration.map_or(0.0, |d| d.value() as f64);

                json!({
                    "OTIO_SCHEMA": "Marker.2",
                    "name": marker.name,
                    "marked_range": {
                        "OTIO_SCHEMA": "TimeRange.1",
                        "start_time": {
                            "OTIO_SCHEMA": "RationalTime.1",
                            "value": marker.position.value() as f64,
                            "rate": fps
                        },
                        "duration": {
                            "OTIO_SCHEMA": "RationalTime.1",
                            "value": dur_val,
                            "rate": fps
                        }
                    },
                    "color": color,
                    "comment": marker.comment,
                    "metadata": marker.metadata
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::Clip;
    use crate::effects::{Effect, EffectType};

    fn tmp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("oximedia-timeline-otio-{name}"))
    }
    use crate::marker::Marker;
    use crate::transition::Transition;
    use crate::types::{Duration, Position};
    use oximedia_core::Rational;
    use std::path::PathBuf;

    fn create_test_timeline() -> Timeline {
        Timeline::new("Test Timeline", Rational::new(24, 1), 48000).expect("should succeed in test")
    }

    #[test]
    fn test_otio_export_empty_timeline() {
        let timeline = create_test_timeline();
        let options = OtioExportOptions::default();
        let result = OtioExporter::export(&timeline, &options);
        assert!(result.is_ok());
        let json_str = result.expect("should succeed in test");
        let parsed: Value = serde_json::from_str(&json_str).expect("should succeed in test");
        assert_eq!(parsed["OTIO_SCHEMA"], "Timeline.1");
        assert_eq!(parsed["name"], "Test Timeline");
    }

    #[test]
    fn test_otio_export_with_clips() {
        let mut timeline = create_test_timeline();
        let track_id = timeline
            .add_video_track("V1")
            .expect("should succeed in test");
        let clip = Clip::new(
            "TestClip".to_string(),
            MediaSource::file(tmp_path("test.mov")),
            Position::new(0),
            Position::new(100),
            Position::new(0),
        )
        .expect("should succeed in test");
        timeline
            .add_clip(track_id, clip)
            .expect("should succeed in test");

        let options = OtioExportOptions::default();
        let json_str = OtioExporter::export(&timeline, &options).expect("should succeed in test");
        let parsed: Value = serde_json::from_str(&json_str).expect("should succeed in test");

        let tracks = &parsed["tracks"]["children"];
        assert_eq!(tracks.as_array().map_or(0, |a| a.len()), 1);

        let track_children = &tracks[0]["children"];
        assert!(track_children.as_array().map_or(0, |a| a.len()) >= 1);
    }

    #[test]
    fn test_otio_export_with_gap() {
        let mut timeline = create_test_timeline();
        let track_id = timeline
            .add_video_track("V1")
            .expect("should succeed in test");
        // Clip starting at frame 50, leaving a gap of 50 frames
        let clip = Clip::new(
            "GapClip".to_string(),
            MediaSource::black(),
            Position::new(0),
            Position::new(50),
            Position::new(50),
        )
        .expect("should succeed in test");
        timeline
            .add_clip(track_id, clip)
            .expect("should succeed in test");

        let options = OtioExportOptions::default();
        let json_str = OtioExporter::export(&timeline, &options).expect("should succeed in test");
        let parsed: Value = serde_json::from_str(&json_str).expect("should succeed in test");

        let children = &parsed["tracks"]["children"][0]["children"];
        // Should have gap + clip
        assert_eq!(children.as_array().map_or(0, |a| a.len()), 2);
        assert_eq!(children[0]["OTIO_SCHEMA"], "Gap.1");
    }

    #[test]
    fn test_otio_export_with_markers() {
        let mut timeline = create_test_timeline();
        timeline
            .markers
            .add_marker(Marker::new(Position::new(50), "TestMarker".to_string()));

        let options = OtioExportOptions::default();
        let json_str = OtioExporter::export(&timeline, &options).expect("should succeed in test");
        let parsed: Value = serde_json::from_str(&json_str).expect("should succeed in test");

        let markers = &parsed["tracks"]["markers"];
        assert_eq!(markers.as_array().map_or(0, |a| a.len()), 1);
        assert_eq!(markers[0]["name"], "TestMarker");
    }

    #[test]
    fn test_otio_export_with_transitions() {
        let mut timeline = create_test_timeline();
        let track_id = timeline
            .add_video_track("V1")
            .expect("should succeed in test");
        let clip = Clip::new(
            "Clip1".to_string(),
            MediaSource::black(),
            Position::new(0),
            Position::new(100),
            Position::new(0),
        )
        .expect("should succeed in test");
        let clip_id = clip.id;
        timeline
            .add_clip(track_id, clip)
            .expect("should succeed in test");
        timeline
            .add_transition(clip_id, Transition::dissolve(Duration::new(24)))
            .expect("should succeed in test");

        let options = OtioExportOptions::default();
        let json_str = OtioExporter::export(&timeline, &options).expect("should succeed in test");
        assert!(json_str.contains("Transition.1"));
        assert!(json_str.contains("CrossDissolve"));
    }

    #[test]
    fn test_otio_export_with_effects() {
        let mut timeline = create_test_timeline();
        let track_id = timeline
            .add_video_track("V1")
            .expect("should succeed in test");
        let mut clip = Clip::new(
            "Clip1".to_string(),
            MediaSource::black(),
            Position::new(0),
            Position::new(100),
            Position::new(0),
        )
        .expect("should succeed in test");
        clip.effects
            .add_effect(Effect::new("Blur".to_string(), EffectType::Blur));
        timeline
            .add_clip(track_id, clip)
            .expect("should succeed in test");

        let options = OtioExportOptions::default();
        let json_str = OtioExporter::export(&timeline, &options).expect("should succeed in test");
        assert!(json_str.contains("Blur"));
        assert!(json_str.contains("Effect.1"));
    }

    #[test]
    fn test_otio_export_no_markers_option() {
        let mut timeline = create_test_timeline();
        timeline
            .markers
            .add_marker(Marker::new(Position::new(50), "Hidden".to_string()));

        let options = OtioExportOptions {
            include_markers: false,
            ..Default::default()
        };
        let json_str = OtioExporter::export(&timeline, &options).expect("should succeed in test");
        let parsed: Value = serde_json::from_str(&json_str).expect("should succeed in test");
        let markers = &parsed["tracks"]["markers"];
        assert_eq!(markers.as_array().map_or(0, |a| a.len()), 0);
    }

    #[test]
    fn test_otio_export_compact() {
        let timeline = create_test_timeline();
        let options = OtioExportOptions {
            pretty_print: false,
            ..Default::default()
        };
        let json_str = OtioExporter::export(&timeline, &options).expect("should succeed in test");
        // Compact JSON should not contain newlines
        assert!(!json_str.contains('\n'));
    }

    #[test]
    fn test_otio_export_media_references() {
        let mut timeline = create_test_timeline();
        let track_id = timeline
            .add_video_track("V1")
            .expect("should succeed in test");

        // File source
        let clip1 = Clip::new(
            "FileClip".to_string(),
            MediaSource::file(tmp_path("test.mov")),
            Position::new(0),
            Position::new(50),
            Position::new(0),
        )
        .expect("should succeed in test");
        timeline
            .add_clip(track_id, clip1)
            .expect("should succeed in test");

        // Color source
        let clip2 = Clip::new(
            "ColorClip".to_string(),
            MediaSource::color(1.0, 0.0, 0.0, 1.0),
            Position::new(0),
            Position::new(50),
            Position::new(50),
        )
        .expect("should succeed in test");
        timeline
            .add_clip(track_id, clip2)
            .expect("should succeed in test");

        let options = OtioExportOptions::default();
        let json_str = OtioExporter::export(&timeline, &options).expect("should succeed in test");
        assert!(json_str.contains("ExternalReference.1"));
        assert!(json_str.contains("GeneratorReference.1"));
    }

    #[test]
    fn test_otio_export_audio_tracks() {
        let mut timeline = create_test_timeline();
        let track_id = timeline
            .add_audio_track("A1")
            .expect("should succeed in test");
        let clip = Clip::new(
            "AudioClip".to_string(),
            MediaSource::black(),
            Position::new(0),
            Position::new(100),
            Position::new(0),
        )
        .expect("should succeed in test");
        timeline
            .add_clip(track_id, clip)
            .expect("should succeed in test");

        let options = OtioExportOptions::default();
        let json_str = OtioExporter::export(&timeline, &options).expect("should succeed in test");
        let parsed: Value = serde_json::from_str(&json_str).expect("should succeed in test");
        let track = &parsed["tracks"]["children"][0];
        assert_eq!(track["kind"], "Audio");
    }

    #[test]
    fn test_otio_export_options_default() {
        let opts = OtioExportOptions::default();
        assert!(opts.include_markers);
        assert!(opts.include_transitions);
        assert!(opts.include_effects);
        assert!(opts.pretty_print);
    }
}
