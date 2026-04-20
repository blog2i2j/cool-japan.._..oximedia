//! Integration tests for the DASH MPD manifest emitter.
//!
//! Verifies that [`emit_mpd`] produces correct, DASH-IF IOP-compliant XML
//! with `SegmentTemplate`, `$Number$`/`$Time$` placeholders, and proper
//! `Representation` IDs for multi-bitrate streams.

use oximedia_container::{
    dash::{
        DashAdaptationSet, DashManifestConfig, DashRepresentation, DashSegmentTemplate,
        DashSegmentTimeline, DashSegmentTimelineEntry,
    },
    emit_mpd,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn two_rep_config() -> DashManifestConfig {
    DashManifestConfig {
        media_presentation_duration: "PT30S".to_string(),
        min_buffer_time: "PT4S".to_string(),
        base_url: None,
        adaptation_sets: vec![DashAdaptationSet {
            id: 1,
            content_type: "video".to_string(),
            mime_type: "video/mp4".to_string(),
            codecs: "av01.0.08M.10".to_string(),
            representations: vec![
                DashRepresentation {
                    id: "video_1080p".to_string(),
                    bandwidth: 4_000_000,
                    width: Some(1920),
                    height: Some(1080),
                    frame_rate: Some("30000/1001".to_string()),
                    audio_sampling_rate: None,
                    segment_template: DashSegmentTemplate {
                        timescale: 90000,
                        duration: Some(270000),
                        initialization: "1080p/init.mp4".to_string(),
                        media: "1080p/seg$Number$.m4s".to_string(),
                        start_number: 1,
                        segment_timeline: None,
                    },
                },
                DashRepresentation {
                    id: "video_720p".to_string(),
                    bandwidth: 2_000_000,
                    width: Some(1280),
                    height: Some(720),
                    frame_rate: Some("30000/1001".to_string()),
                    audio_sampling_rate: None,
                    segment_template: DashSegmentTemplate {
                        timescale: 90000,
                        duration: Some(270000),
                        initialization: "720p/init.mp4".to_string(),
                        media: "720p/seg$Number$.m4s".to_string(),
                        start_number: 1,
                        segment_timeline: None,
                    },
                },
            ],
        }],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_dash_on_demand_profile_present() {
    let config = two_rep_config();
    let mpd = emit_mpd(&config);
    assert!(
        mpd.contains("urn:mpeg:dash:profile:isoff-on-demand:2011"),
        "MPD must contain the DASH-IF IOP on-demand profile URI.\nGot:\n{mpd}"
    );
}

#[test]
fn test_dash_segment_template_element_present() {
    let config = two_rep_config();
    let mpd = emit_mpd(&config);
    assert!(
        mpd.contains("SegmentTemplate"),
        "MPD must contain a <SegmentTemplate> element.\nGot:\n{mpd}"
    );
}

#[test]
fn test_dash_number_placeholder_present() {
    let config = two_rep_config();
    let mpd = emit_mpd(&config);
    assert!(
        mpd.contains("$Number$"),
        "SegmentTemplate media attribute must contain $Number$.\nGot:\n{mpd}"
    );
}

#[test]
fn test_dash_both_representation_ids_present() {
    let config = two_rep_config();
    let mpd = emit_mpd(&config);
    assert!(
        mpd.contains("video_1080p"),
        "MPD must contain representation id='video_1080p'.\nGot:\n{mpd}"
    );
    assert!(
        mpd.contains("video_720p"),
        "MPD must contain representation id='video_720p'.\nGot:\n{mpd}"
    );
}

#[test]
fn test_dash_correct_profiles_attribute_key() {
    let config = two_rep_config();
    let mpd = emit_mpd(&config);
    assert!(
        mpd.contains("profiles="),
        "Root MPD element must have a profiles= attribute.\nGot:\n{mpd}"
    );
}

#[test]
fn test_dash_type_static() {
    let config = two_rep_config();
    let mpd = emit_mpd(&config);
    assert!(
        mpd.contains("type=\"static\""),
        "MPD must declare type=\"static\" for on-demand content.\nGot:\n{mpd}"
    );
}

#[test]
fn test_dash_media_presentation_duration() {
    let config = two_rep_config();
    let mpd = emit_mpd(&config);
    assert!(
        mpd.contains("mediaPresentationDuration=\"PT30S\""),
        "MPD must carry the configured mediaPresentationDuration.\nGot:\n{mpd}"
    );
}

#[test]
fn test_dash_period_element_present() {
    let config = two_rep_config();
    let mpd = emit_mpd(&config);
    assert!(
        mpd.contains("<Period"),
        "MPD must contain a <Period> element.\nGot:\n{mpd}"
    );
}

#[test]
fn test_dash_adaptation_set_content_type() {
    let config = two_rep_config();
    let mpd = emit_mpd(&config);
    assert!(
        mpd.contains("contentType=\"video\""),
        "AdaptationSet must carry contentType=\"video\".\nGot:\n{mpd}"
    );
}

#[test]
fn test_dash_bandwidth_attributes() {
    let config = two_rep_config();
    let mpd = emit_mpd(&config);
    assert!(
        mpd.contains("bandwidth=\"4000000\""),
        "1080p Representation must carry bandwidth=\"4000000\".\nGot:\n{mpd}"
    );
    assert!(
        mpd.contains("bandwidth=\"2000000\""),
        "720p Representation must carry bandwidth=\"2000000\".\nGot:\n{mpd}"
    );
}

#[test]
fn test_dash_base_url_included() {
    let mut config = two_rep_config();
    config.base_url = Some("https://cdn.example.com/stream/".to_string());
    let mpd = emit_mpd(&config);
    assert!(
        mpd.contains("<BaseURL>https://cdn.example.com/stream/</BaseURL>"),
        "MPD must include configured BaseURL.\nGot:\n{mpd}"
    );
}

#[test]
fn test_dash_no_base_url_when_none() {
    let config = two_rep_config(); // no base_url set
    let mpd = emit_mpd(&config);
    assert!(
        !mpd.contains("<BaseURL>"),
        "MPD must NOT include <BaseURL> when none is configured.\nGot:\n{mpd}"
    );
}

#[test]
fn test_dash_audio_representation() {
    let config = DashManifestConfig {
        media_presentation_duration: "PT10S".to_string(),
        min_buffer_time: "PT2S".to_string(),
        base_url: None,
        adaptation_sets: vec![DashAdaptationSet {
            id: 2,
            content_type: "audio".to_string(),
            mime_type: "audio/mp4".to_string(),
            codecs: "opus".to_string(),
            representations: vec![DashRepresentation {
                id: "audio_opus_stereo".to_string(),
                bandwidth: 128_000,
                width: None,
                height: None,
                frame_rate: None,
                audio_sampling_rate: Some(48000),
                segment_template: DashSegmentTemplate {
                    timescale: 48000,
                    duration: Some(96000),
                    initialization: "audio/init.mp4".to_string(),
                    media: "audio/seg$Number$.m4s".to_string(),
                    start_number: 1,
                    segment_timeline: None,
                },
            }],
        }],
    };
    let mpd = emit_mpd(&config);
    assert!(
        mpd.contains("audio_opus_stereo"),
        "audio repr ID must be present"
    );
    assert!(
        mpd.contains("audioSamplingRate=\"48000\""),
        "audio sampling rate must be present"
    );
    assert!(
        mpd.contains("contentType=\"audio\""),
        "contentType audio must be present"
    );
}

#[test]
fn test_dash_segment_timeline_emitted() {
    let config = DashManifestConfig {
        media_presentation_duration: "PT10S".to_string(),
        min_buffer_time: "PT2S".to_string(),
        base_url: None,
        adaptation_sets: vec![DashAdaptationSet {
            id: 1,
            content_type: "video".to_string(),
            mime_type: "video/mp4".to_string(),
            codecs: "av01.0.04M.08".to_string(),
            representations: vec![DashRepresentation {
                id: "v1".to_string(),
                bandwidth: 1_000_000,
                width: Some(1280),
                height: Some(720),
                frame_rate: None,
                audio_sampling_rate: None,
                segment_template: DashSegmentTemplate {
                    timescale: 90000,
                    duration: None,
                    initialization: "v1/init.mp4".to_string(),
                    media: "v1/seg$Time$.m4s".to_string(),
                    start_number: 1,
                    segment_timeline: Some(DashSegmentTimeline {
                        entries: vec![
                            DashSegmentTimelineEntry {
                                t: Some(0),
                                d: 270000,
                                r: 4,
                            },
                            DashSegmentTimelineEntry {
                                t: None,
                                d: 180000,
                                r: 0,
                            },
                        ],
                    }),
                },
            }],
        }],
    };
    let mpd = emit_mpd(&config);
    assert!(
        mpd.contains("SegmentTimeline"),
        "SegmentTimeline element must be present"
    );
    assert!(
        mpd.contains("<S"),
        "S elements must be present inside SegmentTimeline"
    );
    assert!(
        mpd.contains("$Time$"),
        "media template with $Time$ should be preserved"
    );
    assert!(
        mpd.contains("d=\"270000\""),
        "segment duration d=270000 must appear"
    );
    assert!(mpd.contains("r=\"4\""), "repeat count r=4 must appear");
}

#[test]
fn test_dash_xml_declaration_present() {
    let config = two_rep_config();
    let mpd = emit_mpd(&config);
    assert!(
        mpd.starts_with("<?xml"),
        "MPD must start with XML declaration.\nGot:\n{mpd}"
    );
    assert!(
        mpd.contains("encoding=\"UTF-8\""),
        "XML declaration must include encoding=UTF-8.\nGot:\n{mpd}"
    );
}

#[test]
fn test_dash_start_number_attribute() {
    let mut config = two_rep_config();
    // Change start_number to 100
    for adapt in &mut config.adaptation_sets {
        for repr in &mut adapt.representations {
            repr.segment_template.start_number = 100;
        }
    }
    let mpd = emit_mpd(&config);
    assert!(
        mpd.contains("startNumber=\"100\""),
        "SegmentTemplate must carry the configured startNumber.\nGot:\n{mpd}"
    );
}
