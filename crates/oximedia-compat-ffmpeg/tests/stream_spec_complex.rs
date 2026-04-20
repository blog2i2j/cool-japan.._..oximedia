//! Integration tests for the extended `-map` stream selector parsing.

use oximedia_compat_ffmpeg::stream_spec::{StreamSelector, StreamSpecError, StreamType};

#[test]
fn test_all_streams_from_file0() {
    let sel = StreamSelector::parse("0").expect("parse 0");
    assert_eq!(
        sel,
        StreamSelector::All {
            file_idx: 0,
            stream_type: None,
            stream_idx: None,
        }
    );
}

#[test]
fn test_all_video_from_file0() {
    let sel = StreamSelector::parse("0:v").expect("parse 0:v");
    assert_eq!(
        sel,
        StreamSelector::All {
            file_idx: 0,
            stream_type: Some(StreamType::Video),
            stream_idx: None,
        }
    );
}

#[test]
fn test_first_video_from_file0() {
    let sel = StreamSelector::parse("0:v:0").expect("parse 0:v:0");
    assert_eq!(
        sel,
        StreamSelector::All {
            file_idx: 0,
            stream_type: Some(StreamType::Video),
            stream_idx: Some(0),
        }
    );
}

#[test]
fn test_second_audio_from_file1() {
    let sel = StreamSelector::parse("1:a:1").expect("parse 1:a:1");
    assert_eq!(
        sel,
        StreamSelector::All {
            file_idx: 1,
            stream_type: Some(StreamType::Audio),
            stream_idx: Some(1),
        }
    );
}

#[test]
fn test_negative_map_excludes() {
    let sel = StreamSelector::parse("-0:a:1").expect("parse -0:a:1");
    match &sel {
        StreamSelector::Exclude(inner) => {
            assert_eq!(
                **inner,
                StreamSelector::All {
                    file_idx: 0,
                    stream_type: Some(StreamType::Audio),
                    stream_idx: Some(1),
                }
            );
        }
        other => panic!("expected Exclude, got {:?}", other),
    }
}

#[test]
fn test_filter_label() {
    let sel = StreamSelector::parse("[out_v]").expect("parse [out_v]");
    assert_eq!(
        sel,
        StreamSelector::ByLabel {
            label: "out_v".to_string(),
        }
    );
}

#[test]
fn test_metadata_selector() {
    let sel = StreamSelector::parse("0:m:language:eng").expect("parse metadata");
    assert_eq!(
        sel,
        StreamSelector::ByMetadata {
            file_idx: 0,
            stream_type: None,
            key: "language".to_string(),
            value: "eng".to_string(),
        }
    );
}

#[test]
fn test_subtitle_streams() {
    let sel = StreamSelector::parse("0:s").expect("parse subtitles");
    assert_eq!(
        sel,
        StreamSelector::All {
            file_idx: 0,
            stream_type: Some(StreamType::Subtitle),
            stream_idx: None,
        }
    );
}

#[test]
fn test_display_roundtrip() {
    let specs = ["0", "0:v", "0:v:0", "1:a:1", "[out_v]", "0:m:language:eng"];
    for &spec in &specs {
        let sel = StreamSelector::parse(spec).expect("parse");
        assert_eq!(
            sel.to_string(),
            spec,
            "display roundtrip failed for '{}'",
            spec
        );
    }
}

#[test]
fn test_invalid_file_index() {
    let err = StreamSelector::parse("x:v").expect_err("should fail");
    assert!(
        matches!(err, StreamSpecError::InvalidInteger(_)),
        "expected InvalidInteger, got {:?}",
        err
    );
}

#[test]
fn test_invalid_type_letter() {
    let err = StreamSelector::parse("0:z").expect_err("should fail");
    assert!(
        matches!(err, StreamSpecError::UnknownStreamType(_)),
        "expected UnknownStreamType, got {:?}",
        err
    );
}
