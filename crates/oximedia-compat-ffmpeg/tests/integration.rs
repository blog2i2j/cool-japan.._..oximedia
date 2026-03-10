//! Integration tests for the full parse_and_translate pipeline.

use oximedia_compat_ffmpeg::diagnostics::DiagnosticKind;
use oximedia_compat_ffmpeg::parse_and_translate;
use oximedia_compat_ffmpeg::ParsedFilter;

fn sv(args: &[&str]) -> Vec<String> {
    args.iter().map(|s| s.to_string()).collect()
}

#[test]
fn test_full_h264_to_webm_translation() {
    let args = sv(&[
        "-i",
        "input.mkv",
        "-c:v",
        "libx264",
        "-crf",
        "28",
        "-c:a",
        "aac",
        "output.webm",
    ]);
    let result = parse_and_translate(&args);

    assert_eq!(result.jobs.len(), 1, "should produce one job");
    let job = &result.jobs[0];
    assert_eq!(job.input_path, "input.mkv");
    assert_eq!(job.output_path, "output.webm");

    // Patent codecs should be substituted to free equivalents
    assert_eq!(job.video_codec.as_deref(), Some("av1"), "libx264 → av1");
    assert_eq!(job.audio_codec.as_deref(), Some("opus"), "aac → opus");

    // CRF should be preserved
    assert!(
        job.crf.map(|c| (c - 28.0).abs() < 0.001).unwrap_or(false),
        "crf=28 should be carried through"
    );

    // Should have patent substitution diagnostics for both codecs
    let has_video_sub = result.diagnostics.iter().any(|d| {
        matches!(&d.kind, DiagnosticKind::PatentCodecSubstituted { from, .. } if from == "libx264")
    });
    let has_audio_sub = result.diagnostics.iter().any(
        |d| matches!(&d.kind, DiagnosticKind::PatentCodecSubstituted { from, .. } if from == "aac"),
    );
    assert!(has_video_sub, "should warn about libx264 substitution");
    assert!(has_audio_sub, "should warn about aac substitution");
}

#[test]
fn test_native_av1_no_patent_warnings() {
    let args = sv(&[
        "-i",
        "input.mkv",
        "-c:v",
        "av1",
        "-c:a",
        "opus",
        "output.webm",
    ]);
    let result = parse_and_translate(&args);

    assert_eq!(result.jobs.len(), 1);
    assert_eq!(result.jobs[0].video_codec.as_deref(), Some("av1"));
    assert_eq!(result.jobs[0].audio_codec.as_deref(), Some("opus"));

    let patent_warnings: Vec<_> = result
        .diagnostics
        .iter()
        .filter(|d| matches!(&d.kind, DiagnosticKind::PatentCodecSubstituted { .. }))
        .collect();
    assert!(
        patent_warnings.is_empty(),
        "native codecs should produce no patent warnings"
    );
}

#[test]
fn test_filter_translation_scale_and_fps() {
    let args = sv(&["-i", "in.mkv", "-vf", "scale=1280:720,fps=30", "out.webm"]);
    let result = parse_and_translate(&args);

    assert_eq!(result.jobs.len(), 1);
    let job = &result.jobs[0];
    assert!(!job.video_filters.is_empty(), "should have video filters");
    assert_eq!(job.video_filters.len(), 2, "should have scale + fps");

    assert!(
        matches!(
            &job.video_filters[0],
            ParsedFilter::Scale { w: 1280, h: 720 }
        ),
        "first filter should be Scale(1280, 720)"
    );
    assert!(
        matches!(&job.video_filters[1], ParsedFilter::Fps { rate } if (*rate - 30.0).abs() < 0.01),
        "second filter should be Fps(30)"
    );
}

#[test]
fn test_copy_codec_passthrough() {
    let args = sv(&["-i", "in.mkv", "-c:v", "copy", "-c:a", "copy", "out.mkv"]);
    let result = parse_and_translate(&args);

    assert_eq!(result.jobs.len(), 1);
    let job = &result.jobs[0];
    assert_eq!(
        job.video_codec.as_deref(),
        Some("copy"),
        "copy should pass through for video"
    );
    assert_eq!(
        job.audio_codec.as_deref(),
        Some("copy"),
        "copy should pass through for audio"
    );

    let patent_count = result
        .diagnostics
        .iter()
        .filter(|d| matches!(&d.kind, DiagnosticKind::PatentCodecSubstituted { .. }))
        .count();
    assert_eq!(
        patent_count, 0,
        "copy passthrough should produce no patent warnings"
    );
}

#[test]
fn test_multi_input_map_spec() {
    let args = sv(&[
        "-i",
        "video.mkv",
        "-i",
        "audio.mkv",
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "vp9",
        "-c:a",
        "vorbis",
        "output.webm",
    ]);
    let result = parse_and_translate(&args);

    assert!(!result.has_errors(), "should not have errors");
    assert_eq!(result.jobs.len(), 1);
    let job = &result.jobs[0];
    assert_eq!(job.map.len(), 2, "should have two map entries");
    assert_eq!(job.video_codec.as_deref(), Some("vp9"));
    assert_eq!(job.audio_codec.as_deref(), Some("vorbis"));
}

#[test]
fn test_no_input_produces_error() {
    let args = sv(&["output.webm"]);
    let result = parse_and_translate(&args);

    assert!(result.has_errors(), "missing -i should produce an error");
    assert!(result.jobs.is_empty(), "no jobs when input is missing");
}

#[test]
fn test_no_output_produces_error() {
    let args = sv(&["-i", "input.mkv"]);
    let result = parse_and_translate(&args);

    assert!(
        result.has_errors(),
        "missing output path should produce an error"
    );
    assert!(result.jobs.is_empty(), "no jobs when output is missing");
}

#[test]
fn test_overwrite_flag_propagated() {
    let args = sv(&["-y", "-i", "in.mkv", "out.webm"]);
    let result = parse_and_translate(&args);

    assert!(!result.has_errors());
    assert!(
        result.jobs[0].overwrite,
        "overwrite flag should be propagated to job"
    );
}

#[test]
fn test_seek_propagated_from_pre_input() {
    let args = sv(&["-ss", "00:01:00", "-i", "in.mkv", "out.webm"]);
    let result = parse_and_translate(&args);

    assert!(!result.has_errors());
    assert_eq!(
        result.jobs[0].seek.as_deref(),
        Some("00:01:00"),
        "pre-input seek should be propagated"
    );
}

#[test]
fn test_no_video_flag_clears_video_codec() {
    let args = sv(&["-i", "in.mkv", "-vn", "audio_only.ogg"]);
    let result = parse_and_translate(&args);

    assert!(!result.has_errors());
    let job = &result.jobs[0];
    assert!(job.no_video, "no_video should be true");
    assert!(
        job.video_codec.is_none(),
        "video_codec should be None when -vn is set"
    );
}

#[test]
fn test_metadata_preserved_through_translation() {
    let args = sv(&[
        "-i",
        "in.mkv",
        "-metadata",
        "title=My Test Video",
        "-metadata",
        "artist=Test Author",
        "out.webm",
    ]);
    let result = parse_and_translate(&args);

    assert!(!result.has_errors());
    let job = &result.jobs[0];
    assert_eq!(
        job.metadata.get("title").map(String::as_str),
        Some("My Test Video")
    );
    assert_eq!(
        job.metadata.get("artist").map(String::as_str),
        Some("Test Author")
    );
}

#[test]
fn test_vp9_vorbis_no_substitution() {
    let args = sv(&["-i", "in.mkv", "-c:v", "vp9", "-c:a", "vorbis", "out.webm"]);
    let result = parse_and_translate(&args);

    assert!(!result.has_errors());
    let job = &result.jobs[0];
    assert_eq!(job.video_codec.as_deref(), Some("vp9"));
    assert_eq!(job.audio_codec.as_deref(), Some("vorbis"));

    let patent_warnings = result
        .diagnostics
        .iter()
        .filter(|d| matches!(&d.kind, DiagnosticKind::PatentCodecSubstituted { .. }))
        .count();
    assert_eq!(
        patent_warnings, 0,
        "vp9+vorbis should not trigger patent substitution"
    );
}

#[test]
fn test_flac_audio_no_substitution() {
    let args = sv(&["-i", "in.mkv", "-vn", "-c:a", "flac", "out.flac"]);
    let result = parse_and_translate(&args);

    assert!(!result.has_errors());
    let job = &result.jobs[0];
    assert_eq!(job.audio_codec.as_deref(), Some("flac"));
    assert!(
        result
            .diagnostics
            .iter()
            .all(|d| !matches!(&d.kind, DiagnosticKind::PatentCodecSubstituted { .. })),
        "flac should not trigger patent substitution"
    );
}

#[test]
fn test_diagnostic_stderr_format() {
    use oximedia_compat_ffmpeg::diagnostics::Diagnostic;
    let d = Diagnostic::patent_substituted("libx264", "av1");
    let msg = d.format_ffmpeg_style("oximedia-ff");
    assert!(
        msg.starts_with("oximedia-ff:"),
        "should start with program name"
    );
    assert!(
        msg.contains("patent") || msg.contains("codec") || msg.contains("libx264"),
        "should mention patent or codec"
    );
}

#[test]
fn test_libx265_hevc_substitution() {
    let args = sv(&["-i", "in.mkv", "-c:v", "libx265", "out.webm"]);
    let result = parse_and_translate(&args);

    assert!(!result.has_errors());
    assert_eq!(
        result.jobs[0].video_codec.as_deref(),
        Some("av1"),
        "hevc → av1"
    );
    let has_sub = result.diagnostics.iter().any(|d| {
        matches!(&d.kind, DiagnosticKind::PatentCodecSubstituted { from, .. } if from == "libx265")
    });
    assert!(has_sub, "should warn about libx265 substitution");
}

#[test]
fn test_libaom_av1_direct_no_substitution() {
    let args = sv(&["-i", "in.mkv", "-c:v", "libaom-av1", "out.webm"]);
    let result = parse_and_translate(&args);

    assert!(!result.has_errors());
    assert_eq!(result.jobs[0].video_codec.as_deref(), Some("av1"));
    let patent_count = result
        .diagnostics
        .iter()
        .filter(|d| matches!(&d.kind, DiagnosticKind::PatentCodecSubstituted { .. }))
        .count();
    assert_eq!(
        patent_count, 0,
        "libaom-av1 is a direct match, no patent warning"
    );
}

#[test]
fn test_audio_filter_loudnorm() {
    let args = sv(&[
        "-i",
        "in.mkv",
        "-af",
        "loudnorm=I=-23:TP=-2:LRA=7",
        "out.webm",
    ]);
    let result = parse_and_translate(&args);

    assert!(!result.has_errors());
    let job = &result.jobs[0];
    assert_eq!(job.audio_filters.len(), 1, "should have one audio filter");
    assert!(
        matches!(&job.audio_filters[0], ParsedFilter::LoudNorm { .. }),
        "audio filter should be LoudNorm"
    );
}

#[test]
fn test_format_flag_propagated() {
    let args = sv(&["-i", "in.mkv", "-f", "webm", "out.webm"]);
    let result = parse_and_translate(&args);

    assert!(!result.has_errors());
    assert_eq!(result.jobs[0].format.as_deref(), Some("webm"));
}

#[test]
fn test_duration_propagated() {
    let args = sv(&["-i", "in.mkv", "-t", "00:02:00", "out.webm"]);
    let result = parse_and_translate(&args);

    assert!(!result.has_errors());
    assert_eq!(result.jobs[0].duration.as_deref(), Some("00:02:00"));
}
