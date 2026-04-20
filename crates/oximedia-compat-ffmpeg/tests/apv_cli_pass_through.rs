// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Integration test: APV codec is recognised and passes through the
//! `parse_and_translate` pipeline without substitution.
//!
//! APV (ISO/IEC 23009-13) is a royalty-free intra-frame video codec.
//! When an FFmpeg command uses `-c:v apv`, OxiMedia should recognise it as
//! a direct match (not a patent substitution) and emit `video_codec = "apv"`.

use oximedia_compat_ffmpeg::{
    codec_map::CodecMap, codec_mapping::CodecMapper, parse_and_translate,
};

fn sv(args: &[&str]) -> Vec<String> {
    args.iter().map(|s| s.to_string()).collect()
}

// ─── codec_map.rs tests ──────────────────────────────────────────────────────

#[test]
fn test_codec_map_apv_lookup_direct() {
    let map = CodecMap::new();
    let entry = map.lookup("apv");
    assert!(entry.is_some(), "CodecMap must contain an 'apv' entry");
    let entry = entry.expect("entry present");
    assert_eq!(entry.oxi_name, "apv", "oxi_name must be 'apv'");
    assert!(
        matches!(
            entry.category,
            oximedia_compat_ffmpeg::codec_map::CodecCategory::DirectMatch
        ),
        "APV must be a DirectMatch, not substituted"
    );
}

#[test]
fn test_codec_map_apv_oxi_name() {
    let map = CodecMap::new();
    // oxi_name returns the OxiMedia name, or the input unchanged if unrecognised.
    assert_eq!(
        map.oxi_name("apv"),
        "apv",
        "CodecMap::oxi_name('apv') must return 'apv'"
    );
}

#[test]
fn test_codec_map_apv_is_supported() {
    let map = CodecMap::new();
    assert!(
        map.is_supported("apv"),
        "CodecMap::is_supported('apv') must be true"
    );
}

// ─── codec_mapping.rs tests ──────────────────────────────────────────────────

#[test]
fn test_codec_mappings_contains_apv() {
    let found = CodecMapper::codec("apv");
    assert!(
        found.is_some(),
        "CODEC_MAPPINGS must contain an entry for 'apv'"
    );
    let mapping = found.expect("mapping present");
    assert_eq!(mapping.ffmpeg_name, "apv");
    assert_eq!(mapping.oximedia_codec, "apv");
    assert!(mapping.is_video, "APV is a video codec");
    assert!(!mapping.is_audio, "APV is not an audio codec");
}

// ─── parse_and_translate tests ───────────────────────────────────────────────

#[test]
fn test_apv_parse_and_translate_direct_match() {
    let args = sv(&["-i", "input.mkv", "-c:v", "apv", "output.mp4"]);
    let result = parse_and_translate(&args);

    assert_eq!(result.jobs.len(), 1);
    let job = &result.jobs[0];
    assert_eq!(job.input_path, "input.mkv");
    assert_eq!(job.output_path, "output.mp4");

    // APV must pass through as-is — it is patent-free.
    assert_eq!(
        job.video_codec.as_deref(),
        Some("apv"),
        "APV must not be substituted: got {:?}",
        job.video_codec
    );
}

#[test]
fn test_apv_no_patent_substitution_diagnostic() {
    use oximedia_compat_ffmpeg::diagnostics::DiagnosticKind;

    let args = sv(&["-i", "input.mkv", "-c:v", "apv", "output.mp4"]);
    let result = parse_and_translate(&args);

    let has_patent_sub = result.diagnostics.iter().any(|d| {
        matches!(
            &d.kind,
            DiagnosticKind::PatentCodecSubstituted { from, .. } if from == "apv"
        )
    });
    assert!(
        !has_patent_sub,
        "APV must not trigger a patent substitution diagnostic"
    );
}

#[test]
fn test_apv_with_audio_codec() {
    // A typical APV + Opus workflow.
    let args = sv(&[
        "-i",
        "input.mp4",
        "-c:v",
        "apv",
        "-c:a",
        "libopus",
        "output.mp4",
    ]);
    let result = parse_and_translate(&args);

    assert_eq!(result.jobs.len(), 1);
    let job = &result.jobs[0];
    assert_eq!(job.video_codec.as_deref(), Some("apv"));
    assert_eq!(job.audio_codec.as_deref(), Some("opus"));
}

#[test]
fn test_apv_case_insensitive_lookup() {
    let map = CodecMap::new();
    // The codec map normalises to lowercase.
    let entry = map.lookup("APV");
    assert!(
        entry.is_some(),
        "CodecMap lookup must be case-insensitive for 'APV'"
    );
}

#[test]
fn test_apv_hyphen_underscore_normalisation() {
    // 'apv' has no hyphens, but the normalisation path should still work.
    let map = CodecMap::new();
    let entry_plain = map.lookup("apv");
    let entry_norm = map.lookup("apv");
    assert_eq!(entry_plain, entry_norm);
}
