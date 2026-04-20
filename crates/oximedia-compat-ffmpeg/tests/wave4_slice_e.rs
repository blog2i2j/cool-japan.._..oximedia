use std::ptr;

use oximedia_compat_ffmpeg::arg_parser::FfmpegArgs;
use oximedia_compat_ffmpeg::codec_map;
use oximedia_compat_ffmpeg::encoder_options::{EncoderProfile, EncoderQualityPreset, EncoderTune};
use oximedia_compat_ffmpeg::filter_shorthand::{parse_af, parse_vf};
use oximedia_compat_ffmpeg::pass::{parse_pass, PassPhase};

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

#[test]
fn codec_map_cache() {
    let first = codec_map::table();
    let second = codec_map::table();

    assert!(ptr::eq(first, second));
}

#[test]
fn encoder_quality_args() {
    let parsed = FfmpegArgs::parse(&args(&[
        "-i",
        "input.mkv",
        "-preset",
        "slow",
        "-tune",
        "film",
        "-profile:v",
        "high",
        "output.webm",
    ]))
    .expect("encoder quality args should parse");

    let output = &parsed.outputs[0];
    assert_eq!(
        output.encoder_quality.preset,
        Some(EncoderQualityPreset::Slow)
    );
    assert_eq!(output.encoder_quality.tune, Some(EncoderTune::Film));
    assert_eq!(output.encoder_quality.profile, Some(EncoderProfile::High));
}

#[test]
fn encoder_quality_args_invalid_preset_returns_err() {
    assert!(FfmpegArgs::parse(&args(&[
        "-i",
        "input.mkv",
        "-preset",
        "banana",
        "output.webm",
    ]))
    .is_err());
}

#[test]
fn encoder_quality_args_invalid_tune_returns_err() {
    assert!(FfmpegArgs::parse(&args(&[
        "-i",
        "input.mkv",
        "-tune",
        "banana",
        "output.webm",
    ]))
    .is_err());
}

#[test]
fn encoder_quality_args_invalid_profile_returns_err() {
    assert!(FfmpegArgs::parse(&args(&[
        "-i",
        "input.mkv",
        "-profile:v",
        "banana",
        "output.webm",
    ]))
    .is_err());
}

#[test]
fn filter_shorthand_parse() {
    let cases = [
        ("scale=1920:1080", 1usize),
        ("format=yuv420p", 1),
        ("fps=30", 1),
        ("crop=1280:720:0:0", 1),
        ("volume=0.5", 1),
        ("aresample=48000", 1),
        ("scale=1280:720,format=yuv420p", 2),
        ("fps=30,format=yuv420p", 2),
        ("scale=854:480,fps=30,format=yuv420p", 3),
        ("volume=1.25,aresample=44100", 2),
    ];

    for (input, expected_count) in cases {
        let graph = if input.starts_with("volume") || input.starts_with("aresample") {
            parse_af(input).expect("audio shorthand should parse")
        } else {
            parse_vf(input).expect("video shorthand should parse")
        };

        assert_eq!(graph.chains.len(), 1, "expected single chain for {input}");
        assert_eq!(
            graph.filter_count(),
            expected_count,
            "wrong node count for {input}"
        );
        assert_eq!(graph.input_labels(), vec!["in"]);
        assert_eq!(graph.output_labels(), vec!["out"]);
    }
}

#[test]
fn two_pass_flag() {
    let stats_path = std::env::temp_dir().join("oximedia-wave4-stats");
    let pass1 = parse_pass(&args(&[
        "-pass",
        "1",
        "-passlogfile",
        stats_path.to_string_lossy().as_ref(),
    ]))
    .expect("pass 1 should parse");
    let pass2 = parse_pass(&args(&["-pass", "2"])).expect("pass 2 should parse");

    assert_eq!(
        pass1,
        Some(PassPhase::First {
            stats_path: stats_path.clone(),
        })
    );
    assert_eq!(
        pass2,
        Some(PassPhase::Second {
            stats_path: "ffmpeg2pass-0.log".into(),
        })
    );
}
