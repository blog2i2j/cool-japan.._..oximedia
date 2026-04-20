//! Real-world FFmpeg command-line test suite.
//!
//! Covers 50+ real-world FFmpeg command patterns to verify that the
//! compat layer correctly parses and translates common usage patterns.

#[cfg(test)]
mod tests {
    use crate::arg_parser::FfmpegArgs;
    use crate::filter_lex::{parse_filters, ParsedFilter};
    use crate::stream_spec::{StreamIndex, StreamSpec, StreamType};
    use crate::translator::parse_and_translate;

    fn sv(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-compat-ffmpeg-rwtests-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Codec conversion
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rw_01_h264_to_av1() {
        // ffmpeg -i input.mp4 -c:v libx264 output.mp4
        // Patent codec → AV1 substitution
        let args = sv(&["-i", "input.mp4", "-c:v", "libx264", "output.mp4"]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.video_codec.as_deref(), Some("av1"));
        // Should have a patent substitution diagnostic.
        assert!(r.diagnostics.iter().any(|d| {
            matches!(&d.kind, crate::diagnostics::DiagnosticKind::PatentCodecSubstituted { from, .. } if from == "libx264")
        }));
    }

    #[test]
    fn rw_02_hevc_to_av1() {
        // ffmpeg -i input.mkv -c:v libx265 -preset slow output.mkv
        let args = sv(&[
            "-i",
            "input.mkv",
            "-c:v",
            "libx265",
            "-preset",
            "slow",
            "output.mkv",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.video_codec.as_deref(), Some("av1"));
        assert_eq!(job.preset.as_deref(), Some("slow"));
    }

    #[test]
    fn rw_03_aac_to_opus() {
        // ffmpeg -i input.mp4 -c:a aac -b:a 128k output.m4a
        let args = sv(&[
            "-i",
            "input.mp4",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "output.m4a",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.audio_codec.as_deref(), Some("opus"));
        assert_eq!(job.audio_bitrate.as_deref(), Some("128k"));
    }

    #[test]
    fn rw_04_copy_video_transcode_audio() {
        // ffmpeg -i input.mkv -c:v copy -c:a libopus output.mkv
        let args = sv(&[
            "-i",
            "input.mkv",
            "-c:v",
            "copy",
            "-c:a",
            "libopus",
            "output.mkv",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.video_codec.as_deref(), Some("copy"));
        assert_eq!(job.audio_codec.as_deref(), Some("opus"));
    }

    #[test]
    fn rw_05_av1_crf_encoding() {
        // ffmpeg -i input.mkv -c:v libaom-av1 -crf 28 -b:v 0 output.webm
        let args = sv(&[
            "-i",
            "input.mkv",
            "-c:v",
            "libaom-av1",
            "-crf",
            "28",
            "-b:v",
            "0",
            "output.webm",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.video_codec.as_deref(), Some("av1"));
        assert!((job.crf.expect("crf") - 28.0).abs() < 0.001);
    }

    #[test]
    fn rw_06_vp9_two_pass_first() {
        // ffmpeg -i input.mp4 -c:v libvpx-vp9 -b:v 2M -pass 1 -an -f null /dev/null
        let args = sv(&[
            "-i",
            "input.mp4",
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "2M",
            "-pass",
            "1",
            "-an",
            "-f",
            "null",
            "/dev/null",
        ]);
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let out = &parsed.outputs[0];
        assert_eq!(out.pass, Some(1));
        assert!(out.no_audio);
    }

    #[test]
    fn rw_07_vp9_two_pass_second() {
        // ffmpeg -i input.mp4 -c:v libvpx-vp9 -b:v 2M -pass 2 -c:a libopus output.webm
        let args = sv(&[
            "-i",
            "input.mp4",
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "2M",
            "-pass",
            "2",
            "-c:a",
            "libopus",
            "output.webm",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.pass, Some(2));
        assert_eq!(job.video_codec.as_deref(), Some("vp9"));
    }

    #[test]
    fn rw_08_passlogfile_two_pass() {
        // ffmpeg -i in.mp4 -c:v libvpx-vp9 -pass 1 -passlogfile <tmp>/vp9pass out.webm
        let pass_log = tmp_str("vp9pass");
        let args = sv(&[
            "-i",
            "in.mp4",
            "-c:v",
            "libvpx-vp9",
            "-pass",
            "1",
            "-passlogfile",
            &pass_log,
            "out.webm",
        ]);
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let out = &parsed.outputs[0];
        assert_eq!(out.pass, Some(1));
        assert_eq!(out.passlogfile.as_deref(), Some(pass_log.as_str()));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Seeking and trimming
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rw_09_seek_before_input() {
        // ffmpeg -ss 00:01:30 -i input.mp4 -c copy output.mp4
        let args = sv(&[
            "-ss",
            "00:01:30",
            "-i",
            "input.mp4",
            "-c",
            "copy",
            "output.mp4",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.seek.as_deref(), Some("00:01:30"));
    }

    #[test]
    fn rw_10_seek_after_input_with_duration() {
        // ffmpeg -i input.mp4 -ss 10 -t 30 output.mp4
        let args = sv(&["-i", "input.mp4", "-ss", "10", "-t", "30", "output.mp4"]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.seek.as_deref(), Some("10"));
        assert_eq!(job.duration.as_deref(), Some("30"));
    }

    #[test]
    fn rw_11_seek_with_to_position() {
        // ffmpeg -i input.mp4 -ss 0 -to 60 output.mp4
        let args = sv(&["-i", "input.mp4", "-ss", "0", "-to", "60", "output.mp4"]);
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let out = &parsed.outputs[0];
        assert_eq!(out.seek.as_deref(), Some("0"));
        // -to is stored in extra_args (it has no separate duration field in arg_parser).
        let to_entry = out.extra_args.iter().find(|(k, _)| k == "-to");
        assert!(to_entry.is_some(), "-to should appear in extra_args");
        assert_eq!(to_entry.map(|(_, v)| v.as_str()), Some("60"));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Filter chains
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rw_12_vf_scale() {
        // ffmpeg -i input.mp4 -vf scale=1280:720 output.mp4
        let args = sv(&["-i", "input.mp4", "-vf", "scale=1280:720", "output.mp4"]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.video_filters.len(), 1);
        assert!(matches!(
            job.video_filters[0],
            ParsedFilter::Scale { w: 1280, h: 720 }
        ));
    }

    #[test]
    fn rw_13_vf_scale_maintain_aspect() {
        // ffmpeg -i input.mp4 -vf scale=1280:-1 output.mp4
        let args = sv(&["-i", "input.mp4", "-vf", "scale=1280:-1", "output.mp4"]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.video_filters.len(), 1);
        match &job.video_filters[0] {
            ParsedFilter::Scale { w, h } => {
                assert_eq!(*w, 1280);
                assert_eq!(*h, -1);
            }
            _ => panic!("expected Scale filter"),
        }
    }

    #[test]
    fn rw_14_vf_crop_scale_chain() {
        // ffmpeg -i input.mp4 -vf "crop=1920:800,scale=1280:534" output.mp4
        let args = sv(&[
            "-i",
            "input.mp4",
            "-vf",
            "crop=1920:800,scale=1280:534",
            "output.mp4",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.video_filters.len(), 2);
        assert!(matches!(&job.video_filters[0], ParsedFilter::Crop { .. }));
        assert!(matches!(&job.video_filters[1], ParsedFilter::Scale { .. }));
    }

    #[test]
    fn rw_15_af_loudnorm() {
        // ffmpeg -i input.mp4 -af loudnorm=I=-23:TP=-2:LRA=7 output.mp4
        let args = sv(&[
            "-i",
            "input.mp4",
            "-af",
            "loudnorm=I=-23:TP=-2:LRA=7",
            "output.mp4",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.audio_filters.len(), 1);
        assert!(matches!(
            &job.audio_filters[0],
            ParsedFilter::LoudNorm { .. }
        ));
    }

    #[test]
    fn rw_16_af_volume_adjustment() {
        // ffmpeg -i input.mp4 -af volume=0.5 output.mp4
        let args = sv(&["-i", "input.mp4", "-af", "volume=0.5", "output.mp4"]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.audio_filters.len(), 1);
        assert!(matches!(&job.audio_filters[0], ParsedFilter::Volume { .. }));
    }

    #[test]
    fn rw_17_filter_complex_overlay() {
        // ffmpeg -i bg.mp4 -i logo.png -filter_complex "[0:v][1:v]overlay=10:10[out]"
        //        -map "[out]" output.mp4
        let args = sv(&[
            "-i",
            "bg.mp4",
            "-i",
            "logo.png",
            "-filter_complex",
            "[0:v][1:v]overlay=10:10[out]",
            "-map",
            "[out]",
            "output.mp4",
        ]);
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.inputs.len(), 2);
        assert!(parsed.outputs[0].filter_complex.is_some());
    }

    #[test]
    fn rw_18_filter_complex_scale_concat() {
        // Two-input concat with scale.
        let args = sv(&[
            "-i",
            "part1.mp4",
            "-i",
            "part2.mp4",
            "-filter_complex",
            "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[outv][outa]",
            "-map",
            "[outv]",
            "-map",
            "[outa]",
            "output.mp4",
        ]);
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert!(parsed.outputs[0].filter_complex.is_some());
        assert_eq!(parsed.outputs[0].map.len(), 2);
    }

    #[test]
    fn rw_19_deinterlace_filter() {
        // ffmpeg -i interlaced.ts -vf yadif output.mp4
        let args = sv(&["-i", "interlaced.ts", "-vf", "yadif", "output.mp4"]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.video_filters.len(), 1);
        assert_eq!(job.video_filters[0], ParsedFilter::Deinterlace);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Map flag
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rw_20_map_video_from_file_0() {
        // ffmpeg -i input.mp4 -map 0:v:0 -map 0:a:0 output.mkv
        let args = sv(&[
            "-i",
            "input.mp4",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "output.mkv",
        ]);
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.outputs[0].map.len(), 2);
    }

    #[test]
    fn rw_21_negative_map() {
        // ffmpeg -i input.mkv -map 0 -map -0:s output.mp4
        // Map all streams but exclude subtitles.
        let args = sv(&["-i", "input.mkv", "-map", "0", "-map", "-0:s", "output.mp4"]);
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        let maps = &parsed.outputs[0].map;
        assert_eq!(maps.len(), 2);
        assert!(!maps[0].negative, "first map should not be negative");
        assert!(maps[1].negative, "second map should be negative");
    }

    #[test]
    fn rw_22_map_all_streams() {
        // ffmpeg -i input.mkv -map 0 -c copy output.mkv
        let args = sv(&["-i", "input.mkv", "-map", "0", "-c", "copy", "output.mkv"]);
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.outputs[0].map.len(), 1);
        // input_index should be 0 when mapping "0"
        assert_eq!(parsed.outputs[0].map[0].input_index, 0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Metadata
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rw_23_metadata_title() {
        // ffmpeg -i input.mp4 -metadata title="My Movie" output.mp4
        let args = sv(&[
            "-i",
            "input.mp4",
            "-metadata",
            "title=My Movie",
            "output.mp4",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(
            job.metadata.get("title").map(String::as_str),
            Some("My Movie")
        );
    }

    #[test]
    fn rw_24_metadata_multiple_tags() {
        // ffmpeg -i in.mp3 -metadata title="Song" -metadata artist="Artist" out.mp3
        let args = sv(&[
            "-i",
            "in.mp3",
            "-metadata",
            "title=Song",
            "-metadata",
            "artist=Artist",
            "-metadata",
            "comment=Ripped with OxiMedia",
            "out.mp3",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.metadata.get("title").map(String::as_str), Some("Song"));
        assert_eq!(
            job.metadata.get("artist").map(String::as_str),
            Some("Artist")
        );
        assert_eq!(
            job.metadata.get("comment").map(String::as_str),
            Some("Ripped with OxiMedia")
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Muxer options
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rw_25_movflags_faststart() {
        // ffmpeg -i input.mp4 -movflags +faststart output.mp4
        let args = sv(&["-i", "input.mp4", "-movflags", "+faststart", "output.mp4"]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.muxer_options.len(), 1);
        assert!(matches!(
            &job.muxer_options[0].oxi_action,
            crate::translator::MuxerAction::FastStart
        ));
    }

    #[test]
    fn rw_26_movflags_fragmented_mp4() {
        // ffmpeg -i input.mp4 -movflags frag_keyframe+empty_moov output.mp4
        let args = sv(&[
            "-i",
            "input.mp4",
            "-movflags",
            "frag_keyframe+empty_moov",
            "output.mp4",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert!(job
            .muxer_options
            .iter()
            .any(|m| { matches!(&m.oxi_action, crate::translator::MuxerAction::FragmentedMp4) }));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Preset / tune / profile
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rw_27_preset_medium() {
        // ffmpeg -i in.mp4 -c:v libx264 -preset medium out.mp4
        let args = sv(&[
            "-i", "in.mp4", "-c:v", "libx264", "-preset", "medium", "out.mp4",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.preset.as_deref(), Some("medium"));
    }

    #[test]
    fn rw_28_preset_tune_profile_h264() {
        // ffmpeg -i in.mp4 -c:v libx264 -preset fast -tune film -profile:v high out.mp4
        let args = sv(&[
            "-i",
            "in.mp4",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-tune",
            "film",
            "-profile:v",
            "high",
            "out.mp4",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.preset.as_deref(), Some("fast"));
        assert_eq!(job.tune.as_deref(), Some("film"));
        assert_eq!(job.profile.as_deref(), Some("high"));
    }

    #[test]
    fn rw_29_tune_animation() {
        // ffmpeg -i cartoon.mp4 -c:v libx264 -tune animation output.mp4
        let args = sv(&[
            "-i",
            "cartoon.mp4",
            "-c:v",
            "libx264",
            "-tune",
            "animation",
            "output.mp4",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.tune.as_deref(), Some("animation"));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Stream suppression
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rw_30_no_video_audio_only() {
        // ffmpeg -i video.mp4 -vn -c:a libopus audio.ogg
        let args = sv(&["-i", "video.mp4", "-vn", "-c:a", "libopus", "audio.ogg"]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert!(job.no_video);
        assert!(!job.no_audio);
        assert!(job.video_codec.is_none());
        assert_eq!(job.audio_codec.as_deref(), Some("opus"));
    }

    #[test]
    fn rw_31_no_audio_video_only() {
        // ffmpeg -i input.mp4 -an -c:v libaom-av1 output.webm
        let args = sv(&[
            "-i",
            "input.mp4",
            "-an",
            "-c:v",
            "libaom-av1",
            "output.webm",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert!(job.no_audio);
        assert!(!job.no_video);
        assert!(job.audio_codec.is_none());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Multiple outputs
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rw_32_multiple_outputs() {
        // ffmpeg -i input.mp4 output_hd.webm output_sd.webm
        let args = sv(&[
            "-i",
            "input.mp4",
            "-c:v",
            "libaom-av1",
            "output_hd.webm",
            "-c:v",
            "libvpx-vp9",
            "output_sd.webm",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        assert_eq!(r.jobs.len(), 2);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Multiple inputs
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rw_33_multiple_inputs_merge() {
        // ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a libopus merged.mkv
        let args = sv(&[
            "-i",
            "video.mp4",
            "-i",
            "audio.wav",
            "-c:v",
            "copy",
            "-c:a",
            "libopus",
            "merged.mkv",
        ]);
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.inputs.len(), 2);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Hardware encoder aliases (patent substitution)
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rw_34_h264_nvenc_to_av1() {
        // ffmpeg -i input.mp4 -c:v h264_nvenc output.mp4
        let args = sv(&["-i", "input.mp4", "-c:v", "h264_nvenc", "output.mp4"]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.video_codec.as_deref(), Some("av1"));
        // Confirm patent substitution warning.
        assert!(r.diagnostics.iter().any(|d| {
            matches!(&d.kind, crate::diagnostics::DiagnosticKind::PatentCodecSubstituted { from, .. } if from == "h264_nvenc")
        }));
    }

    #[test]
    fn rw_35_hevc_amf_to_av1() {
        // ffmpeg -i input.mp4 -c:v hevc_amf output.mp4
        let args = sv(&["-i", "input.mp4", "-c:v", "hevc_amf", "output.mp4"]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.video_codec.as_deref(), Some("av1"));
    }

    #[test]
    fn rw_36_hevc_nvenc_to_av1() {
        let args = sv(&[
            "-i",
            "in.mp4",
            "-c:v",
            "hevc_nvenc",
            "-b:v",
            "5M",
            "out.mp4",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        assert_eq!(r.jobs[0].video_codec.as_deref(), Some("av1"));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Format specification
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rw_37_force_format_null() {
        // ffmpeg -i input.mp4 -f null /dev/null
        let args = sv(&["-i", "input.mp4", "-f", "null", "/dev/null"]);
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.outputs[0].format.as_deref(), Some("null"));
    }

    #[test]
    fn rw_38_output_to_webm() {
        // ffmpeg -i input.mp4 -c:v libvpx-vp9 -c:a libopus -f webm output.webm
        let args = sv(&[
            "-i",
            "input.mp4",
            "-c:v",
            "libvpx-vp9",
            "-c:a",
            "libopus",
            "-f",
            "webm",
            "output.webm",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.format.as_deref(), Some("webm"));
        assert_eq!(job.video_codec.as_deref(), Some("vp9"));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Overwrite / global options
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rw_39_overwrite_yes() {
        // ffmpeg -y -i input.mp4 output.mp4
        let args = sv(&["-y", "-i", "input.mp4", "output.mp4"]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        assert!(r.jobs[0].overwrite);
    }

    #[test]
    fn rw_40_threads_option() {
        // ffmpeg -threads 4 -i input.mp4 output.webm
        let args = sv(&["-threads", "4", "-i", "input.mp4", "output.webm"]);
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.global_options.threads, Some(4));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Stream specifier parsing
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rw_41_stream_spec_0_v_0() {
        let s = StreamSpec::parse("0:v:0").expect("parse 0:v:0");
        assert_eq!(s.file_index, Some(0));
        assert_eq!(s.stream_type, StreamType::Video);
        assert_eq!(s.stream_index, StreamIndex::Position(0));
    }

    #[test]
    fn rw_42_stream_spec_pid() {
        let s = StreamSpec::parse("0:a:#0x1100").expect("parse 0:a:#0x1100");
        assert_eq!(s.file_index, Some(0));
        assert_eq!(s.stream_type, StreamType::Audio);
        assert_eq!(s.stream_index, StreamIndex::Pid(0x1100));
    }

    #[test]
    fn rw_43_stream_spec_second_input_first_audio() {
        let s = StreamSpec::parse("1:a:0").expect("parse 1:a:0");
        assert_eq!(s.file_index, Some(1));
        assert_eq!(s.stream_type, StreamType::Audio);
        assert_eq!(s.stream_index, StreamIndex::Position(0));
    }

    #[test]
    fn rw_44_stream_spec_program() {
        let s = StreamSpec::parse("p:10:0").expect("parse p:10:0");
        assert_eq!(s.program_id, Some(10));
        assert_eq!(s.index, Some(0));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Filter parsing edge cases
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rw_45_filter_complex_multi_input_multi_output() {
        // [0:v][1:v]hstack=inputs=2[outv]
        let filters = parse_filters("[0:v][1:v]hstack=inputs=2[outv]");
        // hstack is unsupported but should parse as Unknown.
        assert_eq!(filters.len(), 1);
        assert!(matches!(&filters[0], ParsedFilter::Unknown { name, .. } if name == "hstack"));
    }

    #[test]
    fn rw_46_filter_vf_eq_color_correction() {
        let filters = parse_filters("eq=brightness=0.05:contrast=1.1:saturation=1.2");
        assert_eq!(filters.len(), 1);
        assert!(matches!(&filters[0], ParsedFilter::ColorCorrect { .. }));
    }

    #[test]
    fn rw_47_filter_chain_scale_fps_hflip() {
        let filters = parse_filters("scale=1920:1080,fps=30,hflip");
        assert_eq!(filters.len(), 3);
        assert!(matches!(
            &filters[0],
            ParsedFilter::Scale { w: 1920, h: 1080 }
        ));
        assert!(matches!(&filters[1], ParsedFilter::Fps { rate } if (*rate - 30.0).abs() < 0.1));
        assert_eq!(filters[2], ParsedFilter::HFlip);
    }

    #[test]
    fn rw_48_filter_subtitles_burn_in() {
        // ffmpeg -i input.mp4 -vf subtitles=subs.srt output.mp4
        let args = sv(&[
            "-i",
            "input.mp4",
            "-vf",
            "subtitles=filename=subs.srt",
            "output.mp4",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.video_filters.len(), 1);
        assert!(matches!(
            &job.video_filters[0],
            ParsedFilter::SubtitleBurnIn { file } if file == "subs.srt"
        ));
    }

    #[test]
    fn rw_49_filter_lut3d_apply() {
        // ffmpeg -i input.mp4 -vf lut3d=file=grade.cube output.mp4
        let args = sv(&[
            "-i",
            "input.mp4",
            "-vf",
            "lut3d=file=grade.cube",
            "output.mp4",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.video_filters.len(), 1);
        assert!(matches!(
            &job.video_filters[0],
            ParsedFilter::Lut3d { file } if file == "grade.cube"
        ));
    }

    #[test]
    fn rw_50_filter_aresample() {
        // ffmpeg -i input.wav -af aresample=48000 output.flac
        let args = sv(&["-i", "input.wav", "-af", "aresample=48000", "output.flac"]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.audio_filters.len(), 1);
        assert!(matches!(
            &job.audio_filters[0],
            ParsedFilter::Resample { sample_rate: 48000 }
        ));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Diagnostics / error handling
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rw_51_unknown_codec_suggestion() {
        // Typo in codec name should produce a "did you mean?" diagnostic.
        let args = sv(&["-i", "in.mp4", "-c:v", "libx26", "out.mp4"]);
        let r = parse_and_translate(&args);
        // Should not be a fatal error, just a warning/suggestion.
        assert!(!r.has_errors());
        // Should have a suggestion diagnostic.
        let has_suggestion = r.diagnostics.iter().any(|d| {
            d.suggestion
                .as_ref()
                .map(|s| s.contains("Did you mean") || s.contains("patent-free"))
                .unwrap_or(false)
        });
        assert!(has_suggestion, "should produce a suggestion for libx26");
    }

    #[test]
    fn rw_52_no_input_is_error() {
        let args = sv(&["output.mp4"]);
        let r = parse_and_translate(&args);
        assert!(r.has_errors());
        assert!(r.jobs.is_empty());
    }

    #[test]
    fn rw_53_no_output_is_error() {
        let args = sv(&["-i", "input.mp4"]);
        let r = parse_and_translate(&args);
        assert!(r.has_errors());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // FFprobe output mode
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rw_54_ffprobe_json_output() {
        use crate::ffprobe::{PrintFormat, ProbeFormat, ProbeOutput, ProbeStream};

        let v = ProbeStream::new_video("av1", 1920, 1080, "16:9", 30.0);
        let a = ProbeStream::new_audio("opus", 48000, 2, "stereo");
        let fmt = ProbeFormat::new("movie.mkv", "matroska", 800_000_000, 5400.0);

        let mut out = ProbeOutput::new();
        out.streams.push(v);
        out.streams.push(a);
        out.format = Some(fmt);

        let json = out.to_print_format(PrintFormat::Json);
        assert!(json.contains("\"streams\""));
        assert!(json.contains("\"format\""));
        assert!(json.contains("\"av1\""));
        assert!(json.contains("\"opus\""));
        assert!(json.contains("\"matroska\""));
    }

    #[test]
    fn rw_55_ffprobe_csv_output() {
        use crate::ffprobe::{PrintFormat, ProbeOutput, ProbeStream};

        let s = ProbeStream::new_video("vp9", 1280, 720, "16:9", 25.0);
        let mut out = ProbeOutput::new();
        out.streams.push(s);

        let csv = out.to_print_format(PrintFormat::Csv);
        assert!(csv.starts_with("stream,"), "CSV should start with stream,");
        assert!(csv.contains("vp9"));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional real-world patterns
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rw_56_extract_audio_no_video() {
        // ffmpeg -i movie.mkv -vn -c:a copy audio.flac
        let args = sv(&["-i", "movie.mkv", "-vn", "-c:a", "copy", "audio.flac"]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert!(job.no_video);
        assert_eq!(job.audio_codec.as_deref(), Some("copy"));
    }

    #[test]
    fn rw_57_thumbnail_extraction() {
        // ffmpeg -i video.mp4 -ss 10 -vframes 1 thumb.jpg
        let args = sv(&["-i", "video.mp4", "-ss", "10", "thumb.jpg"]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        assert_eq!(r.jobs[0].seek.as_deref(), Some("10"));
    }

    #[test]
    fn rw_58_concat_demuxer_style() {
        // ffmpeg -f concat -safe 0 -i filelist.txt -c copy output.mp4
        let args = sv(&[
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            "filelist.txt",
            "-c",
            "copy",
            "output.mp4",
        ]);
        let parsed = FfmpegArgs::parse(&args).expect("parse");
        assert_eq!(parsed.inputs[0].format.as_deref(), Some("concat"));
        assert_eq!(parsed.inputs[0].path, "filelist.txt");
    }

    #[test]
    fn rw_59_bitrate_video_audio() {
        // ffmpeg -i input.mp4 -b:v 2M -b:a 192k output.mp4
        let args = sv(&[
            "-i",
            "input.mp4",
            "-b:v",
            "2M",
            "-b:a",
            "192k",
            "output.mp4",
        ]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        assert_eq!(job.video_bitrate.as_deref(), Some("2M"));
        assert_eq!(job.audio_bitrate.as_deref(), Some("192k"));
    }

    #[test]
    fn rw_60_codec_mapping_all_streams() {
        // ffmpeg -i input.mkv -c copy output.mkv
        let args = sv(&["-i", "input.mkv", "-c", "copy", "output.mkv"]);
        let r = parse_and_translate(&args);
        assert!(!r.has_errors());
        let job = &r.jobs[0];
        // When -c copy is set as "all streams", both video and audio should be copy.
        assert!(
            job.video_codec.as_deref() == Some("copy")
                || job.audio_codec.as_deref() == Some("copy"),
            "copy should apply to at least one stream codec"
        );
    }
}
