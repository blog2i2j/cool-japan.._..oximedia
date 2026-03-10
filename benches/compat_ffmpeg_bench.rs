//! Benchmarks for the FFmpeg CLI compatibility layer.
//!
//! Covers:
//! - `FfmpegArgs::parse` across a range of command-line complexities
//! - `CodecMap::new` (construction) and `CodecMap::lookup` (individual queries)
//! - `parse_filters` / `parse_filter_graph` for filtergraph strings
//! - `parse_and_translate` end-to-end pipeline

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oximedia_compat_ffmpeg::{
    arg_parser::FfmpegArgs,
    codec_map::CodecMap,
    filter_lex::{parse_filter_graph, parse_filters},
    parse_and_translate,
};

// ── Helpers ────────────────────────────────────────────────────────────────────

fn make_args(args: &[&str]) -> Vec<String> {
    args.iter().map(|s| s.to_string()).collect()
}

// ── Arg parser benchmarks ─────────────────────────────────────────────────────

fn bench_arg_parser(c: &mut Criterion) {
    let mut group = c.benchmark_group("arg_parser");

    // Simple single-input transcode — patent-free codecs specified directly.
    let simple = make_args(&[
        "-i",
        "input.mkv",
        "-c:v",
        "av1",
        "-c:a",
        "opus",
        "output.webm",
    ]);
    group.bench_function("simple_transcode", |b| {
        b.iter(|| FfmpegArgs::parse(black_box(&simple)));
    });

    // Patent codec names that trigger substitution diagnostics.
    let patent = make_args(&[
        "-i",
        "input.mkv",
        "-c:v",
        "libx264",
        "-crf",
        "28",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "output.webm",
    ]);
    group.bench_function("patent_codecs", |b| {
        b.iter(|| FfmpegArgs::parse(black_box(&patent)));
    });

    // Complex filter chain with video and audio filters.
    let filtered = make_args(&[
        "-i",
        "input.mkv",
        "-vf",
        "scale=1280:720,fps=30,hflip,yadif",
        "-af",
        "loudnorm=I=-23:TP=-1.5:LRA=7,volume=2.0",
        "output.webm",
    ]);
    group.bench_function("filter_chain", |b| {
        b.iter(|| FfmpegArgs::parse(black_box(&filtered)));
    });

    // Multiple inputs with explicit stream mapping.
    let multi_input = make_args(&[
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
    group.bench_function("multi_input_map", |b| {
        b.iter(|| FfmpegArgs::parse(black_box(&multi_input)));
    });

    // Dense option set: global flags, pre-seek, codec, quality, audio, filters, metadata.
    let complex = make_args(&[
        "-y",
        "-loglevel",
        "warning",
        "-threads",
        "8",
        "-ss",
        "00:01:30",
        "-i",
        "input.mkv",
        "-c:v",
        "libx265",
        "-crf",
        "18",
        "-c:a",
        "libfdk_aac",
        "-b:a",
        "256k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-vf",
        "scale=3840:2160,fps=60",
        "-af",
        "loudnorm=I=-23:TP=-1.5",
        "-metadata",
        "title=My Film",
        "-metadata",
        "artist=Director",
        "-t",
        "3600",
        "-f",
        "matroska",
        "output.mkv",
    ]);
    group.bench_function("complex_command", |b| {
        b.iter(|| FfmpegArgs::parse(black_box(&complex)));
    });

    // Many metadata key/value pairs — exercises HashMap insertion in OutputSpec.
    let many_meta: Vec<String> = {
        let mut v = vec!["-i".into(), "input.mkv".into()];
        for i in 0..20 {
            v.push("-metadata".into());
            v.push(format!("tag{}=value{}", i, i));
        }
        v.push("-c:v".into());
        v.push("av1".into());
        v.push("output.webm".into());
        v
    };
    group.throughput(Throughput::Elements(20));
    group.bench_function("many_metadata_20", |b| {
        b.iter(|| FfmpegArgs::parse(black_box(&many_meta)));
    });

    group.finish();
}

// ── Codec-map benchmarks ───────────────────────────────────────────────────────

fn bench_codec_map_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec_map");

    // Construction builds the full HashMap from the static table.
    group.bench_function("new", |b| b.iter(|| black_box(CodecMap::new())));

    group.finish();
}

fn bench_codec_map_lookup(c: &mut Criterion) {
    // Build the map once outside the timed loop.
    let map = CodecMap::new();

    let mut group = c.benchmark_group("codec_map_lookup");

    group.bench_function("av1_direct", |b| b.iter(|| map.lookup(black_box("av1"))));

    group.bench_function("libaom_av1_hyphenated", |b| {
        b.iter(|| map.lookup(black_box("libaom-av1")));
    });

    group.bench_function("libx264_patent_substituted", |b| {
        b.iter(|| map.lookup(black_box("libx264")));
    });

    group.bench_function("aac_patent_substituted", |b| {
        b.iter(|| map.lookup(black_box("aac")));
    });

    group.bench_function("copy_passthrough", |b| {
        b.iter(|| map.lookup(black_box("copy")));
    });

    group.bench_function("unknown_nonexistent", |b| {
        b.iter(|| map.lookup(black_box("nonexistent_codec_xyz_abc")));
    });

    // Helper predicates.
    group.bench_function("is_supported_av1", |b| {
        b.iter(|| map.is_supported(black_box("av1")));
    });

    group.bench_function("is_patent_substituted_libx264", |b| {
        b.iter(|| map.is_patent_substituted(black_box("libx264")));
    });

    // Throughput: resolve 10 representative codec names in sequence.
    let codecs = [
        "libx264", "libx265", "aac", "mp3", "av1", "opus", "vp9", "flac", "copy", "vorbis",
    ];
    group.throughput(Throughput::Elements(codecs.len() as u64));
    group.bench_function("lookup_10_codecs_sequential", |b| {
        b.iter(|| {
            let mut count = 0usize;
            for codec in &codecs {
                if map.lookup(black_box(codec)).is_some() {
                    count += 1;
                }
            }
            black_box(count)
        });
    });

    group.finish();
}

// ── Filter-parser benchmarks ───────────────────────────────────────────────────

fn bench_filter_parser(c: &mut Criterion) {
    let mut group = c.benchmark_group("filter_parser");

    // Single filter, no options.
    group.bench_function("hflip_no_args", |b| {
        b.iter(|| parse_filters(black_box("hflip")));
    });

    // Single filter with positional args.
    group.bench_function("scale_positional", |b| {
        b.iter(|| parse_filters(black_box("scale=1280:720")));
    });

    // Single filter with named args.
    group.bench_function("loudnorm_named", |b| {
        b.iter(|| parse_filters(black_box("loudnorm=I=-23:TP=-1.5:LRA=7")));
    });

    // Linear chain of 4 video filters.
    group.bench_function("chain_4_video_filters", |b| {
        b.iter(|| parse_filters(black_box("scale=1920:1080,fps=30,hflip,yadif")));
    });

    // Linear chain mixing video and audio via semicolon separation.
    group.bench_function("chain_6_mixed_filters", |b| {
        b.iter(|| {
            parse_filters(black_box(
                "scale=1920:1080,fps=30,hflip,yadif;loudnorm=I=-23:TP=-1.5,volume=2.0",
            ))
        });
    });

    // Complex 8-filter pipeline including LUT and subtitle burn-in.
    group.bench_function("complex_8_filters", |b| {
        b.iter(|| parse_filters(black_box(
            "scale=1920:1080,fps=30,hflip,yadif,lut3d=file=log.cube,eq=brightness=0.05:contrast=1.1:saturation=0.9,subtitles=filename=subs.srt,crop=1920:800:0:140"
        )));
    });

    // Unknown filter — exercises the `Unknown` fallback path.
    group.bench_function("unknown_filter_with_args", |b| {
        b.iter(|| parse_filters(black_box("someunknownfilter=param1=val1:param2=val2")));
    });

    // Low-level `parse_filter_graph` with labelled pad syntax.
    group.bench_function("filter_graph_labelled_pads", |b| {
        b.iter(|| {
            parse_filter_graph(black_box(
                "[0:v]scale=1280:720[s];[s]fps=24[out];[0:a]loudnorm=I=-23:TP=-1.5[aout]",
            ))
        });
    });

    // Throughput variant: count filters parsed in complex chain.
    let long_chain = "scale=1280:720,fps=30,hflip,yadif,lut3d=file=log.cube,volume=2.0,aresample=48000,crop=1280:544:0:88";
    group.throughput(Throughput::Bytes(long_chain.len() as u64));
    group.bench_function("long_chain_throughput", |b| {
        b.iter(|| parse_filters(black_box(long_chain)));
    });

    group.finish();
}

// ── End-to-end translate benchmarks ───────────────────────────────────────────

fn bench_translate(c: &mut Criterion) {
    let mut group = c.benchmark_group("translate");

    // Minimal: one input, one output, codec substitution.
    let simple = make_args(&[
        "-i",
        "input.mkv",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "output.webm",
    ]);
    group.bench_function("simple_with_substitution", |b| {
        b.iter(|| parse_and_translate(black_box(&simple)));
    });

    // Richer: codec, quality, filters, metadata — all components exercised.
    let complex = make_args(&[
        "-i",
        "input.mkv",
        "-c:v",
        "libx265",
        "-crf",
        "18",
        "-vf",
        "scale=1920:1080,fps=30,lut3d=file=rec709.cube",
        "-c:a",
        "aac",
        "-af",
        "loudnorm=I=-23:TP=-1.5",
        "-metadata",
        "title=Benchmark Test",
        "output.webm",
    ]);
    group.bench_function("complex_with_filters_and_metadata", |b| {
        b.iter(|| parse_and_translate(black_box(&complex)));
    });

    // Multi-output: two outputs with different codec settings.
    let multi_out = make_args(&[
        "-i",
        "input.mkv",
        "-c:v",
        "av1",
        "-c:a",
        "opus",
        "output_hi.webm",
        "-c:v",
        "vp9",
        "-c:a",
        "vorbis",
        "-b:v",
        "1M",
        "output_lo.webm",
    ]);
    group.bench_function("multi_output", |b| {
        b.iter(|| parse_and_translate(black_box(&multi_out)));
    });

    // Multi-input with stream mapping and filters.
    let multi_in = make_args(&[
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
        "-crf",
        "33",
        "-vf",
        "scale=1280:720,fps=25",
        "-c:a",
        "opus",
        "-b:a",
        "96k",
        "output.webm",
    ]);
    group.bench_function("multi_input_with_map_and_filters", |b| {
        b.iter(|| parse_and_translate(black_box(&multi_in)));
    });

    // Stress: large metadata table + many flags.
    let stress: Vec<String> = {
        let mut v = vec![
            "-y".into(),
            "-threads".into(),
            "16".into(),
            "-loglevel".into(),
            "quiet".into(),
            "-ss".into(),
            "00:00:30".into(),
            "-i".into(),
            "source.mkv".into(),
            "-c:v".into(),
            "libx265".into(),
            "-crf".into(),
            "20".into(),
            "-vf".into(),
            "scale=3840:2160,fps=60,hflip,lut3d=file=hdr.cube".into(),
            "-c:a".into(),
            "libfdk_aac".into(),
            "-b:a".into(),
            "320k".into(),
            "-ar".into(),
            "96000".into(),
            "-ac".into(),
            "6".into(),
            "-af".into(),
            "loudnorm=I=-23:TP=-1:LRA=11,aresample=96000".into(),
            "-t".into(),
            "7200".into(),
            "-f".into(),
            "matroska".into(),
        ];
        for i in 0..10 {
            v.push("-metadata".into());
            v.push(format!("key{}=val{}", i, i));
        }
        v.push("output_stress.mkv".into());
        v
    };
    group.bench_function("stress_all_options", |b| {
        b.iter(|| parse_and_translate(black_box(&stress)));
    });

    group.finish();
}

// ── Parametric input-size scaling ─────────────────────────────────────────────

fn bench_arg_parser_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("arg_parser_scaling");

    // Scale the number of `-metadata` pairs to measure linear growth.
    for n in [1usize, 5, 10, 25, 50] {
        let args: Vec<String> = {
            let mut v = vec!["-i".into(), "input.mkv".into()];
            for i in 0..n {
                v.push("-metadata".into());
                v.push(format!("key{}=value{}", i, i));
            }
            v.push("output.webm".into());
            v
        };
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("metadata_pairs", n), &args, |b, args| {
            b.iter(|| FfmpegArgs::parse(black_box(args)));
        });
    }

    group.finish();
}

// ── CV2-compat colour conversion microbenchmarks ──────────────────────────────
//
// These benchmark the same pixel-level algorithms used by the cv2_compat module,
// implemented inline to avoid exposing internal-only functions from oximedia-cv.
// They complement the existing `cv_bench.rs` suite with BT.601 / YUV420p paths
// that match the colour science in the Python cv2 compatibility bridge.

fn bench_color_conversion_bt601(c: &mut Criterion) {
    let mut group = c.benchmark_group("cv2_compat_color_conversion");

    // ── 1080p ─────────────────────────────────────────────────────────────────
    let w_hd = 1920usize;
    let h_hd = 1080usize;
    let rgb_1080p: Vec<u8> = (0..w_hd * h_hd * 3).map(|i| (i % 256) as u8).collect();

    group.throughput(Throughput::Bytes((w_hd * h_hd * 3) as u64));

    group.bench_function("rgb_to_gray_bt601_1080p", |b| {
        b.iter(|| {
            // BT.601 luma: Y = 0.299R + 0.587G + 0.114B  (integer fixed-point)
            let gray: Vec<u8> = black_box(&rgb_1080p)
                .chunks_exact(3)
                .map(|px| {
                    let r = u32::from(px[0]);
                    let g = u32::from(px[1]);
                    let b = u32::from(px[2]);
                    ((r * 299 + g * 587 + b * 114) / 1000) as u8
                })
                .collect();
            black_box(gray)
        });
    });

    group.bench_function("bgr_to_yuv420p_1080p", |b| {
        b.iter(|| {
            // Convert BGR → YUV420p (planar), matching cv2.cvtColor BGR2YUV_I420.
            let data = black_box(&rgb_1080p); // treated as BGR for this benchmark
            let pixels = w_hd * h_hd;
            let mut y_plane = vec![0u8; pixels];
            let mut u_plane = vec![0u8; pixels / 4];
            let mut v_plane = vec![0u8; pixels / 4];

            for row in 0..h_hd {
                for col in 0..w_hd {
                    let idx = (row * w_hd + col) * 3;
                    let b = i32::from(data[idx]);
                    let g = i32::from(data[idx + 1]);
                    let r = i32::from(data[idx + 2]);
                    // BT.601 limited-range
                    let y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
                    y_plane[row * w_hd + col] = y.clamp(16, 235) as u8;
                    if row % 2 == 0 && col % 2 == 0 {
                        let u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                        let v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                        let uv_idx = (row / 2) * (w_hd / 2) + (col / 2);
                        u_plane[uv_idx] = u.clamp(16, 240) as u8;
                        v_plane[uv_idx] = v.clamp(16, 240) as u8;
                    }
                }
            }
            black_box((y_plane, u_plane, v_plane))
        });
    });

    // ── 4K ────────────────────────────────────────────────────────────────────
    let w_4k = 3840usize;
    let h_4k = 2160usize;
    let rgb_4k: Vec<u8> = (0..w_4k * h_4k * 3).map(|i| (i % 256) as u8).collect();

    group.throughput(Throughput::Bytes((w_4k * h_4k * 3) as u64));

    group.bench_function("rgb_to_gray_bt601_4k", |b| {
        b.iter(|| {
            let gray: Vec<u8> = black_box(&rgb_4k)
                .chunks_exact(3)
                .map(|px| {
                    let r = u32::from(px[0]);
                    let g = u32::from(px[1]);
                    let b = u32::from(px[2]);
                    ((r * 299 + g * 587 + b * 114) / 1000) as u8
                })
                .collect();
            black_box(gray)
        });
    });

    group.finish();
}

fn bench_color_conversion_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("cv2_compat_scaling");

    let resolutions = [
        ("320x240", 320usize, 240usize),
        ("640x480", 640, 480),
        ("1280x720", 1280, 720),
        ("1920x1080", 1920, 1080),
        ("3840x2160", 3840, 2160),
    ];

    for (name, w, h) in &resolutions {
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i % 256) as u8).collect();

        group.throughput(Throughput::Elements((w * h) as u64));
        group.bench_with_input(
            BenchmarkId::new("rgb_to_gray_bt601", name),
            &(rgb, *w, *h),
            |b, (rgb, w, h)| {
                b.iter(|| {
                    let gray: Vec<u8> = black_box(rgb)
                        .chunks_exact(3)
                        .map(|px| {
                            let r = u32::from(px[0]);
                            let g = u32::from(px[1]);
                            let bv = u32::from(px[2]);
                            ((r * 299 + g * 587 + bv * 114) / 1000) as u8
                        })
                        .collect();
                    black_box((gray, *w, *h));
                });
            },
        );
    }

    group.finish();
}

// ── criterion_group / criterion_main ──────────────────────────────────────────

criterion_group!(
    arg_parser_benches,
    bench_arg_parser,
    bench_arg_parser_scaling,
);
criterion_group!(
    codec_map_benches,
    bench_codec_map_construction,
    bench_codec_map_lookup,
);
criterion_group!(filter_parser_benches, bench_filter_parser,);
criterion_group!(translate_benches, bench_translate,);
criterion_group!(
    cv2_compat_benches,
    bench_color_conversion_bt601,
    bench_color_conversion_scaling,
);
criterion_main!(
    arg_parser_benches,
    codec_map_benches,
    filter_parser_benches,
    translate_benches,
    cv2_compat_benches,
);
