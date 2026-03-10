//! Benchmarks for oximedia-analysis.
//!
//! These benchmarks measure the performance of various analysis operations
//! on different video resolutions and frame rates.

use oximedia_analysis::{
    audio::AudioAnalyzer, black::BlackFrameDetector, color::ColorAnalyzer,
    content::ContentClassifier, motion::MotionAnalyzer, quality::QualityAssessor,
    scene::SceneDetector, temporal::TemporalAnalyzer, thumbnail::ThumbnailSelector, AnalysisConfig,
    Analyzer,
};
use oximedia_core::types::Rational;
use std::time::Duration;

fn main() {
    println!("OxiMedia Analysis Benchmarks\n");

    // Run all benchmarks
    benchmark_scene_detection();
    benchmark_quality_assessment();
    benchmark_black_frame_detection();
    benchmark_content_classification();
    benchmark_thumbnail_selection();
    benchmark_motion_analysis();
    benchmark_color_analysis();
    benchmark_temporal_analysis();
    benchmark_audio_analysis();
    benchmark_full_pipeline();
    benchmark_different_resolutions();
    benchmark_memory_usage();
}

fn benchmark_scene_detection() {
    println!("=== Scene Detection Benchmark ===");

    let mut detector = SceneDetector::new(0.3);
    let width = 1920;
    let height = 1080;

    let start = std::time::Instant::now();

    for frame_num in 0..300 {
        let brightness = if frame_num % 100 < 50 { 50u8 } else { 200u8 };
        let frame = vec![brightness; width * height];
        detector
            .process_frame(&frame, width, height, frame_num)
            .expect("unexpected None/Err");
    }

    let _scenes = detector.finalize();
    let elapsed = start.elapsed();

    println!("  Frames processed: 300");
    println!("  Resolution: {}x{}", width, height);
    println!("  Time: {:?}", elapsed);
    println!("  FPS: {:.2}", 300.0 / elapsed.as_secs_f64());
    println!("  Time per frame: {:?}\n", elapsed / 300);
}

fn benchmark_quality_assessment() {
    println!("=== Quality Assessment Benchmark ===");

    let mut assessor = QualityAssessor::new();
    let width = 1920;
    let height = 1080;

    let start = std::time::Instant::now();

    for frame_num in 0..300 {
        let frame = vec![128u8; width * height];
        assessor
            .process_frame(&frame, width, height, frame_num)
            .expect("unexpected None/Err");
    }

    let _stats = assessor.finalize();
    let elapsed = start.elapsed();

    println!("  Frames processed: 300");
    println!("  Resolution: {}x{}", width, height);
    println!("  Time: {:?}", elapsed);
    println!("  FPS: {:.2}", 300.0 / elapsed.as_secs_f64());
    println!("  Time per frame: {:?}\n", elapsed / 300);
}

fn benchmark_black_frame_detection() {
    println!("=== Black Frame Detection Benchmark ===");

    let mut detector = BlackFrameDetector::new(16, 10);
    let width = 1920;
    let height = 1080;

    let start = std::time::Instant::now();

    for frame_num in 0..300 {
        let brightness = if (100..150).contains(&frame_num) {
            0u8
        } else {
            128u8
        };
        let frame = vec![brightness; width * height];
        detector
            .process_frame(&frame, width, height, frame_num)
            .expect("unexpected None/Err");
    }

    let _segments = detector.finalize();
    let elapsed = start.elapsed();

    println!("  Frames processed: 300");
    println!("  Resolution: {}x{}", width, height);
    println!("  Time: {:?}", elapsed);
    println!("  FPS: {:.2}", 300.0 / elapsed.as_secs_f64());
    println!("  Time per frame: {:?}\n", elapsed / 300);
}

fn benchmark_content_classification() {
    println!("=== Content Classification Benchmark ===");

    let mut classifier = ContentClassifier::new();
    let width = 1920;
    let height = 1080;

    let start = std::time::Instant::now();

    for frame_num in 0..300 {
        let frame = vec![128u8; width * height];
        classifier
            .process_frame(&frame, width, height, frame_num)
            .expect("unexpected None/Err");
    }

    let _classification = classifier.finalize();
    let elapsed = start.elapsed();

    println!("  Frames processed: 300");
    println!("  Resolution: {}x{}", width, height);
    println!("  Time: {:?}", elapsed);
    println!("  FPS: {:.2}", 300.0 / elapsed.as_secs_f64());
    println!("  Time per frame: {:?}\n", elapsed / 300);
}

fn benchmark_thumbnail_selection() {
    println!("=== Thumbnail Selection Benchmark ===");

    let mut selector = ThumbnailSelector::new(10);
    let width = 1920;
    let height = 1080;

    let start = std::time::Instant::now();

    for frame_num in 0..300 {
        let frame = vec![128u8; width * height];
        selector
            .process_frame(&frame, width, height, frame_num)
            .expect("unexpected None/Err");
    }

    let _thumbnails = selector.finalize();
    let elapsed = start.elapsed();

    println!("  Frames processed: 300");
    println!("  Resolution: {}x{}", width, height);
    println!("  Time: {:?}", elapsed);
    println!("  FPS: {:.2}", 300.0 / elapsed.as_secs_f64());
    println!("  Time per frame: {:?}\n", elapsed / 300);
}

fn benchmark_motion_analysis() {
    println!("=== Motion Analysis Benchmark ===");

    let mut analyzer = MotionAnalyzer::new();
    let width = 1920;
    let height = 1080;

    let start = std::time::Instant::now();

    for frame_num in 0..300 {
        let mut frame = vec![100u8; width * height];

        // Add moving content
        let bar_x = (frame_num * 20) % width;
        for y in 0..height {
            for x in bar_x..bar_x + 50.min(width - bar_x) {
                frame[y * width + x] = 200;
            }
        }

        analyzer
            .process_frame(&frame, width, height, frame_num)
            .expect("unexpected None/Err");
    }

    let _stats = analyzer.finalize();
    let elapsed = start.elapsed();

    println!("  Frames processed: 300");
    println!("  Resolution: {}x{}", width, height);
    println!("  Time: {:?}", elapsed);
    println!("  FPS: {:.2}", 300.0 / elapsed.as_secs_f64());
    println!("  Time per frame: {:?}\n", elapsed / 300);
}

fn benchmark_color_analysis() {
    println!("=== Color Analysis Benchmark ===");

    let mut analyzer = ColorAnalyzer::new(5);
    let width = 1920;
    let height = 1080;
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;

    let start = std::time::Instant::now();

    for frame_num in 0..300 {
        let y_plane = vec![128u8; width * height];
        let u_plane = vec![128u8; uv_width * uv_height];
        let v_plane = vec![128u8; uv_width * uv_height];
        analyzer
            .process_frame(&y_plane, &u_plane, &v_plane, width, height, frame_num)
            .expect("unexpected None/Err");
    }

    let _analysis = analyzer.finalize();
    let elapsed = start.elapsed();

    println!("  Frames processed: 300");
    println!("  Resolution: {}x{}", width, height);
    println!("  Time: {:?}", elapsed);
    println!("  FPS: {:.2}", 300.0 / elapsed.as_secs_f64());
    println!("  Time per frame: {:?}\n", elapsed / 300);
}

fn benchmark_temporal_analysis() {
    println!("=== Temporal Analysis Benchmark ===");

    let mut analyzer = TemporalAnalyzer::new();
    let width = 1920;
    let height = 1080;

    let start = std::time::Instant::now();

    for frame_num in 0..300 {
        let frame = vec![128u8; width * height];
        analyzer
            .process_frame(&frame, width, height, frame_num)
            .expect("unexpected None/Err");
    }

    let _analysis = analyzer.finalize();
    let elapsed = start.elapsed();

    println!("  Frames processed: 300");
    println!("  Resolution: {}x{}", width, height);
    println!("  Time: {:?}", elapsed);
    println!("  FPS: {:.2}", 300.0 / elapsed.as_secs_f64());
    println!("  Time per frame: {:?}\n", elapsed / 300);
}

fn benchmark_audio_analysis() {
    println!("=== Audio Analysis Benchmark ===");

    let mut analyzer = AudioAnalyzer::new(-60.0, Duration::from_millis(500));
    let sample_rate = 48000;

    let start = std::time::Instant::now();

    // Generate 10 seconds of audio
    let mut audio = Vec::new();
    for i in 0..(sample_rate * 10) {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
        audio.push(sample);
        audio.push(sample); // Stereo
    }

    analyzer
        .process_samples(&audio, sample_rate)
        .expect("sample processing should succeed");
    let _analysis = analyzer.finalize();
    let elapsed = start.elapsed();

    println!("  Duration: 10 seconds");
    println!("  Sample rate: {} Hz", sample_rate);
    println!("  Samples: {}", audio.len());
    println!("  Time: {:?}", elapsed);
    println!("  Real-time factor: {:.2}x\n", 10.0 / elapsed.as_secs_f64());
}

fn benchmark_full_pipeline() {
    println!("=== Full Analysis Pipeline Benchmark ===");

    let config = AnalysisConfig::default()
        .with_scene_detection(true)
        .with_quality_assessment(true)
        .with_black_frame_detection(true)
        .with_content_classification(true)
        .with_thumbnail_generation(10)
        .with_motion_analysis(true)
        .with_color_analysis(true)
        .with_temporal_analysis(true)
        .with_audio_analysis(true);

    let mut analyzer = Analyzer::new(config);

    let width = 1920;
    let height = 1080;
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;

    let start = std::time::Instant::now();

    // Process video
    for _frame_num in 0..300 {
        let y_plane = vec![128u8; width * height];
        let u_plane = vec![128u8; uv_width * uv_height];
        let v_plane = vec![128u8; uv_width * uv_height];
        analyzer
            .process_video_frame(
                &y_plane,
                &u_plane,
                &v_plane,
                width,
                height,
                Rational::new(30, 1),
            )
            .expect("unexpected None/Err");
    }

    // Process audio
    let mut audio = Vec::new();
    for i in 0..(48000 * 10) {
        let t = i as f32 / 48000.0;
        let sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
        audio.push(sample);
        audio.push(sample);
    }
    analyzer
        .process_audio_samples(&audio, 48000)
        .expect("audio processing should succeed");

    let _results = analyzer.finalize();
    let elapsed = start.elapsed();

    println!("  Video frames: 300");
    println!("  Resolution: {}x{}", width, height);
    println!("  Audio duration: 10 seconds");
    println!("  Total time: {:?}", elapsed);
    println!("  Video FPS: {:.2}", 300.0 / elapsed.as_secs_f64());
    println!(
        "  Real-time factor (audio): {:.2}x\n",
        10.0 / elapsed.as_secs_f64()
    );
}

fn benchmark_different_resolutions() {
    println!("=== Resolution Scaling Benchmark ===\n");

    let resolutions = vec![
        (640, 480, "480p"),
        (1280, 720, "720p"),
        (1920, 1080, "1080p"),
        (3840, 2160, "4K"),
    ];

    for (width, height, name) in resolutions {
        let mut detector = SceneDetector::new(0.3);

        let start = std::time::Instant::now();

        for frame_num in 0..100 {
            let frame = vec![128u8; width * height];
            detector
                .process_frame(&frame, width, height, frame_num)
                .expect("unexpected None/Err");
        }

        let _scenes = detector.finalize();
        let elapsed = start.elapsed();

        println!("  Resolution: {} ({}x{})", name, width, height);
        println!("    Time: {:?}", elapsed);
        println!("    FPS: {:.2}", 100.0 / elapsed.as_secs_f64());
        println!("    Time per frame: {:?}", elapsed / 100);
        println!(
            "    Megapixels/sec: {:.2}\n",
            (width * height * 100) as f64 / 1_000_000.0 / elapsed.as_secs_f64()
        );
    }
}

fn benchmark_memory_usage() {
    println!("=== Memory Usage Analysis ===\n");

    // Measure memory for different analyzers
    println!("  Estimated memory usage per analyzer:");

    // Scene detector (stores histogram and edges)
    let scene_mem = std::mem::size_of::<SceneDetector>()
        + 256 * std::mem::size_of::<usize>() // Histogram
        + 1920 * 1080 * std::mem::size_of::<bool>(); // Edge map
    println!(
        "    SceneDetector: ~{:.2} MB",
        scene_mem as f64 / 1_000_000.0
    );

    // Quality assessor (stores previous frame)
    let quality_mem = std::mem::size_of::<QualityAssessor>() + 1920 * 1080;
    println!(
        "    QualityAssessor: ~{:.2} MB",
        quality_mem as f64 / 1_000_000.0
    );

    // Motion analyzer (stores previous frame)
    let motion_mem = std::mem::size_of::<MotionAnalyzer>() + 1920 * 1080;
    println!(
        "    MotionAnalyzer: ~{:.2} MB",
        motion_mem as f64 / 1_000_000.0
    );

    // Color analyzer (stores samples)
    let color_samples = (1920 / 8) * (1080 / 8) * 300; // Sampled pixels over 300 frames
    let color_mem = std::mem::size_of::<ColorAnalyzer>() + color_samples * 3;
    println!(
        "    ColorAnalyzer: ~{:.2} MB",
        color_mem as f64 / 1_000_000.0
    );

    // Temporal analyzer
    let temporal_mem = std::mem::size_of::<TemporalAnalyzer>()
        + 1920 * 1080 // Previous frame
        + 300 * std::mem::size_of::<f64>(); // Brightness history
    println!(
        "    TemporalAnalyzer: ~{:.2} MB",
        temporal_mem as f64 / 1_000_000.0
    );

    // Audio analyzer
    let audio_mem =
        std::mem::size_of::<AudioAnalyzer>() + 48000 * 10 * std::mem::size_of::<f32>() * 2; // 10 sec stereo
    println!(
        "    AudioAnalyzer: ~{:.2} MB",
        audio_mem as f64 / 1_000_000.0
    );

    let total = scene_mem + quality_mem + motion_mem + color_mem + temporal_mem + audio_mem;
    println!("\n  Total estimated: ~{:.2} MB", total as f64 / 1_000_000.0);
    println!("  Note: Actual memory usage may vary based on video content and duration.");
}
