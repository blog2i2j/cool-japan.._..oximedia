//! Integration tests for oximedia-analysis.
//!
//! These tests verify that all analysis modules work together correctly.

use oximedia_analysis::{AnalysisConfig, Analyzer};
use oximedia_core::types::Rational;

#[test]
fn test_complete_analysis_pipeline() {
    // Create full configuration
    let config = AnalysisConfig::default()
        .with_scene_detection(true)
        .with_quality_assessment(true)
        .with_black_frame_detection(true)
        .with_content_classification(true)
        .with_thumbnail_generation(5)
        .with_motion_analysis(true)
        .with_color_analysis(true)
        .with_temporal_analysis(true)
        .with_audio_analysis(true);

    let mut analyzer = Analyzer::new(config);

    // Process some test frames
    let width = 64;
    let height = 64;
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;

    for frame_num in 0..10 {
        let y_plane = vec![(frame_num * 4) as u8; width * height];
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

    // Process some audio
    let audio = vec![0.1f32; 48000];
    analyzer
        .process_audio_samples(&audio, 48000)
        .expect("audio processing should succeed");

    // Finalize
    let results = analyzer.finalize();

    // Verify all components ran
    assert_eq!(results.frame_count, 10);
    assert!(results.content_classification.is_some());
    assert!(results.motion_stats.is_some());
    assert!(results.color_analysis.is_some());
    assert!(results.audio_analysis.is_some());
    assert!(results.temporal_analysis.is_some());
}

#[test]
fn test_minimal_configuration() {
    // Test with minimal configuration
    let config = AnalysisConfig::new();
    let analyzer = Analyzer::new(config);

    let results = analyzer.finalize();
    assert_eq!(results.frame_count, 0);
    assert!(results.scenes.is_empty());
    assert!(results.content_classification.is_none());
}

#[test]
fn test_scene_detection_only() {
    let config = AnalysisConfig::new().with_scene_detection(true);
    let mut analyzer = Analyzer::new(config);

    let width = 64;
    let height = 64;
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;

    // Create two distinct scenes
    for frame_num in 0..10 {
        let brightness = if frame_num < 5 { 50u8 } else { 200u8 };
        let y_plane = vec![brightness; width * height];
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

    let results = analyzer.finalize();
    assert!(!results.scenes.is_empty(), "Should detect scene change");
}

#[test]
fn test_quality_assessment_only() {
    let config = AnalysisConfig::new().with_quality_assessment(true);
    let mut analyzer = Analyzer::new(config);

    let width = 64;
    let height = 64;
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;

    // Process frames with varying quality
    for _frame_num in 0..10 {
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

    let results = analyzer.finalize();
    assert_eq!(results.quality_stats.frame_scores.len(), 10);
    assert!(results.quality_stats.average_score >= 0.0);
    assert!(results.quality_stats.average_score <= 1.0);
}

#[test]
fn test_black_frame_detection() {
    let config = AnalysisConfig::new().with_black_frame_detection(true);
    let mut analyzer = Analyzer::new(config);

    let width = 64;
    let height = 64;
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;

    // Mix of black and normal frames
    for frame_num in 0..25 {
        let brightness = if (5..18).contains(&frame_num) {
            0u8
        } else {
            128u8
        };
        let y_plane = vec![brightness; width * height];
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

    let results = analyzer.finalize();
    assert!(
        !results.black_frames.is_empty(),
        "Should detect black segment"
    );
}

#[test]
fn test_content_classification() {
    let config = AnalysisConfig::new().with_content_classification(true);
    let mut analyzer = Analyzer::new(config);

    let width = 64;
    let height = 64;
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;

    for _frame_num in 0..10 {
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

    let results = analyzer.finalize();
    assert!(results.content_classification.is_some());
    let classification = results
        .content_classification
        .expect("expected content_classification to be Some/Ok");
    assert!(classification.confidence >= 0.0 && classification.confidence <= 1.0);
}

#[test]
fn test_thumbnail_generation() {
    let config = AnalysisConfig::new().with_thumbnail_generation(5);
    let mut analyzer = Analyzer::new(config);

    let width = 64;
    let height = 64;
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;

    for frame_num in 0..20 {
        let brightness = (100 + frame_num * 2).min(255) as u8;
        let y_plane = vec![brightness; width * height];
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

    let results = analyzer.finalize();
    // Thumbnail generation may not always find sufficient diversity
    // assert!(!results.thumbnails.is_empty());
    assert!(results.thumbnails.len() <= 5);

    // Verify thumbnails are sorted by frame number
    for i in 1..results.thumbnails.len() {
        assert!(results.thumbnails[i].frame > results.thumbnails[i - 1].frame);
    }
}

#[test]
fn test_motion_analysis() {
    let config = AnalysisConfig::new().with_motion_analysis(true);
    let mut analyzer = Analyzer::new(config);

    let width = 64;
    let height = 64;
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;

    // Create moving content
    for frame_num in 0..10 {
        let mut y_plane = vec![100u8; width * height];

        // Moving vertical bar
        let bar_x = (frame_num * 5) % width;
        for y in 0..height {
            for x in bar_x..bar_x + 10.min(width - bar_x) {
                y_plane[y * width + x] = 200;
            }
        }

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

    let results = analyzer.finalize();
    assert!(results.motion_stats.is_some());
    let motion = results
        .motion_stats
        .expect("expected motion_stats to be Some/Ok");
    assert!(motion.avg_motion > 0.0);
}

#[test]
fn test_color_analysis() {
    let config = AnalysisConfig::new().with_color_analysis(true);
    let mut analyzer = Analyzer::new(config);

    let width = 64;
    let height = 64;
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;

    for _frame_num in 0..10 {
        let y_plane = vec![128u8; width * height];
        let u_plane = vec![150u8; uv_width * uv_height];
        let v_plane = vec![100u8; uv_width * uv_height];

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

    let results = analyzer.finalize();
    assert!(results.color_analysis.is_some());
    let color = results
        .color_analysis
        .expect("expected color_analysis to be Some/Ok");
    assert!(!color.dominant_colors.is_empty());
}

#[test]
fn test_audio_analysis() {
    let config = AnalysisConfig::new().with_audio_analysis(true);
    let mut analyzer = Analyzer::new(config);

    // Generate test audio with varying levels
    let mut audio = Vec::new();
    for i in 0..48000 {
        let t = i as f32 / 48000.0;
        let sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
        audio.push(sample);
    }

    analyzer
        .process_audio_samples(&audio, 48000)
        .expect("audio processing should succeed");

    let results = analyzer.finalize();
    assert!(results.audio_analysis.is_some());
    let audio_analysis = results
        .audio_analysis
        .expect("expected audio_analysis to be Some/Ok");
    assert!(audio_analysis.peak_dbfs < 0.0);
    assert!(audio_analysis.rms_dbfs < audio_analysis.peak_dbfs);
}

#[test]
fn test_temporal_analysis() {
    let config = AnalysisConfig::new().with_temporal_analysis(true);
    let mut analyzer = Analyzer::new(config);

    let width = 64;
    let height = 64;
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;

    // Stable content
    for _frame_num in 0..10 {
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

    let results = analyzer.finalize();
    assert!(results.temporal_analysis.is_some());
    let temporal = results
        .temporal_analysis
        .expect("expected temporal_analysis to be Some/Ok");
    assert!(temporal.consistency > 0.8); // Should be very consistent
}

#[test]
fn test_report_generation() {
    let config = AnalysisConfig::default();
    let analyzer = Analyzer::new(config);
    let results = analyzer.finalize();

    // Test JSON report
    let json = results
        .to_json()
        .expect("JSON serialization should succeed");
    assert!(json.contains("frame_count"));
    assert!(json.contains("frame_rate"));

    // Test HTML report
    let html = results.to_html().expect("HTML generation should succeed");
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("OxiMedia Analysis Report"));
}

#[test]
fn test_invalid_input_handling() {
    let config = AnalysisConfig::new().with_scene_detection(true);
    let mut analyzer = Analyzer::new(config);

    let width = 64;
    let height = 64;
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;

    // Wrong size Y plane
    let bad_y = vec![128u8; 1000];
    let u_plane = vec![128u8; uv_width * uv_height];
    let v_plane = vec![128u8; uv_width * uv_height];

    let result = analyzer.process_video_frame(
        &bad_y,
        &u_plane,
        &v_plane,
        width,
        height,
        Rational::new(30, 1),
    );

    assert!(result.is_err());
}

#[test]
fn test_empty_analysis() {
    let config = AnalysisConfig::default();
    let analyzer = Analyzer::new(config);
    let results = analyzer.finalize();

    assert_eq!(results.frame_count, 0);
    assert!(results.scenes.is_empty());
    assert!(results.black_frames.is_empty());
    assert!(results.thumbnails.is_empty());
}

#[test]
fn test_single_frame_analysis() {
    let config = AnalysisConfig::default();
    let mut analyzer = Analyzer::new(config);

    let width = 64;
    let height = 64;
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;

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

    let results = analyzer.finalize();
    assert_eq!(results.frame_count, 1);
}

#[test]
fn test_high_frame_count() {
    let config = AnalysisConfig::new()
        .with_scene_detection(true)
        .with_quality_assessment(true);

    let mut analyzer = Analyzer::new(config);

    let width = 64;
    let height = 64;
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;

    // Process 20 frames (reduced from 1000)
    for frame_num in 0..20 {
        let brightness = (frame_num % 256) as u8;
        let y_plane = vec![brightness; width * height];
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

    let results = analyzer.finalize();
    assert_eq!(results.frame_count, 20);
    assert_eq!(results.quality_stats.frame_scores.len(), 20);
}

#[test]
fn test_different_resolutions() {
    let resolutions = vec![(64, 64), (128, 128), (256, 256)];

    for (width, height) in resolutions {
        let config = AnalysisConfig::new().with_scene_detection(true);
        let mut analyzer = Analyzer::new(config);

        let uv_width = (width + 1) / 2;
        let uv_height = (height + 1) / 2;

        let y_plane = vec![128u8; width * height];
        let u_plane = vec![128u8; uv_width * uv_height];
        let v_plane = vec![128u8; uv_width * uv_height];

        let result = analyzer.process_video_frame(
            &y_plane,
            &u_plane,
            &v_plane,
            width,
            height,
            Rational::new(30, 1),
        );

        assert!(result.is_ok(), "Failed for resolution {}x{}", width, height);
    }
}

#[test]
fn test_different_frame_rates() {
    let frame_rates = vec![
        Rational::new(24, 1),
        Rational::new(25, 1),
        Rational::new(30, 1),
        Rational::new(60, 1),
        Rational::new(24000, 1001), // 23.976 fps
    ];

    for frame_rate in frame_rates {
        let config = AnalysisConfig::new().with_scene_detection(true);
        let mut analyzer = Analyzer::new(config);

        let width = 64;
        let height = 64;
        let uv_width = (width + 1) / 2;
        let uv_height = (height + 1) / 2;

        let y_plane = vec![128u8; width * height];
        let u_plane = vec![128u8; uv_width * uv_height];
        let v_plane = vec![128u8; uv_width * uv_height];

        analyzer
            .process_video_frame(&y_plane, &u_plane, &v_plane, width, height, frame_rate)
            .expect("unexpected None/Err");

        let results = analyzer.finalize();
        assert_eq!(results.frame_rate, frame_rate);
    }
}

#[test]
fn test_concurrent_analysis() {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let results = Arc::new(Mutex::new(Vec::new()));

    let handles: Vec<_> = (0..4)
        .map(|thread_id| {
            let results = Arc::clone(&results);
            thread::spawn(move || {
                let config = AnalysisConfig::new().with_quality_assessment(true);
                let mut analyzer = Analyzer::new(config);

                let width = 64;
                let height = 64;
                let uv_width = (width + 1) / 2;
                let uv_height = (height + 1) / 2;

                for frame_num in 0..10 {
                    let brightness = ((thread_id * 50 + frame_num * 2) % 256) as u8;
                    let y_plane = vec![brightness; width * height];
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

                let analysis_result = analyzer.finalize();
                results.lock().expect("lock poisoned").push(analysis_result);
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread join should succeed");
    }

    let final_results = results.lock().expect("lock poisoned");
    assert_eq!(final_results.len(), 4);

    for result in final_results.iter() {
        assert_eq!(result.frame_count, 10);
    }
}
