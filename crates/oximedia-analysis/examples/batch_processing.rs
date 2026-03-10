//! Batch processing example.
//!
//! This example demonstrates how to analyze multiple video files in parallel
//! using rayon for efficient batch processing.

use oximedia_analysis::{AnalysisConfig, Analyzer};
use oximedia_core::types::Rational;
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Represents a video file to be analyzed.
struct VideoFile {
    path: PathBuf,
    width: usize,
    height: usize,
    frame_count: usize,
}

/// Analysis job result.
struct AnalysisJob {
    file_path: PathBuf,
    success: bool,
    scene_count: usize,
    quality_score: f64,
    duration_ms: u128,
    error: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OxiMedia Batch Analysis Example\n");

    // Create a list of video files to analyze
    let video_files = vec![
        VideoFile {
            path: PathBuf::from("/tmp/video1.yuv"),
            width: 1920,
            height: 1080,
            frame_count: 300,
        },
        VideoFile {
            path: PathBuf::from("/tmp/video2.yuv"),
            width: 1920,
            height: 1080,
            frame_count: 450,
        },
        VideoFile {
            path: PathBuf::from("/tmp/video3.yuv"),
            width: 1280,
            height: 720,
            frame_count: 600,
        },
        VideoFile {
            path: PathBuf::from("/tmp/video4.yuv"),
            width: 1920,
            height: 1080,
            frame_count: 900,
        },
    ];

    println!("Files to analyze: {}\n", video_files.len());

    // Create analysis configuration
    let config = AnalysisConfig::default()
        .with_scene_detection(true)
        .with_quality_assessment(true)
        .with_black_frame_detection(true)
        .with_content_classification(true)
        .with_thumbnail_generation(5);

    // Progress tracking
    let progress = Arc::new(Mutex::new(0usize));
    let total = video_files.len();

    // Process files in parallel
    println!("Starting parallel analysis...");

    let results: Vec<AnalysisJob> = video_files
        .par_iter()
        .map(|video_file| {
            let result = analyze_video_file(video_file, &config);

            // Update progress
            let mut prog = progress.lock().expect("lock poisoned");
            *prog += 1;
            println!(
                "[{}/{}] Analyzed: {}",
                *prog,
                total,
                video_file.path.display()
            );

            result
        })
        .collect();

    // Print summary
    println!("\n=== Batch Analysis Summary ===\n");

    let successful = results.iter().filter(|r| r.success).count();
    let failed = results.len() - successful;

    println!("Total files: {}", results.len());
    println!("Successful: {}", successful);
    println!("Failed: {}", failed);
    println!();

    // Print individual results
    println!("{:-<100}", "");
    println!(
        "{:<30} {:<10} {:<12} {:<12} {:<15}",
        "File", "Status", "Scenes", "Quality", "Duration (ms)"
    );
    println!("{:-<100}", "");

    for result in &results {
        let status = if result.success { "OK" } else { "FAILED" };
        let filename = result
            .file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        println!(
            "{:<30} {:<10} {:<12} {:<12.2} {:<15}",
            filename, status, result.scene_count, result.quality_score, result.duration_ms
        );

        if let Some(ref error) = result.error {
            println!("  Error: {}", error);
        }
    }

    println!("{:-<100}", "");

    // Statistics
    if successful > 0 {
        let avg_scenes =
            results.iter().map(|r| r.scene_count).sum::<usize>() as f64 / successful as f64;

        let avg_quality = results
            .iter()
            .filter(|r| r.success)
            .map(|r| r.quality_score)
            .sum::<f64>()
            / successful as f64;

        let avg_duration =
            results.iter().map(|r| r.duration_ms).sum::<u128>() as f64 / successful as f64;

        println!("\nStatistics:");
        println!("  Average scenes per video: {:.1}", avg_scenes);
        println!("  Average quality score: {:.2}", avg_quality);
        println!("  Average processing time: {:.0} ms", avg_duration);
    }

    // Generate batch report
    generate_batch_report(&results)?;

    println!("\nBatch report saved to: /tmp/batch_analysis_report.html");

    Ok(())
}

/// Analyze a single video file.
fn analyze_video_file(video_file: &VideoFile, config: &AnalysisConfig) -> AnalysisJob {
    let start = std::time::Instant::now();

    // Create analyzer
    let mut analyzer = Analyzer::new(config.clone());

    // For this example, we generate synthetic data instead of reading files
    let result = (|| -> Result<_, Box<dyn std::error::Error>> {
        // Process frames
        for frame_num in 0..video_file.frame_count {
            let (y_plane, u_plane, v_plane) =
                generate_synthetic_frame(frame_num, video_file.width, video_file.height);

            analyzer.process_video_frame(
                &y_plane,
                &u_plane,
                &v_plane,
                video_file.width,
                video_file.height,
                Rational::new(30, 1),
            )?;
        }

        let results = analyzer.finalize();
        Ok(results)
    })();

    let duration_ms = start.elapsed().as_millis();

    match result {
        Ok(results) => AnalysisJob {
            file_path: video_file.path.clone(),
            success: true,
            scene_count: results.scenes.len(),
            quality_score: results.quality_stats.average_score,
            duration_ms,
            error: None,
        },
        Err(e) => AnalysisJob {
            file_path: video_file.path.clone(),
            success: false,
            scene_count: 0,
            quality_score: 0.0,
            duration_ms,
            error: Some(e.to_string()),
        },
    }
}

/// Generate synthetic frame data.
fn generate_synthetic_frame(
    frame_num: usize,
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let uv_width = (width + 1) / 2;
    let uv_height = (height + 1) / 2;

    // Vary brightness over time to create scene changes
    let scene_id = frame_num / 100;
    let base_brightness = match scene_id % 5 {
        0 => 50,
        1 => 120,
        2 => 200,
        3 => 80,
        _ => 150,
    };

    let y_plane = vec![base_brightness; width * height];
    let u_plane = vec![128u8; uv_width * uv_height];
    let v_plane = vec![128u8; uv_width * uv_height];

    (y_plane, u_plane, v_plane)
}

/// Generate HTML report for batch analysis.
fn generate_batch_report(results: &[AnalysisJob]) -> Result<(), Box<dyn std::error::Error>> {
    let mut html = String::new();

    html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
    html.push_str("<title>Batch Analysis Report</title>\n");
    html.push_str("<style>\n");
    html.push_str("body { font-family: Arial, sans-serif; margin: 20px; }\n");
    html.push_str("table { border-collapse: collapse; width: 100%; }\n");
    html.push_str("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
    html.push_str("th { background-color: #4CAF50; color: white; }\n");
    html.push_str("tr:nth-child(even) { background-color: #f2f2f2; }\n");
    html.push_str(".success { color: green; font-weight: bold; }\n");
    html.push_str(".failed { color: red; font-weight: bold; }\n");
    html.push_str(".stats { background-color: #e7f3ff; padding: 15px; margin: 20px 0; border-radius: 5px; }\n");
    html.push_str("</style>\n");
    html.push_str("</head>\n<body>\n");

    html.push_str("<h1>Batch Video Analysis Report</h1>\n");

    // Summary
    let successful = results.iter().filter(|r| r.success).count();
    let failed = results.len() - successful;

    html.push_str("<div class=\"stats\">\n");
    html.push_str("<h2>Summary</h2>\n");
    html.push_str(&format!("<p>Total files analyzed: {}</p>\n", results.len()));
    html.push_str(&format!("<p>Successful: {}</p>\n", successful));
    html.push_str(&format!("<p>Failed: {}</p>\n", failed));

    if successful > 0 {
        let avg_quality = results
            .iter()
            .filter(|r| r.success)
            .map(|r| r.quality_score)
            .sum::<f64>()
            / successful as f64;

        html.push_str(&format!(
            "<p>Average quality score: {:.2}</p>\n",
            avg_quality
        ));
    }

    html.push_str("</div>\n");

    // Results table
    html.push_str("<h2>Results</h2>\n");
    html.push_str("<table>\n");
    html.push_str(
        "<tr><th>File</th><th>Status</th><th>Scenes</th><th>Quality</th><th>Duration (ms)</th></tr>\n",
    );

    for result in results {
        let status_class = if result.success { "success" } else { "failed" };
        let status_text = if result.success { "OK" } else { "FAILED" };

        let filename = result
            .file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        html.push_str(&format!(
            "<tr><td>{}</td><td class=\"{}\">{}</td><td>{}</td><td>{:.2}</td><td>{}</td></tr>\n",
            filename,
            status_class,
            status_text,
            result.scene_count,
            result.quality_score,
            result.duration_ms
        ));
    }

    html.push_str("</table>\n");

    // Errors
    let errors: Vec<_> = results.iter().filter(|r| !r.success).collect();
    if !errors.is_empty() {
        html.push_str("<h2>Errors</h2>\n");
        html.push_str("<ul>\n");
        for error_result in errors {
            let filename = error_result
                .file_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");

            html.push_str(&format!(
                "<li><strong>{}</strong>: {}</li>\n",
                filename,
                error_result
                    .error
                    .as_ref()
                    .unwrap_or(&"Unknown error".to_string())
            ));
        }
        html.push_str("</ul>\n");
    }

    html.push_str("</body>\n</html>");

    std::fs::write("/tmp/batch_analysis_report.html", html)?;

    Ok(())
}

/// Example of analyzing with custom settings per file.
#[allow(dead_code)]
fn analyze_with_custom_settings() {
    println!("=== Custom Settings Per File ===\n");

    // Different configs for different content types
    let _hd_config = AnalysisConfig::default()
        .with_scene_detection(true)
        .with_quality_assessment(true);

    let _sd_config = AnalysisConfig::new()
        .with_scene_detection(true)
        .with_black_frame_detection(true);

    println!("HD Content: Using full quality assessment");
    println!("SD Content: Using simplified analysis\n");

    // Apply different configs based on resolution
    // (implementation details omitted for brevity)
}

/// Example of progress reporting with detailed statistics.
#[allow(dead_code)]
fn analyze_with_progress_reporting() {
    println!("=== Progress Reporting Example ===\n");

    let total_frames = 1000;
    let update_interval = 50;

    for frame_num in 0..total_frames {
        // Process frame...

        if frame_num % update_interval == 0 {
            let progress = (frame_num as f64 / total_frames as f64) * 100.0;
            print!("\rProgress: {:.1}%", progress);
            std::io::Write::flush(&mut std::io::stdout()).expect("unexpected None/Err");
        }
    }

    println!("\rProgress: 100.0%  ");
}

/// Example of memory-efficient streaming analysis.
#[allow(dead_code)]
fn analyze_large_file_streaming() {
    println!("=== Streaming Analysis Example ===\n");

    let config = AnalysisConfig::default();
    let analyzer = Analyzer::new(config);

    // Process in chunks
    const CHUNK_SIZE: usize = 100;
    let total_frames = 10000;

    for chunk_start in (0..total_frames).step_by(CHUNK_SIZE) {
        let chunk_end = (chunk_start + CHUNK_SIZE).min(total_frames);

        println!("Processing frames {}-{}...", chunk_start, chunk_end - 1);

        // Process chunk...
        for frame_num in chunk_start..chunk_end {
            // Generate and process frame
            let _ = frame_num;
        }

        // Periodic reporting
        let progress = (chunk_end as f64 / total_frames as f64) * 100.0;
        println!("  Overall progress: {:.1}%", progress);
    }

    let _results = analyzer.finalize();
    println!("\nStreaming analysis complete!");
}

/// Example of combining analysis with other processing.
#[allow(dead_code)]
fn analyze_with_processing_pipeline() {
    println!("=== Processing Pipeline Example ===\n");

    // Hypothetical pipeline:
    // 1. Decode video
    // 2. Analyze quality
    // 3. Apply corrections if needed
    // 4. Re-encode

    println!("Pipeline stages:");
    println!("  1. Decode");
    println!("  2. Analyze");
    println!("  3. Correct (if needed)");
    println!("  4. Encode\n");

    let quality_threshold = 0.7;

    // Analyze
    let quality_score = 0.65; // Simulated

    if quality_score < quality_threshold {
        println!("Quality score: {:.2}", quality_score);
        println!("  → Below threshold, applying corrections...");
        // Apply corrections...
        println!("  → Corrections applied");
    } else {
        println!("Quality score: {:.2}", quality_score);
        println!("  → Quality acceptable, no corrections needed");
    }
}

/// Example of distributed batch processing.
#[allow(dead_code)]
fn distributed_batch_analysis() {
    println!("=== Distributed Processing Example ===\n");

    // In a real system, this would distribute across multiple machines
    let worker_count = 4;
    let files_per_worker = 10;

    println!("Workers: {}", worker_count);
    println!("Files per worker: {}", files_per_worker);
    println!("Total files: {}\n", worker_count * files_per_worker);

    for worker_id in 0..worker_count {
        println!("Worker {}: Processing files...", worker_id);
        // Distribute work...
        std::thread::sleep(std::time::Duration::from_millis(100));
        println!("Worker {}: Complete", worker_id);
    }

    println!("\nAll workers finished!");
}
