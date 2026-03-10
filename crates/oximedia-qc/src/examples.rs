//! Examples and usage patterns for oximedia-qc.
//!
//! This module provides documented examples of common QC workflows.

#![allow(dead_code)]

use crate::{
    audio, codec_validation, format, standards, temporal, video, BatchProcessor, ProfileManager,
    QcPreset, QcProfile, QualityControl, Thresholds,
};

/// Example: Basic file validation.
///
/// Validates a single file using the basic preset.
pub fn example_basic_validation() {
    let _qc = QualityControl::with_preset(QcPreset::Basic);

    // In production:
    // let report = qc.validate("video.mkv").expect("operation should succeed");
    // if report.overall_passed {
    //     println!("Validation passed!");
    // } else {
    //     println!("Validation failed:");
    //     for error in report.errors() {
    //         println!("  - {}", error.message);
    //     }
    // }
}

/// Example: Custom validation rules.
///
/// Creates a custom QC configuration with specific rules.
pub fn example_custom_rules() {
    let mut qc = QualityControl::new();

    // Add video quality rules
    qc.add_rule(Box::new(video::VideoCodecValidation));
    qc.add_rule(Box::new(
        video::ResolutionValidation::new()
            .with_min_resolution(1920, 1080)
            .with_max_resolution(3840, 2160),
    ));
    qc.add_rule(Box::new(video::FrameRateValidation::new()));

    // Add audio quality rules
    qc.add_rule(Box::new(audio::AudioCodecValidation));
    qc.add_rule(Box::new(audio::SampleRateValidation::new()));

    // Add temporal checks
    qc.add_rule(Box::new(temporal::DroppedFrameDetection::new()));
    qc.add_rule(Box::new(temporal::DuplicateFrameDetection::new()));

    // In production:
    // let report = qc.validate("video.mkv").expect("operation should succeed");
}

/// Example: Broadcast delivery validation.
///
/// Validates a file for broadcast delivery with strict requirements.
pub fn example_broadcast_validation() {
    let thresholds = Thresholds::new()
        .with_min_video_bitrate(10_000_000)
        .with_loudness_target(-23.0);

    let mut qc = QualityControl::with_thresholds(thresholds);

    // Video requirements
    qc.add_rule(Box::new(video::VideoCodecValidation));
    qc.add_rule(Box::new(
        video::ResolutionValidation::new().with_min_resolution(1280, 720), // HD minimum
    ));
    qc.add_rule(Box::new(video::InterlacingDetection));
    qc.add_rule(Box::new(video::BlackFrameDetection::default()));

    // Audio requirements
    qc.add_rule(Box::new(audio::AudioCodecValidation));
    qc.add_rule(Box::new(audio::LoudnessCompliance::ebu_r128(thresholds)));
    qc.add_rule(Box::new(audio::ClippingDetection::new()));
    qc.add_rule(Box::new(audio::SilenceDetection::new(&thresholds)));

    // Standards compliance
    qc.add_rule(Box::new(standards::EbuR128Validator::new()));
    qc.add_rule(Box::new(standards::SmpteValidator::new()));

    // In production:
    // let report = qc.validate("broadcast.mxf").expect("operation should succeed");
}

/// Example: Streaming platform validation.
///
/// Validates files for YouTube/Vimeo upload.
pub fn example_streaming_validation() {
    let _qc = QualityControl::with_preset(QcPreset::YouTube);

    // In production:
    // let report = qc.validate_streaming("upload.webm").expect("operation should succeed");
    //
    // // Export report as JSON
    // let json = report.to_json().expect("operation should succeed");
    // std::fs::write("qc_report.json", json).expect("operation should succeed");
}

/// Example: Batch processing multiple files.
///
/// Processes a directory of files in parallel.
pub fn example_batch_processing() {
    let qc = QualityControl::with_preset(QcPreset::Comprehensive);
    let _processor = BatchProcessor::new(qc)
        .with_parallel_jobs(4)
        .with_detailed_reports(false);

    // In production:
    // let results = processor.process_directory(
    //     Path::new("/media/videos"),
    //     "*.mkv"
    // ).expect("operation should succeed");
    //
    // println!("{}", results.summary());
    // println!("Passed: {}/{}", results.passed, results.total_files);
}

/// Example: Using QC profiles.
///
/// Loads and uses predefined QC profiles.
pub fn example_qc_profiles() {
    let manager = ProfileManager::new();

    // List available profiles
    let _profiles = manager.list_profiles();
    // println!("Available profiles: {:?}", profiles);

    // Get Netflix profile
    if let Some(netflix_profile) = manager.get_profile("netflix") {
        // Apply profile rules to QC
        // let qc = apply_profile(netflix_profile);
        let _ = netflix_profile;
    }

    // Create custom profile
    let custom = QcProfile::new("my_custom", "Custom validation profile")
        .with_rule("video_codec_validation")
        .with_rule("audio_codec_validation")
        .with_thresholds(
            Thresholds::new()
                .with_min_video_bitrate(5_000_000)
                .with_loudness_target(-24.0),
        );

    // In production:
    // let json = custom.to_json().expect("operation should succeed");
    // std::fs::write("custom_profile.json", json).expect("operation should succeed");
    let _ = custom;
}

/// Example: Database integration.
///
/// Stores and retrieves QC results from database.
#[cfg(feature = "database")]
#[allow(unused_imports)] // Used in commented example code
pub fn example_database_integration() {
    use std::path::Path; // Used in commented example code

    // Open database
    // let mut db = QcDatabase::open("qc_results.db").expect("operation should succeed");

    // Run QC and store results
    // let _qc = QualityControl::with_preset(QcPreset::Comprehensive);
    // let report = qc.validate("video.mkv").expect("operation should succeed");
    // let report_id = db.store_report(&report).expect("operation should succeed");

    // Retrieve historical results
    // let reports = db.get_reports_for_file("video.mkv").expect("operation should succeed");
    // for report in reports {
    //     println!("Report from: {}", report.timestamp);
    // }

    // Get statistics
    // let stats = db.get_file_statistics("video.mkv").expect("operation should succeed");
    // println!("Total runs: {}", stats.total_runs);
    // println!("Pass rate: {:.1}%",
    //     100.0 * stats.passed_runs as f64 / stats.total_runs as f64
    // );
}

/// Example: Format-specific validation.
///
/// Validates MP4 container compliance.
pub fn example_mp4_validation() {
    let mut qc = QualityControl::new();

    qc.add_rule(Box::new(
        format::Mp4Validator::new()
            .with_fast_start_check(true)
            .with_fragmentation_check(true),
    ));

    // In production:
    // let report = qc.validate("video.mp4").expect("operation should succeed");
}

/// Example: Matroska/WebM validation.
///
/// Validates Matroska container compliance.
pub fn example_matroska_validation() {
    let mut qc = QualityControl::new();

    qc.add_rule(Box::new(
        format::MatroskaValidator::new()
            .with_cues_check(true)
            .with_seekhead_check(true),
    ));

    // In production:
    // let report = qc.validate("video.mkv").expect("operation should succeed");
}

/// Example: MXF validation.
///
/// Validates MXF files for broadcast delivery.
pub fn example_mxf_validation() {
    let mut qc = QualityControl::new();

    qc.add_rule(Box::new(
        format::MxfValidator::new()
            .with_operational_pattern(format::mxf::OperationalPattern::Op1a)
            .with_profile(format::mxf::MxfProfile::As11),
    ));

    // In production:
    // let report = qc.validate("broadcast.mxf").expect("operation should succeed");
}

/// Example: MPEG-TS validation.
///
/// Validates MPEG Transport Stream files.
pub fn example_mpegts_validation() {
    let mut qc = QualityControl::new();

    qc.add_rule(Box::new(
        format::MpegTsValidator::new()
            .with_continuity_check(true)
            .with_pcr_check(true)
            .with_max_cc_errors(0),
    ));

    // In production:
    // let report = qc.validate("stream.ts").expect("operation should succeed");
}

/// Example: Codec bitstream validation.
///
/// Validates AV1 bitstream compliance.
pub fn example_av1_validation() {
    let mut qc = QualityControl::new();

    qc.add_rule(Box::new(
        codec_validation::Av1BitstreamValidator::new()
            .with_profile(codec_validation::av1::Av1Profile::Main)
            .with_level(codec_validation::av1::Av1Level::Level4_0),
    ));

    // In production:
    // let report = qc.validate("video.mkv").expect("operation should succeed");
}

/// Example: Professional standards validation.
///
/// Validates against EBU R128 and DPP standards.
pub fn example_professional_standards() {
    let mut qc = QualityControl::new();

    // EBU R128 loudness
    qc.add_rule(Box::new(
        standards::EbuR128Validator::new()
            .with_target_loudness(-23.0)
            .with_max_true_peak(-1.0),
    ));

    // SMPTE standards
    qc.add_rule(Box::new(standards::SmpteValidator::new()));

    // DPP compliance
    qc.add_rule(Box::new(standards::DppValidator::new()));

    // In production:
    // let report = qc.validate("programme.mxf").expect("operation should succeed");
}

/// Example: Temporal quality checks.
///
/// Detects dropped frames, duplicates, and timecode issues.
pub fn example_temporal_checks() {
    let mut qc = QualityControl::new();

    qc.add_rule(Box::new(
        temporal::DroppedFrameDetection::new().with_tolerance(0),
    ));

    qc.add_rule(Box::new(
        temporal::DuplicateFrameDetection::new().with_max_consecutive(3),
    ));

    qc.add_rule(Box::new(temporal::TimecodeContinuity::new()));
    qc.add_rule(Box::new(temporal::DurationAccuracy::new()));

    // In production:
    // let report = qc.validate("video.mkv").expect("operation should succeed");
}

/// Example: Report generation and export.
///
/// Generates reports in various formats.
#[cfg(feature = "json")]
pub fn example_report_generation() {
    let _qc = QualityControl::with_preset(QcPreset::Comprehensive);

    // In production:
    // let report = qc.validate("video.mkv").expect("operation should succeed");
    //
    // // Export as JSON
    // let json = report.to_json().expect("operation should succeed");
    // std::fs::write("report.json", json).expect("operation should succeed");
    //
    // // Export as XML
    // #[cfg(feature = "xml")]
    // {
    //     let xml = report.to_xml().expect("operation should succeed");
    //     std::fs::write("report.xml", xml).expect("operation should succeed");
    // }
    //
    // // Print text summary
    // println!("{}", report.summary());
    //
    // // Print detailed report
    // println!("{}", report);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_examples_compile() {
        // Just verify examples compile
        example_basic_validation();
        example_custom_rules();
        example_broadcast_validation();
        example_streaming_validation();
        example_batch_processing();
        example_qc_profiles();
        example_mp4_validation();
        example_matroska_validation();
        example_mxf_validation();
        example_mpegts_validation();
        example_av1_validation();
        example_professional_standards();
        example_temporal_checks();
    }
}
