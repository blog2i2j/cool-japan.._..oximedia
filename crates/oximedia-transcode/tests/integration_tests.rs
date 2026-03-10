//! Integration tests for the transcode crate.

use oximedia_transcode::*;

#[test]
fn test_transcoder_creation() {
    let transcoder = Transcoder::new()
        .input("/tmp/test_input.mp4")
        .output("/tmp/test_output.mp4")
        .video_codec("vp9")
        .audio_codec("opus");

    // Verify configuration was applied
    assert!(transcoder.config().input.is_some());
    assert!(transcoder.config().output.is_some());
}

#[test]
fn test_preset_application() {
    let preset = presets::youtube::youtube_1080p();
    assert_eq!(preset.width, Some(1920));
    assert_eq!(preset.height, Some(1080));
    assert_eq!(preset.video_codec, Some("h264".to_string()));
}

#[test]
fn test_abr_ladder_creation() {
    let ladder = AbrLadder::hls_standard();
    assert!(ladder.rung_count() > 0);
    assert_eq!(ladder.strategy, AbrStrategy::AppleHls);
}

#[test]
fn test_quality_modes() {
    assert_eq!(QualityMode::Low.to_crf(), 28);
    assert_eq!(QualityMode::Medium.to_crf(), 23);
    assert_eq!(QualityMode::High.to_crf(), 20);
}

#[test]
fn test_progress_tracker() {
    let tracker = ProgressTracker::new(1000, 1);
    tracker.update_frame(500);
    let info = tracker.get_info();
    assert_eq!(info.current_frame, 500);
}

#[test]
fn test_multipass_config() {
    let config = MultiPassConfig::new(MultiPassMode::TwoPass, "/tmp/stats.log");
    assert_eq!(config.mode, MultiPassMode::TwoPass);
    assert!(config.mode.requires_stats());
}

#[test]
fn test_audio_normalization() {
    let normalizer = AudioNormalizer::with_standard(LoudnessStandard::EbuR128);
    assert_eq!(normalizer.target_lufs(), -23.0);
}

#[test]
fn test_validation() {
    use validation::*;

    // Valid formats
    assert!(InputValidator::validate_format("test.mp4").is_ok());
    assert!(InputValidator::validate_format("test.mkv").is_ok());

    // Invalid format
    assert!(InputValidator::validate_format("test.xyz").is_err());
}

#[test]
fn test_hw_accel_detection() {
    let available = detect_available_hw_accel();
    assert!(!available.is_empty());
    assert!(available.contains(&HwAccelType::None));
}

#[test]
fn test_codec_config() {
    let config = H264Config::new()
        .profile(H264Profile::High)
        .level("4.0")
        .refs(3)
        .bframes(3)
        .build();

    assert_eq!(config.codec, "h264");
    assert_eq!(config.profile, Some("high".to_string()));
}

#[test]
fn test_video_filters() {
    let filter = VideoFilter::new()
        .scale(1920, 1080)
        .deinterlace()
        .denoise(3.0);

    assert_eq!(filter.len(), 3);
    assert!(!filter.is_empty());
}

#[test]
fn test_audio_filters() {
    let filter = AudioFilter::new()
        .volume(1.5)
        .resample(48000)
        .normalize(-23.0);

    assert_eq!(filter.len(), 3);
}

#[test]
fn test_parallel_encoder() {
    let config = ParallelConfig::with_max_parallel(4);
    assert_eq!(config.max_parallel, 4);
    assert!(config.validate().is_ok());
}

#[test]
fn test_transcode_job() {
    let config = TranscodeJobConfig::new(TranscodeConfig::default());
    let job = TranscodeJob::new(config);

    assert_eq!(job.status, TranscodeStatus::Queued);
    assert_eq!(job.retry_count, 0);
}

#[test]
fn test_job_queue() {
    let mut queue = JobQueue::new(2);
    let config = TranscodeJobConfig::new(TranscodeConfig::default());
    queue.enqueue(TranscodeJob::new(config));

    assert_eq!(queue.len(), 1);
    assert!(!queue.is_empty());
}

#[test]
fn test_utils_format_duration() {
    use oximedia_transcode::utils::format_duration;
    assert_eq!(format_duration(90.0), "01:30");
    assert_eq!(format_duration(3665.0), "01:01:05");
}

#[test]
fn test_utils_format_file_size() {
    use oximedia_transcode::utils::format_file_size;
    assert_eq!(format_file_size(1024), "1.00 KB");
    assert_eq!(format_file_size(1024 * 1024), "1.00 MB");
}

#[test]
fn test_utils_aspect_ratio() {
    use oximedia_transcode::utils::calculate_aspect_ratio;
    assert_eq!(calculate_aspect_ratio(1920, 1080), (16, 9));
    assert_eq!(calculate_aspect_ratio(1280, 720), (16, 9));
}

#[test]
fn test_preset_categories() {
    // YouTube presets
    let yt_720p = presets::youtube::youtube_720p();
    assert_eq!(yt_720p.width, Some(1280));
    assert_eq!(yt_720p.height, Some(720));

    // Vimeo presets
    let vimeo_hd = presets::vimeo::vimeo_hd();
    assert_eq!(vimeo_hd.width, Some(1280));

    // Broadcast presets
    let prores = presets::broadcast::prores_proxy_hd();
    assert_eq!(prores.width, Some(1280));

    // Archive presets
    let archive = presets::archive::high_quality_vp9();
    assert!(archive.width.is_none()); // Preserves source
}

#[test]
fn test_streaming_presets() {
    let hls = presets::streaming::hls_ladder();
    assert!(hls.rung_count() > 0);

    let twitch = presets::streaming::twitch_1080p60();
    assert_eq!(twitch.frame_rate, Some((60, 1)));
}

#[test]
fn test_social_media_presets() {
    use presets::social;

    let instagram = social::instagram_feed();
    assert_eq!(instagram.width, Some(1080));
    assert_eq!(instagram.height, Some(1080));

    let tiktok = social::tiktok();
    assert_eq!(tiktok.width, Some(1080));
    assert_eq!(tiktok.height, Some(1920));
}

#[test]
fn test_rate_control_modes() {
    let crf = RateControlMode::Crf(23);
    assert!(crf.is_constant_quality());
    assert!(!crf.is_bitrate_mode());

    let cbr = RateControlMode::Cbr(5_000_000);
    assert!(!cbr.is_constant_quality());
    assert!(cbr.is_bitrate_mode());
}

#[test]
fn test_loudness_standards() {
    assert_eq!(LoudnessStandard::EbuR128.target_lufs(), -23.0);
    assert_eq!(LoudnessStandard::AtscA85.target_lufs(), -24.0);
    assert_eq!(LoudnessStandard::Spotify.target_lufs(), -14.0);
}

#[test]
fn test_quality_presets() {
    assert_eq!(QualityPreset::UltraFast.as_str(), "ultrafast");
    assert_eq!(QualityPreset::Medium.as_str(), "medium");
    assert_eq!(QualityPreset::VerySlow.as_str(), "veryslow");
}

#[test]
fn test_tune_modes() {
    assert_eq!(TuneMode::Film.as_str(), "film");
    assert_eq!(TuneMode::Animation.as_str(), "animation");
    assert_eq!(TuneMode::ZeroLatency.as_str(), "zerolatency");
}

#[test]
fn test_codec_suggestions() {
    use oximedia_transcode::utils::{suggest_audio_codec, suggest_video_codec};

    assert_eq!(suggest_video_codec("mp4"), Some("h264".to_string()));
    assert_eq!(suggest_video_codec("webm"), Some("vp9".to_string()));
    assert_eq!(suggest_audio_codec("mp4"), Some("aac".to_string()));
    assert_eq!(suggest_audio_codec("webm"), Some("opus".to_string()));
}

#[test]
fn test_resolution_utilities() {
    use oximedia_transcode::utils::{is_standard_resolution, resolution_name};

    assert!(is_standard_resolution(1920, 1080));
    assert!(is_standard_resolution(3840, 2160));
    assert!(!is_standard_resolution(1000, 1000));

    assert_eq!(resolution_name(1920, 1080), "Full HD (1080p)");
    assert_eq!(resolution_name(3840, 2160), "4K (2160p)");
}

#[test]
fn test_abr_ladder_filtering() {
    let ladder = AbrLadder::hls_standard().filter_by_source(1280, 720);
    let highest = ladder.highest_quality().expect("highest should be valid");
    assert_eq!(highest.height, 720);
}

#[test]
fn test_job_priority() {
    assert!(JobPriority::Critical > JobPriority::High);
    assert!(JobPriority::High > JobPriority::Normal);
    assert!(JobPriority::Normal > JobPriority::Low);
}

#[test]
fn test_hw_accel_config() {
    let config = HwAccelConfig::new(HwAccelType::Nvenc)
        .allow_fallback(false)
        .decode(true)
        .encode(true)
        .device_id(0);

    assert_eq!(config.preferred_type, HwAccelType::Nvenc);
    assert!(!config.allow_fallback);
    assert!(config.decode);
}

#[test]
fn test_opus_config() {
    let config = OpusConfig::new()
        .application(OpusApplication::Audio)
        .complexity(10)
        .vbr(true)
        .build();

    assert_eq!(config.codec, "opus");
}

#[test]
fn test_av1_config() {
    let config = Av1Config::new()
        .cpu_used(6)
        .usage(Av1Usage::Good)
        .row_mt(true)
        .build();

    assert_eq!(config.codec, "av1");
}

#[test]
fn test_vp9_config() {
    let config = Vp9Config::new().cpu_used(4).row_mt(true).build();

    assert_eq!(config.codec, "vp9");
}
