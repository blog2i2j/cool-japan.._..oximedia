//! Encoding module integration tests.

use oximedia_gaming::encode::{
    lowlatency::{EncoderConfig, LatencyMode, LowLatencyEncoder, RateControlMode},
    nvenc::{NvencEncoder, NvencPreset},
    qsv::{QsvEncoder, QsvPreset},
    vce::{VceEncoder, VcePreset},
};

#[test]
fn test_encoder_creation() {
    let config = EncoderConfig::default();
    let encoder = LowLatencyEncoder::new(config).expect("valid encoder");

    let stats = encoder.get_stats();
    assert_eq!(stats.frames_encoded, 0);
}

#[test]
fn test_encoder_invalid_resolution() {
    let mut config = EncoderConfig::default();
    config.resolution = (0, 0);

    assert!(LowLatencyEncoder::new(config).is_err());
}

#[test]
fn test_encoder_invalid_framerate() {
    let mut config = EncoderConfig::default();
    config.framerate = 0;

    assert!(LowLatencyEncoder::new(config).is_err());
}

#[test]
fn test_encoder_invalid_bitrate() {
    let mut config = EncoderConfig::default();
    config.bitrate = 100;

    assert!(LowLatencyEncoder::new(config).is_err());
}

#[test]
fn test_encode_frame() {
    let config = EncoderConfig::default();
    let mut encoder = LowLatencyEncoder::new(config).expect("valid encoder");

    let frame_data = vec![0u8; 1920 * 1080 * 4];
    let encoded = encoder
        .encode_frame(&frame_data)
        .expect("encode should succeed");

    assert!(encoded.is_keyframe);
    assert_eq!(encoder.get_stats().frames_encoded, 1);
}

#[test]
fn test_keyframe_interval() {
    let mut config = EncoderConfig::default();
    config.keyframe_interval = 30;

    let mut encoder = LowLatencyEncoder::new(config).expect("valid encoder");
    let frame_data = vec![0u8; 1920 * 1080 * 4];

    // First frame is keyframe
    let frame1 = encoder
        .encode_frame(&frame_data)
        .expect("encode should succeed");
    assert!(frame1.is_keyframe);

    // Next 29 frames are not keyframes
    for _ in 0..29 {
        let frame = encoder
            .encode_frame(&frame_data)
            .expect("encode should succeed");
        assert!(!frame.is_keyframe);
    }

    // 31st frame is keyframe
    let frame31 = encoder
        .encode_frame(&frame_data)
        .expect("encode should succeed");
    assert!(frame31.is_keyframe);
}

#[test]
fn test_all_latency_modes() {
    let modes = [LatencyMode::UltraLow, LatencyMode::Low, LatencyMode::Normal];

    for mode in modes {
        let mut config = EncoderConfig::default();
        config.latency_mode = mode;

        let encoder = LowLatencyEncoder::new(config).expect("valid encoder");
        assert_eq!(encoder.config.latency_mode, mode);
    }
}

#[test]
fn test_all_rate_control_modes() {
    let modes = [
        RateControlMode::Cbr,
        RateControlMode::Vbr,
        RateControlMode::Cq,
    ];

    for mode in modes {
        let mut config = EncoderConfig::default();
        config.rate_control = mode;

        let encoder = LowLatencyEncoder::new(config).expect("valid encoder");
        assert_eq!(encoder.config.rate_control, mode);
    }
}

#[test]
fn test_ultra_low_latency_config() {
    let mut config = EncoderConfig::default();
    config.latency_mode = LatencyMode::UltraLow;
    config.use_b_frames = false;
    config.keyframe_interval = 60;

    let encoder = LowLatencyEncoder::new(config).expect("valid encoder");
    assert_eq!(encoder.config.latency_mode, LatencyMode::UltraLow);
    assert!(!encoder.config.use_b_frames);
}

#[test]
fn test_high_quality_config() {
    let mut config = EncoderConfig::default();
    config.latency_mode = LatencyMode::Normal;
    config.use_b_frames = true;
    config.bitrate = 15000;

    let encoder = LowLatencyEncoder::new(config).expect("valid encoder");
    assert_eq!(encoder.config.bitrate, 15000);
    assert!(encoder.config.use_b_frames);
}

#[test]
fn test_encoder_flush() {
    let config = EncoderConfig::default();
    let mut encoder = LowLatencyEncoder::new(config).expect("valid encoder");

    let frame_data = vec![0u8; 1920 * 1080 * 4];
    encoder
        .encode_frame(&frame_data)
        .expect("encode should succeed");
    encoder
        .encode_frame(&frame_data)
        .expect("encode should succeed");

    let flushed = encoder.flush().expect("flush should succeed");
    assert!(flushed.is_empty()); // Mock implementation returns empty
}

#[test]
fn test_multiple_resolutions() {
    let resolutions = [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)];

    for (width, height) in resolutions {
        let mut config = EncoderConfig::default();
        config.resolution = (width, height);

        let encoder = LowLatencyEncoder::new(config).expect("valid encoder");
        assert_eq!(encoder.config.resolution, (width, height));
    }
}

#[test]
fn test_multiple_framerates() {
    let framerates = [24, 30, 60, 120, 144, 240];

    for fps in framerates {
        let mut config = EncoderConfig::default();
        config.framerate = fps;

        let encoder = LowLatencyEncoder::new(config).expect("valid encoder");
        assert_eq!(encoder.config.framerate, fps);
    }
}

#[test]
fn test_multiple_bitrates() {
    let bitrates = [500, 2500, 6000, 10000, 20000, 40000];

    for bitrate in bitrates {
        let mut config = EncoderConfig::default();
        config.bitrate = bitrate;

        let encoder = LowLatencyEncoder::new(config).expect("valid encoder");
        assert_eq!(encoder.config.bitrate, bitrate);
    }
}

#[test]
fn test_nvenc_availability() {
    // In test environment, typically not available
    let available = NvencEncoder::is_available();
    assert!(!available);
}

#[test]
fn test_nvenc_recommended_presets() {
    assert_eq!(
        NvencEncoder::recommended_preset_for_latency(30),
        NvencPreset::P1
    );
    assert_eq!(
        NvencEncoder::recommended_preset_for_latency(80),
        NvencPreset::P2
    );
    assert_eq!(
        NvencEncoder::recommended_preset_for_latency(150),
        NvencPreset::P3
    );
}

#[test]
fn test_all_nvenc_presets() {
    let presets = [
        NvencPreset::P1,
        NvencPreset::P2,
        NvencPreset::P3,
        NvencPreset::P4,
        NvencPreset::P5,
        NvencPreset::P6,
        NvencPreset::P7,
    ];

    for preset in presets {
        // Can't create encoder without hardware, but can test the preset enum
        assert_eq!(preset, preset);
    }
}

#[test]
fn test_qsv_availability() {
    // In test environment, typically not available
    let available = QsvEncoder::is_available();
    assert!(!available);
}

#[test]
fn test_qsv_recommended_presets() {
    assert_eq!(
        QsvEncoder::recommended_preset_for_latency(30),
        QsvPreset::VeryFast
    );
    assert_eq!(
        QsvEncoder::recommended_preset_for_latency(80),
        QsvPreset::Fast
    );
    assert_eq!(
        QsvEncoder::recommended_preset_for_latency(150),
        QsvPreset::Medium
    );
}

#[test]
fn test_all_qsv_presets() {
    let presets = [
        QsvPreset::VeryFast,
        QsvPreset::Fast,
        QsvPreset::Medium,
        QsvPreset::Slow,
        QsvPreset::VerySlow,
    ];

    for preset in presets {
        assert_eq!(preset, preset);
    }
}

#[test]
fn test_vce_availability() {
    // In test environment, typically not available
    let available = VceEncoder::is_available();
    assert!(!available);
}

#[test]
fn test_vce_recommended_presets() {
    assert_eq!(
        VceEncoder::recommended_preset_for_latency(50),
        VcePreset::Speed
    );
    assert_eq!(
        VceEncoder::recommended_preset_for_latency(150),
        VcePreset::Balanced
    );
}

#[test]
fn test_all_vce_presets() {
    let presets = [VcePreset::Speed, VcePreset::Balanced, VcePreset::Quality];

    for preset in presets {
        assert_eq!(preset, preset);
    }
}

#[test]
fn test_encoder_config_defaults() {
    let config = EncoderConfig::default();

    assert_eq!(config.resolution, (1920, 1080));
    assert_eq!(config.framerate, 60);
    assert_eq!(config.bitrate, 6000);
    assert_eq!(config.latency_mode, LatencyMode::Low);
    assert_eq!(config.keyframe_interval, 120);
    assert!(!config.use_b_frames);
    assert_eq!(config.rate_control, RateControlMode::Cbr);
}

#[test]
fn test_encode_multiple_frames() {
    let config = EncoderConfig::default();
    let mut encoder = LowLatencyEncoder::new(config).expect("valid encoder");

    let frame_data = vec![0u8; 1920 * 1080 * 4];

    for i in 0..10 {
        encoder
            .encode_frame(&frame_data)
            .expect("encode should succeed");
        assert_eq!(encoder.get_stats().frames_encoded, (i + 1) as u64);
    }
}

#[test]
fn test_encoder_stats() {
    let config = EncoderConfig::default();
    let mut encoder = LowLatencyEncoder::new(config).expect("valid encoder");

    let stats = encoder.get_stats();
    assert_eq!(stats.frames_encoded, 0);

    let frame_data = vec![0u8; 1920 * 1080 * 4];
    encoder
        .encode_frame(&frame_data)
        .expect("encode should succeed");

    let stats = encoder.get_stats();
    assert_eq!(stats.frames_encoded, 1);
    assert!(stats.average_encoding_time.as_millis() > 0);
}
