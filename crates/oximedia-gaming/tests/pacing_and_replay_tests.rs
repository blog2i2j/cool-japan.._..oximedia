//! Frame pacing and replay buffer integration tests.

use oximedia_gaming::{
    pacing::{
        buffer::{BufferConfig, FrameBuffer},
        frame::{FramePacer, PacingMode},
    },
    replay::{
        buffer::{ReplayBuffer, ReplayConfig},
        save::{ReplaySaver, SaveFormat},
    },
};
use std::time::Duration;

// Frame Pacing Tests

#[test]
fn test_frame_pacer_creation() {
    let pacer = FramePacer::new(60, PacingMode::Fixed).expect("valid frame pacer");
    assert_eq!(pacer.frame_count(), 0);
}

#[test]
fn test_invalid_fps() {
    assert!(FramePacer::new(0, PacingMode::Fixed).is_err());
    assert!(FramePacer::new(300, PacingMode::Fixed).is_err());
}

#[test]
fn test_valid_fps_range() {
    let fps_values = [1, 24, 30, 60, 120, 144, 240];

    for fps in fps_values {
        assert!(FramePacer::new(fps, PacingMode::Fixed).is_ok());
    }
}

#[tokio::test]
async fn test_wait_for_next_frame() {
    let mut pacer = FramePacer::new(60, PacingMode::Fixed).expect("valid frame pacer");

    let timing = pacer
        .wait_for_next_frame()
        .await
        .expect("frame timing should succeed");
    assert_eq!(timing.frame_number, 1);
}

#[test]
fn test_frame_pacer_reset() {
    let mut pacer = FramePacer::new(60, PacingMode::Fixed).expect("valid frame pacer");
    pacer.frame_count = 100;

    pacer.reset();
    assert_eq!(pacer.frame_count(), 0);
}

#[test]
fn test_set_target_fps() {
    let mut pacer = FramePacer::new(60, PacingMode::Fixed).expect("valid frame pacer");

    pacer.set_target_fps(120).expect("set fps should succeed");
    assert!(pacer.set_target_fps(0).is_err());
    assert!(pacer.set_target_fps(300).is_err());
}

#[test]
fn test_all_pacing_modes() {
    let modes = [
        PacingMode::Fixed,
        PacingMode::Variable,
        PacingMode::Adaptive,
    ];

    for mode in modes {
        let pacer = FramePacer::new(60, mode).expect("valid frame pacer");
        assert_eq!(pacer.frame_count(), 0);
    }
}

#[test]
fn test_target_frame_time_calculation() {
    let pacer = FramePacer::new(60, PacingMode::Fixed).expect("valid frame pacer");
    let expected = Duration::from_secs_f64(1.0 / 60.0);

    assert!((pacer.target_frame_time().as_secs_f64() - expected.as_secs_f64()).abs() < 0.0001);
}

#[test]
fn test_high_framerate_pacer() {
    let pacer = FramePacer::new(144, PacingMode::Fixed).expect("valid frame pacer");
    let expected = Duration::from_secs_f64(1.0 / 144.0);

    assert!((pacer.target_frame_time().as_secs_f64() - expected.as_secs_f64()).abs() < 0.0001);
}

// Frame Buffer Tests

#[test]
fn test_frame_buffer_creation() {
    let config = BufferConfig::default();
    let buffer: FrameBuffer<u32> = FrameBuffer::new(config).expect("valid frame buffer");

    assert!(buffer.is_empty());
    assert_eq!(buffer.len(), 0);
}

#[test]
fn test_buffer_invalid_config() {
    let config = BufferConfig {
        min_size: 10,
        max_size: 5,
        target_size: 7,
    };

    let result: Result<FrameBuffer<u32>, _> = FrameBuffer::new(config);
    assert!(result.is_err());
}

#[test]
fn test_buffer_target_out_of_range() {
    let config = BufferConfig {
        min_size: 2,
        max_size: 10,
        target_size: 15,
    };

    let result: Result<FrameBuffer<u32>, _> = FrameBuffer::new(config);
    assert!(result.is_err());
}

#[test]
fn test_buffer_push_pop() {
    let config = BufferConfig::default();
    let mut buffer: FrameBuffer<u32> = FrameBuffer::new(config).expect("valid frame buffer");

    buffer.push(1).expect("push should succeed");
    buffer.push(2).expect("push should succeed");
    buffer.push(3).expect("push should succeed");

    assert_eq!(buffer.len(), 3);
    assert_eq!(buffer.pop(), Some(1));
    assert_eq!(buffer.pop(), Some(2));
    assert_eq!(buffer.pop(), Some(3));
    assert!(buffer.is_empty());
}

#[test]
fn test_buffer_full() {
    let config = BufferConfig {
        min_size: 1,
        max_size: 3,
        target_size: 2,
    };
    let mut buffer: FrameBuffer<u32> = FrameBuffer::new(config).expect("valid frame buffer");

    buffer.push(1).expect("push should succeed");
    buffer.push(2).expect("push should succeed");
    buffer.push(3).expect("push should succeed");

    assert!(buffer.is_full());
    assert!(buffer.push(4).is_err());
}

#[test]
fn test_buffer_underrun() {
    let config = BufferConfig {
        min_size: 3,
        max_size: 10,
        target_size: 5,
    };
    let mut buffer: FrameBuffer<u32> = FrameBuffer::new(config).expect("valid frame buffer");

    assert!(buffer.is_underrunning());

    buffer.push(1).expect("push should succeed");
    buffer.push(2).expect("push should succeed");
    assert!(buffer.is_underrunning());

    buffer.push(3).expect("push should succeed");
    assert!(!buffer.is_underrunning());
}

#[test]
fn test_buffer_utilization() {
    let config = BufferConfig {
        min_size: 1,
        max_size: 10,
        target_size: 5,
    };
    let mut buffer: FrameBuffer<u32> = FrameBuffer::new(config).expect("valid frame buffer");

    assert_eq!(buffer.utilization(), 0.0);

    for i in 0..5 {
        buffer.push(i).expect("push should succeed");
    }

    assert_eq!(buffer.utilization(), 0.5);

    for i in 5..10 {
        buffer.push(i).expect("push should succeed");
    }

    assert_eq!(buffer.utilization(), 1.0);
}

#[test]
fn test_buffer_peek() {
    let config = BufferConfig::default();
    let mut buffer: FrameBuffer<u32> = FrameBuffer::new(config).expect("valid frame buffer");

    buffer.push(42).expect("push should succeed");
    assert_eq!(buffer.peek(), Some(&42));
    assert_eq!(buffer.len(), 1);

    buffer.pop();
    assert_eq!(buffer.peek(), None);
}

#[test]
fn test_buffer_clear() {
    let config = BufferConfig::default();
    let mut buffer: FrameBuffer<u32> = FrameBuffer::new(config).expect("valid frame buffer");

    buffer.push(1).expect("push should succeed");
    buffer.push(2).expect("push should succeed");
    buffer.push(3).expect("push should succeed");

    buffer.clear();
    assert!(buffer.is_empty());
    assert_eq!(buffer.len(), 0);
}

// Replay Buffer Tests

#[test]
fn test_replay_buffer_creation() {
    let config = ReplayConfig::default();
    let buffer = ReplayBuffer::new(config).expect("valid replay buffer");

    assert!(!buffer.is_enabled());
}

#[test]
fn test_replay_buffer_invalid_duration() {
    let config = ReplayConfig {
        duration: 1,
        bitrate: 10000,
        audio_enabled: true,
        framerate: 60,
    };

    assert!(ReplayBuffer::new(config).is_err());

    let config = ReplayConfig {
        duration: 400,
        bitrate: 10000,
        audio_enabled: true,
        framerate: 60,
    };

    assert!(ReplayBuffer::new(config).is_err());
}

#[test]
fn test_replay_buffer_valid_durations() {
    let durations = [5, 10, 30, 60, 120, 180, 300];

    for duration in durations {
        let config = ReplayConfig {
            duration,
            bitrate: 10000,
            audio_enabled: true,
            framerate: 60,
        };

        assert!(ReplayBuffer::new(config).is_ok());
    }
}

#[test]
fn test_replay_buffer_enable_disable() {
    let config = ReplayConfig::default();
    let mut buffer = ReplayBuffer::new(config).expect("valid replay buffer");

    assert!(!buffer.is_enabled());

    buffer.enable().expect("enable should succeed");
    assert!(buffer.is_enabled());

    buffer.disable();
    assert!(!buffer.is_enabled());
}

#[test]
fn test_replay_buffer_duration() {
    let config = ReplayConfig {
        duration: 60,
        bitrate: 10000,
        audio_enabled: true,
        framerate: 60,
    };

    let buffer = ReplayBuffer::new(config).expect("valid replay buffer");
    assert_eq!(buffer.duration(), Duration::from_mins(1));
}

#[test]
fn test_replay_config_defaults() {
    let config = ReplayConfig::default();

    assert_eq!(config.duration, 30);
    assert_eq!(config.bitrate, 10000);
    assert!(config.audio_enabled);
}

// Replay Saver Tests

#[test]
fn test_replay_saver_creation() {
    let saver = ReplaySaver::new(SaveFormat::WebM);
    assert_eq!(saver.format(), SaveFormat::WebM);
}

#[test]
fn test_all_save_formats() {
    let formats = [SaveFormat::WebM, SaveFormat::Mkv, SaveFormat::Mp4];

    for format in formats {
        let saver = ReplaySaver::new(format);
        assert_eq!(saver.format(), format);
    }
}

#[tokio::test]
async fn test_save_replay() {
    let saver = ReplaySaver::default();
    let p = std::env::temp_dir().join("oximedia-gaming-pacing-replay.webm");
    saver
        .save(p.to_string_lossy().as_ref())
        .await
        .expect("save should succeed");
}

#[test]
fn test_replay_saver_defaults() {
    let saver = ReplaySaver::default();
    assert_eq!(saver.format(), SaveFormat::WebM);
}

#[tokio::test]
async fn test_save_with_different_formats() {
    let tmp = std::env::temp_dir();
    let webm = tmp.join("oximedia-gaming-pacing-test.webm");
    let mkv = tmp.join("oximedia-gaming-pacing-test.mkv");
    let mp4 = tmp.join("oximedia-gaming-pacing-test.mp4");
    let webm_s = webm.to_string_lossy().into_owned();
    let mkv_s = mkv.to_string_lossy().into_owned();
    let mp4_s = mp4.to_string_lossy().into_owned();
    let formats = [
        (SaveFormat::WebM, webm_s.as_str()),
        (SaveFormat::Mkv, mkv_s.as_str()),
        (SaveFormat::Mp4, mp4_s.as_str()),
    ];

    for (format, path) in formats {
        let saver = ReplaySaver::new(format);
        saver.save(path).await.expect("save should succeed");
    }
}

// Integration Tests

#[tokio::test]
async fn test_pacing_with_buffer() {
    let mut pacer = FramePacer::new(60, PacingMode::Fixed).expect("valid frame pacer");
    let buffer_config = BufferConfig::default();
    let mut buffer: FrameBuffer<u64> = FrameBuffer::new(buffer_config).expect("valid frame buffer");

    for _ in 0..5 {
        let timing = pacer
            .wait_for_next_frame()
            .await
            .expect("frame timing should succeed");
        buffer
            .push(timing.frame_number)
            .expect("push should succeed");
    }

    assert_eq!(buffer.len(), 5);
    assert_eq!(pacer.frame_count(), 5);
}

#[test]
fn test_replay_buffer_with_high_bitrate() {
    let config = ReplayConfig {
        duration: 60,
        bitrate: 50000, // 50 Mbps for high quality
        audio_enabled: true,
        framerate: 60,
    };

    let buffer = ReplayBuffer::new(config).expect("valid replay buffer");
    assert_eq!(buffer.duration(), Duration::from_mins(1));
}

#[test]
fn test_buffer_config_defaults() {
    let config = BufferConfig::default();

    assert_eq!(config.min_size, 2);
    assert_eq!(config.max_size, 10);
    assert_eq!(config.target_size, 5);
}
