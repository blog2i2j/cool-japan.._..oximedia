//! Capture module integration tests.

use oximedia_gaming::capture::{
    cursor::CursorCapture,
    game::{GameCapture, GameProfile},
    screen::{CaptureConfig, CaptureRegion, ScreenCapture},
};

#[test]
fn test_screen_capture_all_regions() {
    let regions = [
        CaptureRegion::PrimaryMonitor,
        CaptureRegion::Monitor(0),
        CaptureRegion::Monitor(1),
        CaptureRegion::Window(12345),
        CaptureRegion::Region {
            x: 0,
            y: 0,
            width: 1920,
            height: 1080,
        },
    ];

    for region in regions {
        let config = CaptureConfig {
            region,
            framerate: 60,
            capture_cursor: true,
        };

        let capture = ScreenCapture::new(config).expect("valid screen capture");
        assert!(!capture.is_capturing());
    }
}

#[test]
fn test_screen_capture_framerate_limits() {
    // Test minimum
    let config = CaptureConfig {
        region: CaptureRegion::PrimaryMonitor,
        framerate: 1,
        capture_cursor: true,
    };
    assert!(ScreenCapture::new(config).is_ok());

    // Test maximum
    let config = CaptureConfig {
        region: CaptureRegion::PrimaryMonitor,
        framerate: 240,
        capture_cursor: true,
    };
    assert!(ScreenCapture::new(config).is_ok());

    // Test invalid (too high)
    let config = CaptureConfig {
        region: CaptureRegion::PrimaryMonitor,
        framerate: 300,
        capture_cursor: true,
    };
    assert!(ScreenCapture::new(config).is_err());

    // Test invalid (zero)
    let config = CaptureConfig {
        region: CaptureRegion::PrimaryMonitor,
        framerate: 0,
        capture_cursor: true,
    };
    assert!(ScreenCapture::new(config).is_err());
}

#[test]
fn test_screen_capture_lifecycle() {
    let config = CaptureConfig::default();
    let mut capture = ScreenCapture::new(config).expect("valid screen capture");

    // Initial state
    assert!(!capture.is_capturing());

    // Start
    capture.start().expect("start should succeed");
    assert!(capture.is_capturing());

    // Pause
    capture.pause().expect("pause should succeed");
    assert!(!capture.is_capturing());

    // Resume
    capture.resume().expect("resume should succeed");
    assert!(capture.is_capturing());

    // Stop
    capture.stop();
    assert!(!capture.is_capturing());
}

#[test]
fn test_screen_capture_double_start() {
    let config = CaptureConfig::default();
    let mut capture = ScreenCapture::new(config).expect("valid screen capture");

    capture.start().expect("start should succeed");
    assert!(capture.start().is_err());

    capture.stop();
}

#[test]
fn test_screen_capture_pause_when_not_running() {
    let config = CaptureConfig::default();
    let mut capture = ScreenCapture::new(config).expect("valid screen capture");

    assert!(capture.pause().is_err());
}

#[test]
fn test_screen_capture_resume_when_not_paused() {
    let config = CaptureConfig::default();
    let mut capture = ScreenCapture::new(config).expect("valid screen capture");

    assert!(capture.resume().is_err());
}

#[test]
fn test_capture_frame() {
    let config = CaptureConfig::default();
    let mut capture = ScreenCapture::new(config).expect("valid screen capture");

    // Should fail when not capturing
    assert!(capture.capture_frame().is_err());

    // Should succeed when capturing
    capture.start().expect("start should succeed");
    let frame = capture
        .capture_frame()
        .expect("capture frame should succeed");
    assert!(frame.width > 0);
    assert!(frame.height > 0);
    assert!(!frame.data.is_empty());

    capture.stop();
}

#[test]
fn test_list_monitors() {
    let monitors = ScreenCapture::list_monitors().expect("list monitors should succeed");
    assert!(!monitors.is_empty());

    for monitor in &monitors {
        assert!(monitor.resolution.0 > 0);
        assert!(monitor.resolution.1 > 0);
        assert!(monitor.refresh_rate > 0);
    }
}

#[test]
fn test_all_game_profiles() {
    let profiles = [
        GameProfile::Fps,
        GameProfile::Moba,
        GameProfile::Fighting,
        GameProfile::Racing,
        GameProfile::Strategy,
        GameProfile::Rpg,
        GameProfile::Platformer,
        GameProfile::Generic,
    ];

    for profile in profiles {
        let capture = GameCapture::new(profile);
        let settings = capture.recommended_settings();

        assert!(settings.target_latency_ms > 0);
        assert!(settings.max_framerate > 0);
        assert!(settings.max_framerate <= 240);
    }
}

#[test]
fn test_fps_game_profile_latency() {
    let capture = GameCapture::new(GameProfile::Fps);
    let settings = capture.recommended_settings();

    // FPS games should have very low latency
    assert!(settings.target_latency_ms <= 50);
    assert!(settings.max_framerate >= 60);
}

#[test]
fn test_fighting_game_profile_latency() {
    let capture = GameCapture::new(GameProfile::Fighting);
    let settings = capture.recommended_settings();

    // Fighting games need frame-perfect timing
    assert!(settings.target_latency_ms <= 20);
}

#[test]
fn test_strategy_game_profile_quality() {
    let capture = GameCapture::new(GameProfile::Strategy);
    let settings = capture.recommended_settings();

    // Strategy games can tolerate higher latency for better quality
    assert!(settings.target_latency_ms >= 80);
    use oximedia_gaming::capture::game::CapturePriority;
    assert_eq!(settings.priority, CapturePriority::Quality);
}

#[test]
fn test_game_capture_attach_detach() {
    let mut capture = GameCapture::new(GameProfile::Generic);

    assert!(!capture.is_active());

    capture.attach(12345).expect("attach should succeed");
    assert!(capture.is_active());

    capture.detach();
    assert!(!capture.is_active());
}

#[test]
fn test_cursor_capture() {
    let mut capture = CursorCapture::new();

    assert!(capture.is_enabled());

    capture.disable();
    assert!(!capture.is_enabled());

    capture.enable();
    assert!(capture.is_enabled());
}

#[test]
fn test_cursor_position_tracking() {
    let mut capture = CursorCapture::new();

    capture.update_position(100, 200);
    let info = capture
        .get_cursor_info()
        .expect("cursor info should succeed");
    assert_eq!(info.position, (100, 200));

    capture.update_position(500, 300);
    let info = capture
        .get_cursor_info()
        .expect("cursor info should succeed");
    assert_eq!(info.position, (500, 300));
}

#[test]
fn test_cursor_visibility() {
    let capture = CursorCapture::new();
    let info = capture
        .get_cursor_info()
        .expect("cursor info should succeed");

    assert!(info.visible);
}

#[test]
fn test_custom_region_capture() {
    let region = CaptureRegion::Region {
        x: 100,
        y: 100,
        width: 1280,
        height: 720,
    };

    let config = CaptureConfig {
        region,
        framerate: 60,
        capture_cursor: false,
    };

    let mut capture = ScreenCapture::new(config).expect("valid screen capture");
    capture.start().expect("start should succeed");

    let frame = capture
        .capture_frame()
        .expect("capture frame should succeed");
    assert_eq!(frame.width, 1280);
    assert_eq!(frame.height, 720);

    capture.stop();
}

#[test]
fn test_high_framerate_capture() {
    let config = CaptureConfig {
        region: CaptureRegion::PrimaryMonitor,
        framerate: 144,
        capture_cursor: true,
    };

    let capture = ScreenCapture::new(config).expect("valid screen capture");
    assert_eq!(capture.config().framerate, 144);
}

#[test]
fn test_capture_config_defaults() {
    let config = CaptureConfig::default();

    assert_eq!(config.region, CaptureRegion::PrimaryMonitor);
    assert_eq!(config.framerate, 60);
    assert!(config.capture_cursor);
}
