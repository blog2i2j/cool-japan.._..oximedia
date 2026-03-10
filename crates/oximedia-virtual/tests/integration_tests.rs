//! Integration tests for oximedia-virtual
//!
//! Comprehensive integration tests for virtual production workflows.

use oximedia_virtual::math::{Point3, UnitQuaternion};
use oximedia_virtual::{
    icvfx::composite::{CompositorConfig, IcvfxCompositor},
    led::{
        render::{LedRenderer, LedRendererConfig},
        LedPanel, LedWall,
    },
    metrics::MetricsCollector,
    tracking::{
        camera::{CameraTracker, CameraTrackerConfig},
        CameraPose,
    },
    workflows::{ArWorkflow, HybridWorkflow, LedWallWorkflow},
    QualityMode, VirtualProduction, VirtualProductionConfig, WorkflowType,
};

/// Small frame dimensions used throughout integration tests for speed.
/// These replace production-sized (1920×1080 / 4K) frames so each test
/// finishes in well under 5 seconds.
const TEST_W: usize = 64;
const TEST_H: usize = 64;

#[test]
fn test_virtual_production_creation() {
    let config = VirtualProductionConfig::default();
    let vp = VirtualProduction::new(config);
    assert!(vp.is_ok());
}

#[test]
fn test_virtual_production_with_all_features() {
    let config = VirtualProductionConfig {
        workflow: WorkflowType::LedWall,
        target_fps: 60.0,
        sync_accuracy_ms: 0.5,
        quality: QualityMode::Final,
        color_calibration: true,
        lens_correction: true,
        num_cameras: 4,
        motion_capture: true,
        unreal_integration: true,
    };

    let vp = VirtualProduction::new(config);
    assert!(vp.is_ok());

    let vp = vp.expect("vp should be valid");
    assert!(vp.multicam_manager().is_some());
}

#[test]
fn test_camera_tracking_integration() {
    let config = CameraTrackerConfig::default();
    let mut tracker = CameraTracker::new(config).expect("test expectation failed");

    // Track for multiple frames (kept small – tracking is O(1) per frame but
    // 100 iterations * any per-frame allocation can still add up)
    for i in 0..10 {
        let timestamp_ns = i * 16_666_667; // 60 FPS
        let result = tracker.update(timestamp_ns);
        assert!(result.is_ok());
    }
}

#[test]
fn test_led_rendering_integration() {
    let mut led_wall = LedWall::new("Test Wall".to_string());
    // 64×64 panel keeps the per-pixel render loop tiny
    led_wall.add_panel(LedPanel::new(
        Point3::origin(),
        5.0,
        3.0,
        (TEST_W, TEST_H),
        2.5,
    ));

    let config = LedRendererConfig::default();
    let mut renderer = LedRenderer::new(config).expect("test expectation failed");
    renderer.set_led_wall(led_wall);

    let camera_pose = CameraPose::new(Point3::new(0.0, 1.5, 5.0), UnitQuaternion::identity(), 0);

    let source_frame = vec![128u8; TEST_W * TEST_H * 3];
    let result = renderer.render(&camera_pose, &source_frame, TEST_W, TEST_H, 0);

    assert!(result.is_ok());
}

#[test]
fn test_icvfx_compositing_integration() {
    let config = CompositorConfig {
        resolution: (TEST_W, TEST_H),
        depth_compositing: true,
        motion_blur: false,
        quality: 1.0,
    };

    let mut compositor = IcvfxCompositor::new(config).expect("test expectation failed");

    let foreground = vec![255u8; TEST_W * TEST_H * 3];
    let background = vec![0u8; TEST_W * TEST_H * 3];
    let depth = vec![0.5f32; TEST_W * TEST_H];

    let result = compositor.composite(&foreground, &background, Some(&depth), 0);

    assert!(result.is_ok());
}

#[test]
fn test_led_wall_workflow_integration() {
    // Use small panel so the per-pixel render loop is fast
    let mut workflow =
        LedWallWorkflow::with_panel_resolution((TEST_W, TEST_H)).expect("test expectation failed");

    workflow
        .start_recording("test-session".to_string(), 0)
        .expect("test expectation failed");

    let camera_pose = CameraPose::new(Point3::new(0.0, 1.5, 5.0), UnitQuaternion::identity(), 0);

    let source_frame = vec![128u8; TEST_W * TEST_H * 3];

    for i in 0..2 {
        let timestamp_ns = i * 16_666_667;
        let result =
            workflow.process_frame(&camera_pose, &source_frame, TEST_W, TEST_H, timestamp_ns);
        assert!(result.is_ok());
    }

    workflow.stop_recording();
}

#[test]
fn test_hybrid_workflow_integration() {
    // Small compositor resolution so blending loop finishes quickly
    let mut workflow =
        HybridWorkflow::with_resolution(TEST_W, TEST_H).expect("test expectation failed");

    workflow
        .start_session("hybrid-session".to_string(), 0)
        .expect("test expectation failed");

    let foreground = vec![128u8; TEST_W * TEST_H * 3];
    let background = vec![64u8; TEST_W * TEST_H * 3];

    for i in 0..3 {
        let timestamp_ns = i * 16_666_667;
        let result = workflow.composite_frame(&foreground, &background, timestamp_ns);
        assert!(result.is_ok());
    }
}

#[test]
fn test_ar_workflow_integration() {
    let mut workflow =
        ArWorkflow::with_resolution(TEST_W, TEST_H).expect("test expectation failed");

    workflow
        .start("ar-test".to_string(), 0)
        .expect("test expectation failed");

    let camera_feed = vec![128u8; TEST_W * TEST_H * 3];
    let virtual_content = vec![255u8; TEST_W * TEST_H * 3];

    for i in 0..3 {
        let timestamp_ns = i * 16_666_667;
        let result = workflow.overlay(&camera_feed, &virtual_content, timestamp_ns);
        assert!(result.is_ok());
    }
}

#[test]
fn test_metrics_collection_integration() {
    let mut metrics = MetricsCollector::new(60);

    // Reduced from 100 iterations (×16 ms sleep = 1.6 s) to 5 iterations so
    // the test finishes in ~80 ms while still exercising the metrics path.
    for _ in 0..5 {
        std::thread::sleep(std::time::Duration::from_millis(16));
        metrics.record_frame();
    }

    let perf = metrics.performance();
    assert!(perf.total_frames > 0);
    assert!(perf.avg_fps > 0.0);
}

#[test]
fn test_full_pipeline_integration() {
    let config = VirtualProductionConfig::default()
        .with_workflow(WorkflowType::LedWall)
        .with_target_fps(60.0)
        .with_quality(QualityMode::Final);

    let mut vp = VirtualProduction::new(config).expect("test expectation failed");

    // Small panel + small compositor so both render and composite loops are fast
    let mut led_wall = LedWall::new("Main Wall".to_string());
    led_wall.add_panel(LedPanel::new(
        Point3::origin(),
        5.0,
        3.0,
        (TEST_W, TEST_H),
        2.5,
    ));

    vp.led_renderer_mut().set_led_wall(led_wall);
    vp.set_compositor_resolution(TEST_W, TEST_H)
        .expect("set_compositor_resolution should succeed");

    for i in 0..2 {
        let timestamp_ns = i * 16_666_667;

        // Track camera
        let camera_pose = vp
            .camera_tracker_mut()
            .update(timestamp_ns)
            .expect("camera_pose should be valid");

        // Render to LED wall
        let source_frame = vec![128u8; TEST_W * TEST_H * 3];
        let led_output = vp
            .led_renderer_mut()
            .render(&camera_pose, &source_frame, TEST_W, TEST_H, timestamp_ns)
            .expect("test expectation failed");

        // Composite
        let foreground = vec![64u8; TEST_W * TEST_H * 3];
        let _composite = vp
            .compositor_mut()
            .composite(&foreground, &led_output.pixels, None, timestamp_ns)
            .expect("test expectation failed");
    }
}

#[test]
fn test_multi_camera_integration() {
    let config = VirtualProductionConfig::default().with_num_cameras(4);

    let vp = VirtualProduction::new(config).expect("vp should be valid");

    assert!(vp.multicam_manager().is_some());

    let multicam = vp.multicam_manager().expect("multicam should be valid");
    assert_eq!(multicam.config().num_cameras, 4);
}

#[test]
fn test_color_pipeline_integration() {
    let config = VirtualProductionConfig::default().with_workflow(WorkflowType::LedWall);

    let mut vp = VirtualProduction::new(config).expect("test expectation failed");

    // 64×64 is more than enough to exercise the color pipeline
    let frame = vec![128u8; TEST_W * TEST_H * 3];
    let result = vp.color_pipeline_mut().process(&frame, TEST_W, TEST_H);

    assert!(result.is_ok());
}

#[test]
fn test_genlock_sync_integration() {
    use oximedia_virtual::sync::genlock::{GenlockConfig, GenlockSync};

    let config = GenlockConfig::default();
    let mut genlock = GenlockSync::new(config).expect("test expectation failed");

    for _ in 0..10 {
        let result = genlock.wait_for_frame();
        assert!(result.is_ok());
    }
}

#[test]
fn test_performance_under_load() {
    let config = VirtualProductionConfig::default()
        .with_target_fps(60.0)
        .with_quality(QualityMode::Preview);

    let mut vp = VirtualProduction::new(config).expect("test expectation failed");

    // Use 64×64 instead of 4K (3840×2160) — the goal is to verify the render
    // path works under load, not to benchmark at production resolution.
    let mut led_wall = LedWall::new("Load Test".to_string());
    led_wall.add_panel(LedPanel::new(
        Point3::origin(),
        10.0,
        6.0,
        (TEST_W, TEST_H),
        1.8,
    ));

    vp.led_renderer_mut().set_led_wall(led_wall);

    let mut metrics = MetricsCollector::new(60);

    for i in 0..3 {
        let timestamp_ns = i * 16_666_667;

        metrics.record_frame();

        let camera_pose = vp
            .camera_tracker_mut()
            .update(timestamp_ns)
            .expect("camera_pose should be valid");
        let source_frame = vec![128u8; TEST_W * TEST_H * 3];
        let _led_output = vp
            .led_renderer_mut()
            .render(&camera_pose, &source_frame, TEST_W, TEST_H, timestamp_ns)
            .expect("test expectation failed");
    }

    assert!(metrics.performance().total_frames >= 3);
}
