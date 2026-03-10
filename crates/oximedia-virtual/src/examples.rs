//! Example workflows and usage patterns for virtual production
//!
//! Provides comprehensive examples for common virtual production scenarios.

#![allow(dead_code)]

use crate::math::{Point3, UnitQuaternion};
use crate::{
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
    QualityMode, Result, VirtualProduction, VirtualProductionConfig, WorkflowType,
};

/// Example: Basic LED wall setup
pub fn example_basic_led_wall() -> Result<()> {
    // Use small resolution in tests for speed; production-sized in normal builds
    #[cfg(test)]
    let (frame_w, frame_h, panel_res) = (64usize, 64usize, (64usize, 64usize));
    #[cfg(not(test))]
    let (frame_w, frame_h, panel_res) = (1920usize, 1080usize, (1920usize, 1080usize));

    // Create LED wall configuration
    let mut led_wall = LedWall::new("Main Wall".to_string());

    // Add LED panels
    led_wall.add_panel(LedPanel::new(
        Point3::new(0.0, 0.0, 0.0),
        5.0,       // 5 meters wide
        3.0,       // 3 meters tall
        panel_res, // Resolution
        2.5,       // Pixel pitch in mm
    ));

    // Create LED renderer
    let config = LedRendererConfig::default();
    let mut renderer = LedRenderer::new(config)?;
    renderer.set_led_wall(led_wall);

    // Create camera tracker
    let tracker_config = CameraTrackerConfig {
        max_latency_ms: f64::MAX / 2.0, // No timeout for example
        ..CameraTrackerConfig::default()
    };
    let mut tracker = CameraTracker::new(tracker_config)?;

    // Example frame rendering loop (fewer iterations in test mode)
    #[cfg(test)]
    let num_frames = 2u64;
    #[cfg(not(test))]
    let num_frames = 3u64;

    for frame_num in 0..num_frames {
        let timestamp_ns = frame_num * 16_666_667; // 60 FPS

        // Update tracking
        let camera_pose = tracker.update(timestamp_ns)?;

        // Render to LED wall
        let source_frame = vec![0u8; frame_w * frame_h * 3]; // Placeholder source
        let _led_output =
            renderer.render(&camera_pose, &source_frame, frame_w, frame_h, timestamp_ns)?;
    }

    Ok(())
}

/// Example: In-camera VFX workflow
pub fn example_icvfx_workflow() -> Result<()> {
    // Use small resolution and fewer frames in tests for speed
    #[cfg(test)]
    let (w, h, num_frames) = (64usize, 64usize, 2u64);
    #[cfg(not(test))]
    let (w, h, num_frames) = (3840usize, 2160usize, 100u64);

    // Create compositor
    let config = CompositorConfig {
        resolution: (w, h),
        depth_compositing: true,
        motion_blur: false,
        quality: 1.0,
    };

    let mut compositor = IcvfxCompositor::new(config)?;

    // Example compositing loop
    for frame_num in 0..num_frames {
        let timestamp_ns = frame_num * 16_666_667;

        // Prepare foreground and background
        let foreground = vec![0u8; w * h * 3];
        let background = vec![255u8; w * h * 3];

        // Optional depth map
        let depth = vec![0.5f32; w * h];

        // Composite
        let _result = compositor.composite(&foreground, &background, Some(&depth), timestamp_ns)?;
    }

    Ok(())
}

/// Example: Multi-camera virtual production
pub fn example_multi_camera_setup() -> Result<()> {
    // Create multi-camera configuration
    let config = VirtualProductionConfig::default()
        .with_num_cameras(4)
        .with_target_fps(60.0)
        .with_quality(QualityMode::Final);

    let vp = VirtualProduction::new(config)?;

    // Get multi-camera manager
    if let Some(multicam) = vp.multicam_manager() {
        println!("Managing {} cameras", multicam.config().num_cameras);
    }

    Ok(())
}

/// Example: LED wall workflow with metrics
pub fn example_led_wall_with_metrics() -> Result<()> {
    // Use small frame size in tests for speed
    #[cfg(test)]
    let (frame_w, frame_h) = (64usize, 64usize);
    #[cfg(not(test))]
    let (frame_w, frame_h) = (1920usize, 1080usize);

    // Create workflow with matching panel resolution
    let mut workflow = LedWallWorkflow::with_panel_resolution((frame_w, frame_h))?;

    // Create metrics collector
    let mut metrics = MetricsCollector::new(60);

    // Start recording session
    workflow.start_recording("session-001".to_string(), 0)?;

    // Processing loop
    for frame_num in 0..3 {
        let timestamp_ns = frame_num * 16_666_667;

        // Record frame timing
        metrics.record_frame();

        // Create camera pose
        let camera_pose = CameraPose::new(
            Point3::new(0.0, 1.5, 5.0),
            UnitQuaternion::identity(),
            timestamp_ns,
        );

        // Process frame
        let source_frame = vec![128u8; frame_w * frame_h * 3];
        let _output =
            workflow.process_frame(&camera_pose, &source_frame, frame_w, frame_h, timestamp_ns)?;

        // Update quality metrics
        if frame_num % 60 == 0 {
            let quality = crate::metrics::QualityMetrics {
                tracking_confidence: 0.95,
                color_accuracy: 0.98,
                sync_accuracy_us: 500,
                brightness_uniformity: 0.97,
            };
            metrics.update_quality(quality);

            // Print metrics report every second
            println!("\n{}", metrics.generate_report());
        }
    }

    // Stop recording
    workflow.stop_recording();

    Ok(())
}

/// Example: Hybrid workflow (LED + green screen)
pub fn example_hybrid_workflow() -> Result<()> {
    // Use small resolution and fewer frames in tests for speed
    #[cfg(test)]
    let (w, h, num_frames) = (64usize, 64usize, 2u64);
    #[cfg(not(test))]
    let (w, h, num_frames) = (1920usize, 1080usize, 100u64);

    let mut workflow = HybridWorkflow::with_resolution(w, h)?;

    // Start session
    workflow.start_session("hybrid-001".to_string(), 0)?;

    // Process frames
    for frame_num in 0..num_frames {
        let timestamp_ns = frame_num * 16_666_667;

        let foreground = vec![0u8; w * h * 3];
        let background = vec![255u8; w * h * 3];

        let _composite = workflow.composite_frame(&foreground, &background, timestamp_ns)?;
    }

    Ok(())
}

/// Example: AR overlay workflow
pub fn example_ar_workflow() -> Result<()> {
    // Use small resolution and fewer frames in tests for speed
    #[cfg(test)]
    let (w, h, num_frames) = (64usize, 64usize, 2u64);
    #[cfg(not(test))]
    let (w, h, num_frames) = (1920usize, 1080usize, 100u64);

    let mut workflow = ArWorkflow::with_resolution(w, h)?;

    // Start AR session
    workflow.start("ar-session-001".to_string(), 0)?;

    // Process AR overlays
    for frame_num in 0..num_frames {
        let timestamp_ns = frame_num * 16_666_667;

        let camera_feed = vec![128u8; w * h * 3];
        let virtual_content = vec![255u8; w * h * 3];

        let _result = workflow.overlay(&camera_feed, &virtual_content, timestamp_ns)?;
    }

    Ok(())
}

/// Example: Real-time camera tracking
pub fn example_camera_tracking() -> Result<()> {
    let config = CameraTrackerConfig {
        update_rate: 120.0, // 120 Hz tracking
        optical_tracking: true,
        imu_tracking: true,
        fusion_weight: 0.7,
        smoothing_window: 5,
        max_latency_ms: 5.0,
    };

    let mut tracker = CameraTracker::new(config)?;

    // Tracking loop
    for frame_num in 0..1000 {
        let timestamp_ns = frame_num * 8_333_333; // 120 Hz

        // Update tracking
        let pose = tracker.update(timestamp_ns)?;

        // Check tracking quality
        if pose.confidence < 0.5 {
            println!("Warning: Low tracking confidence: {}", pose.confidence);
        }

        // Log position every 100 frames
        if frame_num % 100 == 0 {
            println!(
                "Frame {}: Position = ({:.2}, {:.2}, {:.2}), Confidence = {:.2}",
                frame_num, pose.position.x, pose.position.y, pose.position.z, pose.confidence
            );
        }
    }

    Ok(())
}

/// Example: Complete virtual production pipeline
pub fn example_complete_pipeline() -> Result<()> {
    // Use small resolution in tests for speed; production 4K in normal builds
    #[cfg(test)]
    let (w, h, panel_res) = (64usize, 64usize, (64usize, 64usize));
    #[cfg(not(test))]
    let (w, h, panel_res) = (3840usize, 2160usize, (3840usize, 2160usize));

    // Create full virtual production system
    let config = VirtualProductionConfig {
        workflow: WorkflowType::LedWall,
        target_fps: 60.0,
        sync_accuracy_ms: 0.5,
        quality: QualityMode::Final,
        color_calibration: true,
        lens_correction: true,
        num_cameras: 1,
        motion_capture: false,
        unreal_integration: false,
    };

    let mut vp = VirtualProduction::new(config)?;
    vp.set_compositor_resolution(w, h)?;

    // Setup LED wall
    let mut led_wall = LedWall::new("Main Volume".to_string());
    led_wall.add_panel(LedPanel::new(
        Point3::new(0.0, 0.0, 0.0),
        10.0,
        6.0,
        panel_res,
        1.8,
    ));

    vp.led_renderer_mut().set_led_wall(led_wall);

    // Initialize metrics
    let mut metrics = MetricsCollector::new(60);

    // Production loop
    println!("Starting virtual production...");

    for frame_num in 0..2 {
        // 1 minute at 60 FPS
        let timestamp_ns = frame_num * 16_666_667;

        // Track camera
        let start_tracking = std::time::Instant::now();
        let camera_pose = vp.camera_tracker_mut().update(timestamp_ns)?;
        metrics.record_tracking_latency(start_tracking.elapsed());

        // Render to LED wall
        let start_render = std::time::Instant::now();
        let source_frame = vec![128u8; w * h * 3];
        let led_output =
            vp.led_renderer_mut()
                .render(&camera_pose, &source_frame, w, h, timestamp_ns)?;
        metrics.record_render_latency(start_render.elapsed());

        // Composite if needed
        let start_composite = std::time::Instant::now();
        let foreground = vec![0u8; w * h * 3];
        let _composite =
            vp.compositor_mut()
                .composite(&foreground, &led_output.pixels, None, timestamp_ns)?;
        metrics.record_composite_latency(start_composite.elapsed());

        // Update metrics
        metrics.record_frame();
        metrics.update_total_latency();

        // Report every second
        if frame_num % 60 == 0 {
            println!("\nFrame {frame_num}/3600");
            println!("{}", metrics.generate_report());
        }
    }

    println!("\nProduction complete!");
    println!("Final metrics:");
    println!("{}", metrics.generate_report());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_basic_led_wall() {
        let result = example_basic_led_wall();
        assert!(result.is_ok());
    }

    #[test]
    fn test_example_icvfx_workflow() {
        let result = example_icvfx_workflow();
        assert!(result.is_ok());
    }

    #[test]
    fn test_example_multi_camera_setup() {
        let result = example_multi_camera_setup();
        assert!(result.is_ok());
    }

    #[test]
    fn test_example_hybrid_workflow() {
        let result = example_hybrid_workflow();
        assert!(result.is_ok());
    }

    #[test]
    fn test_example_ar_workflow() {
        let result = example_ar_workflow();
        assert!(result.is_ok());
    }

    #[test]
    fn test_example_camera_tracking() {
        let result = example_camera_tracking();
        assert!(result.is_ok());
    }
}
