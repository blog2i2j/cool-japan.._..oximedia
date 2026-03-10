//! Basic playout server example
//!
//! This example demonstrates how to create and configure a broadcast playout server
//! with scheduling, graphics, and monitoring capabilities.

use chrono::Utc;
use oximedia_playout::{
    graphics::{GraphicsConfig, GraphicsEngine, GraphicsLayer, Position},
    monitoring::{Alert, AlertSeverity, AlertType, Monitor, MonitorConfig, OnAirStatus},
    output::{OutputConfig, OutputManager, OutputSettings, OutputType, RTMPSettings},
    playback::{PlaybackConfig, PlaybackEngine},
    playlist::{Playlist, PlaylistItem, PlaylistManager},
    scheduler::{CuePoint, CueType, ScheduledEvent, Scheduler, SchedulerConfig, Transition},
    AudioFormat, PlayoutConfig, PlayoutServer, VideoFormat,
};
use std::path::PathBuf;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("=== OxiMedia Playout Server Example ===\n");

    // 1. Create server configuration
    println!("1. Configuring playout server...");
    let mut config = PlayoutConfig::default();
    config.video_format = VideoFormat::HD1080p25;
    config.audio_format = AudioFormat {
        sample_rate: 48000,
        channels: 2,
        bit_depth: 24,
    };
    config.genlock_enabled = false;
    config.clock_source = "internal".to_string();
    config.buffer_size = 10;
    config.max_latency_ms = 100;
    config.monitoring_enabled = true;
    config.monitoring_port = 8080;
    println!("   ✓ Video: {:?}", config.video_format);
    println!(
        "   ✓ Audio: {}Hz, {} channels, {} bit",
        config.audio_format.sample_rate,
        config.audio_format.channels,
        config.audio_format.bit_depth
    );

    // 2. Create playout server
    println!("\n2. Creating playout server...");
    let server = PlayoutServer::new(config.clone()).await?;
    println!("   ✓ Server created");

    // 3. Setup scheduler
    println!("\n3. Setting up scheduler...");
    let scheduler_config = SchedulerConfig {
        auto_schedule: true,
        lookahead_hours: 24,
        scte35_enabled: true,
        default_fill: Some(PathBuf::from("/var/oximedia/fill.mxf")),
        macro_expansion: true,
        frame_tolerance: 1,
    };
    let scheduler = Scheduler::new(scheduler_config);
    println!("   ✓ Scheduler configured");

    // Add some scheduled events
    let event1 = ScheduledEvent::new_content(
        Utc::now() + chrono::Duration::seconds(60),
        PathBuf::from("/var/oximedia/content/show1.mxf"),
        Some(75000), // 50 minutes at 25fps
        Transition::FadeFromBlack {
            duration_frames: 25,
        },
        Transition::FadeToBlack {
            duration_frames: 25,
        },
    );
    scheduler.add_event(event1);
    println!("   ✓ Added scheduled event");

    // 4. Setup playlist
    println!("\n4. Creating playlist...");
    let playlist_manager = PlaylistManager::new();
    let mut playlist = Playlist::new("Daily Broadcast".to_string());

    // Add items to playlist
    let mut item1 = PlaylistItem::new(
        "Morning News".to_string(),
        PathBuf::from("/var/oximedia/content/news_morning.mxf"),
    );
    item1.duration_frames = Some(45000); // 30 minutes
    item1.cue_points.push(CuePoint {
        id: "ad_break_1".to_string(),
        name: "First Ad Break".to_string(),
        frame_offset: 15000, // 10 minutes in
        cue_type: CueType::AdBreak,
        data: std::collections::HashMap::new(),
    });
    playlist.add_item(item1);

    let item2 = PlaylistItem::new(
        "Feature Film".to_string(),
        PathBuf::from("/var/oximedia/content/film.mxf"),
    );
    playlist.add_item(item2);

    let playlist_id = playlist_manager.add_playlist(playlist);
    playlist_manager.set_active(playlist_id)?;
    println!(
        "   ✓ Playlist created with {} items",
        playlist_manager
            .get_active()
            .expect("get_active should succeed")
            .items
            .len()
    );

    // 5. Setup playback engine
    println!("\n5. Initializing playback engine...");
    let playback_config = PlaybackConfig::from_playout_config(&config);
    let playback_engine = PlaybackEngine::new(playback_config)?;
    println!("   ✓ Playback engine ready");
    println!("   ✓ Frame rate: {:.2} fps", config.video_format.fps());
    println!("   ✓ Buffer size: {} frames", config.buffer_size);

    // 6. Setup outputs
    println!("\n6. Configuring outputs...");
    let output_manager = OutputManager::new();

    // Add RTMP output (for streaming)
    let rtmp_config = OutputConfig {
        output_type: OutputType::RTMP,
        name: "YouTube Stream".to_string(),
        video_format: config.video_format.clone(),
        audio_format: config.audio_format.clone(),
        settings: OutputSettings::RTMP(RTMPSettings {
            url: "rtmp://a.rtmp.youtube.com/live2".to_string(),
            stream_key: "your-stream-key".to_string(),
            video_bitrate_kbps: 5000,
            audio_bitrate_kbps: 128,
            video_codec: "h264".to_string(),
            audio_codec: "aac".to_string(),
            keyframe_interval: 50,
            low_latency: false,
            timeout_seconds: 30,
        }),
        enabled: true,
        priority: 100,
    };
    output_manager.add_output(rtmp_config);
    println!("   ✓ Added RTMP output");

    // 7. Setup graphics
    println!("\n7. Setting up graphics overlay...");
    let graphics_config = GraphicsConfig::default();
    let graphics_engine = GraphicsEngine::new(graphics_config)?;

    // Add station logo
    let logo = GraphicsLayer::logo(
        "Station Logo".to_string(),
        PathBuf::from("/var/oximedia/graphics/logo.png"),
        Position::top_right(),
    );
    let _logo_id = graphics_engine.add_layer(logo)?;
    println!("   ✓ Added station logo");

    // Add lower third
    let lower_third = GraphicsLayer::lower_third(
        "Lower Third".to_string(),
        "John Smith".to_string(),
        "News Reporter".to_string(),
        Position::new(0.1, 0.85),
        "Arial".to_string(),
        48,
    );
    let lt_id = graphics_engine.add_layer(lower_third)?;
    graphics_engine.hide_layer(lt_id)?; // Start hidden
    println!("   ✓ Added lower third (hidden)");

    // Add ticker
    let ticker = GraphicsLayer::ticker(
        "News Ticker".to_string(),
        "Breaking news: OxiMedia playout server now available!".to_string(),
        Position::new(0.0, 0.95),
        "Arial".to_string(),
        32,
        0.01,
    );
    graphics_engine.add_layer(ticker)?;
    println!("   ✓ Added news ticker");

    // 8. Setup monitoring
    println!("\n8. Initializing monitoring...");
    let monitor_config = MonitorConfig {
        port: 8080,
        audio_meters: true,
        waveform: false,
        vectorscope: false,
        alert_history_size: 100,
        metrics_retention_seconds: 3600,
    };
    let monitor = Monitor::new(monitor_config)?;

    // Update on-air status
    let on_air_status = OnAirStatus {
        on_air: true,
        current_program: Some("Daily Broadcast".to_string()),
        current_item: Some("Morning News".to_string()),
        time_on_air: 0,
        timecode: "00:00:00:00".to_string(),
        frame_number: 0,
    };
    monitor.update_on_air(on_air_status);
    println!("   ✓ Monitoring active on port {}", 8080);

    // Raise a test alert
    let alert = Alert::new(
        AlertSeverity::Info,
        AlertType::Custom("System startup".to_string()),
        "Playout server started successfully".to_string(),
    );
    monitor.raise_alert(alert);
    println!("   ✓ Alert system ready");

    // 9. Display system status
    println!("\n9. System Status:");
    println!("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Server State: {:?}", server.state().await);
    println!("   Playback State: {:?}", playback_engine.get_state());
    println!(
        "   Buffer Level: {}/{}",
        playback_engine.buffer_level(),
        config.buffer_size
    );
    println!("   Scheduled Events: {}", scheduler.event_count());
    println!("   Active Alerts: {}", monitor.get_active_alerts().len());
    let visible_layers = graphics_engine.get_visible_layers();
    println!("   Graphics Layers: {} visible", visible_layers.len());
    println!("   Outputs: {}", output_manager.list_outputs().len());
    println!("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // 10. Start the server
    println!("\n10. Starting playout server...");
    server.start().await?;
    println!("    ✓ Server started");

    // Simulate some operation
    println!("\n11. Simulating playout operations...");
    for i in 0..5 {
        tokio::time::sleep(Duration::from_secs(1)).await;

        // Update graphics animations
        graphics_engine.update_animations();

        // Show lower third after 2 seconds
        if i == 2 {
            graphics_engine.show_layer(lt_id)?;
            println!("    → Lower third displayed");
        }

        // Hide lower third after 4 seconds
        if i == 4 {
            graphics_engine.hide_layer(lt_id)?;
            println!("    → Lower third hidden");
        }

        println!("    ⏱  Frame: {}", playback_engine.current_frame());
    }

    // 12. Display final statistics
    println!("\n12. Statistics:");
    let stats = playback_engine.get_stats();
    println!("   Frames played: {}", stats.frames_played);
    println!("   Frames dropped: {}", stats.frames_dropped);
    println!("   Drop rate: {:.2}%", stats.drop_rate());
    println!("   Uptime: {} seconds", stats.uptime_seconds);

    // 13. Export monitoring data
    println!("\n13. Exporting monitoring data...");
    let json_data = monitor.export_metrics()?;
    println!("   ✓ Exported {} bytes of monitoring data", json_data.len());

    // 14. Cleanup
    println!("\n14. Shutting down...");
    server.stop().await?;
    println!("    ✓ Server stopped");

    println!("\n=== Playout server example completed successfully ===\n");

    Ok(())
}
