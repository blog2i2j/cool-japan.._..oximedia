//! Replay buffer example with instant replay.

use oximedia_gaming::{CaptureSource, EncoderPreset, GameStreamer, StreamConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiMedia Gaming - Replay Buffer Example ===\n");

    // Configure streaming with replay buffer
    let config = StreamConfig::builder()
        .source(CaptureSource::PrimaryMonitor)
        .resolution(1920, 1080)
        .framerate(60)
        .encoder_preset(EncoderPreset::HighQuality)
        .bitrate(10000)
        .replay_buffer(30) // 30 second replay buffer
        .build()?;

    println!("Configuration:");
    println!(
        "  Resolution: {}x{}",
        config.resolution.0, config.resolution.1
    );
    println!("  Framerate: {} fps", config.framerate);
    println!(
        "  Replay Buffer: {} seconds\n",
        config
            .replay_buffer_seconds
            .expect("replay buffer seconds should be set")
    );

    // Create streamer
    let mut streamer = GameStreamer::new(config).await?;

    // Enable replay buffer
    println!("Enabling replay buffer...");
    streamer.enable_replay_buffer(30)?;
    println!("Replay buffer enabled!\n");

    // Start streaming
    streamer.start().await?;
    println!("Stream started with replay buffer\n");

    // Simulate gameplay
    println!("Simulating gameplay...");
    for i in 1..=15 {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        println!("[{}s] Playing...", i);

        // Simulate "epic moment" at 10 seconds
        if i == 10 {
            println!("\n🎮 EPIC MOMENT! Saving replay...");
            streamer.save_replay("/tmp/epic_moment.mp4").await?;
            println!("✓ Replay saved to /tmp/epic_moment.mp4\n");
        }
    }

    // Stop streaming
    streamer.stop().await?;
    println!("\nStream stopped!");

    Ok(())
}
