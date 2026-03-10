use super::generate::SpriteGenerationResult;
use super::*;

pub(super) async fn generate_webvtt(
    output_path: &Path,
    thumbnails: &[ThumbnailMetadata],
    options: &SpriteSheetOptions,
) -> Result<()> {
    info!("Generating WebVTT file: {}", output_path.display());

    let mut vtt_content = String::from("WEBVTT\n\n");

    let sprite_filename = options
        .output
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("sprite.png");

    for (i, thumb) in thumbnails.iter().enumerate() {
        let start_time = thumb.timestamp;

        // Calculate end time (halfway to next thumbnail or video end)
        let end_time = if i + 1 < thumbnails.len() {
            (thumb.timestamp + thumbnails[i + 1].timestamp) / 2.0
        } else {
            thumb.timestamp + 5.0 // Add 5 seconds for last thumbnail
        };

        // Format timestamps
        let start = format_vtt_timestamp(start_time);
        let end = format_vtt_timestamp(end_time);

        // Write cue
        vtt_content.push_str(&format!("Cue {}\n", i + 1));
        vtt_content.push_str(&format!("{} --> {}\n", start, end));
        vtt_content.push_str(&format!(
            "{}#xywh={},{},{},{}\n\n",
            sprite_filename, thumb.x, thumb.y, thumb.width, thumb.height
        ));
    }

    tokio::fs::write(output_path, vtt_content)
        .await
        .context("Failed to write WebVTT file")?;

    info!("WebVTT file generated successfully");

    Ok(())
}

/// Format timestamp for WebVTT (HH:MM:SS.mmm).
pub(super) fn format_vtt_timestamp(seconds: f64) -> String {
    let hours = (seconds / 3600.0) as u32;
    let minutes = ((seconds % 3600.0) / 60.0) as u32;
    let secs = (seconds % 60.0) as u32;
    let millis = ((seconds % 1.0) * 1000.0) as u32;

    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, secs, millis)
}

/// Generate JSON manifest file.
pub(super) async fn generate_manifest(
    output_path: &Path,
    result: &SpriteGenerationResult,
    options: &SpriteSheetOptions,
) -> Result<()> {
    info!("Generating JSON manifest: {}", output_path.display());

    // Simulated video metadata (would be read from demuxer in production).
    let video_duration = 120.0;
    let video_fps = 30.0;

    let manifest = SpriteSheetManifest {
        sprite_file: options.output.display().to_string(),
        video_file: options.input.display().to_string(),
        sprite_width: result.sprite_width,
        sprite_height: result.sprite_height,
        thumbnails: result.thumbnails.clone(),
        config: options.config.clone(),
        video_duration,
        video_fps,
        generated_at: chrono::Utc::now().to_rfc3339(),
    };

    let json = serde_json::to_string_pretty(&manifest)?;

    tokio::fs::write(output_path, json)
        .await
        .context("Failed to write JSON manifest")?;

    info!("JSON manifest generated successfully");

    Ok(())
}

/// Create JSON result for output.
pub(super) fn create_json_result(
    result: &SpriteGenerationResult,
    options: &SpriteSheetOptions,
    processing_time: f64,
) -> Result<SpriteSheetResult> {
    Ok(SpriteSheetResult {
        success: true,
        sprite_path: options.output.display().to_string(),
        sprite_width: result.sprite_width,
        sprite_height: result.sprite_height,
        thumbnail_count: result.thumbnails.len(),
        thumbnail_width: options.config.thumbnail_width,
        thumbnail_height: options.config.thumbnail_height,
        columns: options.config.columns,
        rows: options.config.rows,
        format: options.config.format.name().to_string(),
        vtt_path: options.vtt_output.as_ref().map(|p| p.display().to_string()),
        manifest_path: options
            .manifest_output
            .as_ref()
            .map(|p| p.display().to_string()),
        processing_time,
    })
}

/// Print sprite sheet generation summary.
pub(super) fn print_generation_summary(
    result: &SpriteGenerationResult,
    options: &SpriteSheetOptions,
    processing_time: f64,
) {
    println!();
    println!("{}", "Sprite Sheet Generation Complete".green().bold());
    println!("{}", "=".repeat(70));

    println!("{:25} {}", "Output File:", options.output.display());
    println!(
        "{:25} {}x{}",
        "Sprite Dimensions:", result.sprite_width, result.sprite_height
    );
    println!("{:25} {}", "Thumbnails Generated:", result.thumbnails.len());
    println!(
        "{:25} {}x{}",
        "Thumbnail Size:", options.config.thumbnail_width, options.config.thumbnail_height
    );
    println!(
        "{:25} {}x{}",
        "Grid Layout:", options.config.columns, options.config.rows
    );
    println!("{:25} {}", "Format:", options.config.format);
    println!("{:25} {:.2}s", "Processing Time:", processing_time);

    if options.generate_vtt {
        let vtt_path = options
            .vtt_output
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| {
                let mut p = options.output.clone();
                p.set_extension("vtt");
                p.display().to_string()
            });
        println!("{:25} {}", "WebVTT File:", vtt_path);
    }

    if options.generate_manifest {
        let manifest_path = options
            .manifest_output
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| {
                let mut p = options.output.clone();
                p.set_extension("json");
                p.display().to_string()
            });
        println!("{:25} {}", "Manifest File:", manifest_path);
    }

    println!();

    // Show timestamp coverage
    if let (Some(first), Some(last)) = (result.thumbnails.first(), result.thumbnails.last()) {
        println!("{}", "Timestamp Coverage:".cyan());

        println!(
            "{:25} {}",
            "  First Thumbnail:",
            format_timestamp(first.timestamp)
        );
        println!(
            "{:25} {}",
            "  Last Thumbnail:",
            format_timestamp(last.timestamp)
        );

        if result.thumbnails.len() > 2 {
            let avg_interval = if result.thumbnails.len() > 1 {
                (last.timestamp - first.timestamp) / (result.thumbnails.len() - 1) as f64
            } else {
                0.0
            };
            println!("{:25} {:.2}s", "  Average Interval:", avg_interval);
        }
    }

    println!("{}", "=".repeat(70));
}

/// Format timestamp as human-readable string.
fn format_timestamp(seconds: f64) -> String {
    let hours = (seconds / 3600.0) as u32;
    let minutes = ((seconds % 3600.0) / 60.0) as u32;
    let secs = seconds % 60.0;

    if hours > 0 {
        format!("{:02}:{:02}:{:05.2}", hours, minutes, secs)
    } else {
        format!("{:02}:{:05.2}", minutes, secs)
    }
}
