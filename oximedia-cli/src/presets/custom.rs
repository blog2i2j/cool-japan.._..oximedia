//! Custom preset loading and saving.
//!
//! Handles loading presets from TOML files and saving user-created presets.

use super::{validate, AudioConfig, FilterConfig, Preset, PresetCategory, VideoConfig};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// TOML preset file format.
#[derive(Debug, Serialize, Deserialize)]
struct PresetFile {
    preset: PresetMetadata,
    video: VideoConfig,
    audio: AudioConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    filters: Option<FilterConfig>,
}

/// Preset metadata section.
#[derive(Debug, Serialize, Deserialize)]
struct PresetMetadata {
    name: String,
    description: String,
    #[serde(default = "default_category")]
    category: String,
    container: String,
    #[serde(default)]
    tags: Vec<String>,
}

fn default_category() -> String {
    "custom".to_string()
}

/// Load a preset from a TOML file.
pub fn load_preset_from_file<P: AsRef<Path>>(path: P) -> Result<Preset> {
    let content = fs::read_to_string(path.as_ref()).context(format!(
        "Failed to read preset file: {}",
        path.as_ref().display()
    ))?;

    let preset_file: PresetFile =
        toml::from_str(&content).context("Failed to parse TOML preset file")?;

    // Convert to internal Preset structure
    let category =
        PresetCategory::from_str(&preset_file.preset.category).unwrap_or(PresetCategory::Custom);

    let preset = Preset {
        name: preset_file.preset.name,
        description: preset_file.preset.description,
        category,
        video: preset_file.video,
        audio: preset_file.audio,
        container: preset_file.preset.container,
        filters: preset_file.filters,
        builtin: false,
        tags: preset_file.preset.tags,
    };

    // Validate preset
    validate::validate_preset(&preset).context("Preset validation failed")?;

    Ok(preset)
}

/// Save a preset to a TOML file.
pub fn save_preset_to_file<P: AsRef<Path>>(preset: &Preset, dir: P) -> Result<()> {
    let path = dir.as_ref().join(format!("{}.toml", preset.name));

    let preset_file = PresetFile {
        preset: PresetMetadata {
            name: preset.name.clone(),
            description: preset.description.clone(),
            category: preset.category.name().to_lowercase(),
            container: preset.container.clone(),
            tags: preset.tags.clone(),
        },
        video: preset.video.clone(),
        audio: preset.audio.clone(),
        filters: preset.filters.clone(),
    };

    let toml_string =
        toml::to_string_pretty(&preset_file).context("Failed to serialize preset to TOML")?;

    fs::write(&path, toml_string)
        .context(format!("Failed to write preset file: {}", path.display()))?;

    Ok(())
}

/// Load all presets from a directory.
#[allow(dead_code)]
pub fn load_presets_from_dir<P: AsRef<Path>>(dir: P) -> Result<Vec<Preset>> {
    let mut presets = Vec::new();

    if !dir.as_ref().exists() {
        return Ok(presets);
    }

    for entry in fs::read_dir(dir.as_ref())? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("toml") {
            match load_preset_from_file(&path) {
                Ok(preset) => presets.push(preset),
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to load preset from {}: {}",
                        path.display(),
                        e
                    );
                }
            }
        }
    }

    Ok(presets)
}

/// Create a new custom preset interactively.
pub fn create_preset_interactive() -> Result<Preset> {
    use std::io::{self, Write};

    println!("Creating a new custom preset...");
    println!();

    // Get name
    print!("Preset name (alphanumeric, hyphens, underscores): ");
    io::stdout().flush()?;
    let mut name = String::new();
    io::stdin().read_line(&mut name)?;
    let name = name.trim().to_string();

    // Get description
    print!("Description: ");
    io::stdout().flush()?;
    let mut description = String::new();
    io::stdin().read_line(&mut description)?;
    let description = description.trim().to_string();

    // Get category
    println!("Category (web, device, quality, archival, streaming, custom): ");
    print!("> ");
    io::stdout().flush()?;
    let mut category_str = String::new();
    io::stdin().read_line(&mut category_str)?;
    let category = PresetCategory::from_str(category_str.trim()).unwrap_or(PresetCategory::Custom);

    // Get video codec
    print!("Video codec (av1, vp9, vp8): ");
    io::stdout().flush()?;
    let mut video_codec = String::new();
    io::stdin().read_line(&mut video_codec)?;
    let video_codec = video_codec.trim().to_string();

    // Get video bitrate or CRF
    print!("Video bitrate (e.g., '5M') or press Enter for CRF mode: ");
    io::stdout().flush()?;
    let mut video_bitrate_str = String::new();
    io::stdin().read_line(&mut video_bitrate_str)?;
    let video_bitrate_str = video_bitrate_str.trim();

    let (video_bitrate, crf) = if video_bitrate_str.is_empty() {
        print!("CRF value (0-63 for VP9/VP8, 0-255 for AV1): ");
        io::stdout().flush()?;
        let mut crf_str = String::new();
        io::stdin().read_line(&mut crf_str)?;
        let crf: u32 = crf_str.trim().parse().unwrap_or(31);
        (None, Some(crf))
    } else {
        (Some(video_bitrate_str.to_string()), None)
    };

    // Get resolution
    print!("Video width (pixels, or press Enter to skip): ");
    io::stdout().flush()?;
    let mut width_str = String::new();
    io::stdin().read_line(&mut width_str)?;
    let width = if width_str.trim().is_empty() {
        None
    } else {
        Some(width_str.trim().parse().unwrap_or(1920))
    };

    print!("Video height (pixels, or press Enter to skip): ");
    io::stdout().flush()?;
    let mut height_str = String::new();
    io::stdin().read_line(&mut height_str)?;
    let height = if height_str.trim().is_empty() {
        None
    } else {
        Some(height_str.trim().parse().unwrap_or(1080))
    };

    // Get frame rate
    print!("Frame rate (e.g., '30', or press Enter to skip): ");
    io::stdout().flush()?;
    let mut fps_str = String::new();
    io::stdin().read_line(&mut fps_str)?;
    let fps = if fps_str.trim().is_empty() {
        None
    } else {
        Some(fps_str.trim().parse().unwrap_or(30.0))
    };

    // Get audio codec
    print!("Audio codec (opus, vorbis, flac, pcm, aac, mp3): ");
    io::stdout().flush()?;
    let mut audio_codec = String::new();
    io::stdin().read_line(&mut audio_codec)?;
    let audio_codec = audio_codec.trim().to_string();

    // Get audio bitrate
    print!("Audio bitrate (e.g., '128k'): ");
    io::stdout().flush()?;
    let mut audio_bitrate = String::new();
    io::stdin().read_line(&mut audio_bitrate)?;
    let audio_bitrate = Some(audio_bitrate.trim().to_string());

    // Get container
    print!("Container format (webm, mkv, ogg): ");
    io::stdout().flush()?;
    let mut container = String::new();
    io::stdin().read_line(&mut container)?;
    let container = container.trim().to_string();

    let preset = Preset {
        name,
        description,
        category,
        video: VideoConfig {
            codec: video_codec,
            bitrate: video_bitrate,
            crf,
            width,
            height,
            fps,
            preset: Some("medium".to_string()),
            pixel_format: Some("yuv420p".to_string()),
            two_pass: false,
            max_bitrate: None,
            min_bitrate: None,
            buffer_size: None,
            keyframe_interval: Some(240),
            min_keyframe_interval: Some(24),
            aspect_ratio: None,
        },
        audio: AudioConfig {
            codec: audio_codec,
            bitrate: audio_bitrate,
            sample_rate: Some(48000),
            channels: Some(2),
            quality: None,
            compression_level: None,
        },
        container,
        filters: None,
        builtin: false,
        tags: vec![],
    };

    // Validate preset
    validate::validate_preset(&preset)?;

    Ok(preset)
}

/// Generate a preset template TOML file.
pub fn generate_template<P: AsRef<Path>>(path: P) -> Result<()> {
    let template = r#"[preset]
name = "my-custom-preset"
description = "My custom transcoding preset"
category = "custom"  # web, device, quality, archival, streaming, custom
container = "webm"   # webm, mkv, ogg, flac, wav
tags = ["custom", "example"]

[video]
codec = "vp9"        # av1, vp9, vp8, theora
# Use either bitrate OR crf (not both)
# bitrate = "5M"     # Target bitrate (e.g., "5M", "2.5M", "500k")
crf = 31             # Constant Rate Factor (0-63 for VP9/VP8, 0-255 for AV1)
width = 1920
height = 1080
fps = 30.0
preset = "medium"    # ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
pixel_format = "yuv420p"
two_pass = true
# max_bitrate = "7M"
# min_bitrate = "3M"
# buffer_size = "10M"
keyframe_interval = 240
min_keyframe_interval = 24
aspect_ratio = "16:9"

[audio]
codec = "opus"       # opus, vorbis, flac, pcm, aac, mp3
bitrate = "128k"     # Audio bitrate (e.g., "128k", "192k", "256k")
sample_rate = 48000  # 8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000
channels = 2         # 1 (mono), 2 (stereo), 6 (5.1), 8 (7.1)
# quality = 5.0      # Codec-specific quality (0-10)
# compression_level = 8  # For FLAC (0-12)

# Optional filters
# [filters]
# video_filters = ["scale=1920:1080", "fps=30"]
# audio_filters = ["volume=0.5"]
# deinterlace = "yadif"
# denoise = "hqdn3d"
"#;

    fs::write(path.as_ref(), template).context(format!(
        "Failed to write template to {}",
        path.as_ref().display()
    ))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_load_preset_from_file() {
        let temp_dir = TempDir::new().expect("TempDir::new should succeed");
        let preset_path = temp_dir.path().join("test-preset.toml");

        let toml_content = r#"
[preset]
name = "test-preset"
description = "Test preset"
category = "custom"
container = "webm"
tags = ["test"]

[video]
codec = "vp9"
crf = 31
width = 1920
height = 1080
fps = 30.0
preset = "medium"
pixel_format = "yuv420p"
two_pass = true
keyframe_interval = 240
min_keyframe_interval = 24
aspect_ratio = "16:9"

[audio]
codec = "opus"
bitrate = "128k"
sample_rate = 48000
channels = 2
"#;

        fs::write(&preset_path, toml_content).expect("fs::write should succeed");

        let preset = load_preset_from_file(&preset_path).expect("load should succeed");
        assert_eq!(preset.name, "test-preset");
        assert_eq!(preset.video.codec, "vp9");
        assert_eq!(preset.audio.codec, "opus");
    }

    #[test]
    fn test_save_preset_to_file() {
        let temp_dir = TempDir::new().expect("TempDir::new should succeed");

        let preset = Preset {
            name: "save-test".to_string(),
            description: "Save test preset".to_string(),
            category: PresetCategory::Custom,
            video: VideoConfig {
                codec: "vp9".to_string(),
                bitrate: None,
                crf: Some(31),
                width: Some(1920),
                height: Some(1080),
                fps: Some(30.0),
                preset: Some("medium".to_string()),
                pixel_format: Some("yuv420p".to_string()),
                two_pass: true,
                max_bitrate: None,
                min_bitrate: None,
                buffer_size: None,
                keyframe_interval: Some(240),
                min_keyframe_interval: Some(24),
                aspect_ratio: Some("16:9".to_string()),
            },
            audio: AudioConfig {
                codec: "opus".to_string(),
                bitrate: Some("128k".to_string()),
                sample_rate: Some(48000),
                channels: Some(2),
                quality: None,
                compression_level: None,
            },
            container: "webm".to_string(),
            filters: None,
            builtin: false,
            tags: vec!["test".to_string()],
        };

        save_preset_to_file(&preset, temp_dir.path()).expect("save should succeed");

        let saved_path = temp_dir.path().join("save-test.toml");
        assert!(saved_path.exists());

        let loaded = load_preset_from_file(&saved_path).expect("load should succeed");
        assert_eq!(loaded.name, preset.name);
    }
}
