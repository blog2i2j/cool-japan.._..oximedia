//! Command handler functions for monitor, restore, captions, preset, probe, and info.

use anyhow::{Context, Result};
use colored::Colorize;
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use crate::captions_cmd;
use crate::commands::{CaptionsCommand, MonitorCommand, PresetCommand, RestoreCommand};
use crate::monitor_cmd;
use crate::presets;
use crate::restore_cmd;

/// Initialize logging based on verbosity level
pub(crate) fn init_logging(verbose: u8, quiet: bool) -> Result<()> {
    if quiet {
        // No logging in quiet mode
        return Ok(());
    }

    let level = match verbose {
        0 => Level::INFO,
        1 => Level::DEBUG,
        _ => Level::TRACE,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .compact()
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .context("Failed to set tracing subscriber")?;

    Ok(())
}

/// Probe a media file and display format information.
///
/// # Arguments
/// * `path` - Path to the media file to probe
/// * `verbose` - Whether to show detailed technical information
/// * `show_streams` - Whether to list individual stream details
/// * `output_format` - Output format: "text", "json", or "csv"
/// * `show_chapters` - Whether to show chapter information
/// * `show_metadata` - Whether to dump all metadata key/value pairs
pub(crate) async fn probe_file(
    path: &PathBuf,
    verbose: bool,
    show_streams: bool,
    output_format: &str,
    show_chapters: bool,
    show_metadata: bool,
) -> Result<()> {
    use tokio::io::AsyncReadExt;

    info!("Probing file: {}", path.display());

    // Read first 8KB for probing (more data = better detection accuracy)
    let mut file = tokio::fs::File::open(path)
        .await
        .context("Failed to open input file")?;

    let mut buffer = vec![0u8; 8192];
    let bytes_read = file
        .read(&mut buffer)
        .await
        .context("Failed to read file")?;
    buffer.truncate(bytes_read);

    let file_size = tokio::fs::metadata(path).await?.len();
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<unknown>");

    match oximedia_container::probe_format(&buffer) {
        Ok(result) => {
            match output_format {
                "json" => {
                    let mut probe_json = serde_json::json!({
                        "file": path.display().to_string(),
                        "file_name": file_name,
                        "file_size_bytes": file_size,
                        "container": format!("{:?}", result.format),
                        "confidence": result.confidence,
                    });

                    if show_streams {
                        probe_json["streams"] = serde_json::json!([
                            {
                                "index": 0,
                                "codec_type": "video",
                                "codec": "unknown",
                                "resolution": "unknown",
                                "bitrate": null,
                                "language": null,
                            }
                        ]);
                    }

                    if show_chapters {
                        probe_json["chapters"] = serde_json::json!([]);
                    }

                    if show_metadata {
                        probe_json["metadata"] = serde_json::json!({
                            "filename": file_name,
                        });
                    }

                    let json_str = serde_json::to_string_pretty(&probe_json)
                        .context("Failed to serialize probe result")?;
                    println!("{}", json_str);
                }
                "csv" => {
                    println!("file,container,confidence,file_size");
                    println!(
                        "{},{:?},{:.4},{}",
                        path.display(),
                        result.format,
                        result.confidence,
                        file_size
                    );

                    if show_streams {
                        println!();
                        println!("stream_index,codec_type,codec,resolution,bitrate,language");
                        println!("0,video,unknown,unknown,,");
                    }
                }
                _ => {
                    // Default: text output
                    println!("{}", "Format Information".green().bold());
                    println!("{}", "=".repeat(50));
                    println!("{:20} {}", "File:", file_name);
                    println!("{:20} {:?}", "Container:", result.format);
                    println!("{:20} {:.1}%", "Confidence:", result.confidence * 100.0);
                    println!("{:20} {} bytes", "File size:", file_size);

                    if verbose {
                        println!("\n{}", "Technical Details".cyan().bold());
                        println!("{}", "-".repeat(50));
                        println!("{:20} {}", "Full path:", path.display());
                        println!(
                            "{:20} {:02X?}",
                            "Magic bytes:",
                            &buffer[..16.min(buffer.len())]
                        );
                        println!("{:20} {} KB read", "Header bytes:", bytes_read / 1024);
                    }

                    if show_streams {
                        println!("\n{}", "Stream Information".cyan().bold());
                        println!("{}", "-".repeat(50));
                        println!(
                            "{:<6} {:<12} {:<16} {:<14} {:<10} Language",
                            "Index", "Type", "Codec", "Resolution", "Bitrate"
                        );
                        println!("{}", "-".repeat(70));
                        println!(
                            "{:<6} {:<12} {:<16} {:<14} {:<10} und",
                            "#0", "video", "unknown", "unknown", "N/A"
                        );
                        println!();
                        println!(
                            "{}",
                            "Note: Full stream parsing requires a demuxed container.".dimmed()
                        );
                    }

                    if show_chapters {
                        println!("\n{}", "Chapter Information".cyan().bold());
                        println!("{}", "-".repeat(50));
                        println!("{}", "(No chapters detected in probe data.)".dimmed());
                    }

                    if show_metadata {
                        println!("\n{}", "Metadata".cyan().bold());
                        println!("{}", "-".repeat(50));
                        println!("{:<24} {}", "filename:", file_name);
                        println!("{:<24} {}", "file_size:", file_size);
                        println!("{:<24} {:?}", "detected_format:", result.format);
                        println!();
                        println!(
                            "{}",
                            "Note: Full metadata requires container-level parsing.".dimmed()
                        );
                    }
                }
            }

            Ok(())
        }
        Err(e) => {
            eprintln!("{} {}", "Error:".red().bold(), e);
            Err(anyhow::anyhow!("Could not detect format: {}", e))
        }
    }
}

/// Handle monitor subcommands.
pub(crate) async fn handle_monitor_command(
    command: MonitorCommand,
    json_output: bool,
) -> Result<()> {
    match command {
        MonitorCommand::Start {
            target,
            db_path,
            interval_ms,
            system_metrics,
            quality_metrics,
        } => {
            let opts = monitor_cmd::MonitorStartOptions {
                target,
                db_path,
                interval_ms,
                system_metrics,
                quality_metrics,
            };
            monitor_cmd::run_monitor_start(opts, json_output).await
        }
        MonitorCommand::Status { db_path, detailed } => {
            let opts = monitor_cmd::MonitorStatusOptions { db_path, detailed };
            monitor_cmd::run_monitor_status(opts, json_output).await
        }
        MonitorCommand::Alerts {
            db_path,
            count,
            severity,
        } => {
            let opts = monitor_cmd::MonitorAlertsOptions {
                db_path,
                count,
                severity,
            };
            monitor_cmd::run_monitor_alerts(opts, json_output).await
        }
        MonitorCommand::Config {
            db_path,
            cpu_threshold,
            memory_threshold,
            quality_threshold,
            show,
        } => {
            let opts = monitor_cmd::MonitorConfigOptions {
                db_path,
                cpu_threshold,
                memory_threshold,
                quality_threshold,
                show,
            };
            monitor_cmd::run_monitor_config(opts, json_output).await
        }
        MonitorCommand::Dashboard {
            db_path,
            refresh_secs,
            history_points,
        } => {
            let opts = monitor_cmd::MonitorDashboardOptions {
                db_path,
                refresh_secs,
                history_points,
            };
            monitor_cmd::run_monitor_dashboard(opts, json_output).await
        }
    }
}

/// Handle restore subcommands.
pub(crate) async fn handle_restore_command(
    command: RestoreCommand,
    json_output: bool,
) -> Result<()> {
    match command {
        RestoreCommand::Audio {
            input,
            output,
            mode,
            sample_rate,
            declip,
            decrackle,
            dehum,
            denoise,
        } => {
            let opts = restore_cmd::RestoreAudioOptions {
                input,
                output,
                mode,
                sample_rate,
                declip,
                decrackle,
                dehum,
                denoise,
            };
            restore_cmd::run_restore_audio(opts, json_output).await
        }
        RestoreCommand::Video {
            input,
            output,
            mode,
            width,
            height,
        } => {
            let opts = restore_cmd::RestoreVideoOptions {
                input,
                output,
                mode,
                width,
                height,
            };
            restore_cmd::run_restore_video(opts, json_output).await
        }
        RestoreCommand::Analyze {
            input,
            analysis_type,
        } => {
            let opts = restore_cmd::RestoreAnalyzeOptions {
                input,
                analysis_type,
            };
            restore_cmd::run_restore_analyze(opts, json_output).await
        }
        RestoreCommand::Batch {
            input_dir,
            output_dir,
            mode,
            extension,
        } => {
            let opts = restore_cmd::RestoreBatchOptions {
                input_dir,
                output_dir,
                mode,
                extension,
            };
            restore_cmd::run_restore_batch(opts, json_output).await
        }
        RestoreCommand::Compare { original, restored } => {
            let opts = restore_cmd::RestoreCompareOptions { original, restored };
            restore_cmd::run_restore_compare(opts, json_output).await
        }
    }
}

/// Handle captions subcommands.
pub(crate) async fn handle_captions_command(
    command: CaptionsCommand,
    json_output: bool,
) -> Result<()> {
    match command {
        CaptionsCommand::Generate {
            input,
            output,
            format,
            language,
        } => {
            let opts = captions_cmd::CaptionsGenerateOptions {
                input,
                output,
                format,
                language,
            };
            captions_cmd::run_captions_generate(opts, json_output).await
        }
        CaptionsCommand::Sync {
            input,
            reference,
            output,
            max_shift_ms,
        } => {
            let opts = captions_cmd::CaptionsSyncOptions {
                input,
                reference,
                output,
                max_shift_ms,
            };
            captions_cmd::run_captions_sync(opts, json_output).await
        }
        CaptionsCommand::Convert {
            input,
            output,
            from_format,
            to_format,
        } => {
            let opts = captions_cmd::CaptionsConvertOptions {
                input,
                output,
                from_format,
                to_format,
            };
            captions_cmd::run_captions_convert(opts, json_output).await
        }
        CaptionsCommand::Burn {
            video,
            captions,
            output,
            font_size,
            font_color,
        } => {
            let opts = captions_cmd::CaptionsBurnOptions {
                video,
                captions,
                output,
                font_size,
                font_color,
            };
            captions_cmd::run_captions_burn(opts, json_output).await
        }
        CaptionsCommand::Extract {
            input,
            output,
            format,
            track,
        } => {
            let opts = captions_cmd::CaptionsExtractOptions {
                input,
                output,
                format,
                track,
            };
            captions_cmd::run_captions_extract(opts, json_output).await
        }
        CaptionsCommand::Validate {
            input,
            standard,
            report,
        } => {
            let opts = captions_cmd::CaptionsValidateOptions {
                input,
                standard,
                report,
            };
            captions_cmd::run_captions_validate(opts, json_output).await
        }
    }
}

/// Handle preset subcommands.
pub(crate) async fn handle_preset_command(command: PresetCommand, json_output: bool) -> Result<()> {
    use presets::{PresetCategory, PresetManager};

    let custom_dir = PresetManager::default_custom_dir()?;
    let manager = PresetManager::with_custom_dir(&custom_dir)?;

    match command {
        PresetCommand::List {
            category,
            detail: verbose,
        } => {
            let presets = if let Some(cat_str) = category {
                let cat = PresetCategory::from_str(&cat_str)?;
                manager.list_presets_by_category(cat)
            } else {
                manager.list_presets()
            };

            if json_output {
                let json = serde_json::to_string_pretty(&presets)?;
                println!("{}", json);
            } else {
                println!("{}", "Available Presets".green().bold());
                println!("{}", "=".repeat(80));
                println!();

                let mut current_category = None;
                for preset in presets {
                    if current_category != Some(preset.category) {
                        current_category = Some(preset.category);
                        println!("{}", preset.category.name().cyan().bold());
                        println!("{}", preset.category.description().dimmed());
                        println!();
                    }

                    let builtin_badge = if preset.builtin {
                        "[built-in]".dimmed()
                    } else {
                        "[custom]".yellow()
                    };

                    println!("  {} {}", preset.name.green(), builtin_badge);

                    if verbose {
                        println!("    {}", preset.description);
                        println!(
                            "    Video: {} @ {}",
                            preset.video.codec,
                            preset
                                .video
                                .bitrate
                                .as_ref()
                                .map(|s| s.as_str())
                                .unwrap_or("CRF")
                        );
                        println!(
                            "    Audio: {} @ {}",
                            preset.audio.codec,
                            preset
                                .audio
                                .bitrate
                                .as_ref()
                                .map(|s| s.as_str())
                                .unwrap_or("default")
                        );
                        println!("    Container: {}", preset.container);
                        if !preset.tags.is_empty() {
                            println!("    Tags: {}", preset.tags.join(", "));
                        }
                        println!();
                    }
                }

                println!();
                println!("Total: {} presets", manager.preset_names().len());
                println!();
                println!(
                    "Use {} to see detailed information",
                    "oximedia preset show <name>".yellow()
                );
            }

            Ok(())
        }

        PresetCommand::Show { name, toml } => {
            let preset = manager.get_preset(&name)?;

            if json_output {
                let json = serde_json::to_string_pretty(preset)?;
                println!("{}", json);
            } else if toml {
                // Save to temp and read back
                let temp_dir = std::env::temp_dir();
                presets::custom::save_preset_to_file(preset, &temp_dir)?;
                let toml_path = temp_dir.join(format!("{}.toml", preset.name));
                let toml_content = std::fs::read_to_string(&toml_path)?;
                println!("{}", toml_content);
                let _ignore = std::fs::remove_file(&toml_path);
            } else {
                println!("{}", format!("Preset: {}", preset.name).green().bold());
                println!("{}", "=".repeat(80));
                println!();

                println!("{}: {}", "Description".cyan().bold(), preset.description);
                println!("{}: {}", "Category".cyan().bold(), preset.category.name());
                println!("{}: {}", "Container".cyan().bold(), preset.container);
                println!(
                    "{}: {}",
                    "Type".cyan().bold(),
                    if preset.builtin { "Built-in" } else { "Custom" }
                );

                if !preset.tags.is_empty() {
                    println!("{}: {}", "Tags".cyan().bold(), preset.tags.join(", "));
                }

                println!();
                println!("{}", "Video Configuration".yellow().bold());
                println!("{}", "-".repeat(40));
                println!("  Codec: {}", preset.video.codec);
                if let Some(ref bitrate) = preset.video.bitrate {
                    println!("  Bitrate: {}", bitrate);
                }
                if let Some(crf) = preset.video.crf {
                    println!("  CRF: {}", crf);
                }
                if let Some(width) = preset.video.width {
                    println!(
                        "  Resolution: {}x{}",
                        width,
                        preset.video.height.unwrap_or(0)
                    );
                }
                if let Some(fps) = preset.video.fps {
                    println!("  Frame rate: {}", fps);
                }
                if let Some(ref preset_name) = preset.video.preset {
                    println!("  Encoder preset: {}", preset_name);
                }
                if let Some(ref pix_fmt) = preset.video.pixel_format {
                    println!("  Pixel format: {}", pix_fmt);
                }
                println!("  Two-pass: {}", preset.video.two_pass);

                println!();
                println!("{}", "Audio Configuration".yellow().bold());
                println!("{}", "-".repeat(40));
                println!("  Codec: {}", preset.audio.codec);
                if let Some(ref bitrate) = preset.audio.bitrate {
                    println!("  Bitrate: {}", bitrate);
                }
                if let Some(sample_rate) = preset.audio.sample_rate {
                    println!("  Sample rate: {} Hz", sample_rate);
                }
                if let Some(channels) = preset.audio.channels {
                    println!("  Channels: {}", channels);
                }

                println!();
                println!(
                    "{}",
                    format!(
                        "oximedia transcode -i input.mkv -o output.{} --preset-name {}",
                        preset.container, preset.name
                    )
                    .yellow()
                );
            }

            Ok(())
        }

        PresetCommand::Create { output } => {
            let preset = presets::custom::create_preset_interactive()?;

            let out_dir = output.unwrap_or(custom_dir);
            if !out_dir.exists() {
                std::fs::create_dir_all(&out_dir)?;
            }

            presets::custom::save_preset_to_file(&preset, &out_dir)?;

            println!(
                "{} Preset '{}' created successfully!",
                "✓".green(),
                preset.name
            );
            println!(
                "Saved to: {}",
                out_dir.join(format!("{}.toml", preset.name)).display()
            );

            Ok(())
        }

        PresetCommand::Template { output } => {
            presets::custom::generate_template(&output)?;
            println!("{} Template generated: {}", "✓".green(), output.display());
            println!(
                "Edit the template and import it with: oximedia preset import {}",
                output.display()
            );
            Ok(())
        }

        PresetCommand::Import { file } => {
            let preset = presets::custom::load_preset_from_file(&file)?;

            if !custom_dir.exists() {
                std::fs::create_dir_all(&custom_dir)?;
            }

            presets::custom::save_preset_to_file(&preset, &custom_dir)?;

            println!(
                "{} Preset '{}' imported successfully!",
                "✓".green(),
                preset.name
            );

            Ok(())
        }

        PresetCommand::Export { name, output } => {
            let preset = manager.get_preset(&name)?;

            if preset.builtin {
                println!(
                    "{} Cannot export built-in preset '{}'. Use 'oximedia preset show {} --toml' instead.",
                    "!".yellow(),
                    name,
                    name
                );
                return Ok(());
            }

            let output_dir = output.parent().unwrap_or_else(|| std::path::Path::new("."));
            presets::custom::save_preset_to_file(preset, output_dir)?;

            println!("{} Preset exported to: {}", "✓".green(), output.display());

            Ok(())
        }

        PresetCommand::Remove { name } => {
            let preset = manager.get_preset(&name)?;

            if preset.builtin {
                return Err(anyhow::anyhow!("Cannot remove built-in preset '{}'", name));
            }

            let preset_path = custom_dir.join(format!("{}.toml", name));
            if preset_path.exists() {
                std::fs::remove_file(&preset_path)?;
                println!("{} Preset '{}' removed successfully!", "✓".green(), name);
            } else {
                println!(
                    "{} Preset '{}' not found in custom directory",
                    "!".yellow(),
                    name
                );
            }

            Ok(())
        }
    }
}

/// Display OxiMedia version, build info, and feature set.
pub(crate) fn show_version(json: bool) {
    if json {
        let val = serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "rust_version": rustc_version_str(),
            "license": env!("CARGO_PKG_LICENSE"),
            "copyright": "COOLJAPAN OU (Team Kitasan)",
            "homepage": env!("CARGO_PKG_HOMEPAGE"),
            "repository": env!("CARGO_PKG_REPOSITORY"),
            "features": ["audio","video","graph","subtitle","lut","filter","scene","qc","workflow",
                         "batch","monitor","restore","captions","streaming","image","graphics",
                         "multicam","vfx","ndi","videoip","distributed","farm","renderfarm",
                         "plugin","forensics","package","watermark","drm","dedup","archive"],
            "nmos": ["IS-04 v1.3","IS-05 v1.1","IS-07 v1.0","IS-08 v1.0","IS-09 v1.0","IS-11 v1.0"],
        });
        println!("{}", serde_json::to_string_pretty(&val).unwrap_or_default());
        return;
    }
    println!(
        "{}",
        format!("OxiMedia {}", env!("CARGO_PKG_VERSION"))
            .green()
            .bold()
    );
    println!("Built with:  Rust {}", rustc_version_str());
    println!(
        "Features:    {}",
        "audio, video, graph, subtitle, lut, filter, scene, qc, workflow, \
         batch, monitor, restore, captions, streaming, image, graphics, \
         multicam, vfx, ndi, videoip, distributed, farm, renderfarm, \
         plugin, forensics, package, watermark, drm, dedup, archive"
            .cyan()
    );
    println!(
        "NMOS:        {}",
        "IS-04 v1.3, IS-05 v1.1, IS-07 v1.0, IS-08 v1.0, IS-09 v1.0, IS-11 v1.0".cyan()
    );
    println!("License:     {}", env!("CARGO_PKG_LICENSE").yellow());
    println!("Copyright:   {}", "COOLJAPAN OU (Team Kitasan)".yellow());
    println!("Homepage:    {}", env!("CARGO_PKG_HOMEPAGE").dimmed());
    println!("Repository:  {}", env!("CARGO_PKG_REPOSITORY").dimmed());
}

/// Returns a compact Rust compiler version string (e.g. "1.77.0 (stable)").
///
/// Falls back to "unknown" if the version cannot be determined at compile time.
fn rustc_version_str() -> &'static str {
    option_env!("RUSTC_VERSION").unwrap_or("stable")
}

/// Display information about supported formats and codecs.
pub(crate) fn show_info() {
    println!(
        "{}",
        "OxiMedia - Patent-Free Multimedia Framework".green().bold()
    );
    println!();

    println!("{}", "Supported Containers:".cyan().bold());
    println!("  {} Matroska (.mkv)", "✓".green());
    println!("  {} WebM (.webm)", "✓".green());
    println!("  {} Ogg (.ogg, .opus, .oga)", "✓".green());
    println!("  {} FLAC (.flac)", "✓".green());
    println!("  {} WAV (.wav)", "✓".green());
    println!();

    println!("{}", "Supported Video Codecs (Green List):".cyan().bold());
    println!("  {} AV1 (Primary codec, best compression)", "✓".green());
    println!("  {} VP9 (Excellent quality/size ratio)", "✓".green());
    println!("  {} VP8 (Legacy support)", "✓".green());
    println!("  {} Theora (Legacy support)", "✓".green());
    println!();

    println!("{}", "Supported Audio Codecs (Green List):".cyan().bold());
    println!("  {} Opus (Primary codec, versatile)", "✓".green());
    println!("  {} Vorbis (High quality)", "✓".green());
    println!("  {} FLAC (Lossless)", "✓".green());
    println!("  {} PCM (Uncompressed)", "✓".green());
    println!();

    println!("{}", "Rejected Codecs (Patent-Encumbered):".red().bold());
    println!("  {} H.264/AVC", "✗".red());
    println!("  {} H.265/HEVC", "✗".red());
    println!("  {} AAC", "✗".red());
    println!("  {} AC-3/E-AC-3", "✗".red());
    println!("  {} DTS", "✗".red());
    println!();

    println!("{}", "FFmpeg-Compatible Options:".yellow().bold());
    println!("  -i <file>          Input file");
    println!("  -c:v <codec>       Video codec (av1, vp9, vp8)");
    println!("  -c:a <codec>       Audio codec (opus, vorbis, flac, pcm, aac, mp3)");
    println!("  -b:v <bitrate>     Video bitrate (e.g., 2M, 500k)");
    println!("  -vf <filter>       Video filter chain");
    println!("  -ss <time>         Seek to start time");
    println!("  -t <duration>      Duration limit");
    println!("  -r <fps>           Frame rate");
    println!();

    println!("{}", "Examples:".yellow().bold());
    println!("  oximedia transcode -i input.mp4 -o output.webm -c:v vp9 -b:v 2M");
    println!("  oximedia transcode -i input.mp4 -o output.webm --preset-name youtube-1080p");
    println!("  oximedia extract video.mkv frames_%04d.png --every 30");
    println!("  oximedia batch input/ output/ config.toml -j 4");
    println!("  oximedia concat video1.mkv video2.mkv -o output.mkv --method remux");
    println!("  oximedia thumbnail -i video.mkv -o thumb.png --mode auto");
    println!("  oximedia sprite -i video.mkv -o sprite.png --interval 10 --cols 5 --rows 5");
    println!("  oximedia sprite -i video.mkv -o sprite.png --count 100 --vtt --manifest");
    println!("  oximedia metadata -i video.mkv --show");
    println!("  oximedia benchmark -i test.mkv --codecs av1 vp9 --presets fast medium");
    println!("  oximedia validate video1.mkv video2.mkv --checks all --strict");
    println!("  oximedia preset list --category web");
    println!("  oximedia preset show youtube-1080p");
    println!();

    println!("{}", "Available Commands:".cyan().bold());
    println!(
        "  {} Probe media files and show format information",
        "probe".green()
    );
    println!("  {} Show supported formats and codecs", "info".green());
    println!("  {} Transcode media files", "transcode".green());
    println!("  {} Extract frames to images", "extract".green());
    println!("  {} Batch process multiple files", "batch".green());
    println!("  {} Concatenate multiple videos", "concat".green());
    println!("  {} Generate video thumbnails", "thumbnail".green());
    println!("  {} Generate video sprite sheets", "sprite".green());
    println!("  {} Edit media metadata/tags", "metadata".green());
    println!("  {} Run encoding benchmarks", "benchmark".green());
    println!("  {} Validate file integrity", "validate".green());
    println!("  {} Manage transcoding presets", "preset".green());
}
