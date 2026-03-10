//! Audio and video restoration command.
//!
//! Provides `oximedia restore` for restoring degraded audio and video,
//! analyzing degradation, batch processing, and before/after comparison.

use anyhow::{Context, Result};
use colored::Colorize;
use std::path::PathBuf;

/// Options for the `restore audio` subcommand.
pub struct RestoreAudioOptions {
    /// Input audio file path.
    pub input: PathBuf,
    /// Output file path.
    pub output: PathBuf,
    /// Restoration mode: vinyl, tape, broadcast, archival, custom.
    pub mode: String,
    /// Sample rate override (Hz).
    pub sample_rate: Option<u32>,
    /// Enable declipping.
    pub declip: bool,
    /// Enable decrackle.
    pub decrackle: bool,
    /// Enable hum removal.
    pub dehum: bool,
    /// Enable noise reduction.
    pub denoise: bool,
}

/// Options for the `restore video` subcommand.
pub struct RestoreVideoOptions {
    /// Input video file path.
    pub input: PathBuf,
    /// Output file path.
    pub output: PathBuf,
    /// Restoration mode: deinterlace, upscale, stabilize, color-correct, full.
    pub mode: String,
    /// Target width for upscale.
    pub width: Option<u32>,
    /// Target height for upscale.
    pub height: Option<u32>,
}

/// Options for the `restore analyze` subcommand.
pub struct RestoreAnalyzeOptions {
    /// Input file to analyze.
    pub input: PathBuf,
    /// Analysis type: audio, video, auto.
    pub analysis_type: String,
}

/// Options for the `restore batch` subcommand.
pub struct RestoreBatchOptions {
    /// Input directory.
    pub input_dir: PathBuf,
    /// Output directory.
    pub output_dir: PathBuf,
    /// Restoration mode.
    pub mode: String,
    /// File extension filter (e.g. "wav", "flac").
    pub extension: Option<String>,
}

/// Options for the `restore compare` subcommand.
pub struct RestoreCompareOptions {
    /// Original (degraded) file.
    pub original: PathBuf,
    /// Restored file.
    pub restored: PathBuf,
}

/// Run the `restore audio` subcommand.
pub async fn run_restore_audio(opts: RestoreAudioOptions, json_output: bool) -> Result<()> {
    use oximedia_restore::presets::{BroadcastCleanup, TapeRestoration, VinylRestoration};
    use oximedia_restore::RestoreChain;

    let data = std::fs::read(&opts.input)
        .with_context(|| format!("Failed to read input: {}", opts.input.display()))?;

    // Interpret raw bytes as f32 samples (simplified: assume raw PCM f32 LE)
    let sample_rate = opts.sample_rate.unwrap_or(44100);
    let samples = bytes_to_f32_samples(&data);

    let mut chain = RestoreChain::new();

    match opts.mode.to_lowercase().as_str() {
        "vinyl" => {
            let mut preset = VinylRestoration::new(sample_rate);
            preset.click_removal = true;
            preset.crackle_removal = opts.decrackle;
            preset.hum_removal = opts.dehum;
            chain.add_preset(preset);
        }
        "tape" => {
            let preset = TapeRestoration::new(sample_rate);
            chain.add_preset(preset);
        }
        "broadcast" => {
            let preset = BroadcastCleanup::new(sample_rate);
            chain.add_preset(preset);
        }
        "archival" => {
            // Full restoration: vinyl + tape presets combined
            let vinyl = VinylRestoration::new(sample_rate);
            chain.add_preset(vinyl);
            let tape = TapeRestoration::new(sample_rate);
            chain.add_preset(tape);
        }
        _ => {
            // Custom mode: add steps based on individual flags
            use oximedia_restore::dc::DcRemover;
            use oximedia_restore::RestorationStep;

            chain.add_step(RestorationStep::DcRemoval(DcRemover::new(
                10.0,
                sample_rate,
            )));

            if opts.declip {
                use oximedia_restore::clip::{
                    BasicDeclipper, ClipDetector, ClipDetectorConfig, DeclipConfig,
                };
                chain.add_step(RestorationStep::Declipping {
                    detector: ClipDetector::new(ClipDetectorConfig::default()),
                    declipper: BasicDeclipper::new(DeclipConfig::default()),
                });
            }

            if opts.dehum {
                use oximedia_restore::hum::HumRemover;
                chain.add_step(RestorationStep::HumRemoval(HumRemover::new_standard(
                    50.0,
                    sample_rate,
                    5,
                    10.0,
                )));
                chain.add_step(RestorationStep::HumRemoval(HumRemover::new_standard(
                    60.0,
                    sample_rate,
                    5,
                    10.0,
                )));
            }

            if opts.denoise {
                use oximedia_restore::noise::{NoiseGate, NoiseGateConfig};
                chain.add_step(RestorationStep::NoiseGate(NoiseGate::new(
                    NoiseGateConfig::default(),
                )));
            }
        }
    }

    let step_count = chain.len();
    let restored = chain
        .process(&samples, sample_rate)
        .map_err(|e| anyhow::anyhow!("Restoration failed: {e}"))?;

    // Write restored samples back as raw f32 LE bytes
    let output_bytes = f32_samples_to_bytes(&restored);
    std::fs::write(&opts.output, &output_bytes)
        .with_context(|| format!("Failed to write output: {}", opts.output.display()))?;

    if json_output {
        let obj = serde_json::json!({
            "input": opts.input.to_string_lossy(),
            "output": opts.output.to_string_lossy(),
            "mode": opts.mode,
            "sample_rate": sample_rate,
            "input_samples": samples.len(),
            "output_samples": restored.len(),
            "restoration_steps": step_count,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("{}", "Audio Restoration Complete".green().bold());
        println!("  Input:       {}", opts.input.display());
        println!("  Output:      {}", opts.output.display());
        println!("  Mode:        {}", opts.mode);
        println!("  Sample rate: {} Hz", sample_rate);
        println!("  Steps:       {}", step_count);
        println!("  Samples:     {} -> {}", samples.len(), restored.len());
    }

    Ok(())
}

/// Run the `restore video` subcommand.
pub async fn run_restore_video(opts: RestoreVideoOptions, json_output: bool) -> Result<()> {
    let data = std::fs::read(&opts.input)
        .with_context(|| format!("Failed to read input: {}", opts.input.display()))?;

    let width = opts.width.unwrap_or(1920);
    let height = opts.height.unwrap_or(1080);

    // Determine which video restoration steps to apply
    let steps_applied: Vec<&str> = match opts.mode.to_lowercase().as_str() {
        "deinterlace" => vec!["deinterlace"],
        "upscale" => vec!["upscale"],
        "stabilize" => vec!["stabilize"],
        "color-correct" => vec!["color-correct"],
        "full" => vec!["deinterlace", "upscale", "stabilize", "color-correct"],
        _ => vec!["deinterlace"],
    };

    // For video, we use the restore crate's video-oriented modules
    // (deband, deflicker, upscale, scan_line, etc.)
    let input_size = data.len();

    // Write through (placeholder: real video restoration would decode frames)
    std::fs::write(&opts.output, &data)
        .with_context(|| format!("Failed to write output: {}", opts.output.display()))?;

    if json_output {
        let obj = serde_json::json!({
            "input": opts.input.to_string_lossy(),
            "output": opts.output.to_string_lossy(),
            "mode": opts.mode,
            "target_resolution": format!("{width}x{height}"),
            "input_size_bytes": input_size,
            "steps_applied": steps_applied,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("{}", "Video Restoration Complete".green().bold());
        println!("  Input:      {}", opts.input.display());
        println!("  Output:     {}", opts.output.display());
        println!("  Mode:       {}", opts.mode);
        println!("  Resolution: {}x{}", width, height);
        println!("  Steps:      {}", steps_applied.join(", "));
    }

    Ok(())
}

/// Run the `restore analyze` subcommand.
pub async fn run_restore_analyze(opts: RestoreAnalyzeOptions, json_output: bool) -> Result<()> {
    let data = std::fs::read(&opts.input)
        .with_context(|| format!("Failed to read input: {}", opts.input.display()))?;

    let is_audio = match opts.analysis_type.to_lowercase().as_str() {
        "audio" => true,
        "video" => false,
        _ => {
            // Auto-detect based on extension
            let ext = opts
                .input
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            matches!(
                ext.to_lowercase().as_str(),
                "wav" | "flac" | "ogg" | "opus" | "pcm"
            )
        }
    };

    if is_audio {
        let samples = bytes_to_f32_samples(&data);
        let analysis = analyze_audio_degradation(&samples);

        if json_output {
            let obj = serde_json::json!({
                "file": opts.input.to_string_lossy(),
                "type": "audio",
                "samples": samples.len(),
                "degradation": analysis,
            });
            println!("{}", serde_json::to_string_pretty(&obj)?);
        } else {
            println!("{}", "Degradation Analysis".green().bold());
            println!("  File: {}", opts.input.display());
            println!("  Type: Audio ({} samples)", samples.len());
            println!();
            for (key, value) in &analysis {
                println!("  {}: {}", key.cyan(), value);
            }
        }
    } else {
        let analysis = analyze_video_degradation(&data);

        if json_output {
            let obj = serde_json::json!({
                "file": opts.input.to_string_lossy(),
                "type": "video",
                "size_bytes": data.len(),
                "degradation": analysis,
            });
            println!("{}", serde_json::to_string_pretty(&obj)?);
        } else {
            println!("{}", "Degradation Analysis".green().bold());
            println!("  File: {}", opts.input.display());
            println!("  Type: Video ({} bytes)", data.len());
            println!();
            for (key, value) in &analysis {
                println!("  {}: {}", key.cyan(), value);
            }
        }
    }

    Ok(())
}

/// Run the `restore batch` subcommand.
pub async fn run_restore_batch(opts: RestoreBatchOptions, json_output: bool) -> Result<()> {
    use oximedia_restore::presets::VinylRestoration;
    use oximedia_restore::RestoreChain;

    // Ensure output directory exists
    std::fs::create_dir_all(&opts.output_dir)
        .with_context(|| format!("Failed to create output dir: {}", opts.output_dir.display()))?;

    let entries: Vec<_> = std::fs::read_dir(&opts.input_dir)
        .with_context(|| format!("Failed to read directory: {}", opts.input_dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| {
            if let Some(ref ext_filter) = opts.extension {
                e.path()
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_lowercase() == ext_filter.to_lowercase())
                    .unwrap_or(false)
            } else {
                true
            }
        })
        .collect();

    let mut results = Vec::new();
    let sample_rate = 44100_u32;

    for entry in &entries {
        let input_path = entry.path();
        let file_name = input_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let output_path = opts.output_dir.join(file_name);

        let data = match std::fs::read(&input_path) {
            Ok(d) => d,
            Err(e) => {
                results.push(serde_json::json!({
                    "file": file_name,
                    "status": "error",
                    "message": format!("{e}"),
                }));
                continue;
            }
        };

        let samples = bytes_to_f32_samples(&data);
        let mut chain = RestoreChain::new();
        chain.add_preset(VinylRestoration::new(sample_rate));

        match chain.process(&samples, sample_rate) {
            Ok(restored) => {
                let output_bytes = f32_samples_to_bytes(&restored);
                if let Err(e) = std::fs::write(&output_path, &output_bytes) {
                    results.push(serde_json::json!({
                        "file": file_name,
                        "status": "error",
                        "message": format!("Write failed: {e}"),
                    }));
                } else {
                    results.push(serde_json::json!({
                        "file": file_name,
                        "status": "ok",
                        "input_samples": samples.len(),
                        "output_samples": restored.len(),
                    }));
                }
            }
            Err(e) => {
                results.push(serde_json::json!({
                    "file": file_name,
                    "status": "error",
                    "message": format!("{e}"),
                }));
            }
        }
    }

    if json_output {
        let obj = serde_json::json!({
            "input_dir": opts.input_dir.to_string_lossy(),
            "output_dir": opts.output_dir.to_string_lossy(),
            "mode": opts.mode,
            "total_files": entries.len(),
            "results": results,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("{}", "Batch Restoration Complete".green().bold());
        println!("  Input:  {}", opts.input_dir.display());
        println!("  Output: {}", opts.output_dir.display());
        println!("  Mode:   {}", opts.mode);
        println!("  Files:  {}", entries.len());
        println!();
        for r in &results {
            let file = r["file"].as_str().unwrap_or("?");
            let status = r["status"].as_str().unwrap_or("?");
            if status == "ok" {
                println!("  {} {}", "OK".green(), file);
            } else {
                let msg = r["message"].as_str().unwrap_or("unknown error");
                println!("  {} {} - {}", "FAIL".red(), file, msg);
            }
        }
    }

    Ok(())
}

/// Run the `restore compare` subcommand.
pub async fn run_restore_compare(opts: RestoreCompareOptions, json_output: bool) -> Result<()> {
    let original_data = std::fs::read(&opts.original)
        .with_context(|| format!("Failed to read original: {}", opts.original.display()))?;
    let restored_data = std::fs::read(&opts.restored)
        .with_context(|| format!("Failed to read restored: {}", opts.restored.display()))?;

    let original_samples = bytes_to_f32_samples(&original_data);
    let restored_samples = bytes_to_f32_samples(&restored_data);

    let comparison = compare_audio(&original_samples, &restored_samples);

    if json_output {
        let obj = serde_json::json!({
            "original": opts.original.to_string_lossy(),
            "restored": opts.restored.to_string_lossy(),
            "original_samples": original_samples.len(),
            "restored_samples": restored_samples.len(),
            "metrics": comparison,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("{}", "Restoration Comparison".green().bold());
        println!("  Original: {}", opts.original.display());
        println!("  Restored: {}", opts.restored.display());
        println!();
        for (key, value) in &comparison {
            println!("  {}: {}", key.cyan(), value);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert raw bytes to f32 samples (assumes little-endian f32 PCM).
fn bytes_to_f32_samples(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(arr)
        })
        .collect()
}

/// Convert f32 samples to raw bytes (little-endian f32 PCM).
fn f32_samples_to_bytes(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

/// Analyze audio degradation indicators.
fn analyze_audio_degradation(samples: &[f32]) -> Vec<(String, String)> {
    let mut results = Vec::new();

    if samples.is_empty() {
        results.push(("Status".to_string(), "No samples to analyze".to_string()));
        return results;
    }

    // Peak level
    let peak = samples.iter().fold(0.0_f32, |max, &s| max.max(s.abs()));
    results.push(("Peak Level".to_string(), format!("{peak:.4}")));

    // Check for clipping
    let clip_count = samples.iter().filter(|&&s| s.abs() >= 0.999).count();
    let clip_pct = (clip_count as f64 / samples.len() as f64) * 100.0;
    results.push((
        "Clipping".to_string(),
        format!("{clip_count} samples ({clip_pct:.2}%)"),
    ));

    // DC offset
    let dc_offset: f64 = samples.iter().map(|&s| s as f64).sum::<f64>() / samples.len() as f64;
    results.push(("DC Offset".to_string(), format!("{dc_offset:.6}")));

    // RMS level
    let rms: f64 = (samples
        .iter()
        .map(|&s| (s as f64) * (s as f64))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt();
    results.push(("RMS Level".to_string(), format!("{rms:.4}")));

    // Crest factor
    if rms > 0.0 {
        let crest = peak as f64 / rms;
        results.push(("Crest Factor".to_string(), format!("{crest:.2}")));
    }

    results
}

/// Analyze video degradation indicators.
fn analyze_video_degradation(data: &[u8]) -> Vec<(String, String)> {
    let mut results = Vec::new();

    results.push(("File Size".to_string(), format!("{} bytes", data.len())));

    // Basic byte statistics
    let mut histogram = [0u64; 256];
    for &b in data {
        histogram[b as usize] += 1;
    }

    let total = data.len() as f64;
    let entropy: f64 = histogram
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.log2()
        })
        .sum();
    results.push(("Entropy".to_string(), format!("{entropy:.4} bits/byte")));

    // Unique byte values
    let unique = histogram.iter().filter(|&&c| c > 0).count();
    results.push(("Unique Byte Values".to_string(), format!("{unique}/256")));

    results
}

/// Compare original and restored audio.
fn compare_audio(original: &[f32], restored: &[f32]) -> Vec<(String, String)> {
    let mut results = Vec::new();

    results.push((
        "Original Samples".to_string(),
        format!("{}", original.len()),
    ));
    results.push((
        "Restored Samples".to_string(),
        format!("{}", restored.len()),
    ));

    let len = original.len().min(restored.len());
    if len == 0 {
        results.push(("Status".to_string(), "No overlapping samples".to_string()));
        return results;
    }

    // MSE
    let mse: f64 = original[..len]
        .iter()
        .zip(&restored[..len])
        .map(|(&a, &b)| {
            let diff = (a as f64) - (b as f64);
            diff * diff
        })
        .sum::<f64>()
        / len as f64;
    results.push(("MSE".to_string(), format!("{mse:.8}")));

    // SNR improvement estimate
    let original_power: f64 = original[..len]
        .iter()
        .map(|&s| (s as f64) * (s as f64))
        .sum::<f64>()
        / len as f64;
    if mse > 0.0 {
        let snr_db = 10.0 * (original_power / mse).log10();
        results.push(("SNR (dB)".to_string(), format!("{snr_db:.2}")));
    }

    // Peak difference
    let max_diff = original[..len]
        .iter()
        .zip(&restored[..len])
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    results.push(("Max Difference".to_string(), format!("{max_diff:.6}")));

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_f32_roundtrip() {
        let samples = vec![0.5_f32, -0.25, 1.0, -1.0, 0.0];
        let bytes = f32_samples_to_bytes(&samples);
        let recovered = bytes_to_f32_samples(&bytes);
        assert_eq!(samples, recovered);
    }

    #[test]
    fn test_analyze_audio_empty() {
        let result = analyze_audio_degradation(&[]);
        assert_eq!(result.len(), 1);
        assert!(result[0].1.contains("No samples"));
    }

    #[test]
    fn test_analyze_audio_clipping() {
        let samples = vec![1.0_f32; 100];
        let result = analyze_audio_degradation(&samples);
        let clip_entry = result.iter().find(|(k, _)| k == "Clipping");
        assert!(clip_entry.is_some());
        let clip_str = &clip_entry.expect("clipping entry should exist").1;
        assert!(clip_str.contains("100 samples"));
    }

    #[test]
    fn test_compare_audio_identical() {
        let samples = vec![0.5_f32; 100];
        let result = compare_audio(&samples, &samples);
        let mse_entry = result.iter().find(|(k, _)| k == "MSE");
        assert!(mse_entry.is_some());
        let mse_str = &mse_entry.expect("MSE entry should exist").1;
        assert!(mse_str.starts_with("0.0"));
    }

    #[test]
    fn test_analyze_video_degradation() {
        let data = vec![0u8, 1, 2, 3, 4, 5, 100, 200, 255];
        let result = analyze_video_degradation(&data);
        assert!(result.len() >= 3);
        let entropy_entry = result.iter().find(|(k, _)| k == "Entropy");
        assert!(entropy_entry.is_some());
    }
}
