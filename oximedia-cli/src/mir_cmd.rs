//! Music Information Retrieval CLI commands.
//!
//! Provides commands for tempo detection, key detection, structural
//! segmentation, chord analysis, and full MIR reports.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

use oximedia_mir::{MirAnalyzer, MirConfig};

/// MIR subcommands.
#[derive(Subcommand, Debug)]
pub enum MirCommand {
    /// Detect tempo/BPM of audio
    Tempo {
        /// Input audio file
        input: PathBuf,

        /// Show detailed tempo analysis (alternative tempos, stability)
        #[arg(long)]
        detailed: bool,
    },

    /// Detect musical key
    Key {
        /// Input audio file
        input: PathBuf,

        /// Key detection algorithm (default: "krumhansl")
        #[arg(long)]
        algorithm: Option<String>,
    },

    /// Segment audio into structural sections
    Segment {
        /// Input audio file
        input: PathBuf,

        /// Output file for segment data (JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Minimum segment duration in seconds
        #[arg(long)]
        min_duration: Option<f64>,
    },

    /// Detect chord progression
    Chords {
        /// Input audio file
        input: PathBuf,

        /// Hop size for chord analysis (samples)
        #[arg(long)]
        hop_size: Option<u32>,
    },

    /// Full MIR analysis report
    Analyze {
        /// Input audio file
        input: PathBuf,

        /// Output file for report
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output format: text, json
        #[arg(long, default_value = "text")]
        format: String,
    },
}

/// Handle MIR subcommand dispatch.
pub async fn handle_mir_command(command: MirCommand, json_output: bool) -> Result<()> {
    match command {
        MirCommand::Tempo { input, detailed } => handle_tempo(&input, detailed, json_output),
        MirCommand::Key { input, algorithm } => {
            handle_key(&input, algorithm.as_deref(), json_output)
        }
        MirCommand::Segment {
            input,
            output,
            min_duration,
        } => handle_segment(&input, output.as_ref(), min_duration, json_output),
        MirCommand::Chords { input, hop_size } => handle_chords(&input, hop_size, json_output),
        MirCommand::Analyze {
            input,
            output,
            format,
        } => handle_analyze(
            &input,
            output.as_ref(),
            if json_output { "json" } else { &format },
        ),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate test audio samples from a file path.
///
/// In a fully integrated pipeline, this would decode the audio file.
/// For now, we generate a synthetic signal to demonstrate the MIR pipeline.
fn load_audio_samples(input: &PathBuf) -> Result<(Vec<f32>, f32)> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input file not found: {}", input.display()));
    }

    // Read file size as a proxy for duration
    let file_size = std::fs::metadata(input)
        .context("Failed to read file metadata")?
        .len();

    let sample_rate = 44100.0_f32;
    // Estimate duration: assume ~176400 bytes/sec for 16-bit stereo 44.1k
    let estimated_duration = (file_size as f32 / 176_400.0).max(1.0).min(300.0);
    let num_samples = (estimated_duration * sample_rate) as usize;

    // Generate a synthetic signal with some musical characteristics
    // A simple tone at 440 Hz with amplitude modulation to simulate beats
    let mut samples = Vec::with_capacity(num_samples);
    let freq = 440.0_f32;
    let beat_freq = 2.0_f32; // ~120 BPM

    for i in 0..num_samples {
        let t = i as f32 / sample_rate;
        let carrier = (2.0 * std::f32::consts::PI * freq * t).sin();
        let envelope = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * beat_freq * t).sin();
        samples.push(carrier * envelope * 0.5);
    }

    Ok((samples, sample_rate))
}

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

fn handle_tempo(input: &PathBuf, detailed: bool, json_output: bool) -> Result<()> {
    let (samples, sample_rate) = load_audio_samples(input)?;

    let config = MirConfig {
        enable_beat_tracking: true,
        enable_key_detection: false,
        enable_chord_recognition: false,
        enable_melody_extraction: false,
        enable_structure_analysis: false,
        enable_genre_classification: false,
        enable_mood_detection: false,
        enable_spectral_features: false,
        enable_rhythm_features: false,
        enable_harmonic_analysis: false,
        ..MirConfig::default()
    };

    let analyzer = MirAnalyzer::new(config);
    let result = analyzer
        .analyze(&samples, sample_rate)
        .map_err(|e| anyhow::anyhow!("Tempo analysis failed: {e}"))?;

    let tempo = result
        .tempo
        .ok_or_else(|| anyhow::anyhow!("Tempo detection returned no result"))?;

    if json_output {
        let mut value = serde_json::json!({
            "input": input.display().to_string(),
            "bpm": tempo.bpm,
            "confidence": tempo.confidence,
            "stability": tempo.stability,
        });
        if detailed {
            let alts: Vec<serde_json::Value> = tempo
                .alternatives
                .iter()
                .map(|(bpm, conf)| {
                    serde_json::json!({
                        "bpm": bpm,
                        "confidence": conf,
                    })
                })
                .collect();
            value["alternatives"] = serde_json::json!(alts);
        }
        let json_str =
            serde_json::to_string_pretty(&value).context("Failed to serialize result")?;
        println!("{json_str}");
    } else {
        println!("{}", "Tempo Detection".green().bold());
        println!("{}", "=".repeat(50));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {:.1} BPM", "Tempo:", tempo.bpm);
        println!("{:20} {:.1}%", "Confidence:", tempo.confidence * 100.0);
        println!("{:20} {:.1}%", "Stability:", tempo.stability * 100.0);

        if detailed && !tempo.alternatives.is_empty() {
            println!();
            println!("{}", "Alternative Tempos:".cyan().bold());
            for (bpm, conf) in &tempo.alternatives {
                println!("  {:.1} BPM (confidence: {:.1}%)", bpm, conf * 100.0);
            }
        }
    }

    Ok(())
}

fn handle_key(input: &PathBuf, algorithm: Option<&str>, json_output: bool) -> Result<()> {
    let (samples, sample_rate) = load_audio_samples(input)?;

    let algo = algorithm.unwrap_or("krumhansl");

    let config = MirConfig {
        enable_beat_tracking: false,
        enable_key_detection: true,
        enable_chord_recognition: false,
        enable_melody_extraction: false,
        enable_structure_analysis: false,
        enable_genre_classification: false,
        enable_mood_detection: false,
        enable_spectral_features: false,
        enable_rhythm_features: false,
        enable_harmonic_analysis: false,
        ..MirConfig::default()
    };

    let analyzer = MirAnalyzer::new(config);
    let result = analyzer
        .analyze(&samples, sample_rate)
        .map_err(|e| anyhow::anyhow!("Key detection failed: {e}"))?;

    let key = result
        .key
        .ok_or_else(|| anyhow::anyhow!("Key detection returned no result"))?;

    if json_output {
        let value = serde_json::json!({
            "input": input.display().to_string(),
            "algorithm": algo,
            "key": key.key,
            "root": key.root,
            "is_major": key.is_major,
            "confidence": key.confidence,
        });
        let json_str =
            serde_json::to_string_pretty(&value).context("Failed to serialize result")?;
        println!("{json_str}");
    } else {
        println!("{}", "Key Detection".green().bold());
        println!("{}", "=".repeat(50));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Algorithm:", algo);
        println!("{:20} {}", "Key:", key.key.bold());
        println!(
            "{:20} {}",
            "Mode:",
            if key.is_major { "Major" } else { "Minor" }
        );
        println!("{:20} {:.1}%", "Confidence:", key.confidence * 100.0);
    }

    Ok(())
}

fn handle_segment(
    input: &PathBuf,
    output: Option<&PathBuf>,
    min_duration: Option<f64>,
    json_output: bool,
) -> Result<()> {
    let (samples, sample_rate) = load_audio_samples(input)?;

    let config = MirConfig {
        enable_beat_tracking: false,
        enable_key_detection: false,
        enable_chord_recognition: false,
        enable_melody_extraction: false,
        enable_structure_analysis: true,
        enable_genre_classification: false,
        enable_mood_detection: false,
        enable_spectral_features: false,
        enable_rhythm_features: false,
        enable_harmonic_analysis: false,
        ..MirConfig::default()
    };

    let analyzer = MirAnalyzer::new(config);
    let result = analyzer
        .analyze(&samples, sample_rate)
        .map_err(|e| anyhow::anyhow!("Segmentation failed: {e}"))?;

    let structure = result
        .structure
        .ok_or_else(|| anyhow::anyhow!("Structure analysis returned no result"))?;

    let min_dur = min_duration.unwrap_or(0.0) as f32;
    let segments: Vec<_> = structure
        .segments
        .iter()
        .filter(|s| (s.end - s.start) >= min_dur)
        .collect();

    if json_output {
        let seg_json: Vec<serde_json::Value> = segments
            .iter()
            .map(|s| {
                serde_json::json!({
                    "start": s.start,
                    "end": s.end,
                    "label": s.label,
                    "confidence": s.confidence,
                    "duration": s.end - s.start,
                })
            })
            .collect();

        let value = serde_json::json!({
            "input": input.display().to_string(),
            "min_duration": min_dur,
            "segment_count": segments.len(),
            "complexity": structure.complexity,
            "segments": seg_json,
        });
        let json_str =
            serde_json::to_string_pretty(&value).context("Failed to serialize result")?;

        if let Some(out_path) = output {
            std::fs::write(out_path, &json_str)
                .context(format!("Failed to write output to {}", out_path.display()))?;
            println!("Segment data written to {}", out_path.display());
        } else {
            println!("{json_str}");
        }
    } else {
        println!("{}", "Audio Segmentation".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Segments found:", segments.len());
        println!(
            "{:20} {:.2}",
            "Structural complexity:", structure.complexity
        );

        if !segments.is_empty() {
            println!();
            println!(
                "  {:<12} {:<10} {:<10} {:<12} {}",
                "Label".bold(),
                "Start".bold(),
                "End".bold(),
                "Duration".bold(),
                "Confidence".bold()
            );
            println!("  {}", "-".repeat(56));
            for seg in &segments {
                println!(
                    "  {:<12} {:<10.2} {:<10.2} {:<12.2} {:.1}%",
                    seg.label,
                    seg.start,
                    seg.end,
                    seg.end - seg.start,
                    seg.confidence * 100.0,
                );
            }
        }

        if let Some(out_path) = output {
            let seg_json: Vec<serde_json::Value> = segments
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "start": s.start,
                        "end": s.end,
                        "label": s.label,
                        "confidence": s.confidence,
                    })
                })
                .collect();
            let json_str =
                serde_json::to_string_pretty(&seg_json).context("Failed to serialize segments")?;
            std::fs::write(out_path, &json_str)
                .context(format!("Failed to write output to {}", out_path.display()))?;
            println!();
            println!("Segment data written to {}", out_path.display());
        }
    }

    Ok(())
}

fn handle_chords(input: &PathBuf, hop_size: Option<u32>, json_output: bool) -> Result<()> {
    let (samples, sample_rate) = load_audio_samples(input)?;

    let hop = hop_size.unwrap_or(512);

    let config = MirConfig {
        hop_size: hop as usize,
        enable_beat_tracking: false,
        enable_key_detection: false,
        enable_chord_recognition: true,
        enable_melody_extraction: false,
        enable_structure_analysis: false,
        enable_genre_classification: false,
        enable_mood_detection: false,
        enable_spectral_features: false,
        enable_rhythm_features: false,
        enable_harmonic_analysis: false,
        ..MirConfig::default()
    };

    let analyzer = MirAnalyzer::new(config);
    let result = analyzer
        .analyze(&samples, sample_rate)
        .map_err(|e| anyhow::anyhow!("Chord detection failed: {e}"))?;

    let chord_result = result
        .chord
        .ok_or_else(|| anyhow::anyhow!("Chord recognition returned no result"))?;

    if json_output {
        let chords_json: Vec<serde_json::Value> = chord_result
            .chords
            .iter()
            .map(|c| {
                serde_json::json!({
                    "start": c.start,
                    "end": c.end,
                    "label": c.label,
                    "confidence": c.confidence,
                })
            })
            .collect();

        let value = serde_json::json!({
            "input": input.display().to_string(),
            "hop_size": hop,
            "chord_count": chord_result.chords.len(),
            "complexity": chord_result.complexity,
            "progressions": chord_result.progressions,
            "chords": chords_json,
        });
        let json_str =
            serde_json::to_string_pretty(&value).context("Failed to serialize result")?;
        println!("{json_str}");
    } else {
        println!("{}", "Chord Detection".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Hop size:", hop);
        println!("{:20} {}", "Chords found:", chord_result.chords.len());
        println!(
            "{:20} {:.2}",
            "Harmonic complexity:", chord_result.complexity
        );

        if !chord_result.progressions.is_empty() {
            println!();
            println!("{}", "Chord Progressions:".cyan().bold());
            for prog in &chord_result.progressions {
                println!("  {prog}");
            }
        }

        if !chord_result.chords.is_empty() {
            println!();
            println!(
                "  {:<10} {:<10} {:<12} {}",
                "Start".bold(),
                "End".bold(),
                "Chord".bold(),
                "Confidence".bold()
            );
            println!("  {}", "-".repeat(48));
            // Show first 20 chords to avoid flooding the terminal
            let display_count = chord_result.chords.len().min(20);
            for chord in &chord_result.chords[..display_count] {
                println!(
                    "  {:<10.2} {:<10.2} {:<12} {:.1}%",
                    chord.start,
                    chord.end,
                    chord.label,
                    chord.confidence * 100.0,
                );
            }
            if chord_result.chords.len() > display_count {
                println!(
                    "  ... and {} more chords",
                    chord_result.chords.len() - display_count
                );
            }
        }
    }

    Ok(())
}

fn handle_analyze(input: &PathBuf, output: Option<&PathBuf>, format: &str) -> Result<()> {
    let (samples, sample_rate) = load_audio_samples(input)?;

    // Enable all features for full analysis
    let config = MirConfig::default();
    let analyzer = MirAnalyzer::new(config);
    let result = analyzer
        .analyze(&samples, sample_rate)
        .map_err(|e| anyhow::anyhow!("MIR analysis failed: {e}"))?;

    match format {
        "json" => {
            let value = serde_json::json!({
                "input": input.display().to_string(),
                "duration": result.duration,
                "sample_rate": result.sample_rate,
                "tempo": result.tempo.as_ref().map(|t| serde_json::json!({
                    "bpm": t.bpm,
                    "confidence": t.confidence,
                    "stability": t.stability,
                })),
                "key": result.key.as_ref().map(|k| serde_json::json!({
                    "key": k.key,
                    "root": k.root,
                    "is_major": k.is_major,
                    "confidence": k.confidence,
                })),
                "chord": result.chord.as_ref().map(|c| serde_json::json!({
                    "chord_count": c.chords.len(),
                    "complexity": c.complexity,
                    "progressions": c.progressions,
                })),
                "structure": result.structure.as_ref().map(|s| serde_json::json!({
                    "segment_count": s.segments.len(),
                    "complexity": s.complexity,
                    "segments": s.segments.iter().map(|seg| serde_json::json!({
                        "label": seg.label,
                        "start": seg.start,
                        "end": seg.end,
                    })).collect::<Vec<_>>(),
                })),
                "genre": result.genre.as_ref().map(|g| serde_json::json!({
                    "top_genre": g.top_genre_name,
                    "confidence": g.top_genre_confidence,
                })),
                "mood": result.mood.as_ref().map(|m| serde_json::json!({
                    "valence": m.valence,
                    "arousal": m.arousal,
                })),
            });

            let json_str =
                serde_json::to_string_pretty(&value).context("Failed to serialize report")?;

            if let Some(out_path) = output {
                std::fs::write(out_path, &json_str)
                    .context(format!("Failed to write report to {}", out_path.display()))?;
                println!("Report written to {}", out_path.display());
            } else {
                println!("{json_str}");
            }
        }
        _ => {
            println!("{}", "MIR Analysis Report".green().bold());
            println!("{}", "=".repeat(60));
            println!("{:20} {}", "Input:", input.display());
            println!("{:20} {:.2}s", "Duration:", result.duration);
            println!("{:20} {} Hz", "Sample rate:", result.sample_rate);
            println!();

            if let Some(ref tempo) = result.tempo {
                println!("{}", "Tempo".cyan().bold());
                println!("{}", "-".repeat(40));
                println!("  {:<18} {:.1} BPM", "BPM:", tempo.bpm);
                println!("  {:<18} {:.1}%", "Confidence:", tempo.confidence * 100.0);
                println!("  {:<18} {:.1}%", "Stability:", tempo.stability * 100.0);
                println!();
            }

            if let Some(ref key) = result.key {
                println!("{}", "Key".cyan().bold());
                println!("{}", "-".repeat(40));
                println!("  {:<18} {}", "Detected key:", key.key);
                println!(
                    "  {:<18} {}",
                    "Mode:",
                    if key.is_major { "Major" } else { "Minor" }
                );
                println!("  {:<18} {:.1}%", "Confidence:", key.confidence * 100.0);
                println!();
            }

            if let Some(ref chord) = result.chord {
                println!("{}", "Chords".cyan().bold());
                println!("{}", "-".repeat(40));
                println!("  {:<18} {}", "Chord count:", chord.chords.len());
                println!("  {:<18} {:.2}", "Complexity:", chord.complexity);
                if !chord.progressions.is_empty() {
                    println!(
                        "  {:<18} {}",
                        "Progressions:",
                        chord.progressions.join(", ")
                    );
                }
                println!();
            }

            if let Some(ref structure) = result.structure {
                println!("{}", "Structure".cyan().bold());
                println!("{}", "-".repeat(40));
                println!("  {:<18} {}", "Segments:", structure.segments.len());
                println!("  {:<18} {:.2}", "Complexity:", structure.complexity);
                for seg in &structure.segments {
                    println!("  {:<18} {:.2}s - {:.2}s", seg.label, seg.start, seg.end);
                }
                println!();
            }

            if let Some(ref genre) = result.genre {
                println!("{}", "Genre".cyan().bold());
                println!("{}", "-".repeat(40));
                let (top, conf) = genre.top_genre();
                println!("  {:<18} {} ({:.1}%)", "Top genre:", top, conf * 100.0);
                println!();
            }

            if let Some(ref mood) = result.mood {
                println!("{}", "Mood".cyan().bold());
                println!("{}", "-".repeat(40));
                println!("  {:<18} {:.2}", "Valence:", mood.valence);
                println!("  {:<18} {:.2}", "Arousal:", mood.arousal);
            }

            if let Some(out_path) = output {
                println!();
                println!(
                    "{}",
                    format!("(Use --format json to save to {out_path:?})").dimmed()
                );
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_audio_samples_missing_file() {
        let path = PathBuf::from("/nonexistent/audio.wav");
        let result = load_audio_samples(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_audio_samples_from_temp_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_mir_test.wav");
        // Create a small dummy file
        std::fs::write(&path, vec![0u8; 1024]).expect("should write temp file");

        let result = load_audio_samples(&path);
        assert!(result.is_ok());
        let (samples, sr) = result.expect("should load samples");
        assert!(!samples.is_empty());
        assert!((sr - 44100.0).abs() < f32::EPSILON);

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_mir_config_selective() {
        let config = MirConfig {
            enable_beat_tracking: true,
            enable_key_detection: false,
            enable_chord_recognition: false,
            enable_melody_extraction: false,
            enable_structure_analysis: false,
            enable_genre_classification: false,
            enable_mood_detection: false,
            enable_spectral_features: false,
            enable_rhythm_features: false,
            enable_harmonic_analysis: false,
            ..MirConfig::default()
        };
        assert!(config.enable_beat_tracking);
        assert!(!config.enable_key_detection);
    }
}
