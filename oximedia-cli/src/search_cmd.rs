//! Search command for `oximedia search`.
//!
//! Provides text search, visual similarity, fingerprint, and index subcommands
//! via `oximedia-search` and `oximedia-forensics`.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// Subcommands for `oximedia search`.
#[derive(Subcommand)]
pub enum SearchCommand {
    /// Full-text search in a media index
    Text {
        /// Search query string
        #[arg(value_name = "QUERY")]
        query: String,

        /// Path to the search index directory
        #[arg(short, long)]
        index: PathBuf,

        /// Maximum number of results
        #[arg(short, long, default_value = "20")]
        limit: usize,

        /// Result offset for pagination
        #[arg(long, default_value = "0")]
        offset: usize,
    },

    /// Find visually similar media files using perceptual hashing
    Similar {
        /// Input image or video file to find matches for
        #[arg(short, long)]
        input: PathBuf,

        /// Path to the search index directory
        #[arg(short = 'x', long)]
        index: PathBuf,

        /// Maximum number of results
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Similarity threshold (0.0 – 1.0)
        #[arg(long, default_value = "0.8")]
        threshold: f32,
    },

    /// Compute content fingerprint for a media file
    Fingerprint {
        /// Input media file
        #[arg(short, long)]
        input: PathBuf,

        /// Output format: text, json
        #[arg(long, default_value = "text")]
        output_format: String,

        /// Compare against another file fingerprint
        #[arg(long)]
        compare: Option<PathBuf>,
    },

    /// Index a media file for later searching
    Index {
        /// Input media file to index
        #[arg(short, long)]
        input: PathBuf,

        /// Index directory (created if not present)
        #[arg(short = 'd', long)]
        index_dir: PathBuf,

        /// Title metadata override
        #[arg(long)]
        title: Option<String>,

        /// Description metadata override
        #[arg(long)]
        description: Option<String>,
    },
}

/// Entry point called from `main.rs`.
pub async fn run_search(command: SearchCommand, json_output: bool) -> Result<()> {
    match command {
        SearchCommand::Text {
            query,
            index,
            limit,
            offset,
        } => cmd_text(&query, &index, limit, offset, json_output),

        SearchCommand::Similar {
            input,
            index,
            limit,
            threshold,
        } => cmd_similar(&input, &index, limit, threshold, json_output),

        SearchCommand::Fingerprint {
            input,
            output_format,
            compare,
        } => cmd_fingerprint(&input, &output_format, compare.as_deref(), json_output),

        SearchCommand::Index {
            input,
            index_dir,
            title,
            description,
        } => cmd_index_file(
            &input,
            &index_dir,
            title.as_deref(),
            description.as_deref(),
            json_output,
        ),
    }
}

// ---------------------------------------------------------------------------
// Subcommand implementations
// ---------------------------------------------------------------------------

fn cmd_text(
    query: &str,
    index_path: &PathBuf,
    limit: usize,
    offset: usize,
    json_output: bool,
) -> Result<()> {
    let _ = (query, index_path, limit, offset, json_output);
    anyhow::bail!(
        "Text search requires the `search-engine` feature. Rebuild oximedia-cli with --features search-engine."
    );
    #[allow(unreachable_code)]
    Ok(())
}

fn cmd_similar(
    input: &PathBuf,
    index_path: &PathBuf,
    limit: usize,
    threshold: f32,
    json_output: bool,
) -> Result<()> {
    let _ = (input, index_path, limit, threshold, json_output);
    anyhow::bail!(
        "Visual similarity search requires the `search-engine` feature. \
         Rebuild oximedia-cli with --features search-engine."
    );
    #[allow(unreachable_code)]
    Ok(())
}

fn cmd_fingerprint(
    input: &PathBuf,
    output_format: &str,
    compare: Option<&std::path::Path>,
    json_output: bool,
) -> Result<()> {
    use oximedia_forensics::fingerprint::FingerprintMatcher;

    if !input.exists() {
        anyhow::bail!("Input file not found: {}", input.display());
    }

    // Compute fingerprint from file data
    let data = std::fs::read(input)
        .with_context(|| format!("Failed to read file: {}", input.display()))?;

    let fp = compute_file_fingerprint(&data);

    let use_json = json_output || output_format.to_lowercase() == "json";

    // Optional comparison
    let comparison = if let Some(cmp_path) = compare {
        if !cmp_path.exists() {
            anyhow::bail!("Compare file not found: {}", cmp_path.display());
        }
        let cmp_data = std::fs::read(cmp_path)
            .with_context(|| format!("Failed to read compare file: {}", cmp_path.display()))?;
        let cmp_fp = compute_file_fingerprint(&cmp_data);
        let score = FingerprintMatcher::match_score(&fp, &cmp_fp);
        let is_match = FingerprintMatcher::is_match(&fp, &cmp_fp, 0.85);
        Some((cmp_path.to_path_buf(), cmp_fp, score, is_match))
    } else {
        None
    };

    if use_json {
        let mut obj = serde_json::json!({
            "file": input.to_string_lossy(),
            "perceptual_hash": format!("{:016x}", fp.perceptual_hash),
            "audio_fingerprint": format!("{:016x}", fp.audio_fingerprint),
            "metadata_hash": fp.metadata_hash,
        });

        if let Some((cmp_path, cmp_fp, score, is_match)) = &comparison {
            obj["comparison"] = serde_json::json!({
                "compare_file": cmp_path.to_string_lossy(),
                "compare_perceptual_hash": format!("{:016x}", cmp_fp.perceptual_hash),
                "compare_audio_fingerprint": format!("{:016x}", cmp_fp.audio_fingerprint),
                "compare_metadata_hash": cmp_fp.metadata_hash,
                "similarity_score": score,
                "is_match": is_match,
            });
        }

        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "Content Fingerprint".green().bold());
    println!("  File:         {}", input.display().to_string().cyan());
    println!("  Perceptual:   {:016x}", fp.perceptual_hash);
    println!("  Audio:        {:016x}", fp.audio_fingerprint);
    println!("  Metadata:     {}", fp.metadata_hash);

    if let Some((cmp_path, cmp_fp, score, is_match)) = &comparison {
        println!();
        println!("{}", "Comparison".cyan().bold());
        println!("  Compare:      {}", cmp_path.display());
        println!("  Perceptual:   {:016x}", cmp_fp.perceptual_hash);
        println!("  Score:        {:.1}%", score * 100.0);
        if *is_match {
            println!(
                "  Match:        {}",
                "YES — files appear similar".green().bold()
            );
        } else {
            println!("  Match:        {}", "NO — files are different".red());
        }
    }

    Ok(())
}

fn cmd_index_file(
    input: &PathBuf,
    index_dir: &PathBuf,
    title: Option<&str>,
    description: Option<&str>,
    json_output: bool,
) -> Result<()> {
    let _ = (input, index_dir, title, description, json_output);
    anyhow::bail!(
        "Search index building requires the `search-engine` feature. \
         Rebuild oximedia-cli with --features search-engine."
    );
    // Original body preserved as reference comment when feature is available:
    // see oximedia-search crate's `SearchEngine`, `IndexDocument`, `VisualFeatures`.
    // This function can be reinstated by feature-gating at the call site when
    // the `search-engine` feature is active.
    #[allow(unreachable_code)]
    Ok(())
}

// ---------------------------------------------------------------------------
// Fingerprint computation helper
// ---------------------------------------------------------------------------

/// Compute a `Fingerprint` from raw file bytes.
fn compute_file_fingerprint(data: &[u8]) -> oximedia_forensics::fingerprint::Fingerprint {
    use oximedia_forensics::fingerprint::Fingerprint;

    // Perceptual hash: treat raw bytes as luma data approximation
    let phash = Fingerprint::compute_perceptual_hash(data, data.len().max(1), 1);

    // Audio fingerprint: reinterpret bytes as i16 samples
    let samples: Vec<i16> = data
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let ahash = Fingerprint::compute_audio_fingerprint(&samples);

    // Metadata hash: first 256 bytes as text proxy
    let meta_str = String::from_utf8_lossy(&data[..data.len().min(256)]);
    let mhash = Fingerprint::compute_metadata_hash(&meta_str);

    Fingerprint::new(phash, ahash, mhash)
}
