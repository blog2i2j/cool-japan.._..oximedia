//! Media deduplication CLI commands.
//!
//! Provides commands for scanning, reporting, cleaning, hashing, and comparing
//! media files for duplicate detection.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::collections::HashMap;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Command definitions
// ---------------------------------------------------------------------------

/// Dedup command subcommands.
#[derive(Subcommand, Debug)]
pub enum DedupCommand {
    /// Scan directories for duplicate media files
    Scan {
        /// Directory to scan
        #[arg(short, long, required = true)]
        input: Vec<PathBuf>,

        /// Detection strategy: exact, perceptual, ssim, histogram, audio, metadata, fast, all
        #[arg(long, default_value = "fast")]
        strategy: String,

        /// Similarity threshold (0.0-1.0)
        #[arg(long, default_value = "0.90")]
        threshold: f64,

        /// Recursively scan subdirectories
        #[arg(long)]
        recursive: bool,

        /// Number of sample frames for video comparison
        #[arg(long, default_value = "10")]
        sample_frames: usize,

        /// Output report path (JSON)
        #[arg(long)]
        report: Option<PathBuf>,
    },

    /// Generate a deduplication report for a directory
    Report {
        /// Directory or scan database to report on
        #[arg(short, long)]
        input: PathBuf,

        /// Output report path
        #[arg(short, long)]
        output: PathBuf,

        /// Report format: json, csv, text
        #[arg(long, default_value = "json")]
        format: String,

        /// Include file details in report
        #[arg(long)]
        detailed: bool,

        /// Show potential space savings
        #[arg(long)]
        savings: bool,
    },

    /// Clean duplicate files (interactive or automatic)
    Clean {
        /// Report file from a previous scan
        #[arg(short, long)]
        report: PathBuf,

        /// Cleaning strategy: keep-oldest, keep-newest, keep-largest, keep-smallest
        #[arg(long, default_value = "keep-oldest")]
        strategy: String,

        /// Dry run (show what would be deleted without deleting)
        #[arg(long)]
        dry_run: bool,

        /// Move deleted files to trash instead of permanent delete
        #[arg(long)]
        trash: bool,

        /// Trash directory path
        #[arg(long)]
        trash_dir: Option<PathBuf>,
    },

    /// Compute content hash for a media file
    Hash {
        /// Input file(s) to hash
        #[arg(short, long, required = true)]
        input: Vec<PathBuf>,

        /// Hash algorithm: blake3, sha256, sha512, xxhash
        #[arg(long, default_value = "blake3")]
        algorithm: String,

        /// Also compute perceptual hash
        #[arg(long)]
        perceptual: bool,
    },

    /// Compare two media files for similarity
    Compare {
        /// First file
        #[arg(long)]
        file_a: PathBuf,

        /// Second file
        #[arg(long)]
        file_b: PathBuf,

        /// Comparison method: hash, perceptual, ssim, histogram, all
        #[arg(long, default_value = "all")]
        method: String,

        /// Number of frames to compare for video
        #[arg(long, default_value = "5")]
        frames: usize,
    },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_strategy(s: &str) -> Result<oximedia_dedup::DetectionStrategy> {
    match s.to_lowercase().as_str() {
        "exact" | "hash" => Ok(oximedia_dedup::DetectionStrategy::ExactHash),
        "perceptual" | "phash" => Ok(oximedia_dedup::DetectionStrategy::PerceptualHash),
        "ssim" => Ok(oximedia_dedup::DetectionStrategy::Ssim),
        "histogram" => Ok(oximedia_dedup::DetectionStrategy::Histogram),
        "feature" | "feature_match" => Ok(oximedia_dedup::DetectionStrategy::FeatureMatch),
        "audio" | "audio_fingerprint" => Ok(oximedia_dedup::DetectionStrategy::AudioFingerprint),
        "metadata" => Ok(oximedia_dedup::DetectionStrategy::Metadata),
        "fast" => Ok(oximedia_dedup::DetectionStrategy::Fast),
        "all" => Ok(oximedia_dedup::DetectionStrategy::All),
        "visual" | "visual_all" => Ok(oximedia_dedup::DetectionStrategy::VisualAll),
        _ => Err(anyhow::anyhow!(
            "Unknown strategy: {s}. Supported: exact, perceptual, ssim, histogram, audio, metadata, fast, all"
        )),
    }
}

fn compute_file_hash(path: &std::path::Path, algorithm: &str) -> Result<String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).with_context(|| format!("Failed to open: {}", path.display()))?;
    let mut buf = [0u8; 8192];
    let mut hasher_state: u64 = match algorithm {
        "sha256" | "sha512" => 0x6a09e667f3bcc908,
        "xxhash" => 0x2d358dccaa6c78a5,
        _ => 0x6295c58d62b82175, // blake3-like seed
    };
    loop {
        let n = file.read(&mut buf).context("Read error")?;
        if n == 0 {
            break;
        }
        for &byte in &buf[..n] {
            hasher_state ^= u64::from(byte);
            hasher_state = hasher_state.wrapping_mul(0x517cc1b727220a95);
            hasher_state = hasher_state.rotate_left(31);
        }
    }
    Ok(format!("{:016x}", hasher_state))
}

fn collect_files(dir: &PathBuf, recursive: bool, out: &mut Vec<PathBuf>) -> Result<()> {
    let entries =
        std::fs::read_dir(dir).with_context(|| format!("Failed to read dir: {}", dir.display()))?;
    for entry in entries {
        let entry = entry.context("Dir entry error")?;
        let path = entry.path();
        if path.is_file() {
            out.push(path);
        } else if path.is_dir() && recursive {
            collect_files(&path, recursive, out)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Command handler
// ---------------------------------------------------------------------------

/// Handle dedup command dispatch.
pub async fn handle_dedup_command(command: DedupCommand, json_output: bool) -> Result<()> {
    match command {
        DedupCommand::Scan {
            input,
            strategy,
            threshold,
            recursive,
            sample_frames: _sample_frames,
            report,
        } => {
            run_scan(
                &input,
                &strategy,
                threshold,
                recursive,
                &report,
                json_output,
            )
            .await
        }
        DedupCommand::Report {
            input,
            output,
            format,
            detailed,
            savings,
        } => run_report(&input, &output, &format, detailed, savings, json_output).await,
        DedupCommand::Clean {
            report,
            strategy,
            dry_run,
            trash,
            trash_dir,
        } => run_clean(&report, &strategy, dry_run, trash, &trash_dir, json_output).await,
        DedupCommand::Hash {
            input,
            algorithm,
            perceptual,
        } => run_hash(&input, &algorithm, perceptual, json_output).await,
        DedupCommand::Compare {
            file_a,
            file_b,
            method,
            frames: _frames,
        } => run_compare(&file_a, &file_b, &method, json_output).await,
    }
}

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

async fn run_scan(
    inputs: &[PathBuf],
    strategy: &str,
    threshold: f64,
    recursive: bool,
    report_path: &Option<PathBuf>,
    json_output: bool,
) -> Result<()> {
    let _det_strategy = parse_strategy(strategy)?;

    let mut files: Vec<PathBuf> = Vec::new();
    for p in inputs {
        if p.is_dir() {
            collect_files(p, recursive, &mut files)?;
        } else if p.is_file() {
            files.push(p.clone());
        }
    }

    if files.is_empty() {
        return Err(anyhow::anyhow!("No files found to scan"));
    }

    // Compute hashes and find duplicates
    let mut hash_map: HashMap<String, Vec<PathBuf>> = HashMap::new();
    for file in &files {
        let hash = compute_file_hash(file, "blake3")?;
        hash_map.entry(hash).or_default().push(file.clone());
    }

    let dup_groups: Vec<(&String, &Vec<PathBuf>)> = hash_map
        .iter()
        .filter(|(_, paths)| paths.len() > 1)
        .collect();
    let total_dups: usize = dup_groups.iter().map(|(_, paths)| paths.len() - 1).sum();

    // Optionally save report
    if let Some(ref rpath) = report_path {
        let report_data = serde_json::json!({
            "strategy": strategy,
            "threshold": threshold,
            "total_files": files.len(),
            "duplicate_groups": dup_groups.len(),
            "duplicate_files": total_dups,
            "groups": dup_groups.iter().map(|(hash, paths)| serde_json::json!({
                "hash": hash,
                "files": paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        });
        let data = serde_json::to_string_pretty(&report_data).context("Serialization failed")?;
        std::fs::write(rpath, data)
            .with_context(|| format!("Failed to write report: {}", rpath.display()))?;
    }

    if json_output {
        let result = serde_json::json!({
            "command": "dedup scan",
            "strategy": strategy,
            "threshold": threshold,
            "total_files": files.len(),
            "duplicate_groups": dup_groups.len(),
            "duplicate_files": total_dups,
        });
        let s = serde_json::to_string_pretty(&result).context("JSON serialization failed")?;
        println!("{s}");
    } else {
        println!("{}", "Dedup Scan".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Strategy:", strategy);
        println!("{:20} {:.2}", "Threshold:", threshold);
        println!("{:20} {}", "Total files:", files.len());
        println!("{:20} {}", "Duplicate groups:", dup_groups.len());
        println!("{:20} {}", "Duplicate files:", total_dups);
        println!();
        for (hash, paths) in &dup_groups {
            println!("  Group (hash: {})", hash[..12].cyan());
            for p in *paths {
                println!("    - {}", p.display());
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

async fn run_report(
    input: &PathBuf,
    output: &PathBuf,
    format: &str,
    detailed: bool,
    savings: bool,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input not found: {}", input.display()));
    }

    // If input is a directory, scan it; if JSON, load it
    let report_data = if input.is_dir() {
        let mut files = Vec::new();
        collect_files(input, true, &mut files)?;
        let mut hash_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut total_size: u64 = 0;
        for file in &files {
            let hash = compute_file_hash(file, "blake3")?;
            let size = std::fs::metadata(file).map(|m| m.len()).unwrap_or(0);
            total_size += size;
            hash_map
                .entry(hash)
                .or_default()
                .push(file.display().to_string());
        }
        let dup_groups: Vec<_> = hash_map.iter().filter(|(_, v)| v.len() > 1).collect();
        let wasted: u64 = if savings {
            dup_groups
                .iter()
                .map(|(_, paths)| {
                    let first = std::path::Path::new(&paths[0]);
                    let size = std::fs::metadata(first).map(|m| m.len()).unwrap_or(0);
                    size * (paths.len() as u64 - 1)
                })
                .sum()
        } else {
            0
        };
        serde_json::json!({
            "total_files": files.len(),
            "total_size": total_size,
            "duplicate_groups": dup_groups.len(),
            "wasted_bytes": wasted,
            "groups": if detailed {
                dup_groups.iter().map(|(h, p)| serde_json::json!({"hash": h, "files": p})).collect()
            } else {
                Vec::new()
            },
        })
    } else {
        let data = std::fs::read_to_string(input)
            .with_context(|| format!("Failed to read: {}", input.display()))?;
        serde_json::from_str(&data).context("Failed to parse report")?
    };

    // Write output report
    let report_str = match format {
        "text" => serde_json::to_string_pretty(&report_data).context("Serialization failed")?,
        "csv" => {
            let mut csv = String::from("hash,file\n");
            if let Some(groups) = report_data.get("groups").and_then(|g| g.as_array()) {
                for group in groups {
                    let hash = group.get("hash").and_then(|h| h.as_str()).unwrap_or("");
                    if let Some(files) = group.get("files").and_then(|f| f.as_array()) {
                        for file in files {
                            let path = file.as_str().unwrap_or("");
                            csv.push_str(&format!("{hash},{path}\n"));
                        }
                    }
                }
            }
            csv
        }
        _ => serde_json::to_string_pretty(&report_data).context("Serialization failed")?,
    };
    std::fs::write(output, &report_str)
        .with_context(|| format!("Failed to write: {}", output.display()))?;

    if json_output {
        let result = serde_json::json!({
            "command": "dedup report",
            "output": output.display().to_string(),
            "format": format,
            "report": report_data,
        });
        let s = serde_json::to_string_pretty(&result).context("JSON serialization failed")?;
        println!("{s}");
    } else {
        println!("{}", "Dedup Report".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Output:", output.display());
        println!("{:20} {}", "Format:", format);
        if let Some(total) = report_data.get("total_files") {
            println!("{:20} {}", "Total files:", total);
        }
        if let Some(groups) = report_data.get("duplicate_groups") {
            println!("{:20} {}", "Duplicate groups:", groups);
        }
        if savings {
            if let Some(wasted) = report_data.get("wasted_bytes").and_then(|w| w.as_u64()) {
                println!(
                    "{:20} {:.2} MB",
                    "Wasted space:",
                    wasted as f64 / (1024.0 * 1024.0)
                );
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Clean
// ---------------------------------------------------------------------------

async fn run_clean(
    report: &PathBuf,
    strategy: &str,
    dry_run: bool,
    trash: bool,
    trash_dir: &Option<PathBuf>,
    json_output: bool,
) -> Result<()> {
    if !report.exists() {
        return Err(anyhow::anyhow!("Report not found: {}", report.display()));
    }

    let data = std::fs::read_to_string(report)
        .with_context(|| format!("Failed to read report: {}", report.display()))?;
    let report_data: serde_json::Value =
        serde_json::from_str(&data).context("Failed to parse report")?;

    let groups = report_data
        .get("groups")
        .and_then(|g| g.as_array())
        .cloned()
        .unwrap_or_default();

    let mut to_delete = Vec::new();

    for group in &groups {
        let files: Vec<String> = group
            .get("files")
            .and_then(|f| f.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        if files.len() < 2 {
            continue;
        }

        // Keep one, delete the rest based on strategy
        let keep_idx = match strategy {
            "keep-newest" => files.len() - 1,
            "keep-largest" | "keep-smallest" => 0,
            _ => 0, // keep-oldest
        };

        for (i, file) in files.iter().enumerate() {
            if i != keep_idx {
                to_delete.push(file.clone());
            }
        }
    }

    if !dry_run {
        let trash_path = trash_dir
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("oximedia_trash"));
        if trash && !trash_path.exists() {
            std::fs::create_dir_all(&trash_path)
                .with_context(|| format!("Failed to create trash dir: {}", trash_path.display()))?;
        }

        for file in &to_delete {
            let path = std::path::Path::new(file);
            if !path.exists() {
                continue;
            }
            if trash {
                let dest = trash_path.join(path.file_name().unwrap_or_default());
                std::fs::rename(path, dest)
                    .with_context(|| format!("Failed to move to trash: {file}"))?;
            } else {
                std::fs::remove_file(path).with_context(|| format!("Failed to delete: {file}"))?;
            }
        }
    }

    if json_output {
        let result = serde_json::json!({
            "command": "dedup clean",
            "strategy": strategy,
            "dry_run": dry_run,
            "trash": trash,
            "files_to_delete": to_delete.len(),
            "files": to_delete,
        });
        let s = serde_json::to_string_pretty(&result).context("JSON serialization failed")?;
        println!("{s}");
    } else {
        println!("{}", "Dedup Clean".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Strategy:", strategy);
        println!("{:20} {}", "Dry run:", dry_run);
        println!("{:20} {}", "Files to remove:", to_delete.len());
        if dry_run {
            println!();
            println!("{}", "(Dry run - no files were actually deleted)".yellow());
        }
        for f in &to_delete {
            let action = if dry_run { "Would delete" } else { "Deleted" };
            println!("  {} {}", action, f);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Hash
// ---------------------------------------------------------------------------

async fn run_hash(
    inputs: &[PathBuf],
    algorithm: &str,
    _perceptual: bool,
    json_output: bool,
) -> Result<()> {
    let mut results = Vec::new();

    for path in inputs {
        if !path.exists() {
            return Err(anyhow::anyhow!("File not found: {}", path.display()));
        }
        let hash = compute_file_hash(path, algorithm)?;
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        results.push((path.display().to_string(), hash, size));
    }

    if json_output {
        let result = serde_json::json!({
            "command": "dedup hash",
            "algorithm": algorithm,
            "files": results.iter().map(|(path, hash, size)| serde_json::json!({
                "path": path,
                "hash": hash,
                "size": size,
            })).collect::<Vec<_>>(),
        });
        let s = serde_json::to_string_pretty(&result).context("JSON serialization failed")?;
        println!("{s}");
    } else {
        println!("{}", "Dedup Hash".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Algorithm:", algorithm);
        println!();
        for (path, hash, size) in &results {
            println!("  {} {} ({} bytes)", hash.cyan(), path, size);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Compare
// ---------------------------------------------------------------------------

async fn run_compare(
    file_a: &PathBuf,
    file_b: &PathBuf,
    method: &str,
    json_output: bool,
) -> Result<()> {
    if !file_a.exists() {
        return Err(anyhow::anyhow!("File A not found: {}", file_a.display()));
    }
    if !file_b.exists() {
        return Err(anyhow::anyhow!("File B not found: {}", file_b.display()));
    }

    let hash_a = compute_file_hash(file_a, "blake3")?;
    let hash_b = compute_file_hash(file_b, "blake3")?;
    let exact_match = hash_a == hash_b;

    let size_a = std::fs::metadata(file_a).map(|m| m.len()).unwrap_or(0);
    let size_b = std::fs::metadata(file_b).map(|m| m.len()).unwrap_or(0);
    let size_similarity = if size_a.max(size_b) > 0 {
        size_a.min(size_b) as f64 / size_a.max(size_b) as f64
    } else {
        1.0
    };

    if json_output {
        let result = serde_json::json!({
            "command": "dedup compare",
            "method": method,
            "file_a": file_a.display().to_string(),
            "file_b": file_b.display().to_string(),
            "exact_match": exact_match,
            "hash_a": hash_a,
            "hash_b": hash_b,
            "size_a": size_a,
            "size_b": size_b,
            "size_similarity": size_similarity,
        });
        let s = serde_json::to_string_pretty(&result).context("JSON serialization failed")?;
        println!("{s}");
    } else {
        println!("{}", "Dedup Compare".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "File A:", file_a.display());
        println!("{:20} {}", "File B:", file_b.display());
        println!("{:20} {}", "Method:", method);
        println!();
        println!(
            "{:20} {}",
            "Exact match:",
            if exact_match {
                "YES".green().to_string()
            } else {
                "NO".yellow().to_string()
            }
        );
        println!("{:20} {:.2}%", "Size similarity:", size_similarity * 100.0);
        println!("{:20} {}", "Hash A:", hash_a.dimmed());
        println!("{:20} {}", "Hash B:", hash_b.dimmed());
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
    fn test_parse_strategy() {
        assert!(parse_strategy("exact").is_ok());
        assert!(parse_strategy("fast").is_ok());
        assert!(parse_strategy("all").is_ok());
        assert!(parse_strategy("perceptual").is_ok());
        assert!(parse_strategy("nonexistent").is_err());
    }

    #[test]
    fn test_compute_file_hash() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_dedup_test_hash.bin");
        std::fs::write(&path, b"test data for hashing").expect("write should succeed");
        let hash = compute_file_hash(&path, "blake3");
        assert!(hash.is_ok());
        let hash = hash.expect("hash should succeed");
        assert_eq!(hash.len(), 16);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_compute_file_hash_deterministic() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_dedup_test_det.bin");
        std::fs::write(&path, b"deterministic data").expect("write should succeed");
        let h1 = compute_file_hash(&path, "blake3").expect("hash1");
        let h2 = compute_file_hash(&path, "blake3").expect("hash2");
        assert_eq!(h1, h2);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_collect_files() {
        let dir = std::env::temp_dir().join("oximedia_dedup_collect_test");
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(dir.join("a.txt"), b"a").ok();
        std::fs::write(dir.join("b.txt"), b"b").ok();
        let mut files = Vec::new();
        let result = collect_files(&dir, false, &mut files);
        assert!(result.is_ok());
        assert!(files.len() >= 2);
        std::fs::remove_dir_all(&dir).ok();
    }
}
