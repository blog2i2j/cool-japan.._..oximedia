//! Proxy media file generation and management CLI commands.
//!
//! Provides commands for generating, listing, linking, inspecting, and cleaning
//! proxy media files for offline/online editing workflows.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Command definitions
// ---------------------------------------------------------------------------

/// Proxy command subcommands.
#[derive(Subcommand, Debug)]
pub enum ProxyCommand {
    /// Generate proxy media files from originals
    Generate {
        /// Input file(s) to generate proxies for
        #[arg(short, long, required = true)]
        input: Vec<PathBuf>,

        /// Output directory for proxy files
        #[arg(short, long)]
        output: PathBuf,

        /// Resolution preset: quarter, half, full
        #[arg(long, default_value = "quarter")]
        resolution: String,

        /// Quality preset: low, medium, high
        #[arg(long, default_value = "medium")]
        quality: String,

        /// Proxy database path (JSON file)
        #[arg(long)]
        db: PathBuf,

        /// Codec for proxy: vp9, av1
        #[arg(long, default_value = "vp9")]
        codec: String,
    },

    /// List all proxies in the database
    List {
        /// Proxy database path (JSON file)
        #[arg(long)]
        db: PathBuf,

        /// Filter by original file path pattern
        #[arg(long)]
        filter: Option<String>,

        /// Show detailed info for each proxy
        #[arg(long)]
        detailed: bool,

        /// Sort by: name, size, date
        #[arg(long, default_value = "name")]
        sort: String,
    },

    /// Link a proxy file to its original
    Link {
        /// Proxy file path
        #[arg(short, long)]
        proxy: PathBuf,

        /// Original file path
        #[arg(short, long)]
        original: PathBuf,

        /// Proxy database path (JSON file)
        #[arg(long)]
        db: PathBuf,
    },

    /// Show info about a proxy file or link
    Info {
        /// Proxy or original file path to look up
        #[arg(short, long)]
        path: PathBuf,

        /// Proxy database path (JSON file)
        #[arg(long)]
        db: PathBuf,
    },

    /// Clean up stale or orphaned proxy files
    Clean {
        /// Proxy database path (JSON file)
        #[arg(long)]
        db: PathBuf,

        /// Output directory where proxies are stored
        #[arg(short, long)]
        output: PathBuf,

        /// Dry run: show what would be cleaned without deleting
        #[arg(long)]
        dry_run: bool,

        /// Remove proxies whose originals are missing
        #[arg(long)]
        remove_orphans: bool,
    },
}

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProxyRecord {
    id: String,
    original_path: String,
    proxy_path: String,
    resolution: String,
    quality: String,
    codec: String,
    original_size_bytes: u64,
    proxy_size_bytes: u64,
    created_at: String,
    checksum: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct ProxyDb {
    version: u32,
    proxies: Vec<ProxyRecord>,
}

// ---------------------------------------------------------------------------
// Persistence helpers
// ---------------------------------------------------------------------------

fn load_db(path: &PathBuf) -> Result<ProxyDb> {
    if !path.exists() {
        return Ok(ProxyDb {
            version: 1,
            ..ProxyDb::default()
        });
    }
    let data = std::fs::read_to_string(path).context("Failed to read proxy database")?;
    let db: ProxyDb = serde_json::from_str(&data).context("Failed to parse proxy database")?;
    Ok(db)
}

fn save_db(path: &PathBuf, db: &ProxyDb) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).context("Failed to create database directory")?;
        }
    }
    let data = serde_json::to_string_pretty(db).context("Failed to serialize proxy database")?;
    std::fs::write(path, data).context("Failed to write proxy database")?;
    Ok(())
}

fn generate_id() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("proxy-{:016x}", now.as_nanos())
}

fn now_timestamp() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", dur.as_secs())
}

fn compute_checksum(path: &std::path::Path) -> Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).context("Failed to open file for checksum")?;
    let mut hasher_state: u64 = 0xcbf29ce484222325;
    let mut buf = [0u8; 8192];
    loop {
        let n = file
            .read(&mut buf)
            .context("Failed to read file for checksum")?;
        if n == 0 {
            break;
        }
        for &byte in &buf[..n] {
            hasher_state ^= u64::from(byte);
            hasher_state = hasher_state.wrapping_mul(0x100000001b3);
        }
    }
    Ok(format!("{:016x}", hasher_state))
}

fn resolution_scale(res: &str) -> f64 {
    match res {
        "quarter" => 0.25,
        "half" => 0.5,
        "full" => 1.0,
        _ => 0.25,
    }
}

// ---------------------------------------------------------------------------
// Command handler
// ---------------------------------------------------------------------------

/// Handle proxy command dispatch.
pub async fn handle_proxy_command(command: ProxyCommand, json_output: bool) -> Result<()> {
    match command {
        ProxyCommand::Generate {
            input,
            output,
            resolution,
            quality,
            db,
            codec,
        } => {
            run_generate(
                &input,
                &output,
                &resolution,
                &quality,
                &db,
                &codec,
                json_output,
            )
            .await
        }
        ProxyCommand::List {
            db,
            filter,
            detailed,
            sort,
        } => run_list(&db, &filter, detailed, &sort, json_output).await,
        ProxyCommand::Link {
            proxy,
            original,
            db,
        } => run_link(&proxy, &original, &db, json_output).await,
        ProxyCommand::Info { path, db } => run_info(&path, &db, json_output).await,
        ProxyCommand::Clean {
            db,
            output,
            dry_run,
            remove_orphans,
        } => run_clean(&db, &output, dry_run, remove_orphans, json_output).await,
    }
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

async fn run_generate(
    inputs: &[PathBuf],
    output: &PathBuf,
    resolution: &str,
    quality: &str,
    db_path: &PathBuf,
    codec: &str,
    json_output: bool,
) -> Result<()> {
    let mut db = load_db(db_path)?;

    if !output.exists() {
        std::fs::create_dir_all(output).context("Failed to create output directory")?;
    }

    let scale = resolution_scale(resolution);
    let mut generated = Vec::new();

    for input_path in inputs {
        if !input_path.is_file() {
            if !json_output {
                println!(
                    "  {} {} (not a file)",
                    "Skip:".yellow(),
                    input_path.display()
                );
            }
            continue;
        }

        let meta = std::fs::metadata(input_path)
            .with_context(|| format!("Failed to read metadata: {}", input_path.display()))?;

        let filename = input_path.file_stem().unwrap_or_default().to_string_lossy();
        let proxy_name = format!("{filename}_proxy_{resolution}.webm");
        let proxy_path = output.join(&proxy_name);

        // Simulate proxy generation: estimate size based on scale and quality
        let quality_factor = match quality {
            "low" => 0.1,
            "medium" => 0.25,
            "high" => 0.5,
            _ => 0.25,
        };
        let estimated_size = (meta.len() as f64 * scale * scale * quality_factor) as u64;

        // Write a placeholder proxy file for the record
        let proxy_content = format!(
            "PROXY:original={},resolution={},quality={},codec={},scale={scale}",
            input_path.display(),
            resolution,
            quality,
            codec
        );
        std::fs::write(&proxy_path, &proxy_content).context("Failed to write proxy file")?;

        let checksum = compute_checksum(input_path)?;

        let record = ProxyRecord {
            id: generate_id(),
            original_path: input_path.to_string_lossy().to_string(),
            proxy_path: proxy_path.to_string_lossy().to_string(),
            resolution: resolution.to_string(),
            quality: quality.to_string(),
            codec: codec.to_string(),
            original_size_bytes: meta.len(),
            proxy_size_bytes: estimated_size,
            created_at: now_timestamp(),
            checksum,
        };

        generated.push(record.clone());
        db.proxies.push(record);
    }

    save_db(db_path, &db)?;

    if json_output {
        let result = serde_json::json!({
            "command": "proxy_generate",
            "output": output.display().to_string(),
            "generated_count": generated.len(),
            "resolution": resolution,
            "quality": quality,
            "codec": codec,
            "proxies": generated.iter().map(|p| serde_json::json!({
                "id": p.id,
                "original": p.original_path,
                "proxy": p.proxy_path,
                "estimated_size": p.proxy_size_bytes,
            })).collect::<Vec<_>>(),
        });
        let s = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
        println!("{s}");
    } else {
        println!("{}", "Proxy Generation".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Output:", output.display());
        println!("{:20} {}", "Resolution:", resolution);
        println!("{:20} {}", "Quality:", quality);
        println!("{:20} {}", "Codec:", codec);
        println!("{:20} {}", "Generated:", generated.len());
        println!();
        for p in &generated {
            println!("  {} {} -> {}", "+".green(), p.original_path, p.proxy_path);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

async fn run_list(
    db_path: &PathBuf,
    filter: &Option<String>,
    detailed: bool,
    sort: &str,
    json_output: bool,
) -> Result<()> {
    let db = load_db(db_path)?;

    let mut proxies: Vec<&ProxyRecord> = db
        .proxies
        .iter()
        .filter(|p| {
            filter.as_ref().map_or(true, |f| {
                let fl = f.to_lowercase();
                p.original_path.to_lowercase().contains(&fl)
                    || p.proxy_path.to_lowercase().contains(&fl)
            })
        })
        .collect();

    match sort {
        "name" => proxies.sort_by(|a, b| a.original_path.cmp(&b.original_path)),
        "size" => proxies.sort_by(|a, b| b.original_size_bytes.cmp(&a.original_size_bytes)),
        "date" => proxies.sort_by(|a, b| b.created_at.cmp(&a.created_at)),
        _ => {}
    }

    if json_output {
        let result = serde_json::json!({
            "command": "proxy_list",
            "total": proxies.len(),
            "proxies": proxies.iter().map(|p| serde_json::json!({
                "id": p.id,
                "original": p.original_path,
                "proxy": p.proxy_path,
                "resolution": p.resolution,
                "quality": p.quality,
                "codec": p.codec,
            })).collect::<Vec<_>>(),
        });
        let s = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
        println!("{s}");
    } else {
        println!("{}", "Proxy List".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Total proxies:", proxies.len());
        println!();
        for p in &proxies {
            println!(
                "  {} [{}] {} -> {}",
                ">".cyan(),
                p.resolution,
                p.original_path,
                p.proxy_path
            );
            if detailed {
                println!(
                    "    Quality: {}, Codec: {}, Original: {} bytes",
                    p.quality, p.codec, p.original_size_bytes
                );
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Link
// ---------------------------------------------------------------------------

async fn run_link(
    proxy: &PathBuf,
    original: &PathBuf,
    db_path: &PathBuf,
    json_output: bool,
) -> Result<()> {
    let mut db = load_db(db_path)?;

    if !proxy.exists() {
        return Err(anyhow::anyhow!("Proxy file not found: {}", proxy.display()));
    }
    if !original.exists() {
        return Err(anyhow::anyhow!(
            "Original file not found: {}",
            original.display()
        ));
    }

    let orig_meta = std::fs::metadata(original).context("Failed to read original metadata")?;
    let proxy_meta = std::fs::metadata(proxy).context("Failed to read proxy metadata")?;
    let checksum = compute_checksum(original)?;

    let record = ProxyRecord {
        id: generate_id(),
        original_path: original.to_string_lossy().to_string(),
        proxy_path: proxy.to_string_lossy().to_string(),
        resolution: "unknown".to_string(),
        quality: "unknown".to_string(),
        codec: "unknown".to_string(),
        original_size_bytes: orig_meta.len(),
        proxy_size_bytes: proxy_meta.len(),
        created_at: now_timestamp(),
        checksum,
    };

    db.proxies.push(record.clone());
    save_db(db_path, &db)?;

    if json_output {
        let result = serde_json::json!({
            "command": "proxy_link",
            "id": record.id,
            "proxy": record.proxy_path,
            "original": record.original_path,
        });
        let s = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
        println!("{s}");
    } else {
        println!("{}", "Proxy Linked".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "ID:", record.id);
        println!("{:20} {}", "Original:", original.display());
        println!("{:20} {}", "Proxy:", proxy.display());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Info
// ---------------------------------------------------------------------------

async fn run_info(path: &PathBuf, db_path: &PathBuf, json_output: bool) -> Result<()> {
    let db = load_db(db_path)?;
    let path_str = path.to_string_lossy();

    let matches: Vec<&ProxyRecord> = db
        .proxies
        .iter()
        .filter(|p| p.original_path == *path_str || p.proxy_path == *path_str)
        .collect();

    if matches.is_empty() {
        return Err(anyhow::anyhow!(
            "No proxy records found for: {}",
            path.display()
        ));
    }

    if json_output {
        let result = serde_json::json!({
            "command": "proxy_info",
            "path": path_str,
            "records": matches.iter().map(|p| serde_json::json!({
                "id": p.id,
                "original": p.original_path,
                "proxy": p.proxy_path,
                "resolution": p.resolution,
                "quality": p.quality,
                "codec": p.codec,
                "original_size": p.original_size_bytes,
                "proxy_size": p.proxy_size_bytes,
                "checksum": p.checksum,
            })).collect::<Vec<_>>(),
        });
        let s = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
        println!("{s}");
    } else {
        println!("{}", "Proxy Info".green().bold());
        println!("{}", "=".repeat(60));
        for p in &matches {
            println!("{:20} {}", "ID:", p.id);
            println!("{:20} {}", "Original:", p.original_path);
            println!("{:20} {}", "Proxy:", p.proxy_path);
            println!("{:20} {}", "Resolution:", p.resolution);
            println!("{:20} {}", "Quality:", p.quality);
            println!("{:20} {}", "Codec:", p.codec);
            println!("{:20} {} bytes", "Original size:", p.original_size_bytes);
            println!("{:20} {} bytes", "Proxy size:", p.proxy_size_bytes);
            println!();
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Clean
// ---------------------------------------------------------------------------

async fn run_clean(
    db_path: &PathBuf,
    _output: &PathBuf,
    dry_run: bool,
    remove_orphans: bool,
    json_output: bool,
) -> Result<()> {
    let mut db = load_db(db_path)?;

    let mut orphaned = Vec::new();
    let mut stale = Vec::new();

    for proxy in &db.proxies {
        let orig_exists = std::path::Path::new(&proxy.original_path).exists();
        let proxy_exists = std::path::Path::new(&proxy.proxy_path).exists();

        if !orig_exists && remove_orphans {
            orphaned.push(proxy.clone());
        }
        if !proxy_exists {
            stale.push(proxy.clone());
        }
    }

    if !dry_run {
        // Remove orphaned proxy files
        for p in &orphaned {
            let proxy_path = std::path::Path::new(&p.proxy_path);
            if proxy_path.exists() {
                std::fs::remove_file(proxy_path)
                    .with_context(|| format!("Failed to remove: {}", p.proxy_path))?;
            }
        }

        // Remove stale and orphaned records from DB
        let remove_ids: std::collections::HashSet<&str> = orphaned
            .iter()
            .chain(stale.iter())
            .map(|p| p.id.as_str())
            .collect();
        db.proxies.retain(|p| !remove_ids.contains(p.id.as_str()));
        save_db(db_path, &db)?;
    }

    if json_output {
        let result = serde_json::json!({
            "command": "proxy_clean",
            "dry_run": dry_run,
            "orphaned_count": orphaned.len(),
            "stale_count": stale.len(),
            "removed": if dry_run { 0 } else { orphaned.len() + stale.len() },
        });
        let s = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
        println!("{s}");
    } else {
        println!("{}", "Proxy Cleanup".green().bold());
        println!("{}", "=".repeat(60));
        if dry_run {
            println!("{}", "(dry run - no changes made)".yellow());
        }
        println!("{:20} {}", "Orphaned proxies:", orphaned.len());
        println!("{:20} {}", "Stale records:", stale.len());
        for p in &orphaned {
            println!("  {} orphan: {}", "-".red(), p.proxy_path);
        }
        for p in &stale {
            println!("  {} stale: {}", "-".yellow(), p.proxy_path);
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
    fn test_generate_id() {
        let id = generate_id();
        assert!(id.starts_with("proxy-"));
    }

    #[test]
    fn test_resolution_scale() {
        assert!((resolution_scale("quarter") - 0.25).abs() < f64::EPSILON);
        assert!((resolution_scale("half") - 0.5).abs() < f64::EPSILON);
        assert!((resolution_scale("full") - 1.0).abs() < f64::EPSILON);
        assert!((resolution_scale("unknown") - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_proxy_record_serialization() {
        let record = ProxyRecord {
            id: "proxy-001".to_string(),
            original_path: "/tmp/original.mov".to_string(),
            proxy_path: "/tmp/proxy.webm".to_string(),
            resolution: "quarter".to_string(),
            quality: "medium".to_string(),
            codec: "vp9".to_string(),
            original_size_bytes: 1_000_000,
            proxy_size_bytes: 100_000,
            created_at: "12345".to_string(),
            checksum: "abc123".to_string(),
        };
        let json = serde_json::to_string(&record);
        assert!(json.is_ok());
        let parsed: Result<ProxyRecord, _> = serde_json::from_str(&json.expect("should serialize"));
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_db_default() {
        let db = ProxyDb::default();
        assert_eq!(db.version, 0);
        assert!(db.proxies.is_empty());
    }

    #[test]
    fn test_db_roundtrip() {
        let db = ProxyDb {
            version: 1,
            proxies: vec![ProxyRecord {
                id: "proxy-001".to_string(),
                original_path: "/tmp/orig.mov".to_string(),
                proxy_path: "/tmp/proxy.webm".to_string(),
                resolution: "half".to_string(),
                quality: "high".to_string(),
                codec: "av1".to_string(),
                original_size_bytes: 500_000,
                proxy_size_bytes: 50_000,
                created_at: "99999".to_string(),
                checksum: "deadbeef".to_string(),
            }],
        };
        let json = serde_json::to_string_pretty(&db).expect("serialize");
        let parsed: ProxyDb = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.proxies.len(), 1);
        assert_eq!(parsed.proxies[0].codec, "av1");
    }
}
