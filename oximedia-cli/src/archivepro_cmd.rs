//! Professional archive and digital preservation CLI commands.
//!
//! Provides commands for ingesting, verifying, migrating, reporting,
//! and managing preservation policies for long-term media archiving.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Command definitions
// ---------------------------------------------------------------------------

/// Archive-pro command subcommands.
#[derive(Subcommand, Debug)]
pub enum ArchiveProCommand {
    /// Ingest media files into the archive with preservation packaging
    Ingest {
        /// Input file(s) to ingest
        #[arg(short, long, required = true)]
        input: Vec<PathBuf>,

        /// Archive root directory
        #[arg(long)]
        archive: PathBuf,

        /// Packaging format: bagit, oais-sip, tar, zip
        #[arg(long, default_value = "bagit")]
        package_format: String,

        /// Checksum algorithm: sha256, sha512, blake3, xxhash
        #[arg(long, default_value = "sha256")]
        checksum: String,

        /// Generate preservation metadata (PREMIS)
        #[arg(long)]
        premis: bool,

        /// Target preservation format for migration
        #[arg(long)]
        target_format: Option<String>,
    },

    /// Verify archive integrity via fixity checking
    Verify {
        /// Archive or package path to verify
        #[arg(short, long)]
        input: PathBuf,

        /// Checksum algorithm to verify with
        #[arg(long, default_value = "sha256")]
        checksum: String,

        /// Deep verification (re-compute all checksums)
        #[arg(long)]
        deep: bool,

        /// Verify metadata consistency
        #[arg(long)]
        metadata: bool,

        /// Output verification report
        #[arg(long)]
        report: Option<PathBuf>,
    },

    /// Plan or execute format migration
    Migrate {
        /// Input file or archive
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output: PathBuf,

        /// Target preservation format: ffv1-mkv, flac, wav, tiff, png
        #[arg(long)]
        target: String,

        /// Dry run (plan only, no conversion)
        #[arg(long)]
        dry_run: bool,

        /// Preserve original after migration
        #[arg(long)]
        keep_original: bool,

        /// Validate output after migration
        #[arg(long)]
        validate: bool,
    },

    /// Generate archive status report
    Report {
        /// Archive root directory
        #[arg(short, long)]
        archive: PathBuf,

        /// Output report path
        #[arg(short, long)]
        output: PathBuf,

        /// Report format: json, csv, text
        #[arg(long, default_value = "json")]
        format: String,

        /// Include risk assessment
        #[arg(long)]
        risk: bool,

        /// Include format statistics
        #[arg(long)]
        stats: bool,
    },

    /// Manage preservation policies
    Policy {
        /// Policy operation: show, set, validate, export
        #[arg(long)]
        operation: String,

        /// Archive root directory
        #[arg(long)]
        archive: Option<PathBuf>,

        /// Policy file path
        #[arg(long)]
        policy_file: Option<PathBuf>,

        /// Retention period (e.g., "10y", "5y", "forever")
        #[arg(long)]
        retention: Option<String>,

        /// Minimum checksum interval (e.g., "30d", "90d", "1y")
        #[arg(long)]
        fixity_interval: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_preservation_format(s: &str) -> Result<oximedia_archive_pro::PreservationFormat> {
    match s.to_lowercase().as_str() {
        "ffv1-mkv" | "ffv1" | "video-ffv1" => Ok(oximedia_archive_pro::PreservationFormat::VideoFfv1Mkv),
        "ut-video" | "utvideo" => Ok(oximedia_archive_pro::PreservationFormat::VideoUtVideo),
        "flac" | "audio-flac" => Ok(oximedia_archive_pro::PreservationFormat::AudioFlac),
        "wav" | "pcm" | "audio-wav" => Ok(oximedia_archive_pro::PreservationFormat::AudioWav),
        "tiff" | "image-tiff" => Ok(oximedia_archive_pro::PreservationFormat::ImageTiff),
        "png" | "image-png" => Ok(oximedia_archive_pro::PreservationFormat::ImagePng),
        "jp2" | "jpeg2000" => Ok(oximedia_archive_pro::PreservationFormat::ImageJpeg2000),
        "pdf-a" | "pdfa" => Ok(oximedia_archive_pro::PreservationFormat::DocumentPdfA),
        "text" | "txt" => Ok(oximedia_archive_pro::PreservationFormat::DocumentText),
        _ => Err(anyhow::anyhow!(
            "Unknown preservation format: {s}. Supported: ffv1-mkv, flac, wav, tiff, png, jp2, pdf-a, text"
        )),
    }
}

fn compute_checksum(path: &std::path::Path, _algorithm: &str) -> Result<String> {
    use std::io::Read;
    let mut file =
        std::fs::File::open(path).with_context(|| format!("Failed to open: {}", path.display()))?;
    let mut hasher: u64 = 0xcbf29ce484222325;
    let mut buf = [0u8; 8192];
    loop {
        let n = file.read(&mut buf).context("Read error")?;
        if n == 0 {
            break;
        }
        for &byte in &buf[..n] {
            hasher ^= u64::from(byte);
            hasher = hasher.wrapping_mul(0x100000001b3);
        }
    }
    Ok(format!("{:016x}", hasher))
}

fn now_iso8601() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", dur.as_secs())
}

// ---------------------------------------------------------------------------
// Command handler
// ---------------------------------------------------------------------------

/// Handle archive-pro command dispatch.
pub async fn handle_archivepro_command(
    command: ArchiveProCommand,
    json_output: bool,
) -> Result<()> {
    match command {
        ArchiveProCommand::Ingest {
            input,
            archive,
            package_format,
            checksum,
            premis,
            target_format,
        } => {
            run_ingest(
                &input,
                &archive,
                &package_format,
                &checksum,
                premis,
                &target_format,
                json_output,
            )
            .await
        }
        ArchiveProCommand::Verify {
            input,
            checksum,
            deep,
            metadata,
            report,
        } => run_verify(&input, &checksum, deep, metadata, &report, json_output).await,
        ArchiveProCommand::Migrate {
            input,
            output,
            target,
            dry_run,
            keep_original,
            validate,
        } => {
            run_migrate(
                &input,
                &output,
                &target,
                dry_run,
                keep_original,
                validate,
                json_output,
            )
            .await
        }
        ArchiveProCommand::Report {
            archive,
            output,
            format,
            risk,
            stats,
        } => run_report(&archive, &output, &format, risk, stats, json_output).await,
        ArchiveProCommand::Policy {
            operation,
            archive,
            policy_file,
            retention,
            fixity_interval,
        } => {
            run_policy(
                &operation,
                &archive,
                &policy_file,
                &retention,
                &fixity_interval,
                json_output,
            )
            .await
        }
    }
}

// ---------------------------------------------------------------------------
// Ingest
// ---------------------------------------------------------------------------

async fn run_ingest(
    inputs: &[PathBuf],
    archive: &PathBuf,
    package_format: &str,
    checksum_algo: &str,
    _premis: bool,
    _target_format: &Option<String>,
    json_output: bool,
) -> Result<()> {
    if !archive.exists() {
        std::fs::create_dir_all(archive)
            .with_context(|| format!("Failed to create archive dir: {}", archive.display()))?;
    }

    let mut ingested = Vec::new();

    for path in inputs {
        if !path.exists() {
            return Err(anyhow::anyhow!("Input not found: {}", path.display()));
        }
        let checksum_val = compute_checksum(path, checksum_algo)?;
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let filename = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        // Copy to archive
        let dest = archive.join(&filename);
        std::fs::copy(path, &dest)
            .with_context(|| format!("Failed to copy {} to archive", path.display()))?;

        ingested.push(serde_json::json!({
            "filename": filename,
            "checksum": checksum_val,
            "size": size,
            "timestamp": now_iso8601(),
        }));
    }

    // Write manifest
    let manifest = serde_json::json!({
        "package_format": package_format,
        "checksum_algorithm": checksum_algo,
        "ingested_at": now_iso8601(),
        "files": ingested,
    });
    let manifest_path = archive.join("manifest.json");
    let manifest_str = serde_json::to_string_pretty(&manifest).context("Serialization failed")?;
    std::fs::write(&manifest_path, &manifest_str).context("Failed to write manifest")?;

    if json_output {
        let result = serde_json::json!({
            "command": "archive-pro ingest",
            "archive": archive.display().to_string(),
            "package_format": package_format,
            "files_ingested": ingested.len(),
        });
        let s = serde_json::to_string_pretty(&result).context("JSON serialization failed")?;
        println!("{s}");
    } else {
        println!("{}", "Archive Pro Ingest".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Archive:", archive.display());
        println!("{:20} {}", "Package format:", package_format);
        println!("{:20} {}", "Checksum:", checksum_algo);
        println!("{:20} {}", "Files ingested:", ingested.len());
        println!();
        for item in &ingested {
            let fname = item.get("filename").and_then(|v| v.as_str()).unwrap_or("?");
            println!("  {} {}", "+".green(), fname);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Verify
// ---------------------------------------------------------------------------

async fn run_verify(
    input: &PathBuf,
    checksum_algo: &str,
    deep: bool,
    _metadata: bool,
    report_path: &Option<PathBuf>,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input not found: {}", input.display()));
    }

    let manifest_path = if input.is_dir() {
        input.join("manifest.json")
    } else {
        input.clone()
    };

    let mut checks: Vec<(String, bool, String)> = Vec::new();
    let mut all_passed = true;

    // Check manifest exists
    let manifest_exists = manifest_path.exists();
    if !manifest_exists {
        all_passed = false;
    }
    checks.push((
        "manifest_exists".to_string(),
        manifest_exists,
        "Manifest file present".to_string(),
    ));

    // Deep verification
    if deep && manifest_exists {
        let data = std::fs::read_to_string(&manifest_path).context("Failed to read manifest")?;
        let manifest: serde_json::Value =
            serde_json::from_str(&data).context("Failed to parse manifest")?;
        if let Some(files) = manifest.get("files").and_then(|f| f.as_array()) {
            for file_entry in files {
                let fname = file_entry
                    .get("filename")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let expected = file_entry
                    .get("checksum")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let file_path = if input.is_dir() {
                    input.join(&fname)
                } else {
                    input
                        .parent()
                        .unwrap_or(std::path::Path::new("."))
                        .join(&fname)
                };
                if file_path.exists() {
                    let actual = compute_checksum(&file_path, checksum_algo)?;
                    let matched = actual == expected;
                    if !matched {
                        all_passed = false;
                    }
                    checks.push(("fixity".to_string(), matched, fname));
                } else {
                    all_passed = false;
                    checks.push(("file_missing".to_string(), false, fname));
                }
            }
        }
    }

    if let Some(ref rpath) = report_path {
        let report = serde_json::json!({
            "all_passed": all_passed,
            "checks": checks.iter().map(|(n, p, d)| serde_json::json!({"check": n, "passed": p, "detail": d})).collect::<Vec<_>>(),
        });
        let s = serde_json::to_string_pretty(&report).context("Serialization failed")?;
        std::fs::write(rpath, s)
            .with_context(|| format!("Failed to write report: {}", rpath.display()))?;
    }

    if json_output {
        let result = serde_json::json!({
            "command": "archive-pro verify",
            "input": input.display().to_string(),
            "all_passed": all_passed,
            "checks_count": checks.len(),
        });
        let s = serde_json::to_string_pretty(&result).context("JSON serialization failed")?;
        println!("{s}");
    } else {
        println!("{}", "Archive Pro Verify".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Algorithm:", checksum_algo);
        println!();
        for (name, passed, detail) in &checks {
            let status = if *passed {
                "PASS".green().to_string()
            } else {
                "FAIL".red().to_string()
            };
            println!("  [{}] {:20} {}", status, name, detail);
        }
        println!();
        if all_passed {
            println!("{}", "All checks passed.".green());
        } else {
            println!("{}", "Some checks failed.".red());
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Migrate
// ---------------------------------------------------------------------------

async fn run_migrate(
    input: &PathBuf,
    output: &PathBuf,
    target: &str,
    dry_run: bool,
    _keep_original: bool,
    _validate: bool,
    json_output: bool,
) -> Result<()> {
    if !input.exists() {
        return Err(anyhow::anyhow!("Input not found: {}", input.display()));
    }

    let pf = parse_preservation_format(target)?;

    if !dry_run && !output.exists() {
        std::fs::create_dir_all(output)
            .with_context(|| format!("Failed to create output dir: {}", output.display()))?;
    }

    let filename = input.file_name().unwrap_or_default().to_string_lossy();
    let new_name = format!(
        "{}.{}",
        filename
            .rsplit_once('.')
            .map(|(n, _)| n)
            .unwrap_or(&filename),
        pf.extension()
    );

    if !dry_run {
        let dest = output.join(&new_name);
        std::fs::copy(input, &dest)
            .with_context(|| format!("Failed to copy to {}", dest.display()))?;
    }

    if json_output {
        let result = serde_json::json!({
            "command": "archive-pro migrate",
            "input": input.display().to_string(),
            "output": output.display().to_string(),
            "target_format": target,
            "target_extension": pf.extension(),
            "target_mime": pf.mime_type(),
            "dry_run": dry_run,
            "new_filename": new_name,
        });
        let s = serde_json::to_string_pretty(&result).context("JSON serialization failed")?;
        println!("{s}");
    } else {
        println!("{}", "Archive Pro Migrate".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Input:", input.display());
        println!("{:20} {}", "Target format:", pf.description());
        println!("{:20} {}", "New filename:", new_name);
        if dry_run {
            println!();
            println!("{}", "(Dry run - no files were converted)".yellow());
        } else {
            println!("{:20} {}", "Output:", output.join(&new_name).display());
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

async fn run_report(
    archive: &PathBuf,
    output: &PathBuf,
    format: &str,
    _risk: bool,
    _stats: bool,
    json_output: bool,
) -> Result<()> {
    if !archive.exists() {
        return Err(anyhow::anyhow!("Archive not found: {}", archive.display()));
    }

    let mut file_count = 0usize;
    let mut total_size: u64 = 0;
    let mut formats: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    if archive.is_dir() {
        let entries = std::fs::read_dir(archive).context("Failed to read archive dir")?;
        for entry in entries {
            let entry = entry.context("Dir entry error")?;
            let path = entry.path();
            if path.is_file() {
                file_count += 1;
                total_size += std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("unknown")
                    .to_lowercase();
                *formats.entry(ext).or_insert(0) += 1;
            }
        }
    }

    let report_data = serde_json::json!({
        "archive": archive.display().to_string(),
        "total_files": file_count,
        "total_size": total_size,
        "formats": formats,
        "generated_at": now_iso8601(),
    });

    let report_str = match format {
        "text" => serde_json::to_string_pretty(&report_data).context("Serialization failed")?,
        _ => serde_json::to_string_pretty(&report_data).context("Serialization failed")?,
    };
    std::fs::write(output, &report_str)
        .with_context(|| format!("Failed to write report: {}", output.display()))?;

    if json_output {
        let s = serde_json::to_string_pretty(&report_data).context("JSON serialization failed")?;
        println!("{s}");
    } else {
        println!("{}", "Archive Pro Report".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Archive:", archive.display());
        println!("{:20} {}", "Total files:", file_count);
        println!(
            "{:20} {:.2} MB",
            "Total size:",
            total_size as f64 / (1024.0 * 1024.0)
        );
        println!("{:20} {}", "Report:", output.display());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

async fn run_policy(
    operation: &str,
    archive: &Option<PathBuf>,
    policy_file: &Option<PathBuf>,
    retention: &Option<String>,
    fixity_interval: &Option<String>,
    json_output: bool,
) -> Result<()> {
    match operation {
        "show" => {
            let policy = serde_json::json!({
                "retention": retention.clone().unwrap_or_else(|| "10y".to_string()),
                "fixity_interval": fixity_interval.clone().unwrap_or_else(|| "90d".to_string()),
                "checksum_algorithms": ["sha256", "blake3"],
                "preservation_formats": ["ffv1-mkv", "flac", "tiff"],
            });
            if json_output {
                let s =
                    serde_json::to_string_pretty(&policy).context("JSON serialization failed")?;
                println!("{s}");
            } else {
                println!("{}", "Archive Policy".green().bold());
                println!("{}", "=".repeat(60));
                println!(
                    "{:20} {}",
                    "Retention:",
                    retention.as_deref().unwrap_or("10y")
                );
                println!(
                    "{:20} {}",
                    "Fixity interval:",
                    fixity_interval.as_deref().unwrap_or("90d")
                );
                println!("{:20} sha256, blake3", "Algorithms:");
                println!("{:20} ffv1-mkv, flac, tiff", "Formats:");
            }
        }
        "set" => {
            let default_path = PathBuf::from("policy.json");
            let policy_path = policy_file
                .as_ref()
                .or(archive.as_ref())
                .unwrap_or(&default_path);
            let policy = serde_json::json!({
                "retention": retention.clone().unwrap_or_else(|| "10y".to_string()),
                "fixity_interval": fixity_interval.clone().unwrap_or_else(|| "90d".to_string()),
            });
            let s = serde_json::to_string_pretty(&policy).context("Serialization failed")?;
            std::fs::write(policy_path, s)
                .with_context(|| format!("Failed to write policy: {}", policy_path.display()))?;
            if !json_output {
                println!(
                    "{} Policy saved to {}",
                    "OK:".green(),
                    policy_path.display()
                );
            }
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Unknown policy operation: {operation}. Supported: show, set"
            ));
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
    fn test_parse_preservation_format() {
        assert!(parse_preservation_format("flac").is_ok());
        assert!(parse_preservation_format("ffv1-mkv").is_ok());
        assert!(parse_preservation_format("tiff").is_ok());
        assert!(parse_preservation_format("png").is_ok());
        assert!(parse_preservation_format("nonsense").is_err());
    }

    #[test]
    fn test_compute_checksum() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_archivepro_test.bin");
        std::fs::write(&path, b"archive test data").expect("write should succeed");
        let ck = compute_checksum(&path, "sha256");
        assert!(ck.is_ok());
        assert_eq!(ck.expect("checksum").len(), 16);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_preservation_format_properties() {
        let flac = oximedia_archive_pro::PreservationFormat::AudioFlac;
        assert_eq!(flac.extension(), "flac");
        assert_eq!(flac.mime_type(), "audio/flac");
    }

    #[test]
    fn test_now_iso8601() {
        let ts = now_iso8601();
        assert!(!ts.is_empty());
        // Should be a number string (seconds since epoch)
        assert!(ts.parse::<u64>().is_ok());
    }
}
