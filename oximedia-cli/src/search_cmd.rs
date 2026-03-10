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
        #[arg(short, long)]
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
    use oximedia_search::{SearchEngine, SearchFilters, SearchQuery, SortOptions};

    if !index_path.exists() {
        anyhow::bail!(
            "Index directory not found: {}. Use `oximedia search index` to create one.",
            index_path.display()
        );
    }

    let engine = SearchEngine::new(index_path)
        .map_err(|e| anyhow::anyhow!("Failed to open search index: {}", e))?;

    let search_query = SearchQuery {
        text: Some(query.to_string()),
        visual: None,
        audio: None,
        filters: SearchFilters::default(),
        limit,
        offset,
        sort: SortOptions::default(),
    };

    let results = engine
        .search(&search_query)
        .map_err(|e| anyhow::anyhow!("Search failed: {}", e))?;

    if json_output {
        let items: Vec<serde_json::Value> = results
            .results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "asset_id": r.asset_id.to_string(),
                    "score": r.score,
                    "title": r.title,
                    "description": r.description,
                    "file_path": r.file_path,
                    "mime_type": r.mime_type,
                    "duration_ms": r.duration_ms,
                    "matched_fields": r.matched_fields,
                })
            })
            .collect();

        let obj = serde_json::json!({
            "query": query,
            "total": results.total,
            "limit": results.limit,
            "offset": results.offset,
            "query_time_ms": results.query_time_ms,
            "results": items,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "Media Text Search".green().bold());
    println!("  Query:  {}", query.cyan());
    println!("  Index:  {}", index_path.display());
    println!(
        "  Found:  {} results ({} ms)",
        results.total, results.query_time_ms
    );
    println!();

    if results.results.is_empty() {
        println!("  {} No matches found", "○".yellow());
        return Ok(());
    }

    for (idx, item) in results.results.iter().enumerate() {
        let title = item.title.as_deref().unwrap_or_else(|| &item.file_path);
        println!(
            "  {}. {} (score: {:.3})",
            idx + 1 + offset,
            title.yellow().bold(),
            item.score
        );
        println!("     {}", item.file_path.dimmed());
        if !item.matched_fields.is_empty() {
            println!("     Matched: {}", item.matched_fields.join(", ").blue());
        }
    }

    Ok(())
}

fn cmd_similar(
    input: &PathBuf,
    index_path: &PathBuf,
    limit: usize,
    threshold: f32,
    json_output: bool,
) -> Result<()> {
    use oximedia_search::{SearchEngine, SearchFilters, SearchQuery, SortOptions};

    if !input.exists() {
        anyhow::bail!("Input file not found: {}", input.display());
    }

    if !index_path.exists() {
        anyhow::bail!(
            "Index directory not found: {}. Use `oximedia search index` to create one.",
            index_path.display()
        );
    }

    // Read input file data for visual search
    let visual_data = std::fs::read(input)
        .with_context(|| format!("Failed to read input file: {}", input.display()))?;

    let engine = SearchEngine::new(index_path)
        .map_err(|e| anyhow::anyhow!("Failed to open search index: {}", e))?;

    let search_query = SearchQuery {
        text: None,
        visual: Some(visual_data),
        audio: None,
        filters: SearchFilters::default(),
        limit,
        offset: 0,
        sort: SortOptions::default(),
    };

    let results = engine
        .search(&search_query)
        .map_err(|e| anyhow::anyhow!("Visual search failed: {}", e))?;

    // Filter by similarity threshold
    let filtered: Vec<_> = results
        .results
        .iter()
        .filter(|r| r.score >= threshold)
        .collect();

    if json_output {
        let items: Vec<serde_json::Value> = filtered
            .iter()
            .map(|r| {
                serde_json::json!({
                    "asset_id": r.asset_id.to_string(),
                    "similarity": r.score,
                    "title": r.title,
                    "file_path": r.file_path,
                    "mime_type": r.mime_type,
                })
            })
            .collect();

        let obj = serde_json::json!({
            "input": input.to_string_lossy(),
            "threshold": threshold,
            "total_candidates": results.total,
            "matches_above_threshold": filtered.len(),
            "query_time_ms": results.query_time_ms,
            "results": items,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "Visual Similarity Search".green().bold());
    println!("  Input:      {}", input.display().to_string().cyan());
    println!("  Index:      {}", index_path.display());
    println!("  Threshold:  {:.0}%", threshold * 100.0);
    println!(
        "  Found:      {} matches ({} ms)",
        filtered.len(),
        results.query_time_ms
    );
    println!();

    if filtered.is_empty() {
        println!(
            "  {} No visually similar files found above threshold",
            "○".yellow()
        );
        return Ok(());
    }

    for (idx, item) in filtered.iter().enumerate() {
        let title = item.title.as_deref().unwrap_or_else(|| &item.file_path);
        println!(
            "  {}. {} (similarity: {:.1}%)",
            idx + 1,
            title.yellow().bold(),
            item.score * 100.0
        );
        println!("     {}", item.file_path.dimmed());
    }

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
    use oximedia_search::index::builder::{IndexDocument, VisualFeatures};
    use oximedia_search::SearchEngine;
    use uuid::Uuid;

    if !input.exists() {
        anyhow::bail!("Input file not found: {}", input.display());
    }

    // Create index directory
    std::fs::create_dir_all(index_dir)
        .with_context(|| format!("Cannot create index directory: {}", index_dir.display()))?;

    let mut engine = SearchEngine::new(index_dir)
        .map_err(|e| anyhow::anyhow!("Failed to open/create search index: {}", e))?;

    // Build index document
    let asset_id = Uuid::new_v4();
    let file_path = input.to_string_lossy().into_owned();
    let doc_title = title
        .map(|s| s.to_string())
        .or_else(|| input.file_stem().map(|s| s.to_string_lossy().into_owned()));

    // Read file to compute fingerprint for visual features
    let data = std::fs::read(input)
        .with_context(|| format!("Failed to read file: {}", input.display()))?;

    let fp = compute_file_fingerprint(&data);

    let visual_features = VisualFeatures {
        phash: fp.perceptual_hash.to_le_bytes().to_vec(),
        color_histogram: Vec::new(),
        edge_histogram: Vec::new(),
        texture_features: Vec::new(),
    };

    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    let doc = IndexDocument {
        asset_id,
        file_path: file_path.clone(),
        title: doc_title.clone(),
        description: description.map(|s| s.to_string()),
        keywords: Vec::new(),
        categories: Vec::new(),
        mime_type: None,
        format: None,
        codec: None,
        resolution: None,
        duration_ms: None,
        file_size: Some(data.len() as i64),
        bitrate: None,
        framerate: None,
        created_at: now_secs,
        modified_at: now_secs,
        transcript: None,
        ocr_text: None,
        visual_features: Some(visual_features),
        audio_fingerprint: Some(fp.audio_fingerprint.to_le_bytes().to_vec()),
        faces: None,
        dominant_colors: None,
        scene_tags: Vec::new(),
        detected_objects: Vec::new(),
        metadata: serde_json::Value::Null,
    };

    engine
        .index_document(&doc)
        .map_err(|e| anyhow::anyhow!("Failed to index document: {}", e))?;

    engine
        .commit()
        .map_err(|e| anyhow::anyhow!("Failed to commit index: {}", e))?;

    if json_output {
        let obj = serde_json::json!({
            "status": "indexed",
            "asset_id": asset_id.to_string(),
            "file": file_path,
            "title": doc_title,
            "description": description,
            "index_dir": index_dir.to_string_lossy(),
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
        return Ok(());
    }

    println!("{}", "Media Indexed".green().bold());
    println!("  File:    {}", input.display().to_string().cyan());
    println!("  ID:      {}", asset_id.to_string().yellow());
    if let Some(t) = &doc_title {
        println!("  Title:   {}", t);
    }
    if let Some(d) = description {
        println!("  Desc:    {}", d);
    }
    println!("  Index:   {}", index_dir.display());
    println!("  {} Successfully indexed", "✓".green().bold());

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
