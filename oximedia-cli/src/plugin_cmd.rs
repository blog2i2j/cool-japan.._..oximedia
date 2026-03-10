//! Plugin management CLI commands.
//!
//! Provides commands for listing, inspecting, and validating plugins.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use std::path::PathBuf;

/// Plugin management subcommands.
#[derive(Subcommand, Debug)]
pub enum PluginCommand {
    /// List all installed/registered plugins
    List {},

    /// Show detailed info about a specific plugin
    Info {
        /// Plugin name
        name: String,
    },

    /// Show all available codecs from plugins
    Codecs {},

    /// Validate a plugin manifest file
    Validate {
        /// Path to plugin.json manifest
        path: PathBuf,
    },

    /// Show plugin search paths
    Paths {},
}

/// Handle plugin subcommands.
pub async fn handle_plugin_command(command: PluginCommand, json: bool) -> Result<()> {
    match command {
        PluginCommand::List {} => handle_list(json),
        PluginCommand::Info { name } => handle_info(&name, json),
        PluginCommand::Codecs {} => handle_codecs(json),
        PluginCommand::Validate { path } => handle_validate(&path, json),
        PluginCommand::Paths {} => handle_paths(json),
    }
}

fn handle_list(json: bool) -> Result<()> {
    let registry = oximedia_plugin::PluginRegistry::new();
    let plugins = registry.list_plugins();

    if json {
        let items: Vec<serde_json::Value> = plugins
            .iter()
            .map(|p| {
                serde_json::json!({
                    "name": p.name,
                    "version": p.version,
                    "author": p.author,
                    "description": p.description,
                    "license": p.license,
                    "patent_encumbered": p.patent_encumbered,
                    "api_version": p.api_version,
                })
            })
            .collect();
        let output =
            serde_json::to_string_pretty(&items).context("Failed to serialize plugin list")?;
        println!("{output}");
        return Ok(());
    }

    if plugins.is_empty() {
        println!("{}", "No plugins loaded.".yellow());
        println!();
        println!("To load plugins, place them in one of the search paths:");
        let paths = oximedia_plugin::PluginRegistry::default_search_paths();
        for p in &paths {
            println!("  {}", p.display());
        }
        println!();
        println!("Or set {} to a custom path.", "OXIMEDIA_PLUGIN_PATH".bold());
        return Ok(());
    }

    println!("{}", "Loaded Plugins:".bold().cyan());
    println!();
    for plugin in &plugins {
        let patent_tag = if plugin.patent_encumbered {
            " [PATENT]".red().to_string()
        } else {
            String::new()
        };
        println!(
            "  {} {} [{}]{}",
            plugin.name.bold(),
            format!("v{}", plugin.version).dimmed(),
            plugin.license,
            patent_tag,
        );
        println!("    {}", plugin.description);
    }
    println!();
    println!("Total: {} plugin(s)", plugins.len().to_string().bold());

    Ok(())
}

fn handle_info(name: &str, json: bool) -> Result<()> {
    let registry = oximedia_plugin::PluginRegistry::new();
    let plugins = registry.list_plugins();

    let plugin = plugins.iter().find(|p| p.name == name);

    match plugin {
        Some(info) => {
            if json {
                let value = serde_json::json!({
                    "name": info.name,
                    "version": info.version,
                    "author": info.author,
                    "description": info.description,
                    "license": info.license,
                    "patent_encumbered": info.patent_encumbered,
                    "api_version": info.api_version,
                });
                let output = serde_json::to_string_pretty(&value)
                    .context("Failed to serialize plugin info")?;
                println!("{output}");
            } else {
                println!("{}", "Plugin Information:".bold().cyan());
                println!();
                println!("  {:<20} {}", "Name:".bold(), info.name);
                println!("  {:<20} {}", "Version:".bold(), info.version);
                println!("  {:<20} {}", "Author:".bold(), info.author);
                println!("  {:<20} {}", "Description:".bold(), info.description);
                println!("  {:<20} {}", "License:".bold(), info.license);
                println!("  {:<20} {}", "API Version:".bold(), info.api_version);
                println!(
                    "  {:<20} {}",
                    "Patent Status:".bold(),
                    if info.patent_encumbered {
                        "Patent-encumbered".red().to_string()
                    } else {
                        "Clean (royalty-free)".green().to_string()
                    }
                );
            }
        }
        None => {
            if json {
                let value = serde_json::json!({
                    "error": format!("Plugin '{name}' not found"),
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&value).context("Failed to serialize error")?
                );
            } else {
                println!(
                    "{} Plugin '{}' not found.",
                    "Error:".red().bold(),
                    name.bold()
                );
                println!(
                    "Use '{}' to see available plugins.",
                    "oximedia plugin list".bold()
                );
            }
        }
    }

    Ok(())
}

fn handle_codecs(json: bool) -> Result<()> {
    let registry = oximedia_plugin::PluginRegistry::new();
    let codecs = registry.list_codecs();

    if json {
        let items: Vec<serde_json::Value> = codecs
            .iter()
            .map(|c| {
                serde_json::json!({
                    "name": c.codec_name,
                    "decode": c.can_decode,
                    "encode": c.can_encode,
                    "pixel_formats": c.pixel_formats,
                })
            })
            .collect();
        let output =
            serde_json::to_string_pretty(&items).context("Failed to serialize codec list")?;
        println!("{output}");
        return Ok(());
    }

    if codecs.is_empty() {
        println!("{}", "No plugin codecs available.".yellow());
        println!("Load plugins to add codec support.");
        return Ok(());
    }

    println!("{}", "Plugin Codecs:".bold().cyan());
    println!();
    println!(
        "  {:<20} {:<10} {:<10} {}",
        "Codec".bold(),
        "Decode".bold(),
        "Encode".bold(),
        "Formats".bold()
    );
    println!("  {}", "-".repeat(60));

    for codec in &codecs {
        let decode_str = if codec.can_decode {
            "yes".green().to_string()
        } else {
            "no".dimmed().to_string()
        };
        let encode_str = if codec.can_encode {
            "yes".green().to_string()
        } else {
            "no".dimmed().to_string()
        };
        let formats = if codec.pixel_formats.is_empty() {
            "-".to_string()
        } else {
            codec.pixel_formats.join(", ")
        };
        println!(
            "  {:<20} {:<10} {:<10} {}",
            codec.codec_name, decode_str, encode_str, formats
        );
    }
    println!();
    println!("Total: {} codec(s)", codecs.len().to_string().bold());

    Ok(())
}

fn handle_validate(path: &PathBuf, json: bool) -> Result<()> {
    let manifest = oximedia_plugin::PluginManifest::from_file(path)
        .context(format!("Failed to read manifest from '{}'", path.display()))?;

    let validation = manifest.validate();

    if json {
        let value = match &validation {
            Ok(()) => serde_json::json!({
                "valid": true,
                "name": manifest.name,
                "version": manifest.version,
                "api_version": manifest.api_version,
                "codecs": manifest.codecs.len(),
            }),
            Err(e) => serde_json::json!({
                "valid": false,
                "error": e.to_string(),
            }),
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&value)
                .context("Failed to serialize validation result")?
        );
        return Ok(());
    }

    match validation {
        Ok(()) => {
            println!("{} Manifest is valid.", "OK".green().bold());
            println!();
            println!("  {:<20} {}", "Name:".bold(), manifest.name);
            println!("  {:<20} {}", "Version:".bold(), manifest.version);
            println!("  {:<20} {}", "API Version:".bold(), manifest.api_version);
            println!("  {:<20} {}", "Library:".bold(), manifest.library);
            println!("  {:<20} {}", "Codecs:".bold(), manifest.codecs.len());
            for codec in &manifest.codecs {
                let mode = match (codec.decode, codec.encode) {
                    (true, true) => "decode+encode",
                    (true, false) => "decode",
                    (false, true) => "encode",
                    (false, false) => "none",
                };
                println!("    - {} ({})", codec.name, mode);
            }
        }
        Err(e) => {
            println!("{} Manifest validation failed:", "FAIL".red().bold());
            println!("  {e}");
        }
    }

    Ok(())
}

fn handle_paths(json: bool) -> Result<()> {
    let paths = oximedia_plugin::PluginRegistry::default_search_paths();

    if json {
        let items: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
        let output = serde_json::to_string_pretty(&items).context("Failed to serialize paths")?;
        println!("{output}");
        return Ok(());
    }

    println!("{}", "Plugin Search Paths:".bold().cyan());
    println!();
    for path in &paths {
        let exists = path.is_dir();
        let status = if exists {
            "exists".green().to_string()
        } else {
            "not found".dimmed().to_string()
        };
        println!("  {} [{}]", path.display(), status);
    }
    println!();
    println!(
        "Set {} to add custom paths (colon-separated).",
        "OXIMEDIA_PLUGIN_PATH".bold()
    );

    Ok(())
}
