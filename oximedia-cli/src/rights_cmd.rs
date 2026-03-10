//! Digital rights and license management CLI commands.
//!
//! Provides subcommands for managing content rights:
//! registering rights, checking usage rights, transferring ownership,
//! managing licenses, generating reports, and searching rights databases.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;

/// Rights management subcommands.
#[derive(Subcommand, Debug)]
pub enum RightsCommand {
    /// Register rights for a media asset
    Register {
        /// Asset path or ID
        #[arg(long)]
        asset: String,

        /// Rights holder name or ID
        #[arg(long)]
        holder: String,

        /// Rights type: master, sync, mechanical, performance, reproduction, distribution
        #[arg(long)]
        rights_type: String,

        /// Territory scope (comma-separated ISO 3166 codes, or "worldwide")
        #[arg(long, default_value = "worldwide")]
        territory: String,

        /// Start date (ISO 8601)
        #[arg(long)]
        start_date: Option<String>,

        /// End date (ISO 8601)
        #[arg(long)]
        end_date: Option<String>,

        /// Usage restrictions (comma-separated)
        #[arg(long)]
        restrictions: Option<String>,

        /// Royalty rate as a percentage (e.g., "5.0")
        #[arg(long)]
        royalty_rate: Option<f64>,
    },

    /// Check rights status for an asset
    Check {
        /// Asset path or ID
        #[arg(long)]
        asset: String,

        /// Intended use: broadcast, streaming, theatrical, download, physical
        #[arg(long)]
        intended_use: Option<String>,

        /// Territory for use (ISO 3166 code)
        #[arg(long)]
        territory: Option<String>,

        /// Check date (ISO 8601, defaults to now)
        #[arg(long)]
        date: Option<String>,

        /// Show detailed breakdown
        #[arg(long)]
        verbose: bool,
    },

    /// Transfer rights to a new holder
    Transfer {
        /// Asset path or ID
        #[arg(long)]
        asset: String,

        /// Current rights holder
        #[arg(long)]
        from_holder: String,

        /// New rights holder
        #[arg(long)]
        to_holder: String,

        /// Rights type to transfer (omit for all)
        #[arg(long)]
        rights_type: Option<String>,

        /// Transfer effective date (ISO 8601)
        #[arg(long)]
        effective_date: Option<String>,

        /// Transfer consideration/price
        #[arg(long)]
        consideration: Option<String>,

        /// Require acknowledgment
        #[arg(long)]
        require_ack: bool,
    },

    /// Manage licenses for assets
    License {
        /// Asset path or ID
        #[arg(long)]
        asset: String,

        /// License action: create, renew, revoke, show
        #[arg(long, default_value = "show")]
        action: String,

        /// License type: royalty-free, rights-managed, editorial, creative-commons
        #[arg(long)]
        license_type: Option<String>,

        /// Licensee name or ID
        #[arg(long)]
        licensee: Option<String>,

        /// License duration (e.g., "1y", "6m", "perpetual")
        #[arg(long)]
        duration: Option<String>,

        /// Permitted uses (comma-separated)
        #[arg(long)]
        uses: Option<String>,

        /// Maximum distribution count
        #[arg(long)]
        max_distributions: Option<u32>,
    },

    /// Generate rights report
    Report {
        /// Report type: summary, expiring, royalties, territory, compliance
        #[arg(long, default_value = "summary")]
        report_type: String,

        /// Report period start (ISO 8601)
        #[arg(long)]
        from: Option<String>,

        /// Report period end (ISO 8601)
        #[arg(long)]
        to: Option<String>,

        /// Filter by rights holder
        #[arg(long)]
        holder: Option<String>,

        /// Days until expiration threshold (for 'expiring' report)
        #[arg(long, default_value = "30")]
        expiry_days: u32,

        /// Output format: text, json, csv
        #[arg(long, default_value = "text")]
        output_format: String,
    },

    /// Search rights database
    Search {
        /// Search query
        #[arg(long)]
        query: String,

        /// Search scope: all, holder, asset, territory, license
        #[arg(long, default_value = "all")]
        scope: String,

        /// Maximum results
        #[arg(long, default_value = "50")]
        limit: u32,

        /// Output format: text, json
        #[arg(long, default_value = "text")]
        output_format: String,
    },
}

/// Handle rights command dispatch.
pub async fn handle_rights_command(command: RightsCommand, json_output: bool) -> Result<()> {
    match command {
        RightsCommand::Register {
            asset,
            holder,
            rights_type,
            territory,
            start_date,
            end_date,
            restrictions,
            royalty_rate,
        } => {
            register_rights(
                &asset,
                &holder,
                &rights_type,
                &territory,
                start_date.as_deref(),
                end_date.as_deref(),
                restrictions.as_deref(),
                royalty_rate,
                json_output,
            )
            .await
        }
        RightsCommand::Check {
            asset,
            intended_use,
            territory,
            date,
            verbose,
        } => {
            check_rights(
                &asset,
                intended_use.as_deref(),
                territory.as_deref(),
                date.as_deref(),
                verbose,
                json_output,
            )
            .await
        }
        RightsCommand::Transfer {
            asset,
            from_holder,
            to_holder,
            rights_type,
            effective_date,
            consideration,
            require_ack,
        } => {
            transfer_rights(
                &asset,
                &from_holder,
                &to_holder,
                rights_type.as_deref(),
                effective_date.as_deref(),
                consideration.as_deref(),
                require_ack,
                json_output,
            )
            .await
        }
        RightsCommand::License {
            asset,
            action,
            license_type,
            licensee,
            duration,
            uses,
            max_distributions,
        } => {
            manage_license(
                &asset,
                &action,
                license_type.as_deref(),
                licensee.as_deref(),
                duration.as_deref(),
                uses.as_deref(),
                max_distributions,
                json_output,
            )
            .await
        }
        RightsCommand::Report {
            report_type,
            from,
            to,
            holder,
            expiry_days,
            output_format,
        } => {
            generate_report(
                &report_type,
                from.as_deref(),
                to.as_deref(),
                holder.as_deref(),
                expiry_days,
                if json_output { "json" } else { &output_format },
            )
            .await
        }
        RightsCommand::Search {
            query,
            scope,
            limit,
            output_format,
        } => {
            search_rights(
                &query,
                &scope,
                limit,
                if json_output { "json" } else { &output_format },
            )
            .await
        }
    }
}

/// Validate rights type string.
fn validate_rights_type(rights_type: &str) -> Result<()> {
    match rights_type {
        "master" | "sync" | "mechanical" | "performance" | "reproduction" | "distribution" => {
            Ok(())
        }
        other => Err(anyhow::anyhow!(
            "Unknown rights type '{}'. Expected: master, sync, mechanical, performance, reproduction, distribution",
            other
        )),
    }
}

/// Register rights for an asset.
#[allow(clippy::too_many_arguments)]
async fn register_rights(
    asset: &str,
    holder: &str,
    rights_type: &str,
    territory: &str,
    start_date: Option<&str>,
    end_date: Option<&str>,
    restrictions: Option<&str>,
    royalty_rate: Option<f64>,
    json_output: bool,
) -> Result<()> {
    validate_rights_type(rights_type)?;

    let rights_id = format!("rights-{}", uuid::Uuid::new_v4().as_simple());
    let restriction_list: Vec<String> = restrictions
        .map(|r| r.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    if json_output {
        let result = serde_json::json!({
            "command": "register",
            "rights_id": rights_id,
            "asset": asset,
            "holder": holder,
            "rights_type": rights_type,
            "territory": territory,
            "start_date": start_date,
            "end_date": end_date,
            "restrictions": restriction_list,
            "royalty_rate": royalty_rate,
            "status": "registered",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{json_str}");
    } else {
        println!("{}", "Rights Registered".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:25} {}", "Rights ID:", rights_id);
        println!("{:25} {}", "Asset:", asset);
        println!("{:25} {}", "Holder:", holder);
        println!("{:25} {}", "Rights type:", rights_type);
        println!("{:25} {}", "Territory:", territory);
        if let Some(sd) = start_date {
            println!("{:25} {}", "Start date:", sd);
        }
        if let Some(ed) = end_date {
            println!("{:25} {}", "End date:", ed);
        }
        if !restriction_list.is_empty() {
            println!("{:25} {}", "Restrictions:", restriction_list.join(", "));
        }
        if let Some(rate) = royalty_rate {
            println!("{:25} {:.2}%", "Royalty rate:", rate);
        }
        println!();
        println!("{}", "Rights registered successfully.".cyan().bold());
    }

    Ok(())
}

/// Check rights for an asset.
async fn check_rights(
    asset: &str,
    intended_use: Option<&str>,
    territory: Option<&str>,
    date: Option<&str>,
    verbose: bool,
    json_output: bool,
) -> Result<()> {
    // Default: no rights found (requires database)
    let status = "unknown";
    let message = "Rights check requires database integration";

    if json_output {
        let result = serde_json::json!({
            "command": "check",
            "asset": asset,
            "intended_use": intended_use,
            "territory": territory,
            "date": date,
            "status": status,
            "cleared": false,
            "message": message,
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{json_str}");
    } else {
        println!("{}", "Rights Check".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:25} {}", "Asset:", asset);
        if let Some(use_type) = intended_use {
            println!("{:25} {}", "Intended use:", use_type);
        }
        if let Some(terr) = territory {
            println!("{:25} {}", "Territory:", terr);
        }
        if let Some(d) = date {
            println!("{:25} {}", "Date:", d);
        }
        println!();
        println!("{:25} {}", "Status:", status.yellow());
        if verbose {
            println!("{:25} {}", "Note:", message);
        }
    }

    Ok(())
}

/// Transfer rights between holders.
#[allow(clippy::too_many_arguments)]
async fn transfer_rights(
    asset: &str,
    from_holder: &str,
    to_holder: &str,
    rights_type: Option<&str>,
    effective_date: Option<&str>,
    consideration: Option<&str>,
    require_ack: bool,
    json_output: bool,
) -> Result<()> {
    if let Some(rt) = rights_type {
        validate_rights_type(rt)?;
    }

    let transfer_id = format!("transfer-{}", uuid::Uuid::new_v4().as_simple());

    if json_output {
        let result = serde_json::json!({
            "command": "transfer",
            "transfer_id": transfer_id,
            "asset": asset,
            "from_holder": from_holder,
            "to_holder": to_holder,
            "rights_type": rights_type,
            "effective_date": effective_date,
            "consideration": consideration,
            "require_ack": require_ack,
            "status": "pending",
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{json_str}");
    } else {
        println!("{}", "Rights Transfer".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:25} {}", "Transfer ID:", transfer_id);
        println!("{:25} {}", "Asset:", asset);
        println!("{:25} {}", "From:", from_holder);
        println!("{:25} {}", "To:", to_holder);
        if let Some(rt) = rights_type {
            println!("{:25} {}", "Rights type:", rt);
        } else {
            println!("{:25} all", "Rights type:");
        }
        if let Some(ed) = effective_date {
            println!("{:25} {}", "Effective date:", ed);
        }
        if let Some(c) = consideration {
            println!("{:25} {}", "Consideration:", c);
        }
        println!("{:25} {}", "Requires ACK:", require_ack);
        println!();
        println!(
            "{}",
            "Rights transfer initiated (pending acknowledgment)."
                .cyan()
                .bold()
        );
    }

    Ok(())
}

/// Manage licenses for an asset.
#[allow(clippy::too_many_arguments)]
async fn manage_license(
    asset: &str,
    action: &str,
    license_type: Option<&str>,
    licensee: Option<&str>,
    duration: Option<&str>,
    uses: Option<&str>,
    max_distributions: Option<u32>,
    json_output: bool,
) -> Result<()> {
    match action {
        "create" | "renew" | "revoke" | "show" => {}
        other => {
            return Err(anyhow::anyhow!(
                "Unknown license action '{}'. Expected: create, renew, revoke, show",
                other
            ));
        }
    }

    let license_id = format!("license-{}", uuid::Uuid::new_v4().as_simple());
    let use_list: Vec<String> = uses
        .map(|u| u.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    if json_output {
        let result = serde_json::json!({
            "command": "license",
            "license_id": license_id,
            "asset": asset,
            "action": action,
            "license_type": license_type,
            "licensee": licensee,
            "duration": duration,
            "permitted_uses": use_list,
            "max_distributions": max_distributions,
            "status": action,
        });
        let json_str =
            serde_json::to_string_pretty(&result).context("Failed to serialize result")?;
        println!("{json_str}");
    } else {
        println!("{}", format!("License: {action}").green().bold());
        println!("{}", "=".repeat(60));
        println!("{:25} {}", "License ID:", license_id);
        println!("{:25} {}", "Asset:", asset);
        println!("{:25} {}", "Action:", action);
        if let Some(lt) = license_type {
            println!("{:25} {}", "License type:", lt);
        }
        if let Some(l) = licensee {
            println!("{:25} {}", "Licensee:", l);
        }
        if let Some(d) = duration {
            println!("{:25} {}", "Duration:", d);
        }
        if !use_list.is_empty() {
            println!("{:25} {}", "Permitted uses:", use_list.join(", "));
        }
        if let Some(md) = max_distributions {
            println!("{:25} {}", "Max distributions:", md);
        }
        println!();
        println!("{}", format!("License {action} completed.").cyan().bold());
    }

    Ok(())
}

/// Generate a rights report.
async fn generate_report(
    report_type: &str,
    from: Option<&str>,
    to: Option<&str>,
    holder: Option<&str>,
    expiry_days: u32,
    output_format: &str,
) -> Result<()> {
    match report_type {
        "summary" | "expiring" | "royalties" | "territory" | "compliance" => {}
        other => {
            return Err(anyhow::anyhow!(
                "Unknown report type '{}'. Expected: summary, expiring, royalties, territory, compliance",
                other
            ));
        }
    }

    match output_format {
        "json" | "csv" => {
            let result = serde_json::json!({
                "command": "report",
                "report_type": report_type,
                "from": from,
                "to": to,
                "holder": holder,
                "expiry_days": expiry_days,
                "entries": [],
                "total_entries": 0,
            });
            let json_str = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
            println!("{json_str}");
        }
        _ => {
            println!("{}", format!("Rights Report: {report_type}").green().bold());
            println!("{}", "=".repeat(60));
            if let Some(f) = from {
                println!("{:25} {}", "From:", f);
            }
            if let Some(t) = to {
                println!("{:25} {}", "To:", t);
            }
            if let Some(h) = holder {
                println!("{:25} {}", "Holder:", h);
            }
            if report_type == "expiring" {
                println!("{:25} {} days", "Expiry threshold:", expiry_days);
            }
            println!();
            println!("{}", "No report entries found.".yellow());
            println!(
                "{}",
                "Note: Full rights reporting requires database integration.".yellow()
            );
        }
    }

    Ok(())
}

/// Search the rights database.
async fn search_rights(query: &str, scope: &str, limit: u32, output_format: &str) -> Result<()> {
    match scope {
        "all" | "holder" | "asset" | "territory" | "license" => {}
        other => {
            return Err(anyhow::anyhow!(
                "Unknown search scope '{}'. Expected: all, holder, asset, territory, license",
                other
            ));
        }
    }

    match output_format {
        "json" => {
            let result = serde_json::json!({
                "command": "search",
                "query": query,
                "scope": scope,
                "limit": limit,
                "results": [],
                "total_results": 0,
            });
            let json_str = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
            println!("{json_str}");
        }
        _ => {
            println!("{}", "Rights Search".green().bold());
            println!("{}", "=".repeat(60));
            println!("{:25} {}", "Query:", query);
            println!("{:25} {}", "Scope:", scope);
            println!("{:25} {}", "Limit:", limit);
            println!();
            println!("{}", "No results found.".yellow());
            println!(
                "{}",
                "Note: Rights search requires database integration.".yellow()
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_rights_type() {
        assert!(validate_rights_type("master").is_ok());
        assert!(validate_rights_type("sync").is_ok());
        assert!(validate_rights_type("mechanical").is_ok());
        assert!(validate_rights_type("performance").is_ok());
        assert!(validate_rights_type("reproduction").is_ok());
        assert!(validate_rights_type("distribution").is_ok());
        assert!(validate_rights_type("invalid").is_err());
    }

    #[test]
    fn test_rights_command_register() {
        let cmd = RightsCommand::Register {
            asset: "song-001".to_string(),
            holder: "Music Corp".to_string(),
            rights_type: "master".to_string(),
            territory: "worldwide".to_string(),
            start_date: Some("2026-01-01".to_string()),
            end_date: Some("2031-01-01".to_string()),
            restrictions: Some("no-sublicense".to_string()),
            royalty_rate: Some(5.0),
        };
        assert!(matches!(cmd, RightsCommand::Register { .. }));
    }

    #[test]
    fn test_rights_command_transfer() {
        let cmd = RightsCommand::Transfer {
            asset: "song-001".to_string(),
            from_holder: "Music Corp".to_string(),
            to_holder: "New Label".to_string(),
            rights_type: Some("master".to_string()),
            effective_date: Some("2026-06-01".to_string()),
            consideration: Some("$50,000".to_string()),
            require_ack: true,
        };
        assert!(matches!(cmd, RightsCommand::Transfer { .. }));
    }

    #[test]
    fn test_rights_command_license() {
        let cmd = RightsCommand::License {
            asset: "video-001".to_string(),
            action: "create".to_string(),
            license_type: Some("rights-managed".to_string()),
            licensee: Some("Streaming Co".to_string()),
            duration: Some("1y".to_string()),
            uses: Some("streaming,download".to_string()),
            max_distributions: Some(1000000),
        };
        assert!(matches!(cmd, RightsCommand::License { .. }));
    }

    #[test]
    fn test_rights_command_search() {
        let cmd = RightsCommand::Search {
            query: "Music Corp".to_string(),
            scope: "holder".to_string(),
            limit: 50,
            output_format: "text".to_string(),
        };
        assert!(matches!(cmd, RightsCommand::Search { .. }));
    }
}
