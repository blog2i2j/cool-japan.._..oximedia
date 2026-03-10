//! Collaborative editing session CLI commands.
//!
//! Provides commands for creating, joining, sharing, commenting on,
//! exporting, and checking the status of real-time collaborative editing sessions.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Command definitions
// ---------------------------------------------------------------------------

/// Collab command subcommands.
#[derive(Subcommand, Debug)]
pub enum CollabCommand {
    /// Create a new collaborative editing session
    Create {
        /// Project name or identifier
        #[arg(short, long)]
        project: String,

        /// Session name/title
        #[arg(short, long)]
        name: String,

        /// Owner username
        #[arg(long)]
        owner: String,

        /// Maximum concurrent users
        #[arg(long, default_value = "10")]
        max_users: usize,

        /// Enable offline editing support
        #[arg(long)]
        offline: bool,

        /// Session database path (JSON file)
        #[arg(long)]
        db: PathBuf,
    },

    /// Join an existing collaborative session
    Join {
        /// Session ID to join
        #[arg(short, long)]
        session_id: String,

        /// Username for joining
        #[arg(short, long)]
        user: String,

        /// Role: owner, editor, viewer
        #[arg(long, default_value = "editor")]
        role: String,

        /// Session database path (JSON file)
        #[arg(long)]
        db: PathBuf,
    },

    /// Share a session with another user
    Share {
        /// Session ID to share
        #[arg(short, long)]
        session_id: String,

        /// Email or username to share with
        #[arg(short, long)]
        target: String,

        /// Permission level: editor, viewer
        #[arg(long, default_value = "viewer")]
        permission: String,

        /// Session database path (JSON file)
        #[arg(long)]
        db: PathBuf,
    },

    /// Add a comment to a collaborative session
    Comment {
        /// Session ID
        #[arg(short, long)]
        session_id: String,

        /// Comment author
        #[arg(long)]
        author: String,

        /// Comment text
        #[arg(short, long)]
        message: String,

        /// Timecode or frame number to attach the comment to
        #[arg(long)]
        timecode: Option<String>,

        /// Session database path (JSON file)
        #[arg(long)]
        db: PathBuf,
    },

    /// Export session data (edits, comments, history)
    Export {
        /// Session ID to export
        #[arg(short, long)]
        session_id: String,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Export format: json, csv, edl
        #[arg(long, default_value = "json")]
        format: String,

        /// Include comment history
        #[arg(long)]
        include_comments: bool,

        /// Include edit history
        #[arg(long)]
        include_edits: bool,

        /// Session database path (JSON file)
        #[arg(long)]
        db: PathBuf,
    },

    /// Show status of a collaborative session
    Status {
        /// Session ID (if omitted, shows all sessions)
        #[arg(short, long)]
        session_id: Option<String>,

        /// Session database path (JSON file)
        #[arg(long)]
        db: PathBuf,

        /// Show detailed status including user activity
        #[arg(long)]
        detailed: bool,
    },
}

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionRecord {
    id: String,
    project: String,
    name: String,
    owner: String,
    max_users: usize,
    offline_enabled: bool,
    created_at: String,
    status: String,
    users: Vec<UserRecord>,
    comments: Vec<CommentRecord>,
    shares: Vec<ShareRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UserRecord {
    username: String,
    role: String,
    joined_at: String,
    active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CommentRecord {
    id: String,
    author: String,
    message: String,
    timecode: Option<String>,
    created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ShareRecord {
    target: String,
    permission: String,
    shared_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CollabDb {
    version: u32,
    sessions: Vec<SessionRecord>,
}

// ---------------------------------------------------------------------------
// Persistence helpers
// ---------------------------------------------------------------------------

fn load_db(path: &PathBuf) -> Result<CollabDb> {
    if !path.exists() {
        return Ok(CollabDb {
            version: 1,
            ..CollabDb::default()
        });
    }
    let data = std::fs::read_to_string(path).context("Failed to read collab database")?;
    let db: CollabDb = serde_json::from_str(&data).context("Failed to parse collab database")?;
    Ok(db)
}

fn save_db(path: &PathBuf, db: &CollabDb) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).context("Failed to create database directory")?;
        }
    }
    let data = serde_json::to_string_pretty(db).context("Failed to serialize collab database")?;
    std::fs::write(path, data).context("Failed to write collab database")?;
    Ok(())
}

fn generate_id(prefix: &str) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{prefix}-{:016x}", now.as_nanos())
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

/// Handle collab command dispatch.
pub async fn handle_collab_command(command: CollabCommand, json_output: bool) -> Result<()> {
    match command {
        CollabCommand::Create {
            project,
            name,
            owner,
            max_users,
            offline,
            db,
        } => {
            run_create(
                &project,
                &name,
                &owner,
                max_users,
                offline,
                &db,
                json_output,
            )
            .await
        }
        CollabCommand::Join {
            session_id,
            user,
            role,
            db,
        } => run_join(&session_id, &user, &role, &db, json_output).await,
        CollabCommand::Share {
            session_id,
            target,
            permission,
            db,
        } => run_share(&session_id, &target, &permission, &db, json_output).await,
        CollabCommand::Comment {
            session_id,
            author,
            message,
            timecode,
            db,
        } => run_comment(&session_id, &author, &message, &timecode, &db, json_output).await,
        CollabCommand::Export {
            session_id,
            output,
            format,
            include_comments,
            include_edits,
            db,
        } => {
            run_export(
                &session_id,
                &output,
                &format,
                include_comments,
                include_edits,
                &db,
                json_output,
            )
            .await
        }
        CollabCommand::Status {
            session_id,
            db,
            detailed,
        } => run_status(&session_id, &db, detailed, json_output).await,
    }
}

// ---------------------------------------------------------------------------
// Create
// ---------------------------------------------------------------------------

async fn run_create(
    project: &str,
    name: &str,
    owner: &str,
    max_users: usize,
    offline: bool,
    db_path: &PathBuf,
    json_output: bool,
) -> Result<()> {
    let mut db = load_db(db_path)?;
    let session_id = generate_id("session");

    let session = SessionRecord {
        id: session_id.clone(),
        project: project.to_string(),
        name: name.to_string(),
        owner: owner.to_string(),
        max_users,
        offline_enabled: offline,
        created_at: now_iso8601(),
        status: "active".to_string(),
        users: vec![UserRecord {
            username: owner.to_string(),
            role: "owner".to_string(),
            joined_at: now_iso8601(),
            active: true,
        }],
        comments: Vec::new(),
        shares: Vec::new(),
    };

    db.sessions.push(session);
    save_db(db_path, &db)?;

    if json_output {
        let result = serde_json::json!({
            "command": "collab_create",
            "session_id": session_id,
            "project": project,
            "name": name,
            "owner": owner,
            "max_users": max_users,
            "offline_enabled": offline,
        });
        let s = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
        println!("{s}");
    } else {
        println!("{}", "Collab Session Created".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Session ID:", session_id);
        println!("{:20} {}", "Project:", project);
        println!("{:20} {}", "Name:", name);
        println!("{:20} {}", "Owner:", owner);
        println!("{:20} {}", "Max users:", max_users);
        println!("{:20} {}", "Offline:", offline);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Join
// ---------------------------------------------------------------------------

async fn run_join(
    session_id: &str,
    user: &str,
    role: &str,
    db_path: &PathBuf,
    json_output: bool,
) -> Result<()> {
    let mut db = load_db(db_path)?;

    let session = db
        .sessions
        .iter_mut()
        .find(|s| s.id == session_id)
        .ok_or_else(|| anyhow::anyhow!("Session not found: {session_id}"))?;

    if session.users.len() >= session.max_users {
        return Err(anyhow::anyhow!(
            "Session full: {}/{} users",
            session.users.len(),
            session.max_users
        ));
    }

    if session.users.iter().any(|u| u.username == user) {
        return Err(anyhow::anyhow!("User already in session: {user}"));
    }

    let valid_role = match role {
        "owner" | "editor" | "viewer" => role.to_string(),
        _ => {
            return Err(anyhow::anyhow!(
                "Invalid role: {role}. Use owner, editor, or viewer"
            ))
        }
    };

    session.users.push(UserRecord {
        username: user.to_string(),
        role: valid_role.clone(),
        joined_at: now_iso8601(),
        active: true,
    });

    let user_count = session.users.len();

    save_db(db_path, &db)?;

    if json_output {
        let result = serde_json::json!({
            "command": "collab_join",
            "session_id": session_id,
            "user": user,
            "role": valid_role,
            "user_count": user_count,
        });
        let s = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
        println!("{s}");
    } else {
        println!("{}", "Joined Session".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Session ID:", session_id);
        println!("{:20} {}", "User:", user);
        println!("{:20} {}", "Role:", valid_role);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Share
// ---------------------------------------------------------------------------

async fn run_share(
    session_id: &str,
    target: &str,
    permission: &str,
    db_path: &PathBuf,
    json_output: bool,
) -> Result<()> {
    let mut db = load_db(db_path)?;

    let session = db
        .sessions
        .iter_mut()
        .find(|s| s.id == session_id)
        .ok_or_else(|| anyhow::anyhow!("Session not found: {session_id}"))?;

    let valid_perm = match permission {
        "editor" | "viewer" => permission.to_string(),
        _ => {
            return Err(anyhow::anyhow!(
                "Invalid permission: {permission}. Use editor or viewer"
            ))
        }
    };

    session.shares.push(ShareRecord {
        target: target.to_string(),
        permission: valid_perm.clone(),
        shared_at: now_iso8601(),
    });

    save_db(db_path, &db)?;

    if json_output {
        let result = serde_json::json!({
            "command": "collab_share",
            "session_id": session_id,
            "target": target,
            "permission": valid_perm,
        });
        let s = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
        println!("{s}");
    } else {
        println!("{}", "Session Shared".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Session ID:", session_id);
        println!("{:20} {}", "Shared with:", target);
        println!("{:20} {}", "Permission:", valid_perm);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Comment
// ---------------------------------------------------------------------------

async fn run_comment(
    session_id: &str,
    author: &str,
    message: &str,
    timecode: &Option<String>,
    db_path: &PathBuf,
    json_output: bool,
) -> Result<()> {
    let mut db = load_db(db_path)?;

    let session = db
        .sessions
        .iter_mut()
        .find(|s| s.id == session_id)
        .ok_or_else(|| anyhow::anyhow!("Session not found: {session_id}"))?;

    let comment_id = generate_id("comment");
    session.comments.push(CommentRecord {
        id: comment_id.clone(),
        author: author.to_string(),
        message: message.to_string(),
        timecode: timecode.clone(),
        created_at: now_iso8601(),
    });

    save_db(db_path, &db)?;

    if json_output {
        let result = serde_json::json!({
            "command": "collab_comment",
            "comment_id": comment_id,
            "session_id": session_id,
            "author": author,
            "message": message,
            "timecode": timecode,
        });
        let s = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
        println!("{s}");
    } else {
        println!("{}", "Comment Added".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Comment ID:", comment_id);
        println!("{:20} {}", "Session ID:", session_id);
        println!("{:20} {}", "Author:", author);
        if let Some(tc) = timecode {
            println!("{:20} {}", "Timecode:", tc);
        }
        println!("{:20} {}", "Message:", message);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

async fn run_export(
    session_id: &str,
    output: &PathBuf,
    format: &str,
    include_comments: bool,
    _include_edits: bool,
    db_path: &PathBuf,
    json_output: bool,
) -> Result<()> {
    let db = load_db(db_path)?;

    let session = db
        .sessions
        .iter()
        .find(|s| s.id == session_id)
        .ok_or_else(|| anyhow::anyhow!("Session not found: {session_id}"))?;

    let export_data = match format {
        "json" => {
            let mut data = serde_json::json!({
                "session_id": session.id,
                "project": session.project,
                "name": session.name,
                "owner": session.owner,
                "status": session.status,
                "users": session.users.iter().map(|u| serde_json::json!({
                    "username": u.username,
                    "role": u.role,
                    "active": u.active,
                })).collect::<Vec<_>>(),
            });
            if include_comments {
                data["comments"] = serde_json::json!(session
                    .comments
                    .iter()
                    .map(|c| {
                        serde_json::json!({
                            "id": c.id,
                            "author": c.author,
                            "message": c.message,
                            "timecode": c.timecode,
                            "created_at": c.created_at,
                        })
                    })
                    .collect::<Vec<_>>());
            }
            serde_json::to_string_pretty(&data).context("Failed to serialize export")?
        }
        "csv" => {
            let mut csv = String::from("type,id,author,message,timecode,created_at\n");
            if include_comments {
                for c in &session.comments {
                    csv.push_str(&format!(
                        "comment,{},{},{},{},{}\n",
                        c.id,
                        c.author,
                        c.message.replace(',', ";"),
                        c.timecode.as_deref().unwrap_or(""),
                        c.created_at
                    ));
                }
            }
            csv
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported export format: {format}. Use json or csv"
            ))
        }
    };

    if let Some(parent) = output.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).context("Failed to create output directory")?;
        }
    }
    std::fs::write(output, &export_data).context("Failed to write export file")?;

    if json_output {
        let result = serde_json::json!({
            "command": "collab_export",
            "session_id": session_id,
            "output": output.display().to_string(),
            "format": format,
            "size_bytes": export_data.len(),
        });
        let s = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
        println!("{s}");
    } else {
        println!("{}", "Session Exported".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Session ID:", session_id);
        println!("{:20} {}", "Output:", output.display());
        println!("{:20} {}", "Format:", format);
        println!("{:20} {} bytes", "Size:", export_data.len());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

async fn run_status(
    session_id: &Option<String>,
    db_path: &PathBuf,
    detailed: bool,
    json_output: bool,
) -> Result<()> {
    let db = load_db(db_path)?;

    let sessions: Vec<&SessionRecord> = if let Some(ref id) = session_id {
        db.sessions.iter().filter(|s| s.id == *id).collect()
    } else {
        db.sessions.iter().collect()
    };

    if sessions.is_empty() {
        if session_id.is_some() {
            return Err(anyhow::anyhow!("Session not found"));
        }
        if !json_output {
            println!("{}", "No active sessions".yellow());
        }
        return Ok(());
    }

    if json_output {
        let result = serde_json::json!({
            "command": "collab_status",
            "sessions": sessions.iter().map(|s| {
                let mut entry = serde_json::json!({
                    "id": s.id,
                    "project": s.project,
                    "name": s.name,
                    "owner": s.owner,
                    "status": s.status,
                    "user_count": s.users.len(),
                    "comment_count": s.comments.len(),
                });
                if detailed {
                    entry["users"] = serde_json::json!(s.users.iter().map(|u| {
                        serde_json::json!({"username": u.username, "role": u.role, "active": u.active})
                    }).collect::<Vec<_>>());
                }
                entry
            }).collect::<Vec<_>>(),
        });
        let s = serde_json::to_string_pretty(&result).context("Failed to serialize")?;
        println!("{s}");
    } else {
        println!("{}", "Collaboration Status".green().bold());
        println!("{}", "=".repeat(60));
        println!("{:20} {}", "Total sessions:", sessions.len());
        println!();

        for s in &sessions {
            println!(
                "  {} {} [{}] - {} ({} users, {} comments)",
                ">".cyan(),
                s.name,
                s.status,
                s.project,
                s.users.len(),
                s.comments.len()
            );
            if detailed {
                for u in &s.users {
                    let status = if u.active { "active" } else { "offline" };
                    println!(
                        "    {} {} ({}) - {}",
                        "-".dimmed(),
                        u.username,
                        u.role,
                        status
                    );
                }
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
    fn test_generate_id() {
        let id = generate_id("session");
        assert!(id.starts_with("session-"));
        assert!(id.len() > 10);
    }

    #[test]
    fn test_db_roundtrip() {
        let db = CollabDb {
            version: 1,
            sessions: vec![SessionRecord {
                id: "session-001".to_string(),
                project: "test".to_string(),
                name: "Test Session".to_string(),
                owner: "alice".to_string(),
                max_users: 10,
                offline_enabled: false,
                created_at: "12345".to_string(),
                status: "active".to_string(),
                users: vec![UserRecord {
                    username: "alice".to_string(),
                    role: "owner".to_string(),
                    joined_at: "12345".to_string(),
                    active: true,
                }],
                comments: Vec::new(),
                shares: Vec::new(),
            }],
        };
        let json = serde_json::to_string(&db);
        assert!(json.is_ok());
        let parsed: Result<CollabDb, _> =
            serde_json::from_str(&json.expect("serialization should succeed"));
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_empty_db_default() {
        let db = CollabDb::default();
        assert_eq!(db.version, 0);
        assert!(db.sessions.is_empty());
    }

    #[test]
    fn test_comment_record_serialization() {
        let comment = CommentRecord {
            id: "c-001".to_string(),
            author: "bob".to_string(),
            message: "Fix color grading".to_string(),
            timecode: Some("01:02:03:04".to_string()),
            created_at: "999".to_string(),
        };
        let json = serde_json::to_string(&comment);
        assert!(json.is_ok());
        let s = json.expect("should serialize");
        assert!(s.contains("Fix color grading"));
        assert!(s.contains("01:02:03:04"));
    }

    #[test]
    fn test_share_record_serialization() {
        let share = ShareRecord {
            target: "bob@example.com".to_string(),
            permission: "editor".to_string(),
            shared_at: "12345".to_string(),
        };
        let json = serde_json::to_string(&share);
        assert!(json.is_ok());
    }
}
