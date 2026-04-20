//! Export review data to CSV.

use crate::{comment::Comment, error::ReviewResult, task::Task, SessionId};
use std::io::Write;

/// Export comments to CSV.
///
/// # Errors
///
/// Returns error if export fails.
pub async fn export_to_csv(
    session_id: SessionId,
    comments: &[Comment],
    output_path: &str,
) -> ReviewResult<()> {
    let mut file = std::fs::File::create(output_path)?;

    // Write CSV header
    writeln!(file, "ID,Frame,Type,Status,Priority,Author,Text,Created At")?;

    // Write comment rows
    for comment in comments {
        writeln!(
            file,
            "{},{},{:?},{:?},{:?},{},\"{}\",{}",
            comment.id,
            comment.frame,
            comment.annotation_type,
            comment.status,
            comment.priority,
            comment.author.name,
            comment.text.replace('\"', "\"\""),
            comment.created_at.format("%Y-%m-%d %H:%M:%S")
        )?;
    }

    let _ = session_id;
    Ok(())
}

/// Export tasks to CSV.
///
/// # Errors
///
/// Returns error if export fails.
pub async fn export_tasks_to_csv(
    session_id: SessionId,
    tasks: &[Task],
    output_path: &str,
) -> ReviewResult<()> {
    let mut file = std::fs::File::create(output_path)?;

    // Write CSV header
    writeln!(
        file,
        "ID,Title,Status,Priority,Assignee,Creator,Created At,Completed At"
    )?;

    // Write task rows
    for task in tasks {
        writeln!(
            file,
            "{},{},{:?},{:?},{},{},{},{}",
            task.id,
            task.title,
            task.status,
            task.priority,
            task.assignee.name,
            task.creator.name,
            task.created_at.format("%Y-%m-%d %H:%M:%S"),
            task.completed_at.map_or_else(
                || "N/A".to_string(),
                |dt| dt.format("%Y-%m-%d %H:%M:%S").to_string()
            )
        )?;
    }

    let _ = session_id;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-review-csv-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    #[tokio::test]
    async fn test_export_to_csv() {
        let session_id = SessionId::new();
        let comments: Vec<Comment> = Vec::new();

        let temp_file = tmp_str("test_comments.csv");
        let result = export_to_csv(session_id, &comments, &temp_file).await;
        assert!(result.is_ok());

        // Clean up
        let _ = std::fs::remove_file(&temp_file);
    }

    #[tokio::test]
    async fn test_export_tasks_to_csv() {
        let session_id = SessionId::new();
        let tasks: Vec<Task> = Vec::new();

        let temp_file = tmp_str("test_tasks.csv");
        let result = export_tasks_to_csv(session_id, &tasks, &temp_file).await;
        assert!(result.is_ok());

        // Clean up
        let _ = std::fs::remove_file(&temp_file);
    }
}
