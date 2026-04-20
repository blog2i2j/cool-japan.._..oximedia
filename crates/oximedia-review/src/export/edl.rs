//! Export review markers to EDL (Edit Decision List).

use crate::{comment::Comment, error::ReviewResult, SessionId};
use std::io::Write;

/// EDL format version.
#[derive(Debug, Clone, Copy)]
pub enum EdlVersion {
    /// CMX 3400.
    Cmx3400,
    /// CMX 3600.
    Cmx3600,
}

/// EDL export configuration.
pub struct EdlExportConfig {
    /// EDL format version.
    pub version: EdlVersion,
    /// Frame rate.
    pub frame_rate: f64,
    /// Title.
    pub title: Option<String>,
}

impl Default for EdlExportConfig {
    fn default() -> Self {
        Self {
            version: EdlVersion::Cmx3600,
            frame_rate: 24.0,
            title: None,
        }
    }
}

/// Export review markers to EDL.
///
/// # Errors
///
/// Returns error if export fails.
pub async fn export_to_edl(
    session_id: SessionId,
    comments: &[Comment],
    config: EdlExportConfig,
    output_path: &str,
) -> ReviewResult<()> {
    let mut file = std::fs::File::create(output_path)?;

    // Write EDL header
    writeln!(
        file,
        "TITLE: {}",
        config.title.unwrap_or_else(|| session_id.to_string())
    )?;
    writeln!(file, "FCM: NON-DROP FRAME")?;
    writeln!(file)?;

    // Write markers for each comment
    for (index, comment) in comments.iter().enumerate() {
        let event_number = index + 1;
        let timecode = frame_to_timecode(comment.frame, config.frame_rate);

        writeln!(
            file,
            "{event_number:03} BL V C {timecode} {timecode} {timecode} {timecode}"
        )?;

        writeln!(file, "* FROM CLIP NAME: REVIEW MARKER")?;
        writeln!(file, "* COMMENT: {}", comment.text)?;
        writeln!(file)?;
    }

    Ok(())
}

/// Convert frame number to timecode string.
fn frame_to_timecode(frame: i64, frame_rate: f64) -> String {
    let total_frames = frame.max(0) as f64;
    let hours = (total_frames / (frame_rate * 3600.0)).floor();
    let minutes = ((total_frames % (frame_rate * 3600.0)) / (frame_rate * 60.0)).floor();
    let seconds = ((total_frames % (frame_rate * 60.0)) / frame_rate).floor();
    let frames = total_frames % frame_rate;

    format!(
        "{:02}:{:02}:{:02}:{:02}",
        hours as u32, minutes as u32, seconds as u32, frames as u32
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_to_timecode() {
        let timecode = frame_to_timecode(0, 24.0);
        assert_eq!(timecode, "00:00:00:00");

        let timecode = frame_to_timecode(24, 24.0);
        assert_eq!(timecode, "00:00:01:00");

        let timecode = frame_to_timecode(1440, 24.0);
        assert_eq!(timecode, "00:01:00:00");
    }

    #[tokio::test]
    async fn test_export_to_edl() {
        let session_id = SessionId::new();
        let comments: Vec<Comment> = Vec::new();
        let config = EdlExportConfig::default();

        let temp_file = std::env::temp_dir()
            .join("oximedia-review-edl-test_markers.edl")
            .to_string_lossy()
            .into_owned();
        let result = export_to_edl(session_id, &comments, config, &temp_file).await;
        assert!(result.is_ok());

        // Clean up
        let _ = std::fs::remove_file(&temp_file);
    }

    #[test]
    fn test_edl_export_config_default() {
        let config = EdlExportConfig::default();
        assert!((config.frame_rate - 24.0).abs() < 0.001);
    }
}
