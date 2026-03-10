//! Export markers to various formats.

use super::Marker;
use crate::error::{ClipError, ClipResult};
use oximedia_core::types::Rational;

/// Exports markers to EDL format.
pub struct MarkerExporter;

impl MarkerExporter {
    /// Exports markers to EDL marker format.
    ///
    /// # Errors
    ///
    /// Returns an error if the export fails.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::format_push_string
    )]
    pub fn to_edl(markers: &[Marker], frame_rate: Rational) -> ClipResult<String> {
        let mut output = String::new();

        for marker in markers {
            let timecode = Self::frames_to_timecode(marker.frame, frame_rate)?;
            output.push_str(&format!(
                "{} {} {}\n",
                timecode, marker.name, marker.marker_type
            ));
        }

        Ok(output)
    }

    /// Exports markers to CSV format.
    ///
    /// # Errors
    ///
    /// Returns an error if the export fails.
    pub fn to_csv(markers: &[Marker], frame_rate: Rational) -> ClipResult<String> {
        let mut output = String::from("Frame,Timecode,Type,Name,Comment\n");

        for marker in markers {
            let timecode = Self::frames_to_timecode(marker.frame, frame_rate)?;
            let comment = marker.comment.as_deref().unwrap_or("");
            output.push_str(&format!(
                "{},{},{},{},{}\n",
                marker.frame, timecode, marker.marker_type, marker.name, comment
            ));
        }

        Ok(output)
    }

    /// Exports markers to JSON format.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(markers: &[Marker]) -> ClipResult<String> {
        serde_json::to_string_pretty(markers).map_err(|e| ClipError::Serialization(e.to_string()))
    }

    /// Converts frames to timecode string.
    #[allow(clippy::unnecessary_wraps)]
    fn frames_to_timecode(frame: i64, frame_rate: Rational) -> ClipResult<String> {
        let fps = frame_rate.to_f64();
        let total_seconds = frame as f64 / fps;

        let hours = (total_seconds / 3600.0) as i64;
        let minutes = ((total_seconds % 3600.0) / 60.0) as i64;
        let seconds = (total_seconds % 60.0) as i64;
        let frames = frame % frame_rate.num;

        Ok(format!("{hours:02}:{minutes:02}:{seconds:02}:{frames:02}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frames_to_timecode() {
        let frame_rate = Rational::new(24, 1);
        let timecode = MarkerExporter::frames_to_timecode(100, frame_rate)
            .expect("frames_to_timecode should succeed");
        assert_eq!(timecode, "00:00:04:04");
    }

    #[test]
    fn test_to_csv() {
        let marker = Marker::chapter(100, "Chapter 1");
        let markers = vec![marker];
        let frame_rate = Rational::new(24, 1);

        let csv = MarkerExporter::to_csv(&markers, frame_rate).expect("to_csv should succeed");
        assert!(csv.contains("Frame,Timecode,Type,Name,Comment"));
        assert!(csv.contains("100,00:00:04:04,Chapter,Chapter 1"));
    }
}
