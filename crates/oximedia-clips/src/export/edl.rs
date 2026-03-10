//! EDL export functionality.

use crate::clip::Clip;
use crate::error::ClipResult;
use oximedia_core::types::Rational;

/// EDL exporter for clips.
#[derive(Debug, Clone)]
pub struct EdlExporter {
    frame_rate: Rational,
}

impl EdlExporter {
    /// Creates a new EDL exporter.
    #[must_use]
    pub fn new(frame_rate: Rational) -> Self {
        Self { frame_rate }
    }

    /// Exports clips to CMX 3600 EDL format.
    ///
    /// # Errors
    ///
    /// Returns an error if the export fails.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::format_push_string
    )]
    pub fn to_edl(&self, clips: &[Clip]) -> ClipResult<String> {
        let mut output = String::from("TITLE: Clip Export\n");
        output.push_str("FCM: NON-DROP FRAME\n\n");

        for (index, clip) in clips.iter().enumerate() {
            let event_num = index + 1;
            let in_point = clip.in_point.unwrap_or(0);
            let out_point = clip.out_point.or(clip.duration).unwrap_or(0);

            let src_in = self.frames_to_timecode(in_point)?;
            let src_out = self.frames_to_timecode(out_point)?;
            let rec_in = self.frames_to_timecode(0)?;
            let rec_out = self.frames_to_timecode(out_point - in_point)?;

            output.push_str(&format!(
                "{event_num:03}  001      V     C        {src_in} {src_out} {rec_in} {rec_out}\n"
            ));

            output.push_str(&format!("* FROM CLIP NAME: {}\n", clip.name));

            if let Some(file_name) = clip.file_path.file_name() {
                output.push_str(&format!("* SOURCE FILE: {}\n", file_name.to_string_lossy()));
            }

            output.push('\n');
        }

        Ok(output)
    }

    #[allow(clippy::unnecessary_wraps)]
    fn frames_to_timecode(&self, frame: i64) -> ClipResult<String> {
        let fps = self.frame_rate.to_f64();
        let total_seconds = frame as f64 / fps;

        let hours = (total_seconds / 3600.0) as i64;
        let minutes = ((total_seconds % 3600.0) / 60.0) as i64;
        let seconds = (total_seconds % 60.0) as i64;
        let frames = frame % self.frame_rate.num;

        Ok(format!("{hours:02}:{minutes:02}:{seconds:02}:{frames:02}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_edl_export() {
        let mut clip = Clip::new(PathBuf::from("/test.mov"));
        clip.set_name("Test Clip");
        clip.set_duration(1000);
        clip.set_in_point(0);
        clip.set_out_point(100);

        let clips = vec![clip];
        let exporter = EdlExporter::new(Rational::new(24, 1));
        let edl = exporter.to_edl(&clips).expect("to_edl should succeed");

        assert!(edl.contains("TITLE: Clip Export"));
        assert!(edl.contains("001  001"));
        assert!(edl.contains("Test Clip"));
    }

    #[test]
    fn test_frames_to_timecode() {
        let exporter = EdlExporter::new(Rational::new(24, 1));
        let tc = exporter
            .frames_to_timecode(100)
            .expect("frames_to_timecode should succeed");
        assert_eq!(tc, "00:00:04:04");
    }
}
