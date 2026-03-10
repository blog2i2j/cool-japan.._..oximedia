//! Caption authoring and editing tools

use crate::error::{CaptionError, Result};
use crate::types::{Caption, CaptionId, CaptionTrack, Duration, Timestamp};

/// Caption editor for professional authoring workflows
pub struct CaptionEditor {
    track: CaptionTrack,
    undo_stack: Vec<EditAction>,
    redo_stack: Vec<EditAction>,
}

/// Edit action for undo/redo
#[derive(Debug, Clone)]
enum EditAction {
    AddCaption(Caption),
    RemoveCaption(Caption),
    ModifyCaption {
        id: CaptionId,
        old: Caption,
        new: Caption,
    },
    ShiftTiming {
        ids: Vec<CaptionId>,
        offset: Duration,
    },
}

impl CaptionEditor {
    /// Create a new caption editor
    #[must_use]
    pub fn new(track: CaptionTrack) -> Self {
        Self {
            track,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        }
    }

    /// Get the caption track
    #[must_use]
    pub fn track(&self) -> &CaptionTrack {
        &self.track
    }

    /// Get a mutable reference to the track
    pub fn track_mut(&mut self) -> &mut CaptionTrack {
        &mut self.track
    }

    /// Add a new caption
    pub fn add_caption(&mut self, caption: Caption) -> Result<()> {
        self.undo_stack
            .push(EditAction::AddCaption(caption.clone()));
        self.redo_stack.clear();
        self.track.add_caption(caption)
    }

    /// Remove a caption by ID
    pub fn remove_caption(&mut self, id: CaptionId) -> Result<()> {
        if let Some(caption) = self.track.get_caption(id) {
            let caption = caption.clone();
            self.undo_stack
                .push(EditAction::RemoveCaption(caption.clone()));
            self.redo_stack.clear();
            self.track.remove_caption(id)
        } else {
            Err(CaptionError::CaptionNotFound(id.to_string()))
        }
    }

    /// Modify a caption
    pub fn modify_caption<F>(&mut self, id: CaptionId, modify_fn: F) -> Result<()>
    where
        F: FnOnce(&mut Caption),
    {
        if let Some(caption) = self.track.get_caption_mut(id) {
            let old = caption.clone();
            modify_fn(caption);
            let new = caption.clone();
            self.undo_stack
                .push(EditAction::ModifyCaption { id, old, new });
            self.redo_stack.clear();
            Ok(())
        } else {
            Err(CaptionError::CaptionNotFound(id.to_string()))
        }
    }

    /// Shift timing of multiple captions
    pub fn shift_timing(&mut self, ids: Vec<CaptionId>, offset: Duration) -> Result<()> {
        self.undo_stack.push(EditAction::ShiftTiming {
            ids: ids.clone(),
            offset: Duration::from_micros(-offset.as_micros()),
        });
        self.redo_stack.clear();

        for id in ids {
            if let Some(caption) = self.track.get_caption_mut(id) {
                caption.start = caption.start.add(offset);
                caption.end = caption.end.add(offset);
            }
        }
        Ok(())
    }

    /// Undo last action
    pub fn undo(&mut self) -> Result<()> {
        if let Some(action) = self.undo_stack.pop() {
            match action {
                EditAction::AddCaption(caption) => {
                    self.track.remove_caption(caption.id)?;
                    self.redo_stack.push(EditAction::AddCaption(caption));
                }
                EditAction::RemoveCaption(caption) => {
                    self.track.add_caption(caption.clone())?;
                    self.redo_stack.push(EditAction::RemoveCaption(caption));
                }
                EditAction::ModifyCaption { id, old, new } => {
                    if let Some(caption) = self.track.get_caption_mut(id) {
                        *caption = old.clone();
                        self.redo_stack.push(EditAction::ModifyCaption {
                            id,
                            old: new,
                            new: old,
                        });
                    }
                }
                EditAction::ShiftTiming { ids, offset } => {
                    for id in &ids {
                        if let Some(caption) = self.track.get_caption_mut(*id) {
                            caption.start = caption.start.add(offset);
                            caption.end = caption.end.add(offset);
                        }
                    }
                    self.redo_stack.push(EditAction::ShiftTiming {
                        ids,
                        offset: Duration::from_micros(-offset.as_micros()),
                    });
                }
            }
            Ok(())
        } else {
            Err(CaptionError::Other("No action to undo".to_string()))
        }
    }

    /// Redo last undone action
    pub fn redo(&mut self) -> Result<()> {
        if let Some(action) = self.redo_stack.pop() {
            match action {
                EditAction::AddCaption(caption) => {
                    self.track.add_caption(caption.clone())?;
                    self.undo_stack.push(EditAction::AddCaption(caption));
                }
                EditAction::RemoveCaption(caption) => {
                    self.track.remove_caption(caption.id)?;
                    self.undo_stack.push(EditAction::RemoveCaption(caption));
                }
                EditAction::ModifyCaption { id, old, new } => {
                    if let Some(caption) = self.track.get_caption_mut(id) {
                        *caption = new.clone();
                        self.undo_stack
                            .push(EditAction::ModifyCaption { id, old, new });
                    }
                }
                EditAction::ShiftTiming { ids, offset } => {
                    for id in &ids {
                        if let Some(caption) = self.track.get_caption_mut(*id) {
                            caption.start = caption.start.add(offset);
                            caption.end = caption.end.add(offset);
                        }
                    }
                    self.undo_stack
                        .push(EditAction::ShiftTiming { ids, offset });
                }
            }
            Ok(())
        } else {
            Err(CaptionError::Other("No action to redo".to_string()))
        }
    }

    /// Can undo?
    #[must_use]
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Can redo?
    #[must_use]
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }
}

/// Timing control utilities
pub struct TimingControl;

impl TimingControl {
    /// Snap caption timing to frame boundaries
    pub fn snap_to_frames(caption: &mut Caption, fps: f64) -> Result<()> {
        let frame_duration_micros = (1_000_000.0 / fps) as i64;

        let start_frames = caption.start.as_micros() / frame_duration_micros;
        caption.start = Timestamp::from_micros(start_frames * frame_duration_micros);

        let end_frames = caption.end.as_micros() / frame_duration_micros;
        caption.end = Timestamp::from_micros(end_frames * frame_duration_micros);

        Ok(())
    }

    /// Snap all captions in track to frames
    pub fn snap_track_to_frames(track: &mut CaptionTrack, fps: f64) -> Result<()> {
        for caption in &mut track.captions {
            Self::snap_to_frames(caption, fps)?;
        }
        Ok(())
    }

    /// Detect and fix overlapping captions
    pub fn fix_overlaps(
        track: &mut CaptionTrack,
        min_gap_frames: u32,
        fps: f64,
    ) -> Result<Vec<CaptionId>> {
        let frame_duration = Duration::from_micros((1_000_000.0 / fps) as i64);
        let min_gap = Duration::from_micros(frame_duration.as_micros() * i64::from(min_gap_frames));

        let mut fixed = Vec::new();

        for i in 0..track.captions.len().saturating_sub(1) {
            let next_start = track.captions[i + 1].start;
            if track.captions[i].end > next_start.sub(min_gap) {
                // Overlap detected, fix it
                track.captions[i].end = next_start.sub(min_gap);
                fixed.push(track.captions[i].id);
            }
        }

        Ok(fixed)
    }

    /// Ripple edit: move all captions after a point by an offset
    pub fn ripple_edit(track: &mut CaptionTrack, from: Timestamp, offset: Duration) -> Result<()> {
        for caption in &mut track.captions {
            if caption.start >= from {
                caption.start = caption.start.add(offset);
                caption.end = caption.end.add(offset);
            }
        }
        Ok(())
    }
}

/// Line breaking utilities for caption text
pub struct LineBreaker;

impl LineBreaker {
    /// Break text into lines with maximum characters per line
    #[must_use]
    pub fn break_lines(text: &str, max_chars_per_line: usize) -> Vec<String> {
        let mut lines = Vec::new();
        let mut current_line = String::new();

        for word in text.split_whitespace() {
            if current_line.is_empty() {
                current_line = word.to_string();
            } else {
                let test_line = format!("{current_line} {word}");
                if test_line.len() <= max_chars_per_line {
                    current_line = test_line;
                } else {
                    lines.push(current_line);
                    current_line = word.to_string();
                }
            }
        }

        if !current_line.is_empty() {
            lines.push(current_line);
        }

        lines
    }

    /// Smart line breaking that tries to balance line lengths
    #[must_use]
    pub fn balanced_break(text: &str, max_chars_per_line: usize, max_lines: usize) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let total_chars: usize = words.iter().map(|w| w.len()).sum::<usize>() + words.len() - 1;

        if total_chars <= max_chars_per_line {
            return vec![text.to_string()];
        }

        // Try to balance lines
        let ideal_chars_per_line = total_chars / max_lines.min(2);
        let mut lines = Vec::new();
        let mut current_line = String::new();

        for word in words {
            if current_line.is_empty() {
                current_line = word.to_string();
            } else {
                let test_line = format!("{current_line} {word}");
                if test_line.len() <= max_chars_per_line
                    && (lines.is_empty() && test_line.len() <= ideal_chars_per_line
                        || !lines.is_empty())
                {
                    current_line = test_line;
                } else if test_line.len() > max_chars_per_line {
                    lines.push(current_line);
                    current_line = word.to_string();
                    if lines.len() >= max_lines - 1 {
                        break;
                    }
                } else {
                    lines.push(current_line);
                    current_line = word.to_string();
                }
            }
        }

        if !current_line.is_empty() {
            lines.push(current_line);
        }

        lines
    }

    /// Apply line breaks to a caption
    pub fn apply_to_caption(caption: &mut Caption, max_chars_per_line: usize, max_lines: usize) {
        let lines = Self::balanced_break(&caption.text, max_chars_per_line, max_lines);
        caption.text = lines.join("\n");
    }
}

/// Reading speed calculator
pub struct ReadingSpeed;

impl ReadingSpeed {
    /// Calculate reading speed in words per minute
    #[must_use]
    pub fn calculate_wpm(caption: &Caption) -> f64 {
        caption.reading_speed_wpm()
    }

    /// Check if reading speed is acceptable (160-180 WPM max recommended)
    #[must_use]
    pub fn is_acceptable(caption: &Caption, max_wpm: f64) -> bool {
        Self::calculate_wpm(caption) <= max_wpm
    }

    /// Suggest minimum duration for text at a given reading speed
    #[must_use]
    pub fn suggest_duration(text: &str, target_wpm: f64) -> Duration {
        let word_count = text.split_whitespace().count() as f64;
        let seconds = (word_count / target_wpm) * 60.0;
        Duration::from_micros((seconds * 1_000_000.0) as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Language;

    #[test]
    fn test_editor_undo_redo() {
        let track = CaptionTrack::new(Language::english());
        let mut editor = CaptionEditor::new(track);

        let caption = Caption::new(
            Timestamp::from_secs(1),
            Timestamp::from_secs(3),
            "Test".to_string(),
        );
        let _id = caption.id;

        editor
            .add_caption(caption)
            .expect("adding caption should succeed");
        assert_eq!(editor.track().count(), 1);
        assert!(editor.can_undo());

        editor.undo().expect("undo should succeed");
        assert_eq!(editor.track().count(), 0);
        assert!(editor.can_redo());

        editor.redo().expect("redo should succeed");
        assert_eq!(editor.track().count(), 1);
    }

    #[test]
    fn test_snap_to_frames() {
        let mut caption = Caption::new(
            Timestamp::from_micros(1_000_123),
            Timestamp::from_micros(3_000_456),
            "Test".to_string(),
        );

        TimingControl::snap_to_frames(&mut caption, 25.0).expect("snap to frames should succeed");

        // Should be snapped to frame boundaries
        assert_eq!(caption.start.as_micros() % 40_000, 0); // 25 fps = 40ms per frame
        assert_eq!(caption.end.as_micros() % 40_000, 0);
    }

    #[test]
    fn test_line_breaking() {
        let text = "This is a long caption that should be broken into multiple lines";
        let lines = LineBreaker::break_lines(text, 30);

        assert!(lines.len() > 1);
        for line in &lines {
            assert!(line.len() <= 30);
        }
    }

    #[test]
    fn test_reading_speed() {
        let caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(10),
            "This is a test caption with ten words here".to_string(),
        );

        let wpm = ReadingSpeed::calculate_wpm(&caption);
        assert!((wpm - 54.0).abs() < 5.0); // ~9 words in 10 seconds = ~54 WPM
    }
}
