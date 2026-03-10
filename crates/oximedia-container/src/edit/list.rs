//! MP4 edit list support.
//!
//! Provides edit lists for frame-accurate editing in MP4 containers.

#![forbid(unsafe_code)]

use oximedia_core::{OxiError, OxiResult};

/// An edit list entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EditEntry {
    /// Segment duration in movie timescale.
    pub segment_duration: i64,
    /// Media time (start time in media timescale).
    /// -1 means empty edit (dwell/pause).
    pub media_time: i64,
    /// Media rate in 16.16 fixed point.
    /// `0x0001_0000` (1.0) is normal playback.
    pub media_rate: i32,
}

impl EditEntry {
    /// Creates a new edit entry.
    #[must_use]
    pub const fn new(segment_duration: i64, media_time: i64, media_rate: i32) -> Self {
        Self {
            segment_duration,
            media_time,
            media_rate,
        }
    }

    /// Creates an empty edit (pause/dwell).
    #[must_use]
    pub const fn empty(duration: i64) -> Self {
        Self {
            segment_duration: duration,
            media_time: -1,
            media_rate: 0x0001_0000,
        }
    }

    /// Creates a normal playback edit.
    #[must_use]
    pub const fn normal(segment_duration: i64, media_time: i64) -> Self {
        Self {
            segment_duration,
            media_time,
            media_rate: 0x0001_0000, // 1.0 in 16.16 fixed point
        }
    }

    /// Returns true if this is an empty edit.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.media_time == -1
    }

    /// Returns the playback rate as a float.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn rate(&self) -> f64 {
        f64::from(self.media_rate) / 65536.0
    }

    /// Sets the playback rate from a float.
    #[allow(clippy::cast_possible_truncation)]
    pub fn set_rate(&mut self, rate: f64) {
        self.media_rate = (rate * 65536.0) as i32;
    }
}

/// Edit list for a track.
#[derive(Debug, Clone)]
pub struct EditList {
    entries: Vec<EditEntry>,
    movie_timescale: u32,
    media_timescale: u32,
}

impl EditList {
    /// Creates a new edit list.
    #[must_use]
    pub const fn new(movie_timescale: u32, media_timescale: u32) -> Self {
        Self {
            entries: Vec::new(),
            movie_timescale,
            media_timescale,
        }
    }

    /// Adds an edit entry.
    pub fn add_entry(&mut self, entry: EditEntry) -> &mut Self {
        self.entries.push(entry);
        self
    }

    /// Adds an empty edit (pause/dwell).
    pub fn add_empty(&mut self, duration: i64) -> &mut Self {
        self.add_entry(EditEntry::empty(duration))
    }

    /// Adds a normal playback edit.
    pub fn add_normal(&mut self, segment_duration: i64, media_time: i64) -> &mut Self {
        self.add_entry(EditEntry::normal(segment_duration, media_time))
    }

    /// Returns all edit entries.
    #[must_use]
    pub fn entries(&self) -> &[EditEntry] {
        &self.entries
    }

    /// Returns the movie timescale.
    #[must_use]
    pub const fn movie_timescale(&self) -> u32 {
        self.movie_timescale
    }

    /// Returns the media timescale.
    #[must_use]
    pub const fn media_timescale(&self) -> u32 {
        self.media_timescale
    }

    /// Returns the total duration of the edit list in movie timescale.
    #[must_use]
    pub fn total_duration(&self) -> i64 {
        self.entries.iter().map(|e| e.segment_duration).sum()
    }

    /// Clears all edit entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns the number of edit entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the edit list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Converts a presentation timestamp to media timestamp.
    #[must_use]
    pub fn presentation_to_media_time(&self, presentation_time: i64) -> Option<i64> {
        let mut current_time = 0i64;

        for entry in &self.entries {
            let next_time = current_time + entry.segment_duration;

            if presentation_time >= current_time && presentation_time < next_time {
                if entry.is_empty() {
                    return None; // Empty edit
                }

                let offset = presentation_time - current_time;
                #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
                let media_offset = ((offset as f64) * f64::from(self.media_timescale)
                    / f64::from(self.movie_timescale)
                    * entry.rate()) as i64;
                return Some(entry.media_time + media_offset);
            }

            current_time = next_time;
        }

        None
    }

    /// Validates the edit list.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the timescales are zero or if any entry has a non-positive duration.
    pub fn validate(&self) -> OxiResult<()> {
        if self.movie_timescale == 0 {
            return Err(OxiError::InvalidData(
                "Movie timescale cannot be zero".into(),
            ));
        }

        if self.media_timescale == 0 {
            return Err(OxiError::InvalidData(
                "Media timescale cannot be zero".into(),
            ));
        }

        for (i, entry) in self.entries.iter().enumerate() {
            if entry.segment_duration <= 0 {
                return Err(OxiError::InvalidData(format!(
                    "Edit entry {i} has non-positive duration"
                )));
            }
        }

        Ok(())
    }
}

/// Builder for creating complex edit lists.
pub struct EditListBuilder {
    edit_list: EditList,
}

impl EditListBuilder {
    /// Creates a new edit list builder.
    #[must_use]
    pub const fn new(movie_timescale: u32, media_timescale: u32) -> Self {
        Self {
            edit_list: EditList::new(movie_timescale, media_timescale),
        }
    }

    /// Adds a clip from the media.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the clip parameters are invalid.
    pub fn add_clip(&mut self, start_time: i64, duration: i64, rate: f64) -> OxiResult<&mut Self> {
        let mut entry = EditEntry::normal(duration, start_time);
        entry.set_rate(rate);
        self.edit_list.add_entry(entry);
        Ok(self)
    }

    /// Adds a pause/dwell.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the pause duration is invalid.
    pub fn add_pause(&mut self, duration: i64) -> OxiResult<&mut Self> {
        self.edit_list.add_empty(duration);
        Ok(self)
    }

    /// Trims the start of the media.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the trim parameters are invalid.
    pub fn trim_start(&mut self, trim_duration: i64) -> OxiResult<&mut Self> {
        // Add an edit that starts playback after the trim point
        self.add_clip(trim_duration, i64::MAX, 1.0)
    }

    /// Trims the end of the media.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the trim parameters are invalid.
    pub fn trim_end(&mut self, total_duration: i64, trim_duration: i64) -> OxiResult<&mut Self> {
        // Add an edit that plays until the trim point
        self.add_clip(0, total_duration - trim_duration, 1.0)
    }

    /// Builds the edit list.
    #[must_use]
    pub fn build(self) -> EditList {
        self.edit_list
    }
}

/// Common edit list patterns.
pub struct EditListPresets;

impl EditListPresets {
    /// Creates an edit list that trims the start.
    #[must_use]
    pub fn trim_start(trim_ms: i64, movie_timescale: u32, media_timescale: u32) -> EditList {
        let mut list = EditList::new(movie_timescale, media_timescale);

        #[allow(clippy::cast_possible_truncation)]
        let media_time = (trim_ms * i64::from(media_timescale)) / 1000;

        list.add_normal(i64::MAX, media_time);
        list
    }

    /// Creates an edit list that trims the end.
    #[must_use]
    pub fn trim_end(
        duration_ms: i64,
        trim_ms: i64,
        movie_timescale: u32,
        media_timescale: u32,
    ) -> EditList {
        let mut list = EditList::new(movie_timescale, media_timescale);

        #[allow(clippy::cast_possible_truncation)]
        let segment_duration = ((duration_ms - trim_ms) * i64::from(movie_timescale)) / 1000;

        list.add_normal(segment_duration, 0);
        list
    }

    /// Creates an edit list with a pause at the start.
    #[must_use]
    pub fn pause_start(pause_ms: i64, movie_timescale: u32, media_timescale: u32) -> EditList {
        let mut list = EditList::new(movie_timescale, media_timescale);

        #[allow(clippy::cast_possible_truncation)]
        let pause_duration = (pause_ms * i64::from(movie_timescale)) / 1000;

        list.add_empty(pause_duration);
        list.add_normal(i64::MAX, 0);
        list
    }

    /// Creates an edit list for slow motion playback.
    #[must_use]
    pub fn slow_motion(rate: f64, movie_timescale: u32, media_timescale: u32) -> EditList {
        let mut list = EditList::new(movie_timescale, media_timescale);
        let mut entry = EditEntry::normal(i64::MAX, 0);
        entry.set_rate(rate);
        list.add_entry(entry);
        list
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edit_entry() {
        let entry = EditEntry::normal(1000, 500);
        assert_eq!(entry.segment_duration, 1000);
        assert_eq!(entry.media_time, 500);
        assert!(!entry.is_empty());
        assert_eq!(entry.rate(), 1.0);

        let empty = EditEntry::empty(1000);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_edit_entry_rate() {
        let mut entry = EditEntry::normal(1000, 0);
        entry.set_rate(2.0);
        assert!((entry.rate() - 2.0).abs() < 0.01);

        entry.set_rate(0.5);
        assert!((entry.rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_edit_list() {
        let mut list = EditList::new(1000, 48000);
        list.add_normal(1000, 0);
        list.add_empty(500);
        list.add_normal(2000, 48000);

        assert_eq!(list.len(), 3);
        assert_eq!(list.total_duration(), 3500);
        assert!(!list.is_empty());
    }

    #[test]
    fn test_edit_list_validation() {
        let list = EditList::new(1000, 48000);
        assert!(list.validate().is_ok());

        let invalid = EditList::new(0, 48000);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_edit_list_builder() {
        let mut builder = EditListBuilder::new(1000, 48000);
        builder
            .add_clip(0, 1000, 1.0)
            .expect("operation should succeed");
        builder.add_pause(500).expect("operation should succeed");
        builder
            .add_clip(48000, 2000, 0.5)
            .expect("operation should succeed");

        let list = builder.build();
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn test_presentation_to_media_time() {
        let mut list = EditList::new(1000, 48000);
        list.add_normal(1000, 48000);

        let media_time = list.presentation_to_media_time(500);
        assert!(media_time.is_some());

        // Empty edit returns None
        list.clear();
        list.add_empty(1000);
        assert!(list.presentation_to_media_time(500).is_none());
    }

    #[test]
    fn test_edit_list_presets() {
        let trim_start = EditListPresets::trim_start(1000, 1000, 48000);
        assert_eq!(trim_start.len(), 1);

        let trim_end = EditListPresets::trim_end(10000, 1000, 1000, 48000);
        assert_eq!(trim_end.len(), 1);

        let pause = EditListPresets::pause_start(500, 1000, 48000);
        assert_eq!(pause.len(), 2);

        let slow_mo = EditListPresets::slow_motion(0.5, 1000, 48000);
        assert_eq!(slow_mo.len(), 1);
    }
}
