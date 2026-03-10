//! Caption merging and splitting utilities

use crate::error::{CaptionError, Result};
use crate::types::{Caption, CaptionTrack, Duration, Timestamp};

/// Caption merger for combining adjacent captions
pub struct CaptionMerger {
    /// Maximum characters in merged caption
    max_chars: usize,
    /// Maximum duration (milliseconds)
    max_duration_ms: i64,
    /// Maximum gap between captions to merge (milliseconds)
    max_gap_ms: i64,
}

impl CaptionMerger {
    /// Create a new caption merger
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_chars: 84, // 2 lines x 42 chars
            max_duration_ms: 7000,
            max_gap_ms: 2000,
        }
    }

    /// Set maximum characters
    #[must_use]
    pub fn with_max_chars(mut self, max_chars: usize) -> Self {
        self.max_chars = max_chars;
        self
    }

    /// Set maximum duration
    #[must_use]
    pub fn with_max_duration(mut self, duration_ms: i64) -> Self {
        self.max_duration_ms = duration_ms;
        self
    }

    /// Set maximum gap
    #[must_use]
    pub fn with_max_gap(mut self, gap_ms: i64) -> Self {
        self.max_gap_ms = gap_ms;
        self
    }

    /// Merge consecutive captions in a track
    pub fn merge_track(&self, track: &mut CaptionTrack) -> Result<usize> {
        let mut merged_captions = Vec::new();
        let mut merge_count = 0;
        let mut i = 0;

        while i < track.captions.len() {
            let mut current = track.captions[i].clone();
            let mut j = i + 1;

            // Try to merge with following captions
            while j < track.captions.len() {
                let next = &track.captions[j];

                if !self.should_merge(&current, next) {
                    break;
                }

                // Merge
                let merged_text = if current.text.ends_with('.')
                    || current.text.ends_with('!')
                    || current.text.ends_with('?')
                {
                    format!("{} {}", current.text, next.text)
                } else {
                    format!("{} {}", current.text, next.text)
                };

                current.text = merged_text;
                current.end = next.end;
                merge_count += 1;
                j += 1;
            }

            merged_captions.push(current);
            i = j;
        }

        track.captions = merged_captions;
        Ok(merge_count)
    }

    fn should_merge(&self, current: &Caption, next: &Caption) -> bool {
        // Check gap
        let gap_ms = next.start.as_millis() - current.end.as_millis();
        if gap_ms > self.max_gap_ms {
            return false;
        }

        // Check merged text length
        let merged_len = current.text.len() + 1 + next.text.len();
        if merged_len > self.max_chars {
            return false;
        }

        // Check merged duration
        let merged_duration = next.end.as_millis() - current.start.as_millis();
        if merged_duration > self.max_duration_ms {
            return false;
        }

        // Check if same speaker
        if current.speaker != next.speaker {
            return false;
        }

        true
    }

    /// Merge specific captions by ID
    pub fn merge_captions(&self, captions: &[Caption]) -> Result<Caption> {
        if captions.is_empty() {
            return Err(CaptionError::Other("No captions to merge".to_string()));
        }

        if captions.len() == 1 {
            return Ok(captions[0].clone());
        }

        let start = captions[0].start;
        let end = captions
            .last()
            .expect("captions is non-empty: length was checked above")
            .end;

        let texts: Vec<&str> = captions.iter().map(|c| c.text.as_str()).collect();
        let merged_text = texts.join(" ");

        // Check constraints
        if merged_text.len() > self.max_chars {
            return Err(CaptionError::Other(format!(
                "Merged caption would exceed max chars: {} > {}",
                merged_text.len(),
                self.max_chars
            )));
        }

        let duration_ms = end.as_millis() - start.as_millis();
        if duration_ms > self.max_duration_ms {
            return Err(CaptionError::Other(format!(
                "Merged caption would exceed max duration: {}ms > {}ms",
                duration_ms, self.max_duration_ms
            )));
        }

        let mut merged = Caption::new(start, end, merged_text);
        merged.style = captions[0].style.clone();
        merged.position = captions[0].position;

        Ok(merged)
    }
}

impl Default for CaptionMerger {
    fn default() -> Self {
        Self::new()
    }
}

/// Caption splitter for dividing long captions
#[allow(dead_code)]
pub struct CaptionSplitter {
    /// Maximum characters per caption
    max_chars: usize,
    /// Maximum duration per caption (milliseconds)
    max_duration_ms: i64,
    /// Minimum duration per caption (milliseconds)
    min_duration_ms: i64,
    /// Split on sentence boundaries
    split_on_sentences: bool,
}

impl CaptionSplitter {
    /// Create a new caption splitter
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_chars: 84,
            max_duration_ms: 7000,
            min_duration_ms: 1000,
            split_on_sentences: true,
        }
    }

    /// Set maximum characters
    #[must_use]
    pub fn with_max_chars(mut self, max_chars: usize) -> Self {
        self.max_chars = max_chars;
        self
    }

    /// Set maximum duration
    #[must_use]
    pub fn with_max_duration(mut self, duration_ms: i64) -> Self {
        self.max_duration_ms = duration_ms;
        self
    }

    /// Set whether to split on sentence boundaries
    #[must_use]
    pub fn with_sentence_splitting(mut self, enabled: bool) -> Self {
        self.split_on_sentences = enabled;
        self
    }

    /// Split long captions in a track
    pub fn split_track(&self, track: &mut CaptionTrack) -> Result<usize> {
        let mut new_captions = Vec::new();
        let mut split_count = 0;

        for caption in &track.captions {
            if self.should_split(caption) {
                let splits = self.split_caption(caption)?;
                split_count += splits.len() - 1;
                new_captions.extend(splits);
            } else {
                new_captions.push(caption.clone());
            }
        }

        track.captions = new_captions;
        Ok(split_count)
    }

    fn should_split(&self, caption: &Caption) -> bool {
        if caption.text.len() > self.max_chars {
            return true;
        }

        let duration_ms = caption.duration().as_millis();
        if duration_ms > self.max_duration_ms {
            return true;
        }

        false
    }

    /// Split a single caption into multiple parts
    pub fn split_caption(&self, caption: &Caption) -> Result<Vec<Caption>> {
        let text = &caption.text;
        let duration = caption.duration();

        let parts = if self.split_on_sentences {
            self.split_by_sentences(text)
        } else {
            self.split_by_length(text)
        };

        if parts.len() == 1 {
            return Ok(vec![caption.clone()]);
        }

        let mut result = Vec::new();
        let part_duration_ms = duration.as_millis() / parts.len() as i64;

        for (i, part) in parts.iter().enumerate() {
            let start = caption
                .start
                .add(Duration::from_millis(i as i64 * part_duration_ms));
            let end = if i == parts.len() - 1 {
                caption.end
            } else {
                start.add(Duration::from_millis(part_duration_ms))
            };

            let mut new_caption = Caption::new(start, end, part.clone());
            new_caption.style = caption.style.clone();
            new_caption.position = caption.position;
            new_caption.speaker = caption.speaker.clone();

            result.push(new_caption);
        }

        Ok(result)
    }

    fn split_by_sentences(&self, text: &str) -> Vec<String> {
        let mut parts = Vec::new();
        let mut current = String::new();

        for sentence in text.split_inclusive(&['.', '!', '?']) {
            let test = if current.is_empty() {
                sentence.to_string()
            } else {
                format!("{current} {sentence}")
            };

            if test.len() > self.max_chars {
                if !current.is_empty() {
                    parts.push(current);
                }
                current = sentence.to_string();
            } else {
                current = test;
            }
        }

        if !current.is_empty() {
            parts.push(current);
        }

        if parts.is_empty() {
            parts.push(text.to_string());
        }

        parts
    }

    fn split_by_length(&self, text: &str) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut parts = Vec::new();
        let mut current = String::new();

        for word in words {
            let test = if current.is_empty() {
                word.to_string()
            } else {
                format!("{current} {word}")
            };

            if test.len() > self.max_chars {
                if !current.is_empty() {
                    parts.push(current);
                }
                current = word.to_string();
            } else {
                current = test;
            }
        }

        if !current.is_empty() {
            parts.push(current);
        }

        parts
    }

    /// Split at a specific timestamp
    pub fn split_at_time(
        &self,
        caption: &Caption,
        split_time: Timestamp,
    ) -> Result<(Caption, Caption)> {
        if split_time <= caption.start || split_time >= caption.end {
            return Err(CaptionError::Other(
                "Split time must be within caption duration".to_string(),
            ));
        }

        // Estimate where to split the text based on timing
        let total_duration = caption.duration().as_millis();
        let split_offset = split_time.duration_since(caption.start).as_millis();
        let split_ratio = split_offset as f64 / total_duration as f64;

        let words: Vec<&str> = caption.text.split_whitespace().collect();
        let split_index = ((words.len() as f64) * split_ratio).round() as usize;
        let split_index = split_index.clamp(1, words.len() - 1);

        let first_text = words[..split_index].join(" ");
        let second_text = words[split_index..].join(" ");

        let mut first = Caption::new(caption.start, split_time, first_text);
        first.style = caption.style.clone();
        first.position = caption.position;
        first.speaker = caption.speaker.clone();

        let mut second = Caption::new(split_time, caption.end, second_text);
        second.style = caption.style.clone();
        second.position = caption.position;
        second.speaker = caption.speaker.clone();

        Ok((first, second))
    }
}

impl Default for CaptionSplitter {
    fn default() -> Self {
        Self::new()
    }
}

/// Smart merge/split based on reading speed
pub struct SmartMergeSplit {
    target_wpm: f64,
    tolerance_wpm: f64,
}

impl SmartMergeSplit {
    /// Create a new smart merge/split processor
    #[must_use]
    pub fn new(target_wpm: f64) -> Self {
        Self {
            target_wpm,
            tolerance_wpm: 20.0,
        }
    }

    /// Process a track to optimize reading speed
    pub fn optimize_track(&self, track: &mut CaptionTrack) -> Result<(usize, usize)> {
        let merger = CaptionMerger::new();
        let splitter = CaptionSplitter::new();

        let mut merge_count = 0;
        let mut split_count = 0;

        // First pass: identify captions that need adjustment
        let mut i = 0;
        while i < track.captions.len() {
            let wpm = track.captions[i].reading_speed_wpm();

            if wpm > self.target_wpm + self.tolerance_wpm {
                // Too fast, try to split
                if let Ok(splits) = splitter.split_caption(&track.captions[i]) {
                    track.captions.remove(i);
                    for (j, split) in splits.into_iter().enumerate() {
                        track.captions.insert(i + j, split);
                    }
                    split_count += 1;
                }
            } else if wpm < self.target_wpm - self.tolerance_wpm {
                // Too slow, try to merge with next
                if i + 1 < track.captions.len() {
                    let next = track.captions[i + 1].clone();
                    if merger.should_merge(&track.captions[i], &next) {
                        if let Ok(merged) =
                            merger.merge_captions(&[track.captions[i].clone(), next])
                        {
                            track.captions[i] = merged;
                            track.captions.remove(i + 1);
                            merge_count += 1;
                        }
                    }
                }
            }

            i += 1;
        }

        Ok((merge_count, split_count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Language;

    #[test]
    fn test_caption_merger() {
        let merger = CaptionMerger::new();

        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(0),
                Timestamp::from_secs(2),
                "First caption".to_string(),
            ))
            .expect("operation should succeed in test");
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(2),
                Timestamp::from_secs(4),
                "Second caption".to_string(),
            ))
            .expect("operation should succeed in test");

        let count = merger
            .merge_track(&mut track)
            .expect("merge track should succeed");
        assert_eq!(count, 1);
        assert_eq!(track.captions.len(), 1);
        assert!(track.captions[0].text.contains("First"));
        assert!(track.captions[0].text.contains("Second"));
    }

    #[test]
    fn test_caption_splitter() {
        let splitter = CaptionSplitter::new()
            .with_max_chars(20)
            .with_sentence_splitting(false);

        let caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(10),
            "This is a very long caption that should be split into multiple parts".to_string(),
        );

        let splits = splitter
            .split_caption(&caption)
            .expect("split caption should succeed");
        assert!(splits.len() > 1);

        for split in &splits {
            assert!(split.text.len() <= 20);
        }
    }

    #[test]
    fn test_split_at_time() {
        let splitter = CaptionSplitter::new();

        let caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(10),
            "First part second part".to_string(),
        );

        let (first, second) = splitter
            .split_at_time(&caption, Timestamp::from_secs(5))
            .expect("operation should succeed in test");

        assert_eq!(first.start, Timestamp::from_secs(0));
        assert_eq!(first.end, Timestamp::from_secs(5));
        assert_eq!(second.start, Timestamp::from_secs(5));
        assert_eq!(second.end, Timestamp::from_secs(10));
    }

    #[test]
    fn test_smart_merge_split() {
        let optimizer = SmartMergeSplit::new(160.0);

        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(0),
                Timestamp::from_secs(1),
                "This is way too fast for comfortable reading speed".to_string(),
            ))
            .expect("operation should succeed in test");

        let (merges, splits) = optimizer
            .optimize_track(&mut track)
            .expect("optimize track should succeed");
        assert!(merges > 0 || splits > 0);
    }

    #[test]
    fn test_merge_with_gap() {
        let merger = CaptionMerger::new().with_max_gap(100);

        let cap1 = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(2),
            "First".to_string(),
        );

        let cap2 = Caption::new(
            Timestamp::from_secs(5), // 3 second gap
            Timestamp::from_secs(7),
            "Second".to_string(),
        );

        assert!(!merger.should_merge(&cap1, &cap2));
    }
}
