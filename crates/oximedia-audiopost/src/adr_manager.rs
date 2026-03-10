//! ADR (Automated Dialogue Replacement) management: loop takes, sync cue points, ADR report.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A timecode position expressed in sample frames at a given sample rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SamplePosition {
    /// Sample index.
    pub sample: u64,
    /// Sample rate (samples per second).
    pub sample_rate: u32,
}

impl SamplePosition {
    /// Create a new sample position.
    pub fn new(sample: u64, sample_rate: u32) -> Self {
        Self {
            sample,
            sample_rate,
        }
    }

    /// Return the position in seconds.
    pub fn as_secs(&self) -> f64 {
        self.sample as f64 / self.sample_rate as f64
    }

    /// Duration between two sample positions (self must be <= other).
    pub fn duration_to(&self, other: &Self) -> Option<f64> {
        if self.sample_rate != other.sample_rate || other.sample < self.sample {
            return None;
        }
        let delta = other.sample - self.sample;
        Some(delta as f64 / self.sample_rate as f64)
    }
}

/// Status of a loop take recording.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TakeStatus {
    /// Take has not been recorded yet.
    Pending,
    /// Take was recorded and is under review.
    Recorded,
    /// Take has been approved for use.
    Approved,
    /// Take was rejected; needs to be re-recorded.
    Rejected,
    /// Take was approved but an alternative was preferred.
    Alternate,
}

/// A single ADR loop take.
#[derive(Debug, Clone)]
pub struct LoopTake {
    /// Take number within the cue (1-based).
    pub take_number: u32,
    /// File path to the recorded audio.
    pub file_path: String,
    /// Status of this take.
    pub status: TakeStatus,
    /// Optional notes from the director.
    pub notes: Option<String>,
    /// Whether this is the selected (print) take.
    pub is_print: bool,
}

impl LoopTake {
    /// Create a new loop take.
    pub fn new(take_number: u32, file_path: &str) -> Self {
        Self {
            take_number,
            file_path: file_path.to_string(),
            status: TakeStatus::Pending,
            notes: None,
            is_print: false,
        }
    }

    /// Mark this take as approved.
    pub fn approve(&mut self) {
        self.status = TakeStatus::Approved;
    }

    /// Mark this take as rejected with optional notes.
    pub fn reject(&mut self, notes: Option<&str>) {
        self.status = TakeStatus::Rejected;
        self.notes = notes.map(|s| s.to_string());
    }

    /// Mark this take as the print take.
    pub fn set_print(&mut self) {
        self.is_print = true;
    }
}

/// An ADR sync cue point within a cue.
#[derive(Debug, Clone)]
pub struct SyncCuePoint {
    /// Cue point label.
    pub label: String,
    /// Position in the original picture.
    pub picture_position: SamplePosition,
    /// Description of what happens at this point.
    pub description: Option<String>,
}

impl SyncCuePoint {
    /// Create a new sync cue point.
    pub fn new(label: &str, position: SamplePosition) -> Self {
        Self {
            label: label.to_string(),
            picture_position: position,
            description: None,
        }
    }

    /// Add a description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }
}

/// An ADR cue representing a line or passage to be replaced.
#[derive(Debug, Clone)]
pub struct AdrCueRecord {
    /// Unique cue identifier.
    pub cue_id: String,
    /// Cue number displayed in the ADR session (e.g., "1A").
    pub cue_number: String,
    /// Character name.
    pub character: String,
    /// The original dialogue text.
    pub dialogue: String,
    /// In-point in the picture.
    pub in_point: SamplePosition,
    /// Out-point in the picture.
    pub out_point: SamplePosition,
    /// All recorded takes.
    pub takes: Vec<LoopTake>,
    /// Sync cue points within this cue.
    pub sync_points: Vec<SyncCuePoint>,
    /// Whether this cue is complete (has an approved print take).
    pub is_complete: bool,
}

impl AdrCueRecord {
    /// Create a new ADR cue record.
    pub fn new(
        cue_id: &str,
        cue_number: &str,
        character: &str,
        dialogue: &str,
        in_point: SamplePosition,
        out_point: SamplePosition,
    ) -> Self {
        Self {
            cue_id: cue_id.to_string(),
            cue_number: cue_number.to_string(),
            character: character.to_string(),
            dialogue: dialogue.to_string(),
            in_point,
            out_point,
            takes: Vec::new(),
            sync_points: Vec::new(),
            is_complete: false,
        }
    }

    /// Duration of the cue in seconds.
    pub fn duration_secs(&self) -> Option<f64> {
        self.in_point.duration_to(&self.out_point)
    }

    /// Add a loop take.
    pub fn add_take(&mut self, take: LoopTake) {
        self.takes.push(take);
    }

    /// Add a sync cue point.
    pub fn add_sync_point(&mut self, point: SyncCuePoint) {
        self.sync_points.push(point);
    }

    /// Get the current print take, if any.
    pub fn print_take(&self) -> Option<&LoopTake> {
        self.takes.iter().find(|t| t.is_print)
    }

    /// Mark the cue as complete (requires a print take).
    pub fn mark_complete(&mut self) -> bool {
        if self.print_take().is_some() {
            self.is_complete = true;
            true
        } else {
            false
        }
    }

    /// Take count.
    pub fn take_count(&self) -> usize {
        self.takes.len()
    }
}

/// An ADR session report for a full recording session.
#[derive(Debug, Default)]
pub struct AdrReport {
    /// Project name.
    pub project: String,
    /// Episode or reel identifier.
    pub reel: String,
    /// All cues in this session.
    cues: Vec<AdrCueRecord>,
}

impl AdrReport {
    /// Create a new ADR report.
    pub fn new(project: &str, reel: &str) -> Self {
        Self {
            project: project.to_string(),
            reel: reel.to_string(),
            cues: Vec::new(),
        }
    }

    /// Add a cue record.
    pub fn add_cue(&mut self, cue: AdrCueRecord) {
        self.cues.push(cue);
    }

    /// Total number of cues.
    pub fn cue_count(&self) -> usize {
        self.cues.len()
    }

    /// Number of completed cues.
    pub fn completed_count(&self) -> usize {
        self.cues.iter().filter(|c| c.is_complete).count()
    }

    /// Completion percentage (0.0–100.0).
    pub fn completion_percent(&self) -> f64 {
        if self.cues.is_empty() {
            return 100.0;
        }
        self.completed_count() as f64 / self.cues.len() as f64 * 100.0
    }

    /// All cues for a given character.
    pub fn cues_for_character(&self, character: &str) -> Vec<&AdrCueRecord> {
        self.cues
            .iter()
            .filter(|c| c.character == character)
            .collect()
    }

    /// Total take count across all cues.
    pub fn total_takes(&self) -> usize {
        self.cues.iter().map(|c| c.take_count()).sum()
    }

    /// Export a simple text summary.
    pub fn summary_text(&self) -> String {
        format!(
            "ADR Report: {} / {}\nCues: {} (complete: {})\nCompletion: {:.1}%\nTotal takes: {}",
            self.project,
            self.reel,
            self.cue_count(),
            self.completed_count(),
            self.completion_percent(),
            self.total_takes(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pos(sample: u64) -> SamplePosition {
        SamplePosition::new(sample, 48000)
    }

    #[test]
    fn test_sample_position_as_secs() {
        let p = pos(48000);
        assert!((p.as_secs() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_sample_position_duration_to() {
        let start = pos(0);
        let end = pos(48000);
        let dur = start.duration_to(&end).expect("duration_to should succeed");
        assert!((dur - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_duration_to_backwards_returns_none() {
        let start = pos(48000);
        let end = pos(0);
        assert!(start.duration_to(&end).is_none());
    }

    #[test]
    fn test_loop_take_new() {
        let take = LoopTake::new(1, "/takes/take1.wav");
        assert_eq!(take.take_number, 1);
        assert_eq!(take.status, TakeStatus::Pending);
        assert!(!take.is_print);
    }

    #[test]
    fn test_loop_take_approve() {
        let mut take = LoopTake::new(1, "/takes/take1.wav");
        take.approve();
        assert_eq!(take.status, TakeStatus::Approved);
    }

    #[test]
    fn test_loop_take_reject_with_notes() {
        let mut take = LoopTake::new(2, "/takes/take2.wav");
        take.reject(Some("Too breathy"));
        assert_eq!(take.status, TakeStatus::Rejected);
        assert_eq!(take.notes.as_deref(), Some("Too breathy"));
    }

    #[test]
    fn test_loop_take_set_print() {
        let mut take = LoopTake::new(3, "/takes/take3.wav");
        take.set_print();
        assert!(take.is_print);
    }

    #[test]
    fn test_sync_cue_point_with_description() {
        let p = SyncCuePoint::new("CUE-A", pos(1000)).with_description("Door opens");
        assert_eq!(p.label, "CUE-A");
        assert_eq!(p.description.as_deref(), Some("Door opens"));
    }

    #[test]
    fn test_adr_cue_record_duration() {
        let cue = AdrCueRecord::new("cue-1", "1A", "Alice", "Hello world", pos(0), pos(96000));
        let dur = cue.duration_secs().expect("duration_secs should succeed");
        assert!((dur - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_adr_cue_record_takes() {
        let mut cue = AdrCueRecord::new("cue-1", "1A", "Alice", "Hello", pos(0), pos(48000));
        cue.add_take(LoopTake::new(1, "/t1.wav"));
        cue.add_take(LoopTake::new(2, "/t2.wav"));
        assert_eq!(cue.take_count(), 2);
    }

    #[test]
    fn test_adr_cue_mark_complete_requires_print() {
        let mut cue = AdrCueRecord::new("cue-1", "1A", "Bob", "Line", pos(0), pos(48000));
        assert!(!cue.mark_complete());
        let mut take = LoopTake::new(1, "/t1.wav");
        take.set_print();
        cue.add_take(take);
        assert!(cue.mark_complete());
        assert!(cue.is_complete);
    }

    #[test]
    fn test_adr_report_new() {
        let report = AdrReport::new("MyFilm", "Reel 1");
        assert_eq!(report.project, "MyFilm");
        assert_eq!(report.cue_count(), 0);
    }

    #[test]
    fn test_adr_report_completion_percent_empty() {
        let report = AdrReport::new("Film", "R1");
        assert!((report.completion_percent() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_adr_report_completion_percent() {
        let mut report = AdrReport::new("Film", "R1");
        let mut cue1 = AdrCueRecord::new("c1", "1A", "Alice", "Hi", pos(0), pos(48000));
        let mut take = LoopTake::new(1, "/t1.wav");
        take.set_print();
        cue1.add_take(take);
        cue1.mark_complete();
        report.add_cue(cue1);
        report.add_cue(AdrCueRecord::new(
            "c2",
            "2A",
            "Bob",
            "Hello",
            pos(0),
            pos(48000),
        ));
        assert!((report.completion_percent() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_adr_report_cues_for_character() {
        let mut report = AdrReport::new("Film", "R1");
        report.add_cue(AdrCueRecord::new(
            "c1",
            "1A",
            "Alice",
            "Hi",
            pos(0),
            pos(48000),
        ));
        report.add_cue(AdrCueRecord::new(
            "c2",
            "1B",
            "Alice",
            "Bye",
            pos(48000),
            pos(96000),
        ));
        report.add_cue(AdrCueRecord::new(
            "c3",
            "1C",
            "Bob",
            "Hey",
            pos(0),
            pos(48000),
        ));
        assert_eq!(report.cues_for_character("Alice").len(), 2);
        assert_eq!(report.cues_for_character("Bob").len(), 1);
    }

    #[test]
    fn test_adr_report_total_takes() {
        let mut report = AdrReport::new("Film", "R1");
        let mut cue = AdrCueRecord::new("c1", "1A", "Alice", "Hi", pos(0), pos(48000));
        cue.add_take(LoopTake::new(1, "/t1.wav"));
        cue.add_take(LoopTake::new(2, "/t2.wav"));
        cue.add_take(LoopTake::new(3, "/t3.wav"));
        report.add_cue(cue);
        assert_eq!(report.total_takes(), 3);
    }

    #[test]
    fn test_summary_text_contains_project() {
        let report = AdrReport::new("TestFilm", "R2");
        let text = report.summary_text();
        assert!(text.contains("TestFilm"));
        assert!(text.contains("R2"));
    }
}
