//! As-run log generation for broadcast compliance.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::time::Duration;

/// Entry in an as-run log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsRunEntry {
    /// Entry ID.
    pub id: String,

    /// Scheduled start time.
    pub scheduled_start: DateTime<Utc>,

    /// Actual start time.
    pub actual_start: DateTime<Utc>,

    /// Scheduled duration.
    pub scheduled_duration: Duration,

    /// Actual duration.
    pub actual_duration: Duration,

    /// Content title.
    pub title: String,

    /// Content ID or file path.
    pub content_id: String,

    /// Channel ID.
    pub channel_id: String,

    /// Whether playback completed successfully.
    pub completed: bool,

    /// Error message if playback failed.
    pub error: Option<String>,

    /// Additional metadata.
    pub metadata: std::collections::HashMap<String, String>,
}

impl AsRunEntry {
    /// Creates a new as-run entry.
    #[must_use]
    pub fn new<S: Into<String>>(
        title: S,
        content_id: S,
        channel_id: S,
        scheduled_start: DateTime<Utc>,
        actual_start: DateTime<Utc>,
    ) -> Self {
        Self {
            id: generate_id(),
            scheduled_start,
            actual_start,
            scheduled_duration: Duration::ZERO,
            actual_duration: Duration::ZERO,
            title: title.into(),
            content_id: content_id.into(),
            channel_id: channel_id.into(),
            completed: false,
            error: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Sets the scheduled duration.
    #[must_use]
    pub const fn with_scheduled_duration(mut self, duration: Duration) -> Self {
        self.scheduled_duration = duration;
        self
    }

    /// Sets the actual duration.
    #[must_use]
    pub const fn with_actual_duration(mut self, duration: Duration) -> Self {
        self.actual_duration = duration;
        self
    }

    /// Marks this entry as completed.
    pub fn mark_completed(&mut self) {
        self.completed = true;
    }

    /// Sets an error message.
    pub fn set_error<S: Into<String>>(&mut self, error: S) {
        self.error = Some(error.into());
        self.completed = false;
    }

    /// Adds custom metadata.
    pub fn add_metadata<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Calculates the time variance from schedule.
    #[must_use]
    pub fn time_variance(&self) -> chrono::Duration {
        self.actual_start - self.scheduled_start
    }

    /// Calculates the duration variance from schedule.
    #[must_use]
    pub fn duration_variance(&self) -> Duration {
        if self.actual_duration > self.scheduled_duration {
            self.actual_duration
                .checked_sub(self.scheduled_duration)
                .unwrap_or(Duration::ZERO)
        } else {
            self.scheduled_duration
                .checked_sub(self.actual_duration)
                .unwrap_or(Duration::ZERO)
        }
    }
}

/// As-run log for tracking actual broadcast content.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct AsRunLog {
    entries: Vec<AsRunEntry>,
}

impl AsRunLog {
    /// Creates a new as-run log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an entry to the log.
    pub fn add_entry(&mut self, entry: AsRunEntry) {
        self.entries.push(entry);
    }

    /// Gets all entries for a specific channel.
    #[must_use]
    pub fn get_entries_for_channel(&self, channel_id: &str) -> Vec<&AsRunEntry> {
        self.entries
            .iter()
            .filter(|e| e.channel_id == channel_id)
            .collect()
    }

    /// Gets entries in a time range.
    #[must_use]
    pub fn get_entries_in_range(
        &self,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
    ) -> Vec<&AsRunEntry> {
        self.entries
            .iter()
            .filter(|e| e.actual_start >= *start && e.actual_start < *end)
            .collect()
    }

    /// Gets all entries.
    #[must_use]
    pub fn get_all_entries(&self) -> &[AsRunEntry] {
        &self.entries
    }

    /// Exports the log to CSV format.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn export_csv<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        // Write header
        writeln!(
            writer,
            "ID,Scheduled Start,Actual Start,Title,Content ID,Channel,Completed,Error,Time Variance (s),Duration Variance (s)"
        )?;

        // Write entries
        for entry in &self.entries {
            let time_var = entry.time_variance().num_seconds();
            let dur_var = entry.duration_variance().as_secs();
            let error = entry.error.as_deref().unwrap_or("");

            writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{},{}",
                entry.id,
                entry.scheduled_start.to_rfc3339(),
                entry.actual_start.to_rfc3339(),
                escape_csv(&entry.title),
                escape_csv(&entry.content_id),
                entry.channel_id,
                entry.completed,
                escape_csv(error),
                time_var,
                dur_var
            )?;
        }

        Ok(())
    }

    /// Exports the log to JSON format.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.entries)
    }

    /// Returns the number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if there are no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("asrun_{timestamp}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asrun_entry() {
        let now = Utc::now();
        let mut entry = AsRunEntry::new("Test Show", "show_001", "channel1", now, now)
            .with_scheduled_duration(Duration::from_secs(3600))
            .with_actual_duration(Duration::from_secs(3605));

        entry.mark_completed();
        assert!(entry.completed);
        assert_eq!(entry.duration_variance(), Duration::from_secs(5));
    }

    #[test]
    fn test_asrun_log() {
        let mut log = AsRunLog::new();
        let now = Utc::now();

        let entry = AsRunEntry::new("Test", "test_001", "channel1", now, now);
        log.add_entry(entry);

        assert_eq!(log.len(), 1);

        let entries = log.get_entries_for_channel("channel1");
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_csv_export() {
        let mut log = AsRunLog::new();
        let now = Utc::now();

        let entry = AsRunEntry::new("Test Show", "test_001", "channel1", now, now)
            .with_scheduled_duration(Duration::from_secs(3600));

        log.add_entry(entry);

        let mut output = Vec::new();
        log.export_csv(&mut output).expect("should succeed in test");

        let csv = String::from_utf8(output).expect("should succeed in test");
        assert!(csv.contains("Test Show"));
        assert!(csv.contains("channel1"));
    }
}
