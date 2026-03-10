//! Job lifecycle event log.

#![allow(dead_code)]

/// Kind of lifecycle event for a job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobEventKind {
    /// Job was submitted to the queue.
    Submitted,
    /// Job started executing.
    Started,
    /// Job execution was paused.
    Paused,
    /// Job execution was resumed.
    Resumed,
    /// Job completed successfully.
    Completed,
    /// Job failed.
    Failed,
    /// Job was cancelled.
    Cancelled,
    /// A retry has been scheduled for the job.
    RetryScheduled,
}

impl JobEventKind {
    /// Returns `true` if this event kind represents a terminal state (the job will not run again).
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// A single lifecycle event for a job.
#[derive(Clone, Debug)]
pub struct JobEvent {
    /// The job this event relates to.
    pub job_id: u64,
    /// The kind of event.
    pub kind: JobEventKind,
    /// Unix epoch timestamp (seconds) when the event occurred.
    pub timestamp_epoch: u64,
    /// Optional human-readable message (empty string if none).
    pub message: String,
}

impl JobEvent {
    /// Create a new `JobEvent` with an empty message.
    #[must_use]
    pub fn new(job_id: u64, kind: JobEventKind, epoch: u64) -> Self {
        Self {
            job_id,
            kind,
            timestamp_epoch: epoch,
            message: String::new(),
        }
    }

    /// Attach a message to this event and return the modified event.
    #[must_use]
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = msg.into();
        self
    }
}

/// Ordered log of job lifecycle events.
#[derive(Clone, Debug, Default)]
pub struct EventLog {
    /// All recorded events, in insertion order.
    pub events: Vec<JobEvent>,
}

impl EventLog {
    /// Create an empty `EventLog`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an event to the log.
    pub fn record(&mut self, event: JobEvent) {
        self.events.push(event);
    }

    /// Return all events for a specific job id, in order.
    #[must_use]
    pub fn events_for_job(&self, id: u64) -> Vec<&JobEvent> {
        self.events.iter().filter(|e| e.job_id == id).collect()
    }

    /// Return the most recent event for a specific job id.
    #[must_use]
    pub fn latest_event(&self, id: u64) -> Option<&JobEvent> {
        self.events.iter().filter(|e| e.job_id == id).last()
    }

    /// Return all terminal events across all jobs.
    #[must_use]
    pub fn terminal_events(&self) -> Vec<&JobEvent> {
        self.events
            .iter()
            .filter(|e| e.kind.is_terminal())
            .collect()
    }

    /// Return the total number of events recorded.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_log() -> EventLog {
        let mut log = EventLog::new();
        log.record(JobEvent::new(1, JobEventKind::Submitted, 1000));
        log.record(JobEvent::new(1, JobEventKind::Started, 1010));
        log.record(JobEvent::new(2, JobEventKind::Submitted, 1005));
        log.record(JobEvent::new(1, JobEventKind::Completed, 1050));
        log.record(JobEvent::new(2, JobEventKind::Failed, 1060));
        log
    }

    // ---------- JobEventKind tests ----------

    #[test]
    fn test_is_terminal_completed() {
        assert!(JobEventKind::Completed.is_terminal());
    }

    #[test]
    fn test_is_terminal_failed() {
        assert!(JobEventKind::Failed.is_terminal());
    }

    #[test]
    fn test_is_terminal_cancelled() {
        assert!(JobEventKind::Cancelled.is_terminal());
    }

    #[test]
    fn test_is_terminal_submitted_false() {
        assert!(!JobEventKind::Submitted.is_terminal());
    }

    #[test]
    fn test_is_terminal_started_false() {
        assert!(!JobEventKind::Started.is_terminal());
    }

    #[test]
    fn test_is_terminal_paused_false() {
        assert!(!JobEventKind::Paused.is_terminal());
    }

    #[test]
    fn test_is_terminal_resumed_false() {
        assert!(!JobEventKind::Resumed.is_terminal());
    }

    #[test]
    fn test_is_terminal_retry_scheduled_false() {
        assert!(!JobEventKind::RetryScheduled.is_terminal());
    }

    // ---------- JobEvent tests ----------

    #[test]
    fn test_job_event_new() {
        let ev = JobEvent::new(42, JobEventKind::Started, 99999);
        assert_eq!(ev.job_id, 42);
        assert_eq!(ev.kind, JobEventKind::Started);
        assert_eq!(ev.timestamp_epoch, 99999);
        assert!(ev.message.is_empty());
    }

    #[test]
    fn test_job_event_with_message() {
        let ev = JobEvent::new(1, JobEventKind::Failed, 0).with_message("OOM");
        assert_eq!(ev.message, "OOM");
    }

    // ---------- EventLog tests ----------

    #[test]
    fn test_event_count() {
        let log = make_log();
        assert_eq!(log.event_count(), 5);
    }

    #[test]
    fn test_events_for_job() {
        let log = make_log();
        let job1_events = log.events_for_job(1);
        assert_eq!(job1_events.len(), 3);
    }

    #[test]
    fn test_events_for_job_order() {
        let log = make_log();
        let evs = log.events_for_job(1);
        assert_eq!(evs[0].kind, JobEventKind::Submitted);
        assert_eq!(evs[1].kind, JobEventKind::Started);
        assert_eq!(evs[2].kind, JobEventKind::Completed);
    }

    #[test]
    fn test_latest_event() {
        let log = make_log();
        let latest = log.latest_event(1).expect("latest should be valid");
        assert_eq!(latest.kind, JobEventKind::Completed);
    }

    #[test]
    fn test_latest_event_missing() {
        let log = make_log();
        assert!(log.latest_event(999).is_none());
    }

    #[test]
    fn test_terminal_events() {
        let log = make_log();
        let terminals = log.terminal_events();
        assert_eq!(terminals.len(), 2);
    }

    #[test]
    fn test_terminal_events_kinds() {
        let log = make_log();
        let terminals = log.terminal_events();
        assert!(terminals.iter().all(|e| e.kind.is_terminal()));
    }

    #[test]
    fn test_empty_log() {
        let log = EventLog::new();
        assert_eq!(log.event_count(), 0);
        assert!(log.events_for_job(1).is_empty());
        assert!(log.latest_event(1).is_none());
        assert!(log.terminal_events().is_empty());
    }
}
