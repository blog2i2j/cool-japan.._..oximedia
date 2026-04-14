// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Recurring job scheduler with cron-like schedule expressions, next-run
//! calculation, missed-run handling, and run history tracking.
//!
//! This module complements [`cron_scheduler`](crate::cron_scheduler) by
//! providing a higher-level recurring-job registry that binds a `JobPayload`
//! template to a schedule and manages the full lifecycle of recurring
//! executions: spawning new job instances, recording run history, and
//! deciding what to do when runs are missed (catch-up or skip).
//!
//! # Design
//! - `RecurringSchedule` defines when a job recurs (interval-based or fixed
//!   weekday/time).
//! - `RecurringJob` stores the schedule together with a `JobPayload` template
//!   and configuration.
//! - `RecurringJobRegistry` manages a collection of `RecurringJob`s, drives
//!   their ticks, and records the outcomes of past runs.
//! - Missed-run policy: `CatchUpAll`, `CatchUpOne`, or `Skip`.
//!
//! # Example
//! ```rust
//! use oximedia_jobs::recurring_job::{
//!     RecurringJob, RecurringJobRegistry, RecurringSchedule, MissedRunPolicy,
//! };
//! use oximedia_jobs::{JobPayload, Priority, TranscodeParams};
//! use std::time::Duration;
//!
//! let params = TranscodeParams {
//!     input: "live.ts".into(), output: "archive.mp4".into(),
//!     video_codec: "h264".into(), audio_codec: "aac".into(),
//!     video_bitrate: 2_000_000, audio_bitrate: 128_000,
//!     resolution: None, framerate: None,
//!     preset: "fast".into(), hw_accel: None,
//! };
//!
//! let job = RecurringJob::new(
//!     "nightly-archive".into(),
//!     Priority::Normal,
//!     JobPayload::Transcode(params),
//!     RecurringSchedule::Interval(Duration::from_secs(3600)),
//!     MissedRunPolicy::CatchUpOne,
//! );
//!
//! let mut registry = RecurringJobRegistry::new();
//! registry.register(job);
//! ```

use crate::job::{Job, JobPayload, Priority};
use chrono::{DateTime, Datelike, Duration, Timelike, Utc, Weekday};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration as StdDuration;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// RecurringSchedule
// ---------------------------------------------------------------------------

/// Defines when a recurring job should next execute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecurringSchedule {
    /// Run every fixed wall-clock interval.
    Interval(#[serde(with = "serde_duration")] StdDuration),
    /// Run once per day at a specific UTC hour and minute.
    DailyAt {
        /// Hour in UTC (0–23).
        hour: u32,
        /// Minute (0–59).
        minute: u32,
    },
    /// Run on a specific weekday at a specific UTC hour and minute.
    WeeklyAt {
        /// Day of the week.
        weekday: SerdeWeekday,
        /// Hour in UTC (0–23).
        hour: u32,
        /// Minute (0–59).
        minute: u32,
    },
    /// Run on a specific day-of-month at a specific UTC hour and minute.
    MonthlyAt {
        /// Day of the month (1–28 for safety across all months).
        day: u32,
        /// Hour in UTC (0–23).
        hour: u32,
        /// Minute (0–59).
        minute: u32,
    },
    /// Run at minute boundaries of a given step (e.g. every 15 minutes).
    EveryNMinutes(u32),
}

/// Serde-compatible weekday wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerdeWeekday {
    Monday,
    Tuesday,
    Wednesday,
    Thursday,
    Friday,
    Saturday,
    Sunday,
}

impl SerdeWeekday {
    fn as_weekday(self) -> Weekday {
        match self {
            Self::Monday => Weekday::Mon,
            Self::Tuesday => Weekday::Tue,
            Self::Wednesday => Weekday::Wed,
            Self::Thursday => Weekday::Thu,
            Self::Friday => Weekday::Fri,
            Self::Saturday => Weekday::Sat,
            Self::Sunday => Weekday::Sun,
        }
    }
}

/// Custom serde helpers for `std::time::Duration`.
mod serde_duration {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(d.as_secs())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let secs = u64::deserialize(d)?;
        Ok(Duration::from_secs(secs))
    }
}

impl RecurringSchedule {
    /// Calculate the next trigger time after `after`.
    ///
    /// Returns `None` only if the schedule is permanently exhausted (not
    /// possible for any current variant — this is reserved for future
    /// one-shot extension).
    #[must_use]
    pub fn next_after(&self, after: DateTime<Utc>) -> Option<DateTime<Utc>> {
        match self {
            Self::Interval(dur) => {
                let secs = dur.as_secs() as i64;
                Some(after + Duration::seconds(secs))
            }
            Self::DailyAt { hour, minute } => {
                let candidate = after
                    .date_naive()
                    .and_hms_opt(*hour, *minute, 0)?;
                let candidate = DateTime::from_naive_utc_and_offset(candidate, Utc);
                if candidate > after {
                    Some(candidate)
                } else {
                    let next_day = after.date_naive().succ_opt()?;
                    let next_dt = next_day.and_hms_opt(*hour, *minute, 0)?;
                    Some(DateTime::from_naive_utc_and_offset(next_dt, Utc))
                }
            }
            Self::WeeklyAt {
                weekday,
                hour,
                minute,
            } => {
                let target = weekday.as_weekday();
                let mut check = after;
                for _ in 0..8 {
                    let candidate_date = check.date_naive();
                    if candidate_date.weekday() == target {
                        let candidate = candidate_date.and_hms_opt(*hour, *minute, 0)?;
                        let candidate = DateTime::from_naive_utc_and_offset(candidate, Utc);
                        if candidate > after {
                            return Some(candidate);
                        }
                    }
                    let next = check.date_naive().succ_opt()?;
                    check = DateTime::from_naive_utc_and_offset(
                        next.and_hms_opt(0, 0, 0)?,
                        Utc,
                    );
                }
                None
            }
            Self::MonthlyAt { day, hour, minute } => {
                let year = after.year();
                let month = after.month();
                let candidate_date =
                    chrono::NaiveDate::from_ymd_opt(year, month, *day)?;
                let candidate = candidate_date.and_hms_opt(*hour, *minute, 0)?;
                let candidate = DateTime::from_naive_utc_and_offset(candidate, Utc);
                if candidate > after {
                    return Some(candidate);
                }
                // Next month
                let (next_year, next_month) = if month == 12 {
                    (year + 1, 1u32)
                } else {
                    (year, month + 1)
                };
                let next_date = chrono::NaiveDate::from_ymd_opt(next_year, next_month, *day)?;
                let next_dt = next_date.and_hms_opt(*hour, *minute, 0)?;
                Some(DateTime::from_naive_utc_and_offset(next_dt, Utc))
            }
            Self::EveryNMinutes(n) => {
                let n = *n as i64;
                let elapsed_mins = after.minute() as i64;
                let next_mins = ((elapsed_mins / n) + 1) * n;
                let extra_hours = next_mins / 60;
                let minute = (next_mins % 60) as u32;
                let base = after
                    .date_naive()
                    .and_hms_opt(after.hour(), 0, 0)?;
                let base = DateTime::from_naive_utc_and_offset(base, Utc);
                Some(base + Duration::hours(extra_hours) + Duration::minutes(minute as i64))
            }
        }
    }

    /// Calculate all trigger times between `start` (exclusive) and `end`
    /// (inclusive), capped at `max_count`.
    #[must_use]
    pub fn triggers_between(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        max_count: usize,
    ) -> Vec<DateTime<Utc>> {
        let mut result = Vec::new();
        let mut cursor = start;
        while result.len() < max_count {
            match self.next_after(cursor) {
                Some(next) if next <= end => {
                    result.push(next);
                    cursor = next;
                }
                _ => break,
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// MissedRunPolicy
// ---------------------------------------------------------------------------

/// What to do when one or more scheduled runs were missed (e.g. the system
/// was down or the job was paused).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissedRunPolicy {
    /// Catch up by triggering once for every missed run.
    CatchUpAll,
    /// Catch up by triggering exactly one run for all missed time.
    CatchUpOne,
    /// Discard all missed runs; resume from now.
    Skip,
}

// ---------------------------------------------------------------------------
// RunOutcome / RunRecord
// ---------------------------------------------------------------------------

/// The outcome of a single recurring-job run instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunOutcome {
    /// The spawned job completed successfully.
    Success,
    /// The spawned job failed with the given message.
    Failed(String),
    /// The run was skipped due to `MissedRunPolicy::Skip`.
    Skipped,
}

/// A historical record of one execution of a recurring job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    /// Unique ID of the spawned `Job` instance (or a sentinel for skipped runs).
    pub job_instance_id: Uuid,
    /// Scheduled trigger time for this run.
    pub scheduled_at: DateTime<Utc>,
    /// Actual time the run was dispatched.
    pub dispatched_at: DateTime<Utc>,
    /// Outcome of this run.
    pub outcome: RunOutcome,
}

// ---------------------------------------------------------------------------
// RecurringJob
// ---------------------------------------------------------------------------

/// A recurring job definition binding a payload template to a schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurringJob {
    /// Unique identifier for this recurring job definition.
    pub id: Uuid,
    /// Human-readable name.
    pub name: String,
    /// Priority assigned to each spawned job instance.
    pub priority: Priority,
    /// Payload template cloned into every job instance.
    pub payload_template: JobPayload,
    /// When to trigger.
    pub schedule: RecurringSchedule,
    /// What to do with missed runs.
    pub missed_run_policy: MissedRunPolicy,
    /// Tags propagated to each spawned job instance.
    pub tags: Vec<String>,
    /// Whether this recurring job is currently active.
    pub enabled: bool,
    /// UTC time when this definition was created.
    pub created_at: DateTime<Utc>,
    /// Last time a run was dispatched (or `None` if never run).
    pub last_run_at: Option<DateTime<Utc>>,
    /// Next scheduled trigger time (recomputed after each dispatch).
    pub next_run_at: Option<DateTime<Utc>>,
    /// Maximum number of runs to keep in history (0 = unlimited).
    pub history_limit: usize,
    /// Run history (most recent first).
    #[serde(default)]
    pub run_history: Vec<RunRecord>,
}

impl RecurringJob {
    /// Create a new recurring job definition.
    #[must_use]
    pub fn new(
        name: String,
        priority: Priority,
        payload: JobPayload,
        schedule: RecurringSchedule,
        missed_run_policy: MissedRunPolicy,
    ) -> Self {
        let now = Utc::now();
        let next = schedule.next_after(now);
        Self {
            id: Uuid::new_v4(),
            name,
            priority,
            payload_template: payload,
            schedule,
            missed_run_policy,
            tags: Vec::new(),
            enabled: true,
            created_at: now,
            last_run_at: None,
            next_run_at: next,
            history_limit: 100,
            run_history: Vec::new(),
        }
    }

    /// Add a tag propagated to spawned job instances.
    #[must_use]
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set the maximum run history entries to keep.
    #[must_use]
    pub fn with_history_limit(mut self, limit: usize) -> Self {
        self.history_limit = limit;
        self
    }

    /// Disable this recurring job (no new runs will be dispatched).
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Re-enable this recurring job.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Compute the runs that should have fired between `last_run` and `now`,
    /// respecting the `MissedRunPolicy`.
    fn compute_due_runs(&self, last_run: DateTime<Utc>, now: DateTime<Utc>) -> Vec<DateTime<Utc>> {
        match self.missed_run_policy {
            MissedRunPolicy::CatchUpAll => {
                self.schedule.triggers_between(last_run, now, 1000)
            }
            MissedRunPolicy::CatchUpOne => {
                // Only the most-recent missed run
                let triggers = self.schedule.triggers_between(last_run, now, 1000);
                triggers.into_iter().last().into_iter().collect()
            }
            MissedRunPolicy::Skip => {
                // No catch-up; just advance next_run to after `now`
                Vec::new()
            }
        }
    }

    /// Spawn a `Job` instance from this definition at the given scheduled time.
    fn spawn_job(&self, scheduled_at: DateTime<Utc>) -> Job {
        let mut job = Job::new(
            self.name.clone(),
            self.priority,
            self.payload_template.clone(),
        );
        job.scheduled_at = Some(scheduled_at);
        for tag in &self.tags {
            job.tags.push(tag.clone());
        }
        job
    }

    /// Tick this recurring job definition against `now`.
    ///
    /// Returns a `Vec<Job>` of job instances that should be submitted to the
    /// queue, and records the dispatches in `run_history`.
    ///
    /// This method is idempotent with respect to duplicate ticks at the same
    /// second: if `now` has not advanced past `next_run_at`, nothing is returned.
    pub fn tick(&mut self, now: DateTime<Utc>) -> Vec<Job> {
        if !self.enabled {
            return Vec::new();
        }
        let Some(next) = self.next_run_at else {
            return Vec::new();
        };
        if now < next {
            return Vec::new();
        }

        // Determine which scheduled times are due
        let due_times: Vec<DateTime<Utc>> = match self.last_run_at {
            Some(last) => self.compute_due_runs(last, now),
            None => {
                // First run ever — trigger for `next`
                self.schedule
                    .triggers_between(self.created_at - Duration::seconds(1), now, 1)
            }
        };

        let mut jobs = Vec::new();

        if due_times.is_empty() && self.missed_run_policy == MissedRunPolicy::Skip {
            // Skip mode: no catch-up, just advance the clock
            self.last_run_at = Some(now);
        } else if due_times.is_empty() {
            // Nothing due yet (shouldn't happen given guard above)
        } else {
            for scheduled_at in &due_times {
                let job = self.spawn_job(*scheduled_at);
                let record = RunRecord {
                    job_instance_id: job.id,
                    scheduled_at: *scheduled_at,
                    dispatched_at: now,
                    outcome: RunOutcome::Success, // caller updates if it fails
                };
                jobs.push(job);
                self.run_history.insert(0, record);
            }
            self.last_run_at = due_times.last().copied();

            // Trim history
            if self.history_limit > 0 && self.run_history.len() > self.history_limit {
                self.run_history.truncate(self.history_limit);
            }
        }

        // Advance next_run_at
        self.next_run_at = self.schedule.next_after(now);
        jobs
    }

    /// Mark the run with `job_instance_id` as having a specific outcome.
    ///
    /// Returns `true` if a matching record was found and updated.
    pub fn record_outcome(&mut self, job_instance_id: Uuid, outcome: RunOutcome) -> bool {
        for record in &mut self.run_history {
            if record.job_instance_id == job_instance_id {
                record.outcome = outcome;
                return true;
            }
        }
        false
    }

    /// Count of successful runs in history.
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.run_history
            .iter()
            .filter(|r| r.outcome == RunOutcome::Success)
            .count()
    }

    /// Count of failed runs in history.
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.run_history
            .iter()
            .filter(|r| matches!(r.outcome, RunOutcome::Failed(_)))
            .count()
    }
}

// ---------------------------------------------------------------------------
// RecurringJobRegistry
// ---------------------------------------------------------------------------

/// A registry of `RecurringJob` definitions.
///
/// Call `tick(now)` periodically to generate due job instances.
#[derive(Debug, Default)]
pub struct RecurringJobRegistry {
    jobs: HashMap<Uuid, RecurringJob>,
}

impl RecurringJobRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a recurring job.  Returns the job's ID.
    pub fn register(&mut self, job: RecurringJob) -> Uuid {
        let id = job.id;
        self.jobs.insert(id, job);
        id
    }

    /// Remove a recurring job by ID.  Returns `true` if it existed.
    pub fn unregister(&mut self, id: Uuid) -> bool {
        self.jobs.remove(&id).is_some()
    }

    /// Tick all registered recurring jobs against `now`.
    ///
    /// Returns a flat list of job instances ready for submission.
    pub fn tick(&mut self, now: DateTime<Utc>) -> Vec<Job> {
        self.jobs.values_mut().flat_map(|rj| rj.tick(now)).collect()
    }

    /// Enable a recurring job by ID.  Returns `true` if found.
    pub fn enable(&mut self, id: Uuid) -> bool {
        if let Some(rj) = self.jobs.get_mut(&id) {
            rj.enable();
            true
        } else {
            false
        }
    }

    /// Disable a recurring job by ID.  Returns `true` if found.
    pub fn disable(&mut self, id: Uuid) -> bool {
        if let Some(rj) = self.jobs.get_mut(&id) {
            rj.disable();
            true
        } else {
            false
        }
    }

    /// Get a reference to a recurring job by ID.
    #[must_use]
    pub fn get(&self, id: Uuid) -> Option<&RecurringJob> {
        self.jobs.get(&id)
    }

    /// Get a mutable reference to a recurring job by ID.
    pub fn get_mut(&mut self, id: Uuid) -> Option<&mut RecurringJob> {
        self.jobs.get_mut(&id)
    }

    /// Number of registered recurring jobs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Returns `true` if there are no registered recurring jobs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Return IDs of jobs whose `next_run_at` is before or at `now`.
    #[must_use]
    pub fn due_job_ids(&self, now: DateTime<Utc>) -> Vec<Uuid> {
        self.jobs
            .values()
            .filter(|rj| rj.enabled)
            .filter(|rj| rj.next_run_at.map_or(false, |t| t <= now))
            .map(|rj| rj.id)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{JobPayload, Priority, TranscodeParams};

    fn params() -> TranscodeParams {
        TranscodeParams {
            input: "in.mp4".into(),
            output: "out.mp4".into(),
            video_codec: "h264".into(),
            audio_codec: "aac".into(),
            video_bitrate: 4_000_000,
            audio_bitrate: 128_000,
            resolution: None,
            framerate: None,
            preset: "fast".into(),
            hw_accel: None,
        }
    }

    fn make_rj(schedule: RecurringSchedule) -> RecurringJob {
        RecurringJob::new(
            "test-rj".into(),
            Priority::Normal,
            JobPayload::Transcode(params()),
            schedule,
            MissedRunPolicy::CatchUpOne,
        )
    }

    #[test]
    fn test_interval_next_after() {
        let schedule = RecurringSchedule::Interval(StdDuration::from_secs(3600));
        let base = Utc::now();
        let next = schedule.next_after(base).expect("next should exist");
        let diff = next - base;
        assert_eq!(diff.num_seconds(), 3600);
    }

    #[test]
    fn test_every_n_minutes_next_after() {
        let schedule = RecurringSchedule::EveryNMinutes(15);
        // Use a known time: 10:07 UTC → next at 10:15
        let base = chrono::DateTime::parse_from_rfc3339("2025-06-01T10:07:00Z")
            .expect("parse should work")
            .with_timezone(&Utc);
        let next = schedule.next_after(base).expect("next should exist");
        assert_eq!(next.minute(), 15);
        assert_eq!(next.hour(), 10);
    }

    #[test]
    fn test_daily_at_next_after_same_day() {
        let schedule = RecurringSchedule::DailyAt { hour: 23, minute: 0 };
        let base = chrono::DateTime::parse_from_rfc3339("2025-06-01T10:00:00Z")
            .expect("parse")
            .with_timezone(&Utc);
        let next = schedule.next_after(base).expect("next");
        assert_eq!(next.hour(), 23);
        assert_eq!(next.minute(), 0);
        assert_eq!(next.day(), 1);
    }

    #[test]
    fn test_daily_at_next_after_next_day() {
        let schedule = RecurringSchedule::DailyAt { hour: 6, minute: 0 };
        let base = chrono::DateTime::parse_from_rfc3339("2025-06-01T10:00:00Z")
            .expect("parse")
            .with_timezone(&Utc);
        let next = schedule.next_after(base).expect("next");
        assert_eq!(next.day(), 2);
        assert_eq!(next.hour(), 6);
    }

    #[test]
    fn test_triggers_between() {
        let schedule = RecurringSchedule::Interval(StdDuration::from_secs(60));
        let start = Utc::now();
        let end = start + Duration::minutes(5);
        let triggers = schedule.triggers_between(start, end, 10);
        assert_eq!(triggers.len(), 5);
    }

    #[test]
    fn test_tick_generates_job() {
        let schedule = RecurringSchedule::Interval(StdDuration::from_secs(1));
        let mut rj = make_rj(schedule);
        // Advance time far enough to trigger
        let now = rj.next_run_at.expect("must have next_run") + Duration::seconds(1);
        let jobs = rj.tick(now);
        assert!(!jobs.is_empty());
        assert_eq!(rj.run_history.len(), jobs.len());
    }

    #[test]
    fn test_tick_does_not_trigger_before_due() {
        let schedule = RecurringSchedule::Interval(StdDuration::from_secs(3600));
        let mut rj = make_rj(schedule);
        let now = rj.next_run_at.expect("must have next_run") - Duration::seconds(1);
        let jobs = rj.tick(now);
        assert!(jobs.is_empty());
    }

    #[test]
    fn test_disable_prevents_tick() {
        let schedule = RecurringSchedule::Interval(StdDuration::from_secs(1));
        let mut rj = make_rj(schedule);
        rj.disable();
        let now = rj.next_run_at.expect("must have next_run") + Duration::seconds(10);
        let jobs = rj.tick(now);
        assert!(jobs.is_empty());
    }

    #[test]
    fn test_enable_restores_ticking() {
        let schedule = RecurringSchedule::Interval(StdDuration::from_secs(1));
        let mut rj = make_rj(schedule);
        rj.disable();
        rj.enable();
        let now = rj.next_run_at.expect("must have next_run") + Duration::seconds(5);
        let jobs = rj.tick(now);
        assert!(!jobs.is_empty());
    }

    #[test]
    fn test_catch_up_one_policy() {
        let schedule = RecurringSchedule::Interval(StdDuration::from_secs(60));
        let mut rj = RecurringJob::new(
            "catch-up-one".into(),
            Priority::Normal,
            JobPayload::Transcode(params()),
            schedule,
            MissedRunPolicy::CatchUpOne,
        );
        // Simulate being 10 minutes late
        let now = rj.next_run_at.expect("must have next_run") + Duration::minutes(10);
        let jobs = rj.tick(now);
        // Should only catch up once, not 10 times
        assert_eq!(jobs.len(), 1, "CatchUpOne should produce exactly 1 catch-up job");
    }

    #[test]
    fn test_record_outcome() {
        let schedule = RecurringSchedule::Interval(StdDuration::from_secs(1));
        let mut rj = make_rj(schedule);
        let now = rj.next_run_at.expect("must have next_run") + Duration::seconds(2);
        let jobs = rj.tick(now);
        assert!(!jobs.is_empty());

        let job_id = jobs[0].id;
        let updated = rj.record_outcome(job_id, RunOutcome::Failed("disk full".into()));
        assert!(updated);
        assert_eq!(rj.failure_count(), 1);
        assert_eq!(rj.success_count(), jobs.len() - 1);
    }

    #[test]
    fn test_registry_register_and_tick() {
        let schedule = RecurringSchedule::Interval(StdDuration::from_secs(1));
        let rj = make_rj(schedule);
        let mut registry = RecurringJobRegistry::new();
        let id = registry.register(rj);

        assert_eq!(registry.len(), 1);
        let rj_ref = registry.get(id).expect("should exist");
        let now = rj_ref.next_run_at.expect("must have next_run") + Duration::seconds(2);

        let jobs = registry.tick(now);
        assert!(!jobs.is_empty());
    }

    #[test]
    fn test_registry_disable() {
        let schedule = RecurringSchedule::Interval(StdDuration::from_secs(1));
        let rj = make_rj(schedule);
        let mut registry = RecurringJobRegistry::new();
        let id = registry.register(rj);

        let rj_ref = registry.get(id).expect("should exist");
        let now = rj_ref.next_run_at.expect("must have next_run") + Duration::seconds(2);

        registry.disable(id);
        let jobs = registry.tick(now);
        assert!(jobs.is_empty());
    }

    #[test]
    fn test_tags_propagated_to_spawned_jobs() {
        let schedule = RecurringSchedule::Interval(StdDuration::from_secs(1));
        let mut rj = RecurringJob::new(
            "tagged".into(),
            Priority::High,
            JobPayload::Transcode(params()),
            schedule,
            MissedRunPolicy::CatchUpOne,
        )
        .with_tag("video")
        .with_tag("nightly");

        let now = rj.next_run_at.expect("must have next_run") + Duration::seconds(2);
        let jobs = rj.tick(now);
        assert!(!jobs.is_empty());
        assert!(jobs[0].tags.contains(&"video".to_string()));
        assert!(jobs[0].tags.contains(&"nightly".to_string()));
    }

    #[test]
    fn test_history_limit_enforced() {
        let schedule = RecurringSchedule::Interval(StdDuration::from_secs(1));
        let mut rj = RecurringJob::new(
            "limited".into(),
            Priority::Normal,
            JobPayload::Transcode(params()),
            schedule,
            MissedRunPolicy::CatchUpAll,
        )
        .with_history_limit(3);

        // Trigger 10 runs at once
        let now = rj.next_run_at.expect("must have next_run") + Duration::seconds(10);
        let _jobs = rj.tick(now);
        assert!(
            rj.run_history.len() <= 3,
            "history should be capped at 3, got {}",
            rj.run_history.len()
        );
    }
}
