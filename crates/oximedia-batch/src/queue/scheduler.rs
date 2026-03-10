//! Job scheduler for delayed and recurring jobs

use crate::error::Result;
use crate::job::BatchJob;
use crate::types::{JobId, Schedule};
use chrono::{DateTime, Datelike, TimeZone, Timelike, Utc};
use std::collections::HashMap;

/// Parsed cron expression with 5 fields: minute hour day month weekday
struct CronExpression {
    /// Allowed minute values (0-59); empty means wildcard
    minutes: Vec<u8>,
    /// Allowed hour values (0-23); empty means wildcard
    hours: Vec<u8>,
    /// Allowed day-of-month values (1-31); empty means wildcard
    days: Vec<u8>,
    /// Allowed month values (1-12); empty means wildcard
    months: Vec<u8>,
    /// Allowed weekday values (0-6, Sunday=0); empty means wildcard
    weekdays: Vec<u8>,
}

impl CronExpression {
    /// Parse a 5-field cron expression string.
    ///
    /// Supports: `*`, `N`, `*/N`, `N-M`, `N,M,K` per field.
    fn parse(expr: &str) -> Option<Self> {
        let fields: Vec<&str> = expr.split_whitespace().collect();
        if fields.len() != 5 {
            return None;
        }

        let minutes = Self::parse_field(fields[0], 0, 59)?;
        let hours = Self::parse_field(fields[1], 0, 23)?;
        let days = Self::parse_field(fields[2], 1, 31)?;
        let months = Self::parse_field(fields[3], 1, 12)?;
        let weekdays = Self::parse_field(fields[4], 0, 6)?;

        Some(Self {
            minutes,
            hours,
            days,
            months,
            weekdays,
        })
    }

    fn parse_field(field: &str, min: u8, max: u8) -> Option<Vec<u8>> {
        if field == "*" {
            return Some(Vec::new()); // empty = wildcard
        }

        let mut values = Vec::new();

        for part in field.split(',') {
            if part == "*" {
                return Some(Vec::new());
            } else if let Some(step_part) = part.strip_prefix("*/") {
                let step: u8 = step_part.parse().ok()?;
                if step == 0 {
                    return None;
                }
                let mut v = min;
                while v <= max {
                    values.push(v);
                    v = v.checked_add(step)?;
                }
            } else if let Some(dash_pos) = part.find('-') {
                let lo: u8 = part[..dash_pos].parse().ok()?;
                let hi: u8 = part[dash_pos + 1..].parse().ok()?;
                if lo > hi || lo < min || hi > max {
                    return None;
                }
                for v in lo..=hi {
                    values.push(v);
                }
            } else {
                let v: u8 = part.parse().ok()?;
                if v < min || v > max {
                    return None;
                }
                values.push(v);
            }
        }

        values.sort_unstable();
        values.dedup();
        Some(values)
    }

    fn matches_field(values: &[u8], value: u8) -> bool {
        values.is_empty() || values.contains(&value)
    }

    /// Compute the next `DateTime<Utc>` at or after `from + 1 minute` that satisfies this cron.
    fn next_after(&self, from: DateTime<Utc>) -> DateTime<Utc> {
        // Start search from the next minute
        let start_secs = from.timestamp() + 60;
        // Truncate to minute boundary
        let start_secs = (start_secs / 60) * 60;

        let mut candidate = match Utc.timestamp_opt(start_secs, 0) {
            chrono::LocalResult::Single(dt) => dt,
            _ => from + chrono::Duration::minutes(1),
        };

        // Search up to 4 years (worst case leap-year cron)
        let limit = from + chrono::Duration::days(366 * 4);

        while candidate <= limit {
            #[allow(clippy::cast_possible_truncation)]
            let month = candidate.month() as u8;
            #[allow(clippy::cast_possible_truncation)]
            let day = candidate.day() as u8;
            #[allow(clippy::cast_possible_truncation)]
            let weekday = candidate.weekday().num_days_from_sunday() as u8;
            #[allow(clippy::cast_possible_truncation)]
            let hour = candidate.hour() as u8;
            #[allow(clippy::cast_possible_truncation)]
            let minute = candidate.minute() as u8;

            if Self::matches_field(&self.months, month)
                && Self::matches_field(&self.days, day)
                && Self::matches_field(&self.weekdays, weekday)
                && Self::matches_field(&self.hours, hour)
                && Self::matches_field(&self.minutes, minute)
            {
                return candidate;
            }

            candidate += chrono::Duration::minutes(1);
        }

        // Fallback: 1 hour from now
        from + chrono::Duration::hours(1)
    }
}

/// Compute next occurrence for a `Schedule::Recurring` expression, or fall back to 1 hour.
fn next_occurrence_for_expression(expr: &str, from: DateTime<Utc>) -> DateTime<Utc> {
    if let Some(cron) = CronExpression::parse(expr) {
        cron.next_after(from)
    } else {
        from + chrono::Duration::hours(1)
    }
}

/// Scheduled job entry
#[derive(Clone)]
struct ScheduledJob {
    job: BatchJob,
    due_time: DateTime<Utc>,
    recurring: bool,
    /// Cron expression string (for recurring jobs)
    cron_expression: Option<String>,
    /// Fallback interval in seconds (used when no cron expression is available)
    interval_secs: u64,
}

/// Job scheduler
pub struct Scheduler {
    scheduled_jobs: HashMap<JobId, ScheduledJob>,
}

impl Scheduler {
    /// Create a new scheduler
    #[must_use]
    pub fn new() -> Self {
        Self {
            scheduled_jobs: HashMap::new(),
        }
    }

    /// Schedule a job for a specific time
    ///
    /// # Arguments
    ///
    /// * `job` - The job to schedule
    /// * `datetime` - When to execute the job
    ///
    /// # Errors
    ///
    /// Returns an error if scheduling fails
    pub fn schedule_at(&mut self, job: BatchJob, datetime: DateTime<Utc>) -> Result<()> {
        let job_id = job.id.clone();
        self.scheduled_jobs.insert(
            job_id,
            ScheduledJob {
                job,
                due_time: datetime,
                recurring: false,
                cron_expression: None,
                interval_secs: 3600,
            },
        );
        Ok(())
    }

    /// Schedule a job after a delay
    ///
    /// # Arguments
    ///
    /// * `job` - The job to schedule
    /// * `delay_secs` - Delay in seconds
    ///
    /// # Errors
    ///
    /// Returns an error if scheduling fails
    pub fn schedule_after(&mut self, job: BatchJob, delay_secs: u64) -> Result<()> {
        let job_id = job.id.clone();
        #[allow(clippy::cast_possible_wrap)]
        let due_time = Utc::now() + chrono::Duration::seconds(delay_secs as i64);
        self.scheduled_jobs.insert(
            job_id,
            ScheduledJob {
                job,
                due_time,
                recurring: false,
                cron_expression: None,
                interval_secs: 3600,
            },
        );
        Ok(())
    }

    /// Schedule a recurring job using its `Schedule` field.
    ///
    /// If the job has a `Schedule::Recurring { expression }`, the cron expression is parsed
    /// and the initial due time is computed from it.  For all other schedule variants the job
    /// is scheduled to run in one hour and will be rescheduled by the same interval on each
    /// `get_due_jobs` call.
    ///
    /// # Errors
    ///
    /// Returns an error if scheduling fails
    pub fn schedule_recurring(&mut self, job: BatchJob) -> Result<()> {
        let job_id = job.id.clone();
        let now = Utc::now();

        let (due_time, cron_expression, interval_secs) = match &job.schedule {
            Schedule::Recurring { expression } => {
                let expr = expression.clone();
                let due = next_occurrence_for_expression(&expr, now);
                (due, Some(expr), 3600u64)
            }
            Schedule::After(secs) => {
                let secs = *secs;
                #[allow(clippy::cast_possible_wrap)]
                let duration = chrono::Duration::seconds(secs as i64);
                (now + duration, None, secs)
            }
            _ => (now + chrono::Duration::hours(1), None, 3600u64),
        };

        self.scheduled_jobs.insert(
            job_id,
            ScheduledJob {
                job,
                due_time,
                recurring: true,
                cron_expression,
                interval_secs,
            },
        );
        Ok(())
    }

    /// Get jobs that are due to run
    #[must_use]
    pub fn get_due_jobs(&mut self) -> Vec<BatchJob> {
        let now = Utc::now();
        let mut due_jobs = Vec::new();
        let mut to_remove = Vec::new();
        let mut recurring_to_reschedule = Vec::new();

        for (job_id, scheduled) in &self.scheduled_jobs {
            if scheduled.due_time <= now {
                due_jobs.push(scheduled.job.clone());
                if scheduled.recurring {
                    recurring_to_reschedule.push(job_id.clone());
                } else {
                    to_remove.push(job_id.clone());
                }
            }
        }

        // Remove non-recurring jobs that are due
        for job_id in to_remove {
            self.scheduled_jobs.remove(&job_id);
        }

        // Reschedule recurring jobs: advance due_time to next occurrence
        for job_id in &recurring_to_reschedule {
            if let Some(scheduled) = self.scheduled_jobs.get_mut(job_id) {
                scheduled.due_time = if let Some(ref expr) = scheduled.cron_expression.clone() {
                    next_occurrence_for_expression(expr, now)
                } else {
                    {
                        #[allow(clippy::cast_possible_wrap)]
                        let interval = chrono::Duration::seconds(scheduled.interval_secs as i64);
                        scheduled.due_time + interval
                    }
                };
            }
        }

        due_jobs
    }

    /// Cancel a scheduled job
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job to cancel
    ///
    /// # Errors
    ///
    /// Returns an error if the job is not found
    pub fn cancel(&mut self, job_id: &JobId) -> Result<()> {
        self.scheduled_jobs.remove(job_id);
        Ok(())
    }

    /// Get the number of scheduled jobs
    #[must_use]
    pub fn len(&self) -> usize {
        self.scheduled_jobs.len()
    }

    /// Check if there are any scheduled jobs
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scheduled_jobs.is_empty()
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::FileOperation;

    #[test]
    fn test_scheduler_creation() {
        let scheduler = Scheduler::new();
        assert_eq!(scheduler.len(), 0);
        assert!(scheduler.is_empty());
    }

    #[test]
    fn test_schedule_at() {
        let mut scheduler = Scheduler::new();

        let job = BatchJob::new(
            "test".to_string(),
            crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        let future = Utc::now() + chrono::Duration::hours(1);
        let result = scheduler.schedule_at(job, future);

        assert!(result.is_ok());
        assert_eq!(scheduler.len(), 1);
    }

    #[test]
    fn test_schedule_after() {
        let mut scheduler = Scheduler::new();

        let job = BatchJob::new(
            "test".to_string(),
            crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        let result = scheduler.schedule_after(job, 3600);

        assert!(result.is_ok());
        assert_eq!(scheduler.len(), 1);
    }

    #[test]
    fn test_get_due_jobs() {
        let mut scheduler = Scheduler::new();

        let job = BatchJob::new(
            "test".to_string(),
            crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        // Schedule in the past
        let past = Utc::now() - chrono::Duration::hours(1);
        scheduler
            .schedule_at(job, past)
            .expect("schedule_at should succeed");

        let due_jobs = scheduler.get_due_jobs();
        assert_eq!(due_jobs.len(), 1);

        // Non-recurring job should be removed
        assert_eq!(scheduler.len(), 0);
    }

    #[test]
    fn test_cancel_scheduled_job() {
        let mut scheduler = Scheduler::new();

        let job = BatchJob::new(
            "test".to_string(),
            crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        let job_id = job.id.clone();
        let future = Utc::now() + chrono::Duration::hours(1);
        scheduler
            .schedule_at(job, future)
            .expect("schedule_at should succeed");

        scheduler.cancel(&job_id).expect("cancel should succeed");
        assert_eq!(scheduler.len(), 0);
    }

    #[test]
    fn test_schedule_recurring() {
        let mut scheduler = Scheduler::new();

        let job = BatchJob::new(
            "test".to_string(),
            crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        let result = scheduler.schedule_recurring(job);
        assert!(result.is_ok());
        assert_eq!(scheduler.len(), 1);
    }

    #[test]
    fn test_schedule_recurring_with_cron() {
        let mut scheduler = Scheduler::new();

        let mut job = BatchJob::new(
            "cron-job".to_string(),
            crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );
        // Every 5 minutes
        job.set_schedule(crate::types::Schedule::Recurring {
            expression: "*/5 * * * *".to_string(),
        });

        let result = scheduler.schedule_recurring(job);
        assert!(result.is_ok());
        assert_eq!(scheduler.len(), 1);
    }

    #[test]
    fn test_recurring_job_rescheduled_after_due() {
        let mut scheduler = Scheduler::new();

        let mut job = BatchJob::new(
            "recurring".to_string(),
            crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );
        job.set_schedule(crate::types::Schedule::After(60));

        let job_id = job.id.clone();

        // Insert directly with a past due_time so it fires immediately
        scheduler.scheduled_jobs.insert(
            job_id.clone(),
            ScheduledJob {
                job,
                due_time: Utc::now() - chrono::Duration::seconds(1),
                recurring: true,
                cron_expression: None,
                interval_secs: 60,
            },
        );

        let due_jobs = scheduler.get_due_jobs();
        // Should fire once
        assert_eq!(due_jobs.len(), 1);
        // Recurring job stays in the map
        assert_eq!(scheduler.len(), 1);
        // Due time should have been advanced (now in the future)
        let rescheduled = scheduler
            .scheduled_jobs
            .get(&job_id)
            .expect("failed to get value");
        assert!(rescheduled.due_time > Utc::now());
    }

    #[test]
    fn test_cron_expression_parse_wildcard() {
        let cron = CronExpression::parse("* * * * *");
        assert!(cron.is_some());
        let c = cron.expect("cron should be valid");
        assert!(c.minutes.is_empty());
        assert!(c.hours.is_empty());
    }

    #[test]
    fn test_cron_expression_parse_step() {
        let cron = CronExpression::parse("*/15 * * * *");
        assert!(cron.is_some());
        let c = cron.expect("cron should be valid");
        assert_eq!(c.minutes, vec![0, 15, 30, 45]);
    }

    #[test]
    fn test_cron_expression_parse_range() {
        let cron = CronExpression::parse("0 9-17 * * *");
        assert!(cron.is_some());
        let c = cron.expect("cron should be valid");
        assert_eq!(c.minutes, vec![0]);
        assert_eq!(c.hours, vec![9, 10, 11, 12, 13, 14, 15, 16, 17]);
    }

    #[test]
    fn test_cron_expression_parse_list() {
        let cron = CronExpression::parse("0,30 * * * *");
        assert!(cron.is_some());
        let c = cron.expect("cron should be valid");
        assert_eq!(c.minutes, vec![0, 30]);
    }

    #[test]
    fn test_cron_expression_invalid_fields() {
        assert!(CronExpression::parse("* * *").is_none());
        assert!(CronExpression::parse("*/0 * * * *").is_none());
    }

    #[test]
    fn test_cron_next_after() {
        // Every hour at minute 0
        let cron = CronExpression::parse("0 * * * *").expect("failed to parse");
        let from = Utc::now();
        let next = cron.next_after(from);
        assert!(next > from);
        assert_eq!(next.minute(), 0);
    }
}
