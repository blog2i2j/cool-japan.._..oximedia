// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Multi-criteria job filtering, querying, and sorted pagination.
//!
//! `JobFilter` is a composable predicate builder that evaluates against `Job`
//! instances.  Filters can be combined with logical AND (`and`) or OR (`or`),
//! enabling rich query expressions without external query languages.
//!
//! # Example
//! ```rust
//! use oximedia_jobs::job_filter::{JobFilter, SortField, SortOrder, JobPage};
//! use oximedia_jobs::{Job, JobPayload, Priority, TranscodeParams, JobStatus};
//!
//! let params = TranscodeParams {
//!     input: "a.mp4".into(), output: "b.mp4".into(),
//!     video_codec: "h264".into(), audio_codec: "aac".into(),
//!     video_bitrate: 4_000_000, audio_bitrate: 128_000,
//!     resolution: None, framerate: None,
//!     preset: "fast".into(), hw_accel: None,
//! };
//! let job = Job::new("encode".into(), Priority::High, JobPayload::Transcode(params));
//!
//! let filter = JobFilter::new()
//!     .with_priority(Priority::High)
//!     .with_status(JobStatus::Pending);
//!
//! assert!(filter.matches(&job));
//! ```

use crate::job::{Job, JobStatus, Priority};
use chrono::{DateTime, Utc};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// FilterCriteria — individual predicates
// ---------------------------------------------------------------------------

/// A single filter criterion applied to a `Job`.
#[derive(Debug, Clone)]
enum FilterCriteria {
    /// Match a specific job status.
    Status(JobStatus),
    /// Match any of the given statuses.
    AnyStatus(Vec<JobStatus>),
    /// Match a specific priority.
    Priority(Priority),
    /// Match jobs with priority at least this level.
    MinPriority(Priority),
    /// Match jobs submitted after this timestamp (inclusive).
    CreatedAfter(DateTime<Utc>),
    /// Match jobs submitted before this timestamp (inclusive).
    CreatedBefore(DateTime<Utc>),
    /// Match jobs whose name contains the given substring (case-insensitive).
    NameContains(String),
    /// Match jobs that carry all of the given tags.
    HasAllTags(Vec<String>),
    /// Match jobs that carry any of the given tags.
    HasAnyTag(Vec<String>),
    /// Match jobs whose progress is at or above the given percentage.
    MinProgress(u8),
    /// Match jobs whose progress is at or below the given percentage.
    MaxProgress(u8),
}

impl FilterCriteria {
    /// Evaluate this criterion against a job.
    fn matches(&self, job: &Job) -> bool {
        match self {
            Self::Status(s) => job.status == *s,
            Self::AnyStatus(statuses) => statuses.contains(&job.status),
            Self::Priority(p) => job.priority == *p,
            Self::MinPriority(p) => job.priority >= *p,
            Self::CreatedAfter(dt) => job.created_at >= *dt,
            Self::CreatedBefore(dt) => job.created_at <= *dt,
            Self::NameContains(substr) => {
                job.name.to_lowercase().contains(&substr.to_lowercase())
            }
            Self::HasAllTags(tags) => {
                let job_tags: HashSet<&String> = job.tags.iter().collect();
                tags.iter().all(|t| job_tags.contains(t))
            }
            Self::HasAnyTag(tags) => {
                let job_tags: HashSet<&String> = job.tags.iter().collect();
                tags.iter().any(|t| job_tags.contains(t))
            }
            Self::MinProgress(p) => job.progress >= *p,
            Self::MaxProgress(p) => job.progress <= *p,
        }
    }
}

// ---------------------------------------------------------------------------
// SortField / SortOrder
// ---------------------------------------------------------------------------

/// Field to sort results by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortField {
    /// Sort by job creation time.
    CreatedAt,
    /// Sort by job name (lexicographic).
    Name,
    /// Sort by job priority.
    Priority,
    /// Sort by job progress percentage.
    Progress,
}

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortOrder {
    /// Ascending (oldest first / A→Z / lowest first).
    #[default]
    Ascending,
    /// Descending (newest first / Z→A / highest first).
    Descending,
}

// ---------------------------------------------------------------------------
// JobPage — paginated result
// ---------------------------------------------------------------------------

/// A page of filtered jobs with pagination metadata.
#[derive(Debug, Clone)]
pub struct JobPage {
    /// Jobs on this page.
    pub jobs: Vec<Job>,
    /// Offset (number of items skipped before this page).
    pub offset: usize,
    /// Page size limit used.
    pub limit: usize,
    /// Total number of jobs matching the filter (across all pages).
    pub total: usize,
}

impl JobPage {
    /// Whether there is a next page available.
    #[must_use]
    pub fn has_next_page(&self) -> bool {
        self.offset + self.jobs.len() < self.total
    }

    /// Offset that would yield the next page (if any).
    #[must_use]
    pub fn next_offset(&self) -> Option<usize> {
        if self.has_next_page() {
            Some(self.offset + self.limit)
        } else {
            None
        }
    }

    /// Number of the current page (0-indexed).
    #[must_use]
    pub fn page_number(&self) -> usize {
        self.offset.checked_div(self.limit).unwrap_or(0)
    }

    /// Total number of pages given the current limit.
    #[must_use]
    pub fn total_pages(&self) -> usize {
        self.total
            .saturating_add(self.limit.saturating_sub(1))
            .checked_div(self.limit)
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// JobFilter — composable, builder-style filter
// ---------------------------------------------------------------------------

/// A composable, multi-criteria filter for job collections.
///
/// Build a filter with the fluent API, then call [`JobFilter::matches`] on
/// individual jobs or [`JobFilter::apply`] to filter-and-sort a collection.
#[derive(Debug, Clone, Default)]
pub struct JobFilter {
    criteria: Vec<FilterCriteria>,
    /// Field to sort results by (defaults to `CreatedAt`).
    pub sort_field: SortField,
    /// Sort direction (defaults to `Ascending`).
    pub sort_order: SortOrder,
}

impl Default for SortField {
    fn default() -> Self {
        Self::CreatedAt
    }
}

impl JobFilter {
    /// Create a new empty filter (matches everything).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // --- Criterion setters --------------------------------------------------

    /// Filter by an exact status.
    #[must_use]
    pub fn with_status(mut self, status: JobStatus) -> Self {
        self.criteria.push(FilterCriteria::Status(status));
        self
    }

    /// Filter to jobs whose status is one of the given values.
    #[must_use]
    pub fn with_any_status(mut self, statuses: Vec<JobStatus>) -> Self {
        self.criteria.push(FilterCriteria::AnyStatus(statuses));
        self
    }

    /// Filter by an exact priority.
    #[must_use]
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.criteria.push(FilterCriteria::Priority(priority));
        self
    }

    /// Filter to jobs with priority at least `min`.
    #[must_use]
    pub fn with_min_priority(mut self, min: Priority) -> Self {
        self.criteria.push(FilterCriteria::MinPriority(min));
        self
    }

    /// Filter to jobs created at or after `dt`.
    #[must_use]
    pub fn created_after(mut self, dt: DateTime<Utc>) -> Self {
        self.criteria.push(FilterCriteria::CreatedAfter(dt));
        self
    }

    /// Filter to jobs created at or before `dt`.
    #[must_use]
    pub fn created_before(mut self, dt: DateTime<Utc>) -> Self {
        self.criteria.push(FilterCriteria::CreatedBefore(dt));
        self
    }

    /// Filter to jobs whose name contains `substr` (case-insensitive).
    #[must_use]
    pub fn name_contains(mut self, substr: impl Into<String>) -> Self {
        self.criteria
            .push(FilterCriteria::NameContains(substr.into()));
        self
    }

    /// Filter to jobs that carry **all** of the given tags.
    #[must_use]
    pub fn has_all_tags(mut self, tags: Vec<String>) -> Self {
        self.criteria.push(FilterCriteria::HasAllTags(tags));
        self
    }

    /// Filter to jobs that carry **any** of the given tags.
    #[must_use]
    pub fn has_any_tag(mut self, tags: Vec<String>) -> Self {
        self.criteria.push(FilterCriteria::HasAnyTag(tags));
        self
    }

    /// Filter to jobs whose progress is at least `pct`.
    #[must_use]
    pub fn min_progress(mut self, pct: u8) -> Self {
        self.criteria.push(FilterCriteria::MinProgress(pct));
        self
    }

    /// Filter to jobs whose progress is at most `pct`.
    #[must_use]
    pub fn max_progress(mut self, pct: u8) -> Self {
        self.criteria.push(FilterCriteria::MaxProgress(pct));
        self
    }

    // --- Logical combinators ------------------------------------------------

    /// Combine this filter with `other` using logical AND.
    #[must_use]
    pub fn and(self, other: JobFilter) -> ComposedFilter {
        ComposedFilter::And(Box::new(self), Box::new(other))
    }

    /// Combine this filter with `other` using logical OR.
    #[must_use]
    pub fn or(self, other: JobFilter) -> ComposedFilter {
        ComposedFilter::Or(Box::new(self), Box::new(other))
    }

    // --- Sort ---------------------------------------------------------------

    /// Set the sort field.
    #[must_use]
    pub fn sort_by(mut self, field: SortField) -> Self {
        self.sort_field = field;
        self
    }

    /// Set the sort direction.
    #[must_use]
    pub fn sort_order(mut self, order: SortOrder) -> Self {
        self.sort_order = order;
        self
    }

    // --- Evaluation ---------------------------------------------------------

    /// Test whether a single `job` matches all criteria in this filter.
    #[must_use]
    pub fn matches(&self, job: &Job) -> bool {
        self.criteria.iter().all(|c| c.matches(job))
    }

    /// Apply the filter to a collection, returning a sorted `Vec<Job>`.
    ///
    /// Clones jobs that pass the filter.
    #[must_use]
    pub fn apply<'a>(&self, jobs: impl IntoIterator<Item = &'a Job>) -> Vec<Job> {
        let mut matched: Vec<Job> = jobs
            .into_iter()
            .filter(|j| self.matches(j))
            .cloned()
            .collect();
        self.sort_in_place(&mut matched);
        matched
    }

    /// Apply filter + sort, then return a single page of results.
    #[must_use]
    pub fn paginate<'a>(
        &self,
        jobs: impl IntoIterator<Item = &'a Job>,
        offset: usize,
        limit: usize,
    ) -> JobPage {
        let mut matched = self.apply(jobs);
        let total = matched.len();
        let page_jobs: Vec<Job> = matched.drain(..).skip(offset).take(limit).collect();
        JobPage {
            jobs: page_jobs,
            offset,
            limit,
            total,
        }
    }

    // --- Internal helpers ---------------------------------------------------

    fn sort_in_place(&self, jobs: &mut Vec<Job>) {
        let order = self.sort_order;
        match self.sort_field {
            SortField::CreatedAt => {
                jobs.sort_by(|a, b| {
                    let cmp = a.created_at.cmp(&b.created_at);
                    if order == SortOrder::Descending {
                        cmp.reverse()
                    } else {
                        cmp
                    }
                });
            }
            SortField::Name => {
                jobs.sort_by(|a, b| {
                    let cmp = a.name.cmp(&b.name);
                    if order == SortOrder::Descending {
                        cmp.reverse()
                    } else {
                        cmp
                    }
                });
            }
            SortField::Priority => {
                jobs.sort_by(|a, b| {
                    let cmp = a.priority.cmp(&b.priority);
                    if order == SortOrder::Descending {
                        cmp.reverse()
                    } else {
                        cmp
                    }
                });
            }
            SortField::Progress => {
                jobs.sort_by(|a, b| {
                    let cmp = a.progress.cmp(&b.progress);
                    if order == SortOrder::Descending {
                        cmp.reverse()
                    } else {
                        cmp
                    }
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ComposedFilter — logical combination of two JobFilters
// ---------------------------------------------------------------------------

/// A logical combination of two [`JobFilter`]s.
pub enum ComposedFilter {
    /// Both sub-filters must match.
    And(Box<JobFilter>, Box<JobFilter>),
    /// Either sub-filter must match.
    Or(Box<JobFilter>, Box<JobFilter>),
}

impl ComposedFilter {
    /// Test whether a job matches this composed filter.
    #[must_use]
    pub fn matches(&self, job: &Job) -> bool {
        match self {
            Self::And(a, b) => a.matches(job) && b.matches(job),
            Self::Or(a, b) => a.matches(job) || b.matches(job),
        }
    }

    /// Apply the composed filter to a collection, returning matching jobs.
    #[must_use]
    pub fn apply<'a>(&self, jobs: impl IntoIterator<Item = &'a Job>) -> Vec<Job> {
        jobs.into_iter()
            .filter(|j| self.matches(j))
            .cloned()
            .collect()
    }
}

// ---------------------------------------------------------------------------
// FilterBuilder — convenience builder for common filter patterns
// ---------------------------------------------------------------------------

/// Convenience builder that constructs common `JobFilter` patterns.
pub struct FilterBuilder;

impl FilterBuilder {
    /// Create a filter that matches only pending jobs.
    #[must_use]
    pub fn pending() -> JobFilter {
        JobFilter::new().with_status(JobStatus::Pending)
    }

    /// Create a filter that matches only running jobs.
    #[must_use]
    pub fn running() -> JobFilter {
        JobFilter::new().with_status(JobStatus::Running)
    }

    /// Create a filter that matches failed jobs.
    #[must_use]
    pub fn failed() -> JobFilter {
        JobFilter::new().with_status(JobStatus::Failed)
    }

    /// Create a filter that matches terminal jobs (completed, failed, or cancelled).
    #[must_use]
    pub fn terminal() -> JobFilter {
        JobFilter::new().with_any_status(vec![
            JobStatus::Completed,
            JobStatus::Failed,
            JobStatus::Cancelled,
        ])
    }

    /// Create a filter that matches high-priority pending jobs.
    #[must_use]
    pub fn high_priority_pending() -> JobFilter {
        JobFilter::new()
            .with_status(JobStatus::Pending)
            .with_priority(Priority::High)
    }

    /// Create a filter for jobs with a given tag, sorted newest-first.
    #[must_use]
    pub fn by_tag(tag: impl Into<String>) -> JobFilter {
        JobFilter::new()
            .has_any_tag(vec![tag.into()])
            .sort_by(SortField::CreatedAt)
            .sort_order(SortOrder::Descending)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{Job, JobPayload, Priority, TranscodeParams};

    fn make_transcode_params() -> TranscodeParams {
        TranscodeParams {
            input: "in.mp4".into(),
            output: "out.mp4".into(),
            video_codec: "h264".into(),
            audio_codec: "aac".into(),
            video_bitrate: 4_000_000,
            audio_bitrate: 128_000,
            resolution: None,
            framerate: None,
            preset: "medium".into(),
            hw_accel: None,
        }
    }

    fn make_job(name: &str, priority: Priority) -> Job {
        Job::new(name.into(), priority, JobPayload::Transcode(make_transcode_params()))
    }

    fn make_tagged_job(name: &str, priority: Priority, tags: Vec<String>) -> Job {
        use crate::job::JobBuilder;
        let mut builder = JobBuilder::new(
            name.into(),
            priority,
            JobPayload::Transcode(make_transcode_params()),
        );
        for tag in tags {
            builder = builder.tag(tag);
        }
        builder.build()
    }

    #[test]
    fn test_empty_filter_matches_everything() {
        let filter = JobFilter::new();
        let job = make_job("test", Priority::Normal);
        assert!(filter.matches(&job));
    }

    #[test]
    fn test_status_filter() {
        let filter = JobFilter::new().with_status(JobStatus::Pending);
        let job = make_job("pending-job", Priority::Normal);
        assert_eq!(job.status, JobStatus::Pending);
        assert!(filter.matches(&job));

        let filter_running = JobFilter::new().with_status(JobStatus::Running);
        assert!(!filter_running.matches(&job));
    }

    #[test]
    fn test_priority_filter() {
        let filter = JobFilter::new().with_priority(Priority::High);
        let high_job = make_job("high", Priority::High);
        let low_job = make_job("low", Priority::Low);
        assert!(filter.matches(&high_job));
        assert!(!filter.matches(&low_job));
    }

    #[test]
    fn test_min_priority_filter() {
        let filter = JobFilter::new().with_min_priority(Priority::Normal);
        let high_job = make_job("high", Priority::High);
        let normal_job = make_job("normal", Priority::Normal);
        let low_job = make_job("low", Priority::Low);
        assert!(filter.matches(&high_job));
        assert!(filter.matches(&normal_job));
        assert!(!filter.matches(&low_job));
    }

    #[test]
    fn test_name_contains_filter_case_insensitive() {
        let filter = JobFilter::new().name_contains("TRANSCODE");
        let job = make_job("transcode video", Priority::Normal);
        assert!(filter.matches(&job));

        let job2 = make_job("thumbnail gen", Priority::Normal);
        assert!(!filter.matches(&job2));
    }

    #[test]
    fn test_tag_filters() {
        let filter_all = JobFilter::new()
            .has_all_tags(vec!["video".into(), "hd".into()]);
        let filter_any = JobFilter::new()
            .has_any_tag(vec!["video".into(), "audio".into()]);

        let job_both = make_tagged_job("j1", Priority::Normal, vec!["video".into(), "hd".into()]);
        let job_one = make_tagged_job("j2", Priority::Normal, vec!["video".into()]);
        let job_none = make_tagged_job("j3", Priority::Normal, vec!["image".into()]);

        assert!(filter_all.matches(&job_both));
        assert!(!filter_all.matches(&job_one)); // missing "hd"
        assert!(filter_any.matches(&job_both));
        assert!(filter_any.matches(&job_one));
        assert!(!filter_any.matches(&job_none));
    }

    #[test]
    fn test_progress_filter() {
        let filter = JobFilter::new().min_progress(50).max_progress(80);
        let mut job = make_job("prog", Priority::Normal);
        job.progress = 60;
        assert!(filter.matches(&job));

        job.progress = 40;
        assert!(!filter.matches(&job));

        job.progress = 90;
        assert!(!filter.matches(&job));
    }

    #[test]
    fn test_apply_and_sort() {
        let jobs = vec![
            make_job("charlie", Priority::Low),
            make_job("alpha", Priority::High),
            make_job("bravo", Priority::Normal),
        ];

        let filter = JobFilter::new()
            .sort_by(SortField::Name)
            .sort_order(SortOrder::Ascending);
        let result = filter.apply(&jobs);
        assert_eq!(result[0].name, "alpha");
        assert_eq!(result[1].name, "bravo");
        assert_eq!(result[2].name, "charlie");
    }

    #[test]
    fn test_paginate() {
        let jobs: Vec<Job> = (0..10)
            .map(|i| make_job(&format!("job-{i:02}"), Priority::Normal))
            .collect();

        let filter = JobFilter::new().sort_by(SortField::Name);
        let page = filter.paginate(&jobs, 0, 3);
        assert_eq!(page.jobs.len(), 3);
        assert_eq!(page.total, 10);
        assert!(page.has_next_page());
        assert_eq!(page.next_offset(), Some(3));
        assert_eq!(page.page_number(), 0);
        assert_eq!(page.total_pages(), 4);

        let page2 = filter.paginate(&jobs, 9, 3);
        assert_eq!(page2.jobs.len(), 1);
        assert!(!page2.has_next_page());
    }

    #[test]
    fn test_composed_and_filter() {
        let f1 = JobFilter::new().with_priority(Priority::High);
        let f2 = JobFilter::new().with_status(JobStatus::Pending);
        let composed = f1.and(f2);

        let job_match = make_job("match", Priority::High);
        assert_eq!(job_match.status, JobStatus::Pending);
        assert!(composed.matches(&job_match));

        let job_no_match = make_job("low-pending", Priority::Low);
        assert!(!composed.matches(&job_no_match));
    }

    #[test]
    fn test_any_status_filter() {
        let filter = JobFilter::new()
            .with_any_status(vec![JobStatus::Failed, JobStatus::Cancelled]);
        let pending_job = make_job("p", Priority::Normal);
        assert!(!filter.matches(&pending_job)); // pending is not in the list

        // Manually check a failed job by testing the predicate directly
        let criteria = FilterCriteria::AnyStatus(vec![JobStatus::Failed, JobStatus::Cancelled]);
        let mut job = make_job("f", Priority::Normal);
        job.status = JobStatus::Failed;
        assert!(criteria.matches(&job));
    }

    #[test]
    fn test_filter_builder_helpers() {
        let job = make_job("hi", Priority::High);
        assert!(FilterBuilder::pending().matches(&job));
        assert!(!FilterBuilder::running().matches(&job));
        assert!(FilterBuilder::high_priority_pending().matches(&job));

        let tagged = make_tagged_job("tagged", Priority::Normal, vec!["video".into()]);
        assert!(FilterBuilder::by_tag("video").matches(&tagged));
        assert!(!FilterBuilder::by_tag("audio").matches(&tagged));
    }

    #[test]
    fn test_sort_by_priority_descending() {
        let jobs = vec![
            make_job("low", Priority::Low),
            make_job("high", Priority::High),
            make_job("normal", Priority::Normal),
        ];
        let filter = JobFilter::new()
            .sort_by(SortField::Priority)
            .sort_order(SortOrder::Descending);
        let result = filter.apply(&jobs);
        assert_eq!(result[0].priority, Priority::High);
        assert_eq!(result[1].priority, Priority::Normal);
        assert_eq!(result[2].priority, Priority::Low);
    }
}
