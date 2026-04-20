//! Watch folder automation for directory-based transcode monitoring.
//!
//! `TranscodeWatcher` polls a source directory for new media files and
//! dispatches transcode jobs automatically.  The implementation is pure Rust
//! (no `inotify`/`kqueue` bindings required) using a polling loop with
//! configurable interval.

#![allow(dead_code)]

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::{Result, TranscodeConfig, TranscodeError};

/// Action to take when a file has been processed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostProcessAction {
    /// Leave the file in the watch folder.
    Leave,
    /// Move the file to the `done` sub-directory.
    MoveToDone,
    /// Delete the file from the watch folder.
    Delete,
}

/// Policy for selecting the output directory of each transcode job.
#[derive(Debug, Clone)]
pub enum OutputLocation {
    /// Place output in a fixed directory.
    Fixed(PathBuf),
    /// Place output next to the input, using the same name with a new extension.
    SiblingWithExtension(String),
    /// Place output in a `done/` sub-directory of the watch folder.
    DoneSubDir,
}

/// Configuration for `TranscodeWatcher`.
#[derive(Debug, Clone)]
pub struct WatchConfig {
    /// Directory to monitor for incoming files.
    pub watch_dir: PathBuf,
    /// File extensions to accept (lower-case, without leading dot).
    pub accepted_extensions: Vec<String>,
    /// How to determine the output path for each new file.
    pub output_location: OutputLocation,
    /// What to do with an input file after a successful transcode.
    pub on_success: PostProcessAction,
    /// What to do with an input file after a failed transcode.
    pub on_failure: PostProcessAction,
    /// How often to scan the watch directory (milliseconds).
    pub poll_interval_ms: u64,
    /// Base `TranscodeConfig` applied to every discovered file.
    pub base_config: TranscodeConfig,
    /// Maximum number of concurrent jobs.
    pub max_concurrent: usize,
}

impl WatchConfig {
    /// Creates a `WatchConfig` with sensible defaults.
    ///
    /// Accepts common video extensions, moves successful files to `done/`,
    /// and polls every 5 seconds.
    #[must_use]
    pub fn new(watch_dir: impl Into<PathBuf>) -> Self {
        Self {
            watch_dir: watch_dir.into(),
            accepted_extensions: vec![
                "mp4".into(),
                "mkv".into(),
                "mov".into(),
                "avi".into(),
                "webm".into(),
                "mxf".into(),
                "ts".into(),
                "m2ts".into(),
            ],
            output_location: OutputLocation::DoneSubDir,
            on_success: PostProcessAction::MoveToDone,
            on_failure: PostProcessAction::Leave,
            poll_interval_ms: 5_000,
            base_config: TranscodeConfig::default(),
            max_concurrent: 2,
        }
    }

    /// Sets the output location policy.
    #[must_use]
    pub fn output_location(mut self, loc: OutputLocation) -> Self {
        self.output_location = loc;
        self
    }

    /// Sets what happens to the source file after a successful transcode.
    #[must_use]
    pub fn on_success(mut self, action: PostProcessAction) -> Self {
        self.on_success = action;
        self
    }

    /// Sets what happens to the source file after a failed transcode.
    #[must_use]
    pub fn on_failure(mut self, action: PostProcessAction) -> Self {
        self.on_failure = action;
        self
    }

    /// Sets the polling interval in milliseconds.
    #[must_use]
    pub fn poll_interval_ms(mut self, ms: u64) -> Self {
        self.poll_interval_ms = ms;
        self
    }

    /// Overrides the base `TranscodeConfig`.
    #[must_use]
    pub fn base_config(mut self, config: TranscodeConfig) -> Self {
        self.base_config = config;
        self
    }

    /// Sets the maximum number of concurrent jobs.
    #[must_use]
    pub fn max_concurrent(mut self, n: usize) -> Self {
        self.max_concurrent = n;
        self
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the watch directory does not exist.
    pub fn validate(&self) -> Result<()> {
        if !self.watch_dir.exists() {
            return Err(TranscodeError::InvalidInput(format!(
                "Watch directory does not exist: {}",
                self.watch_dir.display()
            )));
        }
        if self.max_concurrent == 0 {
            return Err(TranscodeError::InvalidInput(
                "max_concurrent must be at least 1".into(),
            ));
        }
        Ok(())
    }
}

/// Status of a single watched file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatchFileStatus {
    /// Waiting to be processed.
    Pending,
    /// Currently being transcoded.
    Processing,
    /// Transcoded successfully.
    Done,
    /// Transcode failed; message contains the reason.
    Failed(String),
}

/// A discovered file entry in the watch queue.
#[derive(Debug, Clone)]
pub struct WatchEntry {
    /// Original source file path.
    pub source: PathBuf,
    /// Computed output path.
    pub output: PathBuf,
    /// Current status.
    pub status: WatchFileStatus,
}

impl WatchEntry {
    /// Creates a new watch entry.
    #[must_use]
    pub fn new(source: PathBuf, output: PathBuf) -> Self {
        Self {
            source,
            output,
            status: WatchFileStatus::Pending,
        }
    }
}

/// Directory-based transcode watcher.
///
/// Call [`TranscodeWatcher::scan`] to detect new files, and
/// [`TranscodeWatcher::drain_pending`] to obtain `TranscodeConfig` values
/// ready for submission to the job queue.
pub struct TranscodeWatcher {
    config: WatchConfig,
    /// Paths already seen (regardless of processing status).
    seen: HashSet<PathBuf>,
    /// Queue of watch entries waiting to be processed.
    queue: Vec<WatchEntry>,
}

impl TranscodeWatcher {
    /// Creates a new watcher from `config`.
    #[must_use]
    pub fn new(config: WatchConfig) -> Self {
        Self {
            config,
            seen: HashSet::new(),
            queue: Vec::new(),
        }
    }

    /// Returns the watcher configuration.
    #[must_use]
    pub fn config(&self) -> &WatchConfig {
        &self.config
    }

    /// Returns the poll interval as a [`Duration`].
    #[must_use]
    pub fn poll_interval(&self) -> Duration {
        Duration::from_millis(self.config.poll_interval_ms)
    }

    /// Scans the watch directory for new eligible files.
    ///
    /// Returns the number of new files enqueued.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read.
    pub fn scan(&mut self) -> Result<usize> {
        let entries = std::fs::read_dir(&self.config.watch_dir).map_err(|e| {
            TranscodeError::IoError(format!(
                "Cannot read watch dir '{}': {e}",
                self.config.watch_dir.display()
            ))
        })?;

        let mut new_count = 0usize;

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if self.seen.contains(&path) {
                continue;
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(str::to_lowercase)
                .unwrap_or_default();
            if !self.config.accepted_extensions.iter().any(|a| a == &ext) {
                continue;
            }

            let output = self.resolve_output(&path);
            self.seen.insert(path.clone());
            self.queue.push(WatchEntry::new(path, output));
            new_count += 1;
        }

        Ok(new_count)
    }

    /// Returns all pending entries (without removing them from the queue).
    #[must_use]
    pub fn pending(&self) -> Vec<&WatchEntry> {
        self.queue
            .iter()
            .filter(|e| e.status == WatchFileStatus::Pending)
            .collect()
    }

    /// Drains all pending entries into a `Vec<TranscodeConfig>` for submission
    /// to the job queue, marking each entry as `Processing`.
    pub fn drain_pending(&mut self) -> Vec<(WatchEntry, TranscodeConfig)> {
        let mut out = Vec::new();

        for entry in &mut self.queue {
            if entry.status != WatchFileStatus::Pending {
                continue;
            }
            entry.status = WatchFileStatus::Processing;

            let mut job = self.config.base_config.clone();
            job.input = entry.source.to_str().map(String::from);
            job.output = entry.output.to_str().map(String::from);

            out.push((entry.clone(), job));
        }

        out
    }

    /// Marks a watch entry as successfully processed and applies the configured
    /// post-process action (move / delete / leave).
    ///
    /// # Errors
    ///
    /// Returns an error if the file move or delete operation fails.
    pub fn mark_done(&mut self, source: &Path) -> Result<()> {
        self.update_status(source, WatchFileStatus::Done);

        match self.config.on_success {
            PostProcessAction::Leave => {}
            PostProcessAction::Delete => {
                std::fs::remove_file(source).map_err(|e| {
                    TranscodeError::IoError(format!("Failed to delete '{}': {e}", source.display()))
                })?;
            }
            PostProcessAction::MoveToDone => {
                self.move_to_done_dir(source)?;
            }
        }

        Ok(())
    }

    /// Marks a watch entry as failed and applies the configured on-failure action.
    ///
    /// # Errors
    ///
    /// Returns an error if the file operation fails.
    pub fn mark_failed(&mut self, source: &Path, reason: &str) -> Result<()> {
        self.update_status(source, WatchFileStatus::Failed(reason.to_string()));

        match self.config.on_failure {
            PostProcessAction::Leave => {}
            PostProcessAction::Delete => {
                std::fs::remove_file(source).map_err(|e| {
                    TranscodeError::IoError(format!("Failed to delete '{}': {e}", source.display()))
                })?;
            }
            PostProcessAction::MoveToDone => {
                self.move_to_done_dir(source)?;
            }
        }

        Ok(())
    }

    /// Returns the total number of queued entries.
    #[must_use]
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Returns the number of entries in each status category.
    #[must_use]
    pub fn status_counts(&self) -> WatchStatusCounts {
        let mut counts = WatchStatusCounts::default();
        for entry in &self.queue {
            match entry.status {
                WatchFileStatus::Pending => counts.pending += 1,
                WatchFileStatus::Processing => counts.processing += 1,
                WatchFileStatus::Done => counts.done += 1,
                WatchFileStatus::Failed(_) => counts.failed += 1,
            }
        }
        counts
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn update_status(&mut self, source: &Path, new_status: WatchFileStatus) {
        for entry in &mut self.queue {
            if entry.source == source {
                entry.status = new_status;
                return;
            }
        }
    }

    fn resolve_output(&self, source: &Path) -> PathBuf {
        match &self.config.output_location {
            OutputLocation::Fixed(dir) => {
                let filename = source
                    .file_name()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("output.mkv"));
                dir.join(filename)
            }
            OutputLocation::SiblingWithExtension(ext) => {
                let mut out = source.to_path_buf();
                out.set_extension(ext.trim_start_matches('.'));
                out
            }
            OutputLocation::DoneSubDir => {
                let done_dir = self.config.watch_dir.join("done");
                let filename = source
                    .file_name()
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("output.mkv"));
                done_dir.join(filename)
            }
        }
    }

    fn move_to_done_dir(&self, source: &Path) -> Result<()> {
        let done_dir = self.config.watch_dir.join("done");
        std::fs::create_dir_all(&done_dir)
            .map_err(|e| TranscodeError::IoError(format!("Cannot create done dir: {e}")))?;
        let dest = done_dir.join(
            source
                .file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new("moved_file")),
        );
        std::fs::rename(source, &dest).map_err(|e| {
            TranscodeError::IoError(format!(
                "Cannot move '{}' → '{}': {e}",
                source.display(),
                dest.display()
            ))
        })
    }
}

/// Snapshot of watch queue status counts.
#[derive(Debug, Clone, Default)]
pub struct WatchStatusCounts {
    /// Number of entries awaiting processing.
    pub pending: usize,
    /// Number of entries currently being transcoded.
    pub processing: usize,
    /// Number of successfully completed entries.
    pub done: usize,
    /// Number of failed entries.
    pub failed: usize,
}

// ─── File stability detection ─────────────────────────────────────────────────

/// Configuration for file stability detection.
///
/// Waits until a file has stopped growing before considering it ready
/// for processing. This prevents partial files (still being copied or
/// written by another process) from entering the transcode queue.
#[derive(Debug, Clone)]
pub struct FileStabilityConfig {
    /// Number of consecutive stable checks required before a file is
    /// considered complete.
    pub required_stable_checks: u32,
    /// Interval between stability checks in milliseconds.
    pub check_interval_ms: u64,
    /// Minimum file size in bytes before stability checks begin.
    pub min_file_size: u64,
}

impl Default for FileStabilityConfig {
    fn default() -> Self {
        Self {
            required_stable_checks: 3,
            check_interval_ms: 2_000,
            min_file_size: 1024,
        }
    }
}

impl FileStabilityConfig {
    /// Creates a new stability config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of required stable checks.
    #[must_use]
    pub fn required_checks(mut self, n: u32) -> Self {
        self.required_stable_checks = n;
        self
    }

    /// Sets the check interval in milliseconds.
    #[must_use]
    pub fn check_interval_ms(mut self, ms: u64) -> Self {
        self.check_interval_ms = ms;
        self
    }

    /// Sets the minimum file size.
    #[must_use]
    pub fn min_file_size(mut self, size: u64) -> Self {
        self.min_file_size = size;
        self
    }
}

/// Tracks stability state for a single file.
#[derive(Debug, Clone)]
pub struct FileStabilityTracker {
    /// Path being tracked.
    path: PathBuf,
    /// Last observed file size.
    last_size: u64,
    /// Number of consecutive stable readings.
    stable_count: u32,
    /// Whether the file has been declared stable.
    is_stable: bool,
}

impl FileStabilityTracker {
    /// Creates a new tracker for the given path.
    #[must_use]
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            last_size: 0,
            stable_count: 0,
            is_stable: false,
        }
    }

    /// Checks the file and updates stability state.
    ///
    /// Returns `true` if the file is now considered stable.
    pub fn check(&mut self, config: &FileStabilityConfig) -> bool {
        if self.is_stable {
            return true;
        }
        let current_size = std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);

        if current_size < config.min_file_size {
            self.stable_count = 0;
            self.last_size = current_size;
            return false;
        }

        if current_size == self.last_size {
            self.stable_count += 1;
        } else {
            self.stable_count = 0;
        }
        self.last_size = current_size;

        if self.stable_count >= config.required_stable_checks {
            self.is_stable = true;
        }
        self.is_stable
    }

    /// Returns true if the file has been declared stable.
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.is_stable
    }

    /// Returns the path being tracked.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the last observed file size.
    #[must_use]
    pub fn last_size(&self) -> u64 {
        self.last_size
    }
}

// ─── Hot folder chains ────────────────────────────────────────────────────────

/// A chain of watch folders where the output of one feeds into the next.
///
/// This enables multi-step processing workflows, for example:
/// 1. Ingest folder → transcode to intermediate format
/// 2. Intermediate folder → apply effects / normalisation
/// 3. Final folder → encode to delivery format
#[derive(Debug, Clone)]
pub struct HotFolderChain {
    /// Ordered list of watch configurations forming the chain.
    stages: Vec<WatchConfig>,
}

impl HotFolderChain {
    /// Creates a new empty chain.
    #[must_use]
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Appends a stage to the chain.
    ///
    /// The output directory of the previous stage should match the watch
    /// directory of this stage for seamless chaining.
    pub fn add_stage(&mut self, config: WatchConfig) {
        self.stages.push(config);
    }

    /// Returns the number of stages in the chain.
    #[must_use]
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Returns the stages as a slice.
    #[must_use]
    pub fn stages(&self) -> &[WatchConfig] {
        &self.stages
    }

    /// Validates that the chain is well-formed.
    ///
    /// Checks that each stage's output directory matches the next stage's
    /// watch directory (for `DoneSubDir` and `Fixed` output locations).
    ///
    /// # Errors
    ///
    /// Returns an error if the chain is empty or directories don't align.
    pub fn validate(&self) -> Result<()> {
        if self.stages.is_empty() {
            return Err(TranscodeError::InvalidInput(
                "Hot folder chain has no stages".into(),
            ));
        }

        for i in 0..self.stages.len().saturating_sub(1) {
            let current = &self.stages[i];
            let next = &self.stages[i + 1];

            let output_dir = match &current.output_location {
                OutputLocation::Fixed(dir) => Some(dir.clone()),
                OutputLocation::DoneSubDir => Some(current.watch_dir.join("done")),
                OutputLocation::SiblingWithExtension(_) => None,
            };

            if let Some(out_dir) = output_dir {
                if out_dir != next.watch_dir {
                    return Err(TranscodeError::InvalidInput(format!(
                        "Stage {} output dir '{}' does not match stage {} watch dir '{}'",
                        i,
                        out_dir.display(),
                        i + 1,
                        next.watch_dir.display()
                    )));
                }
            }
        }

        Ok(())
    }
}

impl Default for HotFolderChain {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Filename pattern matching ────────────────────────────────────────────────

/// Pattern-based file filter for selective watch folder processing.
///
/// Uses simple glob-like patterns (not full regex, to avoid a regex dependency)
/// to match filenames. Supports `*` wildcard and case-insensitive matching.
#[derive(Debug, Clone)]
pub struct FilenamePattern {
    /// The raw pattern string.
    pattern: String,
    /// Whether matching is case-insensitive.
    case_insensitive: bool,
}

impl FilenamePattern {
    /// Creates a new filename pattern.
    #[must_use]
    pub fn new(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            case_insensitive: true,
        }
    }

    /// Sets case sensitivity.
    #[must_use]
    pub fn case_insensitive(mut self, ci: bool) -> Self {
        self.case_insensitive = ci;
        self
    }

    /// Tests whether the given filename matches this pattern.
    ///
    /// Supports `*` as a wildcard matching zero or more characters.
    #[must_use]
    pub fn matches(&self, filename: &str) -> bool {
        let (pat, name) = if self.case_insensitive {
            (self.pattern.to_lowercase(), filename.to_lowercase())
        } else {
            (self.pattern.clone(), filename.to_string())
        };
        Self::glob_match(&pat, &name)
    }

    /// Simple glob matching with `*` wildcard.
    fn glob_match(pattern: &str, text: &str) -> bool {
        let pat_chars: Vec<char> = pattern.chars().collect();
        let txt_chars: Vec<char> = text.chars().collect();
        let (plen, tlen) = (pat_chars.len(), txt_chars.len());

        // DP approach for wildcard matching
        let mut dp = vec![vec![false; tlen + 1]; plen + 1];
        dp[0][0] = true;

        // Handle leading *
        for (i, &pc) in pat_chars.iter().enumerate() {
            if pc == '*' {
                dp[i + 1][0] = dp[i][0];
            } else {
                break;
            }
        }

        for i in 1..=plen {
            for j in 1..=tlen {
                if pat_chars[i - 1] == '*' {
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                } else if pat_chars[i - 1] == '?' || pat_chars[i - 1] == txt_chars[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }

        dp[plen][tlen]
    }

    /// Returns the raw pattern string.
    #[must_use]
    pub fn pattern(&self) -> &str {
        &self.pattern
    }
}

// ─── Retry with exponential backoff ───────────────────────────────────────────

/// Configuration for retry with exponential backoff.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Initial delay before the first retry (milliseconds).
    pub initial_delay_ms: u64,
    /// Multiplier applied to the delay after each retry.
    pub backoff_multiplier: f64,
    /// Maximum delay between retries (milliseconds).
    pub max_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 1_000,
            backoff_multiplier: 2.0,
            max_delay_ms: 30_000,
        }
    }
}

impl RetryConfig {
    /// Creates a new retry config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of retries.
    #[must_use]
    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// Sets the initial delay in milliseconds.
    #[must_use]
    pub fn initial_delay_ms(mut self, ms: u64) -> Self {
        self.initial_delay_ms = ms;
        self
    }

    /// Sets the backoff multiplier.
    #[must_use]
    pub fn backoff_multiplier(mut self, m: f64) -> Self {
        self.backoff_multiplier = m;
        self
    }

    /// Sets the maximum delay in milliseconds.
    #[must_use]
    pub fn max_delay_ms(mut self, ms: u64) -> Self {
        self.max_delay_ms = ms;
        self
    }

    /// Computes the delay for the given attempt number (0-based).
    #[must_use]
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::from_millis(self.initial_delay_ms);
        }
        let delay = self.initial_delay_ms as f64 * self.backoff_multiplier.powi(attempt as i32);
        let clamped = delay.min(self.max_delay_ms as f64) as u64;
        Duration::from_millis(clamped)
    }
}

/// Tracks retry state for a single file.
#[derive(Debug, Clone)]
pub struct RetryTracker {
    /// Path of the file being retried.
    pub path: PathBuf,
    /// Number of attempts made so far.
    pub attempts: u32,
    /// Last error message.
    pub last_error: Option<String>,
}

impl RetryTracker {
    /// Creates a new retry tracker.
    #[must_use]
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            attempts: 0,
            last_error: None,
        }
    }

    /// Records a failed attempt.
    pub fn record_failure(&mut self, error: &str) {
        self.attempts += 1;
        self.last_error = Some(error.to_string());
    }

    /// Returns whether more retries are allowed given the config.
    #[must_use]
    pub fn can_retry(&self, config: &RetryConfig) -> bool {
        self.attempts < config.max_retries
    }

    /// Returns the delay before the next retry.
    #[must_use]
    pub fn next_delay(&self, config: &RetryConfig) -> Duration {
        config.delay_for_attempt(self.attempts)
    }
}

// ─── Watch folder statistics ──────────────────────────────────────────────────

/// Statistics for a watch folder's processing activity.
#[derive(Debug, Clone, Default)]
pub struct WatchFolderStats {
    /// Total number of files processed successfully.
    pub processed_count: u64,
    /// Total number of files that failed processing.
    pub error_count: u64,
    /// Total processing time in milliseconds across all successful jobs.
    pub total_processing_time_ms: u64,
    /// Total bytes processed (input file sizes).
    pub total_bytes_processed: u64,
    /// Minimum processing time in milliseconds.
    pub min_processing_time_ms: Option<u64>,
    /// Maximum processing time in milliseconds.
    pub max_processing_time_ms: Option<u64>,
}

impl WatchFolderStats {
    /// Creates a new empty statistics tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a successful processing event.
    pub fn record_success(&mut self, processing_time_ms: u64, file_size_bytes: u64) {
        self.processed_count += 1;
        self.total_processing_time_ms += processing_time_ms;
        self.total_bytes_processed += file_size_bytes;

        self.min_processing_time_ms = Some(
            self.min_processing_time_ms
                .map_or(processing_time_ms, |m| m.min(processing_time_ms)),
        );
        self.max_processing_time_ms = Some(
            self.max_processing_time_ms
                .map_or(processing_time_ms, |m| m.max(processing_time_ms)),
        );
    }

    /// Records a failed processing event.
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }

    /// Returns the average processing time in milliseconds, or `None` if no files processed.
    #[must_use]
    pub fn avg_processing_time_ms(&self) -> Option<u64> {
        if self.processed_count == 0 {
            return None;
        }
        Some(self.total_processing_time_ms / self.processed_count)
    }

    /// Returns the success rate as a fraction [0.0, 1.0].
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total = self.processed_count + self.error_count;
        if total == 0 {
            return 1.0;
        }
        self.processed_count as f64 / total as f64
    }

    /// Returns the average throughput in bytes per second, or `None` if no data.
    #[must_use]
    pub fn avg_throughput_bps(&self) -> Option<f64> {
        if self.total_processing_time_ms == 0 || self.total_bytes_processed == 0 {
            return None;
        }
        let secs = self.total_processing_time_ms as f64 / 1000.0;
        Some(self.total_bytes_processed as f64 / secs)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;
    use std::fs;

    fn make_temp_dir(suffix: &str) -> PathBuf {
        let dir = temp_dir().join(format!("oximedia_watch_test_{suffix}"));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn touch(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, b"fake media").expect("create temp file");
        path
    }

    #[test]
    fn test_watch_config_new() {
        let cfg =
            WatchConfig::new(std::env::temp_dir().join("oximedia-transcode-watch-folder-watch"));
        assert!(!cfg.accepted_extensions.is_empty());
        assert_eq!(cfg.max_concurrent, 2);
        assert_eq!(cfg.poll_interval_ms, 5_000);
    }

    #[test]
    fn test_watch_config_validate_missing_dir() {
        let cfg = WatchConfig::new("/nonexistent/path/for/oximedia_test");
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_watch_config_validate_ok() {
        let dir = make_temp_dir("cfg_ok");
        let cfg = WatchConfig::new(&dir);
        assert!(cfg.validate().is_ok());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_scan_detects_new_files() {
        let dir = make_temp_dir("scan");
        touch(&dir, "video.mp4");
        touch(&dir, "clip.mkv");
        touch(&dir, "readme.txt"); // ignored

        let cfg = WatchConfig::new(&dir);
        let mut watcher = TranscodeWatcher::new(cfg);
        let count = watcher.scan().expect("scan ok");
        assert_eq!(count, 2);
        assert_eq!(watcher.queue_len(), 2);

        // Second scan should not re-enqueue
        let count2 = watcher.scan().expect("scan ok");
        assert_eq!(count2, 0);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_drain_pending_creates_configs() {
        let dir = make_temp_dir("drain");
        touch(&dir, "a.mp4");

        let cfg = WatchConfig::new(&dir);
        let mut watcher = TranscodeWatcher::new(cfg);
        watcher.scan().expect("scan ok");

        let drained = watcher.drain_pending();
        assert_eq!(drained.len(), 1);
        let (entry, job) = &drained[0];
        assert!(entry.source.ends_with("a.mp4"));
        assert!(job.input.is_some());
        assert!(job.output.is_some());

        // After drain, pending count should be 0
        assert_eq!(watcher.pending().len(), 0);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_mark_done_updates_status() {
        let dir = make_temp_dir("mark_done");
        let file = touch(&dir, "b.mp4");

        let cfg = WatchConfig::new(&dir).on_success(PostProcessAction::Leave);
        let mut watcher = TranscodeWatcher::new(cfg);
        watcher.scan().expect("scan ok");
        watcher.drain_pending();

        watcher.mark_done(&file).expect("mark done ok");

        let counts = watcher.status_counts();
        assert_eq!(counts.done, 1);
        assert_eq!(counts.failed, 0);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_mark_failed_updates_status() {
        let dir = make_temp_dir("mark_failed");
        let file = touch(&dir, "c.mp4");

        let cfg = WatchConfig::new(&dir).on_failure(PostProcessAction::Leave);
        let mut watcher = TranscodeWatcher::new(cfg);
        watcher.scan().expect("scan ok");
        watcher.drain_pending();

        watcher
            .mark_failed(&file, "codec not found")
            .expect("mark failed ok");

        let counts = watcher.status_counts();
        assert_eq!(counts.failed, 1);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_status_counts() {
        let dir = make_temp_dir("counts");
        touch(&dir, "x.mp4");
        touch(&dir, "y.mkv");

        let cfg = WatchConfig::new(&dir);
        let mut watcher = TranscodeWatcher::new(cfg);
        watcher.scan().expect("scan ok");

        let counts = watcher.status_counts();
        assert_eq!(counts.pending, 2);
        assert_eq!(counts.processing, 0);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_poll_interval() {
        let cfg = WatchConfig::new("/tmp").poll_interval_ms(2000);
        let watcher = TranscodeWatcher::new(cfg);
        assert_eq!(watcher.poll_interval(), Duration::from_secs(2));
    }

    #[test]
    fn test_output_location_sibling() {
        let dir = make_temp_dir("sibling");
        touch(&dir, "d.mp4");

        let cfg = WatchConfig::new(&dir)
            .output_location(OutputLocation::SiblingWithExtension("mkv".into()));
        let mut watcher = TranscodeWatcher::new(cfg);
        watcher.scan().expect("scan ok");

        let entry = &watcher.queue[0];
        assert!(entry
            .output
            .extension()
            .map(|e| e == "mkv")
            .unwrap_or(false));

        fs::remove_dir_all(&dir).ok();
    }

    // ── File stability tests ─────────────────────────────────────────────────

    #[test]
    fn test_stability_config_defaults() {
        let cfg = FileStabilityConfig::default();
        assert_eq!(cfg.required_stable_checks, 3);
        assert_eq!(cfg.check_interval_ms, 2000);
        assert_eq!(cfg.min_file_size, 1024);
    }

    #[test]
    fn test_stability_config_builder() {
        let cfg = FileStabilityConfig::new()
            .required_checks(5)
            .check_interval_ms(1000)
            .min_file_size(4096);
        assert_eq!(cfg.required_stable_checks, 5);
        assert_eq!(cfg.check_interval_ms, 1000);
        assert_eq!(cfg.min_file_size, 4096);
    }

    #[test]
    fn test_stability_tracker_stable_file() {
        let dir = make_temp_dir("stability");
        let path = dir.join("stable.mp4");
        // Write a file larger than default min_file_size
        fs::write(&path, vec![0u8; 2048]).expect("write ok");

        let cfg = FileStabilityConfig::new().required_checks(2);
        let mut tracker = FileStabilityTracker::new(path);

        // First check: sets baseline
        assert!(!tracker.check(&cfg));
        // Second check: size unchanged → stable_count = 1
        assert!(!tracker.check(&cfg));
        // Third check: stable_count = 2 → stable!
        assert!(tracker.check(&cfg));
        assert!(tracker.is_stable());
        assert_eq!(tracker.last_size(), 2048);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_stability_tracker_growing_file() {
        let dir = make_temp_dir("growing");
        let path = dir.join("growing.mp4");
        fs::write(&path, vec![0u8; 2048]).expect("write ok");

        let cfg = FileStabilityConfig::new().required_checks(2);
        let mut tracker = FileStabilityTracker::new(path.clone());

        tracker.check(&cfg); // baseline
        tracker.check(&cfg); // stable 1

        // File grows
        fs::write(&path, vec![0u8; 4096]).expect("grow ok");
        assert!(!tracker.check(&cfg)); // reset

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_stability_tracker_too_small() {
        let dir = make_temp_dir("small");
        let path = dir.join("tiny.mp4");
        fs::write(&path, b"x").expect("write ok");

        let cfg = FileStabilityConfig::new().min_file_size(1024);
        let mut tracker = FileStabilityTracker::new(path);

        for _ in 0..10 {
            assert!(!tracker.check(&cfg));
        }

        fs::remove_dir_all(&dir).ok();
    }

    // ── Hot folder chain tests ───────────────────────────────────────────────

    #[test]
    fn test_hot_folder_chain_empty() {
        let chain = HotFolderChain::new();
        assert_eq!(chain.stage_count(), 0);
        assert!(chain.validate().is_err());
    }

    #[test]
    fn test_hot_folder_chain_single_stage() {
        let dir = make_temp_dir("chain1");
        let mut chain = HotFolderChain::new();
        chain.add_stage(WatchConfig::new(&dir));
        assert_eq!(chain.stage_count(), 1);
        assert!(chain.validate().is_ok());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_hot_folder_chain_two_stages_aligned() {
        let dir1 = make_temp_dir("chain2a");
        let dir2 = dir1.join("done");
        fs::create_dir_all(&dir2).expect("create done dir");

        let mut chain = HotFolderChain::new();
        chain.add_stage(WatchConfig::new(&dir1)); // output = dir1/done
        chain.add_stage(WatchConfig::new(&dir2)); // watch = dir1/done
        assert_eq!(chain.stage_count(), 2);
        assert!(chain.validate().is_ok());

        fs::remove_dir_all(&dir1).ok();
    }

    #[test]
    fn test_hot_folder_chain_misaligned() {
        let dir1 = make_temp_dir("chain3a");
        let dir2 = make_temp_dir("chain3b");

        let mut chain = HotFolderChain::new();
        chain.add_stage(WatchConfig::new(&dir1)); // output = dir1/done
        chain.add_stage(WatchConfig::new(&dir2)); // watch = dir2 (mismatch)
        assert!(chain.validate().is_err());

        fs::remove_dir_all(&dir1).ok();
        fs::remove_dir_all(&dir2).ok();
    }

    // ── Filename pattern tests ───────────────────────────────────────────────

    #[test]
    fn test_filename_pattern_exact() {
        let p = FilenamePattern::new("video.mp4");
        assert!(p.matches("video.mp4"));
        assert!(p.matches("VIDEO.MP4")); // case insensitive
        assert!(!p.matches("audio.mp4"));
    }

    #[test]
    fn test_filename_pattern_wildcard() {
        let p = FilenamePattern::new("*.mp4");
        assert!(p.matches("video.mp4"));
        assert!(p.matches("CLIP.MP4"));
        assert!(!p.matches("video.mkv"));
    }

    #[test]
    fn test_filename_pattern_wildcard_prefix() {
        let p = FilenamePattern::new("raw_*");
        assert!(p.matches("raw_clip.mp4"));
        assert!(p.matches("raw_"));
        assert!(!p.matches("clip_raw.mp4"));
    }

    #[test]
    fn test_filename_pattern_multiple_wildcards() {
        let p = FilenamePattern::new("*_final_*");
        assert!(p.matches("clip_final_v2.mp4"));
        assert!(!p.matches("clip_draft_v2.mp4"));
    }

    #[test]
    fn test_filename_pattern_case_sensitive() {
        let p = FilenamePattern::new("Video.mp4").case_insensitive(false);
        assert!(p.matches("Video.mp4"));
        assert!(!p.matches("video.mp4"));
    }

    // ── Retry config tests ───────────────────────────────────────────────────

    #[test]
    fn test_retry_config_defaults() {
        let cfg = RetryConfig::default();
        assert_eq!(cfg.max_retries, 3);
        assert_eq!(cfg.initial_delay_ms, 1000);
        assert!((cfg.backoff_multiplier - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_retry_delay_exponential() {
        let cfg = RetryConfig::new()
            .initial_delay_ms(1000)
            .backoff_multiplier(2.0)
            .max_delay_ms(10_000);

        assert_eq!(cfg.delay_for_attempt(0), Duration::from_secs(1));
        assert_eq!(cfg.delay_for_attempt(1), Duration::from_secs(2));
        assert_eq!(cfg.delay_for_attempt(2), Duration::from_secs(4));
        assert_eq!(cfg.delay_for_attempt(3), Duration::from_secs(8));
        // Clamped to max
        assert_eq!(cfg.delay_for_attempt(4), Duration::from_secs(10));
    }

    #[test]
    fn test_retry_tracker() {
        let cfg = RetryConfig::new().max_retries(3);
        let mut tracker = RetryTracker::new(
            std::env::temp_dir().join("oximedia-transcode-watch-folder-test.mp4"),
        );

        assert!(tracker.can_retry(&cfg));
        assert_eq!(tracker.attempts, 0);

        tracker.record_failure("codec error");
        assert_eq!(tracker.attempts, 1);
        assert_eq!(tracker.last_error.as_deref(), Some("codec error"));
        assert!(tracker.can_retry(&cfg));

        tracker.record_failure("timeout");
        tracker.record_failure("timeout");
        assert!(!tracker.can_retry(&cfg));
    }

    // ── Watch folder statistics tests ────────────────────────────────────────

    #[test]
    fn test_stats_empty() {
        let stats = WatchFolderStats::new();
        assert_eq!(stats.processed_count, 0);
        assert_eq!(stats.error_count, 0);
        assert!(stats.avg_processing_time_ms().is_none());
        assert!((stats.success_rate() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_stats_record_success() {
        let mut stats = WatchFolderStats::new();
        stats.record_success(1000, 10_000_000);
        stats.record_success(2000, 20_000_000);

        assert_eq!(stats.processed_count, 2);
        assert_eq!(stats.total_processing_time_ms, 3000);
        assert_eq!(stats.avg_processing_time_ms(), Some(1500));
        assert_eq!(stats.min_processing_time_ms, Some(1000));
        assert_eq!(stats.max_processing_time_ms, Some(2000));
        assert_eq!(stats.total_bytes_processed, 30_000_000);
    }

    #[test]
    fn test_stats_success_rate() {
        let mut stats = WatchFolderStats::new();
        stats.record_success(100, 1000);
        stats.record_success(100, 1000);
        stats.record_error();

        let rate = stats.success_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_stats_throughput() {
        let mut stats = WatchFolderStats::new();
        stats.record_success(1000, 1_000_000); // 1 MB in 1 second

        let bps = stats.avg_throughput_bps().expect("should have throughput");
        assert!((bps - 1_000_000.0).abs() < 1.0);
    }
}
