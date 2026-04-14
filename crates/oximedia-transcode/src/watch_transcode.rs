//! Watch folder automation for auto-transcoding new media files.
//!
//! `TranscodeWatcher` monitors a directory for new media files and creates
//! transcode jobs automatically. Key features:
//!
//! - **Debounce logic**: waits until a file has stopped growing before
//!   submitting it (prevents partial-file processing).
//! - **Include/exclude patterns**: configurable glob-like filename patterns.
//! - **Profile-based transcoding**: associates a `TranscodeProfile` with
//!   the watcher for automatic job creation.
//! - **Concurrent job limits**: throttles how many files are processed
//!   simultaneously.

#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::{Result, TranscodeConfig, TranscodeError};

// ─── Transcode profile ───────────────────────────────────────────────────────

/// A named transcode profile for automatic job creation.
///
/// Associates human-readable settings with the underlying `TranscodeConfig`.
#[derive(Debug, Clone)]
pub struct TranscodeProfile {
    /// Profile name (e.g. "broadcast_hd", "web_720p").
    pub name: String,
    /// Description of the profile.
    pub description: String,
    /// Output file extension (without leading dot).
    pub output_extension: String,
    /// Output directory (if `None`, uses sibling with new extension).
    pub output_dir: Option<PathBuf>,
    /// Base transcode configuration applied to each job.
    pub config: TranscodeConfig,
}

impl TranscodeProfile {
    /// Creates a new profile with the given name and config.
    #[must_use]
    pub fn new(name: impl Into<String>, config: TranscodeConfig) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            output_extension: "mkv".into(),
            output_dir: None,
            config,
        }
    }

    /// Sets the description (builder-style).
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Sets the output extension (builder-style).
    #[must_use]
    pub fn with_output_extension(mut self, ext: impl Into<String>) -> Self {
        self.output_extension = ext.into();
        self
    }

    /// Sets the output directory (builder-style).
    #[must_use]
    pub fn with_output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.output_dir = Some(dir.into());
        self
    }

    /// Computes the output path for a given source file.
    #[must_use]
    pub fn output_path_for(&self, source: &Path) -> PathBuf {
        let stem = source
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let filename = format!("{stem}.{}", self.output_extension);

        match &self.output_dir {
            Some(dir) => dir.join(filename),
            None => {
                let parent = source.parent().unwrap_or_else(|| Path::new("."));
                parent.join(filename)
            }
        }
    }

    /// Creates a `TranscodeConfig` for the given source file.
    #[must_use]
    pub fn create_job_config(&self, source: &Path) -> TranscodeConfig {
        let mut config = self.config.clone();
        config.input = source.to_str().map(String::from);
        config.output = self.output_path_for(source).to_str().map(String::from);
        config
    }
}

// ─── Glob pattern matcher ────────────────────────────────────────────────────

/// A simple glob pattern for include/exclude matching.
///
/// Supports `*` (match any characters) and `?` (match single character).
/// Matching is case-insensitive by default.
#[derive(Debug, Clone)]
pub struct GlobPattern {
    pattern: String,
    case_insensitive: bool,
}

impl GlobPattern {
    /// Creates a new glob pattern.
    #[must_use]
    pub fn new(pattern: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            case_insensitive: true,
        }
    }

    /// Sets case sensitivity.
    #[must_use]
    pub fn case_sensitive(mut self) -> Self {
        self.case_insensitive = false;
        self
    }

    /// Tests whether the given string matches this pattern.
    #[must_use]
    pub fn matches(&self, text: &str) -> bool {
        let (pat, txt) = if self.case_insensitive {
            (self.pattern.to_lowercase(), text.to_lowercase())
        } else {
            (self.pattern.clone(), text.to_string())
        };
        Self::glob_match(&pat, &txt)
    }

    /// Returns the raw pattern string.
    #[must_use]
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    fn glob_match(pattern: &str, text: &str) -> bool {
        let pat: Vec<char> = pattern.chars().collect();
        let txt: Vec<char> = text.chars().collect();
        let (plen, tlen) = (pat.len(), txt.len());

        let mut dp = vec![vec![false; tlen + 1]; plen + 1];
        dp[0][0] = true;

        for (i, &pc) in pat.iter().enumerate() {
            if pc == '*' {
                dp[i + 1][0] = dp[i][0];
            } else {
                break;
            }
        }

        for i in 1..=plen {
            for j in 1..=tlen {
                if pat[i - 1] == '*' {
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                } else if pat[i - 1] == '?' || pat[i - 1] == txt[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }

        dp[plen][tlen]
    }
}

// ─── Debounce tracker ────────────────────────────────────────────────────────

/// Tracks file stability (debounce) to detect when a file has stopped growing.
#[derive(Debug, Clone)]
struct DebounceEntry {
    /// Last observed file size.
    last_size: u64,
    /// Timestamp of last size change.
    last_change: Instant,
    /// Number of consecutive stable checks.
    stable_count: u32,
}

/// Configuration for debounce logic.
#[derive(Debug, Clone)]
pub struct DebounceConfig {
    /// How long a file must remain at the same size to be considered stable.
    pub stable_duration: Duration,
    /// Number of consecutive stable checks required.
    pub required_stable_checks: u32,
    /// Minimum file size before debounce checks begin.
    pub min_file_size: u64,
}

impl Default for DebounceConfig {
    fn default() -> Self {
        Self {
            stable_duration: Duration::from_secs(3),
            required_stable_checks: 2,
            min_file_size: 1024,
        }
    }
}

impl DebounceConfig {
    /// Creates a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the stable duration.
    #[must_use]
    pub fn with_stable_duration(mut self, dur: Duration) -> Self {
        self.stable_duration = dur;
        self
    }

    /// Sets the required stable checks.
    #[must_use]
    pub fn with_required_checks(mut self, n: u32) -> Self {
        self.required_stable_checks = n;
        self
    }

    /// Sets the minimum file size.
    #[must_use]
    pub fn with_min_size(mut self, size: u64) -> Self {
        self.min_file_size = size;
        self
    }
}

// ─── Watcher configuration ──────────────────────────────────────────────────

/// Configuration for `WatchTranscoder`.
#[derive(Debug, Clone)]
pub struct WatchTranscoderConfig {
    /// Directory to monitor.
    pub watch_dir: PathBuf,
    /// Include patterns (if non-empty, only matching files are processed).
    pub include_patterns: Vec<GlobPattern>,
    /// Exclude patterns (matching files are skipped even if they match include).
    pub exclude_patterns: Vec<GlobPattern>,
    /// Debounce configuration.
    pub debounce: DebounceConfig,
    /// Transcode profile to apply.
    pub profile: TranscodeProfile,
    /// Poll interval for directory scanning.
    pub poll_interval: Duration,
    /// Maximum concurrent transcode jobs.
    pub max_concurrent: usize,
    /// Whether to process existing files on startup.
    pub process_existing: bool,
}

impl WatchTranscoderConfig {
    /// Creates a config with sensible defaults.
    #[must_use]
    pub fn new(watch_dir: impl Into<PathBuf>, profile: TranscodeProfile) -> Self {
        Self {
            watch_dir: watch_dir.into(),
            include_patterns: vec![
                GlobPattern::new("*.mp4"),
                GlobPattern::new("*.mkv"),
                GlobPattern::new("*.mov"),
                GlobPattern::new("*.avi"),
                GlobPattern::new("*.webm"),
                GlobPattern::new("*.mxf"),
                GlobPattern::new("*.ts"),
            ],
            exclude_patterns: vec![
                GlobPattern::new(".*"),       // hidden files
                GlobPattern::new("*~"),       // temp/backup files
                GlobPattern::new("*.part"),   // partial downloads
                GlobPattern::new("*.tmp"),    // temp files
            ],
            debounce: DebounceConfig::default(),
            profile,
            poll_interval: Duration::from_secs(5),
            max_concurrent: 2,
            process_existing: false,
        }
    }

    /// Adds an include pattern.
    #[must_use]
    pub fn include(mut self, pattern: impl Into<String>) -> Self {
        self.include_patterns.push(GlobPattern::new(pattern));
        self
    }

    /// Adds an exclude pattern.
    #[must_use]
    pub fn exclude(mut self, pattern: impl Into<String>) -> Self {
        self.exclude_patterns.push(GlobPattern::new(pattern));
        self
    }

    /// Sets the debounce configuration.
    #[must_use]
    pub fn with_debounce(mut self, debounce: DebounceConfig) -> Self {
        self.debounce = debounce;
        self
    }

    /// Sets the poll interval.
    #[must_use]
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Sets the maximum number of concurrent jobs.
    #[must_use]
    pub fn with_max_concurrent(mut self, n: usize) -> Self {
        self.max_concurrent = n;
        self
    }

    /// Enables or disables processing of existing files on startup.
    #[must_use]
    pub fn with_process_existing(mut self, enable: bool) -> Self {
        self.process_existing = enable;
        self
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the watch directory doesn't exist or config is invalid.
    pub fn validate(&self) -> Result<()> {
        if !self.watch_dir.exists() {
            return Err(TranscodeError::InvalidInput(format!(
                "Watch directory does not exist: {}",
                self.watch_dir.display()
            )));
        }
        if self.max_concurrent == 0 {
            return Err(TranscodeError::InvalidInput(
                "max_concurrent must be >= 1".into(),
            ));
        }
        Ok(())
    }
}

// ─── Watch job ───────────────────────────────────────────────────────────────

/// Status of a watched file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatchJobStatus {
    /// File detected, waiting for debounce to complete.
    Debouncing,
    /// File is stable, ready to be transcoded.
    Ready,
    /// Currently being transcoded.
    Transcoding,
    /// Transcode completed successfully.
    Completed,
    /// Transcode failed.
    Failed(String),
    /// File was excluded by pattern matching.
    Excluded,
}

/// A single watch job entry.
#[derive(Debug, Clone)]
pub struct WatchJob {
    /// Source file path.
    pub source: PathBuf,
    /// Output file path (determined by profile).
    pub output: PathBuf,
    /// Current status.
    pub status: WatchJobStatus,
    /// When the file was first detected.
    pub detected_at: Instant,
    /// Transcode config for this job (populated when Ready).
    pub config: Option<TranscodeConfig>,
}

impl WatchJob {
    /// Creates a new watch job.
    fn new(source: PathBuf, output: PathBuf) -> Self {
        Self {
            source,
            output,
            status: WatchJobStatus::Debouncing,
            detected_at: Instant::now(),
            config: None,
        }
    }
}

// ─── WatchTranscoder ─────────────────────────────────────────────────────────

/// Watch folder transcoder that monitors a directory and auto-creates jobs.
///
/// Call [`WatchTranscoder::scan`] to detect new files and update debounce
/// state, then [`WatchTranscoder::drain_ready`] to collect jobs ready for
/// processing.
pub struct WatchTranscoder {
    config: WatchTranscoderConfig,
    /// All known files and their states.
    jobs: Vec<WatchJob>,
    /// Debounce tracking per file path.
    debounce_state: HashMap<PathBuf, DebounceEntry>,
    /// Number of currently active (transcoding) jobs.
    active_count: usize,
}

impl WatchTranscoder {
    /// Creates a new watcher.
    #[must_use]
    pub fn new(config: WatchTranscoderConfig) -> Self {
        Self {
            config,
            jobs: Vec::new(),
            debounce_state: HashMap::new(),
            active_count: 0,
        }
    }

    /// Returns the watcher configuration.
    #[must_use]
    pub fn config(&self) -> &WatchTranscoderConfig {
        &self.config
    }

    /// Returns all jobs.
    #[must_use]
    pub fn jobs(&self) -> &[WatchJob] {
        &self.jobs
    }

    /// Returns the number of active (transcoding) jobs.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Returns whether a file matches the include/exclude filters.
    #[must_use]
    pub fn should_process(&self, filename: &str) -> bool {
        // Check exclude first.
        for pattern in &self.config.exclude_patterns {
            if pattern.matches(filename) {
                return false;
            }
        }
        // If include patterns are specified, file must match at least one.
        if self.config.include_patterns.is_empty() {
            return true;
        }
        self.config
            .include_patterns
            .iter()
            .any(|p| p.matches(filename))
    }

    /// Scans the watch directory for new files and updates debounce state.
    ///
    /// Returns the number of newly discovered files.
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

            // Already known? Skip — debounce is updated in the loop below.
            if self.jobs.iter().any(|j| j.source == path) {
                continue;
            }

            // Check pattern matching.
            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");

            if !self.should_process(filename) {
                continue;
            }

            let output = self.config.profile.output_path_for(&path);
            let job = WatchJob::new(path.clone(), output);
            self.jobs.push(job);

            // Initialize debounce.
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            self.debounce_state.insert(
                path,
                DebounceEntry {
                    last_size: size,
                    last_change: Instant::now(),
                    stable_count: 0,
                },
            );

            new_count += 1;
        }

        // Update debounce for all debouncing entries.
        let debouncing_paths: Vec<PathBuf> = self
            .jobs
            .iter()
            .filter(|j| j.status == WatchJobStatus::Debouncing)
            .map(|j| j.source.clone())
            .collect();

        for path in debouncing_paths {
            self.update_debounce(&path);
        }

        Ok(new_count)
    }

    /// Updates debounce state for a file, promoting to Ready if stable.
    fn update_debounce(&mut self, path: &Path) {
        let current_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

        if let Some(entry) = self.debounce_state.get_mut(path) {
            if current_size < self.config.debounce.min_file_size {
                entry.stable_count = 0;
                entry.last_size = current_size;
                return;
            }

            if current_size == entry.last_size {
                entry.stable_count += 1;
            } else {
                entry.stable_count = 0;
                entry.last_change = Instant::now();
            }
            entry.last_size = current_size;

            if entry.stable_count >= self.config.debounce.required_stable_checks {
                // Promote to Ready.
                for job in &mut self.jobs {
                    if job.source == path && job.status == WatchJobStatus::Debouncing {
                        job.status = WatchJobStatus::Ready;
                        job.config = Some(self.config.profile.create_job_config(path));
                    }
                }
            }
        }
    }

    /// Drains all jobs that are Ready and below the concurrency limit.
    ///
    /// Returns `(source_path, TranscodeConfig)` pairs.
    pub fn drain_ready(&mut self) -> Vec<(PathBuf, TranscodeConfig)> {
        let mut result = Vec::new();

        for job in &mut self.jobs {
            if self.active_count >= self.config.max_concurrent {
                break;
            }
            if job.status == WatchJobStatus::Ready {
                if let Some(ref config) = job.config {
                    result.push((job.source.clone(), config.clone()));
                    job.status = WatchJobStatus::Transcoding;
                    self.active_count += 1;
                }
            }
        }

        result
    }

    /// Marks a job as completed.
    pub fn mark_completed(&mut self, source: &Path) {
        for job in &mut self.jobs {
            if job.source == source && job.status == WatchJobStatus::Transcoding {
                job.status = WatchJobStatus::Completed;
                self.active_count = self.active_count.saturating_sub(1);
                return;
            }
        }
    }

    /// Marks a job as failed.
    pub fn mark_failed(&mut self, source: &Path, reason: impl Into<String>) {
        for job in &mut self.jobs {
            if job.source == source && job.status == WatchJobStatus::Transcoding {
                job.status = WatchJobStatus::Failed(reason.into());
                self.active_count = self.active_count.saturating_sub(1);
                return;
            }
        }
    }

    /// Returns counts of jobs in each status.
    #[must_use]
    pub fn status_summary(&self) -> WatchStatusSummary {
        let mut summary = WatchStatusSummary::default();
        for job in &self.jobs {
            match &job.status {
                WatchJobStatus::Debouncing => summary.debouncing += 1,
                WatchJobStatus::Ready => summary.ready += 1,
                WatchJobStatus::Transcoding => summary.transcoding += 1,
                WatchJobStatus::Completed => summary.completed += 1,
                WatchJobStatus::Failed(_) => summary.failed += 1,
                WatchJobStatus::Excluded => summary.excluded += 1,
            }
        }
        summary
    }

    /// Returns the total number of tracked jobs.
    #[must_use]
    pub fn total_jobs(&self) -> usize {
        self.jobs.len()
    }
}

/// Summary of job status counts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WatchStatusSummary {
    /// Files waiting for debounce.
    pub debouncing: usize,
    /// Files ready for transcoding.
    pub ready: usize,
    /// Files currently being transcoded.
    pub transcoding: usize,
    /// Successfully completed jobs.
    pub completed: usize,
    /// Failed jobs.
    pub failed: usize,
    /// Excluded by pattern.
    pub excluded: usize,
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_profile() -> TranscodeProfile {
        TranscodeProfile::new("test", TranscodeConfig::default())
            .with_output_extension("webm")
            .with_description("Test profile")
    }

    #[test]
    fn test_glob_pattern_basic() {
        let pat = GlobPattern::new("*.mp4");
        assert!(pat.matches("video.mp4"));
        assert!(pat.matches("Video.MP4")); // case insensitive
        assert!(!pat.matches("video.mkv"));
    }

    #[test]
    fn test_glob_pattern_question_mark() {
        let pat = GlobPattern::new("file?.mp4");
        assert!(pat.matches("file1.mp4"));
        assert!(pat.matches("fileA.mp4"));
        assert!(!pat.matches("file12.mp4"));
    }

    #[test]
    fn test_glob_pattern_case_sensitive() {
        let pat = GlobPattern::new("*.MP4").case_sensitive();
        assert!(pat.matches("video.MP4"));
        assert!(!pat.matches("video.mp4"));
    }

    #[test]
    fn test_glob_pattern_wildcard_middle() {
        let pat = GlobPattern::new("pre*suf.txt");
        assert!(pat.matches("pre_middle_suf.txt"));
        assert!(pat.matches("presuf.txt"));
        assert!(!pat.matches("prefix.txt"));
    }

    #[test]
    fn test_should_process_include_exclude() {
        let profile = make_test_profile();
        let config = WatchTranscoderConfig::new(std::env::temp_dir(), profile);
        let watcher = WatchTranscoder::new(config);

        // Included extensions.
        assert!(watcher.should_process("video.mp4"));
        assert!(watcher.should_process("clip.mkv"));
        assert!(watcher.should_process("clip.mov"));

        // Not included.
        assert!(!watcher.should_process("document.pdf"));
        assert!(!watcher.should_process("image.png"));

        // Excluded patterns.
        assert!(!watcher.should_process(".hidden.mp4"));
        assert!(!watcher.should_process("backup.mp4~"));
        assert!(!watcher.should_process("download.mp4.part"));
        assert!(!watcher.should_process("temp.mp4.tmp"));
    }

    #[test]
    fn test_transcode_profile_output_path() {
        let profile = make_test_profile();
        let source = Path::new("/media/input/clip.mp4");
        let output = profile.output_path_for(source);
        assert_eq!(output, PathBuf::from("/media/input/clip.webm"));
    }

    #[test]
    fn test_transcode_profile_output_dir() {
        let profile = make_test_profile()
            .with_output_dir("/media/output");
        let source = Path::new("/media/input/clip.mp4");
        let output = profile.output_path_for(source);
        assert_eq!(output, PathBuf::from("/media/output/clip.webm"));
    }

    #[test]
    fn test_transcode_profile_create_job() {
        let profile = make_test_profile();
        let source = Path::new("/media/clip.mp4");
        let config = profile.create_job_config(source);

        assert_eq!(config.input, Some("/media/clip.mp4".to_string()));
        assert_eq!(config.output, Some("/media/clip.webm".to_string()));
    }

    #[test]
    fn test_watcher_scan_with_real_dir() {
        let temp_dir = std::env::temp_dir().join("oximedia_watch_test_scan");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");

        // Create test files.
        let mp4_path = temp_dir.join("test.mp4");
        let txt_path = temp_dir.join("readme.txt");
        std::fs::write(&mp4_path, b"fake mp4 content that is long enough to pass min size check 1234567890 1234567890").expect("write mp4");
        std::fs::write(&txt_path, b"not a video").expect("write txt");

        let profile = make_test_profile();
        let config = WatchTranscoderConfig::new(&temp_dir, profile)
            .with_process_existing(true);
        let mut watcher = WatchTranscoder::new(config);

        let new_count = watcher.scan().expect("scan should succeed");
        // Should find the .mp4 but not the .txt.
        assert_eq!(new_count, 1);
        assert_eq!(watcher.total_jobs(), 1);
        assert_eq!(watcher.jobs()[0].status, WatchJobStatus::Debouncing);

        // Cleanup.
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_debounce_promotes_to_ready() {
        let temp_dir = std::env::temp_dir().join("oximedia_watch_test_debounce");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");

        let mp4_path = temp_dir.join("stable.mp4");
        std::fs::write(&mp4_path, vec![0u8; 2048]).expect("write mp4");

        let profile = make_test_profile();
        let debounce = DebounceConfig::new()
            .with_required_checks(2)
            .with_min_size(512);
        let config = WatchTranscoderConfig::new(&temp_dir, profile)
            .with_debounce(debounce);
        let mut watcher = WatchTranscoder::new(config);

        // First scan discovers the file and runs one debounce check (stable_count=1).
        watcher.scan().expect("scan 1");
        assert_eq!(watcher.jobs()[0].status, WatchJobStatus::Debouncing);

        // Second scan: file unchanged -> stable_count = 2 >= required (2) -> Ready.
        watcher.scan().expect("scan 2");
        assert_eq!(watcher.jobs()[0].status, WatchJobStatus::Ready);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_drain_ready_respects_concurrency() {
        let temp_dir = std::env::temp_dir().join("oximedia_watch_test_drain");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");

        // Create several files.
        for i in 0..5 {
            let path = temp_dir.join(format!("file{i}.mp4"));
            std::fs::write(&path, vec![0u8; 2048]).expect("write");
        }

        let profile = make_test_profile();
        let debounce = DebounceConfig::new()
            .with_required_checks(1)
            .with_min_size(512);
        let config = WatchTranscoderConfig::new(&temp_dir, profile)
            .with_debounce(debounce)
            .with_max_concurrent(2);
        let mut watcher = WatchTranscoder::new(config);

        // Scan twice to pass debounce.
        watcher.scan().expect("scan 1");
        watcher.scan().expect("scan 2");

        let ready = watcher.drain_ready();
        // Should only drain max_concurrent = 2.
        assert_eq!(ready.len(), 2);
        assert_eq!(watcher.active_count(), 2);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_mark_completed_and_failed() {
        let profile = make_test_profile();
        let config = WatchTranscoderConfig::new(std::env::temp_dir(), profile);
        let mut watcher = WatchTranscoder::new(config);

        // Manually add jobs for testing.
        let source1 = PathBuf::from("/tmp/a.mp4");
        let source2 = PathBuf::from("/tmp/b.mp4");
        watcher.jobs.push(WatchJob {
            source: source1.clone(),
            output: PathBuf::from("/tmp/a.webm"),
            status: WatchJobStatus::Transcoding,
            detected_at: Instant::now(),
            config: None,
        });
        watcher.jobs.push(WatchJob {
            source: source2.clone(),
            output: PathBuf::from("/tmp/b.webm"),
            status: WatchJobStatus::Transcoding,
            detected_at: Instant::now(),
            config: None,
        });
        watcher.active_count = 2;

        watcher.mark_completed(&source1);
        assert_eq!(watcher.active_count(), 1);
        assert_eq!(watcher.jobs[0].status, WatchJobStatus::Completed);

        watcher.mark_failed(&source2, "codec error");
        assert_eq!(watcher.active_count(), 0);
        assert_eq!(
            watcher.jobs[1].status,
            WatchJobStatus::Failed("codec error".to_string())
        );
    }

    #[test]
    fn test_status_summary() {
        let profile = make_test_profile();
        let config = WatchTranscoderConfig::new(std::env::temp_dir(), profile);
        let mut watcher = WatchTranscoder::new(config);

        watcher.jobs.push(WatchJob {
            source: PathBuf::from("/a.mp4"),
            output: PathBuf::from("/a.webm"),
            status: WatchJobStatus::Debouncing,
            detected_at: Instant::now(),
            config: None,
        });
        watcher.jobs.push(WatchJob {
            source: PathBuf::from("/b.mp4"),
            output: PathBuf::from("/b.webm"),
            status: WatchJobStatus::Ready,
            detected_at: Instant::now(),
            config: None,
        });
        watcher.jobs.push(WatchJob {
            source: PathBuf::from("/c.mp4"),
            output: PathBuf::from("/c.webm"),
            status: WatchJobStatus::Completed,
            detected_at: Instant::now(),
            config: None,
        });

        let summary = watcher.status_summary();
        assert_eq!(summary.debouncing, 1);
        assert_eq!(summary.ready, 1);
        assert_eq!(summary.completed, 1);
        assert_eq!(summary.transcoding, 0);
        assert_eq!(summary.failed, 0);
    }

    #[test]
    fn test_debounce_config_builder() {
        let config = DebounceConfig::new()
            .with_stable_duration(Duration::from_secs(10))
            .with_required_checks(5)
            .with_min_size(4096);

        assert_eq!(config.stable_duration, Duration::from_secs(10));
        assert_eq!(config.required_stable_checks, 5);
        assert_eq!(config.min_file_size, 4096);
    }

    #[test]
    fn test_watch_config_validation() {
        let profile = make_test_profile();
        let config = WatchTranscoderConfig::new("/nonexistent/path/12345", profile.clone());
        assert!(config.validate().is_err());

        let config2 = WatchTranscoderConfig::new(std::env::temp_dir(), profile)
            .with_max_concurrent(0);
        assert!(config2.validate().is_err());
    }
}
