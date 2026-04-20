//! Parallel batch repair of independent media files using rayon.
//!
//! This module provides `ParallelRepairEngine` which wraps the standard
//! `RepairEngine` and processes multiple files concurrently using rayon's
//! thread pool. Each file is repaired independently; failures in one file
//! do not affect others.
//!
//! Concurrency is configurable: by default it uses all available CPU cores,
//! but can be limited to a specific number of threads.

#![allow(dead_code)]

use crate::{RepairEngine, RepairError, RepairOptions, RepairResult};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Outcome of a single file in a parallel batch.
#[derive(Debug)]
pub enum BatchItemOutcome {
    /// Repair succeeded.
    Success(RepairResult),
    /// Repair failed with an error.
    Failed {
        /// Path to the file that failed.
        path: PathBuf,
        /// Error message.
        error: String,
    },
}

/// Aggregate statistics for a parallel batch repair.
#[derive(Debug, Clone)]
pub struct BatchStats {
    /// Total number of files processed.
    pub total: usize,
    /// Number of files successfully repaired.
    pub succeeded: usize,
    /// Number of files that failed repair.
    pub failed: usize,
    /// Total issues detected across all files.
    pub total_issues_detected: usize,
    /// Total issues fixed across all files.
    pub total_issues_fixed: usize,
    /// Wall-clock duration for the entire batch.
    pub duration: std::time::Duration,
}

/// Progress callback signature: (completed_count, total_count, current_path).
pub type ProgressCallback = Box<dyn Fn(usize, usize, &std::path::Path) + Send + Sync>;

/// Configuration for the parallel repair engine.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Maximum number of threads to use (0 = use all cores).
    pub max_threads: usize,
    /// Whether to stop processing on the first failure.
    pub fail_fast: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_threads: 0,
            fail_fast: false,
        }
    }
}

/// Parallel repair engine for batch processing of media files.
pub struct ParallelRepairEngine {
    engine: Arc<RepairEngine>,
    config: ParallelConfig,
    progress: Option<Arc<ProgressCallback>>,
}

impl std::fmt::Debug for ParallelRepairEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelRepairEngine")
            .field("engine", &self.engine)
            .field("config", &self.config)
            .field("has_progress_callback", &self.progress.is_some())
            .finish()
    }
}

impl ParallelRepairEngine {
    /// Create a new parallel repair engine with default configuration.
    pub fn new() -> Self {
        Self {
            engine: Arc::new(RepairEngine::new()),
            config: ParallelConfig::default(),
            progress: None,
        }
    }

    /// Create a parallel repair engine with custom configuration.
    pub fn with_config(config: ParallelConfig) -> Self {
        Self {
            engine: Arc::new(RepairEngine::new()),
            config,
            progress: None,
        }
    }

    /// Create a parallel repair engine with a custom inner engine.
    pub fn with_engine(engine: RepairEngine, config: ParallelConfig) -> Self {
        Self {
            engine: Arc::new(engine),
            config,
            progress: None,
        }
    }

    /// Set a progress callback that is invoked after each file is processed.
    pub fn set_progress(&mut self, callback: ProgressCallback) {
        self.progress = Some(Arc::new(callback));
    }

    /// Repair multiple files in parallel.
    ///
    /// Returns a vector of outcomes (one per file) and aggregate statistics.
    /// The order of outcomes matches the order of input paths.
    pub fn repair_batch(
        &self,
        paths: &[PathBuf],
        options: &RepairOptions,
    ) -> crate::Result<(Vec<BatchItemOutcome>, BatchStats)> {
        let start = std::time::Instant::now();

        if paths.is_empty() {
            return Ok((
                Vec::new(),
                BatchStats {
                    total: 0,
                    succeeded: 0,
                    failed: 0,
                    total_issues_detected: 0,
                    total_issues_fixed: 0,
                    duration: start.elapsed(),
                },
            ));
        }

        // Build rayon thread pool with configured parallelism
        let num_threads = if self.config.max_threads == 0 {
            rayon::current_num_threads()
        } else {
            self.config.max_threads
        };

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| RepairError::RepairFailed(format!("Failed to create thread pool: {e}")))?;

        let completed = Arc::new(Mutex::new(0usize));
        let fail_flag = Arc::new(Mutex::new(false));

        let outcomes: Vec<BatchItemOutcome> = pool.install(|| {
            use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

            paths
                .par_iter()
                .map(|path| {
                    // Check fail-fast flag
                    if self.config.fail_fast {
                        let failed = fail_flag.lock().unwrap_or_else(|e| e.into_inner());
                        if *failed {
                            return BatchItemOutcome::Failed {
                                path: path.clone(),
                                error: "Skipped due to fail-fast".to_string(),
                            };
                        }
                    }

                    let outcome = match self.engine.repair_file(path, options) {
                        Ok(result) => BatchItemOutcome::Success(result),
                        Err(e) => {
                            if self.config.fail_fast {
                                let mut failed =
                                    fail_flag.lock().unwrap_or_else(|e| e.into_inner());
                                *failed = true;
                            }
                            BatchItemOutcome::Failed {
                                path: path.clone(),
                                error: e.to_string(),
                            }
                        }
                    };

                    // Update progress
                    if let Some(ref callback) = self.progress {
                        let mut count = completed.lock().unwrap_or_else(|e| e.into_inner());
                        *count += 1;
                        callback(*count, paths.len(), path);
                    }

                    outcome
                })
                .collect()
        });

        // Compute aggregate stats
        let mut succeeded = 0usize;
        let mut failed = 0usize;
        let mut total_issues_detected = 0usize;
        let mut total_issues_fixed = 0usize;

        for outcome in &outcomes {
            match outcome {
                BatchItemOutcome::Success(result) => {
                    if result.success {
                        succeeded += 1;
                    } else {
                        failed += 1;
                    }
                    total_issues_detected += result.issues_detected;
                    total_issues_fixed += result.issues_fixed;
                }
                BatchItemOutcome::Failed { .. } => {
                    failed += 1;
                }
            }
        }

        let stats = BatchStats {
            total: paths.len(),
            succeeded,
            failed,
            total_issues_detected,
            total_issues_fixed,
            duration: start.elapsed(),
        };

        Ok((outcomes, stats))
    }

    /// Analyze multiple files in parallel without repairing.
    pub fn analyze_batch(
        &self,
        paths: &[PathBuf],
    ) -> crate::Result<Vec<(PathBuf, crate::Result<Vec<crate::Issue>>)>> {
        if paths.is_empty() {
            return Ok(Vec::new());
        }

        let num_threads = if self.config.max_threads == 0 {
            rayon::current_num_threads()
        } else {
            self.config.max_threads
        };

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| RepairError::RepairFailed(format!("Failed to create thread pool: {e}")))?;

        let results = pool.install(|| {
            use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

            paths
                .par_iter()
                .map(|path| {
                    let analysis = self.engine.analyze(path);
                    (path.clone(), analysis)
                })
                .collect()
        });

        Ok(results)
    }
}

impl Default for ParallelRepairEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_file(name: &str, data: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!("oximedia_parallel_test_{}", name));
        let mut f = std::fs::File::create(&path).expect("create temp file");
        f.write_all(data).expect("write temp file");
        path
    }

    #[test]
    fn test_parallel_engine_creation() {
        let engine = ParallelRepairEngine::new();
        assert!(!engine.config.fail_fast);
        assert_eq!(engine.config.max_threads, 0);
    }

    #[test]
    fn test_parallel_engine_with_config() {
        let config = ParallelConfig {
            max_threads: 2,
            fail_fast: true,
        };
        let engine = ParallelRepairEngine::with_config(config);
        assert!(engine.config.fail_fast);
        assert_eq!(engine.config.max_threads, 2);
    }

    #[test]
    fn test_parallel_engine_debug() {
        let engine = ParallelRepairEngine::new();
        let debug = format!("{:?}", engine);
        assert!(debug.contains("ParallelRepairEngine"));
    }

    #[test]
    fn test_empty_batch() {
        let engine = ParallelRepairEngine::new();
        let options = RepairOptions {
            create_backup: false,
            verify_after_repair: false,
            ..Default::default()
        };
        let (outcomes, stats) = engine
            .repair_batch(&[], &options)
            .expect("empty batch should succeed");
        assert!(outcomes.is_empty());
        assert_eq!(stats.total, 0);
        assert_eq!(stats.succeeded, 0);
        assert_eq!(stats.failed, 0);
    }

    #[test]
    fn test_batch_with_nonexistent_files() {
        let engine = ParallelRepairEngine::with_config(ParallelConfig {
            max_threads: 2,
            fail_fast: false,
        });
        let options = RepairOptions {
            create_backup: false,
            verify_after_repair: false,
            ..Default::default()
        };
        let tmp = std::env::temp_dir();
        let paths = vec![
            tmp.join("oximedia-repair-parallel-nonexistent_repair_test_1.bin"),
            tmp.join("oximedia-repair-parallel-nonexistent_repair_test_2.bin"),
        ];
        let (outcomes, stats) = engine
            .repair_batch(&paths, &options)
            .expect("batch should complete");
        assert_eq!(stats.total, 2);
        assert_eq!(stats.failed, 2);
        for outcome in &outcomes {
            assert!(matches!(outcome, BatchItemOutcome::Failed { .. }));
        }
    }

    #[test]
    fn test_batch_with_valid_files() {
        let paths = vec![
            temp_file("batch_a.bin", &[0u8; 64]),
            temp_file("batch_b.bin", &[0u8; 64]),
        ];
        let engine = ParallelRepairEngine::with_config(ParallelConfig {
            max_threads: 2,
            fail_fast: false,
        });
        let options = RepairOptions {
            create_backup: false,
            verify_after_repair: false,
            ..Default::default()
        };
        let (outcomes, stats) = engine
            .repair_batch(&paths, &options)
            .expect("batch should succeed");
        assert_eq!(stats.total, 2);
        assert_eq!(outcomes.len(), 2);
        for path in &paths {
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn test_batch_progress_callback() {
        let paths = vec![
            temp_file("progress_a.bin", &[0u8; 32]),
            temp_file("progress_b.bin", &[0u8; 32]),
        ];
        let progress_count = Arc::new(Mutex::new(0usize));
        let progress_clone = Arc::clone(&progress_count);

        let mut engine = ParallelRepairEngine::with_config(ParallelConfig {
            max_threads: 1,
            fail_fast: false,
        });
        engine.set_progress(Box::new(move |completed, total, _path| {
            let mut count = progress_clone.lock().unwrap_or_else(|e| e.into_inner());
            *count = completed;
            assert!(completed <= total);
        }));

        let options = RepairOptions {
            create_backup: false,
            verify_after_repair: false,
            ..Default::default()
        };
        let _ = engine.repair_batch(&paths, &options);

        let final_count = progress_count.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(*final_count, 2);

        for path in &paths {
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn test_analyze_batch_empty() {
        let engine = ParallelRepairEngine::new();
        let results = engine.analyze_batch(&[]).expect("should succeed");
        assert!(results.is_empty());
    }

    #[test]
    fn test_analyze_batch_with_files() {
        let paths = vec![
            temp_file("analyze_a.bin", &[0u8; 64]),
            temp_file("analyze_b.bin", &[0u8; 64]),
        ];
        let engine = ParallelRepairEngine::with_config(ParallelConfig {
            max_threads: 2,
            fail_fast: false,
        });
        let results = engine.analyze_batch(&paths).expect("should succeed");
        assert_eq!(results.len(), 2);
        for path in &paths {
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn test_batch_stats_fields() {
        let stats = BatchStats {
            total: 10,
            succeeded: 7,
            failed: 3,
            total_issues_detected: 15,
            total_issues_fixed: 12,
            duration: std::time::Duration::from_secs(5),
        };
        assert_eq!(stats.total, 10);
        assert_eq!(stats.succeeded, 7);
        assert_eq!(stats.failed, 3);
    }

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        assert_eq!(config.max_threads, 0);
        assert!(!config.fail_fast);
    }

    #[test]
    fn test_fail_fast_mode() {
        let engine = ParallelRepairEngine::with_config(ParallelConfig {
            max_threads: 1, // sequential to make fail-fast deterministic
            fail_fast: true,
        });
        let options = RepairOptions {
            create_backup: false,
            verify_after_repair: false,
            ..Default::default()
        };
        // First file doesn't exist, second should be skipped
        let tmp = std::env::temp_dir();
        let paths = vec![
            tmp.join("oximedia-repair-parallel-nonexistent_ff_1.bin"),
            tmp.join("oximedia-repair-parallel-nonexistent_ff_2.bin"),
        ];
        let (outcomes, stats) = engine
            .repair_batch(&paths, &options)
            .expect("batch should complete");
        assert_eq!(stats.total, 2);
        assert_eq!(outcomes.len(), 2);
        // All should have failed
        assert!(outcomes
            .iter()
            .all(|o| matches!(o, BatchItemOutcome::Failed { .. })));
    }

    #[test]
    fn test_with_engine() {
        let inner = RepairEngine::with_temp_dir(std::env::temp_dir());
        let engine = ParallelRepairEngine::with_engine(inner, ParallelConfig::default());
        assert_eq!(engine.config.max_threads, 0);
    }
}
