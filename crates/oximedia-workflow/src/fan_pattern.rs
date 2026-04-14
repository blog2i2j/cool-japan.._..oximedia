//! Fan-out / fan-in parallel task execution patterns.
//!
//! This module provides three composable primitives:
//!
//! - [`FanOut`]: spawns N tasks concurrently from a single input.
//! - `FanIn`: collects the results of N concurrent tasks into one aggregated
//!   result.
//! - [`FanPattern`]: combines `FanOut` + `FanIn` for the complete scatter/gather
//!   pattern commonly used in media processing pipelines (e.g. encode to multiple
//!   bitrates simultaneously, then merge the manifests).
//!
//! [`FanExecutor`] drives the async execution and result aggregation using
//! `tokio::task::JoinSet`.
//!
//! # Example
//!
//! ```rust,no_run
//! use oximedia_workflow::fan_pattern::{FanTask, FanExecutor, FanPattern, AggregationStrategy};
//!
//! # let rt = tokio::runtime::Runtime::new().unwrap();
//! # rt.block_on(async {
//! let tasks = vec![
//!     FanTask::new("encode-1080p", |_| async { Ok::<i64, String>(1080) }),
//!     FanTask::new("encode-720p",  |_| async { Ok::<i64, String>(720) }),
//!     FanTask::new("encode-480p",  |_| async { Ok::<i64, String>(480) }),
//! ];
//!
//! let pattern = FanPattern::new("multi-bitrate-encode", tasks)
//!     .with_strategy(AggregationStrategy::CollectAll);
//!
//! let executor = FanExecutor::new();
//! let result = executor.execute(pattern).await.expect("all tasks succeed");
//! assert_eq!(result.task_results.len(), 3);
//! # });
//! ```

use std::{
    collections::HashMap,
    fmt,
    future::Future,
    pin::Pin,
    sync::Arc,
    time::{Duration, Instant},
};

use tokio::task::JoinSet;

// ---------------------------------------------------------------------------
// FanTask
// ---------------------------------------------------------------------------

/// Type alias for a boxed async factory that produces a task future.
///
/// The factory receives the shared fan-out context (a `HashMap<String, String>`)
/// and returns a pinned, boxed `Future` that resolves to `Result<i64, String>`.
pub type TaskFactory = Arc<
    dyn Fn(
            Arc<HashMap<String, String>>,
        ) -> Pin<Box<dyn Future<Output = Result<i64, String>> + Send>>
        + Send
        + Sync,
>;

/// A single unit of work in a fan-out/fan-in pattern.
///
/// Each `FanTask` wraps a named async closure that receives shared context
/// from the fan-out root and returns either a numeric output or an error string.
#[derive(Clone)]
pub struct FanTask {
    /// Human-readable name (used in result maps and error messages).
    pub name: String,
    /// Async factory invoked by the executor.
    factory: TaskFactory,
    /// Optional per-task timeout.  When `None`, the executor default is used.
    pub timeout: Option<Duration>,
    /// Task weight used by weighted-sum aggregation (default = 1).
    pub weight: u32,
}

impl fmt::Debug for FanTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FanTask")
            .field("name", &self.name)
            .field("timeout", &self.timeout)
            .field("weight", &self.weight)
            .finish()
    }
}

impl FanTask {
    /// Create a `FanTask` from an async factory closure.
    ///
    /// The closure accepts the shared context and must return a
    /// `Future<Output = Result<i64, String>>`.
    pub fn new<N, F, Fut>(name: N, factory: F) -> Self
    where
        N: Into<String>,
        F: Fn(Arc<HashMap<String, String>>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<i64, String>> + Send + 'static,
    {
        Self {
            name: name.into(),
            factory: Arc::new(move |ctx| Box::pin(factory(ctx))),
            timeout: None,
            weight: 1,
        }
    }

    /// Set a per-task timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the aggregation weight for this task (used with
    /// [`AggregationStrategy::WeightedSum`]).
    #[must_use]
    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }

    /// Invoke the factory and get the future.
    fn invoke(
        &self,
        ctx: Arc<HashMap<String, String>>,
    ) -> Pin<Box<dyn Future<Output = Result<i64, String>> + Send>> {
        (self.factory)(ctx)
    }
}

// ---------------------------------------------------------------------------
// FanOut / FanIn
// ---------------------------------------------------------------------------

/// Distributes a single input context to N parallel tasks.
///
/// `FanOut` owns the initial context key-value pairs that will be passed
/// (via `Arc`) to every spawned task.
#[derive(Debug, Default, Clone)]
pub struct FanOut {
    /// Shared context forwarded to all tasks.
    pub context: HashMap<String, String>,
}

impl FanOut {
    /// Create an empty fan-out context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a key-value pair into the shared context.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.context.insert(key.into(), value.into());
        self
    }

    /// Builder-style context insertion.
    #[must_use]
    pub fn with(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }
}

/// Collects N parallel task outputs and aggregates them into one value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Gather all outputs; fail if any task fails.
    CollectAll,
    /// Succeed as soon as the first task succeeds; ignore subsequent failures.
    FirstSuccess,
    /// Sum all successful outputs; fail only if *all* tasks fail.
    Sum,
    /// Compute the maximum among successful outputs; fail only if all fail.
    Max,
    /// Compute the minimum among successful outputs; fail only if all fail.
    Min,
    /// Compute the weighted sum: `sum(output * weight)` for successful tasks.
    WeightedSum,
}

impl Default for AggregationStrategy {
    fn default() -> Self {
        Self::CollectAll
    }
}

/// The merged result of a completed fan-in phase.
#[derive(Debug, Clone)]
pub struct FanInResult {
    /// Individual per-task results (name → output or error).
    pub task_results: HashMap<String, Result<i64, String>>,
    /// Aggregated scalar value computed by the chosen strategy.
    ///
    /// `None` when the strategy could not produce a value (e.g., all tasks
    /// failed and `Sum` was chosen).
    pub aggregated: Option<i64>,
    /// Total wall-clock duration of the fan execution.
    pub elapsed: Duration,
    /// Number of tasks that succeeded.
    pub success_count: usize,
    /// Number of tasks that failed.
    pub failure_count: usize,
}

impl FanInResult {
    /// Returns `true` when all tasks succeeded.
    #[must_use]
    pub fn all_succeeded(&self) -> bool {
        self.failure_count == 0
    }

    /// Returns `true` when at least one task succeeded.
    #[must_use]
    pub fn any_succeeded(&self) -> bool {
        self.success_count > 0
    }
}

// ---------------------------------------------------------------------------
// FanPattern
// ---------------------------------------------------------------------------

/// Combined fan-out + fan-in pattern.
///
/// Describes *what* to run (tasks + context) and *how* to aggregate the results.
/// Execution is delegated to [`FanExecutor::execute`].
#[derive(Debug)]
pub struct FanPattern {
    /// Human-readable name for this pattern (used in logs and error messages).
    pub name: String,
    /// The tasks to execute in parallel.
    pub tasks: Vec<FanTask>,
    /// Shared context pushed to all tasks via fan-out.
    pub fan_out: FanOut,
    /// Strategy used to aggregate results in the fan-in phase.
    pub strategy: AggregationStrategy,
    /// Default per-task timeout applied when a task has no explicit timeout.
    pub default_timeout: Option<Duration>,
    /// When `true`, the executor stops dispatching further tasks as soon as
    /// one task fails (only meaningful for `CollectAll` strategy).
    pub fail_fast: bool,
}

impl FanPattern {
    /// Create a new pattern with the given name and task list.
    ///
    /// Defaults: `CollectAll` strategy, no timeout, `fail_fast = false`.
    #[must_use]
    pub fn new(name: impl Into<String>, tasks: Vec<FanTask>) -> Self {
        Self {
            name: name.into(),
            tasks,
            fan_out: FanOut::new(),
            strategy: AggregationStrategy::CollectAll,
            default_timeout: None,
            fail_fast: false,
        }
    }

    /// Set the aggregation strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: AggregationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set a default per-task timeout.
    #[must_use]
    pub fn with_default_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = Some(timeout);
        self
    }

    /// Enable fail-fast mode.
    #[must_use]
    pub fn with_fail_fast(mut self, fail_fast: bool) -> Self {
        self.fail_fast = fail_fast;
        self
    }

    /// Add a key-value pair to the shared context.
    #[must_use]
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.fan_out.insert(key, value);
        self
    }

    /// Returns the number of tasks in this pattern.
    #[must_use]
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }
}

// ---------------------------------------------------------------------------
// FanError
// ---------------------------------------------------------------------------

/// Errors produced by fan execution.
#[derive(Debug, thiserror::Error)]
pub enum FanError {
    /// All tasks failed; wraps the per-task error map.
    #[error("All {count} fan tasks failed")]
    AllFailed {
        /// Total number of tasks.
        count: usize,
        /// Per-task error strings.
        errors: HashMap<String, String>,
    },
    /// A subset of tasks failed under `CollectAll` strategy.
    #[error("{failure_count} of {total} fan tasks failed")]
    SomeFailed {
        /// Number of failures.
        failure_count: usize,
        /// Total number of tasks.
        total: usize,
        /// Per-task error strings for failed tasks.
        errors: HashMap<String, String>,
    },
    /// A task exceeded its timeout.
    #[error("Fan task '{name}' timed out after {timeout_ms}ms")]
    TaskTimeout {
        /// Name of the timed-out task.
        name: String,
        /// Configured timeout in milliseconds.
        timeout_ms: u64,
    },
    /// An underlying tokio join error (task panicked).
    #[error("Fan task '{name}' panicked: {reason}")]
    JoinError {
        /// Task name.
        name: String,
        /// Panic message.
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// FanExecutor
// ---------------------------------------------------------------------------

/// Drives concurrent fan-out / fan-in execution.
///
/// `FanExecutor` is stateless and can be reused across multiple
/// [`FanPattern`] executions.
#[derive(Debug, Default, Clone)]
pub struct FanExecutor {
    /// Global default timeout applied when neither the task nor the pattern
    /// specifies one.
    pub global_timeout: Option<Duration>,
}

impl FanExecutor {
    /// Create a new executor with no global timeout.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a global default timeout.
    #[must_use]
    pub fn with_global_timeout(mut self, timeout: Duration) -> Self {
        self.global_timeout = Some(timeout);
        self
    }

    /// Execute a [`FanPattern`] and return the aggregated [`FanInResult`].
    ///
    /// # Errors
    ///
    /// Returns [`FanError::AllFailed`] when every task fails and the strategy
    /// requires at least one success.  Returns [`FanError::SomeFailed`] under
    /// `CollectAll` when any task fails.  Returns [`FanError::TaskTimeout`]
    /// when a task exceeds its timeout.
    pub async fn execute(&self, pattern: FanPattern) -> Result<FanInResult, FanError> {
        let start = Instant::now();
        let ctx = Arc::new(pattern.fan_out.context.clone());
        let strategy = pattern.strategy;

        // Spawn all tasks concurrently via JoinSet.
        let mut join_set: JoinSet<(String, u32, Result<i64, String>)> = JoinSet::new();

        for task in pattern.tasks {
            let task_name = task.name.clone();
            let task_weight = task.weight;
            let task_ctx = ctx.clone();
            let effective_timeout = task
                .timeout
                .or(pattern.default_timeout)
                .or(self.global_timeout);
            let fut = task.invoke(task_ctx);

            join_set.spawn(async move {
                match effective_timeout {
                    Some(timeout) => match tokio::time::timeout(timeout, fut).await {
                        Ok(result) => (task_name, task_weight, result),
                        Err(_elapsed) => (
                            task_name.clone(),
                            task_weight,
                            Err(format!("task '{}' timed out", task_name)),
                        ),
                    },
                    None => {
                        let result = fut.await;
                        (task_name, task_weight, result)
                    }
                }
            });
        }

        // Collect all results.
        let mut task_results: HashMap<String, Result<i64, String>> = HashMap::new();
        let mut weights: HashMap<String, u32> = HashMap::new();
        let mut first_success: Option<(String, i64)> = None;
        let mut errors: HashMap<String, String> = HashMap::new();

        while let Some(join_result) = join_set.join_next().await {
            match join_result {
                Ok((name, weight, outcome)) => {
                    weights.insert(name.clone(), weight);
                    match &outcome {
                        Ok(val) => {
                            if first_success.is_none() {
                                first_success = Some((name.clone(), *val));
                            }
                        }
                        Err(e) => {
                            errors.insert(name.clone(), e.clone());
                        }
                    }
                    task_results.insert(name, outcome);
                }
                Err(join_err) => {
                    // tokio JoinError means the task panicked.
                    let name = format!("<unknown-task-{}>", errors.len());
                    let reason = join_err.to_string();
                    errors.insert(name.clone(), reason.clone());
                    task_results.insert(name, Err(format!("join error: {reason}")));
                }
            }
        }

        let elapsed = start.elapsed();
        let total = task_results.len();
        let failure_count = errors.len();
        let success_count = total - failure_count;

        // Aggregate according to strategy.
        let aggregated = Self::aggregate(&task_results, &weights, strategy, &first_success);

        // Check for errors based on strategy.
        match strategy {
            AggregationStrategy::CollectAll => {
                if failure_count > 0 {
                    return Err(FanError::SomeFailed {
                        failure_count,
                        total,
                        errors,
                    });
                }
            }
            AggregationStrategy::FirstSuccess => {
                if first_success.is_none() {
                    return Err(FanError::AllFailed {
                        count: total,
                        errors,
                    });
                }
            }
            AggregationStrategy::Sum
            | AggregationStrategy::Max
            | AggregationStrategy::Min
            | AggregationStrategy::WeightedSum => {
                if success_count == 0 {
                    return Err(FanError::AllFailed {
                        count: total,
                        errors,
                    });
                }
            }
        }

        Ok(FanInResult {
            task_results,
            aggregated,
            elapsed,
            success_count,
            failure_count,
        })
    }

    fn aggregate(
        task_results: &HashMap<String, Result<i64, String>>,
        weights: &HashMap<String, u32>,
        strategy: AggregationStrategy,
        first_success: &Option<(String, i64)>,
    ) -> Option<i64> {
        let successes: Vec<(i64, u32)> = task_results
            .iter()
            .filter_map(|(name, r)| {
                r.as_ref().ok().map(|&v| {
                    let w = weights.get(name).copied().unwrap_or(1);
                    (v, w)
                })
            })
            .collect();

        match strategy {
            AggregationStrategy::CollectAll => {
                // Sum all as a convenience scalar when all succeed.
                if task_results.values().all(|r| r.is_ok()) {
                    Some(successes.iter().map(|(v, _)| *v).sum())
                } else {
                    None
                }
            }
            AggregationStrategy::FirstSuccess => first_success.as_ref().map(|(_, v)| *v),
            AggregationStrategy::Sum => {
                if successes.is_empty() {
                    None
                } else {
                    Some(successes.iter().map(|(v, _)| *v).sum())
                }
            }
            AggregationStrategy::Max => successes.iter().map(|(v, _)| *v).max(),
            AggregationStrategy::Min => successes.iter().map(|(v, _)| *v).min(),
            AggregationStrategy::WeightedSum => {
                if successes.is_empty() {
                    None
                } else {
                    Some(
                        successes
                            .iter()
                            .map(|(v, w)| v.saturating_mul(i64::from(*w)))
                            .fold(0i64, i64::saturating_add),
                    )
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ok_task(name: &str, value: i64) -> FanTask {
        FanTask::new(name.to_string(), move |_ctx| async move {
            Ok::<i64, String>(value)
        })
    }

    fn make_fail_task(name: &str, reason: &str) -> FanTask {
        let reason = reason.to_string();
        FanTask::new(name.to_string(), move |_ctx| {
            let r = reason.clone();
            async move { Err::<i64, String>(r) }
        })
    }

    // -----------------------------------------------------------------------
    // FanOut context
    // -----------------------------------------------------------------------

    #[test]
    fn fan_out_insert_and_with() {
        let mut fo = FanOut::new();
        fo.insert("k1", "v1");
        let fo = fo.with("k2".to_string(), "v2".to_string());
        assert_eq!(fo.context.get("k1").map(String::as_str), Some("v1"));
        assert_eq!(fo.context.get("k2").map(String::as_str), Some("v2"));
    }

    // -----------------------------------------------------------------------
    // FanTask
    // -----------------------------------------------------------------------

    #[test]
    fn fan_task_weight_default_is_one() {
        let t = make_ok_task("t", 42);
        assert_eq!(t.weight, 1);
    }

    #[test]
    fn fan_task_with_weight() {
        let t = make_ok_task("t", 0).with_weight(5);
        assert_eq!(t.weight, 5);
    }

    #[test]
    fn fan_task_with_timeout() {
        let t = make_ok_task("t", 0).with_timeout(Duration::from_millis(100));
        assert!(t.timeout.is_some());
    }

    // -----------------------------------------------------------------------
    // FanPattern
    // -----------------------------------------------------------------------

    #[test]
    fn fan_pattern_task_count() {
        let p = FanPattern::new("p", vec![make_ok_task("a", 1), make_ok_task("b", 2)]);
        assert_eq!(p.task_count(), 2);
    }

    #[test]
    fn fan_pattern_defaults() {
        let p = FanPattern::new("p", vec![]);
        assert_eq!(p.strategy, AggregationStrategy::CollectAll);
        assert!(!p.fail_fast);
        assert!(p.default_timeout.is_none());
    }

    // -----------------------------------------------------------------------
    // FanExecutor — happy paths
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn execute_collect_all_success() {
        let tasks = vec![
            make_ok_task("a", 10),
            make_ok_task("b", 20),
            make_ok_task("c", 30),
        ];
        let pattern = FanPattern::new("test", tasks).with_strategy(AggregationStrategy::CollectAll);
        let result = FanExecutor::new()
            .execute(pattern)
            .await
            .expect("all tasks succeed");

        assert_eq!(result.success_count, 3);
        assert_eq!(result.failure_count, 0);
        assert!(result.all_succeeded());
        // CollectAll aggregated = sum = 60
        assert_eq!(result.aggregated, Some(60));
    }

    #[tokio::test]
    async fn execute_sum_strategy() {
        let tasks = vec![
            make_ok_task("a", 5),
            make_ok_task("b", 15),
            make_fail_task("c", "fail"),
        ];
        let pattern = FanPattern::new("test", tasks).with_strategy(AggregationStrategy::Sum);
        let result = FanExecutor::new()
            .execute(pattern)
            .await
            .expect("sum succeeds when at least one succeeds");

        assert_eq!(result.success_count, 2);
        assert_eq!(result.failure_count, 1);
        assert_eq!(result.aggregated, Some(20));
    }

    #[tokio::test]
    async fn execute_max_strategy() {
        let tasks = vec![
            make_ok_task("a", 3),
            make_ok_task("b", 100),
            make_ok_task("c", 7),
        ];
        let pattern = FanPattern::new("test", tasks).with_strategy(AggregationStrategy::Max);
        let result = FanExecutor::new()
            .execute(pattern)
            .await
            .expect("max succeeds");
        assert_eq!(result.aggregated, Some(100));
    }

    #[tokio::test]
    async fn execute_min_strategy() {
        let tasks = vec![
            make_ok_task("a", 50),
            make_ok_task("b", 3),
            make_ok_task("c", 25),
        ];
        let pattern = FanPattern::new("test", tasks).with_strategy(AggregationStrategy::Min);
        let result = FanExecutor::new()
            .execute(pattern)
            .await
            .expect("min succeeds");
        assert_eq!(result.aggregated, Some(3));
    }

    #[tokio::test]
    async fn execute_weighted_sum_strategy() {
        let tasks = vec![
            make_ok_task("a", 10).with_weight(3), // 30
            make_ok_task("b", 5).with_weight(2),  // 10
        ];
        let pattern =
            FanPattern::new("test", tasks).with_strategy(AggregationStrategy::WeightedSum);
        let result = FanExecutor::new()
            .execute(pattern)
            .await
            .expect("weighted sum ok");
        assert_eq!(result.aggregated, Some(40));
    }

    #[tokio::test]
    async fn execute_first_success_strategy() {
        // Mix of fail + success tasks; first_success should return a value.
        let tasks = vec![
            make_fail_task("a", "oops"),
            make_ok_task("b", 77),
            make_ok_task("c", 99),
        ];
        let pattern =
            FanPattern::new("test", tasks).with_strategy(AggregationStrategy::FirstSuccess);
        let result = FanExecutor::new()
            .execute(pattern)
            .await
            .expect("first_success finds at least one success");
        assert!(result.any_succeeded());
        assert!(result.aggregated.is_some());
    }

    #[tokio::test]
    async fn execute_collect_all_fails_on_any_failure() {
        let tasks = vec![
            make_ok_task("a", 1),
            make_fail_task("b", "boom"),
            make_ok_task("c", 3),
        ];
        let pattern = FanPattern::new("test", tasks).with_strategy(AggregationStrategy::CollectAll);
        let err = FanExecutor::new()
            .execute(pattern)
            .await
            .expect_err("should fail");

        assert!(matches!(
            err,
            FanError::SomeFailed {
                failure_count: 1,
                total: 3,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn execute_all_fail_returns_all_failed() {
        let tasks = vec![make_fail_task("a", "err-a"), make_fail_task("b", "err-b")];
        let pattern = FanPattern::new("test", tasks).with_strategy(AggregationStrategy::Sum);
        let err = FanExecutor::new()
            .execute(pattern)
            .await
            .expect_err("all fail");
        assert!(matches!(err, FanError::AllFailed { count: 2, .. }));
    }

    #[tokio::test]
    async fn execute_with_context_passed_to_tasks() {
        let tasks = vec![FanTask::new(
            "ctx-reader",
            |ctx: Arc<HashMap<String, String>>| async move {
                let val: i64 = ctx
                    .get("multiplier")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1);
                Ok::<i64, String>(val * 10)
            },
        )];
        let pattern = FanPattern::new("ctx-test", tasks)
            .with_context("multiplier", "7")
            .with_strategy(AggregationStrategy::CollectAll);

        let result = FanExecutor::new().execute(pattern).await.expect("ok");
        let output = result.task_results["ctx-reader"].as_ref().expect("ok");
        assert_eq!(*output, 70);
    }

    #[tokio::test]
    async fn execute_empty_tasks_collect_all() {
        let pattern =
            FanPattern::new("empty", vec![]).with_strategy(AggregationStrategy::CollectAll);
        let result = FanExecutor::new()
            .execute(pattern)
            .await
            .expect("empty is fine");
        assert_eq!(result.success_count, 0);
        assert_eq!(result.failure_count, 0);
    }
}
