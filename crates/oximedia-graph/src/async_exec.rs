//! Async graph execution mode using tokio tasks for I/O-bound source/sink nodes.
//!
//! This module provides an async executor that dispatches each graph node as an
//! independent tokio task.  I/O-bound source and sink nodes benefit the most
//! because they can yield the thread while waiting for data, allowing other
//! nodes to make progress concurrently.
//!
//! # Design
//!
//! The [`AsyncExecutor`] maps an [`ExecutionPlan`] onto tokio tasks:
//!
//! - Each **stage** becomes a `tokio::task::JoinSet` that runs all nodes in
//!   the stage concurrently.
//! - Stages are still sequentially ordered, so data-flow dependencies are
//!   respected.
//! - An optional timeout per stage prevents I/O-bound nodes from stalling the
//!   whole graph.

#![allow(dead_code)]

use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinSet;
use tokio::time::timeout;

use crate::scheduler::{ExecutionPlan, ExecutionStage, ParallelNodeResult, ParallelRunStats};

/// Configuration for the async graph executor.
#[derive(Debug, Clone)]
pub struct AsyncExecutorConfig {
    /// Optional per-stage timeout.  A stage that does not complete within this
    /// duration is cancelled and all remaining nodes are marked as failed.
    pub stage_timeout: Option<Duration>,
    /// If `true`, a stage failure (even a partial one) aborts all subsequent
    /// stages.
    pub fail_on_stage_error: bool,
}

impl Default for AsyncExecutorConfig {
    fn default() -> Self {
        Self {
            stage_timeout: Some(Duration::from_secs(30)),
            fail_on_stage_error: false,
        }
    }
}

/// Async graph executor using tokio tasks.
///
/// # Example
///
/// ```rust,no_run
/// use oximedia_graph::async_exec::{AsyncExecutor, AsyncExecutorConfig};
/// use oximedia_graph::scheduler::ExecutionPlan;
///
/// async fn run_graph(plan: ExecutionPlan) {
///     let config = AsyncExecutorConfig::default();
///     let (results, stats) = AsyncExecutor::run(&plan, config, |node_id| async move {
///         // Perform async I/O for this node.
///         Ok::<(), String>(())
///     }).await;
/// }
/// ```
pub struct AsyncExecutor;

impl AsyncExecutor {
    /// Execute the plan asynchronously.
    ///
    /// `executor` is an async factory that receives a node ID (owned `String`)
    /// and returns a future resolving to `Result<(), String>`.
    ///
    /// Returns the collected per-node results and aggregate statistics.
    pub async fn run<F, Fut>(
        plan: &ExecutionPlan,
        config: AsyncExecutorConfig,
        executor: F,
    ) -> (Vec<ParallelNodeResult>, ParallelRunStats)
    where
        F: Fn(String) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<(), String>> + Send + 'static,
    {
        let mut all_results: Vec<ParallelNodeResult> = Vec::new();
        let mut stats = ParallelRunStats::default();
        let executor = Arc::new(executor);

        'stages: for stage in &plan.stages {
            if stage.nodes.is_empty() {
                continue;
            }
            stats.stages_executed += 1;
            stats.max_concurrency = stats.max_concurrency.max(stage.nodes.len());

            let stage_results = Self::run_stage(stage, &config, Arc::clone(&executor)).await;
            let failures = stage_results.iter().filter(|r| !r.success).count();
            stats.nodes_executed += stage_results.len();
            stats.failures += failures;

            let abort = config.fail_on_stage_error && failures > 0;
            all_results.extend(stage_results);

            if abort {
                break 'stages;
            }
        }

        (all_results, stats)
    }

    /// Execute a single stage: spawn all nodes as tokio tasks, with optional timeout.
    async fn run_stage<F, Fut>(
        stage: &ExecutionStage,
        config: &AsyncExecutorConfig,
        executor: Arc<F>,
    ) -> Vec<ParallelNodeResult>
    where
        F: Fn(String) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<(), String>> + Send + 'static,
    {
        let mut set: JoinSet<ParallelNodeResult> = JoinSet::new();

        for node_id in &stage.nodes {
            let exec = Arc::clone(&executor);
            let nid = node_id.clone();
            set.spawn(async move {
                let result = exec(nid.clone()).await;
                match result {
                    Ok(()) => ParallelNodeResult {
                        node_id: nid,
                        success: true,
                        error: None,
                    },
                    Err(e) => ParallelNodeResult {
                        node_id: nid,
                        success: false,
                        error: Some(e),
                    },
                }
            });
        }

        let collect_future = async {
            let mut results = Vec::new();
            while let Some(join_result) = set.join_next().await {
                match join_result {
                    Ok(node_result) => results.push(node_result),
                    Err(e) => results.push(ParallelNodeResult {
                        node_id: "unknown".to_string(),
                        success: false,
                        error: Some(format!("task panic: {e}")),
                    }),
                }
            }
            results
        };

        match config.stage_timeout {
            Some(dur) => timeout(dur, collect_future)
                .await
                .unwrap_or_else(|_| {
                    // Stage timed out: mark remaining nodes as failed.
                    stage
                        .nodes
                        .iter()
                        .map(|id| ParallelNodeResult {
                            node_id: id.clone(),
                            success: false,
                            error: Some("stage timeout".to_string()),
                        })
                        .collect()
                }),
            None => collect_future.await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::ExecutionStage;

    fn make_plan(stages: Vec<Vec<&str>>) -> ExecutionPlan {
        ExecutionPlan {
            stages: stages
                .into_iter()
                .map(|nodes| ExecutionStage {
                    nodes: nodes.iter().map(|s| s.to_string()).collect(),
                    estimated_cpu_threads: nodes.len() as u32,
                    estimated_memory_mb: nodes.len() as u64 * 64,
                })
                .collect(),
        }
    }

    #[tokio::test]
    async fn test_async_all_succeed() {
        let plan = make_plan(vec![vec!["a", "b"], vec!["c"]]);
        let config = AsyncExecutorConfig::default();
        let (results, stats) = AsyncExecutor::run(&plan, config, |_node_id| async {
            Ok::<(), String>(())
        })
        .await;
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.success));
        assert_eq!(stats.nodes_executed, 3);
        assert_eq!(stats.failures, 0);
    }

    #[tokio::test]
    async fn test_async_partial_failure() {
        let plan = make_plan(vec![vec!["ok1", "fail1", "ok2"]]);
        let config = AsyncExecutorConfig::default();
        let (results, stats) = AsyncExecutor::run(&plan, config, |node_id| async move {
            if node_id == "fail1" {
                Err("simulated failure".to_string())
            } else {
                Ok(())
            }
        })
        .await;
        assert_eq!(stats.failures, 1);
        assert_eq!(results.iter().filter(|r| !r.success).count(), 1);
    }

    #[tokio::test]
    async fn test_async_fail_on_stage_error_aborts_remaining() {
        let plan = make_plan(vec![vec!["fail-node"], vec!["should-not-run"]]);
        let config = AsyncExecutorConfig {
            fail_on_stage_error: true,
            stage_timeout: None,
        };
        let (results, stats) = AsyncExecutor::run(&plan, config, |node_id| async move {
            if node_id == "fail-node" {
                Err("stage error".to_string())
            } else {
                Ok(())
            }
        })
        .await;
        // Second stage should be skipped.
        assert!(!results
            .iter()
            .any(|r| r.node_id == "should-not-run"));
        assert_eq!(stats.stages_executed, 1);
    }

    #[tokio::test]
    async fn test_async_empty_plan() {
        let plan = ExecutionPlan { stages: vec![] };
        let config = AsyncExecutorConfig::default();
        let (results, stats) =
            AsyncExecutor::run(&plan, config, |_| async { Ok::<(), String>(()) }).await;
        assert!(results.is_empty());
        assert_eq!(stats.stages_executed, 0);
    }
}
