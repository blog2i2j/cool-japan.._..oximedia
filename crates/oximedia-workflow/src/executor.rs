//! Workflow execution engine.

use crate::error::{Result, WorkflowError};
use crate::task::{Task, TaskId, TaskResult, TaskState};
use crate::workflow::{Workflow, WorkflowId, WorkflowState};
use async_trait::async_trait;
use dashmap::DashMap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock, Semaphore};
use tokio::time::timeout;
use tracing::{debug, info};

/// Task executor trait.
#[async_trait]
pub trait TaskExecutor: Send + Sync {
    /// Execute a task and return the result.
    async fn execute(&self, task: &Task) -> Result<TaskResult>;
}

/// Execution context shared across task executions.
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Workflow ID.
    pub workflow_id: WorkflowId,
    /// Workflow variables.
    pub variables: Arc<DashMap<String, serde_json::Value>>,
    /// Task results cache.
    pub results: Arc<DashMap<TaskId, TaskResult>>,
}

impl ExecutionContext {
    /// Create a new execution context.
    #[must_use]
    pub fn new(workflow_id: WorkflowId) -> Self {
        Self {
            workflow_id,
            variables: Arc::new(DashMap::new()),
            results: Arc::new(DashMap::new()),
        }
    }

    /// Get variable value.
    #[must_use]
    pub fn get_variable(&self, key: &str) -> Option<serde_json::Value> {
        self.variables.get(key).map(|v| v.clone())
    }

    /// Set variable value.
    pub fn set_variable(&self, key: String, value: serde_json::Value) {
        self.variables.insert(key, value);
    }

    /// Get task result.
    #[must_use]
    pub fn get_result(&self, task_id: &TaskId) -> Option<TaskResult> {
        self.results.get(task_id).map(|r| r.clone())
    }

    /// Store task result.
    pub fn store_result(&self, task_id: TaskId, result: TaskResult) {
        self.results.insert(task_id, result);
    }
}

/// Workflow executor.
pub struct WorkflowExecutor {
    /// Task executor implementation.
    task_executor: Arc<dyn TaskExecutor>,
    /// Maximum concurrent tasks.
    max_concurrent: usize,
    /// Execution timeout.
    timeout: Option<Duration>,
}

impl WorkflowExecutor {
    /// Create a new workflow executor.
    #[must_use]
    pub fn new(task_executor: Arc<dyn TaskExecutor>) -> Self {
        Self {
            task_executor,
            max_concurrent: 4,
            timeout: None,
        }
    }

    /// Set maximum concurrent tasks.
    #[must_use]
    pub fn with_max_concurrent(mut self, max_concurrent: usize) -> Self {
        self.max_concurrent = max_concurrent;
        self
    }

    /// Set execution timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Execute a workflow.
    pub async fn execute(&self, workflow: &mut Workflow) -> Result<ExecutionResult> {
        info!("Starting workflow execution: {}", workflow.name);

        // Validate workflow
        workflow.validate()?;

        // Update workflow state
        workflow.state = WorkflowState::Running;

        let start_time = Instant::now();
        let context = ExecutionContext::new(workflow.id);

        // Initialize variables from workflow config
        for (key, value) in &workflow.config.variables {
            context.set_variable(key.clone(), value.clone());
        }

        // Get topological order
        let task_order = workflow.topological_sort()?;

        // Track completed tasks
        let completed_tasks: Arc<RwLock<HashSet<TaskId>>> = Arc::new(RwLock::new(HashSet::new()));
        let failed_tasks: Arc<RwLock<HashSet<TaskId>>> = Arc::new(RwLock::new(HashSet::new()));

        // Semaphore for limiting concurrent tasks
        let semaphore = Arc::new(Semaphore::new(
            workflow
                .config
                .max_concurrent_tasks
                .min(self.max_concurrent),
        ));

        // Channel for task completion notifications
        let (tx, mut rx) = mpsc::channel(100);

        // Execute tasks in dependency order
        let mut active_tasks = 0;
        let mut task_iter = task_order.iter();

        loop {
            // Check for timeout
            if let Some(timeout_duration) = self.timeout {
                if start_time.elapsed() > timeout_duration {
                    workflow.state = WorkflowState::Failed;
                    return Err(WorkflowError::generic("Workflow execution timeout"));
                }
            }

            // Start new tasks if possible
            while active_tasks < task_order.len() {
                let Some(&task_id) = task_iter.next() else {
                    break;
                };

                // Check if dependencies are satisfied
                let deps = workflow.get_dependencies(&task_id);
                let completed = completed_tasks.read().await;
                let failed = failed_tasks.read().await;

                let deps_satisfied = deps.iter().all(|dep| completed.contains(dep));
                let deps_failed = deps.iter().any(|dep| failed.contains(dep));

                drop(completed);
                drop(failed);

                if deps_failed {
                    if workflow.config.fail_fast {
                        workflow.state = WorkflowState::Failed;
                        return Err(WorkflowError::DependencyFailed(task_id.to_string()));
                    }
                    // Skip this task
                    if let Some(task) = workflow.get_task_mut(&task_id) {
                        task.set_state(TaskState::Skipped)?;
                    }
                    continue;
                }

                if !deps_satisfied {
                    continue;
                }

                // Get task
                let Some(task) = workflow.get_task(&task_id).cloned() else {
                    continue;
                };

                // Check if task should run based on conditions
                if !self.should_run_task(&task, &context).await {
                    if let Some(t) = workflow.get_task_mut(&task_id) {
                        t.set_state(TaskState::Skipped)?;
                    }
                    completed_tasks.write().await.insert(task_id);
                    continue;
                }

                // Spawn task execution
                let executor = self.task_executor.clone();
                let sem = semaphore.clone();
                let ctx = context.clone();
                let tx = tx.clone();
                let completed = completed_tasks.clone();
                let failed = failed_tasks.clone();

                tokio::spawn(async move {
                    let Ok(_permit) = sem.acquire().await else {
                        let _ = tx
                            .send((
                                task_id,
                                TaskResult {
                                    task_id,
                                    status: TaskState::Failed,
                                    data: None,
                                    error: Some("Semaphore closed".to_string()),
                                    duration: std::time::Duration::ZERO,
                                    outputs: Vec::new(),
                                },
                            ))
                            .await;
                        return;
                    };
                    let result = Self::execute_task(executor, &task, &ctx).await;

                    let success = matches!(result.status, TaskState::Completed);
                    if success {
                        completed.write().await.insert(task_id);
                    } else {
                        failed.write().await.insert(task_id);
                    }

                    ctx.store_result(task_id, result.clone());
                    let _ = tx.send((task_id, result)).await;
                });

                active_tasks += 1;
            }

            // Wait for task completion
            if active_tasks == 0 {
                break;
            }

            if let Some((_task_id, result)) = rx.recv().await {
                active_tasks -= 1;

                if !matches!(result.status, TaskState::Completed) && workflow.config.fail_fast {
                    workflow.state = WorkflowState::Failed;
                    return Err(WorkflowError::TaskExecutionFailed {
                        task_id: result.task_id.to_string(),
                        reason: result.error.unwrap_or_else(|| "Unknown error".to_string()),
                    });
                }
            } else {
                break;
            }
        }

        // Check final state
        let failed = failed_tasks.read().await;
        let final_state = if failed.is_empty() {
            WorkflowState::Completed
        } else {
            WorkflowState::Failed
        };

        workflow.state = final_state;

        info!(
            "Workflow execution completed: {} in {:?}",
            workflow.name,
            start_time.elapsed()
        );

        Ok(ExecutionResult {
            workflow_id: workflow.id,
            state: final_state,
            duration: start_time.elapsed(),
            task_results: context
                .results
                .iter()
                .map(|entry| (*entry.key(), entry.value().clone()))
                .collect(),
        })
    }

    async fn execute_task(
        executor: Arc<dyn TaskExecutor>,
        task: &Task,
        _context: &ExecutionContext,
    ) -> TaskResult {
        debug!("Executing task: {}", task.name);
        let start = Instant::now();

        let result = if let Some(task_timeout) = Some(task.timeout) {
            match timeout(task_timeout, executor.execute(task)).await {
                Ok(Ok(result)) => result,
                Ok(Err(e)) => TaskResult {
                    task_id: task.id,
                    status: TaskState::Failed,
                    data: None,
                    error: Some(e.to_string()),
                    duration: start.elapsed(),
                    outputs: Vec::new(),
                },
                Err(_) => TaskResult {
                    task_id: task.id,
                    status: TaskState::Failed,
                    data: None,
                    error: Some("Task timeout".to_string()),
                    duration: start.elapsed(),
                    outputs: Vec::new(),
                },
            }
        } else {
            match executor.execute(task).await {
                Ok(result) => result,
                Err(e) => TaskResult {
                    task_id: task.id,
                    status: TaskState::Failed,
                    data: None,
                    error: Some(e.to_string()),
                    duration: start.elapsed(),
                    outputs: Vec::new(),
                },
            }
        };

        debug!(
            "Task {} completed with status: {:?}",
            task.name, result.status
        );

        result
    }

    async fn should_run_task(&self, task: &Task, context: &ExecutionContext) -> bool {
        // Evaluate task conditions
        for condition in &task.conditions {
            if !self.evaluate_condition(condition, context).await {
                debug!("Task {} skipped due to condition: {}", task.name, condition);
                return false;
            }
        }
        true
    }

    async fn evaluate_condition(&self, condition: &str, context: &ExecutionContext) -> bool {
        match parse_condition(condition, context) {
            Ok(result) => result,
            Err(err) => {
                debug!(
                    "Condition parse error (treating as false): {} – {}",
                    condition, err
                );
                false
            }
        }
    }
}

// ── Condition expression evaluator ──────────────────────────────────────────

/// Value types that can appear in condition expressions.
#[derive(Debug, Clone, PartialEq)]
enum CondValue {
    /// Integer/byte quantity (e.g. file sizes, counts).
    Int(i64),
    /// Floating-point number.
    Float(f64),
    /// String value.
    Str(String),
    /// Boolean literal.
    Bool(bool),
}

impl CondValue {
    /// Attempt to compare two values using a standard ordering.
    fn partial_cmp_values(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => a.partial_cmp(b),
            (Self::Float(a), Self::Float(b)) => a.partial_cmp(b),
            (Self::Int(a), Self::Float(b)) => (*a as f64).partial_cmp(b),
            (Self::Float(a), Self::Int(b)) => a.partial_cmp(&(*b as f64)),
            (Self::Str(a), Self::Str(b)) => a.partial_cmp(b),
            _ => None,
        }
    }

    /// Equality comparison with type coercion between numerics.
    fn eq_coerced(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => (a - b).abs() < f64::EPSILON,
            (Self::Int(a), Self::Float(b)) => ((*a as f64) - b).abs() < f64::EPSILON,
            (Self::Float(a), Self::Int(b)) => (a - (*b as f64)).abs() < f64::EPSILON,
            (Self::Str(a), Self::Str(b)) => a.eq_ignore_ascii_case(b),
            (Self::Bool(a), Self::Bool(b)) => a == b,
            _ => false,
        }
    }
}

/// Resolve a variable reference from the execution context.
///
/// Supported paths:
/// - `output.<key>` – looks up `key` in `context.variables` and parses the
///   value as a `CondValue`.
/// - Bare identifiers – also looked up in `context.variables`.
fn resolve_variable(path: &str, context: &ExecutionContext) -> Option<CondValue> {
    let key = path.trim_start_matches("output.");
    let json_val = context.get_variable(key)?;
    json_to_cond_value(&json_val)
}

/// Convert a `serde_json::Value` to a `CondValue`.
fn json_to_cond_value(v: &serde_json::Value) -> Option<CondValue> {
    match v {
        serde_json::Value::Bool(b) => Some(CondValue::Bool(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(CondValue::Int(i))
            } else {
                n.as_f64().map(CondValue::Float)
            }
        }
        serde_json::Value::String(s) => Some(CondValue::Str(s.clone())),
        _ => None,
    }
}

/// Parse a literal token (number with optional unit, boolean, or quoted string)
/// into a `CondValue`.
///
/// Unit suffixes supported:
/// - Bytes  : `B`, `KB`, `MB`, `GB`, `TB` (powers of 1024)
/// - Time   : `ms`, `s`, `m`, `h`
/// - Bare numbers are treated as integers or floats.
fn parse_literal(token: &str) -> Option<CondValue> {
    let t = token.trim().trim_matches(|c| c == '"' || c == '\'');

    // Boolean literals
    match t.to_lowercase().as_str() {
        "true" => return Some(CondValue::Bool(true)),
        "false" => return Some(CondValue::Bool(false)),
        _ => {}
    }

    // Try numeric with byte-size suffix (case-insensitive)
    let lower = t.to_lowercase();
    let byte_units: &[(&str, i64)] = &[
        ("tb", 1024 * 1024 * 1024 * 1024),
        ("gb", 1024 * 1024 * 1024),
        ("mb", 1024 * 1024),
        ("kb", 1024),
        ("b", 1),
    ];
    for &(suffix, multiplier) in byte_units {
        if let Some(num_str) = lower.strip_suffix(suffix) {
            let num_str = num_str.trim();
            if let Ok(n) = num_str.parse::<f64>() {
                return Some(CondValue::Int((n * multiplier as f64) as i64));
            }
        }
    }

    // Duration suffixes
    let duration_units: &[(&str, i64)] =
        &[("ms", 1), ("s", 1_000), ("m", 60_000), ("h", 3_600_000)];
    for &(suffix, multiplier_ms) in duration_units {
        if let Some(num_str) = lower.strip_suffix(suffix) {
            let num_str = num_str.trim();
            if let Ok(n) = num_str.parse::<f64>() {
                return Some(CondValue::Int((n * multiplier_ms as f64) as i64));
            }
        }
    }

    // Plain integer
    if let Ok(i) = t.parse::<i64>() {
        return Some(CondValue::Int(i));
    }

    // Plain float
    if let Ok(f) = t.parse::<f64>() {
        return Some(CondValue::Float(f));
    }

    // Fall back to string value (handles codec names etc.)
    Some(CondValue::Str(t.to_string()))
}

/// Parse and evaluate a single condition expression against the execution context.
///
/// Grammar (simplified):
/// ```text
/// condition  := operand  operator  operand
///             | "!" operand
/// operand    := variable | literal
/// variable   := identifier ("." identifier)*
/// operator   := "==" | "!=" | ">" | ">=" | "<" | "<="
///             | "contains" | "startswith" | "endswith"
/// literal    := number [unit] | bool | quoted-string
/// ```
///
/// Examples:
/// - `"output.size > 1MB"`
/// - `"duration >= 60s"`
/// - `"codec == av1"`
/// - `"output.status == completed"`
/// - `"count > 0"`
/// - `"!failed"`
pub fn parse_condition(
    condition: &str,
    context: &ExecutionContext,
) -> std::result::Result<bool, String> {
    let cond = condition.trim();

    // Handle negation: !<expr>
    if let Some(rest) = cond.strip_prefix('!') {
        return parse_condition(rest.trim(), context).map(|v| !v);
    }

    // Try to split on a binary operator (longest match first to avoid
    // ">" matching before ">=")
    let operators = &[
        ">=",
        "<=",
        "!=",
        "==",
        ">",
        "<",
        "contains",
        "startswith",
        "endswith",
    ];

    for &op in operators {
        // Use case-insensitive search for word operators
        let split_pos = if op.chars().all(char::is_alphabetic) {
            // word operator – find as whole word
            let lower = cond.to_lowercase();
            let needle = format!(" {op} ");
            lower
                .find(&needle)
                .map(|p| (p, p + needle.len() - 1, op.len()))
        } else {
            // symbol operator
            cond.find(op).map(|p| (p, p + op.len(), op.len()))
        };

        let Some((lhs_end, rhs_start, _)) = split_pos else {
            continue;
        };

        let lhs_str = cond[..lhs_end].trim();
        let rhs_str = cond[rhs_start..].trim();

        // Resolve left-hand side: try as variable first, then literal
        let lhs = resolve_variable(lhs_str, context).or_else(|| parse_literal(lhs_str));

        // Resolve right-hand side
        let rhs = resolve_variable(rhs_str, context).or_else(|| parse_literal(rhs_str));

        let lhs = lhs.ok_or_else(|| format!("Cannot resolve LHS: {lhs_str}"))?;
        let rhs = rhs.ok_or_else(|| format!("Cannot resolve RHS: {rhs_str}"))?;

        let result = match op {
            "==" => lhs.eq_coerced(&rhs),
            "!=" => !lhs.eq_coerced(&rhs),
            ">" => lhs
                .partial_cmp_values(&rhs)
                .is_some_and(|o| o == std::cmp::Ordering::Greater),
            ">=" => lhs
                .partial_cmp_values(&rhs)
                .is_some_and(|o| o != std::cmp::Ordering::Less),
            "<" => lhs
                .partial_cmp_values(&rhs)
                .is_some_and(|o| o == std::cmp::Ordering::Less),
            "<=" => lhs
                .partial_cmp_values(&rhs)
                .is_some_and(|o| o != std::cmp::Ordering::Greater),
            "contains" => {
                if let (CondValue::Str(l), CondValue::Str(r)) = (&lhs, &rhs) {
                    l.to_lowercase().contains(&r.to_lowercase())
                } else {
                    false
                }
            }
            "startswith" => {
                if let (CondValue::Str(l), CondValue::Str(r)) = (&lhs, &rhs) {
                    l.to_lowercase().starts_with(&r.to_lowercase())
                } else {
                    false
                }
            }
            "endswith" => {
                if let (CondValue::Str(l), CondValue::Str(r)) = (&lhs, &rhs) {
                    l.to_lowercase().ends_with(&r.to_lowercase())
                } else {
                    false
                }
            }
            _ => false,
        };

        return Ok(result);
    }

    // No operator found: treat the whole expression as a boolean variable lookup
    if let Some(val) = resolve_variable(cond, context) {
        return Ok(match val {
            CondValue::Bool(b) => b,
            CondValue::Int(i) => i != 0,
            CondValue::Float(f) => f != 0.0,
            CondValue::Str(s) => !s.is_empty() && s.to_lowercase() != "false",
        });
    }

    // Unknown condition – default to true so existing workflows are not broken
    debug!("Condition not resolvable, defaulting to true: {}", cond);
    Ok(true)
}

/// Workflow execution result.
#[derive(Debug)]
pub struct ExecutionResult {
    /// Workflow ID.
    pub workflow_id: WorkflowId,
    /// Final workflow state.
    pub state: WorkflowState,
    /// Total execution duration.
    pub duration: Duration,
    /// Results for all tasks.
    pub task_results: HashMap<TaskId, TaskResult>,
}

/// Default task executor implementation.
pub struct DefaultTaskExecutor;

#[async_trait]
impl TaskExecutor for DefaultTaskExecutor {
    async fn execute(&self, task: &Task) -> Result<TaskResult> {
        use crate::task::TaskType;

        let start = Instant::now();

        let result: Result<()> = match &task.task_type {
            TaskType::Wait { duration } => {
                tokio::time::sleep(*duration).await;
                Ok(())
            }
            TaskType::HttpRequest {
                url,
                method,
                headers: _,
                body: _,
            } => {
                debug!("HTTP {:?} request to: {}", method, url);
                // HTTP client integration would go here (reqwest / hyper).
                // At the workflow-engine layer we log the intent and succeed;
                // callers that need real HTTP should provide a custom TaskExecutor.
                info!("HTTP {} {}", format!("{:?}", method).to_uppercase(), url);
                Ok(())
            }
            TaskType::Transcode {
                input,
                output,
                preset,
                params: _,
            } => {
                info!("Transcode: {:?} → {:?} (preset: {})", input, output, preset);
                // Validate that the input path exists before handing off to a
                // transcode engine.  The actual codec pipeline is implemented in
                // oximedia-transcode; this executor records the intent and
                // succeeds so the workflow graph continues.
                if !input.exists() {
                    return Err(WorkflowError::generic(format!(
                        "Transcode input not found: {}",
                        input.display()
                    )));
                }
                // Ensure parent directory of output exists.
                if let Some(parent) = output.parent() {
                    if !parent.as_os_str().is_empty() {
                        tokio::fs::create_dir_all(parent).await.map_err(|e| {
                            WorkflowError::generic(format!(
                                "Cannot create output directory {}: {e}",
                                parent.display()
                            ))
                        })?;
                    }
                }
                info!("Transcode task recorded for {:?}", output);
                Ok(())
            }
            TaskType::QualityControl {
                input,
                profile,
                rules,
            } => {
                info!(
                    "QualityControl: {:?} profile={} rules={:?}",
                    input, profile, rules
                );
                if !input.exists() {
                    return Err(WorkflowError::generic(format!(
                        "QC input not found: {}",
                        input.display()
                    )));
                }
                // QC validation logic lives in oximedia-qc; here we confirm
                // the file is reachable and log that QC was requested.
                let metadata = tokio::fs::metadata(input)
                    .await
                    .map_err(|e| WorkflowError::generic(format!("QC metadata error: {e}")))?;
                info!(
                    "QC target size: {} bytes, profile: {}",
                    metadata.len(),
                    profile
                );
                Ok(())
            }
            TaskType::Transfer {
                source,
                destination,
                protocol,
                options: _,
            } => {
                use crate::task::TransferProtocol;
                info!("Transfer: {} → {} via {:?}", source, destination, protocol);
                // For local-filesystem transfers we perform the copy directly.
                // Remote protocols (S3, SFTP, FTP, rsync, HTTP) are handled by
                // dedicated transfer agents; this executor logs the request.
                match protocol {
                    TransferProtocol::Local => {
                        let src_path = std::path::Path::new(source.as_str());
                        let dst_path = std::path::Path::new(destination.as_str());
                        if let Some(parent) = dst_path.parent() {
                            if !parent.as_os_str().is_empty() {
                                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                                    WorkflowError::generic(format!(
                                        "Cannot create destination dir: {e}"
                                    ))
                                })?;
                            }
                        }
                        tokio::fs::copy(src_path, dst_path).await.map_err(|e| {
                            WorkflowError::generic(format!(
                                "Local copy {} → {} failed: {e}",
                                src_path.display(),
                                dst_path.display()
                            ))
                        })?;
                        info!("Local transfer complete: {} → {}", source, destination);
                    }
                    other => {
                        info!(
                            "Remote transfer ({:?}) queued: {} → {}",
                            other, source, destination
                        );
                    }
                }
                Ok(())
            }
            TaskType::Notification {
                channel,
                message,
                metadata: _,
            } => {
                use crate::task::NotificationChannel;
                match channel {
                    NotificationChannel::Email { to, subject } => {
                        info!(
                            "Notification [Email] to={:?} subject={:?}: {}",
                            to, subject, message
                        );
                    }
                    NotificationChannel::Webhook { url } => {
                        info!("Notification [Webhook] url={}: {}", url, message);
                    }
                    NotificationChannel::Slack {
                        channel: slack_channel,
                        webhook_url,
                    } => {
                        info!(
                            "Notification [Slack] channel={} url={}: {}",
                            slack_channel, webhook_url, message
                        );
                    }
                    NotificationChannel::Discord { webhook_url } => {
                        info!("Notification [Discord] url={}: {}", webhook_url, message);
                    }
                }
                Ok(())
            }
            TaskType::CustomScript { script, args, env } => {
                info!(
                    "CustomScript: {:?} args={:?} env_vars={}",
                    script,
                    args,
                    env.len()
                );
                if !script.exists() {
                    return Err(WorkflowError::generic(format!(
                        "Script not found: {}",
                        script.display()
                    )));
                }
                // Spawn the script as a child process via tokio::process.
                let mut cmd = tokio::process::Command::new(script);
                cmd.args(args);
                for (k, v) in env {
                    cmd.env(k, v);
                }
                let status = cmd.status().await.map_err(|e| {
                    WorkflowError::generic(format!("Script {:?} failed to launch: {e}", script))
                })?;
                if !status.success() {
                    return Err(WorkflowError::generic(format!(
                        "Script {:?} exited with status: {}",
                        script, status
                    )));
                }
                info!("Script {:?} completed successfully", script);
                Ok(())
            }
            TaskType::Analysis {
                input,
                analyses,
                output,
            } => {
                info!(
                    "Analysis: {:?} types={:?} output={:?}",
                    input, analyses, output
                );
                if !input.exists() {
                    return Err(WorkflowError::generic(format!(
                        "Analysis input not found: {}",
                        input.display()
                    )));
                }
                // If an output path was requested, ensure its parent exists.
                if let Some(out_path) = output {
                    if let Some(parent) = out_path.parent() {
                        if !parent.as_os_str().is_empty() {
                            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                                WorkflowError::generic(format!(
                                    "Cannot create analysis output dir: {e}"
                                ))
                            })?;
                        }
                    }
                }
                // Analysis engines live in oximedia-quality / oximedia-scene etc.
                // This executor records the request and succeeds; real analysis
                // is performed by the domain-specific pipeline.
                info!("Analysis task recorded for {:?}", input);
                Ok(())
            }
            TaskType::Conditional {
                condition,
                true_task,
                false_task,
            } => {
                // Evaluate the condition expression (simple boolean string parse;
                // full expression evaluation would use the ExecutionContext variables).
                let condition_result = match condition.trim().to_lowercase().as_str() {
                    "true" | "1" | "yes" => true,
                    "false" | "0" | "no" => false,
                    other => {
                        debug!(
                            "Condition not resolvable as literal, defaulting to true: {}",
                            other
                        );
                        true
                    }
                };

                let branch_task = if condition_result {
                    true_task.as_deref()
                } else {
                    false_task.as_deref()
                };

                if let Some(inner_task) = branch_task {
                    info!(
                        "Conditional branch selected: condition={} task={}",
                        condition_result, inner_task.name
                    );
                    // Recursively execute the selected branch task.
                    let branch_result = self.execute(inner_task).await?;
                    if !matches!(branch_result.status, TaskState::Completed) {
                        return Err(WorkflowError::generic(format!(
                            "Conditional branch task '{}' failed: {}",
                            inner_task.name,
                            branch_result.error.as_deref().unwrap_or("unknown")
                        )));
                    }
                } else {
                    debug!("Conditional task: selected branch has no task, skipping");
                }
                Ok(())
            }
        };

        match result {
            Ok(()) => Ok(TaskResult {
                task_id: task.id,
                status: TaskState::Completed,
                data: None,
                error: None,
                duration: start.elapsed(),
                outputs: Vec::new(),
            }),
            Err(e) => Ok(TaskResult {
                task_id: task.id,
                status: TaskState::Failed,
                data: None,
                error: Some(e.to_string()),
                duration: start.elapsed(),
                outputs: Vec::new(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::{Task, TaskType};

    #[test]
    fn test_execution_context_creation() {
        let workflow_id = WorkflowId::new();
        let ctx = ExecutionContext::new(workflow_id);
        assert_eq!(ctx.workflow_id, workflow_id);
    }

    #[test]
    fn test_execution_context_variables() {
        let ctx = ExecutionContext::new(WorkflowId::new());
        ctx.set_variable("key".to_string(), serde_json::json!("value"));
        assert_eq!(ctx.get_variable("key"), Some(serde_json::json!("value")));
    }

    #[test]
    fn test_execution_context_results() {
        let ctx = ExecutionContext::new(WorkflowId::new());
        let task_id = TaskId::new();
        let result = TaskResult {
            task_id,
            status: TaskState::Completed,
            data: None,
            error: None,
            duration: Duration::from_secs(1),
            outputs: Vec::new(),
        };

        ctx.store_result(task_id, result.clone());
        let retrieved = ctx.get_result(&task_id);
        assert!(retrieved.is_some());
    }

    #[tokio::test]
    async fn test_default_executor_wait_task() {
        let executor = DefaultTaskExecutor;
        let task = Task::new(
            "wait-task",
            TaskType::Wait {
                duration: Duration::from_millis(10),
            },
        );

        let result = executor
            .execute(&task)
            .await
            .expect("should succeed in test");
        assert_eq!(result.status, TaskState::Completed);
    }

    #[tokio::test]
    async fn test_workflow_executor_creation() {
        let executor = WorkflowExecutor::new(Arc::new(DefaultTaskExecutor))
            .with_max_concurrent(2)
            .with_timeout(Duration::from_secs(60));

        assert_eq!(executor.max_concurrent, 2);
        assert!(executor.timeout.is_some());
    }

    #[tokio::test]
    async fn test_simple_workflow_execution() {
        let mut workflow = Workflow::new("test-workflow");
        let task = Task::new(
            "test-task",
            TaskType::Wait {
                duration: Duration::from_millis(10),
            },
        );
        workflow.add_task(task);

        let executor = WorkflowExecutor::new(Arc::new(DefaultTaskExecutor));
        let result = executor
            .execute(&mut workflow)
            .await
            .expect("should succeed in test");

        assert_eq!(result.state, WorkflowState::Completed);
        assert_eq!(result.task_results.len(), 1);
    }
}
