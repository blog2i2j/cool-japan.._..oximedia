#![allow(dead_code)]
//! Directed acyclic graph (DAG) for expressing GPU/CPU task dependencies.
//!
//! Allows scheduling of compute operations with explicit dependency edges
//! so that tasks can be parallelized where possible and serialized where
//! data dependencies require it.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

/// Unique identifier for a task node in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Task({})", self.0)
    }
}

/// The type of execution backend a task should run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskBackend {
    /// Execute on the GPU via compute shaders.
    Gpu,
    /// Execute on the CPU.
    Cpu,
    /// Let the scheduler decide based on load and capabilities.
    Auto,
}

/// Current execution state of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskState {
    /// Task has not started.
    Pending,
    /// Task dependencies are satisfied and it is ready to run.
    Ready,
    /// Task is currently executing.
    Running,
    /// Task completed successfully.
    Completed,
    /// Task failed.
    Failed,
    /// Task was cancelled.
    Cancelled,
}

impl TaskState {
    /// Returns true if the task is in a terminal state.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    /// Returns true if the task has finished successfully.
    #[must_use]
    pub fn is_success(self) -> bool {
        matches!(self, Self::Completed)
    }
}

/// A single compute task in the dependency graph.
#[derive(Debug, Clone)]
pub struct TaskNode {
    /// Unique identifier.
    pub id: TaskId,
    /// Human-readable label for debugging.
    pub label: String,
    /// Preferred execution backend.
    pub backend: TaskBackend,
    /// Current state.
    pub state: TaskState,
    /// Estimated cost in abstract work units (higher = more expensive).
    pub estimated_cost: u64,
    /// IDs of tasks that this task depends on (must complete before this starts).
    pub dependencies: HashSet<TaskId>,
    /// IDs of tasks that depend on this task.
    pub dependents: HashSet<TaskId>,
}

impl TaskNode {
    /// Creates a new pending task with no dependencies.
    #[must_use]
    pub fn new(id: TaskId, label: String, backend: TaskBackend) -> Self {
        Self {
            id,
            label,
            backend,
            state: TaskState::Pending,
            estimated_cost: 1,
            dependencies: HashSet::new(),
            dependents: HashSet::new(),
        }
    }

    /// Sets the estimated cost.
    #[must_use]
    pub fn with_cost(mut self, cost: u64) -> Self {
        self.estimated_cost = cost;
        self
    }

    /// Returns true if all dependencies are satisfied (completed).
    #[must_use]
    pub fn dependencies_satisfied(&self, completed: &HashSet<TaskId>) -> bool {
        self.dependencies.iter().all(|dep| completed.contains(dep))
    }

    /// Returns the number of unresolved dependencies.
    #[must_use]
    pub fn pending_dependency_count(&self, completed: &HashSet<TaskId>) -> usize {
        self.dependencies
            .iter()
            .filter(|dep| !completed.contains(dep))
            .count()
    }
}

/// Error type for task graph operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskGraphError {
    /// A task with the given ID already exists.
    DuplicateTask(TaskId),
    /// Referenced task was not found.
    TaskNotFound(TaskId),
    /// Adding the dependency would create a cycle.
    CycleDetected,
    /// The graph is in an invalid state for the operation.
    InvalidState(String),
}

impl fmt::Display for TaskGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateTask(id) => write!(f, "duplicate task: {id}"),
            Self::TaskNotFound(id) => write!(f, "task not found: {id}"),
            Self::CycleDetected => write!(f, "cycle detected in task graph"),
            Self::InvalidState(msg) => write!(f, "invalid state: {msg}"),
        }
    }
}

/// A directed acyclic graph (DAG) of compute tasks with dependency tracking.
#[derive(Debug)]
pub struct TaskGraph {
    /// All tasks keyed by ID.
    tasks: HashMap<TaskId, TaskNode>,
    /// Auto-incrementing ID counter.
    next_id: u64,
}

impl TaskGraph {
    /// Creates a new empty task graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
            next_id: 0,
        }
    }

    /// Adds a new task to the graph and returns its assigned ID.
    pub fn add_task(&mut self, label: &str, backend: TaskBackend) -> TaskId {
        let id = TaskId(self.next_id);
        self.next_id += 1;
        let node = TaskNode::new(id, label.to_string(), backend);
        self.tasks.insert(id, node);
        id
    }

    /// Adds a new task with an estimated cost.
    pub fn add_task_with_cost(&mut self, label: &str, backend: TaskBackend, cost: u64) -> TaskId {
        let id = TaskId(self.next_id);
        self.next_id += 1;
        let node = TaskNode::new(id, label.to_string(), backend).with_cost(cost);
        self.tasks.insert(id, node);
        id
    }

    /// Adds a dependency edge: `task` depends on `dependency`.
    ///
    /// # Errors
    ///
    /// Returns an error if either task is not found or if the dependency
    /// would create a cycle.
    pub fn add_dependency(
        &mut self,
        task: TaskId,
        dependency: TaskId,
    ) -> Result<(), TaskGraphError> {
        if !self.tasks.contains_key(&task) {
            return Err(TaskGraphError::TaskNotFound(task));
        }
        if !self.tasks.contains_key(&dependency) {
            return Err(TaskGraphError::TaskNotFound(dependency));
        }

        // Check for cycle: would `dependency` be reachable from `task`'s dependents?
        if self.would_create_cycle(task, dependency) {
            return Err(TaskGraphError::CycleDetected);
        }

        self.tasks
            .get_mut(&task)
            .unwrap_or_else(|| unreachable!("task guaranteed to exist after contains_key check"))
            .dependencies
            .insert(dependency);
        self.tasks
            .get_mut(&dependency)
            .unwrap_or_else(|| {
                unreachable!("dependency guaranteed to exist after contains_key check")
            })
            .dependents
            .insert(task);
        Ok(())
    }

    /// Checks if adding an edge from `task` depending on `dep` would create a cycle.
    fn would_create_cycle(&self, task: TaskId, dep: TaskId) -> bool {
        if task == dep {
            return true;
        }
        // BFS from `task` through dependents to see if `dep` is reachable
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(task);
        while let Some(current) = queue.pop_front() {
            if current == dep {
                return true;
            }
            if visited.insert(current) {
                if let Some(node) = self.tasks.get(&current) {
                    for &dependent in &node.dependents {
                        queue.push_back(dependent);
                    }
                }
            }
        }
        false
    }

    /// Returns the total number of tasks.
    #[must_use]
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Returns a reference to a task by ID.
    #[must_use]
    pub fn get_task(&self, id: TaskId) -> Option<&TaskNode> {
        self.tasks.get(&id)
    }

    /// Returns a mutable reference to a task by ID.
    pub fn get_task_mut(&mut self, id: TaskId) -> Option<&mut TaskNode> {
        self.tasks.get_mut(&id)
    }

    /// Returns all tasks that are ready to execute (dependencies satisfied, state is Pending).
    #[must_use]
    pub fn ready_tasks(&self) -> Vec<TaskId> {
        let completed: HashSet<TaskId> = self
            .tasks
            .values()
            .filter(|t| t.state == TaskState::Completed)
            .map(|t| t.id)
            .collect();

        self.tasks
            .values()
            .filter(|t| t.state == TaskState::Pending && t.dependencies_satisfied(&completed))
            .map(|t| t.id)
            .collect()
    }

    /// Marks a task as running.
    ///
    /// # Errors
    ///
    /// Returns error if the task is not found or not in Pending/Ready state.
    pub fn start_task(&mut self, id: TaskId) -> Result<(), TaskGraphError> {
        let task = self
            .tasks
            .get_mut(&id)
            .ok_or(TaskGraphError::TaskNotFound(id))?;
        if task.state != TaskState::Pending && task.state != TaskState::Ready {
            return Err(TaskGraphError::InvalidState(format!(
                "cannot start task in state {:?}",
                task.state
            )));
        }
        task.state = TaskState::Running;
        Ok(())
    }

    /// Marks a task as completed.
    ///
    /// # Errors
    ///
    /// Returns error if the task is not found or not Running.
    pub fn complete_task(&mut self, id: TaskId) -> Result<(), TaskGraphError> {
        let task = self
            .tasks
            .get_mut(&id)
            .ok_or(TaskGraphError::TaskNotFound(id))?;
        if task.state != TaskState::Running {
            return Err(TaskGraphError::InvalidState(format!(
                "cannot complete task in state {:?}",
                task.state
            )));
        }
        task.state = TaskState::Completed;
        Ok(())
    }

    /// Marks a task as failed.
    ///
    /// # Errors
    ///
    /// Returns error if the task is not found.
    pub fn fail_task(&mut self, id: TaskId) -> Result<(), TaskGraphError> {
        let task = self
            .tasks
            .get_mut(&id)
            .ok_or(TaskGraphError::TaskNotFound(id))?;
        task.state = TaskState::Failed;
        Ok(())
    }

    /// Returns true if all tasks have reached a terminal state.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.tasks.values().all(|t| t.state.is_terminal())
    }

    /// Returns true if any task has failed.
    #[must_use]
    pub fn has_failures(&self) -> bool {
        self.tasks.values().any(|t| t.state == TaskState::Failed)
    }

    /// Returns the total estimated cost of all tasks.
    #[must_use]
    pub fn total_cost(&self) -> u64 {
        self.tasks.values().map(|t| t.estimated_cost).sum()
    }

    /// Returns a topological ordering of the tasks, or an error if the graph has a cycle.
    ///
    /// # Errors
    ///
    /// Returns `CycleDetected` if the graph contains a cycle.
    pub fn topological_sort(&self) -> Result<Vec<TaskId>, TaskGraphError> {
        let mut in_degree: HashMap<TaskId, usize> = HashMap::new();
        for task in self.tasks.values() {
            in_degree.entry(task.id).or_insert(0);
            for &dep in &task.dependents {
                *in_degree.entry(dep).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<TaskId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut result = Vec::new();
        while let Some(id) = queue.pop_front() {
            result.push(id);
            if let Some(node) = self.tasks.get(&id) {
                for &dependent in &node.dependents {
                    if let Some(deg) = in_degree.get_mut(&dependent) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(dependent);
                        }
                    }
                }
            }
        }

        if result.len() != self.tasks.len() {
            return Err(TaskGraphError::CycleDetected);
        }
        Ok(result)
    }

    /// Computes the critical path length (longest path through the graph by cost).
    #[must_use]
    pub fn critical_path_cost(&self) -> u64 {
        if self.tasks.is_empty() {
            return 0;
        }
        let sorted = match self.topological_sort() {
            Ok(s) => s,
            Err(_) => return 0,
        };

        let mut longest: HashMap<TaskId, u64> = HashMap::new();
        for &id in &sorted {
            let task = &self.tasks[&id];
            let max_dep = task
                .dependencies
                .iter()
                .filter_map(|d| longest.get(d))
                .copied()
                .max()
                .unwrap_or(0);
            longest.insert(id, max_dep + task.estimated_cost);
        }

        longest.values().copied().max().unwrap_or(0)
    }
}

impl Default for TaskGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_id_display() {
        let id = TaskId(42);
        assert_eq!(format!("{id}"), "Task(42)");
    }

    #[test]
    fn test_task_state_terminal() {
        assert!(!TaskState::Pending.is_terminal());
        assert!(!TaskState::Ready.is_terminal());
        assert!(!TaskState::Running.is_terminal());
        assert!(TaskState::Completed.is_terminal());
        assert!(TaskState::Failed.is_terminal());
        assert!(TaskState::Cancelled.is_terminal());
    }

    #[test]
    fn test_task_state_success() {
        assert!(TaskState::Completed.is_success());
        assert!(!TaskState::Failed.is_success());
        assert!(!TaskState::Pending.is_success());
    }

    #[test]
    fn test_add_task() {
        let mut g = TaskGraph::new();
        let id = g.add_task("resize", TaskBackend::Gpu);
        assert_eq!(g.task_count(), 1);
        let task = g.get_task(id).expect("task should be valid");
        assert_eq!(task.label, "resize");
        assert_eq!(task.backend, TaskBackend::Gpu);
        assert_eq!(task.state, TaskState::Pending);
    }

    #[test]
    fn test_add_task_with_cost() {
        let mut g = TaskGraph::new();
        let id = g.add_task_with_cost("heavy", TaskBackend::Cpu, 100);
        assert_eq!(
            g.get_task(id)
                .expect("get_task should succeed")
                .estimated_cost,
            100
        );
    }

    #[test]
    fn test_add_dependency() {
        let mut g = TaskGraph::new();
        let a = g.add_task("a", TaskBackend::Auto);
        let b = g.add_task("b", TaskBackend::Auto);
        g.add_dependency(b, a)
            .expect("add_dependency should succeed");

        let node_b = g.get_task(b).expect("node_b should be valid");
        assert!(node_b.dependencies.contains(&a));
        let node_a = g.get_task(a).expect("node_a should be valid");
        assert!(node_a.dependents.contains(&b));
    }

    #[test]
    fn test_cycle_detection_self() {
        let mut g = TaskGraph::new();
        let a = g.add_task("a", TaskBackend::Auto);
        let result = g.add_dependency(a, a);
        assert_eq!(result, Err(TaskGraphError::CycleDetected));
    }

    #[test]
    fn test_cycle_detection_indirect() {
        let mut g = TaskGraph::new();
        let a = g.add_task("a", TaskBackend::Auto);
        let b = g.add_task("b", TaskBackend::Auto);
        let c = g.add_task("c", TaskBackend::Auto);
        g.add_dependency(b, a)
            .expect("add_dependency should succeed");
        g.add_dependency(c, b)
            .expect("add_dependency should succeed");
        let result = g.add_dependency(a, c);
        assert_eq!(result, Err(TaskGraphError::CycleDetected));
    }

    #[test]
    fn test_ready_tasks_no_deps() {
        let mut g = TaskGraph::new();
        let a = g.add_task("a", TaskBackend::Gpu);
        let b = g.add_task("b", TaskBackend::Cpu);
        let ready = g.ready_tasks();
        assert!(ready.contains(&a));
        assert!(ready.contains(&b));
    }

    #[test]
    fn test_ready_tasks_with_deps() {
        let mut g = TaskGraph::new();
        let a = g.add_task("a", TaskBackend::Auto);
        let b = g.add_task("b", TaskBackend::Auto);
        g.add_dependency(b, a)
            .expect("add_dependency should succeed");

        let ready = g.ready_tasks();
        assert!(ready.contains(&a));
        assert!(!ready.contains(&b)); // b depends on a

        g.start_task(a).expect("start_task should succeed");
        g.complete_task(a).expect("complete_task should succeed");

        let ready2 = g.ready_tasks();
        assert!(ready2.contains(&b));
    }

    #[test]
    fn test_start_and_complete_task() {
        let mut g = TaskGraph::new();
        let a = g.add_task("a", TaskBackend::Auto);
        g.start_task(a).expect("start_task should succeed");
        assert_eq!(
            g.get_task(a).expect("get_task should succeed").state,
            TaskState::Running
        );
        g.complete_task(a).expect("complete_task should succeed");
        assert_eq!(
            g.get_task(a).expect("get_task should succeed").state,
            TaskState::Completed
        );
    }

    #[test]
    fn test_fail_task() {
        let mut g = TaskGraph::new();
        let a = g.add_task("a", TaskBackend::Auto);
        g.fail_task(a).expect("fail_task should succeed");
        assert_eq!(
            g.get_task(a).expect("get_task should succeed").state,
            TaskState::Failed
        );
        assert!(g.has_failures());
    }

    #[test]
    fn test_is_complete() {
        let mut g = TaskGraph::new();
        let a = g.add_task("a", TaskBackend::Auto);
        assert!(!g.is_complete());
        g.start_task(a).expect("start_task should succeed");
        g.complete_task(a).expect("complete_task should succeed");
        assert!(g.is_complete());
    }

    #[test]
    fn test_total_cost() {
        let mut g = TaskGraph::new();
        g.add_task_with_cost("a", TaskBackend::Gpu, 10);
        g.add_task_with_cost("b", TaskBackend::Cpu, 20);
        assert_eq!(g.total_cost(), 30);
    }

    #[test]
    fn test_topological_sort() {
        let mut g = TaskGraph::new();
        let a = g.add_task("a", TaskBackend::Auto);
        let b = g.add_task("b", TaskBackend::Auto);
        let c = g.add_task("c", TaskBackend::Auto);
        g.add_dependency(b, a)
            .expect("add_dependency should succeed");
        g.add_dependency(c, b)
            .expect("add_dependency should succeed");

        let sorted = g.topological_sort().expect("sorted should be valid");
        let pos_a = sorted
            .iter()
            .position(|&x| x == a)
            .expect("pos_a should be valid");
        let pos_b = sorted
            .iter()
            .position(|&x| x == b)
            .expect("pos_b should be valid");
        let pos_c = sorted
            .iter()
            .position(|&x| x == c)
            .expect("pos_c should be valid");
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_critical_path_cost() {
        let mut g = TaskGraph::new();
        let a = g.add_task_with_cost("a", TaskBackend::Auto, 5);
        let b = g.add_task_with_cost("b", TaskBackend::Auto, 10);
        let c = g.add_task_with_cost("c", TaskBackend::Auto, 3);
        g.add_dependency(b, a)
            .expect("add_dependency should succeed");
        g.add_dependency(c, a)
            .expect("add_dependency should succeed");
        // Paths: a->b = 15, a->c = 8
        assert_eq!(g.critical_path_cost(), 15);
    }

    #[test]
    fn test_critical_path_empty() {
        let g = TaskGraph::new();
        assert_eq!(g.critical_path_cost(), 0);
    }

    #[test]
    fn test_dependency_not_found() {
        let mut g = TaskGraph::new();
        let a = g.add_task("a", TaskBackend::Auto);
        let bad = TaskId(999);
        assert_eq!(
            g.add_dependency(a, bad),
            Err(TaskGraphError::TaskNotFound(bad))
        );
    }

    #[test]
    fn test_start_task_not_found() {
        let mut g = TaskGraph::new();
        assert!(g.start_task(TaskId(999)).is_err());
    }

    // ── Scheduling overhead and large graph tests ──────────────────────────

    #[test]
    #[ignore] // slow
    fn task_graph_100_node_chain_schedules_quickly() {
        let mut g = TaskGraph::new();
        let ids: Vec<_> = (0..100)
            .map(|i| g.add_task(&format!("task_{i}"), TaskBackend::Auto))
            .collect();
        for i in 0..99 {
            g.add_dependency(ids[i + 1], ids[i])
                .expect("add_dependency should succeed");
        }

        let start = std::time::Instant::now();
        let order = g.topological_sort().expect("sort should succeed");
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_millis() < 100,
            "100-node topological sort took {}ms (must be < 100ms)",
            elapsed.as_millis()
        );
        assert_eq!(order.len(), 100);
    }

    #[test]
    fn task_graph_single_task() {
        let mut g = TaskGraph::new();
        let id = g.add_task("solo", TaskBackend::Auto);
        let order = g.topological_sort().expect("sort should succeed");
        assert_eq!(order.len(), 1);
        assert_eq!(order[0], id);
    }

    #[test]
    fn task_graph_two_task_dependency_order() {
        let mut g = TaskGraph::new();
        let a = g.add_task("a", TaskBackend::Cpu);
        let b = g.add_task("b", TaskBackend::Cpu);
        g.add_dependency(b, a)
            .expect("add_dependency should succeed");

        let order = g.topological_sort().expect("sort should succeed");
        let pos_a = order.iter().position(|&x| x == a).expect("a in order");
        let pos_b = order.iter().position(|&x| x == b).expect("b in order");
        assert!(pos_a < pos_b, "a must come before b");
    }

    #[test]
    fn task_graph_diamond_dag_correct_order() {
        // Diamond: A → (B, C) → D
        let mut g = TaskGraph::new();
        let a = g.add_task("A", TaskBackend::Auto);
        let b = g.add_task("B", TaskBackend::Auto);
        let c = g.add_task("C", TaskBackend::Auto);
        let d = g.add_task("D", TaskBackend::Auto);

        g.add_dependency(b, a).expect("B depends on A");
        g.add_dependency(c, a).expect("C depends on A");
        g.add_dependency(d, b).expect("D depends on B");
        g.add_dependency(d, c).expect("D depends on C");

        let order = g.topological_sort().expect("diamond has no cycle");
        assert_eq!(order.len(), 4);

        let pos = |id: TaskId| order.iter().position(|&x| x == id).expect("id in order");
        assert!(pos(a) < pos(b), "A before B");
        assert!(pos(a) < pos(c), "A before C");
        assert!(pos(b) < pos(d), "B before D");
        assert!(pos(c) < pos(d), "C before D");
    }

    #[test]
    fn task_graph_cycle_detection_returns_error() {
        let mut g = TaskGraph::new();
        let a = g.add_task("a", TaskBackend::Auto);
        let b = g.add_task("b", TaskBackend::Auto);
        let c = g.add_task("c", TaskBackend::Auto);

        g.add_dependency(b, a).expect("b depends on a");
        g.add_dependency(c, b).expect("c depends on b");

        // Adding a → c would create cycle a→b→c→a
        let result = g.add_dependency(a, c);
        assert_eq!(
            result,
            Err(TaskGraphError::CycleDetected),
            "should detect cycle"
        );
    }

    #[test]
    fn task_graph_50_node_fan_out_in() {
        // Fan-out from root to 48 leaves, then fan-in to a sink
        let mut g = TaskGraph::new();
        let root = g.add_task("root", TaskBackend::Gpu);
        let leaves: Vec<_> = (0..48)
            .map(|i| g.add_task(&format!("leaf_{i}"), TaskBackend::Cpu))
            .collect();
        let sink = g.add_task("sink", TaskBackend::Auto);

        for &leaf in &leaves {
            g.add_dependency(leaf, root).expect("leaf depends on root");
            g.add_dependency(sink, leaf).expect("sink depends on leaf");
        }

        let order = g.topological_sort().expect("no cycle in fan graph");
        assert_eq!(order.len(), 50);

        let pos_root = order.iter().position(|&x| x == root).expect("root");
        let pos_sink = order.iter().position(|&x| x == sink).expect("sink");
        assert_eq!(pos_root, 0, "root must be first");
        assert_eq!(pos_sink, 49, "sink must be last");
    }

    #[test]
    fn task_graph_get_task_mut_updates_state() {
        let mut g = TaskGraph::new();
        let id = g.add_task("task", TaskBackend::Auto);
        if let Some(task) = g.get_task_mut(id) {
            task.estimated_cost = 999;
        }
        assert_eq!(
            g.get_task(id).expect("task should exist").estimated_cost,
            999
        );
    }

    #[test]
    fn task_graph_pending_dependency_count() {
        let mut g = TaskGraph::new();
        let a = g.add_task("a", TaskBackend::Auto);
        let b = g.add_task("b", TaskBackend::Auto);
        g.add_dependency(b, a).expect("b depends on a");

        let completed = std::collections::HashSet::new();
        let node_b = g.get_task(b).expect("b exists");
        assert_eq!(node_b.pending_dependency_count(&completed), 1);

        let mut completed2 = std::collections::HashSet::new();
        completed2.insert(a);
        assert_eq!(node_b.pending_dependency_count(&completed2), 0);
    }
}
