//! gRPC communication layer

mod service;

pub use service::FarmCoordinatorService;

use crate::{JobState, Priority, TaskState, WorkerState};

/// Convert protobuf `JobState` to internal `JobState`
#[allow(dead_code)]
#[must_use]
pub fn pb_to_job_state(state: i32) -> JobState {
    // Simple mapping based on protobuf enum values
    match state {
        0 => JobState::Pending,
        1 => JobState::Queued,
        2 => JobState::Running,
        3 => JobState::Completed,
        4 => JobState::Failed,
        5 => JobState::Cancelled,
        6 => JobState::Paused,
        _ => JobState::Pending,
    }
}

/// Convert internal `JobState` to protobuf `JobState`
#[allow(dead_code)]
#[must_use]
pub fn job_state_to_pb(state: JobState) -> i32 {
    match state {
        JobState::Pending => 0,
        JobState::Queued => 1,
        JobState::Running => 2,
        JobState::Completed => 3,
        JobState::Failed => 4,
        JobState::Cancelled => 5,
        JobState::Paused => 6,
        JobState::CompletedWithWarnings => 7,
    }
}

/// Convert protobuf `WorkerState` to internal `WorkerState`
#[allow(dead_code)]
#[must_use]
pub fn pb_to_worker_state(state: i32) -> WorkerState {
    match state {
        0 => WorkerState::Idle,
        1 => WorkerState::Busy,
        2 => WorkerState::Overloaded,
        3 => WorkerState::Draining,
        4 => WorkerState::Offline,
        _ => WorkerState::Offline,
    }
}

/// Convert internal `WorkerState` to protobuf `WorkerState`
#[allow(dead_code)]
#[must_use]
pub fn worker_state_to_pb(state: WorkerState) -> i32 {
    match state {
        WorkerState::Idle => 0,
        WorkerState::Busy => 1,
        WorkerState::Overloaded => 2,
        WorkerState::Draining => 3,
        WorkerState::Offline => 4,
    }
}

/// Convert protobuf Priority to internal Priority
#[allow(dead_code)]
#[must_use]
pub fn pb_to_priority(priority: i32) -> Priority {
    match priority {
        0 => Priority::Low,
        1 => Priority::Normal,
        2 => Priority::High,
        3 => Priority::Critical,
        _ => Priority::Normal,
    }
}

/// Convert internal Priority to protobuf Priority
#[allow(dead_code)]
#[must_use]
pub fn priority_to_pb(priority: Priority) -> i32 {
    match priority {
        Priority::Low => 0,
        Priority::Normal => 1,
        Priority::High => 2,
        Priority::Critical => 3,
    }
}

/// Convert protobuf `TaskState` to internal `TaskState`
#[allow(dead_code)]
#[must_use]
pub fn pb_to_task_state(state: i32) -> TaskState {
    match state {
        0 => TaskState::Pending,
        1 => TaskState::Assigned,
        2 => TaskState::Running,
        3 => TaskState::Completed,
        4 => TaskState::Failed,
        _ => TaskState::Pending,
    }
}

/// Convert internal `TaskState` to protobuf `TaskState`
#[allow(dead_code)]
#[must_use]
pub fn task_state_to_pb(state: TaskState) -> i32 {
    match state {
        TaskState::Pending => 0,
        TaskState::Assigned => 1,
        TaskState::Running => 2,
        TaskState::Completed => 3,
        TaskState::Failed => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_state_conversion() {
        let states = vec![
            JobState::Pending,
            JobState::Queued,
            JobState::Running,
            JobState::Completed,
            JobState::Failed,
            JobState::Cancelled,
            JobState::Paused,
        ];

        for state in states {
            let pb = job_state_to_pb(state);
            let back = pb_to_job_state(pb);
            assert_eq!(state, back);
        }
    }

    #[test]
    fn test_worker_state_conversion() {
        let states = vec![
            WorkerState::Idle,
            WorkerState::Busy,
            WorkerState::Overloaded,
            WorkerState::Draining,
            WorkerState::Offline,
        ];

        for state in states {
            let pb = worker_state_to_pb(state);
            let back = pb_to_worker_state(pb);
            assert_eq!(state, back);
        }
    }

    #[test]
    fn test_priority_conversion() {
        let priorities = vec![
            Priority::Low,
            Priority::Normal,
            Priority::High,
            Priority::Critical,
        ];

        for priority in priorities {
            let pb = priority_to_pb(priority);
            let back = pb_to_priority(pb);
            assert_eq!(priority, back);
        }
    }

    #[test]
    fn test_task_state_conversion() {
        let states = vec![
            TaskState::Pending,
            TaskState::Assigned,
            TaskState::Running,
            TaskState::Completed,
            TaskState::Failed,
        ];

        for state in states {
            let pb = task_state_to_pb(state);
            let back = pb_to_task_state(pb);
            assert_eq!(state, back);
        }
    }
}
