//! Shared undo/redo history
//!
//! This module implements shared undo/redo functionality with change attribution,
//! history branching, and compaction.

use crate::crdt::TimelineOp;
use crate::{CollabError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// History entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub id: Uuid,
    pub operation: TimelineOp,
    pub user_id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub parent_id: Option<Uuid>,
}

impl HistoryEntry {
    /// Create a new history entry
    pub fn new(operation: TimelineOp, user_id: Uuid, parent_id: Option<Uuid>) -> Self {
        Self {
            id: Uuid::new_v4(),
            operation,
            user_id,
            timestamp: chrono::Utc::now(),
            parent_id,
        }
    }

    /// Check if entry is older than a duration
    pub fn is_older_than(&self, duration: chrono::Duration) -> bool {
        let now = chrono::Utc::now();
        let age = now.signed_duration_since(self.timestamp);
        age > duration
    }
}

/// Undo/redo stack for a user
#[derive(Debug, Clone)]
struct UserStack {
    undo_stack: VecDeque<Uuid>,
    redo_stack: VecDeque<Uuid>,
}

impl UserStack {
    fn new() -> Self {
        Self {
            undo_stack: VecDeque::new(),
            redo_stack: VecDeque::new(),
        }
    }

    fn push_undo(&mut self, entry_id: Uuid) {
        self.undo_stack.push_back(entry_id);
        // Clear redo stack when new operation is added
        self.redo_stack.clear();
    }

    fn pop_undo(&mut self) -> Option<Uuid> {
        self.undo_stack.pop_back()
    }

    fn push_redo(&mut self, entry_id: Uuid) {
        self.redo_stack.push_back(entry_id);
    }

    fn pop_redo(&mut self) -> Option<Uuid> {
        self.redo_stack.pop_back()
    }

    fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }
}

/// History branch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryBranch {
    pub id: Uuid,
    pub name: String,
    pub base_entry_id: Uuid,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl HistoryBranch {
    /// Create a new history branch
    pub fn new(name: String, base_entry_id: Uuid) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            base_entry_id,
            created_at: chrono::Utc::now(),
        }
    }
}

/// History manager
pub struct HistoryManager {
    entries: Arc<RwLock<HashMap<Uuid, HistoryEntry>>>,
    user_stacks: Arc<RwLock<HashMap<Uuid, UserStack>>>,
    branches: Arc<RwLock<HashMap<Uuid, HistoryBranch>>>,
    max_size: usize,
    current_branch: Arc<RwLock<Option<Uuid>>>,
}

impl HistoryManager {
    /// Create a new history manager
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            user_stacks: Arc::new(RwLock::new(HashMap::new())),
            branches: Arc::new(RwLock::new(HashMap::new())),
            max_size,
            current_branch: Arc::new(RwLock::new(None)),
        }
    }

    /// Add an operation to history
    pub async fn add_operation(&self, operation: TimelineOp) -> Result<()> {
        let user_id = operation.user_id;

        // Get parent (last entry for this user)
        let parent_id = {
            let stacks = self.user_stacks.read().await;
            stacks
                .get(&user_id)
                .and_then(|stack| stack.undo_stack.back().copied())
        };

        // Create history entry
        let entry = HistoryEntry::new(operation, user_id, parent_id);
        let entry_id = entry.id;

        // Add to entries
        self.entries.write().await.insert(entry_id, entry);

        // Add to user's undo stack
        let mut stacks = self.user_stacks.write().await;
        let stack = stacks.entry(user_id).or_insert_with(UserStack::new);
        stack.push_undo(entry_id);

        // Check size and compact if needed
        if self.entries.read().await.len() > self.max_size {
            drop(stacks); // Release lock before compacting
            self.compact(self.max_size / 2).await?;
        }

        Ok(())
    }

    /// Undo last operation for a user
    pub async fn undo(&self, user_id: Uuid) -> Result<Option<TimelineOp>> {
        let mut stacks = self.user_stacks.write().await;
        let stack = stacks
            .get_mut(&user_id)
            .ok_or(CollabError::UserNotFound(user_id))?;

        if !stack.can_undo() {
            return Ok(None);
        }

        let entry_id = stack
            .pop_undo()
            .ok_or(CollabError::InvalidOperation("Nothing to undo".to_string()))?;

        // Move to redo stack
        stack.push_redo(entry_id);

        // Get the operation to undo
        let entries = self.entries.read().await;
        let entry = entries
            .get(&entry_id)
            .ok_or(CollabError::InvalidOperation("Entry not found".to_string()))?;

        Ok(Some(self.create_inverse_operation(&entry.operation)?))
    }

    /// Redo last undone operation for a user
    pub async fn redo(&self, user_id: Uuid) -> Result<Option<TimelineOp>> {
        let mut stacks = self.user_stacks.write().await;
        let stack = stacks
            .get_mut(&user_id)
            .ok_or(CollabError::UserNotFound(user_id))?;

        if !stack.can_redo() {
            return Ok(None);
        }

        let entry_id = stack
            .pop_redo()
            .ok_or(CollabError::InvalidOperation("Nothing to redo".to_string()))?;

        // Move back to undo stack
        stack.push_undo(entry_id);

        // Get the operation to redo
        let entries = self.entries.read().await;
        let entry = entries
            .get(&entry_id)
            .ok_or(CollabError::InvalidOperation("Entry not found".to_string()))?;

        Ok(Some(entry.operation.clone()))
    }

    /// Create inverse operation for undo
    fn create_inverse_operation(&self, op: &TimelineOp) -> Result<TimelineOp> {
        use crate::crdt::OpType;

        let inverse_type = match op.op_type {
            OpType::InsertClip => OpType::DeleteClip,
            OpType::DeleteClip => OpType::InsertClip,
            OpType::InsertTrack => OpType::DeleteTrack,
            OpType::DeleteTrack => OpType::InsertTrack,
            // For move and update operations, we need to store the previous state
            // For now, we'll return an error as this requires more complex state tracking
            _ => {
                return Err(CollabError::InvalidOperation(
                    "Cannot create inverse for this operation type".to_string(),
                ))
            }
        };

        Ok(TimelineOp {
            id: Uuid::new_v4(),
            op_type: inverse_type,
            user_id: op.user_id,
            timestamp: chrono::Utc::now(),
            data: op.data.clone(),
            version: op.version,
        })
    }

    /// Check if user can undo
    pub async fn can_undo(&self, user_id: Uuid) -> bool {
        self.user_stacks
            .read()
            .await
            .get(&user_id)
            .map(|stack| stack.can_undo())
            .unwrap_or(false)
    }

    /// Check if user can redo
    pub async fn can_redo(&self, user_id: Uuid) -> bool {
        self.user_stacks
            .read()
            .await
            .get(&user_id)
            .map(|stack| stack.can_redo())
            .unwrap_or(false)
    }

    /// Get history for a user
    pub async fn get_user_history(&self, user_id: Uuid) -> Vec<HistoryEntry> {
        let entries = self.entries.read().await;
        let stacks = self.user_stacks.read().await;

        if let Some(stack) = stacks.get(&user_id) {
            stack
                .undo_stack
                .iter()
                .filter_map(|id| entries.get(id).cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get all history entries
    pub async fn get_all_entries(&self) -> Vec<HistoryEntry> {
        let entries = self.entries.read().await;
        entries.values().cloned().collect()
    }

    /// Create a branch from current state
    pub async fn create_branch(&self, name: String, base_entry_id: Uuid) -> Result<Uuid> {
        let branch = HistoryBranch::new(name, base_entry_id);
        let branch_id = branch.id;

        self.branches.write().await.insert(branch_id, branch);
        *self.current_branch.write().await = Some(branch_id);

        Ok(branch_id)
    }

    /// Switch to a branch
    pub async fn switch_branch(&self, branch_id: Uuid) -> Result<()> {
        let branches = self.branches.read().await;
        if !branches.contains_key(&branch_id) {
            return Err(CollabError::InvalidOperation(
                "Branch not found".to_string(),
            ));
        }

        *self.current_branch.write().await = Some(branch_id);
        Ok(())
    }

    /// Get current branch
    pub async fn get_current_branch(&self) -> Option<HistoryBranch> {
        let current = *self.current_branch.read().await;
        if let Some(branch_id) = current {
            self.branches.read().await.get(&branch_id).cloned()
        } else {
            None
        }
    }

    /// List all branches
    pub async fn list_branches(&self) -> Vec<HistoryBranch> {
        self.branches.read().await.values().cloned().collect()
    }

    /// Compact history to reduce size
    pub async fn compact(&self, target_size: usize) -> Result<()> {
        let mut entries = self.entries.write().await;

        if entries.len() <= target_size {
            return Ok(());
        }

        // Sort entries by timestamp
        let mut sorted_entries: Vec<_> = entries
            .iter()
            .map(|(id, entry)| (*id, entry.timestamp))
            .collect();
        sorted_entries.sort_by_key(|(_, ts)| *ts);

        // Remove oldest entries
        let to_remove = entries.len() - target_size;
        let remove_ids: Vec<Uuid> = sorted_entries
            .iter()
            .take(to_remove)
            .map(|(id, _)| *id)
            .collect();

        for id in remove_ids {
            entries.remove(&id);
        }

        // Clean up user stacks
        let mut stacks = self.user_stacks.write().await;
        for stack in stacks.values_mut() {
            stack.undo_stack.retain(|id| entries.contains_key(id));
            stack.redo_stack.retain(|id| entries.contains_key(id));
        }

        tracing::info!(
            "Compacted history from {} to {} entries",
            entries.len() + to_remove,
            entries.len()
        );

        Ok(())
    }

    /// Remove old entries beyond a time threshold
    pub async fn remove_old_entries(&self, duration: chrono::Duration) -> Result<usize> {
        let mut entries = self.entries.write().await;
        let initial_count = entries.len();

        entries.retain(|_, entry| !entry.is_older_than(duration));

        // Clean up user stacks
        let mut stacks = self.user_stacks.write().await;
        for stack in stacks.values_mut() {
            stack.undo_stack.retain(|id| entries.contains_key(id));
            stack.redo_stack.retain(|id| entries.contains_key(id));
        }

        let removed = initial_count - entries.len();
        Ok(removed)
    }

    /// Get history size
    pub async fn size(&self) -> usize {
        self.entries.read().await.len()
    }

    /// Clear all history
    pub async fn clear(&self) -> Result<()> {
        self.entries.write().await.clear();
        self.user_stacks.write().await.clear();
        self.branches.write().await.clear();
        *self.current_branch.write().await = None;
        Ok(())
    }

    /// Get operations by user
    pub async fn get_user_operations(&self, user_id: Uuid) -> Vec<TimelineOp> {
        let entries = self.entries.read().await;
        entries
            .values()
            .filter(|entry| entry.user_id == user_id)
            .map(|entry| entry.operation.clone())
            .collect()
    }

    /// Get operations in time range
    #[allow(clippy::too_many_arguments)]
    pub async fn get_operations_in_range(
        &self,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Vec<HistoryEntry> {
        let entries = self.entries.read().await;
        entries
            .values()
            .filter(|entry| entry.timestamp >= start && entry.timestamp <= end)
            .cloned()
            .collect()
    }

    /// Export history to JSON
    pub async fn export_json(&self) -> Result<String> {
        let entries = self.get_all_entries().await;
        serde_json::to_string_pretty(&entries).map_err(CollabError::SerializationError)
    }

    /// Import history from JSON
    pub async fn import_json(&self, json: &str) -> Result<()> {
        let entries: Vec<HistoryEntry> =
            serde_json::from_str(json).map_err(CollabError::SerializationError)?;

        let mut entry_map = self.entries.write().await;
        for entry in entries {
            entry_map.insert(entry.id, entry);
        }

        Ok(())
    }
}

/// History statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryStats {
    pub total_entries: usize,
    pub entries_by_user: HashMap<Uuid, usize>,
    pub entries_by_type: HashMap<String, usize>,
    pub oldest_entry: Option<chrono::DateTime<chrono::Utc>>,
    pub newest_entry: Option<chrono::DateTime<chrono::Utc>>,
}

impl HistoryStats {
    /// Calculate statistics from history manager
    pub async fn from_manager(manager: &HistoryManager) -> Self {
        let entries = manager.get_all_entries().await;
        let total_entries = entries.len();

        let mut entries_by_user: HashMap<Uuid, usize> = HashMap::new();
        let mut entries_by_type: HashMap<String, usize> = HashMap::new();
        let mut oldest_entry: Option<chrono::DateTime<chrono::Utc>> = None;
        let mut newest_entry: Option<chrono::DateTime<chrono::Utc>> = None;

        for entry in entries {
            *entries_by_user.entry(entry.user_id).or_insert(0) += 1;
            *entries_by_type
                .entry(format!("{:?}", entry.operation.op_type))
                .or_insert(0) += 1;

            if oldest_entry.map_or(true, |t| entry.timestamp < t) {
                oldest_entry = Some(entry.timestamp);
            }

            if newest_entry.map_or(true, |t| entry.timestamp > t) {
                newest_entry = Some(entry.timestamp);
            }
        }

        Self {
            total_entries,
            entries_by_user,
            entries_by_type,
            oldest_entry,
            newest_entry,
        }
    }
}

/// History visualizer
pub struct HistoryVisualizer;

impl HistoryVisualizer {
    /// Generate timeline visualization as ASCII art
    pub fn visualize_timeline(entries: &[HistoryEntry]) -> String {
        let mut output = String::new();
        output.push_str("History Timeline:\n");
        output.push_str("================\n\n");

        for (idx, entry) in entries.iter().enumerate() {
            output.push_str(&format!(
                "{}: [{:?}] by {} at {}\n",
                idx + 1,
                entry.operation.op_type,
                entry.user_id,
                entry.timestamp.format("%Y-%m-%d %H:%M:%S")
            ));

            if let Some(parent_id) = entry.parent_id {
                output.push_str(&format!("   └─ parent: {}\n", parent_id));
            }
        }

        output
    }

    /// Generate DOT graph for branching visualization
    pub fn generate_dot_graph(entries: &[HistoryEntry], branches: &[HistoryBranch]) -> String {
        let mut output = String::new();
        output.push_str("digraph History {\n");
        output.push_str("  rankdir=TB;\n");
        output.push_str("  node [shape=box];\n\n");

        // Add nodes for entries
        for entry in entries {
            output.push_str(&format!(
                "  \"{}\" [label=\"{:?}\\n{}\"];\n",
                entry.id,
                entry.operation.op_type,
                entry.timestamp.format("%H:%M:%S")
            ));
        }

        // Add edges for parent relationships
        for entry in entries {
            if let Some(parent_id) = entry.parent_id {
                output.push_str(&format!("  \"{}\" -> \"{}\";\n", parent_id, entry.id));
            }
        }

        // Add branch nodes
        for branch in branches {
            output.push_str(&format!(
                "  \"branch_{}\" [label=\"Branch: {}\" shape=ellipse style=filled fillcolor=lightblue];\n",
                branch.id, branch.name
            ));
            output.push_str(&format!(
                "  \"{}\" -> \"branch_{}\" [style=dashed];\n",
                branch.base_entry_id, branch.id
            ));
        }

        output.push_str("}\n");
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crdt::OpType;

    fn create_test_op(user_id: Uuid, version: u64) -> TimelineOp {
        TimelineOp {
            id: Uuid::new_v4(),
            op_type: OpType::InsertClip,
            user_id,
            timestamp: chrono::Utc::now(),
            data: serde_json::json!({"test": "data"}),
            version,
        }
    }

    #[tokio::test]
    async fn test_add_operation() {
        let manager = HistoryManager::new(1000);
        let user_id = Uuid::new_v4();
        let op = create_test_op(user_id, 1);

        manager
            .add_operation(op)
            .await
            .expect("collab test operation should succeed");
        assert_eq!(manager.size().await, 1);
    }

    #[tokio::test]
    async fn test_undo_redo() {
        let manager = HistoryManager::new(1000);
        let user_id = Uuid::new_v4();

        let op1 = create_test_op(user_id, 1);
        manager
            .add_operation(op1)
            .await
            .expect("collab test operation should succeed");

        assert!(manager.can_undo(user_id).await);
        assert!(!manager.can_redo(user_id).await);

        manager
            .undo(user_id)
            .await
            .expect("collab test operation should succeed");
        assert!(!manager.can_undo(user_id).await);
        assert!(manager.can_redo(user_id).await);

        manager
            .redo(user_id)
            .await
            .expect("collab test operation should succeed");
        assert!(manager.can_undo(user_id).await);
        assert!(!manager.can_redo(user_id).await);
    }

    #[tokio::test]
    async fn test_compact() {
        let manager = HistoryManager::new(10);
        let user_id = Uuid::new_v4();

        // Add 20 operations
        for i in 0..20 {
            let op = create_test_op(user_id, i);
            manager
                .add_operation(op)
                .await
                .expect("collab test operation should succeed");
        }

        // Compact to 5 entries
        manager
            .compact(5)
            .await
            .expect("collab test operation should succeed");
        assert_eq!(manager.size().await, 5);
    }

    #[tokio::test]
    async fn test_branches() {
        let manager = HistoryManager::new(1000);
        let user_id = Uuid::new_v4();
        let op = create_test_op(user_id, 1);

        manager
            .add_operation(op.clone())
            .await
            .expect("collab test operation should succeed");

        let branch_id = manager
            .create_branch("feature".to_string(), op.id)
            .await
            .expect("collab test operation should succeed");

        let current = manager
            .get_current_branch()
            .await
            .expect("collab test operation should succeed");
        assert_eq!(current.id, branch_id);
        assert_eq!(current.name, "feature");
    }

    #[tokio::test]
    async fn test_history_stats() {
        let manager = HistoryManager::new(1000);
        let user1 = Uuid::new_v4();
        let user2 = Uuid::new_v4();

        manager
            .add_operation(create_test_op(user1, 1))
            .await
            .expect("collab test operation should succeed");
        manager
            .add_operation(create_test_op(user1, 2))
            .await
            .expect("collab test operation should succeed");
        manager
            .add_operation(create_test_op(user2, 3))
            .await
            .expect("collab test operation should succeed");

        let stats = HistoryStats::from_manager(&manager).await;
        assert_eq!(stats.total_entries, 3);
        assert_eq!(stats.entries_by_user.get(&user1), Some(&2));
        assert_eq!(stats.entries_by_user.get(&user2), Some(&1));
    }

    #[tokio::test]
    async fn test_export_import() {
        let manager = HistoryManager::new(1000);
        let user_id = Uuid::new_v4();

        manager
            .add_operation(create_test_op(user_id, 1))
            .await
            .expect("collab test operation should succeed");
        manager
            .add_operation(create_test_op(user_id, 2))
            .await
            .expect("collab test operation should succeed");

        let json = manager
            .export_json()
            .await
            .expect("collab test operation should succeed");

        let manager2 = HistoryManager::new(1000);
        manager2
            .import_json(&json)
            .await
            .expect("collab test operation should succeed");

        assert_eq!(manager2.size().await, 2);
    }
}
