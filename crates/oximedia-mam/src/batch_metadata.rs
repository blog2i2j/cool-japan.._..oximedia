//! Batch metadata operations for the MAM asset layer.
//!
//! Provides an efficient in-memory staging layer that accumulates metadata
//! update commands and flushes them as a single logical batch, dramatically
//! reducing the number of database round-trips for bulk-update scenarios
//! (e.g. re-tagging thousands of assets after an ingest run).
//!
//! The module is **database-agnostic at the core** — it does not call `sqlx`
//! directly.  Callers provide a [`BatchExecutor`] implementation that
//! translates the flushed [`MetadataCommand`]s into concrete SQL or other
//! persistence calls.  A no-op [`NoopExecutor`] is shipped for testing.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_mam::batch_metadata::{
//!     BatchBuffer, MetadataCommand, MetadataField, NoopExecutor,
//! };
//! use uuid::Uuid;
//!
//! let executor = NoopExecutor::default();
//! let mut buffer = BatchBuffer::new(executor, 100);
//!
//! let asset_id = Uuid::new_v4();
//! buffer.push(MetadataCommand::Set {
//!     asset_id,
//!     field: MetadataField::Title("Documentary – Part 1".into()),
//! });
//!
//! // When the buffer is full or explicitly flushed the commands are executed.
//! let stats = buffer.flush().expect("flush failed");
//! assert_eq!(stats.commands_executed, 1);
//! ```

use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// MetadataField
// ---------------------------------------------------------------------------

/// A single typed metadata field that can be updated on an asset.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataField {
    /// Asset title.
    Title(String),
    /// Asset description.
    Description(String),
    /// Replace the full keyword list.
    Keywords(Vec<String>),
    /// Append one keyword to the existing list.
    AddKeyword(String),
    /// Remove one keyword from the existing list.
    RemoveKeyword(String),
    /// Replace the category list.
    Categories(Vec<String>),
    /// Copyright statement.
    Copyright(String),
    /// SPDX license identifier.
    License(String),
    /// Creator / author name.
    Creator(String),
    /// A single custom key-value pair.
    Custom { key: String, value: serde_json::Value },
    /// Asset lifecycle status string (e.g. "archived", "active").
    Status(String),
}

impl MetadataField {
    /// Short discriminant name, used for de-duplication keys.
    #[must_use]
    pub fn field_name(&self) -> &'static str {
        match self {
            Self::Title(_) => "title",
            Self::Description(_) => "description",
            Self::Keywords(_) => "keywords",
            Self::AddKeyword(_) => "add_keyword",
            Self::RemoveKeyword(_) => "remove_keyword",
            Self::Categories(_) => "categories",
            Self::Copyright(_) => "copyright",
            Self::License(_) => "license",
            Self::Creator(_) => "creator",
            Self::Custom { .. } => "custom",
            Self::Status(_) => "status",
        }
    }
}

// ---------------------------------------------------------------------------
// MetadataCommand
// ---------------------------------------------------------------------------

/// A single metadata mutation command targeting one asset.
#[derive(Debug, Clone)]
pub enum MetadataCommand {
    /// Set or overwrite a field.
    Set { asset_id: Uuid, field: MetadataField },
    /// Delete a custom key from an asset's custom_metadata map.
    DeleteCustomKey { asset_id: Uuid, key: String },
    /// Clear all custom metadata for an asset.
    ClearCustomMetadata { asset_id: Uuid },
}

impl MetadataCommand {
    /// Return the asset this command targets.
    #[must_use]
    pub fn asset_id(&self) -> Uuid {
        match self {
            Self::Set { asset_id, .. }
            | Self::DeleteCustomKey { asset_id, .. }
            | Self::ClearCustomMetadata { asset_id } => *asset_id,
        }
    }
}

// ---------------------------------------------------------------------------
// BatchStats
// ---------------------------------------------------------------------------

/// Statistics produced after a flush.
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Number of commands that were sent to the executor.
    pub commands_executed: usize,
    /// Number of distinct assets touched.
    pub assets_touched: usize,
    /// Number of commands that were deduplicated (dropped as redundant).
    pub commands_deduped: usize,
}

// ---------------------------------------------------------------------------
// BatchExecutor trait
// ---------------------------------------------------------------------------

/// Persistence back-end for [`BatchBuffer`].
///
/// Implementors translate a slice of [`MetadataCommand`]s into actual
/// database writes (or any other side effect).
pub trait BatchExecutor {
    /// Execute a batch of commands.
    ///
    /// # Errors
    ///
    /// Returns a string description of any error that occurred.
    fn execute_batch(&mut self, commands: &[MetadataCommand]) -> Result<(), String>;
}

// ---------------------------------------------------------------------------
// NoopExecutor
// ---------------------------------------------------------------------------

/// A no-op executor used in testing.  Records every command it receives.
#[derive(Debug, Default)]
pub struct NoopExecutor {
    /// All commands received across all flush calls.
    pub received: Vec<MetadataCommand>,
}

impl BatchExecutor for NoopExecutor {
    fn execute_batch(&mut self, commands: &[MetadataCommand]) -> Result<(), String> {
        self.received.extend_from_slice(commands);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FailingExecutor
// ---------------------------------------------------------------------------

/// An executor that always returns an error.  Used to test error propagation.
#[derive(Debug, Default)]
pub struct FailingExecutor;

impl BatchExecutor for FailingExecutor {
    fn execute_batch(&mut self, _commands: &[MetadataCommand]) -> Result<(), String> {
        Err("simulated executor failure".into())
    }
}

// ---------------------------------------------------------------------------
// DeduplicationPolicy
// ---------------------------------------------------------------------------

/// Controls how the buffer handles duplicate commands for the same
/// (asset, field) pair before flushing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeduplicationPolicy {
    /// Keep all commands in insertion order (no deduplication).
    None,
    /// For each (asset_id, field_name) pair keep only the **last** command.
    LastWins,
    /// For each (asset_id, field_name) pair keep only the **first** command.
    FirstWins,
}

// ---------------------------------------------------------------------------
// BatchBuffer
// ---------------------------------------------------------------------------

/// In-memory staging buffer for metadata commands.
///
/// Accumulates [`MetadataCommand`]s and flushes them in a single batch when
/// the buffer is full or when [`BatchBuffer::flush`] is called explicitly.
pub struct BatchBuffer<E: BatchExecutor> {
    executor: E,
    capacity: usize,
    commands: Vec<MetadataCommand>,
    dedup_policy: DeduplicationPolicy,
    total_flushed: usize,
    total_deduped: usize,
}

impl<E: BatchExecutor> BatchBuffer<E> {
    /// Create a new buffer backed by `executor` with the given capacity.
    ///
    /// When the buffer reaches `capacity` commands it automatically flushes.
    #[must_use]
    pub fn new(executor: E, capacity: usize) -> Self {
        Self {
            executor,
            capacity: capacity.max(1),
            commands: Vec::new(),
            dedup_policy: DeduplicationPolicy::LastWins,
            total_flushed: 0,
            total_deduped: 0,
        }
    }

    /// Override the deduplication policy (default: [`DeduplicationPolicy::LastWins`]).
    pub fn with_dedup_policy(mut self, policy: DeduplicationPolicy) -> Self {
        self.dedup_policy = policy;
        self
    }

    /// Push a command into the buffer.  If the buffer is now at capacity the
    /// commands are flushed automatically.
    ///
    /// # Errors
    ///
    /// Propagates errors from the executor if an auto-flush is triggered.
    pub fn push(&mut self, cmd: MetadataCommand) -> Result<(), String> {
        self.commands.push(cmd);
        if self.commands.len() >= self.capacity {
            self.flush().map(|_| ())
        } else {
            Ok(())
        }
    }

    /// Push multiple commands at once.
    ///
    /// # Errors
    ///
    /// Returns on the first auto-flush error.
    pub fn push_many(&mut self, cmds: impl IntoIterator<Item = MetadataCommand>) -> Result<(), String> {
        for cmd in cmds {
            self.push(cmd)?;
        }
        Ok(())
    }

    /// Number of commands currently staged in the buffer.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.commands.len()
    }

    /// Total commands sent to the executor across all flushes.
    #[must_use]
    pub fn total_flushed(&self) -> usize {
        self.total_flushed
    }

    /// Total commands dropped by deduplication.
    #[must_use]
    pub fn total_deduped(&self) -> usize {
        self.total_deduped
    }

    /// Flush all staged commands immediately.
    ///
    /// # Errors
    ///
    /// Propagates errors from the executor.
    pub fn flush(&mut self) -> Result<BatchStats, String> {
        if self.commands.is_empty() {
            return Ok(BatchStats::default());
        }

        let raw_count = self.commands.len();
        let deduped = self.apply_dedup();
        let deduped_count = raw_count - deduped.len();
        let assets_touched: std::collections::HashSet<Uuid> =
            deduped.iter().map(|c| c.asset_id()).collect();
        let commands_executed = deduped.len();

        self.executor.execute_batch(&deduped)?;

        self.total_flushed += commands_executed;
        self.total_deduped += deduped_count;
        self.commands.clear();

        Ok(BatchStats {
            commands_executed,
            assets_touched: assets_touched.len(),
            commands_deduped: deduped_count,
        })
    }

    /// Apply the configured deduplication policy and return the resulting
    /// command list.
    fn apply_dedup(&self) -> Vec<MetadataCommand> {
        match self.dedup_policy {
            DeduplicationPolicy::None => self.commands.clone(),
            DeduplicationPolicy::LastWins => {
                // Walk in order; track last position for each (asset, field_name) key.
                let mut last_index: HashMap<(Uuid, &'static str), usize> = HashMap::new();
                for (i, cmd) in self.commands.iter().enumerate() {
                    if let MetadataCommand::Set { asset_id, field } = cmd {
                        last_index.insert((*asset_id, field.field_name()), i);
                    }
                }
                self.commands
                    .iter()
                    .enumerate()
                    .filter(|(i, cmd)| {
                        if let MetadataCommand::Set { asset_id, field } = cmd {
                            last_index.get(&(*asset_id, field.field_name())) == Some(i)
                        } else {
                            true
                        }
                    })
                    .map(|(_, c)| c.clone())
                    .collect()
            }
            DeduplicationPolicy::FirstWins => {
                let mut seen: HashMap<(Uuid, &'static str), bool> = HashMap::new();
                self.commands
                    .iter()
                    .filter(|cmd| {
                        if let MetadataCommand::Set { asset_id, field } = cmd {
                            seen.insert((*asset_id, field.field_name()), true).is_none()
                        } else {
                            true
                        }
                    })
                    .cloned()
                    .collect()
            }
        }
    }

    /// Consume the buffer, returning the underlying executor.
    ///
    /// Any unflushed commands are discarded.  Call [`flush`][Self::flush]
    /// before calling this if you need them persisted.
    pub fn into_executor(self) -> E {
        self.executor
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn asset() -> Uuid {
        Uuid::new_v4()
    }

    #[test]
    fn test_push_and_flush_basic() {
        let exec = NoopExecutor::default();
        let mut buf = BatchBuffer::new(exec, 100);

        let id = asset();
        buf.push(MetadataCommand::Set {
            asset_id: id,
            field: MetadataField::Title("Test".into()),
        })
        .expect("push failed");

        let stats = buf.flush().expect("flush failed");
        assert_eq!(stats.commands_executed, 1);
        assert_eq!(stats.assets_touched, 1);
        assert_eq!(stats.commands_deduped, 0);
    }

    #[test]
    fn test_auto_flush_on_capacity() {
        let exec = NoopExecutor::default();
        let mut buf = BatchBuffer::new(exec, 3).with_dedup_policy(DeduplicationPolicy::None);

        let id = asset();
        for i in 0..3_u32 {
            let title = format!("asset {i}");
            buf.push(MetadataCommand::Set {
                asset_id: id,
                field: MetadataField::Title(title),
            })
            .expect("push failed");
        }
        // Buffer should have auto-flushed when the third command was pushed.
        assert_eq!(buf.pending_count(), 0);
        assert_eq!(buf.total_flushed(), 3);
    }

    #[test]
    fn test_dedup_last_wins() {
        let exec = NoopExecutor::default();
        let mut buf = BatchBuffer::new(exec, 100).with_dedup_policy(DeduplicationPolicy::LastWins);

        let id = asset();
        buf.push(MetadataCommand::Set {
            asset_id: id,
            field: MetadataField::Title("First".into()),
        })
        .expect("push");
        buf.push(MetadataCommand::Set {
            asset_id: id,
            field: MetadataField::Title("Second".into()),
        })
        .expect("push");
        buf.push(MetadataCommand::Set {
            asset_id: id,
            field: MetadataField::Title("Third".into()),
        })
        .expect("push");

        let stats = buf.flush().expect("flush");
        // Only the last Title command should survive.
        assert_eq!(stats.commands_executed, 1);
        assert_eq!(stats.commands_deduped, 2);

        let exec = buf.into_executor();
        assert_eq!(exec.received.len(), 1);
        if let MetadataCommand::Set { field: MetadataField::Title(t), .. } = &exec.received[0] {
            assert_eq!(t, "Third");
        } else {
            panic!("unexpected command");
        }
    }

    #[test]
    fn test_dedup_first_wins() {
        let exec = NoopExecutor::default();
        let mut buf = BatchBuffer::new(exec, 100).with_dedup_policy(DeduplicationPolicy::FirstWins);

        let id = asset();
        buf.push(MetadataCommand::Set {
            asset_id: id,
            field: MetadataField::Title("First".into()),
        })
        .expect("push");
        buf.push(MetadataCommand::Set {
            asset_id: id,
            field: MetadataField::Title("Second".into()),
        })
        .expect("push");

        let stats = buf.flush().expect("flush");
        assert_eq!(stats.commands_executed, 1);
        assert_eq!(stats.commands_deduped, 1);

        let exec = buf.into_executor();
        if let MetadataCommand::Set { field: MetadataField::Title(t), .. } = &exec.received[0] {
            assert_eq!(t, "First");
        } else {
            panic!("unexpected command");
        }
    }

    #[test]
    fn test_dedup_none_keeps_all() {
        let exec = NoopExecutor::default();
        let mut buf = BatchBuffer::new(exec, 100).with_dedup_policy(DeduplicationPolicy::None);

        let id = asset();
        for _ in 0..5 {
            buf.push(MetadataCommand::Set {
                asset_id: id,
                field: MetadataField::Title("x".into()),
            })
            .expect("push");
        }
        let stats = buf.flush().expect("flush");
        assert_eq!(stats.commands_executed, 5);
        assert_eq!(stats.commands_deduped, 0);
    }

    #[test]
    fn test_flush_empty_buffer_returns_zero_stats() {
        let exec = NoopExecutor::default();
        let mut buf = BatchBuffer::new(exec, 100);
        let stats = buf.flush().expect("flush");
        assert_eq!(stats.commands_executed, 0);
        assert_eq!(stats.assets_touched, 0);
    }

    #[test]
    fn test_multiple_assets_counted_correctly() {
        let exec = NoopExecutor::default();
        let mut buf = BatchBuffer::new(exec, 100);

        for _ in 0..5 {
            buf.push(MetadataCommand::Set {
                asset_id: asset(),
                field: MetadataField::Title("x".into()),
            })
            .expect("push");
        }
        let stats = buf.flush().expect("flush");
        assert_eq!(stats.assets_touched, 5);
    }

    #[test]
    fn test_executor_error_propagated() {
        let exec = FailingExecutor;
        let mut buf = BatchBuffer::new(exec, 100);
        buf.push(MetadataCommand::Set {
            asset_id: asset(),
            field: MetadataField::Title("x".into()),
        })
        .expect("push pre-flush");
        let result = buf.flush();
        assert!(result.is_err());
    }

    #[test]
    fn test_push_many_helper() {
        let exec = NoopExecutor::default();
        let mut buf = BatchBuffer::new(exec, 100);
        let cmds: Vec<_> = (0..8)
            .map(|i| MetadataCommand::Set {
                asset_id: asset(),
                field: MetadataField::Title(format!("asset {i}")),
            })
            .collect();
        buf.push_many(cmds).expect("push_many");
        let stats = buf.flush().expect("flush");
        assert_eq!(stats.commands_executed, 8);
    }

    #[test]
    fn test_delete_custom_key_command_not_deduped() {
        let exec = NoopExecutor::default();
        let mut buf = BatchBuffer::new(exec, 100).with_dedup_policy(DeduplicationPolicy::LastWins);
        let id = asset();
        // Two delete-key commands should both survive (non-Set commands are exempt).
        buf.push(MetadataCommand::DeleteCustomKey {
            asset_id: id,
            key: "genre".into(),
        })
        .expect("push");
        buf.push(MetadataCommand::DeleteCustomKey {
            asset_id: id,
            key: "genre".into(),
        })
        .expect("push");
        let stats = buf.flush().expect("flush");
        assert_eq!(stats.commands_executed, 2);
        assert_eq!(stats.commands_deduped, 0);
    }

    #[test]
    fn test_metadata_field_names_are_stable() {
        assert_eq!(MetadataField::Title("".into()).field_name(), "title");
        assert_eq!(MetadataField::Description("".into()).field_name(), "description");
        assert_eq!(MetadataField::Keywords(vec![]).field_name(), "keywords");
        assert_eq!(MetadataField::Status("".into()).field_name(), "status");
        assert_eq!(
            MetadataField::Custom {
                key: "k".into(),
                value: serde_json::Value::Null
            }
            .field_name(),
            "custom"
        );
    }
}
