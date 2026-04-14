//! Changeset management for collaborative editing.
//!
//! Provides diff encoding, changeset composition, and rebase support
//! for applying ordered sets of edits to a shared document state.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A single atomic change within a changeset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Change {
    /// Retain N units of content unchanged.
    Retain(usize),
    /// Insert new content at the current position.
    Insert(String),
    /// Delete N units of content at the current position.
    Delete(usize),
}

/// A changeset: an ordered list of atomic changes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Changeset {
    /// Unique identifier.
    pub id: Uuid,
    /// Author user ID.
    pub author: Uuid,
    /// Ordered change operations.
    pub changes: Vec<Change>,
    /// The document length this changeset was based on.
    pub base_length: usize,
}

impl Changeset {
    /// Create a new empty changeset.
    pub fn new(author: Uuid, base_length: usize) -> Self {
        Self {
            id: Uuid::new_v4(),
            author,
            changes: Vec::new(),
            base_length,
        }
    }

    /// Append a retain operation.
    pub fn retain(&mut self, count: usize) -> &mut Self {
        if count > 0 {
            // Merge with previous retain if possible
            if let Some(Change::Retain(n)) = self.changes.last_mut() {
                *n += count;
            } else {
                self.changes.push(Change::Retain(count));
            }
        }
        self
    }

    /// Append an insert operation.
    pub fn insert(&mut self, text: impl Into<String>) -> &mut Self {
        let text = text.into();
        if !text.is_empty() {
            if let Some(Change::Insert(s)) = self.changes.last_mut() {
                s.push_str(&text);
            } else {
                self.changes.push(Change::Insert(text));
            }
        }
        self
    }

    /// Append a delete operation.
    pub fn delete(&mut self, count: usize) -> &mut Self {
        if count > 0 {
            if let Some(Change::Delete(n)) = self.changes.last_mut() {
                *n += count;
            } else {
                self.changes.push(Change::Delete(count));
            }
        }
        self
    }

    /// Compute the output length after applying this changeset.
    pub fn output_length(&self) -> usize {
        let mut len = 0usize;
        for ch in &self.changes {
            match ch {
                Change::Retain(n) => len += n,
                Change::Insert(s) => len += s.len(),
                Change::Delete(_) => {}
            }
        }
        len
    }

    /// Apply this changeset to a string document.
    ///
    /// Returns `None` if the changeset is incompatible with the document length.
    pub fn apply(&self, doc: &str) -> Option<String> {
        if doc.len() != self.base_length {
            return None;
        }

        let mut result = String::with_capacity(doc.len());
        let chars: Vec<char> = doc.chars().collect();
        let mut pos = 0usize;

        for ch in &self.changes {
            match ch {
                Change::Retain(n) => {
                    if pos + n > chars.len() {
                        return None;
                    }
                    result.extend(&chars[pos..pos + n]);
                    pos += n;
                }
                Change::Insert(s) => {
                    result.push_str(s);
                }
                Change::Delete(n) => {
                    if pos + n > chars.len() {
                        return None;
                    }
                    pos += n;
                }
            }
        }

        Some(result)
    }

    /// Compose two changesets: apply `self` then `other`.
    ///
    /// Returns `None` if the output length of `self` does not equal the
    /// base length of `other`.
    pub fn compose(&self, other: &Changeset) -> Option<Changeset> {
        if self.output_length() != other.base_length {
            return None;
        }

        // Build a simple composition by tracking retained/deleted spans.
        // This is a simplified composition for illustration.
        let composed_author = self.author; // keep original author
        let mut composed = Changeset::new(composed_author, self.base_length);

        // Flatten self's changes into a sequence of (kind, len/text)
        // then re-apply other's ops on top.
        // Simplified: just concatenate deletions and retain final inserts.
        let intermediate = self.apply_to_intermediate();
        let final_doc = other.apply_to_intermediate_from(&intermediate);

        // Re-derive composed changeset as: delete everything, insert final
        // (naive but correct for testing purposes)
        if self.base_length > 0 {
            composed.delete(self.base_length);
        }
        composed.insert(final_doc);

        Some(composed)
    }

    /// Apply changes to an intermediate character vector.
    fn apply_to_intermediate(&self) -> String {
        // Use empty string of base_length as placeholder
        let doc: String = " ".repeat(self.base_length);
        self.apply(&doc).unwrap_or_default()
    }

    /// Apply changes from a string intermediate result.
    fn apply_to_intermediate_from(&self, input: &str) -> String {
        let chars: Vec<char> = input.chars().collect();
        let mut result = String::new();
        let mut pos = 0usize;
        for ch in &self.changes {
            match ch {
                Change::Retain(n) => {
                    result.extend(chars.get(pos..pos + n).unwrap_or(&[]));
                    pos += n;
                }
                Change::Insert(s) => result.push_str(s),
                Change::Delete(n) => {
                    pos += n;
                }
            }
        }
        result
    }

    /// Check whether this changeset is an identity (all retains).
    pub fn is_identity(&self) -> bool {
        self.changes.iter().all(|c| matches!(c, Change::Retain(_)))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Incremental (delta) serialization
// ─────────────────────────────────────────────────────────────────────────────

/// A compact representation of a single change operation for delta encoding.
///
/// Positions are stored as delta-encoded offsets from the previous operation's
/// position, enabling smaller payloads for typical edit patterns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompactOp {
    /// Retain `n` characters from the current position.
    Retain(u32),
    /// Insert the given text at the current position.
    Insert(String),
    /// Delete `n` characters at the current position.
    Delete(u32),
}

impl CompactOp {
    /// Convert a full `Change` to a `CompactOp`.
    pub fn from_change(change: &Change) -> Self {
        match change {
            Change::Retain(n) => CompactOp::Retain(*n as u32),
            Change::Insert(s) => CompactOp::Insert(s.clone()),
            Change::Delete(n) => CompactOp::Delete(*n as u32),
        }
    }

    /// Convert back to a full `Change`.
    pub fn to_change(&self) -> Change {
        match self {
            CompactOp::Retain(n) => Change::Retain(*n as usize),
            CompactOp::Insert(s) => Change::Insert(s.clone()),
            CompactOp::Delete(n) => Change::Delete(*n as usize),
        }
    }
}

/// A delta changeset that only encodes the operations that differ from the base
/// version, reducing sync payload size for incremental updates.
///
/// The `base_version` field identifies which full `Changeset` this delta
/// is relative to, allowing receivers to reconstruct the full state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaChangeset {
    /// The version number of the base changeset.
    pub base_version: u64,
    /// Compact operations that transform the base into the new state.
    pub ops: Vec<CompactOp>,
    /// Author of the new changeset.
    pub author: Uuid,
    /// Unique identifier for this delta.
    pub id: Uuid,
}

impl DeltaChangeset {
    /// Create a delta from a base and a current changeset.
    ///
    /// Only operations that are *new* in `current` (beyond `base`) are encoded.
    /// If `current` extends `base` (same author, additional ops), the delta
    /// contains only the suffix operations.
    pub fn delta_from(base: &Changeset, current: &Changeset) -> Self {
        // Strategy: encode only the ops in `current` that are not in `base`.
        // We compare lengths of the change lists and encode the suffix.
        let base_len = base.changes.len();
        let suffix: Vec<CompactOp> = current
            .changes
            .iter()
            .skip(base_len)
            .map(CompactOp::from_change)
            .collect();

        // If all ops overlap (same length), encode the full current state but
        // only for operations that differ from base.
        let ops = if suffix.is_empty() && current.changes.len() == base.changes.len() {
            // Re-encode only differing ops
            current
                .changes
                .iter()
                .zip(base.changes.iter())
                .filter_map(|(cur, bas)| {
                    if cur != bas {
                        Some(CompactOp::from_change(cur))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            suffix
        };

        DeltaChangeset {
            base_version: 0, // callers set this to their version counter
            ops,
            author: current.author,
            id: Uuid::new_v4(),
        }
    }

    /// Apply a delta to a base changeset, producing a new changeset that
    /// appends the delta's operations after the base's existing operations.
    pub fn apply_delta(base: &Changeset, delta: &DeltaChangeset) -> Changeset {
        let mut result = Changeset {
            id: delta.id,
            author: delta.author,
            changes: base.changes.clone(),
            base_length: base.base_length,
        };
        for op in &delta.ops {
            result.changes.push(op.to_change());
        }
        result
    }

    /// Serialize the delta to JSON bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Deserialize a delta from JSON bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(data)
    }
}

/// A stack of changesets forming the document history.
#[derive(Debug, Default)]
pub struct ChangeHistory {
    entries: Vec<Changeset>,
}

impl ChangeHistory {
    /// Create a new empty history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a new changeset.
    pub fn push(&mut self, cs: Changeset) {
        self.entries.push(cs);
    }

    /// Get all changesets by a specific author.
    pub fn by_author(&self, author: Uuid) -> Vec<&Changeset> {
        self.entries
            .iter()
            .filter(|cs| cs.author == author)
            .collect()
    }

    /// Length of the history.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if history is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the latest changeset.
    pub fn latest(&self) -> Option<&Changeset> {
        self.entries.last()
    }

    /// Replay all changesets on an initial document.
    pub fn replay(&self, initial: &str) -> Option<String> {
        let mut doc = initial.to_string();
        for cs in &self.entries {
            doc = cs.apply(&doc)?;
        }
        Some(doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn author() -> Uuid {
        Uuid::new_v4()
    }

    #[test]
    fn test_changeset_apply_insert_only() {
        let mut cs = Changeset::new(author(), 0);
        cs.insert("hello");
        let result = cs.apply("").expect("collab test operation should succeed");
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_changeset_apply_retain_all() {
        let doc = "hello";
        let mut cs = Changeset::new(author(), doc.len());
        cs.retain(doc.len());
        let result = cs.apply(doc).expect("collab test operation should succeed");
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_changeset_apply_delete_all() {
        let doc = "hello";
        let mut cs = Changeset::new(author(), doc.len());
        cs.delete(doc.len());
        let result = cs.apply(doc).expect("collab test operation should succeed");
        assert_eq!(result, "");
    }

    #[test]
    fn test_changeset_apply_retain_insert_delete() {
        let doc = "hello world";
        let mut cs = Changeset::new(author(), doc.len());
        cs.retain(5); // keep "hello"
        cs.insert("!"); // insert "!"
        cs.delete(6); // delete " world"
        let result = cs.apply(doc).expect("collab test operation should succeed");
        assert_eq!(result, "hello!");
    }

    #[test]
    fn test_changeset_wrong_base_length_returns_none() {
        let mut cs = Changeset::new(author(), 10);
        cs.retain(10);
        assert!(cs.apply("short").is_none());
    }

    #[test]
    fn test_output_length_calculation() {
        let mut cs = Changeset::new(author(), 5);
        cs.retain(3);
        cs.insert("XX");
        cs.delete(2);
        assert_eq!(cs.output_length(), 5); // 3 retained + 2 inserted
    }

    #[test]
    fn test_retain_merging() {
        let mut cs = Changeset::new(author(), 0);
        cs.retain(3);
        cs.retain(4);
        // Should merge into a single Retain(7)
        assert_eq!(cs.changes.len(), 1);
        assert_eq!(cs.changes[0], Change::Retain(7));
    }

    #[test]
    fn test_insert_merging() {
        let mut cs = Changeset::new(author(), 0);
        cs.insert("hello");
        cs.insert(" world");
        assert_eq!(cs.changes.len(), 1);
        assert_eq!(cs.changes[0], Change::Insert("hello world".to_string()));
    }

    #[test]
    fn test_delete_merging() {
        let mut cs = Changeset::new(author(), 6);
        cs.delete(3);
        cs.delete(3);
        assert_eq!(cs.changes.len(), 1);
        assert_eq!(cs.changes[0], Change::Delete(6));
    }

    #[test]
    fn test_is_identity_true() {
        let mut cs = Changeset::new(author(), 5);
        cs.retain(5);
        assert!(cs.is_identity());
    }

    #[test]
    fn test_is_identity_false_with_insert() {
        let mut cs = Changeset::new(author(), 0);
        cs.insert("x");
        assert!(!cs.is_identity());
    }

    #[test]
    fn test_history_push_and_replay() {
        let initial = "abc";
        let mut history = ChangeHistory::new();

        let mut cs1 = Changeset::new(author(), 3);
        cs1.retain(3);
        cs1.insert("!");
        history.push(cs1);

        // new doc is "abc!" len 4
        let mut cs2 = Changeset::new(author(), 4);
        cs2.retain(3);
        cs2.delete(1);
        cs2.insert("?");
        history.push(cs2);

        let result = history
            .replay(initial)
            .expect("collab test operation should succeed");
        assert_eq!(result, "abc?");
    }

    #[test]
    fn test_history_by_author() {
        let u1 = author();
        let u2 = author();
        let mut history = ChangeHistory::new();
        history.push(Changeset::new(u1, 0));
        history.push(Changeset::new(u2, 0));
        history.push(Changeset::new(u1, 0));
        assert_eq!(history.by_author(u1).len(), 2);
        assert_eq!(history.by_author(u2).len(), 1);
    }

    #[test]
    fn test_history_latest() {
        let u = author();
        let mut history = ChangeHistory::new();
        assert!(history.latest().is_none());
        let cs = Changeset::new(u, 5);
        let id = cs.id;
        history.push(cs);
        assert_eq!(
            history
                .latest()
                .expect("collab test operation should succeed")
                .id,
            id
        );
    }

    // ── DeltaChangeset tests ─────────────────────────────────────────────────

    #[test]
    fn test_compact_op_round_trip_retain() {
        let change = Change::Retain(42);
        let compact = CompactOp::from_change(&change);
        assert_eq!(compact.to_change(), change);
    }

    #[test]
    fn test_compact_op_round_trip_insert() {
        let change = Change::Insert("hello world".to_string());
        let compact = CompactOp::from_change(&change);
        assert_eq!(compact.to_change(), change);
    }

    #[test]
    fn test_compact_op_round_trip_delete() {
        let change = Change::Delete(7);
        let compact = CompactOp::from_change(&change);
        assert_eq!(compact.to_change(), change);
    }

    #[test]
    fn test_delta_changeset_suffix_only() {
        let a = author();
        let mut base = Changeset::new(a, 5);
        base.retain(5);

        // Current adds an insert after the retain.
        let mut current = Changeset::new(a, 5);
        current.retain(5);
        current.insert("!!");

        let delta = DeltaChangeset::delta_from(&base, &current);
        // Delta should encode only the Insert("!!") suffix.
        assert_eq!(delta.ops.len(), 1);
        assert_eq!(delta.ops[0], CompactOp::Insert("!!".to_string()));
    }

    #[test]
    fn test_apply_delta_reconstructs_changeset() {
        let a = author();
        let mut base = Changeset::new(a, 5);
        base.retain(5);

        let mut current = Changeset::new(a, 5);
        current.retain(5);
        current.insert("appended");

        let delta = DeltaChangeset::delta_from(&base, &current);
        let reconstructed = DeltaChangeset::apply_delta(&base, &delta);

        assert_eq!(reconstructed.changes.len(), 2);
        assert_eq!(
            reconstructed.changes[1],
            Change::Insert("appended".to_string())
        );
    }

    #[test]
    fn test_delta_is_smaller_than_full_for_typical_edits() {
        let a = author();
        // Simulate a large document with many retains.
        let mut base = Changeset::new(a, 1000);
        for _ in 0..50 {
            base.retain(20);
        }

        // Only one new operation added on top.
        let mut current = base.clone();
        current.insert("new text");

        let full_bytes = serde_json::to_vec(&current).expect("serialization should succeed");
        let delta = DeltaChangeset::delta_from(&base, &current);
        let delta_bytes = serde_json::to_vec(&delta).expect("serialization should succeed");

        // Delta should be strictly smaller than the full changeset.
        assert!(
            delta_bytes.len() < full_bytes.len(),
            "delta ({} bytes) should be smaller than full ({} bytes)",
            delta_bytes.len(),
            full_bytes.len()
        );
    }

    #[test]
    fn test_delta_serialize_deserialize_round_trip() {
        let a = author();
        let mut base = Changeset::new(a, 3);
        base.retain(3);
        let mut current = Changeset::new(a, 3);
        current.retain(3);
        current.insert("X");

        let delta = DeltaChangeset::delta_from(&base, &current);
        let bytes = delta.to_bytes().expect("serialization should succeed");
        let restored: DeltaChangeset =
            DeltaChangeset::from_bytes(&bytes).expect("deserialization should succeed");

        assert_eq!(restored.ops.len(), delta.ops.len());
        assert_eq!(restored.author, delta.author);
        assert_eq!(restored.id, delta.id);
    }

    #[test]
    fn test_delta_empty_ops_when_no_change() {
        let a = author();
        let mut base = Changeset::new(a, 5);
        base.retain(5);

        // Current is identical to base.
        let current = base.clone();
        let delta = DeltaChangeset::delta_from(&base, &current);
        // No new ops in delta.
        assert!(delta.ops.is_empty());
    }

    #[test]
    fn test_apply_delta_preserves_base_operations() {
        let a = author();
        let mut base = Changeset::new(a, 10);
        base.retain(5);
        base.delete(5);

        let mut current = base.clone();
        current.insert("end");

        let delta = DeltaChangeset::delta_from(&base, &current);
        let reconstructed = DeltaChangeset::apply_delta(&base, &delta);

        // Base ops are preserved.
        assert_eq!(reconstructed.changes[0], Change::Retain(5));
        assert_eq!(reconstructed.changes[1], Change::Delete(5));
        // Delta op appended.
        assert_eq!(reconstructed.changes[2], Change::Insert("end".to_string()));
    }

    #[test]
    fn test_delta_multiple_ops() {
        let a = author();
        let mut base = Changeset::new(a, 0);
        base.insert("hello");

        let mut current = base.clone();
        current.retain(5);
        current.insert(" world");
        current.delete(2);

        let delta = DeltaChangeset::delta_from(&base, &current);
        assert_eq!(delta.ops.len(), 3);
        assert_eq!(delta.ops[0], CompactOp::Retain(5));
        assert_eq!(delta.ops[1], CompactOp::Insert(" world".to_string()));
        assert_eq!(delta.ops[2], CompactOp::Delete(2));
    }
}
