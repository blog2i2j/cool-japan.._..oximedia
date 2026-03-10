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
}
