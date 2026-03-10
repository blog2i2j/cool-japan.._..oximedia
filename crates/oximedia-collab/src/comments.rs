//! Threaded comment system for collaborative video review.
//!
//! Supports threaded replies, emoji reactions, comment resolution, and full-text search.

use std::collections::HashMap;

/// Strongly-typed comment identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub struct CommentId(pub u64);

/// A single comment in the system
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Comment {
    pub id: CommentId,
    pub parent_id: Option<CommentId>,
    pub author: String,
    pub content: String,
    pub timestamp_ms: u64,
    pub resolved: bool,
    /// emoji → list of user IDs who reacted
    pub reactions: HashMap<String, Vec<String>>,
}

impl Comment {
    /// Create a new top-level comment
    pub fn new(
        id: CommentId,
        author: impl Into<String>,
        content: impl Into<String>,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            id,
            parent_id: None,
            author: author.into(),
            content: content.into(),
            timestamp_ms,
            resolved: false,
            reactions: HashMap::new(),
        }
    }

    /// Create a reply to another comment
    pub fn reply(
        id: CommentId,
        parent_id: CommentId,
        author: impl Into<String>,
        content: impl Into<String>,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            id,
            parent_id: Some(parent_id),
            author: author.into(),
            content: content.into(),
            timestamp_ms,
            resolved: false,
            reactions: HashMap::new(),
        }
    }

    /// Add a reaction from a user (idempotent)
    pub fn add_reaction(&mut self, emoji: &str, user: &str) {
        let users = self.reactions.entry(emoji.to_string()).or_default();
        if !users.contains(&user.to_string()) {
            users.push(user.to_string());
        }
    }

    /// Remove a reaction from a user
    pub fn remove_reaction(&mut self, emoji: &str, user: &str) {
        if let Some(users) = self.reactions.get_mut(emoji) {
            users.retain(|u| u != user);
            if users.is_empty() {
                self.reactions.remove(emoji);
            }
        }
    }
}

/// A thread anchored at a root comment
pub struct CommentThread {
    pub root: CommentId,
    pub comments: Vec<Comment>,
}

impl CommentThread {
    /// Create a new thread with a root comment
    pub fn new(root: Comment) -> Self {
        let root_id = root.id;
        Self {
            root: root_id,
            comments: vec![root],
        }
    }

    /// Add a comment to the thread (must be a reply to an existing comment in this thread)
    pub fn add(&mut self, comment: Comment) {
        self.comments.push(comment);
    }

    /// Return all direct replies to the given comment id
    pub fn replies_to(&self, id: CommentId) -> Vec<&Comment> {
        self.comments
            .iter()
            .filter(|c| c.parent_id == Some(id))
            .collect()
    }

    /// Mark the entire thread as resolved
    pub fn resolve_thread(&mut self) {
        for c in self.comments.iter_mut() {
            c.resolved = true;
        }
    }

    /// True when all comments in the thread are resolved
    pub fn is_resolved(&self) -> bool {
        self.comments.iter().all(|c| c.resolved)
    }

    /// Return the root comment
    pub fn root_comment(&self) -> Option<&Comment> {
        self.comments.iter().find(|c| c.id == self.root)
    }
}

/// Where in the project a comment is anchored
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CommentAnchor {
    /// Frame number the comment is attached to
    pub frame: u64,
    /// Optional track identifier
    pub track_id: Option<String>,
    /// Timecode string (e.g. "01:02:03:04")
    pub timecode: String,
}

impl CommentAnchor {
    /// Create a new anchor
    pub fn new(frame: u64, timecode: impl Into<String>) -> Self {
        Self {
            frame,
            track_id: None,
            timecode: timecode.into(),
        }
    }

    /// Attach to a specific track
    pub fn on_track(mut self, track_id: impl Into<String>) -> Self {
        self.track_id = Some(track_id.into());
        self
    }
}

/// Central store for all comments
pub struct CommentStore {
    comments: Vec<Comment>,
    next_id: u64,
}

impl CommentStore {
    /// Create an empty store
    pub fn new() -> Self {
        Self {
            comments: Vec::new(),
            next_id: 1,
        }
    }

    /// Allocate the next unique CommentId
    pub fn next_id(&mut self) -> CommentId {
        let id = CommentId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Add a comment to the store
    pub fn add_comment(&mut self, comment: Comment) {
        self.comments.push(comment);
    }

    /// Delete a comment by id. Returns true if a comment was removed.
    pub fn delete_comment(&mut self, id: CommentId) -> bool {
        let before = self.comments.len();
        self.comments.retain(|c| c.id != id);
        self.comments.len() < before
    }

    /// Mark a comment as resolved. Returns true if found.
    pub fn resolve(&mut self, id: CommentId) -> bool {
        if let Some(c) = self.comments.iter_mut().find(|c| c.id == id) {
            c.resolved = true;
            true
        } else {
            false
        }
    }

    /// Full-text search across comment content (case-insensitive)
    pub fn search(&self, query: &str) -> Vec<&Comment> {
        let q = query.to_lowercase();
        self.comments
            .iter()
            .filter(|c| c.content.to_lowercase().contains(&q))
            .collect()
    }

    /// Return a reference to a comment by id
    pub fn get(&self, id: CommentId) -> Option<&Comment> {
        self.comments.iter().find(|c| c.id == id)
    }

    /// Mutable reference to a comment by id
    pub fn get_mut(&mut self, id: CommentId) -> Option<&mut Comment> {
        self.comments.iter_mut().find(|c| c.id == id)
    }

    /// Total number of comments in the store
    pub fn len(&self) -> usize {
        self.comments.len()
    }

    /// True when no comments are stored
    pub fn is_empty(&self) -> bool {
        self.comments.is_empty()
    }
}

impl Default for CommentStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_comment(id: u64, content: &str) -> Comment {
        Comment::new(CommentId(id), "alice", content, id * 1000)
    }

    #[test]
    fn test_add_and_get_comment() {
        let mut store = CommentStore::new();
        let c = make_comment(1, "Great shot!");
        store.add_comment(c);
        assert!(store.get(CommentId(1)).is_some());
    }

    #[test]
    fn test_delete_comment() {
        let mut store = CommentStore::new();
        store.add_comment(make_comment(1, "hello"));
        assert!(store.delete_comment(CommentId(1)));
        assert!(store.get(CommentId(1)).is_none());
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut store = CommentStore::new();
        assert!(!store.delete_comment(CommentId(99)));
    }

    #[test]
    fn test_resolve_comment() {
        let mut store = CommentStore::new();
        store.add_comment(make_comment(1, "fix this"));
        assert!(store.resolve(CommentId(1)));
        assert!(
            store
                .get(CommentId(1))
                .expect("collab test operation should succeed")
                .resolved
        );
    }

    #[test]
    fn test_search() {
        let mut store = CommentStore::new();
        store.add_comment(make_comment(1, "color correction needed"));
        store.add_comment(make_comment(2, "audio is too loud"));
        store.add_comment(make_comment(3, "Color grading looks great"));
        let results = store.search("color");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_reaction_add_remove() {
        let mut c = make_comment(1, "nice");
        c.add_reaction("👍", "alice");
        c.add_reaction("👍", "bob");
        assert_eq!(c.reactions["👍"].len(), 2);
        c.remove_reaction("👍", "alice");
        assert_eq!(c.reactions["👍"].len(), 1);
        c.remove_reaction("👍", "bob");
        assert!(!c.reactions.contains_key("👍"));
    }

    #[test]
    fn test_reaction_idempotent() {
        let mut c = make_comment(1, "cool");
        c.add_reaction("❤️", "alice");
        c.add_reaction("❤️", "alice");
        assert_eq!(c.reactions["❤️"].len(), 1);
    }

    #[test]
    fn test_thread_replies() {
        let root = make_comment(1, "Root comment");
        let mut thread = CommentThread::new(root);
        let reply = Comment::reply(CommentId(2), CommentId(1), "bob", "I agree!", 2000);
        thread.add(reply);
        let replies = thread.replies_to(CommentId(1));
        assert_eq!(replies.len(), 1);
        assert_eq!(replies[0].author, "bob");
    }

    #[test]
    fn test_thread_resolve() {
        let root = make_comment(1, "Issue");
        let mut thread = CommentThread::new(root);
        let reply = Comment::reply(CommentId(2), CommentId(1), "bob", "Fixed!", 2000);
        thread.add(reply);
        assert!(!thread.is_resolved());
        thread.resolve_thread();
        assert!(thread.is_resolved());
    }

    #[test]
    fn test_thread_root_comment() {
        let root = make_comment(5, "Root");
        let thread = CommentThread::new(root);
        assert_eq!(
            thread
                .root_comment()
                .expect("collab test operation should succeed")
                .id,
            CommentId(5)
        );
    }

    #[test]
    fn test_comment_anchor() {
        let anchor = CommentAnchor::new(120, "00:00:05:00").on_track("video-1");
        assert_eq!(anchor.frame, 120);
        assert_eq!(anchor.track_id.as_deref(), Some("video-1"));
        assert_eq!(anchor.timecode, "00:00:05:00");
    }

    #[test]
    fn test_next_id_increments() {
        let mut store = CommentStore::new();
        let id1 = store.next_id();
        let id2 = store.next_id();
        assert_ne!(id1, id2);
        assert_eq!(id2.0, id1.0 + 1);
    }
}
