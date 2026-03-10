//! User presence and activity tracking for collaborative sessions.
//!
//! Tracks online/away/busy/offline state, cursor positions, and user activity
//! broadcasts with a ring buffer of the last 50 broadcasts.

use std::collections::VecDeque;

/// Online status of a user
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum PresenceState {
    Online,
    Away,
    Busy,
    Offline,
}

impl PresenceState {
    /// True when the user is considered active (Online or Busy)
    pub fn is_active(self) -> bool {
        matches!(self, PresenceState::Online | PresenceState::Busy)
    }
}

/// Presence record for a single user
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct UserPresence {
    pub user_id: String,
    pub display_name: String,
    pub avatar_color: [u8; 3],
    pub cursor_position: Option<(f64, f64)>,
    pub last_seen_ms: u64,
    state: PresenceState,
}

impl UserPresence {
    /// Create a new UserPresence in the Online state
    pub fn new(
        user_id: impl Into<String>,
        display_name: impl Into<String>,
        avatar_color: [u8; 3],
        now_ms: u64,
    ) -> Self {
        Self {
            user_id: user_id.into(),
            display_name: display_name.into(),
            avatar_color,
            cursor_position: None,
            last_seen_ms: now_ms,
            state: PresenceState::Online,
        }
    }

    /// Current presence state
    pub fn state(&self) -> PresenceState {
        self.state
    }

    /// True when this user is actively online or busy
    pub fn is_active(&self) -> bool {
        self.state.is_active()
    }
}

/// Manages presence for all users in a session
pub struct PresenceManager {
    users: Vec<UserPresence>,
}

impl PresenceManager {
    /// Create an empty PresenceManager
    pub fn new() -> Self {
        Self { users: Vec::new() }
    }

    /// Register a user if not already present, or update if existing
    pub fn register_user(
        &mut self,
        user_id: &str,
        display_name: &str,
        avatar_color: [u8; 3],
        now_ms: u64,
    ) {
        if !self.users.iter().any(|u| u.user_id == user_id) {
            self.users.push(UserPresence::new(
                user_id,
                display_name,
                avatar_color,
                now_ms,
            ));
        }
    }

    /// Update the presence state of a user
    pub fn update_presence(&mut self, user_id: &str, state: PresenceState) {
        if let Some(u) = self.users.iter_mut().find(|u| u.user_id == user_id) {
            u.state = state;
        }
    }

    /// Update the cursor position of a user
    pub fn update_cursor(&mut self, user_id: &str, x: f64, y: f64) {
        if let Some(u) = self.users.iter_mut().find(|u| u.user_id == user_id) {
            u.cursor_position = Some((x, y));
        }
    }

    /// Return all users that are currently active (Online or Busy)
    pub fn active_users(&self) -> Vec<&UserPresence> {
        self.users.iter().filter(|u| u.is_active()).collect()
    }

    /// Return the presence record for a user, if registered
    pub fn get_user(&self, user_id: &str) -> Option<&UserPresence> {
        self.users.iter().find(|u| u.user_id == user_id)
    }

    /// Total number of registered users
    pub fn user_count(&self) -> usize {
        self.users.len()
    }
}

impl Default for PresenceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Actions a user can perform, broadcast to other participants
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum UserAction {
    Typing,
    Selecting(String),
    Editing(String),
    Viewing,
    Commenting,
}

impl UserAction {
    /// Human-readable description of the action
    pub fn description(&self) -> String {
        match self {
            UserAction::Typing => "is typing".to_string(),
            UserAction::Selecting(what) => format!("is selecting {}", what),
            UserAction::Editing(what) => format!("is editing {}", what),
            UserAction::Viewing => "is viewing".to_string(),
            UserAction::Commenting => "is commenting".to_string(),
        }
    }
}

/// A single activity broadcast from a user
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ActivityBroadcast {
    pub user_id: String,
    pub action: UserAction,
    pub timestamp_ms: u64,
}

impl ActivityBroadcast {
    /// Create a new broadcast
    pub fn new(user_id: impl Into<String>, action: UserAction, timestamp_ms: u64) -> Self {
        Self {
            user_id: user_id.into(),
            action,
            timestamp_ms,
        }
    }
}

/// Ring buffer channel holding up to the last 50 activity broadcasts
pub struct PresenceChannel {
    buffer: VecDeque<ActivityBroadcast>,
    capacity: usize,
}

impl PresenceChannel {
    /// Create a channel with the default capacity of 50
    pub fn new() -> Self {
        Self {
            buffer: VecDeque::new(),
            capacity: 50,
        }
    }

    /// Create a channel with a custom capacity (useful for testing)
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::new(),
            capacity,
        }
    }

    /// Push a broadcast, evicting the oldest when at capacity
    pub fn push(&mut self, broadcast: ActivityBroadcast) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(broadcast);
    }

    /// Return all broadcasts in chronological order
    pub fn broadcasts(&self) -> Vec<&ActivityBroadcast> {
        self.buffer.iter().collect()
    }

    /// Current number of broadcasts in the ring buffer
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// True when the ring buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

impl Default for PresenceChannel {
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

    #[test]
    fn test_presence_state_is_active() {
        assert!(PresenceState::Online.is_active());
        assert!(PresenceState::Busy.is_active());
        assert!(!PresenceState::Away.is_active());
        assert!(!PresenceState::Offline.is_active());
    }

    #[test]
    fn test_register_and_get_user() {
        let mut mgr = PresenceManager::new();
        mgr.register_user("alice", "Alice", [255, 0, 0], 1_000);
        let u = mgr
            .get_user("alice")
            .expect("collab test operation should succeed");
        assert_eq!(u.display_name, "Alice");
        assert_eq!(u.avatar_color, [255, 0, 0]);
    }

    #[test]
    fn test_register_twice_does_not_duplicate() {
        let mut mgr = PresenceManager::new();
        mgr.register_user("alice", "Alice", [0; 3], 0);
        mgr.register_user("alice", "Alice2", [1; 3], 1); // should be ignored
        assert_eq!(mgr.user_count(), 1);
    }

    #[test]
    fn test_update_presence() {
        let mut mgr = PresenceManager::new();
        mgr.register_user("bob", "Bob", [0; 3], 0);
        mgr.update_presence("bob", PresenceState::Away);
        assert_eq!(
            mgr.get_user("bob")
                .expect("collab test operation should succeed")
                .state(),
            PresenceState::Away
        );
    }

    #[test]
    fn test_active_users() {
        let mut mgr = PresenceManager::new();
        mgr.register_user("a", "A", [0; 3], 0);
        mgr.register_user("b", "B", [0; 3], 0);
        mgr.update_presence("b", PresenceState::Offline);
        let active = mgr.active_users();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].user_id, "a");
    }

    #[test]
    fn test_update_cursor() {
        let mut mgr = PresenceManager::new();
        mgr.register_user("c", "C", [0; 3], 0);
        mgr.update_cursor("c", 10.5, 20.3);
        let pos = mgr
            .get_user("c")
            .expect("collab test operation should succeed")
            .cursor_position
            .expect("collab test operation should succeed");
        assert!((pos.0 - 10.5).abs() < 1e-9);
        assert!((pos.1 - 20.3).abs() < 1e-9);
    }

    #[test]
    fn test_user_action_description() {
        assert_eq!(UserAction::Typing.description(), "is typing");
        assert_eq!(UserAction::Viewing.description(), "is viewing");
        assert_eq!(UserAction::Commenting.description(), "is commenting");
        assert!(UserAction::Selecting("clip-1".into())
            .description()
            .contains("clip-1"));
        assert!(UserAction::Editing("track-2".into())
            .description()
            .contains("track-2"));
    }

    #[test]
    fn test_presence_channel_push_and_read() {
        let mut ch = PresenceChannel::new();
        ch.push(ActivityBroadcast::new("alice", UserAction::Typing, 1_000));
        ch.push(ActivityBroadcast::new("bob", UserAction::Viewing, 2_000));
        assert_eq!(ch.len(), 2);
        let broadcasts = ch.broadcasts();
        assert_eq!(broadcasts[0].user_id, "alice");
        assert_eq!(broadcasts[1].user_id, "bob");
    }

    #[test]
    fn test_presence_channel_evicts_oldest() {
        let mut ch = PresenceChannel::with_capacity(3);
        for i in 0u64..4 {
            ch.push(ActivityBroadcast::new(
                format!("user-{}", i),
                UserAction::Viewing,
                i * 1000,
            ));
        }
        assert_eq!(ch.len(), 3);
        // user-0 should have been evicted
        assert!(!ch.broadcasts().iter().any(|b| b.user_id == "user-0"));
        assert!(ch.broadcasts().iter().any(|b| b.user_id == "user-3"));
    }

    #[test]
    fn test_presence_channel_empty() {
        let ch = PresenceChannel::new();
        assert!(ch.is_empty());
        assert_eq!(ch.len(), 0);
    }
}
