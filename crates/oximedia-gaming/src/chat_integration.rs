#![allow(dead_code)]
//! Chat integration system for game streaming.
//!
//! Provides a unified chat message model, command parsing, moderation
//! primitives, and emote tracking for live game streams.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Unique identifier for a chat user.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChatUserId(pub String);

/// A single chat message received during a live stream.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Unique message identifier.
    pub id: u64,
    /// Author of the message.
    pub user: ChatUserId,
    /// Display name shown in chat.
    pub display_name: String,
    /// Raw text content.
    pub text: String,
    /// Role of the user in the channel.
    pub role: UserRole,
    /// Timestamp relative to stream start.
    pub timestamp: Duration,
    /// Whether the message has been deleted by moderation.
    pub deleted: bool,
}

/// Role a chat user holds in the channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UserRole {
    /// Regular viewer.
    Viewer,
    /// Subscriber.
    Subscriber,
    /// VIP badge holder.
    Vip,
    /// Channel moderator.
    Moderator,
    /// Channel owner / broadcaster.
    Broadcaster,
}

/// A parsed chat command (messages starting with `!`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatCommand {
    /// The command name (without the leading `!`).
    pub name: String,
    /// Arguments following the command name.
    pub args: Vec<String>,
}

/// Result of a moderation check on a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModerationAction {
    /// Message is fine.
    Allow,
    /// Message should be deleted / filtered.
    Delete,
    /// User should be timed-out for the given seconds.
    Timeout(u32),
    /// User should be permanently banned.
    Ban,
}

/// Configuration for the chat moderation filter.
#[derive(Debug, Clone)]
pub struct ModerationConfig {
    /// Maximum message length before it is flagged.
    pub max_message_length: usize,
    /// Maximum number of messages a user can send in the rate window.
    pub rate_limit_count: u32,
    /// Rate limit sliding window duration.
    pub rate_limit_window: Duration,
    /// Banned words / phrases (case-insensitive check).
    pub banned_phrases: Vec<String>,
    /// Whether links are allowed.
    pub allow_links: bool,
}

impl Default for ModerationConfig {
    fn default() -> Self {
        Self {
            max_message_length: 500,
            rate_limit_count: 20,
            rate_limit_window: Duration::from_secs(30),
            banned_phrases: Vec::new(),
            allow_links: false,
        }
    }
}

/// Lightweight emote usage tracker.
#[derive(Debug, Clone)]
pub struct EmoteTracker {
    counts: HashMap<String, u64>,
}

impl EmoteTracker {
    /// Create a new, empty emote tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
        }
    }

    /// Record one usage of the given emote name.
    pub fn record(&mut self, emote: &str) {
        *self.counts.entry(emote.to_string()).or_insert(0) += 1;
    }

    /// Return the total count for a specific emote.
    #[must_use]
    pub fn count(&self, emote: &str) -> u64 {
        self.counts.get(emote).copied().unwrap_or(0)
    }

    /// Return the top-N emotes by usage count.
    #[must_use]
    pub fn top_emotes(&self, n: usize) -> Vec<(String, u64)> {
        let mut entries: Vec<_> = self.counts.iter().map(|(k, &v)| (k.clone(), v)).collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(n);
        entries
    }

    /// Reset all counts to zero.
    pub fn reset(&mut self) {
        self.counts.clear();
    }
}

impl Default for EmoteTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-user rate-limiter state.
#[derive(Debug)]
struct UserRateState {
    timestamps: Vec<Instant>,
}

/// Chat moderation engine.
#[derive(Debug)]
pub struct ChatModerator {
    config: ModerationConfig,
    user_rates: HashMap<ChatUserId, UserRateState>,
}

impl ChatModerator {
    /// Create a new moderator with the given config.
    #[must_use]
    pub fn new(config: ModerationConfig) -> Self {
        Self {
            config,
            user_rates: HashMap::new(),
        }
    }

    /// Check a message against moderation rules.
    pub fn check(&mut self, msg: &ChatMessage) -> ModerationAction {
        // Broadcasters and moderators bypass checks.
        if msg.role == UserRole::Broadcaster || msg.role == UserRole::Moderator {
            return ModerationAction::Allow;
        }

        // Length check.
        if msg.text.len() > self.config.max_message_length {
            return ModerationAction::Delete;
        }

        // Banned-phrase check (case-insensitive).
        let lower = msg.text.to_lowercase();
        for phrase in &self.config.banned_phrases {
            if lower.contains(&phrase.to_lowercase()) {
                return ModerationAction::Timeout(600);
            }
        }

        // Link check.
        if !self.config.allow_links && (lower.contains("http://") || lower.contains("https://")) {
            return ModerationAction::Delete;
        }

        // Rate-limit check.
        let now = Instant::now();
        let state = self
            .user_rates
            .entry(msg.user.clone())
            .or_insert_with(|| UserRateState {
                timestamps: Vec::new(),
            });
        state
            .timestamps
            .retain(|t| now.duration_since(*t) < self.config.rate_limit_window);
        state.timestamps.push(now);
        if state.timestamps.len() > self.config.rate_limit_count as usize {
            return ModerationAction::Timeout(30);
        }

        ModerationAction::Allow
    }

    /// Reset all tracked rate-limit state.
    pub fn reset_rates(&mut self) {
        self.user_rates.clear();
    }
}

/// Parse a chat message into a command, if it starts with `!`.
///
/// Returns `None` if the text does not start with `!` or is just `!`.
#[must_use]
pub fn parse_command(text: &str) -> Option<ChatCommand> {
    let trimmed = text.trim();
    if !trimmed.starts_with('!') || trimmed.len() < 2 {
        return None;
    }
    let without_bang = &trimmed[1..];
    let mut parts = without_bang.split_whitespace();
    let name = parts.next()?.to_string();
    let args: Vec<String> = parts.map(String::from).collect();
    Some(ChatCommand { name, args })
}

/// Compute a simple chat-activity rate (messages per second) over a window.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn chat_rate(messages: &[ChatMessage], window: Duration) -> f64 {
    if messages.is_empty() || window.is_zero() {
        return 0.0;
    }
    let count = messages.iter().filter(|m| m.timestamp <= window).count();
    count as f64 / window.as_secs_f64()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_msg(id: u64, user: &str, text: &str, role: UserRole) -> ChatMessage {
        ChatMessage {
            id,
            user: ChatUserId(user.to_string()),
            display_name: user.to_string(),
            text: text.to_string(),
            role,
            timestamp: Duration::from_secs(id),
            deleted: false,
        }
    }

    #[test]
    fn test_parse_command_basic() {
        let cmd = parse_command("!uptime").expect("valid command parse");
        assert_eq!(cmd.name, "uptime");
        assert!(cmd.args.is_empty());
    }

    #[test]
    fn test_parse_command_with_args() {
        let cmd = parse_command("!ban user123 spamming").expect("valid command parse");
        assert_eq!(cmd.name, "ban");
        assert_eq!(cmd.args, vec!["user123", "spamming"]);
    }

    #[test]
    fn test_parse_command_no_bang() {
        assert!(parse_command("hello world").is_none());
    }

    #[test]
    fn test_parse_command_just_bang() {
        assert!(parse_command("!").is_none());
    }

    #[test]
    fn test_moderation_allow() {
        let config = ModerationConfig::default();
        let mut moderator = ChatModerator::new(config);
        let msg = make_msg(1, "alice", "Hello everyone!", UserRole::Viewer);
        assert_eq!(moderator.check(&msg), ModerationAction::Allow);
    }

    #[test]
    fn test_moderation_delete_long_message() {
        let config = ModerationConfig {
            max_message_length: 10,
            ..ModerationConfig::default()
        };
        let mut moderator = ChatModerator::new(config);
        let msg = make_msg(1, "bob", "this is way too long a message", UserRole::Viewer);
        assert_eq!(moderator.check(&msg), ModerationAction::Delete);
    }

    #[test]
    fn test_moderation_banned_phrase() {
        let config = ModerationConfig {
            banned_phrases: vec!["badword".to_string()],
            ..ModerationConfig::default()
        };
        let mut moderator = ChatModerator::new(config);
        let msg = make_msg(1, "carol", "you are a BADWORD user", UserRole::Viewer);
        assert_eq!(moderator.check(&msg), ModerationAction::Timeout(600));
    }

    #[test]
    fn test_moderation_links_blocked() {
        let config = ModerationConfig {
            allow_links: false,
            ..ModerationConfig::default()
        };
        let mut moderator = ChatModerator::new(config);
        let msg = make_msg(1, "dan", "check https://example.com", UserRole::Viewer);
        assert_eq!(moderator.check(&msg), ModerationAction::Delete);
    }

    #[test]
    fn test_moderation_broadcaster_bypass() {
        let config = ModerationConfig {
            max_message_length: 5,
            ..ModerationConfig::default()
        };
        let mut moderator = ChatModerator::new(config);
        let msg = make_msg(
            1,
            "streamer",
            "a very long message indeed",
            UserRole::Broadcaster,
        );
        assert_eq!(moderator.check(&msg), ModerationAction::Allow);
    }

    #[test]
    fn test_emote_tracker_record_and_count() {
        let mut tracker = EmoteTracker::new();
        tracker.record("Kappa");
        tracker.record("Kappa");
        tracker.record("PogChamp");
        assert_eq!(tracker.count("Kappa"), 2);
        assert_eq!(tracker.count("PogChamp"), 1);
        assert_eq!(tracker.count("LUL"), 0);
    }

    #[test]
    fn test_emote_tracker_top() {
        let mut tracker = EmoteTracker::new();
        for _ in 0..5 {
            tracker.record("Kappa");
        }
        for _ in 0..3 {
            tracker.record("PogChamp");
        }
        tracker.record("LUL");
        let top = tracker.top_emotes(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "Kappa");
        assert_eq!(top[0].1, 5);
    }

    #[test]
    fn test_emote_tracker_reset() {
        let mut tracker = EmoteTracker::new();
        tracker.record("Kappa");
        tracker.reset();
        assert_eq!(tracker.count("Kappa"), 0);
    }

    #[test]
    fn test_chat_rate_empty() {
        let rate = chat_rate(&[], Duration::from_secs(10));
        assert!((rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_chat_rate_some_messages() {
        let msgs = vec![
            make_msg(1, "a", "hi", UserRole::Viewer),
            make_msg(2, "b", "yo", UserRole::Viewer),
            make_msg(3, "c", "hey", UserRole::Viewer),
        ];
        let rate = chat_rate(&msgs, Duration::from_secs(10));
        assert!((rate - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_user_role_equality() {
        assert_eq!(UserRole::Viewer, UserRole::Viewer);
        assert_ne!(UserRole::Viewer, UserRole::Moderator);
    }

    #[test]
    fn test_moderation_config_default() {
        let cfg = ModerationConfig::default();
        assert_eq!(cfg.max_message_length, 500);
        assert!(!cfg.allow_links);
        assert!(cfg.banned_phrases.is_empty());
    }
}
