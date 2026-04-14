//! Chat message overlay renderer for game streaming.
//!
//! Manages an incoming chat message queue with per-user rate limiting,
//! configurable display duration, screen-position anchoring, colour coding
//! per badge level, and a read-only snapshot of the currently visible
//! messages for the compositor to render.
//!
//! # Example
//!
//! ```rust
//! use oximedia_gaming::chat_overlay::{
//!     ChatOverlay, ChatOverlayConfig, ChatMessage, BadgeLevel, OverlayAnchor,
//! };
//! use std::time::{Duration, Instant};
//!
//! let cfg = ChatOverlayConfig::default();
//! let mut overlay = ChatOverlay::new(cfg);
//!
//! overlay.push_message(ChatMessage {
//!     id: 1,
//!     user: "viewer42".to_string(),
//!     text: "PogChamp!".to_string(),
//!     badge: BadgeLevel::Subscriber,
//!     color_rgb: [0x00, 0xFF, 0x80],
//!     received_at: Instant::now(),
//! });
//!
//! let visible = overlay.visible_messages(Instant::now());
//! println!("{} messages on screen", visible.len());
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Badge levels
// ---------------------------------------------------------------------------

/// Viewer badge / privilege level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BadgeLevel {
    /// Regular viewer with no subscription.
    Viewer = 0,
    /// Active subscriber.
    Subscriber = 1,
    /// Channel moderator.
    Moderator = 2,
    /// Channel VIP.
    Vip = 3,
    /// Channel owner / broadcaster.
    Broadcaster = 4,
}

impl Default for BadgeLevel {
    fn default() -> Self {
        Self::Viewer
    }
}

// ---------------------------------------------------------------------------
// Overlay position anchor
// ---------------------------------------------------------------------------

/// Anchor position for the chat overlay on the video frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayAnchor {
    /// Bottom-left corner (typical Twitch overlay position).
    BottomLeft,
    /// Bottom-right corner.
    BottomRight,
    /// Top-left corner.
    TopLeft,
    /// Top-right corner.
    TopRight,
    /// Custom pixel offset (x, y) from the top-left of the frame.
    Custom(u32, u32),
}

impl Default for OverlayAnchor {
    fn default() -> Self {
        Self::BottomLeft
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`ChatOverlay`].
#[derive(Debug, Clone)]
pub struct ChatOverlayConfig {
    /// Maximum number of messages visible on screen at once.
    pub max_visible: usize,
    /// How long each message stays on screen.
    pub display_duration: Duration,
    /// Maximum messages queued (older messages are dropped if full).
    pub queue_capacity: usize,
    /// Per-user rate limit: minimum gap between two accepted messages from the
    /// same user.
    pub per_user_rate_limit: Duration,
    /// Minimum badge level required to bypass per-user rate limiting.
    pub rate_limit_bypass_badge: BadgeLevel,
    /// Anchor point for the overlay.
    pub anchor: OverlayAnchor,
    /// Font size hint (pixels) for the renderer.
    pub font_size: u32,
    /// Whether to show badges visually.
    pub show_badges: bool,
    /// Whether to filter messages that are empty after whitespace trimming.
    pub filter_empty: bool,
}

impl Default for ChatOverlayConfig {
    fn default() -> Self {
        Self {
            max_visible: 8,
            display_duration: Duration::from_secs(8),
            queue_capacity: 256,
            per_user_rate_limit: Duration::from_secs(2),
            rate_limit_bypass_badge: BadgeLevel::Moderator,
            anchor: OverlayAnchor::BottomLeft,
            font_size: 24,
            show_badges: true,
            filter_empty: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Chat message
// ---------------------------------------------------------------------------

/// A single chat message to be displayed in the overlay.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Unique message identifier (e.g. from the platform API).
    pub id: u64,
    /// Sender's display name.
    pub user: String,
    /// Message body.
    pub text: String,
    /// Sender's badge level.
    pub badge: BadgeLevel,
    /// Preferred render colour as `[R, G, B]`.
    pub color_rgb: [u8; 3],
    /// Timestamp when the message was received locally.
    pub received_at: Instant,
}

/// A message that has passed the rate-limiter and is waiting in the display
/// queue.
#[derive(Debug, Clone)]
pub struct QueuedMessage {
    /// The original message.
    pub message: ChatMessage,
    /// When this message was admitted to the queue.
    pub queued_at: Instant,
}

// ---------------------------------------------------------------------------
// Rate-limiter state per user
// ---------------------------------------------------------------------------

struct UserState {
    last_admitted: Instant,
}

// ---------------------------------------------------------------------------
// Chat overlay
// ---------------------------------------------------------------------------

/// Chat overlay manager.
///
/// Accepts incoming [`ChatMessage`]s, enforces per-user rate limiting and
/// queue capacity limits, and exposes the set of messages that should
/// currently be visible on screen via [`visible_messages`][`Self::visible_messages`].
pub struct ChatOverlay {
    config: ChatOverlayConfig,
    /// Messages currently in the display queue (oldest first).
    queue: std::collections::VecDeque<QueuedMessage>,
    /// Per-user rate-limit state.
    user_states: HashMap<String, UserState>,
    /// Running count of messages admitted since creation.
    total_admitted: u64,
    /// Running count of messages rejected (rate-limited or queue-full).
    total_rejected: u64,
}

impl ChatOverlay {
    /// Create a new overlay manager with the supplied configuration.
    #[must_use]
    pub fn new(config: ChatOverlayConfig) -> Self {
        Self {
            config,
            queue: std::collections::VecDeque::new(),
            user_states: HashMap::new(),
            total_admitted: 0,
            total_rejected: 0,
        }
    }

    /// Attempt to add a message to the overlay queue.
    ///
    /// Returns `true` if the message was admitted, `false` if it was
    /// rejected (rate limited, empty after filtering, or queue full).
    pub fn push_message(&mut self, msg: ChatMessage) -> bool {
        // --- empty message filter ---
        if self.config.filter_empty && msg.text.trim().is_empty() {
            self.total_rejected += 1;
            return false;
        }

        // --- per-user rate limiting ---
        let now = msg.received_at;
        let bypass = msg.badge >= self.config.rate_limit_bypass_badge;
        if !bypass {
            if let Some(state) = self.user_states.get(&msg.user) {
                if now.duration_since(state.last_admitted) < self.config.per_user_rate_limit {
                    self.total_rejected += 1;
                    return false;
                }
            }
        }

        // --- queue capacity ---
        if self.queue.len() >= self.config.queue_capacity {
            // Drop the oldest message to make room
            self.queue.pop_front();
        }

        let user_key = msg.user.clone();
        let queued_at = now;
        self.queue.push_back(QueuedMessage {
            message: msg,
            queued_at,
        });
        self.user_states
            .insert(user_key, UserState { last_admitted: now });
        self.total_admitted += 1;
        true
    }

    /// Evict expired messages from the queue.
    ///
    /// Should be called periodically (e.g. once per frame) to keep the queue
    /// clean. Returns the number of messages evicted.
    pub fn evict_expired(&mut self, now: Instant) -> usize {
        let before = self.queue.len();
        self.queue
            .retain(|qm| now.duration_since(qm.queued_at) < self.config.display_duration);
        before - self.queue.len()
    }

    /// Returns a snapshot of the messages that should currently be rendered.
    ///
    /// Performs implicit expiry before returning. Messages are returned
    /// oldest-first, capped at [`ChatOverlayConfig::max_visible`].
    #[must_use]
    pub fn visible_messages(&mut self, now: Instant) -> Vec<&QueuedMessage> {
        self.evict_expired(now);
        let skip = self.queue.len().saturating_sub(self.config.max_visible);
        self.queue.iter().skip(skip).collect()
    }

    /// Peek at visible messages without mutating state (does not evict expired).
    ///
    /// Useful for read-only inspection in tests.
    #[must_use]
    pub fn peek_visible(&self, now: Instant) -> Vec<&QueuedMessage> {
        let not_expired: Vec<&QueuedMessage> = self
            .queue
            .iter()
            .filter(|qm| now.duration_since(qm.queued_at) < self.config.display_duration)
            .collect();
        let skip = not_expired.len().saturating_sub(self.config.max_visible);
        not_expired.into_iter().skip(skip).collect()
    }

    /// Total messages admitted since creation.
    #[must_use]
    pub fn total_admitted(&self) -> u64 {
        self.total_admitted
    }

    /// Total messages rejected since creation.
    #[must_use]
    pub fn total_rejected(&self) -> u64 {
        self.total_rejected
    }

    /// Current queue length (including messages that may have expired but
    /// haven't been evicted yet).
    #[must_use]
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Clear the entire queue.
    pub fn clear(&mut self) {
        self.queue.clear();
    }

    /// Access the current configuration.
    #[must_use]
    pub fn config(&self) -> &ChatOverlayConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: ChatOverlayConfig) {
        self.config = config;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn base_msg(id: u64, user: &str, text: &str, t: Instant) -> ChatMessage {
        ChatMessage {
            id,
            user: user.to_string(),
            text: text.to_string(),
            badge: BadgeLevel::Viewer,
            color_rgb: [0xFF, 0xFF, 0xFF],
            received_at: t,
        }
    }

    #[test]
    fn test_basic_message_admitted() {
        let mut overlay = ChatOverlay::new(ChatOverlayConfig::default());
        let now = Instant::now();
        let admitted = overlay.push_message(base_msg(1, "alice", "hello", now));
        assert!(admitted);
        assert_eq!(overlay.total_admitted(), 1);
        assert_eq!(overlay.total_rejected(), 0);
    }

    #[test]
    fn test_empty_message_rejected() {
        let mut overlay = ChatOverlay::new(ChatOverlayConfig::default());
        let now = Instant::now();
        let admitted = overlay.push_message(base_msg(1, "alice", "   ", now));
        assert!(!admitted);
        assert_eq!(overlay.total_rejected(), 1);
    }

    #[test]
    fn test_per_user_rate_limit() {
        let mut overlay = ChatOverlay::new(ChatOverlayConfig {
            per_user_rate_limit: Duration::from_secs(5),
            ..ChatOverlayConfig::default()
        });
        let now = Instant::now();
        // First message admitted
        assert!(overlay.push_message(base_msg(1, "bob", "msg1", now)));
        // Second message immediately after: should be rejected
        assert!(!overlay.push_message(base_msg(2, "bob", "msg2", now)));
        assert_eq!(overlay.total_rejected(), 1);
    }

    #[test]
    fn test_rate_limit_bypassed_for_moderator() {
        let mut overlay = ChatOverlay::new(ChatOverlayConfig {
            per_user_rate_limit: Duration::from_secs(100),
            rate_limit_bypass_badge: BadgeLevel::Moderator,
            ..ChatOverlayConfig::default()
        });
        let now = Instant::now();
        let mut mod_msg = base_msg(1, "mod1", "msg1", now);
        mod_msg.badge = BadgeLevel::Moderator;
        assert!(overlay.push_message(mod_msg.clone()));

        let mut mod_msg2 = base_msg(2, "mod1", "msg2", now);
        mod_msg2.badge = BadgeLevel::Moderator;
        // Should be admitted despite 0-elapsed time since bypass applies
        assert!(overlay.push_message(mod_msg2));
        assert_eq!(overlay.total_admitted(), 2);
    }

    #[test]
    fn test_visible_messages_capped_at_max() {
        let cfg = ChatOverlayConfig {
            max_visible: 3,
            display_duration: Duration::from_mins(1),
            per_user_rate_limit: Duration::ZERO,
            ..ChatOverlayConfig::default()
        };
        let mut overlay = ChatOverlay::new(cfg);
        let now = Instant::now();
        for i in 0..10u64 {
            overlay.push_message(base_msg(i, &format!("user{i}"), "hi", now));
        }
        let visible = overlay.visible_messages(now);
        assert_eq!(visible.len(), 3);
    }

    #[test]
    fn test_expired_messages_evicted() {
        let cfg = ChatOverlayConfig {
            display_duration: Duration::from_millis(50),
            ..ChatOverlayConfig::default()
        };
        let mut overlay = ChatOverlay::new(cfg);
        let now = Instant::now();
        overlay.push_message(base_msg(1, "alice", "old message", now));
        // Advance time well past display_duration
        let later = now + Duration::from_millis(200);
        let evicted = overlay.evict_expired(later);
        assert_eq!(evicted, 1);
        assert_eq!(overlay.queue_len(), 0);
    }

    #[test]
    fn test_queue_capacity_drops_oldest() {
        let cfg = ChatOverlayConfig {
            queue_capacity: 3,
            per_user_rate_limit: Duration::ZERO,
            display_duration: Duration::from_mins(1),
            ..ChatOverlayConfig::default()
        };
        let mut overlay = ChatOverlay::new(cfg);
        let now = Instant::now();
        for i in 0..5u64 {
            overlay.push_message(base_msg(i, &format!("u{i}"), "msg", now));
        }
        // Queue is capped at 3 (oldest two were dropped)
        assert_eq!(overlay.queue_len(), 3);
    }

    #[test]
    fn test_clear_empties_queue() {
        let mut overlay = ChatOverlay::new(ChatOverlayConfig::default());
        let now = Instant::now();
        overlay.push_message(base_msg(1, "alice", "hi", now));
        overlay.clear();
        assert_eq!(overlay.queue_len(), 0);
    }

    #[test]
    fn test_badge_level_ordering() {
        assert!(BadgeLevel::Broadcaster > BadgeLevel::Vip);
        assert!(BadgeLevel::Vip > BadgeLevel::Moderator);
        assert!(BadgeLevel::Moderator > BadgeLevel::Subscriber);
        assert!(BadgeLevel::Subscriber > BadgeLevel::Viewer);
    }

    #[test]
    fn test_multiple_users_independent_rate_limits() {
        let cfg = ChatOverlayConfig {
            per_user_rate_limit: Duration::from_secs(10),
            ..ChatOverlayConfig::default()
        };
        let mut overlay = ChatOverlay::new(cfg);
        let now = Instant::now();
        // alice sends
        assert!(overlay.push_message(base_msg(1, "alice", "hi", now)));
        // bob should not be rate-limited
        assert!(overlay.push_message(base_msg(2, "bob", "hey", now)));
        // alice tries again immediately: rejected
        assert!(!overlay.push_message(base_msg(3, "alice", "again", now)));
        assert_eq!(overlay.total_admitted(), 2);
        assert_eq!(overlay.total_rejected(), 1);
    }

    #[test]
    fn test_peek_visible_does_not_mutate() {
        let cfg = ChatOverlayConfig {
            display_duration: Duration::from_mins(1),
            per_user_rate_limit: Duration::ZERO,
            ..ChatOverlayConfig::default()
        };
        let mut overlay = ChatOverlay::new(cfg);
        let now = Instant::now();
        overlay.push_message(base_msg(1, "alice", "hi", now));
        let before_len = overlay.queue_len();
        let _ = overlay.peek_visible(now);
        assert_eq!(overlay.queue_len(), before_len);
    }
}
