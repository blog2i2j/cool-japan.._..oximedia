//! Twitch streaming integration.
//!
//! Provides Twitch chat parsing, event subscriptions, and stream health monitoring.

use crate::GamingResult;

/// Twitch stream configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TwitchConfig {
    /// Channel name
    pub channel_name: String,
    /// Stream key
    pub stream_key: String,
    /// Stream category (game name)
    pub category: String,
    /// Stream language code (e.g. "en")
    pub language: String,
}

/// Live stream information from Twitch API.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct TwitchStreamInfo {
    /// Current viewer count
    pub viewer_count: u64,
    /// Total follower count
    pub follower_count: u64,
    /// Whether the channel is currently live
    pub is_live: bool,
    /// Stream uptime in seconds
    pub uptime_secs: u64,
}

/// A parsed Twitch IRC chat message.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TwitchChatMessage {
    /// Username of the sender
    pub username: String,
    /// Message text
    pub message: String,
    /// Badge list (e.g. ["broadcaster", "subscriber/6"])
    pub badges: Vec<String>,
    /// Hex color code (e.g. "#FF4500"), if set
    pub color: Option<String>,
}

/// Twitch `EventSub` event type.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TwitchEventType {
    /// New follower
    Follow,
    /// New subscriber
    Subscribe,
    /// Bits cheer
    Cheer,
    /// Raid from another channel
    Raid,
    /// Channel point redemption
    ChannelPointRedeem,
    /// Poll ended
    PollEnd,
}

/// A Twitch `EventSub` webhook event.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TwitchEventSub {
    /// Type of the event
    pub event_type: TwitchEventType,
    /// Raw event data as key-value pairs
    pub data: std::collections::HashMap<String, String>,
}

/// Stream health monitor tracking encoder metrics.
#[derive(Debug, Clone, Default)]
pub struct StreamHealthMonitor {
    /// Number of dropped frames since last reset
    pub frames_dropped: u64,
    /// Current bitrate in kbps
    pub bitrate_kbps: f32,
    /// Maximum acceptable dropped-frame percentage (0.0–1.0)
    dropped_threshold: f32,
    /// Total frames rendered (used to compute drop ratio)
    total_frames: u64,
}

/// Twitch integration for stream metadata.
pub struct TwitchIntegration {
    config: TwitchConfig,
}

impl TwitchConfig {
    /// Create a new `TwitchConfig`.
    #[must_use]
    pub fn new(
        channel_name: impl Into<String>,
        stream_key: impl Into<String>,
        category: impl Into<String>,
        language: impl Into<String>,
    ) -> Self {
        Self {
            channel_name: channel_name.into(),
            stream_key: stream_key.into(),
            category: category.into(),
            language: language.into(),
        }
    }
}

/// IRC chat message parser for Twitch messages.
pub struct TwitchChatParser;

impl TwitchChatParser {
    /// Parse a raw `IRCv3` Twitch message line into a [`TwitchChatMessage`].
    ///
    /// Handles the `@badge-info=...;badges=...;color=...;display-name=...` tag prefix.
    ///
    /// Returns `None` if the line is not a recognised PRIVMSG.
    #[must_use]
    pub fn parse_irc_message(line: &str) -> Option<TwitchChatMessage> {
        // Expect format: @tags :user!user@user.tmi.twitch.tv PRIVMSG #channel :message
        let line = line.trim();
        if line.is_empty() {
            return None;
        }

        // Split off leading IRCv3 tag block
        let (tags_str, rest) = if let Some(stripped) = line.strip_prefix('@') {
            // Find the space that ends the tag block
            let pos = stripped.find(' ')?;
            (&stripped[..pos], stripped[pos + 1..].trim_start())
        } else {
            ("", line)
        };

        // Verify this is a PRIVMSG
        if !rest.contains("PRIVMSG") {
            return None;
        }

        // Parse username from the source prefix `:user!user@...`
        let username = if let Some(source) = rest.strip_prefix(':') {
            let end = source.find('!')?;
            source[..end].to_string()
        } else {
            return None;
        };

        // Extract message body after the final `:`
        let message = {
            // Find the PRIVMSG portion
            let privmsg_pos = rest.find("PRIVMSG")?;
            let after_privmsg = &rest[privmsg_pos + "PRIVMSG".len()..];
            // Skip channel and find the colon-prefixed message
            let colon_pos = after_privmsg.find(':')?;
            after_privmsg[colon_pos + 1..].to_string()
        };

        // Parse IRCv3 tags
        let mut badges: Vec<String> = Vec::new();
        let mut color: Option<String> = None;
        let mut display_name = username.clone();

        for tag in tags_str.split(';') {
            if let Some((key, value)) = tag.split_once('=') {
                match key {
                    "badges" => {
                        if !value.is_empty() {
                            badges = value.split(',').map(String::from).collect();
                        }
                    }
                    "color" => {
                        if !value.is_empty() {
                            color = Some(value.to_string());
                        }
                    }
                    "display-name" => {
                        if !value.is_empty() {
                            display_name = value.to_string();
                        }
                    }
                    _ => {}
                }
            }
        }

        Some(TwitchChatMessage {
            username: display_name,
            message,
            badges,
            color,
        })
    }
}

impl StreamHealthMonitor {
    /// Create a new `StreamHealthMonitor`.
    ///
    /// `dropped_threshold` is the maximum acceptable ratio of dropped frames
    /// (0.0–1.0, e.g. 0.05 means 5%).
    #[must_use]
    pub fn new(dropped_threshold: f32) -> Self {
        Self {
            frames_dropped: 0,
            bitrate_kbps: 0.0,
            dropped_threshold,
            total_frames: 0,
        }
    }

    /// Record a newly rendered frame, optionally flagging it as dropped.
    pub fn record_frame(&mut self, dropped: bool) {
        self.total_frames += 1;
        if dropped {
            self.frames_dropped += 1;
        }
    }

    /// Update the current measured bitrate.
    pub fn set_bitrate(&mut self, kbps: f32) {
        self.bitrate_kbps = kbps;
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        self.frames_dropped = 0;
        self.total_frames = 0;
        self.bitrate_kbps = 0.0;
    }

    /// Return `true` when the stream is considered healthy:
    /// - Drop ratio is below the configured threshold
    /// - Bitrate is positive
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        let drop_ratio = if self.total_frames == 0 {
            0.0_f32
        } else {
            self.frames_dropped as f32 / self.total_frames as f32
        };
        drop_ratio <= self.dropped_threshold && self.bitrate_kbps > 0.0
    }

    /// Return the current drop ratio (0.0–1.0).
    #[must_use]
    pub fn drop_ratio(&self) -> f32 {
        if self.total_frames == 0 {
            0.0
        } else {
            self.frames_dropped as f32 / self.total_frames as f32
        }
    }
}

impl TwitchIntegration {
    /// Create a new Twitch integration.
    #[must_use]
    pub fn new(config: TwitchConfig) -> Self {
        Self { config }
    }

    /// Update stream title (stored in category field for backward compat).
    pub fn update_title(&mut self, title: String) -> GamingResult<()> {
        self.config.category = title;
        Ok(())
    }

    /// Update game category.
    pub fn update_category(&mut self, category: String) -> GamingResult<()> {
        self.config.category = category;
        Ok(())
    }

    /// Get the channel name.
    #[must_use]
    pub fn channel_name(&self) -> &str {
        &self.config.channel_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- TwitchChatParser tests ---

    #[test]
    fn test_parse_basic_privmsg() {
        let line = "@badges=broadcaster/1;color=#FF4500;display-name=TestUser :testuser!testuser@testuser.tmi.twitch.tv PRIVMSG #channel :Hello world";
        let msg = TwitchChatParser::parse_irc_message(line).expect("valid IRC message");
        assert_eq!(msg.username, "TestUser");
        assert_eq!(msg.message, "Hello world");
        assert_eq!(msg.color, Some("#FF4500".to_string()));
        assert!(msg.badges.contains(&"broadcaster/1".to_string()));
    }

    #[test]
    fn test_parse_empty_line() {
        assert!(TwitchChatParser::parse_irc_message("").is_none());
    }

    #[test]
    fn test_parse_non_privmsg() {
        let line = ":tmi.twitch.tv 001 testuser :Welcome";
        assert!(TwitchChatParser::parse_irc_message(line).is_none());
    }

    #[test]
    fn test_parse_no_color() {
        let line = "@badges=subscriber/6;color=;display-name=SubUser :subuser!subuser@subuser.tmi.twitch.tv PRIVMSG #chan :Nice stream!";
        let msg = TwitchChatParser::parse_irc_message(line).expect("valid IRC message");
        assert_eq!(msg.username, "SubUser");
        assert!(msg.color.is_none());
    }

    #[test]
    fn test_parse_multiple_badges() {
        let line = "@badges=moderator/1,subscriber/12;color=#00FF7F;display-name=ModUser :moduser!moduser@moduser.tmi.twitch.tv PRIVMSG #chan :GG";
        let msg = TwitchChatParser::parse_irc_message(line).expect("valid IRC message");
        assert_eq!(msg.badges.len(), 2);
        assert!(msg.badges.contains(&"moderator/1".to_string()));
    }

    #[test]
    fn test_parse_no_badges() {
        let line = "@badges=;color=#0000FF;display-name=Viewer :viewer!viewer@viewer.tmi.twitch.tv PRIVMSG #chan :Hello";
        let msg = TwitchChatParser::parse_irc_message(line).expect("valid IRC message");
        assert!(msg.badges.is_empty());
    }

    // --- StreamHealthMonitor tests ---

    #[test]
    fn test_health_monitor_initially_unhealthy() {
        // No bitrate set yet
        let monitor = StreamHealthMonitor::new(0.05);
        assert!(!monitor.is_healthy());
    }

    #[test]
    fn test_health_monitor_healthy() {
        let mut monitor = StreamHealthMonitor::new(0.05);
        monitor.set_bitrate(6000.0);
        for _ in 0..100 {
            monitor.record_frame(false);
        }
        assert!(monitor.is_healthy());
    }

    #[test]
    fn test_health_monitor_too_many_drops() {
        let mut monitor = StreamHealthMonitor::new(0.05);
        monitor.set_bitrate(6000.0);
        for i in 0..100 {
            monitor.record_frame(i % 5 == 0); // 20% drop rate
        }
        assert!(!monitor.is_healthy());
    }

    #[test]
    fn test_health_monitor_reset() {
        let mut monitor = StreamHealthMonitor::new(0.05);
        monitor.set_bitrate(6000.0);
        monitor.record_frame(true);
        monitor.reset();
        assert_eq!(monitor.frames_dropped, 0);
        assert_eq!(monitor.bitrate_kbps, 0.0);
    }

    #[test]
    fn test_drop_ratio_empty() {
        let monitor = StreamHealthMonitor::new(0.05);
        assert_eq!(monitor.drop_ratio(), 0.0);
    }

    #[test]
    fn test_event_type_equality() {
        assert_eq!(TwitchEventType::Follow, TwitchEventType::Follow);
        assert_ne!(TwitchEventType::Cheer, TwitchEventType::Raid);
    }

    #[test]
    fn test_twitch_config_new() {
        let cfg = TwitchConfig::new("my_channel", "sk-live-xxx", "Fortnite", "en");
        assert_eq!(cfg.channel_name, "my_channel");
        assert_eq!(cfg.language, "en");
    }

    #[test]
    fn test_twitch_integration_update_category() {
        let config = TwitchConfig {
            channel_name: "streamer".to_string(),
            stream_key: "test_key".to_string(),
            category: "Just Chatting".to_string(),
            language: "en".to_string(),
        };
        let mut integration = TwitchIntegration::new(config);
        integration
            .update_category("Fortnite".to_string())
            .expect("update category should succeed");
    }

    #[test]
    fn test_stream_health_exact_threshold() {
        let mut monitor = StreamHealthMonitor::new(0.1);
        monitor.set_bitrate(3000.0);
        // 10 frames, 1 dropped = exactly 10%
        for i in 0..10 {
            monitor.record_frame(i == 0);
        }
        assert!(monitor.is_healthy()); // equal to threshold, still healthy
    }
}
