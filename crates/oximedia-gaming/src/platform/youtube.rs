//! `YouTube` Live streaming integration.
//!
//! Provides stream configuration, live chat filtering, and adaptive bitrate support.

use crate::GamingResult;

/// `YouTube` Live stream configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct YoutubeStreamConfig {
    /// Stream title
    pub title: String,
    /// Stream description
    pub description: String,
    /// Privacy status
    pub privacy: PrivacyStatus,
    /// `YouTube` category ID (e.g. 20 = Gaming)
    pub category_id: u32,
}

/// `YouTube` privacy status for a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivacyStatus {
    /// Visible to everyone
    Public,
    /// Only accessible via direct link
    Unlisted,
    /// Visible only to the owner
    Private,
}

/// A live chat message from `YouTube` Live.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct YoutubeLiveChat {
    /// Unique message identifier
    pub message_id: String,
    /// Display name of the author
    pub author: String,
    /// Message text
    pub message: String,
    /// Unix timestamp in milliseconds
    pub timestamp_ms: u64,
}

/// Spam detection filter for `YouTube` Live chat.
pub struct YoutubeChatFilter {
    /// Fraction of repeated characters that triggers spam detection (0.0–1.0)
    repeated_char_threshold: f32,
    /// Fraction of uppercase characters that triggers spam detection (0.0–1.0)
    caps_threshold: f32,
    /// Known spam substrings
    spam_patterns: Vec<String>,
}

/// `YouTube` Live stream statistics.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct YoutubeStreamStats {
    /// Current concurrent viewer count
    pub concurrent_viewers: u64,
    /// Total chat messages received
    pub total_chat_messages: u64,
    /// Like count
    pub likes: u64,
    /// Dislike count (estimated / historical)
    pub dislikes: u64,
}

/// Video quality levels for adaptive bitrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoQuality {
    /// 144p
    P144,
    /// 240p
    P240,
    /// 360p
    P360,
    /// 480p
    P480,
    /// 720p
    P720,
    /// 1080p
    P1080,
    /// 1440p (2K)
    P1440,
    /// 2160p (4K)
    P2160,
}

/// Adaptive bitrate configuration for `YouTube` Live.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct YoutubeAdaptiveBitrate {
    /// Target video quality
    pub video_quality: VideoQuality,
    /// Audio bitrate in kbps
    pub audio_quality: u32,
}

/// `YouTube` Gaming integration.
pub struct YouTubeIntegration {
    config: YouTubeConfig,
}

/// `YouTube` configuration (legacy simple form).
#[derive(Debug, Clone)]
pub struct YouTubeConfig {
    /// Stream key
    pub stream_key: String,
    /// Stream title
    pub title: String,
    /// Description
    pub description: String,
    /// Privacy level
    pub privacy: String,
}

// ── implementations ─────────────────────────────────────────────────────────

impl YoutubeStreamConfig {
    /// Create a new stream config.
    #[must_use]
    pub fn new(
        title: impl Into<String>,
        description: impl Into<String>,
        privacy: PrivacyStatus,
        category_id: u32,
    ) -> Self {
        Self {
            title: title.into(),
            description: description.into(),
            privacy,
            category_id,
        }
    }
}

impl VideoQuality {
    /// Return the vertical resolution in pixels.
    #[must_use]
    pub fn resolution_p(&self) -> u32 {
        match self {
            Self::P144 => 144,
            Self::P240 => 240,
            Self::P360 => 360,
            Self::P480 => 480,
            Self::P720 => 720,
            Self::P1080 => 1080,
            Self::P1440 => 1440,
            Self::P2160 => 2160,
        }
    }

    /// Suggested video bitrate in kbps for this quality level.
    #[must_use]
    pub fn suggested_bitrate_kbps(&self) -> u32 {
        match self {
            Self::P144 => 400,
            Self::P240 => 700,
            Self::P360 => 1000,
            Self::P480 => 2500,
            Self::P720 => 5000,
            Self::P1080 => 8000,
            Self::P1440 => 16000,
            Self::P2160 => 35000,
        }
    }
}

impl YoutubeAdaptiveBitrate {
    /// Create a new adaptive bitrate config.
    #[must_use]
    pub fn new(video_quality: VideoQuality, audio_quality: u32) -> Self {
        Self {
            video_quality,
            audio_quality,
        }
    }

    /// Total bitrate (video + audio) in kbps.
    #[must_use]
    pub fn total_bitrate_kbps(&self) -> u32 {
        self.video_quality.suggested_bitrate_kbps() + self.audio_quality
    }
}

impl Default for YoutubeChatFilter {
    fn default() -> Self {
        Self {
            repeated_char_threshold: 0.50,
            caps_threshold: 0.80,
            spam_patterns: vec![
                "FREE SUBS".to_string(),
                "sub4sub".to_string(),
                "bit.ly".to_string(),
                "tinyurl".to_string(),
                "FREE GIFT".to_string(),
            ],
        }
    }
}

impl YoutubeChatFilter {
    /// Create a new filter with custom thresholds.
    #[must_use]
    pub fn new(repeated_char_threshold: f32, caps_threshold: f32) -> Self {
        Self {
            repeated_char_threshold,
            caps_threshold,
            spam_patterns: Self::default().spam_patterns,
        }
    }

    /// Add a custom spam pattern.
    pub fn add_pattern(&mut self, pattern: impl Into<String>) {
        self.spam_patterns.push(pattern.into());
    }

    /// Filter spam from a slice of messages.
    ///
    /// Returns the **indices** of messages that are classified as spam.
    #[must_use]
    pub fn filter_spam(&self, messages: &[YoutubeLiveChat]) -> Vec<usize> {
        messages
            .iter()
            .enumerate()
            .filter_map(|(i, m)| {
                if self.is_spam(&m.message) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return `true` if the given text is considered spam.
    #[must_use]
    pub fn is_spam(&self, text: &str) -> bool {
        if text.is_empty() {
            return false;
        }

        // 1. Repeated character check: count the most frequent character
        let char_count = text.chars().count();
        let max_freq = {
            let mut counts = std::collections::HashMap::new();
            for ch in text.chars() {
                if !ch.is_whitespace() {
                    *counts.entry(ch).or_insert(0u32) += 1;
                }
            }
            counts.values().copied().max().unwrap_or(0)
        };
        if char_count > 0 && (max_freq as f32 / char_count as f32) > self.repeated_char_threshold {
            return true;
        }

        // 2. All-caps check (only for messages with enough alphabetic chars)
        let alpha_chars: Vec<char> = text.chars().filter(|c| c.is_alphabetic()).collect();
        if alpha_chars.len() >= 4 {
            let upper_count = alpha_chars.iter().filter(|c| c.is_uppercase()).count();
            if (upper_count as f32 / alpha_chars.len() as f32) > self.caps_threshold {
                return true;
            }
        }

        // 3. Known spam pattern check (case-insensitive)
        let lower = text.to_lowercase();
        for pattern in &self.spam_patterns {
            if lower.contains(&pattern.to_lowercase()) {
                return true;
            }
        }

        false
    }
}

impl YouTubeIntegration {
    /// Create a new `YouTube` integration.
    #[must_use]
    pub fn new(config: YouTubeConfig) -> Self {
        Self { config }
    }

    /// Update stream title.
    pub fn update_title(&mut self, title: String) -> GamingResult<()> {
        self.config.title = title;
        Ok(())
    }

    /// Update description.
    pub fn update_description(&mut self, description: String) -> GamingResult<()> {
        self.config.description = description;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_msg(id: &str, author: &str, text: &str) -> YoutubeLiveChat {
        YoutubeLiveChat {
            message_id: id.to_string(),
            author: author.to_string(),
            message: text.to_string(),
            timestamp_ms: 0,
        }
    }

    // --- YoutubeChatFilter tests ---

    #[test]
    fn test_filter_spam_clean_messages() {
        let filter = YoutubeChatFilter::default();
        let messages = vec![
            make_msg("1", "Alice", "Great stream!"),
            make_msg("2", "Bob", "Love the gameplay"),
        ];
        let spam_indices = filter.filter_spam(&messages);
        assert!(spam_indices.is_empty());
    }

    #[test]
    fn test_filter_spam_known_pattern() {
        let filter = YoutubeChatFilter::default();
        let messages = vec![make_msg("1", "Spammer", "FREE SUBS click here")];
        let spam = filter.filter_spam(&messages);
        assert_eq!(spam, vec![0]);
    }

    #[test]
    fn test_filter_spam_all_caps() {
        let filter = YoutubeChatFilter::default();
        let messages = vec![make_msg("1", "Shouter", "WATCH THIS NOW CLICK HERE")];
        let spam = filter.filter_spam(&messages);
        assert_eq!(spam, vec![0]);
    }

    #[test]
    fn test_filter_repeated_chars() {
        let filter = YoutubeChatFilter::default();
        // "aaaaaaaaaa" -> same char > 50% of total
        let messages = vec![make_msg("1", "Bot", "aaaaaaaaaa")];
        let spam = filter.filter_spam(&messages);
        assert_eq!(spam, vec![0]);
    }

    #[test]
    fn test_filter_empty_message() {
        let filter = YoutubeChatFilter::default();
        assert!(!filter.is_spam(""));
    }

    #[test]
    fn test_filter_custom_pattern() {
        let mut filter = YoutubeChatFilter::default();
        filter.add_pattern("my_spam");
        assert!(filter.is_spam("check out my_spam dot com"));
    }

    #[test]
    fn test_mixed_batch() {
        let filter = YoutubeChatFilter::default();
        let messages = vec![
            make_msg("1", "Alice", "Nice play!"),
            make_msg("2", "Bot1", "FREE GIFT click now"),
            make_msg("3", "Charlie", "Good game, well played!"),
            make_msg("4", "Bot2", "bit.ly/scam"),
        ];
        let spam = filter.filter_spam(&messages);
        assert_eq!(spam, vec![1, 3]);
    }

    // --- VideoQuality tests ---

    #[test]
    fn test_video_quality_resolutions() {
        assert_eq!(VideoQuality::P720.resolution_p(), 720);
        assert_eq!(VideoQuality::P2160.resolution_p(), 2160);
        assert_eq!(VideoQuality::P144.resolution_p(), 144);
    }

    #[test]
    fn test_video_quality_bitrates_ordered() {
        // Higher quality should always have higher suggested bitrate
        assert!(
            VideoQuality::P1080.suggested_bitrate_kbps()
                > VideoQuality::P720.suggested_bitrate_kbps()
        );
        assert!(
            VideoQuality::P2160.suggested_bitrate_kbps()
                > VideoQuality::P1080.suggested_bitrate_kbps()
        );
    }

    #[test]
    fn test_adaptive_bitrate_total() {
        let abr = YoutubeAdaptiveBitrate::new(VideoQuality::P1080, 192);
        assert_eq!(abr.total_bitrate_kbps(), 8000 + 192);
    }

    #[test]
    fn test_privacy_status_equality() {
        assert_eq!(PrivacyStatus::Public, PrivacyStatus::Public);
        assert_ne!(PrivacyStatus::Public, PrivacyStatus::Private);
    }

    #[test]
    fn test_stream_config_creation() {
        let cfg =
            YoutubeStreamConfig::new("My Stream", "A great stream", PrivacyStatus::Public, 20);
        assert_eq!(cfg.title, "My Stream");
        assert_eq!(cfg.privacy, PrivacyStatus::Public);
        assert_eq!(cfg.category_id, 20);
    }

    #[test]
    fn test_youtube_integration_update() {
        let config = YouTubeConfig {
            stream_key: "test_key".to_string(),
            title: "Test Stream".to_string(),
            description: "Test Description".to_string(),
            privacy: "public".to_string(),
        };
        let mut integration = YouTubeIntegration::new(config);
        integration
            .update_title("New Title".to_string())
            .expect("update title should succeed");
        integration
            .update_description("New Desc".to_string())
            .expect("should succeed");
    }
}
