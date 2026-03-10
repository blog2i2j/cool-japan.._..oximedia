#![allow(dead_code)]
//! Screen reader accessibility support for media player interfaces.
//!
//! This module provides ARIA-like role definitions, live region announcements,
//! and semantic descriptions for media UI elements so that screen reader
//! software can present the interface effectively to blind users.

use std::collections::VecDeque;
use std::fmt;

/// ARIA-like role for a UI element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AriaRole {
    /// A clickable button.
    Button,
    /// A range slider (volume, seek).
    Slider,
    /// A progress bar.
    ProgressBar,
    /// A timer/clock display.
    Timer,
    /// A status indicator.
    Status,
    /// A region that receives live updates.
    LiveRegion,
    /// A group container.
    Group,
    /// A toolbar container.
    Toolbar,
    /// A dialog overlay.
    Dialog,
    /// A menu.
    Menu,
    /// A menu item.
    MenuItem,
    /// A checkbox toggle.
    Checkbox,
    /// A generic region.
    Region,
}

impl fmt::Display for AriaRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Button => write!(f, "button"),
            Self::Slider => write!(f, "slider"),
            Self::ProgressBar => write!(f, "progressbar"),
            Self::Timer => write!(f, "timer"),
            Self::Status => write!(f, "status"),
            Self::LiveRegion => write!(f, "log"),
            Self::Group => write!(f, "group"),
            Self::Toolbar => write!(f, "toolbar"),
            Self::Dialog => write!(f, "dialog"),
            Self::Menu => write!(f, "menu"),
            Self::MenuItem => write!(f, "menuitem"),
            Self::Checkbox => write!(f, "checkbox"),
            Self::Region => write!(f, "region"),
        }
    }
}

/// Politeness level for live region announcements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LivePoliteness {
    /// The update is not announced unless the user navigates to it.
    Off,
    /// The update is announced at the next graceful opportunity.
    Polite,
    /// The update is announced immediately, interrupting current speech.
    Assertive,
}

impl fmt::Display for LivePoliteness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Off => write!(f, "off"),
            Self::Polite => write!(f, "polite"),
            Self::Assertive => write!(f, "assertive"),
        }
    }
}

/// Describes a single UI element for screen reader consumption.
#[derive(Debug, Clone)]
pub struct AccessibleElement {
    /// Unique identifier for the element.
    pub id: String,
    /// ARIA-like role.
    pub role: AriaRole,
    /// Accessible label (what the screen reader reads).
    pub label: String,
    /// Optional description for additional context.
    pub description: Option<String>,
    /// Whether the element is currently disabled.
    pub disabled: bool,
    /// Whether the element is currently hidden from the accessibility tree.
    pub hidden: bool,
    /// Optional current value (for sliders, progress bars).
    pub value: Option<String>,
    /// Optional minimum value.
    pub value_min: Option<String>,
    /// Optional maximum value.
    pub value_max: Option<String>,
    /// Whether the element is in a pressed/checked state.
    pub pressed: Option<bool>,
}

impl AccessibleElement {
    /// Create a new accessible element.
    #[must_use]
    pub fn new(id: impl Into<String>, role: AriaRole, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            role,
            label: label.into(),
            description: None,
            disabled: false,
            hidden: false,
            value: None,
            value_min: None,
            value_max: None,
            pressed: None,
        }
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set disabled state.
    #[must_use]
    pub fn with_disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    /// Set hidden state.
    #[must_use]
    pub fn with_hidden(mut self, hidden: bool) -> Self {
        self.hidden = hidden;
        self
    }

    /// Set a current value (for sliders/progress).
    #[must_use]
    pub fn with_value(mut self, value: impl Into<String>) -> Self {
        self.value = Some(value.into());
        self
    }

    /// Set a value range (min/max).
    #[must_use]
    pub fn with_range(mut self, min: impl Into<String>, max: impl Into<String>) -> Self {
        self.value_min = Some(min.into());
        self.value_max = Some(max.into());
        self
    }

    /// Set pressed/checked state.
    #[must_use]
    pub fn with_pressed(mut self, pressed: bool) -> Self {
        self.pressed = Some(pressed);
        self
    }

    /// Generate the full screen reader announcement for this element.
    #[must_use]
    pub fn announce(&self) -> String {
        let mut parts = Vec::new();
        parts.push(self.label.clone());
        parts.push(self.role.to_string());

        if let Some(v) = &self.value {
            parts.push(format!("value {v}"));
        }
        if let Some(pressed) = self.pressed {
            parts.push(if pressed {
                "pressed".to_string()
            } else {
                "not pressed".to_string()
            });
        }
        if self.disabled {
            parts.push("disabled".to_string());
        }

        parts.join(", ")
    }
}

/// A live region announcement to be spoken by the screen reader.
#[derive(Debug, Clone)]
pub struct Announcement {
    /// The text to be spoken.
    pub text: String,
    /// The politeness level.
    pub politeness: LivePoliteness,
    /// Timestamp in milliseconds when the announcement was created.
    pub timestamp_ms: u64,
}

impl Announcement {
    /// Create a new polite announcement.
    #[must_use]
    pub fn polite(text: impl Into<String>, timestamp_ms: u64) -> Self {
        Self {
            text: text.into(),
            politeness: LivePoliteness::Polite,
            timestamp_ms,
        }
    }

    /// Create a new assertive announcement.
    #[must_use]
    pub fn assertive(text: impl Into<String>, timestamp_ms: u64) -> Self {
        Self {
            text: text.into(),
            politeness: LivePoliteness::Assertive,
            timestamp_ms,
        }
    }
}

/// Manages a queue of screen reader announcements.
#[derive(Debug, Clone)]
pub struct AnnouncementQueue {
    /// Queue of pending announcements.
    queue: VecDeque<Announcement>,
    /// Maximum queue size before old entries are dropped.
    max_size: usize,
}

impl AnnouncementQueue {
    /// Create a new announcement queue.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            max_size: max_size.max(1),
        }
    }

    /// Push a new announcement. Drops oldest if over capacity.
    pub fn push(&mut self, announcement: Announcement) {
        if self.queue.len() >= self.max_size {
            self.queue.pop_front();
        }
        self.queue.push_back(announcement);
    }

    /// Pop the next announcement to be spoken.
    pub fn pop(&mut self) -> Option<Announcement> {
        // Assertive announcements have priority
        if let Some(idx) = self
            .queue
            .iter()
            .position(|a| a.politeness == LivePoliteness::Assertive)
        {
            return self.queue.remove(idx);
        }
        self.queue.pop_front()
    }

    /// Get the number of pending announcements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Check whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Clear all pending announcements.
    pub fn clear(&mut self) {
        self.queue.clear();
    }
}

impl Default for AnnouncementQueue {
    fn default() -> Self {
        Self::new(50)
    }
}

/// Generates standard media player screen reader announcements.
#[derive(Debug)]
pub struct MediaAnnouncer;

impl MediaAnnouncer {
    /// Announce playback state change.
    #[must_use]
    pub fn playback_state(playing: bool, timestamp_ms: u64) -> Announcement {
        let text = if playing { "Playing" } else { "Paused" };
        Announcement::assertive(text, timestamp_ms)
    }

    /// Announce volume change.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn volume_change(level: u32, max_level: u32, timestamp_ms: u64) -> Announcement {
        let pct = if max_level == 0 {
            0
        } else {
            #[allow(clippy::cast_possible_truncation)]
            #[allow(clippy::cast_sign_loss)]
            {
                ((f64::from(level) / f64::from(max_level)) * 100.0).round() as u32
            }
        };
        Announcement::polite(format!("Volume {pct} percent"), timestamp_ms)
    }

    /// Announce current time position.
    #[must_use]
    pub fn time_position(
        current_seconds: u64,
        total_seconds: u64,
        timestamp_ms: u64,
    ) -> Announcement {
        let cur_min = current_seconds / 60;
        let cur_sec = current_seconds % 60;
        let tot_min = total_seconds / 60;
        let tot_sec = total_seconds % 60;
        Announcement::polite(
            format!("{cur_min}:{cur_sec:02} of {tot_min}:{tot_sec:02}"),
            timestamp_ms,
        )
    }

    /// Announce mute state.
    #[must_use]
    pub fn mute_state(muted: bool, timestamp_ms: u64) -> Announcement {
        let text = if muted { "Muted" } else { "Unmuted" };
        Announcement::assertive(text, timestamp_ms)
    }

    /// Announce caption toggle.
    #[must_use]
    pub fn caption_state(enabled: bool, timestamp_ms: u64) -> Announcement {
        let text = if enabled {
            "Captions on"
        } else {
            "Captions off"
        };
        Announcement::polite(text, timestamp_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aria_role_display() {
        assert_eq!(AriaRole::Button.to_string(), "button");
        assert_eq!(AriaRole::Slider.to_string(), "slider");
        assert_eq!(AriaRole::ProgressBar.to_string(), "progressbar");
        assert_eq!(AriaRole::Timer.to_string(), "timer");
    }

    #[test]
    fn test_live_politeness_display() {
        assert_eq!(LivePoliteness::Off.to_string(), "off");
        assert_eq!(LivePoliteness::Polite.to_string(), "polite");
        assert_eq!(LivePoliteness::Assertive.to_string(), "assertive");
    }

    #[test]
    fn test_accessible_element_basic() {
        let elem = AccessibleElement::new("play_btn", AriaRole::Button, "Play");
        assert_eq!(elem.id, "play_btn");
        assert_eq!(elem.role, AriaRole::Button);
        assert_eq!(elem.label, "Play");
        assert!(!elem.disabled);
        assert!(!elem.hidden);
    }

    #[test]
    fn test_accessible_element_announce() {
        let elem = AccessibleElement::new("play_btn", AriaRole::Button, "Play");
        let text = elem.announce();
        assert!(text.contains("Play"));
        assert!(text.contains("button"));
    }

    #[test]
    fn test_accessible_element_slider() {
        let elem = AccessibleElement::new("vol", AriaRole::Slider, "Volume")
            .with_value("75")
            .with_range("0", "100");
        let text = elem.announce();
        assert!(text.contains("Volume"));
        assert!(text.contains("slider"));
        assert!(text.contains("value 75"));
    }

    #[test]
    fn test_accessible_element_disabled() {
        let elem = AccessibleElement::new("btn", AriaRole::Button, "Save").with_disabled(true);
        let text = elem.announce();
        assert!(text.contains("disabled"));
    }

    #[test]
    fn test_accessible_element_pressed() {
        let elem = AccessibleElement::new("mute", AriaRole::Button, "Mute").with_pressed(true);
        let text = elem.announce();
        assert!(text.contains("pressed"));
    }

    #[test]
    fn test_announcement_polite() {
        let a = Announcement::polite("Now playing", 1000);
        assert_eq!(a.politeness, LivePoliteness::Polite);
        assert_eq!(a.text, "Now playing");
    }

    #[test]
    fn test_announcement_assertive() {
        let a = Announcement::assertive("Error occurred", 2000);
        assert_eq!(a.politeness, LivePoliteness::Assertive);
    }

    #[test]
    fn test_queue_push_pop() {
        let mut q = AnnouncementQueue::new(10);
        q.push(Announcement::polite("Hello", 100));
        q.push(Announcement::polite("World", 200));
        assert_eq!(q.len(), 2);
        let a = q.pop().expect("a should be valid");
        assert_eq!(a.text, "Hello");
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_queue_assertive_priority() {
        let mut q = AnnouncementQueue::new(10);
        q.push(Announcement::polite("Low priority", 100));
        q.push(Announcement::assertive("High priority", 200));
        q.push(Announcement::polite("Also low", 300));
        let a = q.pop().expect("a should be valid");
        assert_eq!(a.text, "High priority");
    }

    #[test]
    fn test_queue_max_size() {
        let mut q = AnnouncementQueue::new(2);
        q.push(Announcement::polite("A", 100));
        q.push(Announcement::polite("B", 200));
        q.push(Announcement::polite("C", 300));
        // "A" should have been dropped
        assert_eq!(q.len(), 2);
        let first = q.pop().expect("first should be valid");
        assert_eq!(first.text, "B");
    }

    #[test]
    fn test_queue_clear() {
        let mut q = AnnouncementQueue::new(10);
        q.push(Announcement::polite("A", 100));
        q.clear();
        assert!(q.is_empty());
    }

    #[test]
    fn test_media_announcer_playback() {
        let a = MediaAnnouncer::playback_state(true, 0);
        assert_eq!(a.text, "Playing");
        assert_eq!(a.politeness, LivePoliteness::Assertive);

        let a = MediaAnnouncer::playback_state(false, 0);
        assert_eq!(a.text, "Paused");
    }

    #[test]
    fn test_media_announcer_volume() {
        let a = MediaAnnouncer::volume_change(50, 100, 0);
        assert_eq!(a.text, "Volume 50 percent");
        assert_eq!(a.politeness, LivePoliteness::Polite);
    }

    #[test]
    fn test_media_announcer_time() {
        let a = MediaAnnouncer::time_position(65, 120, 0);
        assert_eq!(a.text, "1:05 of 2:00");
    }

    #[test]
    fn test_media_announcer_mute() {
        let a = MediaAnnouncer::mute_state(true, 0);
        assert_eq!(a.text, "Muted");
        let a = MediaAnnouncer::mute_state(false, 0);
        assert_eq!(a.text, "Unmuted");
    }

    #[test]
    fn test_media_announcer_captions() {
        let a = MediaAnnouncer::caption_state(true, 0);
        assert_eq!(a.text, "Captions on");
        let a = MediaAnnouncer::caption_state(false, 0);
        assert_eq!(a.text, "Captions off");
    }
}
