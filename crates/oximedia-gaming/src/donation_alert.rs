//! Donation and subscription alert system for game streams.
//!
//! Provides a priority-ordered alert queue, TTS text preparation, display
//! duration calculation, and deduplication.  Alerts can originate from
//! donations, subscriptions, cheers/bits, raids, and custom events.
//!
//! # Design
//!
//! - [`AlertQueue`] is the primary entry point.  Alerts are enqueued with a
//!   [`Priority`] and dequeued in priority order (highest first) with
//!   FIFO tie-breaking within the same priority tier.
//! - [`TtsPreparation`] expands a raw alert into display text and a
//!   pronunciation-friendly TTS string (e.g. "$5.00" → "five dollars").
//! - [`DisplayDuration`] calculates how long an alert overlay should be
//!   shown based on alert type and message length.
//! - All library code is `unwrap()`-free.

use std::collections::VecDeque;
use std::time::Duration;

// ---------------------------------------------------------------------------
// AlertKind
// ---------------------------------------------------------------------------

/// Classification of a streamer alert.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AlertKind {
    /// One-time monetary donation.
    Donation,
    /// New subscriber (Tier 1/2/3 or equivalent).
    NewSubscriber,
    /// Subscription renewed.
    Resubscription {
        /// Number of months subscribed.
        months: u32,
    },
    /// Gift subscription sent to another viewer.
    GiftSubscription {
        /// Number of subs gifted.
        count: u32,
    },
    /// Platform cheers / bits / equivalent micro-transaction.
    Cheer {
        /// Amount of bits/cheers.
        amount: u64,
    },
    /// Channel raid from another streamer.
    Raid {
        /// Number of raiders.
        raiders: u64,
    },
    /// Host event (another channel hosting this stream).
    Host {
        /// Viewer count brought by the host.
        viewers: u64,
    },
    /// Follow / subscribe-free event.
    Follow,
    /// Arbitrary custom alert.
    Custom {
        /// Short identifier for the custom alert type.
        type_id: String,
    },
}

impl AlertKind {
    /// Human-readable name for display purposes.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Donation => "Donation",
            Self::NewSubscriber => "New Subscriber",
            Self::Resubscription { .. } => "Resubscription",
            Self::GiftSubscription { .. } => "Gift Subscription",
            Self::Cheer { .. } => "Cheer",
            Self::Raid { .. } => "Raid",
            Self::Host { .. } => "Host",
            Self::Follow => "Follow",
            Self::Custom { .. } => "Custom Alert",
        }
    }
}

// ---------------------------------------------------------------------------
// Priority
// ---------------------------------------------------------------------------

/// Alert display priority — higher variants are shown before lower ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Low-importance alerts (follows, small cheers).
    Low = 0,
    /// Standard alerts (new subscribers, small donations).
    Normal = 1,
    /// Important alerts (large donations, gift subs, raids).
    High = 2,
    /// Critical/interrupting alerts shown immediately.
    Critical = 3,
}

impl Priority {
    /// Determine a sensible default priority for a given alert kind and
    /// optional monetary amount in cents.
    #[must_use]
    pub fn auto_classify(kind: &AlertKind, amount_cents: Option<u64>) -> Self {
        match kind {
            AlertKind::Follow => Self::Low,
            AlertKind::Cheer { amount } => {
                if *amount >= 1000 {
                    Self::High
                } else {
                    Self::Normal
                }
            }
            AlertKind::Donation => match amount_cents {
                Some(c) if c >= 5000 => Self::Critical, // $50+
                Some(c) if c >= 1000 => Self::High,     // $10+
                _ => Self::Normal,
            },
            AlertKind::Raid { raiders } => {
                if *raiders >= 100 {
                    Self::Critical
                } else if *raiders >= 20 {
                    Self::High
                } else {
                    Self::Normal
                }
            }
            AlertKind::GiftSubscription { count } => {
                if *count >= 10 {
                    Self::High
                } else {
                    Self::Normal
                }
            }
            AlertKind::NewSubscriber | AlertKind::Resubscription { .. } => Self::Normal,
            AlertKind::Host { viewers } => {
                if *viewers >= 50 {
                    Self::High
                } else {
                    Self::Normal
                }
            }
            AlertKind::Custom { .. } => Self::Normal,
        }
    }
}

// ---------------------------------------------------------------------------
// Alert
// ---------------------------------------------------------------------------

/// A single streamer alert ready to be displayed.
#[derive(Debug, Clone)]
pub struct Alert {
    /// Monotonically increasing sequence number (for FIFO within same priority).
    pub(crate) seq: u64,
    /// Alert classification.
    pub kind: AlertKind,
    /// Display priority.
    pub priority: Priority,
    /// Viewer / donor display name.
    pub sender_name: String,
    /// Optional personal message attached to the alert.
    pub message: Option<String>,
    /// Optional monetary amount in cents (e.g. 500 = $5.00).
    pub amount_cents: Option<u64>,
    /// Currency code (ISO 4217), e.g. "USD".
    pub currency: Option<String>,
    /// Elapsed stream seconds when the alert was enqueued.
    pub elapsed_secs: u64,
}

impl Alert {
    /// Display amount as a human-readable string, e.g. "$5.00" or "€10.50".
    #[must_use]
    pub fn formatted_amount(&self) -> Option<String> {
        let cents = self.amount_cents?;
        let symbol = match self.currency.as_deref() {
            Some("USD") | None => "$",
            Some("EUR") => "€",
            Some("GBP") => "£",
            Some("JPY") => "¥",
            Some("CAD") => "CA$",
            Some("AUD") => "AU$",
            Some(other) => {
                // Return amount with raw code prefix
                return Some(format!("{} {}.{:02}", other, cents / 100, cents % 100));
            }
        };
        Some(format!("{}{}.{:02}", symbol, cents / 100, cents % 100))
    }
}

// ---------------------------------------------------------------------------
// TtsPreparation
// ---------------------------------------------------------------------------

/// Result of preparing an alert for text-to-speech playback.
#[derive(Debug, Clone)]
pub struct TtsPreparation {
    /// Full display text shown on the overlay (may contain symbols).
    pub display_text: String,
    /// TTS-friendly string safe for synthesis engines.
    pub tts_text: String,
    /// Estimated speech duration.
    pub estimated_duration: Duration,
}

impl TtsPreparation {
    /// Prepare TTS and display text for an alert.
    #[must_use]
    pub fn prepare(alert: &Alert) -> Self {
        let amount_str = alert.formatted_amount().unwrap_or_default();
        let amount_spoken = Self::amount_to_words(alert.amount_cents, alert.currency.as_deref());

        let (display_text, tts_text) = match &alert.kind {
            AlertKind::Donation => {
                let disp = if amount_str.is_empty() {
                    format!("{} donated!", alert.sender_name)
                } else {
                    format!("{} donated {}!", alert.sender_name, amount_str)
                };
                let tts = if amount_spoken.is_empty() {
                    format!("{} donated!", alert.sender_name)
                } else {
                    format!("{} donated {}!", alert.sender_name, amount_spoken)
                };
                (disp, tts)
            }
            AlertKind::NewSubscriber => (
                format!("{} just subscribed!", alert.sender_name),
                format!("{} just subscribed!", alert.sender_name),
            ),
            AlertKind::Resubscription { months } => (
                format!(
                    "{} resubscribed for {} month{}!",
                    alert.sender_name,
                    months,
                    if *months == 1 { "" } else { "s" }
                ),
                format!(
                    "{} resubscribed for {} month{}!",
                    alert.sender_name,
                    months,
                    if *months == 1 { "" } else { "s" }
                ),
            ),
            AlertKind::GiftSubscription { count } => (
                format!(
                    "{} gifted {} sub{}!",
                    alert.sender_name,
                    count,
                    if *count == 1 { "" } else { "s" }
                ),
                format!(
                    "{} gifted {} subscription{}!",
                    alert.sender_name,
                    count,
                    if *count == 1 { "" } else { "s" }
                ),
            ),
            AlertKind::Cheer { amount } => (
                format!("{} cheered {} bits!", alert.sender_name, amount),
                format!("{} cheered {} bits!", alert.sender_name, amount),
            ),
            AlertKind::Raid { raiders } => (
                format!(
                    "{} is raiding with {} viewer{}!",
                    alert.sender_name,
                    raiders,
                    if *raiders == 1 { "" } else { "s" }
                ),
                format!(
                    "{} is raiding with {} viewer{}!",
                    alert.sender_name,
                    raiders,
                    if *raiders == 1 { "" } else { "s" }
                ),
            ),
            AlertKind::Host { viewers } => (
                format!(
                    "{} is hosting with {} viewer{}!",
                    alert.sender_name,
                    viewers,
                    if *viewers == 1 { "" } else { "s" }
                ),
                format!(
                    "{} is hosting with {} viewer{}!",
                    alert.sender_name,
                    viewers,
                    if *viewers == 1 { "" } else { "s" }
                ),
            ),
            AlertKind::Follow => (
                format!("{} followed!", alert.sender_name),
                format!("{} followed!", alert.sender_name),
            ),
            AlertKind::Custom { type_id } => (
                format!("[{}] {}", type_id, alert.sender_name),
                format!("Custom alert from {}!", alert.sender_name),
            ),
        };

        // Append personal message if present.
        let (mut display_text, mut tts_text) = (display_text, tts_text);
        if let Some(msg) = &alert.message {
            if !msg.is_empty() {
                display_text.push_str(&format!(" — {msg}"));
                tts_text.push_str(&format!(" They say: {msg}"));
            }
        }

        let estimated_duration = Self::estimate_speech_duration(&tts_text);

        Self {
            display_text,
            tts_text,
            estimated_duration,
        }
    }

    /// Convert a monetary amount in cents to spoken English words.
    /// Returns an empty string if the amount is `None`.
    fn amount_to_words(amount_cents: Option<u64>, currency: Option<&str>) -> String {
        let cents = match amount_cents {
            Some(c) => c,
            None => return String::new(),
        };
        let dollars = cents / 100;
        let rem_cents = cents % 100;
        let unit = match currency {
            Some("EUR") => ("euro", "euro cent"),
            Some("GBP") => ("pound", "pence"),
            _ => ("dollar", "cent"),
        };
        if rem_cents == 0 {
            format!(
                "{} {}{}",
                dollars,
                unit.0,
                if dollars == 1 { "" } else { "s" }
            )
        } else {
            format!("{} {} and {} {}", dollars, unit.0, rem_cents, unit.1)
        }
    }

    /// Estimate speech duration at ~150 words per minute.
    fn estimate_speech_duration(text: &str) -> Duration {
        let word_count = text.split_whitespace().count();
        // 150 wpm → 0.4 seconds per word
        let secs = (word_count as f64 * 0.4).max(1.0);
        Duration::from_secs_f64(secs)
    }
}

// ---------------------------------------------------------------------------
// DisplayDuration
// ---------------------------------------------------------------------------

/// Calculates how long an alert overlay should remain visible.
#[derive(Debug, Clone, Copy)]
pub struct DisplayDuration;

impl DisplayDuration {
    /// Base display time for an alert kind (seconds).
    #[must_use]
    pub fn base_secs(kind: &AlertKind) -> u64 {
        match kind {
            AlertKind::Follow => 4,
            AlertKind::Cheer { .. } => 5,
            AlertKind::NewSubscriber | AlertKind::Resubscription { .. } => 6,
            AlertKind::GiftSubscription { .. } => 7,
            AlertKind::Donation => 8,
            AlertKind::Raid { .. } | AlertKind::Host { .. } => 10,
            AlertKind::Custom { .. } => 5,
        }
    }

    /// Calculate total display duration including time for reading the message.
    ///
    /// Adds ~0.05 seconds per character in the personal message, capped at 15s.
    #[must_use]
    pub fn calculate(kind: &AlertKind, message: Option<&str>) -> Duration {
        let base = Self::base_secs(kind);
        let extra = message.map(|m| (m.len() as f64 * 0.05) as u64).unwrap_or(0);
        let total = (base + extra).min(15);
        Duration::from_secs(total)
    }
}

// ---------------------------------------------------------------------------
// AlertBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing [`Alert`] values.
#[derive(Debug, Default)]
pub struct AlertBuilder {
    kind: Option<AlertKind>,
    priority: Option<Priority>,
    sender_name: String,
    message: Option<String>,
    amount_cents: Option<u64>,
    currency: Option<String>,
    elapsed_secs: u64,
}

impl AlertBuilder {
    /// Create a new alert builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the alert kind.
    #[must_use]
    pub fn kind(mut self, kind: AlertKind) -> Self {
        self.kind = Some(kind);
        self
    }

    /// Override the auto-classified priority.
    #[must_use]
    pub fn priority(mut self, priority: Priority) -> Self {
        self.priority = Some(priority);
        self
    }

    /// Set the sender (donor/subscriber) display name.
    #[must_use]
    pub fn sender(mut self, name: impl Into<String>) -> Self {
        self.sender_name = name.into();
        self
    }

    /// Attach a personal message.
    #[must_use]
    pub fn message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Set the monetary amount in cents.
    #[must_use]
    pub fn amount_cents(mut self, cents: u64) -> Self {
        self.amount_cents = Some(cents);
        self
    }

    /// Set the currency (ISO 4217 code).
    #[must_use]
    pub fn currency(mut self, code: impl Into<String>) -> Self {
        self.currency = Some(code.into());
        self
    }

    /// Set the elapsed stream seconds when this alert occurred.
    #[must_use]
    pub fn elapsed_secs(mut self, secs: u64) -> Self {
        self.elapsed_secs = secs;
        self
    }

    /// Build the alert.  Returns `None` if no kind was set.
    #[must_use]
    pub fn build(self, seq: u64) -> Option<Alert> {
        let kind = self.kind?;
        let priority = self
            .priority
            .unwrap_or_else(|| Priority::auto_classify(&kind, self.amount_cents));
        Some(Alert {
            seq,
            kind,
            priority,
            sender_name: self.sender_name,
            message: self.message,
            amount_cents: self.amount_cents,
            currency: self.currency,
            elapsed_secs: self.elapsed_secs,
        })
    }
}

// ---------------------------------------------------------------------------
// AlertQueue
// ---------------------------------------------------------------------------

/// Priority-ordered alert queue with deduplication.
///
/// Alerts are returned in descending priority order with FIFO tie-breaking
/// within the same priority tier.  Up to `capacity` pending alerts are stored;
/// once full, newly enqueued low-priority alerts may be dropped.
#[derive(Debug)]
pub struct AlertQueue {
    /// Internal storage — unsorted insertion, sorted on dequeue.
    pending: VecDeque<Alert>,
    /// Maximum number of pending alerts.
    capacity: usize,
    /// Monotonic sequence counter.
    seq: u64,
    /// History of displayed alerts (capped at `history_cap`).
    history: VecDeque<Alert>,
    history_cap: usize,
}

impl AlertQueue {
    /// Create a new alert queue.
    ///
    /// - `capacity`: maximum pending alerts before dropping new ones.
    /// - `history_cap`: maximum displayed-alert history to keep.
    #[must_use]
    pub fn new(capacity: usize, history_cap: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            capacity: capacity.max(1),
            seq: 0,
            history: VecDeque::new(),
            history_cap: history_cap.max(1),
        }
    }

    /// Enqueue an alert.  Returns `false` if the queue is full and the alert
    /// was dropped (low-priority alerts are dropped first).
    pub fn enqueue(&mut self, mut alert: Alert) -> bool {
        if self.pending.len() >= self.capacity {
            // Try to drop a lower-priority alert to make room.
            if let Some(pos) = self.lowest_priority_position(&alert) {
                self.pending.remove(pos);
            } else {
                // New alert is the lowest priority — drop it.
                return false;
            }
        }
        alert.seq = self.seq;
        self.seq = self.seq.saturating_add(1);
        self.pending.push_back(alert);
        true
    }

    /// Dequeue the next alert to display (highest priority, then oldest seq).
    ///
    /// Moves it into the history log.  Returns `None` if there are no pending
    /// alerts.
    pub fn dequeue(&mut self) -> Option<Alert> {
        if self.pending.is_empty() {
            return None;
        }
        // Find the index of the highest-priority alert (lowest seq wins tie).
        let best_idx = self
            .pending
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.priority.cmp(&b.priority).then_with(|| b.seq.cmp(&a.seq)) // lower seq = older = higher precedence
            })
            .map(|(i, _)| i)?;

        let alert = self.pending.remove(best_idx)?;
        // Push to history
        if self.history.len() >= self.history_cap {
            self.history.pop_front();
        }
        self.history.push_back(alert.clone());
        Some(alert)
    }

    /// Number of alerts waiting to be displayed.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Peek at the next alert without removing it.
    #[must_use]
    pub fn peek_next(&self) -> Option<&Alert> {
        self.pending
            .iter()
            .max_by(|a, b| a.priority.cmp(&b.priority).then_with(|| b.seq.cmp(&a.seq)))
    }

    /// History of the last `history_cap` displayed alerts.
    #[must_use]
    pub fn history(&self) -> &VecDeque<Alert> {
        &self.history
    }

    /// Clear all pending alerts.
    pub fn clear(&mut self) {
        self.pending.clear();
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Position in `pending` of the alert with the lowest priority (oldest seq
    /// wins tie) — a candidate for eviction to make room for a higher-priority
    /// incoming alert.  Returns `None` if the incoming alert is lower than all
    /// queued alerts.
    fn lowest_priority_position(&self, incoming: &Alert) -> Option<usize> {
        let (idx, candidate) = self.pending.iter().enumerate().min_by(|(_, a), (_, b)| {
            a.priority.cmp(&b.priority).then_with(|| a.seq.cmp(&b.seq)) // higher seq = newer = evict first
        })?;

        if candidate.priority < incoming.priority {
            Some(idx)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_alert(kind: AlertKind, priority: Priority, sender: &str, seq: u64) -> Alert {
        Alert {
            seq,
            kind,
            priority,
            sender_name: sender.to_string(),
            message: None,
            amount_cents: None,
            currency: None,
            elapsed_secs: 0,
        }
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_auto_classify_follow_is_low() {
        let p = Priority::auto_classify(&AlertKind::Follow, None);
        assert_eq!(p, Priority::Low);
    }

    #[test]
    fn test_auto_classify_large_donation_is_critical() {
        let p = Priority::auto_classify(&AlertKind::Donation, Some(10_000));
        assert_eq!(p, Priority::Critical);
    }

    #[test]
    fn test_auto_classify_raid_sizes() {
        let small = Priority::auto_classify(&AlertKind::Raid { raiders: 5 }, None);
        let big = Priority::auto_classify(&AlertKind::Raid { raiders: 200 }, None);
        assert_eq!(small, Priority::Normal);
        assert_eq!(big, Priority::Critical);
    }

    #[test]
    fn test_queue_dequeue_priority_order() {
        let mut q = AlertQueue::new(10, 20);
        q.enqueue(make_alert(AlertKind::Follow, Priority::Low, "viewer1", 0));
        q.enqueue(make_alert(AlertKind::Donation, Priority::High, "donor1", 1));
        q.enqueue(make_alert(
            AlertKind::NewSubscriber,
            Priority::Normal,
            "sub1",
            2,
        ));

        let first = q.dequeue().expect("should dequeue");
        assert_eq!(first.priority, Priority::High);
        let second = q.dequeue().expect("should dequeue");
        assert_eq!(second.priority, Priority::Normal);
        let third = q.dequeue().expect("should dequeue");
        assert_eq!(third.priority, Priority::Low);
        assert!(q.dequeue().is_none());
    }

    #[test]
    fn test_queue_capacity_drops_low_priority() {
        let mut q = AlertQueue::new(2, 5);
        let a1 = make_alert(AlertKind::Donation, Priority::High, "a", 0);
        let a2 = make_alert(AlertKind::Donation, Priority::High, "b", 1);
        let low = make_alert(AlertKind::Follow, Priority::Low, "c", 2);
        assert!(q.enqueue(a1));
        assert!(q.enqueue(a2));
        // Queue full; low priority should be dropped
        assert!(!q.enqueue(low));
        assert_eq!(q.pending_count(), 2);
    }

    #[test]
    fn test_queue_capacity_evicts_lower_for_higher() {
        let mut q = AlertQueue::new(2, 5);
        let low1 = make_alert(AlertKind::Follow, Priority::Low, "a", 0);
        let low2 = make_alert(AlertKind::Follow, Priority::Low, "b", 1);
        let high = make_alert(AlertKind::Raid { raiders: 200 }, Priority::Critical, "c", 2);
        assert!(q.enqueue(low1));
        assert!(q.enqueue(low2));
        // High-priority should evict one of the low ones
        assert!(q.enqueue(high));
        assert_eq!(q.pending_count(), 2);
        let next = q.peek_next().expect("should have next");
        assert_eq!(next.priority, Priority::Critical);
    }

    #[test]
    fn test_tts_preparation_donation() {
        let alert = Alert {
            seq: 0,
            kind: AlertKind::Donation,
            priority: Priority::High,
            sender_name: "Alice".to_string(),
            message: Some("Thank you!".to_string()),
            amount_cents: Some(500),
            currency: Some("USD".to_string()),
            elapsed_secs: 120,
        };
        let tts = TtsPreparation::prepare(&alert);
        assert!(tts.display_text.contains("Alice"));
        assert!(tts.display_text.contains("$5.00"));
        assert!(tts.tts_text.contains("5 dollars"));
        assert!(tts.tts_text.contains("Thank you"));
        assert!(tts.estimated_duration > Duration::ZERO);
    }

    #[test]
    fn test_tts_preparation_raid() {
        let alert = Alert {
            seq: 1,
            kind: AlertKind::Raid { raiders: 50 },
            priority: Priority::High,
            sender_name: "Raider".to_string(),
            message: None,
            amount_cents: None,
            currency: None,
            elapsed_secs: 300,
        };
        let tts = TtsPreparation::prepare(&alert);
        assert!(tts.tts_text.contains("50"));
        assert!(tts.display_text.contains("Raider"));
    }

    #[test]
    fn test_display_duration_increases_with_message() {
        let short = DisplayDuration::calculate(&AlertKind::Donation, None);
        let long_msg = "A".repeat(100);
        let long = DisplayDuration::calculate(&AlertKind::Donation, Some(&long_msg));
        assert!(long >= short);
        // Capped at 15 seconds
        assert!(long <= Duration::from_secs(15));
    }

    #[test]
    fn test_alert_builder_auto_priority() {
        let alert = AlertBuilder::new()
            .kind(AlertKind::Donation)
            .sender("Bob")
            .amount_cents(2000) // $20 → High
            .currency("USD")
            .build(0)
            .expect("should build");
        assert_eq!(alert.priority, Priority::High);
        assert_eq!(alert.sender_name, "Bob");
    }

    #[test]
    fn test_formatted_amount_usd() {
        let alert = Alert {
            seq: 0,
            kind: AlertKind::Donation,
            priority: Priority::Normal,
            sender_name: "Bob".to_string(),
            message: None,
            amount_cents: Some(1050),
            currency: Some("USD".to_string()),
            elapsed_secs: 0,
        };
        assert_eq!(alert.formatted_amount(), Some("$10.50".to_string()));
    }

    #[test]
    fn test_formatted_amount_eur() {
        let alert = Alert {
            seq: 0,
            kind: AlertKind::Donation,
            priority: Priority::Normal,
            sender_name: "Carlo".to_string(),
            message: None,
            amount_cents: Some(200),
            currency: Some("EUR".to_string()),
            elapsed_secs: 0,
        };
        assert_eq!(alert.formatted_amount(), Some("€2.00".to_string()));
    }

    #[test]
    fn test_queue_history_recorded() {
        let mut q = AlertQueue::new(10, 5);
        q.enqueue(make_alert(AlertKind::Follow, Priority::Low, "u1", 0));
        q.dequeue();
        assert_eq!(q.history().len(), 1);
    }

    #[test]
    fn test_queue_clear() {
        let mut q = AlertQueue::new(10, 5);
        q.enqueue(make_alert(AlertKind::Follow, Priority::Low, "u1", 0));
        q.enqueue(make_alert(AlertKind::Follow, Priority::Low, "u2", 1));
        q.clear();
        assert!(q.is_empty());
    }

    #[test]
    fn test_alert_kind_display_names() {
        assert_eq!(AlertKind::Donation.display_name(), "Donation");
        assert_eq!(AlertKind::Follow.display_name(), "Follow");
        assert_eq!(AlertKind::Raid { raiders: 10 }.display_name(), "Raid");
        assert_eq!(
            AlertKind::Custom {
                type_id: "milestone".to_string()
            }
            .display_name(),
            "Custom Alert"
        );
    }
}
