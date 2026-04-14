//! Real-time data binding for the scoreboard renderer.
//!
//! This module provides the [`DataSource`] trait, typed update events
//! ([`ScoreUpdate`], [`ScoreField`], [`ScoreValue`]), and a reference
//! in-memory implementation ([`MemoryDataSource`]) that can be used for
//! testing or as a template for external API / shared-memory bridges.
//!
//! # Integration with [`ScoreboardRenderer`]
//!
//! Attach any [`DataSource`] to a [`ScoreboardRenderer`] with
//! [`ScoreboardRenderer::attach_source`]. Every call to
//! [`ScoreboardRenderer::render_frame`] will first drain pending updates
//! from the source before producing the RGBA overlay buffer.
//!
//! ```rust
//! use oximedia_graphics::scoreboard::{ScoreboardConfig, SportType, TeamScore, GameClock};
//! use oximedia_graphics::scoreboard_datasource::{
//!     ScoreboardRenderer, MemoryDataSource, ScoreUpdate, ScoreField, ScoreValue,
//! };
//!
//! let config = ScoreboardConfig::new(
//!     SportType::Basketball,
//!     TeamScore::new("HOME", 0, [200, 0, 0, 255]),
//!     TeamScore::new("AWAY", 0, [0, 0, 200, 255]),
//!     GameClock::new(12, 0, 1),
//!     true,
//! );
//! let mut renderer = ScoreboardRenderer::new(config);
//! let mut source = MemoryDataSource::new();
//! source.push(ScoreUpdate {
//!     field: ScoreField::HomeScore,
//!     value: ScoreValue::Integer(3),
//!     timestamp_ms: 1000,
//! });
//! renderer.attach_source(Box::new(source));
//! let _pixels = renderer.render_frame(1920, 1080);
//! ```

use std::collections::HashMap;

use crate::scoreboard::{GameClock, ScoreboardConfig, ScoreboardUpdate, ScoreboardRenderer as StaticRenderer};

// ---------------------------------------------------------------------------
// Public data types
// ---------------------------------------------------------------------------

/// A key-value update event for scoreboard data.
#[derive(Debug, Clone)]
pub struct ScoreUpdate {
    /// Which field to update.
    pub field: ScoreField,
    /// The new value for the field.
    pub value: ScoreValue,
    /// Wall-clock timestamp in milliseconds at which this update was produced.
    pub timestamp_ms: u64,
}

/// Which field in the scoreboard to update.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScoreField {
    /// Home team integer score.
    HomeScore,
    /// Away team integer score.
    AwayScore,
    /// Home team display name.
    HomeName,
    /// Away team display name.
    AwayName,
    /// Current period / quarter / half (integer).
    Period,
    /// Clock string (e.g. `"12:34"`).
    Clock,
    /// Arbitrary named field for sponsor banners, custom stats, etc.
    Custom(String),
}

/// A typed value for a score field.
#[derive(Debug, Clone, PartialEq)]
pub enum ScoreValue {
    /// Signed 64-bit integer (scores, periods, …).
    Integer(i64),
    /// UTF-8 text (team names, clock strings, …).
    Text(String),
    /// 64-bit floating-point (win probability, distance, …).
    Float(f64),
}

// ---------------------------------------------------------------------------
// DataSource trait
// ---------------------------------------------------------------------------

/// Trait for live data sources that push [`ScoreUpdate`] events.
///
/// Implement this trait to connect any real-time feed (REST polling, WebSocket,
/// shared memory ring-buffer, …) to a [`ScoreboardRenderer`].
pub trait DataSource: Send {
    /// Drain all pending updates accumulated since the last call.
    ///
    /// The returned `Vec` is empty when there is nothing new. Implementations
    /// must clear their internal queue so subsequent calls do not return the
    /// same events twice.
    fn poll_updates(&mut self) -> Vec<ScoreUpdate>;

    /// Whether the underlying connection / feed is still healthy.
    ///
    /// A renderer that calls `poll_updates` on a disconnected source may
    /// choose to display a "feed lost" indicator or freeze the last state.
    fn is_connected(&self) -> bool;
}

// ---------------------------------------------------------------------------
// MemoryDataSource — reference in-memory implementation
// ---------------------------------------------------------------------------

/// An in-memory [`DataSource`] suitable for unit tests and integration demos.
///
/// Push updates via [`MemoryDataSource::push`]; they will be returned on the
/// next [`DataSource::poll_updates`] call and then cleared.
pub struct MemoryDataSource {
    pending: Vec<ScoreUpdate>,
    connected: bool,
}

impl MemoryDataSource {
    /// Create a new, empty, connected `MemoryDataSource`.
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            connected: true,
        }
    }

    /// Enqueue an update to be returned on the next [`DataSource::poll_updates`].
    pub fn push(&mut self, update: ScoreUpdate) {
        self.pending.push(update);
    }

    /// Mark this source as disconnected.
    ///
    /// After calling `disconnect`, [`DataSource::is_connected`] will return
    /// `false` and the renderer will stop polling this source.
    pub fn disconnect(&mut self) {
        self.connected = false;
    }
}

impl Default for MemoryDataSource {
    fn default() -> Self {
        Self::new()
    }
}

impl DataSource for MemoryDataSource {
    fn poll_updates(&mut self) -> Vec<ScoreUpdate> {
        // Drain and return, leaving an empty queue.
        std::mem::take(&mut self.pending)
    }

    fn is_connected(&self) -> bool {
        self.connected
    }
}

// ---------------------------------------------------------------------------
// ScoreboardRenderer — stateful renderer with live data binding
// ---------------------------------------------------------------------------

/// A stateful scoreboard renderer that optionally pulls live data from an
/// attached [`DataSource`] on every call to [`render_frame`].
///
/// This type wraps the immutable [`StaticRenderer`] and adds:
/// - internal mutable state (scores, names, period, clock string, custom map)
/// - an optional boxed [`DataSource`] that is polled each frame
/// - [`apply_update`] / [`apply_updates`] for synchronous manual updates
///
/// [`render_frame`]: ScoreboardRenderer::render_frame
pub struct ScoreboardRenderer {
    config: ScoreboardConfig,
    /// Custom field store (e.g. "sponsor", "possession", …).
    custom_fields: HashMap<String, ScoreValue>,
    /// Optionally-attached live data source.
    source: Option<Box<dyn DataSource>>,
}

impl ScoreboardRenderer {
    /// Create a new renderer from the given static configuration.
    pub fn new(config: ScoreboardConfig) -> Self {
        Self {
            config,
            custom_fields: HashMap::new(),
            source: None,
        }
    }

    // ------------------------------------------------------------------
    // Data-source attachment
    // ------------------------------------------------------------------

    /// Attach a live [`DataSource`].
    ///
    /// The source will be polled on every call to [`render_frame`]. Any
    /// previously attached source is replaced.
    pub fn attach_source(&mut self, source: Box<dyn DataSource>) {
        self.source = Some(source);
    }

    /// Detach the current [`DataSource`], if any.
    ///
    /// After calling this method, [`render_frame`] will no longer poll for
    /// updates; the renderer keeps its last known state.
    pub fn detach_source(&mut self) {
        self.source = None;
    }

    // ------------------------------------------------------------------
    // Manual update application
    // ------------------------------------------------------------------

    /// Apply a single [`ScoreUpdate`] to the internal state immediately.
    pub fn apply_update(&mut self, update: ScoreUpdate) {
        match update.field {
            ScoreField::HomeScore => {
                if let ScoreValue::Integer(v) = update.value {
                    // Saturate at zero; u32 cannot represent negative scores.
                    self.config.home.score = u32::try_from(v).unwrap_or(0);
                }
            }
            ScoreField::AwayScore => {
                if let ScoreValue::Integer(v) = update.value {
                    self.config.away.score = u32::try_from(v).unwrap_or(0);
                }
            }
            ScoreField::HomeName => {
                if let ScoreValue::Text(name) = update.value {
                    self.config.home.name = name;
                }
            }
            ScoreField::AwayName => {
                if let ScoreValue::Text(name) = update.value {
                    self.config.away.name = name;
                }
            }
            ScoreField::Period => {
                if let ScoreValue::Integer(v) = update.value {
                    let period = u32::try_from(v).unwrap_or(1).max(1);
                    self.config.clock.period = period;
                    // Keep GameClock's own period field in sync.
                    self.config
                        .clock
                        .apply_scoreboard_update(ScoreboardUpdate::PeriodChange(period));
                }
            }
            ScoreField::Clock => {
                if let ScoreValue::Text(clock_str) = update.value {
                    // Parse "MM:SS" or "M:SS" into the GameClock minutes/seconds.
                    if let Some((mins, secs)) = parse_clock_string(&clock_str) {
                        self.config.clock.minutes = mins;
                        self.config.clock.seconds = secs;
                    }
                }
            }
            ScoreField::Custom(key) => {
                self.custom_fields.insert(key, update.value);
            }
        }
    }

    /// Apply a batch of [`ScoreUpdate`] events in order.
    pub fn apply_updates(&mut self, updates: &[ScoreUpdate]) {
        for update in updates {
            self.apply_update(update.clone());
        }
    }

    // ------------------------------------------------------------------
    // Rendering
    // ------------------------------------------------------------------

    /// Render the scoreboard as an RGBA overlay strip.
    ///
    /// Before rendering, any attached [`DataSource`] is polled and all pending
    /// updates are applied so the output always reflects the freshest state.
    ///
    /// Returns `Vec<u8>` of RGBA data with length `width × bar_height × 4`.
    pub fn render_frame(&mut self, width: u32, height: u32) -> Vec<u8> {
        // 1. Poll the attached source (if connected).
        if let Some(source) = self.source.as_mut() {
            if source.is_connected() {
                let updates = source.poll_updates();
                for u in updates {
                    self.apply_update(u);
                }
            }
        }

        // 2. Delegate pixel rendering to the existing static renderer.
        StaticRenderer::render(&self.config, width, height)
    }

    // ------------------------------------------------------------------
    // State accessors (useful for tests and HUD overlays)
    // ------------------------------------------------------------------

    /// Current home team score.
    pub fn home_score(&self) -> u32 {
        self.config.home.score
    }

    /// Current away team score.
    pub fn away_score(&self) -> u32 {
        self.config.away.score
    }

    /// Current home team name.
    pub fn home_name(&self) -> &str {
        &self.config.home.name
    }

    /// Current away team name.
    pub fn away_name(&self) -> &str {
        &self.config.away.name
    }

    /// Current period.
    pub fn period(&self) -> u32 {
        self.config.clock.period
    }

    /// Current clock as `"MM:SS"` string.
    pub fn clock_string(&self) -> String {
        self.config.clock.format()
    }

    /// Retrieve a stored custom field value, if present.
    pub fn custom_field(&self, key: &str) -> Option<&ScoreValue> {
        self.custom_fields.get(key)
    }
}

// ---------------------------------------------------------------------------
// Helper: GameClock extension (keep PeriodChange in sync)
// ---------------------------------------------------------------------------

/// Internal extension trait so we can call `apply_scoreboard_update` on `GameClock`.
trait GameClockExt {
    fn apply_scoreboard_update(&mut self, update: ScoreboardUpdate);
}

impl GameClockExt for GameClock {
    fn apply_scoreboard_update(&mut self, update: ScoreboardUpdate) {
        if let ScoreboardUpdate::PeriodChange(p) = update {
            self.period = p;
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: parse "MM:SS" clock string
// ---------------------------------------------------------------------------

/// Parse a `"MM:SS"` or `"M:SS"` string into `(minutes, seconds)`.
///
/// Returns `None` if the string is malformed.
fn parse_clock_string(s: &str) -> Option<(u32, u32)> {
    let mut parts = s.splitn(2, ':');
    let mins_str = parts.next()?;
    let secs_str = parts.next()?;
    let mins = mins_str.trim().parse::<u32>().ok()?;
    let secs = secs_str.trim().parse::<u32>().ok()?;
    if secs >= 60 {
        return None;
    }
    Some((mins, secs))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoreboard::{GameClock, ScoreboardConfig, SportType, TeamScore};

    fn make_config() -> ScoreboardConfig {
        ScoreboardConfig::new(
            SportType::Basketball,
            TeamScore::new("HOME", 0, [200, 0, 0, 255]),
            TeamScore::new("AWAY", 0, [0, 0, 200, 255]),
            GameClock::new(12, 0, 1),
            true,
        )
    }

    fn make_renderer() -> ScoreboardRenderer {
        ScoreboardRenderer::new(make_config())
    }

    // 1. New MemoryDataSource returns no updates.
    #[test]
    fn test_memory_source_new_empty() {
        let mut src = MemoryDataSource::new();
        let updates = src.poll_updates();
        assert!(updates.is_empty(), "expected no updates on fresh source");
    }

    // 2. push → poll returns them; subsequent poll returns empty.
    #[test]
    fn test_memory_source_push_poll() {
        let mut src = MemoryDataSource::new();
        src.push(ScoreUpdate {
            field: ScoreField::HomeScore,
            value: ScoreValue::Integer(5),
            timestamp_ms: 100,
        });
        src.push(ScoreUpdate {
            field: ScoreField::AwayScore,
            value: ScoreValue::Integer(3),
            timestamp_ms: 200,
        });
        let first = src.poll_updates();
        assert_eq!(first.len(), 2);
        let second = src.poll_updates();
        assert!(second.is_empty(), "queue must be drained after first poll");
    }

    // 3. New source is connected.
    #[test]
    fn test_memory_source_connected() {
        let src = MemoryDataSource::new();
        assert!(src.is_connected());
    }

    // 4. After disconnect, is_connected returns false.
    #[test]
    fn test_memory_source_disconnect() {
        let mut src = MemoryDataSource::new();
        src.disconnect();
        assert!(!src.is_connected());
    }

    // 5. Applying HomeScore update changes rendered score.
    #[test]
    fn test_apply_update_home_score() {
        let mut r = make_renderer();
        r.apply_update(ScoreUpdate {
            field: ScoreField::HomeScore,
            value: ScoreValue::Integer(7),
            timestamp_ms: 0,
        });
        assert_eq!(r.home_score(), 7);
    }

    // 6. Applying AwayScore update changes rendered away score.
    #[test]
    fn test_apply_update_away_score() {
        let mut r = make_renderer();
        r.apply_update(ScoreUpdate {
            field: ScoreField::AwayScore,
            value: ScoreValue::Integer(4),
            timestamp_ms: 0,
        });
        assert_eq!(r.away_score(), 4);
    }

    // 7. Applying HomeName update changes team name.
    #[test]
    fn test_apply_update_home_name() {
        let mut r = make_renderer();
        r.apply_update(ScoreUpdate {
            field: ScoreField::HomeName,
            value: ScoreValue::Text("Lakers".to_string()),
            timestamp_ms: 0,
        });
        assert_eq!(r.home_name(), "Lakers");
    }

    // 8. Applying multiple updates in one batch call works.
    #[test]
    fn test_apply_updates_batch() {
        let mut r = make_renderer();
        let updates = vec![
            ScoreUpdate {
                field: ScoreField::HomeScore,
                value: ScoreValue::Integer(10),
                timestamp_ms: 0,
            },
            ScoreUpdate {
                field: ScoreField::AwayScore,
                value: ScoreValue::Integer(8),
                timestamp_ms: 1,
            },
            ScoreUpdate {
                field: ScoreField::AwayName,
                value: ScoreValue::Text("Celtics".to_string()),
                timestamp_ms: 2,
            },
        ];
        r.apply_updates(&updates);
        assert_eq!(r.home_score(), 10);
        assert_eq!(r.away_score(), 8);
        assert_eq!(r.away_name(), "Celtics");
    }

    // 9. Attached source is polled during render_frame.
    #[test]
    fn test_attach_source_polls_on_render() {
        let mut r = make_renderer();
        let mut src = MemoryDataSource::new();
        src.push(ScoreUpdate {
            field: ScoreField::HomeScore,
            value: ScoreValue::Integer(99),
            timestamp_ms: 0,
        });
        r.attach_source(Box::new(src));
        // Before render, score should still be 0.
        assert_eq!(r.home_score(), 0);
        // render_frame() must poll the source and apply the update.
        let pixels = r.render_frame(320, 240);
        assert!(!pixels.is_empty());
        assert_eq!(r.home_score(), 99, "score must be updated after render_frame");
    }

    // 10. After detach, further source pushes don't affect render.
    #[test]
    fn test_detach_source_stops_polling() {
        let mut r = make_renderer();
        let mut src = MemoryDataSource::new();
        // Push before detach (update not yet polled).
        src.push(ScoreUpdate {
            field: ScoreField::HomeScore,
            value: ScoreValue::Integer(42),
            timestamp_ms: 0,
        });
        r.attach_source(Box::new(src));
        r.detach_source();
        // Render should NOT apply the queued update because source was detached.
        r.render_frame(320, 240);
        assert_eq!(r.home_score(), 0, "detached source must not affect render");
    }

    // 11. Integer, Text, Float all stored correctly.
    #[test]
    fn test_score_value_types() {
        let int_val = ScoreValue::Integer(42);
        let txt_val = ScoreValue::Text("hello".to_string());
        let flt_val = ScoreValue::Float(3.14);

        assert_eq!(int_val, ScoreValue::Integer(42));
        assert_eq!(txt_val, ScoreValue::Text("hello".to_string()));
        // Float comparison via pattern matching to avoid direct f64 equality.
        if let ScoreValue::Float(f) = flt_val {
            assert!((f - 3.14_f64).abs() < 1e-10);
        } else {
            panic!("expected Float variant");
        }
    }

    // 12. Custom("sponsor") field stored and retrievable.
    #[test]
    fn test_custom_field() {
        let mut r = make_renderer();
        r.apply_update(ScoreUpdate {
            field: ScoreField::Custom("sponsor".to_string()),
            value: ScoreValue::Text("Acme Corp".to_string()),
            timestamp_ms: 0,
        });
        let stored = r.custom_field("sponsor");
        assert!(stored.is_some());
        assert_eq!(
            stored.expect("custom field must exist"),
            &ScoreValue::Text("Acme Corp".to_string())
        );
        // Unknown key returns None.
        assert!(r.custom_field("missing").is_none());
    }

    // Bonus: Period and Clock field updates.
    #[test]
    fn test_apply_update_period() {
        let mut r = make_renderer();
        r.apply_update(ScoreUpdate {
            field: ScoreField::Period,
            value: ScoreValue::Integer(3),
            timestamp_ms: 0,
        });
        assert_eq!(r.period(), 3);
    }

    #[test]
    fn test_apply_update_clock_string() {
        let mut r = make_renderer();
        r.apply_update(ScoreUpdate {
            field: ScoreField::Clock,
            value: ScoreValue::Text("05:30".to_string()),
            timestamp_ms: 0,
        });
        assert_eq!(r.clock_string(), "05:30");
    }

    #[test]
    fn test_parse_clock_invalid_ignored() {
        let mut r = make_renderer();
        // Initial clock is 12:00.
        // Sending a malformed string should leave clock unchanged.
        r.apply_update(ScoreUpdate {
            field: ScoreField::Clock,
            value: ScoreValue::Text("not-a-clock".to_string()),
            timestamp_ms: 0,
        });
        assert_eq!(r.clock_string(), "12:00");
    }

    #[test]
    fn test_memory_source_default() {
        let src = MemoryDataSource::default();
        assert!(src.is_connected());
    }
}
