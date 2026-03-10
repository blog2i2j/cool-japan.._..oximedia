#![allow(dead_code)]

//! Dashboard widget system for composing monitoring displays.
//!
//! Provides configurable widgets that render metric data for dashboards,
//! including sparklines, gauge arcs, status indicators, and summary tables.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, SystemTime};

// ---------------------------------------------------------------------------
// Widget types
// ---------------------------------------------------------------------------

/// Unique identifier for a widget instance.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WidgetId(String);

impl WidgetId {
    /// Create a new widget identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Return the inner string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for WidgetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Layout position on a dashboard grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridPosition {
    /// Column index (0-based).
    pub col: u32,
    /// Row index (0-based).
    pub row: u32,
    /// Column span.
    pub col_span: u32,
    /// Row span.
    pub row_span: u32,
}

impl GridPosition {
    /// Create a new grid position.
    #[must_use]
    pub fn new(col: u32, row: u32, col_span: u32, row_span: u32) -> Self {
        Self {
            col,
            row,
            col_span: col_span.max(1),
            row_span: row_span.max(1),
        }
    }

    /// Return the total number of grid cells occupied.
    #[must_use]
    pub fn cell_count(&self) -> u32 {
        self.col_span * self.row_span
    }

    /// Check whether two grid positions overlap.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        let self_right = self.col + self.col_span;
        let self_bottom = self.row + self.row_span;
        let other_right = other.col + other.col_span;
        let other_bottom = other.row + other.row_span;

        self.col < other_right
            && self_right > other.col
            && self.row < other_bottom
            && self_bottom > other.row
    }
}

/// Kind of widget to display.
#[derive(Debug, Clone, PartialEq)]
pub enum WidgetKind {
    /// Sparkline chart showing recent time-series values.
    Sparkline {
        /// Maximum number of data points retained.
        max_points: usize,
        /// Color hex string (e.g. `#00ff00`).
        color: String,
    },
    /// Gauge arc (0 – 100 %).
    Gauge {
        /// Minimum value shown on the gauge.
        min_value: f64,
        /// Maximum value shown on the gauge.
        max_value: f64,
    },
    /// Simple status indicator (ok / warning / critical / unknown).
    StatusIndicator,
    /// Summary table with key-value rows.
    SummaryTable {
        /// Column headers.
        headers: Vec<String>,
    },
    /// Plain text/markdown widget.
    Text {
        /// Whether the content is markdown.
        is_markdown: bool,
    },
}

/// Severity thresholds for colouring a widget.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WidgetThresholds {
    /// Value at or above which the widget turns *warning*.
    pub warning: f64,
    /// Value at or above which the widget turns *critical*.
    pub critical: f64,
}

impl WidgetThresholds {
    /// Create new thresholds.
    #[must_use]
    pub fn new(warning: f64, critical: f64) -> Self {
        Self { warning, critical }
    }

    /// Evaluate a value against the thresholds.
    #[must_use]
    pub fn evaluate(&self, value: f64) -> ThresholdLevel {
        if value >= self.critical {
            ThresholdLevel::Critical
        } else if value >= self.warning {
            ThresholdLevel::Warning
        } else {
            ThresholdLevel::Normal
        }
    }
}

/// Result of evaluating a value against thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdLevel {
    /// Value is within normal range.
    Normal,
    /// Value has crossed the warning threshold.
    Warning,
    /// Value has crossed the critical threshold.
    Critical,
}

/// A single data point pushed to a widget.
#[derive(Debug, Clone)]
pub struct DataPoint {
    /// Timestamp of the data point.
    pub timestamp: SystemTime,
    /// Numeric value.
    pub value: f64,
    /// Optional label.
    pub label: Option<String>,
}

impl DataPoint {
    /// Create a data point with the current time.
    #[must_use]
    pub fn now(value: f64) -> Self {
        Self {
            timestamp: SystemTime::now(),
            value,
            label: None,
        }
    }

    /// Create a data point with a label.
    pub fn with_label(value: f64, label: impl Into<String>) -> Self {
        Self {
            timestamp: SystemTime::now(),
            value,
            label: Some(label.into()),
        }
    }
}

/// Configuration for a dashboard widget.
#[derive(Debug, Clone)]
pub struct WidgetConfig {
    /// Widget identifier.
    pub id: WidgetId,
    /// Human-readable title.
    pub title: String,
    /// Kind of widget.
    pub kind: WidgetKind,
    /// Grid position.
    pub position: GridPosition,
    /// Optional thresholds for colour coding.
    pub thresholds: Option<WidgetThresholds>,
    /// Refresh interval.
    pub refresh_interval: Duration,
}

impl WidgetConfig {
    /// Create a new widget configuration.
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        kind: WidgetKind,
        position: GridPosition,
    ) -> Self {
        Self {
            id: WidgetId::new(id),
            title: title.into(),
            kind,
            position,
            thresholds: None,
            refresh_interval: Duration::from_secs(5),
        }
    }

    /// Set thresholds.
    #[must_use]
    pub fn with_thresholds(mut self, thresholds: WidgetThresholds) -> Self {
        self.thresholds = Some(thresholds);
        self
    }

    /// Set refresh interval.
    #[must_use]
    pub fn with_refresh_interval(mut self, interval: Duration) -> Self {
        self.refresh_interval = interval;
        self
    }
}

/// Runtime state of a widget including buffered data.
#[derive(Debug)]
pub struct WidgetState {
    /// Configuration snapshot.
    pub config: WidgetConfig,
    /// Buffered data points (most recent last).
    pub data: Vec<DataPoint>,
    /// Maximum retained points (ring behaviour).
    pub max_points: usize,
    /// Last update time.
    pub last_updated: Option<SystemTime>,
}

impl WidgetState {
    /// Create from config with default buffer size.
    #[must_use]
    pub fn from_config(config: WidgetConfig) -> Self {
        let max_points = match &config.kind {
            WidgetKind::Sparkline { max_points, .. } => *max_points,
            _ => 100,
        };
        Self {
            config,
            data: Vec::new(),
            max_points,
            last_updated: None,
        }
    }

    /// Push a new data point; trims if over capacity.
    pub fn push(&mut self, point: DataPoint) {
        self.data.push(point);
        if self.data.len() > self.max_points {
            self.data.remove(0);
        }
        self.last_updated = Some(SystemTime::now());
    }

    /// Return the latest value, if any.
    #[must_use]
    pub fn latest_value(&self) -> Option<f64> {
        self.data.last().map(|p| p.value)
    }

    /// Compute the mean of all buffered values.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        if self.data.is_empty() {
            return None;
        }
        let sum: f64 = self.data.iter().map(|p| p.value).sum();
        Some(sum / self.data.len() as f64)
    }

    /// Return (min, max) over buffered data.
    #[must_use]
    pub fn min_max(&self) -> Option<(f64, f64)> {
        if self.data.is_empty() {
            return None;
        }
        let mut mn = f64::MAX;
        let mut mx = f64::MIN;
        for p in &self.data {
            if p.value < mn {
                mn = p.value;
            }
            if p.value > mx {
                mx = p.value;
            }
        }
        Some((mn, mx))
    }

    /// Clear all buffered data.
    pub fn clear(&mut self) {
        self.data.clear();
        self.last_updated = None;
    }

    /// Return how many data points are buffered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Evaluate the latest value against configured thresholds.
    #[must_use]
    pub fn threshold_level(&self) -> Option<ThresholdLevel> {
        let val = self.latest_value()?;
        let thresholds = self.config.thresholds.as_ref()?;
        Some(thresholds.evaluate(val))
    }
}

/// A dashboard composed of multiple widgets.
#[derive(Debug)]
pub struct Dashboard {
    /// Dashboard name.
    pub name: String,
    /// Grid columns.
    pub columns: u32,
    /// Grid rows.
    pub rows: u32,
    /// Widgets keyed by id.
    widgets: HashMap<String, WidgetState>,
}

impl Dashboard {
    /// Create an empty dashboard.
    pub fn new(name: impl Into<String>, columns: u32, rows: u32) -> Self {
        Self {
            name: name.into(),
            columns: columns.max(1),
            rows: rows.max(1),
            widgets: HashMap::new(),
        }
    }

    /// Add a widget. Returns `false` if the id already exists.
    pub fn add_widget(&mut self, config: WidgetConfig) -> bool {
        let key = config.id.as_str().to_owned();
        if self.widgets.contains_key(&key) {
            return false;
        }
        self.widgets.insert(key, WidgetState::from_config(config));
        true
    }

    /// Remove a widget by id. Returns `true` if found.
    pub fn remove_widget(&mut self, id: &str) -> bool {
        self.widgets.remove(id).is_some()
    }

    /// Push a data point to a specific widget. Returns `false` if not found.
    pub fn push_data(&mut self, widget_id: &str, point: DataPoint) -> bool {
        if let Some(w) = self.widgets.get_mut(widget_id) {
            w.push(point);
            true
        } else {
            false
        }
    }

    /// Get an immutable reference to a widget state.
    #[must_use]
    pub fn widget(&self, id: &str) -> Option<&WidgetState> {
        self.widgets.get(id)
    }

    /// Return the number of widgets.
    #[must_use]
    pub fn widget_count(&self) -> usize {
        self.widgets.len()
    }

    /// Return all widget ids.
    #[must_use]
    pub fn widget_ids(&self) -> Vec<String> {
        self.widgets.keys().cloned().collect()
    }

    /// Check if any widget is in critical state.
    #[must_use]
    pub fn has_critical(&self) -> bool {
        self.widgets
            .values()
            .any(|w| w.threshold_level() == Some(ThresholdLevel::Critical))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_widget_id_display() {
        let id = WidgetId::new("cpu_gauge");
        assert_eq!(id.as_str(), "cpu_gauge");
        assert_eq!(format!("{id}"), "cpu_gauge");
    }

    #[test]
    fn test_grid_position_cell_count() {
        let pos = GridPosition::new(0, 0, 3, 2);
        assert_eq!(pos.cell_count(), 6);
    }

    #[test]
    fn test_grid_position_overlap() {
        let a = GridPosition::new(0, 0, 2, 2);
        let b = GridPosition::new(1, 1, 2, 2);
        assert!(a.overlaps(&b));

        let c = GridPosition::new(3, 3, 1, 1);
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_grid_position_no_overlap_adjacent() {
        let a = GridPosition::new(0, 0, 2, 2);
        let b = GridPosition::new(2, 0, 2, 2);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_thresholds_normal() {
        let t = WidgetThresholds::new(70.0, 90.0);
        assert_eq!(t.evaluate(50.0), ThresholdLevel::Normal);
    }

    #[test]
    fn test_thresholds_warning() {
        let t = WidgetThresholds::new(70.0, 90.0);
        assert_eq!(t.evaluate(75.0), ThresholdLevel::Warning);
    }

    #[test]
    fn test_thresholds_critical() {
        let t = WidgetThresholds::new(70.0, 90.0);
        assert_eq!(t.evaluate(95.0), ThresholdLevel::Critical);
    }

    #[test]
    fn test_widget_state_push_and_trim() {
        let cfg = WidgetConfig::new(
            "test",
            "Test",
            WidgetKind::Sparkline {
                max_points: 3,
                color: "#ff0000".into(),
            },
            GridPosition::new(0, 0, 1, 1),
        );
        let mut state = WidgetState::from_config(cfg);
        state.push(DataPoint::now(1.0));
        state.push(DataPoint::now(2.0));
        state.push(DataPoint::now(3.0));
        state.push(DataPoint::now(4.0));
        assert_eq!(state.len(), 3);
        assert_eq!(state.latest_value(), Some(4.0));
    }

    #[test]
    fn test_widget_state_mean() {
        let cfg = WidgetConfig::new(
            "m",
            "Mean",
            WidgetKind::Gauge {
                min_value: 0.0,
                max_value: 100.0,
            },
            GridPosition::new(0, 0, 1, 1),
        );
        let mut state = WidgetState::from_config(cfg);
        state.push(DataPoint::now(10.0));
        state.push(DataPoint::now(20.0));
        state.push(DataPoint::now(30.0));
        let mean = state.mean().expect("mean should succeed");
        assert!((mean - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_widget_state_min_max() {
        let cfg = WidgetConfig::new(
            "mm",
            "MinMax",
            WidgetKind::StatusIndicator,
            GridPosition::new(0, 0, 1, 1),
        );
        let mut state = WidgetState::from_config(cfg);
        state.push(DataPoint::now(5.0));
        state.push(DataPoint::now(15.0));
        state.push(DataPoint::now(10.0));
        let (mn, mx) = state.min_max().expect("min_max should succeed");
        assert!((mn - 5.0).abs() < 1e-9);
        assert!((mx - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_widget_state_clear() {
        let cfg = WidgetConfig::new(
            "c",
            "C",
            WidgetKind::StatusIndicator,
            GridPosition::new(0, 0, 1, 1),
        );
        let mut state = WidgetState::from_config(cfg);
        state.push(DataPoint::now(1.0));
        assert!(!state.is_empty());
        state.clear();
        assert!(state.is_empty());
        assert!(state.latest_value().is_none());
    }

    #[test]
    fn test_dashboard_add_remove_widget() {
        let mut dash = Dashboard::new("Test Dashboard", 12, 8);
        let cfg = WidgetConfig::new(
            "w1",
            "Widget 1",
            WidgetKind::StatusIndicator,
            GridPosition::new(0, 0, 2, 2),
        );
        assert!(dash.add_widget(cfg));
        assert_eq!(dash.widget_count(), 1);

        // duplicate id should fail
        let cfg2 = WidgetConfig::new(
            "w1",
            "Dup",
            WidgetKind::StatusIndicator,
            GridPosition::new(5, 5, 1, 1),
        );
        assert!(!dash.add_widget(cfg2));
        assert_eq!(dash.widget_count(), 1);

        assert!(dash.remove_widget("w1"));
        assert_eq!(dash.widget_count(), 0);
    }

    #[test]
    fn test_dashboard_push_data() {
        let mut dash = Dashboard::new("D", 12, 8);
        let cfg = WidgetConfig::new(
            "cpu",
            "CPU",
            WidgetKind::Gauge {
                min_value: 0.0,
                max_value: 100.0,
            },
            GridPosition::new(0, 0, 1, 1),
        );
        dash.add_widget(cfg);
        assert!(dash.push_data("cpu", DataPoint::now(42.0)));
        assert!(!dash.push_data("missing", DataPoint::now(0.0)));
        assert_eq!(
            dash.widget("cpu")
                .expect("widget should succeed")
                .latest_value(),
            Some(42.0)
        );
    }

    #[test]
    fn test_dashboard_has_critical() {
        let mut dash = Dashboard::new("D", 4, 4);
        let cfg = WidgetConfig::new(
            "temp",
            "Temp",
            WidgetKind::Gauge {
                min_value: 0.0,
                max_value: 120.0,
            },
            GridPosition::new(0, 0, 1, 1),
        )
        .with_thresholds(WidgetThresholds::new(70.0, 90.0));
        dash.add_widget(cfg);
        dash.push_data("temp", DataPoint::now(50.0));
        assert!(!dash.has_critical());

        dash.push_data("temp", DataPoint::now(95.0));
        assert!(dash.has_critical());
    }

    #[test]
    fn test_data_point_with_label() {
        let dp = DataPoint::with_label(3.14, "pi");
        assert!((dp.value - 3.14).abs() < 1e-9);
        assert_eq!(dp.label.as_deref(), Some("pi"));
    }

    #[test]
    fn test_widget_config_refresh_interval() {
        let cfg = WidgetConfig::new(
            "x",
            "X",
            WidgetKind::Text { is_markdown: true },
            GridPosition::new(0, 0, 1, 1),
        )
        .with_refresh_interval(Duration::from_secs(10));
        assert_eq!(cfg.refresh_interval, Duration::from_secs(10));
    }
}
