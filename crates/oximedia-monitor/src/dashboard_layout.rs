//! Dashboard layout template system for configuring monitoring views.
//!
//! Provides a structured, panel-based model for defining monitoring dashboards
//! that can be serialised to/from JSON and validated for correctness.
//!
//! # Concepts
//!
//! - [`DashboardPanel`] — an individual visualisation widget (time-series
//!   graph, gauge, counter, heatmap, etc.) with a metric query string.
//! - [`DashboardRow`] — a horizontal group of panels.
//! - [`DashboardLayout`] — a complete, named dashboard definition composed
//!   of rows and panels, plus free-form tags.
//! - [`TemplateLibrary`] — built-in templates for common media-server
//!   monitoring scenarios.
//!
//! # Example
//!
//! ```rust
//! use oximedia_monitor::dashboard_layout::{TemplateLibrary, DashboardLayout};
//!
//! let tpl = TemplateLibrary::media_server_overview();
//! assert!(tpl.validate().is_empty(), "Built-in template must be valid");
//! println!("Panels: {}", tpl.total_panels());
//! ```

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// PanelType
// ---------------------------------------------------------------------------

/// Visual type of a dashboard panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PanelType {
    /// A time-series line or area chart.
    TimeSeriesGraph,
    /// A single-stat circular or linear gauge.
    Gauge,
    /// A monotonically increasing counter display.
    Counter,
    /// A two-dimensional heat map (e.g. latency distribution over time).
    Heatmap,
    /// A list of currently firing or recently resolved alerts.
    AlertList,
    /// A free-form text / Markdown panel.
    Text,
}

impl PanelType {
    /// Human-readable display name.
    #[must_use]
    pub fn display_name(self) -> &'static str {
        match self {
            Self::TimeSeriesGraph => "Time Series Graph",
            Self::Gauge => "Gauge",
            Self::Counter => "Counter",
            Self::Heatmap => "Heatmap",
            Self::AlertList => "Alert List",
            Self::Text => "Text",
        }
    }

    /// Returns `true` if this panel type requires a non-empty `metric_query`.
    #[must_use]
    pub fn requires_query(self) -> bool {
        !matches!(self, Self::Text | Self::AlertList)
    }
}

// ---------------------------------------------------------------------------
// DashboardPanel
// ---------------------------------------------------------------------------

/// A single visualisation widget within a dashboard row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DashboardPanel {
    /// Unique identifier for this panel within the dashboard.
    pub id: String,
    /// Display title shown above the panel.
    pub title: String,
    /// Visual type of this panel.
    pub panel_type: PanelType,
    /// Metric query expression (e.g. Prometheus PromQL, metric name).
    /// May be empty for `Text` and `AlertList` panels.
    pub metric_query: String,
    /// Width in abstract grid units (typically 1–24).
    pub width_units: u8,
    /// Height in abstract grid units (typically 1–12).
    pub height_units: u8,
}

impl DashboardPanel {
    /// Create a new panel.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        panel_type: PanelType,
        metric_query: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            panel_type,
            metric_query: metric_query.into(),
            width_units: 12,
            height_units: 6,
        }
    }

    /// Set explicit grid dimensions.
    #[must_use]
    pub fn with_size(mut self, width_units: u8, height_units: u8) -> Self {
        self.width_units = width_units;
        self.height_units = height_units;
        self
    }

    /// Validate this panel, returning a list of validation error messages.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.id.is_empty() {
            errors.push(format!("Panel '{}': id must not be empty", self.title));
        }
        if self.title.is_empty() {
            errors.push(format!("Panel id='{}': title must not be empty", self.id));
        }
        if self.panel_type.requires_query() && self.metric_query.is_empty() {
            errors.push(format!(
                "Panel '{}' ({}): metric_query must not be empty for panel type '{}'",
                self.title,
                self.id,
                self.panel_type.display_name()
            ));
        }
        if self.width_units == 0 {
            errors.push(format!(
                "Panel '{}': width_units must be >= 1",
                self.title
            ));
        }
        if self.height_units == 0 {
            errors.push(format!(
                "Panel '{}': height_units must be >= 1",
                self.title
            ));
        }

        errors
    }
}

// ---------------------------------------------------------------------------
// DashboardRow
// ---------------------------------------------------------------------------

/// A horizontal grouping of panels within a dashboard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DashboardRow {
    /// Row section title (may be empty for unlabelled rows).
    pub title: String,
    /// Panels in this row, displayed left-to-right.
    pub panels: Vec<DashboardPanel>,
}

impl DashboardRow {
    /// Create a new row.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            panels: Vec::new(),
        }
    }

    /// Append a panel to this row.
    pub fn push(&mut self, panel: DashboardPanel) {
        self.panels.push(panel);
    }

    /// Number of panels in this row.
    #[must_use]
    pub fn panel_count(&self) -> usize {
        self.panels.len()
    }

    /// Validate all panels in this row.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        self.panels.iter().flat_map(DashboardPanel::validate).collect()
    }
}

// ---------------------------------------------------------------------------
// DashboardLayout
// ---------------------------------------------------------------------------

/// A complete, named monitoring dashboard definition.
///
/// Serialisable to/from JSON so definitions can be stored in files or
/// databases and loaded at runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DashboardLayout {
    /// Dashboard name / identifier (should be unique within a library).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Rows of panels.
    pub rows: Vec<DashboardRow>,
    /// Free-form tags for categorisation (e.g. `"audio"`, `"encoding"`).
    pub tags: Vec<String>,
}

impl DashboardLayout {
    /// Create an empty dashboard layout.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            rows: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// Append a row to this dashboard.
    pub fn push_row(&mut self, row: DashboardRow) {
        self.rows.push(row);
    }

    /// Add a tag.
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        self.tags.push(tag.into());
    }

    /// Total number of panels across all rows.
    #[must_use]
    pub fn total_panels(&self) -> usize {
        self.rows.iter().map(DashboardRow::panel_count).sum()
    }

    /// Collect all panels whose type matches `panel_type`.
    #[must_use]
    pub fn panels_of_type(&self, panel_type: PanelType) -> Vec<&DashboardPanel> {
        self.rows
            .iter()
            .flat_map(|r| r.panels.iter())
            .filter(|p| p.panel_type == panel_type)
            .collect()
    }

    /// Validate the entire dashboard.
    ///
    /// Returns a (possibly empty) list of human-readable error descriptions.
    /// An empty list means the dashboard is considered valid.
    #[must_use]
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.name.is_empty() {
            errors.push("Dashboard name must not be empty".to_string());
        }
        if self.rows.is_empty() {
            errors.push(format!(
                "Dashboard '{}': must contain at least one row",
                self.name
            ));
        }

        // Check for duplicate panel ids across the entire dashboard.
        let mut seen_ids: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for row in &self.rows {
            for panel in &row.panels {
                if !panel.id.is_empty() && !seen_ids.insert(panel.id.as_str()) {
                    errors.push(format!(
                        "Dashboard '{}': duplicate panel id '{}'",
                        self.name, panel.id
                    ));
                }
            }
        }

        // Delegate to row / panel validation.
        for row in &self.rows {
            errors.extend(row.validate());
        }

        errors
    }

    /// Serialise to a pretty-printed JSON string.
    ///
    /// # Errors
    ///
    /// Returns a `serde_json::Error` if serialisation fails (should not
    /// happen for well-formed structs).
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialise from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns a `serde_json::Error` if the JSON is malformed or the schema
    /// does not match.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

// ---------------------------------------------------------------------------
// TemplateLibrary
// ---------------------------------------------------------------------------

/// Built-in dashboard layout templates for common media-server scenarios.
pub struct TemplateLibrary;

impl TemplateLibrary {
    /// A high-level overview dashboard for a media server.
    ///
    /// Covers CPU, memory, active sessions, encoding throughput, and alerts.
    #[must_use]
    pub fn media_server_overview() -> DashboardLayout {
        let mut layout = DashboardLayout::new(
            "media_server_overview",
            "High-level health and throughput overview for a media server node",
        );
        layout.add_tag("infrastructure");
        layout.add_tag("media");

        // Row 1 — System health
        let mut row1 = DashboardRow::new("System Health");
        row1.push(
            DashboardPanel::new(
                "cpu_usage",
                "CPU Usage",
                PanelType::Gauge,
                "system_cpu_usage_percent",
            )
            .with_size(6, 4),
        );
        row1.push(
            DashboardPanel::new(
                "memory_usage",
                "Memory Usage",
                PanelType::Gauge,
                "system_memory_used_bytes",
            )
            .with_size(6, 4),
        );
        row1.push(
            DashboardPanel::new(
                "disk_io",
                "Disk I/O",
                PanelType::TimeSeriesGraph,
                "rate(system_disk_io_bytes_total[1m])",
            )
            .with_size(12, 4),
        );
        layout.push_row(row1);

        // Row 2 — Encoding metrics
        let mut row2 = DashboardRow::new("Encoding Pipeline");
        row2.push(
            DashboardPanel::new(
                "encode_fps",
                "Encoding FPS",
                PanelType::TimeSeriesGraph,
                "encoding_frames_per_second",
            )
            .with_size(12, 6),
        );
        row2.push(
            DashboardPanel::new(
                "encode_jobs",
                "Active Jobs",
                PanelType::Counter,
                "encoding_active_jobs",
            )
            .with_size(6, 6),
        );
        row2.push(
            DashboardPanel::new(
                "encode_errors",
                "Encoding Errors",
                PanelType::Counter,
                "encoding_error_count_total",
            )
            .with_size(6, 6),
        );
        layout.push_row(row2);

        // Row 3 — Alerts
        let mut row3 = DashboardRow::new("Alerts");
        row3.push(
            DashboardPanel::new("active_alerts", "Active Alerts", PanelType::AlertList, "")
                .with_size(24, 8),
        );
        layout.push_row(row3);

        layout
    }

    /// Dashboard focused on the encoding pipeline quality metrics.
    ///
    /// Covers PSNR, SSIM, VMAF, bitrate, and latency.
    #[must_use]
    pub fn encoding_pipeline() -> DashboardLayout {
        let mut layout = DashboardLayout::new(
            "encoding_pipeline",
            "Quality and performance metrics for the encoding pipeline",
        );
        layout.add_tag("encoding");
        layout.add_tag("quality");

        // Row 1 — Quality scores
        let mut row1 = DashboardRow::new("Quality Metrics");
        row1.push(
            DashboardPanel::new(
                "psnr",
                "PSNR",
                PanelType::TimeSeriesGraph,
                "encoding_quality_psnr",
            )
            .with_size(8, 6),
        );
        row1.push(
            DashboardPanel::new(
                "ssim",
                "SSIM",
                PanelType::TimeSeriesGraph,
                "encoding_quality_ssim",
            )
            .with_size(8, 6),
        );
        row1.push(
            DashboardPanel::new(
                "vmaf",
                "VMAF",
                PanelType::Gauge,
                "encoding_quality_vmaf",
            )
            .with_size(8, 6),
        );
        layout.push_row(row1);

        // Row 2 — Bitrate & latency
        let mut row2 = DashboardRow::new("Throughput");
        row2.push(
            DashboardPanel::new(
                "video_bitrate",
                "Video Bitrate",
                PanelType::TimeSeriesGraph,
                "encoding_video_bitrate_bps",
            )
            .with_size(12, 6),
        );
        row2.push(
            DashboardPanel::new(
                "encode_latency",
                "Encode Latency (ms)",
                PanelType::Heatmap,
                "histogram_quantile(0.99, encoding_latency_ms_bucket)",
            )
            .with_size(12, 6),
        );
        layout.push_row(row2);

        // Row 3 — Notes
        let mut row3 = DashboardRow::new("Notes");
        row3.push(
            DashboardPanel::new(
                "pipeline_notes",
                "Pipeline Notes",
                PanelType::Text,
                "",
            )
            .with_size(24, 4),
        );
        layout.push_row(row3);

        layout
    }

    /// Dashboard for monitoring storage subsystem health.
    ///
    /// Covers disk usage, I/O rates, inode usage, and SMART status.
    #[must_use]
    pub fn storage_health() -> DashboardLayout {
        let mut layout = DashboardLayout::new(
            "storage_health",
            "Disk, I/O, and storage subsystem health metrics",
        );
        layout.add_tag("storage");
        layout.add_tag("infrastructure");

        // Row 1 — Capacity
        let mut row1 = DashboardRow::new("Capacity");
        row1.push(
            DashboardPanel::new(
                "disk_used_pct",
                "Disk Used %",
                PanelType::Gauge,
                "system_disk_used_percent",
            )
            .with_size(8, 5),
        );
        row1.push(
            DashboardPanel::new(
                "disk_free_bytes",
                "Disk Free",
                PanelType::Gauge,
                "system_disk_free_bytes",
            )
            .with_size(8, 5),
        );
        row1.push(
            DashboardPanel::new(
                "inode_used_pct",
                "Inode Used %",
                PanelType::Gauge,
                "system_inode_used_percent",
            )
            .with_size(8, 5),
        );
        layout.push_row(row1);

        // Row 2 — I/O rates
        let mut row2 = DashboardRow::new("I/O Activity");
        row2.push(
            DashboardPanel::new(
                "disk_read_bps",
                "Read Throughput",
                PanelType::TimeSeriesGraph,
                "rate(system_disk_read_bytes_total[1m])",
            )
            .with_size(12, 6),
        );
        row2.push(
            DashboardPanel::new(
                "disk_write_bps",
                "Write Throughput",
                PanelType::TimeSeriesGraph,
                "rate(system_disk_write_bytes_total[1m])",
            )
            .with_size(12, 6),
        );
        layout.push_row(row2);

        // Row 3 — Latency heat map
        let mut row3 = DashboardRow::new("I/O Latency");
        row3.push(
            DashboardPanel::new(
                "io_latency_heatmap",
                "I/O Latency Distribution",
                PanelType::Heatmap,
                "histogram_quantile(0.95, system_io_latency_ms_bucket)",
            )
            .with_size(24, 8),
        );
        layout.push_row(row3);

        layout
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Built-in template validation ------------------------------------

    #[test]
    fn test_media_server_overview_is_valid() {
        let tpl = TemplateLibrary::media_server_overview();
        let errors = tpl.validate();
        assert!(
            errors.is_empty(),
            "media_server_overview must be valid; got: {errors:?}"
        );
    }

    #[test]
    fn test_encoding_pipeline_is_valid() {
        let tpl = TemplateLibrary::encoding_pipeline();
        let errors = tpl.validate();
        assert!(
            errors.is_empty(),
            "encoding_pipeline must be valid; got: {errors:?}"
        );
    }

    #[test]
    fn test_storage_health_is_valid() {
        let tpl = TemplateLibrary::storage_health();
        let errors = tpl.validate();
        assert!(
            errors.is_empty(),
            "storage_health must be valid; got: {errors:?}"
        );
    }

    // ---- total_panels ---------------------------------------------------

    #[test]
    fn test_media_server_overview_panel_count() {
        let tpl = TemplateLibrary::media_server_overview();
        // Row1: 3 panels + Row2: 3 panels + Row3: 1 panel = 7 panels.
        assert_eq!(tpl.total_panels(), 7);
    }

    #[test]
    fn test_encoding_pipeline_panel_count() {
        let tpl = TemplateLibrary::encoding_pipeline();
        // Row1: 3 panels + Row2: 2 panels + Row3: 1 panel = 6 panels.
        assert_eq!(tpl.total_panels(), 6);
    }

    #[test]
    fn test_storage_health_panel_count() {
        let tpl = TemplateLibrary::storage_health();
        // Row1: 3 panels + Row2: 2 panels + Row3: 1 panel = 6 panels.
        assert_eq!(tpl.total_panels(), 6);
    }

    // ---- panels_of_type -------------------------------------------------

    #[test]
    fn test_panels_of_type_gauge() {
        let tpl = TemplateLibrary::media_server_overview();
        let gauges = tpl.panels_of_type(PanelType::Gauge);
        // cpu_usage + memory_usage = 2 gauges.
        assert_eq!(gauges.len(), 2);
        let titles: Vec<&str> = gauges.iter().map(|p| p.title.as_str()).collect();
        assert!(titles.contains(&"CPU Usage"));
        assert!(titles.contains(&"Memory Usage"));
    }

    #[test]
    fn test_panels_of_type_alert_list() {
        let tpl = TemplateLibrary::media_server_overview();
        let alert_panels = tpl.panels_of_type(PanelType::AlertList);
        assert_eq!(alert_panels.len(), 1);
        assert_eq!(alert_panels[0].id, "active_alerts");
    }

    #[test]
    fn test_panels_of_type_heatmap() {
        let tpl = TemplateLibrary::encoding_pipeline();
        let heatmaps = tpl.panels_of_type(PanelType::Heatmap);
        assert_eq!(heatmaps.len(), 1);
        assert_eq!(heatmaps[0].id, "encode_latency");
    }

    #[test]
    fn test_panels_of_type_returns_empty_for_absent() {
        let layout = DashboardLayout::new("empty", "no panels");
        // No panels, so any type query returns empty.
        assert!(layout.panels_of_type(PanelType::TimeSeriesGraph).is_empty());
    }

    // ---- validate error detection ----------------------------------------

    #[test]
    fn test_validate_empty_name_error() {
        let mut layout = DashboardLayout::new("", "no name");
        let mut row = DashboardRow::new("Row");
        row.push(DashboardPanel::new("p1", "Panel", PanelType::Gauge, "some_metric"));
        layout.push_row(row);
        let errors = layout.validate();
        assert!(
            errors.iter().any(|e| e.contains("name must not be empty")),
            "Should report empty name error"
        );
    }

    #[test]
    fn test_validate_empty_rows_error() {
        let layout = DashboardLayout::new("my-dashboard", "no rows");
        let errors = layout.validate();
        assert!(
            errors.iter().any(|e| e.contains("at least one row")),
            "Should report empty rows error"
        );
    }

    #[test]
    fn test_validate_panel_requires_query() {
        let mut layout = DashboardLayout::new("dash", "desc");
        let mut row = DashboardRow::new("Row 1");
        // TimeSeriesGraph requires a metric_query — leave it empty.
        row.push(DashboardPanel::new(
            "p1",
            "Empty Query Panel",
            PanelType::TimeSeriesGraph,
            "",
        ));
        layout.push_row(row);
        let errors = layout.validate();
        assert!(
            errors.iter().any(|e| e.contains("metric_query")),
            "Should report missing metric_query error"
        );
    }

    #[test]
    fn test_validate_duplicate_panel_ids() {
        let mut layout = DashboardLayout::new("dash", "desc");
        let mut row = DashboardRow::new("Row 1");
        row.push(DashboardPanel::new("p1", "Panel A", PanelType::Gauge, "metric_a"));
        row.push(DashboardPanel::new("p1", "Panel B", PanelType::Gauge, "metric_b"));
        layout.push_row(row);
        let errors = layout.validate();
        assert!(
            errors.iter().any(|e| e.contains("duplicate panel id")),
            "Should report duplicate panel id error"
        );
    }

    #[test]
    fn test_validate_zero_width_error() {
        let mut layout = DashboardLayout::new("dash", "desc");
        let mut row = DashboardRow::new("Row 1");
        let panel = DashboardPanel {
            id: "p1".to_string(),
            title: "Panel".to_string(),
            panel_type: PanelType::Gauge,
            metric_query: "m".to_string(),
            width_units: 0,
            height_units: 4,
        };
        row.push(panel);
        layout.push_row(row);
        let errors = layout.validate();
        assert!(
            errors.iter().any(|e| e.contains("width_units")),
            "Should report zero width error"
        );
    }

    // ---- JSON round-trip -------------------------------------------------

    #[test]
    fn test_json_round_trip() {
        let original = TemplateLibrary::encoding_pipeline();
        let json = original.to_json_pretty().expect("serialise ok");
        let restored = DashboardLayout::from_json(&json).expect("deserialise ok");
        assert_eq!(original.name, restored.name);
        assert_eq!(original.total_panels(), restored.total_panels());
    }

    #[test]
    fn test_json_contains_panel_type() {
        let tpl = TemplateLibrary::storage_health();
        let json = tpl.to_json_pretty().expect("serialise ok");
        assert!(
            json.contains("time_series_graph") || json.contains("TimeSeriesGraph"),
            "JSON must contain panel type identifier"
        );
    }

    // ---- PanelType helpers -----------------------------------------------

    #[test]
    fn test_panel_type_requires_query() {
        assert!(PanelType::TimeSeriesGraph.requires_query());
        assert!(PanelType::Gauge.requires_query());
        assert!(PanelType::Counter.requires_query());
        assert!(PanelType::Heatmap.requires_query());
        assert!(!PanelType::AlertList.requires_query());
        assert!(!PanelType::Text.requires_query());
    }

    #[test]
    fn test_panel_type_display_name_non_empty() {
        for pt in [
            PanelType::TimeSeriesGraph,
            PanelType::Gauge,
            PanelType::Counter,
            PanelType::Heatmap,
            PanelType::AlertList,
            PanelType::Text,
        ] {
            assert!(
                !pt.display_name().is_empty(),
                "display_name() must not be empty"
            );
        }
    }

    // ---- DashboardRow helpers -------------------------------------------

    #[test]
    fn test_row_panel_count() {
        let mut row = DashboardRow::new("R");
        assert_eq!(row.panel_count(), 0);
        row.push(DashboardPanel::new("p1", "P1", PanelType::Gauge, "m"));
        row.push(DashboardPanel::new("p2", "P2", PanelType::Counter, "n"));
        assert_eq!(row.panel_count(), 2);
    }

    // ---- Tags -----------------------------------------------------------

    #[test]
    fn test_tags_stored() {
        let tpl = TemplateLibrary::media_server_overview();
        assert!(tpl.tags.contains(&"infrastructure".to_string()));
        assert!(tpl.tags.contains(&"media".to_string()));
    }
}
