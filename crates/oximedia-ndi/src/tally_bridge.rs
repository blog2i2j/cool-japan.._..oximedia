//! NDI tally bridge — convert NDI tally messages to/from generic tally states
//! and aggregate tally across multiple simultaneous NDI sources.
//!
//! The tally bridge sits between the low-level [`crate::tally_bus`] (which deals with
//! NDI wire bytes) and higher-level switcher or automation systems that may use their
//! own tally representation.  It provides:
//!
//! * [`GenericTallyState`] — switcher-agnostic program/preview/off enum compatible with
//!   Ross Video, Blackmagic, and GVG-style tally systems.
//! * [`TallyBridgeMapping`] — bidirectional conversion between NDI wire bytes and
//!   `GenericTallyState`.
//! * [`MultiSourceTallyAggregator`] — collects tally from N NDI sources and
//!   produces a merged output tally.
//! * [`TallyBridgeEvent`] — change notification emitted when any source changes state.
//! * [`TallyBridgeConfig`] — policy controls (e.g. priority ordering for conflict resolution).

#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// GenericTallyState
// ─────────────────────────────────────────────────────────────────────────────

/// Switcher-agnostic tally state used by external systems.
///
/// This is deliberately kept separate from [`crate::tally_bus::TallyLight`] so that
/// the bridge can evolve independently of the NDI wire format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenericTallyState {
    /// No active tally.
    Off,
    /// Source is on preview (green).
    Preview,
    /// Source is on program (red / on-air).
    Program,
    /// Source is simultaneously on program and preview.
    ProgramAndPreview,
}

impl GenericTallyState {
    /// Returns `true` when the source is on program.
    pub fn is_program(self) -> bool {
        matches!(self, Self::Program | Self::ProgramAndPreview)
    }

    /// Returns `true` when the source is on preview.
    pub fn is_preview(self) -> bool {
        matches!(self, Self::Preview | Self::ProgramAndPreview)
    }

    /// Returns `true` when any tally is active.
    pub fn is_active(self) -> bool {
        self != Self::Off
    }

    /// Display-friendly name of this state.
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Preview => "Preview",
            Self::Program => "Program",
            Self::ProgramAndPreview => "Program+Preview",
        }
    }
}

impl Default for GenericTallyState {
    fn default() -> Self {
        Self::Off
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TallyBridgeMapping
// ─────────────────────────────────────────────────────────────────────────────

/// Bidirectional conversion table between NDI tally bytes and [`GenericTallyState`].
///
/// The NDI wire encoding uses a 2-bit byte where bit 0 = program, bit 1 = preview.
pub struct TallyBridgeMapping;

impl TallyBridgeMapping {
    /// Decode a raw NDI tally byte into a [`GenericTallyState`].
    ///
    /// ```
    /// use oximedia_ndi::tally_bridge::{TallyBridgeMapping, GenericTallyState};
    ///
    /// assert_eq!(TallyBridgeMapping::from_ndi_byte(0x01), GenericTallyState::Program);
    /// assert_eq!(TallyBridgeMapping::from_ndi_byte(0x02), GenericTallyState::Preview);
    /// assert_eq!(TallyBridgeMapping::from_ndi_byte(0x03), GenericTallyState::ProgramAndPreview);
    /// assert_eq!(TallyBridgeMapping::from_ndi_byte(0x00), GenericTallyState::Off);
    /// ```
    pub fn from_ndi_byte(byte: u8) -> GenericTallyState {
        let prog = (byte & 0x01) != 0;
        let prev = (byte & 0x02) != 0;
        match (prog, prev) {
            (true, true) => GenericTallyState::ProgramAndPreview,
            (true, false) => GenericTallyState::Program,
            (false, true) => GenericTallyState::Preview,
            (false, false) => GenericTallyState::Off,
        }
    }

    /// Encode a [`GenericTallyState`] into the NDI tally wire byte.
    ///
    /// ```
    /// use oximedia_ndi::tally_bridge::{TallyBridgeMapping, GenericTallyState};
    ///
    /// assert_eq!(TallyBridgeMapping::to_ndi_byte(GenericTallyState::Program), 0x01);
    /// assert_eq!(TallyBridgeMapping::to_ndi_byte(GenericTallyState::Preview), 0x02);
    /// assert_eq!(TallyBridgeMapping::to_ndi_byte(GenericTallyState::ProgramAndPreview), 0x03);
    /// assert_eq!(TallyBridgeMapping::to_ndi_byte(GenericTallyState::Off), 0x00);
    /// ```
    pub fn to_ndi_byte(state: GenericTallyState) -> u8 {
        let prog: u8 = if state.is_program() { 0x01 } else { 0x00 };
        let prev: u8 = if state.is_preview() { 0x02 } else { 0x00 };
        prog | prev
    }

    /// Parse a tally state from an NDI XML metadata string.
    ///
    /// Expected format: `<tally program="true|false" preview="true|false"/>`
    pub fn from_xml(xml: &str) -> GenericTallyState {
        let prog = xml.contains("program=\"true\"");
        let prev = xml.contains("preview=\"true\"");
        match (prog, prev) {
            (true, true) => GenericTallyState::ProgramAndPreview,
            (true, false) => GenericTallyState::Program,
            (false, true) => GenericTallyState::Preview,
            (false, false) => GenericTallyState::Off,
        }
    }

    /// Serialize a [`GenericTallyState`] to an NDI XML tally element.
    pub fn to_xml(state: GenericTallyState) -> String {
        format!(
            r#"<tally program="{}" preview="{}"/>"#,
            state.is_program(),
            state.is_preview()
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TallySourcePriority
// ─────────────────────────────────────────────────────────────────────────────

/// Priority level for a tally source used in conflict resolution.
///
/// Higher priority sources win when multiple sources report conflicting states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TallySourcePriority {
    /// Lowest priority; state can always be overridden.
    Low = 0,
    /// Normal priority (default for auto-discovered sources).
    Normal = 1,
    /// High priority; overrides Normal and Low.
    High = 2,
    /// Master priority; overrides all others.
    Master = 3,
}

impl Default for TallySourcePriority {
    fn default() -> Self {
        Self::Normal
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TallyBridgeConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Policy configuration for the tally bridge.
#[derive(Debug, Clone)]
pub struct TallyBridgeConfig {
    /// When `true`, a source on program in any input sets the aggregate to Program.
    /// When `false`, program state requires the highest-priority source to agree.
    pub any_program_wins: bool,

    /// Maximum number of sources that can be registered simultaneously.
    pub max_sources: usize,
}

impl Default for TallyBridgeConfig {
    fn default() -> Self {
        Self {
            any_program_wins: true,
            max_sources: 64,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TallyBridgeEvent
// ─────────────────────────────────────────────────────────────────────────────

/// Emitted when the tally state of a source changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TallyBridgeEvent {
    /// Source name.
    pub source: String,
    /// Previous state.
    pub old_state: GenericTallyState,
    /// New state.
    pub new_state: GenericTallyState,
}

// ─────────────────────────────────────────────────────────────────────────────
// SourceEntry (internal)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SourceEntry {
    state: GenericTallyState,
    priority: TallySourcePriority,
}

// ─────────────────────────────────────────────────────────────────────────────
// MultiSourceTallyAggregator
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregates tally state from multiple NDI sources into a single output state.
///
/// # Aggregation logic
///
/// With `any_program_wins = true` (default):
/// * If **any** registered source is on program → aggregate is Program.
/// * Else if any source is on preview → aggregate is Preview.
/// * Else → Off.
///
/// With `any_program_wins = false`:
/// * Only the highest-priority source's program state is used.
/// * Preview follows the same rule independently.
#[derive(Debug)]
pub struct MultiSourceTallyAggregator {
    sources: HashMap<String, SourceEntry>,
    config: TallyBridgeConfig,
    event_log: Vec<TallyBridgeEvent>,
}

impl MultiSourceTallyAggregator {
    /// Create an aggregator with default configuration.
    pub fn new() -> Self {
        Self::with_config(TallyBridgeConfig::default())
    }

    /// Create an aggregator with custom configuration.
    pub fn with_config(config: TallyBridgeConfig) -> Self {
        Self {
            sources: HashMap::new(),
            config,
            event_log: Vec::new(),
        }
    }

    /// Register a source with a given priority.
    ///
    /// Returns `Err` if the source registry is full.
    pub fn register(
        &mut self,
        source: &str,
        priority: TallySourcePriority,
    ) -> Result<(), String> {
        if !self.sources.contains_key(source)
            && self.sources.len() >= self.config.max_sources
        {
            return Err(format!(
                "tally bridge: source registry full ({} sources)",
                self.config.max_sources
            ));
        }
        self.sources.entry(source.to_string()).or_insert(SourceEntry {
            state: GenericTallyState::Off,
            priority,
        });
        Ok(())
    }

    /// Remove a source from the aggregator.
    pub fn unregister(&mut self, source: &str) {
        self.sources.remove(source);
    }

    /// Update the tally state for a named source.
    ///
    /// Records a [`TallyBridgeEvent`] if the state actually changed.
    /// Returns `Err` if the source is not registered.
    pub fn update(
        &mut self,
        source: &str,
        new_state: GenericTallyState,
    ) -> Result<(), String> {
        let entry = self
            .sources
            .get_mut(source)
            .ok_or_else(|| format!("tally bridge: unknown source '{source}'"))?;

        let old_state = entry.state;
        if old_state != new_state {
            self.event_log.push(TallyBridgeEvent {
                source: source.to_string(),
                old_state,
                new_state,
            });
            entry.state = new_state;
        }
        Ok(())
    }

    /// Update via raw NDI tally byte.
    pub fn update_from_byte(&mut self, source: &str, byte: u8) -> Result<(), String> {
        let state = TallyBridgeMapping::from_ndi_byte(byte);
        self.update(source, state)
    }

    /// Update via NDI XML tally string.
    pub fn update_from_xml(&mut self, source: &str, xml: &str) -> Result<(), String> {
        let state = TallyBridgeMapping::from_xml(xml);
        self.update(source, state)
    }

    /// Get the current state of a specific source.
    pub fn source_state(&self, source: &str) -> Option<GenericTallyState> {
        self.sources.get(source).map(|e| e.state)
    }

    /// Compute the merged aggregate tally across all registered sources.
    pub fn aggregate(&self) -> GenericTallyState {
        if self.config.any_program_wins {
            self.aggregate_any_wins()
        } else {
            self.aggregate_priority_based()
        }
    }

    fn aggregate_any_wins(&self) -> GenericTallyState {
        let mut prog = false;
        let mut prev = false;
        for entry in self.sources.values() {
            if entry.state.is_program() {
                prog = true;
            }
            if entry.state.is_preview() {
                prev = true;
            }
        }
        match (prog, prev) {
            (true, true) => GenericTallyState::ProgramAndPreview,
            (true, false) => GenericTallyState::Program,
            (false, true) => GenericTallyState::Preview,
            (false, false) => GenericTallyState::Off,
        }
    }

    fn aggregate_priority_based(&self) -> GenericTallyState {
        let max_priority = self
            .sources
            .values()
            .map(|e| e.priority)
            .max()
            .unwrap_or(TallySourcePriority::Low);

        let mut prog = false;
        let mut prev = false;

        for entry in self.sources.values().filter(|e| e.priority == max_priority) {
            if entry.state.is_program() {
                prog = true;
            }
            if entry.state.is_preview() {
                prev = true;
            }
        }

        match (prog, prev) {
            (true, true) => GenericTallyState::ProgramAndPreview,
            (true, false) => GenericTallyState::Program,
            (false, true) => GenericTallyState::Preview,
            (false, false) => GenericTallyState::Off,
        }
    }

    /// Drain all pending change events.
    pub fn drain_events(&mut self) -> Vec<TallyBridgeEvent> {
        std::mem::take(&mut self.event_log)
    }

    /// All sources currently on program.
    pub fn program_sources(&self) -> Vec<&str> {
        self.sources
            .iter()
            .filter(|(_, e)| e.state.is_program())
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// All sources currently on preview.
    pub fn preview_sources(&self) -> Vec<&str> {
        self.sources
            .iter()
            .filter(|(_, e)| e.state.is_preview())
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// Number of registered sources.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Reset all tally states to Off and clear the event log.
    pub fn reset(&mut self) {
        for entry in self.sources.values_mut() {
            entry.state = GenericTallyState::Off;
        }
        self.event_log.clear();
    }
}

impl Default for MultiSourceTallyAggregator {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── GenericTallyState ─────────────────────────────────────────────────────

    #[test]
    fn test_generic_state_predicates() {
        assert!(GenericTallyState::Program.is_program());
        assert!(GenericTallyState::ProgramAndPreview.is_program());
        assert!(!GenericTallyState::Preview.is_program());
        assert!(!GenericTallyState::Off.is_program());

        assert!(GenericTallyState::Preview.is_preview());
        assert!(GenericTallyState::ProgramAndPreview.is_preview());
        assert!(!GenericTallyState::Program.is_preview());
        assert!(!GenericTallyState::Off.is_preview());

        assert!(!GenericTallyState::Off.is_active());
        assert!(GenericTallyState::Program.is_active());
    }

    #[test]
    fn test_generic_state_labels() {
        assert_eq!(GenericTallyState::Off.label(), "Off");
        assert_eq!(GenericTallyState::Program.label(), "Program");
        assert_eq!(GenericTallyState::Preview.label(), "Preview");
        assert_eq!(GenericTallyState::ProgramAndPreview.label(), "Program+Preview");
    }

    // ── TallyBridgeMapping ────────────────────────────────────────────────────

    #[test]
    fn test_mapping_byte_roundtrip() {
        for state in [
            GenericTallyState::Off,
            GenericTallyState::Program,
            GenericTallyState::Preview,
            GenericTallyState::ProgramAndPreview,
        ] {
            let byte = TallyBridgeMapping::to_ndi_byte(state);
            let decoded = TallyBridgeMapping::from_ndi_byte(byte);
            assert_eq!(decoded, state, "roundtrip failed for {state:?}");
        }
    }

    #[test]
    fn test_mapping_xml_roundtrip() {
        for state in [
            GenericTallyState::Off,
            GenericTallyState::Program,
            GenericTallyState::Preview,
            GenericTallyState::ProgramAndPreview,
        ] {
            let xml = TallyBridgeMapping::to_xml(state);
            let decoded = TallyBridgeMapping::from_xml(&xml);
            assert_eq!(decoded, state, "XML roundtrip failed for {state:?}");
        }
    }

    #[test]
    fn test_mapping_known_bytes() {
        assert_eq!(TallyBridgeMapping::from_ndi_byte(0x00), GenericTallyState::Off);
        assert_eq!(TallyBridgeMapping::from_ndi_byte(0x01), GenericTallyState::Program);
        assert_eq!(TallyBridgeMapping::from_ndi_byte(0x02), GenericTallyState::Preview);
        assert_eq!(TallyBridgeMapping::from_ndi_byte(0x03), GenericTallyState::ProgramAndPreview);
    }

    // ── MultiSourceTallyAggregator ────────────────────────────────────────────

    #[test]
    fn test_aggregator_register_and_update() {
        let mut agg = MultiSourceTallyAggregator::new();
        agg.register("Cam1", TallySourcePriority::Normal).unwrap();
        agg.update("Cam1", GenericTallyState::Program).unwrap();
        assert_eq!(agg.source_state("Cam1"), Some(GenericTallyState::Program));
    }

    #[test]
    fn test_aggregator_any_wins_program() {
        let mut agg = MultiSourceTallyAggregator::new();
        agg.register("A", TallySourcePriority::Normal).unwrap();
        agg.register("B", TallySourcePriority::Normal).unwrap();
        agg.update("A", GenericTallyState::Off).unwrap();
        agg.update("B", GenericTallyState::Program).unwrap();
        assert_eq!(agg.aggregate(), GenericTallyState::Program);
    }

    #[test]
    fn test_aggregator_all_off() {
        let mut agg = MultiSourceTallyAggregator::new();
        agg.register("A", TallySourcePriority::Normal).unwrap();
        agg.register("B", TallySourcePriority::Normal).unwrap();
        assert_eq!(agg.aggregate(), GenericTallyState::Off);
    }

    #[test]
    fn test_aggregator_events_emitted_on_change() {
        let mut agg = MultiSourceTallyAggregator::new();
        agg.register("Cam1", TallySourcePriority::Normal).unwrap();
        agg.update("Cam1", GenericTallyState::Program).unwrap();
        let events = agg.drain_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].new_state, GenericTallyState::Program);
        assert_eq!(events[0].old_state, GenericTallyState::Off);
    }

    #[test]
    fn test_aggregator_no_event_on_same_state() {
        let mut agg = MultiSourceTallyAggregator::new();
        agg.register("Cam1", TallySourcePriority::Normal).unwrap();
        agg.update("Cam1", GenericTallyState::Program).unwrap();
        agg.drain_events();
        agg.update("Cam1", GenericTallyState::Program).unwrap();
        assert!(agg.drain_events().is_empty());
    }

    #[test]
    fn test_aggregator_update_from_byte() {
        let mut agg = MultiSourceTallyAggregator::new();
        agg.register("Cam1", TallySourcePriority::Normal).unwrap();
        agg.update_from_byte("Cam1", 0x01).unwrap();
        assert_eq!(agg.source_state("Cam1"), Some(GenericTallyState::Program));
    }

    #[test]
    fn test_aggregator_priority_based_conflict() {
        let mut cfg = TallyBridgeConfig::default();
        cfg.any_program_wins = false;
        let mut agg = MultiSourceTallyAggregator::with_config(cfg);

        agg.register("Low", TallySourcePriority::Low).unwrap();
        agg.register("High", TallySourcePriority::High).unwrap();

        // Low source says Program; High source says Off
        agg.update("Low", GenericTallyState::Program).unwrap();
        agg.update("High", GenericTallyState::Off).unwrap();

        // With priority-based, only the High source controls outcome
        assert_eq!(agg.aggregate(), GenericTallyState::Off);
    }

    #[test]
    fn test_aggregator_reset() {
        let mut agg = MultiSourceTallyAggregator::new();
        agg.register("Cam1", TallySourcePriority::Normal).unwrap();
        agg.update("Cam1", GenericTallyState::Program).unwrap();
        agg.reset();
        assert_eq!(agg.source_state("Cam1"), Some(GenericTallyState::Off));
        assert_eq!(agg.aggregate(), GenericTallyState::Off);
        assert!(agg.drain_events().is_empty());
    }

    #[test]
    fn test_aggregator_program_and_preview_sources() {
        let mut agg = MultiSourceTallyAggregator::new();
        agg.register("A", TallySourcePriority::Normal).unwrap();
        agg.register("B", TallySourcePriority::Normal).unwrap();
        agg.register("C", TallySourcePriority::Normal).unwrap();
        agg.update("A", GenericTallyState::Program).unwrap();
        agg.update("B", GenericTallyState::Preview).unwrap();
        agg.update("C", GenericTallyState::Off).unwrap();

        let prog = agg.program_sources();
        let prev = agg.preview_sources();
        assert_eq!(prog.len(), 1);
        assert!(prog.contains(&"A"));
        assert_eq!(prev.len(), 1);
        assert!(prev.contains(&"B"));
    }

    #[test]
    fn test_aggregator_unregister() {
        let mut agg = MultiSourceTallyAggregator::new();
        agg.register("Cam1", TallySourcePriority::Normal).unwrap();
        agg.unregister("Cam1");
        assert_eq!(agg.source_count(), 0);
        assert!(agg.source_state("Cam1").is_none());
    }
}
