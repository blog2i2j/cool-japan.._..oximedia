//! NDI tally state manager
//!
//! This module provides a high-level, self-contained tally manager that tracks
//! the on-air and preview status of NDI sources independently of the lower-level
//! private `tally` module.

#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]

/// The tally state of an NDI source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TallyStateKind {
    /// Neither on-air nor in preview.
    Off,
    /// Red light — the source is currently on air.
    OnAir,
    /// Green light — the source is currently in preview.
    Preview,
    /// Both on-air and in preview simultaneously.
    OnAirAndPreview,
}

impl TallyStateKind {
    /// Returns `true` when the source is on air (red tally active).
    pub fn is_on_air(&self) -> bool {
        matches!(self, Self::OnAir | Self::OnAirAndPreview)
    }

    /// Returns `true` when the source is in preview (green tally active).
    pub fn is_preview(&self) -> bool {
        matches!(self, Self::Preview | Self::OnAirAndPreview)
    }
}

/// Tally information for a single NDI source.
#[derive(Debug, Clone)]
pub struct TallyInfo {
    /// Name of the NDI source.
    pub source_name: String,
    /// Current tally state.
    pub state: TallyStateKind,
    /// Unix epoch (seconds) of the last state update.
    pub last_update_epoch: u64,
}

impl TallyInfo {
    /// Create a new `TallyInfo`.
    pub fn new(source_name: impl Into<String>, state: TallyStateKind, epoch: u64) -> Self {
        Self {
            source_name: source_name.into(),
            state,
            last_update_epoch: epoch,
        }
    }

    /// Returns `true` when the source is currently on air.
    pub fn is_live(&self) -> bool {
        self.state.is_on_air()
    }
}

/// An in-memory manager for NDI tally states.
#[derive(Debug, Default)]
pub struct TallyManager {
    /// All tracked sources.
    pub tallies: Vec<TallyInfo>,
}

impl TallyManager {
    /// Create an empty `TallyManager`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update (or insert) the tally state for a source.
    pub fn set_tally(&mut self, source_name: impl Into<String>, state: TallyStateKind, epoch: u64) {
        let name = source_name.into();
        if let Some(existing) = self.tallies.iter_mut().find(|t| t.source_name == name) {
            existing.state = state;
            existing.last_update_epoch = epoch;
        } else {
            self.tallies.push(TallyInfo::new(name, state, epoch));
        }
    }

    /// Retrieve the `TallyInfo` for a named source.
    pub fn get_tally(&self, source_name: &str) -> Option<&TallyInfo> {
        self.tallies.iter().find(|t| t.source_name == source_name)
    }

    /// Return all sources that are currently on air.
    pub fn on_air_sources(&self) -> Vec<&TallyInfo> {
        self.tallies
            .iter()
            .filter(|t| t.state.is_on_air())
            .collect()
    }

    /// Return all sources that are currently in preview.
    pub fn preview_sources(&self) -> Vec<&TallyInfo> {
        self.tallies
            .iter()
            .filter(|t| t.state.is_preview())
            .collect()
    }

    /// Returns `true` when at least one source is on air.
    pub fn any_on_air(&self) -> bool {
        self.tallies.iter().any(|t| t.state.is_on_air())
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- TallyStateKind ---

    #[test]
    fn test_off_not_on_air() {
        assert!(!TallyStateKind::Off.is_on_air());
    }

    #[test]
    fn test_off_not_preview() {
        assert!(!TallyStateKind::Off.is_preview());
    }

    #[test]
    fn test_on_air_is_on_air() {
        assert!(TallyStateKind::OnAir.is_on_air());
    }

    #[test]
    fn test_on_air_not_preview() {
        assert!(!TallyStateKind::OnAir.is_preview());
    }

    #[test]
    fn test_preview_not_on_air() {
        assert!(!TallyStateKind::Preview.is_on_air());
    }

    #[test]
    fn test_preview_is_preview() {
        assert!(TallyStateKind::Preview.is_preview());
    }

    #[test]
    fn test_on_air_and_preview_is_both() {
        assert!(TallyStateKind::OnAirAndPreview.is_on_air());
        assert!(TallyStateKind::OnAirAndPreview.is_preview());
    }

    // --- TallyInfo ---

    #[test]
    fn test_tally_info_is_live_on_air() {
        let t = TallyInfo::new("Cam1", TallyStateKind::OnAir, 0);
        assert!(t.is_live());
    }

    #[test]
    fn test_tally_info_is_live_off() {
        let t = TallyInfo::new("Cam1", TallyStateKind::Off, 0);
        assert!(!t.is_live());
    }

    // --- TallyManager ---

    #[test]
    fn test_set_and_get_tally() {
        let mut mgr = TallyManager::new();
        mgr.set_tally("Cam1", TallyStateKind::OnAir, 1000);
        let info = mgr.get_tally("Cam1").expect("expected tally to exist");
        assert!(info.is_live());
    }

    #[test]
    fn test_get_tally_missing() {
        let mgr = TallyManager::new();
        assert!(mgr.get_tally("Ghost").is_none());
    }

    #[test]
    fn test_set_tally_updates_existing() {
        let mut mgr = TallyManager::new();
        mgr.set_tally("Cam1", TallyStateKind::OnAir, 1000);
        mgr.set_tally("Cam1", TallyStateKind::Off, 2000);
        let info = mgr.get_tally("Cam1").expect("expected tally to exist");
        assert_eq!(info.state, TallyStateKind::Off);
        assert_eq!(info.last_update_epoch, 2000);
    }

    #[test]
    fn test_on_air_sources() {
        let mut mgr = TallyManager::new();
        mgr.set_tally("Cam1", TallyStateKind::OnAir, 0);
        mgr.set_tally("Cam2", TallyStateKind::Preview, 0);
        mgr.set_tally("Cam3", TallyStateKind::Off, 0);
        assert_eq!(mgr.on_air_sources().len(), 1);
        assert_eq!(mgr.on_air_sources()[0].source_name, "Cam1");
    }

    #[test]
    fn test_preview_sources() {
        let mut mgr = TallyManager::new();
        mgr.set_tally("Cam1", TallyStateKind::Preview, 0);
        mgr.set_tally("Cam2", TallyStateKind::OnAirAndPreview, 0);
        assert_eq!(mgr.preview_sources().len(), 2);
    }

    #[test]
    fn test_any_on_air_true() {
        let mut mgr = TallyManager::new();
        mgr.set_tally("Cam1", TallyStateKind::OnAir, 0);
        assert!(mgr.any_on_air());
    }

    #[test]
    fn test_any_on_air_false() {
        let mut mgr = TallyManager::new();
        mgr.set_tally("Cam1", TallyStateKind::Preview, 0);
        assert!(!mgr.any_on_air());
    }

    #[test]
    fn test_any_on_air_empty() {
        let mgr = TallyManager::new();
        assert!(!mgr.any_on_air());
    }
}
