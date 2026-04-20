#![allow(dead_code)]
//! # Signal Chain
//!
//! Models the ordered processing chain that a video/audio signal passes
//! through on its way from source to output: input → process stages →
//! output. Supports stage insertion, removal, bypass, and chain validation.

use std::collections::VecDeque;

/// Processing stage category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageKind {
    /// Input source (capture card, file reader, stream receiver).
    Input,
    /// Video processing (scale, colour correct, deinterlace …).
    VideoProcess,
    /// Audio processing (normalise, mix, EQ …).
    AudioProcess,
    /// Graphics/CG overlay.
    Overlay,
    /// Monitoring / analysis tap.
    Monitor,
    /// Encoding / compression stage.
    Encode,
    /// Output delivery (SDI, network, file).
    Output,
}

impl StageKind {
    /// Return `true` for stages that must appear at most once.
    pub fn is_singleton(&self) -> bool {
        matches!(self, StageKind::Input | StageKind::Output)
    }
}

/// A single stage in the signal chain.
#[derive(Debug, Clone)]
pub struct ChainStage {
    /// Unique ID within this chain.
    pub id: u32,
    /// Display name.
    pub name: String,
    /// Stage category.
    pub kind: StageKind,
    /// Whether this stage is bypassed (signal passes through unmodified).
    pub bypassed: bool,
    /// Arbitrary key/value parameters for the stage processor.
    pub params: Vec<(String, String)>,
}

impl ChainStage {
    /// Create a new active (non-bypassed) stage.
    pub fn new(id: u32, name: impl Into<String>, kind: StageKind) -> Self {
        Self {
            id,
            name: name.into(),
            kind,
            bypassed: false,
            params: Vec::new(),
        }
    }

    /// Set a parameter value, overwriting any existing value for this key.
    pub fn set_param(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        if let Some(entry) = self.params.iter_mut().find(|(k, _)| k == &key) {
            entry.1 = value.into();
        } else {
            self.params.push((key, value.into()));
        }
    }

    /// Get a parameter value by key.
    pub fn get_param(&self, key: &str) -> Option<&str> {
        self.params
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Toggle the bypass state.
    pub fn toggle_bypass(&mut self) {
        self.bypassed = !self.bypassed;
    }
}

/// Errors that can occur when manipulating a signal chain.
#[derive(Debug, Clone, PartialEq)]
pub enum ChainError {
    /// Singleton stage (Input/Output) already present.
    SingletonViolation(StageKind),
    /// No stage with this ID found.
    NotFound(u32),
    /// The chain is missing a required stage.
    InvalidChain(String),
    /// Insertion index is out of bounds.
    IndexOutOfBounds(usize),
}

impl std::fmt::Display for ChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChainError::SingletonViolation(k) => write!(f, "{k:?} stage already exists"),
            ChainError::NotFound(id) => write!(f, "Stage {id} not found"),
            ChainError::InvalidChain(msg) => write!(f, "Invalid chain: {msg}"),
            ChainError::IndexOutOfBounds(i) => write!(f, "Index {i} out of bounds"),
        }
    }
}

/// Result type for chain operations.
pub type ChainResult<T> = Result<T, ChainError>;

/// Ordered, validated signal processing chain.
pub struct SignalChain {
    stages: VecDeque<ChainStage>,
    next_id: u32,
}

impl SignalChain {
    /// Create an empty chain.
    pub fn new() -> Self {
        Self {
            stages: VecDeque::new(),
            next_id: 1,
        }
    }

    /// Append a stage to the end of the chain. Returns the assigned stage ID.
    pub fn push(&mut self, name: impl Into<String>, kind: StageKind) -> ChainResult<u32> {
        self.check_singleton(kind)?;
        let id = self.alloc_id();
        self.stages.push_back(ChainStage::new(id, name, kind));
        Ok(id)
    }

    /// Insert a stage at position `index`. Returns the assigned stage ID.
    pub fn insert(
        &mut self,
        index: usize,
        name: impl Into<String>,
        kind: StageKind,
    ) -> ChainResult<u32> {
        if index > self.stages.len() {
            return Err(ChainError::IndexOutOfBounds(index));
        }
        self.check_singleton(kind)?;
        let id = self.alloc_id();
        self.stages.insert(index, ChainStage::new(id, name, kind));
        Ok(id)
    }

    /// Remove the stage with the given ID.
    pub fn remove(&mut self, id: u32) -> ChainResult<ChainStage> {
        let pos = self
            .stages
            .iter()
            .position(|s| s.id == id)
            .ok_or(ChainError::NotFound(id))?;
        self.stages.remove(pos).ok_or(ChainError::NotFound(id))
    }

    /// Bypass or un-bypass a stage by ID.
    pub fn set_bypass(&mut self, id: u32, bypassed: bool) -> ChainResult<()> {
        self.stages
            .iter_mut()
            .find(|s| s.id == id)
            .ok_or(ChainError::NotFound(id))
            .map(|s| s.bypassed = bypassed)
    }

    /// Get an immutable reference to a stage.
    pub fn get(&self, id: u32) -> Option<&ChainStage> {
        self.stages.iter().find(|s| s.id == id)
    }

    /// Get a mutable reference to a stage.
    pub fn get_mut(&mut self, id: u32) -> Option<&mut ChainStage> {
        self.stages.iter_mut().find(|s| s.id == id)
    }

    /// Validate that the chain has exactly one Input and one Output.
    pub fn validate(&self) -> ChainResult<()> {
        let inputs = self
            .stages
            .iter()
            .filter(|s| s.kind == StageKind::Input)
            .count();
        let outputs = self
            .stages
            .iter()
            .filter(|s| s.kind == StageKind::Output)
            .count();
        if inputs != 1 {
            return Err(ChainError::InvalidChain(format!(
                "Expected 1 Input stage, found {inputs}"
            )));
        }
        if outputs != 1 {
            return Err(ChainError::InvalidChain(format!(
                "Expected 1 Output stage, found {outputs}"
            )));
        }
        Ok(())
    }

    /// Return active (non-bypassed) stages in order.
    pub fn active_stages(&self) -> Vec<&ChainStage> {
        self.stages.iter().filter(|s| !s.bypassed).collect()
    }

    /// Total number of stages (including bypassed).
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Return `true` if the chain has no stages.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Return all stages in order.
    pub fn stages(&self) -> impl Iterator<Item = &ChainStage> {
        self.stages.iter()
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn check_singleton(&self, kind: StageKind) -> ChainResult<()> {
        if kind.is_singleton() && self.stages.iter().any(|s| s.kind == kind) {
            Err(ChainError::SingletonViolation(kind))
        } else {
            Ok(())
        }
    }
}

impl Default for SignalChain {
    fn default() -> Self {
        Self::new()
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_valid_chain() -> (SignalChain, u32, u32) {
        let mut c = SignalChain::new();
        let inp = c
            .push("Capture", StageKind::Input)
            .expect("should succeed in test");
        let out = c
            .push("SDI-Out", StageKind::Output)
            .expect("should succeed in test");
        (c, inp, out)
    }

    #[test]
    fn test_push_returns_id() {
        let mut c = SignalChain::new();
        let id = c
            .push("Capture", StageKind::Input)
            .expect("should succeed in test");
        assert_eq!(id, 1);
    }

    #[test]
    fn test_singleton_input_rejected() {
        let mut c = SignalChain::new();
        c.push("Capture1", StageKind::Input)
            .expect("should succeed in test");
        let err = c.push("Capture2", StageKind::Input).unwrap_err();
        assert!(matches!(
            err,
            ChainError::SingletonViolation(StageKind::Input)
        ));
    }

    #[test]
    fn test_singleton_output_rejected() {
        let mut c = SignalChain::new();
        c.push("Out1", StageKind::Output)
            .expect("should succeed in test");
        assert!(matches!(
            c.push("Out2", StageKind::Output).unwrap_err(),
            ChainError::SingletonViolation(_)
        ));
    }

    #[test]
    fn test_insert_in_middle() {
        let mut c = SignalChain::new();
        c.push("In", StageKind::Input)
            .expect("should succeed in test");
        c.push("Out", StageKind::Output)
            .expect("should succeed in test");
        let mid_id = c
            .insert(1, "ColorCorrect", StageKind::VideoProcess)
            .expect("should succeed in test");
        let stages: Vec<&ChainStage> = c.stages().collect();
        assert_eq!(stages[1].id, mid_id);
    }

    #[test]
    fn test_insert_out_of_bounds() {
        let mut c = SignalChain::new();
        let err = c.insert(99, "X", StageKind::VideoProcess).unwrap_err();
        assert!(matches!(err, ChainError::IndexOutOfBounds(99)));
    }

    #[test]
    fn test_remove_stage() {
        let (mut c, inp, _) = build_valid_chain();
        // Input is singleton but we can remove it (no singleton check on remove)
        let removed = c.remove(inp).expect("should succeed in test");
        assert_eq!(removed.name, "Capture");
    }

    #[test]
    fn test_remove_not_found() {
        let mut c = SignalChain::new();
        assert!(matches!(
            c.remove(99).unwrap_err(),
            ChainError::NotFound(99)
        ));
    }

    #[test]
    fn test_bypass_stage() {
        let mut c = SignalChain::new();
        let id = c
            .push("In", StageKind::Input)
            .expect("should succeed in test");
        c.set_bypass(id, true).expect("should succeed in test");
        assert!(c.get(id).expect("should succeed in test").bypassed);
    }

    #[test]
    fn test_active_stages_excludes_bypassed() {
        let mut c = SignalChain::new();
        let id = c
            .push("In", StageKind::Input)
            .expect("should succeed in test");
        c.push("Out", StageKind::Output)
            .expect("should succeed in test");
        c.set_bypass(id, true).expect("should succeed in test");
        assert_eq!(c.active_stages().len(), 1);
    }

    #[test]
    fn test_validate_valid_chain() {
        let (c, _, _) = build_valid_chain();
        assert!(c.validate().is_ok());
    }

    #[test]
    fn test_validate_missing_input() {
        let mut c = SignalChain::new();
        c.push("Out", StageKind::Output)
            .expect("should succeed in test");
        assert!(matches!(
            c.validate().unwrap_err(),
            ChainError::InvalidChain(_)
        ));
    }

    #[test]
    fn test_validate_missing_output() {
        let mut c = SignalChain::new();
        c.push("In", StageKind::Input)
            .expect("should succeed in test");
        assert!(matches!(
            c.validate().unwrap_err(),
            ChainError::InvalidChain(_)
        ));
    }

    #[test]
    fn test_stage_param_set_get() {
        let mut c = SignalChain::new();
        let id = c
            .push("EQ", StageKind::AudioProcess)
            .expect("should succeed in test");
        {
            let stage = c.get_mut(id).expect("should succeed in test");
            stage.set_param("gain_db", "6.0");
        }
        let stage = c.get(id).expect("should succeed in test");
        assert_eq!(stage.get_param("gain_db"), Some("6.0"));
    }

    #[test]
    fn test_stage_param_overwrite() {
        let mut stage = ChainStage::new(1, "EQ", StageKind::AudioProcess);
        stage.set_param("gain_db", "6.0");
        stage.set_param("gain_db", "12.0");
        assert_eq!(stage.get_param("gain_db"), Some("12.0"));
    }

    #[test]
    fn test_toggle_bypass() {
        let mut stage = ChainStage::new(1, "Scale", StageKind::VideoProcess);
        assert!(!stage.bypassed);
        stage.toggle_bypass();
        assert!(stage.bypassed);
        stage.toggle_bypass();
        assert!(!stage.bypassed);
    }

    #[test]
    fn test_stage_kind_singleton() {
        assert!(StageKind::Input.is_singleton());
        assert!(StageKind::Output.is_singleton());
        assert!(!StageKind::VideoProcess.is_singleton());
    }

    #[test]
    fn test_chain_len() {
        let (c, _, _) = build_valid_chain();
        assert_eq!(c.len(), 2);
    }
}
