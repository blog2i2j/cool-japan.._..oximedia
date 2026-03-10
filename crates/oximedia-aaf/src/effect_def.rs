//! AAF effect definitions
//!
//! Effect type registry, operation group definitions, and effect slots
//! for AAF effects (SMPTE ST 377-1 Section 14).

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// Category of an AAF effect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EffectCategory {
    /// Transition between two clips
    Transition,
    /// Filter applied to a single clip
    Filter,
    /// Compositing effect (mix, key)
    Compositing,
    /// Motion effect (speed change, freeze)
    Motion,
    /// Color correction
    ColorCorrection,
    /// Custom/user-defined
    Custom,
}

/// Definition of a single effect type
#[derive(Debug, Clone)]
pub struct EffectTypeDef {
    pub auid: String,
    pub name: String,
    pub description: String,
    pub category: EffectCategory,
    pub is_time_warp: bool,
    pub bypass_override: Option<u32>,
    pub num_inputs: usize,
    pub parameter_names: Vec<String>,
}

impl EffectTypeDef {
    /// Create a new effect type definition
    #[must_use]
    pub fn new(
        auid: impl Into<String>,
        name: impl Into<String>,
        category: EffectCategory,
        num_inputs: usize,
    ) -> Self {
        Self {
            auid: auid.into(),
            name: name.into(),
            description: String::new(),
            category,
            is_time_warp: false,
            bypass_override: None,
            num_inputs,
            parameter_names: Vec::new(),
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Mark as time warp
    pub fn as_time_warp(mut self) -> Self {
        self.is_time_warp = true;
        self
    }

    /// Add a parameter name
    pub fn add_parameter(mut self, name: impl Into<String>) -> Self {
        self.parameter_names.push(name.into());
        self
    }
}

/// Registry of effect type definitions
#[derive(Debug, Clone, Default)]
pub struct EffectTypeRegistry {
    by_auid: HashMap<String, EffectTypeDef>,
    by_name: HashMap<String, String>, // name -> auid
}

impl EffectTypeRegistry {
    /// Create a new empty registry
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry pre-populated with common AAF effects
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();
        reg.register(
            EffectTypeDef::new(
                "urn:smpte:ul:060e2b34.04010101.0e040302.00000000",
                "VideoDissolve",
                EffectCategory::Transition,
                2,
            )
            .with_description("Linear video dissolve transition")
            .add_parameter("Level"),
        );
        reg.register(
            EffectTypeDef::new(
                "urn:smpte:ul:060e2b34.04010101.0e040303.00000000",
                "SMPTEVideoWipe",
                EffectCategory::Transition,
                2,
            )
            .with_description("SMPTE wipe transition")
            .add_parameter("Level")
            .add_parameter("WipeCode"),
        );
        reg.register(
            EffectTypeDef::new(
                "urn:smpte:ul:060e2b34.04010101.0e040401.00000000",
                "VideoSpeedControl",
                EffectCategory::Motion,
                1,
            )
            .with_description("Speed change effect")
            .as_time_warp()
            .add_parameter("SpeedRatio"),
        );
        reg.register(
            EffectTypeDef::new(
                "urn:smpte:ul:060e2b34.04010101.0e040501.00000000",
                "VideoColorCorrect",
                EffectCategory::ColorCorrection,
                1,
            )
            .with_description("Basic color correction")
            .add_parameter("Brightness")
            .add_parameter("Contrast")
            .add_parameter("Saturation"),
        );
        reg
    }

    /// Register an effect type definition
    pub fn register(&mut self, def: EffectTypeDef) {
        let auid = def.auid.clone();
        self.by_name.insert(def.name.clone(), auid.clone());
        self.by_auid.insert(auid, def);
    }

    /// Look up by AUID
    #[must_use]
    pub fn get_by_auid(&self, auid: &str) -> Option<&EffectTypeDef> {
        self.by_auid.get(auid)
    }

    /// Look up by name
    #[must_use]
    pub fn get_by_name(&self, name: &str) -> Option<&EffectTypeDef> {
        self.by_name.get(name).and_then(|a| self.by_auid.get(a))
    }

    /// Number of registered effects
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_auid.len()
    }

    /// Whether the registry is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_auid.is_empty()
    }

    /// All registered AUIDs
    #[must_use]
    pub fn auids(&self) -> Vec<&str> {
        self.by_auid.keys().map(String::as_str).collect()
    }

    /// Effects by category
    #[must_use]
    pub fn by_category(&self, category: EffectCategory) -> Vec<&EffectTypeDef> {
        self.by_auid
            .values()
            .filter(|e| e.category == category)
            .collect()
    }
}

/// A slot in an operation group holding a segment
#[derive(Debug, Clone)]
pub struct EffectSlot {
    pub slot_id: u32,
    pub segment_kind: String,
    pub length: i64,
    pub attributes: HashMap<String, String>,
}

impl EffectSlot {
    /// Create a new effect slot
    #[must_use]
    pub fn new(slot_id: u32, segment_kind: impl Into<String>, length: i64) -> Self {
        Self {
            slot_id,
            segment_kind: segment_kind.into(),
            length,
            attributes: HashMap::new(),
        }
    }

    /// Set an attribute
    pub fn set_attr(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.insert(key.into(), value.into());
    }
}

/// An operation group: wraps an effect definition and its input slots
#[derive(Debug, Clone)]
pub struct OperationGroupDef {
    pub effect_auid: String,
    pub length: i64,
    pub is_time_warp: bool,
    pub input_segments: Vec<EffectSlot>,
    pub parameters: HashMap<String, String>,
}

impl OperationGroupDef {
    /// Create a new operation group
    #[must_use]
    pub fn new(effect_auid: impl Into<String>, length: i64, is_time_warp: bool) -> Self {
        Self {
            effect_auid: effect_auid.into(),
            length,
            is_time_warp,
            input_segments: Vec::new(),
            parameters: HashMap::new(),
        }
    }

    /// Add an input segment slot
    pub fn add_input(&mut self, slot: EffectSlot) {
        self.input_segments.push(slot);
    }

    /// Set a parameter value
    pub fn set_parameter(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.parameters.insert(name.into(), value.into());
    }

    /// Get a parameter value
    #[must_use]
    pub fn get_parameter(&self, name: &str) -> Option<&str> {
        self.parameters.get(name).map(String::as_str)
    }

    /// Number of input slots
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.input_segments.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_type_def_new() {
        let def = EffectTypeDef::new("auid-1", "Dissolve", EffectCategory::Transition, 2);
        assert_eq!(def.name, "Dissolve");
        assert_eq!(def.num_inputs, 2);
        assert!(!def.is_time_warp);
        assert!(def.parameter_names.is_empty());
    }

    #[test]
    fn test_effect_type_def_builder() {
        let def = EffectTypeDef::new("auid-2", "SpeedRamp", EffectCategory::Motion, 1)
            .with_description("Speed ramp effect")
            .as_time_warp()
            .add_parameter("SpeedRatio");
        assert!(def.is_time_warp);
        assert_eq!(def.description, "Speed ramp effect");
        assert_eq!(def.parameter_names.len(), 1);
        assert_eq!(def.parameter_names[0], "SpeedRatio");
    }

    #[test]
    fn test_registry_register_and_lookup() {
        let mut reg = EffectTypeRegistry::new();
        let def = EffectTypeDef::new("auid-abc", "MyEffect", EffectCategory::Filter, 1);
        reg.register(def);
        assert_eq!(reg.len(), 1);
        assert!(reg.get_by_auid("auid-abc").is_some());
        assert!(reg.get_by_name("MyEffect").is_some());
        assert!(reg.get_by_auid("missing").is_none());
    }

    #[test]
    fn test_registry_with_defaults() {
        let reg = EffectTypeRegistry::with_defaults();
        assert!(reg.len() >= 4);
        assert!(reg.get_by_name("VideoDissolve").is_some());
        assert!(reg.get_by_name("VideoSpeedControl").is_some());
        let speed = reg
            .get_by_name("VideoSpeedControl")
            .expect("speed should be valid");
        assert!(speed.is_time_warp);
    }

    #[test]
    fn test_registry_by_category() {
        let reg = EffectTypeRegistry::with_defaults();
        let transitions = reg.by_category(EffectCategory::Transition);
        assert!(transitions.len() >= 2);
        let motions = reg.by_category(EffectCategory::Motion);
        assert!(!motions.is_empty());
    }

    #[test]
    fn test_registry_is_empty() {
        let reg = EffectTypeRegistry::new();
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_auids() {
        let mut reg = EffectTypeRegistry::new();
        reg.register(EffectTypeDef::new("a1", "E1", EffectCategory::Filter, 1));
        reg.register(EffectTypeDef::new("a2", "E2", EffectCategory::Filter, 1));
        let auids = reg.auids();
        assert_eq!(auids.len(), 2);
    }

    #[test]
    fn test_effect_slot_creation() {
        let mut slot = EffectSlot::new(1, "SourceClip", 100);
        slot.set_attr("role", "foreground");
        assert_eq!(slot.slot_id, 1);
        assert_eq!(slot.length, 100);
        assert_eq!(
            slot.attributes.get("role").map(String::as_str),
            Some("foreground")
        );
    }

    #[test]
    fn test_operation_group_def() {
        let mut og = OperationGroupDef::new("auid-dissolve", 50, false);
        og.add_input(EffectSlot::new(1, "SourceClip", 50));
        og.add_input(EffectSlot::new(2, "SourceClip", 50));
        og.set_parameter("Level", "0.5");
        assert_eq!(og.input_count(), 2);
        assert_eq!(og.get_parameter("Level"), Some("0.5"));
        assert!(og.get_parameter("Missing").is_none());
    }

    #[test]
    fn test_operation_group_time_warp() {
        let og = OperationGroupDef::new("auid-speed", 100, true);
        assert!(og.is_time_warp);
    }

    #[test]
    fn test_effect_category_equality() {
        assert_eq!(EffectCategory::Transition, EffectCategory::Transition);
        assert_ne!(EffectCategory::Transition, EffectCategory::Filter);
    }

    #[test]
    fn test_color_correct_parameters() {
        let reg = EffectTypeRegistry::with_defaults();
        let cc = reg
            .get_by_name("VideoColorCorrect")
            .expect("cc should be valid");
        assert!(cc.parameter_names.contains(&"Brightness".to_string()));
        assert!(cc.parameter_names.contains(&"Contrast".to_string()));
        assert!(cc.parameter_names.contains(&"Saturation".to_string()));
    }

    #[test]
    fn test_wipe_parameters() {
        let reg = EffectTypeRegistry::with_defaults();
        let wipe = reg
            .get_by_name("SMPTEVideoWipe")
            .expect("wipe should be valid");
        assert_eq!(wipe.num_inputs, 2);
        assert!(wipe.parameter_names.contains(&"WipeCode".to_string()));
    }
}
