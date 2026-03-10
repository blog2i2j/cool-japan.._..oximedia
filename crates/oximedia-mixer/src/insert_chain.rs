#![allow(dead_code)]
//! Insert effect chain management for mixer channels.
//!
//! This module models the per-channel insert chain found on professional
//! mixing consoles. Each channel has a fixed number of insert slots (default 8).
//! Each slot can hold one effect processor and can be individually bypassed,
//! reordered, or replaced without affecting other slots.

use std::fmt;

/// Maximum number of insert slots per chain.
pub const MAX_INSERT_SLOTS: usize = 8;

/// Unique identifier for an insert slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InsertSlotId(
    /// Numeric slot index.
    pub usize,
);

impl fmt::Display for InsertSlotId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InsertSlot({})", self.0)
    }
}

/// Category of effect that can occupy an insert slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertEffectKind {
    /// Parametric or graphic equalizer.
    Eq,
    /// Compressor / limiter / gate.
    Dynamics,
    /// Reverb processor.
    Reverb,
    /// Delay / echo.
    Delay,
    /// Chorus / flanger / phaser.
    Modulation,
    /// Saturation / distortion.
    Distortion,
    /// De-esser.
    DeEsser,
    /// Custom / third-party effect.
    Custom,
}

/// Represents a single effect loaded into an insert slot.
#[derive(Debug, Clone)]
pub struct InsertEffect {
    /// Human-readable name of the effect.
    pub name: String,
    /// Category of this effect.
    pub kind: InsertEffectKind,
    /// Wet/dry mix ratio (0.0 = fully dry, 1.0 = fully wet).
    pub mix: f64,
    /// Effect-specific parameter key-value pairs.
    pub params: Vec<(String, f64)>,
}

impl InsertEffect {
    /// Create a new insert effect.
    #[must_use]
    pub fn new(name: String, kind: InsertEffectKind) -> Self {
        Self {
            name,
            kind,
            mix: 1.0,
            params: Vec::new(),
        }
    }

    /// Set a named parameter value.
    pub fn set_param(&mut self, key: &str, value: f64) {
        for (k, v) in &mut self.params {
            if k == key {
                *v = value;
                return;
            }
        }
        self.params.push((key.to_string(), value));
    }

    /// Get a named parameter value.
    #[must_use]
    pub fn get_param(&self, key: &str) -> Option<f64> {
        self.params.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
    }

    /// Set wet/dry mix.
    pub fn set_mix(&mut self, mix: f64) {
        self.mix = mix.clamp(0.0, 1.0);
    }
}

/// A single slot in the insert chain.
#[derive(Debug, Clone)]
pub struct InsertSlot {
    /// Slot identifier.
    pub id: InsertSlotId,
    /// Loaded effect, if any.
    pub effect: Option<InsertEffect>,
    /// Whether this slot is bypassed.
    pub bypass: bool,
    /// Pre-fader (true) or post-fader (false) placement.
    pub pre_fader: bool,
}

impl InsertSlot {
    /// Create a new empty slot.
    #[must_use]
    pub fn new(index: usize) -> Self {
        Self {
            id: InsertSlotId(index),
            effect: None,
            bypass: false,
            pre_fader: false,
        }
    }

    /// Load an effect into this slot, returning any previously loaded effect.
    pub fn load(&mut self, effect: InsertEffect) -> Option<InsertEffect> {
        self.effect.replace(effect)
    }

    /// Unload the effect from this slot.
    pub fn unload(&mut self) -> Option<InsertEffect> {
        self.effect.take()
    }

    /// Check whether an effect is loaded.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.effect.is_some()
    }

    /// Check whether this slot is active (loaded and not bypassed).
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.effect.is_some() && !self.bypass
    }
}

/// Per-channel insert chain holding up to `MAX_INSERT_SLOTS` slots.
#[derive(Debug, Clone)]
pub struct InsertChain {
    /// Ordered list of insert slots.
    pub slots: Vec<InsertSlot>,
    /// Global bypass for the entire chain.
    pub bypass: bool,
    /// Label for this chain (usually the channel name).
    pub label: String,
}

impl InsertChain {
    /// Create a new insert chain with the default number of empty slots.
    #[must_use]
    pub fn new(label: String) -> Self {
        let slots = (0..MAX_INSERT_SLOTS).map(InsertSlot::new).collect();
        Self {
            slots,
            bypass: false,
            label,
        }
    }

    /// Create a chain with a custom number of slots.
    #[must_use]
    pub fn with_capacity(label: String, num_slots: usize) -> Self {
        let slots = (0..num_slots).map(InsertSlot::new).collect();
        Self {
            slots,
            bypass: false,
            label,
        }
    }

    /// Load an effect into a specific slot index.
    ///
    /// Returns the previously loaded effect if any, or `None`.
    /// Returns `None` without modification if the index is out of range.
    pub fn load_at(&mut self, index: usize, effect: InsertEffect) -> Option<InsertEffect> {
        self.slots.get_mut(index).and_then(|slot| slot.load(effect))
    }

    /// Unload the effect from a specific slot.
    pub fn unload_at(&mut self, index: usize) -> Option<InsertEffect> {
        self.slots.get_mut(index).and_then(InsertSlot::unload)
    }

    /// Bypass a specific slot.
    pub fn bypass_slot(&mut self, index: usize, bypass: bool) {
        if let Some(slot) = self.slots.get_mut(index) {
            slot.bypass = bypass;
        }
    }

    /// Swap two slots by index.
    pub fn swap_slots(&mut self, a: usize, b: usize) {
        if a < self.slots.len() && b < self.slots.len() && a != b {
            self.slots.swap(a, b);
            // Update IDs to match new positions
            self.slots[a].id = InsertSlotId(a);
            self.slots[b].id = InsertSlotId(b);
        }
    }

    /// Move a slot from one position to another, shifting others accordingly.
    pub fn move_slot(&mut self, from: usize, to: usize) {
        if from >= self.slots.len() || to >= self.slots.len() || from == to {
            return;
        }
        let slot = self.slots.remove(from);
        self.slots.insert(to, slot);
        // Re-index all slots
        for (i, s) in self.slots.iter_mut().enumerate() {
            s.id = InsertSlotId(i);
        }
    }

    /// Count how many slots have effects loaded.
    #[must_use]
    pub fn loaded_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_loaded()).count()
    }

    /// Count how many slots are active (loaded and not bypassed).
    #[must_use]
    pub fn active_count(&self) -> usize {
        if self.bypass {
            return 0;
        }
        self.slots.iter().filter(|s| s.is_active()).count()
    }

    /// Get an iterator over active (non-bypassed, loaded) slots in order.
    pub fn active_slots(&self) -> impl Iterator<Item = &InsertSlot> {
        let chain_bypass = self.bypass;
        self.slots
            .iter()
            .filter(move |s| !chain_bypass && s.is_active())
    }

    /// Clear all slots.
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            slot.unload();
            slot.bypass = false;
        }
    }

    /// Get the number of slots.
    #[must_use]
    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    /// Get a slot by index.
    #[must_use]
    pub fn get_slot(&self, index: usize) -> Option<&InsertSlot> {
        self.slots.get(index)
    }

    /// Get a mutable slot by index.
    pub fn get_slot_mut(&mut self, index: usize) -> Option<&mut InsertSlot> {
        self.slots.get_mut(index)
    }

    /// Collect the names of all loaded effects in chain order.
    #[must_use]
    pub fn effect_names(&self) -> Vec<String> {
        self.slots
            .iter()
            .filter_map(|s| s.effect.as_ref().map(|e| e.name.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_effect(name: &str, kind: InsertEffectKind) -> InsertEffect {
        InsertEffect::new(name.to_string(), kind)
    }

    #[test]
    fn test_insert_effect_new() {
        let e = InsertEffect::new("Comp".into(), InsertEffectKind::Dynamics);
        assert_eq!(e.name, "Comp");
        assert_eq!(e.kind, InsertEffectKind::Dynamics);
        assert!((e.mix - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_insert_effect_params() {
        let mut e = make_effect("EQ", InsertEffectKind::Eq);
        e.set_param("freq", 1000.0);
        assert!(
            (e.get_param("freq").expect("get_param should succeed") - 1000.0).abs() < f64::EPSILON
        );
        e.set_param("freq", 2000.0);
        assert!(
            (e.get_param("freq").expect("get_param should succeed") - 2000.0).abs() < f64::EPSILON
        );
        assert!(e.get_param("gain").is_none());
    }

    #[test]
    fn test_insert_effect_mix_clamp() {
        let mut e = make_effect("Reverb", InsertEffectKind::Reverb);
        e.set_mix(1.5);
        assert!((e.mix - 1.0).abs() < f64::EPSILON);
        e.set_mix(-0.5);
        assert!(e.mix.abs() < f64::EPSILON);
    }

    #[test]
    fn test_insert_slot_load_unload() {
        let mut slot = InsertSlot::new(0);
        assert!(!slot.is_loaded());
        slot.load(make_effect("Comp", InsertEffectKind::Dynamics));
        assert!(slot.is_loaded());
        assert!(slot.is_active());
        let removed = slot.unload();
        assert!(removed.is_some());
        assert!(!slot.is_loaded());
    }

    #[test]
    fn test_insert_slot_bypass() {
        let mut slot = InsertSlot::new(0);
        slot.load(make_effect("EQ", InsertEffectKind::Eq));
        slot.bypass = true;
        assert!(slot.is_loaded());
        assert!(!slot.is_active());
    }

    #[test]
    fn test_insert_chain_creation() {
        let chain = InsertChain::new("Vocals".into());
        assert_eq!(chain.num_slots(), MAX_INSERT_SLOTS);
        assert_eq!(chain.loaded_count(), 0);
        assert_eq!(chain.active_count(), 0);
    }

    #[test]
    fn test_insert_chain_with_capacity() {
        let chain = InsertChain::with_capacity("FX".into(), 4);
        assert_eq!(chain.num_slots(), 4);
    }

    #[test]
    fn test_insert_chain_load_at() {
        let mut chain = InsertChain::new("Ch1".into());
        chain.load_at(0, make_effect("Comp", InsertEffectKind::Dynamics));
        chain.load_at(1, make_effect("EQ", InsertEffectKind::Eq));
        assert_eq!(chain.loaded_count(), 2);
        assert_eq!(chain.active_count(), 2);
    }

    #[test]
    fn test_insert_chain_swap_slots() {
        let mut chain = InsertChain::new("Ch1".into());
        chain.load_at(0, make_effect("Comp", InsertEffectKind::Dynamics));
        chain.load_at(1, make_effect("EQ", InsertEffectKind::Eq));
        chain.swap_slots(0, 1);
        let names = chain.effect_names();
        assert_eq!(names[0], "EQ");
        assert_eq!(names[1], "Comp");
    }

    #[test]
    fn test_insert_chain_move_slot() {
        let mut chain = InsertChain::new("Ch1".into());
        chain.load_at(0, make_effect("A", InsertEffectKind::Eq));
        chain.load_at(1, make_effect("B", InsertEffectKind::Dynamics));
        chain.load_at(2, make_effect("C", InsertEffectKind::Reverb));
        chain.move_slot(0, 2);
        let names = chain.effect_names();
        assert_eq!(names, vec!["B", "C", "A"]);
    }

    #[test]
    fn test_insert_chain_global_bypass() {
        let mut chain = InsertChain::new("Ch1".into());
        chain.load_at(0, make_effect("Comp", InsertEffectKind::Dynamics));
        assert_eq!(chain.active_count(), 1);
        chain.bypass = true;
        assert_eq!(chain.active_count(), 0);
    }

    #[test]
    fn test_insert_chain_clear() {
        let mut chain = InsertChain::new("Ch1".into());
        chain.load_at(0, make_effect("A", InsertEffectKind::Eq));
        chain.load_at(3, make_effect("B", InsertEffectKind::Delay));
        chain.clear();
        assert_eq!(chain.loaded_count(), 0);
    }

    #[test]
    fn test_insert_chain_effect_names() {
        let mut chain = InsertChain::new("Ch1".into());
        chain.load_at(0, make_effect("Comp", InsertEffectKind::Dynamics));
        chain.load_at(2, make_effect("Delay", InsertEffectKind::Delay));
        let names = chain.effect_names();
        assert_eq!(names, vec!["Comp", "Delay"]);
    }

    #[test]
    fn test_insert_slot_id_display() {
        let id = InsertSlotId(3);
        assert_eq!(format!("{id}"), "InsertSlot(3)");
    }

    #[test]
    fn test_insert_chain_bypass_slot() {
        let mut chain = InsertChain::new("Ch1".into());
        chain.load_at(0, make_effect("Comp", InsertEffectKind::Dynamics));
        chain.bypass_slot(0, true);
        assert!(chain.get_slot(0).expect("get_slot should succeed").bypass);
        assert_eq!(chain.active_count(), 0);
    }
}
