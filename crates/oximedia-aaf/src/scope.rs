//! AAF nested scope and scope reference management
//!
//! Provides types for managing nested scope structures in AAF compositions,
//! allowing segments to reference other segments within a scope chain.
//! Scope references are used for multi-layer effects and nested compositions
//! per SMPTE ST 377-1 Section 12.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// Identifies a position within the scope chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeRef {
    /// Number of scope levels to traverse upward (0 = current scope).
    pub relative_scope: u32,
    /// Slot index within the referenced scope.
    pub relative_slot: u32,
}

impl ScopeRef {
    /// Create a new scope reference.
    #[must_use]
    pub const fn new(relative_scope: u32, relative_slot: u32) -> Self {
        Self {
            relative_scope,
            relative_slot,
        }
    }

    /// Scope reference to the current scope, slot 0.
    #[must_use]
    pub const fn current_scope_slot_zero() -> Self {
        Self::new(0, 0)
    }

    /// Whether this references the current (innermost) scope.
    #[must_use]
    pub const fn is_current_scope(&self) -> bool {
        self.relative_scope == 0
    }

    /// Depth of traversal (alias for `relative_scope`).
    #[must_use]
    pub const fn depth(&self) -> u32 {
        self.relative_scope
    }
}

/// The kind of segment that may appear within a scope slot.
#[derive(Debug, Clone, PartialEq)]
pub enum ScopeSegmentKind {
    /// A source clip reference.
    SourceClip {
        /// Mob ID being referenced.
        mob_id: String,
        /// Track/slot ID within the referenced mob.
        track_id: u32,
        /// Start position in edit units.
        start: i64,
        /// Length in edit units.
        length: i64,
    },
    /// A filler (gap / silence / black).
    Filler {
        /// Duration of the filler in edit units.
        length: i64,
    },
    /// A nested scope.
    NestedScope {
        /// The inner scope.
        scope: NestedScope,
    },
    /// A scope reference to another layer.
    Reference {
        /// The scope reference coordinates.
        scope_ref: ScopeRef,
        /// Length in edit units.
        length: i64,
    },
}

impl ScopeSegmentKind {
    /// Duration of this segment in edit units.
    #[must_use]
    pub fn length(&self) -> i64 {
        match self {
            Self::SourceClip { length, .. } => *length,
            Self::Filler { length } => *length,
            Self::NestedScope { scope } => scope.duration(),
            Self::Reference { length, .. } => *length,
        }
    }
}

/// A single slot within a scope.
#[derive(Debug, Clone, PartialEq)]
pub struct ScopeSlot {
    /// Slot index within this scope (0-based).
    pub index: u32,
    /// Human-readable label.
    pub label: String,
    /// The segment in this slot.
    pub segment: ScopeSegmentKind,
}

impl ScopeSlot {
    /// Create a new scope slot.
    #[must_use]
    pub fn new(index: u32, label: impl Into<String>, segment: ScopeSegmentKind) -> Self {
        Self {
            index,
            label: label.into(),
            segment,
        }
    }

    /// Duration of the segment in this slot.
    #[must_use]
    pub fn duration(&self) -> i64 {
        self.segment.length()
    }
}

/// A nested scope containing one or more slots.
///
/// Nested scopes are used in AAF to model multi-layer effects where one
/// segment can reference another through scope references.
#[derive(Debug, Clone, PartialEq)]
pub struct NestedScope {
    /// Human-readable scope name.
    pub name: String,
    /// The slots within this scope, in order.
    pub slots: Vec<ScopeSlot>,
}

impl NestedScope {
    /// Create a new empty nested scope.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            slots: Vec::new(),
        }
    }

    /// Add a slot to this scope.
    pub fn add_slot(&mut self, slot: ScopeSlot) {
        self.slots.push(slot);
    }

    /// Number of slots.
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Whether this scope has no slots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Duration of the scope (max of all slot durations).
    #[must_use]
    pub fn duration(&self) -> i64 {
        self.slots.iter().map(|s| s.duration()).max().unwrap_or(0)
    }

    /// Get slot by index.
    #[must_use]
    pub fn get_slot(&self, index: u32) -> Option<&ScopeSlot> {
        self.slots.iter().find(|s| s.index == index)
    }

    /// Resolve a scope reference within this scope.
    ///
    /// Returns the slot at the given `relative_slot` if `relative_scope` is 0.
    /// Does not traverse parent scopes (that requires the full scope chain).
    #[must_use]
    pub fn resolve_local(&self, scope_ref: &ScopeRef) -> Option<&ScopeSlot> {
        if scope_ref.relative_scope != 0 {
            return None;
        }
        self.get_slot(scope_ref.relative_slot)
    }
}

/// A scope chain for resolving scope references through nested scopes.
///
/// The chain is ordered from innermost (index 0) to outermost (last).
#[derive(Debug, Default)]
pub struct ScopeChain {
    /// Scopes from inner to outer.
    scopes: Vec<NestedScope>,
}

impl ScopeChain {
    /// Create an empty scope chain.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a new innermost scope onto the chain.
    pub fn push(&mut self, scope: NestedScope) {
        self.scopes.insert(0, scope);
    }

    /// Pop the innermost scope.
    pub fn pop(&mut self) -> Option<NestedScope> {
        if self.scopes.is_empty() {
            None
        } else {
            Some(self.scopes.remove(0))
        }
    }

    /// Depth of the chain.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.scopes.len()
    }

    /// Resolve a scope reference.
    ///
    /// Uses `relative_scope` to index into the chain (0 = innermost).
    #[must_use]
    pub fn resolve(&self, scope_ref: &ScopeRef) -> Option<&ScopeSlot> {
        let idx = scope_ref.relative_scope as usize;
        let scope = self.scopes.get(idx)?;
        scope.get_slot(scope_ref.relative_slot)
    }

    /// Whether the chain is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.scopes.is_empty()
    }

    /// Get scope at a given depth (0 = innermost).
    #[must_use]
    pub fn scope_at(&self, depth: usize) -> Option<&NestedScope> {
        self.scopes.get(depth)
    }
}

/// Validates a nested scope structure for common errors.
#[derive(Debug, Default)]
pub struct ScopeValidator;

impl ScopeValidator {
    /// Validate a nested scope, returning a list of error messages.
    ///
    /// Checks for:
    /// - Duplicate slot indices
    /// - Negative durations
    /// - Dangling scope references within the local scope
    #[must_use]
    pub fn validate(scope: &NestedScope) -> Vec<String> {
        let mut errors = Vec::new();
        let mut seen_indices: HashMap<u32, usize> = HashMap::new();

        for (i, slot) in scope.slots.iter().enumerate() {
            // Check for duplicate slot indices
            if let Some(prev) = seen_indices.get(&slot.index) {
                errors.push(format!(
                    "Duplicate slot index {} at positions {} and {}",
                    slot.index, prev, i
                ));
            }
            seen_indices.insert(slot.index, i);

            // Check for negative durations
            if slot.duration() < 0 {
                errors.push(format!(
                    "Slot {} has negative duration: {}",
                    slot.index,
                    slot.duration()
                ));
            }

            // Check local scope references
            if let ScopeSegmentKind::Reference { scope_ref, .. } = &slot.segment {
                if scope_ref.is_current_scope() && scope.get_slot(scope_ref.relative_slot).is_none()
                {
                    errors.push(format!(
                        "Slot {} references non-existent local slot {}",
                        slot.index, scope_ref.relative_slot
                    ));
                }
            }
        }

        errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_ref_creation() {
        let r = ScopeRef::new(2, 3);
        assert_eq!(r.relative_scope, 2);
        assert_eq!(r.relative_slot, 3);
        assert!(!r.is_current_scope());
        assert_eq!(r.depth(), 2);
    }

    #[test]
    fn test_scope_ref_current_scope() {
        let r = ScopeRef::current_scope_slot_zero();
        assert!(r.is_current_scope());
        assert_eq!(r.depth(), 0);
        assert_eq!(r.relative_slot, 0);
    }

    #[test]
    fn test_scope_segment_source_clip_length() {
        let seg = ScopeSegmentKind::SourceClip {
            mob_id: "mob-1".into(),
            track_id: 1,
            start: 0,
            length: 100,
        };
        assert_eq!(seg.length(), 100);
    }

    #[test]
    fn test_scope_segment_filler_length() {
        let seg = ScopeSegmentKind::Filler { length: 50 };
        assert_eq!(seg.length(), 50);
    }

    #[test]
    fn test_scope_slot_creation() {
        let slot = ScopeSlot::new(0, "Background", ScopeSegmentKind::Filler { length: 200 });
        assert_eq!(slot.index, 0);
        assert_eq!(slot.label, "Background");
        assert_eq!(slot.duration(), 200);
    }

    #[test]
    fn test_nested_scope_empty() {
        let scope = NestedScope::new("Empty");
        assert!(scope.is_empty());
        assert_eq!(scope.slot_count(), 0);
        assert_eq!(scope.duration(), 0);
    }

    #[test]
    fn test_nested_scope_duration_max_slot() {
        let mut scope = NestedScope::new("FX");
        scope.add_slot(ScopeSlot::new(
            0,
            "BG",
            ScopeSegmentKind::Filler { length: 100 },
        ));
        scope.add_slot(ScopeSlot::new(
            1,
            "FG",
            ScopeSegmentKind::SourceClip {
                mob_id: "m".into(),
                track_id: 1,
                start: 0,
                length: 200,
            },
        ));
        assert_eq!(scope.duration(), 200);
    }

    #[test]
    fn test_nested_scope_get_slot() {
        let mut scope = NestedScope::new("S");
        scope.add_slot(ScopeSlot::new(
            5,
            "Layer5",
            ScopeSegmentKind::Filler { length: 10 },
        ));
        assert!(scope.get_slot(5).is_some());
        assert!(scope.get_slot(0).is_none());
    }

    #[test]
    fn test_nested_scope_resolve_local() {
        let mut scope = NestedScope::new("S");
        scope.add_slot(ScopeSlot::new(
            0,
            "L0",
            ScopeSegmentKind::Filler { length: 10 },
        ));
        let r = ScopeRef::new(0, 0);
        assert!(scope.resolve_local(&r).is_some());

        let r_parent = ScopeRef::new(1, 0);
        assert!(scope.resolve_local(&r_parent).is_none());
    }

    #[test]
    fn test_scope_chain_push_pop() {
        let mut chain = ScopeChain::new();
        assert!(chain.is_empty());

        chain.push(NestedScope::new("Inner"));
        chain.push(NestedScope::new("Innermost"));
        assert_eq!(chain.depth(), 2);

        let popped = chain.pop().expect("popped should be valid");
        assert_eq!(popped.name, "Innermost");
        assert_eq!(chain.depth(), 1);
    }

    #[test]
    fn test_scope_chain_resolve() {
        let mut chain = ScopeChain::new();

        let mut outer = NestedScope::new("Outer");
        outer.add_slot(ScopeSlot::new(
            0,
            "OuterBG",
            ScopeSegmentKind::Filler { length: 300 },
        ));

        let mut inner = NestedScope::new("Inner");
        inner.add_slot(ScopeSlot::new(
            0,
            "InnerFG",
            ScopeSegmentKind::Filler { length: 100 },
        ));

        // Push outer first, then inner (inner is at index 0)
        chain.push(outer);
        chain.push(inner);

        // Resolve innermost slot 0
        let r0 = ScopeRef::new(0, 0);
        let slot = chain.resolve(&r0).expect("slot should be valid");
        assert_eq!(slot.label, "InnerFG");

        // Resolve outer slot 0
        let r1 = ScopeRef::new(1, 0);
        let slot = chain.resolve(&r1).expect("slot should be valid");
        assert_eq!(slot.label, "OuterBG");
    }

    #[test]
    fn test_scope_chain_resolve_missing() {
        let chain = ScopeChain::new();
        let r = ScopeRef::new(0, 0);
        assert!(chain.resolve(&r).is_none());
    }

    #[test]
    fn test_scope_validator_valid() {
        let mut scope = NestedScope::new("OK");
        scope.add_slot(ScopeSlot::new(
            0,
            "A",
            ScopeSegmentKind::Filler { length: 10 },
        ));
        scope.add_slot(ScopeSlot::new(
            1,
            "B",
            ScopeSegmentKind::Filler { length: 20 },
        ));
        let errors = ScopeValidator::validate(&scope);
        assert!(errors.is_empty(), "Expected no errors, got: {errors:?}");
    }

    #[test]
    fn test_scope_validator_duplicate_index() {
        let mut scope = NestedScope::new("Dup");
        scope.add_slot(ScopeSlot::new(
            0,
            "A",
            ScopeSegmentKind::Filler { length: 10 },
        ));
        scope.add_slot(ScopeSlot::new(
            0,
            "B",
            ScopeSegmentKind::Filler { length: 20 },
        ));
        let errors = ScopeValidator::validate(&scope);
        assert!(!errors.is_empty());
        assert!(errors[0].contains("Duplicate"));
    }

    #[test]
    fn test_scope_validator_dangling_reference() {
        let mut scope = NestedScope::new("Dangle");
        scope.add_slot(ScopeSlot::new(
            0,
            "A",
            ScopeSegmentKind::Reference {
                scope_ref: ScopeRef::new(0, 99),
                length: 10,
            },
        ));
        let errors = ScopeValidator::validate(&scope);
        assert!(!errors.is_empty());
        assert!(errors[0].contains("non-existent"));
    }

    #[test]
    fn test_nested_scope_in_segment() {
        let mut inner = NestedScope::new("Inner");
        inner.add_slot(ScopeSlot::new(
            0,
            "X",
            ScopeSegmentKind::Filler { length: 42 },
        ));
        let seg = ScopeSegmentKind::NestedScope { scope: inner };
        assert_eq!(seg.length(), 42);
    }

    #[test]
    fn test_scope_chain_scope_at() {
        let mut chain = ScopeChain::new();
        chain.push(NestedScope::new("Outer"));
        chain.push(NestedScope::new("Inner"));
        assert_eq!(
            chain.scope_at(0).expect("scope_at should succeed").name,
            "Inner"
        );
        assert_eq!(
            chain.scope_at(1).expect("scope_at should succeed").name,
            "Outer"
        );
        assert!(chain.scope_at(2).is_none());
    }
}
