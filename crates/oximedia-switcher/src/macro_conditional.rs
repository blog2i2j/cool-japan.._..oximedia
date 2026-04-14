//! Conditional branching extensions for the macro engine.
//!
//! This module extends the switcher's macro system with `if/else` style
//! conditional logic.  Conditions can be evaluated against the live tally
//! state, input availability, keyer on-air state, or arbitrary numeric
//! comparisons.
//!
//! # Architecture
//!
//! A conditional block consists of:
//!
//! 1. A [`Condition`] that evaluates to `true` or `false`.
//! 2. A `then` command list executed when the condition is `true`.
//! 3. An optional `else` command list executed when the condition is `false`.
//!
//! Conditions can be composed with `And`, `Or`, and `Not` operators to form
//! arbitrarily complex expressions.
//!
//! # Example
//!
//! ```rust
//! use oximedia_switcher::macro_conditional::{
//!     Condition, ConditionalBlock, ConditionEvaluator, SwitcherSnapshot,
//!     ConditionalAction,
//! };
//!
//! // "If input 1 is on program, then cut on ME 0."
//! let block = ConditionalBlock::new(
//!     Condition::InputOnProgram { input_id: 1 },
//!     vec![ConditionalAction::Cut { me_row: 0 }],
//! );
//!
//! let snapshot = SwitcherSnapshot {
//!     program_inputs: vec![1],
//!     preview_inputs: vec![2],
//!     available_inputs: vec![1, 2, 3],
//!     keyers_on_air: vec![],
//!     dsks_on_air: vec![],
//!     transition_active: vec![false],
//! };
//!
//! let evaluator = ConditionEvaluator::new();
//! let result = evaluator.evaluate(&block, &snapshot);
//! assert!(result.condition_met);
//! assert_eq!(result.actions.len(), 1);
//! ```

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ────────────────────────────────────────────────────────────────────────────
// Errors
// ────────────────────────────────────────────────────────────────────────────

/// Errors from the conditional macro subsystem.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ConditionalError {
    /// The condition expression tree is too deeply nested.
    #[error("Condition nesting depth {0} exceeds maximum {1}")]
    NestingTooDeep(usize, usize),

    /// An empty action list was supplied for the then-branch.
    #[error("Conditional block must have at least one then-action")]
    EmptyThenActions,

    /// Referenced input ID is invalid.
    #[error("Input {0} referenced in condition is out of range")]
    InvalidInputId(usize),

    /// Referenced keyer ID is invalid.
    #[error("Keyer {0} referenced in condition is out of range")]
    InvalidKeyerId(usize),

    /// Referenced M/E row is invalid.
    #[error("M/E row {0} referenced in condition is out of range")]
    InvalidMeRow(usize),
}

// ────────────────────────────────────────────────────────────────────────────
// Condition expression tree
// ────────────────────────────────────────────────────────────────────────────

/// Maximum allowed nesting depth for composed conditions.
pub const MAX_CONDITION_DEPTH: usize = 16;

/// A condition that can be evaluated against the live switcher state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Condition {
    /// True if the given input is currently on program on any M/E row.
    InputOnProgram { input_id: usize },

    /// True if the given input is currently on preview on any M/E row.
    InputOnPreview { input_id: usize },

    /// True if the given input is available (connected / hot-plugged).
    InputAvailable { input_id: usize },

    /// True if the given input is NOT available (disconnected).
    InputUnavailable { input_id: usize },

    /// True if the upstream keyer is currently on-air.
    KeyerOnAir { keyer_id: usize },

    /// True if the downstream keyer is currently on-air.
    DskOnAir { dsk_id: usize },

    /// True if a transition is actively running on the given M/E row.
    TransitionActive { me_row: usize },

    /// True if NO transition is running on the given M/E row.
    TransitionIdle { me_row: usize },

    /// Logical AND of two sub-conditions.
    And(Box<Condition>, Box<Condition>),

    /// Logical OR of two sub-conditions.
    Or(Box<Condition>, Box<Condition>),

    /// Logical NOT of a sub-condition.
    Not(Box<Condition>),

    /// Always true (useful as a default / fallback).
    Always,

    /// Always false (useful for disabling a branch without removing it).
    Never,
}

impl Condition {
    /// Logical AND combinator.
    pub fn and(self, other: Condition) -> Self {
        Condition::And(Box::new(self), Box::new(other))
    }

    /// Logical OR combinator.
    pub fn or(self, other: Condition) -> Self {
        Condition::Or(Box::new(self), Box::new(other))
    }

    /// Logical NOT combinator.
    pub fn not(self) -> Self {
        Condition::Not(Box::new(self))
    }

    /// Compute the nesting depth of this condition tree.
    pub fn depth(&self) -> usize {
        match self {
            Condition::And(a, b) | Condition::Or(a, b) => 1 + a.depth().max(b.depth()),
            Condition::Not(inner) => 1 + inner.depth(),
            _ => 0,
        }
    }

    /// Validate that the condition tree does not exceed the maximum depth.
    pub fn validate(&self) -> Result<(), ConditionalError> {
        let d = self.depth();
        if d > MAX_CONDITION_DEPTH {
            return Err(ConditionalError::NestingTooDeep(d, MAX_CONDITION_DEPTH));
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Actions
// ────────────────────────────────────────────────────────────────────────────

/// An action that can be emitted by a conditional block.
///
/// These map 1:1 to `MacroCommand` variants but are kept as a separate enum
/// so that the conditional subsystem is decoupled from the macro engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConditionalAction {
    /// Select a program source on an M/E row.
    SelectProgram { me_row: usize, input: usize },
    /// Select a preview source on an M/E row.
    SelectPreview { me_row: usize, input: usize },
    /// Perform a cut on an M/E row.
    Cut { me_row: usize },
    /// Perform an auto transition on an M/E row.
    Auto { me_row: usize },
    /// Set keyer on-air state.
    SetKeyerOnAir { keyer_id: usize, on_air: bool },
    /// Set DSK on-air state.
    SetDskOnAir { dsk_id: usize, on_air: bool },
    /// Select aux bus source.
    SelectAux { aux_id: usize, input: usize },
    /// Wait for a given duration (milliseconds).
    Wait { duration_ms: u64 },
    /// Run another macro by ID.
    RunMacro { macro_id: usize },
}

// ────────────────────────────────────────────────────────────────────────────
// Conditional block
// ────────────────────────────────────────────────────────────────────────────

/// A conditional block: condition + then-actions + optional else-actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalBlock {
    /// The condition to evaluate.
    pub condition: Condition,
    /// Actions to execute if the condition is true.
    pub then_actions: Vec<ConditionalAction>,
    /// Actions to execute if the condition is false (optional).
    pub else_actions: Vec<ConditionalAction>,
    /// Human-readable label for this block.
    pub label: String,
    /// Whether this block is enabled (disabled blocks are skipped).
    pub enabled: bool,
}

impl ConditionalBlock {
    /// Create a new conditional block with only a then-branch.
    pub fn new(condition: Condition, then_actions: Vec<ConditionalAction>) -> Self {
        Self {
            condition,
            then_actions,
            else_actions: Vec::new(),
            label: String::new(),
            enabled: true,
        }
    }

    /// Create a conditional block with both then and else branches.
    pub fn with_else(
        condition: Condition,
        then_actions: Vec<ConditionalAction>,
        else_actions: Vec<ConditionalAction>,
    ) -> Self {
        Self {
            condition,
            then_actions,
            else_actions,
            label: String::new(),
            enabled: true,
        }
    }

    /// Set the label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Disable this block.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable this block.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Validate the block.
    pub fn validate(&self) -> Result<(), ConditionalError> {
        if self.then_actions.is_empty() {
            return Err(ConditionalError::EmptyThenActions);
        }
        self.condition.validate()?;
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Switcher snapshot (for evaluation)
// ────────────────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of the switcher state used for condition
/// evaluation.  This avoids needing a direct reference to the `Switcher`
/// struct and keeps the conditional system decoupled.
#[derive(Debug, Clone, Default)]
pub struct SwitcherSnapshot {
    /// Input IDs currently on program (across all M/E rows).
    pub program_inputs: Vec<usize>,
    /// Input IDs currently on preview (across all M/E rows).
    pub preview_inputs: Vec<usize>,
    /// Input IDs that are currently available / connected.
    pub available_inputs: Vec<usize>,
    /// Upstream keyer IDs that are currently on-air.
    pub keyers_on_air: Vec<usize>,
    /// Downstream keyer IDs that are currently on-air.
    pub dsks_on_air: Vec<usize>,
    /// Per-M/E-row: whether a transition is in progress.
    pub transition_active: Vec<bool>,
}

// ────────────────────────────────────────────────────────────────────────────
// Evaluator
// ────────────────────────────────────────────────────────────────────────────

/// Result of evaluating a conditional block.
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Whether the condition was met.
    pub condition_met: bool,
    /// The actions that should be executed (from the chosen branch).
    pub actions: Vec<ConditionalAction>,
    /// Label of the block (for logging/debugging).
    pub label: String,
}

/// Evaluates conditions against a [`SwitcherSnapshot`].
pub struct ConditionEvaluator {
    /// Maximum recursion depth.
    max_depth: usize,
}

impl Default for ConditionEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConditionEvaluator {
    /// Create a new evaluator with default settings.
    pub fn new() -> Self {
        Self {
            max_depth: MAX_CONDITION_DEPTH,
        }
    }

    /// Create an evaluator with a custom max depth.
    pub fn with_max_depth(max_depth: usize) -> Self {
        Self { max_depth }
    }

    /// Evaluate a single condition against a snapshot.
    pub fn eval_condition(&self, condition: &Condition, snapshot: &SwitcherSnapshot) -> bool {
        self.eval_condition_inner(condition, snapshot, 0)
    }

    /// Evaluate a conditional block and return the result.
    pub fn evaluate(
        &self,
        block: &ConditionalBlock,
        snapshot: &SwitcherSnapshot,
    ) -> EvaluationResult {
        if !block.enabled {
            return EvaluationResult {
                condition_met: false,
                actions: Vec::new(),
                label: block.label.clone(),
            };
        }

        let met = self.eval_condition(&block.condition, snapshot);
        let actions = if met {
            block.then_actions.clone()
        } else {
            block.else_actions.clone()
        };

        EvaluationResult {
            condition_met: met,
            actions,
            label: block.label.clone(),
        }
    }

    /// Evaluate a sequence of conditional blocks, returning all results.
    pub fn evaluate_all(
        &self,
        blocks: &[ConditionalBlock],
        snapshot: &SwitcherSnapshot,
    ) -> Vec<EvaluationResult> {
        blocks.iter().map(|b| self.evaluate(b, snapshot)).collect()
    }

    /// Collect all actions from a sequence of conditional blocks.
    pub fn collect_actions(
        &self,
        blocks: &[ConditionalBlock],
        snapshot: &SwitcherSnapshot,
    ) -> Vec<ConditionalAction> {
        self.evaluate_all(blocks, snapshot)
            .into_iter()
            .flat_map(|r| r.actions)
            .collect()
    }

    // ── Internal recursive evaluator ────────────────────────────────────────

    fn eval_condition_inner(
        &self,
        condition: &Condition,
        snapshot: &SwitcherSnapshot,
        depth: usize,
    ) -> bool {
        if depth > self.max_depth {
            return false;
        }

        match condition {
            Condition::InputOnProgram { input_id } => snapshot.program_inputs.contains(input_id),
            Condition::InputOnPreview { input_id } => snapshot.preview_inputs.contains(input_id),
            Condition::InputAvailable { input_id } => snapshot.available_inputs.contains(input_id),
            Condition::InputUnavailable { input_id } => {
                !snapshot.available_inputs.contains(input_id)
            }
            Condition::KeyerOnAir { keyer_id } => snapshot.keyers_on_air.contains(keyer_id),
            Condition::DskOnAir { dsk_id } => snapshot.dsks_on_air.contains(dsk_id),
            Condition::TransitionActive { me_row } => snapshot
                .transition_active
                .get(*me_row)
                .copied()
                .unwrap_or(false),
            Condition::TransitionIdle { me_row } => !snapshot
                .transition_active
                .get(*me_row)
                .copied()
                .unwrap_or(true),
            Condition::And(a, b) => {
                self.eval_condition_inner(a, snapshot, depth + 1)
                    && self.eval_condition_inner(b, snapshot, depth + 1)
            }
            Condition::Or(a, b) => {
                self.eval_condition_inner(a, snapshot, depth + 1)
                    || self.eval_condition_inner(b, snapshot, depth + 1)
            }
            Condition::Not(inner) => !self.eval_condition_inner(inner, snapshot, depth + 1),
            Condition::Always => true,
            Condition::Never => false,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot_basic() -> SwitcherSnapshot {
        SwitcherSnapshot {
            program_inputs: vec![1, 3],
            preview_inputs: vec![2],
            available_inputs: vec![1, 2, 3, 4],
            keyers_on_air: vec![0],
            dsks_on_air: vec![1],
            transition_active: vec![false, true],
        }
    }

    #[test]
    fn test_input_on_program() {
        let eval = ConditionEvaluator::new();
        let snap = snapshot_basic();
        assert!(eval.eval_condition(&Condition::InputOnProgram { input_id: 1 }, &snap));
        assert!(!eval.eval_condition(&Condition::InputOnProgram { input_id: 2 }, &snap));
    }

    #[test]
    fn test_input_on_preview() {
        let eval = ConditionEvaluator::new();
        let snap = snapshot_basic();
        assert!(eval.eval_condition(&Condition::InputOnPreview { input_id: 2 }, &snap));
        assert!(!eval.eval_condition(&Condition::InputOnPreview { input_id: 1 }, &snap));
    }

    #[test]
    fn test_input_available() {
        let eval = ConditionEvaluator::new();
        let snap = snapshot_basic();
        assert!(eval.eval_condition(&Condition::InputAvailable { input_id: 4 }, &snap));
        assert!(!eval.eval_condition(&Condition::InputAvailable { input_id: 99 }, &snap));
    }

    #[test]
    fn test_input_unavailable() {
        let eval = ConditionEvaluator::new();
        let snap = snapshot_basic();
        assert!(eval.eval_condition(&Condition::InputUnavailable { input_id: 99 }, &snap));
        assert!(!eval.eval_condition(&Condition::InputUnavailable { input_id: 1 }, &snap));
    }

    #[test]
    fn test_keyer_on_air() {
        let eval = ConditionEvaluator::new();
        let snap = snapshot_basic();
        assert!(eval.eval_condition(&Condition::KeyerOnAir { keyer_id: 0 }, &snap));
        assert!(!eval.eval_condition(&Condition::KeyerOnAir { keyer_id: 5 }, &snap));
    }

    #[test]
    fn test_dsk_on_air() {
        let eval = ConditionEvaluator::new();
        let snap = snapshot_basic();
        assert!(eval.eval_condition(&Condition::DskOnAir { dsk_id: 1 }, &snap));
        assert!(!eval.eval_condition(&Condition::DskOnAir { dsk_id: 0 }, &snap));
    }

    #[test]
    fn test_transition_active_idle() {
        let eval = ConditionEvaluator::new();
        let snap = snapshot_basic();
        assert!(!eval.eval_condition(&Condition::TransitionActive { me_row: 0 }, &snap));
        assert!(eval.eval_condition(&Condition::TransitionActive { me_row: 1 }, &snap));
        assert!(eval.eval_condition(&Condition::TransitionIdle { me_row: 0 }, &snap));
        assert!(!eval.eval_condition(&Condition::TransitionIdle { me_row: 1 }, &snap));
    }

    #[test]
    fn test_and_combinator() {
        let eval = ConditionEvaluator::new();
        let snap = snapshot_basic();
        let cond = Condition::InputOnProgram { input_id: 1 }
            .and(Condition::InputOnPreview { input_id: 2 });
        assert!(eval.eval_condition(&cond, &snap));

        let cond2 = Condition::InputOnProgram { input_id: 1 }
            .and(Condition::InputOnProgram { input_id: 99 });
        assert!(!eval.eval_condition(&cond2, &snap));
    }

    #[test]
    fn test_or_combinator() {
        let eval = ConditionEvaluator::new();
        let snap = snapshot_basic();
        let cond = Condition::InputOnProgram { input_id: 99 }
            .or(Condition::InputOnPreview { input_id: 2 });
        assert!(eval.eval_condition(&cond, &snap));

        let cond2 = Condition::InputOnProgram { input_id: 99 }
            .or(Condition::InputOnPreview { input_id: 99 });
        assert!(!eval.eval_condition(&cond2, &snap));
    }

    #[test]
    fn test_not_combinator() {
        let eval = ConditionEvaluator::new();
        let snap = snapshot_basic();
        let cond = Condition::InputOnProgram { input_id: 1 }.not();
        assert!(!eval.eval_condition(&cond, &snap));

        let cond2 = Condition::InputOnProgram { input_id: 99 }.not();
        assert!(eval.eval_condition(&cond2, &snap));
    }

    #[test]
    fn test_always_never() {
        let eval = ConditionEvaluator::new();
        let snap = snapshot_basic();
        assert!(eval.eval_condition(&Condition::Always, &snap));
        assert!(!eval.eval_condition(&Condition::Never, &snap));
    }

    #[test]
    fn test_conditional_block_then_branch() {
        let eval = ConditionEvaluator::new();
        let snap = snapshot_basic();
        let block = ConditionalBlock::new(
            Condition::InputOnProgram { input_id: 1 },
            vec![ConditionalAction::Cut { me_row: 0 }],
        );
        let result = eval.evaluate(&block, &snap);
        assert!(result.condition_met);
        assert_eq!(result.actions.len(), 1);
        assert_eq!(result.actions[0], ConditionalAction::Cut { me_row: 0 });
    }

    #[test]
    fn test_conditional_block_else_branch() {
        let eval = ConditionEvaluator::new();
        let snap = snapshot_basic();
        let block = ConditionalBlock::with_else(
            Condition::InputOnProgram { input_id: 99 },
            vec![ConditionalAction::Cut { me_row: 0 }],
            vec![ConditionalAction::Auto { me_row: 0 }],
        );
        let result = eval.evaluate(&block, &snap);
        assert!(!result.condition_met);
        assert_eq!(result.actions.len(), 1);
        assert_eq!(result.actions[0], ConditionalAction::Auto { me_row: 0 });
    }

    #[test]
    fn test_disabled_block_returns_empty() {
        let eval = ConditionEvaluator::new();
        let snap = snapshot_basic();
        let mut block = ConditionalBlock::new(
            Condition::Always,
            vec![ConditionalAction::Cut { me_row: 0 }],
        );
        block.disable();
        let result = eval.evaluate(&block, &snap);
        assert!(!result.condition_met);
        assert!(result.actions.is_empty());
    }

    #[test]
    fn test_validate_empty_then_actions() {
        let block = ConditionalBlock::new(Condition::Always, vec![]);
        assert!(matches!(
            block.validate(),
            Err(ConditionalError::EmptyThenActions)
        ));
    }

    #[test]
    fn test_condition_depth() {
        let simple = Condition::Always;
        assert_eq!(simple.depth(), 0);

        let nested = Condition::And(
            Box::new(Condition::Or(
                Box::new(Condition::Always),
                Box::new(Condition::Never),
            )),
            Box::new(Condition::Not(Box::new(Condition::Always))),
        );
        assert_eq!(nested.depth(), 2);
    }

    #[test]
    fn test_collect_actions_from_multiple_blocks() {
        let eval = ConditionEvaluator::new();
        let snap = snapshot_basic();
        let blocks = vec![
            ConditionalBlock::new(
                Condition::InputOnProgram { input_id: 1 },
                vec![ConditionalAction::Cut { me_row: 0 }],
            ),
            ConditionalBlock::new(
                Condition::InputOnPreview { input_id: 2 },
                vec![ConditionalAction::Auto { me_row: 1 }],
            ),
            ConditionalBlock::new(
                Condition::InputOnProgram { input_id: 99 },
                vec![ConditionalAction::Cut { me_row: 2 }],
            ),
        ];
        let actions = eval.collect_actions(&blocks, &snap);
        // First two blocks match, third does not.
        assert_eq!(actions.len(), 2);
    }

    #[test]
    fn test_complex_nested_condition() {
        let eval = ConditionEvaluator::new();
        let snap = snapshot_basic();
        // (input 1 on program AND input 2 on preview) OR (keyer 0 on-air)
        let cond = Condition::InputOnProgram { input_id: 1 }
            .and(Condition::InputOnPreview { input_id: 2 })
            .or(Condition::KeyerOnAir { keyer_id: 0 });
        assert!(eval.eval_condition(&cond, &snap));
    }

    #[test]
    fn test_evaluate_all() {
        let eval = ConditionEvaluator::new();
        let snap = snapshot_basic();
        let blocks = vec![
            ConditionalBlock::new(
                Condition::Always,
                vec![ConditionalAction::Cut { me_row: 0 }],
            ),
            ConditionalBlock::new(Condition::Never, vec![ConditionalAction::Cut { me_row: 1 }]),
        ];
        let results = eval.evaluate_all(&blocks, &snap);
        assert_eq!(results.len(), 2);
        assert!(results[0].condition_met);
        assert!(!results[1].condition_met);
    }

    #[test]
    fn test_block_with_label() {
        let block = ConditionalBlock::new(
            Condition::Always,
            vec![ConditionalAction::Cut { me_row: 0 }],
        )
        .with_label("my-block");
        assert_eq!(block.label, "my-block");
    }

    #[test]
    fn test_out_of_bounds_transition_defaults_safely() {
        let eval = ConditionEvaluator::new();
        let snap = snapshot_basic(); // 2 M/E rows
                                     // Query M/E row 99, which is out of bounds
        assert!(!eval.eval_condition(&Condition::TransitionActive { me_row: 99 }, &snap));
        // TransitionIdle for out-of-bounds should also be false (no data)
        assert!(!eval.eval_condition(&Condition::TransitionIdle { me_row: 99 }, &snap));
    }
}
