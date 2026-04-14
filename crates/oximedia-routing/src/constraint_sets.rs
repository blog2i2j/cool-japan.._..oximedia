#![allow(dead_code)]
//! NMOS IS-11 constraint sets for sender/receiver format negotiation.
//!
//! IS-11 allows receivers (and sometimes senders) to publish a set of
//! format constraints.  A sender is compatible with a receiver only if at
//! least one of the receiver's constraint sets is fully satisfied by the
//! sender's [`SenderCapabilities`].
//!
//! # Design
//!
//! - [`ParamConstraint`] — a bounded constraint on a single numeric or
//!   string parameter (minimum, maximum, enumeration, or exact match).
//! - [`ConstraintSet`] — a named collection of parameter constraints that
//!   must *all* be satisfied simultaneously (logical AND).
//! - [`SenderCapabilities`] — a flat description of what a sender can
//!   produce (resolution, frame rate, codec, sample rate, …).
//! - [`ConstraintEngine`] — evaluates whether a sender satisfies a
//!   receiver's constraint sets (any-of semantics: logical OR across sets).
//! - [`NegotiationResult`] — the outcome of a compatibility check.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// ParamConstraint
// ---------------------------------------------------------------------------

/// A constraint on a single named parameter.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamConstraint {
    /// The parameter must be at least `min` (inclusive).
    Minimum(f64),
    /// The parameter must be at most `max` (inclusive).
    Maximum(f64),
    /// The parameter must fall within [`min`, `max`] (inclusive on both).
    Range { min: f64, max: f64 },
    /// The parameter must equal one of the listed values (string enum).
    Enum(Vec<String>),
    /// The parameter must exactly equal the given string.
    ExactString(String),
    /// The parameter must exactly equal the given number (within ε).
    ExactNumber(f64),
}

impl ParamConstraint {
    /// Evaluates the constraint against a string value.  Numeric constraints
    /// attempt to parse the string as f64; string constraints compare directly.
    pub fn matches_str(&self, value: &str) -> bool {
        match self {
            Self::Enum(options) => options.iter().any(|o| o == value),
            Self::ExactString(s) => s == value,
            Self::Minimum(min) => value.parse::<f64>().map(|v| v >= *min).unwrap_or(false),
            Self::Maximum(max) => value.parse::<f64>().map(|v| v <= *max).unwrap_or(false),
            Self::Range { min, max } => value
                .parse::<f64>()
                .map(|v| v >= *min && v <= *max)
                .unwrap_or(false),
            Self::ExactNumber(n) => value
                .parse::<f64>()
                .map(|v| (v - n).abs() < 1e-9)
                .unwrap_or(false),
        }
    }

    /// Evaluates the constraint against a numeric value.
    pub fn matches_number(&self, value: f64) -> bool {
        match self {
            Self::Minimum(min) => value >= *min,
            Self::Maximum(max) => value <= *max,
            Self::Range { min, max } => value >= *min && value <= *max,
            Self::ExactNumber(n) => (value - n).abs() < 1e-9,
            Self::Enum(options) => options.iter().any(|o| {
                o.parse::<f64>()
                    .map(|n| (n - value).abs() < 1e-9)
                    .unwrap_or(false)
            }),
            Self::ExactString(s) => s
                .parse::<f64>()
                .map(|n| (n - value).abs() < 1e-9)
                .unwrap_or(false),
        }
    }
}

impl fmt::Display for ParamConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Minimum(n) => write!(f, ">={n}"),
            Self::Maximum(n) => write!(f, "<={n}"),
            Self::Range { min, max } => write!(f, "[{min}, {max}]"),
            Self::Enum(opts) => write!(f, "one_of({opts:?})"),
            Self::ExactString(s) => write!(f, "=={s:?}"),
            Self::ExactNumber(n) => write!(f, "=={n}"),
        }
    }
}

// ---------------------------------------------------------------------------
// ConstraintSet
// ---------------------------------------------------------------------------

/// A named set of per-parameter constraints (logical AND).
///
/// All constraints within a set must be satisfied for the set to match.
#[derive(Debug, Clone, Default)]
pub struct ConstraintSet {
    /// Human-readable name for this set.
    pub name: String,
    /// Map from parameter name to its constraint.
    pub constraints: HashMap<String, ParamConstraint>,
}

impl ConstraintSet {
    /// Creates an empty constraint set.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            constraints: HashMap::new(),
        }
    }

    /// Adds a parameter constraint.
    pub fn add(mut self, param: impl Into<String>, constraint: ParamConstraint) -> Self {
        self.constraints.insert(param.into(), constraint);
        self
    }

    /// Returns the number of parameter constraints in this set.
    pub fn param_count(&self) -> usize {
        self.constraints.len()
    }

    /// Returns `true` if there are no parameter constraints.
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }
}

impl fmt::Display for ConstraintSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ConstraintSet('{}', {} params)",
            self.name,
            self.constraints.len()
        )
    }
}

// ---------------------------------------------------------------------------
// SenderCapabilities
// ---------------------------------------------------------------------------

/// Flat description of a sender's current capabilities.
///
/// Both numeric and string parameters are stored in separate maps to avoid
/// parse overhead during constraint evaluation.
#[derive(Debug, Clone, Default)]
pub struct SenderCapabilities {
    /// Numeric parameters (width, height, sample_rate, channels, …).
    pub numeric: HashMap<String, f64>,
    /// String parameters (codec, color_space, transport, …).
    pub string: HashMap<String, String>,
}

impl SenderCapabilities {
    /// Creates an empty capability set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts a numeric capability.
    pub fn with_numeric(mut self, param: impl Into<String>, value: f64) -> Self {
        self.numeric.insert(param.into(), value);
        self
    }

    /// Inserts a string capability.
    pub fn with_string(mut self, param: impl Into<String>, value: impl Into<String>) -> Self {
        self.string.insert(param.into(), value.into());
        self
    }

    /// Returns the numeric value for a parameter, or `None`.
    pub fn get_numeric(&self, param: &str) -> Option<f64> {
        self.numeric.get(param).copied()
    }

    /// Returns the string value for a parameter, or `None`.
    pub fn get_string(&self, param: &str) -> Option<&str> {
        self.string.get(param).map(|s| s.as_str())
    }

    /// Evaluates a single [`ParamConstraint`] against this capability set.
    fn satisfies_param(&self, param: &str, constraint: &ParamConstraint) -> bool {
        // Try numeric first, then string
        if let Some(n) = self.numeric.get(param) {
            return constraint.matches_number(*n);
        }
        if let Some(s) = self.string.get(param) {
            return constraint.matches_str(s.as_str());
        }
        // Parameter not present in capabilities — constraint not satisfied
        false
    }

    /// Returns `true` if all constraints in the given set are satisfied.
    pub fn satisfies_set(&self, set: &ConstraintSet) -> bool {
        set.constraints
            .iter()
            .all(|(param, constraint)| self.satisfies_param(param, constraint))
    }
}

// ---------------------------------------------------------------------------
// NegotiationResult
// ---------------------------------------------------------------------------

/// Result of a constraint-set compatibility check.
#[derive(Debug, Clone)]
pub struct NegotiationResult {
    /// Whether the sender satisfies at least one receiver constraint set.
    pub compatible: bool,
    /// The name of the first matching constraint set, if any.
    pub matched_set: Option<String>,
    /// Constraint sets that were evaluated but did not match, with reasons.
    pub failures: Vec<NegotiationFailure>,
}

impl NegotiationResult {
    /// Returns `true` if sender and receiver are compatible.
    pub fn is_compatible(&self) -> bool {
        self.compatible
    }
}

impl fmt::Display for NegotiationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.compatible {
            write!(
                f,
                "compatible (matched '{}')",
                self.matched_set.as_deref().unwrap_or("?")
            )
        } else {
            write!(f, "incompatible ({} sets failed)", self.failures.len())
        }
    }
}

/// Detail on why a specific constraint set failed to match.
#[derive(Debug, Clone)]
pub struct NegotiationFailure {
    /// Name of the constraint set that failed.
    pub set_name: String,
    /// Parameter names whose constraints were not satisfied.
    pub failed_params: Vec<String>,
}

// ---------------------------------------------------------------------------
// ConstraintEngine
// ---------------------------------------------------------------------------

/// Evaluates sender/receiver compatibility against NMOS IS-11 constraint sets.
///
/// A receiver publishes one or more [`ConstraintSet`]s.  The engine returns
/// [`NegotiationResult::compatible`] = `true` when the sender satisfies at
/// least one set in its entirety (OR across sets, AND within a set).
#[derive(Debug, Default)]
pub struct ConstraintEngine {
    /// Named constraint sets (typically the receiver's requirements).
    sets: Vec<ConstraintSet>,
}

impl ConstraintEngine {
    /// Creates an engine with no constraint sets.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a constraint set.
    pub fn add_set(&mut self, set: ConstraintSet) {
        self.sets.push(set);
    }

    /// Returns the number of constraint sets registered.
    pub fn set_count(&self) -> usize {
        self.sets.len()
    }

    /// Removes all constraint sets.
    pub fn clear(&mut self) {
        self.sets.clear();
    }

    /// Evaluates whether `capabilities` satisfies at least one constraint set.
    ///
    /// Returns a detailed [`NegotiationResult`] describing matches and failures.
    pub fn evaluate(&self, capabilities: &SenderCapabilities) -> NegotiationResult {
        let mut failures = Vec::new();
        let mut matched_set = None;

        for set in &self.sets {
            let failed_params: Vec<String> = set
                .constraints
                .iter()
                .filter_map(|(param, constraint)| {
                    if !capabilities.satisfies_param_pub(param, constraint) {
                        Some(param.clone())
                    } else {
                        None
                    }
                })
                .collect();

            if failed_params.is_empty() {
                // This set is fully satisfied — compatible!
                matched_set = Some(set.name.clone());
                break;
            } else {
                failures.push(NegotiationFailure {
                    set_name: set.name.clone(),
                    failed_params,
                });
            }
        }

        NegotiationResult {
            compatible: matched_set.is_some(),
            matched_set,
            failures,
        }
    }

    /// Returns `true` if the sender satisfies *all* constraint sets (AND semantics).
    /// Useful for validating against mandatory capability baselines.
    pub fn evaluate_all(&self, capabilities: &SenderCapabilities) -> bool {
        self.sets.iter().all(|set| capabilities.satisfies_set(set))
    }

    /// Returns a list of parameter names that appear in any constraint set.
    pub fn constrained_params(&self) -> Vec<String> {
        let mut params: Vec<String> = self
            .sets
            .iter()
            .flat_map(|s| s.constraints.keys().cloned())
            .collect();
        params.sort();
        params.dedup();
        params
    }
}

// ---------------------------------------------------------------------------
// Internal helper: expose satisfies_param publicly for the engine
// ---------------------------------------------------------------------------

impl SenderCapabilities {
    /// Same as `satisfies_param` but accessible from sibling types.
    pub fn satisfies_param_pub(&self, param: &str, constraint: &ParamConstraint) -> bool {
        self.satisfies_param(param, constraint)
    }
}

// ---------------------------------------------------------------------------
// FormatNegotiator
// ---------------------------------------------------------------------------

/// High-level helper that wraps a [`ConstraintEngine`] with receiver
/// constraint sets and exposes a single `negotiate` entry point.
#[derive(Debug, Default)]
pub struct FormatNegotiator {
    engine: ConstraintEngine,
    receiver_id: String,
}

impl FormatNegotiator {
    /// Creates a negotiator for the given receiver id.
    pub fn new(receiver_id: impl Into<String>) -> Self {
        Self {
            engine: ConstraintEngine::new(),
            receiver_id: receiver_id.into(),
        }
    }

    /// Returns the receiver id.
    pub fn receiver_id(&self) -> &str {
        &self.receiver_id
    }

    /// Registers a constraint set.
    pub fn add_constraint_set(&mut self, set: ConstraintSet) {
        self.engine.add_set(set);
    }

    /// Negotiates format compatibility between this receiver and a sender.
    pub fn negotiate(&self, sender: &SenderCapabilities) -> NegotiationResult {
        self.engine.evaluate(sender)
    }

    /// Returns the number of constraint sets.
    pub fn set_count(&self) -> usize {
        self.engine.set_count()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn hd_video_sender() -> SenderCapabilities {
        SenderCapabilities::new()
            .with_numeric("width", 1920.0)
            .with_numeric("height", 1080.0)
            .with_numeric("frame_rate", 50.0)
            .with_string("codec", "raw")
            .with_string("color_space", "BT.709")
    }

    fn audio_sender() -> SenderCapabilities {
        SenderCapabilities::new()
            .with_numeric("sample_rate", 48000.0)
            .with_numeric("channels", 2.0)
            .with_numeric("bit_depth", 24.0)
            .with_string("codec", "L24")
    }

    #[test]
    fn test_param_constraint_minimum_match() {
        let c = ParamConstraint::Minimum(1920.0);
        assert!(c.matches_number(1920.0));
        assert!(c.matches_number(3840.0));
        assert!(!c.matches_number(1280.0));
    }

    #[test]
    fn test_param_constraint_maximum_match() {
        let c = ParamConstraint::Maximum(60.0);
        assert!(c.matches_number(50.0));
        assert!(c.matches_number(60.0));
        assert!(!c.matches_number(120.0));
    }

    #[test]
    fn test_param_constraint_range_match() {
        let c = ParamConstraint::Range {
            min: 25.0,
            max: 60.0,
        };
        assert!(c.matches_number(50.0));
        assert!(!c.matches_number(24.0));
        assert!(!c.matches_number(61.0));
    }

    #[test]
    fn test_param_constraint_enum_match() {
        let c = ParamConstraint::Enum(vec!["raw".to_string(), "H.264".to_string()]);
        assert!(c.matches_str("raw"));
        assert!(c.matches_str("H.264"));
        assert!(!c.matches_str("HEVC"));
    }

    #[test]
    fn test_param_constraint_exact_string_match() {
        let c = ParamConstraint::ExactString("BT.709".to_string());
        assert!(c.matches_str("BT.709"));
        assert!(!c.matches_str("BT.2020"));
    }

    #[test]
    fn test_sender_satisfies_set_all_match() {
        let sender = hd_video_sender();
        let set = ConstraintSet::new("HD 1080p50")
            .add("width", ParamConstraint::ExactNumber(1920.0))
            .add("height", ParamConstraint::ExactNumber(1080.0))
            .add("frame_rate", ParamConstraint::Minimum(50.0))
            .add("codec", ParamConstraint::ExactString("raw".to_string()));
        assert!(sender.satisfies_set(&set));
    }

    #[test]
    fn test_sender_fails_set_when_param_missing() {
        let sender = hd_video_sender();
        let set = ConstraintSet::new("Requires audio")
            .add("sample_rate", ParamConstraint::ExactNumber(48000.0));
        assert!(!sender.satisfies_set(&set));
    }

    #[test]
    fn test_constraint_engine_compatible_first_set_matches() {
        let sender = hd_video_sender();
        let mut engine = ConstraintEngine::new();
        engine.add_set(ConstraintSet::new("SD").add("width", ParamConstraint::ExactNumber(720.0)));
        engine.add_set(
            ConstraintSet::new("HD")
                .add("width", ParamConstraint::ExactNumber(1920.0))
                .add("height", ParamConstraint::ExactNumber(1080.0)),
        );
        let result = engine.evaluate(&sender);
        assert!(result.is_compatible());
        assert_eq!(result.matched_set.as_deref(), Some("HD"));
    }

    #[test]
    fn test_constraint_engine_incompatible_no_sets_match() {
        let sender = hd_video_sender();
        let mut engine = ConstraintEngine::new();
        engine
            .add_set(ConstraintSet::new("4K only").add("width", ParamConstraint::Minimum(3840.0)));
        let result = engine.evaluate(&sender);
        assert!(!result.is_compatible());
        assert_eq!(result.failures.len(), 1);
        assert!(result.failures[0]
            .failed_params
            .contains(&"width".to_string()));
    }

    #[test]
    fn test_constraint_engine_empty_set_always_matches() {
        let sender = audio_sender();
        let mut engine = ConstraintEngine::new();
        engine.add_set(ConstraintSet::new("unconstrained"));
        let result = engine.evaluate(&sender);
        assert!(result.is_compatible());
    }

    #[test]
    fn test_format_negotiator_audio() {
        let sender = audio_sender();
        let mut neg = FormatNegotiator::new("recv-audio");
        neg.add_constraint_set(
            ConstraintSet::new("48k stereo L24")
                .add("sample_rate", ParamConstraint::ExactNumber(48000.0))
                .add("channels", ParamConstraint::Minimum(2.0))
                .add("codec", ParamConstraint::ExactString("L24".to_string())),
        );
        let result = neg.negotiate(&sender);
        assert!(result.is_compatible());
    }

    #[test]
    fn test_constraint_engine_evaluate_all() {
        let sender = hd_video_sender();
        let mut engine = ConstraintEngine::new();
        engine.add_set(
            ConstraintSet::new("width check").add("width", ParamConstraint::Minimum(1920.0)),
        );
        engine.add_set(
            ConstraintSet::new("codec check")
                .add("codec", ParamConstraint::ExactString("raw".to_string())),
        );
        // Both sets satisfied → evaluate_all = true
        assert!(engine.evaluate_all(&sender));
    }

    #[test]
    fn test_constrained_params_deduplication() {
        let mut engine = ConstraintEngine::new();
        engine.add_set(
            ConstraintSet::new("s1")
                .add("width", ParamConstraint::Minimum(1280.0))
                .add("codec", ParamConstraint::ExactString("raw".to_string())),
        );
        engine.add_set(
            ConstraintSet::new("s2")
                .add("width", ParamConstraint::Minimum(1920.0))
                .add("height", ParamConstraint::ExactNumber(1080.0)),
        );
        let params = engine.constrained_params();
        // Should have codec, height, width (sorted, dedup)
        assert_eq!(params, vec!["codec", "height", "width"]);
    }

    #[test]
    fn test_negotiation_result_display_compatible() {
        let result = NegotiationResult {
            compatible: true,
            matched_set: Some("HD".to_string()),
            failures: Vec::new(),
        };
        let s = format!("{result}");
        assert!(s.contains("compatible"));
        assert!(s.contains("HD"));
    }

    #[test]
    fn test_negotiation_result_display_incompatible() {
        let result = NegotiationResult {
            compatible: false,
            matched_set: None,
            failures: vec![NegotiationFailure {
                set_name: "HD".to_string(),
                failed_params: vec!["width".to_string()],
            }],
        };
        let s = format!("{result}");
        assert!(s.contains("incompatible"));
    }
}
