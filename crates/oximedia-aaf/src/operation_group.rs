//! AAF OperationGroup module
//!
//! Represents AAF effects, transitions, and mattes via `OperationDef` and `OperationGroup`.

#[allow(dead_code)]
/// Category of an AAF operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationCategory {
    /// Video effect
    VideoEffect,
    /// Audio effect
    AudioEffect,
    /// Transition between segments
    Transition,
    /// Matte effect
    Matte,
}

impl OperationCategory {
    /// Returns `true` if this is an effect (video or audio), not a transition
    #[must_use]
    pub fn is_effect(&self) -> bool {
        matches!(
            self,
            OperationCategory::VideoEffect | OperationCategory::AudioEffect
        )
    }
}

#[allow(dead_code)]
/// Definition of a single operation parameter
#[derive(Debug, Clone)]
pub struct ParameterDef {
    /// Parameter identifier
    pub pid: u32,
    /// Human-readable name
    pub name: String,
    /// Data type name (e.g. "Rational", "Boolean", "Int32")
    pub data_type: String,
}

impl ParameterDef {
    /// Create a new `ParameterDef`
    #[must_use]
    pub fn new(pid: u32, name: String, data_type: String) -> Self {
        Self {
            pid,
            name,
            data_type,
        }
    }

    /// Returns `true` when the data type is a numeric type
    #[must_use]
    pub fn is_numeric(&self) -> bool {
        matches!(
            self.data_type.as_str(),
            "Int8"
                | "Int16"
                | "Int32"
                | "Int64"
                | "UInt8"
                | "UInt16"
                | "UInt32"
                | "UInt64"
                | "Float"
                | "Double"
                | "Rational"
        )
    }
}

#[allow(dead_code)]
/// Definition of an AAF operation (effect or transition type)
#[derive(Debug, Clone)]
pub struct OperationDef {
    /// Operation definition identifier
    pub op_def_id: u32,
    /// Human-readable name
    pub name: String,
    /// Category
    pub category: OperationCategory,
    /// Parameter definitions
    pub parameters: Vec<ParameterDef>,
}

impl OperationDef {
    /// Create a new `OperationDef`
    #[must_use]
    pub fn new(
        op_def_id: u32,
        name: String,
        category: OperationCategory,
        parameters: Vec<ParameterDef>,
    ) -> Self {
        Self {
            op_def_id,
            name,
            category,
            parameters,
        }
    }

    /// Return the number of parameters
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.parameters.len()
    }

    /// Find a parameter by name (case-insensitive)
    #[must_use]
    pub fn find_param(&self, name: &str) -> Option<&ParameterDef> {
        let lower = name.to_lowercase();
        self.parameters
            .iter()
            .find(|p| p.name.to_lowercase() == lower)
    }
}

#[allow(dead_code)]
/// An instantiation of an AAF operation applied to segment(s)
#[derive(Debug, Clone)]
pub struct OperationGroup {
    /// The operation definition
    pub op_def: OperationDef,
    /// Length in edit units
    pub length: u64,
    /// Number of input segments
    pub input_segments: u32,
}

impl OperationGroup {
    /// Create a new `OperationGroup`
    #[must_use]
    pub fn new(op_def: OperationDef, length: u64, input_segments: u32) -> Self {
        Self {
            op_def,
            length,
            input_segments,
        }
    }

    /// Returns `true` when the underlying operation is a transition
    #[must_use]
    pub fn is_transition(&self) -> bool {
        matches!(self.op_def.category, OperationCategory::Transition)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_video_effect_def() -> OperationDef {
        OperationDef::new(
            1,
            "VideoDissolve".into(),
            OperationCategory::VideoEffect,
            vec![
                ParameterDef::new(1, "Level".into(), "Rational".into()),
                ParameterDef::new(2, "Reverse".into(), "Boolean".into()),
            ],
        )
    }

    fn make_transition_def() -> OperationDef {
        OperationDef::new(
            2,
            "Wipe".into(),
            OperationCategory::Transition,
            vec![ParameterDef::new(1, "Progress".into(), "Rational".into())],
        )
    }

    #[test]
    fn test_operation_category_video_is_effect() {
        assert!(OperationCategory::VideoEffect.is_effect());
    }

    #[test]
    fn test_operation_category_audio_is_effect() {
        assert!(OperationCategory::AudioEffect.is_effect());
    }

    #[test]
    fn test_operation_category_transition_not_effect() {
        assert!(!OperationCategory::Transition.is_effect());
    }

    #[test]
    fn test_operation_category_matte_not_effect() {
        assert!(!OperationCategory::Matte.is_effect());
    }

    #[test]
    fn test_parameter_def_is_numeric_rational() {
        let p = ParameterDef::new(1, "Level".into(), "Rational".into());
        assert!(p.is_numeric());
    }

    #[test]
    fn test_parameter_def_is_numeric_boolean() {
        let p = ParameterDef::new(2, "Flag".into(), "Boolean".into());
        assert!(!p.is_numeric());
    }

    #[test]
    fn test_parameter_def_is_numeric_string() {
        let p = ParameterDef::new(3, "Label".into(), "String".into());
        assert!(!p.is_numeric());
    }

    #[test]
    fn test_operation_def_param_count() {
        let def = make_video_effect_def();
        assert_eq!(def.param_count(), 2);
    }

    #[test]
    fn test_operation_def_find_param_found() {
        let def = make_video_effect_def();
        let p = def.find_param("level");
        assert!(p.is_some());
        assert_eq!(p.expect("test expectation failed").pid, 1);
    }

    #[test]
    fn test_operation_def_find_param_case_insensitive() {
        let def = make_video_effect_def();
        assert!(def.find_param("REVERSE").is_some());
    }

    #[test]
    fn test_operation_def_find_param_not_found() {
        let def = make_video_effect_def();
        assert!(def.find_param("nonexistent").is_none());
    }

    #[test]
    fn test_operation_group_is_transition_false() {
        let def = make_video_effect_def();
        let grp = OperationGroup::new(def, 100, 1);
        assert!(!grp.is_transition());
    }

    #[test]
    fn test_operation_group_is_transition_true() {
        let def = make_transition_def();
        let grp = OperationGroup::new(def, 50, 2);
        assert!(grp.is_transition());
    }

    #[test]
    fn test_operation_group_length() {
        let def = make_video_effect_def();
        let grp = OperationGroup::new(def, 240, 1);
        assert_eq!(grp.length, 240);
    }

    #[test]
    fn test_operation_group_input_segments() {
        let def = make_transition_def();
        let grp = OperationGroup::new(def, 30, 2);
        assert_eq!(grp.input_segments, 2);
    }
}
