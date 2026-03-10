//! AAF Dictionary
//!
//! This module implements the AAF dictionary which defines:
//! - Classes (object types)
//! - Properties (object attributes)
//! - Types (data types)
//! - Data definitions (essence types)
//!
//! The dictionary enables AAF extensibility and provides metadata about the object model.

use crate::structured_storage::StorageReader;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::io::{Read, Seek};
use uuid::Uuid;

/// AUID (AAF Unique Identifier) - 16-byte identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Auid([u8; 16]);

impl Auid {
    /// Create a new AUID from bytes
    #[must_use]
    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        Self(*bytes)
    }

    /// Create from UUID
    #[must_use]
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(*uuid.as_bytes())
    }

    /// Convert to UUID
    #[must_use]
    pub fn to_uuid(&self) -> Uuid {
        Uuid::from_bytes(self.0)
    }

    /// Get bytes
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Create a null AUID (all zeros)
    #[must_use]
    pub fn null() -> Self {
        Self([0u8; 16])
    }

    /// Check if this is a null AUID
    #[must_use]
    pub fn is_null(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }

    /// Check if this represents a picture data definition
    #[must_use]
    pub fn is_picture(&self) -> bool {
        *self == Self::PICTURE || *self == Self::PICTURE_WITH_MATTE
    }

    /// Check if this represents a sound data definition
    #[must_use]
    pub fn is_sound(&self) -> bool {
        *self == Self::SOUND
    }

    /// Check if this represents a timecode data definition
    #[must_use]
    pub fn is_timecode(&self) -> bool {
        *self == Self::TIMECODE || *self == Self::EDGECODE
    }

    // Standard AAF data definitions
    pub const PICTURE: Auid = Auid([
        0x01, 0x03, 0x02, 0x01, 0x01, 0x00, 0x00, 0x00, 0x06, 0x0e, 0x2b, 0x34, 0x04, 0x01, 0x01,
        0x01,
    ]);

    pub const SOUND: Auid = Auid([
        0x01, 0x03, 0x02, 0x02, 0x01, 0x00, 0x00, 0x00, 0x06, 0x0e, 0x2b, 0x34, 0x04, 0x01, 0x01,
        0x01,
    ]);

    pub const TIMECODE: Auid = Auid([
        0x01, 0x03, 0x02, 0x01, 0x02, 0x00, 0x00, 0x00, 0x06, 0x0e, 0x2b, 0x34, 0x04, 0x01, 0x01,
        0x01,
    ]);

    pub const EDGECODE: Auid = Auid([
        0x01, 0x03, 0x02, 0x01, 0x03, 0x00, 0x00, 0x00, 0x06, 0x0e, 0x2b, 0x34, 0x04, 0x01, 0x01,
        0x01,
    ]);

    pub const PICTURE_WITH_MATTE: Auid = Auid([
        0x05, 0x03, 0x02, 0x01, 0x01, 0x00, 0x00, 0x00, 0x06, 0x0e, 0x2b, 0x34, 0x04, 0x01, 0x01,
        0x01,
    ]);

    pub const AUXILIARY: Auid = Auid([
        0x01, 0x03, 0x02, 0x03, 0x01, 0x00, 0x00, 0x00, 0x06, 0x0e, 0x2b, 0x34, 0x04, 0x01, 0x01,
        0x01,
    ]);

    pub const DATA: Auid = Auid([
        0x01, 0x03, 0x02, 0x03, 0x02, 0x00, 0x00, 0x00, 0x06, 0x0e, 0x2b, 0x34, 0x04, 0x01, 0x01,
        0x01,
    ]);

    // Class IDs
    pub const CLASS_HEADER: Auid = Auid([
        0x0d, 0x01, 0x01, 0x01, 0x01, 0x01, 0x2f, 0x00, 0x06, 0x0e, 0x2b, 0x34, 0x02, 0x06, 0x01,
        0x01,
    ]);

    pub const CLASS_COMPOSITION_MOB: Auid = Auid([
        0x0d, 0x01, 0x01, 0x01, 0x01, 0x01, 0x30, 0x00, 0x06, 0x0e, 0x2b, 0x34, 0x02, 0x06, 0x01,
        0x01,
    ]);

    pub const CLASS_MASTER_MOB: Auid = Auid([
        0x0d, 0x01, 0x01, 0x01, 0x01, 0x01, 0x32, 0x00, 0x06, 0x0e, 0x2b, 0x34, 0x02, 0x06, 0x01,
        0x01,
    ]);

    pub const CLASS_SOURCE_MOB: Auid = Auid([
        0x0d, 0x01, 0x01, 0x01, 0x01, 0x01, 0x33, 0x00, 0x06, 0x0e, 0x2b, 0x34, 0x02, 0x06, 0x01,
        0x01,
    ]);

    pub const CLASS_TIMELINE_MOB_SLOT: Auid = Auid([
        0x0d, 0x01, 0x01, 0x01, 0x01, 0x01, 0x37, 0x00, 0x06, 0x0e, 0x2b, 0x34, 0x02, 0x06, 0x01,
        0x01,
    ]);

    pub const CLASS_SEQUENCE: Auid = Auid([
        0x0d, 0x01, 0x01, 0x01, 0x01, 0x01, 0x0f, 0x00, 0x06, 0x0e, 0x2b, 0x34, 0x02, 0x06, 0x01,
        0x01,
    ]);

    pub const CLASS_SOURCE_CLIP: Auid = Auid([
        0x0d, 0x01, 0x01, 0x01, 0x01, 0x01, 0x11, 0x00, 0x06, 0x0e, 0x2b, 0x34, 0x02, 0x06, 0x01,
        0x01,
    ]);

    pub const CLASS_FILLER: Auid = Auid([
        0x0d, 0x01, 0x01, 0x01, 0x01, 0x01, 0x09, 0x00, 0x06, 0x0e, 0x2b, 0x34, 0x02, 0x06, 0x01,
        0x01,
    ]);

    pub const CLASS_TRANSITION: Auid = Auid([
        0x0d, 0x01, 0x01, 0x01, 0x01, 0x01, 0x12, 0x00, 0x06, 0x0e, 0x2b, 0x34, 0x02, 0x06, 0x01,
        0x01,
    ]);

    pub const CLASS_OPERATION_GROUP: Auid = Auid([
        0x0d, 0x01, 0x01, 0x01, 0x01, 0x01, 0x0a, 0x00, 0x06, 0x0e, 0x2b, 0x34, 0x02, 0x06, 0x01,
        0x01,
    ]);
}

impl fmt::Display for Auid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uuid())
    }
}

impl Default for Auid {
    fn default() -> Self {
        Self::null()
    }
}

impl From<Uuid> for Auid {
    fn from(uuid: Uuid) -> Self {
        Self::from_uuid(uuid)
    }
}

impl From<Auid> for Uuid {
    fn from(auid: Auid) -> Self {
        auid.to_uuid()
    }
}

/// AAF Dictionary
#[derive(Debug, Clone)]
pub struct Dictionary {
    /// Class definitions
    classes: HashMap<Auid, ClassDefinition>,
    /// Property definitions
    properties: HashMap<Auid, PropertyDefinition>,
    /// Type definitions
    types: HashMap<Auid, TypeDefinition>,
    /// Data definitions
    data_definitions: HashMap<Auid, DataDefinition>,
}

impl Dictionary {
    /// Create a new dictionary
    #[must_use]
    pub fn new() -> Self {
        let mut dict = Self {
            classes: HashMap::new(),
            properties: HashMap::new(),
            types: HashMap::new(),
            data_definitions: HashMap::new(),
        };

        // Add baseline AAF types
        dict.add_baseline_types();
        dict.add_baseline_classes();
        dict.add_baseline_properties();
        dict.add_baseline_data_definitions();

        dict
    }

    /// Add a class definition
    pub fn add_class(&mut self, class: ClassDefinition) {
        self.classes.insert(class.auid, class);
    }

    /// Add a property definition
    pub fn add_property(&mut self, property: PropertyDefinition) {
        self.properties.insert(property.auid, property);
    }

    /// Add a type definition
    pub fn add_type(&mut self, type_def: TypeDefinition) {
        self.types.insert(type_def.auid, type_def);
    }

    /// Add a data definition
    pub fn add_data_definition(&mut self, data_def: DataDefinition) {
        self.data_definitions.insert(data_def.auid, data_def);
    }

    /// Get a class definition by AUID
    #[must_use]
    pub fn get_class(&self, auid: &Auid) -> Option<&ClassDefinition> {
        self.classes.get(auid)
    }

    /// Get a property definition by AUID
    #[must_use]
    pub fn get_property(&self, auid: &Auid) -> Option<&PropertyDefinition> {
        self.properties.get(auid)
    }

    /// Get a type definition by AUID
    #[must_use]
    pub fn get_type(&self, auid: &Auid) -> Option<&TypeDefinition> {
        self.types.get(auid)
    }

    /// Get a data definition by AUID
    #[must_use]
    pub fn get_data_definition(&self, auid: &Auid) -> Option<&DataDefinition> {
        self.data_definitions.get(auid)
    }

    /// Get a class by name
    #[must_use]
    pub fn get_class_by_name(&self, name: &str) -> Option<&ClassDefinition> {
        self.classes.values().find(|c| c.name == name)
    }

    /// Get a property by name
    #[must_use]
    pub fn get_property_by_name(&self, name: &str) -> Option<&PropertyDefinition> {
        self.properties.values().find(|p| p.name == name)
    }

    /// Get a type by name
    #[must_use]
    pub fn get_type_by_name(&self, name: &str) -> Option<&TypeDefinition> {
        self.types.values().find(|t| t.name == name)
    }

    /// Add baseline type definitions
    fn add_baseline_types(&mut self) {
        let types = vec![
            TypeDefinition::new(Auid::null(), "Boolean", TypeKind::Integer),
            TypeDefinition::new(Auid::null(), "Int8", TypeKind::Integer),
            TypeDefinition::new(Auid::null(), "UInt8", TypeKind::Integer),
            TypeDefinition::new(Auid::null(), "Int16", TypeKind::Integer),
            TypeDefinition::new(Auid::null(), "UInt16", TypeKind::Integer),
            TypeDefinition::new(Auid::null(), "Int32", TypeKind::Integer),
            TypeDefinition::new(Auid::null(), "UInt32", TypeKind::Integer),
            TypeDefinition::new(Auid::null(), "Int64", TypeKind::Integer),
            TypeDefinition::new(Auid::null(), "UInt64", TypeKind::Integer),
            TypeDefinition::new(Auid::null(), "Float", TypeKind::Float),
            TypeDefinition::new(Auid::null(), "Double", TypeKind::Float),
            TypeDefinition::new(Auid::null(), "String", TypeKind::String),
            TypeDefinition::new(Auid::null(), "AUID", TypeKind::Record),
            TypeDefinition::new(Auid::null(), "MobID", TypeKind::Record),
            TypeDefinition::new(Auid::null(), "Position", TypeKind::Integer),
            TypeDefinition::new(Auid::null(), "Length", TypeKind::Integer),
        ];

        for type_def in types {
            self.types.insert(type_def.auid, type_def);
        }
    }

    /// Add baseline class definitions
    fn add_baseline_classes(&mut self) {
        let classes = vec![
            ClassDefinition::new(Auid::CLASS_HEADER, "Header", None),
            ClassDefinition::new(Auid::CLASS_COMPOSITION_MOB, "CompositionMob", None),
            ClassDefinition::new(Auid::CLASS_MASTER_MOB, "MasterMob", None),
            ClassDefinition::new(Auid::CLASS_SOURCE_MOB, "SourceMob", None),
            ClassDefinition::new(Auid::CLASS_TIMELINE_MOB_SLOT, "TimelineMobSlot", None),
            ClassDefinition::new(Auid::CLASS_SEQUENCE, "Sequence", None),
            ClassDefinition::new(Auid::CLASS_SOURCE_CLIP, "SourceClip", None),
            ClassDefinition::new(Auid::CLASS_FILLER, "Filler", None),
            ClassDefinition::new(Auid::CLASS_TRANSITION, "Transition", None),
            ClassDefinition::new(Auid::CLASS_OPERATION_GROUP, "OperationGroup", None),
        ];

        for class in classes {
            self.classes.insert(class.auid, class);
        }
    }

    /// Add baseline property definitions
    fn add_baseline_properties(&mut self) {
        // Properties are typically added dynamically as they're encountered
        // This is a placeholder for common properties
    }

    /// Add baseline data definitions
    fn add_baseline_data_definitions(&mut self) {
        let data_defs = vec![
            DataDefinition::new(Auid::PICTURE, "Picture"),
            DataDefinition::new(Auid::SOUND, "Sound"),
            DataDefinition::new(Auid::TIMECODE, "Timecode"),
            DataDefinition::new(Auid::EDGECODE, "Edgecode"),
        ];

        for data_def in data_defs {
            self.data_definitions.insert(data_def.auid, data_def);
        }
    }
}

impl Default for Dictionary {
    fn default() -> Self {
        Self::new()
    }
}

/// Class definition
#[derive(Debug, Clone)]
pub struct ClassDefinition {
    /// Class AUID
    pub auid: Auid,
    /// Class name
    pub name: String,
    /// Parent class AUID
    pub parent: Option<Auid>,
    /// Properties
    pub properties: Vec<Auid>,
    /// Is concrete (can be instantiated)
    pub is_concrete: bool,
}

impl ClassDefinition {
    /// Create a new class definition
    pub fn new(auid: Auid, name: impl Into<String>, parent: Option<Auid>) -> Self {
        Self {
            auid,
            name: name.into(),
            parent,
            properties: Vec::new(),
            is_concrete: true,
        }
    }

    /// Add a property to this class
    pub fn add_property(&mut self, property_auid: Auid) {
        if !self.properties.contains(&property_auid) {
            self.properties.push(property_auid);
        }
    }

    /// Check if this class has a property
    #[must_use]
    pub fn has_property(&self, property_auid: &Auid) -> bool {
        self.properties.contains(property_auid)
    }
}

/// Property definition
#[derive(Debug, Clone)]
pub struct PropertyDefinition {
    /// Property AUID
    pub auid: Auid,
    /// Property name
    pub name: String,
    /// Type AUID
    pub type_auid: Auid,
    /// Is optional
    pub is_optional: bool,
    /// Local identification (for binary encoding)
    pub local_id: Option<u16>,
}

impl PropertyDefinition {
    /// Create a new property definition
    pub fn new(auid: Auid, name: impl Into<String>, type_auid: Auid) -> Self {
        Self {
            auid,
            name: name.into(),
            type_auid,
            is_optional: false,
            local_id: None,
        }
    }

    /// Set if this property is optional
    #[must_use]
    pub fn with_optional(mut self, optional: bool) -> Self {
        self.is_optional = optional;
        self
    }

    /// Set local ID
    #[must_use]
    pub fn with_local_id(mut self, local_id: u16) -> Self {
        self.local_id = Some(local_id);
        self
    }
}

/// Type definition
#[derive(Debug, Clone)]
pub struct TypeDefinition {
    /// Type AUID
    pub auid: Auid,
    /// Type name
    pub name: String,
    /// Type kind
    pub kind: TypeKind,
}

impl TypeDefinition {
    /// Create a new type definition
    pub fn new(auid: Auid, name: impl Into<String>, kind: TypeKind) -> Self {
        Self {
            auid,
            name: name.into(),
            kind,
        }
    }

    /// Get the type name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Type kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeKind {
    /// Integer type
    Integer,
    /// Floating point type
    Float,
    /// String type
    String,
    /// Record (struct) type
    Record,
    /// Enumeration type
    Enum,
    /// Fixed array type
    FixedArray,
    /// Variable array type
    VariableArray,
    /// Set type
    Set,
    /// Strong reference
    StrongRef,
    /// Weak reference
    WeakRef,
    /// Opaque type
    Opaque,
}

/// Data definition
#[derive(Debug, Clone)]
pub struct DataDefinition {
    /// Data definition AUID
    pub auid: Auid,
    /// Name
    pub name: String,
    /// Description
    pub description: Option<String>,
}

impl DataDefinition {
    /// Create a new data definition
    pub fn new(auid: Auid, name: impl Into<String>) -> Self {
        Self {
            auid,
            name: name.into(),
            description: None,
        }
    }

    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Read dictionary from AAF file
pub fn read_dictionary<R: Read + Seek>(_storage: &mut StorageReader<R>) -> Result<Dictionary> {
    // In a real implementation, we would read the dictionary from the
    // "MetaDictionary" stream in the AAF file
    // For now, return a baseline dictionary
    Ok(Dictionary::new())
}

/// Dictionary builder for extensibility
pub struct DictionaryBuilder {
    dictionary: Dictionary,
}

impl DictionaryBuilder {
    /// Create a new dictionary builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            dictionary: Dictionary::new(),
        }
    }

    /// Add a custom class
    pub fn add_custom_class(
        mut self,
        auid: Auid,
        name: impl Into<String>,
        parent: Option<Auid>,
    ) -> Self {
        let class = ClassDefinition::new(auid, name, parent);
        self.dictionary.add_class(class);
        self
    }

    /// Add a custom property
    pub fn add_custom_property(
        mut self,
        auid: Auid,
        name: impl Into<String>,
        type_auid: Auid,
    ) -> Self {
        let property = PropertyDefinition::new(auid, name, type_auid);
        self.dictionary.add_property(property);
        self
    }

    /// Add a custom type
    pub fn add_custom_type(mut self, auid: Auid, name: impl Into<String>, kind: TypeKind) -> Self {
        let type_def = TypeDefinition::new(auid, name, kind);
        self.dictionary.add_type(type_def);
        self
    }

    /// Build the dictionary
    #[must_use]
    pub fn build(self) -> Dictionary {
        self.dictionary
    }
}

impl Default for DictionaryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auid_creation() {
        let bytes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let auid = Auid::from_bytes(&bytes);
        assert_eq!(auid.as_bytes(), &bytes);
    }

    #[test]
    fn test_auid_null() {
        let auid = Auid::null();
        assert!(auid.is_null());
    }

    #[test]
    fn test_auid_uuid_conversion() {
        let uuid = Uuid::new_v4();
        let auid = Auid::from_uuid(uuid);
        assert_eq!(auid.to_uuid(), uuid);
    }

    #[test]
    fn test_auid_data_definitions() {
        assert!(Auid::PICTURE.is_picture());
        assert!(Auid::SOUND.is_sound());
        assert!(Auid::TIMECODE.is_timecode());
    }

    #[test]
    fn test_dictionary_creation() {
        let dict = Dictionary::new();
        assert!(!dict.classes.is_empty());
        assert!(!dict.types.is_empty());
    }

    #[test]
    fn test_dictionary_get_class() {
        let dict = Dictionary::new();
        let class = dict.get_class(&Auid::CLASS_HEADER);
        assert!(class.is_some());
        assert_eq!(class.expect("test expectation failed").name, "Header");
    }

    #[test]
    fn test_dictionary_get_class_by_name() {
        let dict = Dictionary::new();
        let class = dict.get_class_by_name("CompositionMob");
        assert!(class.is_some());
        assert_eq!(
            class.expect("test expectation failed").auid,
            Auid::CLASS_COMPOSITION_MOB
        );
    }

    #[test]
    fn test_class_definition() {
        let mut class = ClassDefinition::new(Auid::null(), "TestClass", None);
        assert_eq!(class.name, "TestClass");
        assert!(class.is_concrete);

        let prop_auid = Auid::null();
        class.add_property(prop_auid);
        assert!(class.has_property(&prop_auid));
    }

    #[test]
    fn test_property_definition() {
        let prop = PropertyDefinition::new(Auid::null(), "TestProp", Auid::null())
            .with_optional(true)
            .with_local_id(0x1234);

        assert_eq!(prop.name, "TestProp");
        assert!(prop.is_optional);
        assert_eq!(prop.local_id, Some(0x1234));
    }

    #[test]
    fn test_type_definition() {
        let type_def = TypeDefinition::new(Auid::null(), "TestType", TypeKind::Integer);
        assert_eq!(type_def.name(), "TestType");
        assert_eq!(type_def.kind, TypeKind::Integer);
    }

    #[test]
    fn test_dictionary_builder() {
        let dict = DictionaryBuilder::new()
            .add_custom_class(Auid::null(), "CustomClass", None)
            .add_custom_property(Auid::null(), "CustomProp", Auid::null())
            .add_custom_type(Auid::null(), "CustomType", TypeKind::String)
            .build();

        assert!(dict.get_class_by_name("CustomClass").is_some());
    }
}
