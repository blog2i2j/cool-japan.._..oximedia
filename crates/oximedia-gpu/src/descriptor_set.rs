#![allow(dead_code)]
//! Descriptor set and layout management for GPU pipeline bindings.

/// The kind of resource bound at a particular descriptor slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingType {
    /// A uniform (constant) buffer.
    UniformBuffer,
    /// A storage buffer that can be read/written by a shader.
    StorageBuffer,
    /// A read-only sampled texture.
    SampledTexture,
    /// A read/write storage texture.
    StorageTexture,
    /// A combined image + sampler.
    CombinedImageSampler,
}

impl BindingType {
    /// Returns `true` if this binding type is backed by a buffer (not a texture).
    #[must_use]
    pub fn is_buffer(&self) -> bool {
        matches!(self, Self::UniformBuffer | Self::StorageBuffer)
    }

    /// Returns `true` if this binding type involves a texture.
    #[must_use]
    pub fn is_texture(&self) -> bool {
        matches!(
            self,
            Self::SampledTexture | Self::StorageTexture | Self::CombinedImageSampler
        )
    }

    /// Returns `true` if the shader can write to this resource.
    #[must_use]
    pub fn is_writable(&self) -> bool {
        matches!(self, Self::StorageBuffer | Self::StorageTexture)
    }
}

/// A single binding within a descriptor set.
#[derive(Debug, Clone)]
pub struct DescriptorBinding {
    /// The binding slot index.
    pub slot: u32,
    /// The type of resource at this slot.
    pub binding_type: BindingType,
    /// The number of array elements (1 for non-array bindings).
    pub count: u32,
    /// An optional debug label.
    pub label: Option<String>,
}

impl DescriptorBinding {
    /// Create a new binding for a single resource.
    #[must_use]
    pub fn new(slot: u32, binding_type: BindingType) -> Self {
        Self {
            slot,
            binding_type,
            count: 1,
            label: None,
        }
    }

    /// Create a new array binding.
    #[must_use]
    pub fn array(slot: u32, binding_type: BindingType, count: u32) -> Self {
        Self {
            slot,
            binding_type,
            count,
            label: None,
        }
    }

    /// Attach a debug label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// A binding is valid when `count` is at least 1.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.count >= 1
    }
}

/// A descriptor set: a collection of resource bindings.
#[derive(Debug, Default)]
pub struct DescriptorSet {
    bindings: Vec<DescriptorBinding>,
}

impl DescriptorSet {
    /// Create an empty descriptor set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a binding to the set, replacing any existing binding at the same slot.
    pub fn add_binding(&mut self, binding: DescriptorBinding) {
        if let Some(existing) = self.bindings.iter_mut().find(|b| b.slot == binding.slot) {
            *existing = binding;
        } else {
            self.bindings.push(binding);
        }
    }

    /// Total number of bindings in the set.
    #[must_use]
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }

    /// Retrieve a binding by its slot number.
    #[must_use]
    pub fn get_binding(&self, slot: u32) -> Option<&DescriptorBinding> {
        self.bindings.iter().find(|b| b.slot == slot)
    }

    /// Returns all buffer bindings.
    #[must_use]
    pub fn buffer_bindings(&self) -> Vec<&DescriptorBinding> {
        self.bindings
            .iter()
            .filter(|b| b.binding_type.is_buffer())
            .collect()
    }

    /// Returns all texture bindings.
    #[must_use]
    pub fn texture_bindings(&self) -> Vec<&DescriptorBinding> {
        self.bindings
            .iter()
            .filter(|b| b.binding_type.is_texture())
            .collect()
    }
}

/// A descriptor layout: the schema that describes a set's bindings
/// without being tied to specific resources.
#[derive(Debug, Default)]
pub struct DescriptorLayout {
    entries: Vec<DescriptorBinding>,
}

impl DescriptorLayout {
    /// Create an empty layout.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a binding entry to the layout.
    pub fn add_entry(&mut self, entry: DescriptorBinding) {
        self.entries.push(entry);
    }

    /// All binding entries in this layout.
    #[must_use]
    pub fn bindings(&self) -> &[DescriptorBinding] {
        &self.entries
    }

    /// Number of binding entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when there are no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binding_type_is_buffer_uniform() {
        assert!(BindingType::UniformBuffer.is_buffer());
    }

    #[test]
    fn test_binding_type_is_buffer_storage() {
        assert!(BindingType::StorageBuffer.is_buffer());
    }

    #[test]
    fn test_binding_type_is_buffer_texture_false() {
        assert!(!BindingType::SampledTexture.is_buffer());
    }

    #[test]
    fn test_binding_type_is_texture() {
        assert!(BindingType::SampledTexture.is_texture());
        assert!(BindingType::StorageTexture.is_texture());
        assert!(BindingType::CombinedImageSampler.is_texture());
    }

    #[test]
    fn test_binding_type_is_writable() {
        assert!(BindingType::StorageBuffer.is_writable());
        assert!(BindingType::StorageTexture.is_writable());
        assert!(!BindingType::UniformBuffer.is_writable());
        assert!(!BindingType::SampledTexture.is_writable());
    }

    #[test]
    fn test_descriptor_binding_is_valid() {
        let b = DescriptorBinding::new(0, BindingType::UniformBuffer);
        assert!(b.is_valid());
    }

    #[test]
    fn test_descriptor_binding_array_count() {
        let b = DescriptorBinding::array(1, BindingType::SampledTexture, 4);
        assert_eq!(b.count, 4);
        assert!(b.is_valid());
    }

    #[test]
    fn test_descriptor_binding_with_label() {
        let b = DescriptorBinding::new(2, BindingType::StorageBuffer).with_label("my_buf");
        assert_eq!(b.label.as_deref(), Some("my_buf"));
    }

    #[test]
    fn test_descriptor_set_add_binding_count() {
        let mut set = DescriptorSet::new();
        set.add_binding(DescriptorBinding::new(0, BindingType::UniformBuffer));
        set.add_binding(DescriptorBinding::new(1, BindingType::SampledTexture));
        assert_eq!(set.binding_count(), 2);
    }

    #[test]
    fn test_descriptor_set_replace_binding() {
        let mut set = DescriptorSet::new();
        set.add_binding(DescriptorBinding::new(0, BindingType::UniformBuffer));
        set.add_binding(DescriptorBinding::new(0, BindingType::StorageBuffer));
        assert_eq!(set.binding_count(), 1);
        assert_eq!(
            set.get_binding(0)
                .expect("binding should exist")
                .binding_type,
            BindingType::StorageBuffer
        );
    }

    #[test]
    fn test_descriptor_set_buffer_bindings() {
        let mut set = DescriptorSet::new();
        set.add_binding(DescriptorBinding::new(0, BindingType::UniformBuffer));
        set.add_binding(DescriptorBinding::new(1, BindingType::SampledTexture));
        set.add_binding(DescriptorBinding::new(2, BindingType::StorageBuffer));
        assert_eq!(set.buffer_bindings().len(), 2);
    }

    #[test]
    fn test_descriptor_layout_bindings() {
        let mut layout = DescriptorLayout::new();
        layout.add_entry(DescriptorBinding::new(0, BindingType::UniformBuffer));
        layout.add_entry(DescriptorBinding::new(1, BindingType::StorageTexture));
        assert_eq!(layout.bindings().len(), 2);
        assert!(!layout.is_empty());
    }

    #[test]
    fn test_descriptor_layout_empty() {
        let layout = DescriptorLayout::new();
        assert!(layout.is_empty());
        assert_eq!(layout.len(), 0);
    }
}
