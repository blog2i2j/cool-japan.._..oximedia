//! GPU resource allocation and lifetime tracking.
#![allow(dead_code)]

use std::collections::HashMap;

/// Category of a GPU resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    /// Device-local buffer (lives on the GPU).
    GpuBuffer,
    /// Host-visible / staging buffer.
    HostBuffer,
    /// 2-D texture / image.
    Texture2D,
    /// 3-D texture / volume.
    Texture3D,
    /// Sampler object.
    Sampler,
    /// Shader resource view / descriptor.
    Descriptor,
}

impl ResourceType {
    /// Returns `true` for resources that reside in device (GPU) memory.
    #[must_use]
    pub fn is_gpu_memory(&self) -> bool {
        matches!(self, Self::GpuBuffer | Self::Texture2D | Self::Texture3D)
    }

    /// Returns `true` for resources that are textures of any dimension.
    #[must_use]
    pub fn is_texture(&self) -> bool {
        matches!(self, Self::Texture2D | Self::Texture3D)
    }
}

/// Opaque handle representing an allocated GPU resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ResourceHandle {
    id: u64,
}

impl ResourceHandle {
    /// Create a new handle from a raw id.
    #[must_use]
    pub(crate) fn new(id: u64) -> Self {
        Self { id }
    }

    /// Returns `true` if this handle has a non-zero id (i.e. was produced by
    /// a successful allocation, not a default / null construction).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.id != 0
    }

    /// Raw numeric id (useful for logging).
    #[must_use]
    pub fn raw_id(&self) -> u64 {
        self.id
    }
}

/// Metadata stored alongside each allocated resource.
#[derive(Debug, Clone)]
struct ResourceMeta {
    resource_type: ResourceType,
    size_bytes: usize,
    label: String,
}

/// Simple GPU resource manager that tracks allocations and their sizes.
pub struct GpuResourceManager {
    resources: HashMap<ResourceHandle, ResourceMeta>,
    next_id: u64,
}

impl GpuResourceManager {
    /// Create a new, empty resource manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
            next_id: 1,
        }
    }

    /// Allocate a new resource.
    ///
    /// Returns a valid [`ResourceHandle`] on success, or an error string if
    /// `size_bytes` is zero.
    pub fn allocate(
        &mut self,
        resource_type: ResourceType,
        size_bytes: usize,
        label: impl Into<String>,
    ) -> Result<ResourceHandle, String> {
        if size_bytes == 0 {
            return Err("Cannot allocate a zero-byte resource".to_string());
        }
        let handle = ResourceHandle::new(self.next_id);
        self.next_id += 1;
        self.resources.insert(
            handle,
            ResourceMeta {
                resource_type,
                size_bytes,
                label: label.into(),
            },
        );
        Ok(handle)
    }

    /// Free a previously allocated resource.
    ///
    /// Returns `true` if the resource existed and was freed, `false` otherwise.
    pub fn free(&mut self, handle: ResourceHandle) -> bool {
        self.resources.remove(&handle).is_some()
    }

    /// Total bytes currently allocated in GPU memory (device-local resources).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn total_gpu_bytes(&self) -> usize {
        self.resources
            .values()
            .filter(|m| m.resource_type.is_gpu_memory())
            .map(|m| m.size_bytes)
            .sum()
    }

    /// Total bytes across all resource types.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.resources.values().map(|m| m.size_bytes).sum()
    }

    /// Number of live allocations.
    #[must_use]
    pub fn allocation_count(&self) -> usize {
        self.resources.len()
    }

    /// Query the type of a resource by handle.
    #[must_use]
    pub fn resource_type(&self, handle: ResourceHandle) -> Option<ResourceType> {
        self.resources.get(&handle).map(|m| m.resource_type)
    }

    /// Query the size of a resource in bytes.
    #[must_use]
    pub fn resource_size(&self, handle: ResourceHandle) -> Option<usize> {
        self.resources.get(&handle).map(|m| m.size_bytes)
    }

    /// Query the label of a resource.
    #[must_use]
    pub fn resource_label(&self, handle: ResourceHandle) -> Option<&str> {
        self.resources.get(&handle).map(|m| m.label.as_str())
    }

    /// Returns `true` if the handle refers to a live allocation.
    #[must_use]
    pub fn is_alive(&self, handle: ResourceHandle) -> bool {
        self.resources.contains_key(&handle)
    }
}

impl Default for GpuResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> GpuResourceManager {
        GpuResourceManager::new()
    }

    // --- ResourceType tests ---

    #[test]
    fn test_gpu_buffer_is_gpu_memory() {
        assert!(ResourceType::GpuBuffer.is_gpu_memory());
    }

    #[test]
    fn test_host_buffer_not_gpu_memory() {
        assert!(!ResourceType::HostBuffer.is_gpu_memory());
    }

    #[test]
    fn test_texture2d_is_gpu_memory() {
        assert!(ResourceType::Texture2D.is_gpu_memory());
    }

    #[test]
    fn test_sampler_not_gpu_memory() {
        assert!(!ResourceType::Sampler.is_gpu_memory());
    }

    #[test]
    fn test_texture3d_is_texture() {
        assert!(ResourceType::Texture3D.is_texture());
    }

    #[test]
    fn test_gpu_buffer_not_texture() {
        assert!(!ResourceType::GpuBuffer.is_texture());
    }

    // --- ResourceHandle tests ---

    #[test]
    fn test_default_handle_invalid() {
        let h = ResourceHandle::default();
        assert!(!h.is_valid());
    }

    #[test]
    fn test_allocated_handle_valid() {
        let mut mgr = make_manager();
        let h = mgr
            .allocate(ResourceType::GpuBuffer, 1024, "buf")
            .expect("allocation should succeed");
        assert!(h.is_valid());
    }

    // --- GpuResourceManager tests ---

    #[test]
    fn test_allocate_returns_valid_handle() {
        let mut mgr = make_manager();
        let h = mgr
            .allocate(ResourceType::Texture2D, 4096, "tex")
            .expect("allocation should succeed");
        assert!(h.is_valid());
    }

    #[test]
    fn test_allocate_zero_bytes_returns_error() {
        let mut mgr = make_manager();
        assert!(mgr.allocate(ResourceType::GpuBuffer, 0, "bad").is_err());
    }

    #[test]
    fn test_allocation_count_increases() {
        let mut mgr = make_manager();
        mgr.allocate(ResourceType::GpuBuffer, 256, "a")
            .expect("allocation should succeed");
        mgr.allocate(ResourceType::HostBuffer, 512, "b")
            .expect("allocation should succeed");
        assert_eq!(mgr.allocation_count(), 2);
    }

    #[test]
    fn test_total_gpu_bytes_only_counts_gpu_resources() {
        let mut mgr = make_manager();
        mgr.allocate(ResourceType::GpuBuffer, 1024, "gpu")
            .expect("allocation should succeed");
        mgr.allocate(ResourceType::HostBuffer, 2048, "host")
            .expect("operation should succeed in test");
        assert_eq!(mgr.total_gpu_bytes(), 1024);
    }

    #[test]
    fn test_total_bytes_counts_all() {
        let mut mgr = make_manager();
        mgr.allocate(ResourceType::GpuBuffer, 1024, "gpu")
            .expect("allocation should succeed");
        mgr.allocate(ResourceType::HostBuffer, 2048, "host")
            .expect("operation should succeed in test");
        assert_eq!(mgr.total_bytes(), 3072);
    }

    #[test]
    fn test_free_removes_resource() {
        let mut mgr = make_manager();
        let h = mgr
            .allocate(ResourceType::GpuBuffer, 512, "buf")
            .expect("allocation should succeed");
        assert!(mgr.free(h));
        assert_eq!(mgr.allocation_count(), 0);
    }

    #[test]
    fn test_free_invalid_handle_returns_false() {
        let mut mgr = make_manager();
        let h = ResourceHandle::default();
        assert!(!mgr.free(h));
    }

    #[test]
    fn test_resource_type_query() {
        let mut mgr = make_manager();
        let h = mgr
            .allocate(ResourceType::Sampler, 64, "s")
            .expect("allocation should succeed");
        assert_eq!(mgr.resource_type(h), Some(ResourceType::Sampler));
    }

    #[test]
    fn test_resource_size_query() {
        let mut mgr = make_manager();
        let h = mgr
            .allocate(ResourceType::Texture2D, 8192, "t")
            .expect("allocation should succeed");
        assert_eq!(mgr.resource_size(h), Some(8192));
    }

    #[test]
    fn test_resource_label_query() {
        let mut mgr = make_manager();
        let h = mgr
            .allocate(ResourceType::GpuBuffer, 128, "my_label")
            .expect("operation should succeed in test");
        assert_eq!(mgr.resource_label(h), Some("my_label"));
    }

    #[test]
    fn test_is_alive_after_free_is_false() {
        let mut mgr = make_manager();
        let h = mgr
            .allocate(ResourceType::GpuBuffer, 256, "buf")
            .expect("allocation should succeed");
        mgr.free(h);
        assert!(!mgr.is_alive(h));
    }
}
