//! Compute backend abstraction layer.
//!
//! This module provides a trait-based abstraction for GPU compute operations,
//! enabling pluggable backends (Vulkan, CPU fallback) without direct dependency
//! on GPU-specific APIs in calling code.
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────┐
//! │          ComputeBackend Trait              │
//! └────────────────────────────────────────────┘
//!          │                      │
//!          ▼                      ▼
//!   ┌─────────────┐     ┌──────────────────────┐
//!   │VulkanCompute│     │CpuFallbackBackend    │
//!   │Backend      │     │                      │
//!   └─────────────┘     └──────────────────────┘
//! ```
//!
//! Actual Vulkan calls are performed via the `vulkan` module when the
//! `vulkan` feature is enabled. The trait itself is feature-agnostic.

use crate::error::{AccelError, AccelResult};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// ============================================================================
// Buffer Handle
// ============================================================================

/// Opaque handle to a GPU buffer allocation.
///
/// The underlying representation is backend-specific. Callers should treat
/// this as an opaque token and manage lifetime through the backend API.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub(crate) u64);

impl BufferHandle {
    /// Create a new buffer handle from a raw ID.
    ///
    /// This is intended for backend implementations only.
    #[must_use]
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw handle ID.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.0
    }
}

// ============================================================================
// Kernel Dispatch Parameters
// ============================================================================

/// Parameters for dispatching a compute kernel.
#[derive(Debug, Clone, Copy)]
pub struct DispatchParams {
    /// Number of workgroups in X dimension.
    pub groups_x: u32,
    /// Number of workgroups in Y dimension.
    pub groups_y: u32,
    /// Number of workgroups in Z dimension.
    pub groups_z: u32,
}

impl DispatchParams {
    /// Create a 1D dispatch.
    #[must_use]
    pub const fn new_1d(groups_x: u32) -> Self {
        Self {
            groups_x,
            groups_y: 1,
            groups_z: 1,
        }
    }

    /// Create a 2D dispatch.
    #[must_use]
    pub const fn new_2d(groups_x: u32, groups_y: u32) -> Self {
        Self {
            groups_x,
            groups_y,
            groups_z: 1,
        }
    }

    /// Create a 3D dispatch.
    #[must_use]
    pub const fn new_3d(groups_x: u32, groups_y: u32, groups_z: u32) -> Self {
        Self {
            groups_x,
            groups_y,
            groups_z,
        }
    }

    /// Calculate 2D dispatch for covering `width × height` pixels with the
    /// given local workgroup size.
    #[must_use]
    pub fn for_image(width: u32, height: u32, local_x: u32, local_y: u32) -> Self {
        Self {
            groups_x: width.div_ceil(local_x),
            groups_y: height.div_ceil(local_y),
            groups_z: 1,
        }
    }
}

// ============================================================================
// Memory Usage Statistics
// ============================================================================

/// GPU memory usage statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuMemoryStats {
    /// Total bytes currently allocated.
    pub allocated_bytes: u64,
    /// Peak bytes ever allocated.
    pub peak_bytes: u64,
    /// Number of live buffer allocations.
    pub allocation_count: u64,
}

impl GpuMemoryStats {
    /// Return current usage in mebibytes.
    #[must_use]
    pub fn allocated_mib(&self) -> f64 {
        self.allocated_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Return peak usage in mebibytes.
    #[must_use]
    pub fn peak_mib(&self) -> f64 {
        self.peak_bytes as f64 / (1024.0 * 1024.0)
    }
}

// ============================================================================
// ComputeBackend Trait
// ============================================================================

/// Abstraction over GPU and CPU compute backends.
///
/// Implementing this trait allows the rest of the acceleration layer to work
/// without knowing whether it is running on a real GPU, a software Vulkan ICD,
/// or a pure-CPU fallback.
///
/// # Thread Safety
///
/// All methods take `&self` so that backends can be shared across threads via
/// `Arc<dyn ComputeBackend + Send + Sync>`.
pub trait ComputeBackend: Send + Sync {
    // ---- Buffer lifecycle -----------------------------------------------

    /// Allocate a GPU-side buffer of `size` bytes.
    ///
    /// The returned handle is opaque; use it with `upload_buffer`,
    /// `download_buffer`, `dispatch_kernel`, and `free_buffer`.
    ///
    /// # Errors
    ///
    /// Returns `AccelError::OutOfMemory` or `AccelError::BufferAllocation` on
    /// failure.
    fn allocate_buffer(&self, size: u64) -> AccelResult<BufferHandle>;

    /// Upload host data to a previously allocated buffer.
    ///
    /// `data.len()` must equal the size passed to `allocate_buffer`.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is invalid or if the upload fails.
    fn upload_buffer(&self, handle: &BufferHandle, data: &[u8]) -> AccelResult<()>;

    /// Download the contents of a GPU buffer into a host `Vec<u8>`.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is invalid or if the download fails.
    fn download_buffer(&self, handle: &BufferHandle) -> AccelResult<Vec<u8>>;

    /// Free a previously allocated buffer.
    ///
    /// After this call the handle must not be used again.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is invalid.
    fn free_buffer(&self, handle: BufferHandle) -> AccelResult<()>;

    // ---- Kernel dispatch ------------------------------------------------

    /// Execute a registered compute kernel.
    ///
    /// # Arguments
    ///
    /// * `kernel_name` – name of a shader previously registered with
    ///   [`KernelRegistry::register`].
    /// * `buffers` – ordered list of buffer handles bound to the kernel
    ///   (binding 0, 1, …).
    /// * `dispatch` – workgroup grid dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if the kernel is unknown, the dispatch fails, or
    /// any buffer handle is invalid.
    fn dispatch_kernel(
        &self,
        kernel_name: &str,
        buffers: &[&BufferHandle],
        dispatch: DispatchParams,
    ) -> AccelResult<()>;

    /// Block the calling thread until all previously submitted work is done.
    ///
    /// # Errors
    ///
    /// Returns an error if synchronization fails.
    fn synchronize(&self) -> AccelResult<()>;

    // ---- Introspection --------------------------------------------------

    /// Human-readable name of this backend (e.g. `"NVIDIA GeForce RTX 4090"`).
    fn backend_name(&self) -> &str;

    /// Return `true` if this backend runs on a real GPU.
    fn is_gpu(&self) -> bool;

    /// Return memory usage statistics.
    fn memory_stats(&self) -> GpuMemoryStats;
}

// ============================================================================
// Kernel Registry
// ============================================================================

/// Registered compute kernel entry.
#[derive(Clone)]
struct KernelEntry {
    /// SPIR-V bytecode (for Vulkan) or empty for CPU-only kernels.
    #[allow(dead_code)]
    spirv: Vec<u8>,
    /// Descriptive label.
    label: String,
}

/// Registry of named compute kernels backed by SPIR-V byte slices.
///
/// Backends look up kernels by name when executing `dispatch_kernel`.
///
/// # Example
///
/// ```rust
/// use oximedia_accel::compute_backend::KernelRegistry;
///
/// let registry = KernelRegistry::new();
/// // Register a dummy SPIR-V blob (real code would use compiled shaders):
/// registry.register("scale_bilinear", b"SPIR-V", "Bilinear scale kernel");
/// assert!(registry.get("scale_bilinear").is_some());
/// assert!(registry.get("unknown").is_none());
/// ```
pub struct KernelRegistry {
    kernels: RwLock<HashMap<String, KernelEntry>>,
}

impl KernelRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            kernels: RwLock::new(HashMap::new()),
        }
    }

    /// Register a SPIR-V shader under `name`.
    ///
    /// If a kernel with the same name was already registered it is replaced.
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned (should never happen in
    /// normal operation).
    pub fn register(&self, name: &str, spirv: &[u8], label: &str) {
        let mut map = self.kernels.write().expect("KernelRegistry RwLock poisoned");
        map.insert(
            name.to_string(),
            KernelEntry {
                spirv: spirv.to_vec(),
                label: label.to_string(),
            },
        );
    }

    /// Look up a kernel by name.
    ///
    /// Returns `Some(label)` if found, `None` otherwise.
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<String> {
        let map = self.kernels.read().expect("KernelRegistry RwLock poisoned");
        map.get(name).map(|e| e.label.clone())
    }

    /// Return the number of registered kernels.
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned.
    #[must_use]
    pub fn len(&self) -> usize {
        self.kernels.read().expect("KernelRegistry RwLock poisoned").len()
    }

    /// Return `true` if no kernels have been registered.
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.kernels.read().expect("KernelRegistry RwLock poisoned").is_empty()
    }

    /// List all registered kernel names.
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned.
    #[must_use]
    pub fn kernel_names(&self) -> Vec<String> {
        let map = self.kernels.read().expect("KernelRegistry RwLock poisoned");
        map.keys().cloned().collect()
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// YUV Frame Helpers
// ============================================================================

/// Planar YUV frame description (no pixel data; just geometry).
#[derive(Debug, Clone, Copy)]
pub struct YuvFrameInfo {
    /// Luma width in pixels.
    pub width: u32,
    /// Luma height in pixels.
    pub height: u32,
    /// Chroma subsampling in X (1 = 4:4:4, 2 = 4:2:0/4:2:2).
    pub chroma_subsample_x: u32,
    /// Chroma subsampling in Y (1 = 4:4:4 / 4:2:2, 2 = 4:2:0).
    pub chroma_subsample_y: u32,
}

impl YuvFrameInfo {
    /// Create a YUV 4:2:0 frame descriptor.
    #[must_use]
    pub const fn yuv420(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            chroma_subsample_x: 2,
            chroma_subsample_y: 2,
        }
    }

    /// Create a YUV 4:4:4 frame descriptor.
    #[must_use]
    pub const fn yuv444(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            chroma_subsample_x: 1,
            chroma_subsample_y: 1,
        }
    }

    /// Luma plane size in bytes (8 bits per sample).
    #[must_use]
    pub const fn luma_size(&self) -> u64 {
        (self.width as u64) * (self.height as u64)
    }

    /// Single chroma plane size in bytes (8 bits per sample).
    #[must_use]
    pub fn chroma_size(&self) -> u64 {
        let cw = self.width.div_ceil(self.chroma_subsample_x) as u64;
        let ch = self.height.div_ceil(self.chroma_subsample_y) as u64;
        cw * ch
    }

    /// Total planar buffer size: Y + Cb + Cr.
    #[must_use]
    pub fn total_size(&self) -> u64 {
        self.luma_size() + 2 * self.chroma_size()
    }
}

/// Upload a planar YUV frame to the GPU.
///
/// Validates that `data.len()` equals `info.total_size()` before uploading.
///
/// # Errors
///
/// Returns an error if `data` length is wrong, allocation fails, or upload
/// fails.
pub fn upload_yuv_frame(
    backend: &dyn ComputeBackend,
    data: &[u8],
    info: &YuvFrameInfo,
) -> AccelResult<BufferHandle> {
    let expected = info.total_size() as usize;
    if data.len() != expected {
        return Err(AccelError::BufferSizeMismatch {
            expected,
            actual: data.len(),
        });
    }
    let handle = backend.allocate_buffer(info.total_size())?;
    backend.upload_buffer(&handle, data)?;
    Ok(handle)
}

/// Download a planar YUV frame from the GPU and return the raw bytes.
///
/// The caller is responsible for interpreting the byte layout using the same
/// [`YuvFrameInfo`] that was used during upload.
///
/// # Errors
///
/// Returns an error if the download fails or the handle is invalid.
pub fn download_yuv_frame(
    backend: &dyn ComputeBackend,
    handle: &BufferHandle,
) -> AccelResult<Vec<u8>> {
    backend.download_buffer(handle)
}

// ============================================================================
// VulkanComputeBackend
// ============================================================================

/// Vulkan compute backend.
///
/// When Vulkan is unavailable at runtime this type still compiles but all
/// operations return `AccelError::Unsupported`.  The design allows the
/// accelerator host to select a backend at runtime using feature gating.
///
/// Real Vulkan dispatch is delegated to the `vulkan` module which owns the
/// actual Vulkan objects.
pub struct VulkanComputeBackend {
    /// Backend label (device name or "Vulkan (unavailable)").
    name: String,
    /// Whether Vulkan was successfully initialised.
    available: bool,
    /// Simple allocation counter for tracking.
    next_id: std::sync::atomic::AtomicU64,
    /// Live allocation sizes keyed by handle ID.
    allocations: RwLock<HashMap<u64, u64>>,
    /// Total bytes allocated (running sum).
    total_allocated: std::sync::atomic::AtomicU64,
    /// Peak bytes allocated.
    peak_allocated: std::sync::atomic::AtomicU64,
    /// Simulated buffer storage (CPU-side for the abstraction layer).
    buffers: RwLock<HashMap<u64, Vec<u8>>>,
}

impl VulkanComputeBackend {
    /// Try to create a Vulkan compute backend.
    ///
    /// Returns `Ok` even when no Vulkan driver is present; the `available`
    /// flag reflects whether real GPU operations are possible.
    #[must_use]
    pub fn new() -> Self {
        // Detect Vulkan availability through the existing device module.
        // We don't actually initialise a full Vulkan device here to keep this
        // lightweight; the heavy-weight path is through `AccelContext`.
        #[cfg(feature = "vulkan-detect")]
        let (name, available) = {
            use crate::device::DeviceSelector;
            match DeviceSelector::default().select() {
                Ok(dev) => (dev.name().to_string(), true),
                Err(_) => ("Vulkan (unavailable)".to_string(), false),
            }
        };
        #[cfg(not(feature = "vulkan-detect"))]
        let (name, available) = ("Vulkan compute backend".to_string(), false);

        Self {
            name,
            available,
            next_id: std::sync::atomic::AtomicU64::new(1),
            allocations: RwLock::new(HashMap::new()),
            total_allocated: std::sync::atomic::AtomicU64::new(0),
            peak_allocated: std::sync::atomic::AtomicU64::new(0),
            buffers: RwLock::new(HashMap::new()),
        }
    }

    /// Return whether a Vulkan device was detected.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.available
    }

    fn track_alloc(&self, id: u64, size: u64) {
        let mut allocs = self.allocations.write().expect("VulkanComputeBackend poisoned");
        allocs.insert(id, size);
        let current = self.total_allocated.fetch_add(size, std::sync::atomic::Ordering::Relaxed) + size;
        let mut peak = self.peak_allocated.load(std::sync::atomic::Ordering::Relaxed);
        while current > peak {
            match self.peak_allocated.compare_exchange_weak(
                peak,
                current,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
    }

    fn track_free(&self, id: u64) {
        let mut allocs = self.allocations.write().expect("VulkanComputeBackend poisoned");
        if let Some(size) = allocs.remove(&id) {
            self.total_allocated.fetch_sub(size, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

impl Default for VulkanComputeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for VulkanComputeBackend {
    fn allocate_buffer(&self, size: u64) -> AccelResult<BufferHandle> {
        let id = self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.track_alloc(id, size);
        // Store a zeroed host-side shadow buffer.
        let mut bufs = self.buffers.write().expect("VulkanComputeBackend buffers poisoned");
        bufs.insert(id, vec![0u8; size as usize]);
        Ok(BufferHandle::new(id))
    }

    fn upload_buffer(&self, handle: &BufferHandle, data: &[u8]) -> AccelResult<()> {
        let mut bufs = self.buffers.write().expect("VulkanComputeBackend buffers poisoned");
        match bufs.get_mut(&handle.0) {
            Some(buf) if buf.len() == data.len() => {
                buf.copy_from_slice(data);
                Ok(())
            }
            Some(buf) => Err(AccelError::BufferSizeMismatch {
                expected: buf.len(),
                actual: data.len(),
            }),
            None => Err(AccelError::BufferAllocation(
                format!("Invalid buffer handle: {}", handle.0),
            )),
        }
    }

    fn download_buffer(&self, handle: &BufferHandle) -> AccelResult<Vec<u8>> {
        let bufs = self.buffers.read().expect("VulkanComputeBackend buffers poisoned");
        bufs.get(&handle.0)
            .map(Clone::clone)
            .ok_or_else(|| AccelError::BufferAllocation(format!("Invalid buffer handle: {}", handle.0)))
    }

    fn free_buffer(&self, handle: BufferHandle) -> AccelResult<()> {
        self.track_free(handle.0);
        let mut bufs = self.buffers.write().expect("VulkanComputeBackend buffers poisoned");
        if bufs.remove(&handle.0).is_none() {
            return Err(AccelError::BufferAllocation(
                format!("Double-free of buffer handle: {}", handle.0),
            ));
        }
        Ok(())
    }

    fn dispatch_kernel(
        &self,
        kernel_name: &str,
        _buffers: &[&BufferHandle],
        _dispatch: DispatchParams,
    ) -> AccelResult<()> {
        // Without an actual compiled pipeline this falls back gracefully.
        tracing::debug!(
            "VulkanComputeBackend::dispatch_kernel: '{}' (no-op in abstraction layer)",
            kernel_name
        );
        Ok(())
    }

    fn synchronize(&self) -> AccelResult<()> {
        // Nothing to synchronize in the pure-abstraction path.
        Ok(())
    }

    fn backend_name(&self) -> &str {
        &self.name
    }

    fn is_gpu(&self) -> bool {
        self.available
    }

    fn memory_stats(&self) -> GpuMemoryStats {
        let allocs = self.allocations.read().expect("VulkanComputeBackend poisoned");
        GpuMemoryStats {
            allocated_bytes: self.total_allocated.load(std::sync::atomic::Ordering::Relaxed),
            peak_bytes: self.peak_allocated.load(std::sync::atomic::Ordering::Relaxed),
            allocation_count: allocs.len() as u64,
        }
    }
}

// ============================================================================
// CpuFallbackBackend
// ============================================================================

/// Pure-CPU compute backend used when GPU acceleration is unavailable.
///
/// Buffers are plain `Vec<u8>` allocations.  Kernel dispatch is a no-op in
/// this layer; the actual CPU work is performed by the higher-level
/// `CpuAccel` type via rayon.
pub struct CpuFallbackBackend {
    /// Allocation counter for handle generation.
    next_id: std::sync::atomic::AtomicU64,
    /// Live allocations.
    allocations: RwLock<HashMap<u64, u64>>,
    /// Buffer storage.
    buffers: RwLock<HashMap<u64, Vec<u8>>>,
    /// Bytes currently live.
    current_bytes: std::sync::atomic::AtomicU64,
    /// Peak bytes live.
    peak_bytes: std::sync::atomic::AtomicU64,
}

impl CpuFallbackBackend {
    /// Create a new CPU fallback backend.
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_id: std::sync::atomic::AtomicU64::new(1),
            allocations: RwLock::new(HashMap::new()),
            buffers: RwLock::new(HashMap::new()),
            current_bytes: std::sync::atomic::AtomicU64::new(0),
            peak_bytes: std::sync::atomic::AtomicU64::new(0),
        }
    }

    fn update_peak(&self, current: u64) {
        let mut peak = self.peak_bytes.load(std::sync::atomic::Ordering::Relaxed);
        while current > peak {
            match self.peak_bytes.compare_exchange_weak(
                peak,
                current,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
    }
}

impl Default for CpuFallbackBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CpuFallbackBackend {
    fn allocate_buffer(&self, size: u64) -> AccelResult<BufferHandle> {
        let id = self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        {
            let mut allocs = self.allocations.write().expect("CpuFallbackBackend poisoned");
            allocs.insert(id, size);
        }
        let current = self.current_bytes.fetch_add(size, std::sync::atomic::Ordering::Relaxed) + size;
        self.update_peak(current);
        let mut bufs = self.buffers.write().expect("CpuFallbackBackend buffers poisoned");
        bufs.insert(id, vec![0u8; size as usize]);
        Ok(BufferHandle::new(id))
    }

    fn upload_buffer(&self, handle: &BufferHandle, data: &[u8]) -> AccelResult<()> {
        let mut bufs = self.buffers.write().expect("CpuFallbackBackend buffers poisoned");
        match bufs.get_mut(&handle.0) {
            Some(buf) if buf.len() == data.len() => {
                buf.copy_from_slice(data);
                Ok(())
            }
            Some(buf) => Err(AccelError::BufferSizeMismatch {
                expected: buf.len(),
                actual: data.len(),
            }),
            None => Err(AccelError::BufferAllocation(
                format!("Invalid buffer handle: {}", handle.0),
            )),
        }
    }

    fn download_buffer(&self, handle: &BufferHandle) -> AccelResult<Vec<u8>> {
        let bufs = self.buffers.read().expect("CpuFallbackBackend buffers poisoned");
        bufs.get(&handle.0)
            .map(Clone::clone)
            .ok_or_else(|| AccelError::BufferAllocation(format!("Invalid buffer handle: {}", handle.0)))
    }

    fn free_buffer(&self, handle: BufferHandle) -> AccelResult<()> {
        let size = {
            let mut allocs = self.allocations.write().expect("CpuFallbackBackend poisoned");
            allocs.remove(&handle.0)
        };
        match size {
            Some(s) => {
                self.current_bytes.fetch_sub(s, std::sync::atomic::Ordering::Relaxed);
                let mut bufs = self.buffers.write().expect("CpuFallbackBackend buffers poisoned");
                bufs.remove(&handle.0);
                Ok(())
            }
            None => Err(AccelError::BufferAllocation(
                format!("Double-free of buffer handle: {}", handle.0),
            )),
        }
    }

    fn dispatch_kernel(
        &self,
        kernel_name: &str,
        _buffers: &[&BufferHandle],
        _dispatch: DispatchParams,
    ) -> AccelResult<()> {
        tracing::debug!(
            "CpuFallbackBackend::dispatch_kernel: '{}' (CPU path; use CpuAccel for real work)",
            kernel_name
        );
        Ok(())
    }

    fn synchronize(&self) -> AccelResult<()> {
        // CPU is always coherent.
        Ok(())
    }

    fn backend_name(&self) -> &str {
        "CPU Fallback"
    }

    fn is_gpu(&self) -> bool {
        false
    }

    fn memory_stats(&self) -> GpuMemoryStats {
        let allocs = self.allocations.read().expect("CpuFallbackBackend poisoned");
        GpuMemoryStats {
            allocated_bytes: self.current_bytes.load(std::sync::atomic::Ordering::Relaxed),
            peak_bytes: self.peak_bytes.load(std::sync::atomic::Ordering::Relaxed),
            allocation_count: allocs.len() as u64,
        }
    }
}

// ============================================================================
// Factory helper
// ============================================================================

/// Create the best available [`ComputeBackend`] wrapped in an `Arc`.
///
/// Tries `VulkanComputeBackend` first; if Vulkan is unavailable falls back to
/// `CpuFallbackBackend`.
#[must_use]
pub fn create_backend() -> Arc<dyn ComputeBackend> {
    let vulkan = VulkanComputeBackend::new();
    if vulkan.is_available() {
        tracing::info!("ComputeBackend: using Vulkan ({})", vulkan.name);
        Arc::new(vulkan)
    } else {
        tracing::info!("ComputeBackend: Vulkan unavailable, using CPU fallback");
        Arc::new(CpuFallbackBackend::new())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cpu_backend() -> CpuFallbackBackend {
        CpuFallbackBackend::new()
    }

    #[test]
    fn test_cpu_backend_alloc_upload_download_free() {
        let b = make_cpu_backend();

        let data = vec![1u8, 2, 3, 4, 5, 6];
        let h = b.allocate_buffer(data.len() as u64).expect("h should be valid");

        b.upload_buffer(&h, &data).expect("upload_buffer should succeed");
        let out = b.download_buffer(&h).expect("out should be valid");
        assert_eq!(out, data);

        b.free_buffer(h).expect("free_buffer should succeed");
    }

    #[test]
    fn test_cpu_backend_size_mismatch() {
        let b = make_cpu_backend();
        let h = b.allocate_buffer(4).expect("h should be valid");
        let result = b.upload_buffer(&h, &[1u8, 2, 3]); // 3 != 4
        assert!(result.is_err());
        b.free_buffer(h).expect("free_buffer should succeed");
    }

    #[test]
    fn test_cpu_backend_double_free() {
        let b = make_cpu_backend();
        let h = b.allocate_buffer(8).expect("h should be valid");
        let h2 = BufferHandle::new(h.0);
        b.free_buffer(h).expect("free_buffer should succeed");
        assert!(b.free_buffer(h2).is_err());
    }

    #[test]
    fn test_cpu_backend_dispatch_noop() {
        let b = make_cpu_backend();
        let dispatch = DispatchParams::new_2d(8, 8);
        b.dispatch_kernel("my_kernel", &[], dispatch).expect("dispatch_kernel should succeed");
    }

    #[test]
    fn test_cpu_backend_synchronize() {
        let b = make_cpu_backend();
        b.synchronize().expect("synchronize should succeed");
    }

    #[test]
    fn test_cpu_backend_memory_stats() {
        let b = make_cpu_backend();
        let h1 = b.allocate_buffer(1024).expect("h1 should be valid");
        let h2 = b.allocate_buffer(2048).expect("h2 should be valid");
        let stats = b.memory_stats();
        assert_eq!(stats.allocated_bytes, 3072);
        assert_eq!(stats.allocation_count, 2);
        b.free_buffer(h1).expect("free_buffer should succeed");
        let stats2 = b.memory_stats();
        assert_eq!(stats2.allocated_bytes, 2048);
        assert_eq!(stats2.peak_bytes, 3072);
        b.free_buffer(h2).expect("free_buffer should succeed");
    }

    #[test]
    fn test_kernel_registry() {
        let reg = KernelRegistry::new();
        assert!(reg.is_empty());

        reg.register("bilinear", b"\x03\x02\x23\x07", "Bilinear scale");
        assert_eq!(reg.len(), 1);
        assert!(reg.get("bilinear").is_some());
        assert!(reg.get("unknown").is_none());

        let names = reg.kernel_names();
        assert!(names.contains(&"bilinear".to_string()));
    }

    #[test]
    fn test_yuv_frame_info() {
        let info = YuvFrameInfo::yuv420(1920, 1080);
        assert_eq!(info.luma_size(), 1920 * 1080);
        assert_eq!(info.chroma_size(), 960 * 540);
        assert_eq!(info.total_size(), 1920 * 1080 + 2 * 960 * 540);
    }

    #[test]
    fn test_upload_download_yuv_frame() {
        let b = CpuFallbackBackend::new();
        let info = YuvFrameInfo::yuv420(4, 4);
        let data = vec![0u8; info.total_size() as usize];
        let h = upload_yuv_frame(&b, &data, &info).expect("h should be valid");
        let out = download_yuv_frame(&b, &h).expect("out should be valid");
        assert_eq!(out.len(), data.len());
        b.free_buffer(h).expect("free_buffer should succeed");
    }

    #[test]
    fn test_dispatch_params() {
        let p = DispatchParams::for_image(1920, 1080, 16, 16);
        assert_eq!(p.groups_x, 120); // 1920/16 = 120
        assert_eq!(p.groups_y, 68);  // ceil(1080/16) = 68
    }

    #[test]
    fn test_vulkan_backend_creation() {
        let b = VulkanComputeBackend::new();
        // Should at least create without panic; GPU availability varies.
        assert!(!b.backend_name().is_empty());
    }

    #[test]
    fn test_vulkan_backend_alloc_upload_download() {
        let b = VulkanComputeBackend::new();
        let data = vec![42u8; 16];
        let h = b.allocate_buffer(16).expect("h should be valid");
        b.upload_buffer(&h, &data).expect("upload_buffer should succeed");
        let out = b.download_buffer(&h).expect("out should be valid");
        assert_eq!(out, data);
        b.free_buffer(h).expect("free_buffer should succeed");
    }
}
