//! GPU buffer management for `oximedia-gpu`.
//!
//! Provides buffer usage flags, a logical `GpuBuffer` abstraction, and a
//! pooled allocator (`GpuBufferPool`) that recycles buffers to reduce
//! allocation overhead.

#![allow(dead_code)]

/// Describes how a GPU buffer will be used by the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferUsage {
    /// Read-only storage, typically for shader inputs.
    StorageRead,
    /// Read/write storage, for shader outputs or in-place ops.
    StorageReadWrite,
    /// Uniform/constant data uploaded from the CPU.
    Uniform,
    /// Staging buffer for CPU→GPU uploads.
    Upload,
    /// Staging buffer for GPU→CPU readback.
    Readback,
    /// Index buffer for draw calls.
    Index,
    /// Vertex buffer for draw calls.
    Vertex,
}

impl BufferUsage {
    /// Returns `true` when the pipeline may write to this buffer.
    #[must_use]
    pub fn is_writable(&self) -> bool {
        matches!(self, Self::StorageReadWrite | Self::Upload | Self::Readback)
    }

    /// Returns `true` when the buffer is used for transferring data between
    /// CPU and GPU (upload or readback).
    #[must_use]
    pub fn is_staging(&self) -> bool {
        matches!(self, Self::Upload | Self::Readback)
    }

    /// Returns a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::StorageRead => "storage_read",
            Self::StorageReadWrite => "storage_read_write",
            Self::Uniform => "uniform",
            Self::Upload => "upload",
            Self::Readback => "readback",
            Self::Index => "index",
            Self::Vertex => "vertex",
        }
    }
}

/// A logical GPU buffer (CPU-side descriptor; no actual WGPU resource here).
#[derive(Debug, Clone)]
pub struct GpuBuffer {
    /// Unique identifier assigned by the pool.
    pub id: u64,
    /// Intended usage.
    pub usage: BufferUsage,
    /// Allocated size in bytes.
    size: usize,
    /// Whether this buffer is currently mapped for CPU access.
    mapped: bool,
    /// Debug label shown in GPU profiling tools.
    pub label: Option<String>,
}

impl GpuBuffer {
    /// Creates a new GPU buffer descriptor.
    #[must_use]
    pub fn new(id: u64, usage: BufferUsage, size: usize) -> Self {
        Self {
            id,
            usage,
            size,
            mapped: false,
            label: None,
        }
    }

    /// Creates a buffer with a debug label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Returns the buffer size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.size
    }

    /// Returns `true` if the buffer is currently mapped for CPU access.
    #[must_use]
    pub fn is_mapped(&self) -> bool {
        self.mapped
    }

    /// Simulates mapping the buffer for CPU access.
    ///
    /// Returns an error string if the buffer usage does not allow mapping.
    pub fn map(&mut self) -> Result<(), String> {
        if !self.usage.is_staging() {
            return Err(format!(
                "Buffer (usage={}) cannot be mapped; only Upload/Readback buffers support mapping.",
                self.usage.label()
            ));
        }
        self.mapped = true;
        Ok(())
    }

    /// Unmaps the buffer (no-op if not mapped).
    pub fn unmap(&mut self) {
        self.mapped = false;
    }

    /// Returns the size in 4-byte aligned units (useful for uniform offsets).
    #[must_use]
    pub fn aligned_size(&self) -> usize {
        (self.size + 3) & !3
    }
}

/// A simple pool that allocates and recycles [`GpuBuffer`] instances.
#[derive(Debug, Default)]
pub struct GpuBufferPool {
    next_id: u64,
    active: Vec<GpuBuffer>,
    free_list: Vec<GpuBuffer>,
}

impl GpuBufferPool {
    /// Creates a new, empty buffer pool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocates a buffer of the given usage and size.
    ///
    /// If a compatible free buffer exists it is reused; otherwise a new one
    /// is created.
    pub fn allocate(&mut self, usage: BufferUsage, size: usize) -> GpuBuffer {
        // Try to recycle a free buffer with the same usage and sufficient size.
        if let Some(pos) = self
            .free_list
            .iter()
            .position(|b| b.usage == usage && b.size_bytes() >= size)
        {
            return self.free_list.remove(pos);
        }
        // Allocate a new buffer.
        let id = self.next_id;
        self.next_id += 1;
        let buf = GpuBuffer::new(id, usage, size);
        self.active.push(buf.clone());
        buf
    }

    /// Returns a buffer to the pool for future reuse.
    pub fn release(&mut self, mut buf: GpuBuffer) {
        buf.unmap(); // ensure it is not left mapped
                     // Remove from active list (best-effort; id may not be present if already released)
        self.active.retain(|b| b.id != buf.id);
        self.free_list.push(buf);
    }

    /// Returns the total number of bytes currently allocated across all active
    /// and free buffers.
    #[must_use]
    pub fn total_allocated(&self) -> usize {
        let active: usize = self.active.iter().map(GpuBuffer::size_bytes).sum();
        let free: usize = self.free_list.iter().map(GpuBuffer::size_bytes).sum();
        active + free
    }

    /// Returns the number of active (in-use) buffers.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Returns the number of buffers waiting in the free list.
    #[must_use]
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_usage_is_writable_storage_rw() {
        assert!(BufferUsage::StorageReadWrite.is_writable());
    }

    #[test]
    fn test_buffer_usage_not_writable_storage_read() {
        assert!(!BufferUsage::StorageRead.is_writable());
    }

    #[test]
    fn test_buffer_usage_is_staging_upload() {
        assert!(BufferUsage::Upload.is_staging());
    }

    #[test]
    fn test_buffer_usage_not_staging_uniform() {
        assert!(!BufferUsage::Uniform.is_staging());
    }

    #[test]
    fn test_buffer_usage_label_non_empty() {
        for usage in [
            BufferUsage::StorageRead,
            BufferUsage::StorageReadWrite,
            BufferUsage::Uniform,
            BufferUsage::Upload,
            BufferUsage::Readback,
            BufferUsage::Index,
            BufferUsage::Vertex,
        ] {
            assert!(!usage.label().is_empty());
        }
    }

    #[test]
    fn test_gpu_buffer_size_bytes() {
        let b = GpuBuffer::new(0, BufferUsage::Uniform, 256);
        assert_eq!(b.size_bytes(), 256);
    }

    #[test]
    fn test_gpu_buffer_not_mapped_by_default() {
        let b = GpuBuffer::new(0, BufferUsage::Upload, 1024);
        assert!(!b.is_mapped());
    }

    #[test]
    fn test_gpu_buffer_map_upload_ok() {
        let mut b = GpuBuffer::new(0, BufferUsage::Upload, 1024);
        assert!(b.map().is_ok());
        assert!(b.is_mapped());
    }

    #[test]
    fn test_gpu_buffer_map_non_staging_err() {
        let mut b = GpuBuffer::new(0, BufferUsage::StorageRead, 512);
        assert!(b.map().is_err());
    }

    #[test]
    fn test_gpu_buffer_unmap_clears_flag() {
        let mut b = GpuBuffer::new(0, BufferUsage::Readback, 512);
        b.map().expect("buffer map should succeed");
        b.unmap();
        assert!(!b.is_mapped());
    }

    #[test]
    fn test_gpu_buffer_aligned_size() {
        let b = GpuBuffer::new(0, BufferUsage::Uniform, 13);
        assert_eq!(b.aligned_size(), 16);
    }

    #[test]
    fn test_pool_allocate_creates_buffer() {
        let mut pool = GpuBufferPool::new();
        let buf = pool.allocate(BufferUsage::StorageRead, 1024);
        assert_eq!(buf.size_bytes(), 1024);
        assert_eq!(buf.usage, BufferUsage::StorageRead);
    }

    #[test]
    fn test_pool_release_and_reuse() {
        let mut pool = GpuBufferPool::new();
        let buf = pool.allocate(BufferUsage::Upload, 512);
        let id = buf.id;
        pool.release(buf);
        let reused = pool.allocate(BufferUsage::Upload, 512);
        assert_eq!(reused.id, id); // same buffer recycled
    }

    #[test]
    fn test_pool_total_allocated() {
        let mut pool = GpuBufferPool::new();
        pool.allocate(BufferUsage::Uniform, 256);
        pool.allocate(BufferUsage::Uniform, 512);
        assert_eq!(pool.total_allocated(), 768);
    }

    #[test]
    fn test_pool_free_count_after_release() {
        let mut pool = GpuBufferPool::new();
        let buf = pool.allocate(BufferUsage::Readback, 1024);
        pool.release(buf);
        assert_eq!(pool.free_count(), 1);
    }
}
