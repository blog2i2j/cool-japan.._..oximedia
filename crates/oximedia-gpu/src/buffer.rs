//! GPU buffer management for staging and device memory

use crate::{GpuDevice, GpuError, Result};
use std::sync::Arc;
use wgpu::{Buffer, BufferUsages};

/// Type of GPU buffer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferType {
    /// Staging buffer for CPU to GPU transfer
    Staging,
    /// Storage buffer for compute operations
    Storage,
    /// Uniform buffer for shader parameters
    Uniform,
    /// Read-back buffer for GPU to CPU transfer
    ReadBack,
}

/// GPU buffer wrapper with automatic memory management
pub struct GpuBuffer {
    buffer: Buffer,
    size: u64,
    buffer_type: BufferType,
    device: Arc<wgpu::Device>,
}

impl GpuBuffer {
    /// Create a new GPU buffer
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `size` - Buffer size in bytes
    /// * `buffer_type` - Type of buffer to create
    ///
    /// # Errors
    ///
    /// Returns an error if buffer creation fails.
    pub fn new(device: &GpuDevice, size: u64, buffer_type: BufferType) -> Result<Self> {
        let usage = Self::buffer_usage(buffer_type);

        let buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("OxiMedia {buffer_type:?} Buffer")),
            size,
            usage,
            mapped_at_creation: false,
        });

        Ok(Self {
            buffer,
            size,
            buffer_type,
            device: Arc::clone(device.device()),
        })
    }

    /// Create a buffer initialized with data
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `data` - Initial data
    /// * `buffer_type` - Type of buffer to create
    ///
    /// # Errors
    ///
    /// Returns an error if buffer creation fails.
    pub fn with_data(device: &GpuDevice, data: &[u8], buffer_type: BufferType) -> Result<Self> {
        let size = data.len() as u64;
        let usage = Self::buffer_usage(buffer_type);

        let buffer = device
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("OxiMedia {buffer_type:?} Buffer")),
                contents: data,
                usage,
            });

        Ok(Self {
            buffer,
            size,
            buffer_type,
            device: Arc::clone(device.device()),
        })
    }

    /// Write data to the buffer
    ///
    /// # Arguments
    ///
    /// * `queue` - GPU queue for data transfer
    /// * `offset` - Offset in bytes
    /// * `data` - Data to write
    ///
    /// # Errors
    ///
    /// Returns an error if the write operation fails or if offset + data size
    /// exceeds buffer size.
    pub fn write(&self, queue: &wgpu::Queue, offset: u64, data: &[u8]) -> Result<()> {
        if offset + data.len() as u64 > self.size {
            return Err(GpuError::InvalidBufferSize {
                expected: self.size as usize,
                actual: (offset + data.len() as u64) as usize,
            });
        }

        queue.write_buffer(&self.buffer, offset, data);
        Ok(())
    }

    /// Read data from the buffer (asynchronously)
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `offset` - Offset in bytes
    /// * `size` - Number of bytes to read
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer type doesn't support reading or if
    /// the read operation fails.
    pub async fn read_async(&self, _device: &GpuDevice, offset: u64, size: u64) -> Result<Vec<u8>> {
        if self.buffer_type != BufferType::ReadBack {
            return Err(GpuError::NotSupported(
                "Can only read from ReadBack buffers".to_string(),
            ));
        }

        if offset + size > self.size {
            return Err(GpuError::InvalidBufferSize {
                expected: self.size as usize,
                actual: (offset + size) as usize,
            });
        }

        let slice = self.buffer.slice(offset..offset + size);
        let (sender, receiver) = futures_channel::oneshot::channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        receiver
            .await
            .map_err(|e| GpuError::BufferMapping(e.to_string()))?
            .map_err(|e| GpuError::BufferMapping(e.to_string()))?;

        let data = slice.get_mapped_range();
        let result = data.to_vec();

        drop(data);
        self.buffer.unmap();

        Ok(result)
    }

    /// Read data from the buffer (blocking)
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `offset` - Offset in bytes
    /// * `size` - Number of bytes to read
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer type doesn't support reading or if
    /// the read operation fails.
    pub fn read(&self, device: &GpuDevice, offset: u64, size: u64) -> Result<Vec<u8>> {
        pollster::block_on(self.read_async(device, offset, size))
    }

    /// Get the underlying WGPU buffer
    #[must_use]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get buffer size in bytes
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get buffer type
    #[must_use]
    pub fn buffer_type(&self) -> BufferType {
        self.buffer_type
    }

    /// Copy data from this buffer to another buffer
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder for recording the copy operation
    /// * `dst` - Destination buffer
    /// * `src_offset` - Source offset in bytes
    /// * `dst_offset` - Destination offset in bytes
    /// * `size` - Number of bytes to copy
    ///
    /// # Errors
    ///
    /// Returns an error if offsets or size are invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_to(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        dst: &Self,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    ) -> Result<()> {
        if src_offset + size > self.size {
            return Err(GpuError::InvalidBufferSize {
                expected: self.size as usize,
                actual: (src_offset + size) as usize,
            });
        }

        if dst_offset + size > dst.size {
            return Err(GpuError::InvalidBufferSize {
                expected: dst.size as usize,
                actual: (dst_offset + size) as usize,
            });
        }

        encoder.copy_buffer_to_buffer(&self.buffer, src_offset, &dst.buffer, dst_offset, size);
        Ok(())
    }

    fn buffer_usage(buffer_type: BufferType) -> BufferUsages {
        match buffer_type {
            BufferType::Staging => BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
            BufferType::Storage => {
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC
            }
            BufferType::Uniform => BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            BufferType::ReadBack => BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        }
    }
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("size", &self.size)
            .field("buffer_type", &self.buffer_type)
            .finish()
    }
}

// Add the missing dependency
use wgpu::util::DeviceExt;
