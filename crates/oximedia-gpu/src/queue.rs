//! Command queue management for GPU operations
//!
//! This module provides abstractions for managing command queues,
//! including command buffer submission, synchronization, and queue families.

use crate::GpuDevice;
use std::sync::Arc;
use wgpu::{CommandBuffer, CommandEncoder, Queue, SubmissionIndex};

/// Queue type for different workload categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueType {
    /// Compute queue for compute operations
    Compute,
    /// Transfer queue for data transfers
    Transfer,
    /// Graphics queue for graphics operations (rarely used in compute-only context)
    Graphics,
}

/// Command queue wrapper with additional functionality
pub struct CommandQueue {
    queue: Arc<Queue>,
    device: Arc<wgpu::Device>,
    queue_type: QueueType,
}

impl CommandQueue {
    /// Create a new command queue
    #[must_use]
    pub fn new(device: &GpuDevice, queue_type: QueueType) -> Self {
        Self {
            queue: Arc::clone(device.queue()),
            device: Arc::clone(device.device()),
            queue_type,
        }
    }

    /// Submit a single command buffer to the queue
    ///
    /// # Arguments
    ///
    /// * `command_buffer` - Command buffer to submit
    ///
    /// # Returns
    ///
    /// Submission index for synchronization
    #[must_use]
    pub fn submit_single(&self, command_buffer: CommandBuffer) -> SubmissionIndex {
        self.queue.submit(Some(command_buffer))
    }

    /// Submit multiple command buffers to the queue
    ///
    /// # Arguments
    ///
    /// * `command_buffers` - Command buffers to submit
    ///
    /// # Returns
    ///
    /// Submission index for synchronization
    #[must_use]
    pub fn submit_many(&self, command_buffers: Vec<CommandBuffer>) -> SubmissionIndex {
        self.queue.submit(command_buffers)
    }

    /// Submit commands created by an encoder
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to finish and submit
    ///
    /// # Returns
    ///
    /// Submission index for synchronization
    #[must_use]
    pub fn submit_encoder(&self, encoder: CommandEncoder) -> SubmissionIndex {
        self.queue.submit(Some(encoder.finish()))
    }

    /// Write data directly to a buffer
    ///
    /// This is a convenience method that bypasses the staging buffer
    /// and directly writes to the destination buffer.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Target buffer
    /// * `offset` - Offset in bytes
    /// * `data` - Data to write
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, offset: u64, data: &[u8]) {
        self.queue.write_buffer(buffer, offset, data);
    }

    /// Wait for all pending operations on this queue to complete
    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Get the queue type
    #[must_use]
    pub fn queue_type(&self) -> QueueType {
        self.queue_type
    }

    /// Get the underlying WGPU queue
    #[must_use]
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }
}

/// Queue manager for multi-queue support
pub struct QueueManager {
    compute_queue: CommandQueue,
    transfer_queue: CommandQueue,
    graphics_queue: CommandQueue,
}

impl QueueManager {
    /// Create a new queue manager
    ///
    /// Note: In wgpu, we typically have a single queue that handles all operations.
    /// This abstraction provides a logical separation for different workload types.
    #[must_use]
    pub fn new(device: &GpuDevice) -> Self {
        Self {
            compute_queue: CommandQueue::new(device, QueueType::Compute),
            transfer_queue: CommandQueue::new(device, QueueType::Transfer),
            graphics_queue: CommandQueue::new(device, QueueType::Graphics),
        }
    }

    /// Get the compute queue
    #[must_use]
    pub fn compute(&self) -> &CommandQueue {
        &self.compute_queue
    }

    /// Get the transfer queue
    #[must_use]
    pub fn transfer(&self) -> &CommandQueue {
        &self.transfer_queue
    }

    /// Get the graphics queue
    #[must_use]
    pub fn graphics(&self) -> &CommandQueue {
        &self.graphics_queue
    }

    /// Get a queue by type
    #[must_use]
    pub fn get_queue(&self, queue_type: QueueType) -> &CommandQueue {
        match queue_type {
            QueueType::Compute => &self.compute_queue,
            QueueType::Transfer => &self.transfer_queue,
            QueueType::Graphics => &self.graphics_queue,
        }
    }

    /// Wait for all queues to complete
    pub fn wait_all(&self) {
        self.compute_queue.wait();
        self.transfer_queue.wait();
        self.graphics_queue.wait();
    }
}

/// Command buffer builder with fluent API
pub struct CommandBufferBuilder {
    encoder: CommandEncoder,
    label: String,
}

impl CommandBufferBuilder {
    /// Create a new command buffer builder
    pub fn new(device: &GpuDevice, label: impl Into<String>) -> Self {
        let label_string = label.into();
        let encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&label_string),
            });

        Self {
            encoder,
            label: label_string,
        }
    }

    /// Get a mutable reference to the encoder
    pub fn encoder(&mut self) -> &mut CommandEncoder {
        &mut self.encoder
    }

    /// Finish building and return the command buffer
    #[must_use]
    pub fn finish(self) -> CommandBuffer {
        self.encoder.finish()
    }

    /// Finish building and submit to a queue
    #[must_use]
    pub fn submit(self, queue: &CommandQueue) -> SubmissionIndex {
        queue.submit_encoder(self.encoder)
    }

    /// Get the label
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// Async command submission handle
pub struct AsyncSubmission {
    submission_index: SubmissionIndex,
    device: Arc<wgpu::Device>,
}

impl AsyncSubmission {
    /// Create a new async submission handle
    #[must_use]
    pub fn new(submission_index: SubmissionIndex, device: Arc<wgpu::Device>) -> Self {
        Self {
            submission_index,
            device,
        }
    }

    /// Wait for this submission to complete
    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Get the submission index
    #[must_use]
    pub fn index(&self) -> &SubmissionIndex {
        &self.submission_index
    }
}

/// Batch command submission for improved performance
pub struct BatchSubmitter {
    command_buffers: Vec<CommandBuffer>,
    max_batch_size: usize,
}

impl BatchSubmitter {
    /// Create a new batch submitter
    ///
    /// # Arguments
    ///
    /// * `max_batch_size` - Maximum number of command buffers to batch
    #[must_use]
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            command_buffers: Vec::with_capacity(max_batch_size),
            max_batch_size,
        }
    }

    /// Add a command buffer to the batch
    ///
    /// If the batch is full, it will be automatically submitted.
    ///
    /// # Arguments
    ///
    /// * `command_buffer` - Command buffer to add
    /// * `queue` - Queue to submit to when batch is full
    ///
    /// # Returns
    ///
    /// Submission index if batch was submitted, None otherwise
    pub fn add(
        &mut self,
        command_buffer: CommandBuffer,
        queue: &CommandQueue,
    ) -> Option<SubmissionIndex> {
        self.command_buffers.push(command_buffer);

        if self.command_buffers.len() >= self.max_batch_size {
            Some(self.flush(queue))
        } else {
            None
        }
    }

    /// Flush all pending command buffers to the queue
    ///
    /// # Arguments
    ///
    /// * `queue` - Queue to submit to
    ///
    /// # Returns
    ///
    /// Submission index
    pub fn flush(&mut self, queue: &CommandQueue) -> SubmissionIndex {
        let buffers = std::mem::take(&mut self.command_buffers);
        queue.submit_many(buffers)
    }

    /// Get the number of pending command buffers
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.command_buffers.len()
    }

    /// Check if the batch is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.command_buffers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_type() {
        assert_eq!(QueueType::Compute, QueueType::Compute);
        assert_ne!(QueueType::Compute, QueueType::Transfer);
        assert_ne!(QueueType::Compute, QueueType::Graphics);
    }

    #[test]
    fn test_batch_submitter() {
        let submitter = BatchSubmitter::new(5);

        assert_eq!(submitter.pending_count(), 0);
        assert!(submitter.is_empty());
    }
}
