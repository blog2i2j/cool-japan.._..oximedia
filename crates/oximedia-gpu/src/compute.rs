//! Compute pipeline management for GPU operations
//!
//! This module provides high-level abstractions for managing compute pipelines,
//! including pipeline creation, caching, and execution.

use crate::{GpuDevice, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{
    BindGroupLayout, CommandEncoder, ComputePass, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, PipelineLayoutDescriptor, ShaderModule,
};

/// Compute pipeline wrapper with metadata
pub struct ComputePipelineHandle {
    pipeline: ComputePipeline,
    workgroup_size: (u32, u32, u32),
    label: String,
}

impl ComputePipelineHandle {
    /// Create a new compute pipeline handle
    #[must_use]
    pub fn new(pipeline: ComputePipeline, workgroup_size: (u32, u32, u32), label: String) -> Self {
        Self {
            pipeline,
            workgroup_size,
            label,
        }
    }

    /// Get the underlying pipeline
    #[must_use]
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.pipeline
    }

    /// Get the workgroup size
    #[must_use]
    pub fn workgroup_size(&self) -> (u32, u32, u32) {
        self.workgroup_size
    }

    /// Get the pipeline label
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// Compute pipeline manager with caching
pub struct ComputePipelineManager {
    device: Arc<wgpu::Device>,
    pipelines: RwLock<HashMap<String, Arc<ComputePipelineHandle>>>,
}

impl ComputePipelineManager {
    /// Create a new compute pipeline manager
    #[must_use]
    pub fn new(device: &GpuDevice) -> Self {
        Self {
            device: Arc::clone(device.device()),
            pipelines: RwLock::new(HashMap::new()),
        }
    }

    /// Get or create a compute pipeline
    ///
    /// # Arguments
    ///
    /// * `key` - Unique key for caching the pipeline
    /// * `label` - Human-readable label for debugging
    /// * `shader` - Compiled shader module
    /// * `entry_point` - Entry point function name
    /// * `bind_group_layout` - Bind group layout for resources
    /// * `workgroup_size` - Workgroup size (x, y, z)
    ///
    /// # Errors
    ///
    /// Returns an error if pipeline creation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn get_or_create(
        &self,
        key: &str,
        label: &str,
        shader: &ShaderModule,
        entry_point: &str,
        bind_group_layout: &BindGroupLayout,
        workgroup_size: (u32, u32, u32),
    ) -> Result<Arc<ComputePipelineHandle>> {
        // Check cache first
        {
            let cache = self.pipelines.read();
            if let Some(pipeline) = cache.get(key) {
                return Ok(Arc::clone(pipeline));
            }
        }

        // Create pipeline
        let pipeline = self.create_pipeline(label, shader, entry_point, bind_group_layout)?;
        let handle = Arc::new(ComputePipelineHandle::new(
            pipeline,
            workgroup_size,
            label.to_string(),
        ));

        // Cache it
        {
            let mut cache = self.pipelines.write();
            cache.insert(key.to_string(), Arc::clone(&handle));
        }

        Ok(handle)
    }

    /// Create a new compute pipeline
    fn create_pipeline(
        &self,
        label: &str,
        shader: &ShaderModule,
        entry_point: &str,
        bind_group_layout: &BindGroupLayout,
    ) -> Result<ComputePipeline> {
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some(&format!("{label} Layout")),
                bind_group_layouts: &[Some(bind_group_layout)],
                immediate_size: 0,
            });

        Ok(self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: shader,
                entry_point: Some(entry_point),
                cache: None,
                compilation_options: Default::default(),
            }))
    }

    /// Clear the pipeline cache
    pub fn clear_cache(&self) {
        let mut cache = self.pipelines.write();
        cache.clear();
    }

    /// Get number of cached pipelines
    pub fn cache_size(&self) -> usize {
        let cache = self.pipelines.read();
        cache.len()
    }
}

/// Compute pass builder for easier command encoding
pub struct ComputePassBuilder<'a> {
    encoder: &'a mut CommandEncoder,
    label: Option<String>,
}

impl<'a> ComputePassBuilder<'a> {
    /// Create a new compute pass builder
    pub fn new(encoder: &'a mut CommandEncoder) -> Self {
        Self {
            encoder,
            label: None,
        }
    }

    /// Set the compute pass label
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Begin the compute pass and execute commands
    ///
    /// # Arguments
    ///
    /// * `f` - Function that configures the compute pass
    pub fn execute<F>(self, f: F)
    where
        F: FnOnce(&mut ComputePass<'_>),
    {
        let mut pass = self.encoder.begin_compute_pass(&ComputePassDescriptor {
            label: self.label.as_deref(),
            timestamp_writes: None,
        });

        f(&mut pass);
    }
}

/// Helper for dispatching compute workgroups
pub struct DispatchHelper;

impl DispatchHelper {
    /// Calculate dispatch dimensions for 1D workload
    ///
    /// # Arguments
    ///
    /// * `count` - Total number of elements
    /// * `workgroup_size` - Workgroup size
    ///
    /// # Returns
    ///
    /// Number of workgroups to dispatch
    #[must_use]
    pub fn dispatch_1d(count: u32, workgroup_size: u32) -> u32 {
        count.div_ceil(workgroup_size)
    }

    /// Calculate dispatch dimensions for 2D workload
    ///
    /// # Arguments
    ///
    /// * `width` - Width of the workload
    /// * `height` - Height of the workload
    /// * `workgroup_size` - Workgroup size (x, y)
    ///
    /// # Returns
    ///
    /// Number of workgroups to dispatch (x, y)
    #[must_use]
    pub fn dispatch_2d(width: u32, height: u32, workgroup_size: (u32, u32)) -> (u32, u32) {
        let x = width.div_ceil(workgroup_size.0);
        let y = height.div_ceil(workgroup_size.1);
        (x, y)
    }

    /// Calculate dispatch dimensions for 3D workload
    ///
    /// # Arguments
    ///
    /// * `width` - Width of the workload
    /// * `height` - Height of the workload
    /// * `depth` - Depth of the workload
    /// * `workgroup_size` - Workgroup size (x, y, z)
    ///
    /// # Returns
    ///
    /// Number of workgroups to dispatch (x, y, z)
    #[must_use]
    pub fn dispatch_3d(
        width: u32,
        height: u32,
        depth: u32,
        workgroup_size: (u32, u32, u32),
    ) -> (u32, u32, u32) {
        let x = width.div_ceil(workgroup_size.0);
        let y = height.div_ceil(workgroup_size.1);
        let z = depth.div_ceil(workgroup_size.2);
        (x, y, z)
    }
}

/// Compute operation executor
pub struct ComputeExecutor<'a> {
    device: &'a GpuDevice,
    encoder: CommandEncoder,
}

impl<'a> ComputeExecutor<'a> {
    /// Create a new compute executor
    #[must_use]
    pub fn new(device: &'a GpuDevice, label: &str) -> Self {
        let encoder = device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });

        Self { device, encoder }
    }

    /// Begin a compute pass
    pub fn begin_pass(&mut self, label: &str) -> ComputePassBuilder<'_> {
        ComputePassBuilder::new(&mut self.encoder).with_label(label)
    }

    // Note: Simple dispatch helper removed due to lifetime complexity.
    // Use begin_pass() directly for compute dispatches.
    // Example:
    // executor.begin_pass("My Compute Pass").execute(|pass| {
    //     pass.set_pipeline(&pipeline);
    //     pass.set_bind_group(0, &bind_group, &[]);
    //     pass.dispatch_workgroups(x, y, z);
    // });

    /// Finish encoding and submit commands
    pub fn submit(self) {
        let command_buffer = self.encoder.finish();
        self.device.queue().submit(Some(command_buffer));
    }

    /// Get a mutable reference to the encoder for advanced operations
    pub fn encoder_mut(&mut self) -> &mut CommandEncoder {
        &mut self.encoder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_1d() {
        assert_eq!(DispatchHelper::dispatch_1d(100, 64), 2);
        assert_eq!(DispatchHelper::dispatch_1d(64, 64), 1);
        assert_eq!(DispatchHelper::dispatch_1d(65, 64), 2);
        assert_eq!(DispatchHelper::dispatch_1d(0, 64), 0);
    }

    #[test]
    fn test_dispatch_2d() {
        assert_eq!(DispatchHelper::dispatch_2d(100, 100, (16, 16)), (7, 7));
        assert_eq!(DispatchHelper::dispatch_2d(16, 16, (16, 16)), (1, 1));
        assert_eq!(DispatchHelper::dispatch_2d(17, 17, (16, 16)), (2, 2));
    }

    #[test]
    fn test_dispatch_3d() {
        assert_eq!(
            DispatchHelper::dispatch_3d(100, 100, 100, (8, 8, 8)),
            (13, 13, 13)
        );
        assert_eq!(DispatchHelper::dispatch_3d(8, 8, 8, (8, 8, 8)), (1, 1, 1));
        assert_eq!(DispatchHelper::dispatch_3d(9, 9, 9, (8, 8, 8)), (2, 2, 2));
    }
}
