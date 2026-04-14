//! Shader compilation and pipeline management

use crate::{GpuDevice, Result};
use std::borrow::Cow;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, ComputePipeline,
    ComputePipelineDescriptor, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderStages,
};

/// Shader source type
pub enum ShaderSource<'a> {
    /// WGSL source code
    Wgsl(Cow<'a, str>),
    /// Embedded shader (included at compile time)
    Embedded(&'a str),
}

/// Shader compiler and pipeline builder
pub struct ShaderCompiler {
    device: std::sync::Arc<wgpu::Device>,
}

impl ShaderCompiler {
    /// Create a new shader compiler
    #[must_use]
    pub fn new(device: &GpuDevice) -> Self {
        Self {
            device: std::sync::Arc::clone(device.device()),
        }
    }

    /// Compile a shader module from source
    ///
    /// # Arguments
    ///
    /// * `label` - Shader label for debugging
    /// * `source` - Shader source code
    ///
    /// # Errors
    ///
    /// Returns an error if shader compilation fails.
    pub fn compile(&self, label: &str, source: ShaderSource<'_>) -> Result<ShaderModule> {
        let source_str = match source {
            ShaderSource::Wgsl(code) => code,
            ShaderSource::Embedded(code) => Cow::Borrowed(code),
        };

        Ok(self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source_str),
        }))
    }

    /// Create a compute pipeline
    ///
    /// # Arguments
    ///
    /// * `label` - Pipeline label for debugging
    /// * `shader` - Compiled shader module
    /// * `entry_point` - Entry point function name
    /// * `bind_group_layout` - Bind group layout for resources
    ///
    /// # Errors
    ///
    /// Returns an error if pipeline creation fails.
    pub fn create_pipeline(
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

    /// Create a bind group layout for compute operations
    ///
    /// # Arguments
    ///
    /// * `label` - Layout label for debugging
    /// * `entries` - Bind group layout entries
    #[must_use]
    pub fn create_bind_group_layout(
        &self,
        label: &str,
        entries: &[BindGroupLayoutEntry],
    ) -> BindGroupLayout {
        self.device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            })
    }

    /// Create a bind group
    ///
    /// # Arguments
    ///
    /// * `label` - Bind group label for debugging
    /// * `layout` - Bind group layout
    /// * `entries` - Bind group entries
    #[must_use]
    pub fn create_bind_group(
        &self,
        label: &str,
        layout: &BindGroupLayout,
        entries: &[BindGroupEntry<'_>],
    ) -> BindGroup {
        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some(label),
            layout,
            entries,
        })
    }
}

/// Helper for creating standard bind group layouts
pub struct BindGroupLayoutBuilder {
    entries: Vec<BindGroupLayoutEntry>,
}

impl BindGroupLayoutBuilder {
    /// Create a new bind group layout builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a storage buffer binding (read-only)
    #[must_use]
    pub fn add_storage_buffer_read_only(mut self, binding: u32) -> Self {
        self.entries.push(BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        self
    }

    /// Add a storage buffer binding (read-write)
    #[must_use]
    pub fn add_storage_buffer(mut self, binding: u32) -> Self {
        self.entries.push(BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        self
    }

    /// Add a uniform buffer binding
    #[must_use]
    pub fn add_uniform_buffer(mut self, binding: u32) -> Self {
        self.entries.push(BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        self
    }

    /// Build the bind group layout entries
    #[must_use]
    pub fn build(self) -> Vec<BindGroupLayoutEntry> {
        self.entries
    }
}

impl Default for BindGroupLayoutBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Precompiled shaders embedded at compile time
pub mod embedded {
    /// Color space conversion shader source
    pub const COLORSPACE_SHADER: &str = include_str!("shaders/colorspace.wgsl");

    /// Image scaling shader source
    pub const SCALE_SHADER: &str = include_str!("shaders/scale.wgsl");

    /// Convolution filter shader source
    pub const FILTER_SHADER: &str = include_str!("shaders/filter.wgsl");

    /// Transform operations shader source
    pub const TRANSFORM_SHADER: &str = include_str!("shaders/transform.wgsl");

    /// Bilateral filter denoising shader source
    pub const BILATERAL_SHADER: &str = include_str!("shaders/bilateral.wgsl");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bind_group_layout_builder() {
        let layout = BindGroupLayoutBuilder::new()
            .add_storage_buffer_read_only(0)
            .add_storage_buffer(1)
            .add_uniform_buffer(2)
            .build();

        assert_eq!(layout.len(), 3);
    }
}
