//! Color conversion kernel implementations.

use crate::buffer::BufferManager;
use crate::cpu_fallback::CpuAccel;
use crate::error::{AccelError, AccelResult};
use crate::shaders::color::{rgb_to_yuv420p, yuv420p_to_rgb};
use crate::traits::HardwareAccel;
use oximedia_core::PixelFormat;
use std::sync::Arc;
use vulkano::buffer::BufferUsage;
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::pipeline::{
    compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::sync::GpuFuture;

/// Color conversion kernel.
pub struct ColorKernel {
    device: Arc<Device>,
    queue: Arc<Queue>,
    buffer_manager: BufferManager,
    descriptor_allocator: Arc<StandardDescriptorSetAllocator>,
    rgb_to_yuv_pipeline: Arc<ComputePipeline>,
    yuv_to_rgb_pipeline: Arc<ComputePipeline>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ColorPushConstants {
    width: u32,
    height: u32,
    channels: u32,
    _padding: u32,
}

impl ColorKernel {
    /// Creates a new color conversion kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if pipeline creation fails.
    /// # Panics
    ///
    /// Panics if the shader entry point "main" is not found.
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        buffer_manager: BufferManager,
    ) -> AccelResult<Self> {
        let descriptor_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        // Create RGB to YUV pipeline
        let rgb_to_yuv_shader = rgb_to_yuv420p::load(device.clone())
            .map_err(|e| AccelError::ShaderCompilation(format!("RGB to YUV shader: {e:?}")))?;

        let rgb_to_yuv_stage = PipelineShaderStageCreateInfo::new(
            rgb_to_yuv_shader.entry_point("main").ok_or_else(|| {
                AccelError::ShaderCompilation(
                    "RGB to YUV shader: entry point 'main' not found".to_string(),
                )
            })?,
        );

        let rgb_to_yuv_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(std::slice::from_ref(
                &rgb_to_yuv_stage,
            ))
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| AccelError::PipelineCreation(format!("RGB to YUV layout: {e:?}")))?,
        )
        .map_err(|e| AccelError::PipelineCreation(format!("RGB to YUV layout creation: {e:?}")))?;

        let rgb_to_yuv_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(rgb_to_yuv_stage, rgb_to_yuv_layout),
        )
        .map_err(|e| AccelError::PipelineCreation(format!("RGB to YUV pipeline: {e:?}")))?;

        // Create YUV to RGB pipeline
        let yuv_to_rgb_shader = yuv420p_to_rgb::load(device.clone())
            .map_err(|e| AccelError::ShaderCompilation(format!("YUV to RGB shader: {e:?}")))?;

        let yuv_to_rgb_stage = PipelineShaderStageCreateInfo::new(
            yuv_to_rgb_shader.entry_point("main").ok_or_else(|| {
                AccelError::ShaderCompilation(
                    "YUV to RGB shader: entry point 'main' not found".to_string(),
                )
            })?,
        );

        let yuv_to_rgb_layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(std::slice::from_ref(
                &yuv_to_rgb_stage,
            ))
            .into_pipeline_layout_create_info(device.clone())
            .map_err(|e| AccelError::PipelineCreation(format!("YUV to RGB layout: {e:?}")))?,
        )
        .map_err(|e| AccelError::PipelineCreation(format!("YUV to RGB layout creation: {e:?}")))?;

        let yuv_to_rgb_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(yuv_to_rgb_stage, yuv_to_rgb_layout),
        )
        .map_err(|e| AccelError::PipelineCreation(format!("YUV to RGB pipeline: {e:?}")))?;

        Ok(Self {
            device,
            queue,
            buffer_manager,
            descriptor_allocator,
            rgb_to_yuv_pipeline,
            yuv_to_rgb_pipeline,
        })
    }

    /// Converts color format.
    ///
    /// GPU-accelerated paths are used for Rgb24<->Yuv420p.
    /// All other supported conversions fall back to the CPU implementation.
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion fails or format is unsupported.
    #[allow(clippy::cast_possible_truncation)]
    pub fn convert(
        &self,
        input: &[u8],
        width: u32,
        height: u32,
        src_format: PixelFormat,
        dst_format: PixelFormat,
    ) -> AccelResult<Vec<u8>> {
        match (src_format, dst_format) {
            // GPU-accelerated paths
            (PixelFormat::Rgb24, PixelFormat::Yuv420p) => self.rgb_to_yuv420p(input, width, height),
            (PixelFormat::Yuv420p, PixelFormat::Rgb24) => self.yuv420p_to_rgb(input, width, height),

            // CPU fallback paths for formats without dedicated GPU shaders
            (src, dst) => CpuAccel::new().convert_color(input, width, height, src, dst),
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn rgb_to_yuv420p(&self, input: &[u8], width: u32, height: u32) -> AccelResult<Vec<u8>> {
        let expected_size = (width * height * 3) as usize;
        if input.len() != expected_size {
            return Err(AccelError::BufferSizeMismatch {
                expected: expected_size,
                actual: input.len(),
            });
        }

        // Create input buffer
        let input_buffer = self
            .buffer_manager
            .create_device_buffer(input.len() as u64, BufferUsage::STORAGE_BUFFER)?;
        self.buffer_manager.upload_data(input, &input_buffer)?;

        // Create output buffers (Y, U, V planes)
        let y_size = (width * height) as usize;
        let uv_size = (width * height / 4) as usize;

        let y_buffer = self
            .buffer_manager
            .create_device_buffer(y_size as u64, BufferUsage::STORAGE_BUFFER)?;
        let u_buffer = self
            .buffer_manager
            .create_device_buffer(uv_size as u64, BufferUsage::STORAGE_BUFFER)?;
        let v_buffer = self
            .buffer_manager
            .create_device_buffer(uv_size as u64, BufferUsage::STORAGE_BUFFER)?;

        // Create descriptor set
        let layout = self
            .rgb_to_yuv_pipeline
            .layout()
            .set_layouts()
            .first()
            .ok_or_else(|| AccelError::PipelineCreation("No descriptor set layout".to_string()))?;

        let descriptor_set = DescriptorSet::new(
            self.descriptor_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, input_buffer),
                WriteDescriptorSet::buffer(1, y_buffer.clone()),
                WriteDescriptorSet::buffer(2, u_buffer.clone()),
                WriteDescriptorSet::buffer(3, v_buffer.clone()),
            ],
            [],
        )
        .map_err(|e| AccelError::PipelineCreation(format!("Descriptor set: {e:?}")))?;

        // Create command buffer
        let mut builder = self.buffer_manager.create_command_buffer()?;

        let push_constants = ColorPushConstants {
            width,
            height,
            channels: 3,
            _padding: 0,
        };

        builder
            .bind_pipeline_compute(self.rgb_to_yuv_pipeline.clone())
            .map_err(|e| AccelError::CommandBuffer(format!("Bind pipeline: {e:?}")))?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.rgb_to_yuv_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .map_err(|e| AccelError::CommandBuffer(format!("Bind descriptor sets: {e:?}")))?
            .push_constants(self.rgb_to_yuv_pipeline.layout().clone(), 0, push_constants)
            .map_err(|e| AccelError::CommandBuffer(format!("Push constants: {e:?}")))?;
        unsafe {
            builder
                .dispatch([width.div_ceil(16), height.div_ceil(16), 1])
                .map_err(|e| AccelError::Dispatch(format!("Dispatch: {e:?}")))?;
        }

        let command_buffer = builder
            .build()
            .map_err(|e| AccelError::CommandBuffer(format!("Build: {e:?}")))?;

        // Execute and wait
        vulkano::sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .map_err(|e| AccelError::Dispatch(format!("Execute: {e:?}")))?
            .then_signal_fence_and_flush()
            .map_err(|e| AccelError::Dispatch(format!("Flush: {e:?}")))?
            .wait(None)
            .map_err(|e| AccelError::Synchronization(format!("Wait: {e:?}")))?;

        // Download results
        let y_data = self.buffer_manager.download_data(&y_buffer)?;
        let u_data = self.buffer_manager.download_data(&u_buffer)?;
        let v_data = self.buffer_manager.download_data(&v_buffer)?;

        // Combine planes
        let mut result = Vec::with_capacity(y_size + uv_size * 2);
        result.extend_from_slice(&y_data);
        result.extend_from_slice(&u_data);
        result.extend_from_slice(&v_data);

        Ok(result)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn yuv420p_to_rgb(&self, input: &[u8], width: u32, height: u32) -> AccelResult<Vec<u8>> {
        let y_size = (width * height) as usize;
        let uv_size = (width * height / 4) as usize;
        let expected_size = y_size + uv_size * 2;

        if input.len() != expected_size {
            return Err(AccelError::BufferSizeMismatch {
                expected: expected_size,
                actual: input.len(),
            });
        }

        // Split input into Y, U, V planes
        let y_plane = &input[..y_size];
        let u_plane = &input[y_size..y_size + uv_size];
        let v_plane = &input[y_size + uv_size..];

        // Create input buffers
        let y_buffer = self
            .buffer_manager
            .create_device_buffer(y_size as u64, BufferUsage::STORAGE_BUFFER)?;
        let u_buffer = self
            .buffer_manager
            .create_device_buffer(uv_size as u64, BufferUsage::STORAGE_BUFFER)?;
        let v_buffer = self
            .buffer_manager
            .create_device_buffer(uv_size as u64, BufferUsage::STORAGE_BUFFER)?;

        self.buffer_manager.upload_data(y_plane, &y_buffer)?;
        self.buffer_manager.upload_data(u_plane, &u_buffer)?;
        self.buffer_manager.upload_data(v_plane, &v_buffer)?;

        // Create output buffer
        let output_size = (width * height * 3) as usize;
        let output_buffer = self
            .buffer_manager
            .create_device_buffer(output_size as u64, BufferUsage::STORAGE_BUFFER)?;

        // Create descriptor set
        let layout = self
            .yuv_to_rgb_pipeline
            .layout()
            .set_layouts()
            .first()
            .ok_or_else(|| AccelError::PipelineCreation("No descriptor set layout".to_string()))?;

        let descriptor_set = DescriptorSet::new(
            self.descriptor_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, y_buffer),
                WriteDescriptorSet::buffer(1, u_buffer),
                WriteDescriptorSet::buffer(2, v_buffer),
                WriteDescriptorSet::buffer(3, output_buffer.clone()),
            ],
            [],
        )
        .map_err(|e| AccelError::PipelineCreation(format!("Descriptor set: {e:?}")))?;

        // Create command buffer
        let mut builder = self.buffer_manager.create_command_buffer()?;

        let push_constants = ColorPushConstants {
            width,
            height,
            channels: 3,
            _padding: 0,
        };

        builder
            .bind_pipeline_compute(self.yuv_to_rgb_pipeline.clone())
            .map_err(|e| AccelError::CommandBuffer(format!("Bind pipeline: {e:?}")))?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.yuv_to_rgb_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .map_err(|e| AccelError::CommandBuffer(format!("Bind descriptor sets: {e:?}")))?
            .push_constants(self.yuv_to_rgb_pipeline.layout().clone(), 0, push_constants)
            .map_err(|e| AccelError::CommandBuffer(format!("Push constants: {e:?}")))?;
        unsafe {
            builder
                .dispatch([width.div_ceil(16), height.div_ceil(16), 1])
                .map_err(|e| AccelError::Dispatch(format!("Dispatch: {e:?}")))?;
        }

        let command_buffer = builder
            .build()
            .map_err(|e| AccelError::CommandBuffer(format!("Build: {e:?}")))?;

        // Execute and wait
        vulkano::sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .map_err(|e| AccelError::Dispatch(format!("Execute: {e:?}")))?
            .then_signal_fence_and_flush()
            .map_err(|e| AccelError::Dispatch(format!("Flush: {e:?}")))?
            .wait(None)
            .map_err(|e| AccelError::Synchronization(format!("Wait: {e:?}")))?;

        // Download result
        self.buffer_manager.download_data(&output_buffer)
    }
}
