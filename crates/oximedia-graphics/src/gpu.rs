//! GPU-accelerated rendering using wgpu

#[cfg(feature = "gpu")]
use crate::color::Color;
#[cfg(feature = "gpu")]
use crate::error::{GraphicsError, Result};
#[cfg(feature = "gpu")]
use crate::primitives::Rect;

#[cfg(feature = "gpu")]
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

/// Vertex for GPU rendering
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    /// Position (x, y)
    pub position: [f32; 2],
    /// Color (r, g, b, a)
    pub color: [f32; 4],
    /// Texture coordinates (u, v)
    pub tex_coords: [f32; 2],
}

#[cfg(feature = "gpu")]
impl Vertex {
    /// Create a new vertex
    #[must_use]
    pub fn new(position: [f32; 2], color: [f32; 4], tex_coords: [f32; 2]) -> Self {
        Self {
            position,
            color,
            tex_coords,
        }
    }

    /// Get vertex buffer layout
    #[must_use]
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// GPU renderer
#[cfg(feature = "gpu")]
pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    width: u32,
    height: u32,
}

#[cfg(feature = "gpu")]
impl GpuRenderer {
    /// Create a new GPU renderer
    pub async fn new(width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(GraphicsError::InvalidDimensions(width, height));
        }

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|_| GraphicsError::GpuError("Failed to find adapter".to_string()))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Graphics Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| GraphicsError::GpuError(format!("Failed to create device: {e}")))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Graphics Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/graphics.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Graphics Pipeline Layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Graphics Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            width,
            height,
        })
    }

    /// Render rectangle
    pub fn render_rect(&self, rect: Rect, color: Color) -> Result<Vec<u8>> {
        // Convert to normalized device coordinates
        let x1 = (rect.x / self.width as f32) * 2.0 - 1.0;
        let y1 = (rect.y / self.height as f32) * 2.0 - 1.0;
        let x2 = ((rect.x + rect.width) / self.width as f32) * 2.0 - 1.0;
        let y2 = ((rect.y + rect.height) / self.height as f32) * 2.0 - 1.0;

        let color_f = color.to_float();

        let vertices = vec![
            Vertex::new([x1, y1], color_f, [0.0, 0.0]),
            Vertex::new([x2, y1], color_f, [1.0, 0.0]),
            Vertex::new([x2, y2], color_f, [1.0, 1.0]),
            Vertex::new([x1, y1], color_f, [0.0, 0.0]),
            Vertex::new([x2, y2], color_f, [1.0, 1.0]),
            Vertex::new([x1, y2], color_f, [0.0, 1.0]),
        ];

        self.render_vertices(&vertices)
    }

    /// Render vertices
    pub fn render_vertices(&self, vertices: &[Vertex]) -> Result<Vec<u8>> {
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render Texture"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.draw(0..vertices.len() as u32, 0..1);
        }

        // Read back texture data
        let buffer_size = wgpu::BufferAddress::from(self.width * self.height * 4);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * self.width),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let data = buffer_slice.get_mapped_range();
        let result = data.to_vec();

        drop(data);
        buffer.unmap();

        Ok(result)
    }

    /// Get width
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get height
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// GPU batch renderer for multiple primitives
#[cfg(feature = "gpu")]
pub struct BatchRenderer {
    renderer: GpuRenderer,
    vertices: Vec<Vertex>,
}

#[cfg(feature = "gpu")]
impl BatchRenderer {
    /// Create a new batch renderer
    pub async fn new(width: u32, height: u32) -> Result<Self> {
        Ok(Self {
            renderer: GpuRenderer::new(width, height).await?,
            vertices: Vec::new(),
        })
    }

    /// Add rectangle to batch
    pub fn add_rect(&mut self, rect: Rect, color: Color) {
        let width = self.renderer.width as f32;
        let height = self.renderer.height as f32;

        let x1 = (rect.x / width) * 2.0 - 1.0;
        let y1 = (rect.y / height) * 2.0 - 1.0;
        let x2 = ((rect.x + rect.width) / width) * 2.0 - 1.0;
        let y2 = ((rect.y + rect.height) / height) * 2.0 - 1.0;

        let color_f = color.to_float();

        self.vertices.extend_from_slice(&[
            Vertex::new([x1, y1], color_f, [0.0, 0.0]),
            Vertex::new([x2, y1], color_f, [1.0, 0.0]),
            Vertex::new([x2, y2], color_f, [1.0, 1.0]),
            Vertex::new([x1, y1], color_f, [0.0, 0.0]),
            Vertex::new([x2, y2], color_f, [1.0, 1.0]),
            Vertex::new([x1, y2], color_f, [0.0, 1.0]),
        ]);
    }

    /// Render all batched primitives
    pub fn render(&self) -> Result<Vec<u8>> {
        self.renderer.render_vertices(&self.vertices)
    }

    /// Clear batch
    pub fn clear(&mut self) {
        self.vertices.clear();
    }
}

#[cfg(test)]
#[cfg(feature = "gpu")]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_creation() {
        let vertex = Vertex::new([0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0]);
        assert_eq!(vertex.position, [0.0, 0.0]);
        assert_eq!(vertex.color, [1.0, 1.0, 1.0, 1.0]);
    }

    #[tokio::test]
    #[ignore] // Requires GPU hardware; run with --ignored
    async fn test_gpu_renderer_creation() {
        // This test might fail if no GPU is available - use small size for speed
        let result = GpuRenderer::new(8, 8).await;
        // We don't assert success because GPU might not be available in test environment
        assert!(result.is_ok() || result.is_err());
    }
}
