//! Render pass configuration and builder for `oximedia-gpu`.
//!
//! Provides load/store operation enums, per-attachment configuration, and a
//! builder pattern for constructing `RenderPassConfig` descriptors.

#![allow(dead_code)]

/// Specifies what happens to an attachment's contents at the start of a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoadOp {
    /// Clear the attachment to a specified value.
    Clear,
    /// Load the existing contents.
    Load,
    /// Don't care about existing contents (undefined after load).
    DontCare,
}

impl LoadOp {
    /// Returns `true` when existing data will be preserved (Load).
    #[must_use]
    pub fn preserves_content(&self) -> bool {
        matches!(self, Self::Load)
    }

    /// Returns a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Clear => "clear",
            Self::Load => "load",
            Self::DontCare => "dont_care",
        }
    }
}

/// Specifies what happens to an attachment's contents at the end of a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StoreOp {
    /// Store the results back to the attachment.
    Store,
    /// Discard the results (transient attachment).
    Discard,
    /// Don't care about the result.
    DontCare,
}

impl StoreOp {
    /// Returns `true` when the pass output will be written back.
    #[must_use]
    pub fn writes_output(&self) -> bool {
        matches!(self, Self::Store)
    }

    /// Returns a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Store => "store",
            Self::Discard => "discard",
            Self::DontCare => "dont_care",
        }
    }
}

/// Pixel format of a render attachment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttachmentFormat {
    /// 8-bit per channel RGBA.
    Rgba8Unorm,
    /// 16-bit per channel RGBA (HDR).
    Rgba16Float,
    /// 32-bit per channel RGBA (HDR).
    Rgba32Float,
    /// 32-bit depth, 8-bit stencil.
    Depth32FloatStencil8,
    /// 32-bit float depth only.
    Depth32Float,
    /// 24-bit depth, 8-bit stencil.
    Depth24PlusStencil8,
}

impl AttachmentFormat {
    /// Returns `true` when this format contains a depth component.
    #[must_use]
    pub fn has_depth(&self) -> bool {
        matches!(
            self,
            Self::Depth32FloatStencil8 | Self::Depth32Float | Self::Depth24PlusStencil8
        )
    }

    /// Returns `true` when this format contains a stencil component.
    #[must_use]
    pub fn has_stencil(&self) -> bool {
        matches!(self, Self::Depth32FloatStencil8 | Self::Depth24PlusStencil8)
    }

    /// Returns the number of bytes per texel.
    #[must_use]
    pub fn bytes_per_texel(&self) -> u32 {
        match self {
            Self::Rgba8Unorm => 4,
            Self::Rgba16Float => 8,
            Self::Rgba32Float => 16,
            Self::Depth32FloatStencil8 => 5,
            Self::Depth32Float => 4,
            Self::Depth24PlusStencil8 => 4,
        }
    }
}

/// Configuration for a single attachment within a render pass.
#[derive(Debug, Clone)]
pub struct AttachmentConfig {
    /// Pixel format.
    pub format: AttachmentFormat,
    /// Operation at the start of the pass.
    pub load_op: LoadOp,
    /// Operation at the end of the pass.
    pub store_op: StoreOp,
    /// Optional MSAA sample count (1 = no MSAA).
    pub sample_count: u32,
    /// Debug label.
    pub label: Option<String>,
}

impl AttachmentConfig {
    /// Creates an attachment configuration.
    #[must_use]
    pub fn new(format: AttachmentFormat, load_op: LoadOp, store_op: StoreOp) -> Self {
        Self {
            format,
            load_op,
            store_op,
            sample_count: 1,
            label: None,
        }
    }

    /// Returns `true` when this attachment is a depth (or depth/stencil) attachment.
    #[must_use]
    pub fn has_depth(&self) -> bool {
        self.format.has_depth()
    }

    /// Returns `true` when MSAA is enabled (sample count > 1).
    #[must_use]
    pub fn is_multisampled(&self) -> bool {
        self.sample_count > 1
    }

    /// Sets the MSAA sample count.
    #[must_use]
    pub fn with_sample_count(mut self, count: u32) -> Self {
        self.sample_count = count;
        self
    }

    /// Attaches a debug label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Describes all attachments used in a single render pass.
#[derive(Debug, Clone)]
pub struct RenderPassConfig {
    /// Color attachments (up to 8 on most hardware).
    pub color_attachments: Vec<AttachmentConfig>,
    /// Optional depth/stencil attachment.
    pub depth_attachment: Option<AttachmentConfig>,
    /// Debug label for the render pass.
    pub label: Option<String>,
}

impl RenderPassConfig {
    /// Returns the total number of attachments (color + optional depth).
    #[must_use]
    pub fn attachment_count(&self) -> usize {
        self.color_attachments.len() + usize::from(self.depth_attachment.is_some())
    }

    /// Returns `true` when a depth attachment is present.
    #[must_use]
    pub fn has_depth_attachment(&self) -> bool {
        self.depth_attachment.is_some()
    }

    /// Returns `true` when all color attachments use the same format.
    #[must_use]
    pub fn has_uniform_color_format(&self) -> bool {
        let mut iter = self.color_attachments.iter().map(|a| a.format);
        match iter.next() {
            None => true,
            Some(first) => iter.all(|f| f == first),
        }
    }
}

/// Fluent builder for [`RenderPassConfig`].
#[derive(Debug, Default)]
pub struct RenderPassBuilder {
    color_attachments: Vec<AttachmentConfig>,
    depth_attachment: Option<AttachmentConfig>,
    label: Option<String>,
}

impl RenderPassBuilder {
    /// Creates a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a color attachment.
    #[must_use]
    pub fn add_color_attachment(mut self, attachment: AttachmentConfig) -> Self {
        self.color_attachments.push(attachment);
        self
    }

    /// Sets the depth/stencil attachment.
    ///
    /// Returns an error string if the attachment format does not contain depth.
    pub fn set_depth_attachment(mut self, attachment: AttachmentConfig) -> Result<Self, String> {
        if !attachment.has_depth() {
            return Err(format!(
                "Format {:?} does not contain a depth component",
                attachment.format
            ));
        }
        self.depth_attachment = Some(attachment);
        Ok(self)
    }

    /// Sets a debug label for the render pass.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Consumes the builder and returns the [`RenderPassConfig`].
    ///
    /// Returns an error if no color attachments have been added.
    pub fn build(self) -> Result<RenderPassConfig, String> {
        if self.color_attachments.is_empty() {
            return Err("RenderPassConfig requires at least one color attachment".into());
        }
        Ok(RenderPassConfig {
            color_attachments: self.color_attachments,
            depth_attachment: self.depth_attachment,
            label: self.label,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_op_preserves_content_load() {
        assert!(LoadOp::Load.preserves_content());
    }

    #[test]
    fn test_load_op_clear_does_not_preserve() {
        assert!(!LoadOp::Clear.preserves_content());
    }

    #[test]
    fn test_store_op_writes_output_store() {
        assert!(StoreOp::Store.writes_output());
    }

    #[test]
    fn test_store_op_discard_no_write() {
        assert!(!StoreOp::Discard.writes_output());
    }

    #[test]
    fn test_attachment_format_has_depth_depth32() {
        assert!(AttachmentFormat::Depth32Float.has_depth());
    }

    #[test]
    fn test_attachment_format_rgba8_no_depth() {
        assert!(!AttachmentFormat::Rgba8Unorm.has_depth());
    }

    #[test]
    fn test_attachment_format_has_stencil_depth24() {
        assert!(AttachmentFormat::Depth24PlusStencil8.has_stencil());
    }

    #[test]
    fn test_attachment_format_bytes_per_texel_rgba32() {
        assert_eq!(AttachmentFormat::Rgba32Float.bytes_per_texel(), 16);
    }

    #[test]
    fn test_attachment_config_has_depth_true() {
        let a = AttachmentConfig::new(
            AttachmentFormat::Depth32Float,
            LoadOp::Clear,
            StoreOp::Store,
        );
        assert!(a.has_depth());
    }

    #[test]
    fn test_attachment_config_not_multisampled_by_default() {
        let a = AttachmentConfig::new(AttachmentFormat::Rgba8Unorm, LoadOp::Clear, StoreOp::Store);
        assert!(!a.is_multisampled());
    }

    #[test]
    fn test_attachment_config_with_sample_count() {
        let a = AttachmentConfig::new(AttachmentFormat::Rgba8Unorm, LoadOp::Clear, StoreOp::Store)
            .with_sample_count(4);
        assert!(a.is_multisampled());
        assert_eq!(a.sample_count, 4);
    }

    #[test]
    fn test_render_pass_builder_build_ok() {
        let color =
            AttachmentConfig::new(AttachmentFormat::Rgba8Unorm, LoadOp::Clear, StoreOp::Store);
        let config = RenderPassBuilder::new()
            .add_color_attachment(color)
            .build()
            .expect("operation should succeed in test");
        assert_eq!(config.attachment_count(), 1);
    }

    #[test]
    fn test_render_pass_builder_build_no_color_err() {
        let result = RenderPassBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_render_pass_builder_with_depth() {
        let color =
            AttachmentConfig::new(AttachmentFormat::Rgba8Unorm, LoadOp::Clear, StoreOp::Store);
        let depth = AttachmentConfig::new(
            AttachmentFormat::Depth32Float,
            LoadOp::Clear,
            StoreOp::Discard,
        );
        let config = RenderPassBuilder::new()
            .add_color_attachment(color)
            .set_depth_attachment(depth)
            .expect("operation should succeed in test")
            .build()
            .expect("operation should succeed in test");
        assert!(config.has_depth_attachment());
        assert_eq!(config.attachment_count(), 2);
    }

    #[test]
    fn test_render_pass_builder_depth_non_depth_format_err() {
        let bad =
            AttachmentConfig::new(AttachmentFormat::Rgba8Unorm, LoadOp::Clear, StoreOp::Store);
        let result = RenderPassBuilder::new().set_depth_attachment(bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_render_pass_uniform_color_format_true_single() {
        let color =
            AttachmentConfig::new(AttachmentFormat::Rgba8Unorm, LoadOp::Clear, StoreOp::Store);
        let config = RenderPassBuilder::new()
            .add_color_attachment(color)
            .build()
            .expect("operation should succeed in test");
        assert!(config.has_uniform_color_format());
    }
}
