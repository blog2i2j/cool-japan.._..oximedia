//! Vertex buffer layout descriptions and management.
//!
//! Defines vertex attribute formats, stride calculations, and an in-memory
//! vertex buffer abstraction for use with GPU render pipelines.

#![allow(dead_code)]

/// The data format and dimensionality of a single vertex attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexAttribute {
    /// Single `f32` component (e.g. a scalar weight).
    Float32,
    /// Two `f32` components (e.g. UV texture coordinates).
    Float32x2,
    /// Three `f32` components (e.g. position XYZ or normal).
    Float32x3,
    /// Four `f32` components (e.g. RGBA colour or XYZW position).
    Float32x4,
    /// Single `u32` (e.g. material index).
    Uint32,
    /// Two `u32` components.
    Uint32x2,
    /// Single `i32`.
    Sint32,
    /// Four `u8` normalised to [0, 1] (e.g. packed colours).
    Unorm8x4,
}

impl VertexAttribute {
    /// Size of this attribute in bytes.
    #[must_use]
    pub const fn byte_size(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float32x2 => 8,
            Self::Float32x3 => 12,
            Self::Float32x4 => 16,
            Self::Uint32 => 4,
            Self::Uint32x2 => 8,
            Self::Sint32 => 4,
            Self::Unorm8x4 => 4,
        }
    }

    /// Number of scalar components in this attribute.
    #[must_use]
    pub const fn component_count(self) -> usize {
        match self {
            Self::Float32 | Self::Uint32 | Self::Sint32 => 1,
            Self::Float32x2 | Self::Uint32x2 => 2,
            Self::Float32x3 => 3,
            Self::Float32x4 | Self::Unorm8x4 => 4,
        }
    }
}

/// A named slot in a vertex layout, pairing a semantic name with its format.
#[derive(Debug, Clone)]
pub struct VertexSlot {
    /// Semantic name used in shaders (e.g. `"POSITION"`, `"TEXCOORD"`).
    pub name: String,
    /// Data format of this slot.
    pub attribute: VertexAttribute,
    /// Byte offset from the start of the vertex record.
    pub offset: usize,
}

/// Describes the memory layout of a single interleaved vertex record.
///
/// Attributes are stored in insertion order; the stride is computed
/// automatically from the sum of all attribute sizes.
#[derive(Debug, Clone, Default)]
pub struct VertexLayout {
    slots: Vec<VertexSlot>,
}

impl VertexLayout {
    /// Create an empty layout.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a named attribute to the layout.
    ///
    /// Returns `&mut self` for builder-style chaining.
    pub fn add(&mut self, name: impl Into<String>, attribute: VertexAttribute) -> &mut Self {
        let offset = self.stride();
        self.slots.push(VertexSlot {
            name: name.into(),
            attribute,
            offset,
        });
        self
    }

    /// Total byte size of one vertex record (sum of all attribute sizes).
    #[must_use]
    pub fn stride(&self) -> usize {
        self.slots.iter().map(|s| s.attribute.byte_size()).sum()
    }

    /// Number of attributes in the layout.
    #[must_use]
    pub fn attribute_count(&self) -> usize {
        self.slots.len()
    }

    /// Iterate over the slots in declaration order.
    #[must_use]
    pub fn slots(&self) -> &[VertexSlot] {
        &self.slots
    }

    /// Return the slot with `name`, if present.
    #[must_use]
    pub fn slot_by_name(&self, name: &str) -> Option<&VertexSlot> {
        self.slots.iter().find(|s| s.name == name)
    }
}

/// Errors that can occur when working with a [`VertexBuffer`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VertexBufferError {
    /// The raw byte data length is not a multiple of the layout stride.
    StrideMismatch {
        /// Stride implied by the layout.
        stride: usize,
        /// Actual byte length of the data.
        data_len: usize,
    },
    /// The buffer is empty and no vertices could be retrieved.
    Empty,
}

impl std::fmt::Display for VertexBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StrideMismatch { stride, data_len } => write!(
                f,
                "data length {data_len} is not a multiple of stride {stride}"
            ),
            Self::Empty => write!(f, "vertex buffer is empty"),
        }
    }
}

impl std::error::Error for VertexBufferError {}

/// An in-memory buffer of interleaved vertex data with an associated layout.
///
/// # Example
///
/// ```
/// use oximedia_gpu::vertex_buffer::{VertexAttribute, VertexLayout, VertexBuffer};
///
/// let mut layout = VertexLayout::new();
/// layout.add("POSITION", VertexAttribute::Float32x3);
/// layout.add("TEXCOORD", VertexAttribute::Float32x2);
///
/// // 1 vertex = 20 bytes
/// let data = vec![0u8; 20];
/// let vb = VertexBuffer::new(layout, data)?;
/// assert_eq!(vb.vertex_count(), 1);
/// assert_eq!(vb.stride(), 20);
/// ```
#[derive(Debug, Clone)]
pub struct VertexBuffer {
    layout: VertexLayout,
    data: Vec<u8>,
}

impl VertexBuffer {
    /// Create a new vertex buffer, validating that `data.len()` is a multiple
    /// of the layout stride.
    ///
    /// # Errors
    ///
    /// Returns [`VertexBufferError::StrideMismatch`] if the data is not aligned
    /// to the layout stride, or [`VertexBufferError::Empty`] if the stride is
    /// zero.
    pub fn new(layout: VertexLayout, data: Vec<u8>) -> Result<Self, VertexBufferError> {
        let stride = layout.stride();
        if stride == 0 {
            return Err(VertexBufferError::Empty);
        }
        if data.len() % stride != 0 {
            return Err(VertexBufferError::StrideMismatch {
                stride,
                data_len: data.len(),
            });
        }
        Ok(Self { layout, data })
    }

    /// Byte stride between consecutive vertex records.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.layout.stride()
    }

    /// Number of complete vertex records stored in the buffer.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        let s = self.stride();
        if s == 0 {
            0
        } else {
            self.data.len() / s
        }
    }

    /// Total size of the raw byte data.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    /// Raw byte slice for the entire buffer.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Return the layout describing this buffer's format.
    #[must_use]
    pub fn layout(&self) -> &VertexLayout {
        &self.layout
    }

    /// Return the raw bytes for vertex at `index`, or `None` if out of range.
    #[must_use]
    pub fn vertex_bytes(&self, index: usize) -> Option<&[u8]> {
        let s = self.stride();
        let start = index * s;
        let end = start + s;
        self.data.get(start..end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attribute_float32_size() {
        assert_eq!(VertexAttribute::Float32.byte_size(), 4);
    }

    #[test]
    fn attribute_float32x3_size() {
        assert_eq!(VertexAttribute::Float32x3.byte_size(), 12);
    }

    #[test]
    fn attribute_unorm8x4_size() {
        assert_eq!(VertexAttribute::Unorm8x4.byte_size(), 4);
    }

    #[test]
    fn attribute_component_counts() {
        assert_eq!(VertexAttribute::Float32.component_count(), 1);
        assert_eq!(VertexAttribute::Float32x2.component_count(), 2);
        assert_eq!(VertexAttribute::Float32x3.component_count(), 3);
        assert_eq!(VertexAttribute::Float32x4.component_count(), 4);
    }

    #[test]
    fn layout_stride_single_attr() {
        let mut l = VertexLayout::new();
        l.add("POS", VertexAttribute::Float32x3);
        assert_eq!(l.stride(), 12);
    }

    #[test]
    fn layout_stride_multiple_attrs() {
        let mut l = VertexLayout::new();
        l.add("POS", VertexAttribute::Float32x3);
        l.add("UV", VertexAttribute::Float32x2);
        assert_eq!(l.stride(), 20);
    }

    #[test]
    fn layout_offsets_are_cumulative() {
        let mut l = VertexLayout::new();
        l.add("POS", VertexAttribute::Float32x3);
        l.add("UV", VertexAttribute::Float32x2);
        assert_eq!(l.slots()[0].offset, 0);
        assert_eq!(l.slots()[1].offset, 12);
    }

    #[test]
    fn layout_slot_by_name() {
        let mut l = VertexLayout::new();
        l.add("NORMAL", VertexAttribute::Float32x3);
        assert!(l.slot_by_name("NORMAL").is_some());
        assert!(l.slot_by_name("UV").is_none());
    }

    #[test]
    fn vertex_buffer_create_ok() {
        let mut l = VertexLayout::new();
        l.add("POS", VertexAttribute::Float32x3);
        let vb =
            VertexBuffer::new(l, vec![0u8; 24]).expect("vertex buffer creation should succeed");
        assert_eq!(vb.vertex_count(), 2);
    }

    #[test]
    fn vertex_buffer_stride_mismatch_error() {
        let mut l = VertexLayout::new();
        l.add("POS", VertexAttribute::Float32x3);
        let err = VertexBuffer::new(l, vec![0u8; 13]).unwrap_err();
        matches!(err, VertexBufferError::StrideMismatch { .. });
    }

    #[test]
    fn vertex_buffer_empty_layout_error() {
        let l = VertexLayout::new();
        let err = VertexBuffer::new(l, vec![]).unwrap_err();
        assert_eq!(err, VertexBufferError::Empty);
    }

    #[test]
    fn vertex_buffer_vertex_bytes_valid() {
        let mut l = VertexLayout::new();
        l.add("POS", VertexAttribute::Uint32);
        let data: Vec<u8> = (0u8..8).collect();
        let vb = VertexBuffer::new(l, data.clone()).expect("vertex buffer creation should succeed");
        assert_eq!(vb.vertex_bytes(0), Some(&data[0..4]));
        assert_eq!(vb.vertex_bytes(1), Some(&data[4..8]));
        assert!(vb.vertex_bytes(2).is_none());
    }

    #[test]
    fn vertex_buffer_byte_len() {
        let mut l = VertexLayout::new();
        l.add("POS", VertexAttribute::Float32x4);
        let vb =
            VertexBuffer::new(l, vec![0u8; 32]).expect("vertex buffer creation should succeed");
        assert_eq!(vb.byte_len(), 32);
    }
}
