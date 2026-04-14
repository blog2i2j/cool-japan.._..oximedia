//! Numpy-compatible buffer protocol support for zero-copy frame access.
//!
//! Provides buffer-backed frame types that can be exposed to Python's buffer
//! protocol, enabling numpy to create views over Rust-owned memory without
//! copying data. This module contains the pure-Rust data structures and
//! layout logic; PyO3 `#[pyclass]` wrappers reference these types.
//!
//! # Design
//!
//! - [`FrameBuffer`] holds raw pixel/sample data with layout metadata.
//! - [`BufferLayout`] describes shape, strides, and element type.
//! - [`BufferView`] provides an immutable slice view with bounds checking.
//! - All types are designed to be zero-copy compatible with numpy's
//!   `__buffer__` / `Py_buffer` protocol.

use std::fmt;

// ---------------------------------------------------------------------------
// ElementType
// ---------------------------------------------------------------------------

/// Element data type for buffer contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 16-bit integer (little-endian).
    U16,
    /// 32-bit IEEE float.
    F32,
    /// 64-bit IEEE float.
    F64,
    /// Signed 16-bit integer (little-endian).
    I16,
    /// Signed 32-bit integer.
    I32,
}

impl ElementType {
    /// Size in bytes of a single element.
    pub fn byte_size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 => 8,
        }
    }

    /// Python struct format character for `Py_buffer.format`.
    pub fn py_format_char(self) -> &'static str {
        match self {
            Self::U8 => "B",
            Self::U16 => "H",
            Self::I16 => "h",
            Self::I32 => "i",
            Self::F32 => "f",
            Self::F64 => "d",
        }
    }

    /// Numpy dtype string.
    pub fn numpy_dtype(self) -> &'static str {
        match self {
            Self::U8 => "uint8",
            Self::U16 => "uint16",
            Self::I16 => "int16",
            Self::I32 => "int32",
            Self::F32 => "float32",
            Self::F64 => "float64",
        }
    }
}

impl fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.numpy_dtype())
    }
}

// ---------------------------------------------------------------------------
// BufferLayout
// ---------------------------------------------------------------------------

/// Describes the memory layout of a multi-dimensional buffer.
///
/// Compatible with numpy's shape/strides model and Python's buffer protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferLayout {
    /// Shape of the buffer (e.g. `[height, width, channels]` for video).
    pub shape: Vec<usize>,
    /// Byte strides for each dimension.
    pub strides: Vec<usize>,
    /// Element data type.
    pub element_type: ElementType,
    /// Whether the data is in C-contiguous (row-major) order.
    pub c_contiguous: bool,
}

impl BufferLayout {
    /// Create a new C-contiguous layout for the given shape and element type.
    pub fn c_contiguous(shape: Vec<usize>, element_type: ElementType) -> Self {
        let ndim = shape.len();
        let mut strides = vec![0usize; ndim];
        if ndim > 0 {
            strides[ndim - 1] = element_type.byte_size();
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        Self {
            shape,
            strides,
            element_type,
            c_contiguous: true,
        }
    }

    /// Create a layout with explicit strides (may not be contiguous).
    pub fn with_strides(
        shape: Vec<usize>,
        strides: Vec<usize>,
        element_type: ElementType,
    ) -> Result<Self, BufferError> {
        if shape.len() != strides.len() {
            return Err(BufferError::ShapeMismatch {
                expected: shape.len(),
                got: strides.len(),
            });
        }
        // Check C-contiguity
        let c_layout = Self::c_contiguous(shape.clone(), element_type);
        let is_c = c_layout.strides == strides;
        Ok(Self {
            shape,
            strides,
            element_type,
            c_contiguous: is_c,
        })
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total byte size of the buffer.
    pub fn total_bytes(&self) -> usize {
        self.num_elements() * self.element_type.byte_size()
    }

    /// Create a video frame layout: `[height, width, channels]`.
    pub fn video_frame(height: u32, width: u32, channels: u32, element_type: ElementType) -> Self {
        Self::c_contiguous(
            vec![height as usize, width as usize, channels as usize],
            element_type,
        )
    }

    /// Create an audio buffer layout: `[samples, channels]`.
    pub fn audio_buffer(samples: usize, channels: u32, element_type: ElementType) -> Self {
        Self::c_contiguous(vec![samples, channels as usize], element_type)
    }

    /// Create a single-channel (planar) video layout: `[height, width]`.
    pub fn video_plane(height: u32, width: u32, element_type: ElementType) -> Self {
        Self::c_contiguous(vec![height as usize, width as usize], element_type)
    }
}

impl fmt::Display for BufferLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BufferLayout(shape={:?}, dtype={}, {})",
            self.shape,
            self.element_type,
            if self.c_contiguous {
                "C-contiguous"
            } else {
                "non-contiguous"
            }
        )
    }
}

// ---------------------------------------------------------------------------
// FrameBuffer
// ---------------------------------------------------------------------------

/// A raw buffer holding frame data with associated layout metadata.
///
/// Designed for zero-copy sharing with numpy via Python's buffer protocol.
#[derive(Debug, Clone)]
pub struct FrameBuffer {
    /// Raw byte storage.
    data: Vec<u8>,
    /// Layout describing shape, strides, and element type.
    layout: BufferLayout,
    /// Presentation timestamp in microseconds (-1 = unset).
    pts_us: i64,
    /// Optional label (e.g. "luma", "chroma_u", "audio_left").
    label: Option<String>,
}

impl FrameBuffer {
    /// Create a zero-filled buffer with the given layout.
    pub fn zeros(layout: BufferLayout) -> Self {
        let size = layout.total_bytes();
        Self {
            data: vec![0u8; size],
            layout,
            pts_us: -1,
            label: None,
        }
    }

    /// Create a buffer from existing data, validating size against layout.
    pub fn from_data(data: Vec<u8>, layout: BufferLayout) -> Result<Self, BufferError> {
        let expected = layout.total_bytes();
        if data.len() != expected {
            return Err(BufferError::SizeMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(Self {
            data,
            layout,
            pts_us: -1,
            label: None,
        })
    }

    /// Set the presentation timestamp.
    pub fn with_pts(mut self, pts_us: i64) -> Self {
        self.pts_us = pts_us;
        self
    }

    /// Set an optional label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get a reference to the raw data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get a mutable reference to the raw data.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get the layout descriptor.
    pub fn layout(&self) -> &BufferLayout {
        &self.layout
    }

    /// Get the presentation timestamp.
    pub fn pts_us(&self) -> i64 {
        self.pts_us
    }

    /// Get the optional label.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Total byte size of the buffer.
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    /// Number of elements (pixels, samples, etc.).
    pub fn num_elements(&self) -> usize {
        self.layout.num_elements()
    }

    /// Create a view into a sub-region by flattened byte offset and length.
    pub fn view(&self, offset: usize, length: usize) -> Result<BufferView<'_>, BufferError> {
        let end = offset.checked_add(length).ok_or(BufferError::OutOfBounds {
            offset,
            length,
            buffer_size: self.data.len(),
        })?;
        if end > self.data.len() {
            return Err(BufferError::OutOfBounds {
                offset,
                length,
                buffer_size: self.data.len(),
            });
        }
        Ok(BufferView {
            data: &self.data[offset..end],
            element_type: self.layout.element_type,
        })
    }

    /// Fill the entire buffer with a single byte value.
    pub fn fill(&mut self, value: u8) {
        self.data.fill(value);
    }
}

// ---------------------------------------------------------------------------
// BufferView
// ---------------------------------------------------------------------------

/// An immutable slice view into a [`FrameBuffer`].
#[derive(Debug)]
pub struct BufferView<'a> {
    /// Slice of the underlying data.
    data: &'a [u8],
    /// Element type for interpreting the bytes.
    element_type: ElementType,
}

impl BufferView<'_> {
    /// Raw byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        self.data
    }

    /// Number of bytes in this view.
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    /// Number of elements of the declared type.
    pub fn element_count(&self) -> usize {
        let elem_size = self.element_type.byte_size();
        if elem_size == 0 {
            return 0;
        }
        self.data.len() / elem_size
    }

    /// Element type.
    pub fn element_type(&self) -> ElementType {
        self.element_type
    }
}

// ---------------------------------------------------------------------------
// MultiPlaneBuffer
// ---------------------------------------------------------------------------

/// Multi-plane buffer for planar video formats (YUV420, etc.).
#[derive(Debug, Clone)]
pub struct MultiPlaneBuffer {
    /// Individual plane buffers.
    planes: Vec<FrameBuffer>,
    /// Video width.
    width: u32,
    /// Video height.
    height: u32,
}

impl MultiPlaneBuffer {
    /// Create a YUV420 planar buffer (3 planes: Y, U, V).
    pub fn yuv420(width: u32, height: u32) -> Self {
        let y_layout = BufferLayout::video_plane(height, width, ElementType::U8);
        let uv_w = (width + 1) / 2;
        let uv_h = (height + 1) / 2;
        let u_layout = BufferLayout::video_plane(uv_h, uv_w, ElementType::U8);
        let v_layout = BufferLayout::video_plane(uv_h, uv_w, ElementType::U8);

        Self {
            planes: vec![
                FrameBuffer::zeros(y_layout).with_label("Y"),
                FrameBuffer::zeros(u_layout).with_label("U"),
                FrameBuffer::zeros(v_layout).with_label("V"),
            ],
            width,
            height,
        }
    }

    /// Create a packed RGB buffer (single plane, 3 channels).
    pub fn rgb(width: u32, height: u32) -> Self {
        let layout = BufferLayout::video_frame(height, width, 3, ElementType::U8);
        Self {
            planes: vec![FrameBuffer::zeros(layout).with_label("RGB")],
            width,
            height,
        }
    }

    /// Number of planes.
    pub fn plane_count(&self) -> usize {
        self.planes.len()
    }

    /// Get a specific plane by index.
    pub fn plane(&self, index: usize) -> Option<&FrameBuffer> {
        self.planes.get(index)
    }

    /// Get a mutable reference to a plane.
    pub fn plane_mut(&mut self, index: usize) -> Option<&mut FrameBuffer> {
        self.planes.get_mut(index)
    }

    /// Video width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Video height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Total bytes across all planes.
    pub fn total_bytes(&self) -> usize {
        self.planes.iter().map(FrameBuffer::byte_size).sum()
    }
}

// ---------------------------------------------------------------------------
// BufferError
// ---------------------------------------------------------------------------

/// Errors related to buffer operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferError {
    /// Data size does not match the expected layout size.
    SizeMismatch {
        /// Expected byte count.
        expected: usize,
        /// Actual byte count.
        got: usize,
    },
    /// Shape and strides dimension count mismatch.
    ShapeMismatch {
        /// Expected dimension count.
        expected: usize,
        /// Actual dimension count.
        got: usize,
    },
    /// Out-of-bounds access.
    OutOfBounds {
        /// Requested offset.
        offset: usize,
        /// Requested length.
        length: usize,
        /// Total buffer size.
        buffer_size: usize,
    },
}

impl fmt::Display for BufferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeMismatch { expected, got } => {
                write!(f, "buffer size mismatch: expected {expected}, got {got}")
            }
            Self::ShapeMismatch { expected, got } => {
                write!(
                    f,
                    "shape/strides dimension mismatch: expected {expected}, got {got}"
                )
            }
            Self::OutOfBounds {
                offset,
                length,
                buffer_size,
            } => write!(
                f,
                "out of bounds: offset={offset}, length={length}, buffer_size={buffer_size}"
            ),
        }
    }
}

impl std::error::Error for BufferError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── ElementType ───────────────────────────────────────────────────────

    #[test]
    fn test_element_type_byte_size() {
        assert_eq!(ElementType::U8.byte_size(), 1);
        assert_eq!(ElementType::U16.byte_size(), 2);
        assert_eq!(ElementType::I16.byte_size(), 2);
        assert_eq!(ElementType::F32.byte_size(), 4);
        assert_eq!(ElementType::I32.byte_size(), 4);
        assert_eq!(ElementType::F64.byte_size(), 8);
    }

    #[test]
    fn test_element_type_py_format() {
        assert_eq!(ElementType::U8.py_format_char(), "B");
        assert_eq!(ElementType::F32.py_format_char(), "f");
        assert_eq!(ElementType::I16.py_format_char(), "h");
    }

    #[test]
    fn test_element_type_numpy_dtype() {
        assert_eq!(ElementType::U8.numpy_dtype(), "uint8");
        assert_eq!(ElementType::F32.numpy_dtype(), "float32");
        assert_eq!(ElementType::F64.numpy_dtype(), "float64");
    }

    #[test]
    fn test_element_type_display() {
        assert_eq!(format!("{}", ElementType::U8), "uint8");
        assert_eq!(format!("{}", ElementType::I32), "int32");
    }

    // ── BufferLayout ──────────────────────────────────────────────────────

    #[test]
    fn test_layout_c_contiguous_1d() {
        let layout = BufferLayout::c_contiguous(vec![100], ElementType::F32);
        assert_eq!(layout.ndim(), 1);
        assert_eq!(layout.num_elements(), 100);
        assert_eq!(layout.total_bytes(), 400);
        assert_eq!(layout.strides, vec![4]);
        assert!(layout.c_contiguous);
    }

    #[test]
    fn test_layout_c_contiguous_3d() {
        // [height=1080, width=1920, channels=3] of U8
        let layout = BufferLayout::c_contiguous(vec![1080, 1920, 3], ElementType::U8);
        assert_eq!(layout.ndim(), 3);
        assert_eq!(layout.num_elements(), 1080 * 1920 * 3);
        assert_eq!(layout.total_bytes(), 1080 * 1920 * 3);
        assert_eq!(layout.strides, vec![1920 * 3, 3, 1]);
    }

    #[test]
    fn test_layout_with_strides_valid() {
        let layout = BufferLayout::with_strides(
            vec![10, 20],
            vec![20, 1],
            ElementType::U8,
        );
        assert!(layout.is_ok());
        let l = layout.expect("should be ok");
        assert!(l.c_contiguous);
    }

    #[test]
    fn test_layout_with_strides_mismatch() {
        let result = BufferLayout::with_strides(
            vec![10, 20],
            vec![20],
            ElementType::U8,
        );
        assert!(matches!(result, Err(BufferError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_layout_video_frame() {
        let layout = BufferLayout::video_frame(1080, 1920, 3, ElementType::U8);
        assert_eq!(layout.shape, vec![1080, 1920, 3]);
        assert_eq!(layout.total_bytes(), 1080 * 1920 * 3);
    }

    #[test]
    fn test_layout_audio_buffer() {
        let layout = BufferLayout::audio_buffer(48000, 2, ElementType::F32);
        assert_eq!(layout.shape, vec![48000, 2]);
        assert_eq!(layout.total_bytes(), 48000 * 2 * 4);
    }

    #[test]
    fn test_layout_video_plane() {
        let layout = BufferLayout::video_plane(720, 1280, ElementType::U8);
        assert_eq!(layout.shape, vec![720, 1280]);
        assert_eq!(layout.total_bytes(), 720 * 1280);
    }

    #[test]
    fn test_layout_display() {
        let layout = BufferLayout::c_contiguous(vec![10, 20], ElementType::F32);
        let s = format!("{layout}");
        assert!(s.contains("shape=[10, 20]"));
        assert!(s.contains("float32"));
        assert!(s.contains("C-contiguous"));
    }

    // ── FrameBuffer ───────────────────────────────────────────────────────

    #[test]
    fn test_frame_buffer_zeros() {
        let layout = BufferLayout::c_contiguous(vec![10], ElementType::U8);
        let buf = FrameBuffer::zeros(layout);
        assert_eq!(buf.byte_size(), 10);
        assert_eq!(buf.num_elements(), 10);
        assert!(buf.data().iter().all(|&b| b == 0));
        assert_eq!(buf.pts_us(), -1);
        assert!(buf.label().is_none());
    }

    #[test]
    fn test_frame_buffer_from_data_ok() {
        let layout = BufferLayout::c_contiguous(vec![4], ElementType::U8);
        let data = vec![1, 2, 3, 4];
        let buf = FrameBuffer::from_data(data, layout);
        assert!(buf.is_ok());
        let b = buf.expect("should succeed");
        assert_eq!(b.data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_frame_buffer_from_data_size_mismatch() {
        let layout = BufferLayout::c_contiguous(vec![4], ElementType::U8);
        let data = vec![1, 2]; // too short
        let err = FrameBuffer::from_data(data, layout);
        assert!(matches!(err, Err(BufferError::SizeMismatch { .. })));
    }

    #[test]
    fn test_frame_buffer_with_pts_and_label() {
        let layout = BufferLayout::c_contiguous(vec![2], ElementType::U8);
        let buf = FrameBuffer::zeros(layout)
            .with_pts(42_000)
            .with_label("luma");
        assert_eq!(buf.pts_us(), 42_000);
        assert_eq!(buf.label(), Some("luma"));
    }

    #[test]
    fn test_frame_buffer_fill() {
        let layout = BufferLayout::c_contiguous(vec![8], ElementType::U8);
        let mut buf = FrameBuffer::zeros(layout);
        buf.fill(0xFF);
        assert!(buf.data().iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn test_frame_buffer_data_mut() {
        let layout = BufferLayout::c_contiguous(vec![4], ElementType::U8);
        let mut buf = FrameBuffer::zeros(layout);
        buf.data_mut()[0] = 42;
        assert_eq!(buf.data()[0], 42);
    }

    // ── BufferView ────────────────────────────────────────────────────────

    #[test]
    fn test_buffer_view_ok() {
        let layout = BufferLayout::c_contiguous(vec![10], ElementType::U8);
        let mut buf = FrameBuffer::zeros(layout);
        buf.data_mut()[2] = 99;
        let view = buf.view(2, 3);
        assert!(view.is_ok());
        let v = view.expect("should succeed");
        assert_eq!(v.byte_len(), 3);
        assert_eq!(v.as_bytes()[0], 99);
        assert_eq!(v.element_count(), 3);
        assert_eq!(v.element_type(), ElementType::U8);
    }

    #[test]
    fn test_buffer_view_out_of_bounds() {
        let layout = BufferLayout::c_contiguous(vec![4], ElementType::U8);
        let buf = FrameBuffer::zeros(layout);
        let result = buf.view(2, 10);
        assert!(matches!(result, Err(BufferError::OutOfBounds { .. })));
    }

    #[test]
    fn test_buffer_view_f32_element_count() {
        let layout = BufferLayout::c_contiguous(vec![4], ElementType::F32);
        let buf = FrameBuffer::zeros(layout);
        let view = buf.view(0, 16).expect("should succeed");
        assert_eq!(view.element_count(), 4); // 16 bytes / 4 bytes per f32
    }

    // ── MultiPlaneBuffer ──────────────────────────────────────────────────

    #[test]
    fn test_multi_plane_yuv420() {
        let mp = MultiPlaneBuffer::yuv420(320, 240);
        assert_eq!(mp.plane_count(), 3);
        assert_eq!(mp.width(), 320);
        assert_eq!(mp.height(), 240);
        // Y plane: 320*240 = 76800
        // U plane: 160*120 = 19200
        // V plane: 160*120 = 19200
        assert_eq!(mp.total_bytes(), 76800 + 19200 + 19200);

        let y = mp.plane(0).expect("should have Y plane");
        assert_eq!(y.label(), Some("Y"));
        let u = mp.plane(1).expect("should have U plane");
        assert_eq!(u.label(), Some("U"));
    }

    #[test]
    fn test_multi_plane_rgb() {
        let mp = MultiPlaneBuffer::rgb(1920, 1080);
        assert_eq!(mp.plane_count(), 1);
        assert_eq!(mp.total_bytes(), 1920 * 1080 * 3);
    }

    #[test]
    fn test_multi_plane_odd_dimensions() {
        // Odd dimensions: 321x241
        let mp = MultiPlaneBuffer::yuv420(321, 241);
        assert_eq!(mp.plane_count(), 3);
        let y = mp.plane(0).expect("Y");
        // Y: 321 * 241 = 77361
        assert_eq!(y.byte_size(), 321 * 241);
        let u = mp.plane(1).expect("U");
        // U: ceil(321/2) * ceil(241/2) = 161 * 121 = 19481
        assert_eq!(u.byte_size(), 161 * 121);
    }

    #[test]
    fn test_multi_plane_mut() {
        let mut mp = MultiPlaneBuffer::yuv420(4, 4);
        let y = mp.plane_mut(0).expect("Y");
        y.fill(128);
        let y_ref = mp.plane(0).expect("Y");
        assert!(y_ref.data().iter().all(|&b| b == 128));
    }

    #[test]
    fn test_multi_plane_out_of_range() {
        let mp = MultiPlaneBuffer::rgb(10, 10);
        assert!(mp.plane(1).is_none());
    }

    // ── BufferError ───────────────────────────────────────────────────────

    #[test]
    fn test_buffer_error_display() {
        let e = BufferError::SizeMismatch {
            expected: 100,
            got: 50,
        };
        assert!(e.to_string().contains("100"));
        assert!(e.to_string().contains("50"));

        let e2 = BufferError::ShapeMismatch {
            expected: 3,
            got: 2,
        };
        assert!(e2.to_string().contains("dimension"));

        let e3 = BufferError::OutOfBounds {
            offset: 10,
            length: 20,
            buffer_size: 15,
        };
        assert!(e3.to_string().contains("out of bounds"));
    }
}
