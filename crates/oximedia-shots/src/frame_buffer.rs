//! Pure Rust frame buffer types replacing ndarray dependency.
//!
//! Provides `FrameBuffer` (3D: height x width x channels) and `GrayImage` / `FloatImage`
//! (2D: height x width) for image processing without external linear algebra crates.

/// A 3-dimensional frame buffer (height x width x channels) backed by a flat `Vec<u8>`.
///
/// This replaces `ndarray::Array3<u8>` for storing RGB video frames.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameBuffer {
    data: Vec<u8>,
    height: usize,
    width: usize,
    channels: usize,
}

impl FrameBuffer {
    /// Create a new frame buffer filled with zeros.
    #[must_use]
    pub fn zeros(height: usize, width: usize, channels: usize) -> Self {
        Self {
            data: vec![0u8; height * width * channels],
            height,
            width,
            channels,
        }
    }

    /// Create a new frame buffer filled with a constant value.
    #[must_use]
    pub fn from_elem(height: usize, width: usize, channels: usize, value: u8) -> Self {
        Self {
            data: vec![value; height * width * channels],
            height,
            width,
            channels,
        }
    }

    /// Create from raw data.
    ///
    /// Returns `None` if the data length does not match `height * width * channels`.
    #[must_use]
    pub fn from_vec(height: usize, width: usize, channels: usize, data: Vec<u8>) -> Option<Self> {
        if data.len() != height * width * channels {
            return None;
        }
        Some(Self {
            data,
            height,
            width,
            channels,
        })
    }

    /// Get a pixel value at (y, x, c).
    #[inline]
    #[must_use]
    pub fn get(&self, y: usize, x: usize, c: usize) -> u8 {
        self.data[(y * self.width + x) * self.channels + c]
    }

    /// Set a pixel value at (y, x, c).
    #[inline]
    pub fn set(&mut self, y: usize, x: usize, c: usize, value: u8) {
        self.data[(y * self.width + x) * self.channels + c] = value;
    }

    /// Get the dimensions as (height, width, channels).
    #[must_use]
    pub fn dim(&self) -> (usize, usize, usize) {
        (self.height, self.width, self.channels)
    }

    /// Get the height.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get the width.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get the number of channels.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Fill the buffer with a constant value.
    pub fn fill(&mut self, value: u8) {
        self.data.fill(value);
    }

    /// Get a reference to the raw data.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Check if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Total number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

/// A 2-dimensional grayscale image backed by a flat `Vec<u8>`.
///
/// This replaces `ndarray::Array2<u8>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrayImage {
    data: Vec<u8>,
    height: usize,
    width: usize,
}

impl GrayImage {
    /// Create a new grayscale image filled with zeros.
    #[must_use]
    pub fn zeros(height: usize, width: usize) -> Self {
        Self {
            data: vec![0u8; height * width],
            height,
            width,
        }
    }

    /// Get a pixel value at (y, x).
    #[inline]
    #[must_use]
    pub fn get(&self, y: usize, x: usize) -> u8 {
        self.data[y * self.width + x]
    }

    /// Set a pixel value at (y, x).
    #[inline]
    pub fn set(&mut self, y: usize, x: usize, value: u8) {
        self.data[y * self.width + x] = value;
    }

    /// Get the dimensions as (height, width).
    #[must_use]
    pub fn dim(&self) -> (usize, usize) {
        (self.height, self.width)
    }

    /// Get the height.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get the width.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }
}

/// A 2-dimensional float image backed by a flat `Vec<f32>`.
///
/// This replaces `ndarray::Array2<f32>`.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatImage {
    data: Vec<f32>,
    height: usize,
    width: usize,
}

impl FloatImage {
    /// Create a new float image filled with zeros.
    #[must_use]
    pub fn zeros(height: usize, width: usize) -> Self {
        Self {
            data: vec![0.0f32; height * width],
            height,
            width,
        }
    }

    /// Get a pixel value at (y, x).
    #[inline]
    #[must_use]
    pub fn get(&self, y: usize, x: usize) -> f32 {
        self.data[y * self.width + x]
    }

    /// Set a pixel value at (y, x).
    #[inline]
    pub fn set(&mut self, y: usize, x: usize, value: f32) {
        self.data[y * self.width + x] = value;
    }

    /// Get the dimensions as (height, width).
    #[must_use]
    pub fn dim(&self) -> (usize, usize) {
        (self.height, self.width)
    }
}

/// Convert an RGB `FrameBuffer` to a `GrayImage` using BT.601 coefficients.
#[must_use]
pub fn to_grayscale(frame: &FrameBuffer) -> GrayImage {
    let (h, w, _) = frame.dim();
    let mut gray = GrayImage::zeros(h, w);
    for y in 0..h {
        for x in 0..w {
            let r = f32::from(frame.get(y, x, 0));
            let g = f32::from(frame.get(y, x, 1));
            let b = f32::from(frame.get(y, x, 2));
            gray.set(y, x, ((r * 0.299) + (g * 0.587) + (b * 0.114)) as u8);
        }
    }
    gray
}

/// Apply Sobel edge detection to a grayscale image.
#[must_use]
pub fn detect_edges(gray: &GrayImage) -> GrayImage {
    let (h, w) = gray.dim();
    let mut edges = GrayImage::zeros(h, w);

    let sobel_x: [[i32; 3]; 3] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
    let sobel_y: [[i32; 3]; 3] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

    for y in 1..(h.saturating_sub(1)) {
        for x in 1..(w.saturating_sub(1)) {
            let mut gx = 0i32;
            let mut gy = 0i32;

            for dy in 0..3 {
                for dx in 0..3 {
                    let pixel = i32::from(gray.get(y + dy - 1, x + dx - 1));
                    gx += pixel * sobel_x[dy][dx];
                    gy += pixel * sobel_y[dy][dx];
                }
            }

            let magnitude = ((gx * gx + gy * gy) as f32).sqrt();
            edges.set(y, x, magnitude.min(255.0) as u8);
        }
    }

    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_buffer_zeros() {
        let fb = FrameBuffer::zeros(10, 20, 3);
        assert_eq!(fb.dim(), (10, 20, 3));
        assert_eq!(fb.get(0, 0, 0), 0);
    }

    #[test]
    fn test_frame_buffer_from_elem() {
        let fb = FrameBuffer::from_elem(5, 5, 3, 128);
        assert_eq!(fb.get(2, 3, 1), 128);
    }

    #[test]
    fn test_frame_buffer_set_get() {
        let mut fb = FrameBuffer::zeros(10, 10, 3);
        fb.set(5, 5, 0, 255);
        assert_eq!(fb.get(5, 5, 0), 255);
        assert_eq!(fb.get(5, 5, 1), 0);
    }

    #[test]
    fn test_gray_image_zeros() {
        let g = GrayImage::zeros(10, 20);
        assert_eq!(g.dim(), (10, 20));
        assert_eq!(g.get(0, 0), 0);
    }

    #[test]
    fn test_gray_image_set_get() {
        let mut g = GrayImage::zeros(10, 10);
        g.set(3, 7, 200);
        assert_eq!(g.get(3, 7), 200);
    }

    #[test]
    fn test_float_image_zeros() {
        let f = FloatImage::zeros(10, 20);
        assert_eq!(f.dim(), (10, 20));
        assert!((f.get(0, 0) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_to_grayscale() {
        let fb = FrameBuffer::from_elem(10, 10, 3, 128);
        let gray = to_grayscale(&fb);
        assert_eq!(gray.dim(), (10, 10));
        // 128 * (0.299 + 0.587 + 0.114) = 128
        assert!((gray.get(5, 5) as i32 - 128).abs() <= 1);
    }

    #[test]
    fn test_detect_edges_uniform() {
        let gray = GrayImage::zeros(20, 20);
        let edges = detect_edges(&gray);
        // Uniform image should have zero edges
        assert_eq!(edges.get(10, 10), 0);
    }

    #[test]
    fn test_frame_buffer_fill() {
        let mut fb = FrameBuffer::zeros(5, 5, 3);
        fb.fill(255);
        assert_eq!(fb.get(0, 0, 0), 255);
        assert_eq!(fb.get(4, 4, 2), 255);
    }
}
