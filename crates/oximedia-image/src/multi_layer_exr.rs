//! Multi-layer/multi-part OpenEXR support for compositing workflows.
//!
//! Provides reading and writing of individual layers within an EXR image,
//! enabling compositing workflows where separate render passes (diffuse,
//! specular, depth, normals, etc.) live in a single file.
//!
//! # Concepts
//!
//! - **Layer**: A named collection of channels (e.g. "beauty", "depth").
//! - **Channel**: A named data series within a layer (e.g. "R", "G", "B", "A", "Z").
//! - **Part**: In multi-part EXR, each part is an independent image with its own
//!   header, data window, and compression. Parts map 1:1 to layers in this API.
//!
//! # Data model
//!
//! ```text
//! ExrDocument
//!  └── parts: Vec<ExrPart>
//!       └── name, data_window, display_window, channels, pixel_data
//! ```

#![allow(dead_code)]

use crate::error::{ImageError, ImageResult};

// ── Channel descriptor ──────────────────────────────────────────────────────

/// Data type for an EXR channel sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExrChannelType {
    /// 16-bit IEEE 754 half-float.
    Half,
    /// 32-bit IEEE 754 single-precision float.
    Float,
    /// 32-bit unsigned integer.
    Uint,
}

impl ExrChannelType {
    /// Byte size per sample.
    #[must_use]
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::Half => 2,
            Self::Float | Self::Uint => 4,
        }
    }

    /// Returns the type code used in the EXR header.
    #[must_use]
    pub const fn type_code(self) -> u32 {
        match self {
            Self::Uint => 0,
            Self::Half => 1,
            Self::Float => 2,
        }
    }

    /// Construct from a type code.
    pub fn from_type_code(code: u32) -> ImageResult<Self> {
        match code {
            0 => Ok(Self::Uint),
            1 => Ok(Self::Half),
            2 => Ok(Self::Float),
            _ => Err(ImageError::invalid_format(format!(
                "Unknown EXR channel type code {code}"
            ))),
        }
    }
}

/// A single channel descriptor within a layer.
#[derive(Debug, Clone, PartialEq)]
pub struct ExrChannel {
    /// Channel name (e.g. "R", "G", "B", "A", "Z", "N.x").
    pub name: String,
    /// Sample data type.
    pub channel_type: ExrChannelType,
    /// Sub-sampling factor in X (1 = full resolution).
    pub x_sampling: u32,
    /// Sub-sampling factor in Y.
    pub y_sampling: u32,
    /// Whether the channel is stored linearly (non-perceptual).
    pub linear: bool,
}

impl ExrChannel {
    /// Create a new channel descriptor with default sampling (1, 1, linear).
    #[must_use]
    pub fn new(name: impl Into<String>, channel_type: ExrChannelType) -> Self {
        Self {
            name: name.into(),
            channel_type,
            x_sampling: 1,
            y_sampling: 1,
            linear: true,
        }
    }

    /// Builder: set sub-sampling.
    #[must_use]
    pub fn with_sampling(mut self, x: u32, y: u32) -> Self {
        self.x_sampling = x;
        self.y_sampling = y;
        self
    }
}

// ── Rectangular region ──────────────────────────────────────────────────────

/// A 2D axis-aligned integer rectangle (data or display window).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExrRect {
    /// Minimum X (inclusive).
    pub x_min: i32,
    /// Minimum Y (inclusive).
    pub y_min: i32,
    /// Maximum X (inclusive).
    pub x_max: i32,
    /// Maximum Y (inclusive).
    pub y_max: i32,
}

impl ExrRect {
    /// Width of the rectangle (inclusive).
    #[must_use]
    pub fn width(&self) -> u32 {
        if self.x_max < self.x_min {
            0
        } else {
            (self.x_max - self.x_min + 1) as u32
        }
    }

    /// Height of the rectangle (inclusive).
    #[must_use]
    pub fn height(&self) -> u32 {
        if self.y_max < self.y_min {
            0
        } else {
            (self.y_max - self.y_min + 1) as u32
        }
    }

    /// Total pixel count.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.width() as usize * self.height() as usize
    }

    /// Whether this rectangle is empty (zero or negative area).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.x_max < self.x_min || self.y_max < self.y_min
    }

    /// Compute the intersection of two rectangles.
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self {
        Self {
            x_min: self.x_min.max(other.x_min),
            y_min: self.y_min.max(other.y_min),
            x_max: self.x_max.min(other.x_max),
            y_max: self.y_max.min(other.y_max),
        }
    }

    /// Compute the bounding union of two rectangles.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Self {
            x_min: self.x_min.min(other.x_min),
            y_min: self.y_min.min(other.y_min),
            x_max: self.x_max.max(other.x_max),
            y_max: self.y_max.max(other.y_max),
        }
    }
}

// ── Compression ─────────────────────────────────────────────────────────────

/// EXR compression method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExrCompression {
    /// No compression.
    None,
    /// Run-length encoding.
    Rle,
    /// Zip (1 scanline).
    ZipSingle,
    /// Zip (16 scanlines).
    Zip,
    /// PIZ wavelet compression.
    Piz,
    /// PXR24 (lossy 24-bit float).
    Pxr24,
    /// B44 (lossy, fixed ratio).
    B44,
    /// B44A (lossy, allows flat areas to be lossless).
    B44a,
    /// DWAA (lossy, variable block).
    Dwaa,
    /// DWAB (lossy, larger blocks).
    Dwab,
}

impl ExrCompression {
    /// Number of scanlines per block for this compression.
    #[must_use]
    pub const fn scanlines_per_block(self) -> u32 {
        match self {
            Self::None | Self::Rle | Self::ZipSingle => 1,
            Self::Zip => 16,
            Self::Piz => 32,
            Self::Pxr24 => 16,
            Self::B44 | Self::B44a => 32,
            Self::Dwaa => 32,
            Self::Dwab => 256,
        }
    }

    /// Encode the compression as the EXR header byte.
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Rle => 1,
            Self::ZipSingle => 2,
            Self::Zip => 3,
            Self::Piz => 4,
            Self::Pxr24 => 5,
            Self::B44 => 6,
            Self::B44a => 7,
            Self::Dwaa => 8,
            Self::Dwab => 9,
        }
    }

    /// Parse from the EXR header byte.
    pub fn from_code(code: u8) -> ImageResult<Self> {
        match code {
            0 => Ok(Self::None),
            1 => Ok(Self::Rle),
            2 => Ok(Self::ZipSingle),
            3 => Ok(Self::Zip),
            4 => Ok(Self::Piz),
            5 => Ok(Self::Pxr24),
            6 => Ok(Self::B44),
            7 => Ok(Self::B44a),
            8 => Ok(Self::Dwaa),
            9 => Ok(Self::Dwab),
            _ => Err(ImageError::invalid_format(format!(
                "Unknown EXR compression code {code}"
            ))),
        }
    }
}

// ── Part (layer) ────────────────────────────────────────────────────────────

/// A single part (layer) in a multi-part EXR document.
#[derive(Debug, Clone)]
pub struct ExrPart {
    /// Part / layer name.
    pub name: String,
    /// Channel descriptors (sorted by name for EXR spec compliance).
    pub channels: Vec<ExrChannel>,
    /// Data window — the bounding box of actual pixel data.
    pub data_window: ExrRect,
    /// Display window — the intended viewing rectangle.
    pub display_window: ExrRect,
    /// Compression method for this part.
    pub compression: ExrCompression,
    /// Pixel aspect ratio (width / height), typically 1.0.
    pub pixel_aspect_ratio: f32,
    /// Raw pixel data per channel, stored as `f32` for processing uniformity.
    /// Outer vec is per-channel (same order as `channels`), inner is row-major.
    pub pixel_data: Vec<Vec<f32>>,
}

impl ExrPart {
    /// Create a new empty part with the given name and window.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        channels: Vec<ExrChannel>,
        data_window: ExrRect,
        display_window: ExrRect,
    ) -> Self {
        let pixel_count = data_window.pixel_count();
        let pixel_data = channels.iter().map(|_| vec![0.0f32; pixel_count]).collect();
        Self {
            name: name.into(),
            channels,
            data_window,
            display_window,
            compression: ExrCompression::Zip,
            pixel_aspect_ratio: 1.0,
            pixel_data,
        }
    }

    /// Width of the data window.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.data_window.width()
    }

    /// Height of the data window.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.data_window.height()
    }

    /// Find a channel index by name.
    #[must_use]
    pub fn channel_index(&self, name: &str) -> Option<usize> {
        self.channels.iter().position(|c| c.name == name)
    }

    /// Get pixel data for a named channel.
    pub fn channel_data(&self, name: &str) -> ImageResult<&[f32]> {
        let idx = self
            .channel_index(name)
            .ok_or_else(|| ImageError::invalid_format(format!("Channel '{name}' not found")))?;
        Ok(&self.pixel_data[idx])
    }

    /// Get mutable pixel data for a named channel.
    pub fn channel_data_mut(&mut self, name: &str) -> ImageResult<&mut [f32]> {
        let idx = self
            .channel_index(name)
            .ok_or_else(|| ImageError::invalid_format(format!("Channel '{name}' not found")))?;
        Ok(&mut self.pixel_data[idx])
    }

    /// Set all pixels in a channel to a constant value.
    pub fn fill_channel(&mut self, name: &str, value: f32) -> ImageResult<()> {
        let data = self.channel_data_mut(name)?;
        for p in data.iter_mut() {
            *p = value;
        }
        Ok(())
    }

    /// Get a pixel value at (x, y) for a named channel.
    pub fn get_pixel(&self, channel: &str, x: u32, y: u32) -> ImageResult<f32> {
        let w = self.width();
        let h = self.height();
        if x >= w || y >= h {
            return Err(ImageError::InvalidDimensions(x, y));
        }
        let data = self.channel_data(channel)?;
        Ok(data[(y as usize) * (w as usize) + (x as usize)])
    }

    /// Set a pixel value at (x, y) for a named channel.
    pub fn set_pixel(&mut self, channel: &str, x: u32, y: u32, value: f32) -> ImageResult<()> {
        let w = self.width();
        let h = self.height();
        if x >= w || y >= h {
            return Err(ImageError::InvalidDimensions(x, y));
        }
        let data = self.channel_data_mut(channel)?;
        data[(y as usize) * (w as usize) + (x as usize)] = value;
        Ok(())
    }

    /// Validate internal consistency.
    pub fn validate(&self) -> ImageResult<()> {
        let expected = self.data_window.pixel_count();
        for (i, ch) in self.channels.iter().enumerate() {
            if i >= self.pixel_data.len() {
                return Err(ImageError::invalid_format(format!(
                    "Missing pixel data for channel '{}'",
                    ch.name
                )));
            }
            if self.pixel_data[i].len() != expected {
                return Err(ImageError::invalid_format(format!(
                    "Channel '{}' has {} samples, expected {expected}",
                    ch.name,
                    self.pixel_data[i].len()
                )));
            }
        }
        Ok(())
    }
}

// ── Document ────────────────────────────────────────────────────────────────

/// A multi-part EXR document containing one or more layers.
#[derive(Debug, Clone)]
pub struct ExrDocument {
    /// Parts / layers.
    pub parts: Vec<ExrPart>,
}

impl ExrDocument {
    /// Create a new empty document.
    #[must_use]
    pub fn new() -> Self {
        Self { parts: Vec::new() }
    }

    /// Add a part (layer) to the document.
    pub fn add_part(&mut self, part: ExrPart) {
        self.parts.push(part);
    }

    /// Number of parts.
    #[must_use]
    pub fn part_count(&self) -> usize {
        self.parts.len()
    }

    /// Find a part by name.
    #[must_use]
    pub fn find_part(&self, name: &str) -> Option<&ExrPart> {
        self.parts.iter().find(|p| p.name == name)
    }

    /// Find a mutable part by name.
    #[must_use]
    pub fn find_part_mut(&mut self, name: &str) -> Option<&mut ExrPart> {
        self.parts.iter_mut().find(|p| p.name == name)
    }

    /// List all part names.
    #[must_use]
    pub fn part_names(&self) -> Vec<&str> {
        self.parts.iter().map(|p| p.name.as_str()).collect()
    }

    /// Compute the overall display window (union of all parts).
    pub fn overall_display_window(&self) -> ImageResult<ExrRect> {
        let first = self
            .parts
            .first()
            .ok_or_else(|| ImageError::invalid_format("No parts in document"))?;
        let mut rect = first.display_window;
        for part in &self.parts[1..] {
            rect = rect.union(&part.display_window);
        }
        Ok(rect)
    }

    /// Validate all parts.
    pub fn validate(&self) -> ImageResult<()> {
        if self.parts.is_empty() {
            return Err(ImageError::invalid_format("Document has no parts"));
        }
        for part in &self.parts {
            part.validate()?;
        }
        // Check for duplicate part names
        let mut names: Vec<&str> = self.parts.iter().map(|p| p.name.as_str()).collect();
        names.sort();
        for pair in names.windows(2) {
            if pair[0] == pair[1] {
                return Err(ImageError::invalid_format(format!(
                    "Duplicate part name: '{}'",
                    pair[0]
                )));
            }
        }
        Ok(())
    }

    /// Merge a specific channel from one part into another part, creating it
    /// if it doesn't exist. This is the primary compositing operation.
    pub fn merge_channel(
        &mut self,
        src_part: &str,
        src_channel: &str,
        dst_part: &str,
        dst_channel: &str,
        alpha: f32,
    ) -> ImageResult<()> {
        // Read source data
        let src = self.find_part(src_part).ok_or_else(|| {
            ImageError::invalid_format(format!("Source part '{src_part}' not found"))
        })?;
        let src_data: Vec<f32> = src.channel_data(src_channel)?.to_vec();
        let src_window = src.data_window;

        // Get or validate destination
        let dst = self.find_part_mut(dst_part).ok_or_else(|| {
            ImageError::invalid_format(format!("Destination part '{dst_part}' not found"))
        })?;

        if dst.data_window != src_window {
            return Err(ImageError::invalid_format(
                "Source and destination data windows must match for merge",
            ));
        }

        // Ensure destination channel exists
        if dst.channel_index(dst_channel).is_none() {
            dst.channels
                .push(ExrChannel::new(dst_channel, ExrChannelType::Float));
            dst.pixel_data
                .push(vec![0.0f32; dst.data_window.pixel_count()]);
        }

        let dst_data = dst.channel_data_mut(dst_channel)?;
        let clamped_alpha = alpha.clamp(0.0, 1.0);
        for (d, s) in dst_data.iter_mut().zip(src_data.iter()) {
            *d = *d * (1.0 - clamped_alpha) + *s * clamped_alpha;
        }

        Ok(())
    }
}

impl Default for ExrDocument {
    fn default() -> Self {
        Self::new()
    }
}

// ── Layer builder helpers ───────────────────────────────────────────────────

/// Helper to create common EXR layer configurations.
pub struct LayerBuilder;

impl LayerBuilder {
    /// Create an RGBA beauty pass layer.
    #[must_use]
    pub fn rgba(name: &str, width: u32, height: u32) -> ExrPart {
        let channels = vec![
            ExrChannel::new("R", ExrChannelType::Half),
            ExrChannel::new("G", ExrChannelType::Half),
            ExrChannel::new("B", ExrChannelType::Half),
            ExrChannel::new("A", ExrChannelType::Half),
        ];
        let window = ExrRect {
            x_min: 0,
            y_min: 0,
            x_max: width as i32 - 1,
            y_max: height as i32 - 1,
        };
        ExrPart::new(name, channels, window, window)
    }

    /// Create a depth (Z) layer.
    #[must_use]
    pub fn depth(name: &str, width: u32, height: u32) -> ExrPart {
        let channels = vec![ExrChannel::new("Z", ExrChannelType::Float)];
        let window = ExrRect {
            x_min: 0,
            y_min: 0,
            x_max: width as i32 - 1,
            y_max: height as i32 - 1,
        };
        ExrPart::new(name, channels, window, window)
    }

    /// Create a normal map layer (N.x, N.y, N.z).
    #[must_use]
    pub fn normals(name: &str, width: u32, height: u32) -> ExrPart {
        let channels = vec![
            ExrChannel::new("N.x", ExrChannelType::Half),
            ExrChannel::new("N.y", ExrChannelType::Half),
            ExrChannel::new("N.z", ExrChannelType::Half),
        ];
        let window = ExrRect {
            x_min: 0,
            y_min: 0,
            x_max: width as i32 - 1,
            y_max: height as i32 - 1,
        };
        ExrPart::new(name, channels, window, window)
    }

    /// Create a custom single-channel layer (e.g., ambient occlusion, ID matte).
    #[must_use]
    pub fn single_channel(
        name: &str,
        channel_name: &str,
        channel_type: ExrChannelType,
        width: u32,
        height: u32,
    ) -> ExrPart {
        let channels = vec![ExrChannel::new(channel_name, channel_type)];
        let window = ExrRect {
            x_min: 0,
            y_min: 0,
            x_max: width as i32 - 1,
            y_max: height as i32 - 1,
        };
        ExrPart::new(name, channels, window, window)
    }
}

// ── Serialisation helpers (binary) ──────────────────────────────────────────

/// Serialize channel pixel data as uncompressed scanlines (f32 → bytes).
pub fn channel_data_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for &v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Deserialize channel pixel data from uncompressed bytes (LE f32).
pub fn bytes_to_channel_data(bytes: &[u8]) -> ImageResult<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(ImageError::invalid_format(
            "Channel data length is not a multiple of 4",
        ));
    }
    let count = bytes.len() / 4;
    let mut data = Vec::with_capacity(count);
    for chunk in bytes.chunks_exact(4) {
        let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        data.push(f32::from_le_bytes(arr));
    }
    Ok(data)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_window(w: u32, h: u32) -> ExrRect {
        ExrRect {
            x_min: 0,
            y_min: 0,
            x_max: w as i32 - 1,
            y_max: h as i32 - 1,
        }
    }

    #[test]
    fn test_exr_rect_dimensions() {
        let r = ExrRect {
            x_min: 0,
            y_min: 0,
            x_max: 99,
            y_max: 49,
        };
        assert_eq!(r.width(), 100);
        assert_eq!(r.height(), 50);
        assert_eq!(r.pixel_count(), 5000);
        assert!(!r.is_empty());
    }

    #[test]
    fn test_exr_rect_empty() {
        let r = ExrRect {
            x_min: 10,
            y_min: 10,
            x_max: 5,
            y_max: 5,
        };
        assert!(r.is_empty());
        assert_eq!(r.width(), 0);
    }

    #[test]
    fn test_exr_rect_intersection() {
        let a = ExrRect {
            x_min: 0,
            y_min: 0,
            x_max: 100,
            y_max: 100,
        };
        let b = ExrRect {
            x_min: 50,
            y_min: 50,
            x_max: 150,
            y_max: 150,
        };
        let c = a.intersect(&b);
        assert_eq!(c.x_min, 50);
        assert_eq!(c.y_min, 50);
        assert_eq!(c.x_max, 100);
        assert_eq!(c.y_max, 100);
        assert_eq!(c.width(), 51);
        assert_eq!(c.height(), 51);
    }

    #[test]
    fn test_exr_rect_union() {
        let a = ExrRect {
            x_min: 10,
            y_min: 20,
            x_max: 50,
            y_max: 60,
        };
        let b = ExrRect {
            x_min: 30,
            y_min: 40,
            x_max: 80,
            y_max: 90,
        };
        let u = a.union(&b);
        assert_eq!(u.x_min, 10);
        assert_eq!(u.y_min, 20);
        assert_eq!(u.x_max, 80);
        assert_eq!(u.y_max, 90);
    }

    #[test]
    fn test_channel_type_round_trip() {
        for code in 0..=2u32 {
            let ct = ExrChannelType::from_type_code(code).expect("valid code");
            assert_eq!(ct.type_code(), code);
        }
        assert!(ExrChannelType::from_type_code(99).is_err());
    }

    #[test]
    fn test_compression_round_trip() {
        for code in 0..=9u8 {
            let comp = ExrCompression::from_code(code).expect("valid code");
            assert_eq!(comp.code(), code);
        }
        assert!(ExrCompression::from_code(42).is_err());
    }

    #[test]
    fn test_part_creation_and_pixel_access() {
        let mut part = LayerBuilder::rgba("beauty", 4, 4);
        assert_eq!(part.width(), 4);
        assert_eq!(part.height(), 4);
        assert_eq!(part.channels.len(), 4);
        assert_eq!(part.pixel_data.len(), 4);
        assert_eq!(part.pixel_data[0].len(), 16);

        part.set_pixel("R", 2, 3, 0.75).expect("set pixel");
        let val = part.get_pixel("R", 2, 3).expect("get pixel");
        assert!((val - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_part_fill_channel() {
        let mut part = LayerBuilder::depth("depth", 8, 8);
        part.fill_channel("Z", 1000.0).expect("fill");
        let data = part.channel_data("Z").expect("get");
        for &v in data {
            assert!((v - 1000.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_part_validate_ok() {
        let part = LayerBuilder::normals("normals", 10, 10);
        part.validate().expect("should be valid");
    }

    #[test]
    fn test_part_validate_bad_length() {
        let mut part = LayerBuilder::rgba("bad", 4, 4);
        part.pixel_data[1] = vec![0.0; 10]; // wrong length
        assert!(part.validate().is_err());
    }

    #[test]
    fn test_document_creation() {
        let mut doc = ExrDocument::new();
        doc.add_part(LayerBuilder::rgba("beauty", 100, 100));
        doc.add_part(LayerBuilder::depth("depth", 100, 100));
        doc.add_part(LayerBuilder::normals("normals", 100, 100));

        assert_eq!(doc.part_count(), 3);
        assert_eq!(doc.part_names(), vec!["beauty", "depth", "normals"]);
        assert!(doc.find_part("beauty").is_some());
        assert!(doc.find_part("missing").is_none());
    }

    #[test]
    fn test_document_validate_duplicate_names() {
        let mut doc = ExrDocument::new();
        doc.add_part(LayerBuilder::rgba("beauty", 10, 10));
        doc.add_part(LayerBuilder::rgba("beauty", 10, 10));
        assert!(doc.validate().is_err());
    }

    #[test]
    fn test_document_validate_empty() {
        let doc = ExrDocument::new();
        assert!(doc.validate().is_err());
    }

    #[test]
    fn test_overall_display_window() {
        let mut doc = ExrDocument::new();
        let mut p1 = LayerBuilder::rgba("a", 100, 100);
        p1.display_window = ExrRect {
            x_min: 0,
            y_min: 0,
            x_max: 99,
            y_max: 99,
        };
        let mut p2 = LayerBuilder::rgba("b", 50, 50);
        p2.display_window = ExrRect {
            x_min: 50,
            y_min: 50,
            x_max: 149,
            y_max: 149,
        };
        doc.add_part(p1);
        doc.add_part(p2);

        let overall = doc.overall_display_window().expect("ok");
        assert_eq!(overall.x_min, 0);
        assert_eq!(overall.y_min, 0);
        assert_eq!(overall.x_max, 149);
        assert_eq!(overall.y_max, 149);
    }

    #[test]
    fn test_merge_channel() {
        let mut doc = ExrDocument::new();
        let mut src = LayerBuilder::single_channel("src", "V", ExrChannelType::Float, 4, 4);
        src.fill_channel("V", 1.0).expect("fill");
        let dst = LayerBuilder::single_channel("dst", "V", ExrChannelType::Float, 4, 4);
        doc.add_part(src);
        doc.add_part(dst);

        doc.merge_channel("src", "V", "dst", "V", 0.5)
            .expect("merge");
        let dst_part = doc.find_part("dst").expect("find");
        let data = dst_part.channel_data("V").expect("get");
        for &v in data {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_merge_creates_missing_channel() {
        let mut doc = ExrDocument::new();
        let mut src = LayerBuilder::single_channel("src", "AO", ExrChannelType::Float, 2, 2);
        src.fill_channel("AO", 0.8).expect("fill");
        let dst = LayerBuilder::rgba("dst", 2, 2);
        doc.add_part(src);
        doc.add_part(dst);

        doc.merge_channel("src", "AO", "dst", "AO", 1.0)
            .expect("merge");
        let dst_part = doc.find_part("dst").expect("find");
        assert!(dst_part.channel_index("AO").is_some());
        let data = dst_part.channel_data("AO").expect("get");
        for &v in data {
            assert!((v - 0.8).abs() < 1e-6);
        }
    }

    #[test]
    fn test_serialization_round_trip() {
        let original: Vec<f32> = vec![1.0, -0.5, 0.0, 3.14159, f32::MAX, f32::MIN_POSITIVE];
        let bytes = channel_data_to_bytes(&original);
        let decoded = bytes_to_channel_data(&bytes).expect("decode");
        assert_eq!(original.len(), decoded.len());
        for (a, b) in original.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_serialization_bad_length() {
        assert!(bytes_to_channel_data(&[1, 2, 3]).is_err());
    }

    #[test]
    fn test_channel_subsampling() {
        let ch = ExrChannel::new("chroma", ExrChannelType::Half).with_sampling(2, 2);
        assert_eq!(ch.x_sampling, 2);
        assert_eq!(ch.y_sampling, 2);
        assert!(ch.linear);
    }

    #[test]
    fn test_layer_builder_normals_channels() {
        let part = LayerBuilder::normals("nrm", 16, 16);
        assert_eq!(part.channels.len(), 3);
        assert!(part.channel_index("N.x").is_some());
        assert!(part.channel_index("N.y").is_some());
        assert!(part.channel_index("N.z").is_some());
    }

    #[test]
    fn test_pixel_out_of_bounds() {
        let part = LayerBuilder::depth("d", 4, 4);
        assert!(part.get_pixel("Z", 4, 0).is_err());
        assert!(part.get_pixel("Z", 0, 4).is_err());
    }

    #[test]
    fn test_channel_not_found() {
        let part = LayerBuilder::depth("d", 4, 4);
        assert!(part.channel_data("nonexistent").is_err());
    }
}
