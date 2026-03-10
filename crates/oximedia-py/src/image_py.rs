//! Python bindings for professional image I/O and processing.
//!
//! Wraps `oximedia-image` for DPX/EXR/TIFF image frame access, image
//! sequences, and basic pixel-level filtering operations.

use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::HashMap;

use oximedia_image::format_detect::FormatDetector;
use oximedia_image::histogram_ops::{Histogram, RgbHistogram};
use oximedia_image::{ColorSpace, ImageData, ImageFrame, PixelType, SequencePattern};

// ---------------------------------------------------------------------------
// PyImageFrame
// ---------------------------------------------------------------------------

/// An in-memory image frame with pixel data and metadata.
#[pyclass]
#[derive(Clone)]
pub struct PyImageFrame {
    /// Image width in pixels.
    #[pyo3(get)]
    pub width: u32,
    /// Image height in pixels.
    #[pyo3(get)]
    pub height: u32,
    /// Number of color channels.
    #[pyo3(get)]
    pub channels: u32,
    /// Bits per component.
    #[pyo3(get)]
    pub bit_depth: u32,
    /// Color space name.
    #[pyo3(get)]
    pub colorspace: String,
    /// Raw pixel data (interleaved).
    data: Vec<u8>,
}

#[pymethods]
impl PyImageFrame {
    /// Create a new image frame from raw pixel data.
    #[new]
    #[pyo3(signature = (width, height, channels, bit_depth, data, colorspace = None))]
    fn new(
        width: u32,
        height: u32,
        channels: u32,
        bit_depth: u32,
        data: Vec<u8>,
        colorspace: Option<&str>,
    ) -> PyResult<Self> {
        let bytes_per_component = ((bit_depth + 7) / 8) as usize;
        let expected =
            (width as usize) * (height as usize) * (channels as usize) * bytes_per_component;

        if data.len() < expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Data too small: need {} bytes for {}x{}x{} @ {}-bit, got {}",
                expected,
                width,
                height,
                channels,
                bit_depth,
                data.len()
            )));
        }

        Ok(Self {
            width,
            height,
            channels,
            bit_depth,
            colorspace: colorspace.unwrap_or("srgb").to_string(),
            data,
        })
    }

    /// Return pixel data as bytes.
    fn data_as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Create a frame from raw pixel data (NumPy-style flat buffer).
    #[classmethod]
    #[pyo3(signature = (data, width, height, channels, bit_depth = None))]
    fn from_numpy(
        _cls: &Bound<'_, PyType>,
        data: Vec<u8>,
        width: u32,
        height: u32,
        channels: u32,
        bit_depth: Option<u32>,
    ) -> PyResult<Self> {
        let depth = bit_depth.unwrap_or(8);
        Self::new(width, height, channels, depth, data, None)
    }

    /// Get pixel value at (x, y) as a list of f64 per channel.
    fn pixel_at(&self, x: u32, y: u32) -> PyResult<Vec<f64>> {
        if x >= self.width || y >= self.height {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Pixel ({}, {}) out of bounds ({}x{})",
                x, y, self.width, self.height
            )));
        }

        let bytes_per_comp = ((self.bit_depth + 7) / 8) as usize;
        let pixel_offset = ((y as usize) * (self.width as usize) + (x as usize))
            * (self.channels as usize)
            * bytes_per_comp;

        let mut values = Vec::with_capacity(self.channels as usize);
        for ch in 0..self.channels as usize {
            let offset = pixel_offset + ch * bytes_per_comp;
            let val = match bytes_per_comp {
                1 => {
                    let v = self.data.get(offset).copied().ok_or_else(|| {
                        pyo3::exceptions::PyIndexError::new_err("Pixel data out of bounds")
                    })?;
                    f64::from(v)
                }
                2 => {
                    if offset + 1 >= self.data.len() {
                        return Err(pyo3::exceptions::PyIndexError::new_err(
                            "Pixel data out of bounds",
                        ));
                    }
                    let v = u16::from_le_bytes([self.data[offset], self.data[offset + 1]]);
                    f64::from(v)
                }
                4 => {
                    if offset + 3 >= self.data.len() {
                        return Err(pyo3::exceptions::PyIndexError::new_err(
                            "Pixel data out of bounds",
                        ));
                    }
                    let bytes = [
                        self.data[offset],
                        self.data[offset + 1],
                        self.data[offset + 2],
                        self.data[offset + 3],
                    ];
                    f64::from(f32::from_le_bytes(bytes))
                }
                _ => 0.0,
            };
            values.push(val);
        }

        Ok(values)
    }

    /// Total number of pixels.
    fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Stride (bytes per row).
    fn stride(&self) -> usize {
        let bytes_per_comp = ((self.bit_depth + 7) / 8) as usize;
        (self.width as usize) * (self.channels as usize) * bytes_per_comp
    }

    fn __repr__(&self) -> String {
        format!(
            "PyImageFrame({}x{}, {}ch, {}-bit, colorspace='{}')",
            self.width, self.height, self.channels, self.bit_depth, self.colorspace
        )
    }
}

// ---------------------------------------------------------------------------
// PyImageSequence
// ---------------------------------------------------------------------------

/// Represents an image file sequence (e.g. DPX or EXR frames).
#[pyclass]
#[derive(Clone)]
pub struct PyImageSequence {
    /// Pattern string used to create the sequence.
    #[pyo3(get)]
    pub pattern: String,
    /// Total number of frames in the range.
    #[pyo3(get)]
    pub frame_count: u32,
    /// First frame number.
    #[pyo3(get)]
    pub start_frame: u32,
    /// Last frame number.
    #[pyo3(get)]
    pub end_frame: u32,
    /// Whether the sequence has missing frames.
    #[pyo3(get)]
    pub has_gaps: bool,
    /// Internal pattern for path generation.
    inner_pattern: Option<SequencePattern>,
    /// List of missing frame numbers.
    gaps: Vec<u32>,
}

#[pymethods]
impl PyImageSequence {
    /// Create a sequence from a pattern and frame range.
    #[classmethod]
    fn from_pattern(
        _cls: &Bound<'_, PyType>,
        pattern: &str,
        start: u32,
        end: u32,
    ) -> PyResult<Self> {
        if start > end {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Start frame ({}) must be <= end frame ({})",
                start, end
            )));
        }

        let seq_pattern = SequencePattern::parse(pattern).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid pattern '{}': {}", pattern, e))
        })?;

        let seq = oximedia_image::ImageSequence::from_pattern(seq_pattern.clone(), start..=end)
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Sequence error: {}", e))
            })?;

        Ok(Self {
            pattern: pattern.to_string(),
            frame_count: end - start + 1,
            start_frame: start,
            end_frame: end,
            has_gaps: !seq.gaps.is_empty(),
            inner_pattern: Some(seq_pattern),
            gaps: seq.gaps,
        })
    }

    /// Detect an existing image sequence from a directory.
    #[classmethod]
    fn detect(_cls: &Bound<'_, PyType>, directory: &str, pattern: &str) -> PyResult<Self> {
        let seq_pattern = SequencePattern::parse(pattern).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid pattern '{}': {}", pattern, e))
        })?;

        let dir_path = std::path::Path::new(directory);
        let seq =
            oximedia_image::ImageSequence::detect(dir_path, seq_pattern.clone()).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Detection failed: {}", e))
            })?;

        let start = *seq.range.start();
        let end = *seq.range.end();

        Ok(Self {
            pattern: pattern.to_string(),
            frame_count: end - start + 1,
            start_frame: start,
            end_frame: end,
            has_gaps: !seq.gaps.is_empty(),
            inner_pattern: Some(seq_pattern),
            gaps: seq.gaps,
        })
    }

    /// Get the file path for a specific frame number.
    fn frame_path(&self, frame: u32) -> PyResult<String> {
        let pat = self.inner_pattern.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Sequence pattern not initialized")
        })?;

        if frame < self.start_frame || frame > self.end_frame {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Frame {} out of range [{}, {}]",
                frame, self.start_frame, self.end_frame
            )));
        }

        Ok(pat.format(frame).display().to_string())
    }

    /// Return list of missing frame numbers.
    fn missing_frames(&self) -> Vec<u32> {
        self.gaps.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyImageSequence(pattern='{}', frames={}-{}, count={}, gaps={})",
            self.pattern,
            self.start_frame,
            self.end_frame,
            self.frame_count,
            if self.has_gaps {
                self.gaps.len().to_string()
            } else {
                "none".to_string()
            }
        )
    }
}

// ---------------------------------------------------------------------------
// PyImageFilter
// ---------------------------------------------------------------------------

/// A composable image filter that can be applied to frames.
#[pyclass]
#[derive(Clone)]
pub struct PyImageFilter {
    filter_type: String,
    params: HashMap<String, f64>,
}

#[pymethods]
impl PyImageFilter {
    /// Create a blur filter.
    #[classmethod]
    fn blur(_cls: &Bound<'_, PyType>, radius: f64) -> PyResult<Self> {
        if radius < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Blur radius must be >= 0",
            ));
        }
        let mut params = HashMap::new();
        params.insert("radius".to_string(), radius);
        Ok(Self {
            filter_type: "blur".to_string(),
            params,
        })
    }

    /// Create a sharpen filter.
    #[classmethod]
    fn sharpen(_cls: &Bound<'_, PyType>, amount: f64) -> PyResult<Self> {
        if amount < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Sharpen amount must be >= 0",
            ));
        }
        let mut params = HashMap::new();
        params.insert("amount".to_string(), amount);
        Ok(Self {
            filter_type: "sharpen".to_string(),
            params,
        })
    }

    /// Create a brightness adjustment filter.
    #[classmethod]
    fn brightness(_cls: &Bound<'_, PyType>, amount: f64) -> PyResult<Self> {
        let mut params = HashMap::new();
        params.insert("amount".to_string(), amount);
        Ok(Self {
            filter_type: "brightness".to_string(),
            params,
        })
    }

    /// Create a contrast adjustment filter.
    #[classmethod]
    fn contrast(_cls: &Bound<'_, PyType>, amount: f64) -> PyResult<Self> {
        if amount < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Contrast must be >= 0",
            ));
        }
        let mut params = HashMap::new();
        params.insert("amount".to_string(), amount);
        Ok(Self {
            filter_type: "contrast".to_string(),
            params,
        })
    }

    /// Create a saturation adjustment filter.
    #[classmethod]
    fn saturation(_cls: &Bound<'_, PyType>, amount: f64) -> PyResult<Self> {
        if amount < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Saturation must be >= 0",
            ));
        }
        let mut params = HashMap::new();
        params.insert("amount".to_string(), amount);
        Ok(Self {
            filter_type: "saturation".to_string(),
            params,
        })
    }

    /// Create a gamma correction filter.
    #[classmethod]
    fn gamma(_cls: &Bound<'_, PyType>, value: f64) -> PyResult<Self> {
        if value <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("Gamma must be > 0"));
        }
        let mut params = HashMap::new();
        params.insert("value".to_string(), value);
        Ok(Self {
            filter_type: "gamma".to_string(),
            params,
        })
    }

    /// Apply this filter to an image frame, returning a new frame.
    fn apply(&self, frame: &PyImageFrame) -> PyResult<PyImageFrame> {
        // Only 8-bit single-channel or RGB for now
        if frame.bit_depth != 8 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Filter currently supports 8-bit images only, got {}-bit",
                frame.bit_depth
            )));
        }

        let mut output_data = frame.data.clone();
        let w = frame.width as usize;
        let h = frame.height as usize;

        match self.filter_type.as_str() {
            "blur" => {
                let radius = self.params.get("radius").copied().unwrap_or(1.0);
                let r = (radius as u32).max(1);
                // Use box blur from oximedia_image for grayscale, or do per-channel
                if frame.channels == 1 {
                    output_data = oximedia_image::filter::box_blur(&frame.data, w, h, r);
                } else {
                    // Apply per channel
                    let ch = frame.channels as usize;
                    let pixel_count = w * h;
                    for c in 0..ch {
                        let mut channel_data = vec![0u8; pixel_count];
                        for i in 0..pixel_count {
                            channel_data[i] = frame.data[i * ch + c];
                        }
                        let blurred = oximedia_image::filter::box_blur(&channel_data, w, h, r);
                        for i in 0..pixel_count {
                            output_data[i * ch + c] = blurred[i];
                        }
                    }
                }
            }
            "sharpen" => {
                let amount = self.params.get("amount").copied().unwrap_or(1.0);
                if frame.channels == 1 {
                    let sharpened = oximedia_image::filter::sharpen(&frame.data, w, h);
                    // Blend original and sharpened by amount
                    for i in 0..output_data.len() {
                        let orig = frame.data[i] as f64;
                        let sharp = sharpened[i] as f64;
                        let val = orig + (sharp - orig) * amount;
                        output_data[i] = val.clamp(0.0, 255.0) as u8;
                    }
                } else {
                    let ch = frame.channels as usize;
                    let pixel_count = w * h;
                    for c in 0..ch {
                        let mut channel_data = vec![0u8; pixel_count];
                        for i in 0..pixel_count {
                            channel_data[i] = frame.data[i * ch + c];
                        }
                        let sharpened = oximedia_image::filter::sharpen(&channel_data, w, h);
                        for i in 0..pixel_count {
                            let orig = frame.data[i * ch + c] as f64;
                            let sharp = sharpened[i] as f64;
                            let val = orig + (sharp - orig) * amount;
                            output_data[i * ch + c] = val.clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
            "brightness" => {
                let amount = self.params.get("amount").copied().unwrap_or(0.0);
                let offset = (amount * 255.0) as i32;
                for byte in &mut output_data {
                    let val = (*byte as i32) + offset;
                    *byte = val.clamp(0, 255) as u8;
                }
            }
            "contrast" => {
                let factor = self.params.get("amount").copied().unwrap_or(1.0);
                for byte in &mut output_data {
                    let val = (((*byte as f64) - 128.0) * factor + 128.0).clamp(0.0, 255.0);
                    *byte = val as u8;
                }
            }
            "saturation" => {
                let factor = self.params.get("amount").copied().unwrap_or(1.0);
                if frame.channels >= 3 {
                    let ch = frame.channels as usize;
                    let pixel_count = w * h;
                    for i in 0..pixel_count {
                        let base = i * ch;
                        let r = output_data[base] as f64;
                        let g = output_data[base + 1] as f64;
                        let b = output_data[base + 2] as f64;
                        // BT.601 luma
                        let luma = 0.299 * r + 0.587 * g + 0.114 * b;
                        output_data[base] = (luma + (r - luma) * factor).clamp(0.0, 255.0) as u8;
                        output_data[base + 1] =
                            (luma + (g - luma) * factor).clamp(0.0, 255.0) as u8;
                        output_data[base + 2] =
                            (luma + (b - luma) * factor).clamp(0.0, 255.0) as u8;
                    }
                }
                // For single channel, saturation has no effect
            }
            "gamma" => {
                let gamma_val = self.params.get("value").copied().unwrap_or(1.0);
                if gamma_val > 0.0 {
                    let inv_gamma = 1.0 / gamma_val;
                    for byte in &mut output_data {
                        let normalized = (*byte as f64) / 255.0;
                        let corrected = normalized.powf(inv_gamma) * 255.0;
                        *byte = corrected.clamp(0.0, 255.0) as u8;
                    }
                }
            }
            other => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Unknown filter type: {}",
                    other
                )));
            }
        }

        Ok(PyImageFrame {
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            bit_depth: frame.bit_depth,
            colorspace: frame.colorspace.clone(),
            data: output_data,
        })
    }

    fn __repr__(&self) -> String {
        let params_str: Vec<String> = self
            .params
            .iter()
            .map(|(k, v)| format!("{}={:.3}", k, v))
            .collect();
        format!(
            "PyImageFilter(type='{}', {})",
            self.filter_type,
            params_str.join(", ")
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Read an image from disk and return a PyImageFrame.
#[pyfunction]
pub fn read_image(path: &str) -> PyResult<PyImageFrame> {
    let file_path = std::path::Path::new(path);
    if !file_path.exists() {
        return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "File not found: {}",
            path
        )));
    }

    // Read header to detect format
    let mut header_buf = vec![0u8; 2048];
    let mut f = std::fs::File::open(file_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to open file: {}", e)))?;
    let bytes_read = std::io::Read::read(&mut f, &mut header_buf)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to read file: {}", e)))?;
    header_buf.truncate(bytes_read);

    let fmt = FormatDetector::detect(&header_buf);

    // Try format-specific readers
    let frame: ImageFrame = match fmt {
        oximedia_image::format_detect::ImageFormat::Dpx => {
            oximedia_image::dpx::read_dpx(file_path, 1).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("DPX read error: {}", e))
            })?
        }
        oximedia_image::format_detect::ImageFormat::Exr => {
            oximedia_image::exr::read_exr(file_path, 1).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("EXR read error: {}", e))
            })?
        }
        oximedia_image::format_detect::ImageFormat::Tiff => {
            oximedia_image::tiff::read_tiff(file_path, 1).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("TIFF read error: {}", e))
            })?
        }
        other => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Unsupported image format: {} (supported: DPX, EXR, TIFF)",
                other.name()
            )));
        }
    };

    let data_bytes = match &frame.data {
        ImageData::Interleaved(bytes) => bytes.to_vec(),
        ImageData::Planar(planes) => {
            let mut combined = Vec::new();
            for plane in planes {
                combined.extend_from_slice(plane);
            }
            combined
        }
    };

    let cs_name = match frame.color_space {
        ColorSpace::LinearRgb => "linear",
        ColorSpace::Srgb => "srgb",
        ColorSpace::Rec709 => "rec709",
        ColorSpace::Rec2020 => "rec2020",
        ColorSpace::DciP3 => "dci-p3",
        ColorSpace::Log => "log",
        ColorSpace::Luma => "luma",
        ColorSpace::YCbCr => "ycbcr",
        ColorSpace::Cmyk => "cmyk",
    };

    Ok(PyImageFrame {
        width: frame.width,
        height: frame.height,
        channels: u32::from(frame.components),
        bit_depth: u32::from(frame.pixel_type.bit_depth()),
        colorspace: cs_name.to_string(),
        data: data_bytes,
    })
}

/// Write a PyImageFrame to disk.
#[pyfunction]
pub fn write_image(frame: &PyImageFrame, path: &str) -> PyResult<()> {
    let file_path = std::path::Path::new(path);
    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let pixel_type = match frame.bit_depth {
        8 => PixelType::U8,
        10 => PixelType::U10,
        12 => PixelType::U12,
        16 => PixelType::U16,
        32 => PixelType::F32,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported bit depth: {}",
                other
            )));
        }
    };

    let color_space = match frame.colorspace.as_str() {
        "linear" | "linear-rgb" => ColorSpace::LinearRgb,
        "srgb" => ColorSpace::Srgb,
        "rec709" => ColorSpace::Rec709,
        "rec2020" => ColorSpace::Rec2020,
        "dci-p3" => ColorSpace::DciP3,
        "log" => ColorSpace::Log,
        "luma" => ColorSpace::Luma,
        "ycbcr" => ColorSpace::YCbCr,
        "cmyk" => ColorSpace::Cmyk,
        _ => ColorSpace::Srgb,
    };

    let components = frame.channels as u8;
    let image_data = ImageData::interleaved(frame.data.clone());

    let image_frame = ImageFrame::new(
        1,
        frame.width,
        frame.height,
        pixel_type,
        components,
        color_space,
        image_data,
    );

    match ext.as_str() {
        "dpx" => {
            oximedia_image::dpx::write_dpx(
                file_path,
                &image_frame,
                oximedia_image::Endian::native(),
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("DPX write error: {}", e))
            })?;
        }
        "exr" => {
            oximedia_image::exr::write_exr(
                file_path,
                &image_frame,
                oximedia_image::exr::ExrCompression::Zip,
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("EXR write error: {}", e))
            })?;
        }
        "tif" | "tiff" => {
            oximedia_image::tiff::write_tiff(
                file_path,
                &image_frame,
                oximedia_image::tiff::TiffCompression::None,
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("TIFF write error: {}", e))
            })?;
        }
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported output format: .{} (supported: dpx, exr, tif/tiff)",
                other
            )));
        }
    }

    Ok(())
}

/// Convert an image file from one format to another.
#[pyfunction]
#[pyo3(signature = (input, output, bit_depth = None, colorspace = None))]
pub fn convert_image(
    input: &str,
    output: &str,
    bit_depth: Option<u32>,
    colorspace: Option<&str>,
) -> PyResult<()> {
    let frame = read_image(input)?;

    let target_depth = bit_depth.unwrap_or(frame.bit_depth);
    let target_cs = colorspace.unwrap_or(&frame.colorspace);

    let converted = PyImageFrame {
        width: frame.width,
        height: frame.height,
        channels: frame.channels,
        bit_depth: target_depth,
        colorspace: target_cs.to_string(),
        data: if target_depth == frame.bit_depth {
            frame.data
        } else {
            // Simple depth conversion for 8<->16 bit
            convert_depth(&frame.data, frame.bit_depth, target_depth)?
        },
    };

    write_image(&converted, output)
}

/// Compute histogram from an image frame.
///
/// Returns a list of channel histograms, each containing 256 bin counts.
#[pyfunction]
#[pyo3(signature = (frame, mode = None))]
pub fn image_histogram(frame: &PyImageFrame, mode: Option<&str>) -> PyResult<Vec<Vec<u64>>> {
    if frame.bit_depth != 8 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Histogram currently supports 8-bit images only",
        ));
    }

    let hist_mode = mode.unwrap_or("rgb");

    match hist_mode {
        "luma" => {
            let mut hist = Histogram::new();
            let ch = frame.channels as usize;
            let pixel_count = (frame.width as usize) * (frame.height as usize);

            for i in 0..pixel_count {
                let luma = if ch >= 3 {
                    let r = frame.data[i * ch] as f64;
                    let g = frame.data[i * ch + 1] as f64;
                    let b = frame.data[i * ch + 2] as f64;
                    (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0) as u8
                } else {
                    frame.data[i * ch]
                };
                hist.accumulate(luma);
            }

            Ok(vec![hist.bins.to_vec()])
        }
        "rgb" | "per-channel" => {
            if frame.channels >= 3 {
                let mut rgb_hist = RgbHistogram::new();
                let ch = frame.channels as usize;
                let pixel_count = (frame.width as usize) * (frame.height as usize);

                for i in 0..pixel_count {
                    let base = i * ch;
                    rgb_hist.accumulate_rgb(
                        frame.data[base],
                        frame.data[base + 1],
                        frame.data[base + 2],
                    );
                }

                Ok(vec![
                    rgb_hist.red.bins.to_vec(),
                    rgb_hist.green.bins.to_vec(),
                    rgb_hist.blue.bins.to_vec(),
                ])
            } else {
                // Single channel
                let mut hist = Histogram::new();
                hist.accumulate_slice(&frame.data);
                Ok(vec![hist.bins.to_vec()])
            }
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown histogram mode '{}'. Valid: rgb, luma, per-channel",
            other
        ))),
    }
}

/// List supported image formats.
#[pyfunction]
pub fn list_supported_formats() -> Vec<String> {
    vec![
        "DPX (Digital Picture Exchange) - .dpx".to_string(),
        "OpenEXR (High Dynamic Range) - .exr".to_string(),
        "TIFF (Tagged Image File Format) - .tif/.tiff".to_string(),
        "PNG (Portable Network Graphics) - .png [detect only]".to_string(),
        "JPEG (Joint Photographic Experts Group) - .jpg/.jpeg [detect only]".to_string(),
        "BMP (Bitmap) - .bmp [detect only]".to_string(),
        "GIF (Graphics Interchange Format) - .gif [detect only]".to_string(),
        "WebP - .webp [detect only]".to_string(),
        "Cineon - .cin [detect only]".to_string(),
        "JPEG 2000 - .jp2 [detect only]".to_string(),
        "HEIF/HEIC - .heic/.heif [detect only]".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register image types and functions onto a Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyImageFrame>()?;
    m.add_class::<PyImageSequence>()?;
    m.add_class::<PyImageFilter>()?;
    m.add_function(wrap_pyfunction!(read_image, m)?)?;
    m.add_function(wrap_pyfunction!(write_image, m)?)?;
    m.add_function(wrap_pyfunction!(convert_image, m)?)?;
    m.add_function(wrap_pyfunction!(image_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(list_supported_formats, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn convert_depth(data: &[u8], from_depth: u32, to_depth: u32) -> PyResult<Vec<u8>> {
    match (from_depth, to_depth) {
        (8, 16) => {
            let mut out = Vec::with_capacity(data.len() * 2);
            for &b in data {
                let val = (u16::from(b) * 257) as u16; // scale 0-255 to 0-65535
                out.extend_from_slice(&val.to_le_bytes());
            }
            Ok(out)
        }
        (16, 8) => {
            if data.len() % 2 != 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "16-bit data must have even length",
                ));
            }
            let mut out = Vec::with_capacity(data.len() / 2);
            for chunk in data.chunks_exact(2) {
                let val = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push((val >> 8) as u8);
            }
            Ok(out)
        }
        (from, to) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Depth conversion from {}-bit to {}-bit not yet supported",
            from, to
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_image_frame_creation() {
        let data = vec![0u8; 300]; // 10x10x3
        let frame = PyImageFrame {
            width: 10,
            height: 10,
            channels: 3,
            bit_depth: 8,
            colorspace: "srgb".to_string(),
            data,
        };
        assert_eq!(frame.pixel_count(), 100);
        assert_eq!(frame.stride(), 30);
    }

    #[test]
    fn test_convert_depth_8_to_16() {
        let data = vec![0, 128, 255];
        let result = convert_depth(&data, 8, 16).expect("should succeed");
        assert_eq!(result.len(), 6);
        // Check 255 -> 65535
        let val = u16::from_le_bytes([result[4], result[5]]);
        assert_eq!(val, 65535);
    }

    #[test]
    fn test_convert_depth_16_to_8() {
        let data: Vec<u8> = vec![0, 0, 0xFF, 0xFF]; // 0 and 65535
        let result = convert_depth(&data, 16, 8).expect("should succeed");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 255);
    }

    #[test]
    fn test_list_supported_formats() {
        let formats = list_supported_formats();
        assert!(!formats.is_empty());
        assert!(formats.iter().any(|f| f.contains("DPX")));
        assert!(formats.iter().any(|f| f.contains("OpenEXR")));
    }

    #[test]
    fn test_repr_image_frame() {
        let frame = PyImageFrame {
            width: 1920,
            height: 1080,
            channels: 3,
            bit_depth: 10,
            colorspace: "rec709".to_string(),
            data: vec![],
        };
        let r = frame.__repr__();
        assert!(r.contains("1920x1080"));
        assert!(r.contains("10-bit"));
    }
}
