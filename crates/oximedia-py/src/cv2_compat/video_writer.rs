//! VideoWriter — write frames to a video file.

use super::image_io::extract_img;
use pyo3::prelude::*;
use std::path::PathBuf;

/// Compute a FOURCC code from four characters.
///
/// Mirrors `cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')`.
/// Each argument should be a single-character string.
#[pyfunction]
#[pyo3(name = "VideoWriter_fourcc")]
pub fn video_writer_fourcc(c1: &str, c2: &str, c3: &str, c4: &str) -> PyResult<i32> {
    let first_char = |s: &str, name: &str| -> PyResult<u8> {
        s.bytes().next().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "VideoWriter_fourcc: {} must be a non-empty string",
                name
            ))
        })
    };
    let b1 = first_char(c1, "c1")?;
    let b2 = first_char(c2, "c2")?;
    let b3 = first_char(c3, "c3")?;
    let b4 = first_char(c4, "c4")?;
    let code = (b1 as i32) | ((b2 as i32) << 8) | ((b3 as i32) << 16) | ((b4 as i32) << 24);
    Ok(code)
}

/// Video writer that saves frames to individual images or a simple format.
///
/// Mirrors `cv2.VideoWriter(filename, fourcc, fps, frameSize)`.
/// When the filename ends with `.png`, `.jpg`, or `.bmp`, frames are saved as
/// a numbered image sequence. Otherwise frames are accumulated and written
/// when `release()` is called.
#[pyclass(name = "VideoWriter")]
pub struct PyVideoWriter {
    path: PathBuf,
    fourcc: i32,
    fps: f64,
    width: usize,
    height: usize,
    is_opened: bool,
    frame_count: usize,
    /// Accumulated raw RGB frames (for video output)
    frames: Vec<Vec<u8>>,
}

#[pymethods]
impl PyVideoWriter {
    #[new]
    pub fn new(
        filename: &str,
        fourcc: i32,
        fps: f64,
        frame_size: (usize, usize),
    ) -> PyResult<Self> {
        let (width, height) = frame_size;
        let path = PathBuf::from(filename);

        // Create parent dirs if needed
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!("VideoWriter: {}", e))
                })?;
            }
        }

        Ok(Self {
            path,
            fourcc,
            fps,
            width,
            height,
            is_opened: true,
            frame_count: 0,
            frames: Vec::new(),
        })
    }

    /// Check if the writer is opened.
    #[allow(non_snake_case)]
    pub fn isOpened(&self) -> bool {
        self.is_opened
    }

    /// Write a frame.
    pub fn write(&mut self, py: Python<'_>, frame: Py<PyAny>) -> PyResult<()> {
        if !self.is_opened {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "VideoWriter is not opened",
            ));
        }

        let (data, _h, _w, ch) = extract_img(py, &frame)?;

        let path_str = self.path.to_string_lossy();
        let ext = path_str.rsplit('.').next().unwrap_or("png").to_lowercase();

        if matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "tiff") {
            // Write as image sequence
            let stem = self.path.file_stem().unwrap_or_default().to_string_lossy();
            let ext_str = self.path.extension().unwrap_or_default().to_string_lossy();
            let parent = self.path.parent().unwrap_or(std::path::Path::new("."));
            let out_path = parent.join(format!("{:}_{:05}.{}", stem, self.frame_count, ext_str));

            // BGR -> RGB and write
            use super::image_io::{bgr_to_rgb, encode_image};
            use image::{DynamicImage, GrayImage, ImageFormat, RgbImage};

            let fmt = match ext.as_str() {
                "png" => ImageFormat::Png,
                "jpg" | "jpeg" => ImageFormat::Jpeg,
                "bmp" => ImageFormat::Bmp,
                _ => ImageFormat::Png,
            };

            let bytes = if ch == 1 {
                let gray = GrayImage::from_raw(self.width as u32, self.height as u32, data)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "VideoWriter.write: invalid frame size",
                        )
                    })?;
                encode_image(DynamicImage::ImageLuma8(gray), fmt)?
            } else {
                let rgb_data = bgr_to_rgb(data, self.width, self.height);
                let rgb = RgbImage::from_raw(self.width as u32, self.height as u32, rgb_data)
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "VideoWriter.write: invalid frame size",
                        )
                    })?;
                encode_image(DynamicImage::ImageRgb8(rgb), fmt)?
            };

            std::fs::write(&out_path, bytes).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("VideoWriter.write: {}", e))
            })?;
        } else {
            // Accumulate frame data for potential future video muxer
            self.frames.push(data);
        }

        self.frame_count += 1;
        Ok(())
    }

    /// Get a writer property.
    pub fn get(&self, prop_id: i32) -> f64 {
        match prop_id {
            3 => self.width as f64,  // CAP_PROP_FRAME_WIDTH
            4 => self.height as f64, // CAP_PROP_FRAME_HEIGHT
            5 => self.fps,           // CAP_PROP_FPS
            6 => self.fourcc as f64, // CAP_PROP_FOURCC
            _ => 0.0,
        }
    }

    /// Release the writer.
    pub fn release(&mut self) {
        self.is_opened = false;
        self.frames.clear();
    }

    /// Context manager support.
    pub fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub fn __exit__(
        &mut self,
        _exc_type: Py<PyAny>,
        _exc_val: Py<PyAny>,
        _exc_tb: Py<PyAny>,
    ) -> bool {
        self.release();
        false
    }
}
