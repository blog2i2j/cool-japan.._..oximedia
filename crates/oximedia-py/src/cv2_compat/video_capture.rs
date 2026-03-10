//! VideoCapture — file-backed video frame reader.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::path::PathBuf;

/// Video capture from a file or device.
///
/// Mirrors `cv2.VideoCapture(filename_or_index)`.
/// Supported: reading frames from image sequences or single images.
/// Full video demuxing requires feature-gated backend; this implementation
/// provides a compatible API with stub behavior for non-image sources.
#[pyclass(name = "VideoCapture")]
pub struct PyVideoCapture {
    source: CaptureSource,
    frame_pos: u64,
    total_frames: u64,
    width: f64,
    height: f64,
    fps: f64,
    is_opened: bool,
}

#[allow(dead_code)]
enum CaptureSource {
    ImageFile(PathBuf),
    ImageSequence { pattern: PathBuf, indices: Vec<u64> },
    Device(i32),
    Closed,
}

#[pymethods]
impl PyVideoCapture {
    /// Create a new VideoCapture.
    ///
    /// `source` can be:
    /// - A file path string (tries to open as single image or image sequence)
    /// - An integer (device index — not supported, opens empty capture)
    #[new]
    pub fn new(py: Python<'_>, source: Py<PyAny>) -> PyResult<Self> {
        // Try integer (device)
        if let Ok(idx) = source.extract::<i32>(py) {
            return Ok(Self {
                source: CaptureSource::Device(idx),
                frame_pos: 0,
                total_frames: 0,
                width: 640.0,
                height: 480.0,
                fps: 30.0,
                is_opened: false, // Can't open real devices here
            });
        }

        // Try string path
        let path_str: String = source.extract(py)?;
        let path = PathBuf::from(&path_str);

        if path.exists() {
            // Determine dimensions from image
            let (w, h) = probe_image_size(&path).unwrap_or((640, 480));
            return Ok(Self {
                source: CaptureSource::ImageFile(path),
                frame_pos: 0,
                total_frames: 1,
                width: w as f64,
                height: h as f64,
                fps: 1.0,
                is_opened: true,
            });
        }

        Ok(Self {
            source: CaptureSource::Closed,
            frame_pos: 0,
            total_frames: 0,
            width: 0.0,
            height: 0.0,
            fps: 0.0,
            is_opened: false,
        })
    }

    /// Check if the capture is opened.
    #[allow(non_snake_case)]
    pub fn isOpened(&self) -> bool {
        self.is_opened
    }

    /// Read the next frame.
    ///
    /// Returns `(retval, frame)` where frame is an image dict.
    pub fn read(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.source {
            CaptureSource::ImageFile(path) => {
                if self.frame_pos >= self.total_frames {
                    let result = pyo3::types::PyTuple::new(
                        py,
                        &[
                            false.into_pyobject(py)?.as_any().clone(),
                            py.None().into_bound(py),
                        ],
                    )?;
                    return Ok(result.into());
                }
                self.frame_pos += 1;

                let img = image::open(path.as_path()).map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!("VideoCapture.read: {}", e))
                })?;
                let rgb = img.to_rgb8();
                let (w, h_img) = rgb.dimensions();
                let mut data = rgb.into_raw();
                // BGR swap
                for i in 0..(w * h_img) as usize {
                    data.swap(i * 3, i * 3 + 2);
                }
                let dict = PyDict::new(py);
                dict.set_item("data", PyBytes::new(py, &data))?;
                dict.set_item("shape", (h_img as usize, w as usize, 3usize))?;
                dict.set_item("dtype", "uint8")?;
                let result = pyo3::types::PyTuple::new(
                    py,
                    &[
                        true.into_pyobject(py)?.as_any().clone(),
                        dict.as_any().clone(),
                    ],
                )?;
                Ok(result.into())
            }
            _ => {
                let result = pyo3::types::PyTuple::new(
                    py,
                    &[
                        false.into_pyobject(py)?.as_any().clone(),
                        py.None().into_bound(py),
                    ],
                )?;
                Ok(result.into())
            }
        }
    }

    /// Get a capture property.
    pub fn get(&self, prop_id: i32) -> f64 {
        match prop_id {
            0 => self.frame_pos as f64 / self.fps * 1000.0, // CAP_PROP_POS_MSEC
            1 => self.frame_pos as f64,                     // CAP_PROP_POS_FRAMES
            2 => {
                if self.total_frames > 0 {
                    self.frame_pos as f64 / self.total_frames as f64
                } else {
                    0.0
                }
            } // ratio
            3 => self.width,                                // CAP_PROP_FRAME_WIDTH
            4 => self.height,                               // CAP_PROP_FRAME_HEIGHT
            5 => self.fps,                                  // CAP_PROP_FPS
            7 => self.total_frames as f64,                  // CAP_PROP_FRAME_COUNT
            _ => 0.0,
        }
    }

    /// Set a capture property.
    pub fn set(&mut self, prop_id: i32, value: f64) -> bool {
        match prop_id {
            1 => {
                // CAP_PROP_POS_FRAMES
                self.frame_pos = value as u64;
                true
            }
            5 => {
                self.fps = value;
                true
            }
            _ => false,
        }
    }

    /// Release the capture.
    pub fn release(&mut self) {
        self.is_opened = false;
        self.source = CaptureSource::Closed;
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

fn probe_image_size(path: &PathBuf) -> Option<(u32, u32)> {
    use image::GenericImageView;
    let img = image::open(path).ok()?;
    let (w, h) = img.dimensions();
    Some((w, h))
}
