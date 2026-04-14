//! Jupyter notebook integration for OxiMedia Python bindings.
//!
//! Provides `show_video_frame`, `show_yuv_planes`, and `show_audio_waveform`
//! functions that display media frames inline when running inside a Jupyter
//! notebook, and fall back gracefully to writing files or printing stats
//! when running in a plain Python REPL.
//!
//! PNG encoding is performed in pure Rust using oxiarc-deflate for zlib compression
//! and an inline CRC32 implementation — no C libraries are required.
//!
//! # Example (Jupyter)
//! ```python
//! import oximedia
//!
//! frame = oximedia.VideoFrame(...)
//! oximedia.show_video_frame(frame)    # IPython.display.Image inline
//! oximedia.show_yuv_planes(frame)     # show each YUV plane separately
//!
//! audio = oximedia.AudioFrame(...)
//! oximedia.show_audio_waveform(audio) # matplotlib waveform or printed stats
//! ```

use std::io::Write as _;

use oxiarc_deflate::ZlibStreamEncoder;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

// ---------------------------------------------------------------------------
// CRC32 (IEEE polynomial) — inline, no external crate required
// ---------------------------------------------------------------------------

/// Build the CRC32 lookup table for the IEEE polynomial (0xEDB88320).
fn build_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            if c & 1 != 0 {
                c = 0xEDB8_8320u32 ^ (c >> 1);
            } else {
                c >>= 1;
            }
        }
        *entry = c;
    }
    table
}

/// Compute a CRC32 checksum over `data`, starting from `crc`.
fn crc32_update(crc: u32, data: &[u8], table: &[u32; 256]) -> u32 {
    let mut c = crc ^ 0xFFFF_FFFF;
    for &b in data {
        c = table[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

/// Compute a standalone CRC32 over `data`.
fn crc32(data: &[u8]) -> u32 {
    let table = build_crc32_table();
    crc32_update(0, data, &table)
}

// ---------------------------------------------------------------------------
// PNG encoder (grayscale, 8-bit)
// ---------------------------------------------------------------------------

/// Write a big-endian u32 into a byte buffer.
fn push_u32_be(buf: &mut Vec<u8>, v: u32) {
    buf.push((v >> 24) as u8);
    buf.push((v >> 16) as u8);
    buf.push((v >> 8) as u8);
    buf.push(v as u8);
}

/// Append a PNG chunk: length(4) + type(4) + data + crc(4).
fn write_png_chunk(buf: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    push_u32_be(buf, data.len() as u32);
    buf.extend_from_slice(chunk_type);
    buf.extend_from_slice(data);
    let crc_data: Vec<u8> = chunk_type.iter().chain(data.iter()).copied().collect();
    push_u32_be(buf, crc32(&crc_data));
}

/// Encode raw 8-bit grayscale pixel data into a valid PNG byte stream.
///
/// The `data` slice must contain exactly `width * height` bytes (row-major,
/// no padding).  If it is shorter, the image is zero-padded; if longer, the
/// excess is ignored.
pub fn write_png_grayscale(width: u32, height: u32, data: &[u8]) -> Vec<u8> {
    let mut png = Vec::new();

    // PNG signature
    png.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

    // IHDR chunk: width(4) height(4) bit_depth(1) color_type(1) compression(1) filter(1) interlace(1)
    let mut ihdr = Vec::with_capacity(13);
    push_u32_be(&mut ihdr, width);
    push_u32_be(&mut ihdr, height);
    ihdr.push(8); // bit depth
    ihdr.push(0); // color type: grayscale
    ihdr.push(0); // compression method: deflate
    ihdr.push(0); // filter method: adaptive
    ihdr.push(0); // interlace method: none
    write_png_chunk(&mut png, b"IHDR", &ihdr);

    // Build the raw scanline data: each row is prefixed with a filter byte (0 = None).
    let row_bytes = width as usize;
    let mut raw = Vec::with_capacity((row_bytes + 1) * height as usize);
    for row in 0..(height as usize) {
        raw.push(0u8); // filter type: None
        let start = row * row_bytes;
        let end = start + row_bytes;
        if start < data.len() {
            let avail = end.min(data.len());
            raw.extend_from_slice(&data[start..avail]);
            // Zero-pad if row extends past available data
            if avail < end {
                raw.extend(std::iter::repeat(0u8).take(end - avail));
            }
        } else {
            raw.extend(std::iter::repeat(0u8).take(row_bytes));
        }
    }

    // Zlib-compress the raw scanlines for the IDAT chunk.
    let mut encoder = ZlibStreamEncoder::new(Vec::new(), 6);
    // If compression fails we fall back to an uncompressed-looking stream.
    let compressed = if encoder.write_all(&raw).is_ok() {
        encoder.finish().unwrap_or_else(|_| raw.clone())
    } else {
        raw.clone()
    };

    write_png_chunk(&mut png, b"IDAT", &compressed);

    // IEND chunk (empty data)
    write_png_chunk(&mut png, b"IEND", &[]);

    png
}

// ---------------------------------------------------------------------------
// IPython display helpers
// ---------------------------------------------------------------------------

/// Try to display `png_bytes` inline via `IPython.display`.
///
/// Returns `Ok(true)` if IPython was available and the image was displayed,
/// `Ok(false)` if IPython is absent (caller should fall back to file I/O).
fn try_ipython_display(py: Python<'_>, png_bytes: &[u8]) -> PyResult<bool> {
    let ipython_display = match py.import("IPython.display") {
        Ok(m) => m,
        Err(_) => return Ok(false),
    };
    let image_class = ipython_display.getattr("Image").map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "IPython.display.Image not found: {e}"
        ))
    })?;

    let bytes_obj = PyBytes::new(py, png_bytes);
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("data", bytes_obj)?;
    let image_obj = image_class.call((), Some(&kwargs)).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "IPython.display.Image() failed: {e}"
        ))
    })?;

    ipython_display
        .call_method1("display", (image_obj,))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "IPython.display.display() failed: {e}"
            ))
        })?;
    Ok(true)
}

/// Write PNG bytes to `path` in the system temp directory and print the path.
fn fallback_write_png(py: Python<'_>, filename: &str, png_bytes: &[u8]) -> PyResult<()> {
    let tmp = std::env::temp_dir().join(filename);
    std::fs::write(&tmp, png_bytes).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "failed to write preview PNG to {}: {e}",
            tmp.display()
        ))
    })?;
    let builtins = py.import("builtins").map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("could not import builtins: {e}"))
    })?;
    builtins.call_method1(
        "print",
        (format!(
            "[oximedia] Saved frame preview to: {}",
            tmp.display()
        ),),
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Public pyfunction: show_video_frame
// ---------------------------------------------------------------------------

/// Display a `VideoFrame` inline in a Jupyter notebook.
///
/// The Y (luminance) plane is encoded as a grayscale PNG and displayed via
/// ``IPython.display.Image``.  When IPython is not available, the PNG is
/// written to a temp file and the path is printed.
///
/// Parameters
/// ----------
/// frame : VideoFrame
///     The video frame to display.
#[pyfunction]
pub fn show_video_frame(py: Python<'_>, frame: &crate::types::VideoFrame) -> PyResult<()> {
    let inner = frame.inner();
    let width = inner.width;
    let height = inner.height;

    // Use the Y (luma) plane — plane 0
    let plane_data = inner
        .planes
        .first()
        .map(|p| p.data.as_slice())
        .unwrap_or(&[]);

    let png = write_png_grayscale(width, height, plane_data);

    if !try_ipython_display(py, &png)? {
        fallback_write_png(py, "oximedia_frame_preview.png", &png)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public pyfunction: show_yuv_planes
// ---------------------------------------------------------------------------

/// Display each YUV plane of a `VideoFrame` separately in a Jupyter notebook.
///
/// Each plane (Y, U, V) is encoded as an individual grayscale PNG.
/// IPython inline display is attempted first; if unavailable the PNGs are
/// written to temp files.
///
/// Parameters
/// ----------
/// frame : VideoFrame
///     The video frame whose planes to display.
#[pyfunction]
pub fn show_yuv_planes(py: Python<'_>, frame: &crate::types::VideoFrame) -> PyResult<()> {
    let inner = frame.inner();
    let plane_names = ["Y", "U", "V", "A"]; // label up to 4 planes

    for (idx, plane) in inner.planes.iter().enumerate() {
        // For chroma planes in YUV 4:2:0, width/height are halved.
        // We use the plane's stride as width approximation for display.
        let plane_width = if idx == 0 {
            inner.width
        } else {
            // Chroma width for 4:2:0
            (inner.width + 1) / 2
        };
        let plane_height = if idx == 0 {
            inner.height
        } else {
            (inner.height + 1) / 2
        };

        let label = plane_names.get(idx).copied().unwrap_or("?");
        let png = write_png_grayscale(plane_width, plane_height, &plane.data);

        if !try_ipython_display(py, &png)? {
            let filename = format!("oximedia_plane_{idx}.png");
            fallback_write_png(py, &filename, &png)?;
        } else {
            // Print a label so the user knows which plane they are seeing
            let builtins = py.import("builtins")?;
            builtins.call_method1("print", (format!("[oximedia] Plane {idx} ({label})"),))?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public pyfunction: show_audio_waveform
// ---------------------------------------------------------------------------

/// Display an `AudioFrame` waveform in a Jupyter notebook.
///
/// Attempts to use matplotlib to render a waveform plot.  If matplotlib is
/// not installed, basic statistics (min, max, RMS) are printed instead.
///
/// Parameters
/// ----------
/// frame : AudioFrame
///     The audio frame to visualise.
#[pyfunction]
pub fn show_audio_waveform(py: Python<'_>, frame: &crate::types::AudioFrame) -> PyResult<()> {
    let inner = frame.inner();
    let duration = inner.duration_seconds();
    let sample_count = inner.sample_count;

    // Attempt to convert samples to f32
    let samples_f32 = inner.to_f32().unwrap_or_default();

    // Try matplotlib path
    match py.import("matplotlib.pyplot") {
        Ok(plt) => {
            let np = py.import("numpy").map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "numpy import failed (required with matplotlib): {e}"
                ))
            })?;

            // Build time axis via np.linspace(0, duration, sample_count)
            let x_arr = np
                .call_method1("linspace", (0.0f64, duration, sample_count))
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "np.linspace failed: {e}"
                    ))
                })?;

            // Convert samples to a Python list for matplotlib
            let py_samples = pyo3::types::PyList::new(py, &samples_f32)?;

            plt.call_method1("figure", (pyo3::types::PyTuple::empty(py),))
                .ok(); // ignore if figure call fails

            plt.call_method1("plot", (x_arr, py_samples)).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("plt.plot() failed: {e}"))
            })?;

            plt.call_method1("xlabel", ("Time (s)",))?;
            plt.call_method1("ylabel", ("Amplitude",))?;
            plt.call_method1("title", ("Audio Waveform",))?;

            // Check if running in notebook (has inline backend)
            plt.call_method0("show").ok(); // best-effort; may be no-op in some backends
        }
        Err(_) => {
            // Fallback: print statistics to stdout
            let builtins = py.import("builtins").map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "could not import builtins: {e}"
                ))
            })?;

            let stats = compute_waveform_stats(&samples_f32);
            builtins.call_method1(
                "print",
                (format!(
                    "[oximedia] AudioFrame waveform stats | samples: {} | duration: {:.3}s | \
                     min: {:.4} | max: {:.4} | rms: {:.4}",
                    sample_count, duration, stats.min, stats.max, stats.rms
                ),),
            )?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Waveform statistics (fallback path)
// ---------------------------------------------------------------------------

struct WaveformStats {
    min: f32,
    max: f32,
    rms: f32,
}

fn compute_waveform_stats(samples: &[f32]) -> WaveformStats {
    if samples.is_empty() {
        return WaveformStats {
            min: 0.0,
            max: 0.0,
            rms: 0.0,
        };
    }
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut sum_sq = 0.0f64;

    for &s in samples {
        if s < min {
            min = s;
        }
        if s > max {
            max = s;
        }
        sum_sq += f64::from(s) * f64::from(s);
    }

    let rms = (sum_sq / samples.len() as f64).sqrt() as f32;
    WaveformStats { min, max, rms }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register Jupyter display functions into the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(show_video_frame, m)?)?;
    m.add_function(wrap_pyfunction!(show_yuv_planes, m)?)?;
    m.add_function(wrap_pyfunction!(show_audio_waveform, m)?)?;
    Ok(())
}
