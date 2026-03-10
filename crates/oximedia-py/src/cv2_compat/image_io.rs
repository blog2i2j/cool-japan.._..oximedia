//! Image I/O functions: imread, imwrite, imdecode, imencode.

use image::{DynamicImage, GrayImage, ImageFormat, RgbImage};
use ndarray::Array3;
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use std::io::Cursor;

/// Load an image from a file path.
///
/// Returns a `numpy.ndarray` with shape `(H, W, C)` and dtype `uint8`.
/// Channels are in BGR order (OpenCV convention).
#[pyfunction]
#[pyo3(name = "imread", signature = (filename, flags=1))]
pub fn imread(py: Python<'_>, filename: &str, flags: i32) -> PyResult<Py<PyAny>> {
    let img = image::open(filename)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("imread: {}", e)))?;

    match flags {
        0 => {
            // IMREAD_GRAYSCALE
            let gray = img.to_luma8();
            let (w, h) = gray.dimensions();
            let data = gray.into_raw();
            make_image_output(py, data, h as usize, w as usize, 1)
        }
        _ => {
            // IMREAD_COLOR (default) — BGR
            let rgb = img.to_rgb8();
            let (w, h) = rgb.dimensions();
            let data = rgb_to_bgr(rgb.into_raw(), w as usize, h as usize);
            make_image_output(py, data, h as usize, w as usize, 3)
        }
    }
}

/// Write an image to a file.
#[pyfunction]
#[pyo3(name = "imwrite", signature = (filename, img))]
pub fn imwrite(py: Python<'_>, filename: &str, img: Py<PyAny>) -> PyResult<bool> {
    let (data, h, w, ch) = extract_img(py, &img)?;
    let fmt = infer_format(filename)?;

    let bytes = if ch == 1 {
        let gray = GrayImage::from_raw(w as u32, h as u32, data).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("imwrite: invalid image dimensions")
        })?;
        encode_image(DynamicImage::ImageLuma8(gray), fmt)?
    } else {
        let rgb_data = bgr_to_rgb(data, w, h);
        let rgb = RgbImage::from_raw(w as u32, h as u32, rgb_data).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("imwrite: invalid image dimensions")
        })?;
        encode_image(DynamicImage::ImageRgb8(rgb), fmt)?
    };

    std::fs::write(filename, bytes)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("imwrite: {}", e)))?;
    Ok(true)
}

/// Decode an image from a byte buffer.
#[pyfunction]
#[pyo3(name = "imdecode", signature = (buf, flags=1))]
pub fn imdecode(py: Python<'_>, buf: &[u8], flags: i32) -> PyResult<Py<PyAny>> {
    let img = image::load_from_memory(buf)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("imdecode: {}", e)))?;

    match flags {
        0 => {
            let gray = img.to_luma8();
            let (w, h) = gray.dimensions();
            let data = gray.into_raw();
            make_image_output(py, data, h as usize, w as usize, 1)
        }
        _ => {
            let rgb = img.to_rgb8();
            let (w, h) = rgb.dimensions();
            let data = rgb_to_bgr(rgb.into_raw(), w as usize, h as usize);
            make_image_output(py, data, h as usize, w as usize, 3)
        }
    }
}

/// Encode an image to a byte buffer.
///
/// Returns `(retval, buf)` tuple where retval=True and buf is bytes.
#[pyfunction]
#[pyo3(name = "imencode", signature = (ext, img, params=None))]
pub fn imencode(
    py: Python<'_>,
    ext: &str,
    img: Py<PyAny>,
    params: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let _ = params; // ignored
    let (data, h, w, ch) = extract_img(py, &img)?;

    let fmt = match ext {
        ".png" | "png" => ImageFormat::Png,
        ".jpg" | ".jpeg" | "jpg" | "jpeg" => ImageFormat::Jpeg,
        ".bmp" | "bmp" => ImageFormat::Bmp,
        ".tiff" | ".tif" | "tiff" => ImageFormat::Tiff,
        _ => ImageFormat::Png,
    };

    let bytes = if ch == 1 {
        let gray = GrayImage::from_raw(w as u32, h as u32, data).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("imencode: invalid dimensions")
        })?;
        encode_image(DynamicImage::ImageLuma8(gray), fmt)?
    } else {
        let rgb_data = bgr_to_rgb(data, w, h);
        let rgb = RgbImage::from_raw(w as u32, h as u32, rgb_data).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("imencode: invalid dimensions")
        })?;
        encode_image(DynamicImage::ImageRgb8(rgb), fmt)?
    };

    let py_bytes = PyBytes::new(py, &bytes);
    let tup = PyTuple::new(
        py,
        &[
            true.into_pyobject(py)?.as_any().clone(),
            py_bytes.as_any().clone(),
        ],
    )?;
    Ok(tup.into())
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Create a numpy `ndarray` with shape `(H, W, C)` and dtype `uint8` from raw bytes.
///
/// This is the primary output format — true numpy arrays, compatible with real OpenCV.
pub(crate) fn make_image_output(
    py: Python<'_>,
    data: Vec<u8>,
    h: usize,
    w: usize,
    ch: usize,
) -> PyResult<Py<PyAny>> {
    let arr = Array3::from_shape_vec((h, w, ch), data).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("image shape error: {}", e))
    })?;
    Ok(arr.into_pyarray(py).into_any().unbind())
}

/// Extract image as `(data, height, width, channels)` from a Python object.
///
/// Accepts our internal dict format or numpy arrays.
pub(crate) fn extract_img(
    py: Python<'_>,
    obj: &Py<PyAny>,
) -> PyResult<(Vec<u8>, usize, usize, usize)> {
    let bound = obj.bind(py);

    // Handle dict-format (our internal format)
    if let Ok(dict) = bound.cast::<PyDict>() {
        let data_item = dict
            .get_item("data")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("image dict missing 'data'"))?;
        let data: Vec<u8> = data_item.extract()?;
        let shape_item = dict
            .get_item("shape")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("image dict missing 'shape'"))?;
        let shape: (usize, usize, usize) = shape_item.extract().or_else(|_| {
            let (h, w): (usize, usize) = shape_item.extract()?;
            Ok::<_, PyErr>((h, w, 1))
        })?;
        return Ok((data, shape.0, shape.1, shape.2));
    }

    // Handle numpy array (has .tobytes() and .shape)
    if bound.hasattr("tobytes")? && bound.hasattr("shape")? {
        let shape_any = bound.getattr("shape")?;
        let (h, w, ch) = shape_any.extract::<(usize, usize, usize)>().or_else(|_| {
            let (h, w): (usize, usize) = shape_any.extract()?;
            Ok::<_, PyErr>((h, w, 1))
        })?;
        let data_bytes = bound.call_method0("tobytes")?;
        let data: Vec<u8> = data_bytes.extract()?;
        return Ok((data, h, w, ch));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected image as dict with 'data'/'shape' keys, or numpy array",
    ))
}

pub(crate) fn rgb_to_bgr(mut data: Vec<u8>, w: usize, h: usize) -> Vec<u8> {
    for y in 0..h {
        for x in 0..w {
            let off = (y * w + x) * 3;
            data.swap(off, off + 2); // R <-> B
        }
    }
    data
}

pub(crate) fn bgr_to_rgb(data: Vec<u8>, w: usize, h: usize) -> Vec<u8> {
    rgb_to_bgr(data, w, h) // symmetric
}

fn infer_format(filename: &str) -> PyResult<ImageFormat> {
    let lower = filename.to_lowercase();
    if lower.ends_with(".png") {
        return Ok(ImageFormat::Png);
    }
    if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        return Ok(ImageFormat::Jpeg);
    }
    if lower.ends_with(".bmp") {
        return Ok(ImageFormat::Bmp);
    }
    if lower.ends_with(".tiff") || lower.ends_with(".tif") {
        return Ok(ImageFormat::Tiff);
    }
    if lower.ends_with(".webp") {
        return Ok(ImageFormat::WebP);
    }
    Ok(ImageFormat::Png)
}

pub(crate) fn encode_image(img: DynamicImage, fmt: ImageFormat) -> PyResult<Vec<u8>> {
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, fmt)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("encode: {}", e)))?;
    Ok(buf.into_inner())
}
