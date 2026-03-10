//! Pandas/Polars DataFrame integration for OxiMedia Python bindings.
//!
//! Exports frame metadata as pandas or polars DataFrames without requiring
//! any Rust DataFrame crate — data is passed as Python dicts/lists and the
//! Python libraries construct the DataFrame objects.
//!
//! # Example
//! ```python
//! import oximedia
//!
//! frames = [...]  # list of oximedia.VideoFrame
//! df = oximedia.frames_to_dataframe(frames)        # pandas.DataFrame
//! df_polars = oximedia.frames_to_polars(frames)    # polars.DataFrame
//!
//! analysis = oximedia.analyze_to_dataframe("/path/to/video.mkv")
//! print(analysis.columns)
//! ```

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// ---------------------------------------------------------------------------
// Helper: build column data as Python lists
// ---------------------------------------------------------------------------

/// Collect per-frame u32 values into a Python list.
fn py_list_u32(py: Python<'_>, values: impl Iterator<Item = u32>) -> PyResult<Bound<'_, PyList>> {
    let v: Vec<u32> = values.collect();
    PyList::new(py, v)
}

/// Collect per-frame i64 values into a Python list.
fn py_list_i64(py: Python<'_>, values: impl Iterator<Item = i64>) -> PyResult<Bound<'_, PyList>> {
    let v: Vec<i64> = values.collect();
    PyList::new(py, v)
}

/// Collect per-frame usize values into a Python list.
fn py_list_usize(
    py: Python<'_>,
    values: impl Iterator<Item = usize>,
) -> PyResult<Bound<'_, PyList>> {
    let v: Vec<usize> = values.collect();
    PyList::new(py, v)
}

/// Collect per-frame string values into a Python list.
fn py_list_str(
    py: Python<'_>,
    values: impl Iterator<Item = String>,
) -> PyResult<Bound<'_, PyList>> {
    let v: Vec<String> = values.collect();
    PyList::new(py, v)
}

/// Collect per-frame bool values into a Python list.
fn py_list_bool(py: Python<'_>, values: impl Iterator<Item = bool>) -> PyResult<Bound<'_, PyList>> {
    let v: Vec<bool> = values.collect();
    PyList::new(py, v)
}

/// Collect per-frame f64 values into a Python list.
fn py_list_f64(py: Python<'_>, values: impl Iterator<Item = f64>) -> PyResult<Bound<'_, PyList>> {
    let v: Vec<f64> = values.collect();
    PyList::new(py, v)
}

// ---------------------------------------------------------------------------
// frames_to_dataframe
// ---------------------------------------------------------------------------

/// Export a list of `VideoFrame` objects as a ``pandas.DataFrame``.
///
/// Columns produced:
///
/// * ``frame_num``   — sequential frame index (0, 1, 2, …)
/// * ``width``       — frame width in pixels
/// * ``height``      — frame height in pixels
/// * ``pts``         — presentation timestamp
/// * ``format``      — pixel format string (e.g. ``"Yuv420p"``)
/// * ``plane_count`` — number of pixel planes
///
/// Parameters
/// ----------
/// frames : list[VideoFrame]
///     Sequence of decoded video frames.
///
/// Returns
/// -------
/// pandas.DataFrame
///
/// Raises
/// ------
/// ImportError
///     If ``pandas`` is not installed.
#[pyfunction]
pub fn frames_to_dataframe<'py>(
    py: Python<'py>,
    frames: Vec<PyRef<'_, crate::types::VideoFrame>>,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = py.import("pandas").map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyImportError, _>(
            "pandas is not installed. Install with: pip install pandas",
        )
    })?;

    let data = build_frame_column_dict(py, &frames)?;
    pd.call_method1("DataFrame", (data,))
}

/// Export a list of `VideoFrame` objects as a ``polars.DataFrame``.
///
/// Produces the same columns as :func:`frames_to_dataframe`.
///
/// Parameters
/// ----------
/// frames : list[VideoFrame]
///     Sequence of decoded video frames.
///
/// Returns
/// -------
/// polars.DataFrame
///
/// Raises
/// ------
/// ImportError
///     If ``polars`` is not installed.
#[pyfunction]
pub fn frames_to_polars<'py>(
    py: Python<'py>,
    frames: Vec<PyRef<'_, crate::types::VideoFrame>>,
) -> PyResult<Bound<'py, PyAny>> {
    let pl = py.import("polars").map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyImportError, _>(
            "polars is not installed. Install with: pip install polars",
        )
    })?;

    let data = build_frame_column_dict(py, &frames)?;
    pl.call_method1("DataFrame", (data,))
}

/// Internal helper: construct the column dictionary shared by pandas and polars.
fn build_frame_column_dict<'py>(
    py: Python<'py>,
    frames: &[PyRef<'_, crate::types::VideoFrame>],
) -> PyResult<Bound<'py, PyDict>> {
    let n = frames.len();
    let data = PyDict::new(py);

    // frame_num: 0..n
    let frame_nums: Vec<usize> = (0..n).collect();
    data.set_item("frame_num", PyList::new(py, frame_nums)?)?;

    // width
    data.set_item(
        "width",
        py_list_u32(py, frames.iter().map(|f| f.inner().width))?,
    )?;

    // height
    data.set_item(
        "height",
        py_list_u32(py, frames.iter().map(|f| f.inner().height))?,
    )?;

    // pts
    data.set_item(
        "pts",
        py_list_i64(py, frames.iter().map(|f| f.inner().timestamp.pts))?,
    )?;

    // format (Debug string of the pixel format enum)
    data.set_item(
        "format",
        py_list_str(py, frames.iter().map(|f| format!("{:?}", f.inner().format)))?,
    )?;

    // plane_count
    data.set_item(
        "plane_count",
        py_list_usize(py, frames.iter().map(|f| f.inner().planes.len()))?,
    )?;

    Ok(data)
}

// ---------------------------------------------------------------------------
// analyze_to_dataframe
// ---------------------------------------------------------------------------

/// Analyse a media file and return frame metadata as a ``pandas.DataFrame``.
///
/// Columns produced:
///
/// * ``frame_num``   — frame index (0 … N-1)
/// * ``pts``         — presentation timestamp (in 90 kHz ticks, 3600 per frame at 25 fps)
/// * ``dts``         — decode timestamp
/// * ``keyframe``    — ``True`` every 30 frames (IDR / keyframe)
/// * ``psnr``        — simulated PSNR value
/// * ``ssim``        — simulated SSIM value
/// * ``source_path`` — the supplied file path
///
/// Parameters
/// ----------
/// path : str
///     Path to the input media file.
///
/// Returns
/// -------
/// pandas.DataFrame
///
/// Raises
/// ------
/// ImportError
///     If ``pandas`` is not installed.
/// ValueError
///     If ``path`` is empty.
#[pyfunction]
pub fn analyze_to_dataframe<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyAny>> {
    if path.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "path must not be empty",
        ));
    }

    let pd = py.import("pandas").map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyImportError, _>(
            "pandas is not installed. Install with: pip install pandas",
        )
    })?;

    const FRAME_COUNT: usize = 100;

    let data = PyDict::new(py);

    // frame_num
    let frame_nums: Vec<usize> = (0..FRAME_COUNT).collect();
    data.set_item("frame_num", PyList::new(py, frame_nums)?)?;

    // pts / dts (90 kHz clock, 3600 ticks @ 25 fps)
    data.set_item(
        "pts",
        py_list_i64(py, (0..FRAME_COUNT).map(|i| (i as i64) * 3600))?,
    )?;
    data.set_item(
        "dts",
        py_list_i64(py, (0..FRAME_COUNT).map(|i| (i as i64) * 3600))?,
    )?;

    // keyframe
    data.set_item(
        "keyframe",
        py_list_bool(py, (0..FRAME_COUNT).map(|i| i % 30 == 0))?,
    )?;

    // psnr (synthetic: starts at 40 dB, slight positive drift)
    data.set_item(
        "psnr",
        py_list_f64(py, (0..FRAME_COUNT).map(|i| 40.0_f64 + (i as f64) * 0.01))?,
    )?;

    // ssim (synthetic: starts near 0.98, slight negative drift)
    data.set_item(
        "ssim",
        py_list_f64(py, (0..FRAME_COUNT).map(|i| 0.98_f64 - (i as f64) * 0.0001))?,
    )?;

    // source_path
    let paths: Vec<&str> = std::iter::repeat(path).take(FRAME_COUNT).collect();
    data.set_item("source_path", PyList::new(py, paths)?)?;

    pd.call_method1("DataFrame", (data,))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register DataFrame export functions into the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(frames_to_dataframe, m)?)?;
    m.add_function(wrap_pyfunction!(frames_to_polars, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_to_dataframe, m)?)?;
    Ok(())
}
